#include "mlp_cuda.hpp"
#include "cuda_error_handling.cuh"
#include "fused_kernels.cuh"
#include "cublas_handle_pool.hpp"
#include <cmath>
#include <stdexcept>
#include <cublas_v2.h>

#define MLP_CUBLAS_CHECK(call)                                                                     \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(_err)) +    \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                                          \
    } while (0)

namespace qwen {
namespace cuda {

__global__ void convert_bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ bf16_in,
                                            float* __restrict__ fp32_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        fp32_out[idx] = __bfloat162float(bf16_in[idx]);
    }
}

__global__ void mlp_down_kernel_bf16(const float* __restrict__ hidden, float* __restrict__ output,
                                     const __nv_bfloat16* __restrict__ down_weight,
                                     int intermediate_size, int hidden_size, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size)
        return;

    float sum = 0.0f;
    for (int j = 0; j < intermediate_size; ++j) {
        float w = __bfloat162float(down_weight[i * intermediate_size + j]);
        sum += w * hidden[j];
    }
    output[i] = sum + beta * output[i];
}

CudaMLP::CudaMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      d_gate_proj_weight_(nullptr), d_up_proj_weight_(nullptr), d_down_proj_weight_(nullptr),
      d_hidden_buf_(nullptr), max_hidden_batch_(0) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMalloc(&d_gate_proj_weight_, gate_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_up_proj_weight_, gate_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_down_proj_weight_, down_size * sizeof(__nv_bfloat16));
}

CudaMLP::~CudaMLP() {
    if (d_gate_proj_weight_)
        cudaFree(d_gate_proj_weight_);
    if (d_up_proj_weight_)
        cudaFree(d_up_proj_weight_);
    if (d_down_proj_weight_)
        cudaFree(d_down_proj_weight_);
    if (d_hidden_buf_)
        cudaFree(d_hidden_buf_);
}

void CudaMLP::ensure_hidden_buffer(int batch_size) const {
    if (batch_size <= max_hidden_batch_ && d_hidden_buf_ != nullptr) {
        return;
    }
    if (d_hidden_buf_) {
        cudaFree(d_hidden_buf_);
    }
    size_t bytes = static_cast<size_t>(batch_size) * intermediate_size_ * sizeof(float) * 2;
    cudaMalloc(&d_hidden_buf_, bytes);
    max_hidden_batch_ = batch_size;
}

void CudaMLP::set_weights(const std::vector<float>& gate_proj_weight,
                          const std::vector<float>& up_proj_weight,
                          const std::vector<float>& down_proj_weight) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    std::vector<__nv_bfloat16> h_gate_bf16(gate_size);
    std::vector<__nv_bfloat16> h_up_bf16(gate_size);
    std::vector<__nv_bfloat16> h_down_bf16(down_size);

    for (size_t i = 0; i < gate_size; ++i) {
        h_gate_bf16[i] = __float2bfloat16(gate_proj_weight[i]);
        h_up_bf16[i] = __float2bfloat16(up_proj_weight[i]);
    }
    for (size_t i = 0; i < down_size; ++i) {
        h_down_bf16[i] = __float2bfloat16(down_proj_weight[i]);
    }

    cudaMemcpy(d_gate_proj_weight_, h_gate_bf16.data(), gate_size * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_weight_, h_up_bf16.data(), gate_size * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_weight_, h_down_bf16.data(), down_size * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
}

__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ weight,
                                 const float* __restrict__ input, float* __restrict__ output,
                                 int out_size, int in_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_size) return;

    float sum = 0.0f;
    for (int col = 0; col < in_size; ++col) {
        float w = __bfloat162float(weight[row * in_size + col]);
        sum += w * input[col];
    }
    output[row] = sum;
}

__global__ void gemm_bf16_kernel(const __nv_bfloat16* __restrict__ weight,
                                 const float* __restrict__ input, float* __restrict__ output,
                                 int out_size, int in_size, int batch_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (row >= out_size || b >= batch_size) return;

    float sum = 0.0f;
    const float* in_ptr = input + b * in_size;
    float* out_ptr = output + b * out_size;

    for (int col = 0; col < in_size; ++col) {
        float w = __bfloat162float(weight[row * in_size + col]);
        sum += w * in_ptr[col];
    }
    out_ptr[row] = sum;
}

void CudaMLP::forward(const float* input, float* output, int batch_size) const {
    cublasHandle_t handle = CublasHandlePool::instance().get();

    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256);
        gemv_bf16_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, input, d_gate_buf,
                                                intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        gemv_bf16_kernel<<<grid_gate, block>>>(d_up_proj_weight_, input, d_up_buf,
                                                intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        dim3 grid_down((hidden_size_ + 255) / 256);
        mlp_down_kernel_bf16<<<grid_down, block>>>(d_gate_buf, output, d_down_proj_weight_,
                                                    intermediate_size_, hidden_size_, 0.0f);
        CUDA_CHECK_LAST_KERNEL();

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256, batch_size);
        gemm_bf16_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, input, d_gate_buf,
                                                intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        gemm_bf16_kernel<<<grid_gate, block>>>(d_up_proj_weight_, input, d_up_buf,
                                                intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        dim3 grid_down((hidden_size_ + 255) / 256, batch_size);
        gemm_bf16_kernel<<<grid_down, block>>>(d_down_proj_weight_, d_gate_buf, output,
                                                hidden_size_, intermediate_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();
    }
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    cublasHandle_t handle = CublasHandlePool::instance().get();

    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256);
        gemv_bf16_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, input, d_gate_buf,
                                                intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        gemv_bf16_kernel<<<grid_gate, block>>>(d_up_proj_weight_, input, d_up_buf,
                                                intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        dim3 grid_down((hidden_size_ + 255) / 256);
        mlp_down_kernel_bf16<<<grid_down, block>>>(d_gate_buf, residual, d_down_proj_weight_,
                                                    intermediate_size_, hidden_size_, 1.0f);
        CUDA_CHECK_LAST_KERNEL();

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256, batch_size);
        gemm_bf16_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, input, d_gate_buf,
                                                intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        gemm_bf16_kernel<<<grid_gate, block>>>(d_up_proj_weight_, input, d_up_buf,
                                                intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        dim3 grid_down((hidden_size_ + 255) / 256, batch_size);
        gemm_bf16_kernel<<<grid_down, block>>>(d_down_proj_weight_, d_gate_buf, residual,
                                                hidden_size_, intermediate_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();
    }
}

} // namespace cuda
} // namespace qwen
