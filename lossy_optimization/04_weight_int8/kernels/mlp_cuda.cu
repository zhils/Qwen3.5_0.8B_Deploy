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

__global__ void gemv_int8_kernel(const int8_t* __restrict__ weight,
                                 const float* __restrict__ weight_scale,
                                 const float* __restrict__ input, float* __restrict__ output,
                                 int out_size, int in_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_size) return;

    float scale = weight_scale[row];
    float sum = 0.0f;
    for (int col = 0; col < in_size; ++col) {
        sum += static_cast<float>(weight[row * in_size + col]) * input[col];
    }
    output[row] = sum * scale;
}

__global__ void gemm_int8_kernel(const int8_t* __restrict__ weight,
                                 const float* __restrict__ weight_scale,
                                 const float* __restrict__ input, float* __restrict__ output,
                                 int out_size, int in_size, int batch_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (row >= out_size || b >= batch_size) return;

    float scale = weight_scale[row];
    float sum = 0.0f;
    const float* in_ptr = input + b * in_size;
    float* out_ptr = output + b * out_size;

    for (int col = 0; col < in_size; ++col) {
        sum += static_cast<float>(weight[row * in_size + col]) * in_ptr[col];
    }
    out_ptr[row] = sum * scale;
}

__global__ void mlp_down_kernel_int8(const float* __restrict__ hidden, float* __restrict__ output,
                                     const int8_t* __restrict__ down_weight,
                                     const float* __restrict__ down_scale,
                                     int intermediate_size, int hidden_size, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size)
        return;

    float scale = down_scale[i];
    float sum = 0.0f;
    for (int j = 0; j < intermediate_size; ++j) {
        sum += static_cast<float>(down_weight[i * intermediate_size + j]) * hidden[j];
    }
    output[i] = sum * scale + beta * output[i];
}

CudaMLP::CudaMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      d_gate_proj_weight_(nullptr), d_up_proj_weight_(nullptr), d_down_proj_weight_(nullptr),
      d_gate_proj_scale_(nullptr), d_up_proj_scale_(nullptr), d_down_proj_scale_(nullptr),
      d_hidden_buf_(nullptr), max_hidden_batch_(0) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMalloc(&d_gate_proj_weight_, gate_size * sizeof(int8_t));
    cudaMalloc(&d_up_proj_weight_, gate_size * sizeof(int8_t));
    cudaMalloc(&d_down_proj_weight_, down_size * sizeof(int8_t));
    cudaMalloc(&d_gate_proj_scale_, intermediate_size_ * sizeof(float));
    cudaMalloc(&d_up_proj_scale_, intermediate_size_ * sizeof(float));
    cudaMalloc(&d_down_proj_scale_, hidden_size_ * sizeof(float));
}

CudaMLP::~CudaMLP() {
    if (d_gate_proj_weight_) cudaFree(d_gate_proj_weight_);
    if (d_up_proj_weight_) cudaFree(d_up_proj_weight_);
    if (d_down_proj_weight_) cudaFree(d_down_proj_weight_);
    if (d_gate_proj_scale_) cudaFree(d_gate_proj_scale_);
    if (d_up_proj_scale_) cudaFree(d_up_proj_scale_);
    if (d_down_proj_scale_) cudaFree(d_down_proj_scale_);
    if (d_hidden_buf_) cudaFree(d_hidden_buf_);
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

    std::vector<int8_t> h_gate_int8(gate_size);
    std::vector<int8_t> h_up_int8(gate_size);
    std::vector<int8_t> h_down_int8(down_size);
    std::vector<float> h_gate_scale(intermediate_size_);
    std::vector<float> h_up_scale(intermediate_size_);
    std::vector<float> h_down_scale(hidden_size_);

    for (int row = 0; row < intermediate_size_; ++row) {
        float max_val = 0.0f;
        for (int col = 0; col < hidden_size_; ++col) {
            max_val = fmaxf(max_val, fabsf(gate_proj_weight[row * hidden_size_ + col]));
        }
        float scale = max_val / 127.0f;
        h_gate_scale[row] = scale;
        for (int col = 0; col < hidden_size_; ++col) {
            h_gate_int8[row * hidden_size_ + col] = static_cast<int8_t>(
                roundf(gate_proj_weight[row * hidden_size_ + col] / scale));
        }
    }

    for (int row = 0; row < intermediate_size_; ++row) {
        float max_val = 0.0f;
        for (int col = 0; col < hidden_size_; ++col) {
            max_val = fmaxf(max_val, fabsf(up_proj_weight[row * hidden_size_ + col]));
        }
        float scale = max_val / 127.0f;
        h_up_scale[row] = scale;
        for (int col = 0; col < hidden_size_; ++col) {
            h_up_int8[row * hidden_size_ + col] = static_cast<int8_t>(
                roundf(up_proj_weight[row * hidden_size_ + col] / scale));
        }
    }

    for (int row = 0; row < hidden_size_; ++row) {
        float max_val = 0.0f;
        for (int col = 0; col < intermediate_size_; ++col) {
            max_val = fmaxf(max_val, fabsf(down_proj_weight[row * intermediate_size_ + col]));
        }
        float scale = max_val / 127.0f;
        h_down_scale[row] = scale;
        for (int col = 0; col < intermediate_size_; ++col) {
            h_down_int8[row * intermediate_size_ + col] = static_cast<int8_t>(
                roundf(down_proj_weight[row * intermediate_size_ + col] / scale));
        }
    }

    cudaMemcpy(d_gate_proj_weight_, h_gate_int8.data(), gate_size * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_weight_, h_up_int8.data(), gate_size * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_weight_, h_down_int8.data(), down_size * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate_proj_scale_, h_gate_scale.data(), intermediate_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_scale_, h_up_scale.data(), intermediate_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_scale_, h_down_scale.data(), hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
}

void CudaMLP::forward(const float* input, float* output, int batch_size) const {
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256);
        gemv_int8_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, d_gate_proj_scale_,
                                                input, d_gate_buf, intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        gemv_int8_kernel<<<grid_gate, block>>>(d_up_proj_weight_, d_up_proj_scale_,
                                                input, d_up_buf, intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        dim3 grid_down((hidden_size_ + 255) / 256);
        mlp_down_kernel_int8<<<grid_down, block>>>(d_gate_buf, output, d_down_proj_weight_,
                                                    d_down_proj_scale_, intermediate_size_, hidden_size_, 0.0f);
        CUDA_CHECK_LAST_KERNEL();

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256, batch_size);
        gemm_int8_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, d_gate_proj_scale_,
                                                input, d_gate_buf, intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        gemm_int8_kernel<<<grid_gate, block>>>(d_up_proj_weight_, d_up_proj_scale_,
                                                input, d_up_buf, intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        dim3 grid_down((hidden_size_ + 255) / 256, batch_size);
        gemm_int8_kernel<<<grid_down, block>>>(d_down_proj_weight_, d_down_proj_scale_,
                                                d_gate_buf, output, hidden_size_, intermediate_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();
    }
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256);
        gemv_int8_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, d_gate_proj_scale_,
                                                input, d_gate_buf, intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        gemv_int8_kernel<<<grid_gate, block>>>(d_up_proj_weight_, d_up_proj_scale_,
                                                input, d_up_buf, intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        dim3 grid_down((hidden_size_ + 255) / 256);
        mlp_down_kernel_int8<<<grid_down, block>>>(d_gate_buf, residual, d_down_proj_weight_,
                                                    d_down_proj_scale_, intermediate_size_, hidden_size_, 1.0f);
        CUDA_CHECK_LAST_KERNEL();

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256, batch_size);
        gemm_int8_kernel<<<grid_gate, block>>>(d_gate_proj_weight_, d_gate_proj_scale_,
                                                input, d_gate_buf, intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        gemm_int8_kernel<<<grid_gate, block>>>(d_up_proj_weight_, d_up_proj_scale_,
                                                input, d_up_buf, intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        dim3 grid_down((hidden_size_ + 255) / 256, batch_size);
        gemm_int8_kernel<<<grid_down, block>>>(d_down_proj_weight_, d_down_proj_scale_,
                                                d_gate_buf, residual, hidden_size_, intermediate_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();
    }
}

} // namespace cuda
} // namespace qwen
