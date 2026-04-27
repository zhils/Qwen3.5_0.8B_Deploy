#include "mlp_cuda.hpp"
#include "cuda_error_handling.cuh"
#include "fused_kernels.cuh"
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

__global__ void mlp_down_kernel(const float* __restrict__ hidden, float* __restrict__ output,
                                const float* __restrict__ down_weight, int intermediate_size,
                                int hidden_size, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size)
        return;

    float sum = 0.0f;
    for (int j = 0; j < intermediate_size; ++j) {
        sum += down_weight[i * intermediate_size + j] * hidden[j];
    }
    output[i] = sum + beta * output[i];
}

CudaMLP::CudaMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      d_gate_proj_weight_(nullptr), d_up_proj_weight_(nullptr), d_down_proj_weight_(nullptr),
      cublas_handle_(nullptr), d_hidden_buf_(nullptr), max_hidden_batch_(0) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMalloc(&d_gate_proj_weight_, gate_size * sizeof(float));
    cudaMalloc(&d_up_proj_weight_, gate_size * sizeof(float));
    cudaMalloc(&d_down_proj_weight_, down_size * sizeof(float));

    MLP_CUBLAS_CHECK(cublasCreate(&cublas_handle_));

    cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH);
}

CudaMLP::~CudaMLP() {
    if (d_gate_proj_weight_)
        cudaFree(d_gate_proj_weight_);
    if (d_up_proj_weight_)
        cudaFree(d_up_proj_weight_);
    if (d_down_proj_weight_)
        cudaFree(d_down_proj_weight_);
    if (cublas_handle_)
        cublasDestroy(cublas_handle_);
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
    // Need space for gate + up + hidden (all [batch_size, intermediate_size])
    size_t bytes = static_cast<size_t>(batch_size) * intermediate_size_ * sizeof(float) * 3;
    cudaMalloc(&d_hidden_buf_, bytes);
    max_hidden_batch_ = batch_size;
}

void CudaMLP::set_weights(const std::vector<float>& gate_proj_weight,
                          const std::vector<float>& up_proj_weight,
                          const std::vector<float>& down_proj_weight) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMemcpy(d_gate_proj_weight_, gate_proj_weight.data(), gate_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_weight_, up_proj_weight.data(), gate_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_weight_, down_proj_weight.data(), down_size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void CudaMLP::forward(const float* input, float* output, int batch_size) const {
    // Always use cuBLAS GEMM path for better performance and Tensor Core utilization
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        const float alpha = 1.0f, beta = 0.0f;

        // gate projection: [1, hidden] × [hidden, inter] → [1, inter]
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, 1, hidden_size_,
                                     &alpha, d_gate_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_gate_buf, intermediate_size_));

        // up projection: [1, hidden] × [hidden, inter] → [1, inter]
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, 1, hidden_size_,
                                     &alpha, d_up_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_up_buf, intermediate_size_));

        // SiLU(gate) * up
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        // down projection: [1, inter] × [inter, hidden] → [1, hidden]
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_size_, 1, intermediate_size_,
                                     &alpha, d_down_proj_weight_, intermediate_size_,
                                     d_gate_buf, intermediate_size_,
                                     &beta, output, hidden_size_));

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        // Batch MLP using cuBLAS GEMM for all projections
        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        const float alpha = 1.0f, beta = 0.0f;

        // gate projection
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, batch_size, hidden_size_,
                                     &alpha, d_gate_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_gate_buf, intermediate_size_));

        // up projection
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, batch_size, hidden_size_,
                                     &alpha, d_up_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_up_buf, intermediate_size_));

        // Element-wise SiLU(gate) * up
        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_hidden_buf_, total_elements);

        // down projection with residual add
        const float alpha_down = 1.0f, beta_down = 1.0f;
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_size_, batch_size, intermediate_size_,
                                     &alpha_down, d_down_proj_weight_, intermediate_size_,
                                     d_hidden_buf_, intermediate_size_,
                                     &beta_down, output, hidden_size_));
    }
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    // Unified path: always use cuBLAS GEMM for all batch sizes
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        const float alpha = 1.0f, beta = 0.0f;

        // gate projection: [1, hidden] × [hidden, inter] → [1, inter]
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, 1, hidden_size_,
                                     &alpha, d_gate_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_gate_buf, intermediate_size_));

        // up projection: [1, hidden] × [hidden, inter] → [1, inter]
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, 1, hidden_size_,
                                     &alpha, d_up_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_up_buf, intermediate_size_));

        // SiLU(gate) * up
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        // down projection with residual: [1, inter] × [inter, hidden] → [1, hidden]
        const float alpha_down = 1.0f, beta_down = 1.0f;
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_size_, 1, intermediate_size_,
                                     &alpha_down, d_down_proj_weight_, intermediate_size_,
                                     d_gate_buf, intermediate_size_,
                                     &beta_down, residual, hidden_size_));

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        // Batch MLP using cuBLAS GEMM for all projections
        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        const float alpha = 1.0f, beta = 0.0f;

        // gate projection
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, batch_size, hidden_size_,
                                     &alpha, d_gate_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_gate_buf, intermediate_size_));

        // up projection
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     intermediate_size_, batch_size, hidden_size_,
                                     &alpha, d_up_proj_weight_, hidden_size_,
                                     input, hidden_size_,
                                     &beta, d_up_buf, intermediate_size_));

        // Element-wise SiLU(gate) * up
        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_hidden_buf_, total_elements);

        // down projection with residual add
        const float alpha_down = 1.0f, beta_down = 1.0f;
        MLP_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_size_, batch_size, intermediate_size_,
                                     &alpha_down, d_down_proj_weight_, intermediate_size_,
                                     d_hidden_buf_, intermediate_size_,
                                     &beta_down, residual, hidden_size_));
    }
}

} // namespace cuda
} // namespace qwen
