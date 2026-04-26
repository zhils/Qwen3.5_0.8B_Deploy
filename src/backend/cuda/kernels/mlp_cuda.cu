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
      cublas_handle_(nullptr) {
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
    float* d_hidden;
    cudaMalloc(&d_hidden, intermediate_size_ * sizeof(float));

    launch_fused_gate_silu_mul(input, d_gate_proj_weight_, d_up_proj_weight_,
                               d_hidden, hidden_size_, intermediate_size_);
    CUDA_CHECK_LAST_KERNEL();

    const float alpha = 1.0f, beta = 0.0f;
    MLP_CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T, intermediate_size_, hidden_size_,
                                 &alpha, d_down_proj_weight_, intermediate_size_, d_hidden, 1,
                                 &beta, output, 1));

    cudaFree(d_hidden);
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    float* d_hidden;
    cudaMalloc(&d_hidden, intermediate_size_ * sizeof(float));

    launch_fused_gate_silu_mul(input, d_gate_proj_weight_, d_up_proj_weight_,
                               d_hidden, hidden_size_, intermediate_size_);
    CUDA_CHECK_LAST_KERNEL();

    const float alpha = 1.0f, beta = 1.0f;
    MLP_CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T, intermediate_size_, hidden_size_,
                                 &alpha, d_down_proj_weight_, intermediate_size_, d_hidden, 1,
                                 &beta, residual, 1));

    cudaFree(d_hidden);
}

} // namespace cuda
} // namespace qwen
