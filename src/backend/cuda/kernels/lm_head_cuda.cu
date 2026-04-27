#include "lm_head_cuda.hpp"
#include "cuda_error_handling.cuh"
#include <stdexcept>
#include <string>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error(std::string("cuBLAS error ") +                                \
                                     std::to_string(static_cast<int>(_err)) + " at " + __FILE__ +  \
                                     ":" + std::to_string(__LINE__));                             \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (call);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) +      \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                                          \
    } while (0)

namespace qwen {
namespace cuda {

__global__ void fp32_to_bf16_kernel(const float* __restrict__ fp32_data,
                                    __nv_bfloat16* __restrict__ bf16_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    bf16_data[i] = __float2bfloat16(fp32_data[i]);
}

__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ bf16_data,
                                    float* __restrict__ fp32_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    fp32_data[i] = __bfloat162float(bf16_data[i]);
}

CudaLMHead::CudaLMHead(int hidden_size, int vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size),
      d_weight_bf16_(nullptr), d_input_bf16_(nullptr), d_output_bf16_(nullptr) {
    size_t weight_size = static_cast<size_t>(vocab_size_) * hidden_size_;
    CUDA_CHECK(cudaMalloc(&d_weight_bf16_, weight_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_input_bf16_, hidden_size_ * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_output_bf16_, vocab_size_ * sizeof(__nv_bfloat16)));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    weight_converted_ = false;
}

CudaLMHead::~CudaLMHead() {
    if (d_weight_bf16_)
        cudaFree(d_weight_bf16_);
    if (d_input_bf16_)
        cudaFree(d_input_bf16_);
    if (d_output_bf16_)
        cudaFree(d_output_bf16_);
    if (cublas_handle_)
        cublasDestroy(cublas_handle_);
}

void CudaLMHead::set_weight(const std::vector<float>& weight) {
    size_t weight_size = static_cast<size_t>(vocab_size_) * hidden_size_;
    if (weight.size() != weight_size) {
        throw std::invalid_argument("CudaLMHead weight size mismatch: expected " +
                                    std::to_string(weight_size) + ", got " +
                                    std::to_string(weight.size()));
    }

    float* d_temp_fp32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_fp32, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_temp_fp32, weight.data(), weight_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((weight_size + 255) / 256);
    fp32_to_bf16_kernel<<<grid, block>>>(d_temp_fp32, d_weight_bf16_, weight_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_temp_fp32));

    weight_converted_ = true;
}

void CudaLMHead::forward(const float* input, float* output) const {
    dim3 block(256);

    dim3 grid_in((hidden_size_ + 255) / 256);
    fp32_to_bf16_kernel<<<grid_in, block>>>(input, d_input_bf16_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, vocab_size_, 1,
                              hidden_size_, &alpha, static_cast<const void*>(d_weight_bf16_),
                              CUDA_R_16BF, hidden_size_, static_cast<const void*>(d_input_bf16_),
                              CUDA_R_16BF, hidden_size_, &beta, static_cast<void*>(d_output_bf16_),
                              CUDA_R_16BF, vocab_size_, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    dim3 grid_out((vocab_size_ + 255) / 256);
    bf16_to_fp32_kernel<<<grid_out, block>>>(d_output_bf16_, output, vocab_size_);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace qwen
