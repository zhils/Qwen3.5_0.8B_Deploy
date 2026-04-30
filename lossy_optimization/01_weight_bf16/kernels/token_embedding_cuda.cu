#include "token_embedding_cuda.hpp"
#include "cuda_error_handling.cuh"
#include <stdexcept>

namespace qwen {
namespace cuda {

__global__ void embedding_lookup_kernel(float* __restrict__ output,
                                        const float* __restrict__ weight, int token_id,
                                        int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size)
        return;
    output[idx] = weight[static_cast<size_t>(token_id) * hidden_size + idx];
}

__global__ void embedding_lookup_bf16_kernel(float* __restrict__ output,
                                              const __nv_bfloat16* __restrict__ weight,
                                              int token_id, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size)
        return;
    __nv_bfloat16 val = weight[static_cast<size_t>(token_id) * hidden_size + idx];
    output[idx] = __bfloat162float(val);
}

__global__ void embedding_lookup_batch_kernel(float* __restrict__ output,
                                              const float* __restrict__ weight,
                                              const int* __restrict__ token_ids, int num_tokens,
                                              int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_size;
    if (idx >= total)
        return;

    int token_idx = idx / hidden_size;
    int dim_idx = idx % hidden_size;
    int token_id = token_ids[token_idx];

    output[idx] = weight[static_cast<size_t>(token_id) * hidden_size + dim_idx];
}

__global__ void embedding_lookup_batch_bf16_kernel(float* __restrict__ output,
                                                    const __nv_bfloat16* __restrict__ weight,
                                                    const int* __restrict__ token_ids,
                                                    int num_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_size;
    if (idx >= total)
        return;

    int token_idx = idx / hidden_size;
    int dim_idx = idx % hidden_size;
    int token_id = token_ids[token_idx];

    __nv_bfloat16 val = weight[static_cast<size_t>(token_id) * hidden_size + dim_idx];
    output[idx] = __bfloat162float(val);
}

CudaTokenEmbedding::CudaTokenEmbedding(int vocab_size, int hidden_size)
    : vocab_size_(vocab_size), hidden_size_(hidden_size), d_weight_fp32_(nullptr),
      d_weight_bf16_(nullptr), owns_weight_(false) {
}

CudaTokenEmbedding::~CudaTokenEmbedding() {
    if (owns_weight_) {
        if (d_weight_fp32_)
            cudaFree(d_weight_fp32_);
        if (d_weight_bf16_)
            cudaFree(d_weight_bf16_);
    }
}

void CudaTokenEmbedding::set_weight(const std::vector<float>& weight) {
    size_t expected = static_cast<size_t>(vocab_size_) * hidden_size_;
    if (weight.size() != expected) {
        throw std::invalid_argument("CudaTokenEmbedding::set_weight: size mismatch");
    }
    
    if (owns_weight_ && d_weight_fp32_) {
        cudaFree(d_weight_fp32_);
    }
    
    cudaError_t err = cudaMalloc(&d_weight_fp32_, expected * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CudaTokenEmbedding: Failed to allocate GPU memory for weights");
    }
    cudaMemcpy(d_weight_fp32_, weight.data(), expected * sizeof(float), cudaMemcpyHostToDevice);
    d_weight_bf16_ = nullptr;
    owns_weight_ = true;
}

void CudaTokenEmbedding::set_weight_bf16_ptr(__nv_bfloat16* d_weight_bf16) {
    if (owns_weight_) {
        if (d_weight_fp32_)
            cudaFree(d_weight_fp32_);
        if (d_weight_bf16_)
            cudaFree(d_weight_bf16_);
        owns_weight_ = false;
    }
    d_weight_bf16_ = d_weight_bf16;
    d_weight_fp32_ = nullptr;
}

void CudaTokenEmbedding::set_weight_fp32_ptr(float* d_weight_fp32) {
    if (owns_weight_) {
        if (d_weight_fp32_)
            cudaFree(d_weight_fp32_);
        if (d_weight_bf16_)
            cudaFree(d_weight_bf16_);
        owns_weight_ = false;
    }
    d_weight_fp32_ = d_weight_fp32;
    d_weight_bf16_ = nullptr;
}

void CudaTokenEmbedding::forward(int token_id, float* d_output) const {
    dim3 block(256);
    dim3 grid((hidden_size_ + 255) / 256);
    
    if (d_weight_bf16_) {
        embedding_lookup_bf16_kernel<<<grid, block>>>(d_output, d_weight_bf16_, token_id, hidden_size_);
    } else if (d_weight_fp32_) {
        embedding_lookup_kernel<<<grid, block>>>(d_output, d_weight_fp32_, token_id, hidden_size_);
    } else {
        throw std::runtime_error("CudaTokenEmbedding: weight not set");
    }
    CUDA_CHECK_LAST_KERNEL();
}

void CudaTokenEmbedding::forward(const std::vector<int>& token_ids, float* d_output) const {
    int num_tokens = static_cast<int>(token_ids.size());

    int* d_token_ids;
    cudaMalloc(&d_token_ids, num_tokens * sizeof(int));
    cudaMemcpy(d_token_ids, token_ids.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);
    int total = num_tokens * hidden_size_;
    dim3 grid((total + 255) / 256);
    
    if (d_weight_bf16_) {
        embedding_lookup_batch_bf16_kernel<<<grid, block>>>(d_output, d_weight_bf16_, d_token_ids,
                                                            num_tokens, hidden_size_);
    } else if (d_weight_fp32_) {
        embedding_lookup_batch_kernel<<<grid, block>>>(d_output, d_weight_fp32_, d_token_ids,
                                                       num_tokens, hidden_size_);
    } else {
        cudaFree(d_token_ids);
        throw std::runtime_error("CudaTokenEmbedding: weight not set");
    }
    CUDA_CHECK_LAST_KERNEL();

    cudaFree(d_token_ids);
}

} // namespace cuda
} // namespace qwen
