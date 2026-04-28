#include "rmsnorm_cuda.hpp"
#include "cuda_error_handling.cuh"
#include <cmath>
#include <stdexcept>

namespace qwen {
namespace cuda {

__global__ void rmsnorm_kernel(const float* __restrict__ input, float* __restrict__ output,
                               const float* __restrict__ weight, int hidden_size, float eps) {
    extern __shared__ float shared_data[];

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float* in_ptr = input + batch_idx * hidden_size;
    float* out_ptr = output + batch_idx * hidden_size;

    float partial_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += block_size) {
        float val = in_ptr[i];
        partial_sum += val * val;
    }

    shared_data[tid] = partial_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(shared_data[0] / hidden_size + eps);

    for (int i = tid; i < hidden_size; i += block_size) {
        out_ptr[i] = (in_ptr[i] / rms) * (1.0f + weight[i]);
    }
}

__global__ void rmsnorm_add_residual_kernel(const float* __restrict__ residual_in,
                                            const float* __restrict__ attn_out,
                                            float* __restrict__ residual_out,
                                            float* __restrict__ normed_out,
                                            const float* __restrict__ weight, int hidden_size,
                                            float eps) {
    extern __shared__ float shared_data[];

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float* in_ptr = residual_in + batch_idx * hidden_size;
    const float* attn_ptr = attn_out + batch_idx * hidden_size;
    float* out_ptr = residual_out + batch_idx * hidden_size;
    float* norm_ptr = normed_out + batch_idx * hidden_size;

    float partial_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += block_size) {
        float val = in_ptr[i] + attn_ptr[i];
        out_ptr[i] = val;
        partial_sum += val * val;
    }

    shared_data[tid] = partial_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(shared_data[0] / hidden_size + eps);

    for (int i = tid; i < hidden_size; i += block_size) {
        norm_ptr[i] = (out_ptr[i] / rms) * (1.0f + weight[i]);
    }
}

CudaRMSNorm::CudaRMSNorm(int hidden_size, float eps)
    : hidden_size_(hidden_size), eps_(eps), d_weight_(nullptr) {
    cudaMalloc(&d_weight_, hidden_size_ * sizeof(float));
}

CudaRMSNorm::~CudaRMSNorm() {
    if (d_weight_)
        cudaFree(d_weight_);
}

void CudaRMSNorm::set_weights(const std::vector<float>& weight) {
    cudaMemcpy(d_weight_, weight.data(), hidden_size_ * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaRMSNorm::forward(const float* input, float* output, int batch_size) const {
    const int block_size = 256;
    const int shared_mem = block_size * sizeof(float);

    rmsnorm_kernel<<<batch_size, block_size, shared_mem>>>(input, output, d_weight_, hidden_size_,
                                                           eps_);
    CUDA_CHECK_LAST_KERNEL();
}

void CudaRMSNorm::forward_with_residual(const float* residual_in, const float* attn_out,
                                        float* residual_out, float* normed_out,
                                        int batch_size) const {
    const int block_size = 256;
    const int shared_mem = block_size * sizeof(float);

    rmsnorm_add_residual_kernel<<<batch_size, block_size, shared_mem>>>(
        residual_in, attn_out, residual_out, normed_out, d_weight_, hidden_size_, eps_);
    CUDA_CHECK_LAST_KERNEL();
}

} // namespace cuda
} // namespace qwen
