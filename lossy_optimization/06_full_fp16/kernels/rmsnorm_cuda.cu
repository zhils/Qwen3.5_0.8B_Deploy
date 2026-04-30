#include "rmsnorm_cuda.hpp"
#include "cuda_error_handling.cuh"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace qwen {
namespace cuda {

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum_rms(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction: first warp reduce, then broadcast
__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];  // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum_rms(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if we are the first warp
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) val = warp_reduce_sum_rms(val);

    return val;
}

// Optimized RMSNorm using warp shuffle and vectorized loads
__global__ void rmsnorm_kernel_optimized(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         const float* __restrict__ weight,
                                         int hidden_size, float eps) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in_ptr = input + batch_idx * hidden_size;
    float* out_ptr = output + batch_idx * hidden_size;

    float partial_sum = 0.0f;

    // Vectorized loads for better memory bandwidth
    const int vec_size = 4;
    const int vec_hidden = hidden_size / vec_size;
    const float4* in_vec = reinterpret_cast<const float4*>(in_ptr);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += blockDim.x) {
        float4 v4 = in_vec[i];
        partial_sum += v4.x * v4.x;
        partial_sum += v4.y * v4.y;
        partial_sum += v4.z * v4.z;
        partial_sum += v4.w * v4.w;
    }

    // Handle remaining elements
    for (int i = vec_hidden * vec_size + tid; i < hidden_size; i += blockDim.x) {
        float v = in_ptr[i];
        partial_sum += v * v;
    }

    // Block-level reduction
    float sum_sq = block_reduce_sum(partial_sum);

    // Broadcast RMS to all threads
    __shared__ float rms_inv;
    if (tid == 0) {
        rms_inv = 1.0f / sqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    // Vectorized write
    float4* out_vec = reinterpret_cast<float4*>(out_ptr);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += blockDim.x) {
        float4 v4 = in_vec[i];
        float4 w4 = w_vec[i];
        float4 o4;
        o4.x = (v4.x * rms_inv) * (1.0f + w4.x);
        o4.y = (v4.y * rms_inv) * (1.0f + w4.y);
        o4.z = (v4.z * rms_inv) * (1.0f + w4.z);
        o4.w = (v4.w * rms_inv) * (1.0f + w4.w);
        out_vec[i] = o4;
    }

    for (int i = vec_hidden * vec_size + tid; i < hidden_size; i += blockDim.x) {
        out_ptr[i] = (in_ptr[i] * rms_inv) * (1.0f + weight[i]);
    }
}

// Optimized RMSNorm with fused residual add
__global__ void rmsnorm_add_residual_kernel_optimized(const float* __restrict__ residual_in,
                                                       const float* __restrict__ attn_out,
                                                       float* __restrict__ residual_out,
                                                       float* __restrict__ normed_out,
                                                       const float* __restrict__ weight,
                                                       int hidden_size, float eps) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in_ptr = residual_in + batch_idx * hidden_size;
    const float* attn_ptr = attn_out + batch_idx * hidden_size;
    float* out_ptr = residual_out + batch_idx * hidden_size;
    float* norm_ptr = normed_out + batch_idx * hidden_size;

    float partial_sum = 0.0f;

    const int vec_size = 4;
    const int vec_hidden = hidden_size / vec_size;

    // Fused: add residual + compute variance
    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += blockDim.x) {
        float4 r4 = reinterpret_cast<const float4*>(in_ptr)[i];
        float4 a4 = reinterpret_cast<const float4*>(attn_ptr)[i];
        float4 sum4;
        sum4.x = r4.x + a4.x;
        sum4.y = r4.y + a4.y;
        sum4.z = r4.z + a4.z;
        sum4.w = r4.w + a4.w;
        reinterpret_cast<float4*>(out_ptr)[i] = sum4;

        partial_sum += sum4.x * sum4.x;
        partial_sum += sum4.y * sum4.y;
        partial_sum += sum4.z * sum4.z;
        partial_sum += sum4.w * sum4.w;
    }

    for (int i = vec_hidden * vec_size + tid; i < hidden_size; i += blockDim.x) {
        float val = in_ptr[i] + attn_ptr[i];
        out_ptr[i] = val;
        partial_sum += val * val;
    }

    float sum_sq = block_reduce_sum(partial_sum);

    __shared__ float rms_inv;
    if (tid == 0) {
        rms_inv = 1.0f / sqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    // Apply RMSNorm
    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += blockDim.x) {
        float4 v4 = reinterpret_cast<const float4*>(out_ptr)[i];
        float4 w4 = reinterpret_cast<const float4*>(weight)[i];
        float4 o4;
        o4.x = (v4.x * rms_inv) * (1.0f + w4.x);
        o4.y = (v4.y * rms_inv) * (1.0f + w4.y);
        o4.z = (v4.z * rms_inv) * (1.0f + w4.z);
        o4.w = (v4.w * rms_inv) * (1.0f + w4.w);
        reinterpret_cast<float4*>(norm_ptr)[i] = o4;
    }

    for (int i = vec_hidden * vec_size + tid; i < hidden_size; i += blockDim.x) {
        norm_ptr[i] = (out_ptr[i] * rms_inv) * (1.0f + weight[i]);
    }
}

// Simple fallback kernels
__global__ void rmsnorm_kernel_simple(const float* __restrict__ input, float* __restrict__ output,
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

__global__ void rmsnorm_add_residual_kernel_simple(const float* __restrict__ residual_in,
                                                    const float* __restrict__ attn_out,
                                                    float* __restrict__ residual_out,
                                                    float* __restrict__ normed_out,
                                                    const float* __restrict__ weight,
                                                    int hidden_size, float eps) {
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

// Backward compatibility wrappers - device function versions for calling from other kernels
__device__ void rmsnorm_kernel_device(const float* __restrict__ input, float* __restrict__ output,
                               const float* __restrict__ weight, int hidden_size, float eps) {
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

    // Use warp shuffle for reduction
    float sum_sq = partial_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float rms_inv;
    if (tid == 0) {
        rms_inv = 1.0f / sqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden_size; i += block_size) {
        out_ptr[i] = (in_ptr[i] * rms_inv) * (1.0f + weight[i]);
    }
}

__global__ void rmsnorm_kernel(const float* __restrict__ input, float* __restrict__ output,
                               const float* __restrict__ weight, int hidden_size, float eps) {
    rmsnorm_kernel_device(input, output, weight, hidden_size, eps);
}

__device__ void rmsnorm_add_residual_kernel_device(const float* __restrict__ residual_in,
                                            const float* __restrict__ attn_out,
                                            float* __restrict__ residual_out,
                                            float* __restrict__ normed_out,
                                            const float* __restrict__ weight,
                                            int hidden_size, float eps) {
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

    float sum_sq = partial_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float rms_inv;
    if (tid == 0) {
        rms_inv = 1.0f / sqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden_size; i += block_size) {
        norm_ptr[i] = (out_ptr[i] * rms_inv) * (1.0f + weight[i]);
    }
}

__global__ void rmsnorm_add_residual_kernel(const float* __restrict__ residual_in,
                                            const float* __restrict__ attn_out,
                                            float* __restrict__ residual_out,
                                            float* __restrict__ normed_out,
                                            const float* __restrict__ weight,
                                            int hidden_size, float eps) {
    rmsnorm_add_residual_kernel_device(residual_in, attn_out, residual_out, normed_out,
                                        weight, hidden_size, eps);
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

    // Check alignment for vectorized loads
    bool aligned = ((uintptr_t)input % 16 == 0) &&
                   ((uintptr_t)output % 16 == 0) &&
                   ((uintptr_t)d_weight_ % 16 == 0) &&
                   (hidden_size_ % 4 == 0);

    if (aligned && hidden_size_ >= 512) {
        rmsnorm_kernel_optimized<<<batch_size, block_size, 0>>>(
            input, output, d_weight_, hidden_size_, eps_);
    } else {
        const int shared_mem = block_size * sizeof(float);
        rmsnorm_kernel_simple<<<batch_size, block_size, shared_mem>>>(
            input, output, d_weight_, hidden_size_, eps_);
    }
    CUDA_CHECK_LAST_KERNEL();
}

void CudaRMSNorm::forward_with_residual(const float* residual_in, const float* attn_out,
                                        float* residual_out, float* normed_out,
                                        int batch_size) const {
    const int block_size = 256;

    bool aligned = ((uintptr_t)residual_in % 16 == 0) &&
                   ((uintptr_t)attn_out % 16 == 0) &&
                   ((uintptr_t)residual_out % 16 == 0) &&
                   ((uintptr_t)normed_out % 16 == 0) &&
                   ((uintptr_t)d_weight_ % 16 == 0) &&
                   (hidden_size_ % 4 == 0);

    if (aligned && hidden_size_ >= 512) {
        rmsnorm_add_residual_kernel_optimized<<<batch_size, block_size, 0>>>(
            residual_in, attn_out, residual_out, normed_out, d_weight_, hidden_size_, eps_);
    } else {
        const int shared_mem = block_size * sizeof(float);
        rmsnorm_add_residual_kernel_simple<<<batch_size, block_size, shared_mem>>>(
            residual_in, attn_out, residual_out, normed_out, d_weight_, hidden_size_, eps_);
    }
    CUDA_CHECK_LAST_KERNEL();
}

} // namespace cuda
} // namespace qwen
