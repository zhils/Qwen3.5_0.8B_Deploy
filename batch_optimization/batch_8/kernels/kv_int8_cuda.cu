#include "kv_int8_cuda.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace qwen {
namespace cuda {

__global__ void kv_quantize_kernel(const float* __restrict__ input, int8_t* __restrict__ output,
                                   float* __restrict__ scale, int num_elements) {
    extern __shared__ float s_max[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= num_elements)
        return;

    float val = input[gid];
    float abs_val = fabsf(val);

    s_max[tid] = abs_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float block_max = s_max[0];
    float block_scale = block_max / 127.0f;
    if (block_scale < 1e-8f)
        block_scale = 1.0f;

    if (tid == 0) {
        scale[blockIdx.x] = block_scale;
    }

    output[gid] = static_cast<int8_t>(roundf(val / block_scale));
}

__global__ void kv_dequantize_kernel(const int8_t* __restrict__ input,
                                     const float* __restrict__ scale, float* __restrict__ output,
                                     int num_elements) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements)
        return;

    float s = scale[blockIdx.x];
    output[gid] = static_cast<float>(input[gid]) * s;
}

__global__ void kv_append_int8_kernel(int8_t* __restrict__ k_cache, int8_t* __restrict__ v_cache,
                                      float* __restrict__ k_scale, float* __restrict__ v_scale,
                                      const float* __restrict__ d_k, const float* __restrict__ d_v,
                                      int layer_idx, int position, int num_kv_heads, int head_dim,
                                      int max_seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_size = num_kv_heads * head_dim;

    if (tid >= kv_size)
        return;

    extern __shared__ float s_max_k[];
    float* s_max_v = s_max_k + blockDim.x;

    float k_val = d_k[tid];
    float v_val = d_v[tid];

    s_max_k[threadIdx.x] = fabsf(k_val);
    s_max_v[threadIdx.x] = fabsf(v_val);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max_k[threadIdx.x] = fmaxf(s_max_k[threadIdx.x], s_max_k[threadIdx.x + s]);
            s_max_v[threadIdx.x] = fmaxf(s_max_v[threadIdx.x], s_max_v[threadIdx.x + s]);
        }
        __syncthreads();
    }

    float k_block_max = s_max_k[0];
    float v_block_max = s_max_v[0];
    float k_s = k_block_max / 127.0f;
    float v_s = v_block_max / 127.0f;
    if (k_s < 1e-8f)
        k_s = 1.0f;
    if (v_s < 1e-8f)
        v_s = 1.0f;

    if (threadIdx.x == 0) {
        int scale_idx = layer_idx * max_seq_len + position;
        k_scale[scale_idx] = k_s;
        v_scale[scale_idx] = v_s;
    }

    int cache_offset = layer_idx * max_seq_len * kv_size + position * kv_size + tid;
    k_cache[cache_offset] = static_cast<int8_t>(roundf(k_val / k_s));
    v_cache[cache_offset] = static_cast<int8_t>(roundf(v_val / v_s));
}

__global__ void kv_get_int8_kernel(const int8_t* __restrict__ k_cache,
                                   const int8_t* __restrict__ v_cache,
                                   const float* __restrict__ k_scale,
                                   const float* __restrict__ v_scale, float* __restrict__ d_k_out,
                                   float* __restrict__ d_v_out, int layer_idx, int position,
                                   int num_kv_heads, int head_dim, int max_seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_size = num_kv_heads * head_dim;

    if (tid >= kv_size)
        return;

    int scale_idx = layer_idx * max_seq_len + position;
    float k_s = k_scale[scale_idx];
    float v_s = v_scale[scale_idx];

    int cache_offset = layer_idx * max_seq_len * kv_size + position * kv_size + tid;
    d_k_out[tid] = static_cast<float>(k_cache[cache_offset]) * k_s;
    d_v_out[tid] = static_cast<float>(v_cache[cache_offset]) * v_s;
}

void CudaKVCacheINT8::reset(int nl, int nkh, int hd, int max_len) {
    clear();

    num_layers = nl;
    num_kv_heads = nkh;
    head_dim = hd;
    max_seq_len = max_len;
    layer_lengths.assign(nl, 0);

    size_t kv_size = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t cache_size = static_cast<size_t>(num_layers) * max_seq_len * kv_size;
    size_t scale_size = static_cast<size_t>(num_layers) * max_seq_len;

    cudaMalloc(&d_k_cache, cache_size * sizeof(int8_t));
    cudaMalloc(&d_v_cache, cache_size * sizeof(int8_t));
    cudaMalloc(&d_k_scale, scale_size * sizeof(float));
    cudaMalloc(&d_v_scale, scale_size * sizeof(float));

    cudaMemset(d_k_cache, 0, cache_size * sizeof(int8_t));
    cudaMemset(d_v_cache, 0, cache_size * sizeof(int8_t));
    cudaMemset(d_k_scale, 0, scale_size * sizeof(float));
    cudaMemset(d_v_scale, 0, scale_size * sizeof(float));

    bytes_allocated = cache_size * 2 * sizeof(int8_t) + scale_size * 2 * sizeof(float);
}

void CudaKVCacheINT8::clear() {
    if (d_k_cache) {
        cudaFree(d_k_cache);
        d_k_cache = nullptr;
    }
    if (d_v_cache) {
        cudaFree(d_v_cache);
        d_v_cache = nullptr;
    }
    if (d_k_scale) {
        cudaFree(d_k_scale);
        d_k_scale = nullptr;
    }
    if (d_v_scale) {
        cudaFree(d_v_scale);
        d_v_scale = nullptr;
    }
    bytes_allocated = 0;
}

void kv_quantize_fp32_to_int8(const float* d_input, int8_t* d_output, float* d_scale,
                              int num_elements, cudaStream_t stream) {
    const int block_size = 256;
    const int elements_per_block = block_size;
    int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;

    size_t shared_mem = block_size * sizeof(float);

    kv_quantize_kernel<<<num_blocks, block_size, shared_mem, stream>>>(d_input, d_output, d_scale,
                                                                       num_elements);
}

void kv_dequantize_int8_to_fp32(const int8_t* d_input, const float* d_scale, float* d_output,
                                int num_elements, cudaStream_t stream) {
    const int block_size = 256;
    const int elements_per_block = block_size;
    int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;

    kv_dequantize_kernel<<<num_blocks, block_size, 0, stream>>>(d_input, d_scale, d_output,
                                                                num_elements);
}

void kv_append_int8(CudaKVCacheINT8& cache, const float* d_k, const float* d_v, int layer_idx,
                    cudaStream_t stream) {
    int kv_size = cache.num_kv_heads * cache.head_dim;
    int position = cache.layer_lengths[layer_idx];

    const int block_size = 256;
    int num_blocks = (kv_size + block_size - 1) / block_size;
    size_t shared_mem = block_size * 2 * sizeof(float);

    kv_append_int8_kernel<<<num_blocks, block_size, shared_mem, stream>>>(
        cache.d_k_cache, cache.d_v_cache, cache.d_k_scale, cache.d_v_scale, d_k, d_v, layer_idx,
        position, cache.num_kv_heads, cache.head_dim, cache.max_seq_len);

    cache.layer_lengths[layer_idx]++;
}

void kv_get_int8(const CudaKVCacheINT8& cache, float* d_k_out, float* d_v_out, int layer_idx,
                 cudaStream_t stream) {
    int kv_size = cache.num_kv_heads * cache.head_dim;
    int position = cache.layer_lengths[layer_idx] - 1;
    if (position < 0)
        return;

    const int block_size = 256;
    int num_blocks = (kv_size + block_size - 1) / block_size;

    kv_get_int8_kernel<<<num_blocks, block_size, 0, stream>>>(
        cache.d_k_cache, cache.d_v_cache, cache.d_k_scale, cache.d_v_scale, d_k_out, d_v_out,
        layer_idx, position, cache.num_kv_heads, cache.head_dim, cache.max_seq_len);
}

} // namespace cuda
} // namespace qwen
