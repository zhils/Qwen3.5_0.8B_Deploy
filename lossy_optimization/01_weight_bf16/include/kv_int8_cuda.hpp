#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>

namespace qwen {
namespace cuda {

struct CudaKVCacheINT8 {
    int8_t* d_k_cache;
    int8_t* d_v_cache;
    float* d_k_scale;
    float* d_v_scale;
    std::vector<int> layer_lengths;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    size_t bytes_allocated;

    void reset(int nl, int nkh, int hd, int max_len);
    void clear();
    int length(int layer_idx) const {
        return layer_lengths[layer_idx];
    }
    size_t memory_bytes() const {
        return bytes_allocated;
    }
};

void kv_quantize_fp32_to_int8(const float* d_input, int8_t* d_output, float* d_scale,
                              int num_elements, cudaStream_t stream = 0);

void kv_dequantize_int8_to_fp32(const int8_t* d_input, const float* d_scale, float* d_output,
                                int num_elements, cudaStream_t stream = 0);

void kv_append_int8(CudaKVCacheINT8& cache, const float* d_k, const float* d_v, int layer_idx,
                    cudaStream_t stream = 0);

void kv_get_int8(const CudaKVCacheINT8& cache, float* d_k_out, float* d_v_out, int layer_idx,
                 cudaStream_t stream = 0);

} // namespace cuda
} // namespace qwen
