#pragma once

#include <cuda_runtime.h>

namespace qwen {
namespace cuda {

/**
 * Flash Attention v2 for the decode phase (single query attending to KV cache).
 *
 * Uses tiled computation with online softmax to avoid materializing the full
 * N x N attention matrix. Memory complexity is O(N) instead of O(N^2).
 *
 * Supports GQA: num_heads Q heads share num_kv_heads KV heads.
 *
 * @param d_Q        Query vectors [num_heads, head_dim] on device
 * @param d_K_cache  Key cache [seq_len, num_kv_heads, head_dim] on device
 * @param d_V_cache  Value cache [seq_len, num_kv_heads, head_dim] on device
 * @param d_output   Output [num_heads, head_dim] on device
 * @param num_heads  Number of query attention heads
 * @param num_kv_heads Number of key/value heads (GQA)
 * @param head_dim   Dimension per head (supports 64, 128, 256)
 * @param seq_len    Current cached sequence length
 */
void flash_attention_decode(const float* d_Q, const float* d_K_cache, const float* d_V_cache,
                            float* d_output, int num_heads, int num_kv_heads, int q_head_dim,
                            int kv_head_dim, int seq_len, float scale = 0.0f);

} // namespace cuda
} // namespace qwen
