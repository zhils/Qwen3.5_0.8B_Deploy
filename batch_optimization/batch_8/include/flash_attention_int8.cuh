#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {
namespace cuda {

/**
 * INT8-aware Flash Attention for the decode phase.
 *
 * This kernel directly reads INT8 KV cache with per-position scales,
 * performing dequantization on-the-fly during attention computation.
 *
 * Benefits:
 * - Eliminates intermediate FP32 buffer for KV cache
 * - Reduces memory bandwidth (INT8 vs FP32 = 4x less data transfer)
 * - Fused dequantization saves kernel launch overhead
 *
 * Memory layout:
 * - K_cache: [num_layers, max_seq_len, num_kv_heads, head_dim] as int8
 * - V_cache: [num_layers, max_seq_len, num_kv_heads, head_dim] as int8
 * - K_scale: [num_layers, max_seq_len] as float (per-position scale)
 * - V_scale: [num_layers, max_seq_len] as float (per-position scale)
 *
 * @param d_Q        Query vectors [num_heads, q_head_dim] in FP32
 * @param d_K_cache  Key cache in INT8
 * @param d_V_cache  Value cache in INT8
 * @param d_K_scale  Per-position K scales
 * @param d_V_scale  Per-position V scales
 * @param d_output   Output [num_heads, kv_head_dim] in FP32
 * @param num_heads  Number of query heads
 * @param num_kv_heads Number of KV heads (GQA)
 * @param q_head_dim Query head dimension
 * @param kv_head_dim KV head dimension
 * @param seq_len    Current sequence length
 * @param layer_idx  Current layer index (for scale lookup)
 * @param max_seq_len Maximum sequence length (for stride calculation)
 * @param scale      Attention scale (default: 1/sqrt(head_dim))
 */
void flash_attention_decode_int8(const float* d_Q, const int8_t* d_K_cache, const int8_t* d_V_cache,
                                 const float* d_K_scale, const float* d_V_scale, float* d_output,
                                 int num_heads, int num_kv_heads, int q_head_dim, int kv_head_dim,
                                 int seq_len, int layer_idx, int max_seq_len, float scale = 0.0f);

/**
 * Batched INT8-aware Flash Attention for multiple layers.
 *
 * @param d_Q        Query vectors [num_heads, q_head_dim]
 * @param d_K_cache  Key cache for all layers
 * @param d_V_cache  Value cache for all layers
 * @param d_K_scale  Per-position K scales for all layers
 * @param d_V_scale  Per-position V scales for all layers
 * @param d_output   Output
 * @param num_heads  Number of query heads
 * @param num_kv_heads Number of KV heads
 * @param q_head_dim Query head dimension
 * @param kv_head_dim KV head dimension
 * @param seq_len    Current sequence length
 * @param layer_idx  Current layer index
 * @param max_seq_len Maximum sequence length
 * @param scale      Attention scale
 */
void flash_attention_decode_int8_batched(const float* d_Q, const int8_t* d_K_cache,
                                         const int8_t* d_V_cache, const float* d_K_scale,
                                         const float* d_V_scale, float* d_output, int num_heads,
                                         int num_kv_heads, int q_head_dim, int kv_head_dim,
                                         int seq_len, int layer_idx, int max_seq_len,
                                         float scale = 0.0f);

} // namespace cuda
} // namespace qwen
