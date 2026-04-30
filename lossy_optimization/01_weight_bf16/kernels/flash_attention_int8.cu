/**
 * INT8-aware Flash Attention Implementation
 *
 * Key innovation: Dequantization is fused into the attention computation,
 * eliminating the need for intermediate FP32 KV buffer.
 *
 * Memory bandwidth savings:
 * - Original: Read INT8 KV -> Write FP32 KV -> Read FP32 KV -> Compute
 * - Fused:    Read INT8 KV + Scale -> Compute (dequantize on-the-fly)
 *
 * This reduces KV memory traffic by ~2x (INT8 + scale vs FP32).
 */

#include "flash_attention_int8.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace qwen {
namespace cuda {

static constexpr int INT8_BLOCK_KV = 32;

template <int Q_DIM, int KV_DIM>
__global__ void
flash_attn_decode_int8_kernel(const float* __restrict__ Q, const int8_t* __restrict__ K_cache,
                              const int8_t* __restrict__ V_cache, const float* __restrict__ K_scale,
                              const float* __restrict__ V_scale, float* __restrict__ output,
                              int num_heads, int num_kv_heads, int seq_len, int layer_idx,
                              int max_seq_len, float scale) {
    const int head_idx = blockIdx.x;
    if (head_idx >= num_heads)
        return;

    const int kv_head = head_idx * num_kv_heads / num_heads;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_k = smem + Q_DIM;
    float* s_v = s_k + INT8_BLOCK_KV * KV_DIM;

    for (int d = tid; d < Q_DIM; d += blockDim.x) {
        s_q[d] = Q[head_idx * Q_DIM + d];
    }
    __syncthreads();

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[KV_DIM];
#pragma unroll
    for (int d = 0; d < KV_DIM; ++d)
        acc[d] = 0.0f;

    const int kv_size = num_kv_heads * KV_DIM;
    const int layer_offset = layer_idx * max_seq_len;

    for (int tile_start = 0; tile_start < seq_len; tile_start += INT8_BLOCK_KV) {
        int tile_end = min(tile_start + INT8_BLOCK_KV, seq_len);
        int tile_len = tile_end - tile_start;

        for (int i = tid; i < tile_len * KV_DIM; i += blockDim.x) {
            int t = i / KV_DIM;
            int d = i % KV_DIM;
            int pos = tile_start + t;

            int kv_offset =
                layer_idx * max_seq_len * kv_size + pos * kv_size + kv_head * KV_DIM + d;
            int scale_idx = layer_offset + pos;

            float k_s = K_scale[scale_idx];
            float v_s = V_scale[scale_idx];

            s_k[t * KV_DIM + d] = static_cast<float>(K_cache[kv_offset]) * k_s;
            s_v[t * KV_DIM + d] = static_cast<float>(V_cache[kv_offset]) * v_s;
        }
        __syncthreads();

        if (tid == 0) {
            float m_tile = -FLT_MAX;
            float scores[INT8_BLOCK_KV];

            for (int t = 0; t < tile_len; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < KV_DIM; ++d) {
                    dot += s_q[d] * s_k[t * KV_DIM + d];
                }
                scores[t] = dot * scale;
                m_tile = fmaxf(m_tile, scores[t]);
            }

            float m_new = fmaxf(m_prev, m_tile);
            float correction = expf(m_prev - m_new);

            for (int d = 0; d < KV_DIM; ++d) {
                acc[d] *= correction;
            }
            l_prev *= correction;

            for (int t = 0; t < tile_len; ++t) {
                float p = expf(scores[t] - m_new);
                l_prev += p;
                for (int d = 0; d < KV_DIM; ++d) {
                    acc[d] += p * s_v[t * KV_DIM + d];
                }
            }

            m_prev = m_new;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (int d = 0; d < KV_DIM; ++d) {
            output[head_idx * KV_DIM + d] = acc[d] * inv_l;
        }
    }
}

template <int Q_DIM, int KV_DIM>
__global__ void flash_attn_decode_int8_optimized_kernel(
    const float* __restrict__ Q, const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache, const float* __restrict__ K_scale,
    const float* __restrict__ V_scale, float* __restrict__ output, int num_heads, int num_kv_heads,
    int seq_len, int layer_idx, int max_seq_len, float scale) {
    const int head_idx = blockIdx.x;
    if (head_idx >= num_heads)
        return;

    const int kv_head = head_idx * num_kv_heads / num_heads;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_k = smem + Q_DIM;
    float* s_v = s_k + INT8_BLOCK_KV * KV_DIM;
    float* s_scales = s_v + INT8_BLOCK_KV * KV_DIM;

    for (int d = tid; d < Q_DIM; d += blockDim.x) {
        s_q[d] = Q[head_idx * Q_DIM + d];
    }
    __syncthreads();

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[KV_DIM];
#pragma unroll
    for (int d = 0; d < KV_DIM; ++d)
        acc[d] = 0.0f;

    const int kv_size = num_kv_heads * KV_DIM;
    const int layer_offset = layer_idx * max_seq_len;

    for (int tile_start = 0; tile_start < seq_len; tile_start += INT8_BLOCK_KV) {
        int tile_end = min(tile_start + INT8_BLOCK_KV, seq_len);
        int tile_len = tile_end - tile_start;

        for (int i = tid; i < tile_len; i += blockDim.x) {
            int pos = tile_start + i;
            int scale_idx = layer_offset + pos;
            s_scales[i * 2] = K_scale[scale_idx];
            s_scales[i * 2 + 1] = V_scale[scale_idx];
        }
        __syncthreads();

        for (int i = tid; i < tile_len * KV_DIM; i += blockDim.x) {
            int t = i / KV_DIM;
            int d = i % KV_DIM;
            int pos = tile_start + t;

            int kv_offset =
                layer_idx * max_seq_len * kv_size + pos * kv_size + kv_head * KV_DIM + d;

            float k_s = s_scales[t * 2];
            float v_s = s_scales[t * 2 + 1];

            s_k[t * KV_DIM + d] = static_cast<float>(K_cache[kv_offset]) * k_s;
            s_v[t * KV_DIM + d] = static_cast<float>(V_cache[kv_offset]) * v_s;
        }
        __syncthreads();

        if (warp_id == 0) {
            float m_tile = -FLT_MAX;
            float scores[INT8_BLOCK_KV];

            for (int t = lane_id; t < tile_len; t += 32) {
                float dot = 0.0f;
                for (int d = 0; d < KV_DIM; ++d) {
                    dot += s_q[d] * s_k[t * KV_DIM + d];
                }
                scores[t] = dot * scale;
            }

            for (int t = lane_id; t < tile_len; t += 32) {
                m_tile = fmaxf(m_tile, scores[t]);
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                m_tile = fmaxf(m_tile, __shfl_down_sync(0xffffffff, m_tile, offset));
            }
            m_tile = __shfl_sync(0xffffffff, m_tile, 0);

            float m_new = fmaxf(m_prev, m_tile);
            float correction = expf(m_prev - m_new);

            for (int d = 0; d < KV_DIM; ++d) {
                acc[d] *= correction;
            }
            l_prev *= correction;

            for (int t = lane_id; t < tile_len; t += 32) {
                float p = expf(scores[t] - m_new);
                for (int d = 0; d < KV_DIM; ++d) {
                    atomicAdd(&acc[d], p * s_v[t * KV_DIM + d]);
                }
            }

            m_prev = m_new;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_l = l_prev;
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_l += __shfl_down_sync(0xffffffff, total_l, offset);
        }
        float inv_l = (total_l > 0.0f) ? (1.0f / total_l) : 0.0f;

        for (int d = 0; d < KV_DIM; ++d) {
            output[head_idx * KV_DIM + d] = acc[d] * inv_l;
        }
    }
}

void flash_attention_decode_int8(const float* d_Q, const int8_t* d_K_cache, const int8_t* d_V_cache,
                                 const float* d_K_scale, const float* d_V_scale, float* d_output,
                                 int num_heads, int num_kv_heads, int q_head_dim, int kv_head_dim,
                                 int seq_len, int layer_idx, int max_seq_len, float scale) {
    float attn_scale = (scale > 0.0f) ? scale : (1.0f / sqrtf(static_cast<float>(q_head_dim)));
    int smem_size =
        (q_head_dim + 2 * INT8_BLOCK_KV * kv_head_dim + INT8_BLOCK_KV * 2) * sizeof(float);
    int block_size = 32;

    auto launch = [&](auto kernel_instance) {
        cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        kernel_instance<<<num_heads, block_size, smem_size>>>(
            d_Q, d_K_cache, d_V_cache, d_K_scale, d_V_scale, d_output, num_heads, num_kv_heads,
            seq_len, layer_idx, max_seq_len, attn_scale);
    };

    if (q_head_dim == 256 && kv_head_dim == 256) {
        launch(flash_attn_decode_int8_kernel<256, 256>);
    } else if (q_head_dim == 256 && kv_head_dim == 128) {
        launch(flash_attn_decode_int8_kernel<256, 128>);
    } else if (q_head_dim == 128 && kv_head_dim == 128) {
        launch(flash_attn_decode_int8_kernel<128, 128>);
    } else if (q_head_dim == 64 && kv_head_dim == 32) {
        launch(flash_attn_decode_int8_kernel<64, 32>);
    } else if (q_head_dim == 64 && kv_head_dim == 64) {
        launch(flash_attn_decode_int8_kernel<64, 64>);
    } else if (q_head_dim == 32 && kv_head_dim == 32) {
        launch(flash_attn_decode_int8_kernel<32, 32>);
    } else {
        fprintf(stderr,
                "flash_attention_decode_int8: unsupported q_head_dim=%d kv_head_dim=%d, fallback "
                "to 256/256\n",
                q_head_dim, kv_head_dim);
        launch(flash_attn_decode_int8_kernel<256, 256>);
    }
}

void flash_attention_decode_int8_batched(const float* d_Q, const int8_t* d_K_cache,
                                         const int8_t* d_V_cache, const float* d_K_scale,
                                         const float* d_V_scale, float* d_output, int num_heads,
                                         int num_kv_heads, int q_head_dim, int kv_head_dim,
                                         int seq_len, int layer_idx, int max_seq_len, float scale) {
    flash_attention_decode_int8(d_Q, d_K_cache, d_V_cache, d_K_scale, d_V_scale, d_output,
                                num_heads, num_kv_heads, q_head_dim, kv_head_dim, seq_len,
                                layer_idx, max_seq_len, scale);
}

} // namespace cuda
} // namespace qwen
