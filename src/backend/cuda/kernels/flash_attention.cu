/**
 * Flash Attention v2 implementation for Qwen3.5 GQA (Grouped Query Attention).
 *
 * Core idea: tile the KV sequence into blocks, compute partial softmax
 * in shared memory, and accumulate the output without materializing the
 * full attention matrix in HBM. This gives O(N) memory instead of O(N^2).
 *
 * v2 improvements:
 * - Warp-level parallelism: Each warp processes a subset of KV tiles
 * - Reduced HBM traffic: Q stays in registers, KV loaded once per tile
 * - Better occupancy: Smaller shared memory footprint per block
 *
 * Reference: Dao, "FlashAttention-2: Faster Attention with Better Parallelism
 * and Work Partitioning" (2023).
 */

#include "flash_attention.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace qwen {
namespace cuda {

// ============================================================================
// Flash Attention v2 - Decode Phase (warp-level parallelism)
// ============================================================================

template <int HEAD_DIM, int KV_TILE>
__global__ void flash_attn_v2_decode_kernel(const float* __restrict__ Q,
                                            const float* __restrict__ K_cache,
                                            const float* __restrict__ V_cache,
                                            float* __restrict__ output, int num_heads,
                                            int num_kv_heads, int seq_len, float scale) {
    const int head_idx = blockIdx.x;
    if (head_idx >= num_heads)
        return;

    const int kv_head = head_idx * num_kv_heads / num_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float q_reg[HEAD_DIM];
    const float* q_ptr = Q + head_idx * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = q_ptr[d];
    }

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = smem + KV_TILE * HEAD_DIM;
    float* s_m = smem + 2 * KV_TILE * HEAD_DIM;
    float* s_l = s_m + num_warps;

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_reg[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        o_reg[d] = 0.0f;
    }

    for (int tile_start = 0; tile_start < seq_len; tile_start += KV_TILE) {
        int tile_end = min(tile_start + KV_TILE, seq_len);
        int tile_len = tile_end - tile_start;

        for (int i = tid; i < tile_len * HEAD_DIM; i += blockDim.x) {
            int t = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            size_t kv_offset = static_cast<size_t>(tile_start + t) * num_kv_heads * HEAD_DIM +
                               kv_head * HEAD_DIM + d;
            s_k[t * HEAD_DIM + d] = K_cache[kv_offset];
            s_v[t * HEAD_DIM + d] = V_cache[kv_offset];
        }
        __syncthreads();

        float local_m = -FLT_MAX;
        float local_scores[KV_TILE];

        int kv_per_warp = (tile_len + num_warps - 1) / num_warps;
        int kv_start = warp_id * kv_per_warp;
        int kv_end = min(kv_start + kv_per_warp, tile_len);

        for (int t = kv_start + lane; t < kv_end; t += 32) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += q_reg[d] * s_k[t * HEAD_DIM + d];
            }
            float score = dot * scale;
            local_scores[t - kv_start] = score;
            local_m = fmaxf(local_m, score);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            local_m = fmaxf(local_m, __shfl_down_sync(0xffffffff, local_m, offset));
        }
        local_m = __shfl_sync(0xffffffff, local_m, 0);

        if (lane == 0) {
            s_m[warp_id] = local_m;
        }
        __syncthreads();

        float tile_m = -FLT_MAX;
        if (tid < num_warps) {
            tile_m = s_m[tid];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_m = fmaxf(tile_m, __shfl_down_sync(0xffffffff, tile_m, offset));
        }
        tile_m = __shfl_sync(0xffffffff, tile_m, 0);

        float local_l = 0.0f;
        for (int t = kv_start + lane; t < kv_end; t += 32) {
            float exp_score = expf(local_scores[t - kv_start] - tile_m);
            local_scores[t - kv_start] = exp_score;
            local_l += exp_score;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            local_l += __shfl_down_sync(0xffffffff, local_l, offset);
        }
        local_l = __shfl_sync(0xffffffff, local_l, 0);

        if (lane == 0) {
            s_l[warp_id] = local_l;
        }
        __syncthreads();

        float tile_l = 0.0f;
        if (tid < num_warps) {
            tile_l = s_l[tid];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_l += __shfl_down_sync(0xffffffff, tile_l, offset);
        }
        tile_l = __shfl_sync(0xffffffff, tile_l, 0);

        float m_new = fmaxf(m_prev, tile_m);
        float exp_old = expf(m_prev - m_new);
        float exp_new = expf(tile_m - m_new);

#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_reg[d] = o_reg[d] * exp_old;
        }
        l_prev = l_prev * exp_old;

        for (int t = kv_start; t < kv_end; ++t) {
            float p_val;
            if (lane == 0) {
                p_val = local_scores[t - kv_start];
            }
            p_val = __shfl_sync(0xffffffff, p_val, 0);

#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                o_reg[d] += p_val * s_v[t * HEAD_DIM + d];
            }
        }

        l_prev += tile_l * exp_new;
        m_prev = m_new;

        __syncthreads();
    }

    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    float* out_ptr = output + head_idx * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out_ptr[d] = o_reg[d] * inv_l;
    }
}

// ============================================================================
// Launch Wrapper
// ============================================================================

void flash_attention_decode(const float* d_Q, const float* d_K_cache, const float* d_V_cache,
                            float* d_output, int num_heads, int num_kv_heads, int q_head_dim,
                            int kv_head_dim, int seq_len, float scale) {
    float attn_scale = (scale > 0.0f) ? scale : (1.0f / sqrtf(static_cast<float>(q_head_dim)));

    int block_size = 128;

    int actual_kv_tile = 64;
    if (kv_head_dim >= 256) {
        actual_kv_tile = 32;
    } else if (kv_head_dim >= 128) {
        actual_kv_tile = 48;
    }
    int smem_size = 2 * actual_kv_tile * kv_head_dim * sizeof(float) + 32 * sizeof(float);

    auto launch = [&](auto kernel_instance) {
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(kernel_instance,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        kernel_instance<<<num_heads, block_size, smem_size>>>(
            d_Q, d_K_cache, d_V_cache, d_output, num_heads, num_kv_heads, seq_len, attn_scale);
    };

    if (kv_head_dim == 256) {
        launch(flash_attn_v2_decode_kernel<256, 32>);
    } else if (kv_head_dim == 128) {
        launch(flash_attn_v2_decode_kernel<128, 48>);
    } else if (kv_head_dim == 64) {
        launch(flash_attn_v2_decode_kernel<64, 64>);
    } else {
        launch(flash_attn_v2_decode_kernel<256, 32>);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attention_decode error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace qwen
