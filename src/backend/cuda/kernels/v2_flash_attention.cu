/**
 * v2.0 Flash Attention v2 Implementation
 *
 * Key improvements over v1:
 * 1. Warp-level parallelism: Each warp processes a subset of KV tiles
 * 2. Reduced HBM traffic: Q stays in registers, KV loaded once per tile
 * 3. Better occupancy: Smaller shared memory footprint per block
 * 4. Online softmax with proper numerical stability
 *
 * Reference: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
 */

#include "flash_attention.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace qwen {
namespace cuda {

// ============================================================================
// Flash Attention v2 - Decode Phase (single query)
// ============================================================================

// Br = number of query rows per block (always 1 for decode)
// Bc = number of key/value columns per tile
// Each warp handles Bc/32 KV positions in parallel for score computation
// Then all warps participate in the reduction and output accumulation

template <int HEAD_DIM, int KV_TILE = 64>
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

    // Load Q into registers (all threads load, then share via shmem)
    float q_reg[HEAD_DIM];
    const float* q_ptr = Q + head_idx * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = q_ptr[d];
    }

    // Shared memory for KV tile and softmax stats
    extern __shared__ float smem[];
    float* s_k = smem;                       // [KV_TILE][HEAD_DIM]
    float* s_v = smem + KV_TILE * HEAD_DIM;  // [KV_TILE][HEAD_DIM]
    float* s_m = smem + 2 * KV_TILE * HEAD_DIM; // [num_warps] - tile max per warp
    float* s_l = s_m + num_warps;            // [num_warps] - tile sum per warp

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_reg[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        o_reg[d] = 0.0f;
    }

    // Iterate over KV sequence in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += KV_TILE) {
        int tile_end = min(tile_start + KV_TILE, seq_len);
        int tile_len = tile_end - tile_start;

        // Load KV tile into shared memory (cooperative)
        for (int i = tid; i < tile_len * HEAD_DIM; i += blockDim.x) {
            int t = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            size_t kv_offset = static_cast<size_t>(tile_start + t) * num_kv_heads * HEAD_DIM +
                               kv_head * HEAD_DIM + d;
            s_k[t * HEAD_DIM + d] = K_cache[kv_offset];
            s_v[t * HEAD_DIM + d] = V_cache[kv_offset];
        }
        __syncthreads();

        // Step 1: Compute attention scores for this tile
        // Each warp handles a subset of KV positions
        float local_m = -FLT_MAX;
        float local_scores[KV_TILE]; // Max 64 elements, but only tile_len valid

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

        // Warp reduce max
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_m = fmaxf(local_m, __shfl_down_sync(0xffffffff, local_m, offset));
        }
        // Broadcast max to all lanes in warp
        local_m = __shfl_sync(0xffffffff, local_m, 0);

        // Store warp max
        if (lane == 0) {
            s_m[warp_id] = local_m;
        }
        __syncthreads();

        // Block reduce max across warps
        float tile_m = -FLT_MAX;
        if (tid < num_warps) {
            tile_m = s_m[tid];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_m = fmaxf(tile_m, __shfl_down_sync(0xffffffff, tile_m, offset));
        }
        tile_m = __shfl_sync(0xffffffff, tile_m, 0);

        // Step 2: Compute exp(score - tile_m) and local sum
        float local_l = 0.0f;
        for (int t = kv_start + lane; t < kv_end; t += 32) {
            float exp_score = expf(local_scores[t - kv_start] - tile_m);
            local_scores[t - kv_start] = exp_score;
            local_l += exp_score;
        }

        // Warp reduce sum
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_l += __shfl_down_sync(0xffffffff, local_l, offset);
        }
        local_l = __shfl_sync(0xffffffff, local_l, 0);

        if (lane == 0) {
            s_l[warp_id] = local_l;
        }
        __syncthreads();

        // Block reduce sum across warps
        float tile_l = 0.0f;
        if (tid < num_warps) {
            tile_l = s_l[tid];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_l += __shfl_down_sync(0xffffffff, tile_l, offset);
        }
        tile_l = __shfl_sync(0xffffffff, tile_l, 0);

        // Step 3: Online softmax update
        float m_new = fmaxf(m_prev, tile_m);
        float exp_old = expf(m_prev - m_new);
        float exp_new = expf(tile_m - m_new);

        // Rescale previous output
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_reg[d] = o_reg[d] * exp_old;
        }
        l_prev = l_prev * exp_old;

        // Accumulate new tile contributions
        // Each warp handles its subset of KV positions
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

    // Final normalization and write output
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

    float* out_ptr = output + head_idx * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out_ptr[d] = o_reg[d] * inv_l;
    }
}

// ============================================================================
// Flash Attention v2 - Prefill Phase (multiple queries)
// ============================================================================

template <int HEAD_DIM, int Q_TILE = 4, int KV_TILE = 64>
__global__ void flash_attn_v2_prefill_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K_cache,
                                             const float* __restrict__ V_cache,
                                             float* __restrict__ output, int num_heads,
                                             int num_kv_heads, int seq_len, float scale) {
    // Each block handles Q_TILE queries
    const int q_tile_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int q_start = q_tile_idx * Q_TILE;
    if (head_idx >= num_heads || q_start >= seq_len)
        return;

    const int kv_head = head_idx * num_kv_heads / num_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const int q_end = min(q_start + Q_TILE, seq_len);
    const int num_q = q_end - q_start;

    // Load Q tile into registers
    float q_reg[Q_TILE][HEAD_DIM];
    for (int q = 0; q < num_q; ++q) {
        const float* q_ptr = Q + (q_start + q) * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_reg[q][d] = q_ptr[d];
        }
    }

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = smem + KV_TILE * HEAD_DIM;

    // Per-query accumulators
    float m_prev[Q_TILE];
    float l_prev[Q_TILE];
    float o_reg[Q_TILE][HEAD_DIM];

    for (int q = 0; q < num_q; ++q) {
        m_prev[q] = -FLT_MAX;
        l_prev[q] = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_reg[q][d] = 0.0f;
        }
    }

    // Iterate over KV tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += KV_TILE) {
        int tile_end = min(tile_start + KV_TILE, seq_len);
        int tile_len = tile_end - tile_start;

        // Load KV tile
        for (int i = tid; i < tile_len * HEAD_DIM; i += blockDim.x) {
            int t = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            size_t kv_offset = static_cast<size_t>(tile_start + t) * num_kv_heads * HEAD_DIM +
                               kv_head * HEAD_DIM + d;
            s_k[t * HEAD_DIM + d] = K_cache[kv_offset];
            s_v[t * HEAD_DIM + d] = V_cache[kv_offset];
        }
        __syncthreads();

        // Process each query in the tile
        for (int q = 0; q < num_q; ++q) {
            // Compute scores for this query against KV tile
            float local_m = -FLT_MAX;
            float local_scores[KV_TILE];

            int kv_per_warp = (tile_len + num_warps - 1) / num_warps;
            int kv_start = warp_id * kv_per_warp;
            int kv_end = min(kv_start + kv_per_warp, tile_len);

            for (int t = kv_start + lane; t < kv_end; t += 32) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q_reg[q][d] * s_k[t * HEAD_DIM + d];
                }
                float score = dot * scale;
                local_scores[t - kv_start] = score;
                local_m = fmaxf(local_m, score);
            }

            // Warp reduce max
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_m = fmaxf(local_m, __shfl_down_sync(0xffffffff, local_m, offset));
            }
            local_m = __shfl_sync(0xffffffff, local_m, 0);

            // Compute exp and sum
            float local_l = 0.0f;
            for (int t = kv_start + lane; t < kv_end; t += 32) {
                float exp_score = expf(local_scores[t - kv_start] - local_m);
                local_scores[t - kv_start] = exp_score;
                local_l += exp_score;
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                local_l += __shfl_down_sync(0xffffffff, local_l, offset);
            }
            local_l = __shfl_sync(0xffffffff, local_l, 0);

            // Online softmax update
            float m_new = fmaxf(m_prev[q], local_m);
            float exp_old = expf(m_prev[q] - m_new);
            float exp_new = expf(local_m - m_new);

            for (int d = 0; d < HEAD_DIM; ++d) {
                o_reg[q][d] = o_reg[q][d] * exp_old;
            }
            l_prev[q] = l_prev[q] * exp_old;

            // Accumulate
            for (int t = kv_start; t < kv_end; ++t) {
                float p_val;
                if (lane == 0) {
                    p_val = local_scores[t - kv_start];
                }
                p_val = __shfl_sync(0xffffffff, p_val, 0);

                for (int d = 0; d < HEAD_DIM; ++d) {
                    o_reg[q][d] += p_val * s_v[t * HEAD_DIM + d];
                }
            }

            l_prev[q] += local_l * exp_new;
            m_prev[q] = m_new;
        }

        __syncthreads();
    }

    // Write output
    for (int q = 0; q < num_q; ++q) {
        float inv_l = (l_prev[q] > 0.0f) ? (1.0f / l_prev[q]) : 0.0f;
        float* out_ptr = output + (q_start + q) * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) {
            out_ptr[d] = o_reg[q][d] * inv_l;
        }
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void flash_attention_v2_decode(const float* d_Q, const float* d_K_cache, const float* d_V_cache,
                               float* d_output, int num_heads, int num_kv_heads, int q_head_dim,
                               int kv_head_dim, int seq_len, float scale) {
    float attn_scale = (scale > 0.0f) ? scale : (1.0f / sqrtf(static_cast<float>(q_head_dim)));

    // Use 128 threads (4 warps) for better occupancy
    int block_size = 128;
    constexpr int KV_TILE = 64;

    int smem_size = 2 * KV_TILE * kv_head_dim * sizeof(float) + 32 * sizeof(float);

    auto launch = [&](auto kernel_instance) {
        cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        kernel_instance<<<num_heads, block_size, smem_size>>>(
            d_Q, d_K_cache, d_V_cache, d_output, num_heads, num_kv_heads, seq_len, attn_scale);
    };

    if (kv_head_dim == 256) {
        launch(flash_attn_v2_decode_kernel<256, KV_TILE>);
    } else if (kv_head_dim == 128) {
        launch(flash_attn_v2_decode_kernel<128, KV_TILE>);
    } else if (kv_head_dim == 64) {
        launch(flash_attn_v2_decode_kernel<64, KV_TILE>);
    } else {
        fprintf(stderr, "flash_attention_v2_decode: unsupported kv_head_dim=%d, fallback to 256\n",
                kv_head_dim);
        launch(flash_attn_v2_decode_kernel<256, KV_TILE>);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attention_v2_decode error: %s\n", cudaGetErrorString(err));
    }
}

void flash_attention_v2_prefill(const float* d_Q, const float* d_K_cache, const float* d_V_cache,
                                float* d_output, int num_heads, int num_kv_heads, int q_head_dim,
                                int kv_head_dim, int seq_len, float scale) {
    float attn_scale = (scale > 0.0f) ? scale : (1.0f / sqrtf(static_cast<float>(q_head_dim)));

    int block_size = 128;
    constexpr int Q_TILE = 4;
    constexpr int KV_TILE = 64;

    int num_q_tiles = (seq_len + Q_TILE - 1) / Q_TILE;
    dim3 grid(num_heads, num_q_tiles);

    int smem_size = 2 * KV_TILE * kv_head_dim * sizeof(float);

    auto launch = [&](auto kernel_instance) {
        cudaFuncSetAttribute(kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
        kernel_instance<<<grid, block_size, smem_size>>>(
            d_Q, d_K_cache, d_V_cache, d_output, num_heads, num_kv_heads, seq_len, attn_scale);
    };

    if (kv_head_dim == 256) {
        launch(flash_attn_v2_prefill_kernel<256, Q_TILE, KV_TILE>);
    } else if (kv_head_dim == 128) {
        launch(flash_attn_v2_prefill_kernel<128, Q_TILE, KV_TILE>);
    } else if (kv_head_dim == 64) {
        launch(flash_attn_v2_prefill_kernel<64, Q_TILE, KV_TILE>);
    } else {
        fprintf(stderr, "flash_attention_v2_prefill: unsupported kv_head_dim=%d, fallback to 256\n",
                kv_head_dim);
        launch(flash_attn_v2_prefill_kernel<256, Q_TILE, KV_TILE>);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attention_v2_prefill error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace qwen
