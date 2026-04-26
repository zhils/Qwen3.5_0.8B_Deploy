/**
 * Fused Kernels for v2.0 Performance Optimization
 * 
 * 1. RMSNorm + ResidualAdd Fusion
 * 2. Gate projection + SiLU + Mul Fusion  
 * 3. FlashAttention v2 Prefill
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace qwen {
namespace cuda {

// ============================================================================
// 1. RMSNorm + ResidualAdd Fusion (Optimized)
// ============================================================================

template <int BLOCK_SIZE>
__global__ void fused_rmsnorm_residual_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ attn_out,
    float* __restrict__ residual_out,
    float* __restrict__ normed_out,
    const float* __restrict__ weight,
    int hidden_size,
    float eps) {
    
    __shared__ float s_sum[BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    
    const float* in_ptr = residual_in + batch_idx * hidden_size;
    const float* attn_ptr = attn_out + batch_idx * hidden_size;
    float* out_ptr = residual_out + batch_idx * hidden_size;
    float* norm_ptr = normed_out + batch_idx * hidden_size;
    
    float partial_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = in_ptr[i] + attn_ptr[i];
        out_ptr[i] = val;
        partial_sum += val * val;
    }
    
    s_sum[tid] = partial_sum;
    __syncthreads();
    
#pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(s_sum[0] / hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        norm_ptr[i] = out_ptr[i] * rms * (1.0f + weight[i]);
    }
}

// ============================================================================
// 2. Gate Projection + SiLU + Mul Fusion (Optimized)
// ============================================================================

template <int BLOCK_SIZE, int VECTOR_SIZE = 4>
__global__ void fused_gate_silu_mul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gate_weight,
    const float* __restrict__ up_weight,
    float* __restrict__ hidden,
    int hidden_size,
    int intermediate_size) {
    
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= intermediate_size) return;
    
    float g = 0.0f, u = 0.0f;
    
    const float* g_w = gate_weight + idx * hidden_size;
    const float* u_w = up_weight + idx * hidden_size;
    
#pragma unroll 4
    for (int j = 0; j < hidden_size; ++j) {
        float x = input[j];
        g += g_w[j] * x;
        u += u_w[j] * x;
    }
    
    float silu_g = g / (1.0f + expf(-g));
    hidden[idx] = silu_g * u;
}

// ============================================================================
// 3. FlashAttention v2 Prefill Kernel
// ============================================================================

template <int HEAD_DIM, int KV_TILE>
__global__ void flash_attn_v2_prefill_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int layer_idx,
    int max_seq_len,
    float scale) {
    
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    
    if (head_idx >= num_heads) return;
    
    const int kv_head = head_idx * num_kv_heads / num_heads;
    
    float q_reg[HEAD_DIM];
    const float* q_ptr = Q + static_cast<size_t>(batch_idx) * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    
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
            size_t kv_offset = static_cast<size_t>(layer_idx) * max_seq_len * num_kv_heads * HEAD_DIM +
                               static_cast<size_t>(tile_start + t) * num_kv_heads * HEAD_DIM +
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
    float* out_ptr = output + static_cast<size_t>(batch_idx) * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out_ptr[d] = o_reg[d] * inv_l;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void launch_fused_rmsnorm_residual(
    const float* residual_in,
    const float* attn_out,
    float* residual_out,
    float* normed_out,
    const float* weight,
    int hidden_size,
    int batch_size,
    float eps,
    cudaStream_t stream) {
    
    const int block_size = 256;
    fused_rmsnorm_residual_kernel<block_size><<<batch_size, block_size, block_size * sizeof(float), stream>>>(
        residual_in, attn_out, residual_out, normed_out, weight, hidden_size, eps);
}

void launch_fused_gate_silu_mul(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* hidden,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream) {
    
    const int block_size = 256;
    int grid_size = (intermediate_size + block_size - 1) / block_size;
    fused_gate_silu_mul_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
        input, gate_weight, up_weight, hidden, hidden_size, intermediate_size);
}

void launch_flash_attn_v2_prefill(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int batch_size,
    int layer_idx,
    int max_seq_len,
    float scale,
    cudaStream_t stream) {
    
    const int block_size = 128;
    
    int kv_tile = 64;
    if (head_dim >= 256) {
        kv_tile = 32;
    } else if (head_dim >= 128) {
        kv_tile = 48;
    }
    
    int smem_size = 2 * kv_tile * head_dim * sizeof(float) + 32 * sizeof(float);
    
    dim3 grid(num_heads, batch_size);
    
    auto launch = [&](auto kernel_instance) {
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(kernel_instance,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        kernel_instance<<<grid, block_size, smem_size, stream>>>(
            Q, K_cache, V_cache, output, num_heads, num_kv_heads, seq_len, layer_idx, max_seq_len, scale);
    };
    
    if (head_dim == 256) {
        if (kv_tile == 32) {
            launch(flash_attn_v2_prefill_kernel<256, 32>);
        } else {
            launch(flash_attn_v2_prefill_kernel<256, 48>);
        }
    } else if (head_dim == 128) {
        launch(flash_attn_v2_prefill_kernel<128, 48>);
    } else if (head_dim == 64) {
        launch(flash_attn_v2_prefill_kernel<64, 64>);
    } else {
        launch(flash_attn_v2_prefill_kernel<256, 32>);
    }
}

} // namespace cuda
} // namespace qwen
