#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace qwen {
namespace cuda {

// Warp-level GEMV using Tensor Core-like operations
// Optimized for decode phase: batch=1, small M, large K/N
// Uses float accumulation for numerical stability

// Forward declarations for CUTLASS-style warp-level GEMV
// These kernels use warp shuffle for reduction instead of shared memory
// Much faster for small output dimensions typical in LLM decode

// Warp-level dot product using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Vectorized load helpers
__device__ __forceinline__ void load_float4(const float* ptr, float4& out) {
    out = *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, const float4& val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Optimized GEMV: each warp computes one output row
// Uses vectorized loads and warp shuffle reduction
// Best for: decode phase where M=1, K=hidden_size, N=out_dim
template <int WARP_SIZE = 32, int VEC_SIZE = 4>
__global__ void cutlass_style_gemv_kernel(
    const float* __restrict__ weight,  // [N, K] row-major
    const float* __restrict__ input,   // [K]
    float* __restrict__ output,        // [N]
    int N, int K) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * num_warps + warp_id;

    if (row >= N) return;

    const float* w_row = weight + row * K;
    float sum = 0.0f;

    // Vectorized dot product
    const int vec_k = K / VEC_SIZE;
    const float4* w_vec = reinterpret_cast<const float4*>(w_row);
    const float4* i_vec = reinterpret_cast<const float4*>(input);

    #pragma unroll 8
    for (int k = lane_id; k < vec_k; k += WARP_SIZE) {
        float4 w4 = w_vec[k];
        float4 i4 = i_vec[k];
        sum += w4.x * i4.x;
        sum += w4.y * i4.y;
        sum += w4.z * i4.z;
        sum += w4.w * i4.w;
    }

    // Handle remaining elements
    int remainder_start = vec_k * VEC_SIZE;
    for (int k = remainder_start + lane_id; k < K; k += WARP_SIZE) {
        sum += w_row[k] * input[k];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[row] = sum;
    }
}

// BF16 version for memory bandwidth reduction
template <int WARP_SIZE = 32>
__global__ void cutlass_style_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight,  // [N, K] row-major
    const __nv_bfloat16* __restrict__ input,   // [K]
    float* __restrict__ output,                // [N]
    int N, int K) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * num_warps + warp_id;

    if (row >= N) return;

    const __nv_bfloat16* w_row = weight + row * K;
    float sum = 0.0f;

    // Process 2 BF16 elements at a time (32-bit loads)
    const int pair_k = K / 2;
    const int2* w_pair = reinterpret_cast<const int2*>(w_row);
    const int2* i_pair = reinterpret_cast<const int2*>(input);

    #pragma unroll 8
    for (int k = lane_id; k < pair_k; k += WARP_SIZE) {
        int2 w2 = w_pair[k];
        int2 i2 = i_pair[k];

        __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162*>(&w2.x);
        __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162*>(&w2.y);
        __nv_bfloat162 i_lo = *reinterpret_cast<__nv_bfloat162*>(&i2.x);
        __nv_bfloat162 i_hi = *reinterpret_cast<__nv_bfloat162*>(&i2.y);

        sum += __bfloat162float(w_lo.x) * __bfloat162float(i_lo.x);
        sum += __bfloat162float(w_lo.y) * __bfloat162float(i_lo.y);
        sum += __bfloat162float(w_hi.x) * __bfloat162float(i_hi.x);
        sum += __bfloat162float(w_hi.y) * __bfloat162float(i_hi.y);
    }

    // Handle remaining elements
    for (int k = (pair_k * 2) + lane_id; k < K; k += WARP_SIZE) {
        sum += __bfloat162float(w_row[k]) * __bfloat162float(input[k]);
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[row] = sum;
    }
}

// GEMV with fused bias add and activation (SiLU)
// For MLP gate_proj + up_proj fusion
template <int WARP_SIZE = 32, int VEC_SIZE = 4>
__global__ void cutlass_style_gemv_fused_silu_kernel(
    const float* __restrict__ gate_weight,  // [N, K]
    const float* __restrict__ up_weight,    // [N, K]
    const float* __restrict__ input,        // [K]
    float* __restrict__ output,             // [N]
    int N, int K) {

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * num_warps + warp_id;

    if (row >= N) return;

    const float* g_row = gate_weight + row * K;
    const float* u_row = up_weight + row * K;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    // Vectorized dot product for both weights
    const int vec_k = K / VEC_SIZE;
    const float4* g_vec = reinterpret_cast<const float4*>(g_row);
    const float4* u_vec = reinterpret_cast<const float4*>(u_row);
    const float4* i_vec = reinterpret_cast<const float4*>(input);

    #pragma unroll 8
    for (int k = lane_id; k < vec_k; k += WARP_SIZE) {
        float4 g4 = g_vec[k];
        float4 u4 = u_vec[k];
        float4 i4 = i_vec[k];

        gate_sum += g4.x * i4.x;
        gate_sum += g4.y * i4.y;
        gate_sum += g4.z * i4.z;
        gate_sum += g4.w * i4.w;

        up_sum += u4.x * i4.x;
        up_sum += u4.y * i4.y;
        up_sum += u4.z * i4.z;
        up_sum += u4.w * i4.w;
    }

    // Handle remaining elements
    int remainder_start = vec_k * VEC_SIZE;
    for (int k = remainder_start + lane_id; k < K; k += WARP_SIZE) {
        gate_sum += g_row[k] * input[k];
        up_sum += u_row[k] * input[k];
    }

    // Warp-level reduction
    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);

    if (lane_id == 0) {
        // SiLU(gate) * up
        float silu = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = silu * up_sum;
    }
}

// Launcher functions
inline void launch_cutlass_gemv(
    const float* weight, const float* input, float* output,
    int N, int K, cudaStream_t stream = 0) {

    const int WARP_SIZE = 32;
    const int warps_per_block = 8;  // 256 threads
    const int block_size = warps_per_block * WARP_SIZE;

    // Check alignment for vectorized loads
    bool aligned = ((uintptr_t)weight % 16 == 0) &&
                   ((uintptr_t)input % 16 == 0) &&
                   (K % 4 == 0);

    if (aligned) {
        int grid = (N + warps_per_block - 1) / warps_per_block;
        cutlass_style_gemv_kernel<32, 4><<<grid, block_size, 0, stream>>>(
            weight, input, output, N, K);
    } else {
        // Fallback to scalar version
        int grid = (N + warps_per_block - 1) / warps_per_block;
        cutlass_style_gemv_kernel<32, 1><<<grid, block_size, 0, stream>>>(
            weight, input, output, N, K);
    }
}

inline void launch_cutlass_gemv_bf16(
    const __nv_bfloat16* weight, const __nv_bfloat16* input, float* output,
    int N, int K, cudaStream_t stream = 0) {

    const int WARP_SIZE = 32;
    const int warps_per_block = 8;
    const int block_size = warps_per_block * WARP_SIZE;
    int grid = (N + warps_per_block - 1) / warps_per_block;

    cutlass_style_gemv_bf16_kernel<32><<<grid, block_size, 0, stream>>>(
        weight, input, output, N, K);
}

inline void launch_cutlass_gemv_fused_silu(
    const float* gate_weight, const float* up_weight,
    const float* input, float* output,
    int N, int K, cudaStream_t stream = 0) {

    const int WARP_SIZE = 32;
    const int warps_per_block = 8;
    const int block_size = warps_per_block * WARP_SIZE;

    bool aligned = ((uintptr_t)gate_weight % 16 == 0) &&
                   ((uintptr_t)up_weight % 16 == 0) &&
                   ((uintptr_t)input % 16 == 0) &&
                   (K % 4 == 0);

    int grid = (N + warps_per_block - 1) / warps_per_block;

    if (aligned) {
        cutlass_style_gemv_fused_silu_kernel<32, 4><<<grid, block_size, 0, stream>>>(
            gate_weight, up_weight, input, output, N, K);
    } else {
        cutlass_style_gemv_fused_silu_kernel<32, 1><<<grid, block_size, 0, stream>>>(
            gate_weight, up_weight, input, output, N, K);
    }
}

} // namespace cuda
} // namespace qwen
