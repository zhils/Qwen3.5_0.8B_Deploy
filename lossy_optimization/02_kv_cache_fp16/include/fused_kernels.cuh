#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qwen {
namespace cuda {

void launch_fused_rmsnorm_residual(
    const float* residual_in,
    const float* attn_out,
    float* residual_out,
    float* normed_out,
    const float* weight,
    int hidden_size,
    int batch_size,
    float eps = 1e-6f,
    cudaStream_t stream = 0);

void launch_fused_gate_silu_mul(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* hidden,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream = 0);

void launch_silu_mul_batch(
    const float* gate,
    const float* up,
    float* hidden,
    int n,
    cudaStream_t stream = 0);

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
    cudaStream_t stream = 0);

void launch_flash_attn_v2_prefill_fp16_cache(
    const float* Q,
    const half* K_cache,
    const half* V_cache,
    float* output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int batch_size,
    int layer_idx,
    int max_seq_len,
    float scale,
    cudaStream_t stream = 0);

} // namespace cuda
} // namespace qwen
