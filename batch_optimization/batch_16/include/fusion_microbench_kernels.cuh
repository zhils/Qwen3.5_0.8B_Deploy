#pragma once

#include <cuda_runtime.h>

namespace qwen {
namespace cuda {
namespace fusion_bench {

void run_fusion1_baseline_q_path(const float* d_input, const float* d_q_weight,
                                 const float* d_q_norm_w, float* d_q, float* d_gate,
                                 int hidden_size, int num_heads, int q_head_dim, int kv_head_dim,
                                 int rotary_dim, float rope_base, int position);

void run_fusion1_fused_q_path(const float* d_input, const float* d_q_weight,
                              const float* d_q_norm_w, float* d_q, float* d_gate, int hidden_size,
                              int num_heads, int q_head_dim, int kv_head_dim, int rotary_dim,
                              float rope_base, int position);

void run_fusion2_baseline_kv_cache(const float* d_input, const float* d_k_w, const float* d_v_w,
                                   const float* d_k_norm_w, float* d_k, float* d_v,
                                   float* d_k_cache, float* d_v_cache, size_t k_offset_elems,
                                   int hidden_size, int num_kv_heads, int kv_head_dim,
                                   int rotary_dim, float rope_base, int position);

void run_fusion2_fused_kv_cache(const float* d_input, const float* d_k_w, const float* d_v_w,
                                const float* d_k_norm_w, float* d_k, float* d_v, float* d_k_cache,
                                float* d_v_cache, size_t k_offset_elems, int hidden_size,
                                int num_kv_heads, int kv_head_dim, int rotary_dim, float rope_base,
                                int position);

void run_fusion3_baseline_attn_core(const float* d_q, const float* k_ptr, const float* v_ptr,
                                    float* d_scores, float* d_attn_out, int num_heads,
                                    int num_kv_heads, int kv_head_dim, int q_head_dim, int seq_len);

void run_fusion3_flash_attn_core(const float* d_q, const float* k_ptr, const float* v_ptr,
                                 float* d_attn_out, int num_heads, int num_kv_heads, int head_dim,
                                 int seq_len);

void run_fusion3_gate_o_fused(float* d_attn_out, const float* d_gate, const float* d_o_weight,
                              float* d_output, int num_heads, int kv_head_dim, int q_head_dim,
                              int total_out, int hidden_size);

void run_fusion3_gate_o_baseline(float* d_attn_out, const float* d_gate, const float* d_o_weight,
                                 float* d_output, int num_heads, int kv_head_dim, int q_head_dim,
                                 int total_out, int hidden_size);

void run_fusion4_rmsnorm_then_linear_head(const float* d_input, const float* d_norm_w,
                                          const float* d_weight_row_major, float* d_out_partial,
                                          float* d_tmp_normed, int hidden_size,
                                          int out_dim_partial);

void run_fusion4_fused_rmsnorm_linear_head(const float* d_input, const float* d_norm_w,
                                           const float* d_weight_row_major, float* d_out_partial,
                                           int hidden_size, int out_dim_partial);

void run_fusion5_mlp_baseline_chain(const float* d_input, const float* d_gate_w,
                                    const float* d_up_w, const float* d_down_w, float* d_gate,
                                    float* d_up, float* d_hidden, float* d_output, int hidden_size,
                                    int intermediate_size);

void run_fusion5_mlp_fused_gate_silu(const float* d_input, const float* d_gate_w,
                                     const float* d_up_w, const float* d_down_w, float* d_hidden,
                                     float* d_output, int hidden_size, int intermediate_size);

void run_fusion6_chain_postnorm_mlp_residual(const float* d_residual_in, const float* d_post_norm_w,
                                             const float* d_gate_w, const float* d_up_w,
                                             const float* d_down_w, float* d_tmp_normed,
                                             float* d_tmp_gate, float* d_tmp_up,
                                             float* d_tmp_hidden, float* d_mlp_out, int hidden_size,
                                             int intermediate_size);

void run_fusion6_fused_postnorm_mlp_residual(const float* d_residual_in, const float* d_post_norm_w,
                                             const float* d_gate_w, const float* d_up_w,
                                             const float* d_down_w, float* d_tmp_normed,
                                             float* d_hidden, float* d_mlp_out, int hidden_size,
                                             int intermediate_size);

void run_fusion7_conv1d_update_baseline(const float* d_mixed, const float* d_conv_w,
                                        float* d_conv_state, float* d_conv_out, int conv_dim,
                                        int conv_kernel);

void run_fusion7_conv1d_update_fused(const float* d_mixed, const float* d_conv_w,
                                     float* d_conv_state, float* d_conv_out, int conv_dim,
                                     int conv_kernel);

void run_fusion7_l2norm_qk_baseline(float* d_q, float* d_k, int num_heads, int key_dim,
                                    float q_scale);

void run_fusion7_l2norm_qk_fused(float* d_q, float* d_k, int num_heads, int key_dim, float q_scale);

} // namespace fusion_bench
} // namespace cuda
} // namespace qwen
