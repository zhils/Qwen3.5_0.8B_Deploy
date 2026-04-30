#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cuda_bf16.h>

namespace qwen {
namespace cuda {

struct CudaLinearAttnState {
    float* d_recurrent_state;
    float* d_conv_state;
    int num_heads;
    int key_dim;
    int value_dim;
    int conv_dim;
    int conv_kernel;

    void reset(int nh, int kd, int vd, int conv_k = 4);
    void clear();
};

class CudaLinearAttention {
  public:
    CudaLinearAttention(int hidden_size = 1024, int num_heads = 16, int key_dim = 128,
                        int value_dim = 128, int conv_kernel = 4);
    ~CudaLinearAttention();

    void set_weights(const std::vector<float>& in_proj_qkv_weight,
                     const std::vector<float>& in_proj_a_weight,
                     const std::vector<float>& in_proj_b_weight,
                     const std::vector<float>& in_proj_z_weight,
                     const std::vector<float>& conv1d_weight,
                     const std::vector<float>& out_proj_weight,
                     const std::vector<float>& a_log = {}, const std::vector<float>& dt_bias = {},
                     const std::vector<float>& norm_weight = {});

    void forward(const float* input, float* output, CudaLinearAttnState& state) const;

    void forward_batch(const float* input, float* output, CudaLinearAttnState& state,
                       int batch_size) const;

    void forward_fused(const float* input, float* output, CudaLinearAttnState& state) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int num_heads() const {
        return num_heads_;
    }

  private:
    int hidden_size_;
    int num_heads_;
    int key_dim_;
    int value_dim_;
    int conv_kernel_;

    __nv_bfloat16* d_in_proj_qkv_weight_;
    __nv_bfloat16* d_in_proj_a_weight_;
    __nv_bfloat16* d_in_proj_b_weight_;
    __nv_bfloat16* d_in_proj_z_weight_;
    __nv_bfloat16* d_conv1d_weight_;
    __nv_bfloat16* d_out_proj_weight_;
    float* d_a_log_;
    float* d_dt_bias_;
    float* d_norm_weight_;
    float* d_in_proj_z_buf_;

    mutable float* d_mixed_qkv_buf_;
    mutable float* d_conv_out_buf_;
    mutable float* d_q_buf_;
    mutable float* d_k_buf_;
    mutable float* d_v_buf_;
    mutable float* d_a_buf_;
    mutable float* d_b_raw_buf_;
    mutable float* d_attn_out_buf_;
    mutable float* d_z_buf_;

    mutable float* d_batch_mixed_qkv_buf_;
    mutable float* d_batch_conv_out_buf_;
    mutable float* d_batch_a_buf_;
    mutable float* d_batch_b_raw_buf_;
    mutable float* d_batch_z_buf_;
    mutable float* d_batch_attn_out_buf_;
    mutable int max_batch_size_;

    void ensure_batch_buffers(int batch_size) const;
};

} // namespace cuda
} // namespace qwen
