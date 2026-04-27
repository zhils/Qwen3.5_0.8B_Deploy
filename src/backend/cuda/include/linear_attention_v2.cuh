#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include "linear_attention_cuda.hpp"

namespace qwen {
namespace cuda {

class CudaLinearAttentionV2 {
  public:
    CudaLinearAttentionV2(int hidden_size = 1024, int num_heads = 16, int key_dim = 128,
                          int value_dim = 128, int conv_kernel = 4);
    ~CudaLinearAttentionV2();

    void set_weights(const std::vector<float>& in_proj_qkv_weight,
                     const std::vector<float>& in_proj_a_weight,
                     const std::vector<float>& in_proj_b_weight,
                     const std::vector<float>& in_proj_z_weight,
                     const std::vector<float>& conv1d_weight,
                     const std::vector<float>& out_proj_weight,
                     const std::vector<float>& a_log = {},
                     const std::vector<float>& dt_bias = {},
                     const std::vector<float>& norm_weight = {});

    void forward(const float* input, float* output, CudaLinearAttnState& state) const;

    void forward_batch(const float* input, float* output, CudaLinearAttnState& state,
                       int batch_size, cudaStream_t stream = 0) const;

    void set_stream(cudaStream_t stream) const;

    int hidden_size() const { return hidden_size_; }
    int num_heads() const { return num_heads_; }

  private:
    int hidden_size_;
    int num_heads_;
    int key_dim_;
    int value_dim_;
    int conv_kernel_;

    float* d_in_proj_qkv_weight_;
    float* d_in_proj_a_weight_;
    float* d_in_proj_b_weight_;
    float* d_in_proj_z_weight_;
    float* d_conv1d_weight_;
    float* d_out_proj_weight_;
    float* d_a_log_;
    float* d_dt_bias_;
    float* d_norm_weight_;

    float* d_mixed_qkv_buf_;
    float* d_conv_out_buf_;
    float* d_a_buf_;
    float* d_b_raw_buf_;
    float* d_attn_out_buf_;
    float* d_z_buf_;

    cublasHandle_t cublas_handle_;
    float* d_cublas_workspace_ = nullptr;

    mutable float* d_batch_mixed_qkv_buf_;
    mutable float* d_batch_conv_out_buf_;
    mutable float* d_batch_a_buf_;
    mutable float* d_batch_b_raw_buf_;
    mutable float* d_batch_z_buf_;
    mutable float* d_batch_attn_out_buf_;
    mutable float* d_batch_conv_state_buf_;
    mutable float* d_scan_g_buf_;
    mutable float* d_scan_b_vec_buf_;
    mutable int max_batch_size_;

    void ensure_batch_buffers(int batch_size) const;
};

} // namespace cuda
} // namespace qwen
