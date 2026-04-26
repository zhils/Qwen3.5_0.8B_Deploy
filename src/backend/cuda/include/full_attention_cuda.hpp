#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>

namespace qwen {
namespace cuda {

struct CudaKVCache {
    float* d_k_cache;
    float* d_v_cache;
    std::vector<int> layer_lengths;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;

    void reset(int nl, int nkh, int hd, int max_len);
    void clear();
    int length(int layer_idx) const {
        return layer_lengths[layer_idx];
    }
};

class CudaFullAttention {
  public:
    CudaFullAttention(int hidden_size = 1024, int num_heads = 8, int num_kv_heads = 2,
                      int q_head_dim = 256, int kv_head_dim = 256);
    ~CudaFullAttention();

    void
    set_weights(const std::vector<float>& q_proj_weight, const std::vector<float>& k_proj_weight,
                const std::vector<float>& v_proj_weight, const std::vector<float>& q_norm_weight,
                const std::vector<float>& k_norm_weight, const std::vector<float>& o_proj_weight);

    void forward(const float* input, float* output, CudaKVCache& kv_cache, int layer_idx,
                 int position) const;

    /**
     * Batch prefill forward: processes multiple tokens simultaneously.
     * input:  [batch_size, hidden_size] contiguous
     * output: [batch_size, hidden_size] contiguous
     * positions: [batch_size] positions for RoPE
     */
    void forward_batch_prefill(const float* input, float* output, CudaKVCache& kv_cache,
                               int layer_idx, const int* positions, int batch_size) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int num_heads() const {
        return num_heads_;
    }

  private:
    int hidden_size_;
    int num_heads_;
    int num_kv_heads_;
    int q_head_dim_;
    int kv_head_dim_;

    float* d_q_proj_weight_;
    float* d_k_proj_weight_;
    float* d_v_proj_weight_;
    float* d_o_proj_weight_;
    float* d_q_norm_weight_;
    float* d_k_norm_weight_;

    mutable float* d_q_buf_;
    mutable float* d_gate_buf_;
    mutable float* d_k_buf_;
    mutable float* d_v_buf_;
    mutable float* d_attn_out_buf_;
    mutable float* d_attn_scores_buf_;
    mutable int max_seq_len_;

    // Batch prefill buffers (lazily allocated)
    mutable float* d_batch_q_buf_;
    mutable float* d_batch_gate_buf_;
    mutable float* d_batch_k_buf_;
    mutable float* d_batch_v_buf_;
    mutable float* d_batch_attn_out_buf_;
    mutable int max_batch_size_;

    void ensure_batch_buffers(int batch_size) const;
};

} // namespace cuda
} // namespace qwen
