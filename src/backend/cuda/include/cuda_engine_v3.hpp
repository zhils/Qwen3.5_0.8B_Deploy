#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <memory>
#include <string>

#include "rmsnorm_cuda.hpp"
#include "mlp_cuda.hpp"
#include "lm_head_cuda.hpp"
#include "token_embedding_cuda.hpp"
#include "full_attention_cuda.hpp"

namespace qwen {
namespace cuda {

struct CudaLayerConfigV3 {
    int hidden_size = 1024;
    int intermediate_size = 3584;

    int num_heads = 8;
    int num_kv_heads = 2;
    int q_head_dim = 256;
    int kv_head_dim = 256;
};

class CudaLayerV3 {
  public:
    CudaLayerV3(int layer_idx, const CudaLayerConfigV3& config);
    ~CudaLayerV3();

    void set_weights(const std::vector<float>& input_norm_weight,
                     const std::vector<float>& post_norm_weight,
                     const std::vector<float>& mlp_gate_w, const std::vector<float>& mlp_up_w,
                     const std::vector<float>& mlp_down_w,
                     const std::vector<float>& full_q_w, const std::vector<float>& full_k_w,
                     const std::vector<float>& full_v_w, const std::vector<float>& full_qn_w,
                     const std::vector<float>& full_kn_w, const std::vector<float>& full_o_w);

    void forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                 float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                 CudaKVCache& kv_cache, int position) const;

    void forward_batch_prefill(const float* d_input, float* d_output, float* d_normed_input_buf,
                               float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                               CudaKVCache& kv_cache, const int* positions, int batch_size) const;

    int layer_idx() const { return layer_idx_; }

  private:
    int layer_idx_;
    CudaLayerConfigV3 config_;

    std::unique_ptr<CudaRMSNorm> input_norm_;
    std::unique_ptr<CudaRMSNorm> post_norm_;
    std::unique_ptr<CudaMLP> mlp_;
    std::unique_ptr<CudaFullAttention> full_attn_;
};

class CudaEngineV3 {
  public:
    CudaEngineV3(int num_layers, int hidden_size, int intermediate_size, int vocab_size,
                 int max_seq_len);
    ~CudaEngineV3();

    void set_layer_weights(int layer_idx, const std::vector<float>& weights_flat);
    void set_final_norm_weight(const std::vector<float>& weight);
    void set_lm_head_weight(const std::vector<float>& weight);
    
    void set_shared_embedding_lmhead_weight(const std::vector<float>& weight);
    
    void set_embedding_weight(const std::vector<float>& weight);

    void forward(const float* d_input, float* d_output, int position);

    void forward_batch_prefill(const float* d_input, float* d_output, const int* positions,
                               int batch_size);
    
    void forward_token(int token_id, float* d_output, int position);
    
    void forward_tokens(const std::vector<int>& token_ids, float* d_output, const int* positions);

    void forward_host(const std::vector<float>& input, std::vector<float>& output, int position);

    std::vector<float> get_output() const;

    void reset_cache();

    std::string get_device_info() const;

    size_t gpu_memory_bytes() const { return gpu_memory_bytes_; }
    bool ready() const { return ready_; }
    
    CudaTokenEmbedding* embedding() { return embedding_.get(); }

  private:
    int num_layers_;
    int hidden_size_;
    int intermediate_size_;
    int vocab_size_;
    int max_seq_len_;

    void ensure_batch_buffers(int batch_size);

    std::vector<std::unique_ptr<CudaLayerV3>> layers_;
    std::unique_ptr<CudaRMSNorm> final_norm_;
    std::unique_ptr<CudaLMHead> lm_head_;
    std::unique_ptr<CudaTokenEmbedding> embedding_;

    CudaKVCache kv_cache_;
    
    __nv_bfloat16* d_shared_embedding_lmhead_weight_;

    float* d_input_buf_;
    float* d_normed_input_;
    float* d_attn_out_;
    float* d_post_normed_;
    float* d_mlp_out_;
    float* d_residual_;
    float* d_output_buf_;
    float* d_lmhead_out_;

    float* d_batch_input_buf_ = nullptr;
    float* d_batch_output_buf_ = nullptr;
    int* d_positions_buf_ = nullptr;
    int max_batch_size_ = 0;

    // Batch prefill intermediate buffers (lazily allocated)
    mutable float* d_batch_normed_input_ = nullptr;
    mutable float* d_batch_attn_out_ = nullptr;
    mutable float* d_batch_post_normed_ = nullptr;
    mutable float* d_batch_mlp_out_ = nullptr;
    mutable int max_batch_intermediate_size_ = 0;

    size_t gpu_memory_bytes_;
    bool ready_;

    void allocate_buffers();
    void free_buffers();
    void ensure_batch_intermediate_buffers(int batch_size) const;
};

} // namespace cuda
} // namespace qwen
