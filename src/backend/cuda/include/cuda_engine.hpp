#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>

#include "rmsnorm_cuda.hpp"
#include "mlp_cuda.hpp"
#include "lm_head_cuda.hpp"
#include "full_attention_cuda.hpp"
#include "linear_attention_cuda.hpp"

namespace qwen {
namespace cuda {

class StreamManager;

struct CudaLayerConfig {
    int hidden_size = 1024;
    int intermediate_size = 3584;
    bool is_linear = true;

    int linear_num_heads = 16;
    int num_kv_heads = 2;
    int key_dim = 128;
    int value_dim = 128;
    int conv_kernel = 4;
    int full_num_heads = 8;
    int q_head_dim = 256;
    int kv_head_dim = 256;
};

class CudaLayer {
  public:
    CudaLayer(int layer_idx, const CudaLayerConfig& config);
    ~CudaLayer();

    void set_weights(const std::vector<float>& input_norm_weight,
                     const std::vector<float>& post_norm_weight,
                     const std::vector<float>& mlp_gate_w, const std::vector<float>& mlp_up_w,
                     const std::vector<float>& mlp_down_w, bool is_linear,
                     const std::vector<float>& lin_qkv_w, const std::vector<float>& lin_a_w,
                     const std::vector<float>& lin_b_w, const std::vector<float>& lin_z_w,
                     const std::vector<float>& lin_conv_w, const std::vector<float>& lin_out_w,
                     const std::vector<float>& lin_a_log, const std::vector<float>& lin_dt_bias,
                     const std::vector<float>& lin_norm_w, const std::vector<float>& full_q_w,
                     const std::vector<float>& full_k_w, const std::vector<float>& full_v_w,
                     const std::vector<float>& full_qn_w, const std::vector<float>& full_kn_w,
                     const std::vector<float>& full_o_w);

    void forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                 float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                 CudaKVCache& kv_cache, CudaLinearAttnState& lin_state, int position) const;

    void forward_batch_prefill(const float* d_input, float* d_output, float* d_normed_input_buf,
                               float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                               CudaKVCache& kv_cache, CudaLinearAttnState& lin_state,
                               const int* positions, int batch_size, int max_seq = 0) const;

    int layer_idx() const {
        return layer_idx_;
    }
    bool is_linear() const {
        return is_linear_;
    }

  private:
    int layer_idx_;
    CudaLayerConfig config_;
    bool is_linear_;

    std::unique_ptr<CudaRMSNorm> input_norm_;
    std::unique_ptr<CudaRMSNorm> post_norm_;
    std::unique_ptr<CudaMLP> mlp_;
    std::unique_ptr<CudaLinearAttention> linear_attn_;
    std::unique_ptr<CudaFullAttention> full_attn_;
};

class CudaEngine {
  public:
    CudaEngine(int num_layers, int hidden_size, int intermediate_size, int vocab_size,
               int max_seq_len);
    ~CudaEngine();

    void set_layer_weights(int layer_idx, const std::vector<float>& weights_flat);
    void set_final_norm_weight(const std::vector<float>& weight);
    void set_lm_head_weight(const std::vector<float>& weight);

    void forward(const float* d_input, float* d_output, int position);

    void forward_batch_prefill(const float* d_input, float* d_output, const int* positions,
                               int batch_size);

    void forward_batch_prefill_graph(const float* d_input, float* d_output, const int* positions,
                                     int batch_size);

    void forward_host(const std::vector<float>& input, std::vector<float>& output, int position);

    std::vector<float> get_output() const;

    void reset_cache();

    std::string get_device_info() const;

    size_t gpu_memory_bytes() const {
        return gpu_memory_bytes_;
    }
    bool ready() const {
        return ready_;
    }

  private:
    int num_layers_;
    int hidden_size_;
    int intermediate_size_;
    int vocab_size_;
    int max_seq_len_;

    void ensure_batch_buffers(int batch_size);

    std::vector<std::unique_ptr<CudaLayer>> layers_;
    std::unique_ptr<CudaRMSNorm> final_norm_;
    std::unique_ptr<CudaLMHead> lm_head_;

    CudaKVCache kv_cache_;
    std::vector<CudaLinearAttnState> linear_states_;

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

    // CUDA Graph for prefill
    cudaGraph_t prefill_graph_ = nullptr;
    cudaGraphExec_t prefill_graph_exec_ = nullptr;
    int prefill_graph_batch_size_ = 0;
    bool prefill_graph_captured_ = false;

    size_t gpu_memory_bytes_;
    bool ready_;

    std::unique_ptr<StreamManager> stream_manager_;

    void allocate_buffers();
    void free_buffers();
};

void set_fa_weights(CudaFullAttention* fa, const std::vector<float>& q_w,
                    const std::vector<float>& k_w, const std::vector<float>& v_w,
                    const std::vector<float>& qn_w, const std::vector<float>& kn_w,
                    const std::vector<float>& o_w, int layer_idx);

} // namespace cuda
} // namespace qwen
