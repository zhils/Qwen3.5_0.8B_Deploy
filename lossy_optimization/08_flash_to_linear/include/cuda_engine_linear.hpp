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
#include "linear_attention_v2.cuh"

namespace qwen {
namespace cuda {

struct CudaLayerConfigLinear {
    int hidden_size = 1024;
    int intermediate_size = 3584;
    int num_heads = 16;
    int key_dim = 128;
    int value_dim = 128;
    int conv_kernel = 4;
};

class CudaLayerLinear {
  public:
    CudaLayerLinear(int layer_idx, const CudaLayerConfigLinear& config);
    ~CudaLayerLinear();

    void set_weights(const std::vector<float>& input_norm_weight,
                     const std::vector<float>& post_norm_weight,
                     const std::vector<float>& mlp_gate_w, const std::vector<float>& mlp_up_w,
                     const std::vector<float>& mlp_down_w,
                     const std::vector<float>& linear_qkv_w, const std::vector<float>& linear_a_w,
                     const std::vector<float>& linear_b_w, const std::vector<float>& linear_z_w,
                     const std::vector<float>& linear_conv_w, const std::vector<float>& linear_out_w);

    void forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                 float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                 CudaLinearAttnState& state) const;

    void forward_batch(const float* d_input, float* d_output, float* d_normed_input_buf,
                       float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                       CudaLinearAttnState& state, int batch_size, cudaStream_t stream = 0) const;

    int layer_idx() const { return layer_idx_; }

  private:
    int layer_idx_;
    CudaLayerConfigLinear config_;

    std::unique_ptr<CudaRMSNorm> input_norm_;
    std::unique_ptr<CudaRMSNorm> post_norm_;
    std::unique_ptr<CudaMLP> mlp_;
    std::unique_ptr<CudaLinearAttentionV2> linear_attn_;
};

class CudaEngineLinear {
  public:
    CudaEngineLinear(int num_layers, int hidden_size, int intermediate_size, int vocab_size);
    ~CudaEngineLinear();

    void set_layer_weights(int layer_idx, const std::vector<float>& weights_flat);
    void set_final_norm_weight(const std::vector<float>& weight);
    void set_lm_head_weight(const std::vector<float>& weight);
    void set_embedding_weight(const std::vector<float>& weight);

    void forward(const float* d_input, float* d_output);
    void forward_batch(const float* d_input, float* d_output, int batch_size, cudaStream_t stream = 0);

    void forward_host(const std::vector<float>& input, std::vector<float>& output);

    std::vector<float> get_output() const;

    void reset_state();

    std::string get_device_info() const;

    size_t gpu_memory_bytes() const { return gpu_memory_bytes_; }
    bool ready() const { return ready_; }

  private:
    int num_layers_;
    int hidden_size_;
    int intermediate_size_;
    int vocab_size_;

    void ensure_batch_buffers(int batch_size);

    std::vector<std::unique_ptr<CudaLayerLinear>> layers_;
    std::unique_ptr<CudaRMSNorm> final_norm_;
    std::unique_ptr<CudaLMHead> lm_head_;
    std::unique_ptr<CudaTokenEmbedding> embedding_;

    std::vector<CudaLinearAttnState> linear_states_;

    float* d_input_buf_;
    float* d_normed_input_;
    float* d_attn_out_;
    float* d_post_normed_;
    float* d_mlp_out_;
    float* d_output_buf_;

    float* d_batch_input_buf_ = nullptr;
    float* d_batch_output_buf_ = nullptr;
    int max_batch_size_ = 0;

    size_t gpu_memory_bytes_;
    bool ready_;

    void allocate_buffers();
    void free_buffers();
};

} // namespace cuda
} // namespace qwen
