#pragma once

#include "language_common.hpp"
#include "language_mlp.hpp"
#include "language_linear_attn.hpp"
#include "language_full_attn.hpp"
#include <memory>
#include <vector>

namespace qwen {

struct LanguageLayerWeights {
    std::vector<float> input_layernorm_weight;
    std::vector<float> post_attention_layernorm_weight;

    std::vector<float> mlp_gate_proj_weight;
    std::vector<float> mlp_up_proj_weight;
    std::vector<float> mlp_down_proj_weight;

    bool is_linear = true;

    std::vector<float> linear_in_proj_qkv_weight;
    std::vector<float> linear_in_proj_a_weight;
    std::vector<float> linear_in_proj_b_weight;
    std::vector<float> linear_in_proj_z_weight;
    std::vector<float> linear_conv1d_weight;
    std::vector<float> linear_A_log;
    std::vector<float> linear_dt_bias;
    std::vector<float> linear_norm_weight;
    std::vector<float> linear_out_proj_weight;

    std::vector<float> full_q_proj_weight;
    std::vector<float> full_k_proj_weight;
    std::vector<float> full_v_proj_weight;
    std::vector<float> full_q_norm_weight;
    std::vector<float> full_k_norm_weight;
    std::vector<float> full_o_proj_weight;
};

class LanguageLayer {
  public:
    LanguageLayer(int layer_idx, int hidden_size = 1024, int intermediate_size = 3584);

    void set_weights(const LanguageLayerWeights& weights);

    std::vector<float> forward(const std::vector<float>& input, LinearAttnState& linear_state,
                               KVCache& kv_cache, int position) const;

    std::vector<float> forward_sequence(const std::vector<float>& input, int seq_len,
                                        LinearAttnState& linear_state, KVCache& kv_cache) const;

    int layer_idx() const {
        return layer_idx_;
    }
    int hidden_size() const {
        return hidden_size_;
    }
    bool is_linear() const {
        return is_linear_;
    }

    std::vector<float> input_layernorm_forward(const std::vector<float>& input) const;
    std::vector<float> attention_forward(const std::vector<float>& input,
                                         LinearAttnState& linear_state, KVCache& kv_cache,
                                         int position) const;
    std::vector<float> post_layernorm_forward(const std::vector<float>& input) const;
    std::vector<float> mlp_forward(const std::vector<float>& input) const;

  private:
    void check_ready() const;

    int layer_idx_;
    int hidden_size_;
    int intermediate_size_;
    bool is_linear_;

    std::unique_ptr<RMSNorm> input_layernorm_;
    std::unique_ptr<RMSNorm> post_layernorm_;
    std::unique_ptr<MLP> mlp_;

    std::unique_ptr<LinearAttention> linear_attn_;
    std::unique_ptr<FullAttention> full_attn_;
};

class LanguageBackbone {
  public:
    LanguageBackbone(int num_layers = 24, int hidden_size = 1024, int intermediate_size = 3584);

    void set_layer_weights(int layer_idx, const LanguageLayerWeights& weights);
    void set_final_norm_weight(std::vector<float> weight);

    std::vector<float> forward(const std::vector<float>& input,
                               std::vector<LinearAttnState>& linear_states, KVCache& kv_cache,
                               int position) const;

    std::vector<float> forward_sequence(const std::vector<float>& input, int seq_len,
                                        std::vector<LinearAttnState>& linear_states,
                                        KVCache& kv_cache) const;

    int num_layers() const {
        return num_layers_;
    }
    int hidden_size() const {
        return hidden_size_;
    }
    const std::vector<std::unique_ptr<LanguageLayer>>& layers() const {
        return layers_;
    }
    std::vector<float> final_norm_forward(const std::vector<float>& input) const {
        return final_norm_->forward(input);
    }

  private:
    void check_ready() const;

    int num_layers_;
    int hidden_size_;
    int intermediate_size_;

    std::vector<std::unique_ptr<LanguageLayer>> layers_;
    std::unique_ptr<RMSNorm> final_norm_;
};

} // namespace qwen
