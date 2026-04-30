#pragma once

#include "language_common.hpp"
#include "language_mlp.hpp"
#include "language_full_attn.hpp"
#include <vector>
#include <memory>

namespace qwen {

struct MTPWeights {
    std::vector<float> pre_fc_norm_hidden_weight;
    std::vector<float> pre_fc_norm_embedding_weight;

    std::vector<float> layer_input_layernorm_weight;
    std::vector<float> layer_post_attention_layernorm_weight;

    std::vector<float> mlp_gate_proj_weight;
    std::vector<float> mlp_up_proj_weight;
    std::vector<float> mlp_down_proj_weight;

    std::vector<float> attn_q_proj_weight;
    std::vector<float> attn_k_proj_weight;
    std::vector<float> attn_v_proj_weight;
    std::vector<float> attn_q_norm_weight;
    std::vector<float> attn_k_norm_weight;
    std::vector<float> attn_o_proj_weight;

    std::vector<float> norm_weight;
    std::vector<float> fc_weight;
};

class MTPHead {
  public:
    MTPHead(int hidden_size = 1024, int intermediate_size = 3584, int fc_output_size = 2048);

    void set_weights(const MTPWeights& weights);

    std::vector<float> forward(const std::vector<float>& hidden_states,
                               const std::vector<float>& embedding_input, KVCache& kv_cache,
                               int position) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int fc_output_size() const {
        return fc_output_size_;
    }

  private:
    void check_ready() const;

    int hidden_size_;
    int intermediate_size_;
    int fc_output_size_;

    std::unique_ptr<RMSNorm> pre_fc_norm_hidden_;
    std::unique_ptr<RMSNorm> pre_fc_norm_embedding_;

    std::unique_ptr<RMSNorm> input_layernorm_;
    std::unique_ptr<FullAttention> attention_;
    std::unique_ptr<RMSNorm> post_attention_layernorm_;
    std::unique_ptr<MLP> mlp_;

    std::unique_ptr<RMSNorm> norm_;
    std::vector<float> fc_weight_;
};

} // namespace qwen
