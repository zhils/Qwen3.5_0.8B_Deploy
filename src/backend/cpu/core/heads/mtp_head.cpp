#include "mtp_head.hpp"
#include <sstream>
#include <algorithm>

namespace qwen {

MTPHead::MTPHead(int hidden_size, int intermediate_size, int fc_output_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      fc_output_size_(fc_output_size) {

    if (hidden_size <= 0 || intermediate_size <= 0 || fc_output_size <= 0) {
        throw std::invalid_argument("All dimensions must be > 0");
    }

    pre_fc_norm_hidden_ = std::make_unique<RMSNorm>(hidden_size);
    pre_fc_norm_embedding_ = std::make_unique<RMSNorm>(hidden_size);

    input_layernorm_ = std::make_unique<RMSNorm>(hidden_size);
    attention_ = std::make_unique<FullAttention>(hidden_size, 8, 2, 512, 256, 10000000.0f, 0.25f);
    post_attention_layernorm_ = std::make_unique<RMSNorm>(hidden_size);
    mlp_ = std::make_unique<MLP>(hidden_size, intermediate_size);

    norm_ = std::make_unique<RMSNorm>(hidden_size);
}

void MTPHead::set_weights(const MTPWeights& weights) {
    pre_fc_norm_hidden_->set_weight(weights.pre_fc_norm_hidden_weight);
    pre_fc_norm_embedding_->set_weight(weights.pre_fc_norm_embedding_weight);

    input_layernorm_->set_weight(weights.layer_input_layernorm_weight);
    post_attention_layernorm_->set_weight(weights.layer_post_attention_layernorm_weight);

    mlp_->set_weights(std::vector<float>(weights.mlp_gate_proj_weight),
                      std::vector<float>(weights.mlp_up_proj_weight),
                      std::vector<float>(weights.mlp_down_proj_weight));

    attention_->set_weights(std::vector<float>(weights.attn_q_proj_weight),
                            std::vector<float>(weights.attn_k_proj_weight),
                            std::vector<float>(weights.attn_v_proj_weight),
                            std::vector<float>(weights.attn_q_norm_weight),
                            std::vector<float>(weights.attn_k_norm_weight),
                            std::vector<float>(weights.attn_o_proj_weight));

    norm_->set_weight(weights.norm_weight);

    size_t expected_fc = static_cast<size_t>(fc_output_size_) * hidden_size_;
    if (weights.fc_weight.size() != expected_fc) {
        std::ostringstream oss;
        oss << "MTP fc weight size mismatch: expected " << expected_fc << ", got "
            << weights.fc_weight.size();
        throw std::invalid_argument(oss.str());
    }
    fc_weight_ = weights.fc_weight;
}

void MTPHead::check_ready() const {
    if (!pre_fc_norm_hidden_ || !pre_fc_norm_embedding_ || !input_layernorm_ || !attention_ ||
        !post_attention_layernorm_ || !mlp_ || !norm_) {
        throw std::runtime_error("MTPHead not properly initialized");
    }
}

std::vector<float> MTPHead::forward(const std::vector<float>& hidden_states,
                                    const std::vector<float>& embedding_input, KVCache& kv_cache,
                                    int position) const {
    check_ready();

    if (static_cast<int>(hidden_states.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "MTPHead hidden_states size mismatch: expected " << hidden_size_ << ", got "
            << hidden_states.size();
        throw std::invalid_argument(oss.str());
    }

    if (static_cast<int>(embedding_input.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "MTPHead embedding_input size mismatch: expected " << hidden_size_ << ", got "
            << embedding_input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> h1 = pre_fc_norm_hidden_->forward(hidden_states);
    std::vector<float> h2 = pre_fc_norm_embedding_->forward(embedding_input);

    std::vector<float> combined(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        combined[i] = h1[i] + h2[i];
    }

    std::vector<float> attn_input = input_layernorm_->forward(combined);
    std::vector<float> attn_out = attention_->forward(attn_input, kv_cache, 0, position);

    std::vector<float> residual_attn(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        residual_attn[i] = combined[i] + attn_out[i];
    }

    std::vector<float> mlp_input = post_attention_layernorm_->forward(residual_attn);
    std::vector<float> mlp_out = mlp_->forward(mlp_input);

    std::vector<float> layer_output(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        layer_output[i] = residual_attn[i] + mlp_out[i];
    }

    std::vector<float> normalized = norm_->forward(layer_output);

    std::vector<float> mtp_output(fc_output_size_, 0.0f);
    for (int o = 0; o < fc_output_size_; ++o) {
        for (int h = 0; h < hidden_size_; ++h) {
            mtp_output[o] += fc_weight_[o * hidden_size_ + h] * normalized[h];
        }
    }

    return mtp_output;
}

} // namespace qwen
