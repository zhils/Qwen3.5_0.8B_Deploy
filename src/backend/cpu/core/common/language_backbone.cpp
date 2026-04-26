#include "language_backbone.hpp"
#include <sstream>

namespace qwen {

LanguageLayer::LanguageLayer(int layer_idx, int hidden_size, int intermediate_size)
    : layer_idx_(layer_idx), hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      is_linear_(is_linear_layer(layer_idx)) {

    input_layernorm_ = std::make_unique<RMSNorm>(hidden_size);
    post_layernorm_ = std::make_unique<RMSNorm>(hidden_size);
    mlp_ = std::make_unique<MLP>(hidden_size, intermediate_size);

    if (is_linear_) {
        linear_attn_ = std::make_unique<LinearAttention>(hidden_size, 16, 128, 128, 4);
    } else {
        full_attn_ =
            std::make_unique<FullAttention>(hidden_size, 8, 2, 256, 256, 10000000.0f, 0.25f);
    }
}

void LanguageLayer::set_weights(const LanguageLayerWeights& weights) {
    input_layernorm_->set_weight(weights.input_layernorm_weight);
    post_layernorm_->set_weight(weights.post_attention_layernorm_weight);
    mlp_->set_weights(std::vector<float>(weights.mlp_gate_proj_weight),
                      std::vector<float>(weights.mlp_up_proj_weight),
                      std::vector<float>(weights.mlp_down_proj_weight));

    if (is_linear_) {
        linear_attn_->set_weights(std::vector<float>(weights.linear_in_proj_qkv_weight),
                                  std::vector<float>(weights.linear_in_proj_a_weight),
                                  std::vector<float>(weights.linear_in_proj_b_weight),
                                  std::vector<float>(weights.linear_in_proj_z_weight),
                                  std::vector<float>(weights.linear_conv1d_weight),
                                  std::vector<float>(weights.linear_A_log),
                                  std::vector<float>(weights.linear_dt_bias),
                                  std::vector<float>(weights.linear_norm_weight),
                                  std::vector<float>(weights.linear_out_proj_weight));
    } else {
        full_attn_->set_weights(std::vector<float>(weights.full_q_proj_weight),
                                std::vector<float>(weights.full_k_proj_weight),
                                std::vector<float>(weights.full_v_proj_weight),
                                std::vector<float>(weights.full_q_norm_weight),
                                std::vector<float>(weights.full_k_norm_weight),
                                std::vector<float>(weights.full_o_proj_weight));
    }
}

void LanguageLayer::check_ready() const {
    if (!input_layernorm_ || !post_layernorm_ || !mlp_) {
        throw std::runtime_error("LanguageLayer not properly initialized");
    }
}

std::vector<float> LanguageLayer::forward(const std::vector<float>& input,
                                          LinearAttnState& linear_state, KVCache& kv_cache,
                                          int position) const {
    check_ready();

    if (static_cast<int>(input.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "LanguageLayer input size mismatch: expected " << hidden_size_ << ", got "
            << input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> h1 = input_layernorm_->forward(input);

    std::vector<float> attn_out;
    if (is_linear_) {
        attn_out = linear_attn_->forward(h1, linear_state);
    } else {
        attn_out = full_attn_->forward(h1, kv_cache, layer_idx_, position);
    }

    std::vector<float> h(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        h[i] = input[i] + attn_out[i];
    }

    std::vector<float> h2 = post_layernorm_->forward(h);

    std::vector<float> mlp_out = mlp_->forward(h2);

    std::vector<float> output(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        output[i] = h[i] + mlp_out[i];
    }

    return output;
}

std::vector<float> LanguageLayer::forward_sequence(const std::vector<float>& input, int seq_len,
                                                   LinearAttnState& linear_state,
                                                   KVCache& kv_cache) const {
    check_ready();

    size_t expected_size = static_cast<size_t>(seq_len) * hidden_size_;
    if (input.size() != expected_size) {
        std::ostringstream oss;
        oss << "LanguageLayer sequence input size mismatch: expected " << expected_size << ", got "
            << input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> output(expected_size);

    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> token_input(hidden_size_);
        std::copy(input.begin() + t * hidden_size_, input.begin() + (t + 1) * hidden_size_,
                  token_input.begin());

        std::vector<float> token_output = forward(token_input, linear_state, kv_cache, t);

        std::copy(token_output.begin(), token_output.end(), output.begin() + t * hidden_size_);
    }

    return output;
}

std::vector<float> LanguageLayer::input_layernorm_forward(const std::vector<float>& input) const {
    check_ready();
    return input_layernorm_->forward(input);
}

std::vector<float> LanguageLayer::attention_forward(const std::vector<float>& input,
                                                    LinearAttnState& linear_state,
                                                    KVCache& kv_cache, int position) const {
    check_ready();
    if (is_linear_) {
        return linear_attn_->forward(input, linear_state);
    } else {
        return full_attn_->forward(input, kv_cache, layer_idx_, position);
    }
}

std::vector<float> LanguageLayer::post_layernorm_forward(const std::vector<float>& input) const {
    check_ready();
    return post_layernorm_->forward(input);
}

std::vector<float> LanguageLayer::mlp_forward(const std::vector<float>& input) const {
    check_ready();
    return mlp_->forward(input);
}

LanguageBackbone::LanguageBackbone(int num_layers, int hidden_size, int intermediate_size)
    : num_layers_(num_layers), hidden_size_(hidden_size), intermediate_size_(intermediate_size) {

    if (num_layers <= 0) {
        throw std::invalid_argument("num_layers must be > 0");
    }

    for (int i = 0; i < num_layers; ++i) {
        layers_.push_back(std::make_unique<LanguageLayer>(i, hidden_size, intermediate_size));
    }

    final_norm_ = std::make_unique<RMSNorm>(hidden_size);
}

void LanguageBackbone::set_layer_weights(int layer_idx, const LanguageLayerWeights& weights) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        std::ostringstream oss;
        oss << "Invalid layer_idx: " << layer_idx;
        throw std::out_of_range(oss.str());
    }
    layers_[layer_idx]->set_weights(weights);
}

void LanguageBackbone::set_final_norm_weight(std::vector<float> weight) {
    final_norm_->set_weight(std::move(weight));
}

void LanguageBackbone::check_ready() const {
    if (layers_.empty() || !final_norm_) {
        throw std::runtime_error("LanguageBackbone not properly initialized");
    }
}

std::vector<float> LanguageBackbone::forward(const std::vector<float>& input,
                                             std::vector<LinearAttnState>& linear_states,
                                             KVCache& kv_cache, int position) const {
    check_ready();

    if (static_cast<int>(input.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "LanguageBackbone input size mismatch: expected " << hidden_size_ << ", got "
            << input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> h = input;

    for (int i = 0; i < num_layers_; ++i) {
        h = layers_[i]->forward(h, linear_states[i], kv_cache, position);
    }

    h = final_norm_->forward(h);

    return h;
}

std::vector<float> LanguageBackbone::forward_sequence(const std::vector<float>& input, int seq_len,
                                                      std::vector<LinearAttnState>& linear_states,
                                                      KVCache& kv_cache) const {
    check_ready();

    size_t expected_size = static_cast<size_t>(seq_len) * hidden_size_;
    if (input.size() != expected_size) {
        std::ostringstream oss;
        oss << "LanguageBackbone sequence input size mismatch: expected " << expected_size
            << ", got " << input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> output(expected_size);

    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> token_input(hidden_size_);
        std::copy(input.begin() + t * hidden_size_, input.begin() + (t + 1) * hidden_size_,
                  token_input.begin());

        std::vector<float> token_output = forward(token_input, linear_states, kv_cache, t);

        std::copy(token_output.begin(), token_output.end(), output.begin() + t * hidden_size_);
    }

    return output;
}

} // namespace qwen
