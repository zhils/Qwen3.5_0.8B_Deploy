#include "language_full_attn.hpp"
#include <cmath>
#include <sstream>
#include <iostream>
#include <memory>

namespace qwen {

FullAttention::FullAttention(int hidden_size, int num_heads, int num_kv_heads, int q_head_dim,
                             int kv_head_dim, float rope_theta, float partial_rotary)
    : hidden_size_(hidden_size), num_heads_(num_heads), num_kv_heads_(num_kv_heads),
      q_head_dim_(q_head_dim), kv_head_dim_(kv_head_dim), rope_theta_(rope_theta),
      partial_rotary_(partial_rotary) {
    if (hidden_size <= 0 || num_heads <= 0 || num_kv_heads <= 0) {
        throw std::invalid_argument("All dimensions must be > 0");
    }
    rotary_dim_ = static_cast<int>(kv_head_dim * partial_rotary);
}

void FullAttention::set_weights(std::vector<float> q_proj_weight, std::vector<float> k_proj_weight,
                                std::vector<float> v_proj_weight, std::vector<float> q_norm_weight,
                                std::vector<float> k_norm_weight,
                                std::vector<float> o_proj_weight) {
    size_t expected_q = static_cast<size_t>(num_heads_) * q_head_dim_ * 2 * hidden_size_;
    size_t expected_kv = static_cast<size_t>(num_kv_heads_) * kv_head_dim_ * hidden_size_;
    size_t expected_q_norm = static_cast<size_t>(kv_head_dim_);
    size_t expected_k_norm = static_cast<size_t>(kv_head_dim_);
    size_t expected_o = static_cast<size_t>(hidden_size_) * num_heads_ * kv_head_dim_;

    if (q_proj_weight.size() != expected_q) {
        std::ostringstream oss;
        oss << "q_proj weight size mismatch: expected " << expected_q << ", got "
            << q_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (k_proj_weight.size() != expected_kv) {
        std::ostringstream oss;
        oss << "k_proj weight size mismatch: expected " << expected_kv << ", got "
            << k_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (v_proj_weight.size() != expected_kv) {
        std::ostringstream oss;
        oss << "v_proj weight size mismatch: expected " << expected_kv << ", got "
            << v_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (q_norm_weight.size() != expected_q_norm) {
        std::ostringstream oss;
        oss << "q_norm weight size mismatch: expected " << expected_q_norm << ", got "
            << q_norm_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (k_norm_weight.size() != expected_k_norm) {
        std::ostringstream oss;
        oss << "k_norm weight size mismatch: expected " << expected_k_norm << ", got "
            << k_norm_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (o_proj_weight.size() != expected_o) {
        std::ostringstream oss;
        oss << "o_proj weight size mismatch: expected " << expected_o << ", got "
            << o_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }

    q_proj_weight_ = std::move(q_proj_weight);
    k_proj_weight_ = std::move(k_proj_weight);
    v_proj_weight_ = std::move(v_proj_weight);
    q_norm_weight_ = std::move(q_norm_weight);
    k_norm_weight_ = std::move(k_norm_weight);
    o_proj_weight_ = std::move(o_proj_weight);
}

void FullAttention::check_ready() const {
    if (q_proj_weight_.empty()) {
        throw std::runtime_error("FullAttention weights not set");
    }
}

void FullAttention::rms_normalize(float* data, int size, const float* weight) const {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_sq += data[i] * data[i];
    }
    float rms = std::sqrt(sum_sq / size + 1e-6f);
    for (int i = 0; i < size; ++i) {
        data[i] = data[i] / rms * (1.0f + weight[i]);
    }
}

void FullAttention::apply_rope(float* q, float* k, int position) const {
    for (int d = 0; d < rotary_dim_; d += 2) {
        float freq = 1.0f / std::pow(rope_theta_, static_cast<float>(d) / rotary_dim_);
        float angle = position * freq;
        float cos_val = std::cos(angle);
        float sin_val = std::sin(angle);

        for (int h = 0; h < num_heads_; ++h) {
            float q0 = q[h * q_head_dim_ + d];
            float q1 = q[h * q_head_dim_ + d + 1];
            q[h * q_head_dim_ + d] = q0 * cos_val - q1 * sin_val;
            q[h * q_head_dim_ + d + 1] = q0 * sin_val + q1 * cos_val;
        }

        for (int h = 0; h < num_kv_heads_; ++h) {
            float k0 = k[h * kv_head_dim_ + d];
            float k1 = k[h * kv_head_dim_ + d + 1];
            k[h * kv_head_dim_ + d] = k0 * cos_val - k1 * sin_val;
            k[h * kv_head_dim_ + d + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

std::vector<float> FullAttention::forward(const std::vector<float>& input, KVCache& kv_cache,
                                          int layer_idx, int position) const {
    check_ready();

    if (static_cast<int>(input.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "FullAttention input size mismatch: expected " << hidden_size_ << ", got "
            << input.size();
        throw std::invalid_argument(oss.str());
    }

    auto q_raw = std::make_unique<std::vector<float>>(num_heads_ * q_head_dim_ * 2, 0.0f);
    for (int i = 0; i < num_heads_ * q_head_dim_ * 2; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            (*q_raw)[i] += q_proj_weight_[i * hidden_size_ + j] * input[j];
        }
    }

    auto q = std::make_unique<std::vector<float>>(num_heads_ * q_head_dim_, 0.0f);
    auto gate = std::make_unique<std::vector<float>>(num_heads_ * q_head_dim_, 0.0f);
    for (int h = 0; h < num_heads_; ++h) {
        for (int d = 0; d < q_head_dim_; ++d) {
            (*q)[h * q_head_dim_ + d] = (*q_raw)[h * (q_head_dim_ * 2) + d];
            (*gate)[h * q_head_dim_ + d] = (*q_raw)[h * (q_head_dim_ * 2) + q_head_dim_ + d];
        }
    }
    q_raw.reset();

    auto k = std::make_unique<std::vector<float>>(num_kv_heads_ * kv_head_dim_, 0.0f);
    for (int i = 0; i < num_kv_heads_ * kv_head_dim_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            (*k)[i] += k_proj_weight_[i * hidden_size_ + j] * input[j];
        }
    }

    auto v = std::make_unique<std::vector<float>>(num_kv_heads_ * kv_head_dim_, 0.0f);
    for (int i = 0; i < num_kv_heads_ * kv_head_dim_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            (*v)[i] += v_proj_weight_[i * hidden_size_ + j] * input[j];
        }
    }

    for (int h = 0; h < num_heads_; ++h) {
        rms_normalize(&(*q)[h * q_head_dim_], kv_head_dim_, q_norm_weight_.data());
    }
    for (int h = 0; h < num_kv_heads_; ++h) {
        rms_normalize(&(*k)[h * kv_head_dim_], kv_head_dim_, k_norm_weight_.data());
    }

    apply_rope(q->data(), k->data(), position);

    kv_cache.append(layer_idx, k->data(), v->data());

    auto output = std::make_unique<std::vector<float>>(num_heads_ * kv_head_dim_, 0.0f);
    int kv_len = kv_cache.length(layer_idx);
    int heads_per_group = num_heads_ / num_kv_heads_;

    for (int h = 0; h < num_heads_; ++h) {
        int kv_h = h / heads_per_group;

        std::vector<float> attn_weights(kv_len, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(kv_head_dim_));

        for (int t = 0; t < kv_len; ++t) {
            const float* k_ptr =
                kv_cache.get_k(layer_idx) + t * num_kv_heads_ * kv_head_dim_ + kv_h * kv_head_dim_;
            float dot = 0.0f;
            for (int d = 0; d < kv_head_dim_; ++d) {
                dot += (*q)[h * q_head_dim_ + d] * k_ptr[d];
            }
            attn_weights[t] = dot * scale;
        }

        float max_attn = attn_weights[0];
        for (int t = 1; t < kv_len; ++t) {
            max_attn = std::max(max_attn, attn_weights[t]);
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            attn_weights[t] = std::exp(attn_weights[t] - max_attn);
            sum_exp += attn_weights[t];
        }
        for (int t = 0; t < kv_len; ++t) {
            attn_weights[t] /= sum_exp;
        }

        for (int t = 0; t < kv_len; ++t) {
            const float* v_ptr =
                kv_cache.get_v(layer_idx) + t * num_kv_heads_ * kv_head_dim_ + kv_h * kv_head_dim_;
            for (int d = 0; d < kv_head_dim_; ++d) {
                (*output)[h * kv_head_dim_ + d] += attn_weights[t] * v_ptr[d];
            }
        }
    }

    for (int h = 0; h < num_heads_; ++h) {
        for (int d = 0; d < kv_head_dim_; ++d) {
            (*output)[h * kv_head_dim_ + d] *=
                1.0f / (1.0f + std::exp(-(*gate)[h * q_head_dim_ + d]));
        }
    }

    auto final_output = std::make_unique<std::vector<float>>(hidden_size_, 0.0f);
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < num_heads_ * kv_head_dim_; ++j) {
            (*final_output)[i] += o_proj_weight_[i * num_heads_ * kv_head_dim_ + j] * (*output)[j];
        }
    }

    return *final_output;
}

std::vector<float> FullAttention::forward_sequence(const std::vector<float>& input, int seq_len,
                                                   KVCache& kv_cache, int layer_idx) const {
    check_ready();

    size_t expected_size = static_cast<size_t>(seq_len) * hidden_size_;
    if (input.size() != expected_size) {
        std::ostringstream oss;
        oss << "FullAttention sequence input size mismatch: expected " << expected_size << ", got "
            << input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> output(expected_size);

    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> token_input(hidden_size_);
        std::copy(input.begin() + t * hidden_size_, input.begin() + (t + 1) * hidden_size_,
                  token_input.begin());

        std::vector<float> token_output = forward(token_input, kv_cache, layer_idx, t);

        std::copy(token_output.begin(), token_output.end(), output.begin() + t * hidden_size_);
    }

    return output;
}

} // namespace qwen
