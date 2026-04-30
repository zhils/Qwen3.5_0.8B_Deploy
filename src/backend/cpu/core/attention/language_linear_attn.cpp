#include "language_linear_attn.hpp"
#include <cmath>
#include <sstream>
#include <algorithm>

namespace qwen {

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float softplus(float x) {
    if (x > 20.0f)
        return x;
    return std::log(1.0f + std::exp(x));
}

static float l2norm_eps = 1e-6f;

LinearAttention::LinearAttention(int hidden_size, int num_heads, int key_dim, int value_dim,
                                 int conv_kernel)
    : hidden_size_(hidden_size), num_heads_(num_heads), key_dim_(key_dim), value_dim_(value_dim),
      conv_kernel_(conv_kernel) {
    if (hidden_size <= 0 || num_heads <= 0 || key_dim <= 0 || value_dim <= 0) {
        throw std::invalid_argument("All dimensions must be > 0");
    }
}

std::vector<float> LinearAttention::linear(const std::vector<float>& input,
                                           const std::vector<float>& weight, int out_size) const {
    std::vector<float> output(out_size, 0.0f);
    for (int i = 0; i < out_size; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            output[i] += weight[i * hidden_size_ + j] * input[j];
        }
    }
    return output;
}

void LinearAttention::set_weights(std::vector<float> in_proj_qkv_weight,
                                  std::vector<float> in_proj_a_weight,
                                  std::vector<float> in_proj_b_weight,
                                  std::vector<float> in_proj_z_weight,
                                  std::vector<float> conv1d_weight, std::vector<float> A_log,
                                  std::vector<float> dt_bias, std::vector<float> norm_weight,
                                  std::vector<float> out_proj_weight) {
    int qkv_size = num_heads_ * key_dim_ + num_heads_ * key_dim_ + num_heads_ * value_dim_;
    size_t expected_qkv = static_cast<size_t>(qkv_size) * hidden_size_;
    size_t expected_a = static_cast<size_t>(num_heads_) * hidden_size_;
    size_t expected_b = static_cast<size_t>(num_heads_) * hidden_size_;
    size_t expected_z = static_cast<size_t>(num_heads_ * value_dim_) * hidden_size_;
    size_t expected_A = static_cast<size_t>(num_heads_);
    size_t expected_dt = static_cast<size_t>(num_heads_);
    size_t expected_norm = static_cast<size_t>(value_dim_);
    size_t expected_out = static_cast<size_t>(hidden_size_) * num_heads_ * value_dim_;

    if (in_proj_qkv_weight.size() != expected_qkv) {
        std::ostringstream oss;
        oss << "in_proj_qkv weight size mismatch: expected " << expected_qkv << ", got "
            << in_proj_qkv_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (in_proj_a_weight.size() != expected_a) {
        std::ostringstream oss;
        oss << "in_proj_a weight size mismatch: expected " << expected_a << ", got "
            << in_proj_a_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (in_proj_b_weight.size() != expected_b) {
        std::ostringstream oss;
        oss << "in_proj_b weight size mismatch: expected " << expected_b << ", got "
            << in_proj_b_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (in_proj_z_weight.size() != expected_z) {
        std::ostringstream oss;
        oss << "in_proj_z weight size mismatch: expected " << expected_z << ", got "
            << in_proj_z_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (A_log.size() != expected_A) {
        std::ostringstream oss;
        oss << "A_log size mismatch: expected " << expected_A << ", got " << A_log.size();
        throw std::invalid_argument(oss.str());
    }
    if (dt_bias.size() != expected_dt) {
        std::ostringstream oss;
        oss << "dt_bias size mismatch: expected " << expected_dt << ", got " << dt_bias.size();
        throw std::invalid_argument(oss.str());
    }
    if (norm_weight.size() != expected_norm) {
        std::ostringstream oss;
        oss << "norm weight size mismatch: expected " << expected_norm << ", got "
            << norm_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (out_proj_weight.size() != expected_out) {
        std::ostringstream oss;
        oss << "out_proj weight size mismatch: expected " << expected_out << ", got "
            << out_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }

    in_proj_qkv_weight_ = std::move(in_proj_qkv_weight);
    in_proj_a_weight_ = std::move(in_proj_a_weight);
    in_proj_b_weight_ = std::move(in_proj_b_weight);
    in_proj_z_weight_ = std::move(in_proj_z_weight);
    conv1d_weight_ = std::move(conv1d_weight);
    A_log_ = std::move(A_log);
    dt_bias_ = std::move(dt_bias);
    norm_weight_ = std::move(norm_weight);
    out_proj_weight_ = std::move(out_proj_weight);
}

void LinearAttention::check_ready() const {
    if (in_proj_qkv_weight_.empty()) {
        throw std::runtime_error("LinearAttention weights not set");
    }
}

std::vector<float> LinearAttention::forward(const std::vector<float>& input,
                                            LinearAttnState& state) const {
    check_ready();

    if (static_cast<int>(input.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "LinearAttention input size mismatch: expected " << hidden_size_ << ", got "
            << input.size();
        throw std::invalid_argument(oss.str());
    }

    int q_dim = num_heads_ * key_dim_;
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;

    std::vector<float> mixed_qkv = linear(input, in_proj_qkv_weight_, conv_dim);

    int state_len = conv_kernel_ - 1;

    std::vector<float> conv_out(conv_dim, 0.0f);
    for (int d = 0; d < conv_dim; ++d) {
        float sum = conv1d_weight_[d * conv_kernel_ + (conv_kernel_ - 1)] * mixed_qkv[d];
        for (int k = 0; k < state_len; ++k) {
            sum += conv1d_weight_[d * conv_kernel_ + k] * state.conv_state[d][k];
        }
        conv_out[d] = silu(sum);
    }

    for (int d = 0; d < conv_dim; ++d) {
        for (int k = state_len - 1; k > 0; --k) {
            state.conv_state[d][k] = state.conv_state[d][k - 1];
        }
        state.conv_state[d][0] = mixed_qkv[d];
    }

    std::vector<float> q(q_dim), k(k_dim), v(v_dim);
    std::copy(conv_out.begin(), conv_out.begin() + q_dim, q.begin());
    std::copy(conv_out.begin() + q_dim, conv_out.begin() + q_dim + k_dim, k.begin());
    std::copy(conv_out.begin() + q_dim + k_dim, conv_out.end(), v.begin());

    std::vector<float> a = linear(input, in_proj_a_weight_, num_heads_);
    std::vector<float> b_raw = linear(input, in_proj_b_weight_, num_heads_);
    std::vector<float> z = linear(input, in_proj_z_weight_, num_heads_ * value_dim_);

    std::vector<float> beta(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        beta[h] = sigmoid(b_raw[h]);
    }

    std::vector<float> g(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        g[h] = -std::exp(A_log_[h]) * softplus(a[h] + dt_bias_[h]);
    }

    float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));

    for (int h = 0; h < num_heads_; ++h) {
        float q_l2 = 0.0f;
        for (int d = 0; d < key_dim_; ++d) {
            float val = q[h * key_dim_ + d];
            q_l2 += val * val;
        }
        q_l2 = std::sqrt(q_l2 + l2norm_eps);
        for (int d = 0; d < key_dim_; ++d) {
            q[h * key_dim_ + d] = q[h * key_dim_ + d] / q_l2 * q_scale;
        }

        float k_l2 = 0.0f;
        for (int d = 0; d < key_dim_; ++d) {
            float val = k[h * key_dim_ + d];
            k_l2 += val * val;
        }
        k_l2 = std::sqrt(k_l2 + l2norm_eps);
        for (int d = 0; d < key_dim_; ++d) {
            k[h * key_dim_ + d] = k[h * key_dim_ + d] / k_l2;
        }
    }

    std::vector<float> output(num_heads_ * value_dim_, 0.0f);

    for (int h = 0; h < num_heads_; ++h) {
        float g_t = std::exp(g[h]);

        for (int kd = 0; kd < key_dim_; ++kd) {
            for (int vd = 0; vd < value_dim_; ++vd) {
                state.recurrent_state[h][kd * value_dim_ + vd] *= g_t;
            }
        }

        std::vector<float> kv_mem(value_dim_, 0.0f);
        for (int vd = 0; vd < value_dim_; ++vd) {
            for (int kd = 0; kd < key_dim_; ++kd) {
                kv_mem[vd] += state.recurrent_state[h][kd * value_dim_ + vd] * k[h * key_dim_ + kd];
            }
        }

        std::vector<float> delta(value_dim_);
        for (int vd = 0; vd < value_dim_; ++vd) {
            delta[vd] = (v[h * value_dim_ + vd] - kv_mem[vd]) * beta[h];
        }

        for (int kd = 0; kd < key_dim_; ++kd) {
            for (int vd = 0; vd < value_dim_; ++vd) {
                state.recurrent_state[h][kd * value_dim_ + vd] += k[h * key_dim_ + kd] * delta[vd];
            }
        }

        for (int vd = 0; vd < value_dim_; ++vd) {
            float sum = 0.0f;
            for (int kd = 0; kd < key_dim_; ++kd) {
                sum += state.recurrent_state[h][kd * value_dim_ + vd] * q[h * key_dim_ + kd];
            }
            output[h * value_dim_ + vd] = sum;
        }
    }

    for (int h = 0; h < num_heads_; ++h) {
        float variance = 0.0f;
        for (int d = 0; d < value_dim_; ++d) {
            variance += output[h * value_dim_ + d] * output[h * value_dim_ + d];
        }
        variance /= value_dim_;
        float inv_rms = 1.0f / std::sqrt(variance + l2norm_eps);
        for (int d = 0; d < value_dim_; ++d) {
            output[h * value_dim_ + d] = output[h * value_dim_ + d] * inv_rms * norm_weight_[d];
        }
    }

    for (int i = 0; i < num_heads_ * value_dim_; ++i) {
        output[i] = output[i] * silu(z[i]);
    }

    std::vector<float> final_output(hidden_size_, 0.0f);
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < num_heads_ * value_dim_; ++j) {
            final_output[i] += out_proj_weight_[i * num_heads_ * value_dim_ + j] * output[j];
        }
    }

    return final_output;
}

std::vector<float> LinearAttention::forward_sequence(const std::vector<float>& input,
                                                     int seq_len) const {
    check_ready();

    size_t expected_size = static_cast<size_t>(seq_len) * hidden_size_;
    if (input.size() != expected_size) {
        std::ostringstream oss;
        oss << "LinearAttention sequence input size mismatch: expected " << expected_size
            << ", got " << input.size();
        throw std::invalid_argument(oss.str());
    }

    LinearAttnState state;
    state.reset(num_heads_, key_dim_, value_dim_, conv_kernel_);

    std::vector<float> output(static_cast<size_t>(seq_len) * hidden_size_);

    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> token_input(hidden_size_);
        std::copy(input.begin() + t * hidden_size_, input.begin() + (t + 1) * hidden_size_,
                  token_input.begin());

        std::vector<float> token_output = forward(token_input, state);

        std::copy(token_output.begin(), token_output.end(), output.begin() + t * hidden_size_);
    }

    return output;
}

} // namespace qwen
