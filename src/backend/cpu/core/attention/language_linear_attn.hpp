#pragma once

#include <vector>

namespace qwen {

struct LinearAttnState {
    std::vector<std::vector<float>> recurrent_state;
    std::vector<std::vector<float>> conv_state;
    int num_heads = 0;
    int key_dim = 0;
    int value_dim = 0;
    int conv_dim = 0;
    int conv_kernel = 0;

    void reset(int nh, int kd, int vd, int conv_k = 4) {
        num_heads = nh;
        key_dim = kd;
        value_dim = vd;
        conv_kernel = conv_k;
        int qkv_per_head = kd * 2 + vd;
        conv_dim = nh * qkv_per_head;
        recurrent_state.assign(nh, std::vector<float>(kd * vd, 0.0f));
        conv_state.assign(conv_dim, std::vector<float>(conv_k - 1, 0.0f));
    }

    void clear() {
        for (auto& s : recurrent_state) {
            std::fill(s.begin(), s.end(), 0.0f);
        }
        for (auto& s : conv_state) {
            std::fill(s.begin(), s.end(), 0.0f);
        }
    }
};

class LinearAttention {
  public:
    LinearAttention(int hidden_size = 1024, int num_heads = 16, int key_dim = 128,
                    int value_dim = 128, int conv_kernel = 4);

    void set_weights(std::vector<float> in_proj_qkv_weight, std::vector<float> in_proj_a_weight,
                     std::vector<float> in_proj_b_weight, std::vector<float> in_proj_z_weight,
                     std::vector<float> conv1d_weight, std::vector<float> A_log,
                     std::vector<float> dt_bias, std::vector<float> norm_weight,
                     std::vector<float> out_proj_weight);

    std::vector<float> forward(const std::vector<float>& input, LinearAttnState& state) const;

    std::vector<float> forward_sequence(const std::vector<float>& input, int seq_len) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int num_heads() const {
        return num_heads_;
    }
    int key_dim() const {
        return key_dim_;
    }
    int value_dim() const {
        return value_dim_;
    }

  private:
    void check_ready() const;
    std::vector<float> linear(const std::vector<float>& input, const std::vector<float>& weight,
                              int out_size) const;

    int hidden_size_;
    int num_heads_;
    int key_dim_;
    int value_dim_;
    int conv_kernel_;

    std::vector<float> in_proj_qkv_weight_;
    std::vector<float> in_proj_a_weight_;
    std::vector<float> in_proj_b_weight_;
    std::vector<float> in_proj_z_weight_;
    std::vector<float> conv1d_weight_;
    std::vector<float> A_log_;
    std::vector<float> dt_bias_;
    std::vector<float> norm_weight_;
    std::vector<float> out_proj_weight_;
};

} // namespace qwen
