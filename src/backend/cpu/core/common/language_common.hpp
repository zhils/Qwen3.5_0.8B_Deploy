#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace qwen {

class RMSNorm {
  public:
    RMSNorm(int hidden_size, float eps = 1e-6f);

    void set_weight(std::vector<float> weight);

    std::vector<float> forward(const std::vector<float>& input) const;

    void forward_inplace(std::vector<float>& input) const;

    void forward_batch(const float* input, float* output, int batch_size) const;

    int hidden_size() const {
        return hidden_size_;
    }
    const std::vector<float>& weight() const {
        return weight_;
    }

  private:
    void check_ready() const;

    int hidden_size_;
    float eps_;
    std::vector<float> weight_;
};

inline bool is_linear_layer(int layer_idx) {
    static const int linear_layers[] = {0,  1,  2,  4,  5,  6,  8,  9,  10,
                                        12, 13, 14, 16, 17, 18, 20, 21, 22};
    for (int i : linear_layers) {
        if (i == layer_idx)
            return true;
    }
    return false;
}

inline bool is_full_layer(int layer_idx) {
    static const int full_layers[] = {3, 7, 11, 15, 19, 23};
    for (int i : full_layers) {
        if (i == layer_idx)
            return true;
    }
    return false;
}

} // namespace qwen
