#pragma once

#include <vector>

namespace qwen {

class MLP {
  public:
    MLP(int hidden_size = 1024, int intermediate_size = 3584);

    void set_weights(std::vector<float> gate_proj_weight, std::vector<float> up_proj_weight,
                     std::vector<float> down_proj_weight);

    std::vector<float> forward(const std::vector<float>& input) const;

    void forward_batch(const float* input, float* output, int batch_size) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int intermediate_size() const {
        return intermediate_size_;
    }

  private:
    void check_ready() const;

    int hidden_size_;
    int intermediate_size_;

    std::vector<float> gate_proj_weight_;
    std::vector<float> up_proj_weight_;
    std::vector<float> down_proj_weight_;
};

} // namespace qwen
