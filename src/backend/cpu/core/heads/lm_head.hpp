#pragma once

#include <vector>

namespace qwen {

class LMHead {
  public:
    LMHead(int hidden_size = 1024, int vocab_size = 248320);

    void set_weight(const std::vector<float>& weight);

    std::vector<float> forward(const std::vector<float>& input) const;

    std::vector<std::pair<int, float>> get_top_tokens(const std::vector<float>& logits,
                                                      int top_k = 10) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int vocab_size() const {
        return vocab_size_;
    }

  private:
    void check_ready() const;

    int hidden_size_;
    int vocab_size_;
    std::vector<float> weight_;
};

} // namespace qwen
