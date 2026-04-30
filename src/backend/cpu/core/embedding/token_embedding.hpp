#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qwen {

class TokenEmbedding {
  public:
    TokenEmbedding(int vocab_size = 248320, int hidden_size = 1024);

    void set_weights(std::vector<float> weight);

    std::vector<float> forward(const std::vector<int>& token_ids) const;

    int vocab_size() const {
        return vocab_size_;
    }
    int hidden_size() const {
        return hidden_size_;
    }

    const std::vector<float>& embeddings() const {
        return weight_;
    }

    std::vector<float> get_embedding(int token_id) const;

  private:
    void check_ready() const;

    int vocab_size_;
    int hidden_size_;
    std::vector<float> weight_;
};

} // namespace qwen
