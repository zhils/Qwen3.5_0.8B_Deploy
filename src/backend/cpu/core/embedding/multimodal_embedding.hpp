#pragma once

#include "vision_transformer.hpp"
#include "token_embedding.hpp"
#include <vector>

namespace qwen {

struct MultimodalSequence {
    std::vector<float> data;
    int num_tokens = 0;
    int hidden_size = 0;
    int num_vision_tokens = 0;
    int num_text_tokens = 0;

    inline float at(int token_idx, int dim_idx) const {
        return data[static_cast<size_t>(token_idx) * hidden_size + dim_idx];
    }

    inline float& at(int token_idx, int dim_idx) {
        return data[static_cast<size_t>(token_idx) * hidden_size + dim_idx];
    }
};

class MultimodalEmbedding {
  public:
    MultimodalEmbedding(int hidden_size = 1024);

    MultimodalSequence concat_vision_text(const Tensor2D& vision_tokens,
                                          const std::vector<int>& text_token_ids,
                                          const TokenEmbedding& text_embedding) const;

    MultimodalSequence
    concat_vision_text(const Tensor2D& vision_tokens,
                       const std::vector<std::vector<float>>& text_embeddings) const;

    MultimodalSequence text_only(const std::vector<int>& text_token_ids,
                                 const TokenEmbedding& text_embedding) const;

    int hidden_size() const {
        return hidden_size_;
    }

  private:
    int hidden_size_;
};

} // namespace qwen
