#include "token_embedding.hpp"
#include "error_handling.hpp"

namespace qwen {

TokenEmbedding::TokenEmbedding(int vocab_size, int hidden_size)
    : vocab_size_(vocab_size), hidden_size_(hidden_size) {
    QWEN_CHECK_POSITIVE(vocab_size, "vocab_size");
    QWEN_CHECK_POSITIVE(hidden_size, "hidden_size");
}

void TokenEmbedding::set_weights(std::vector<float> weight) {
    size_t expected_size = static_cast<size_t>(vocab_size_) * hidden_size_;
    QWEN_CHECK_WEIGHT_SIZE(expected_size, weight.size(), "TokenEmbedding");
    weight_ = std::move(weight);
}

void TokenEmbedding::check_ready() const {
    QWEN_CHECK_NOT_EMPTY(weight_, "TokenEmbedding");
}

std::vector<float> TokenEmbedding::get_embedding(int token_id) const {
    check_ready();
    QWEN_CHECK(token_id >= 0 && token_id < vocab_size_,
               "token_id out of range: " + std::to_string(token_id) +
                   ", vocab_size=" + std::to_string(vocab_size_));

    std::vector<float> embedding(hidden_size_);
    size_t offset = static_cast<size_t>(token_id) * hidden_size_;
    std::copy(weight_.begin() + offset, weight_.begin() + offset + hidden_size_, embedding.begin());
    return embedding;
}

std::vector<float> TokenEmbedding::forward(const std::vector<int>& token_ids) const {
    check_ready();

    for (int id : token_ids) {
        QWEN_CHECK(id >= 0 && id < vocab_size_,
                   "token_id out of range: " + std::to_string(id) +
                       ", vocab_size=" + std::to_string(vocab_size_));
    }

    std::vector<float> embeddings(token_ids.size() * hidden_size_);
    for (size_t i = 0; i < token_ids.size(); ++i) {
        size_t src_offset = static_cast<size_t>(token_ids[i]) * hidden_size_;
        size_t dst_offset = i * hidden_size_;
        std::copy(weight_.begin() + src_offset, weight_.begin() + src_offset + hidden_size_,
                  embeddings.begin() + dst_offset);
    }

    return embeddings;
}

} // namespace qwen
