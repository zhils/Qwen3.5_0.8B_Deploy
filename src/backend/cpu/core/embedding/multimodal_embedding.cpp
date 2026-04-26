#include "multimodal_embedding.hpp"
#include <sstream>
#include <stdexcept>

namespace qwen {

MultimodalEmbedding::MultimodalEmbedding(int hidden_size) : hidden_size_(hidden_size) {
    if (hidden_size <= 0) {
        throw std::invalid_argument("hidden_size must be > 0");
    }
}

MultimodalSequence
MultimodalEmbedding::concat_vision_text(const Tensor2D& vision_tokens,
                                        const std::vector<int>& text_token_ids,
                                        const TokenEmbedding& text_embedding) const {

    if (vision_tokens.d != hidden_size_) {
        std::ostringstream oss;
        oss << "vision_tokens hidden size mismatch: expected " << hidden_size_ << ", got "
            << vision_tokens.d;
        throw std::invalid_argument(oss.str());
    }

    if (text_embedding.hidden_size() != hidden_size_) {
        std::ostringstream oss;
        oss << "text_embedding hidden size mismatch: expected " << hidden_size_ << ", got "
            << text_embedding.hidden_size();
        throw std::invalid_argument(oss.str());
    }

    int num_vision = vision_tokens.n;
    int num_text = static_cast<int>(text_token_ids.size());
    int total_tokens = num_vision + num_text;

    MultimodalSequence result;
    result.num_tokens = total_tokens;
    result.hidden_size = hidden_size_;
    result.num_vision_tokens = num_vision;
    result.num_text_tokens = num_text;
    result.data.resize(static_cast<size_t>(total_tokens) * hidden_size_);

    for (int i = 0; i < num_vision; ++i) {
        for (int d = 0; d < hidden_size_; ++d) {
            result.at(i, d) = vision_tokens.at(i, d);
        }
    }

    for (int i = 0; i < num_text; ++i) {
        std::vector<float> text_emb = text_embedding.get_embedding(text_token_ids[i]);
        int token_idx = num_vision + i;
        for (int d = 0; d < hidden_size_; ++d) {
            result.at(token_idx, d) = text_emb[d];
        }
    }

    return result;
}

MultimodalSequence MultimodalEmbedding::concat_vision_text(
    const Tensor2D& vision_tokens, const std::vector<std::vector<float>>& text_embeddings) const {

    if (vision_tokens.d != hidden_size_) {
        std::ostringstream oss;
        oss << "vision_tokens hidden size mismatch: expected " << hidden_size_ << ", got "
            << vision_tokens.d;
        throw std::invalid_argument(oss.str());
    }

    for (const auto& emb : text_embeddings) {
        if (static_cast<int>(emb.size()) != hidden_size_) {
            std::ostringstream oss;
            oss << "text_embedding size mismatch: expected " << hidden_size_ << ", got "
                << emb.size();
            throw std::invalid_argument(oss.str());
        }
    }

    int num_vision = vision_tokens.n;
    int num_text = static_cast<int>(text_embeddings.size());
    int total_tokens = num_vision + num_text;

    MultimodalSequence result;
    result.num_tokens = total_tokens;
    result.hidden_size = hidden_size_;
    result.num_vision_tokens = num_vision;
    result.num_text_tokens = num_text;
    result.data.resize(static_cast<size_t>(total_tokens) * hidden_size_);

    for (int i = 0; i < num_vision; ++i) {
        for (int d = 0; d < hidden_size_; ++d) {
            result.at(i, d) = vision_tokens.at(i, d);
        }
    }

    for (int i = 0; i < num_text; ++i) {
        int token_idx = num_vision + i;
        for (int d = 0; d < hidden_size_; ++d) {
            result.at(token_idx, d) = text_embeddings[i][d];
        }
    }

    return result;
}

MultimodalSequence MultimodalEmbedding::text_only(const std::vector<int>& text_token_ids,
                                                  const TokenEmbedding& text_embedding) const {

    if (text_embedding.hidden_size() != hidden_size_) {
        std::ostringstream oss;
        oss << "text_embedding hidden size mismatch: expected " << hidden_size_ << ", got "
            << text_embedding.hidden_size();
        throw std::invalid_argument(oss.str());
    }

    int num_text = static_cast<int>(text_token_ids.size());

    MultimodalSequence result;
    result.num_tokens = num_text;
    result.hidden_size = hidden_size_;
    result.num_vision_tokens = 0;
    result.num_text_tokens = num_text;
    result.data.resize(static_cast<size_t>(num_text) * hidden_size_);

    for (int i = 0; i < num_text; ++i) {
        std::vector<float> text_emb = text_embedding.get_embedding(text_token_ids[i]);
        for (int d = 0; d < hidden_size_; ++d) {
            result.at(i, d) = text_emb[d];
        }
    }

    return result;
}

} // namespace qwen
