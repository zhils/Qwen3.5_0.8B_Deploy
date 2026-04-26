#include "lm_head.hpp"
#include "error_handling.hpp"
#include <algorithm>
#include <queue>

namespace qwen {

LMHead::LMHead(int hidden_size, int vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size) {
    QWEN_CHECK_POSITIVE(hidden_size, "hidden_size");
    QWEN_CHECK_POSITIVE(vocab_size, "vocab_size");
}

void LMHead::set_weight(const std::vector<float>& weight) {
    size_t expected_size = static_cast<size_t>(vocab_size_) * hidden_size_;
    QWEN_CHECK_WEIGHT_SIZE(expected_size, weight.size(), "LMHead");
    weight_ = weight;
}

void LMHead::check_ready() const {
    QWEN_CHECK_NOT_EMPTY(weight_, "LMHead");
}

std::vector<float> LMHead::forward(const std::vector<float>& input) const {
    check_ready();
    QWEN_CHECK_DIM(hidden_size_, input.size());

    std::vector<float> logits(vocab_size_, 0.0f);

    for (int v = 0; v < vocab_size_; ++v) {
        float dot = 0.0f;
        for (int h = 0; h < hidden_size_; ++h) {
            dot += weight_[v * hidden_size_ + h] * input[h];
        }
        logits[v] = dot;
    }

    return logits;
}

std::vector<std::pair<int, float>> LMHead::get_top_tokens(const std::vector<float>& logits,
                                                          int top_k) const {
    QWEN_CHECK_DIM(vocab_size_, logits.size());
    QWEN_CHECK(top_k > 0, "top_k must be positive");

    auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    };
    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, decltype(cmp)>
        pq(cmp);

    for (int i = 0; i < vocab_size_; ++i) {
        pq.push({i, logits[i]});
        if (static_cast<int>(pq.size()) > top_k) {
            pq.pop();
        }
    }

    std::vector<std::pair<int, float>> result;
    while (!pq.empty()) {
        result.push_back(pq.top());
        pq.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

} // namespace qwen
