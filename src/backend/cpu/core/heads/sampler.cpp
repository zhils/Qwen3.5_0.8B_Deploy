#include "sampler.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>

namespace qwen {

Sampler::Sampler(int vocab_size, SamplingStrategy strategy, float temperature, int top_k,
                 float top_p, int seed)
    : vocab_size_(vocab_size), strategy_(strategy), temperature_(temperature), top_k_(top_k),
      top_p_(top_p), rng_(seed) {
    if (vocab_size <= 0) {
        throw std::invalid_argument("vocab_size must be > 0");
    }
    if (temperature <= 0.0f) {
        throw std::invalid_argument("temperature must be > 0");
    }
}

void Sampler::softmax(std::vector<float>& logits) const {
    float max_val = *std::max_element(logits.begin(), logits.end());

    for (auto& logit : logits) {
        logit = std::exp(logit - max_val);
    }

    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    for (auto& logit : logits) {
        logit /= sum;
    }
}

void Sampler::apply_temperature(std::vector<float>& logits) const {
    if (temperature_ != 1.0f) {
        for (auto& logit : logits) {
            logit /= temperature_;
        }
    }
}

int Sampler::sample_greedy(const std::vector<float>& logits) const {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int Sampler::sample_top_k(const std::vector<float>& logits) const {
    std::vector<std::pair<int, float>> indexed_logits(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        indexed_logits[i] = {i, logits[i]};
    }

    auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    };
    std::partial_sort(indexed_logits.begin(), indexed_logits.begin() + top_k_, indexed_logits.end(),
                      cmp);

    std::vector<float> top_k_logits(top_k_);
    for (int i = 0; i < top_k_; ++i) {
        top_k_logits[i] = indexed_logits[i].second;
    }

    softmax(top_k_logits);

    std::discrete_distribution<int> dist(top_k_logits.begin(), top_k_logits.end());
    return indexed_logits[dist(rng_)].first;
}

int Sampler::sample_top_p(const std::vector<float>& logits) const {
    std::vector<std::pair<int, float>> indexed_logits(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        indexed_logits[i] = {i, logits[i]};
    }

    auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    };
    std::sort(indexed_logits.begin(), indexed_logits.end(), cmp);

    std::vector<float> sorted_logits(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        sorted_logits[i] = indexed_logits[i].second;
    }

    softmax(sorted_logits);

    float cumulative_prob = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < vocab_size_; ++i) {
        cumulative_prob += sorted_logits[i];
        last_idx = i;
        if (cumulative_prob >= top_p_) {
            break;
        }
    }

    std::vector<float> filtered_probs(last_idx + 1);
    std::vector<int> filtered_indices(last_idx + 1);
    for (int i = 0; i <= last_idx; ++i) {
        filtered_probs[i] = sorted_logits[i];
        filtered_indices[i] = indexed_logits[i].first;
    }

    float prob_sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
    for (auto& p : filtered_probs) {
        p /= prob_sum;
    }

    std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
    return filtered_indices[dist(rng_)];
}

int Sampler::sample_temperature(const std::vector<float>& logits) const {
    std::vector<float> scaled_logits(logits.begin(), logits.end());
    apply_temperature(scaled_logits);
    softmax(scaled_logits);

    std::discrete_distribution<int> dist(scaled_logits.begin(), scaled_logits.end());
    return dist(rng_);
}

int Sampler::sample(const std::vector<float>& logits) const {
    if (static_cast<int>(logits.size()) != vocab_size_) {
        std::ostringstream oss;
        oss << "Logits size mismatch: expected " << vocab_size_ << ", got " << logits.size();
        throw std::invalid_argument(oss.str());
    }

    switch (strategy_) {
    case SamplingStrategy::GREEDY:
        return sample_greedy(logits);
    case SamplingStrategy::TOP_K:
        return sample_top_k(logits);
    case SamplingStrategy::TOP_P:
        return sample_top_p(logits);
    case SamplingStrategy::TEMPERATURE:
        return sample_temperature(logits);
    default:
        return sample_greedy(logits);
    }
}

std::vector<int> Sampler::sample_batch(const std::vector<float>& logits, int batch_size) const {
    std::vector<int> tokens(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        tokens[i] = sample(logits);
    }
    return tokens;
}

} // namespace qwen
