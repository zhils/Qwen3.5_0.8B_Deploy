#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace qwen {

enum class SamplingStrategy { GREEDY, TOP_K, TOP_P, TEMPERATURE };

class Sampler {
  public:
    Sampler(int vocab_size = 248320, SamplingStrategy strategy = SamplingStrategy::GREEDY,
            float temperature = 1.0f, int top_k = 50, float top_p = 0.9f, int seed = 42);

    int sample(const std::vector<float>& logits) const;

    std::vector<int> sample_batch(const std::vector<float>& logits, int batch_size) const;

    void set_temperature(float temp) {
        temperature_ = temp;
    }
    void set_top_k(int k) {
        top_k_ = k;
    }
    void set_top_p(float p) {
        top_p_ = p;
    }
    void set_strategy(SamplingStrategy strategy) {
        strategy_ = strategy;
    }

    float temperature() const {
        return temperature_;
    }
    int top_k() const {
        return top_k_;
    }
    float top_p() const {
        return top_p_;
    }
    SamplingStrategy strategy() const {
        return strategy_;
    }

  private:
    int sample_greedy(const std::vector<float>& logits) const;
    int sample_top_k(const std::vector<float>& logits) const;
    int sample_top_p(const std::vector<float>& logits) const;
    int sample_temperature(const std::vector<float>& logits) const;

    void softmax(std::vector<float>& logits) const;
    void apply_temperature(std::vector<float>& logits) const;

    int vocab_size_;
    SamplingStrategy strategy_;
    float temperature_;
    int top_k_;
    float top_p_;
    mutable std::mt19937 rng_;
};

} // namespace qwen
