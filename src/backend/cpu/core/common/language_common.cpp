#include "language_common.hpp"
#include "error_handling.hpp"
#include <sstream>

namespace qwen {

RMSNorm::RMSNorm(int hidden_size, float eps) : hidden_size_(hidden_size), eps_(eps) {
    QWEN_CHECK_POSITIVE(hidden_size, "hidden_size");
    QWEN_CHECK_POSITIVE(static_cast<int>(eps), "eps");
}

void RMSNorm::set_weight(std::vector<float> weight) {
    QWEN_CHECK_WEIGHT_SIZE(hidden_size_, weight.size(), "RMSNorm");
    weight_ = std::move(weight);
}

void RMSNorm::check_ready() const {
    QWEN_CHECK_NOT_EMPTY(weight_, "RMSNorm");
}

std::vector<float> RMSNorm::forward(const std::vector<float>& input) const {
    check_ready();
    QWEN_CHECK_DIM(hidden_size_, input.size());

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size_; ++i) {
        sum_sq += input[i] * input[i];
    }

    float rms = std::sqrt(sum_sq / hidden_size_ + eps_);

    std::vector<float> output(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        output[i] = (input[i] / rms) * (1.0f + weight_[i]);
    }

    return output;
}

void RMSNorm::forward_inplace(std::vector<float>& input) const {
    check_ready();
    QWEN_CHECK_DIM(hidden_size_, input.size());

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size_; ++i) {
        sum_sq += input[i] * input[i];
    }

    float rms = std::sqrt(sum_sq / hidden_size_ + eps_);

    for (int i = 0; i < hidden_size_; ++i) {
        input[i] = (input[i] / rms) * (1.0f + weight_[i]);
    }
}

void RMSNorm::forward_batch(const float* input, float* output, int batch_size) const {
    check_ready();
    QWEN_CHECK_NOT_NULL(input, "input");
    QWEN_CHECK_NOT_NULL(output, "output");
    QWEN_CHECK_POSITIVE(batch_size, "batch_size");

    for (int b = 0; b < batch_size; ++b) {
        const float* in_ptr = input + b * hidden_size_;
        float* out_ptr = output + b * hidden_size_;

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_size_; ++i) {
            sum_sq += in_ptr[i] * in_ptr[i];
        }

        float rms = std::sqrt(sum_sq / hidden_size_ + eps_);

        for (int i = 0; i < hidden_size_; ++i) {
            out_ptr[i] = (in_ptr[i] / rms) * (1.0f + weight_[i]);
        }
    }
}

} // namespace qwen
