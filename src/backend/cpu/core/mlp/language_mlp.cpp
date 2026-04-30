#include "language_mlp.hpp"
#include <cmath>
#include <sstream>

namespace qwen {

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

MLP::MLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size) {
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("hidden_size and intermediate_size must be > 0");
    }
}

void MLP::set_weights(std::vector<float> gate_proj_weight, std::vector<float> up_proj_weight,
                      std::vector<float> down_proj_weight) {
    size_t expected_gate = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t expected_down = static_cast<size_t>(hidden_size_) * intermediate_size_;

    if (gate_proj_weight.size() != expected_gate) {
        std::ostringstream oss;
        oss << "gate_proj weight size mismatch: expected " << expected_gate << ", got "
            << gate_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (up_proj_weight.size() != expected_gate) {
        std::ostringstream oss;
        oss << "up_proj weight size mismatch: expected " << expected_gate << ", got "
            << up_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (down_proj_weight.size() != expected_down) {
        std::ostringstream oss;
        oss << "down_proj weight size mismatch: expected " << expected_down << ", got "
            << down_proj_weight.size();
        throw std::invalid_argument(oss.str());
    }

    gate_proj_weight_ = std::move(gate_proj_weight);
    up_proj_weight_ = std::move(up_proj_weight);
    down_proj_weight_ = std::move(down_proj_weight);
}

void MLP::check_ready() const {
    if (gate_proj_weight_.empty() || up_proj_weight_.empty() || down_proj_weight_.empty()) {
        throw std::runtime_error("MLP weights not set");
    }
}

std::vector<float> MLP::forward(const std::vector<float>& input) const {
    check_ready();

    if (static_cast<int>(input.size()) != hidden_size_) {
        std::ostringstream oss;
        oss << "MLP input size mismatch: expected " << hidden_size_ << ", got " << input.size();
        throw std::invalid_argument(oss.str());
    }

    std::vector<float> gate(intermediate_size_);
    std::vector<float> up(intermediate_size_);

    for (int i = 0; i < intermediate_size_; ++i) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < hidden_size_; ++j) {
            float x = input[j];
            g += gate_proj_weight_[i * hidden_size_ + j] * x;
            u += up_proj_weight_[i * hidden_size_ + j] * x;
        }
        gate[i] = silu(g);
        up[i] = u;
    }

    std::vector<float> hidden(intermediate_size_);
    for (int i = 0; i < intermediate_size_; ++i) {
        hidden[i] = gate[i] * up[i];
    }

    std::vector<float> output(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size_; ++j) {
            sum += down_proj_weight_[i * intermediate_size_ + j] * hidden[j];
        }
        output[i] = sum;
    }

    return output;
}

void MLP::forward_batch(const float* input, float* output, int batch_size) const {
    check_ready();

    for (int b = 0; b < batch_size; ++b) {
        const float* in_ptr = input + b * hidden_size_;
        float* out_ptr = output + b * hidden_size_;

        std::vector<float> gate(intermediate_size_);
        std::vector<float> up(intermediate_size_);

        for (int i = 0; i < intermediate_size_; ++i) {
            float g = 0.0f, u = 0.0f;
            for (int j = 0; j < hidden_size_; ++j) {
                float x = in_ptr[j];
                g += gate_proj_weight_[i * hidden_size_ + j] * x;
                u += up_proj_weight_[i * hidden_size_ + j] * x;
            }
            gate[i] = silu(g);
            up[i] = u;
        }

        std::vector<float> hidden(intermediate_size_);
        for (int i = 0; i < intermediate_size_; ++i) {
            hidden[i] = gate[i] * up[i];
        }

        for (int i = 0; i < hidden_size_; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < intermediate_size_; ++j) {
                sum += down_proj_weight_[i * intermediate_size_ + j] * hidden[j];
            }
            out_ptr[i] = sum;
        }
    }
}

} // namespace qwen
