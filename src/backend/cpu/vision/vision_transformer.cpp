#include "vision_transformer.hpp"
#include <sstream>
#include <algorithm>
#include <numeric>

namespace qwen {

VisionLayerNorm::VisionLayerNorm(int hidden_size, float eps)
    : hidden_size_(hidden_size), eps_(eps) {
    if (hidden_size <= 0) {
        throw std::invalid_argument("hidden_size must be > 0");
    }
}

void VisionLayerNorm::set_weights(std::vector<float> weight, std::vector<float> bias) {
    if (weight.size() != static_cast<size_t>(hidden_size_)) {
        std::ostringstream oss;
        oss << "weight size mismatch, expected " << hidden_size_ << ", got " << weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (bias.size() != static_cast<size_t>(hidden_size_)) {
        std::ostringstream oss;
        oss << "bias size mismatch, expected " << hidden_size_ << ", got " << bias.size();
        throw std::invalid_argument(oss.str());
    }
    weight_ = std::move(weight);
    bias_ = std::move(bias);
}

Tensor2D VisionLayerNorm::forward(const Tensor2D& x) const {
    if (weight_.empty()) {
        throw std::runtime_error("LayerNorm weights not loaded");
    }
    if (x.d != hidden_size_) {
        throw std::invalid_argument("input dim mismatch");
    }

    Tensor2D out(x.n, x.d);

    for (int i = 0; i < x.n; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < x.d; ++j) {
            mean += x.at(i, j);
        }
        mean /= x.d;

        float var = 0.0f;
        for (int j = 0; j < x.d; ++j) {
            float diff = x.at(i, j) - mean;
            var += diff * diff;
        }
        var /= x.d;

        float inv_std = 1.0f / std::sqrt(var + eps_);

        for (int j = 0; j < x.d; ++j) {
            float normalized = (x.at(i, j) - mean) * inv_std;
            out.at(i, j) = normalized * weight_[j] + bias_[j];
        }
    }

    return out;
}

VisionMultiHeadAttention::VisionMultiHeadAttention(int hidden_size, int num_heads)
    : hidden_size_(hidden_size), num_heads_(num_heads), head_dim_(hidden_size / num_heads) {
    if (hidden_size <= 0 || num_heads <= 0) {
        throw std::invalid_argument("hidden_size and num_heads must be > 0");
    }
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }
}

void VisionMultiHeadAttention::set_weights(std::vector<float> qkv_weight,
                                           std::vector<float> qkv_bias,
                                           std::vector<float> proj_weight,
                                           std::vector<float> proj_bias) {

    size_t expected_qkv_weight = static_cast<size_t>(3) * hidden_size_ * hidden_size_;
    if (qkv_weight.size() != expected_qkv_weight) {
        std::ostringstream oss;
        oss << "qkv_weight size mismatch, expected " << expected_qkv_weight << ", got "
            << qkv_weight.size();
        throw std::invalid_argument(oss.str());
    }

    size_t expected_proj_weight = static_cast<size_t>(hidden_size_) * hidden_size_;
    if (proj_weight.size() != expected_proj_weight) {
        std::ostringstream oss;
        oss << "proj_weight size mismatch, expected " << expected_proj_weight << ", got "
            << proj_weight.size();
        throw std::invalid_argument(oss.str());
    }

    qkv_weight_ = std::move(qkv_weight);
    qkv_bias_ = std::move(qkv_bias);
    proj_weight_ = std::move(proj_weight);
    proj_bias_ = std::move(proj_bias);
}

Tensor2D VisionMultiHeadAttention::forward(const Tensor2D& x) const {
    if (qkv_weight_.empty()) {
        throw std::runtime_error("MultiHeadAttention weights not loaded");
    }
    if (x.d != hidden_size_) {
        throw std::invalid_argument("input dim mismatch");
    }

    int n = x.n;
    int h = num_heads_;
    int d = head_dim_;

    std::vector<float> qkv(n * 3 * hidden_size_, 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 3 * hidden_size_; ++k) {
            float sum = qkv_bias_.empty() ? 0.0f : qkv_bias_[k];
            for (int j = 0; j < hidden_size_; ++j) {
                sum += x.at(i, j) * qkv_weight_[k * hidden_size_ + j];
            }
            qkv[i * 3 * hidden_size_ + k] = sum;
        }
    }

    std::vector<float> attn_out(n * hidden_size_, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int head = 0; head < h; ++head) {
        std::vector<float> scores(n * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float score = 0.0f;
                for (int dk = 0; dk < d; ++dk) {
                    int q_idx = i * 3 * hidden_size_ + head * d + dk;
                    int k_idx = j * 3 * hidden_size_ + hidden_size_ + head * d + dk;
                    score += qkv[q_idx] * qkv[k_idx];
                }
                scores[i * n + j] = score * scale;
            }
        }

        std::vector<float> attn_weights(n * n);
        for (int i = 0; i < n; ++i) {
            float max_score = scores[i * n];
            for (int j = 1; j < n; ++j) {
                max_score = std::max(max_score, scores[i * n + j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < n; ++j) {
                attn_weights[i * n + j] = std::exp(scores[i * n + j] - max_score);
                sum_exp += attn_weights[i * n + j];
            }

            for (int j = 0; j < n; ++j) {
                attn_weights[i * n + j] /= sum_exp;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int dk = 0; dk < d; ++dk) {
                float val = 0.0f;
                for (int j = 0; j < n; ++j) {
                    int v_idx = j * 3 * hidden_size_ + 2 * hidden_size_ + head * d + dk;
                    val += attn_weights[i * n + j] * qkv[v_idx];
                }
                attn_out[i * hidden_size_ + head * d + dk] = val;
            }
        }
    }

    Tensor2D out(n, hidden_size_);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            float sum = proj_bias_.empty() ? 0.0f : proj_bias_[j];
            for (int k = 0; k < hidden_size_; ++k) {
                sum += attn_out[i * hidden_size_ + k] * proj_weight_[j * hidden_size_ + k];
            }
            out.at(i, j) = sum;
        }
    }

    return out;
}

VisionMLP::VisionMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size) {
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("hidden_size and intermediate_size must be > 0");
    }
}

float VisionMLP::gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void VisionMLP::set_weights(std::vector<float> fc1_weight, std::vector<float> fc1_bias,
                            std::vector<float> fc2_weight, std::vector<float> fc2_bias) {

    size_t expected_fc1 = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t expected_fc2 = static_cast<size_t>(hidden_size_) * intermediate_size_;

    if (fc1_weight.size() != expected_fc1) {
        std::ostringstream oss;
        oss << "fc1_weight size mismatch, expected " << expected_fc1 << ", got "
            << fc1_weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (fc2_weight.size() != expected_fc2) {
        std::ostringstream oss;
        oss << "fc2_weight size mismatch, expected " << expected_fc2 << ", got "
            << fc2_weight.size();
        throw std::invalid_argument(oss.str());
    }

    fc1_weight_ = std::move(fc1_weight);
    fc1_bias_ = std::move(fc1_bias);
    fc2_weight_ = std::move(fc2_weight);
    fc2_bias_ = std::move(fc2_bias);
}

Tensor2D VisionMLP::forward(const Tensor2D& x) const {
    if (fc1_weight_.empty() || fc2_weight_.empty()) {
        throw std::runtime_error("MLP weights not loaded");
    }
    if (x.d != hidden_size_) {
        throw std::invalid_argument("input dim mismatch");
    }

    int n = x.n;

    std::vector<float> hidden(n * intermediate_size_);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < intermediate_size_; ++j) {
            float sum = fc1_bias_.empty() ? 0.0f : fc1_bias_[j];
            for (int k = 0; k < hidden_size_; ++k) {
                sum += x.at(i, k) * fc1_weight_[j * hidden_size_ + k];
            }
            hidden[i * intermediate_size_ + j] = gelu(sum);
        }
    }

    Tensor2D out(n, hidden_size_);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            float sum = fc2_bias_.empty() ? 0.0f : fc2_bias_[j];
            for (int k = 0; k < intermediate_size_; ++k) {
                sum += hidden[i * intermediate_size_ + k] * fc2_weight_[j * intermediate_size_ + k];
            }
            out.at(i, j) = sum;
        }
    }

    return out;
}

VisionBlock::VisionBlock(int hidden_size, int num_heads, int intermediate_size)
    : norm1_(hidden_size), attn_(hidden_size, num_heads), norm2_(hidden_size),
      mlp_(hidden_size, intermediate_size) {}

void VisionBlock::set_weights(std::vector<float> norm1_weight, std::vector<float> norm1_bias,
                              std::vector<float> qkv_weight, std::vector<float> qkv_bias,
                              std::vector<float> proj_weight, std::vector<float> proj_bias,
                              std::vector<float> norm2_weight, std::vector<float> norm2_bias,
                              std::vector<float> fc1_weight, std::vector<float> fc1_bias,
                              std::vector<float> fc2_weight, std::vector<float> fc2_bias) {

    norm1_.set_weights(std::move(norm1_weight), std::move(norm1_bias));
    attn_.set_weights(std::move(qkv_weight), std::move(qkv_bias), std::move(proj_weight),
                      std::move(proj_bias));
    norm2_.set_weights(std::move(norm2_weight), std::move(norm2_bias));
    mlp_.set_weights(std::move(fc1_weight), std::move(fc1_bias), std::move(fc2_weight),
                     std::move(fc2_bias));
}

Tensor2D VisionBlock::forward(const Tensor2D& x) const {
    Tensor2D norm1_out = norm1_.forward(x);

    Tensor2D attn_out = attn_.forward(norm1_out);

    Tensor2D residual1(x.n, x.d);
    for (size_t i = 0; i < x.data.size(); ++i) {
        residual1.data[i] = x.data[i] + attn_out.data[i];
    }

    Tensor2D norm2_out = norm2_.forward(residual1);

    Tensor2D mlp_out = mlp_.forward(norm2_out);

    Tensor2D out(x.n, x.d);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = residual1.data[i] + mlp_out.data[i];
    }

    return out;
}

VisionTransformer::VisionTransformer(int hidden_size, int num_heads, int intermediate_size,
                                     int depth)
    : hidden_size_(hidden_size), num_heads_(num_heads), intermediate_size_(intermediate_size),
      depth_(depth) {

    if (depth <= 0) {
        throw std::invalid_argument("depth must be > 0");
    }

    blocks_.reserve(depth);
    for (int i = 0; i < depth; ++i) {
        blocks_.emplace_back(hidden_size, num_heads, intermediate_size);
    }
}

void VisionTransformer::set_pos_embed(std::vector<float> pos_embed) {
    pos_embed_ = std::move(pos_embed);
}

void VisionTransformer::set_block_weights(
    int block_idx, std::vector<float> norm1_weight, std::vector<float> norm1_bias,
    std::vector<float> qkv_weight, std::vector<float> qkv_bias, std::vector<float> proj_weight,
    std::vector<float> proj_bias, std::vector<float> norm2_weight, std::vector<float> norm2_bias,
    std::vector<float> fc1_weight, std::vector<float> fc1_bias, std::vector<float> fc2_weight,
    std::vector<float> fc2_bias) {

    if (block_idx < 0 || block_idx >= depth_) {
        throw std::out_of_range("block_idx out of range");
    }

    blocks_[block_idx].set_weights(
        std::move(norm1_weight), std::move(norm1_bias), std::move(qkv_weight), std::move(qkv_bias),
        std::move(proj_weight), std::move(proj_bias), std::move(norm2_weight),
        std::move(norm2_bias), std::move(fc1_weight), std::move(fc1_bias), std::move(fc2_weight),
        std::move(fc2_bias));
}

Tensor2D VisionTransformer::forward(const Tensor2D& x) const {
    if (x.d != hidden_size_) {
        throw std::invalid_argument("input dim mismatch");
    }

    Tensor2D out = x;

    if (!pos_embed_.empty()) {
        size_t needed_size = static_cast<size_t>(x.n) * hidden_size_;
        if (pos_embed_.size() < needed_size) {
            throw std::invalid_argument("pos_embed size too small");
        }
        for (size_t i = 0; i < out.data.size(); ++i) {
            out.data[i] += pos_embed_[i];
        }
    }

    for (int i = 0; i < depth_; ++i) {
        out = blocks_[i].forward(out);
    }

    return out;
}

VisualMerger::VisualMerger(int in_hidden_size, int out_hidden_size, int intermediate_size,
                           int spatial_merge_size)
    : in_hidden_size_(in_hidden_size), out_hidden_size_(out_hidden_size),
      intermediate_size_(intermediate_size), spatial_merge_size_(spatial_merge_size),
      norm_(in_hidden_size) {

    if (in_hidden_size <= 0 || out_hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("hidden sizes must be > 0");
    }
    if (spatial_merge_size <= 0) {
        throw std::invalid_argument("spatial_merge_size must be > 0");
    }
}

void VisualMerger::set_weights(std::vector<float> norm_weight, std::vector<float> norm_bias,
                               std::vector<float> fc1_weight, std::vector<float> fc1_bias,
                               std::vector<float> fc2_weight, std::vector<float> fc2_bias) {

    norm_.set_weights(std::move(norm_weight), std::move(norm_bias));

    size_t expected_fc1_weight = static_cast<size_t>(intermediate_size_) * intermediate_size_;
    if (fc1_weight.size() != expected_fc1_weight) {
        std::ostringstream oss;
        oss << "fc1_weight size mismatch, expected " << expected_fc1_weight << ", got "
            << fc1_weight.size();
        throw std::invalid_argument(oss.str());
    }

    size_t expected_fc1_bias = static_cast<size_t>(intermediate_size_);
    if (!fc1_bias.empty() && fc1_bias.size() != expected_fc1_bias) {
        std::ostringstream oss;
        oss << "fc1_bias size mismatch, expected " << expected_fc1_bias << ", got "
            << fc1_bias.size();
        throw std::invalid_argument(oss.str());
    }

    size_t expected_fc2_weight = static_cast<size_t>(out_hidden_size_) * intermediate_size_;
    if (fc2_weight.size() != expected_fc2_weight) {
        std::ostringstream oss;
        oss << "fc2_weight size mismatch, expected " << expected_fc2_weight << ", got "
            << fc2_weight.size();
        throw std::invalid_argument(oss.str());
    }

    size_t expected_fc2_bias = static_cast<size_t>(out_hidden_size_);
    if (!fc2_bias.empty() && fc2_bias.size() != expected_fc2_bias) {
        std::ostringstream oss;
        oss << "fc2_bias size mismatch, expected " << expected_fc2_bias << ", got "
            << fc2_bias.size();
        throw std::invalid_argument(oss.str());
    }

    fc1_weight_ = std::move(fc1_weight);
    fc1_bias_ = std::move(fc1_bias);
    fc2_weight_ = std::move(fc2_weight);
    fc2_bias_ = std::move(fc2_bias);
}

float VisualMerger::gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

Tensor2D VisualMerger::forward(const Tensor2D& x, int grid_h, int grid_w) const {
    if (x.d != in_hidden_size_) {
        throw std::invalid_argument("input dim mismatch");
    }

    Tensor2D norm_out = norm_.forward(x);

    int merge_h = grid_h / spatial_merge_size_;
    int merge_w = grid_w / spatial_merge_size_;
    int merged_dim = in_hidden_size_ * spatial_merge_size_ * spatial_merge_size_;

    int merged_tokens = merge_h * merge_w;
    std::vector<float> merged(merged_tokens * merged_dim, 0.0f);

    for (int mh = 0; mh < merge_h; ++mh) {
        for (int mw = 0; mw < merge_w; ++mw) {
            int merged_idx = mh * merge_w + mw;
            for (int sh = 0; sh < spatial_merge_size_; ++sh) {
                for (int sw = 0; sw < spatial_merge_size_; ++sw) {
                    int orig_h = mh * spatial_merge_size_ + sh;
                    int orig_w = mw * spatial_merge_size_ + sw;
                    int orig_idx = orig_h * grid_w + orig_w;

                    int offset = (sh * spatial_merge_size_ + sw) * in_hidden_size_;
                    for (int d = 0; d < in_hidden_size_; ++d) {
                        merged[merged_idx * merged_dim + offset + d] = norm_out.at(orig_idx, d);
                    }
                }
            }
        }
    }

    std::vector<float> hidden(merged_tokens * intermediate_size_);
    for (int i = 0; i < merged_tokens; ++i) {
        for (int j = 0; j < intermediate_size_; ++j) {
            float sum = fc1_bias_.empty() ? 0.0f : fc1_bias_[j];
            for (int k = 0; k < merged_dim; ++k) {
                sum += merged[i * merged_dim + k] * fc1_weight_[j * merged_dim + k];
            }
            hidden[i * intermediate_size_ + j] = gelu(sum);
        }
    }

    Tensor2D out(merged_tokens, out_hidden_size_);
    for (int i = 0; i < merged_tokens; ++i) {
        for (int j = 0; j < out_hidden_size_; ++j) {
            float sum = fc2_bias_.empty() ? 0.0f : fc2_bias_[j];
            for (int k = 0; k < intermediate_size_; ++k) {
                sum += hidden[i * intermediate_size_ + k] * fc2_weight_[j * intermediate_size_ + k];
            }
            out.at(i, j) = sum;
        }
    }

    return out;
}

} // namespace qwen
