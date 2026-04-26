#include "vision_patch_embedding.hpp"

#include <sstream>

namespace qwen {

VisionPatchEmbedding::VisionPatchEmbedding(int embed_dim, int in_channels, int patch_size,
                                           int temporal_patch, bool use_bias)
    : embed_dim_(embed_dim), in_channels_(in_channels), patch_size_(patch_size),
      temporal_patch_(temporal_patch), use_bias_(use_bias) {
    if (embed_dim_ <= 0) {
        throw std::invalid_argument("embed_dim must be > 0");
    }
    if (in_channels_ != 3) {
        throw std::invalid_argument("Qwen3.5 vision patch embedding expects in_channels=3");
    }
    if (patch_size_ != 16) {
        throw std::invalid_argument("Qwen3.5 vision patch embedding expects patch_size=16");
    }
    if (temporal_patch_ != 2) {
        throw std::invalid_argument("Qwen3.5 vision patch embedding expects temporal_patch=2");
    }
}

void VisionPatchEmbedding::set_weights(std::vector<float> weight, std::vector<float> bias) {
    const size_t expected_weight = static_cast<size_t>(embed_dim_) * in_channels_ *
                                   temporal_patch_ * patch_size_ * patch_size_;
    if (weight.size() != expected_weight) {
        std::ostringstream oss;
        oss << "weight size mismatch, expected " << expected_weight << ", got " << weight.size();
        throw std::invalid_argument(oss.str());
    }
    if (use_bias_) {
        if (bias.empty()) {
            bias.resize(static_cast<size_t>(embed_dim_), 0.0f);
        } else if (bias.size() != static_cast<size_t>(embed_dim_)) {
            std::ostringstream oss;
            oss << "bias size mismatch, expected " << embed_dim_ << ", got " << bias.size();
            throw std::invalid_argument(oss.str());
        }
    }
    weight_ = std::move(weight);
    bias_ = std::move(bias);
}

void VisionPatchEmbedding::check_ready() const {
    if (weight_.empty()) {
        throw std::runtime_error("weights are not loaded");
    }
    if (use_bias_ && bias_.size() != static_cast<size_t>(embed_dim_)) {
        throw std::runtime_error("bias is not loaded correctly");
    }
}

Tensor3D VisionPatchEmbedding::forward(const Tensor5D& x) const {
    check_ready();
    if (x.c != in_channels_) {
        throw std::invalid_argument("input channels mismatch");
    }
    if (x.t % temporal_patch_ != 0) {
        throw std::invalid_argument("input T must be divisible by temporal_patch (2)");
    }
    if (x.h % patch_size_ != 0 || x.w % patch_size_ != 0) {
        throw std::invalid_argument("input H/W must be divisible by patch_size (16)");
    }

    const int nt = x.t / temporal_patch_;
    const int nh = x.h / patch_size_;
    const int nw = x.w / patch_size_;
    const int n = nt * nh * nw;

    Tensor3D out(x.b, n, embed_dim_);

    for (int bi = 0; bi < x.b; ++bi) {
        for (int tt = 0; tt < nt; ++tt) {
            for (int yy = 0; yy < nh; ++yy) {
                for (int xx = 0; xx < nw; ++xx) {
                    const int token_idx = (tt * nh + yy) * nw + xx;
                    for (int od = 0; od < embed_dim_; ++od) {
                        float acc = use_bias_ ? bias_[static_cast<size_t>(od)] : 0.0f;
                        for (int ic = 0; ic < in_channels_; ++ic) {
                            for (int kt = 0; kt < temporal_patch_; ++kt) {
                                const int t_in = tt * temporal_patch_ + kt;
                                for (int ky = 0; ky < patch_size_; ++ky) {
                                    const int y_in = yy * patch_size_ + ky;
                                    for (int kx = 0; kx < patch_size_; ++kx) {
                                        const int x_in = xx * patch_size_ + kx;
                                        const float xv = x.at(bi, t_in, ic, y_in, x_in);
                                        const float wv = weight_[weight_index(od, ic, kt, ky, kx)];
                                        acc += xv * wv;
                                    }
                                }
                            }
                        }
                        out.at(bi, token_idx, od) = acc;
                    }
                }
            }
        }
    }

    return out;
}

} // namespace qwen
