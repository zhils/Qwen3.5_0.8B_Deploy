#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qwen {

// Simple contiguous tensor wrapper for [B, T, C, H, W].
struct Tensor5D {
    int b = 0;
    int t = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    std::vector<float> data;

    Tensor5D() = default;
    Tensor5D(int batch, int time, int channels, int height, int width)
        : b(batch), t(time), c(channels), h(height), w(width),
          data(static_cast<size_t>(batch) * time * channels * height * width, 0.0f) {}

    inline size_t index(int bi, int ti, int ci, int yi, int xi) const {
        return ((((static_cast<size_t>(bi) * t + ti) * c + ci) * h + yi) * w + xi);
    }

    inline float at(int bi, int ti, int ci, int yi, int xi) const {
        return data[index(bi, ti, ci, yi, xi)];
    }

    inline float& at(int bi, int ti, int ci, int yi, int xi) {
        return data[index(bi, ti, ci, yi, xi)];
    }
};

// Output wrapper for [B, N, D], where N = num_patches.
struct Tensor3D {
    int b = 0;
    int n = 0;
    int d = 0;
    std::vector<float> data;

    Tensor3D() = default;
    Tensor3D(int batch, int tokens, int dim)
        : b(batch), n(tokens), d(dim), data(static_cast<size_t>(batch) * tokens * dim, 0.0f) {}

    inline size_t index(int bi, int ni, int di) const {
        return ((static_cast<size_t>(bi) * n + ni) * d + di);
    }

    inline float at(int bi, int ni, int di) const {
        return data[index(bi, ni, di)];
    }

    inline float& at(int bi, int ni, int di) {
        return data[index(bi, ni, di)];
    }
};

class VisionPatchEmbedding {
  public:
    // Qwen3.5-0.8B constraints:
    //   patch_size = 16
    //   temporal_patch = 2
    //   in_channels = 3
    VisionPatchEmbedding(int embed_dim, int in_channels = 3, int patch_size = 16,
                         int temporal_patch = 2, bool use_bias = true);

    // Weight shape: [embed_dim, in_channels, temporal_patch, patch_size, patch_size]
    void set_weights(std::vector<float> weight, std::vector<float> bias = {});

    // Input layout: [B, T, C, H, W], C must equal in_channels.
    // Output layout: [B, N, embed_dim], where:
    //   Nt = T / temporal_patch
    //   Nh = H / patch_size
    //   Nw = W / patch_size
    //   N  = Nt * Nh * Nw
    Tensor3D forward(const Tensor5D& x) const;

    int embed_dim() const {
        return embed_dim_;
    }
    int in_channels() const {
        return in_channels_;
    }
    int patch_size() const {
        return patch_size_;
    }
    int temporal_patch() const {
        return temporal_patch_;
    }

  private:
    inline size_t weight_index(int od, int ic, int kt, int ky, int kx) const {
        const size_t c1 = static_cast<size_t>(in_channels_);
        const size_t t1 = static_cast<size_t>(temporal_patch_);
        const size_t p1 = static_cast<size_t>(patch_size_);
        return (((((static_cast<size_t>(od) * c1 + ic) * t1 + kt) * p1 + ky) * p1) + kx);
    }

    void check_ready() const;

    int embed_dim_;
    int in_channels_;
    int patch_size_;
    int temporal_patch_;
    bool use_bias_;

    std::vector<float> weight_;
    std::vector<float> bias_;
};

} // namespace qwen
