#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cmath>

namespace qwen {

struct Tensor2D {
    int n = 0;
    int d = 0;
    std::vector<float> data;

    Tensor2D() = default;
    Tensor2D(int tokens, int dim)
        : n(tokens), d(dim), data(static_cast<size_t>(tokens) * dim, 0.0f) {}

    inline size_t index(int ni, int di) const {
        return static_cast<size_t>(ni) * d + di;
    }

    inline float at(int ni, int di) const {
        return data[index(ni, di)];
    }

    inline float& at(int ni, int di) {
        return data[index(ni, di)];
    }
};

class VisionLayerNorm {
  public:
    VisionLayerNorm(int hidden_size, float eps = 1e-6f);

    void set_weights(std::vector<float> weight, std::vector<float> bias);

    Tensor2D forward(const Tensor2D& x) const;

    int hidden_size() const {
        return hidden_size_;
    }

  private:
    int hidden_size_;
    float eps_;
    std::vector<float> weight_;
    std::vector<float> bias_;
};

class VisionMultiHeadAttention {
  public:
    VisionMultiHeadAttention(int hidden_size, int num_heads);

    void set_weights(std::vector<float> qkv_weight, std::vector<float> qkv_bias,
                     std::vector<float> proj_weight, std::vector<float> proj_bias);

    Tensor2D forward(const Tensor2D& x) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int num_heads() const {
        return num_heads_;
    }
    int head_dim() const {
        return head_dim_;
    }

  private:
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    std::vector<float> qkv_weight_;
    std::vector<float> qkv_bias_;
    std::vector<float> proj_weight_;
    std::vector<float> proj_bias_;
};

class VisionMLP {
  public:
    VisionMLP(int hidden_size, int intermediate_size);

    void set_weights(std::vector<float> fc1_weight, std::vector<float> fc1_bias,
                     std::vector<float> fc2_weight, std::vector<float> fc2_bias);

    Tensor2D forward(const Tensor2D& x) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int intermediate_size() const {
        return intermediate_size_;
    }

  private:
    static float gelu(float x);

    int hidden_size_;
    int intermediate_size_;

    std::vector<float> fc1_weight_;
    std::vector<float> fc1_bias_;
    std::vector<float> fc2_weight_;
    std::vector<float> fc2_bias_;
};

class VisionBlock {
  public:
    VisionBlock(int hidden_size, int num_heads, int intermediate_size);

    void set_weights(std::vector<float> norm1_weight, std::vector<float> norm1_bias,
                     std::vector<float> qkv_weight, std::vector<float> qkv_bias,
                     std::vector<float> proj_weight, std::vector<float> proj_bias,
                     std::vector<float> norm2_weight, std::vector<float> norm2_bias,
                     std::vector<float> fc1_weight, std::vector<float> fc1_bias,
                     std::vector<float> fc2_weight, std::vector<float> fc2_bias);

    Tensor2D forward(const Tensor2D& x) const;

  private:
    VisionLayerNorm norm1_;
    VisionMultiHeadAttention attn_;
    VisionLayerNorm norm2_;
    VisionMLP mlp_;
};

class VisionTransformer {
  public:
    VisionTransformer(int hidden_size = 768, int num_heads = 12, int intermediate_size = 3072,
                      int depth = 12);

    void set_pos_embed(std::vector<float> pos_embed);

    void set_block_weights(int block_idx, std::vector<float> norm1_weight,
                           std::vector<float> norm1_bias, std::vector<float> qkv_weight,
                           std::vector<float> qkv_bias, std::vector<float> proj_weight,
                           std::vector<float> proj_bias, std::vector<float> norm2_weight,
                           std::vector<float> norm2_bias, std::vector<float> fc1_weight,
                           std::vector<float> fc1_bias, std::vector<float> fc2_weight,
                           std::vector<float> fc2_bias);

    Tensor2D forward(const Tensor2D& x) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int num_heads() const {
        return num_heads_;
    }
    int intermediate_size() const {
        return intermediate_size_;
    }
    int depth() const {
        return depth_;
    }

  private:
    int hidden_size_;
    int num_heads_;
    int intermediate_size_;
    int depth_;

    std::vector<float> pos_embed_;
    std::vector<VisionBlock> blocks_;
};

class VisualMerger {
  public:
    VisualMerger(int in_hidden_size = 768, int out_hidden_size = 1024, int intermediate_size = 3072,
                 int spatial_merge_size = 2);

    void set_weights(std::vector<float> norm_weight, std::vector<float> norm_bias,
                     std::vector<float> fc1_weight, std::vector<float> fc1_bias,
                     std::vector<float> fc2_weight, std::vector<float> fc2_bias);

    Tensor2D forward(const Tensor2D& x, int grid_h, int grid_w) const;

    int in_hidden_size() const {
        return in_hidden_size_;
    }
    int out_hidden_size() const {
        return out_hidden_size_;
    }
    int intermediate_size() const {
        return intermediate_size_;
    }
    int spatial_merge_size() const {
        return spatial_merge_size_;
    }

  private:
    static float gelu(float x);

    int in_hidden_size_;
    int out_hidden_size_;
    int intermediate_size_;
    int spatial_merge_size_;

    VisionLayerNorm norm_;
    std::vector<float> fc1_weight_;
    std::vector<float> fc1_bias_;
    std::vector<float> fc2_weight_;
    std::vector<float> fc2_bias_;
};

} // namespace qwen
