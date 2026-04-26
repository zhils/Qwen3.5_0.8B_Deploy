#pragma once

#include <algorithm>
#include <vector>

namespace qwen {

struct KVCache {
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<int> layer_lengths;
    int num_layers = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int max_seq_len = 0;

    void reset(int nl, int nkh, int hd, int max_len) {
        num_layers = nl;
        num_kv_heads = nkh;
        head_dim = hd;
        max_seq_len = max_len;
        layer_lengths.assign(num_layers, 0);// 每一层记录长度

        size_t cache_size = static_cast<size_t>(num_layers) * max_seq_len * num_kv_heads * head_dim;
        k_cache.assign(cache_size, 0.0f);
        v_cache.assign(cache_size, 0.0f);
    }

    void clear() {
        std::fill(k_cache.begin(), k_cache.end(), 0.0f);
        std::fill(v_cache.begin(), v_cache.end(), 0.0f);
        std::fill(layer_lengths.begin(), layer_lengths.end(), 0);
    }

    void append(int layer_idx, const float* k, const float* v, int seq_len = 1) {
        int& layer_len = layer_lengths[layer_idx];
        size_t offset = static_cast<size_t>(layer_idx) * max_seq_len * num_kv_heads * head_dim
                      + static_cast<size_t>(layer_len) * num_kv_heads * head_dim;

        std::copy(k, k + seq_len * num_kv_heads * head_dim, k_cache.begin() + offset);
        std::copy(v, v + seq_len * num_kv_heads * head_dim, v_cache.begin() + offset);

        layer_len += seq_len;
    }

    int length(int layer_idx) const { return layer_lengths[layer_idx]; }

    const float* get_k(int layer_idx) const {
        return k_cache.data() + static_cast<size_t>(layer_idx) * max_seq_len * num_kv_heads * head_dim;
    }

    const float* get_v(int layer_idx) const {
        return v_cache.data() + static_cast<size_t>(layer_idx) * max_seq_len * num_kv_heads * head_dim;
    }
};

class FullAttention {
public:
    FullAttention(
        int hidden_size = 1024,
        int num_heads = 8,
        int num_kv_heads = 2,
        int q_head_dim = 512,
        int kv_head_dim = 256,
        float rope_theta = 10000000.0f,
        float partial_rotary = 0.25f
    );

    void set_weights(
        std::vector<float> q_proj_weight,
        std::vector<float> k_proj_weight,
        std::vector<float> v_proj_weight,
        std::vector<float> q_norm_weight,
        std::vector<float> k_norm_weight,
        std::vector<float> o_proj_weight
    );

    std::vector<float> forward(
        const std::vector<float>& input,
        KVCache& kv_cache,
        int layer_idx,
        int position
    ) const;

    std::vector<float> forward_sequence(
        const std::vector<float>& input,
        int seq_len,
        KVCache& kv_cache,
        int layer_idx
    ) const;

    int hidden_size() const { return hidden_size_; }
    int num_heads() const { return num_heads_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int q_head_dim() const { return q_head_dim_; }
    int kv_head_dim() const { return kv_head_dim_; }

    const float* get_q_weight() const { return q_proj_weight_.data(); }
    const float* get_k_weight() const { return k_proj_weight_.data(); }
    const float* get_v_weight() const { return v_proj_weight_.data(); }
    const float* get_o_weight() const { return o_proj_weight_.data(); }

private:
    void check_ready() const;
    void apply_rope(float* q, float* k, int position) const;
    void rms_normalize(float* data, int size, const float* weight) const;

    int hidden_size_;
    int num_heads_;
    int num_kv_heads_;
    int q_head_dim_;
    int kv_head_dim_;
    float rope_theta_;
    float partial_rotary_;
    int rotary_dim_;

    std::vector<float> q_proj_weight_;
    std::vector<float> k_proj_weight_;
    std::vector<float> v_proj_weight_;
    std::vector<float> q_norm_weight_;
    std::vector<float> k_norm_weight_;
    std::vector<float> o_proj_weight_;
};

}  // namespace qwen
