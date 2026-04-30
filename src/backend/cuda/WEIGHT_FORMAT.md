# CUDA 权重扁平化格式规范

> CudaEngine::set_layer_weights() 的权重拼接顺序定义

---

## 概述

`CudaEngine::set_layer_weights(int layer_idx, const std::vector<float>& flat_weights)` 接受一维扁平化的权重向量。不同层类型 (Linear/Full) 的拼接顺序如下。

---

## 公共前缀 (所有层)

| 偏移量 | 字段名 | 大小 | 说明 |
|--------|--------|------|------|
| 0 | input_layernorm_weight | hidden_size (1024) | 输入 RMSNorm 权重 |
| +hs | post_attention_layernorm_weight | hidden_size (1024) | 后 RMSNorm 权重 |
| +2hs | mlp_gate_proj_weight | intermediate × hidden (3584 × 1024) | MLP gate 投影 |
| +2hs + isz×hs | mlp_up_proj_weight | intermediate × hidden (3584 × 1024) | MLP up 投影 |
| +2hs + 2×isz×hs | mlp_down_proj_weight | hidden × intermediate (1024 × 3584) | MLP down 投影 |

公共前缀大小 = 2×1024 + 3×3584×1024 = 11,018,240 floats

---

## LinearAttention 层 (is_linear=true)

| 偏移量 | 字段名 | 大小 | 说明 |
|--------|--------|------|------|
| +公共前缀 | linear_in_proj_qkv_weight | conv_dim × hidden | QKV 联合投影 |
| +... | linear_in_proj_a_weight | num_heads × hidden | a 投影 |
| +... | linear_in_proj_b_weight | num_heads × hidden | b 投影 |
| +... | linear_in_proj_z_weight | z_dim × hidden | z 门控投影 |
| +... | linear_conv1d_weight | conv_dim × conv_kernel | 1D 卷积权重 |
| +... | linear_out_proj_weight | hidden × z_dim | 输出投影 |

其中：
- conv_dim = num_heads × (key_dim × 2 + value_dim) = 16 × (128×2 + 128) = 6144
- z_dim = num_heads × value_dim = 16 × 128 = 2048

---

## FullAttention 层 (is_linear=false)

| 偏移量 | 字段名 | 大小 | 说明 |
|--------|--------|------|------|
| +公共前缀 | full_q_proj_weight | num_heads × q_head_dim × 2 × hidden | Q 投影 (含门控) |
| +... | full_k_proj_weight | num_kv_heads × kv_head_dim × hidden | K 投影 |
| +... | full_v_proj_weight | num_kv_heads × kv_head_dim × hidden | V 投影 |
| +... | full_q_norm_weight | kv_head_dim | Q RMSNorm 权重 |
| +... | full_k_norm_weight | kv_head_dim | K RMSNorm 权重 |
| +... | full_o_proj_weight | hidden × num_heads × kv_head_dim | 输出投影 |

其中：
- Q 投影 ×2 是因为同时输出查询向量和门控向量
- q_head_dim = kv_head_dim = 256
- num_heads = 8, num_kv_heads = 2

---

## C++ 常量定义

```cpp
// 建议在代码中使用以下常量
namespace WeightLayout {
    constexpr int HIDDEN = 1024;
    constexpr int INTERMEDIATE = 3584;
    constexpr int NUM_HEADS = 16;      // Linear
    constexpr int KEY_DIM = 128;       // Linear
    constexpr int VALUE_DIM = 128;     // Linear
    constexpr int CONV_KERNEL = 4;     // Linear
    constexpr int Q_HEADS = 8;         // Full
    constexpr int KV_HEADS = 2;        // Full
    constexpr int HEAD_DIM = 256;      // Full

    // 公共前缀偏移量
    constexpr size_t INPUT_NORM = 0;
    constexpr size_t POST_NORM = HIDDEN;
    constexpr size_t MLP_GATE = 2 * HIDDEN;
    constexpr size_t MLP_UP = MLP_GATE + INTERMEDIATE * HIDDEN;
    constexpr size_t MLP_DOWN = MLP_UP + INTERMEDIATE * HIDDEN;
    constexpr size_t COMMON_PREFIX = MLP_DOWN + HIDDEN * INTERMEDIATE;
}
```

---

*文档版本：1.0 | 日期：2026-04-09*
