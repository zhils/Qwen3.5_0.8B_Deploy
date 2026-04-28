# Qwen3.5-0.8B Deploy 无损精度内存优化分析

> 文档版本: v1.0
> 更新日期: 2026-04-28
> 适用范围: CUDA Engine v3.3

---

## 1. 概述

本文档分析在不损失计算精度的前提下，项目中的内存优化空间。所有优化方案保持 **FP32 计算精度**，仅通过内存布局优化、Buffer 复用、动态分配等手段减少显存占用。

---

## 2. 当前内存构成回顾

根据 [memory_consumption_analysis.md](memory_consumption_analysis.md)，v3.3 版本 (batch=1) 内存占用约 **8,772 MB**：

| 类别 | 内存 (MB) | 占比 |
|------|-----------|------|
| 模型权重 | 3,693 | 42.1% |
| LM Head 双精度 | ~1,437 | 16.4% |
| 其他开销 | ~1,683 | 19.2% |
| 激活 Buffer | ~1,000 | 11.4% |
| KV Cache | 402 | 4.6% |
| cuBLAS Workspace | ~384 | 4.4% |
| CUDA Graph | ~200 | 2.3% |
| Linear State | 13 | 0.1% |

---

## 3. 无损精度优化方案

### 3.1 高优先级优化

#### 3.1.1 Embedding 与 LM Head 权重共享

**当前问题：**
- Token Embedding 占用 958 MB
- LM Head 占用 958 MB
- **Qwen 模型中 Embedding 和 LM Head 通常共享同一份权重**（Tied Embedding）

**优化方案：**
- 检测模型是否使用 tied embedding
- 如果权重相同，GPU 上只保留一份
- 两个组件共用同一个 `d_weight_` 指针

**预期收益：** 节省 **958 MB**

**实现位置：**
- [src/backend/cuda/kernels/token_embedding_cuda.cu](src/backend/cuda/kernels/token_embedding_cuda.cu)
- [src/backend/cuda/kernels/lm_head_cuda.cu](src/backend/cuda/kernels/lm_head_cuda.cu)

**代码修改示例：**
```cpp
// 在 CudaEngine 中添加共享权重指针
class CudaEngine {
    float* d_shared_embedding_lmhead_weight_ = nullptr;  // 共享权重
    bool use_tied_embedding_ = true;
    
    // 设置权重时，如果启用 tied embedding，只分配一次
    void set_shared_weight(const std::vector<float>& weight) {
        if (use_tied_embedding_) {
            cudaMalloc(&d_shared_embedding_lmhead_weight_, weight_size);
            cudaMemcpy(d_shared_embedding_lmhead_weight_, ...);
            
            // 同时设置给 embedding 和 lm_head
            embedding.set_weight_ptr(d_shared_embedding_lmhead_weight_);
            lm_head.set_weight_ptr(d_shared_embedding_lmhead_weight_);
        }
    }
};
```

---

#### 3.1.2 消除 LM Head 双精度存储

**当前问题：**
- `CudaLMHead` 同时保留 FP32 和 BF16 两份权重
- `d_weight_bf16_` 占用 479 MB
- 但 BF16 权重是转换后的副本，原始 FP32 权重在加载后通常不再需要

**优化方案：**
- 仅保留 BF16 权重（用于计算）
- 或仅保留 FP32 权重（按需在线转换）
- 推荐：仅保留 BF16，因为 LM Head 已经使用 BF16 Tensor Core 计算

**预期收益：** 节省 **479 MB**（BF16 路径）或 **958 MB**（FP32 路径）

**实现位置：** [src/backend/cuda/kernels/lm_head_cuda.cu](src/backend/cuda/kernels/lm_head_cuda.cu)

**代码修改：**
```cpp
// 当前：同时保留 FP32（临时）和 BF16
// 优化后：仅保留 BF16
class CudaLMHead {
    __nv_bfloat16* d_weight_bf16_;  // 仅保留 BF16
    // 删除 FP32 临时存储
};
```

---

#### 3.1.3 统一 cuBLAS Handle（已完成观察）

**当前问题：**
- `CublasHandlePool` 使用单例模式，已经共享一个 handle
- 但 `CudaLMHead` 单独创建了一个 `cublas_handle_`

**优化方案：**
- `CudaLMHead` 也使用 `CublasHandlePool::instance().get()`
- 删除独立的 `cublas_handle_` 成员

**预期收益：** 节省 **~4-32 MB** workspace

**实现位置：** [src/backend/cuda/kernels/lm_head_cuda.cu](src/backend/cuda/kernels/lm_head_cuda.cu)

---

#### 3.1.4 Buffer 复用与统一分配

**当前问题：**
- `CudaEngine` 分配了多个独立的单序列 buffer：
  - `d_input_buf_`, `d_normed_input_`, `d_attn_out_`, `d_post_normed_`, `d_mlp_out_`, `d_residual_`, `d_output_buf_`
- 每个 4 KB，共 28 KB — 虽然不大，但分配开销存在

**优化方案：**
- 统一分配一个大 buffer，通过偏移使用
- 减少 CUDA 内存分配器碎片化

**预期收益：** 节省少量显存，主要减少分配开销

**实现位置：** [src/backend/cuda/kernels/cuda_engine.cu](src/backend/cuda/kernels/cuda_engine.cu)

---

#### 3.1.5 MLP Hidden Buffer 优化

**当前问题：**
- `d_hidden_buf_` 分配 `batch × intermediate × 2`（gate + up）
- 但 `forward` 函数中 batch_size=1 时，每次都 `cudaMalloc` gate_buf 和 up_buf
- 没有复用预分配的 buffer

**优化方案：**
- batch_size=1 时也使用预分配的 `d_hidden_buf_`
- 或统一使用 `d_hidden_buf_` 处理所有 batch size

**预期收益：** 减少频繁的 `cudaMalloc/cudaFree` 调用开销

**实现位置：** [src/backend/cuda/kernels/mlp_cuda.cu](src/backend/cuda/kernels/mlp_cuda.cu)

---

#### 3.1.6 KV Cache 动态按需分配

**当前问题：**
- `CudaKVCache::reset` 预分配 `num_layers × max_seq_len × num_kv_heads × head_dim`
- 默认 `max_seq_len = 8192` 或 `262144`
- 实际序列长度通常远小于最大值

**优化方案：**
- 初始分配较小空间（如 1024）
- 当序列增长时，按需重新分配（类似 std::vector 的扩容策略）
- 使用 2 倍扩容策略，均摊 O(1) 分配成本

**预期收益：**
- 短序列场景节省 **90%+** KV Cache 内存
- 例如 512 长度序列：从 402 MB → ~25 MB

**实现位置：** [src/backend/cuda/kernels/full_attention_cuda.cu](src/backend/cuda/kernels/full_attention_cuda.cu)

**代码修改示例：**
```cpp
struct CudaKVCache {
    // 动态扩容
    void ensure_capacity(int layer_idx, int required_seq_len);
    
private:
    int allocated_seq_len = 0;  // 实际分配的长度
    
    void grow(int new_seq_len) {
        if (new_seq_len <= allocated_seq_len) return;
        // 2倍扩容
        int new_capacity = std::max(new_seq_len, allocated_seq_len * 2);
        // 重新分配并拷贝数据...
    }
};
```

---

#### 3.1.7 Linear Attention Batch Buffer 延迟分配优化

**当前问题：**
- `forward_batch` 中每次调用都 `cudaMalloc` `d_batch_conv_state`
- 这是临时 buffer，应该预分配

**优化方案：**
- 在 `CudaLinearAttention` 中添加 `d_batch_conv_state_` 成员
- 类似其他 batch buffer 的 `ensure_batch_buffers` 模式

**预期收益：** 减少分配开销，避免内存碎片

**实现位置：** [src/backend/cuda/kernels/linear_attention_cuda.cu](src/backend/cuda/kernels/linear_attention_cuda.cu)

---

#### 3.1.8 Token Embedding Batch 临时 Buffer 优化

**当前问题：**
- `CudaTokenEmbedding::forward(const std::vector<int>&)` 每次调用都 `cudaMalloc` `d_token_ids`
- 这是 Host → Device 的临时拷贝

**优化方案：**
- 预分配一个固定大小的 `d_token_ids_buf_`
- 或使用 pinned memory + async copy

**预期收益：** 减少小内存频繁分配

**实现位置：** [src/backend/cuda/kernels/token_embedding_cuda.cu](src/backend/cuda/kernels/token_embedding_cuda.cu)

---

### 3.2 中优先级优化

#### 3.2.1 Full Attention 内部 Buffer 精简

**当前问题：**
- `CudaFullAttention` 分配了 `d_attn_scores_buf_`：[num_heads, max_seq_len] = 8 MB
- 但 FlashAttention v2 实现在线 softmax，不需要存储完整的 scores 矩阵
- 检查 `d_attn_scores_buf_` 是否实际被使用

**优化方案：**
- 如果未使用，删除该 buffer
- 如果用于其他路径，改为按需分配

**预期收益：** 节省 **8 MB**

**实现位置：** [src/backend/cuda/kernels/full_attention_cuda.cu](src/backend/cuda/kernels/full_attention_cuda.cu)

---

#### 3.2.2 Attention Batch Buffer 与 Engine Buffer 复用

**当前问题：**
- `CudaFullAttention` 有自己的 batch buffer：`d_batch_q_buf_`, `d_batch_k_buf_` 等
- `CudaEngine` 也有 batch buffer：`d_batch_input_buf_`, `d_batch_output_buf_`
- 这些 buffer 在时间上不重叠，可以复用

**优化方案：**
- 统一分配一个大的 "workspace" buffer
- 各模块通过偏移使用，避免重复分配

**预期收益：** 节省 **~10-20 MB**

---

#### 3.2.3 CUDA Graph 内存优化

**当前问题：**
- CUDA Graph 捕获时会缓存 kernel 参数和依赖关系
- 占用约 100-500 MB

**优化方案：**
- 仅在 decode 阶段使用 CUDA Graph（已完成）
- prefill 阶段不使用 Graph（已完成）
- 这是合理的 trade-off

**预期收益：** 当前实现已较优，无需额外优化

---

### 3.3 低优先级优化

#### 3.3.1 Linear Attention State 统一分配

**当前问题：**
- 每层 Linear Attention 的 State 独立分配
- 12 层共 13.3 MB，虽然不大，但分配开销存在

**优化方案：**
- 统一分配一个大块，每层通过偏移访问

**预期收益：** 节省少量显存和分配开销

---

## 4. 无损优化收益汇总

| 优化项 | 节省内存 | 实现难度 | 风险 | 精度影响 |
|--------|----------|----------|------|----------|
| Embedding/LM Head 共享 | **958 MB** | 低 | 低 | 无 |
| 消除 LM Head 双精度 | **479 MB** | 低 | 低 | 无 |
| KV Cache 动态分配 | **~300-380 MB** | 中 | 中 | 无 |
| 统一 cuBLAS Handle | **~4-32 MB** | 低 | 低 | 无 |
| Buffer 复用/统一分配 | **~50-100 MB** | 中 | 低 | 无 |
| 删除未使用 scores buffer | **8 MB** | 低 | 低 | 无 |
| Embedding batch buffer 预分配 | 微量 | 低 | 低 | 无 |
| Linear Attn batch state 预分配 | 微量 | 低 | 低 | 无 |
| **总计** | **~1,800 - 2,000 MB** | - | - | **无损** |

**优化后预期内存：** 8,772 MB → **~6,800 MB**（节省约 20-23%）

---

## 5. 实施建议

### Phase 1: 快速收益（1-2 天）
1. **Embedding/LM Head 共享** — 最大单笔收益
2. **消除 LM Head 双精度** — 简单安全
3. **统一 cuBLAS Handle** — 一行代码修改

### Phase 2: 中等收益（2-3 天）
4. **KV Cache 动态分配** — 需要仔细测试
5. **Buffer 复用** — 需要重构内存管理

### Phase 3: 精细优化（1 天）
6. **删除未使用 buffer**
7. **预分配临时 buffer**

---

## 6. 验证方法

每项优化后应验证：
1. **精度验证**：运行 `v2_kernel_accuracy_validate` 或 `e2e_validate`
2. **内存验证**：使用 `GPUMemoryProfiler` 对比优化前后内存占用
3. **性能验证**：运行 `performance_test` 确保无性能退化

---

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| [src/backend/cuda/kernels/cuda_engine.cu](src/backend/cuda/kernels/cuda_engine.cu) | Engine Buffer 分配 |
| [src/backend/cuda/kernels/lm_head_cuda.cu](src/backend/cuda/kernels/lm_head_cuda.cu) | LM Head 双精度权重 |
| [src/backend/cuda/kernels/token_embedding_cuda.cu](src/backend/cuda/kernels/token_embedding_cuda.cu) | Embedding 权重 |
| [src/backend/cuda/kernels/full_attention_cuda.cu](src/backend/cuda/kernels/full_attention_cuda.cu) | KV Cache 分配 |
| [src/backend/cuda/kernels/mlp_cuda.cu](src/backend/cuda/kernels/mlp_cuda.cu) | MLP Hidden Buffer |
| [src/backend/cuda/kernels/linear_attention_cuda.cu](src/backend/cuda/kernels/linear_attention_cuda.cu) | Linear Attn Buffer |
