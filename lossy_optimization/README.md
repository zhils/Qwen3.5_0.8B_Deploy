# 有损优化实验目录

本目录包含 9 种不同的有损精度优化实现，每种优化都在独立的子文件夹中实现。

## 目录结构

```
lossy_optimization/
├── 01_weight_bf16/      # 权重 BF16 量化
├── 02_kv_cache_fp16/    # KV Cache FP16
├── 03_kv_cache_int8/    # KV Cache INT8
├── 04_weight_int8/      # 权重 INT8
├── 05_weight_int4/      # 权重 INT4
├── 06_full_fp16/        # 权重 FP16 + KV Cache FP16
├── 07_full_int8/        # 权重 INT8 + KV Cache INT8
├── 08_flash_to_linear/  # Flash Attention → Linear Attention
└── 09_linear_to_flash/  # Linear Attention → Flash Attention
```

## 各优化说明

### 01_weight_bf16 - 权重 BF16 量化

**优化内容**：将所有权重从 FP32 转为 BF16 存储

**预期效果**：
- 权重内存减少 50%
- 精度损失极小（BF16 与 FP32 动态范围相同）
- 适合 Ampere+ 架构 GPU

### 02_kv_cache_fp16 - KV Cache FP16

**优化内容**：将 KV Cache 从 FP32 转为 FP16 存储

**预期效果**：
- KV Cache 内存减少 50%
- 精度损失较小
- 适合所有支持 FP16 的 GPU

### 03_kv_cache_int8 - KV Cache INT8

**优化内容**：将 KV Cache 从 FP32 转为 INT8 存储，使用 per-token 量化

**预期效果**：
- KV Cache 内存减少 75%
- 精度损失中等
- 需要额外的 scale 存储

### 04_weight_int8 - 权重 INT8

**优化内容**：将所有权重从 FP32 转为 INT8 存储，使用 per-channel 量化

**预期效果**：
- 权重内存减少 75%
- 精度损失较大
- 需要 INT8 Tensor Core 支持

### 05_weight_int4 - 权重 INT4

**优化内容**：将所有权重从 FP32 转为 INT4 存储，使用分组量化

**预期效果**：
- 权重内存减少 87.5%
- 精度损失最大
- 需要特殊的解压缩 kernel

### 06_full_fp16 - 权重 FP16 + KV Cache FP16

**优化内容**：整条链路使用 FP16

**预期效果**：
- 总内存减少约 50%
- 精度损失较小
- 充分利用 FP16 Tensor Core

### 07_full_int8 - 权重 INT8 + KV Cache INT8

**优化内容**：整条链路使用 INT8

**预期效果**：
- 总内存减少约 75%
- 精度损失较大
- 充分利用 INT8 Tensor Core

### 08_flash_to_linear - Flash Attention 替换为 Linear Attention

**优化内容**：使用 Linear Attention 替换 Flash/Full Attention

**预期效果**：
| 指标 | Flash Attention | Linear Attention |
|------|-----------------|------------------|
| Prefill 复杂度 | O(n²) | O(n) |
| Decode 复杂度 | O(n) | O(1) |
| 内存占用 | O(n) KV Cache | O(1) 状态 |
| 长序列支持 | 受限 | 优秀 |

**适用场景**：
- 长序列推理（> 4K tokens）
- 内存受限场景
- 流式生成

**实现文件**：
- `include/cuda_engine_linear.hpp` - Linear Attention Engine
- `kernels/cuda_engine_linear.cu` - Engine 实现

**状态管理**：
- 使用 `CudaLinearAttnState` 替代 `CudaKVCache`
- 状态大小固定，不随序列长度增长

### 09_linear_to_flash - Linear Attention 替换为 Flash Attention

**优化内容**：使用 Flash Attention 替换 Linear Attention

**预期效果**：
| 指标 | Linear Attention | Flash Attention |
|------|------------------|-----------------|
| 精度 | 近似 | 精确 |
| Prefill 速度 | 快 | 中等 |
| Decode 速度 | 快 | 中等 |
| 短序列性能 | 一般 | 优秀 |

**适用场景**：
- 需要精确注意力的场景
- 短序列推理（< 4K tokens）
- 质量优先场景

**实现文件**：
- `include/cuda_engine_flash.hpp` - Flash Attention Engine
- `kernels/cuda_engine_flash.cu` - Engine 实现

**状态管理**：
- 使用 `CudaKVCache` 存储 KV 对
- 支持动态扩容

## 性能对比

### 内存节省对比

| 优化方案 | 权重内存 | KV Cache 内存 | 总内存节省 |
|---------|---------|--------------|-----------|
| 原始 FP32 | 100% | 100% | 0% |
| 01_weight_bf16 | 50% | 100% | ~40% |
| 02_kv_cache_fp16 | 100% | 50% | ~10% |
| 03_kv_cache_int8 | 100% | 25% | ~15% |
| 04_weight_int8 | 25% | 100% | ~60% |
| 05_weight_int4 | 12.5% | 100% | ~70% |
| 06_full_fp16 | 50% | 50% | ~50% |
| 07_full_int8 | 25% | 25% | ~75% |

### Attention 替换性能对比

| 方案 | Prefill 延迟 | Decode 延迟 | 内存 | 精度 |
|------|-------------|-------------|------|------|
| Flash Attention | O(n²) | O(n) | O(n) | 精确 |
| Linear Attention | O(n) | O(1) | O(1) | 近似 |

### Prefill vs Decode 平衡分析

```
序列长度 vs 推荐方案:

短序列 (< 1K):    Flash Attention (精确 + 速度)
中序列 (1K-4K):   Flash Attention (精确)
长序列 (4K-16K):  Linear Attention (内存 + 速度)
超长序列 (> 16K): Linear Attention (必需)
```

## 编译说明

每个子文件夹包含独立的 CMakeLists.txt，可以单独编译：

```bash
cd lossy_optimization/08_flash_to_linear
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
make -j4
```

## 使用示例

### Flash to Linear Attention

```cpp
#include "cuda_engine_linear.hpp"

qwen::cuda::CudaEngineLinear engine(num_layers, hidden_size, intermediate_size, vocab_size);

// 设置权重...

// 单步推理
engine.forward(d_input, d_output);

// 批量推理
engine.forward_batch(d_batch_input, d_batch_output, batch_size);

// 重置状态（新对话）
engine.reset_state();
```

### Linear to Flash Attention

```cpp
#include "cuda_engine_flash.hpp"

qwen::cuda::CudaEngineFlash engine(num_layers, hidden_size, intermediate_size, vocab_size, max_seq_len);

// 设置权重...

// 单步推理
engine.forward(d_input, d_output, position);

// 批量 Prefill
engine.forward_batch_prefill(d_batch_input, d_batch_output, positions, batch_size);

// 重置缓存（新对话）
engine.reset_cache();
```

## 注意事项

1. **BF16** 需要 Ampere+ 架构 (RTX 30 系列)
2. **INT8 量化** 需要校准数据集
3. **INT4 量化** 需要特殊的打包/解包 kernel
4. **Linear Attention** 精度损失需要通过验证测试评估
5. **Flash Attention** 长序列受显存限制
6. 精度损失需要通过验证测试评估
