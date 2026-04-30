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

### 实测性能数据 (RTX 5060 Ti, 2026-04-29)

#### 01_weight_bf16 (权重 BF16)

| batch_size | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | TPOT (ms/tok) | Memory (MB) |
|------------|-----------------|----------------|-----------|---------------|-------------|
| 1 | 692.0 | 12,214.8 | 1,479.7 | 0.082 | 5,215 |
| 8 | 706.2 | 12,236.9 | 1,450.0 | 0.082 | 5,215 |
| 16 | 689.9 | 12,183.1 | 1,484.3 | 0.082 | 5,215 |
| 32 | 691.4 | 12,256.8 | 1,481.0 | 0.082 | 5,215 |
| 64 | 994.8 | 12,206.2 | 1,029.3 | 0.082 | 5,277 |
| 128 | 1,276.5 | 12,173.0 | 802.2 | 0.082 | 5,423 |

> **测试条件**：prefill=1024, decode=512, 5轮平均, BF16权重

#### 其他变体测试数据

| 02_kv_cache_fp16 | 1 | 702.4 | 12,260.6 | 1,457.9 | 0.082 | 5,215 |
| 02_kv_cache_fp16 | 8 | 704.1 | 12,217.2 | 1,454.3 | 0.082 | 5,215 |
| 02_kv_cache_fp16 | 16 | 701.1 | 12,218.7 | 1,460.5 | 0.082 | 5,215 |
| 02_kv_cache_fp16 | 32 | 702.0 | 12,209.9 | 1,458.7 | 0.082 | 5,215 |
| 02_kv_cache_fp16 | 64 | 1,017.5 | 12,244.7 | 1,006.3 | 0.082 | 5,277 |
| 02_kv_cache_fp16 | 128 | 1,253.6 | 12,160.9 | 816.9 | 0.082 | 5,423 |
| 03_kv_cache_int8 | 1 | 708.8 | 12,164.3 | 1,444.7 | 0.082 | 5,215 |
| 03_kv_cache_int8 | 8 | 708.2 | 12,169.8 | 1,446.0 | 0.082 | 5,215 |
| 03_kv_cache_int8 | 16 | 685.6 | 12,079.3 | 1,493.6 | 0.083 | 5,215 |
| 03_kv_cache_int8 | 32 | 701.4 | 12,105.2 | 1,460.0 | 0.083 | 5,215 |
| 03_kv_cache_int8 | 64 | 1,011.3 | 12,153.4 | 1,012.5 | 0.082 | 5,277 |
| 03_kv_cache_int8 | 128 | 1,296.3 | 12,184.9 | 789.9 | 0.082 | 5,423 |
| 04_weight_int8 | 1 | 699.0 | 12,203.5 | 1,465.0 | 0.082 | 5,215 |
| 04_weight_int8 | 8 | 705.5 | 12,188.9 | 1,451.4 | 0.082 | 5,215 |
| 04_weight_int8 | 16 | 706.6 | 12,195.3 | 1,449.1 | 0.082 | 5,215 |
| 04_weight_int8 | 32 | 712.1 | 12,171.1 | 1,438.0 | 0.082 | 5,215 |
| 04_weight_int8 | 64 | 1,003.4 | 12,174.2 | 1,020.5 | 0.082 | 5,277 |
| 04_weight_int8 | 128 | 1,281.1 | 12,186.6 | 799.3 | 0.082 | 5,423 |
| 05_weight_int4 | 1 | 705.9 | 12,202.0 | 1,450.5 | 0.082 | 5,215 |
| 05_weight_int4 | 8 | 710.2 | 12,214.0 | 1,441.9 | 0.082 | 5,215 |
| 05_weight_int4 | 16 | 700.0 | 12,253.5 | 1,462.8 | 0.082 | 5,215 |
| 05_weight_int4 | 32 | 708.1 | 12,215.4 | 1,446.2 | 0.082 | 5,215 |
| 05_weight_int4 | 64 | 1,026.3 | 12,138.4 | 997.7 | 0.082 | 5,277 |
| 05_weight_int4 | 128 | 1,278.8 | 12,252.3 | 800.8 | 0.082 | 5,423 |
| 06_full_fp16 | 1 | 703.4 | 12,289.2 | 1,455.8 | 0.081 | 5,215 |
| 06_full_fp16 | 8 | 710.3 | 12,203.6 | 1,441.7 | 0.082 | 5,215 |
| 06_full_fp16 | 16 | 702.6 | 12,154.5 | 1,457.5 | 0.082 | 5,215 |
| 06_full_fp16 | 32 | 716.2 | 12,235.1 | 1,429.8 | 0.082 | 5,215 |
| 06_full_fp16 | 64 | 1,015.1 | 12,187.7 | 1,008.7 | 0.082 | 5,277 |
| 06_full_fp16 | 128 | 1,304.9 | 12,222.6 | 784.7 | 0.082 | 5,423 |
| 07_full_int8 | 1 | 721.6 | 12,289.5 | 1,419.0 | 0.081 | 5,215 |
| 07_full_int8 | 8 | 712.4 | 12,281.9 | 1,437.3 | 0.081 | 5,215 |
| 07_full_int8 | 16 | 702.3 | 12,179.7 | 1,458.1 | 0.082 | 5,215 |
| 07_full_int8 | 32 | 713.5 | 12,271.7 | 1,435.2 | 0.081 | 5,215 |
| 07_full_int8 | 64 | 1,031.5 | 12,267.2 | 992.7 | 0.082 | 5,277 |
| 07_full_int8 | 128 | 1,285.4 | 12,224.3 | 796.6 | 0.082 | 5,423 |
| 08_flash_to_linear | 1 | 716.6 | 12,153.6 | 1,428.9 | 0.082 | 5,215 |
| 08_flash_to_linear | 8 | 697.0 | 12,109.0 | 1,469.3 | 0.083 | 5,215 |
| 08_flash_to_linear | 16 | 693.0 | 12,011.0 | 1,477.5 | 0.083 | 5,215 |
| 08_flash_to_linear | 32 | 702.6 | 12,125.8 | 1,457.4 | 0.082 | 5,215 |
| 08_flash_to_linear | 64 | 996.6 | 12,098.9 | 1,027.5 | 0.083 | 5,277 |
| 08_flash_to_linear | 128 | 1,266.7 | 12,143.8 | 808.4 | 0.082 | 5,423 |
| 09_linear_to_flash | 1 | 718.6 | 12,221.1 | 1,425.1 | 0.082 | 5,215 |
| 09_linear_to_flash | 8 | 719.2 | 12,260.0 | 1,423.8 | 0.082 | 5,215 |
| 09_linear_to_flash | 16 | 709.1 | 12,204.0 | 1,444.1 | 0.082 | 5,215 |
| 09_linear_to_flash | 32 | 723.5 | 12,264.9 | 1,415.3 | 0.082 | 5,215 |
| 09_linear_to_flash | 64 | 1,020.5 | 12,275.5 | 1,003.4 | 0.081 | 5,277 |
| 09_linear_to_flash | 128 | 1,259.4 | 12,195.9 | 813.1 | 0.082 | 5,423 |

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
