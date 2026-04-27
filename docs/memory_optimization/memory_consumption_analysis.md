# Qwen3.5-0.8B Deploy 内存消耗分析

> 文档版本: v1.0
> 更新日期: 2026-04-27
> 适用范围: CUDA Engine v2.0 / v3.0 / v3.3

---

## 1. 概述

本文档详细分析 Qwen3.5-0.8B Deploy 项目在 GPU 推理过程中的内存消耗构成，包括权重内存、激活内存、KV Cache、中间 Buffer 等各类内存占用，并提供优化方向与 trade-off 分析。

### 1.1 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_hidden_layers | 24 | 共 24 层 |
| hidden_size | 1024 | 隐藏层维度 |
| intermediate_size | 3584 | MLP 中间层维度 |
| vocab_size | 248320 | 词表大小 |
| num_attention_heads | 8 | Full Attention 头数 |
| num_key_value_heads | 2 | GQA KV 头数 |
| head_dim | 256 | 注意力头维度 |
| max_position_embeddings | 262144 | 最大序列长度 |
| linear_attention_key_dim | 128 | Linear Attention Key 维度 |
| linear_attention_value_dim | 128 | Linear Attention Value 维度 |
| conv_kernel_size | 4 | Conv1D 卷积核大小 |

### 1.2 Layer Pattern

每 4 层一轮回（v2.0 及之后版本）：
```
Layer 0-1:  LinearAttention
Layer 2-3:  FullAttention (GQA)
Layer 4-5:  LinearAttention
Layer 6-7:  FullAttention (GQA)
...
Layer 20-21: LinearAttention
Layer 22-23: FullAttention (GQA)
```

即：12 层 Linear Attention + 12 层 Full Attention（v2.1 调整为 50/50 比例）

---

## 2. 内存消耗总览

### 2.1 RTX 3080 Ti 实测数据 (batch=1)

| 指标 | 数值 |
|------|------|
| **GPU VRAM 总容量** | 11,910 MB |
| **Deploy 实际占用** | 8,772 MB |
| **vLLM 占用** | ~9,500 MB |
| **可用余量** | ~3,138 MB |

### 2.2 RTX 5060 Ti 实测数据

| 指标 | 数值 |
|------|------|
| **GPU VRAM 总容量** | 16,384 MB |
| **Deploy 实际占用** | ~10,445 MB |
| **可用余量** | ~5,939 MB |

---

## 3. 内存消耗详细分解

### 3.1 权重内存 (Model Weights)

#### 3.1.1 每层权重计算

**Linear Attention Layer 权重：**

| 权重 | 维度 | 参数量 | FP32 内存 |
|------|------|--------|-----------|
| in_proj_qkv | [conv_dim, hidden] | 5,120 × 1,024 = 5,242,880 | 20.0 MB |
| in_proj_a | [num_heads, hidden] | 16 × 1,024 = 16,384 | 0.06 MB |
| in_proj_b | [num_heads, hidden] | 16 × 1,024 = 16,384 | 0.06 MB |
| in_proj_z | [z_dim, hidden] | 2,048 × 1,024 = 2,097,152 | 8.0 MB |
| conv1d | [conv_dim, conv_kernel] | 5,120 × 4 = 20,480 | 0.08 MB |
| out_proj | [hidden, z_dim] | 1,024 × 2,048 = 2,097,152 | 8.0 MB |
| a_log | [num_heads] | 16 | 0.00006 MB |
| dt_bias | [num_heads] | 16 | 0.00006 MB |
| norm_weight | [value_dim] | 128 | 0.0005 MB |
| **Linear Attention 小计** | - | **9,474,512** | **36.1 MB** |

> 注：conv_dim = num_heads × (key_dim × 2 + value_dim) = 16 × (128 × 2 + 128) = 5,120
> z_dim = num_heads × value_dim = 16 × 128 = 2,048

**Full Attention Layer 权重：**

| 权重 | 维度 | 参数量 | FP32 内存 |
|------|------|--------|-----------|
| q_proj | [num_heads × q_head_dim × 2, hidden] | 4,096 × 1,024 = 4,194,304 | 16.0 MB |
| k_proj | [num_kv_heads × kv_head_dim, hidden] | 512 × 1,024 = 524,288 | 2.0 MB |
| v_proj | [num_kv_heads × kv_head_dim, hidden] | 512 × 1,024 = 524,288 | 2.0 MB |
| o_proj | [hidden, num_heads × kv_head_dim] | 1,024 × 2,048 = 2,097,152 | 8.0 MB |
| q_norm | [kv_head_dim] | 256 | 0.001 MB |
| k_norm | [kv_head_dim] | 256 | 0.001 MB |
| **Full Attention 小计** | - | **7,340,544** | **28.0 MB** |

**MLP 权重（每层相同）：**

| 权重 | 维度 | 参数量 | FP32 内存 |
|------|------|--------|-----------|
| gate_proj | [intermediate, hidden] | 3,584 × 1,024 = 3,670,016 | 14.0 MB |
| up_proj | [intermediate, hidden] | 3,584 × 1,024 = 3,670,016 | 14.0 MB |
| down_proj | [hidden, intermediate] | 1,024 × 3,584 = 3,670,016 | 14.0 MB |
| **MLP 小计** | - | **11,010,048** | **42.0 MB** |

**RMSNorm 权重（每层 2 个）：**

| 权重 | 维度 | 参数量 | FP32 内存 |
|------|------|--------|-----------|
| input_norm | [hidden] | 1,024 | 0.004 MB |
| post_norm | [hidden] | 1,024 | 0.004 MB |
| **RMSNorm 小计** | - | **2,048** | **0.008 MB** |

#### 3.1.2 模型总权重内存

**v2.0 / v3.x 版本（12 Linear + 12 Full）：**

| 组件 | 层数 | 每层内存 | 总内存 |
|------|------|----------|--------|
| Linear Attention | 12 | 36.1 MB | 433.2 MB |
| Full Attention | 12 | 28.0 MB | 336.0 MB |
| MLP | 24 | 42.0 MB | 1,008.0 MB |
| RMSNorm | 24 | 0.008 MB | 0.2 MB |
| **Layer 总权重** | - | - | **1,777.4 MB** |

**其他权重：**

| 组件 | 维度 | 参数量 | FP32 内存 |
|------|------|--------|-----------|
| Token Embedding | [vocab, hidden] | 248,320 × 1,024 | 958.0 MB |
| LM Head | [hidden, vocab] | 1,024 × 248,320 | 958.0 MB |
| Final RMSNorm | [hidden] | 1,024 | 0.004 MB |
| **其他权重小计** | - | - | **1,916.0 MB** |

**FP32 总权重内存：约 3,693 MB (~3.6 GB)**

#### 3.1.3 权重内存优化空间

| 优化方案 | 内存节省 | 说明 |
|----------|----------|------|
| FP16/BF16 量化 | **50%** (~1.85 GB) | 权重减半，decode 速度提升 20-30% |
| INT8 量化 | **75%** (~2.77 GB) | 需校准，可能轻微影响精度 |
| INT4 量化 | **87.5%** (~3.23 GB) | 精度损失较大，需仔细评估 |

---

### 3.2 KV Cache 内存

#### 3.2.1 KV Cache 计算

```
KV Cache 内存 = num_layers × max_seq_len × num_kv_heads × head_dim × 2 (K+V) × sizeof(float)
```

**Full Attention KV Cache：**

| 配置 | 计算 | 内存 |
|------|------|------|
| 单层 | 1 × 262,144 × 2 × 256 × 2 × 4 | 1,073 MB |
| 12 层 Full Attention | 12 × 262,144 × 2 × 256 × 2 × 4 | **12,884 MB (~12.6 GB)** |

**当前实现：**

| 版本 | KV Cache 分配策略 | 实际占用 |
|------|-------------------|----------|
| v2.0 / v3.x | 预分配 max_seq_len=8192（实际） | 12 × 8,192 × 2 × 256 × 2 × 4 = **402 MB** |
| v3.0 (All Flash) | 24 层全部分配 | 24 × 8,192 × 2 × 256 × 2 × 4 = **805 MB** |

> 注：代码中 `max_seq_len_` 在 `CudaFullAttention` 构造函数中设置为 8192，但实际 `CudaKVCache::reset` 使用传入的 `max_seq_len`（默认 262144）。如果按 262144 分配，KV Cache 将占用约 12.6 GB。

#### 3.2.2 KV Cache 优化空间

| 优化方案 | 内存节省 | 说明 |
|----------|----------|------|
| 动态分配（按需增长） | **90%+** | 根据实际序列长度分配，避免预分配浪费 |
| Paged KV Cache | **80%+** | 分页管理，支持更长上下文 |
| FP16/BF16 KV Cache | **50%** | KV Cache 减半 |
| INT8 KV Cache | **75%** | 已实验，见 `kv_int8_cuda.cu` |
| GQA 已优化 | - | 已从 8 heads 减少到 2 heads，节省 75% |

---

### 3.3 Linear Attention State 内存

#### 3.3.1 State 构成

每个 Linear Attention Layer 需要维护两个状态：

| State | 维度 | 每层内存 |
|-------|------|----------|
| Recurrent State | [num_heads, key_dim, value_dim] | 16 × 128 × 128 × 4 = 1.0 MB |
| Conv State | [conv_dim, conv_kernel - 1] | 5,120 × 3 × 4 = 0.06 MB |
| **每层小计** | - | **~1.06 MB** |

**12 层 Linear Attention 总 State 内存：**

| 组件 | 计算 | 内存 |
|------|------|------|
| Recurrent State | 12 × 16 × 128 × 128 × 4 | 12.6 MB |
| Conv State | 12 × 5,120 × 3 × 4 | 0.7 MB |
| **State 总内存** | - | **~13.3 MB** |

#### 3.3.2 State 优化空间

| 优化方案 | 内存节省 | 说明 |
|----------|----------|------|
| FP16 State | **50%** | 状态值范围可控，可安全量化 |
| 统一分配 | 少量 | 避免每层独立分配开销 |

---

### 3.4 激活/中间 Buffer 内存

#### 3.4.1 Engine 级别 Buffer

**CudaEngine (v2.0) 单序列 Buffer：**

| Buffer | 维度 | 内存 |
|--------|------|------|
| d_input_buf | [hidden] | 4 KB |
| d_normed_input | [hidden] | 4 KB |
| d_attn_out | [hidden] | 4 KB |
| d_post_normed | [hidden] | 4 KB |
| d_mlp_out | [hidden] | 4 KB |
| d_residual | [hidden] | 4 KB |
| d_output_buf | [hidden] | 4 KB |
| d_lmhead_out | [vocab] | 958 MB |
| **单序列小计** | - | **~958 MB** |

**Batch Buffer（动态分配）：**

| Buffer | 维度 | batch=128 内存 |
|--------|------|----------------|
| d_batch_input_buf | [batch, hidden] | 512 KB |
| d_batch_output_buf | [batch, hidden] | 512 KB |
| d_positions_buf | [batch] | 512 B |
| **Batch 小计** | - | **~1 MB** |

#### 3.4.2 Full Attention 内部 Buffer

| Buffer | 维度 | 内存 |
|--------|------|------|
| d_q_buf | [num_heads, q_head_dim] | 8 KB |
| d_gate_buf | [num_heads, q_head_dim] | 8 KB |
| d_k_buf | [num_kv_heads, kv_head_dim] | 2 KB |
| d_v_buf | [num_kv_heads, kv_head_dim] | 2 KB |
| d_attn_out_buf | [num_heads, kv_head_dim] | 8 KB |
| d_attn_scores_buf | [num_heads, max_seq_len] | 8 MB (8192) |
| **单序列小计** | - | **~8.03 MB** |

**Batch Prefill Buffer（动态分配，max_batch_size）：**

| Buffer | 维度 | batch=128 内存 |
|--------|------|----------------|
| d_batch_q_buf | [batch, num_heads, q_head_dim] | 1 MB |
| d_batch_gate_buf | [batch, num_heads, q_head_dim] | 1 MB |
| d_batch_k_buf | [batch, num_kv_heads, kv_head_dim] | 256 KB |
| d_batch_v_buf | [batch, num_kv_heads, kv_head_dim] | 256 KB |
| d_batch_attn_out_buf | [batch, num_heads, kv_head_dim] | 1 MB |
| **Batch 小计** | - | **~3.5 MB** |

#### 3.4.3 Linear Attention 内部 Buffer

| Buffer | 维度 | 内存 |
|--------|------|------|
| d_mixed_qkv_buf | [conv_dim] | 20 KB |
| d_conv_out_buf | [conv_dim] | 20 KB |
| d_q_buf | [num_heads, key_dim] | 8 KB |
| d_k_buf | [num_heads, key_dim] | 8 KB |
| d_v_buf | [num_heads, value_dim] | 8 KB |
| d_a_buf | [num_heads] | 64 B |
| d_b_raw_buf | [num_heads] | 64 B |
| d_attn_out_buf | [z_dim] | 8 KB |
| d_z_buf | [z_dim] | 8 KB |
| **单序列小计** | - | **~80 KB** |

**Batch Buffer（动态分配）：**

| Buffer | 维度 | batch=128 内存 |
|--------|------|----------------|
| d_batch_mixed_qkv_buf | [batch, conv_dim] | 2.5 MB |
| d_batch_conv_out_buf | [batch, conv_dim] | 2.5 MB |
| d_batch_a_buf | [batch, num_heads] | 8 KB |
| d_batch_b_raw_buf | [batch, num_heads] | 8 KB |
| d_batch_z_buf | [batch, z_dim] | 1 MB |
| d_batch_attn_out_buf | [batch, z_dim] | 1 MB |
| **Batch 小计** | - | **~7 MB** |

#### 3.4.4 MLP 内部 Buffer

| Buffer | 维度 | 内存 |
|--------|------|------|
| d_hidden_buf（batch 时） | [batch, intermediate × 3] | batch=128: 5.4 MB |

#### 3.4.5 LM Head Buffer

| Buffer | 维度 | 内存 |
|--------|------|------|
| d_weight_fp32 | [hidden, vocab] | 958 MB |
| d_weight_bf16 | [hidden, vocab] | 479 MB |
| d_input_bf16 | [hidden] | 2 KB |
| d_output_bf16 | [vocab] | 479 MB |
| **LM Head 小计** | - | **~1,916 MB** |

> 注：LM Head 同时保留了 FP32 和 BF16 两份权重，可优化为仅保留一份。

---

### 3.5 CUDA Graph 内存

| 组件 | 内存占用 | 说明 |
|------|----------|------|
| CUDA Graph 结构 | ~10-50 MB | 与图复杂度相关 |
| Graph Exec 缓存 | ~100-500 MB | 捕获的 kernel 参数和依赖 |
| **CUDA Graph 小计** | **~100-500 MB** | vLLM 实测约 0.44 GB |

---

### 3.6 cuBLAS Workspace

| 组件 | 内存占用 | 说明 |
|------|----------|------|
| cuBLAS Handle Workspace | 默认 4-32 MB/handle | 每层独立 handle |
| 总 Workspace (24 层) | **~96-768 MB** | 可统一为单个 handle 节省 |

---

## 4. 内存消耗汇总

### 4.1 v3.3 版本内存汇总 (batch=1)

| 类别 | 内存 (MB) | 占比 | 优化优先级 |
|------|-----------|------|------------|
| **模型权重** | 3,693 | 42.1% | 高 |
| ├─ Layer 权重 | 1,777 | 20.3% | 中 |
| ├─ Embedding | 958 | 10.9% | 高 |
| ├─ LM Head | 958 | 10.9% | 高 |
| └─ Final Norm | 0.004 | ~0% | 低 |
| **KV Cache** | 402 | 4.6% | 高 |
| **Linear State** | 13 | 0.1% | 低 |
| **激活 Buffer** | ~1,000 | 11.4% | 中 |
| ├─ Engine Buffer | ~958 | 10.9% | 中 |
| ├─ Attention Buffer | ~8 | 0.1% | 低 |
| └─ Linear Buffer | ~0.08 | ~0% | 低 |
| **LM Head 双精度** | ~1,437 | 16.4% | 高 |
| **CUDA Graph** | ~200 | 2.3% | 低 |
| **cuBLAS Workspace** | ~384 | 4.4% | 中 |
| **其他开销** | ~1,683 | 19.2% | - |
| **总计 (RTX 3080 Ti 实测)** | **8,772** | **100%** | - |

> 注：实际占用与理论计算存在差异，主要由于 CUDA Runtime、驱动开销、内存对齐、以及动态分配未完全统计。

### 4.2 内存占用饼图（理论估算）

```
模型权重 (3,693 MB)     ████████████████████████████████████████  42.1%
LM Head 双精度 (1,437)   ████████████████                          16.4%
其他开销 (1,683)         ██████████████████                        19.2%
激活 Buffer (~1,000)     ███████████                               11.4%
KV Cache (402)           ███                                        4.6%
cuBLAS Workspace (384)   ███                                        4.4%
CUDA Graph (~200)        ██                                         2.3%
Linear State (13)        ~0%                                        0.1%
```

---

## 5. 内存优化策略

### 5.1 高优先级优化

#### 5.1.1 权重 FP16/BF16 量化

**预期收益：**
- 权重内存减半：3,693 MB → 1,847 MB（节省 1,846 MB）
- Decode 速度提升 20-30%（带宽瓶颈缓解）

**实现方案：**
- 权重加载时转换为 BF16
- GEMM 使用 BF16 Tensor Core
- 已在 `lm_head_cuda.cu` 中实现 BF16 路径，可扩展到所有层

**Trade-off：**
- 轻微精度损失（BF16 通常可接受）
- 需要修改所有 kernel 支持 BF16 输入

#### 5.1.2 LM Head + Embedding 统一与量化

**当前问题：**
- Embedding 和 LM Head 各占用 958 MB，共 1,916 MB
- LM Head 同时保留 FP32 和 BF16 两份权重

**优化方案：**
- Embedding 和 LM Head 共享同一份权重（Qwen 模型中两者相同）
- 量化为 BF16：958 MB → 479 MB
- 仅保留一份：479 MB（节省 1,437 MB）

**预期收益：** 节省 **1,437 MB**

#### 5.1.3 KV Cache 动态分配

**当前问题：**
- 预分配 max_seq_len=8192 或 262144
- 实际序列长度通常远小于最大值

**优化方案：**
- 根据实际输入长度动态分配 KV Cache
- 使用 Paged KV Cache 分页管理

**预期收益：**
- 短序列场景节省 90%+ KV Cache 内存
- 支持更长上下文（按需扩展）

### 5.2 中优先级优化

#### 5.2.1 统一 cuBLAS Handle

**当前问题：**
- 每层独立创建 cuBLAS handle（24 个）
- 每个 handle 有独立 workspace

**优化方案：**
- 全局共享单个 cuBLAS handle
- 或使用 handle pool（线程安全）

**预期收益：** 节省 **~300-700 MB** workspace

#### 5.2.2 激活 Buffer 复用

**当前问题：**
- Engine 级别 buffer 与 Layer 级别 buffer 分离
- Batch 和单序列 buffer 分离

**优化方案：**
- 统一分配大块内存，各组件按需偏移使用
- Batch=1 时复用 batch buffer

**预期收益：** 节省 **~500 MB**

#### 5.2.3 MLP 中间 Buffer 优化

**当前问题：**
- `d_hidden_buf_` 分配 `batch × intermediate × 3`
- 实际只需要 `batch × intermediate × 2`（gate + up，down 可直接写输出）

**预期收益：** 节省 **~30% MLP buffer**

### 5.3 低优先级优化

#### 5.3.1 Linear Attention State FP16

**预期收益：** 节省 **~6.6 MB**（收益较小）

#### 5.3.2 CUDA Graph 内存优化

**预期收益：** 有限，Graph 本身内存占用不大

---

## 6. 优化收益汇总

| 优化项 | 节省内存 | 实现难度 | 风险 |
|--------|----------|----------|------|
| 权重 BF16 量化 | 1,846 MB | 中 | 低 |
| LM Head 统一+量化 | 1,437 MB | 低 | 低 |
| KV Cache 动态分配 | 300-400 MB | 中 | 中 |
| 统一 cuBLAS Handle | 300-700 MB | 低 | 低 |
| Buffer 复用 | ~500 MB | 中 | 低 |
| **总计** | **~4,400 MB** | - | - |

**优化后预期内存：** 8,772 MB → **~4,400 MB**（节省约 50%）

---

## 7. 各版本内存对比

| 版本 | 权重精度 | KV Cache 策略 | 显存占用 (RTX 3080 Ti) | 说明 |
|------|----------|---------------|------------------------|------|
| v1.0 | FP32 | 预分配 | ~9,500 MB | 基础实现 |
| v2.0 | FP32 | 预分配 8192 | ~8,800 MB | Gated Delta 优化 |
| v3.0 | FP32 | 预分配 8192 | ~9,100 MB | All Flash Attention |
| v3.3 | FP32 | 预分配 8192 | **8,772 MB** | Kernel Memory Opt |
| **v4.0 (目标)** | **BF16** | **动态分配** | **~4,500 MB** | 全面量化+动态内存 |

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| [src/backend/cuda/kernels/cuda_engine.cu](src/backend/cuda/kernels/cuda_engine.cu) | Engine 内存分配 |
| [src/backend/cuda/kernels/full_attention_cuda.cu](src/backend/cuda/kernels/full_attention_cuda.cu) | Full Attention Buffer |
| [src/backend/cuda/kernels/linear_attention_cuda.cu](src/backend/cuda/kernels/linear_attention_cuda.cu) | Linear Attention Buffer |
| [src/backend/cuda/kernels/mlp_cuda.cu](src/backend/cuda/kernels/mlp_cuda.cu) | MLP Buffer |
| [src/backend/cuda/kernels/lm_head_cuda.cu](src/backend/cuda/kernels/lm_head_cuda.cu) | LM Head 双精度权重 |
| [docs/optimization/PERFORMANCE_TEST_TEMPLATE.md](docs/optimization/PERFORMANCE_TEST_TEMPLATE.md) | 性能测试数据 |

---

## 9. 附录

### 9.1 内存计算公式

```
权重内存 = 参数量 × sizeof(dtype)
KV Cache = num_layers × seq_len × num_kv_heads × head_dim × 2 × sizeof(float)
Linear State = num_layers × num_heads × key_dim × value_dim × sizeof(float)
Activation = batch_size × hidden_size × sizeof(float)
```

### 9.2 单位换算

| 单位 | 换算 |
|------|------|
| 1 MB | 1,024 KB |
| 1 GB | 1,024 MB |
| FP32 | 4 bytes |
| FP16/BF16 | 2 bytes |
| INT8 | 1 byte |
