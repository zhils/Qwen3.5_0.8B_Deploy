# Qwen3.5-0.8B Deploy 有损精度内存优化分析

> 文档版本: v1.0
> 更新日期: 2026-04-28
> 适用范围: CUDA Engine v3.3

---

## 1. 概述

本文档分析在**可接受精度损失**的前提下，项目中的内存优化空间。有损优化通过降低数据精度（FP16/BF16/INT8/INT4）来减少显存占用，同时可能提升推理速度（因为带宽需求降低）。

所有方案都需要经过**精度验证**和** perplexity 评估**后才能上线。

---

## 2. 当前精度基线

| 组件 | 当前精度 | 内存占用 |
|------|----------|----------|
| 模型权重 | FP32 | 3,693 MB |
| KV Cache | FP32 | 402 MB |
| Linear State | FP32 | 13 MB |
| 激活值 | FP32 | ~1,000 MB |
| LM Head 计算 | BF16 (已有) | 479 MB |

---

## 3. 有损优化方案

### 3.1 权重量化（最大收益）

#### 3.1.1 全局 BF16 权重量化

**方案：**
- 将所有 FP32 权重转换为 BF16
- 使用 BF16 Tensor Core 进行 GEMM 计算

**内存收益：**
- 权重内存减半：3,693 MB → **1,847 MB**（节省 **1,846 MB**）

**精度影响：**
- BF16 有 8 位指数（与 FP32 相同），3 位尾数
- 对于 0.8B 小模型，通常**无明显精度损失**
- 需要验证 perplexity 变化 < 1%

**性能影响：**
- Decode 阶段带宽减半，速度提升 **20-30%**
- Prefill 阶段 Tensor Core 利用率更高

**实现难度：** 中
**风险：** 低

**实现要点：**
1. 权重加载时转换为 BF16
2. 所有 GEMM 使用 `cublasGemmEx` with `CUDA_R_16BF`
3. Kernel 输入/输出改为 BF16 或保持 FP32（根据精度需求）

**参考实现：**
- `lm_head_cuda.cu` 已实现 BF16 路径，可作为模板
- 需要扩展到：MLP、Attention、RMSNorm、Embedding

---

#### 3.1.2 分层混合精度量化

**方案：**
- 对不敏感层使用 INT8，敏感层保持 BF16/FP32
- 例如：Embedding 和 LM Head 保持高精度（对输出影响大）
- MLP 和 Attention 投影使用 INT8

**内存收益：**
- 部分权重降至 1 byte：额外节省 **~25-50%**
- 总体权重内存：3,693 MB → **~900-1,400 MB**

**精度影响：**
- 需要逐层敏感度分析
- 通常 MLP 权重对 INT8 容忍度较高
- Attention Q/K/V 投影可能需要保持 BF16

**实现难度：** 高
**风险：** 中

**实现要点：**
1. 权重敏感度分析（逐层评估 INT8 影响）
2. 使用 INT8 GEMM（cuBLAS 支持）
3. 需要校准（calibration）确定缩放因子

---

#### 3.1.3 INT4 权重量化（极致压缩）

**方案：**
- 使用 INT4 存储权重（0.5 bytes/参数）
- 解包到 INT8/BF16 后计算

**内存收益：**
- 权重内存降至 1/8：3,693 MB → **~460 MB**
- 加上 INT4 → BF16 查找表，实际约 **~550 MB**

**精度影响：**
- 0.8B 小模型可能对 INT4 敏感
- 预计 perplexity 上升 2-5%
- 需要分组量化（per-channel or per-group）

**实现难度：** 很高
**风险：** 高

**实现要点：**
1. 权重打包（2 个 INT4 存到一个 uint8_t）
2. 反包 kernel（INT4 → BF16）
3. 或使用 NVIDIA 的 CUTLASS INT4 kernel

---

### 3.2 KV Cache 量化

#### 3.2.1 FP16/BF16 KV Cache

**方案：**
- KV Cache 从 FP32 降至 FP16 或 BF16

**内存收益：**
- KV Cache 减半：402 MB → **201 MB**（节省 **201 MB**）
- 若 max_seq_len=262144：12.6 GB → **6.3 GB**

**精度影响：**
- KV Cache 存储的是经过投影后的特征
- 通常对精度不敏感，FP16/BF16 足够
- 已有实验代码：`kv_int8_cuda.cu`

**性能影响：**
- Decode 阶段 KV Cache 读取带宽减半
- 速度提升 **10-15%**

**实现难度：** 低-中
**风险：** 低

**实现要点：**
1. `CudaKVCache` 改为 `__half*` 或 `__nv_bfloat16*`
2. FlashAttention kernel 支持 FP16/BF16
3. KV 投影后转换为 FP16/BF16

**参考文件：**
- [src/backend/cuda/kernels/kv_int8_cuda.cu](src/backend/cuda/kernels/kv_int8_cuda.cu)（已有 INT8 实验）

---

#### 3.2.2 INT8 KV Cache

**方案：**
- KV Cache 使用 INT8 存储
- 每通道（per-channel）缩放因子

**内存收益：**
- KV Cache 降至 1/4：402 MB → **~100 MB**

**精度影响：**
- 需要校准缩放因子
- 通常对 decode 质量影响较小
- 已有实验代码支持

**实现难度：** 中
**风险：** 中

**实现要点：**
1. 存储 INT8 KV + 缩放因子
2. 读取时反量化为 FP16/BF16 后计算
3. 或修改 FlashAttention 支持 INT8 输入

---

### 3.3 激活值量化

#### 3.3.1 动态 INT8 激活量化

**方案：**
- 每层输入/输出使用 INT8
- 动态计算缩放因子（per-token）

**内存收益：**
- 激活 buffer 减半或更多：~1,000 MB → **~250-500 MB**

**精度影响：**
- 动态量化通常比静态量化精度高
- 但 kernel 实现复杂

**实现难度：** 很高
**风险：** 中-高

---

### 3.4 Linear Attention State 量化

#### 3.4.1 FP16 State

**方案：**
- Recurrent State 和 Conv State 使用 FP16

**内存收益：**
- 13 MB → **6.5 MB**（节省 **6.5 MB**）

**精度影响：**
- Linear Attention 的 state 是累加状态
- FP16 可能累积误差，需要评估

**实现难度：** 低
**风险：** 低-中

---

### 3.5 综合量化策略

#### 3.5.1 推荐配置：BF16 全局 + FP16 KV Cache

| 组件 | 精度 | 内存 |
|------|------|------|
| 权重 | BF16 | 1,847 MB |
| KV Cache | FP16 | 201 MB |
| Linear State | FP16 | 6.5 MB |
| 激活 | FP16 | ~500 MB |
| Embedding/LM Head 共享 | BF16 | 479 MB |
| **总计（估算）** | - | **~3,000 MB** |

**相比当前 8,772 MB，节省约 65%！**

#### 3.5.2 激进配置：INT8 权重 + INT8 KV Cache

| 组件 | 精度 | 内存 |
|------|------|------|
| 权重 | INT8 | ~923 MB |
| KV Cache | INT8 | ~100 MB |
| Linear State | FP16 | 6.5 MB |
| 激活 | FP16 | ~500 MB |
| **总计（估算）** | - | **~1,500 MB** |

**相比当前 8,772 MB，节省约 83%！**

---

## 4. 有损优化收益汇总

| 优化项 | 节省内存 | 实现难度 | 精度风险 | 速度提升 |
|--------|----------|----------|----------|----------|
| 权重 BF16 量化 | **1,846 MB** | 中 | 低 | +20-30% |
| KV Cache FP16 | **201 MB** | 低 | 很低 | +10-15% |
| KV Cache INT8 | **~300 MB** | 中 | 低 | +15-20% |
| 权重 INT8 | **~2,770 MB** | 高 | 中 | +30-40% |
| 权重 INT4 | **~3,230 MB** | 很高 | 高 | +40-50% |
| Linear State FP16 | **6.5 MB** | 低 | 低 | 微量 |
| 激活动态量化 | **~500 MB** | 很高 | 中 | 微量 |
| **BF16 综合方案** | **~5,500 MB** | 中 | 低 | **+20-30%** |
| **INT8 综合方案** | **~7,200 MB** | 高 | 中 | **+30-40%** |

---

## 5. 精度验证策略

### 5.1 单元验证
- 每层输出对比：FP32 vs 量化后
- Max diff < 1e-3（BF16）或 < 1e-2（INT8）

### 5.2 端到端验证
- 使用标准 prompt 测试输出一致性
- 对比生成文本的语义等价性

### 5.3 Perplexity 评估
- 在验证集上计算 perplexity
- BF16: 目标 perplexity 变化 < 1%
- INT8: 目标 perplexity 变化 < 5%

### 5.4 下游任务评估
- 如有条件，在标准 benchmark 上评估
- 例如：HellaSwag, ARC, MMLU 等

---

## 6. 实施路线图

### Phase 1: BF16 基础量化（1-2 周）
1. **Embedding BF16 量化**
2. **MLP BF16 量化**
3. **Attention 投影 BF16 量化**
4. **RMSNorm 保持 FP32**（对精度敏感）
5. **精度验证**

### Phase 2: KV Cache FP16（3-5 天）
1. **修改 CudaKVCache 为 FP16**
2. **修改 FlashAttention 支持 FP16 KV**
3. **精度验证**

### Phase 3: INT8 实验（2-3 周）
1. **权重敏感度分析**
2. **INT8 GEMM 集成**
3. **校准流程**
4. **全面精度验证**

### Phase 4: 极致优化（可选）
1. **INT4 实验**
2. **激活量化**
3. **Paged KV Cache**

---

## 7. 与无损优化的叠加效果

| 方案 | 无损优化 | 有损优化 | 合计节省 | 最终内存 |
|------|----------|----------|----------|----------|
| 仅无损 | ~1,800 MB | - | ~1,800 MB | ~6,800 MB |
| 仅 BF16 | - | ~3,500 MB | ~3,500 MB | ~5,200 MB |
| 无损 + BF16 | ~1,800 MB | ~3,500 MB | ~5,300 MB | **~3,400 MB** |
| 无损 + INT8 | ~1,800 MB | ~5,500 MB | ~7,300 MB | **~1,400 MB** |

**最佳实践：先实施无损优化，再叠加有损优化。**

---

## 8. 相关文件与参考

| 文件 | 说明 |
|------|------|
| [src/backend/cuda/kernels/lm_head_cuda.cu](src/backend/cuda/kernels/lm_head_cuda.cu) | BF16 GEMM 参考实现 |
| [src/backend/cuda/kernels/kv_int8_cuda.cu](src/backend/cuda/kernels/kv_int8_cuda.cu) | INT8 KV Cache 实验 |
| [src/backend/cuda/include/kv_int8_cuda.hpp](src/backend/cuda/include/kv_int8_cuda.hpp) | INT8 KV Cache 接口 |
| [docs/memory_optimization/memory_consumption_analysis.md](memory_consumption_analysis.md) | 内存基线分析 |
| [docs/memory_optimization/lossless_optimization_analysis.md](lossless_optimization_analysis.md) | 无损优化方案 |

---

## 9. 参考工具

### 9.1 NVIDIA 官方工具
- **TensorRT**: 自动 INT8/FP16 量化，支持 Qwen 模型
- **CUDA Toolkit**: `cuda_fp16.h`, `cuda_bf16.h`
- **cuBLAS**: `cublasGemmEx` 支持混合精度

### 9.2 开源量化框架
- **AutoGPTQ**: GPTQ 量化，支持 4-bit
- **llama.cpp**: GGUF 量化格式参考
- **vLLM**: FP8/INT8 KV Cache 实现参考

### 9.3 本项目工具
- [src/backend/cuda/tools/gpu_memory_profiler.hpp](src/backend/cuda/tools/gpu_memory_profiler.hpp): 内存分析工具
- `performance_test`: 性能基准测试
- `v2_kernel_accuracy_validate`: 精度验证
