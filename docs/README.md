# Qwen3.5-0.8B C++/CUDA 推理引擎 - 文档中心

本文档中心提供项目的所有技术文档，按类别组织，方便快速定位所需信息。

## 核心性能指标

| 测试条件 | Prefill TTFT | Prefill 吞吐 | Decode TPOT | Decode 吞吐 |
|---------|-------------|-------------|-------------|-------------|
| RTX 5060 Ti, prefill=1024, decode=512 | **1,948 ms** | **525.6 tok/s** | **0.062 ms/tok** | **16,248 tok/s** |

### 性能演进

| 版本 | Prefill 吞吐 | Decode 吞吐 | TTFT | 主要优化 |
|------|-------------|-------------|------|---------|
| v1.0 (CUDA Baseline) | 17.5 tok/s | 12.5 tok/s | 58,400 ms | CUDA 基础实现，单 token 串行 |
| v2.0 (FlashAttention) | 86.4 tok/s | 15,774 tok/s | 11,856 ms | FlashAttention v2 + Tensor Core + Batch Prefill |
| **v3.0 (Batch GEMM)** | **525.6 tok/s** | **16,248 tok/s** | **1,948 ms** | **Batch Linear Attention + cuBLAS GEMM + Kernel Fusion** |

---

## 快速导航

### 新手入门
- [项目架构概览](implementation/qwen3_5_0_8b_architecture.md) - 了解整体设计
- [CUDA 优化日志](implementation/cuda_optimization_log.md) - 详细优化记录与性能数据
- [快速开始指南](../README.md) - 项目主 README

### 模型架构详解
- [00_总览与索引](qwen3_5_0_8b_details/00_总览与索引.md) - 模型架构总览
- [01_总体架构_实现级流程](qwen3_5_0_8b_details/01_总体架构_实现级流程.md) - 端到端执行流程
- [02_文本主干_24层调度与参数映射](qwen3_5_0_8b_details/02_文本主干_24层调度与参数映射.md) - 24层调度逻辑
- [03_LinearAttention_GatedDeltaNet_细节](qwen3_5_0_8b_details/03_LinearAttention_GatedDeltaNet_细节.md) - 线性注意力实现
- [04_FullAttention_GQA_RoPE_细节](qwen3_5_0_8b_details/04_FullAttention_GQA_RoPE_细节.md) - 全注意力实现
- [05_MLP_层归一化_输出头](qwen3_5_0_8b_details/05_MLP_层归一化_输出头.md) - MLP 和归一化
- [06_视觉编码器与Merger_细节](qwen3_5_0_8b_details/06_视觉编码器与Merger_细节.md) - 视觉处理模块
- [07_MTP分支与推测解码_参数加载](qwen3_5_0_8b_details/07_MTP分支与推测解码_参数加载.md) - MTP 分支

### 性能优化与基准测试
- [CUDA 优化日志](implementation/cuda_optimization_log.md) - 完整优化记录（推荐）
- [BF16 GEMM 基准测试](implementation/bf16_prefill_benchmark.md) - BF16 矩阵乘法（已归档）
- [Batch POC 基准测试](implementation/batch_poc_benchmark.md) - 批处理 POC 测试（已归档）
- [基准测试自动化](implementation/benchmark_automation_principle_and_implementation.md) - 自动化测试框架

### 开发与调试
- [精度验证方法](implementation/accuracy_validation_methodology.md) - 精度验证流程
- [项目结构规范](implementation/项目结构体设计规范.md) - 代码组织规范

---

## 文档目录结构

```
docs/
├── README.md                          # 本文档（统一入口）
├── implementation/                    # 实现相关文档
│   ├── cuda_optimization_log.md       # CUDA 优化详细记录（核心文档）
│   ├── qwen3_5_0_8b_architecture.md   # 项目架构概览
│   ├── accuracy_validation_methodology.md  # 精度验证方法
│   ├── bf16_prefill_benchmark.md      # BF16 基准测试（已归档）
│   ├── batch_poc_benchmark.md         # 批处理 POC 测试（已归档）
│   ├── benchmark_automation_principle_and_implementation.md  # 自动化测试
│   └── 项目结构体设计规范.md          # 代码组织规范
└── qwen3_5_0_8b_details/              # 模型架构详解
    ├── 00_总览与索引.md
    ├── 01_总体架构_实现级流程.md
    ├── 02_文本主干_24层调度与参数映射.md
    ├── 03_LinearAttention_GatedDeltaNet_细节.md
    ├── 04_FullAttention_GQA_RoPE_细节.md
    ├── 05_MLP_层归一化_输出头.md
    ├── 06_视觉编码器与Merger_细节.md
    └── 07_MTP分支与推测解码_参数加载.md
```

---

## 推荐阅读路径

### 路径 1：了解项目全貌（适合新贡献者）
1. [项目架构概览](implementation/qwen3_5_0_8b_architecture.md)
2. [CUDA 优化日志](implementation/cuda_optimization_log.md)
3. [模型架构总览](qwen3_5_0_8b_details/00_总览与索引.md)

### 路径 2：深入技术细节（适合开发者）
1. [总体架构实现流程](qwen3_5_0_8b_details/01_总体架构_实现级流程.md)
2. [LinearAttention 细节](qwen3_5_0_8b_details/03_LinearAttention_GatedDeltaNet_细节.md)
3. [FullAttention 细节](qwen3_5_0_8b_details/04_FullAttention_GQA_RoPE_细节.md)
4. [CUDA 优化日志](implementation/cuda_optimization_log.md)

### 路径 3：性能优化（适合优化工程师）
1. [CUDA 优化日志](implementation/cuda_optimization_log.md)
2. [BF16 GEMM 基准测试](implementation/bf16_prefill_benchmark.md)（已归档）
3. [Batch POC 基准测试](implementation/batch_poc_benchmark.md)（已归档）

---

## 关键 CUDA 优化

| 优化项 | 效果 | 文档 |
|--------|------|------|
| Batch Linear Attention cuBLAS GEMM | Prefill +482% | [cuda_optimization_log.md](implementation/cuda_optimization_log.md) |
| Flash Attention v2 | 内存复杂度 O(1) | [cuda_optimization_log.md](implementation/cuda_optimization_log.md) |
| Kernel Fusion | 减少 kernel launch | [cuda_optimization_log.md](implementation/cuda_optimization_log.md) |
| CUDA Graph Decode | 消除 launch 开销 | [cuda_optimization_log.md](implementation/cuda_optimization_log.md) |

---

## 相关链接

- [项目主 README](../README.md)
- [GitHub 仓库](https://github.com/zhils/Qwen3.5_0.8B_Deploy)
- [CMakeLists.txt](../CMakeLists.txt)

---

最后更新：2026-04-27
