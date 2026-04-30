***

# Qwen3.5-0.8B CUDA推理引擎 - 项目概述

<br />

## 项目背景

本项目是 Qwen3.5-0.8B 模型的高性能 CUDA 推理引擎，采用 C++ + CUDA 实现。

### 核心性能指标

| 测试条件                   | Prefill TTFT | Prefill 吞吐  | Decode TPOT  | Decode 吞吐    |
| ---------------------- | ------------ | ----------- | ------------ | ------------ |
| RTX 5060 Ti, batch=128 | 1,497 ms     | 684.0 tok/s | 0.086 ms/tok | 11,682 tok/s |
| RTX 5060 Ti, batch=1   | 2,149 ms     | 444.2 tok/s | 0.062 ms/tok | 16,133 tok/s |
| RTX 3080 Ti, batch=1   | 2,167 ms     | 574.5 tok/s | 0.099 ms/tok | 10,147 tok/s |

### 与 llama.cpp 对比 (RTX 5060 Ti)

| 指标        | 本项目 (v3.3)   | llama.cpp  | 优势        |
| --------- | ------------ | ---------- | --------- |
| Decode 吞吐 | 16,133 tok/s | 191.79 t/s | **84.1x** |
| TPOT      | 0.062 ms     | 5.21 ms    | **84.0x** |

## 项目结构

```
Qwen3.5_0.8B_Deploy/
├── src/
│   ├── backend/
│   │   ├── cpu/                    # CPU 后端（参考实现）
│   │   │   ├── core/              # 核心模块
│   │   │   │   ├── attention/      # Linear/Full Attention
│   │   │   │   ├── common/         # Language Backbone
│   │   │   │   ├── embedding/      # Token Embedding
│   │   │   │   ├── heads/          # LM Head, MTP, Sampler
│   │   │   │   └── mlp/            # MLP Layer
│   │   │   └── vision/             # Vision Encoder
│   │   └── cuda/                  # CUDA 后端（主路径）
│   │       ├── include/           # 头文件
│   │       └── kernels/           # CUDA Kernels
│   │           ├── cuda_engine.cu      # 引擎主控
│   │           ├── cuda_engine_v3.cu   # V3 引擎
│   │           ├── flash_attention.cu  # Flash Attention
│   │           ├── linear_attention_cuda.cu  # Linear Attention
│   │           ├── linear_attention_v2.cu    # V2 优化
│   │           ├── fused_kernels.cu         # 融合 Kernel
│   │           ├── mlp_cuda.cu              # MLP
│   │           └── rmsnorm_cuda.cu          # RMSNorm
│   ├── tests/
│   │   ├── unit/          # 单元测试
│   │   └── integration/   # 集成测试
│   └── docs/
│       ├── implementation/      # 实现文档
│       └── qwen3_5_0_8b_details/  # 模型架构详解
├── lossy_optimization/   # 有损优化实验（不动）
├── others/              # 归档内容
├── CMakeLists.txt
└── README.md
```

## 技术栈

| 组件        | 技术                                    |
| --------- | ------------------------------------- |
| 语言        | C++17, CUDA                           |
| BLAS      | cuBLAS (Tensor Core TF32)             |
| Attention | Flash Attention v2 + Linear Attention |
| 内存        | BF16 权重, INT8 KV Cache, Paged KV      |
| 优化        | Kernel Fusion, CUDA Graph, Batch GEMM |

## 模型配置

| 参数                      | 值                                 |
| ----------------------- | --------------------------------- |
| Hidden Size             | 1024                              |
| Intermediate Size       | 3584                              |
| Num Layers              | 24 (18 Linear + 6 Full Attention) |
| Attention Heads         | 16                                |
| Head Dim                | 64                                |
| Max Position Embeddings | 32768                             |
| Vocab Size              | 151936                            |

## 关键文件

| 文件                                                       | 用途                     |
| -------------------------------------------------------- | ---------------------- |
| `src/backend/cuda/kernels/cuda_engine_v3.cu`             | V3 引擎主入口               |
| `src/backend/cuda/kernels/cuda_engine.cu`                | V2 引擎                  |
| `src/backend/cuda/kernels/linear_attention_cuda.cu`      | Batch Linear Attention |
| `src/backend/cuda/kernels/flash_attention.cu`            | Flash Attention        |
| `src/backend/cuda/kernels/fused_kernels.cu`              | 融合 Kernels             |
| `docs/implementation/accuracy_validation_methodology.md` | 精度验证方法论                |

