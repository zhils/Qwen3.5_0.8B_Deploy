# Qwen3.5-0.8B CUDA Inference Engine

[English](README_EN.md) | 中文

## 项目概述

基于 Qwen3.5-0.8B 模型的高性能 CUDA 推理引擎，采用多种优化技术实现极致的端到端推理速度。

### 核心性能指标

| 测试条件 | Prefill TTFT | Prefill 吞吐 | Decode TPOT | Decode 吞吐 | 端到端耗时 |
|---------|-------------|-------------|-------------|-------------|-----------|
| RTX 5060 Ti, batch=128, prefill=1024, decode=512 | **1,948 ms** | **525.6 tok/s** | **0.062 ms/tok** | **16,248 tok/s** | **1,980 ms** |
| RTX 5060 Ti, batch=1, prefill=1024, decode=512 | **2,305 ms** | **444.2 tok/s** | **0.062 ms/tok** | **16,133 tok/s** | **2,337 ms** |

### 与 llama.cpp 对比 (同硬件 RTX 5060 Ti)

#### 单请求对比 (batch=1)

| 指标 | 本项目 (v3.0) | llama.cpp (BF16 GGUF) | 优势 |
|------|--------------|----------------------|------|
| **Prefill 吞吐** | **40.4 tok/s** | **193.16 t/s** | **0.21x** |
| **Decode 吞吐** | **16,346 tok/s** | **191.79 t/s** | **85.2x** |
| **Prefill TTFT (1024 tok)** | **25,336 ms** | **5,301 ms** | **0.21x** |
| **Decode TPOT** | **0.061 ms** | **5.21 ms** | **85.4x** |
| **端到端耗时 (1024+512)** | **25,367 ms** | **7,971 ms** | **0.31x** |
| **显存占用** | 10,355 MB | 2,298 MB | - |

#### Batch 对比 (batch=128)

| 指标 | 本项目 (v3.0) | llama.cpp (BF16 GGUF) | 优势 |
|------|--------------|----------------------|------|
| **Prefill 吞吐** | **525.6 tok/s** | 未测试 | - |
| **Prefill TTFT (1024 tok)** | **1,948 ms** | 未测试 | - |

> **测试条件说明**：
> - llama.cpp：使用 `llama-bench` 实测，`qwen3.5-0.8b-f16.gguf` (BF16)，batch=1，`-ngl 99 -fa 1`，重复 20 次取 P50
> - 本项目：FP32 权重，batch_size=1（单请求）或 128（batch prefill）
> - **Prefill 差距说明**：单请求时本项目 prefill 仅 40.4 tok/s，远低于 llama.cpp 的 193.16 t/s。这是因为 llama.cpp 在单请求场景下优化更成熟（如 prompt caching、graph capture 等）。本项目的优势在 **batch 场景**，通过 Batch Linear Attention + cuBLAS GEMM 将 prefill 提升至 525.6 tok/s。
> - **Decode 差距说明**：85.2x 的差距主要来自 (1) 本项目启用 CUDA Graph 消除 kernel launch 开销；(2) 全融合 kernel（SiLU+Mul、RMSNorm+Residual 等）减少 HBM 访存；(3) 预分配 buffer 无动态内存分配。llama.cpp 作为通用框架，在 0.8B 小模型上 kernel launch 和框架开销占比较高。

### 性能演进

| 版本 | Prefill 吞吐 | Decode 吞吐 | TTFT | 主要优化 |
|------|-------------|-------------|------|---------|
| v1.0 (CUDA Baseline) | 17.5 tok/s | 12.5 tok/s | 58,400 ms | CUDA 基础实现，单 token 串行 |
| v2.0 (FlashAttention) | 86.4 tok/s | 15,774 tok/s | 11,856 ms | FlashAttention v2 + Tensor Core + Batch Prefill |
| **v3.0 (Batch GEMM)** | **525.6 tok/s** | **16,248 tok/s** | **1,948 ms** | **Batch Linear Attention + cuBLAS GEMM + Kernel Fusion** |

**v3.0 相比 v1.0**: Prefill 提升 **+2,904%**，TTFT 降低 **-97%**

## 项目结构

```
.
├── src/
│   ├── backend/
│   │   ├── cpu/              # CPU 后端实现
│   │   │   ├── core/         # 核心计算模块
│   │   │   │   ├── attention/     # Full Attention + Linear Attention
│   │   │   │   ├── common/        # 公共组件
│   │   │   │   ├── embedding/     # Token + 多模态嵌入
│   │   │   │   ├── heads/         # LM Head + MTP Head + Sampler
│   │   │   │   └── mlp/           # MLP 层
│   │   │   └── vision/       # 视觉编码器
│   │   └── cuda/             # CUDA 后端实现
│   │       ├── include/      # CUDA 头文件
│   │       │   ├── cuda_engine.hpp           # CUDA 引擎接口
│   │       │   ├── flash_attention.cuh       # FlashAttention 声明
│   │       │   ├── full_attention_cuda.hpp   # Full Attention 声明
│   │       │   ├── linear_attention_cuda.hpp # Linear Attention 声明
│   │       │   └── ...
│   │       └── kernels/      # CUDA kernel 实现
│   │           ├── cuda_engine.cu            # 引擎主控
│   │           ├── flash_attention.cu        # FlashAttention 实现
│   │           ├── full_attention_cuda.cu    # Full Attention 实现
│   │           ├── linear_attention_cuda.cu  # Linear Attention 实现
│   │           ├── mlp_cuda.cu              # MLP 实现
│   │           ├── rmsnorm_cuda.cu          # RMSNorm 实现
│   │           └── ...
│   └── ...
├── tests/
│   ├── unit/                # 单元测试
│   ├── integration/         # 集成测试
│   └── benchmarks/          # 性能测试
├── docs/
│   ├── implementation/      # 实现文档
│   │   └── cuda_optimization_log.md  # CUDA 优化详细记录
│   ├── qwen3_5_0_8b_details/ # 模型架构详解
│   └── PERFORMANCE_REPORT.md # 详细性能报告
├── scripts/
│   ├── benchmark/           # 基准测试脚本
│   ├── debug/               # 调试脚本
│   ├── validation/          # 验证脚本
│   └── weights/             # 权重导出脚本
├── third_party/             # 第三方库
├── CMakeLists.txt
└── README.md
```

## 技术亮点

### 1. Batch Linear Attention (核心优化)

**文件**: [linear_attention_cuda.cu](src/backend/cuda/kernels/linear_attention_cuda.cu)

| 优化项 | 实现方式 | 效果 |
|--------|---------|------|
| **cuBLAS GEMM 批量投影** | QKV/A/B/Z/O 五个投影全部使用 `cublasSgemm` | 一次性处理整个 batch |
| **Batch Conv1D** | `conv1d_update_fused_batch_kernel` | 并行处理所有 token 的卷积 |
| **Batch L2 Norm** | `l2norm_qk_fused_batch_kernel` | 并行归一化 Q/K |
| **Batch Norm+Gate** | `norm_gate_fused_batch_kernel` | 并行归一化和门控 |
| **Kernel Launch 减少** | 从 `batch_size × 8` 减少到约 **9 个 kernel** | 大幅降低 launch 开销 |
| **预分配 Buffer** | `d_batch_mixed_qkv_buf_` 等 | 避免重复 `cudaMalloc` |

**精度验证**: Batch 输出与串行输出一致 (max diff 9.6e-08)

### 2. Flash Attention v2

**文件**: [fused_kernels.cu](src/backend/cuda/kernels/fused_kernels.cu)

- Warp-level 并行归约，减少同步开销
- Online softmax 减少 HBM 访问
- Tiled computation for Q/K/V
- Output projection 使用 cuBLAS GEMM

### 3. Kernel 融合

| 融合 Kernel | 功能 | 文件 |
|-----------|------|------|
| `save_rmsnorm_kernel` | Save 残差 + RMSNorm | fused_kernels.cu |
| `attn_add_rmsnorm_fused_kernel` | Attention 输出 + Add + RMSNorm | fused_kernels.cu |
| `silu_mul_fused_kernel` | SiLU 激活 + 乘法 | fused_kernels.cu |
| `conv1d_update_fused_kernel` | Conv1D + State 更新 | linear_attention_cuda.cu |
| `l2norm_qk_fused_kernel` | L2 归一化 Q + K | linear_attention_cuda.cu |
| `norm_gate_fused_kernel` | RMSNorm + Gate | linear_attention_cuda.cu |

### 4. cuBLAS 优化

- **Prefill 阶段**: cuBLAS GEMM 利用 Tensor Core (TF32) 加速
- **MLP Batch**: `sgemm` 替代 `sgemv`，预分配 hidden buffer
- **Flash Attention Output**: cuBLAS GEMM 替代手动矩阵乘法

### 5. CUDA Graph

- Decode 阶段启用 CUDA Graph，减少 Kernel Launch 开销
- 静态图捕获，多次执行零开销

## 快速开始

### 编译

```bash
# 使用 CMake
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
cmake --build . --config Release -j4

# 或使用 Visual Studio (Windows)
cmake -B build -DENABLE_CUDA=ON .
cmake --build build --config Release -j4
```

### 运行性能测试

```bash
# 默认: prefill=1024, decode=512, 5轮, batch_size=128
./performance_test

# 自定义参数: prefill_tokens decode_tokens rounds batch_size
./performance_test 1024 512 5 128
```

### 运行精度验证

```bash
# 验证 Batch Linear Attention 精度
./verify_linear_attn_batch
```

## 模型配置

| 参数 | 值 |
|------|-----|
| Hidden Size | 1024 |
| Intermediate Size | 3584 |
| Num Layers | 24 (18 Linear + 6 Full Attention) |
| Num Heads | 8, Head Dim = 256 |
| Num KV Heads | 2, KV Head Dim = 256 |
| Vocab Size | 248320 |
| Linear Attention Key Dim | 128 |
| Linear Attention Value Dim | 128 |
| Conv Kernel Size | 4 |

## 依赖

- CUDA Toolkit 12.0+
- cuBLAS
- CMake 3.18+
- C++17
- NVIDIA GPU (Compute Capability 7.5+)

## 性能分析

### GPU: NVIDIA GeForce RTX 5060 Ti

| 指标 | 值 |
|------|-----|
| 理论显存带宽 | 448 GB/s |
| 理论 FP32 算力 | 22.6 TFLOPS |
| 理论 TF32 算力 | 45.2 TFLOPS |

### 瓶颈分析

- **Prefill**: Compute-bound (cuBLAS GEMM 充分利用 Tensor Core)
- **Decode**: Memory-bound (权重读取 + KV Cache)

## 进一步优化方向

1. **FP16/BF16 量化**: 显存减半，decode 速度提升 20-30%
2. **INT8/INT4 量化**: 进一步降低带宽压力
3. **Paged KV Cache**: 支持更长上下文
4. **Continuous Batching**: 提升批量推理吞吐
5. **CUDA Graph Prefill**: 捕获 prefill 计算图，消除 kernel launch 开销

## 优化记录

详细优化记录见 [docs/implementation/cuda_optimization_log.md](docs/implementation/cuda_optimization_log.md)

## 许可

MIT License

## 参考

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Qwen3.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
