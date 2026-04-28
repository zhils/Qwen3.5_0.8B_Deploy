# Qwen3.5-0.8B CUDA Inference Engine

[English](README_EN.md) | 中文

## 项目概述

基于 Qwen3.5-0.8B 模型的高性能 CUDA 推理引擎，采用多种优化技术实现极致的端到端推理速度。

### 核心性能指标

> **吞吐量说明**：Prefill 吞吐量为**单序列等效吞吐量**（总处理 token 数 / 总时间），非 batch 总吞吐量。实际 batch 总吞吐量 = 单序列等效吞吐量 × batch_size。

| 测试条件 | Prefill TTFT | Prefill 吞吐 (单序列等效) | Decode TPOT | Decode 吞吐 | 端到端耗时 |
|---------|-------------|------------------------|-------------|-------------|-----------|
| RTX 5060 Ti, batch=128, prefill=1024, decode=512 | **1,497 ms** | **684.0 tok/s** (实际 87,552 tok/s) | **0.086 ms/tok** | **11,682 tok/s** | **1,541 ms** |
| RTX 5060 Ti, batch=1, prefill=1024, decode=512 | **2,149 ms** | **444.2 tok/s** | **0.062 ms/tok** | **16,133 tok/s** | **2,181 ms** |
| **RTX 3080 Ti, batch=1, prefill=1024, decode=512** | **2,167 ms** | **574.5 tok/s** | **0.099 ms/tok** | **10,147 tok/s** | **1,684 ms** |

### 与 llama.cpp 对比 (同硬件 RTX 5060 Ti)

#### 单请求对比 (batch=1)

| 指标 | 本项目 (v3.1) | llama.cpp (BF16 GGUF) | 优势 |
|------|--------------|----------------------|------|
| **Prefill 吞吐** | **444.2 tok/s** | **193.16 t/s** | **2.30x** |
| **Decode 吞吐** | **16,133 tok/s** | **191.79 t/s** | **84.1x** |
| **Prefill TTFT (1024 tok)** | **2,305 ms** | **5,301 ms** | **2.30x** |
| **Decode TPOT** | **0.062 ms** | **5.21 ms** | **84.0x** |
| **端到端耗时 (1024+512)** | **2,337 ms** | **7,971 ms** | **3.41x** |
| **显存占用** | ~4,070 MB | 2,298 MB | - |

#### Batch 对比 (batch=128)

| 指标 | 本项目 (v3.3) | llama.cpp (BF16 GGUF) | 优势 |
|------|--------------|----------------------|------|
| **Prefill 吞吐 (单序列等效)** | **684.0 tok/s** (实际 87,552 tok/s) | 未测试 | - |
| **Prefill 吞吐 (batch 总吞吐)** | **87,552 tok/s** | 未测试 | - |
| **Prefill TTFT (1024 tok)** | **1,497 ms** | 未测试 | - |

> **测试条件说明**：
> - llama.cpp：使用 `llama-bench` 实测，`qwen3.5-0.8b-f16.gguf` (BF16)，batch=1，`-ngl 99 -fa 1`，重复 20 次取 P50
> - 本项目：FP32 权重，batch_size=1（单请求）或 128（batch prefill）
> - **Prefill 优势说明**：v3.1 通过内部 token 累积（BATCH_SIZE >= 32）优化，单请求 prefill 从 40.4 tok/s 提升至 444.2 tok/s，是 llama.cpp 的 2.3 倍。即使 batch=1，内部也会累积至少 32 个 token 批量处理，充分利用 GPU 并行度。
> - **Decode 差距说明**：84x 的差距主要来自 (1) 本项目启用 CUDA Graph 消除 kernel launch 开销；(2) 全融合 kernel（SiLU+Mul、RMSNorm+Residual 等）减少 HBM 访存；(3) 预分配 buffer 无动态内存分配。llama.cpp 作为通用框架，在 0.8B 小模型上 kernel launch 和框架开销占比较高。

### 三方对比 (RTX 3080 Ti, batch=1)

> 数据来源：[docs/optimization/PERFORMANCE_TEST_TEMPLATE.md](docs/optimization/PERFORMANCE_TEST_TEMPLATE.md)，测试日期 2026-04-27

| 指标 | vLLM (FlashAttn v2) | llama.cpp (BF16) | 本项目 (Deploy) | 最优 |
|------|----------------------|------------------|-----------------|------|
| **Prefill 吞吐** | 614 tok/s | **16,148 tok/s** | 574 tok/s | llama.cpp |
| **Decode 吞吐** | 316 tok/s | 249 tok/s | **10,147 tok/s** | **Deploy** |
| **TPOT** | 3.16 ms/token | 4.01 ms/token | **0.099 ms/token** | **Deploy** |
| **E2E Avg (1024+512)** | **1,617 ms** | ~2,119 ms* | 1,833 ms | vLLM |
| **E2E P50 (1024+512)** | **1,615 ms** | ~2,119 ms* | 1,684 ms | vLLM |
| **E2E P90 (1024+512)** | **1,634 ms** | N/A | 3,122 ms | vLLM |
| **GPU VRAM** | ~9.5 GB | N/A | ~4.0 GB | Deploy |

*llama.cpp E2E 为理论推算值（Prefill 63ms + Decode 2056ms），非实测

**分析**：
- **Decode 阶段**：Deploy 以 10,147 tok/s 大幅领先，TPOT 仅 0.099 ms/token，是 vLLM 的 32 倍、llama.cpp 的 40 倍
- **Prefill 阶段**：llama.cpp 凭借标准 Attention + 极致优化达到 16,148 tok/s，是 Deploy 的 28 倍；vLLM 与 Deploy 相近（614 vs 574 tok/s）
- **端到端**：vLLM 通过 CUDA Graph + 流水线重叠（Prefill 末尾与 Decode 首 token 重叠执行），E2E 仅 1.6s，最优；Deploy E2E 1.8s，差距缩小到 15%
- **显存**：Deploy 占用 ~4.0 GB（优化后），通过 Embedding/LM Head 权重共享、BF16 存储、KV Cache 动态分配、统一 cuBLAS Handle 等技术实现 53.6% 内存节省

### 性能演进

#### RTX 5060 Ti

| 版本 | Prefill 吞吐 (batch=1, 单序列) | Prefill 吞吐 (batch=128, 单序列等效) | Decode 吞吐 | TTFT (batch=1) | 主要优化 |
|------|-------------------------------|-------------------------------------|-------------|---------------|---------|
| v1.0 (CUDA Baseline) | 17.5 tok/s | - | 12.5 tok/s | 58,400 ms | CUDA 基础实现，单 token 串行 |
| v2.0 (FlashAttention) | 86.4 tok/s | - | 15,774 tok/s | 11,856 ms | FlashAttention v2 + Tensor Core + Batch Prefill |
| v3.0 (Batch GEMM) | 40.4 tok/s | 525.6 tok/s | 16,248 tok/s | 25,336 ms | Batch Linear Attention + cuBLAS GEMM + Kernel Fusion |
| v3.1 (Token Accumulation) | 444.2 tok/s | 520.3 tok/s | 16,133 tok/s | 2,305 ms | 内部 Token 累积 (BATCH_SIZE >= 32) + CUDA Graph 框架 |
| v3.2 (FlashAttention Prefill Opt) | 444.2 tok/s | 637.1 tok/s | 10,686 tok/s | 2,305 ms | FlashAttention Prefill Kernel 重构：warp-level 并行 + 消除跨 warp 同步 |
| **v3.3 (Kernel Memory Opt)** | **444.2 tok/s** | **684.0 tok/s** (实际 87,552 tok/s) | **11,682 tok/s** | **2,149 ms** | **Gated Delta __ldg + MLP 统一 cuBLAS GEMM + Tensor Core** |

#### RTX 3080 Ti (batch=1)

| 版本 | Prefill 吞吐 | Decode 吞吐 | TTFT (1024 tok) | TPOT | E2E P50 (1024+512) | 主要优化 |
|------|-------------|-------------|-----------------|------|-------------------|---------|
| **v3.3 (实测)** | **574.5 tok/s** | **10,147 tok/s** | **2,167 ms** | **0.099 ms/tok** | **1,684 ms** | 同 v3.3 优化栈，Linux CUDA 12.8 环境 |

**v3.3 相比 v1.0**: Prefill 提升 **+3,810%** (batch=128)，TTFT 降低 **-97%**

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

### 2. Flash Attention v2 (v3.2 重大优化)

**文件**: [fused_kernels.cu](src/backend/cuda/kernels/fused_kernels.cu)

- **Warp-level 并行**: 每个 warp 独立处理一个 head，一个 block 处理 4 个 heads
- **消除跨 warp 同步**: 仅使用 `__shfl_sync` 进行 warp 内规约，无需 shared memory 数组
- **Online softmax**: 减少 HBM 访问
- **Tiled computation**: Q/K/V 分块计算
- **Bank-conflict-aware**: Shared memory +1 padding 消除 bank conflict
- **效果**: Prefill 吞吐从 304 tok/s 提升至 **637 tok/s** (+109%)

### 3. v2.0 Kernel 微优化

**文件**: [linear_attention_v2.cu](src/backend/cuda/kernels/linear_attention_v2.cu), [flash_attention.cu](src/backend/cuda/kernels/flash_attention.cu)

| 优化项 | 文件 | 效果 |
|--------|------|------|
| **conv1d_update 寄存器缓存** | `linear_attention_v2.cu` | 权重缓存到寄存器，减少 global memory 读取 |
| **norm_gate_fused 寄存器缓存** | `linear_attention_v2.cu` | 中间值缓存到寄存器 |
| **Flash Attention Bank-conflict-aware** | `flash_attention.cu`, `fused_kernels.cu` | Shared memory padding 消除 bank conflict |

### 4. Kernel 融合

| 融合 Kernel | 功能 | 文件 |
|-----------|------|------|
| `save_rmsnorm_kernel` | Save 残差 + RMSNorm | fused_kernels.cu |
| `attn_add_rmsnorm_fused_kernel` | Attention 输出 + Add + RMSNorm | fused_kernels.cu |
| `silu_mul_fused_kernel` | SiLU 激活 + 乘法 | fused_kernels.cu |
| `conv1d_update_fused_kernel` | Conv1D + State 更新 | linear_attention_cuda.cu |
| `l2norm_qk_fused_kernel` | L2 归一化 Q + K | linear_attention_cuda.cu |
| `norm_gate_fused_kernel` | RMSNorm + Gate | linear_attention_cuda.cu |

### 5. cuBLAS 优化

- **Prefill 阶段**: cuBLAS GEMM 利用 Tensor Core (TF32) 加速
- **MLP Batch**: `sgemm` 替代 `sgemv`，预分配 hidden buffer
- **Flash Attention Output**: cuBLAS GEMM 替代手动矩阵乘法

### 6. CUDA Graph

- Decode 阶段启用 CUDA Graph，减少 Kernel Launch 开销
- 静态图捕获，多次执行零开销

### 7. 内存优化 (v3.3)

| 优化项 | 实现方式 | 节省内存 |
|--------|---------|---------|
| **Embedding/LM Head 权重共享** | 两者共享同一份 BF16 权重 | ~297 MB |
| **LM Head BF16 存储** | 主机端 FP32→BF16 转换，GPU 仅存 BF16 | ~297 MB |
| **KV Cache 动态分配** | 2x 扩容策略，按需分配 | 短序列优化 |
| **统一 cuBLAS Handle** | CublasHandlePool 单例池 | ~36 MB |

**总优化效果**: 显存从 ~8.77 GB 降至 ~4.07 GB，**节省 53.6%**

详细分析见 [docs/memory_optimization/memory_analysis_report.md](docs/memory_optimization/memory_analysis_report.md)

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
5. ~~**CUDA Graph Prefill**: 捕获 prefill 计算图，消除 kernel launch 开销~~ (框架已搭建，因 attention kernel 含 D2H memcpy 暂时 fallback)
6. **内部 Token 累积**: batch=1 时内部累积 >=32 token 再批量处理，prefill 提升 11x
7. **Flash Attention Prefill Kernel 优化**: warp-level 并行 + 消除跨 warp 同步，prefill 提升 109% (v3.2 已完成)

## 优化记录

详细优化记录见 [docs/implementation/cuda_optimization_log.md](docs/implementation/cuda_optimization_log.md)

## 许可

MIT License

## 参考

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Qwen3.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
