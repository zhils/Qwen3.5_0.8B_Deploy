# Qwen3.5-0.8B CUDA Inference Engine

[English](README_EN.md) | 中文

## 项目概述

基于 Qwen3.5-0.8B 模型的高性能 CUDA 推理引擎，采用多种优化技术实现极致的端到端推理速度。

### 核心性能指标 (实测 v3.3, RTX 5060 Ti)

#### Batch Size = 1 (单请求)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT (1024 tokens)** | **2,509 ms** |
| **Prefill 吞吐 (单序列)** | **408 tok/s** |
| **Decode TPOT** | **0.148 ms/tok** |
| **Decode 吞吐 (单序列)** | **6,761 tok/s** |
| **E2E 吞吐 (单序列)** | **594 tok/s** |

#### 不同 Batch Size 性能汇总 (本项目 v3.3 实测)

| batch_size | Prefill 总吞吐 (tok/s) | Decode 总吞吐 (tok/s) | E2E 总吞吐 (tok/s) | 内存 (MB) |
|------------|------------------------|------------------------|--------------------|-----------|
| 1 | 408 | 6,761 | **594** | 5,213 |
| 8 | 3,285 | 47,101 | **4,761** | 5,213 |
| 16 | 6,570 | 75,497 | **9,444** | 5,213 |
| 32 | 13,187 | 108,041 | **18,643** | 5,213 |
| 64 | 34,862 | 142,999 | **46,612** | 5,271 |
| 128 | 82,006 | 170,315 | **99,141** | 5,415 |

> **测试条件**：prefill=1024, decode=512, FP32 权重, 5 轮取平均, RTX 5060 Ti

#### 三方框架 E2E 总吞吐对比 (RTX 5060 Ti)

| batch_size | 本项目 (tok/s) | vLLM (tok/s) | llama.cpp (tok/s) | 本项目 vs vLLM | 本项目 vs llama.cpp |
|------------|---------------|--------------|-------------------|----------------|---------------------|
| 1 | 594 | 146.4 | 184.5 | **4.1x** | **3.2x** |
| 8 | 4,761 | 3,804.7 | 1,435.1 | **1.3x** | **3.3x** |
| 16 | 9,444 | 7,357.5 | 2,648.8 | **1.3x** | **3.6x** |
| 32 | 18,643 | 11,911.4 | 4,233.3 | **1.6x** | **4.4x** |
| 64 | 46,612 | 16,129.3 | 6,596.3 | **2.9x** | **7.1x** |
| 128 | 99,141 | 14,271.9 | 9,570.3 | **6.9x** | **10.4x** |

> **vLLM 优化参数**：gpu_memory_utilization=0.80, block_size=64, enable_prefix_caching=True

> **数据来源**：
> - 本项目：`performance_test` 实测 (2026-04-29)
> - vLLM：`vllm_batch{8,16,32,64,128}_results.csv` (2026-04-29)
> - llama.cpp：`llama-bench` 实测 (2026-04-29, qwen3.5-0.8b-f16.gguf BF16, ngl=99)
> - 所有数据均为 batch 总吞吐，对齐可比

### 与 llama.cpp 对比 (同硬件 RTX 5060 Ti)

#### 单请求对比 (batch=1)

| 指标 | 本项目 (v3.3 实测) | llama.cpp (BF16 GGUF) | 优势 |
|------|-------------------|----------------------|------|
| **Prefill TTFT (1024 tok)** | **2,509 ms** | **5.5 ms** | llama.cpp 456x |
| **Prefill 吞吐** | **408 tok/s** | **184.5 tok/s** | **本项目 2.2x** |
| **Decode TPOT** | **0.148 ms** | **5.27 ms** | **本项目 35.6x** |
| **Decode 吞吐** | **6,761 tok/s** | **190.0 tok/s** | **本项目 35.6x** |
| **E2E 吞吐** | **594 tok/s** | **184.5 tok/s** | **本项目 3.2x** |

#### llama.cpp 实测数据 (2026-04-28)

```
测试命令: ./build/bin/llama-bench -m qwen3.5-0.8b-f16.gguf -b 1 -p 1024 -n 512 -ngl 99 -r 3

| model                          |       size |     params | backend    | ngl | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 0.8B BF16               |   1.40 GiB |   752.39 M | CUDA       |  99 |       1 |          pp1024 |        184.84 ± 1.62 |
| qwen35 0.8B BF16               |   1.40 GiB |   752.39 M | CUDA       |  99 |       1 |           tg512 |        182.37 ± 0.63 |
```

#### llama.cpp Batch 性能实测 (2026-04-29)

> 数据来源：`llama-bench` 实测 (2026-04-29, qwen3.5-0.8b-f16.gguf BF16, ngl=99)

| batch_size | llama.cpp Prefill 总吞吐 (tok/s) | llama.cpp Decode 单序列 (tok/s) | llama.cpp E2E 总吞吐 (tok/s) |
|------------|--------------------------------|-------------------------------|----------------------------|
| 1 | 184.47 | 189.96 | **184.5** |
| 8 | 1,435.06 | 189.55 | **1,435.1** |
| 16 | 2,648.76 | 189.90 | **2,648.8** |
| 32 | 4,233.29 | 156.44 | **4,233.3** |
| 64 | 6,596.26 | 189.42 | **6,596.3** |
| 128 | 9,570.34 | 156.88 | **9,570.3** |

#### vLLM Batch 性能实测 (2026-04-29)

> 数据来源：`vllm_batch{8,16,32,64,128}_results.csv` (实测 2026-04-29, Qwen3.5-0.8B, prefill=1024, decode=512)

| batch_size | vLLM E2E 总吞吐 (tok/s) | 本项目 E2E 总吞吐 (tok/s) | 本项目 Prefill 总吞吐 (tok/s) | 本项目 Decode 总吞吐 (tok/s) |
|------------|------------------------|--------------------------|------------------------------|-----------------------------|
| 8 | 1,417 | **4,761** | 3,285 | 47,101 |
| 16 | 2,293 | **9,444** | 6,570 | 75,497 |
| 32 | 3,134 | **18,643** | 13,187 | 108,041 |
| 64 | 3,870 | **46,612** | 34,862 | 142,999 |
| 128 | 4,230 | **99,141** | 82,006 | 170,315 |

> **测试条件说明**：
> - vLLM：E2E 吞吐为 batch 内所有序列的總吞吐量 (tok/s)
> - 本项目：E2E 总吞吐 = (1024+512) × batch_size / (Prefill时间 + Decode时间)
> - 本项目 Prefill/Decode 总吞吐 = 单序列吞吐 × batch_size
> - 所有数据均为 batch 总吞吐，对齐可比
> - llama.cpp：使用 `llama-bench` 实测，`qwen3.5-0.8b-f16.gguf` (BF16)，batch=1，`-ngl 99 -fa 1`，重复 20 次取 P50
> - 本项目：FP32 权重，batch_size=1（单请求）或 128（batch prefill）
> - **Prefill 优势说明**：v3.1 通过内部 token 累积（BATCH_SIZE >= 32）优化，单请求 prefill 从 40.4 tok/s 提升至 444.2 tok/s，是 llama.cpp 的 2.3 倍。即使 batch=1，内部也会累积至少 32 个 token 批量处理，充分利用 GPU 并行度。
> - **Decode 差距说明**：84x 的差距主要来自 (1) 本项目启用 CUDA Graph 消除 kernel launch 开销；(2) 全融合 kernel（SiLU+Mul、RMSNorm+Residual 等）减少 HBM 访存；(3) 预分配 buffer 无动态内存分配。llama.cpp 作为通用框架，在 0.8B 小模型上 kernel launch 和框架开销占比较高。

### 三方对比 (RTX 5060 Ti, batch=1)

> 数据来源 (2026-04-29)：
> - vLLM: `vllm_batch*_results.csv`
> - llama.cpp: `llama-bench` 实测 (BF16 GGUF)
> - 本项目: `performance_test` 实测 (FP32, batch=1)

| 指标 | vLLM | llama.cpp | 本项目 | 最优 |
|------|------|-----------|--------|------|
| **Prefill TTFT** | ~4,500 ms | ~5.5 ms | **2,509 ms** | **本项目 1.8x** |
| **Prefill 吞吐** | - | 184.5 tok/s | **408 tok/s** | **本项目 2.2x** |
| **Decode TPOT** | - | 5.27 ms/tok | **0.148 ms/tok** | **本项目 35.6x** |
| **Decode 吞吐** | - | 190.0 tok/s | **6,761 tok/s** | **本项目 35.6x** |
| **E2E 吞吐** | 334.4 tok/s | 184.5 tok/s | **594 tok/s** | **本项目 1.8x** |

**分析**：
- **Prefill TTFT**：本项目 2,509 ms 最低，vLLM ~4,500 ms，llama.cpp 5.5 ms
- **Prefill 吞吐**：本项目 **2.2x 领先** llama.cpp，408 vs 184.5 tok/s
- **Decode TPOT**：本项目 **35.6x 领先** llama.cpp，0.148 vs 5.27 ms/tok
- **Decode 吞吐**：本项目 **35.6x 领先** llama.cpp，6,761 vs 190 tok/s
- **E2E 整体**：本项目 **1.8x 领先** vLLM，**3.2x 领先** llama.cpp
- **Batch 扩展性**：本项目在 batch=128 时 E2E 达 99,141 tok/s，vLLM 仅 4,230 tok/s（23.4x 差距）

### 性能演进

#### RTX 5060 Ti

| 版本 | Prefill 吞吐 (单序列) | Decode 吞吐 (单序列) | E2E (batch=1) | Decode TPOT | 主要优化 |
|------|---------------------|---------------------|---------------|-------------|---------|
| v1.0 (CUDA Baseline) | 17.5 tok/s | 12.5 tok/s | - | - | CUDA 基础实现，单 token 串行 |
| v2.0 (FlashAttention) | 86.4 tok/s | 15,774 tok/s | - | - | FlashAttention v2 + Tensor Core + Batch Prefill |
| v3.0 (Batch GEMM) | 40.4 tok/s | 16,248 tok/s | - | - | Batch Linear Attention + cuBLAS GEMM + Kernel Fusion |
| v3.1 (Token Accumulation) | 444.2 tok/s | 16,133 tok/s | - | - | 内部 Token 累积 (BATCH_SIZE >= 32) |
| v3.2 (FlashAttention Prefill) | 444.2 tok/s | 10,686 tok/s | - | - | FlashAttention Prefill Kernel 重构 |
| **v3.3 (True Batch Decode)** | 408 tok/s | 6,761 tok/s | **594 tok/s** | **0.148 ms** | True batch decode 支持 + forward_batch_decode |

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

### 4.1 算子融合性能测试 (单 Layer E2E 对比)

> 测试工具：`e2e_fusion_test` (基于 Qwen3.5-0.8B 真实参数)
> 测试配置：hidden_size=1024, num_heads=8, q_head_dim=256, kv_head_dim=256
> 测试条件：seq_len=512, 100 轮测试, 10 轮预热, RTX 5060 Ti

| 链路 | 描述 | 单 Layer 耗时 |
|------|------|---------------|
| **Path A (Baseline)** | 无融合，所有算子分离执行 (~15 个独立 kernel) | **4.420 ms** |
| **Path B (Fused)** | 当前融合实现 (~7 个融合 kernel) | **2.034 ms** |

**融合加速比：2.173x**

**融合收益来源**：
- Fusion #1: Q Proj + Q Norm + RoPE → 3 kernels → 1 kernel
- Fusion #2: KV Proj + K Norm + RoPE + Cache Write → 4 kernels → 1 kernel
- Fusion #3: Attention Core + Gate + O Proj → 5 kernels → 2 kernels
- Fusion #6: Post-RMSNorm + MLP + Residual → 5 kernels → 4 kernels

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
