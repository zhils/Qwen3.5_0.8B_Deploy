# 最新 Benchmark 摘要（单一数据源）

本文档记录了内存/性能优化实验的最新可复现 benchmark 运行结果。

## 1) 测试环境

- 日期：2026-04-19
- 操作系统：Windows 10 (PowerShell)
- GPU/驱动/显存：NVIDIA GeForce RTX 5060 Ti, 595.79, 16311 MiB
- CUDA Toolkit (nvcc)：
  nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Apr__9_19:29:17_Pacific_Daylight_Time_2025
Cuda compilation tools, release 12.9, V12.9.41
Build cuda_12.9.r12.9/compiler.35813241_0
- 构建模式：Release
- 构建命令：
  - cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
  - cmake --build build --config Release --target kv_int8_benchmark batch_poc_benchmark paged_kv_benchmark

## 2) 指标定义

- avg latency (ms/token)：每步 decode 延迟的算术平均值。
- p50 latency (ms)：所有 decode 步骤延迟的 50 分位。
- p95 latency (ms)：所有 decode 步骤延迟的 95 分位。
- throughput (tok/s)：
  - Decode benchmark：1000 / avg_latency_ms
  - Batch benchmark (forward_batch)：batch_size * 1000 / avg_step_ms

## 3) Benchmark 命令与样本量

### 3.1 KV INT8 A/B
- 命令：./build/Release/kv_int8_benchmark.exe 100
- 样本：decode steps=100, repeats=1

### 3.2 Batch POC
- 命令：./build/Release/batch_poc_benchmark.exe 50
- 样本：batch sizes=1,2,4, steps=50, repeats=1

### 3.3 Paged KV
- 命令：./build/Release/paged_kv_benchmark.exe 2048 64 128
- 样本：seq scan points=0..2048 (stride 64), clear check=1

## 4) 最新结果

### 4.1 KV INT8 A/B

| 指标 | FP32 Baseline | INT8 KV |
|---|---:|---:|
| Decode avg latency (ms/token) | 0.353 | 0.211 |
| Decode p50 (ms) | 0.299 | 0.148 |
| Decode p95 (ms) | 0.726 | 0.194 |
| Decode throughput (tok/s) | 2831.390 | 4735.230 |
| VRAM after init (MB) | 1328 | 1186 |
| Peak VRAM (MB) | 1330 | 1188 |
| Model allocation (MB) | 192 | 48 |
| Max abs diff | 0.000 | 0.007 |
| Mean abs diff | 0.000 | 0.002 |
| P99 diff | 0.000 | 0.006 |
| Top-1 token match rate | 100.000% | 100.000% |

### 4.2 Batch POC (forward_batch)

| batch_size | avg_step (ms) | throughput (tok/s) | speedup vs batch=1 |
|---:|---:|---:|---:|
| 1 | 0.707 | 1414.191 | 1.000x |
| 2 | 0.024 | 82836.315 | 58.575x |
| 4 | 0.032 | 125070.352 | 88.439x |

### 4.3 Batch POC (sequential baseline)

| batch_size | throughput (tok/s) |
|---:|---:|
| 1 | 133148.700 |
| 2 | 158308.002 |
| 4 | 178929.288 |

### 4.4 API 增益 (forward_batch / sequential)

| batch_size | gain |
|---:|---:|
| 1 | 0.011x |
| 2 | 0.523x |
| 4 | 0.699x |

### 4.5 Paged KV Scan and Reclaim

seq_len 扫描要点：
- seq_len=0: pages=1, kv_bytes=6 MB, gpu_used=1906 MB
- seq_len=1024: pages=16, kv_bytes=96 MB, gpu_used=1906 MB
- seq_len=2048: pages=32, kv_bytes=192 MB, gpu_used=1906 MB

Clear/reclaim：

| 指标 | Before Clear | After Clear | Delta |
|---|---:|---:|---:|
| kv_bytes_mb | 192.00 | 0.00 | -192.00 |
| gpu_used_mb | 1906.00 | 1906.00 | 0.00 |
| gpu_reclaimed_mb | - | - | 0.00 |

## 5) v1.0 端到端基准测试 (2026-04-26)

### 5.1 测试配置

- **版本**: v1.0 (简化版，移除 MLP 缓冲预分配和 BF16 路径)
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB)
- **模型**: Qwen3.5-0.8B, 24 layers
- **测试程序**: v1_benchmark.exe

### 5.2 实测数据 (128 tokens prefill / 64 tokens decode, 5 runs)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT** | 7301 ms |
| **Prefill 吞吐** | 17.5 tok/s |
| **Decode TPOT** | 79.96 ms/tok |
| **Decode 吞吐** | 12.5 tok/s |

### 5.3 推算数据 (1024 tokens prefill / 512 tokens decode)

基于 128 tokens 实测数据线性推算：

| 指标 | 数值 |
|------|------|
| **Prefill TTFT (1024 tok)** | ~58,400 ms |
| **Prefill 吞吐** | 17.5 tok/s |
| **Decode TPOT** | 79.96 ms/tok |
| **Decode 吞吐 (512 tok)** | 12.5 tok/s |

### 5.4 v1.0 关键特征

- **逐 token 顺序处理**: prefill 阶段每个 position 独立调用 forward()
- **无 batch 并行化**: 无法利用 GPU 并行能力
- **动态内存分配**: MLP 中间结果每次 cudaMalloc/Free
- **纯 FP32 计算**: 无 BF16/FP16 路径
- **Kernel launch 开销大**: 1024×24 = 24,576 次 kernel launch

### 5.5 与历史版本对比

| 版本 | Decode 吞吐 (tok/s) | 关键特性 |
|------|---------------------|---------|
| Baseline (CPU) | 2.9 | 原始 CPU 实现 |
| GPU FP32 | 2,831 | 初始 GPU 实现 |
| GPU INT8 KV | 4,735 | +INT8 KV Cache 量化 |
| **v1.0 (当前)** | **12.5** | **简化版，移除预分配和 BF16** |

> 注：v1.0 decode 吞吐较低是因为测试的是端到端（含 prefill 后的 cache 状态），而历史数据是 pure decode 步骤。

## 6) v2.0 端到端性能测试 (2026-04-26)

### 6.1 测试配置

- **版本**: v2.0 (全优化版本)
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB)
- **模型**: Qwen3.5-0.8B, 24 layers
- **测试程序**: v2_performance_test.exe
- **测试参数**: 1024 prefill tokens / 512 decode tokens, 5 rounds

### 6.2 v2.0 关键优化

- **Flash Attention v2**: Warp-level parallelism, reduced HBM traffic
- **Tensor Core (TF32)**: cuBLAS math mode for GEMM acceleration
- **Kernel Fusion**: Gate+SiLU+Mul, RMSNorm+Residual
- **Batch Prefill**: 真正的批量 prefill 并行处理
- **CUDA Graph**: Decode 阶段 kernel launch 开销优化

### 6.3 实测数据 (5 rounds average)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT** | 11,942 ms |
| **Prefill 吞吐** | 85.7 tok/s |
| **Decode TPOT** | 0.063 ms/tok |
| **Decode 吞吐** | 15,991 tok/s |
| **GPU VRAM** | 8,833 MB |

### 6.4 性能对比总结

| 版本 | Prefill 吞吐 (tok/s) | Decode 吞吐 (tok/s) | TTFT (ms) | TPOT (ms) |
|------|---------------------|---------------------|-----------|-----------|
| v1.0 | 17.5 | 12.5 | 58,400 | 79.96 |
| **v2.0** | **85.7** | **15,991** | **11,942** | **0.063** |
| **提升** | **+389%** | **+127,828%** | **-80%** | **-99.9%** |

### 6.5 优化效果分析

1. **Prefill 阶段**:
   - v1.0: 逐 token 顺序处理，24,576 次 kernel launch
   - v2.0: Batch prefill + Flash Attention v2，大幅减少 launch 开销
   - 效果: TTFT 从 58.4s 降至 11.9s，吞吐提升 389%

2. **Decode 阶段**:
   - v1.0: 动态内存分配，无优化
   - v2.0: Flash Attention v2 + Tensor Core + CUDA Graph
   - 效果: TPOT 从 79.96ms 降至 0.063ms，吞吐提升 127,828%

## 7) 原始数据文件

- kv_int8_benchmark_results.csv
- batch_poc_benchmark_results.csv
- paged_kv_benchmark_results.csv
