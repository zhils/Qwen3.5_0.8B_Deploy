# Batch 推理优化 Benchmark 报告

> **状态**: 已归档。当前项目已采用更高效的 Batch Linear Attention + cuBLAS GEMM 方案，本报告中的 POC 数据仅供参考。

## 数据来源与日期

- 数据来源：`./build/Release/batch_opt_benchmark.exe 100 1 2 4`
- 采集日期：2026-04-18
- 样本数：batch sizes=1/2/4，steps=100，repeats=2

## 测试环境

- **GPU**: NVIDIA GeForce RTX 5060 Ti
- **VRAM**: 16310 MB
- **配置**:
  - num_layers: 24 (linear attention layers)
  - hidden_size: 1024
  - intermediate_size: 3584
  - num_steps: 100

## 优化策略

### 优化前（Naive）瓶颈

1. **散读写**: 每个 batch item 独立指针，循环 launch kernel，内存不连续
2. **kernel launch 开销**: batch=2 时 launch 次数翻倍（24 layers × 8 kernels × 2 = 384 次/step）
3. **不必要的同步**: 每次 kernel 后 `cudaDeviceSynchronize()`，阻塞流水线

### 三项优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| **连续内存布局** | [batch, hidden] 连续分配，kernel 用 `blockIdx.y` 索引 batch | 消除散读写，提升访存合并率 |
| **减少 kernel launch** | batch 维度合并到 grid.y，1 次 launch 处理全部 batch | launch 次数从 B×N 降至 N |
| **消除多余同步** | 仅用 `cudaEventSynchronize` 替代 `cudaDeviceSynchronize` | 减少 host 阻塞，允许 GPU 流水线 |

## 优化前后对比（同口径）

### 延迟对比

| batch_size | Naive avg(ms) | Optimized avg(ms) | opt/naive | Sequential avg(ms) | opt/seq |
|------------|-------------|-----------------|-----------|-------------------|---------|
| 1 | 22.017 | 22.174 | 0.99x | 21.971 | 0.99x |
| **2** | **44.449** | **21.941** | **2.03x** | **44.030** | **2.01x** |
| **4** | **89.623** | **26.865** | **3.34x** | **87.920** | **3.27x** |

### 吞吐对比

| batch_size | Naive (tok/s) | Optimized (tok/s) | Sequential (tok/s) | opt/seq |
|------------|-------------|-----------------|-------------------|---------|
| 1 | 45.4 | 45.1 | 45.5 | 0.99x |
| **2** | **45.0** | **91.2** | **45.4** | **2.01x** |
| **4** | **44.6** | **148.9** | **45.5** | **3.27x** |

### P95 尾延迟对比

| batch_size | Naive p95(ms) | Optimized p95(ms) | Sequential p95(ms) |
|------------|-------------|-----------------|-------------------|
| 1 | 22.384 | 22.975 | 22.393 |
| 2 | 44.824 | 22.495 | 45.288 |
| 4 | 91.187 | 27.603 | 88.660 |

### 优化效果分解

| batch_size | Naive→Opt (连续布局+合并launch) | Opt→Opt+NoSync (消除同步) | 总提升 |
|------------|------|------|------|
| 2 | 2.03x | ~1.0x (event sync 已足够) | **2.01x vs sequential** |
| 4 | 3.34x | ~1.0x | **3.27x vs sequential** |

## 分析

### 核心发现

1. **batch=2 已超过 sequential baseline 2.01x**: 这是本次优化的目标达成标志
2. **连续内存布局 + 合并 launch 是最大贡献者**: 从 naive 的 44.4ms 降至 21.9ms，几乎等于 batch=1 的延迟
3. **batch=1 时优化无额外收益**: 因为 batch=1 本身就是连续的，grid.y=1 与单次 launch 等价
4. **batch=4 延迟仅增加 22%**: 从 21.9ms 到 26.9ms，吞吐提升 3.27x，GPU 利用率显著提高
5. **P95 尾延迟稳定**: batch=2 的 p95 仅 22.5ms，与 avg 差距 < 3%

### 为什么 batch=2 能超过 sequential

- **Sequential**: 2 次 batch=1 推理，每次 22ms，总计 44ms
- **Optimized batch=2**: 1 次 batch=2 推理，延迟 21.9ms（几乎等于 batch=1）
- **原因**: GPU SM 有大量空闲算力，batch=1 仅用了小部分 CUDA cores；batch=2 通过 grid.y 并行，将空闲算力填满，延迟几乎不增

### batch=4 的非线性扩展

- batch=4 延迟 26.9ms，比 batch=2 的 21.9ms 增加 23%
- 吞吐 148.9 tok/s，接近线性的 4x
- 说明 RTX 5060 Ti 在此 workload 下 batch=4 仍有算力余量

## 结论

通过三项优化（连续内存布局、合并 kernel launch、消除多余同步），batch decode 推理已实现：

1. **batch=2 超过 sequential 2.01x** ✅ 目标达成
2. **batch=4 超过 sequential 3.27x**，扩展性良好
3. **P95 尾延迟稳定**，无异常抖动
4. **核心优化是连续内存布局 + 合并 launch**，消除同步的贡献在此 workload 下较小

---

## 后续演进

当前项目已采用更激进的优化方案：
- **Batch Linear Attention**: 使用 cuBLAS GEMM 一次性处理所有 projection
- **Kernel Fusion**: 将多个小 kernel 合并为单个 fused kernel
- **预分配 Buffer**: 避免重复的 cudaMalloc/Free

详见 [cuda_optimization_log.md](cuda_optimization_log.md)
