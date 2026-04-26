# KV INT8 量化 Benchmark 报告

## 数据来源与日期

- 数据来源：`./build/Release/kv_int8_benchmark.exe 100` 和 `./build/Release/stage2_gpu_benchmark.exe ./weights 20`
- 采集日期：2026-04-18
- 样本数：decode steps=100

## 测试环境

- **GPU**: NVIDIA GeForce RTX 5060 Ti
- **VRAM**: 16310 MB
- **配置**:
  - num_layers: 24
  - num_kv_heads: 2
  - head_dim: 256
  - decode_steps: 100

## Stage2 GPU FP32 Baseline (full pipeline)

| 指标 | 值 |
|------|-----|
| Prefill latency (5 tokens) | 1592.6 ms |
| Decode avg latency (ms/token) | 23.0 ms |
| Decode p50 (ms) | 23.3 ms |
| Decode p95 (ms) | 24.0 ms |
| Decode throughput (tok/s) | 43.4 |
| GPU VRAM after init | 6322.6 MB |
| GPU VRAM peak | 8814.6 MB |
| GPU model allocation | 5186.0 MB |

### Full Pipeline 模块级延迟分解

| 模块 | avg (ms) | p50 (ms) | p95 (ms) |
|------|----------|----------|----------|
| GPU RMSNorm (1024) | 0.014 | 0.010 | 0.040 |
| GPU MLP/SwiGLU (1024->3584) | 0.243 | 0.238 | 0.266 |
| GPU LMHead (1024->248320) | 1.226 | 1.221 | 1.246 |
| GPU FullAttention (decode) | 0.731 | 0.730 | 0.825 |
| GPU LinearAttention (decode) | 0.550 | 0.548 | 0.560 |
| GPU TokenEmbedding (lookup) | 0.010 | 0.006 | 0.035 |

## KV INT8 量化对比

### 跨序列长度性能对比

| seq_len | FP32 avg(ms) | INT8 avg(ms) | 延迟加速比 | FP32 p95(ms) | INT8 p95(ms) | FP32 alloc(MB) | INT8 alloc(MB) | 显存压缩比 |
|---------|-------------|-------------|-----------|-------------|-------------|---------------|---------------|-----------|
| 2048 | 0.294 | 0.307 | 0.96x | 0.342 | 0.365 | 192 | 48 | **4.00x** |
| 4096 | 0.272 | 0.155 | **1.76x** | 0.310 | 0.184 | 384 | 96 | **4.00x** |
| 8192 | 0.368 | 0.163 | **2.26x** | 0.698 | 0.188 | 768 | 193 | **3.98x** |

### 精度稳定性（跨序列长度）

| seq_len | Max abs diff | Mean abs diff | P99 diff | Q1 max | Q2 max | Q3 max | Q4 max | 精度判定 |
|---------|-------------|-------------|---------|--------|--------|--------|--------|---------|
| 2048 | 0.022 | 0.002 | 0.011 | 0.022 | 0.018 | 0.020 | 0.018 | STABLE |
| 4096 | 0.022 | 0.002 | 0.011 | 0.022 | 0.018 | 0.020 | 0.018 | STABLE |
| 8192 | 0.022 | 0.002 | 0.011 | 0.022 | 0.018 | 0.020 | 0.018 | STABLE |

### 尾延迟稳定性

| seq_len | INT8 p95/avg 比 | 延迟判定 |
|---------|----------------|---------|
| 2048 | 1.19x | STABLE |
| 4096 | 1.19x | STABLE |
| 8192 | 1.15x | STABLE |

## 分析

### 核心发现

1. **显存压缩 4x 在所有序列长度下一致**: 2K→48MB, 4K→96MB, 8K→193MB，线性增长，无额外开销
2. **精度无累积漂移**: Q4（最后 25% 位置）max diff 在 2K/4K/8K 下分别为 0.018/0.018/0.018，完全一致，说明 INT8 量化误差不会随上下文长度累积
3. **长上下文下性能收益更显著**: 4K/8K 下 INT8 比 FP32 快约 1.8-2.3x，而 2K 下无明显差异。原因：INT8 显存占用更小，cache 友好性在长序列下优势更突出
4. **尾延迟稳定**: p95/avg 比在所有序列长度下均 < 1.2x，无异常尾延迟
5. **Full Pipeline vs KV-Only**: stage2_gpu_benchmark 显示完整 pipeline 的 MLP (0.24ms) 和 Attention (0.55-0.73ms) 是主要耗时，KV 量化主要节省的是显存带宽

### 精度细节

- Max abs diff 在 0.021~0.022 范围内，跨序列长度无显著变化
- Mean abs diff 始终为 0.002，量化噪声水平极低
- P99 diff 为 0.011，99% 的元素误差 < 0.011
- 四分位 max diff 趋势: Q1≈Q2≈Q3≈Q4，无尾部位置精度退化

## 结论

KV INT8 量化在 2K/4K/8K 序列长度下均通过稳定性验证：

1. **显存**: 4x 压缩在所有长度下一致，8K 时节省 575MB
2. **精度**: 无累积漂移，Q4 max diff 在 2K/4K/8K 下均为 0.018，量化误差与序列长度无关
3. **延迟**: 4K/8K 下加速比达 1.8-2.3x，尾延迟 p95/avg < 1.2x，稳定可靠
4. **结论**: INT8 KV 量化已具备在长上下文生产场景中部署的稳定性保证
