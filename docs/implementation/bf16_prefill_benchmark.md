# BF16 Prefill Benchmark 报告

## 数据来源与日期

- 数据来源：`./build/Release/bf16_gemm_benchmark.exe D:/deploy/c++deploy/weights`
- 结果文件：`bf16_prefill_results.csv`
- 结果文件（整模）：`bf16_prefill_full_model_results.csv`
- 结果文件（转换开销）：`bf16_conversion_overhead_results.csv`
- 采集日期：2026-04-20

---

## 测试目的

此前 BF16 对比主要集中在 decode 场景。  
本次新增 prefill 压测（单层端到端）用于验证“BF16 更适合长序列预填充”的收益。

---

## 测试配置

- GPU：NVIDIA GeForce RTX 5060 Ti
- Compute Capability：12.0
- BF16 Tensor Core：支持
- 模块：单层 FullAttention + RMSNorm + MLP（layer_3 权重）
- prefill_len：128 tokens
- warmup：3
- iterations：30

---

## 结果

### 1) 单层端到端 prefill（FullAttn+RMSNorm+MLP，len=128）

| 指标 | FP32 | BF16 | 变化 |
|---|---:|---:|---:|
| Prefill total avg (ms) | 242.294 | 230.669 | **1.0504x** |
| Prefill total p50 (ms) | 242.667 | 233.018 | 1.041x |
| Prefill total p95 (ms) | 270.220 | 275.369 | 0.981x |
| Prefill per-token avg (ms/token) | 1.8929 | 1.8021 | **-4.80%** |

结论：

- 在本次单层 prefill 场景下，BF16 相比 FP32 有稳定正收益。
- 平均总时延降低约 11.6 ms（242.294 -> 230.669 ms）。
- 每 token 平均 prefill 时延降低约 4.80%。

### 2) 整模 prefill（24 层 CudaEngine，len=32）

| 指标 | FP32 | BF16 | 变化 |
|---|---:|---:|---:|
| Prefill total avg (ms) | 639.839 | 634.372 | **1.0086x** |
| Prefill total p50 (ms) | 639.771 | 634.400 | 1.0085x |
| Prefill total p95 (ms) | 639.999 | 634.493 | 1.0087x |
| Prefill per-token avg (ms/token) | 19.9950 | 19.8241 | **-0.85%** |

结论：

- 在整模 prefill（当前实现）下，BF16 已相对 FP32 小幅转正（1.0086x）。
- 这说明通过减少不必要转换后，BF16 系统级收益可以逐步显现，但当前提升幅度仍有限。
- 后续仍需继续优化 dtype 转换、Kernel 组织与整链路调度，进一步放大整模收益。

### 3) 转换开销占比估算（整模 prefill）

| 指标 | 数值 |
|---|---:|
| Conversion-only prefill total avg (ms) | 26.416 |
| BF16 full-model prefill total avg (ms) | 634.372 |
| Conversion / BF16 prefill | **4.1641%** |

解读：

- `FP32<->BF16` 转换本身有可见成本，但按当前测量约占整模 BF16 prefill 的 4.16%。
- 这意味着“整模仅小幅领先”仍受转换和链路结构共同影响，仍有继续优化空间。

---

## 说明与边界

- 本次是“单层端到端 prefill”验证，不是完整 24 层整模 prefill。
- decode 场景下 BF16 不一定总是优势，本次结果再次说明 BF16 更适合在 prefill 大 GEMM 密集阶段发挥。
- 本次已新增整模 prefill A/B（`bf16_prefill_full_model_results.csv`），得到“整模暂未受益”的结果。
- 下一步建议聚焦于：减少 FP32<->BF16 转换次数、扩大 BF16 覆盖路径、优化整模 kernel 编排后再复测。

