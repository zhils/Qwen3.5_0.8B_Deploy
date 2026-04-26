# BF16 Prefill Benchmark 报告

> **状态**: 已归档。BF16 优化收益有限（整模仅 0.86%），当前项目以 FP32 + cuBLAS GEMM 为主。

## 数据来源与日期

- 数据来源：`./build/Release/bf16_gemm_benchmark.exe`
- 采集日期：2026-04-20

---

## 测试目的

验证 BF16 在 prefill 场景的收益。

---

## 测试配置

- GPU：NVIDIA GeForce RTX 5060 Ti
- Compute Capability：12.0
- BF16 Tensor Core：支持
- 模块：单层 FullAttention + RMSNorm + MLP
- prefill_len：128 tokens

---

## 结果

### 1) 单层端到端 prefill

| 指标 | FP32 | BF16 | 变化 |
|---|---:|---:|---:|
| Prefill total avg (ms) | 242.294 | 230.669 | **1.0504x** |
| Prefill per-token avg (ms/token) | 1.8929 | 1.8021 | **-4.80%** |

### 2) 整模 prefill（24 层，len=32）

| 指标 | FP32 | BF16 | 变化 |
|---|---:|---:|---:|
| Prefill total avg (ms) | 639.839 | 634.372 | **1.0086x** |
| Prefill per-token avg (ms/token) | 19.9950 | 19.8241 | **-0.85%** |

### 3) 转换开销占比

| 指标 | 数值 |
|---|---:|
| Conversion-only prefill total avg (ms) | 26.416 |
| BF16 full-model prefill total avg (ms) | 634.372 |
| Conversion / BF16 prefill | **4.16%** |

---

## 结论

- 单层 prefill：BF16 收益约 4.8%
- 整模 prefill：BF16 收益仅 0.86%，转换开销占 4.16%
- **当前项目未采用 BF16**，以 FP32 + cuBLAS TF32 Tensor Core 为主

---

## 后续演进

当前项目通过 Batch Linear Attention + cuBLAS GEMM 实现 prefill 吞吐 **525.6 tok/s**，远超 BF16 优化收益。

详见 [cuda_optimization_log.md](cuda_optimization_log.md)
