# 基准自动化与单一真相源（SSOT）实现

## 1) 原理

高性能项目里最常见的问题不是“没跑 benchmark”，而是：

- 口径不一致
- 命令不可复现
- 文档与原始结果不一致

因此需要自动化链路把“执行 -> 解析 -> 汇总 -> 发布”串成闭环，并固定指标定义。

## 2) 本项目实现

核心脚本：

- `scripts/benchmark/run_all_memory_benchmarks.ps1`

功能：

1. 自动检查并构建三个基准目标
   - `kv_int8_benchmark`
   - `batch_poc_benchmark`
   - `paged_kv_benchmark`
2. 统一执行命令并采集输出
3. 解析 csv/控制台结果
4. 自动生成：
   - `docs/latest_benchmark_summary.md`
   - `docs/latest_benchmark_summary.csv`

该 summary 已包含：

- 环境信息（GPU/驱动/CUDA/构建参数）
- 实验命令行
- 样本数
- 核心指标定义（avg/p50/p95/throughput）

## 3) 收益

- **一致性收益**：同一指标在多份文档中以 `latest_benchmark_summary.md` 为唯一口径。
- **效率收益**：一条脚本跑完三类内存优化实验，显著减少人工整理时间。
- **可追溯收益**：原始 csv 与 markdown 汇总同步产出，便于复查与面试举证。

## 4) 当前边界

- 目前自动化主要覆盖“内存优化专题”（KV INT8 / Batch POC / Paged KV）。
- 后续可扩展到 LMHead BF16、MLP BF16、阶段 3 融合微基准，形成统一性能看板。

