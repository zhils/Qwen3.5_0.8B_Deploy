# Qwen3.5-0.8B C++/CUDA 推理引擎 - 文档中心

本文档中心提供项目的所有技术文档，按类别组织，方便快速定位所需信息。

## 📖 快速导航

### 新手入门
- [项目架构概览](implementation/qwen3_5_0_8b_architecture.md) - 了解整体设计
- [性能报告](implementation/performance_report.md) - 查看基准测试结果
- [快速开始指南](../README.md) - 项目主 README

### 模型架构详解
- [00_总览与索引](qwen3_5_0_8b_details/00_总览与索引.md) - 模型架构总览
- [01_总体架构_实现级流程](qwen3_5_0_8b_details/01_总体架构_实现级流程.md) - 端到端执行流程
- [02_文本主干_24层调度与参数映射](qwen3_5_0_8b_details/02_文本主干_24层调度与参数映射.md) - 24层调度逻辑
- [03_LinearAttention_GatedDeltaNet_细节](qwen3_5_0_8b_details/03_LinearAttention_GatedDeltaNet_细节.md) - 线性注意力实现
- [04_FullAttention_GQA_RoPE_细节](qwen3_5_0_8b_details/04_FullAttention_GQA_RoPE_细节.md) - 全注意力实现
- [05_MLP_层归一化_输出头](qwen3_5_0_8b_details/05_MLP_层归一化_输出头.md) - MLP 和归一化
- [06_视觉编码器与Merger_细节](qwen3_5_0_8b_details/06_视觉编码器与Merger_细节.md) - 视觉处理模块
- [07_MTP分支与推测解码_参数加载](qwen3_5_0_8b_details/07_MTP分支与推测解码_参数加载.md) - MTP 分支

### 性能优化与基准测试
- [性能对比分析](implementation/performance_comparison.md) - CPU vs GPU vs 其他框架
- [最新基准测试摘要](implementation/latest_benchmark_summary.md) - 最新性能数据
- [性能报告](implementation/performance_report.md) - 详细性能分析
- [Phase3 基线测试](implementation/phase3_baseline.md) - 第三阶段基线
- [Phase3 优化报告](implementation/phase3_optimization_report.md) - 第三阶段优化结果
- [内存优化报告](implementation/memory_optimization_report.md) - 内存使用优化
- [CUDA 加速方案复盘](implementation/CUDA加速方案项目报告与复盘.md) - CUDA 优化总结

### 微基准测试
- [KV INT8 基准测试](implementation/kv_int8_benchmark.md) - INT8 量化 KV Cache
- [Batch POC 基准测试](implementation/batch_poc_benchmark.md) - 批处理优化
- [Paged KV 基准测试](implementation/paged_kv_benchmark.md) - Paged KV Cache
- [BF16 GEMM 基准测试](implementation/bf16_prefill_benchmark.md) - BF16 矩阵乘法
- [基准测试自动化](implementation/benchmark_automation_principle_and_implementation.md) - 自动化测试框架

### 开发与调试
- [代码规范](implementation/code_style_guide.md) - 编码规范和最佳实践
- [性能分析指南](implementation/profiling_guide.md) - 如何分析性能
- [精度验证方法](implementation/accuracy_validation_methodology.md) - 精度验证流程
- [项目结构规范](implementation/项目结构体设计规范.md) - 代码组织规范
- [项目结构规划](implementation/PROJECT_STRUCTURE_PLAN.md) - 项目规划文档

## 📁 文档目录结构

```
docs/
├── README.md                          # 本文档（统一入口）
├── implementation/                    # 实现相关文档
│   ├── 性能报告/
│   │   ├── performance_report.md
│   │   ├── latest_benchmark_summary.md
│   │   ├── phase3_baseline.md
│   │   ├── phase3_optimization_report.md
│   │   └── memory_optimization_report.md
│   ├── 基准测试/
│   │   ├── kv_int8_benchmark.md
│   │   ├── batch_poc_benchmark.md
│   │   ├── paged_kv_benchmark.md
│   │   ├── bf16_prefill_benchmark.md
│   │   └── benchmark_automation_principle_and_implementation.md
│   ├── 架构设计/
│   │   ├── qwen3_5_0_8b_architecture.md
│   │   ├── PROJECT_STRUCTURE_PLAN.md
│   │   └── 项目结构体设计规范.md
│   └── 开发指南/
│       ├── profiling_guide.md
│       ├── accuracy_validation_methodology.md
│       └── CUDA加速方案项目报告与复盘.md
└── qwen3_5_0_8b_details/              # 模型架构详解
    ├── 00_总览与索引.md
    ├── 01_总体架构_实现级流程.md
    ├── 02_文本主干_24层调度与参数映射.md
    ├── 03_LinearAttention_GatedDeltaNet_细节.md
    ├── 04_FullAttention_GQA_RoPE_细节.md
    ├── 05_MLP_层归一化_输出头.md
    ├── 06_视觉编码器与Merger_细节.md
    └── 07_MTP分支与推测解码_参数加载.md
```

## 🎯 推荐阅读路径

### 路径 1：了解项目全貌（适合新贡献者）
1. [项目架构概览](implementation/qwen3_5_0_8b_architecture.md)
2. [性能报告](implementation/performance_report.md)
3. [模型架构总览](qwen3_5_0_8b_details/00_总览与索引.md)

### 路径 2：深入技术细节（适合开发者）
1. [总体架构实现流程](qwen3_5_0_8b_details/01_总体架构_实现级流程.md)
2. [LinearAttention 细节](qwen3_5_0_8b_details/03_LinearAttention_GatedDeltaNet_细节.md)
3. [FullAttention 细节](qwen3_5_0_8b_details/04_FullAttention_GQA_RoPE_细节.md)
4. [CUDA 加速方案复盘](implementation/CUDA加速方案项目报告与复盘.md)

### 路径 3：性能优化（适合优化工程师）
1. [最新基准测试摘要](implementation/latest_benchmark_summary.md)
2. [Phase3 优化报告](implementation/phase3_optimization_report.md)
3. [内存优化报告](implementation/memory_optimization_report.md)
4. [性能分析指南](implementation/profiling_guide.md)

## 📊 关键性能指标

| 指标 | 数值 | 测试环境 |
|------|------|----------|
| CPU 推理速度 | 详见 [性能报告](implementation/performance_report.md) | 48GB 内存 |
| CUDA 加速比 | 详见 [CUDA 复盘报告](implementation/CUDA加速方案项目报告与复盘.md) | NVIDIA GPU |
| 内存占用 | 详见 [内存优化报告](implementation/memory_optimization_report.md) | - |
| 精度误差 | < 0.04 (Max Diff) | 与 PyTorch 对比 |

## 🔗 相关链接

- [项目主 README](../README.md)
- [英文 README](../README_EN.md)
- [CI/CD 工作流](../.github/workflows/ci.yml)
- [CMakeLists.txt](../CMakeLists.txt)
- [第三方库说明](../third_party/README.md)

## 📝 文档维护

本文档由项目维护者定期更新。如有文档缺失或需要更新，请提交 Issue 或 Pull Request。

最后更新：2026-04-20
