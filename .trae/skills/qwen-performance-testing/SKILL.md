---
name: "qwen-performance-testing"
description: "Qwen3.5-0.8B CUDA推理引擎性能测试方法。Invoke when user asks for performance testing, benchmarking, or performance optimization."
---

# Qwen3.5-0.8B CUDA推理引擎 - 性能测试方法

## 1. 核心性能指标

### 关键指标定义

| 指标 | 英文 | 说明 |
|------|------|------|
| TTFT | Time To First Token | 首个token生成时间 |
| TPOT | Time Per Output Token | 每个输出token耗时 |
| 吞吐量 | Throughput | 单位时间处理token数 |

### 性能目标

| 阶段 | 指标 | 目标值 |
|------|------|--------|
| **Prefill** | 吞吐量 | > 400 tok/s (batch=1) |
| **Decode** | 吞吐量 | > 10,000 tok/s |
| **Decode** | TPOT | < 0.1 ms/tok |
| **端到端** | E2E (1024+512) | < 2.5s |

## 2. 测试工具

### 2.1 性能测试可执行文件

```bash
# V1 性能测试
./performance_test

# V3 性能测试（推荐）
./performance_test_v3

# 自定义参数
./performance_test <prefill_tokens> <decode_tokens> <rounds> <batch_size>
./performance_test 1024 512 5 128  # 默认配置
```

### 2.2 内存分析

```bash
# GPU 内存分析
./memory_analysis
```

## 3. 测试配置

### 3.1 标准测试配置

| 配置 | prefill_tokens | decode_tokens | rounds | batch_size |
|------|----------------|---------------|--------|------------|
| **小请求** | 128 | 64 | 10 | 1 |
| **中等请求** | 512 | 256 | 5 | 8 |
| **大请求** | 1024 | 512 | 5 | 128 |
| **超大请求** | 2048 | 1024 | 3 | 64 |

### 3.2 硬件配置

| 硬件 | 说明 |
|------|------|
| RTX 5060 Ti | 主要测试硬件 |
| RTX 3080 Ti | 对比硬件 |
| CUDA 12.x | 推荐版本 |

## 4. 测试流程

### 4.1 快速性能测试

```bash
# 1. 编译
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
cmake --build . --config Release -j4

# 2. 运行标准测试
./performance_test_v3 1024 512 5 128

# 3. 检查输出
# - Prefill TTFT: < 2.2s (batch=1) 或 < 1.5s (batch=128)
# - Prefill 吞吐: > 400 tok/s
# - Decode TPOT: < 0.1 ms
# - Decode 吞吐: > 10,000 tok/s
```

### 4.2 批量性能测试

```bash
# 测试不同 batch size
for bs in 1 8 16 32 64 128; do
    echo "Testing batch_size=$bs"
    ./performance_test 1024 512 5 $bs
done
```

### 4.3 对比测试

```bash
# vs llama.cpp
# llama-bench -m qwen3.5-0.8b-f16.gguf -ngl 99 -fa 1

# vs vLLM
# python -m vllm.entrypoints.openai.api_server ...
```

## 5. 性能分析方法

### 5.1 Profiling 工具

```bash
# NVIDIA Nsight Systems
nsys profile --trace=cuda,nvtx ./performance_test_v3 1024 512 5 128

# NVIDIA Nsight Compute
ncu --set full ./performance_test_v3 1024 512 5 128
```

### 5.2 关键 Kernel 分析

| Kernel | 重要性 | 优化目标 |
|--------|--------|---------|
| `flash_attention_kernel` | **Critical** | Prefill 吞吐 |
| `linear_attention_kernel` | **Critical** | Decode 吞吐 |
| `sgemm` (cuBLAS) | **High** | MLP, QKV 投影 |
| `rmsnorm_kernel` | Medium | 延迟优化 |

### 5.3 性能瓶颈识别

```
Prefill 瓶颈:
├── Attention 计算 → Flash Attention 优化
├── GEMM 计算 → cuBLAS Tensor Core
└── Kernel Launch → CUDA Graph

Decode 瓶颈:
├── Attention 计算 → Linear Attention
├── Sampling → GPU Sampler
└── Memory Copy → Paged KV Cache
```

## 6. 性能回归检测

### 回归阈值

| 指标 | 回归阈值 |
|------|---------|
| Prefill 吞吐 | -5% |
| Decode 吞吐 | -5% |
| TPOT | +10% |
| 显存占用 | +10% |

### 检测方法

```bash
# 记录基准性能
echo "Baseline: Prefill=444.2, Decode=16133" > baseline.txt

# 测试新版本
./performance_test_v3 1024 512 5 1 > new_result.txt

# 对比
grep -E "Prefill|Decode|TPOT" new_result.txt
```

## 7. 输出格式

### 标准输出示例

```
========================================
Qwen3.5-0.8B CUDA Engine Performance Test
========================================
Config: prefill=1024, decode=512, batch=128, rounds=5

--- Prefill Stage ---
TTFT: 1497 ms
Throughput: 684.0 tok/s (actual: 87552 tok/s)

--- Decode Stage ---
TPOT: 0.086 ms/tok
Throughput: 11682 tok/s

--- End-to-End ---
Total Time: 1541 ms
========================================
```

## 8. 常见问题

### Q: Prefill 吞吐低？
- 检查 Flash Attention kernel 是否启用
- 验证 cuBLAS Tensor Core 是否激活
- 检查 batch size 是否足够大

### Q: Decode TPOT 高？
- 检查是否使用 Linear Attention
- 验证 CUDA Graph 是否启用
- 检查 KV Cache 是否有压缩

### Q: 显存占用高？
- 检查权重是否为 BF16
- 验证 KV Cache 分配策略
- 检查是否有内存泄漏
