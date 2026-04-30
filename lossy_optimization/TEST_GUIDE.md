# Lossy Optimization 性能测试文档

## 测试环境

- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **测试框架**: CUDA
- **测试日期**: 2026-04-29

## 测试配置

| 参数 | 值 |
|------|-----|
| Prefill tokens | 1024 |
| Decode tokens | 512 |
| 测试轮次 | 5 |
| Batch sizes | 1, 8, 16, 32, 64, 128 |

## 测试指标

| 指标 | 说明 | 单位 |
|------|------|------|
| **Prefill throughput** | Prefill 阶段总吞吐 | tokens/sec |
| **Decode throughput** | Decode 阶段总吞吐 | tokens/sec |
| **E2E throughput** | 端到端总吞吐 | tokens/sec |
| **TTFT** | Time To First Token (首个token生成时间) | ms |
| **TPOT** | Time Per Output Token (每个输出token耗时) | ms/token |
| **Memory** | GPU 显存峰值 | MB |

## 测试对象

| 编号 | 优化方案 | 描述 |
|------|----------|------|
| 01_weight_bf16 | 权重 BF16 量化 | 权重 FP32 → BF16 |
| 02_kv_cache_fp16 | KV Cache FP16 | KV Cache FP32 → FP16 |
| 03_kv_cache_int8 | KV Cache INT8 | KV Cache FP32 → INT8 |
| 04_weight_int8 | 权重 INT8 量化 | 权重 FP32 → INT8 |
| 05_weight_int4 | 权重 INT4 量化 | 权重 FP32 → INT4 |
| 06_full_fp16 | FP16 全链路 | 权重 + KV Cache 均用 FP16 |
| 07_full_int8 | INT8 全链路 | 权重 + KV Cache 均用 INT8 |
| 08_flash_to_linear | Linear Attention | Flash → Linear Attention |
| 09_linear_to_flash | Flash Attention | Linear → Flash Attention |

## 测试脚本

### 自动测试脚本

```bash
cd /mnt/d/deploy/Qwen3.5_0.8B_Deploy/lossy_optimization
chmod +x run_all_tests.sh
./run_all_tests.sh
```

该脚本会自动遍历所有 9 个优化变体，测试所有 batch size 并生成汇总表格。

### 单独测试某个变体

```bash
cd /mnt/d/deploy/Qwen3.5_0.8B_Deploy/lossy_optimization/01_weight_bf16/build
./performance_test <prefill_tokens> <decode_tokens> <num_rounds> <batch_size>

# 示例
./performance_test 1024 512 5 8
```

### 测试特定 batch size

```bash
cd /mnt/d/deploy/Qwen3.5_0.8B_Deploy/lossy_optimization/01_weight_bf16/build

# 测试 batch=1
./performance_test 1024 512 5 1

# 测试 batch=8
./performance_test 1024 512 5 8

# 测试 batch=16
./performance_test 1024 512 5 16

# 测试 batch=32
./performance_test 1024 512 5 32

# 测试 batch=64
./performance_test 1024 512 5 64

# 测试 batch=128
./performance_test 1024 512 5 128
```

## 预期输出格式

```
=======================================================================
   Qwen3.5-0.8B v2.0 Performance Test
=======================================================================
 GPU: NVIDIA GeForce RTX 5060 Ti
 CC:  12.0
 VRAM: 16310 MB

=======================================================================
   v2.0 Performance Test (All Optimizations Enabled)
=======================================================================
   Prefill tokens:     1024
   Decode tokens:      512
   Batch size:         8
   Rounds:             5

[Loading random weights...]

[Running 5 round(s)...]

--- Prefill (1024 tokens, batch=8) ---
  Total time:      2494.067 ms
  TTFT:            2494.067 ms
  Single thrpt:     410.574 tokens/sec
  Batch thrpt:     3284.595 tokens/sec

--- Decode (512 tokens, batch=8) ---
  TPOT:               0.170 ms/token
  Single thrpt:    5887.594 tokens/sec
  Batch thrpt:    47100.749 tokens/sec

--- E2E (batch=8) ---
  Total time:      2581.029 ms
  E2E thrpt:       4760.892 tokens/sec

--- Memory ---
  GPU VRAM used:   5212.562 MB
  GPU VRAM total: 16310.562 MB
```

## 结果汇总表格式

测试完成后，结果汇总表格式如下：

| Variant | Batch | Prefill (tok/s) | Decode (tok/s) | E2E (tok/s) | TTFT (ms) | TPOT (ms/tok) | Memory (MB) |
|---------|-------|-----------------|----------------|--------------|------------|---------------|-------------|
| 01_weight_bf16 | 1 | - | - | - | - | - | - |
| 01_weight_bf16 | 8 | - | - | - | - | - | - |
| ... | ... | ... | ... | ... | ... | ... | ... |

## 内存消耗参考

| 优化方案 | 理论内存节省 | 预估 VRAM 使用 |
|----------|-------------|----------------|
| 原始 FP32 | 0% | ~5200 MB |
| 01_weight_bf16 | ~40% | ~3200 MB |
| 02_kv_cache_fp16 | ~10% | ~4700 MB |
| 03_kv_cache_int8 | ~15% | ~4400 MB |
| 04_weight_int8 | ~60% | ~2100 MB |
| 05_weight_int4 | ~70% | ~1600 MB |
| 06_full_fp16 | ~50% | ~2600 MB |
| 07_full_int8 | ~75% | ~1300 MB |

## 注意事项

1. **BF16** 需要 Ampere+ 架构 (RTX 30 系列及以上)
2. **INT8/INT4 量化** 需要对应的量化权重文件
3. **Linear Attention** 变体 (08, 09) 的性能特征与传统 Attention 不同
4. 测试使用随机权重，不加载真实模型权重
5. 每个 batch size 测试 5 轮取平均

## 故障排查

### 编译错误

如果遇到编译错误，确保 CUDA 环境正确：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
```

### 运行错误

如果 `performance_test` 无法运行：

```bash
# 检查 CUDA 设备
nvidia-smi

# 检查动态库链接
ldd ./performance_test
```

### 显存不足

如果遇到 OOM 错误：

1. 减小 batch size
2. 减少 max_seq_len
3. 使用量化版本 (INT8/INT4)
