# Prompt 测试协议

> 本文档定义阶段3性能测量的标准输入协议，确保所有优化前后对比基于相同口径。

---

## 1. 三档 Prompt 定义

| 档位 | Token 长度 | 用途 | 输入构造 |
|------|-----------|------|---------|
| **短** | 1~8 tokens | 纯 decode 性能测量 | 单个 token 或短句 |
| **中** | 256~512 tokens | 典型对话场景 | 重复填充至目标长度 |
| **长** | 1024~2048 tokens | 长上下文压力测试 | 重复填充至目标长度（按显存可调） |

### 短 Prompt 示例

```
Token IDs: [1024]
Length: 1 token
```

### 中 Prompt 示例

```
Token IDs: [1024, 2048, 3012, 4567, ...] (重复至 256 tokens)
Length: 256 tokens
```

### 长 Prompt 示例

```
Token IDs: [1024, 2048, 3012, 4567, ...] (重复至 1024 tokens)
Length: 1024 tokens
```

---

## 2. 随机种子与采样策略

| 参数 | 值 | 说明 |
|------|---|------|
| 随机种子 | 42 | 所有实验固定 |
| 生成长度 | 32 tokens | decode 阶段统一 |
| 采样策略 1 | **Greedy (argmax)** | 确定性，用于精度验证 |
| 采样策略 2 | Top-k (k=50) + Temperature (0.7) | 随机性，用于多样性验证 |

---

## 3. 统计口径

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| TTFT | Time To First Token | 从输入提交到第一个输出 token 的延迟 |
| Decode ms/token | 平均每 token 生成时间 | 总 decode 时间 / 生成的 token 数 |
| tok/s | 每秒生成 token 数 | 1000 / decode_ms_per_token |
| P50/P95/P99 | 延迟百分位 | 逐 token 延迟排序后取百分位 |
| Peak VRAM | 峰值显存占用 | `cudaMemGetInfo` 采样最小 free 值 |
| Peak Host WS | 峰值主机内存 | 进程 Working Set (Windows) |

---

## 4. 计时方式

- **GPU 计时**: `cudaEventRecord` / `cudaEventElapsedTime` (μs 精度)
- **Wall-clock 计时**: `std::chrono::high_resolution_clock` (ms 精度)
- **Warmup**: 前 5 次迭代不计入统计
- **Iterations**: 至少 50 次取平均

---

## 5. 可复现命令

```bash
# 短 prompt 基准
performance_benchmark.exe --seq_len 1 --warmup 5 --iterations 50

# 中 prompt 基准
performance_benchmark.exe --seq_len 256 --warmup 5 --iterations 50

# 长 prompt 基准
performance_benchmark.exe --seq_len 1024 --warmup 5 --iterations 50
```
