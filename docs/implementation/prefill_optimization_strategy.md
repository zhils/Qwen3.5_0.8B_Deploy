# v2.0 Prefill 进一步优化策略

## 当前状态 (v3.2)

| 指标 | 数值 |
|------|------|
| Prefill 吞吐 (batch=128) | 637 tok/s |
| Prefill TTFT (1024 tok) | 1607 ms |
| Decode 吞吐 (batch=1) | 16,133 tok/s |
| Decode TPOT | 0.062 ms |

## 目标

在可接受地降低 decode 吞吐的前提下，进一步提升 prefill 吞吐。

---

## 策略 1: Chunk-based Gated Delta Parallelization

### 原理

当前 Gated Delta 的瓶颈在于 recurrent state 的串行依赖：

```
for t in range(seq_len):      # 128 次串行
    for h in range(num_heads):  # 16 并行
        state[h] = f(state[h], q[t,h], k[t,h], v[t,h])
```

**Chunk-based 方案**: 将 tokens 分成 chunks（如 16 tokens/chunk），chunk 内并行计算：

```
for chunk in range(num_chunks):  # 8 chunks
    # Chunk 内并行: 16 tokens × 16 heads
    parallel_compute_chunk(chunk)
    # 串行更新 state
    update_recurrent_state(chunk)
```

### 实现要点

1. **每个 chunk 独立计算 attention scores**
   - 利用 chunk 内 tokens 的独立性
   - 使用 shared memory 缓存 chunk 的 Q/K/V

2. **最后串行合并 state**
   - 只保留 chunk 边界处的 state
   - 减少串行步骤数: 128 → 8

### 预期效果

| 指标 | 当前 | 优化后 | 变化 |
|------|------|--------|------|
| Prefill 吞吐 | 637 tok/s | 900-1100 tok/s | +40-70% |
| Decode 吞吐 | 16,133 tok/s | 15,500 tok/s | -4% |
| 实现难度 | - | 高 | - |

### 权衡

- **收益**: Prefill 大幅提升
- **代价**: Decode 略降（chunk 边界处理开销）
- **风险**: 精度可能受影响（近似计算）

---

## 策略 2: FP16/BF16 混合精度

### 原理

当前使用 FP32，但 RTX 5060 Ti 的 Tensor Core 支持 FP16/BF16 加速：

| 精度 | 带宽 | Tensor Core 速度 | 适用场景 |
|------|------|-----------------|----------|
| FP32 | 100% | 1x | 当前 |
| TF32 | 100% | 2x | 已启用 |
| BF16 | 50% | 2x | 推荐 |
| FP16 | 50% | 2x | 推荐 |

### 实现要点

1. **权重存储 FP16/BF16**
   - 显存减半
   - 加载带宽减半

2. **GEMM 使用 FP16 Tensor Core**
   - cuBLAS: `CUBLAS_COMPUTE_16F`
   - 速度提升 2x

3. **Attention 计算保持 FP32**
   - softmax 数值稳定性
   - 仅在 GEMM 使用 FP16

### 预期效果

| 指标 | 当前 | 优化后 | 变化 |
|------|------|--------|------|
| Prefill 吞吐 | 637 tok/s | 950-1200 tok/s | +50-90% |
| Decode 吞吐 | 16,133 tok/s | 20,000-25,000 tok/s | +25-55% |
| 显存占用 | 10.6 GB | 6-7 GB | -40% |
| 实现难度 | - | 中 | - |

### 权衡

- **收益**: Prefill 和 Decode 都提升
- **代价**: 需要修改所有 kernel 和 cuBLAS 调用
- **风险**: 精度损失（需验证）

---

## 策略 3: 调整 Linear/Full Attention 比例

### 原理

当前比例: 50/50 (Linear:Full = 12:12 layers)

Full Attention 的 prefill 比 Linear Attention 更快（因为可以并行），但 decode 更慢（需要访问 KV cache）。

### 方案对比

| 比例 | Linear:Full | Prefill 预估 | Decode 预估 | 适用场景 |
|------|------------|-------------|------------|----------|
| 75/25 (原始) | 18:6 | 550 tok/s | 17,500 tok/s | Decode 优先 |
| 50/50 (当前) | 12:12 | 637 tok/s | 16,133 tok/s | 平衡 |
| 25/75 | 6:18 | 750 tok/s | 12,000 tok/s | Prefill 优先 |
| 0/100 (v3.0) | 0:24 | 900 tok/s | 3,000 tok/s | 仅 prefill |

### 预期效果 (25/75)

| 指标 | 当前 (50/50) | 优化后 (25/75) | 变化 |
|------|-------------|---------------|------|
| Prefill 吞吐 | 637 tok/s | 750 tok/s | +18% |
| Decode 吞吐 | 16,133 tok/s | 12,000 tok/s | -26% |
| 实现难度 | - | 极低 | - |

### 权衡

- **收益**: 无需代码改动，仅修改配置
- **代价**: Decode 显著下降
- **风险**: 模型精度可能受影响（架构改变）

---

## 策略 4: CUDA Graph for Prefill

### 原理

当前 CUDA Graph 仅用于 decode。Prefill 阶段有大量 kernel launch 开销。

**障碍**: `forward_batch_prefill` 中有动态操作：
- `cudaMemcpyDeviceToHost` (获取 max_seq)
- CPU 循环处理 KV cache

**解决方案**:
1. 预计算 max_seq（已完成）
2. 确保所有 kernel 使用相同 stream
3. 预分配所有中间 buffer

### 预期效果

| 指标 | 当前 | 优化后 | 变化 |
|------|------|--------|------|
| Prefill 吞吐 | 637 tok/s | 700-750 tok/s | +10-18% |
| Decode 吞吐 | 16,133 tok/s | 16,133 tok/s | 无 |
| 实现难度 | - | 中 | - |

### 权衡

- **收益**: Prefill 提升，Decode 不变
- **代价**: 需要重构 prefill 路径
- **风险**: Graph 捕获失败（动态内存分配）

---

## 推荐执行顺序

| 优先级 | 策略 | 成本 | Prefill 收益 | Decode 影响 | 综合推荐 |
|--------|------|------|-------------|------------|----------|
| 1 | **FP16/BF16** | 中 | +50-90% | +25-55% | ⭐⭐⭐⭐⭐ |
| 2 | **CUDA Graph** | 中 | +10-18% | 无 | ⭐⭐⭐⭐ |
| 3 | **Chunk-based** | 高 | +40-70% | -4% | ⭐⭐⭐ |
| 4 | **调整比例** | 极低 | +18% | -26% | ⭐⭐ |

---

## 下一步行动

### 立即执行 (本周)
1. **FP16/BF16 量化**
   - 修改 `CudaMLP`, `CudaLinearAttention`, `CudaFullAttention` 的权重存储
   - 修改 cuBLAS 调用使用 `CUBLAS_COMPUTE_16F`
   - 验证精度

### 短期 (下周)
2. **CUDA Graph Prefill**
   - 重构 `forward_batch_prefill_graph`
   - 消除所有 D2H memcpy
   - 预分配所有 buffer

### 中期 (下月)
3. **Chunk-based Gated Delta**
   - 设计 chunk 并行算法
   - 实现并验证精度
   - 调优 chunk 大小

---

## 性能目标

| 版本 | Prefill (batch=128) | Decode (batch=1) | 实现策略 |
|------|--------------------|------------------|----------|
| v3.2 (当前) | 637 tok/s | 16,133 tok/s | - |
| v3.3 (FP16) | 1000 tok/s | 20,000 tok/s | FP16 + CUDA Graph |
| v3.4 (Chunk) | 1200 tok/s | 19,000 tok/s | + Chunk-based |
| v4.0 (目标) | 1500 tok/s | 22,000 tok/s | + 更多优化 |
