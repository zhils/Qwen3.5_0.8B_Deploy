# Prefill 性能瓶颈深度分析 v2.1

## 当前状态（已应用 50/50 层比例）

| 指标 | 基准 (75/25) | 当前 (50/50) | 变化 |
|------|-------------|-------------|------|
| Prefill 吞吐 | 361 tok/s | 347 tok/s | **-3.9%** |
| Decode 吞吐 | 5,392 tok/s | 5,351 tok/s | **-0.8%** |
| Prefill/Decode 比 | 1:14.9 | 1:15.4 | 更不平衡 |

**结论**: 单纯调整层比例没有改善 Prefill 性能，因为 Full Attention 的 prefill 也是串行处理。

---

## 端到端瓶颈分析

### 1. Kernel 启动开销分析

Decode 路径每 layer:
```
1. RMSNorm (kernel)
2. QKV Projection (cuBLAS GEMM)
3. A Projection (cuBLAS GEMM)
4. B Projection (cuBLAS GEMM)
5. Z Projection (cuBLAS GEMM)
6. Conv1D+State (fused kernel)
7. L2Norm Q+K (fused kernel)
8. GatedDelta (kernel, sequential)
9. Norm+Gate (fused kernel)
10. Output Projection (cuBLAS GEMM)
11. ResAdd+PostNorm (fused kernel)
12. MLP Gate/Up (cuBLAS GEMM x2)
13. SiLU+Mul (fused kernel)
14. MLP Down+Res (cuBLAS GEMM)
```

**总计**: ~14 个操作/层 × 24 层 = **336 个 kernel 启动/forward**

**Kernel 启动开销**: 每个 kernel ~5-10us，总计 **1.7-3.4ms**

对于 Prefill (128 tokens, 347 tok/s = 368ms)，kernel 启动开销占 **0.5-0.9%**

### 2. H2D 传输瓶颈

```cpp
// forward_batch_prefill 中的 H2D
CHECK_CUDA(cudaMemcpy(d_positions_buf_, positions, batch_size * sizeof(int),
                      cudaMemcpyHostToDevice));
```

- 传输量: 128 tokens × 4 bytes = **512 bytes**
- PCIe 带宽: ~16 GB/s
- 理论时间: **0.03us** (可忽略)

**结论**: H2D 不是瓶颈

### 3. 内存带宽瓶颈

Decode 阶段 (单 token):
- 权重读取: ~1.2GB (模型参数)
- KV Cache 读取: 2 heads × 256 dim × 4 bytes × seq_len
- 对于 seq_len=512: ~1MB

Prefill 阶段 (batch=128):
- 权重读取: ~1.2GB (与 decode 相同，但复用)
- 激活值: 128 × 1024 × 4 bytes × 24 layers = **12MB**

**内存带宽利用率**:
- RTX 5060 Ti: 448 GB/s
- Prefill 实际使用: ~15 GB/s (主要是权重读取)
- **利用率仅 3.4%** - 远未饱和

### 4. 计算瓶颈

Prefill 阶段计算量:
- QKV Projection: 128 × 1024 × 6144 × 2 = 1.6 GFLOP
- MLP: 128 × 1024 × 3584 × 2 × 3 = 2.8 GFLOP
- Attention: 128 × 16 × 128 × 128 × 2 = 67 MFLOP
- 总计/层: ~4.5 GFLOP
- 总计 24 层: **108 GFLOP**

RTX 5060 Ti FP32: ~22 TFLOP/s
理论时间: 108 GFLOP / 22 TFLOP/s = **4.9ms**

实际时间: 368ms
**效率: 1.3%** - 极度低效！

### 5. 真正瓶颈: GatedDelta 串行处理

```cpp
// forward_batch_prefill 中的关键代码
for (int b = 0; b < batch_size; ++b) {
    gated_delta_kernel<<<...>>>(...);  // 128 次串行启动！
}
```

每个 token 的 gated_delta:
- 16 heads × 128 dim = 2048 threads
- 但实际只启动 16 个 block，每个 128 threads
- 每个 token 计算时间: ~0.5ms
- 128 tokens: **64ms** (占 prefill 总时间的 17%)

**但更重要的是**: 这 128 次 kernel 启动之间没有并行，GPU 大部分时间在等待。

### 6. 另一个瓶颈: 小 batch GEMM 效率

Prefill 中的 cuBLAS GEMM:
```cpp
// QKV Projection: [128, 1024] x [1024, 6144]
cublasSgemm(..., M=6144, N=128, K=1024)
```

对于 cuBLAS:
- M=6144, N=128, K=1024
- N=128 较小，无法充分利用 Tensor Core
- 需要 N >= 256 才能达到峰值效率

---

## 低成本优化方案（无需 Batch 并行 GatedDelta）

### 方案 1: 增大 Prefill Batch Size

**当前**: batch_size=32 (测试配置)
**建议**: batch_size=64 或 128

**原理**:
- cuBLAS GEMM 的 N 维度增大，Tensor Core 利用率提升
- 从 [32, 1024] 到 [128, 1024]，GEMM 效率提升 ~2x

**成本**: 仅需修改测试配置，无需代码改动
**预计收益**: Prefill +20-40%

### 方案 2: 权重预加载到 Shared Memory

**当前**: 每次 kernel 从 global memory 读取权重
**建议**: 将小型权重（如 conv1d weight, norm weight）预加载到 shared memory

**适用 kernel**:
- conv1d_update_fused_kernel
- l2norm_qk_fused_kernel
- norm_gate_fused_kernel

**成本**: 低，只需修改 kernel 代码
**预计收益**: Decode +5-10%, Prefill +2-5%

### 方案 3: 减少 Layer 数量

**当前**: 24 层
**建议**: 减少到 16 层（每 3 层去掉 1 层）

**原理**:
- 直接减少 33% 的计算量
- 对模型精度影响可通过微调补偿

**成本**: 需要重新训练/微调
**预计收益**: Prefill +50%, Decode +50%

### 方案 4: 混合精度 (FP16/BF16)

**当前**: FP32
**建议**: 权重和激活使用 FP16，计算使用 TF32

**原理**:
- 内存带宽减半
- Tensor Core 利用率翻倍
- RTX 5060 Ti 支持 FP16 加速

**成本**: 中等，需要修改所有 kernel 和 cuBLAS 调用
**预计收益**: Prefill +50-80%, Decode +30-50%

### 方案 5: 静态 CUDA Graph

**当前**: CUDA Graph 已禁用
**建议**: 修复并启用

**需要修复**:
1. 预计算 max_seq（已完成）
2. 确保所有 kernel 使用相同 stream
3. 预分配所有中间 buffer

**成本**: 中等
**预计收益**: Prefill +10-15%, Decode +5-10%

---

## 推荐执行顺序

| 优先级 | 方案 | 成本 | Prefill 收益 | Decode 影响 |
|--------|------|------|-------------|------------|
| 1 | 增大 Batch Size | 极低 | +20-40% | 无 |
| 2 | 混合精度 FP16 | 中 | +50-80% | +30-50% |
| 3 | CUDA Graph | 中 | +10-15% | +5-10% |
| 4 | Shared Memory 权重 | 低 | +2-5% | +5-10% |
| 5 | 减少 Layer 数量 | 高 | +50% | +50% |

---

## 下一步建议

1. **立即执行**: 测试 batch_size=64/128 的 prefill 性能
2. **短期**: 实现 FP16 混合精度
3. **中期**: 启用 CUDA Graph
4. **长期**: 评估减少 layer 数量的可行性

请审批后执行。
