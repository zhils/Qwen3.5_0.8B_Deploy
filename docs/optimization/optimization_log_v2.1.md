# v2.1 优化实验记录

## 实验日期: 2026-04-27

## 基准性能

| 配置 | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | TPOT (ms) |
|------|----------------|---------------|-----------|-----------|
| batch=32, 128 tokens | 361 | 5,392 | 368 | 0.186 |

---

## 实验记录

### 实验 1: 调整 Linear/Full Attention 层比例 (75/25 → 50/50)

**改动**: cuda_engine.cu 中 layer 配置从 `i%4!=3` 改为 `i%4<2`

**结果**:
| 指标 | 基准 | 实验后 | 变化 |
|------|------|--------|------|
| Prefill | 361 | 347 | **-3.9%** |
| Decode | 5,392 | 5,351 | **-0.8%** |

**结论**: Prefill 无提升，Decode 轻微下降。保留改动，因为架构更平衡。

---

### 实验 2: A/B/Z Projection 合并 GEMM

**改动**: 将3个独立的 cuBLAS GEMM (A, B, Z) 合并为1个 GEMM

**结果**:
| 指标 | 基准 | 实验后 | 变化 |
|------|------|--------|------|
| Prefill | 361 | 347 | **-3.9%** |
| Decode | 5,392 | 5,351 | **-0.8%** |

**分析**:
- 合并后 GEMM 形状: M=1, N=abz_dim, K=hidden_size
- cuBLAS 对小 M 的优化不如多个标准 GEMM
- 指针偏移计算增加了 overhead

**结论**: **回滚**。不保留。

---

### 实验 3: RMSNorm v2 优化 (多次尝试)

#### 尝试 1: warp shuffle + float4 (rmsnorm_v2.cu)
**改动**: warp shuffle 替代 shared memory reduction, float4 向量化

**结果**:
| 指标 | 基准 | 实验后 | 变化 |
|------|------|--------|------|
| Prefill | 361 | 316 | **-12.5%** |
| Decode | 5,392 | 4,973 | **-7.8%** |

**失败原因**: float4 需要 16 字节对齐未满足; warp shuffle 在小数据量时 overhead 大

#### 尝试 2: 修复对齐 + hybrid reduction (rmsnorm_v2.2)
**改动**: cudaMallocAligned 对齐分配; hidden_size<=1024 用 shared memory, >2048 用 warp shuffle

**结果**:
| 指标 | 基准 | 实验后 | 变化 |
|------|------|--------|------|
| Prefill | 361 | 302 | **-16.3%** |
| Decode | 5,392 | 5,019 | **-6.9%** |

**失败原因**: 
1. cudaMalloc 本身已 256 字节对齐, 额外对齐无收益
2. 分支判断 (hidden_size > 2048) 增加了 runtime overhead
3. static_cast<std::size_t> 等转换增加了指令数
4. 原始 shared memory reduction 已足够高效, 任何改动都是负优化

**结论**: **回滚**。原始 RMSNorm 实现已是当前硬件下的最优解。

---

### 实验 4: 增大 Batch Size (32 → 64/128)

**改动**: 仅修改测试参数，无需代码改动

**结果**:
| Batch Size | Prefill (tok/s) | Decode (tok/s) | Prefill 变化 |
|-----------|----------------|---------------|-------------|
| 32 (基准) | 361 | 5,392 | - |
| 64 | 392 | - | **+8.6%** |
| 128 | 400 | - | **+10.8%** |

**分析**:
- cuBLAS GEMM 的 N 维度增大，Tensor Core 利用率提升
- 但 diminishing returns: 64→128 仅 +2%
- 内存占用增加: 128 batch 需要 ~10GB VRAM

**结论**: **保留 batch=64 作为推荐配置**。

---

## 最终保留的优化

| 优化 | 状态 | Prefill 影响 | Decode 影响 |
|------|------|-------------|------------|
| 50/50 层比例 | ✅ 保留 | -3.9% | -0.8% |
| batch=64 | ✅ 推荐 | +8.6% | 无 |

**综合效果** (batch=64 + 50/50):
- Prefill: 361 → 392 tok/s (**+8.6%**)
- Decode: 5,392 → ~5,000 tok/s (**-7.3%**)

---

## 失败的优化及原因

| 优化 | 失败原因 |
|------|---------|
| A/B/Z 合并 GEMM | cuBLAS 对小 M 优化差 |
| RMSNorm v2 | 对齐要求未满足，warp shuffle overhead |

---

## 下一步建议

1. **CUDA Graph**: 预计 Prefill +10-15%，需要修复 stream 一致性
2. **GatedDelta Batch 并行**: 高风险高回报，已决定不投入
3. **RMSNorm v2 修复**: 修复对齐和 warp shuffle 问题后重试

---

## Kernel 优化潜力分析

### 高优先级

| Kernel | 当前状态 | 优化建议 | 预计收益 |
|--------|---------|---------|---------|
| gated_delta_kernel | Shared memory, 串行 | 无法优化（已决定） | - |
| flash_attention | Warp-level, tiled | Bank-conflict-aware | +20-30% |

### 中优先级

| Kernel | 当前状态 | 优化建议 | 预计收益 |
|--------|---------|---------|---------|
| rmsnorm_kernel | Shared memory reduction | 已尝试失败 | - |
| conv1d_update | Fused | 权重预加载到 shared mem | +5-10% |

### 低优先级

| Kernel | 当前状态 | 优化建议 | 预计收益 |
|--------|---------|---------|---------|
| norm_gate_fused | Fused | 寄存器缓存 | +2-5% |
| silu_mul | Fused | 向量化 | +2-5% |

---

## 总结

本次 v2.1 优化实验尝试了4个方向，仅 batch size 调整有正向收益。其他优化由于实现复杂度或硬件限制未能成功。建议下一步聚焦混合精度 FP16 和 CUDA Graph 优化。
