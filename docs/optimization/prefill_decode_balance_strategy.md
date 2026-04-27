# Prefill vs Decode 性能平衡策略分析

## 当前性能基线

| 指标 | Prefill | Decode | 差距 |
|------|---------|--------|------|
| 吞吐 (tok/s) | 444.2 | 16,133 | Decode 快 **36x** |
| 延迟 (ms/token) | 2.25 | 0.062 | Prefill 慢 **36x** |
| 计算强度 | 高 (GEMM主导) | 低 (内存带宽主导) |

**核心问题**: Prefill 和 Decode 的吞吐严重不平衡。Decode 太快而 Prefill 太慢。

---

## 端到端算子流程分析

### Decode 路径 (单 token)

```
Input [1,1024] --GPU--> RMSNorm [1,1024] --GPU--> QKV_Proj(cuBLAS) [1,6144]
  --GPU--> Conv1D+State(Fused) [1,6144] --GPU--> L2Norm_QK(Fused) [1,2048]
  --GPU--> A/B/Z Proj [1,16/16/2048] --GPU--> GatedDelta(SharedMem) [1,2048]
  --GPU--> Norm+Gate(Fused) [1,2048] --GPU--> Out_Proj [1,1024]
  --GPU--> ResAdd+PostNorm(Fused) [1,1024]
  --GPU--> MLP: Gate/Up(cuBLAS) [1,3584] --GPU--> SiLU+Mul(Fused) [1,3584]
  --GPU--> Down(cuBLAS,beta=1) [1,1024]
```

**Decode 每 layer kernel 数**: ~12 个 kernel 启动

### Prefill 路径 (batch=128)

```
Input [128,1024] --H2D--> d_Input --GPU--> RMSNorm [128,1024]
  --GPU--> QKV_Proj_Batch(cuBLAS) [128,6144] --GPU--> Conv1D_Batch(Fused) [128,6144]
  --GPU--> L2Norm_QK_Batch(Fused) [128,2048] --GPU--> A/B/Z Proj_Batch(cuBLAS)
  --GPU--> GatedDelta_Seq(LOOP!) [128,2048] --GPU--> Norm+Gate_Batch(Fused) [128,2048]
  --GPU--> Out_Proj_Batch(cuBLAS) [128,1024] --GPU--> ResAdd+PostNorm [128,1024]
  --GPU--> MLP_Batch(cuBLASx3+Fused) [128,1024]
```

**Prefill 每 layer kernel 数**: ~12 个 kernel 启动 + H2D transfer

---

## 已完成的融合优化 (绿色)

| 优化 | 状态 | 收益 |
|------|------|------|
| Conv1D + State Update | Fused | 减少 1 个 kernel |
| L2Norm Q + K | Fused | 减少 1 个 kernel |
| Norm + Gate | Fused | 减少 1 个 kernel |
| ResAdd + PostNorm | Fused | 减少 1 个 kernel |
| SiLU + Mul | Fused | 减少 1 个 kernel |
| MLP Down + Residual | beta=1 | 减少 1 个 kernel |

---

## 剩余的融合机会 (红色)

### 机会 1: GatedDelta + Norm + Gate 三合一
**当前**: 2 个 kernel (gated_delta_kernel -> norm_gate_fused_kernel)
**建议**: 1 个 kernel (gated_delta_norm_gate_fused)
**收益**: 减少 1 个 kernel 启动 (~5-10us), 消除 attn_out 中间 buffer
**风险**: 寄存器压力增加，可能降低 occupancy
**预计提速**: 2-5%

### 机会 2: QKV Projection + Conv1D + L2Norm 链式融合
**当前**: 3 个独立 kernel (cuBLAS GEMM -> conv1d -> l2norm)
**建议**: 1 个 fused kernel
**收益**: 大幅减少 kernel 启动开销和中间数据搬运
**风险**: 实现复杂，cuBLAS GEMM 难以与 custom kernel 融合
**预计提速**: 5-10%

### 机会 3: A/B/Z Projection 合并为单一 GEMM
**当前**: 3 个 cublasSgemv (A, B, Z 分别投影)
**建议**: 1 个 cublasSgemm 输出 [1, NH+NH+Z_DIM]
**收益**: 3 个 kernel -> 1 个 kernel
**预计提速**: 5-8%

---

## Prefill 瓶颈深度分析

### 瓶颈 1: GatedDelta 串行处理 (最关键)

```cpp
// current code
for (int b = 0; b < batch_size; ++b) {
    gated_delta_kernel<<<...>>>(...);  // 每个 token 串行！
}
```

**问题**: batch prefill 时，gated_delta 仍然逐个 token 处理，因为 recurrent state 有依赖。
**影响**: 128 个 token 需要 128 次 kernel 启动，无法利用 batch 并行。

**解决方案**:
- **方案 A**: 使用 prefix scan 并行化 (已尝试，实现复杂)
- **方案 B**: 将 tokens 分 chunk，chunk 内串行但 chunk 间并行
- **方案 C**: 降低 Linear Attention 层数，增加 Full Attention 层数

### 瓶颈 2: H2D Transfer for positions

```cpp
cudaMemcpy(d_positions_buf_, positions, batch_size * sizeof(int), cudaMemcpyHostToDevice);
```

**问题**: 每次 prefill 都需要 H2D transfer
**解决方案**: 预分配并重用 device buffer

### 瓶颈 3: Kernel 启动开销

Decode 和 Prefill 都有 ~12 个 kernel/层 × 24 层 = 288 个 kernel/forward
**解决方案**: CUDA Graph (已部分实现但禁用)

---

## 平衡策略提案

### 策略 1: 降低 Decode 速度，提升 Prefill (推荐)

**思路**: 当前 Decode 16,133 tok/s 远快于 Prefill 444 tok/s。可以适当牺牲 Decode 性能来提升 Prefill。

**具体措施**:
1. **减少 KV Cache 优化**: 当前 Decode 快是因为 KV Cache 命中率高。可以降低 KV Cache 精度 (FP16/BF16) 或压缩，这会影响 Decode 但几乎不影响 Prefill
2. **降低 Decode 的 batch 处理**: 当前 decode 是单 token，如果增加 decode 的延迟容忍度，可以将更多资源分配给 prefill
3. **动态负载均衡**: 根据当前是 prefill 还是 decode 阶段，动态调整 GPU 资源分配

**预计效果**:
- Decode: 16,133 -> ~12,000 tok/s (-25%)
- Prefill: 444 -> ~600 tok/s (+35%)

### 策略 2: 架构调整 - 增加 Full Attention 层比例

**当前**: 3/4 层是 Linear Attention，1/4 是 Full Attention
**建议**: 调整为 1/2 Linear + 1/2 Full Attention

**理由**:
- Full Attention 的 prefill 可以通过 Flash Attention 高效并行
- Linear Attention 的 decode 更快，但 prefill 是瓶颈
- 增加 Full Attention 层会提升 prefill 吞吐，降低 decode 吞吐

**预计效果**:
- Decode: 16,133 -> ~10,000 tok/s (-38%)
- Prefill: 444 -> ~700 tok/s (+58%)

### 策略 3: Batch Prefill 优化 - 真正的 Batch 并行

**当前**: batch prefill 只是数据 batch，但 gated_delta 仍串行
**建议**: 实现真正的 batch 并行 gated_delta

**技术方案**:
```cpp
// 将 recurrent state 从 per-token 改为 per-batch
// 使用独立的 state  per batch element，消除串行依赖
__global__ void batch_gated_delta_parallel(...) {
    int b = blockIdx.y;  // batch index
    int h = blockIdx.x;  // head index
    // 每个 batch element 独立计算，无依赖
}
```

**预计效果**:
- Prefill: 444 -> ~900 tok/s (+103%)
- Decode: 不受影响

### 策略 4: CUDA Graph 启用

**当前**: CUDA Graph 已禁用
**建议**: 修复并启用 CUDA Graph

**需要修复**:
1. 消除 Full Attention 中的 D2H memcpy (已预计算 max_seq)
2. 确保所有 kernel 使用相同的 stream
3. 预分配所有中间 buffer

**预计效果**:
- Prefill: 444 -> ~500 tok/s (+13%)
- Decode: 不受影响

---

## 策略对比

| 策略 | Prefill 提升 | Decode 影响 | 实现难度 | 风险 |
|------|-------------|------------|---------|------|
| 1. 降低 Decode 优化 | +35% | -25% | 低 | 低 |
| 2. 增加 Full Attn 比例 | +58% | -38% | 中 | 中 (精度影响) |
| 3. Batch 并行 GatedDelta | +103% | 无 | 高 | 高 (实现复杂) |
| 4. CUDA Graph | +13% | 无 | 中 | 低 |

---

## 推荐执行顺序

1. **第一步**: 策略 4 (CUDA Graph) - 低风险，立即收益
2. **第二步**: 策略 1 (降低 Decode 优化) - 简单调整，平衡性能
3. **第三步**: 策略 2 (架构调整) - 如果前两步不够，再调整架构
4. **第四步**: 策略 3 (Batch 并行) - 最后尝试，风险最高但收益最大

---

## 需要您审批的决策

1. **是否接受降低 Decode 性能来提升 Prefill？**
   - 目标平衡: Prefill ~600 tok/s, Decode ~12,000 tok/s

2. **是否调整 Linear/Full Attention 层比例？**
   - 当前: 75% Linear + 25% Full
   - 建议: 50% Linear + 50% Full

3. **是否投入高成本实现 Batch 并行 GatedDelta？**
   - 预计开发时间: 1-2 周
   - 预计收益: Prefill 翻倍

请审批后执行。
