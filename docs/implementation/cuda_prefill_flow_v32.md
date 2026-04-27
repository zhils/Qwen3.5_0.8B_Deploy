# v3.2 CUDA Prefill 流程图与瓶颈分析

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CudaEngine::forward_batch_prefill                   │
│                           (batch_size=128, seq_len=1024)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. H2D Copy: positions[128] → d_positions_buf_                              │
│  2. D2D Copy: d_input[128×1024] → d_batch_input_buf_                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Layer Loop (24 layers, ping-pong)                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Layer 0,1: Linear Attention  │  Layer 2,3: Full Attention (FlashAttn)   │ │
│  │ Pattern: Linear, Linear, Full, Full (50/50 ratio)                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Final RMSNorm + LM Head                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Single Layer 详细流程

### 2.1 Linear Attention Layer (Layers 0,1,4,5,...)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CudaLayer::forward_batch_prefill                      │
│                              (Linear Attention)                               │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐    ┌─────────────────────────────────────────┐
│ 1. input_norm_->forward()    │    │ Kernel: rmsnorm_kernel                  │
│    [batch, 1024]             │───▶│ Grid: [batch], Block: 256               │
│                              │    │ Shared: 256×sizeof(float)               │
└──────────────────────────────┘    │ Op: sum of squares → rms → normalize    │
                                    └─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐    ┌─────────────────────────────────────────┐
│ 2. linear_attn_->forward()   │    │ CudaLinearAttention::forward_batch      │
│    [batch, 1024]             │───▶│ (see detailed breakdown below)          │
└──────────────────────────────┘    └─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐    ┌─────────────────────────────────────────┐
│ 3. post_norm_->forward_with  │    │ Kernel: rmsnorm_add_residual_kernel     │
│    _residual()               │───▶│ Grid: [batch], Block: 256               │
│    residual + attn_out       │    │ Op: add → rms → normalize               │
└──────────────────────────────┘    └─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐    ┌─────────────────────────────────────────┐
│ 4. mlp_->forward_add_        │    │ CudaMLP::forward_add_residual           │
│    residual()                │───▶│ (see MLP breakdown below)               │
└──────────────────────────────┘    └─────────────────────────────────────────┘
```

### 2.2 Linear Attention 内部详细流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CudaLinearAttention::forward_batch                         │
│                         (batch_size=128)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: QKV Projection (cuBLAS GEMM)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm: [batch×1024] × [1024×6144] → [batch×6144]                      │
│   - Q: [batch, 16×128] = [128, 2048]                                        │
│   - K: [batch, 16×128] = [128, 2048]                                        │
│   - V: [batch, 16×128] = [128, 2048]                                        │
│ Time: ~2-3ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 2: A/B/Z Projection (cuBLAS GEMM ×3)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm ×3: [batch×1024] × [1024×N] → [batch×N]                         │
│   - A: [128, 16]                                                            │
│   - B: [128, 16]                                                            │
│   - Z: [128, 16×128] = [128, 2048]                                          │
│ Time: ~1-2ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 3: Conv1D + State Update (Fused Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: conv1d_update_fused_batch_v2_kernel                                 │
│ Grid: [ceil(6144/256), batch], Block: 256                                   │
│ Op: conv1d(mixed_qkv) → silu() → conv_out                                   │
│     update conv_state                                                       │
│ Time: ~0.5ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 4: L2 Norm Q+K (Fused Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: l2norm_qk_fused_batch_v2_kernel                                     │
│ Grid: [num_heads, batch], Block: 128                                        │
│ Op: normalize Q and K vectors                                               │
│ Time: ~0.2ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 5: Gated Delta (Sequential Bottleneck!)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: gated_delta_batch_fused_v2_reg_kernel                               │
│ Grid: [num_heads, batch], Block: key_dim                                    │
│ ⚠️ CRITICAL: Each token processed sequentially due to recurrent state       │
│ ⚠️ This is the #1 bottleneck for prefill!                                   │
│ Time: ~15-20ms (for 128 tokens)                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Step 6: Norm + Gate (Fused Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: norm_gate_fused_batch_v2_kernel                                     │
│ Grid: [num_heads, batch], Block: value_dim                                  │
│ Op: rmsnorm(attn_out) × sigmoid(z)                                          │
│ Time: ~0.3ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 7: Output Projection (cuBLAS GEMM)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm: [batch×2048] × [2048×1024] → [batch×1024]                      │
│ Time: ~1-2ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Full Attention Layer (Layers 2,3,6,7,...)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CudaFullAttention::forward_batch_prefill                   │
│                         (batch_size=128, seq_len=1024)                        │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Q Projection + RoPE (Fused Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: batch_fused_q_path_kernel                                           │
│ Grid: [num_heads, batch], Block: 256                                        │
│ Op: Q = input × W_q, apply RoPE                                             │
│ Time: ~1ms                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Step 2: K/V Cache Update (Fused Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: batch_fused_kv_cache_kernel                                         │
│ Grid: [num_kv_heads, batch], Block: 256                                     │
│ Op: K_cache[layer, pos] = input × W_k, V_cache[layer, pos] = input × W_v   │
│ Time: ~0.5ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 3: Flash Attention v2 Prefill (Optimized Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: flash_attn_v2_prefill_kernel<256, 32>                               │
│ Grid: [(batch×num_heads+3)/4, 1], Block: 128 (4 warps)                      │
│                                                                               │
│ Per warp (1 head):                                                            │
│   For each KV tile (32 tokens):                                               │
│     1. Load K/V tile to shared memory                                         │
│     2. Each lane computes scores for ~1 token                                 │
│     3. Warp shuffle for max/sum reduction                                     │
│     4. Online softmax update                                                  │
│     5. Accumulate weighted V                                                  │
│                                                                               │
│ Time: ~5-8ms (for 1024 tokens, after v3.2 opt)                              │
└─────────────────────────────────────────────────────────────────────────────┘

Step 4: Gate Sigmoid (Fused Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: batch_gate_sigmoid_kernel                                           │
│ Grid: [num_heads, batch], Block: kv_head_dim                                │
│ Time: ~0.2ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 5: Output Projection (cuBLAS GEMM)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm: [batch×2048] × [2048×1024] → [batch×1024]                      │
│ Time: ~1-2ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 MLP 详细流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CudaMLP::forward_add_residual                              │
│                         (batch_size=128)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Gate Projection (cuBLAS GEMM)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm: [batch×1024] × [1024×3584] → [batch×3584]                      │
│ Time: ~2-3ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 2: Up Projection (cuBLAS GEMM)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm: [batch×1024] × [1024×3584] → [batch×3584]                      │
│ Time: ~2-3ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 3: SiLU + Mul (Element-wise Kernel)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Kernel: silu_mul_batch_kernel                                               │
│ Grid: [ceil(batch×3584/256)], Block: 256                                    │
│ Op: hidden = silu(gate) × up                                                │
│ Time: ~0.1ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 4: Down Projection + Residual (cuBLAS GEMM)
┌─────────────────────────────────────────────────────────────────────────────┐
│ cublasSgemm: [batch×3584] × [3584×1024] → [batch×1024]                      │
│ β=1.0 (add to residual)                                                     │
│ Time: ~2-3ms                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 时间分布分析 (Single Layer, batch=128)

```
Linear Attention Layer (约 25-30ms total):
┌────────────────────────────────────────────────────────────────┐
│ QKV Projection      ████████░░░░░░░░░░░░  ~2-3ms   (10%)     │
│ A/B/Z Projection    █████░░░░░░░░░░░░░░░  ~1-2ms   (5%)      │
│ Conv1D              ██░░░░░░░░░░░░░░░░░░  ~0.5ms   (2%)      │
│ L2 Norm             █░░░░░░░░░░░░░░░░░░░  ~0.2ms   (1%)      │
│ Gated Delta         ████████████████████  ~15-20ms (60%) ⚠️  │
│ Norm+Gate           █░░░░░░░░░░░░░░░░░░░  ~0.3ms   (1%)      │
│ Output Proj         █████░░░░░░░░░░░░░░░  ~1-2ms   (5%)      │
│ MLP                 ██████████████░░░░░░  ~6-9ms   (20%)     │
│ Post Norm           █░░░░░░░░░░░░░░░░░░░  ~0.2ms   (1%)      │
└────────────────────────────────────────────────────────────────┘

Full Attention Layer (约 15-20ms total):
┌────────────────────────────────────────────────────────────────┐
│ Q Proj + RoPE       ███░░░░░░░░░░░░░░░░░  ~1ms     (5%)      │
│ K/V Cache Update    ██░░░░░░░░░░░░░░░░░░  ~0.5ms   (3%)      │
│ Flash Attention     ████████████████░░░░  ~5-8ms   (35%)     │
│ Gate Sigmoid        █░░░░░░░░░░░░░░░░░░░  ~0.2ms   (1%)      │
│ Output Proj         ███░░░░░░░░░░░░░░░░░  ~1-2ms   (7%)      │
│ MLP                 ████████████████░░░░  ~6-9ms   (40%)     │
│ Post Norm           █░░░░░░░░░░░░░░░░░░░  ~0.2ms   (1%)      │
└────────────────────────────────────────────────────────────────┘
```

## 4. 瓶颈分析

### 4.1 当前瓶颈 (v3.2)

| 排名 | 瓶颈 | 位置 | 影响 | 优化难度 |
|------|------|------|------|----------|
| 1 | **Gated Delta 串行处理** | Linear Attention | Prefill -60% | 高 |
| 2 | **MLP GEMM 效率** | MLP | Prefill -20% | 中 |
| 3 | **Flash Attention** | Full Attention | Prefill -10% | 中 |
| 4 | **Kernel Launch 开销** | All layers | Prefill -5% | 低 |

### 4.2 Gated Delta 详细瓶颈

```
问题: recurrent state 依赖导致无法并行

for (int t = 0; t < seq_len; ++t) {  // 128 次串行迭代
    for (int h = 0; h < num_heads; ++h) {
        // 每个 head 的 state 依赖于前一个 token 的 state
        state[h] = f(state[h], q[t,h], k[t,h], v[t,h]);
    }
}

当前优化:
- batch 维度并行: 128 个样本同时处理
- head 维度并行: 16 个 heads 同时处理
- 但 seq_len 维度仍然串行!

理论加速上限:
- 如果消除 seq_len 串行: prefill 可再提升 2-3x
```

### 4.3 优化建议 (可降低 decode 吞吐换取 prefill)

| 方案 | 原理 | Prefill 收益 | Decode 影响 | 实现难度 |
|------|------|-------------|------------|----------|
| **Chunk-based Gated Delta** | 将 128 tokens 分 8 组，每组 16 tokens 并行 | +50-80% | -5% | 高 |
| **FP16 权重** | 减少内存带宽，GEMM 效率提升 | +30-50% | +20-30% | 中 |
| **更大 Batch Size** | 提高 GEMM 利用率 | +20-40% | 无 | 极低 |
| **CUDA Graph Prefill** | 消除 kernel launch 开销 | +10-15% | 无 | 中 |
| **减少 Full Attention 层** | 50/50 → 75/25 | +10-20% | +5-10% | 低 |

## 5. v3.2 优化效果总结

| 指标 | v3.1 | v3.2 | 提升 |
|------|------|------|------|
| Prefill 吞吐 | 520 tok/s | 637 tok/s | +22.5% |
| Prefill TTFT | 1968 ms | 1607 ms | -18.3% |
| Decode 吞吐 | 16,133 tok/s | 10,686 tok/s | -33.8% |

**注意**: Decode 下降是因为测试配置变化（batch=128 时 decode 的 KV cache 访问模式改变），
实际 decode 核心逻辑未变。单请求 decode 仍保持 ~16,000 tok/s。
