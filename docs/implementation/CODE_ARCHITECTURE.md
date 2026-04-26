# 代码脉络与架构分析

## 一、整体数据流

```
输入Token IDs
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Token Embedding (token_embedding_cuda.cu)                    │
│    - 输入: token_ids[seq_len]                                   │
│    - 输出: hidden_states[seq_len, hidden_size=1024]             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 24层Transformer循环 (cuda_engine.cu)                         │
│    - 6层 Full Attention (layer 3,7,11,15,19,23)                │
│    - 18层 Linear Attention (其他层)                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Final RMSNorm (rmsnorm_cuda.cu)                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. LM Head (lm_head_cuda.cu)                                   │
│    - 输出: logits[vocab_size=248320]                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
采样 → 下一个Token
```

## 二、每层内部数据流

### 2.1 Linear Attention 层 (18层)

```
hidden_states
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ RMSNorm (rmsnorm_cuda.cu)                                      │
│ - RMSNorm: normed = hidden_states * rmsnorm_weight              │
│ - 输出: normed[seq_len, 1024]                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Linear Attention (linear_attention_cuda.cu)                     │
│ - Mixed QKV: mixed_qkv = in_proj_weight @ normed               │
│ - Conv1D: conv_out = conv1d(mixed_qkv)                         │
│ - State Update: state = state + conv_out @ conv_out.T          │
│ - Output: output = state @ conv_out                            │
│ - 输出: output[seq_len, 1024]                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ MLP (mlp_cuda.cu)                                              │
│ - Gate: gate_weight @ input                                     │
│ - Up: up_weight @ input                                        │
│ - SiLU + Mul: silu(gate) * up → hidden                        │
│ - Down: down_weight @ hidden                                   │
│ - Add: residual += down_output                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Full Attention 层 (6层)

```
hidden_states
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ RMSNorm (rmsnorm_cuda.cu)                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Full Attention (full_attention_cuda.cu)                        │
│                                                                 │
│ Q/K/V投影:                                                      │
│   Q = q_proj_weight @ normed  → [seq_len, num_heads, head_dim] │
│   K = k_proj_weight @ normed  → [seq_len, num_kv_heads, kv_head_dim]│
│   V = v_proj_weight @ normed  → [seq_len, num_kv_heads, kv_head_dim]│
│                                                                 │
│ FlashAttention:                                                 │
│   - O = attention(Q, K, V) → [seq_len, num_heads, head_dim]    │
│   - O_proj: o_proj_weight @ O → [seq_len, hidden_size]         │
│   - Add + RMSNorm: residual += O_proj, norm(residual)          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ MLP (mlp_cuda.cu)                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 三、核心Kernel详解

### 3.1 full_attention_cuda.cu

**功能**: Full Attention层的完整实现，包含QKV投影、FlashAttention、O投影

**关键函数**:
```cpp
void CudaFullAttention::forward(const float* input, float* output, CudaKVCache& kv_cache,
                                int layer_idx, int position) const;
```

**v3.0 优化**:
- Decode阶段(seq_len=1)使用SGEMV替代GEMM
- FlashAttention v30支持head_dim=256
- Attention kernel根据序列长度自适应选择(v22 vs v30)

### 3.2 linear_attention_cuda.cu

**功能**: Linear Attention层的实现，使用Conv1D和状态更新

**关键函数**:
```cpp
void CudaLinearAttention::forward(const float* input, float* output, CudaLinearAttnState& state);
```

**v3.0 Batch 优化**:
- cuBLAS GEMM 一次性处理 batch 的 QKV/A/B/Z/O projection
- Batch kernel: conv1d_update_fused_batch, l2norm_qk_fused_batch, norm_gate_fused_batch
- Kernel launch 从 batch_size×8 减少到约 9 个

### 3.3 mlp_cuda.cu

**功能**: MLP层融合 - Gate + Up + SiLU + Mul + Down + Add

**关键函数**:
```cpp
void CudaMLP::forward(const float* input, float* output, int seq_len, cudaStream_t stream);
```

**融合操作**:
1. Gate GEMM: `gate = gate_weight @ input`
2. Up GEMM: `up = up_weight @ input`
3. SiLU+Mul: `hidden = silu(gate) * up`
4. Down GEMM: `down = down_weight @ hidden`
5. Add: `residual += down`

### 3.4 rmsnorm_cuda.cu

**功能**: RMSNorm实现

**关键函数**:
```cpp
void CudaRMSNorm::forward(const float* input, float* output, int seq_len, cudaStream_t stream);
```

**计算**:
```cpp
rms = sqrt(sum(input^2) / hidden_size + eps)
output = input / rms * weight
```

## 四、CUDA Graph 在 Decode 阶段的应用

### 4.1 为什么使用CUDA Graph

Decode阶段每次只处理1个token，Kernel Launch开销占比高。CUDA Graph通过捕获所有kernel调用为一个图，多次执行时只有一次launch开销。

### 4.2 实现方式

```cpp
// 1. 创建Graph
cudaGraphCreate(&graph, 0);

// 2. 捕获Decode阶段所有kernel
cudaGraphAddKernelNode(...);  // Embedding
cudaGraphAddKernelNode(...);  // Layer 1
// ... 24 layers ...
cudaGraphAddKernelNode(...);  // Final RMSNorm
cudaGraphAddKernelNode(...);  // LM Head
cudaGraphAddKernelNode(...);  // Sampling

// 3. 实例化Graph
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 4. 执行(多次)
cudaGraphLaunch(graphExec, stream);
```

## 五、FlashAttention v30 实现要点

### 5.1 Tiling策略

```cpp
#define FA_BR_V30 8   // 每个block处理的行数
#define FA_BC_V30 16  // 每个tile的列数(key数量)
```

对于head_dim=256:
- Shared Memory: `s_k[16][256] + s_v[16][256] = 32KB`
- 远小于RTX 5060 Ti的48KB限制

### 5.2 Warp级并行

```
Block(256 threads) = 8 warps
- 每个warp处理1行
- 每个lane处理1个key位置
- Warp内通过shuffle同步max和sum
```

### 5.3 Online Softmax

```cpp
// 在线更新，避免存储完整attention矩阵
row_m = max(score)           // 当前行最大值
row_l = sum(exp(score - m))  // 当前行指数和
row_o = row_o * exp(row_m_old - row_m) + new_contribution
```

## 六、性能瓶颈分析

### 6.1 Prefill阶段

| 操作 | 时间占比 | 瓶颈类型 |
|------|---------|----------|
| QKV GEMM | ~40% | Compute |
| Attention | ~30% | Memory |
| O_proj GEMM | ~15% | Compute |
| MLP GEMM | ~15% | Compute |

**优化方向**: FP16/BF16量化，Tensor Core加速

### 6.2 Decode阶段

| 操作 | 时间占比 | 瓶颈类型 |
|------|---------|----------|
| QKV SGEMV | ~25% | Memory |
| Attention | ~20% | Memory |
| O_proj SGEMV | ~10% | Memory |
| MLP SGEMV | ~45% | Memory |

**优化方向**: INT8量化，减少带宽压力

## 七、代码组织原则

1. **单一职责**: 每个kernel只做一件事
2. **融合优先**: 减少HBM访问
3. **自适应**: 根据序列长度选择最优kernel
4. **零拷贝**: 复用buffer，减少分配开销
