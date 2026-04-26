# CUDA Kernel 优化记录

## 当前性能数据 (RTX 5060 Ti, 1024 prefill / 512 decode)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT** | 1,948 ms |
| **Prefill 吞吐** | 525.6 tok/s |
| **Decode TPOT** | 0.062 ms/tok |
| **Decode 吞吐** | 16,248 tok/s |
| **GPU VRAM** | 10,707 MB |

---

## 已完成的优化

### 1. Flash Attention v2
**文件**: [fused_kernels.cu](../src/backend/cuda/kernels/fused_kernels.cu)
- Warp-level parallelism
- Reduced HBM traffic
- Tiled computation for Q/K/V

### 2. Tensor Core (TF32)
**文件**: [mlp_cuda.cu](../src/backend/cuda/kernels/mlp_cuda.cu), [lm_head_cuda.cu](../src/backend/cuda/kernels/lm_head_cuda.cu)
- cuBLAS math mode: `CUBLAS_TF32_TENSOR_OP_MATH`
- GEMM acceleration for MLP projections

### 3. Kernel Fusion
**文件**: [fused_kernels.cu](../src/backend/cuda/kernels/fused_kernels.cu), [linear_attention_fused.cu](../src/backend/cuda/kernels/linear_attention_fused.cu)
- Gate + SiLU + Mul fusion
- RMSNorm + Residual fusion
- Conv1D + State update fusion
- L2 norm Q + K fusion
- Norm + Gate fusion
- Linear Attention 全融合内核 (forward_fused): 将 projection、conv1d、norm、gated delta、output projection 合并为单个 kernel

### 4. Batch Prefill
**文件**: [full_attention_cuda.cu](../src/backend/cuda/kernels/full_attention_cuda.cu)
- `batch_fused_q_path_kernel`: Batch Q projection + RoPE
- `batch_fused_kv_cache_kernel`: Batch KV cache update

### 5. Pinned Memory
**文件**: [performance_test.cu](../src/backend/cuda/performance_test.cu)
- `cudaMallocHost` for H2D transfer optimization

### 6. Flash Attention Prefill cuBLAS
**文件**: [full_attention_cuda.cu](../src/backend/cuda/kernels/full_attention_cuda.cu)
- Output projection 使用 cuBLAS `sgemm` 替代手动 kernel

### 7. Linear Attention 消除 cudaMemcpy
**文件**: [linear_attention_cuda.cu](../src/backend/cuda/kernels/linear_attention_cuda.cu)
- 移除 `forward()` 中 3 次 `cudaMemcpyDeviceToDevice`，改用指针偏移直接访问 conv_out 中的 Q/K/V

### 8. MLP Batch cuBLAS GEMM
**文件**: [mlp_cuda.cu](../src/backend/cuda/kernels/mlp_cuda.cu), [fused_kernels.cu](../src/backend/cuda/kernels/fused_kernels.cu)
- Batch 场景使用 cuBLAS `sgemm` 替代 `cublasSgemv`
- 预分配 hidden buffer
- 添加 `silu_mul_batch` kernel

### 9. Batch Linear Attention cuBLAS GEMM (核心优化)
**文件**: [linear_attention_cuda.cu](../src/backend/cuda/kernels/linear_attention_cuda.cu)
- 使用 cuBLAS GEMM 一次性处理 batch 的 QKV/A/B/Z/O projection
- 添加 batch kernel: conv1d_update_fused_batch, l2norm_qk_fused_batch, norm_gate_fused_batch
- 从 batch_size×8 个 kernel 减少到约 9 个 kernel
- 预分配 batch buffer，避免重复 cudaMalloc
- 保持 gated_delta 串行（recurrent state 依赖）
- **效果**: Prefill 吞吐 90.3 → 525.6 tok/s (+482%)
- **精度验证**: batch 输出与串行输出一致 (max diff 9.6e-08)

---

## 待优化项目

### P1: CUDA Graph Prefill (预期 +10-20% Prefill)
**优化方案**: 捕获 prefill graph，消除 kernel launch 开销

### P2: 异步流水线 (预期 +10-15% Prefill)
**优化方案**: 双缓冲实现 H2D 与计算重叠

### P3: FP16/BF16 量化 (预期 +20-30% Decode)
**优化方案**: 权重和激活使用 FP16/BF16，利用 Tensor Core

### P4: Paged KV Cache (预期 -50% 显存)
**优化方案**: 按需分配 KV Cache 页面，支持长上下文

---

## Kernel 文件清单

| 文件 | 功能 | 优化状态 |
|------|------|---------|
| [rmsnorm_cuda.cu](../src/backend/cuda/kernels/rmsnorm_cuda.cu) | RMSNorm | 已优化 |
| [mlp_cuda.cu](../src/backend/cuda/kernels/mlp_cuda.cu) | MLP (cuBLAS GEMM) | 已优化 |
| [full_attention_cuda.cu](../src/backend/cuda/kernels/full_attention_cuda.cu) | Full Attention + Batch Prefill | 已优化 |
| [linear_attention_cuda.cu](../src/backend/cuda/kernels/linear_attention_cuda.cu) | Linear Attention (GatedDeltaNet) | 已优化 |
| [lm_head_cuda.cu](../src/backend/cuda/kernels/lm_head_cuda.cu) | LM Head (cuBLAS GEMM) | 已优化 |
| [token_embedding_cuda.cu](../src/backend/cuda/kernels/token_embedding_cuda.cu) | Token Embedding | 已优化 |
| [fused_kernels.cu](../src/backend/cuda/kernels/fused_kernels.cu) | Fused Kernels | 已优化 |
| [flash_attention.cu](../src/backend/cuda/kernels/flash_attention.cu) | Flash Attention v1 | 备用 |
| [gpu_sampler_argmax.cu](../src/backend/cuda/kernels/gpu_sampler_argmax.cu) | Argmax Sampler | 已优化 |

---

## 性能对比

| 版本 | Prefill 吞吐 (tok/s) | Decode 吞吐 (tok/s) | TTFT (ms) | TPOT (ms) | 主要优化 |
|------|---------------------|---------------------|-----------|-----------|---------|
| v1.0 (CUDA Baseline) | 17.5 | 12.5 | 58,400 | 79.96 | CUDA 基础实现，单 token 串行处理 |
| v2.0 | 86.4 | 15,774 | 11,856 | 0.063 | FlashAttention v2 + Tensor Core + Batch Prefill |
| **v3.0 (当前)** | **525.6** | **16,248** | **1,948** | **0.062** | **Batch Linear Attention + cuBLAS GEMM + Kernel Fusion** |
| **提升(v1→v3)** | **+2,904%** | **+129,884%** | **-97%** | **-99.9%** | |
