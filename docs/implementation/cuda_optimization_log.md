# CUDA Kernel 优化记录

## 当前性能数据 (RTX 5060 Ti, 1024 prefill / 512 decode)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT** | 11,856 ms |
| **Prefill 吞吐** | 86.4 tok/s |
| **Decode TPOT** | 0.063 ms/tok |
| **Decode 吞吐** | 15,774 tok/s |
| **GPU VRAM** | 8,833 MB |

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
**文件**: [fused_kernels.cu](../src/backend/cuda/kernels/fused_kernels.cu)
- Gate + SiLU + Mul fusion
- RMSNorm + Residual fusion
- Conv1D + State update fusion
- L2 norm Q + K fusion
- Norm + Gate fusion

### 4. Batch Prefill
**文件**: [full_attention_cuda.cu](../src/backend/cuda/kernels/full_attention_cuda.cu)
- `batch_fused_q_path_kernel`: Batch Q projection + RoPE
- `batch_fused_kv_cache_kernel`: Batch KV cache update

### 5. Pinned Memory
**文件**: [performance_test.cu](../src/backend/cuda/performance_test.cu)
- `cudaMallocHost` for H2D transfer optimization

---

## 待优化项目

### P0: Batch Embedding Lookup (预期 +50-100% Prefill)
**当前问题**: 每个 token 单独调用 embedding lookup
**优化方案**: 批量 embedding lookup，消除 D2H 传输

### P1: 增大 Batch Size (预期 +30-50% Prefill)
**当前**: BATCH_SIZE = 64
**优化方案**: BATCH_SIZE = 256 或更大

### P2: Flash Attention Prefill cuBLAS (预期 +20-30% Prefill)
**当前**: 手动矩阵乘法
**优化方案**: 使用 cuBLAS GEMM 进行批量矩阵乘法

### P3: CUDA Graph Prefill (预期 +10-20% Prefill)
**优化方案**: 捕获 prefill graph，消除 kernel launch 开销

### P4: 异步流水线 (预期 +10-15% Prefill)
**优化方案**: 双缓冲实现 H2D 与计算重叠

---

## Kernel 文件清单

| 文件 | 功能 | 优化状态 |
|------|------|---------|
| [rmsnorm_cuda.cu](../src/backend/cuda/kernels/rmsnorm_cuda.cu) | RMSNorm | 已优化 |
| [mlp_cuda.cu](../src/backend/cuda/kernels/mlp_cuda.cu) | MLP (cuBLAS GEMM) | 已优化 |
| [full_attention_cuda.cu](../src/backend/cuda/kernels/full_attention_cuda.cu) | Full Attention + Batch Prefill | 已优化 |
| [linear_attention_cuda.cu](../src/backend/cuda/kernels/linear_attention_cuda.cu) | Linear Attention (GatedDeltaNet) | 已优化 |
| [lm_head_cuda.cu](../src/backend/cuda/kernels/lm_head_cuda.cu) | LM Head (cuBLAS GEMM) | 已优化 |
| [token_embedding_cuda.cu](../src/backend/cuda/kernels/token_embedding_cuda.cu) | Token Embedding | 待优化 |
| [fused_kernels.cu](../src/backend/cuda/kernels/fused_kernels.cu) | Fused Kernels | 已优化 |
| [flash_attention.cu](../src/backend/cuda/kernels/flash_attention.cu) | Flash Attention v1 | 备用 |
| [gpu_sampler_argmax.cu](../src/backend/cuda/kernels/gpu_sampler_argmax.cu) | Argmax Sampler | 已优化 |

---

## 性能对比

| 版本 | Prefill 吞吐 (tok/s) | Decode 吞吐 (tok/s) | TTFT (ms) | TPOT (ms) |
|------|---------------------|---------------------|-----------|-----------|
| v1.0 | 17.5 | 12.5 | 58,400 | 79.96 |
| **当前** | **86.4** | **15,774** | **11,856** | **0.063** |
| **提升** | **+394%** | **+126,190%** | **-80%** | **-99.9%** |
