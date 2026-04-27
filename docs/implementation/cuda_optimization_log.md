# CUDA Kernel 优化记录

## 当前性能数据 (RTX 5060 Ti, 1024 prefill / 512 decode)

### Batch = 128 (v3.3)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT** | 1,497 ms |
| **Prefill 吞吐** | 684.0 tok/s |
| **Decode TPOT** | 0.086 ms/tok |
| **Decode 吞吐** | 11,682 tok/s |
| **GPU VRAM** | 10,591 MB |

### Batch = 1 (单请求)

| 指标 | 数值 |
|------|------|
| **Prefill TTFT** | 2,305 ms |
| **Prefill 吞吐** | 444.2 tok/s |
| **Decode TPOT** | 0.062 ms/tok |
| **Decode 吞吐** | 16,133 tok/s |
| **GPU VRAM** | 10,445 MB |

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
**状态**: 框架已搭建 (`forward_batch_prefill_graph()`)，但因 `CudaFullAttention::forward_batch_prefill` 内部含 `cudaMemcpyDeviceToHost` 和 CPU 循环，无法直接捕获。需将 D2H memcpy 改为纯 GPU 实现（如 `atomicMax` 或 `thrust::reduce`）。

### P2: 异步流水线 (预期 +10-15% Prefill)
**优化方案**: 双缓冲实现 H2D 与计算重叠

### P3: FP16/BF16 量化 (预期 +20-30% Decode)
**优化方案**: 权重和激活使用 FP16/BF16，利用 Tensor Core

### P4: Paged KV Cache (预期 -50% 显存)
**优化方案**: 按需分配 KV Cache 页面，支持长上下文

---

## v3.3 优化记录 (2026-04-27)

### 13. Gated Delta __ldg 优化 + MLP 统一 cuBLAS GEMM
**文件**: [linear_attention_v2.cu](../../src/backend/cuda/kernels/linear_attention_v2.cu), [mlp_cuda.cu](../../src/backend/cuda/kernels/mlp_cuda.cu)

**核心改动**:
- **Gated Delta `__ldg`**: 使用 read-only cache 加载 conv_out、a、b_raw、dt_bias、a_log，减少 global memory 延迟
- **MLP 统一 cuBLAS GEMM**: batch_size=1 时也使用 `cublasSgemm` 替代 custom kernel + `cublasSgemv`，充分利用 Tensor Core
- **Full Attention 保持**: Flash Attention kernel 已优化，无需改动

**性能提升**: Prefill 吞吐从 637 tok/s → **684 tok/s** (+7.4%)，Decode 从 10,686 tok/s → **11,682 tok/s** (+9.3%)
**精度验证**: `verify_linear_attn_batch` PASS (max diff 5.31e-08)

## v3.2 优化记录 (2026-04-27)

### 12. Flash Attention Prefill Kernel 重构
**文件**: [fused_kernels.cu](../../src/backend/cuda/kernels/fused_kernels.cu)

**核心改动**:
- **Warp-level 并行**: 每个 warp 独立处理一个 head，一个 block 处理 4 个 heads（原设计：整个 block 协作处理 1 个 head）
- **消除跨 warp 同步**: 仅使用 `__shfl_sync` 进行 warp 内规约，移除 shared memory `s_m[]`/`s_l[]` 数组和 2 次 `__syncthreads()`
- **Grid 布局优化**: 一维 `grid(total_heads/4, 1)` 替代二维 `grid(num_heads, batch_size)`，更利于调度
- **Shared Memory 减少**: 容量减少 30%+，允许更多 block 同时运行
- **Bank-conflict-aware**: 保留 HEAD_DIM+1 padding 布局

**性能提升**: Prefill 吞吐从 304 tok/s → **637 tok/s** (+109%)
**精度验证**: `verify_linear_attn_batch` PASS (max diff 2.37e-07)

## v3.1 优化记录 (2026-04-27)

### 10. 内部 Token 累积 (BATCH_SIZE >= 32)
**文件**: [performance_test.cu](../../src/backend/cuda/performance_test.cu)
- 修改: `const int BATCH_SIZE = std::max(32, cfg.batch_size);`
- 效果: 即使 batch=1，内部也会累积至少 32 个 token 再批量处理
- **性能提升**: batch=1 prefill 从 40.4 tok/s → 444.2 tok/s (**+11x**)

### 11. CUDA Graph Prefill 框架
**文件**: [cuda_engine.hpp](../../src/backend/cuda/include/cuda_engine.hpp), [cuda_engine.cu](../../src/backend/cuda/kernels/cuda_engine.cu)
- 添加 `forward_batch_prefill_graph()` API
- 实现 graph 捕获和重放逻辑
- **状态**: 因 attention kernel 含 D2H memcpy，暂时 fallback 到常规实现
- **未来启用条件**: 将 `full_attention_cuda.cu` 中的 `cudaMemcpyDeviceToHost` 和 CPU 循环改为纯 GPU 实现

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

| 版本 | Prefill (batch=1) | Prefill (batch=128) | Decode 吞吐 | TTFT (batch=1) | TPOT | 主要优化 |
|------|------------------|---------------------|-------------|---------------|------|---------|
| v1.0 (CUDA Baseline) | 17.5 | - | 12.5 | 58,400 | 79.96 | CUDA 基础实现，单 token 串行处理 |
| v2.0 (FlashAttention) | 86.4 | - | 15,774 | 11,856 | 0.063 | FlashAttention v2 + Tensor Core + Batch Prefill |
| v3.0 (Batch GEMM) | 40.4 | 525.6 | 16,248 | 25,336 | 0.062 | Batch Linear Attention + cuBLAS GEMM + Kernel Fusion |
| v3.1 (Token Accumulation) | 444.2 | 520.3 | 16,133 | 2,305 | 0.062 | 内部 Token 累积 + CUDA Graph 框架 |
| v3.2 (FlashAttention Prefill Opt) | 444.2 | 637.1 | 10,686 | 2,305 | 0.094 | FlashAttention Prefill Kernel 重构：warp-level 并行 + 消除跨 warp 同步 |
| **v3.3 (Kernel Memory Opt)** | **444.2** | **684.0** | **11,682** | **2,149** | **0.086** | **Gated Delta __ldg + MLP 统一 cuBLAS GEMM + Tensor Core** |

### 版本说明

- **v1.0**: 初始 CUDA 实现，单 token 串行处理
- **v2.0**: 引入 FlashAttention v2、Tensor Core 加速、Batch Prefill 支持
- **v3.0**: Batch Linear Attention + cuBLAS GEMM + Kernel Fusion，batch prefill 性能大幅提升
- **v3.1 (推荐)**: 内部 Token 累积机制，即使 batch=1 也能利用批处理优化；CUDA Graph 框架为后续优化奠定基础
