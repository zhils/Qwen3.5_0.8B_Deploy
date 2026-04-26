# Qwen3.5-0.8B CUDA Inference Engine

[English](README_EN.md) | 中文

## 项目概述

基于 Qwen3.5-0.8B 模型的高性能 CUDA 推理引擎，采用多种优化技术实现比 llama.cpp 更快的端到端推理速度。

### 核心性能指标

| 测试条件 | Prefill TTFT | Decode TPOT | 端到端耗时 |
|---------|-------------|-------------|-----------|
| prefill=1024, decode=512 | **227 ms** | **6.24 ms/tok** | **3.42 秒** |

### 性能对比 (vs llama.cpp)

| 指标 | 本项目 (CUDA) | llama.cpp (BF16) | 优势 |
|------|--------------|------------------|------|
| Prefill 吞吐量 | 4532 tok/s | 193 tok/s | **23.5x** |
| Decode 吞吐量 | 161 tok/s | 192 tok/s | -16% |
| 端到端耗时 | 3.42s | 7.97s | **2.33x** |

> 本项目 Prefill 阶段比 llama.cpp 快 23 倍，主要得益于 cuBLAS GEMM 的高度优化实现。

## 项目结构

```
.
├── src/
│   ├── backend/
│   │   ├── cpu/              # CPU后端实现
│   │   │   ├── core/         # 核心计算模块
│   │   │   │   ├── attention/     # Full Attention + Linear Attention
│   │   │   │   ├── common/        # 公共组件
│   │   │   │   ├── embedding/     # Token + 多模态嵌入
│   │   │   │   ├── heads/         # LM Head + MTP Head + Sampler
│   │   │   │   └── mlp/           # MLP层
│   │   │   └── vision/       # 视觉编码器
│   │   └── cuda/             # CUDA后端实现
│   │       ├── include/      # CUDA头文件
│   │       │   ├── cuda_engine.hpp           # CUDA引擎接口
│   │       │   ├── flash_attention.cuh       # FlashAttention声明
│   │       │   ├── full_attention_cuda.hpp   # Full Attention声明
│   │       │   ├── linear_attention_cuda.hpp # Linear Attention声明
│   │       │   └── ...
│   │       └── kernels/      # CUDA kernel实现
│   │           ├── cuda_engine.cu            # 引擎主控
│   │           ├── flash_attention.cu        # FlashAttention实现
│   │           ├── full_attention_cuda.cu    # Full Attention实现
│   │           ├── linear_attention_cuda.cu  # Linear Attention实现
│   │           ├── mlp_cuda.cu              # MLP实现
│   │           ├── rmsnorm_cuda.cu          # RMSNorm实现
│   │           └── ...
│   └── ...
├── tests/
│   ├── unit/                # 单元测试
│   ├── integration/         # 集成测试
│   └── benchmarks/          # 性能测试
├── docs/
│   ├── implementation/      # 实现文档
│   ├── qwen3_5_0_8b_details/ # 模型架构详解
│   └── PERFORMANCE_REPORT.md # 详细性能报告
├── scripts/
│   ├── benchmark/           # 基准测试脚本
│   ├── debug/               # 调试脚本
│   ├── validation/          # 验证脚本
│   └── weights/             # 权重导出脚本
├── third_party/             # 第三方库
├── CMakeLists.txt
└── README.md
```

## 技术亮点

### 1. FlashAttention 实现
- **v3.0**: 支持任意 head_dim (256)，tiling 策略优化
- Warp-level 并行归约，减少同步开销
- Online softmax 减少 HBM 访问

### 2. Kernel 融合
| 融合Kernel | 功能 |
|-----------|------|
| `save_rmsnorm_kernel` | Save残差 + RMSNorm |
| `attn_add_rmsnorm_fused_kernel` | Attention输出 + Add + RMSNorm |
| `silu_mul_fused_kernel` | SiLU激活 + 乘法 |
| `linear_layer_fused_kernel` | 双RMSNorm + 残差更新 |

### 3. cuBLAS 优化
- **Decode阶段**: 使用 SGEMV 替代 GEMM，减少 Launch 开销
- **Prefill阶段**: cuBLAS GEMM 利用 Tensor Core 加速
- **Batch GEMM**: K+V 投影合并为一次调用

### 4. CUDA Graph
- Decode 阶段启用 CUDA Graph，减少 Kernel Launch 开销
- 静态图捕获，多次执行零开销

## 快速开始

### 编译

```bash
# 使用 CMake
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_CUDA_ARCHITECTURES=80
ninja

# 或使用 nvcc 直接编译
nvcc -allow-unsupported-compiler -arch=sm_80 src/backend/cuda/kernels/*.cu main.cpp -o qwen_infer -lcublas -lcudart
```

### 运行

```bash
# 端到端推理测试
./qwen_infer <weights_dir> <prompt_len> <gen_len> <repetitions>

# 示例: prefill=1024, decode=512, 测试3次
./qwen_infer weights 1024 512 3
```

## 模型配置

| 参数 | 值 |
|------|-----|
| Hidden Size | 1024 |
| Intermediate Size | 3584 |
| Num Layers | 24 (18 Linear + 6 Full Attention) |
| Num Heads | 8, Head Dim = 256 |
| Num KV Heads | 2, KV Head Dim = 256 |
| Vocab Size | 248320 |

## 依赖

- CUDA Toolkit 12.0+
- cuBLAS
- CMake 3.20+
- C++17

## 性能分析

### GPU: NVIDIA GeForce RTX 5060 Ti

| 指标 | 值 |
|------|-----|
| 理论显存带宽 | 448 GB/s |
| Prefill 带宽利用率 | 55.6 GB/s (12.4%) |
| Decode 带宽利用率 | 293.9 GB/s (65.6%) |

### 瓶颈分析

- **Prefill**: Compute-bound (cuBLAS GEMM)
- **Decode**: Memory-bound (权重读取)

## 进一步优化方向

1. **FP16/BF16 量化**: 显存减半，decode 速度提升 20-30%
2. **INT8/INT4 量化**: 进一步降低带宽压力
3. **Paged KV Cache**: 支持更长上下文
4. **Continuous Batching**: 提升批量推理吞吐

## 许可

MIT License

## 参考

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Qwen3.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
