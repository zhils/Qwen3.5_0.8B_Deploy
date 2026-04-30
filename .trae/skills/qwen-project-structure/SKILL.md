---
name: "qwen-project-structure"
description: "Qwen3.5-0.8B CUDA推理引擎项目目录结构详解。Invoke when user asks about project structure, file organization, or directory layout."
---

# Qwen3.5-0.8B CUDA推理引擎 - 项目结构详解

## 根目录结构

```
Qwen3.5_0.8B_Deploy/
├── src/                          # 源代码
├── tests/                        # 测试代码
├── docs/                         # 文档
├── scripts/                      # 脚本工具
├── lossy_optimization/           # 有损优化实验（不动）
├── others/                      # 归档内容
├── CMakeLists.txt               # 主构建配置
├── README.md                    # 项目说明
└── README_EN.md                 # 英文说明
```

## src/ 目录结构

```
src/
└── backend/
    ├── cpu/                      # CPU 后端（参考实现）
    │   ├── core/                # 核心计算模块
    │   │   ├── common/          # 公共组件
    │   │   │   ├── language_backbone.cpp/hpp      # 语言主干
    │   │   │   ├── language_common.cpp/hpp        # 公共工具
    │   │   │   └── error_handling.hpp             # 错误处理
    │   │   ├── attention/       # 注意力机制
    │   │   │   ├── language_linear_attn.cpp/hpp  # Linear Attention
    │   │   │   └── language_full_attn.cpp/hpp     # Full Attention
    │   │   ├── embedding/      # 嵌入层
    │   │   │   ├── token_embedding.cpp/hpp         # Token 嵌入
    │   │   │   └── multimodal_embedding.cpp/hpp   # 多模态嵌入
    │   │   ├── heads/          # 输出头
    │   │   │   ├── lm_head.cpp/hpp                # LM Head
    │   │   │   ├── sampler.cpp/hpp                # 采样器
    │   │   │   └── mtp_head.cpp/hpp               # MTP Head
    │   │   └── mlp/            # MLP 层
    │   │       └── language_mlp.cpp/hpp
    │   └── vision/              # 视觉编码器
    │       ├── vision_patch_embedding.cpp/hpp  # Patch Embedding
    │       ├── vision_transformer.cpp/hpp     # ViT
    │       └── qwen35_image_preprocess.cpp/hpp # 图像预处理
    │
    └── cuda/                    # CUDA 后端（主路径）
        ├── include/             # 头文件
        │   ├── cuda_engine.hpp          # CUDA 引擎接口
        │   ├── cuda_engine_v3.hpp       # V3 引擎接口
        │   ├── flash_attention.cuh      # Flash Attention
        │   ├── full_attention_cuda.hpp  # Full Attention
        │   ├── linear_attention_cuda.hpp # Linear Attention
        │   ├── linear_attention_v2.cuh  # V2 优化
        │   ├── mlp_cuda.hpp            # MLP
        │   ├── rmsnorm_cuda.hpp        # RMSNorm
        │   ├── fused_kernels.cuh        # 融合 Kernels
        │   ├── kv_int8_cuda.hpp        # INT8 KV Cache
        │   ├── paged_kv.hpp            # Paged KV
        │   ├── cuda_utils.cuh          # CUDA 工具
        │   ├── cuda_error_handling.cuh  # 错误处理
        │   ├── cuda_ops.cuh            # CUDA 算子
        │   ├── bf16_gemm.cuh           # BF16 GEMM
        │   └── cublas_handle_pool.hpp  # cuBLAS 句柄池
        │
        ├── kernels/             # CUDA Kernel 实现
        │   ├── cuda_engine.cu         # V2 引擎
        │   ├── cuda_engine_v3.cu     # V3 引擎
        │   ├── flash_attention.cu     # Flash Attention
        │   ├── linear_attention_cuda.cu  # Linear Attention
        │   ├── linear_attention_v2.cu    # V2 优化
        │   ├── linear_attention_fused.cu # 融合 Linear Attn
        │   ├── full_attention_cuda.cu    # Full Attention
        │   ├── fused_kernels.cu         # 融合 Kernels
        │   ├── mlp_cuda.cu               # MLP
        │   ├── rmsnorm_cuda.cu           # RMSNorm
        │   ├── lm_head_cuda.cu           # LM Head
        │   ├── token_embedding_cuda.cu  # Token Embedding
        │   ├── vision_patch_embedding_cuda.cu # Vision Patch
        │   ├── kv_int8_cuda.cu          # INT8 KV Cache
        │   ├── paged_kv.cpp             # Paged KV
        │   ├── gpu_sampler_argmax.cu    # 采样器
        │   ├── fa_wrapper.cpp           # FA 封装
        │   ├── fa_set_weights.cu        # FA 权重设置
        │   └── nvcc_smoke_test.cu       # NVCC 测试
        │
        ├── tools/               # 工具
        │   ├── gpu_memory_profiler.cpp/hpp  # GPU 内存分析
        │   └── cuda_op_flowchart.hpp       # CUDA 流程图
        │
        ├── performance_test.cu        # 性能测试 V1
        ├── performance_test_v3.cu     # 性能测试 V3
        ├── performance_benchmark.cpp   # 基准测试
        ├── memory_analysis.cu         # 内存分析
        ├── verify_linear_attn_batch.cu # Linear Attn 验证
        └── WEIGHT_FORMAT.md           # 权重格式说明
```

## tests/ 目录结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_rmsnorm.cpp      # RMSNorm 测试
│   ├── test_rmsnorm.cu       # RMSNorm CUDA 测试
│   ├── test_flash_attention.cu  # Flash Attention 测试
│   ├── test_linear_attn.cpp    # Linear Attention 测试
│   ├── test_full_attn.cpp      # Full Attention 测试
│   ├── test_kvcache.cpp        # KV Cache 测试
│   └── test_backbone.cpp       # 主干测试
│
├── integration/              # 集成测试
│   ├── e2e_inference_test.cpp # 端到端推理测试
│   ├── v2_kernel_accuracy_validate.cu # Kernel 精度验证
│   ├── cpu_benchmark.cpp       # CPU 基准测试
│   └── stage2_benchmark.cpp    # Stage2 基准测试
│
└── tools/                    # 工具
    ├── cuda_op_flowchart_demo.cpp
    └── cuda_pipeline_flowchart_demo.cpp
```

## docs/ 目录结构

```
docs/
├── README.md                         # 文档索引
├── implementation/                    # 实现文档
│   ├── CODE_ARCHITECTURE.md          # 代码架构
│   ├── cuda_optimization_log.md      # CUDA 优化日志
│   ├── cuda_prefill_flow_v32.md     # V3.2 Prefill 流程
│   ├── prefill_optimization_strategy.md # Prefill 优化策略
│   ├── qwen3_5_0_8b_architecture.md  # 模型架构
│   ├── accuracy_validation_methodology.md # 精度验证方法论
│   └── WEIGHT_FORMAT.md              # 权重格式
│
├── qwen3_5_0_8b_details/             # 模型细节
│   ├── 00_总览与索引.md
│   ├── 01_总体架构_实现级流程.md
│   ├── 02_文本主干_24层调度与参数映射.md
│   ├── 03_LinearAttention_GatedDeltaNet_细节.md
│   ├── 04_FullAttention_GQA_RoPE_细节.md
│   ├── 05_MLP_层归一化_输出头.md
│   ├── 06_视觉编码器与Merger_细节.md
│   └── 07_MTP分支与推测解码_参数加载.md
│
└── memory_optimization/              # 内存优化
    └── memory_consumption_analysis.md # 内存消耗分析
```

## lossy_optimization/ 目录

有损优化实验目录，按精度损失程度分层：

```
lossy_optimization/
├── 01_weight_bf16/      # BF16 权重（最小损失）
├── 02_kv_cache_fp16/   # FP16 KV Cache
├── 03_kv_cache_int8/   # INT8 KV Cache
├── 04_weight_int8/     # INT8 权重
├── 05_weight_int4/     # INT4 权重（较大损失）
├── 06_full_fp16/       # 全 FP16
├── 07_weight_only/     # 仅权重压缩
├── 08_flash_to_linear/ # Flash→Linear 切换
└── 09_linear_to_flash/ # Linear→Flash 切换
```

## others/ 目录

归档内容，包含：

```
others/
├── batch_optimization/       # Batch 优化实验
│   ├── batch_8/
│   ├── batch_16/
│   ├── batch_32/
│   ├── batch_64/
│   └── batch_128/
├── archived_tests/           # 归档测试
├── archived_docs/            # 归档文档
├── archived_scripts/         # 归档脚本
├── archived_benchmarks/       # 归档基准测试
├── weights/                  # 权重导出脚本
├── interview/                # 面试相关
└── *.md                      # 散落文档
```

## CMakeLists.txt 构建结构

```cmake
# 核心库
add_library(qwen_core STATIC ${CORE_SOURCES} ${VISION_SOURCES})

# CUDA 库 (ENABLE_CUDA=ON 时)
add_library(qwen_cuda STATIC ${CUDA_KERNEL_SOURCES})

# 可执行文件
- performance_test      # V1 性能测试
- performance_test_v3   # V3 性能测试
- memory_analysis       # 内存分析
- verify_linear_attn_batch  # Linear Attn 验证
- v2_kernel_accuracy_validate # Kernel 精度验证
```
