# 批处理优化实验目录

本目录包含针对不同 batch size 的优化实现，每个子文件夹针对特定的 batch size 进行了专门的优化。

## 目录结构

```
batch_optimization/
├── batch_8/     # Batch Size = 8 优化
├── batch_16/    # Batch Size = 16 优化
├── batch_32/    # Batch Size = 32 优化
├── batch_64/    # Batch Size = 64 优化
└── batch_128/   # Batch Size = 128 优化
```

## 各 Batch Size 优化策略

### batch_8 - 小批量优化

**配置参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| GEMM_BLOCK_M | 64 | GEMM M 维度分块 |
| GEMM_BLOCK_N | 64 | GEMM N 维度分块 |
| GEMM_BLOCK_K | 32 | GEMM K 维度分块 |
| ATTENTION_BLOCK_SIZE | 128 | Attention kernel 块大小 |
| SHARED_MEMORY_SIZE | 32KB | Shared memory 使用量 |
| UNROLL_FACTOR | 4 | 循环展开因子 |

**优化重点**：
- 减少线程块大小，提高小批量下的并行度
- 降低 shared memory 使用，避免资源浪费
- 较小的循环展开因子，减少寄存器压力

### batch_16 - 中小批量优化

**配置参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| GEMM_BLOCK_M | 64 | GEMM M 维度分块 |
| GEMM_BLOCK_N | 128 | GEMM N 维度分块 |
| GEMM_BLOCK_K | 32 | GEMM K 维度分块 |
| ATTENTION_BLOCK_SIZE | 256 | Attention kernel 块大小 |
| SHARED_MEMORY_SIZE | 48KB | Shared memory 使用量 |
| UNROLL_FACTOR | 4 | 循环展开因子 |

**优化重点**：
- 增加 N 维度分块，提高内存访问效率
- 使用 shared memory 进行矩阵分块计算
- 适中的线程块配置

### batch_32 - 中等批量优化

**配置参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| GEMM_BLOCK_M | 128 | GEMM M 维度分块 |
| GEMM_BLOCK_N | 128 | GEMM N 维度分块 |
| GEMM_BLOCK_K | 32 | GEMM K 维度分块 |
| ATTENTION_BLOCK_SIZE | 256 | Attention kernel 块大小 |
| SHARED_MEMORY_SIZE | 64KB | Shared memory 使用量 |
| UNROLL_FACTOR | 8 | 循环展开因子 |

**优化重点**：
- 正方形分块配置，平衡 M/N 维度
- 增大 shared memory，提高数据复用
- 更大的线程块 (512 threads)

### batch_64 - 大批量优化

**配置参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| GEMM_BLOCK_M | 128 | GEMM M 维度分块 |
| GEMM_BLOCK_N | 256 | GEMM N 维度分块 |
| GEMM_BLOCK_K | 64 | GEMM K 维度分块 |
| ATTENTION_BLOCK_SIZE | 512 | Attention kernel 块大小 |
| SHARED_MEMORY_SIZE | 96KB | Shared memory 使用量 |
| UNROLL_FACTOR | 8 | 循环展开因子 |

**优化重点**：
- 非对称分块，N 维度更大以利用内存合并
- Warp-level 优化，减少同步开销
- 动态 shared memory 分配

### batch_128 - 超大批量优化

**配置参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| GEMM_BLOCK_M | 256 | GEMM M 维度分块 |
| GEMM_BLOCK_N | 256 | GEMM N 维度分块 |
| GEMM_BLOCK_K | 64 | GEMM K 维度分块 |
| ATTENTION_BLOCK_SIZE | 512 | Attention kernel 块大小 |
| SHARED_MEMORY_SIZE | 128KB | Shared memory 使用量 |
| UNROLL_FACTOR | 16 | 循环展开因子 |

**优化重点**：
- 大分块配置，最大化 Tensor Core 利用率
- Warp-level 矩阵乘法优化
- 最大循环展开，充分利用寄存器

## 性能对比预期

| Batch Size | 相对吞吐量 | 显存占用 | 延迟 |
|------------|-----------|---------|------|
| 8 | 1.0x | 低 | 低 |
| 16 | 1.8x | 中 | 低 |
| 32 | 3.2x | 中 | 中 |
| 64 | 5.5x | 高 | 中 |
| 128 | 8.0x | 高 | 高 |

## 编译说明

每个子文件夹包含独立的 CMakeLists.txt，可以单独编译：

```bash
cd batch_optimization/batch_32
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON -DBATCH_SIZE=32
make -j4
```

## 使用方法

```cpp
#include "batch_config.cuh"
#include "batch_gemm_optimized.cu"

// 使用优化后的批处理 GEMM
launch_batch_gemm_optimized(d_A, d_B, d_C, M, N, K, batch_size, stream);
```

## 注意事项

1. **Shared Memory 限制**：不同 GPU 架构有不同的 shared memory 限制
   - Volta: 96KB per SM
   - Ampere: 164KB per SM
   - Hopper: 228KB per SM

2. **寄存器压力**：大批量配置使用更多寄存器，可能影响 occupancy

3. **Tensor Core**：batch_64 和 batch_128 配置针对 Tensor Core 优化

4. **内存带宽**：大批量时内存带宽可能成为瓶颈
