#pragma once

#define OPTIMIZED_BATCH_SIZE 32

#define GEMM_BLOCK_M 128
#define GEMM_BLOCK_N 128
#define GEMM_BLOCK_K 32

#define ATTENTION_BLOCK_SIZE 256
#define MLP_BLOCK_SIZE 512

#define PREFILL_WARP_COUNT 8
#define DECODE_WARP_COUNT 8

#define SHARED_MEMORY_SIZE 65536

#define USE_BATCH_SPECIFIC_KERNELS 1
#define UNROLL_FACTOR 8

namespace qwen {
namespace config {
    constexpr int BATCH_SIZE = OPTIMIZED_BATCH_SIZE;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    constexpr int MAX_SHARED_MEM = SHARED_MEMORY_SIZE;
}
}
