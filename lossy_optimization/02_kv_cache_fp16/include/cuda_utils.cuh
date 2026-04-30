#pragma once

namespace qwen {
namespace cuda {

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <int BLOCK_SIZE>
__device__ inline float blockReduceSum(float val) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    s_data[tid] = val;
    __syncthreads();

#pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    return s_data[0];
}

} // namespace cuda
} // namespace qwen
