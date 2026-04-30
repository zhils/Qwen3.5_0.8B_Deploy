#pragma once

namespace qwen {
namespace cuda {

static __global__ void add_residual_kernel(const float* __restrict__ input,
                                           const float* __restrict__ attn_out,
                                           float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    output[idx] = input[idx] + attn_out[idx];
}

} // namespace cuda
} // namespace qwen
