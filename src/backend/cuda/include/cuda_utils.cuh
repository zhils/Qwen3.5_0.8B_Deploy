#pragma once

namespace qwen {
namespace cuda {

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

} // namespace cuda
} // namespace qwen
