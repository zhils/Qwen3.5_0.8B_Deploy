#include "linear_attention_cuda.hpp"
#include "cuda_utils.cuh"
#include "cuda_error_handling.cuh"
#include <cmath>

namespace qwen {
namespace cuda {

void CudaLinearAttention::forward_fused(const float* input, float* output,
                                        CudaLinearAttnState& state) const {
    forward(input, output, state);
}

} // namespace cuda
} // namespace qwen
