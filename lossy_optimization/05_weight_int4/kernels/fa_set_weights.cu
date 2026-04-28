#include "full_attention_cuda.hpp"
#include <cuda_runtime.h>

namespace qwen {
namespace cuda {

void set_fa_weights(CudaFullAttention* fa, const std::vector<float>& q_w,
                    const std::vector<float>& k_w, const std::vector<float>& v_w,
                    const std::vector<float>& qn_w, const std::vector<float>& kn_w,
                    const std::vector<float>& o_w, int layer_idx) {
    (void)layer_idx;
    fa->set_weights(q_w, k_w, v_w, qn_w, kn_w, o_w);
}

} // namespace cuda
} // namespace qwen
