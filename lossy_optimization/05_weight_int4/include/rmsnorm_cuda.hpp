#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace qwen {
namespace cuda {

class CudaRMSNorm {
  public:
    CudaRMSNorm(int hidden_size = 1024, float eps = 1e-6f);
    ~CudaRMSNorm();

    void set_weights(const std::vector<float>& weight);

    void forward(const float* input, float* output, int batch_size = 1) const;

    void forward_with_residual(const float* residual_in, const float* attn_out, float* residual_out,
                               float* normed_out, int batch_size = 1) const;

    int hidden_size() const {
        return hidden_size_;
    }

  private:
    int hidden_size_;
    float eps_;
    float* d_weight_;
};

} // namespace cuda
} // namespace qwen
