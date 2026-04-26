#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

namespace qwen {
namespace cuda {

class CudaMLP {
  public:
    CudaMLP(int hidden_size = 1024, int intermediate_size = 3584);
    ~CudaMLP();

    void set_weights(const std::vector<float>& gate_proj_weight,
                     const std::vector<float>& up_proj_weight,
                     const std::vector<float>& down_proj_weight);

    void forward(const float* input, float* output, int batch_size = 1) const;

    void forward_add_residual(const float* input, float* residual, int batch_size = 1) const;

    int hidden_size() const {
        return hidden_size_;
    }

  private:
    int hidden_size_;
    int intermediate_size_;

    float* d_gate_proj_weight_;
    float* d_up_proj_weight_;
    float* d_down_proj_weight_;

    cublasHandle_t cublas_handle_;
};

} // namespace cuda
} // namespace qwen
