#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>

namespace qwen {
namespace cuda {

class CudaLMHead {
  public:
    CudaLMHead(int hidden_size = 1024, int vocab_size = 248320);
    ~CudaLMHead();

    void set_weight(const std::vector<float>& weight);

    void forward(const float* input, float* output) const;

    int hidden_size() const {
        return hidden_size_;
    }
    int vocab_size() const {
        return vocab_size_;
    }

  private:
    int hidden_size_;
    int vocab_size_;
    __nv_bfloat16* d_weight_bf16_;
    __nv_bfloat16* d_input_bf16_;
    __nv_bfloat16* d_output_bf16_;
    mutable cublasHandle_t cublas_handle_;
    bool weight_converted_;
};

} // namespace cuda
} // namespace qwen
