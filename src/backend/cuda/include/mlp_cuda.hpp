#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdint>

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

    // INT8 native forward (input/output are INT8 with scales)
    void forward_int8(const int8_t* input, int8_t* output, float input_scale, float* output_scale,
                      int batch_size = 1) const;

    int hidden_size() const {
        return hidden_size_;
    }

  private:
    int hidden_size_;
    int intermediate_size_;

    // INT8 weights (per-row scale)
    int8_t* d_gate_proj_weight_;
    int8_t* d_up_proj_weight_;
    int8_t* d_down_proj_weight_;
    
    // Per-row scales for weights
    float* d_gate_proj_scale_;
    float* d_up_proj_scale_;
    float* d_down_proj_scale_;

    // INT8 intermediate buffers
    mutable int8_t* d_hidden_buf_int8_;
    mutable int8_t* d_gate_buf_int8_;
    mutable int8_t* d_up_buf_int8_;
    mutable float* d_gate_buf_scale_;
    mutable float* d_up_buf_scale_;
    mutable int max_hidden_batch_;

    // FP32 buffers for output
    mutable float* d_output_buf_fp32_;
    
    void ensure_buffers(int batch_size) const;
};

} // namespace cuda
} // namespace qwen
