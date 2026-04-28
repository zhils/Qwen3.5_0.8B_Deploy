#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

namespace qwen {
namespace cuda {

class CudaTokenEmbedding {
  public:
    CudaTokenEmbedding(int vocab_size = 248320, int hidden_size = 1024);
    ~CudaTokenEmbedding();

    void set_weight(const std::vector<float>& weight);
    
    void set_weight_bf16_ptr(__nv_bfloat16* d_weight_bf16);
    
    void set_weight_fp32_ptr(float* d_weight_fp32);

    void forward(int token_id, float* d_output) const;
    void forward(const std::vector<int>& token_ids, float* d_output) const;

    int vocab_size() const {
        return vocab_size_;
    }
    int hidden_size() const {
        return hidden_size_;
    }
    
    float* d_weight_fp32() { return d_weight_fp32_; }
    __nv_bfloat16* d_weight_bf16() { return d_weight_bf16_; }

  private:
    int vocab_size_;
    int hidden_size_;
    float* d_weight_fp32_;
    __nv_bfloat16* d_weight_bf16_;
    bool owns_weight_;
};

} // namespace cuda
} // namespace qwen
