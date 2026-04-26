#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace qwen {
namespace cuda {

class CudaTokenEmbedding {
  public:
    CudaTokenEmbedding(int vocab_size = 248320, int hidden_size = 1024);
    ~CudaTokenEmbedding();

    void set_weight(const std::vector<float>& weight);

    void forward(int token_id, float* d_output) const;
    void forward(const std::vector<int>& token_ids, float* d_output) const;

    int vocab_size() const {
        return vocab_size_;
    }
    int hidden_size() const {
        return hidden_size_;
    }

  private:
    int vocab_size_;
    int hidden_size_;
    float* d_weight_;
};

} // namespace cuda
} // namespace qwen
