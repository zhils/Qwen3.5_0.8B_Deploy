#pragma once

namespace qwen {
namespace cuda {

class GpuGreedyArgmaxSampler {
  public:
    explicit GpuGreedyArgmaxSampler(int vocab_size);
    ~GpuGreedyArgmaxSampler();

    GpuGreedyArgmaxSampler(const GpuGreedyArgmaxSampler&) = delete;
    GpuGreedyArgmaxSampler& operator=(const GpuGreedyArgmaxSampler&) = delete;

    int sample(const float* d_logits);

  private:
    int vocab_size_;
    int num_blocks_;
    float* d_block_max_;
    int* d_block_idx_;
};

} // namespace cuda
} // namespace qwen
