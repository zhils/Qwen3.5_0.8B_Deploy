#include "../include/gpu_sampler_argmax.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace qwen {
namespace cuda {

namespace {

constexpr int kBlockSize = 256;

__global__ void block_argmax_kernel(const float* logits, int n, float* block_max, int* block_idx) {
    __shared__ float s_vals[kBlockSize];
    __shared__ int s_idx[kBlockSize];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float v = -1e30f;
    int idx = -1;
    if (gid < n) {
        v = logits[gid];
        idx = gid;
    }
    s_vals[tid] = v;
    s_idx[tid] = idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_vals[tid + stride] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = s_vals[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}

void check_cuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

} // namespace

GpuGreedyArgmaxSampler::GpuGreedyArgmaxSampler(int vocab_size)
    : vocab_size_(vocab_size), num_blocks_((vocab_size + kBlockSize - 1) / kBlockSize),
      d_block_max_(nullptr), d_block_idx_(nullptr) {
    check_cuda(cudaMalloc(&d_block_max_, static_cast<size_t>(num_blocks_) * sizeof(float)),
               "cudaMalloc d_block_max");
    check_cuda(cudaMalloc(&d_block_idx_, static_cast<size_t>(num_blocks_) * sizeof(int)),
               "cudaMalloc d_block_idx");
}

GpuGreedyArgmaxSampler::~GpuGreedyArgmaxSampler() {
    if (d_block_max_)
        cudaFree(d_block_max_);
    if (d_block_idx_)
        cudaFree(d_block_idx_);
}

int GpuGreedyArgmaxSampler::sample(const float* d_logits) {
    block_argmax_kernel<<<num_blocks_, kBlockSize>>>(d_logits, vocab_size_, d_block_max_,
                                                     d_block_idx_);
    check_cuda(cudaGetLastError(), "block_argmax_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "block_argmax_kernel sync");

    std::vector<float> h_max(num_blocks_);
    std::vector<int> h_idx(num_blocks_);
    check_cuda(cudaMemcpy(h_max.data(), d_block_max_,
                          static_cast<size_t>(num_blocks_) * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy h_max");
    check_cuda(cudaMemcpy(h_idx.data(), d_block_idx_,
                          static_cast<size_t>(num_blocks_) * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy h_idx");

    float best = -1e30f;
    int best_idx = 0;
    for (int i = 0; i < num_blocks_; ++i) {
        if (h_max[i] > best) {
            best = h_max[i];
            best_idx = h_idx[i];
        }
    }
    return best_idx;
}

} // namespace cuda
} // namespace qwen
