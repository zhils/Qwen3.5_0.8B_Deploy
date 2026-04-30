#include "lm_head_cuda.hpp"
#include "cuda_error_handling.cuh"
#include "cutlass_gemm_wrapper.cuh"
#include <stdexcept>
#include <string>
#include <cuda_bf16.h>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (call);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) +      \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                                          \
    } while (0)

namespace qwen {
namespace cuda {

static __global__ void fp32_to_bf16_kernel(const float* __restrict__ fp32_data,
                                    __nv_bfloat16* __restrict__ bf16_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    bf16_data[i] = __float2bfloat16(fp32_data[i]);
}

static __global__ void bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ bf16_data,
                                    float* __restrict__ fp32_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    fp32_data[i] = __bfloat162float(bf16_data[i]);
}

// Optimized LM Head GEMV using shared memory tiling
// Vocab size is typically very large (e.g., 151936), hidden_size ~ 1024-2048
template <int TILE_M, int TILE_K, int THREADS>
__global__ void lm_head_gemv_kernel_optimized(const __nv_bfloat16* __restrict__ weight,
                                              const __nv_bfloat16* __restrict__ input,
                                              float* __restrict__ output,
                                              int vocab_size, int hidden_size) {
    __shared__ float s_input[TILE_K];
    __shared__ float s_weight[TILE_M][TILE_K + 1];

    const int row_base = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;

    const int rows_per_thread = (TILE_M + THREADS - 1) / THREADS;
    const int my_row_start = tid * rows_per_thread;
    const int my_row_end = min(my_row_start + rows_per_thread, TILE_M);

    float accum[8] = {0.0f};

    for (int k_tile = 0; k_tile < hidden_size; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, hidden_size);
        int k_len = k_end - k_tile;

        // Load input into shared memory (convert bf16 to float)
        for (int k = tid; k < k_len; k += THREADS) {
            s_input[k] = __bfloat162float(input[k_tile + k]);
        }

        // Load weight tile
        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < vocab_size) {
                const __nv_bfloat16* w_row = weight + global_row * hidden_size + k_tile;
                for (int k = 0; k < k_len; ++k) {
                    s_weight[r][k] = __bfloat162float(w_row[k]);
                }
            }
        }

        __syncthreads();

        // Compute
        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < vocab_size) {
                float sum = accum[r - my_row_start];
                #pragma unroll 4
                for (int k = 0; k < k_len; ++k) {
                    sum += s_weight[r][k] * s_input[k];
                }
                accum[r - my_row_start] = sum;
            }
        }

        __syncthreads();
    }

    for (int r = my_row_start; r < my_row_end; ++r) {
        int global_row = row_base + r;
        if (global_row < vocab_size) {
            output[global_row] = accum[r - my_row_start];
        }
    }
}

// Simple fallback kernel
__global__ void lm_head_gemv_kernel_simple(const __nv_bfloat16* __restrict__ weight,
                                           const __nv_bfloat16* __restrict__ input,
                                           float* __restrict__ output,
                                           int vocab_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < hidden_size; ++j) {
        sum += __bfloat162float(weight[i * hidden_size + j]) * __bfloat162float(input[j]);
    }
    output[i] = sum;
}

static void launch_lm_head_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* input,
                                float* output, int vocab_size, int hidden_size,
                                cudaStream_t stream = 0) {
    // For small vocab, use simple kernel
    if (vocab_size < 128 || hidden_size < 128) {
        int block = 256;
        int grid = (vocab_size + block - 1) / block;
        lm_head_gemv_kernel_simple<<<grid, block, 0, stream>>>(weight, input, output, vocab_size, hidden_size);
        return;
    }

    // Use CUTLASS-style warp-level GEMV for BF16
    bool aligned = ((uintptr_t)weight % 16 == 0) &&
                   ((uintptr_t)input % 16 == 0) &&
                   (hidden_size % 4 == 0);

    if (aligned) {
        launch_cutlass_gemv_bf16(weight, input, output, vocab_size, hidden_size, stream);
    } else {
        // Fallback to shared memory tiling
        const int TILE_M = 32;
        const int TILE_K = 256;
        const int THREADS = 256;
        int grid = (vocab_size + TILE_M - 1) / TILE_M;
        lm_head_gemv_kernel_optimized<TILE_M, TILE_K, THREADS>
            <<<grid, THREADS, 0, stream>>>(weight, input, output, vocab_size, hidden_size);
    }
}

CudaLMHead::CudaLMHead(int hidden_size, int vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size),
      d_weight_bf16_(nullptr), d_input_bf16_(nullptr), d_output_bf16_(nullptr),
      owns_weight_(false), weight_set_(false) {
    CUDA_CHECK(cudaMalloc(&d_input_bf16_, hidden_size_ * sizeof(__nv_bfloat16)));
}

CudaLMHead::~CudaLMHead() {
    if (owns_weight_ && d_weight_bf16_) {
        cudaFree(d_weight_bf16_);
    }
    if (d_input_bf16_)
        cudaFree(d_input_bf16_);
    if (d_output_bf16_)
        cudaFree(d_output_bf16_);
}

void CudaLMHead::set_weight(const std::vector<float>& weight) {
    size_t weight_size = static_cast<size_t>(vocab_size_) * hidden_size_;
    if (weight.size() != weight_size) {
        throw std::invalid_argument("CudaLMHead weight size mismatch: expected " +
                                    std::to_string(weight_size) + ", got " +
                                    std::to_string(weight.size()));
    }

    if (owns_weight_ && d_weight_bf16_) {
        cudaFree(d_weight_bf16_);
    }

    CUDA_CHECK(cudaMalloc(&d_weight_bf16_, weight_size * sizeof(__nv_bfloat16)));

    std::vector<__nv_bfloat16> h_weight_bf16(weight_size);
    for (size_t i = 0; i < weight_size; ++i) {
        h_weight_bf16[i] = __float2bfloat16(weight[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_weight_bf16_, h_weight_bf16.data(), 
                          weight_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    owns_weight_ = true;
    weight_set_ = true;
}

void CudaLMHead::set_weight_bf16_ptr(__nv_bfloat16* d_weight_bf16) {
    if (owns_weight_ && d_weight_bf16_) {
        cudaFree(d_weight_bf16_);
    }
    d_weight_bf16_ = d_weight_bf16;
    owns_weight_ = false;
    weight_set_ = true;
}

void CudaLMHead::forward(const float* input, float* output) const {
    if (!weight_set_ || !d_weight_bf16_) {
        throw std::runtime_error("CudaLMHead: weight not set");
    }

    dim3 block(256);
    dim3 grid_in((hidden_size_ + 255) / 256);
    fp32_to_bf16_kernel<<<grid_in, block>>>(input, d_input_bf16_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());

    launch_lm_head_gemv(d_weight_bf16_, d_input_bf16_, output, vocab_size_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace qwen
