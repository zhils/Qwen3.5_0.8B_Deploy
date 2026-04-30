#include "mlp_cuda.hpp"
#include "cuda_error_handling.cuh"
#include "fused_kernels.cuh"
#include "cutlass_gemm_wrapper.cuh"
#include <cmath>
#include <stdexcept>

namespace qwen {
namespace cuda {

// Optimized GEMV kernel using shared memory tiling
// Each block computes a tile of output elements
// TILE_M = number of output elements per block
// TILE_K = number of input elements loaded into shared memory at a time

template <int TILE_M, int TILE_K, int THREADS>
__global__ void mlp_gemv_kernel_optimized(const float* __restrict__ weight,
                                          const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int out_dim, int in_dim) {
    __shared__ float s_input[TILE_K];
    __shared__ float s_weight[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts

    const int row_base = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;

    // Each thread handles one or more output rows
    const int rows_per_thread = (TILE_M + THREADS - 1) / THREADS;
    const int my_row_start = tid * rows_per_thread;
    const int my_row_end = min(my_row_start + rows_per_thread, TILE_M);

    // Initialize accumulators
    float accum[8] = {0.0f};  // Max 8 rows per thread

    // Iterate over input tiles
    for (int k_tile = 0; k_tile < in_dim; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, in_dim);
        int k_len = k_end - k_tile;

        // Load input tile into shared memory (coalesced)
        for (int k = tid; k < k_len; k += THREADS) {
            s_input[k] = input[k_tile + k];
        }

        // Load weight tile into shared memory
        // Each thread loads its assigned rows
        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_dim) {
                const float* w_row = weight + global_row * in_dim + k_tile;
                for (int k = 0; k < k_len; ++k) {
                    s_weight[r][k] = w_row[k];
                }
            }
        }

        __syncthreads();

        // Compute partial dot products
        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_dim) {
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

    // Write results
    for (int r = my_row_start; r < my_row_end; ++r) {
        int global_row = row_base + r;
        if (global_row < out_dim) {
            output[global_row] = accum[r - my_row_start];
        }
    }
}

// Even more optimized: use float4 vectorized loads
// For dimensions that are multiples of 4
template <int TILE_M, int TILE_K, int THREADS>
__global__ void mlp_gemv_kernel_vec4(const float* __restrict__ weight,
                                     const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int out_dim, int in_dim) {
    __shared__ float s_input[TILE_K];
    __shared__ float s_weight[TILE_M][TILE_K + 1];

    const int row_base = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;

    const int rows_per_thread = (TILE_M + THREADS - 1) / THREADS;
    const int my_row_start = tid * rows_per_thread;
    const int my_row_end = min(my_row_start + rows_per_thread, TILE_M);

    float accum[8] = {0.0f};

    for (int k_tile = 0; k_tile < in_dim; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, in_dim);
        int k_len = k_end - k_tile;

        // Vectorized input load
        const float4* input4 = reinterpret_cast<const float4*>(input + k_tile);
        for (int k = tid; k < k_len / 4; k += THREADS) {
            float4 v = input4[k];
            int base = k * 4;
            s_input[base] = v.x;
            s_input[base + 1] = v.y;
            s_input[base + 2] = v.z;
            s_input[base + 3] = v.w;
        }
        // Handle remainder
        for (int k = tid + (k_len / 4) * 4; k < k_len; k += THREADS) {
            s_input[k] = input[k_tile + k];
        }

        // Load weights
        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_dim) {
                const float* w_row = weight + global_row * in_dim + k_tile;
                for (int k = 0; k < k_len; ++k) {
                    s_weight[r][k] = w_row[k];
                }
            }
        }

        __syncthreads();

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_dim) {
                float sum = accum[r - my_row_start];
                #pragma unroll 8
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
        if (global_row < out_dim) {
            output[global_row] = accum[r - my_row_start];
        }
    }
}

// Fallback simple kernel for very small dimensions
__global__ void mlp_gemv_kernel_simple(const float* __restrict__ weight,
                                       const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int out_dim, int in_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < in_dim; ++j) {
        sum += weight[i * in_dim + j] * input[j];
    }
    output[i] = sum;
}

__global__ void mlp_gemv_add_residual_kernel_simple(const float* __restrict__ weight,
                                                    const float* __restrict__ input,
                                                    float* __restrict__ output,
                                                    int out_dim, int in_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < in_dim; ++j) {
        sum += weight[i * in_dim + j] * input[j];
    }
    output[i] = sum + output[i];
}

// Optimized add-residual version
template <int TILE_M, int TILE_K, int THREADS>
__global__ void mlp_gemv_add_residual_kernel_optimized(const float* __restrict__ weight,
                                                       const float* __restrict__ input,
                                                       float* __restrict__ output,
                                                       int out_dim, int in_dim) {
    __shared__ float s_input[TILE_K];
    __shared__ float s_weight[TILE_M][TILE_K + 1];

    const int row_base = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;

    const int rows_per_thread = (TILE_M + THREADS - 1) / THREADS;
    const int my_row_start = tid * rows_per_thread;
    const int my_row_end = min(my_row_start + rows_per_thread, TILE_M);

    float accum[8] = {0.0f};

    for (int k_tile = 0; k_tile < in_dim; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, in_dim);
        int k_len = k_end - k_tile;

        for (int k = tid; k < k_len; k += THREADS) {
            s_input[k] = input[k_tile + k];
        }

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_dim) {
                const float* w_row = weight + global_row * in_dim + k_tile;
                for (int k = 0; k < k_len; ++k) {
                    s_weight[r][k] = w_row[k];
                }
            }
        }

        __syncthreads();

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_dim) {
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
        if (global_row < out_dim) {
            output[global_row] = accum[r - my_row_start] + output[global_row];
        }
    }
}

// Launcher that selects the best kernel based on dimensions
static void launch_mlp_gemv(const float* weight, const float* input, float* output,
                            int out_dim, int in_dim, cudaStream_t stream = 0) {
    // For small dimensions, use simple kernel
    if (out_dim < 128 || in_dim < 128) {
        int block = 256;
        int grid = (out_dim + block - 1) / block;
        mlp_gemv_kernel_simple<<<grid, block, 0, stream>>>(weight, input, output, out_dim, in_dim);
        return;
    }

    // Use CUTLASS-style warp-level GEMV for better performance on Blackwell
    // This uses warp shuffle reduction instead of shared memory atomics
    bool aligned = ((uintptr_t)weight % 16 == 0) &&
                   ((uintptr_t)input % 16 == 0) &&
                   (in_dim % 4 == 0);

    if (aligned) {
        launch_cutlass_gemv(weight, input, output, out_dim, in_dim, stream);
    } else {
        // Fallback to shared memory tiling
        const int TILE_M = 32;
        const int TILE_K = 256;
        const int THREADS = 256;
        int grid = (out_dim + TILE_M - 1) / TILE_M;
        mlp_gemv_kernel_optimized<TILE_M, TILE_K, THREADS>
            <<<grid, THREADS, 0, stream>>>(weight, input, output, out_dim, in_dim);
    }
}

static void launch_mlp_gemv_add_residual(const float* weight, const float* input, float* output,
                                         int out_dim, int in_dim, cudaStream_t stream = 0) {
    if (out_dim < 128 || in_dim < 128) {
        int block = 256;
        int grid = (out_dim + block - 1) / block;
        mlp_gemv_add_residual_kernel_simple<<<grid, block, 0, stream>>>(weight, input, output, out_dim, in_dim);
        return;
    }

    const int TILE_M = 32;
    const int TILE_K = 256;
    const int THREADS = 256;
    int grid = (out_dim + TILE_M - 1) / TILE_M;

    mlp_gemv_add_residual_kernel_optimized<TILE_M, TILE_K, THREADS>
        <<<grid, THREADS, 0, stream>>>(weight, input, output, out_dim, in_dim);
}

CudaMLP::CudaMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      d_gate_proj_weight_(nullptr), d_up_proj_weight_(nullptr), d_down_proj_weight_(nullptr),
      d_hidden_buf_(nullptr), max_hidden_batch_(0) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMalloc(&d_gate_proj_weight_, gate_size * sizeof(float));
    cudaMalloc(&d_up_proj_weight_, gate_size * sizeof(float));
    cudaMalloc(&d_down_proj_weight_, down_size * sizeof(float));
}

CudaMLP::~CudaMLP() {
    if (d_gate_proj_weight_)
        cudaFree(d_gate_proj_weight_);
    if (d_up_proj_weight_)
        cudaFree(d_up_proj_weight_);
    if (d_down_proj_weight_)
        cudaFree(d_down_proj_weight_);
    if (d_hidden_buf_)
        cudaFree(d_hidden_buf_);
}

void CudaMLP::ensure_hidden_buffer(int batch_size) const {
    if (batch_size <= max_hidden_batch_ && d_hidden_buf_ != nullptr) {
        return;
    }
    if (d_hidden_buf_) {
        cudaFree(d_hidden_buf_);
    }
    size_t bytes = static_cast<size_t>(batch_size) * intermediate_size_ * sizeof(float) * 2;
    cudaMalloc(&d_hidden_buf_, bytes);
    max_hidden_batch_ = batch_size;
}

void CudaMLP::set_weights(const std::vector<float>& gate_proj_weight,
                          const std::vector<float>& up_proj_weight,
                          const std::vector<float>& down_proj_weight) {
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMemcpy(d_gate_proj_weight_, gate_proj_weight.data(), gate_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_weight_, up_proj_weight.data(), gate_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_weight_, down_proj_weight.data(), down_size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void CudaMLP::forward(const float* input, float* output, int batch_size) const {
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        launch_mlp_gemv(d_gate_proj_weight_, input, d_gate_buf,
                        intermediate_size_, hidden_size_);
        launch_mlp_gemv(d_up_proj_weight_, input, d_up_buf,
                        intermediate_size_, hidden_size_);

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);

        launch_mlp_gemv(d_down_proj_weight_, d_gate_buf, output,
                        hidden_size_, intermediate_size_);

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        for (int b = 0; b < batch_size; ++b) {
            launch_mlp_gemv(d_gate_proj_weight_,
                            input + b * hidden_size_,
                            d_gate_buf + b * intermediate_size_,
                            intermediate_size_, hidden_size_);
            launch_mlp_gemv(d_up_proj_weight_,
                            input + b * hidden_size_,
                            d_up_buf + b * intermediate_size_,
                            intermediate_size_, hidden_size_);
        }

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        for (int b = 0; b < batch_size; ++b) {
            launch_mlp_gemv(d_down_proj_weight_,
                            d_gate_buf + b * intermediate_size_,
                            output + b * hidden_size_,
                            hidden_size_, intermediate_size_);
        }
    }
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        launch_mlp_gemv(d_gate_proj_weight_, input, d_gate_buf,
                        intermediate_size_, hidden_size_);
        launch_mlp_gemv(d_up_proj_weight_, input, d_up_buf,
                        intermediate_size_, hidden_size_);

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);

        launch_mlp_gemv_add_residual(d_down_proj_weight_, d_gate_buf, residual,
                                     hidden_size_, intermediate_size_);

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        for (int b = 0; b < batch_size; ++b) {
            launch_mlp_gemv(d_gate_proj_weight_,
                            input + b * hidden_size_,
                            d_gate_buf + b * intermediate_size_,
                            intermediate_size_, hidden_size_);
            launch_mlp_gemv(d_up_proj_weight_,
                            input + b * hidden_size_,
                            d_up_buf + b * intermediate_size_,
                            intermediate_size_, hidden_size_);
        }

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        for (int b = 0; b < batch_size; ++b) {
            launch_mlp_gemv_add_residual(d_down_proj_weight_,
                                         d_gate_buf + b * intermediate_size_,
                                         residual + b * hidden_size_,
                                         hidden_size_, intermediate_size_);
        }
    }
}

} // namespace cuda
} // namespace qwen
