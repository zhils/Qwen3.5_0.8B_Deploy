#include "linear_attention_cuda.hpp"
#include "cuda_utils.cuh"
#include "cuda_error_handling.cuh"
#include "cutlass_gemm_wrapper.cuh"
#include <cmath>
#include <cstdio>

namespace qwen {
namespace cuda {

// Optimized linear projection using shared memory tiling
template <int TILE_M, int TILE_K, int THREADS>
__global__ void linear_proj_kernel_optimized(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             const float* __restrict__ weight, int hidden_size,
                                             int out_size) {
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

        for (int k = tid; k < k_len; k += THREADS) {
            s_input[k] = input[k_tile + k];
        }

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_size) {
                const float* w_row = weight + global_row * hidden_size + k_tile;
                for (int k = 0; k < k_len; ++k) {
                    s_weight[r][k] = w_row[k];
                }
            }
        }

        __syncthreads();

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < out_size) {
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
        if (global_row < out_size) {
            output[global_row] = accum[r - my_row_start];
        }
    }
}

// Simple fallback
__global__ void linear_proj_kernel_simple(const float* __restrict__ input, float* __restrict__ output,
                                          const float* __restrict__ weight, int hidden_size,
                                          int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < hidden_size; ++j) {
        sum += weight[idx * hidden_size + j] * input[j];
    }
    output[idx] = sum;
}

static void launch_linear_proj(const float* input, float* output, const float* weight,
                               int hidden_size, int out_size, cudaStream_t stream = 0) {
    if (out_size < 128 || hidden_size < 128) {
        int block = 256;
        int grid = (out_size + block - 1) / block;
        linear_proj_kernel_simple<<<grid, block, 0, stream>>>(input, output, weight, hidden_size, out_size);
        return;
    }

    // Use CUTLASS-style warp-level GEMV
    bool aligned = ((uintptr_t)weight % 16 == 0) &&
                   ((uintptr_t)input % 16 == 0) &&
                   (hidden_size % 4 == 0);

    if (aligned) {
        launch_cutlass_gemv(weight, input, output, out_size, hidden_size, stream);
    } else {
        // Fallback to shared memory tiling
        const int TILE_M = 32;
        const int TILE_K = 256;
        const int THREADS = 256;
        int grid = (out_size + TILE_M - 1) / TILE_M;
        linear_proj_kernel_optimized<TILE_M, TILE_K, THREADS>
            <<<grid, THREADS, 0, stream>>>(input, output, weight, hidden_size, out_size);
    }
}

// Keep old name for backward compatibility - device function wrapper
__device__ void linear_proj_kernel_device(const float* __restrict__ input, float* __restrict__ output,
                                   const float* __restrict__ weight, int hidden_size,
                                   int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < hidden_size; ++j) {
        sum += weight[idx * hidden_size + j] * input[j];
    }
    output[idx] = sum;
}

__global__ void linear_proj_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   const float* __restrict__ weight, int hidden_size,
                                   int out_size) {
    linear_proj_kernel_device(input, output, weight, hidden_size, out_size);
}

// Optimized conv1d kernel using shared memory for weight caching
// Each block processes a tile of channels, caching weights in shared memory
// This reduces global memory reads when conv_kernel is small (typical: 4)
template <int MAX_KERNEL_SIZE = 8>
__global__ void conv1d_kernel_optimized(const float* __restrict__ mixed_qkv,
                                        float* __restrict__ conv_out,
                                        const float* __restrict__ conv_weight,
                                        const float* __restrict__ conv_state,
                                        int conv_dim, int conv_kernel) {
    extern __shared__ float s_weights[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int d_start = blockIdx.x * block_size;
    const int d = d_start + tid;

    // Preload weights for this block into shared memory
    // Each thread loads its own channel's weights
    if (d < conv_dim) {
        const float* w_ptr = conv_weight + d * conv_kernel;
        float* s_w = s_weights + tid * MAX_KERNEL_SIZE;
        #pragma unroll
        for (int k = 0; k < conv_kernel; ++k) {
            s_w[k] = w_ptr[k];
        }
    }
    __syncthreads();

    if (d >= conv_dim) return;

    const float* s_w = s_weights + tid * MAX_KERNEL_SIZE;
    const float* state_ptr = conv_state + d * (conv_kernel - 1);

    float sum = s_w[conv_kernel - 1] * mixed_qkv[d];
    #pragma unroll
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += s_w[k] * state_ptr[k];
    }
    conv_out[d] = silu(sum);
}

// Simple fallback kernel for large kernel sizes
__global__ void conv1d_kernel(const float* __restrict__ mixed_qkv, float* __restrict__ conv_out,
                              const float* __restrict__ conv_weight,
                              const float* __restrict__ conv_state, int conv_dim, int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;

    float sum = conv_weight[d * conv_kernel + (conv_kernel - 1)] * mixed_qkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += conv_weight[d * conv_kernel + k] * conv_state[d * (conv_kernel - 1) + k];
    }
    conv_out[d] = silu(sum);
}

__global__ void update_conv_state_kernel(const float* __restrict__ mixed_qkv,
                                         float* __restrict__ conv_state, int conv_dim,
                                         int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;

    for (int k = (conv_kernel - 2); k > 0; --k) {
        conv_state[d * (conv_kernel - 1) + k] = conv_state[d * (conv_kernel - 1) + k - 1];
    }
    conv_state[d * (conv_kernel - 1)] = mixed_qkv[d];
}

// Optimized gated_delta kernel with better shared memory usage
// Caches k, q vectors in shared memory to reduce global memory reads
// Uses warp shuffle for partial reductions where applicable
template <int MAX_KEY_DIM = 64>
__global__ void gated_delta_kernel_optimized(const float* __restrict__ k, const float* __restrict__ v,
                                              const float* __restrict__ q, const float* __restrict__ a,
                                              const float* __restrict__ b, const float* __restrict__ a_log,
                                              const float* __restrict__ dt_bias,
                                              float* __restrict__ recurrent_state,
                                              float* __restrict__ attn_out, int num_heads, int key_dim,
                                              int value_dim) {
    int head_idx = blockIdx.x;
    int tidx = threadIdx.x;
    if (head_idx >= num_heads || tidx >= value_dim)
        return;

    extern __shared__ float smem[];
    float* s_params = smem;           // 2 floats: sigmoid(b), g_t
    float* s_k = smem + 2;            // key_dim floats: cached k vector
    float* s_q = s_k + key_dim;       // key_dim floats: cached q vector
    float* s_kv_mem = s_q + key_dim;  // value_dim floats: kv accumulator
    float* s_delta = s_kv_mem + value_dim; // value_dim floats: delta

    int state_base = head_idx * key_dim * value_dim;
    const int k_offset = head_idx * key_dim;

    // Thread 0 computes gating parameters and caches k, q vectors
    if (tidx == 0) {
        s_params[0] = 1.0f / (1.0f + expf(-b[head_idx])); // sigmoid(b)
        float sp = a[head_idx] + dt_bias[head_idx];
        sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));  // softplus
        s_params[1] = expf(-expf(a_log[head_idx]) * sp); // g_t

        // Cache k and q vectors in shared memory
        const float* k_ptr = k + k_offset;
        const float* q_ptr = q + k_offset;
        #pragma unroll 4
        for (int kd = 0; kd < key_dim; ++kd) {
            s_k[kd] = k_ptr[kd];
            s_q[kd] = q_ptr[kd];
        }
    }
    __syncthreads();

    float g_t = s_params[1];

    // Apply gating to recurrent state
    #pragma unroll 4
    for (int kd = 0; kd < key_dim; ++kd) {
        recurrent_state[state_base + kd * value_dim + tidx] *= g_t;
    }
    __syncthreads();

    // Compute kv = recurrent_state @ k
    float kv = 0.0f;
    #pragma unroll 4
    for (int kd = 0; kd < key_dim; ++kd) {
        kv += recurrent_state[state_base + kd * value_dim + tidx] * s_k[kd];
    }
    s_kv_mem[tidx] = kv;
    __syncthreads();

    // Compute delta
    float delta = (v[head_idx * value_dim + tidx] - s_kv_mem[tidx]) * s_params[0];
    s_delta[tidx] = delta;
    __syncthreads();

    // Update recurrent state: state += k^T @ delta
    #pragma unroll 4
    for (int kd = 0; kd < key_dim; ++kd) {
        recurrent_state[state_base + kd * value_dim + tidx] += s_k[kd] * delta;
    }
    __syncthreads();

    // Compute output: attn_out = recurrent_state @ q
    float sum = 0.0f;
    #pragma unroll 4
    for (int kd = 0; kd < key_dim; ++kd) {
        sum += recurrent_state[state_base + kd * value_dim + tidx] * s_q[kd];
    }
    attn_out[head_idx * value_dim + tidx] = sum;
}

// Original kernel for fallback
__global__ void gated_delta_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                   const float* __restrict__ q, const float* __restrict__ a,
                                   const float* __restrict__ b, const float* __restrict__ a_log,
                                   const float* __restrict__ dt_bias,
                                   float* __restrict__ recurrent_state,
                                   float* __restrict__ attn_out, int num_heads, int key_dim,
                                   int value_dim) {
    int head_idx = blockIdx.x;
    int tidx = threadIdx.x;
    if (head_idx >= num_heads || tidx >= value_dim)
        return;

    extern __shared__ float smem[];
    float* s_params = smem;
    float* s_kv_mem = smem + 2;
    float* s_delta = s_kv_mem + value_dim;

    int state_base = head_idx * key_dim * value_dim;

    if (tidx == 0) {
        s_params[0] = 1.0f / (1.0f + expf(-b[head_idx])); // sigmoid(b)
        float sp = a[head_idx] + dt_bias[head_idx];
        sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));  // softplus
        s_params[1] = expf(-expf(a_log[head_idx]) * sp); // g_t = exp(-exp(A_log) * softplus(...))
    }
    __syncthreads();

    float g_t = s_params[1];

    for (int kd = 0; kd < key_dim; ++kd) {
        recurrent_state[state_base + kd * value_dim + tidx] *= g_t;
    }
    __syncthreads();

    s_kv_mem[tidx] = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        s_kv_mem[tidx] +=
            recurrent_state[state_base + kd * value_dim + tidx] * k[head_idx * key_dim + kd];
    }
    __syncthreads();

    s_delta[tidx] = (v[head_idx * value_dim + tidx] - s_kv_mem[tidx]) * s_params[0];
    __syncthreads();

    for (int kd = 0; kd < key_dim; ++kd) {
        recurrent_state[state_base + kd * value_dim + tidx] +=
            k[head_idx * key_dim + kd] * s_delta[tidx];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        sum += recurrent_state[state_base + kd * value_dim + tidx] * q[head_idx * key_dim + kd];
    }
    attn_out[head_idx * value_dim + tidx] = sum;
}

__global__ void l2_normalize_heads_kernel(float* __restrict__ data, int num_heads, int head_dim,
                                          float scale, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    float l2 = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float v = data[h * head_dim + d];
        l2 += v * v;
    }
    l2 = sqrtf(l2 + eps);
    for (int d = 0; d < head_dim; ++d) {
        data[h * head_dim + d] = data[h * head_dim + d] / l2 * scale;
    }
}

__global__ void group_rms_norm_kernel(float* __restrict__ data,
                                      const float* __restrict__ norm_weight, int num_heads,
                                      int head_dim, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;

    float variance = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float v = data[h * head_dim + d];
        variance += v * v;
    }
    variance /= head_dim;
    float inv_rms = 1.0f / sqrtf(variance + eps);
    for (int d = 0; d < head_dim; ++d) {
        data[h * head_dim + d] *= inv_rms * norm_weight[d];
    }
}

__global__ void silu_gate_kernel(float* __restrict__ data, const float* __restrict__ z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    float zv = z[idx];
    float s = zv / (1.0f + expf(-zv));
    data[idx] *= s;
}

// Optimized output projection using shared memory tiling
template <int TILE_M, int TILE_K, int THREADS>
__global__ void la_output_proj_kernel_optimized(const float* __restrict__ attn_out,
                                                float* __restrict__ output,
                                                const float* __restrict__ out_weight, int input_dim,
                                                int output_dim) {
    __shared__ float s_attn[TILE_K];
    __shared__ float s_weight[TILE_M][TILE_K + 1];

    const int row_base = blockIdx.x * TILE_M;
    const int tid = threadIdx.x;

    const int rows_per_thread = (TILE_M + THREADS - 1) / THREADS;
    const int my_row_start = tid * rows_per_thread;
    const int my_row_end = min(my_row_start + rows_per_thread, TILE_M);

    float accum[8] = {0.0f};

    for (int k_tile = 0; k_tile < input_dim; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, input_dim);
        int k_len = k_end - k_tile;

        for (int k = tid; k < k_len; k += THREADS) {
            s_attn[k] = attn_out[k_tile + k];
        }

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < output_dim) {
                const float* w_row = out_weight + global_row * input_dim + k_tile;
                for (int k = 0; k < k_len; ++k) {
                    s_weight[r][k] = w_row[k];
                }
            }
        }

        __syncthreads();

        for (int r = my_row_start; r < my_row_end; ++r) {
            int global_row = row_base + r;
            if (global_row < output_dim) {
                float sum = accum[r - my_row_start];
                #pragma unroll 4
                for (int k = 0; k < k_len; ++k) {
                    sum += s_weight[r][k] * s_attn[k];
                }
                accum[r - my_row_start] = sum;
            }
        }

        __syncthreads();
    }

    for (int r = my_row_start; r < my_row_end; ++r) {
        int global_row = row_base + r;
        if (global_row < output_dim) {
            output[global_row] = accum[r - my_row_start];
        }
    }
}

// Simple fallback
__global__ void la_output_proj_kernel_simple(const float* __restrict__ attn_out,
                                             float* __restrict__ output,
                                             const float* __restrict__ out_weight, int input_dim,
                                             int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_dim) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < input_dim; ++j) {
        sum += out_weight[idx * input_dim + j] * attn_out[j];
    }
    output[idx] = sum;
}

static void launch_la_output_proj(const float* attn_out, float* output, const float* out_weight,
                                  int input_dim, int output_dim, cudaStream_t stream = 0) {
    if (output_dim < 128 || input_dim < 128) {
        int block = 256;
        int grid = (output_dim + block - 1) / block;
        la_output_proj_kernel_simple<<<grid, block, 0, stream>>>(attn_out, output, out_weight, input_dim, output_dim);
        return;
    }

    // Use CUTLASS-style warp-level GEMV
    bool aligned = ((uintptr_t)out_weight % 16 == 0) &&
                   ((uintptr_t)attn_out % 16 == 0) &&
                   (input_dim % 4 == 0);

    if (aligned) {
        launch_cutlass_gemv(out_weight, attn_out, output, output_dim, input_dim, stream);
    } else {
        // Fallback to shared memory tiling
        const int TILE_M = 32;
        const int TILE_K = 256;
        const int THREADS = 256;
        int grid = (output_dim + TILE_M - 1) / TILE_M;
        la_output_proj_kernel_optimized<TILE_M, TILE_K, THREADS>
            <<<grid, THREADS, 0, stream>>>(attn_out, output, out_weight, input_dim, output_dim);
    }
}

// Keep old name for backward compatibility - device function wrapper
__device__ void la_output_proj_kernel_device(const float* __restrict__ attn_out, float* __restrict__ output,
                                      const float* __restrict__ out_weight, int input_dim,
                                      int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_dim) return;
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < input_dim; ++j) {
        sum += out_weight[idx * input_dim + j] * attn_out[j];
    }
    output[idx] = sum;
}

__global__ void la_output_proj_kernel(const float* __restrict__ attn_out, float* __restrict__ output,
                                      const float* __restrict__ out_weight, int input_dim,
                                      int output_dim) {
    la_output_proj_kernel_device(attn_out, output, out_weight, input_dim, output_dim);
}

__global__ void conv1d_update_fused_kernel(const float* __restrict__ mixed_qkv,
                                           float* __restrict__ conv_out,
                                           float* __restrict__ conv_state,
                                           const float* __restrict__ conv_weight, int conv_dim,
                                           int conv_kernel) {
    extern __shared__ float s_conv_weights[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int d_start = blockIdx.x * block_size;
    const int d = d_start + tid;

    // Cache weights in shared memory for small kernels (typical: 4)
    if (conv_kernel <= 8 && d < conv_dim) {
        const float* w_ptr = conv_weight + d * conv_kernel;
        float* s_w = s_conv_weights + tid * 8;  // Assume MAX_KERNEL_SIZE = 8
        #pragma unroll
        for (int k = 0; k < conv_kernel; ++k) {
            s_w[k] = w_ptr[k];
        }
    }
    __syncthreads();

    if (d >= conv_dim) return;

    float sum;
    if (conv_kernel <= 8) {
        const float* s_w = s_conv_weights + tid * 8;
        sum = s_w[conv_kernel - 1] * mixed_qkv[d];
        #pragma unroll
        for (int k = 0; k < conv_kernel - 1; ++k) {
            sum += s_w[k] * conv_state[d * (conv_kernel - 1) + k];
        }
    } else {
        sum = conv_weight[d * conv_kernel + (conv_kernel - 1)] * mixed_qkv[d];
        for (int k = 0; k < conv_kernel - 1; ++k) {
            sum += conv_weight[d * conv_kernel + k] * conv_state[d * (conv_kernel - 1) + k];
        }
    }
    conv_out[d] = silu(sum);
    for (int k = (conv_kernel - 2); k > 0; --k) {
        conv_state[d * (conv_kernel - 1) + k] = conv_state[d * (conv_kernel - 1) + k - 1];
    }
    conv_state[d * (conv_kernel - 1)] = mixed_qkv[d];
}

__global__ void l2norm_qk_fused_kernel(float* __restrict__ d_q, float* __restrict__ d_k,
                                       int num_heads, int key_dim, float q_scale, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    float l2q = 0.0f, l2k = 0.0f;
    for (int d = 0; d < key_dim; ++d) {
        float vq = d_q[h * key_dim + d];
        float vk = d_k[h * key_dim + d];
        l2q += vq * vq;
        l2k += vk * vk;
    }
    l2q = sqrtf(l2q + eps);
    l2k = sqrtf(l2k + eps);
    for (int d = 0; d < key_dim; ++d) {
        d_q[h * key_dim + d] = d_q[h * key_dim + d] / l2q * q_scale;
        d_k[h * key_dim + d] /= l2k;
    }
}

__global__ void norm_gate_fused_kernel(float* __restrict__ data,
                                       const float* __restrict__ norm_weight,
                                       const float* __restrict__ z, int num_heads, int value_dim,
                                       float eps) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= num_heads)
        return;

    extern __shared__ float smem[];
    if (tid == 0) {
        float variance = 0.0f;
        for (int d = 0; d < value_dim; ++d) {
            float v = data[h * value_dim + d];
            variance += v * v;
        }
        variance /= value_dim;
        float inv_rms = 1.0f / sqrtf(variance + eps);
        smem[0] = inv_rms;
    }
    __syncthreads();
    float inv_rms = smem[0];

    for (int d = tid; d < value_dim; d += blockDim.x) {
        float normed = data[h * value_dim + d] * inv_rms * norm_weight[d];
        float zv = z[h * value_dim + d];
        float s = zv / (1.0f + expf(-zv));
        data[h * value_dim + d] = normed * s;
    }
}

// Batch version kernels
__global__ void conv1d_update_fused_batch_kernel(const float* __restrict__ mixed_qkv,
                                                  float* __restrict__ conv_out,
                                                  float* __restrict__ conv_state,
                                                  const float* __restrict__ conv_weight,
                                                  int conv_dim, int conv_kernel, int batch_size) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= conv_dim || b >= batch_size) return;

    const float* mqkv = mixed_qkv + b * conv_dim;
    float* cout = conv_out + b * conv_dim;
    float* cstate = conv_state + b * conv_dim * (conv_kernel - 1);
    float sum = conv_weight[d * conv_kernel + (conv_kernel - 1)] * mqkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += conv_weight[d * conv_kernel + k] * cstate[d * (conv_kernel - 1) + k];
    }
    cout[d] = silu(sum);
    for (int k = (conv_kernel - 2); k > 0; --k) {
        cstate[d * (conv_kernel - 1) + k] = cstate[d * (conv_kernel - 1) + k - 1];
    }
    cstate[d * (conv_kernel - 1)] = mqkv[d];
}

__global__ void l2norm_qk_fused_batch_kernel(float* __restrict__ d_q, float* __restrict__ d_k,
                                              int num_heads, int key_dim, float q_scale, float eps,
                                              int batch_size) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= num_heads || b >= batch_size) return;

    float l2q = 0.0f, l2k = 0.0f;
    int q_offset = (b * num_heads + h) * key_dim;
    for (int d = 0; d < key_dim; ++d) {
        float vq = d_q[q_offset + d];
        float vk = d_k[q_offset + d];
        l2q += vq * vq;
        l2k += vk * vk;
    }
    l2q = sqrtf(l2q + eps);
    l2k = sqrtf(l2k + eps);
    for (int d = 0; d < key_dim; ++d) {
        d_q[q_offset + d] = d_q[q_offset + d] / l2q * q_scale;
        d_k[q_offset + d] /= l2k;
    }
}

__global__ void norm_gate_fused_batch_kernel(float* __restrict__ data,
                                              const float* __restrict__ norm_weight,
                                              const float* __restrict__ z, int num_heads,
                                              int value_dim, float eps, int batch_size) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= num_heads || b >= batch_size) return;

    extern __shared__ float smem[];
    int offset = (b * num_heads + h) * value_dim;
    if (tid == 0) {
        float variance = 0.0f;
        for (int d = 0; d < value_dim; ++d) {
            float v = data[offset + d];
            variance += v * v;
        }
        variance /= value_dim;
        smem[0] = 1.0f / sqrtf(variance + eps);
    }
    __syncthreads();
    float inv_rms = smem[0];

    for (int d = tid; d < value_dim; d += blockDim.x) {
        float normed = data[offset + d] * inv_rms * norm_weight[d];
        float zv = z[offset + d];
        float s = zv / (1.0f + expf(-zv));
        data[offset + d] = normed * s;
    }
}

void CudaLinearAttnState::reset(int nh, int kd, int vd, int conv_k) {
    num_heads = nh;
    key_dim = kd;
    value_dim = vd;
    conv_kernel = conv_k;
    int qkv_per_head = kd * 2 + vd;
    conv_dim = nh * qkv_per_head;

    size_t rec_size = static_cast<size_t>(nh) * kd * vd;
    size_t conv_size = static_cast<size_t>(conv_dim) * (conv_k - 1);

    cudaMalloc(&d_recurrent_state, rec_size * sizeof(float));
    cudaMalloc(&d_conv_state, conv_size * sizeof(float));

    cudaMemset(d_recurrent_state, 0, rec_size * sizeof(float));
    cudaMemset(d_conv_state, 0, conv_size * sizeof(float));
}

void CudaLinearAttnState::clear() {
    size_t rec_size = static_cast<size_t>(num_heads) * key_dim * value_dim;
    size_t conv_size = static_cast<size_t>(conv_dim) * (conv_kernel - 1);

    cudaMemset(d_recurrent_state, 0, rec_size * sizeof(float));
    cudaMemset(d_conv_state, 0, conv_size * sizeof(float));
}

CudaLinearAttention::CudaLinearAttention(int hidden_size, int num_heads, int key_dim, int value_dim,
                                         int conv_kernel)
    : hidden_size_(hidden_size), num_heads_(num_heads), key_dim_(key_dim), value_dim_(value_dim),
      conv_kernel_(conv_kernel), d_batch_mixed_qkv_buf_(nullptr),
      d_batch_conv_out_buf_(nullptr), d_batch_a_buf_(nullptr), d_batch_b_raw_buf_(nullptr),
      d_batch_z_buf_(nullptr), d_batch_attn_out_buf_(nullptr), max_batch_size_(0) {
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    cudaMalloc(&d_in_proj_qkv_weight_,
               static_cast<size_t>(conv_dim) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_in_proj_a_weight_,
               static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_in_proj_b_weight_,
               static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_in_proj_z_weight_, static_cast<size_t>(z_dim) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_conv1d_weight_, static_cast<size_t>(conv_dim) * conv_kernel_ * sizeof(float));
    cudaMalloc(&d_out_proj_weight_, static_cast<size_t>(hidden_size_) * z_dim * sizeof(float));
    cudaMalloc(&d_a_log_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_dt_bias_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_norm_weight_, static_cast<size_t>(value_dim_) * sizeof(float));
    cudaMemset(d_norm_weight_, 0, static_cast<size_t>(value_dim_) * sizeof(float));

    cudaMalloc(&d_mixed_qkv_buf_, static_cast<size_t>(conv_dim) * sizeof(float));
    cudaMalloc(&d_conv_out_buf_, static_cast<size_t>(conv_dim) * sizeof(float));
    cudaMalloc(&d_q_buf_, static_cast<size_t>(num_heads_) * key_dim_ * sizeof(float));
    cudaMalloc(&d_k_buf_, static_cast<size_t>(k_dim) * sizeof(float));
    cudaMalloc(&d_v_buf_, static_cast<size_t>(v_dim) * sizeof(float));
    cudaMalloc(&d_a_buf_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_b_raw_buf_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_attn_out_buf_, static_cast<size_t>(z_dim) * sizeof(float));
    cudaMalloc(&d_z_buf_, static_cast<size_t>(z_dim) * sizeof(float));
}

CudaLinearAttention::~CudaLinearAttention() {
    if (d_in_proj_qkv_weight_)
        cudaFree(d_in_proj_qkv_weight_);
    if (d_in_proj_a_weight_)
        cudaFree(d_in_proj_a_weight_);
    if (d_in_proj_b_weight_)
        cudaFree(d_in_proj_b_weight_);
    if (d_in_proj_z_weight_)
        cudaFree(d_in_proj_z_weight_);
    if (d_conv1d_weight_)
        cudaFree(d_conv1d_weight_);
    if (d_out_proj_weight_)
        cudaFree(d_out_proj_weight_);
    if (d_a_log_)
        cudaFree(d_a_log_);
    if (d_dt_bias_)
        cudaFree(d_dt_bias_);
    if (d_norm_weight_)
        cudaFree(d_norm_weight_);

    if (d_mixed_qkv_buf_)
        cudaFree(d_mixed_qkv_buf_);
    if (d_conv_out_buf_)
        cudaFree(d_conv_out_buf_);
    if (d_q_buf_)
        cudaFree(d_q_buf_);
    if (d_k_buf_)
        cudaFree(d_k_buf_);
    if (d_v_buf_)
        cudaFree(d_v_buf_);
    if (d_a_buf_)
        cudaFree(d_a_buf_);
    if (d_b_raw_buf_)
        cudaFree(d_b_raw_buf_);
    if (d_attn_out_buf_)
        cudaFree(d_attn_out_buf_);
    if (d_z_buf_)
        cudaFree(d_z_buf_);

    if (d_batch_mixed_qkv_buf_)
        cudaFree(d_batch_mixed_qkv_buf_);
    if (d_batch_conv_out_buf_)
        cudaFree(d_batch_conv_out_buf_);
    if (d_batch_a_buf_)
        cudaFree(d_batch_a_buf_);
    if (d_batch_b_raw_buf_)
        cudaFree(d_batch_b_raw_buf_);
    if (d_batch_z_buf_)
        cudaFree(d_batch_z_buf_);
    if (d_batch_attn_out_buf_)
        cudaFree(d_batch_attn_out_buf_);
}

void CudaLinearAttention::ensure_batch_buffers(int batch_size) const {
    if (batch_size <= max_batch_size_ && d_batch_mixed_qkv_buf_ != nullptr) {
        return;
    }
    if (d_batch_mixed_qkv_buf_)
        cudaFree(d_batch_mixed_qkv_buf_);
    if (d_batch_conv_out_buf_)
        cudaFree(d_batch_conv_out_buf_);
    if (d_batch_a_buf_)
        cudaFree(d_batch_a_buf_);
    if (d_batch_b_raw_buf_)
        cudaFree(d_batch_b_raw_buf_);
    if (d_batch_z_buf_)
        cudaFree(d_batch_z_buf_);
    if (d_batch_attn_out_buf_)
        cudaFree(d_batch_attn_out_buf_);

    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    cudaMalloc(&d_batch_mixed_qkv_buf_, static_cast<size_t>(batch_size) * conv_dim * sizeof(float));
    cudaMalloc(&d_batch_conv_out_buf_, static_cast<size_t>(batch_size) * conv_dim * sizeof(float));
    cudaMalloc(&d_batch_a_buf_, static_cast<size_t>(batch_size) * num_heads_ * sizeof(float));
    cudaMalloc(&d_batch_b_raw_buf_, static_cast<size_t>(batch_size) * num_heads_ * sizeof(float));
    cudaMalloc(&d_batch_z_buf_, static_cast<size_t>(batch_size) * z_dim * sizeof(float));
    cudaMalloc(&d_batch_attn_out_buf_, static_cast<size_t>(batch_size) * z_dim * sizeof(float));
    max_batch_size_ = batch_size;
}

void CudaLinearAttention::set_weights(
    const std::vector<float>& in_proj_qkv_weight, const std::vector<float>& in_proj_a_weight,
    const std::vector<float>& in_proj_b_weight, const std::vector<float>& in_proj_z_weight,
    const std::vector<float>& conv1d_weight, const std::vector<float>& out_proj_weight,
    const std::vector<float>& a_log, const std::vector<float>& dt_bias,
    const std::vector<float>& norm_weight) {
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    cudaMemcpy(d_in_proj_qkv_weight_, in_proj_qkv_weight.data(),
               static_cast<size_t>(conv_dim) * hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_proj_a_weight_, in_proj_a_weight.data(),
               static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_proj_b_weight_, in_proj_b_weight.data(),
               static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_proj_z_weight_, in_proj_z_weight.data(),
               static_cast<size_t>(z_dim) * hidden_size_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1d_weight_, conv1d_weight.data(),
               static_cast<size_t>(conv_dim) * conv_kernel_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_proj_weight_, out_proj_weight.data(),
               static_cast<size_t>(hidden_size_) * z_dim * sizeof(float), cudaMemcpyHostToDevice);
    if (!a_log.empty()) {
        cudaMemcpy(d_a_log_, a_log.data(), static_cast<size_t>(num_heads_) * sizeof(float),
                   cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_a_log_, 0, static_cast<size_t>(num_heads_) * sizeof(float));
    }
    if (!dt_bias.empty()) {
        cudaMemcpy(d_dt_bias_, dt_bias.data(), static_cast<size_t>(num_heads_) * sizeof(float),
                   cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_dt_bias_, 0, static_cast<size_t>(num_heads_) * sizeof(float));
    }
    if (!norm_weight.empty()) {
        cudaMemcpy(d_norm_weight_, norm_weight.data(),
                   static_cast<size_t>(value_dim_) * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        std::vector<float> ones(value_dim_, 1.0f);
        cudaMemcpy(d_norm_weight_, ones.data(), static_cast<size_t>(value_dim_) * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
}

void CudaLinearAttention::forward(const float* input, float* output,
                                  CudaLinearAttnState& state) const {
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    float* d_mixed_qkv = d_mixed_qkv_buf_;
    float* d_conv_out = d_conv_out_buf_;
    // Q, K, V pointers directly into conv_out buffer (no memcpy needed)
    // d_q = d_conv_out, d_k = d_conv_out + num_heads * key_dim, d_v = d_conv_out + num_heads * key_dim + k_dim
    float* d_a = d_a_buf_;
    float* d_b_raw = d_b_raw_buf_;

    launch_linear_proj(input, d_mixed_qkv, d_in_proj_qkv_weight_, hidden_size_, conv_dim);
    CUDA_CHECK_LAST_KERNEL();

    // Opt-5: Fused conv1d + state update — 1 kernel instead of 2
    // Use optimized conv1d with shared memory weight caching for small kernels
    dim3 block(256);
    dim3 conv_grid((conv_dim + 255) / 256);
    if (conv_kernel_ <= 8) {
        size_t shmem = 256 * 8 * sizeof(float);  // 256 threads * 8 floats per thread
        conv1d_update_fused_kernel<<<conv_grid, block, shmem>>>(d_mixed_qkv, d_conv_out, state.d_conv_state,
                                                         d_conv1d_weight_, conv_dim, conv_kernel_);
    } else {
        conv1d_update_fused_kernel<<<conv_grid, block>>>(d_mixed_qkv, d_conv_out, state.d_conv_state,
                                                         d_conv1d_weight_, conv_dim, conv_kernel_);
    }
    CUDA_CHECK_LAST_KERNEL();

    // Extract Q, K, V directly from conv_out without memcpy
    // d_q = d_conv_out[0 : num_heads * key_dim]
    // d_k = d_conv_out[num_heads * key_dim : num_heads * key_dim + k_dim]
    // d_v = d_conv_out[num_heads * key_dim + k_dim : end]

    // Opt-5: Fused L2 norm Q+K — 1 kernel instead of 2
    // Pass d_conv_out directly as q and k pointers with offsets
    const float l2norm_eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    l2norm_qk_fused_kernel<<<num_heads_, 1>>>(d_conv_out, d_conv_out + num_heads_ * key_dim_,
                                               num_heads_, key_dim_, q_scale, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    launch_linear_proj(input, d_a, d_in_proj_a_weight_, hidden_size_, num_heads_);
    CUDA_CHECK_LAST_KERNEL();
    launch_linear_proj(input, d_b_raw, d_in_proj_b_weight_, hidden_size_, num_heads_);
    CUDA_CHECK_LAST_KERNEL();

    launch_linear_proj(input, d_z_buf_, d_in_proj_z_weight_, hidden_size_, z_dim);
    CUDA_CHECK_LAST_KERNEL();

    float* d_attn_out = d_attn_out_buf_;

    dim3 update_grid(num_heads_);
    // Use optimized kernel with cached k/q vectors for small key_dim
    if (key_dim_ <= 64) {
        size_t shared_mem = (2 + 2 * key_dim_ + 2 * value_dim_) * sizeof(float);
        gated_delta_kernel_optimized<<<update_grid, value_dim_, shared_mem>>>(
            d_conv_out + num_heads_ * key_dim_,
            d_conv_out + num_heads_ * key_dim_ + k_dim,
            d_conv_out,
            d_a, d_b_raw, d_a_log_, d_dt_bias_, state.d_recurrent_state, d_attn_out,
            num_heads_, key_dim_, value_dim_);
    } else {
        size_t shared_mem = (2 + 2 * value_dim_) * sizeof(float);
        gated_delta_kernel<<<update_grid, value_dim_, shared_mem>>>(
            d_conv_out + num_heads_ * key_dim_,
            d_conv_out + num_heads_ * key_dim_ + k_dim,
            d_conv_out,
            d_a, d_b_raw, d_a_log_, d_dt_bias_, state.d_recurrent_state, d_attn_out,
            num_heads_, key_dim_, value_dim_);
    }
    CUDA_CHECK_LAST_KERNEL();

    // Opt-5: Fused norm + gate — 1 kernel instead of 2
    size_t norm_gate_shmem = sizeof(float);
    norm_gate_fused_kernel<<<num_heads_, 128, norm_gate_shmem>>>(
        d_attn_out, d_norm_weight_, d_z_buf_, num_heads_, value_dim_, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    launch_la_output_proj(d_attn_out, output, d_out_proj_weight_, z_dim, hidden_size_);
    CUDA_CHECK_LAST_KERNEL();
}

void CudaLinearAttention::forward_batch(const float* input, float* output,
                                        CudaLinearAttnState& state, int batch_size) const {
    if (batch_size == 1) {
        forward(input, output, state);
        return;
    }

    ensure_batch_buffers(batch_size);

    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    float* d_mixed_qkv = d_batch_mixed_qkv_buf_;
    float* d_conv_out = d_batch_conv_out_buf_;
    float* d_a = d_batch_a_buf_;
    float* d_b_raw = d_batch_b_raw_buf_;
    float* d_z = d_batch_z_buf_;
    float* d_attn_out = d_batch_attn_out_buf_;

    float* d_batch_conv_state = nullptr;
    size_t conv_state_per_token = static_cast<size_t>(conv_dim) * (conv_kernel_ - 1);
    cudaMalloc(&d_batch_conv_state, conv_state_per_token * batch_size * sizeof(float));
    cudaMemcpy(d_batch_conv_state, state.d_conv_state,
               conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice);
    for (int b = 1; b < batch_size; ++b) {
        cudaMemcpy(d_batch_conv_state + b * conv_state_per_token, state.d_conv_state,
                   conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    for (int b = 0; b < batch_size; ++b) {
        launch_linear_proj(input + b * hidden_size_, d_mixed_qkv + b * conv_dim,
                           d_in_proj_qkv_weight_, hidden_size_, conv_dim);
    }
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        launch_linear_proj(input + b * hidden_size_, d_a + b * num_heads_,
                           d_in_proj_a_weight_, hidden_size_, num_heads_);
        launch_linear_proj(input + b * hidden_size_, d_b_raw + b * num_heads_,
                           d_in_proj_b_weight_, hidden_size_, num_heads_);
    }
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        launch_linear_proj(input + b * hidden_size_, d_z + b * z_dim,
                           d_in_proj_z_weight_, hidden_size_, z_dim);
    }
    CUDA_CHECK_LAST_KERNEL();

    dim3 conv_block(256);
    dim3 conv_grid((conv_dim + 255) / 256, batch_size);
    conv1d_update_fused_batch_kernel<<<conv_grid, conv_block>>>(
        d_mixed_qkv, d_conv_out, d_batch_conv_state, d_conv1d_weight_,
        conv_dim, conv_kernel_, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    dim3 l2_grid(num_heads_, batch_size);
    const float l2norm_eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    l2norm_qk_fused_batch_kernel<<<l2_grid, 1>>>(
        d_conv_out, d_conv_out + num_heads_ * key_dim_,
        num_heads_, key_dim_, q_scale, l2norm_eps, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        dim3 update_grid(num_heads_);
        if (key_dim_ <= 64) {
            size_t shared_mem = (2 + 2 * key_dim_ + 2 * value_dim_) * sizeof(float);
            gated_delta_kernel_optimized<<<update_grid, value_dim_, shared_mem>>>(
                d_conv_out + b * conv_dim + num_heads_ * key_dim_,
                d_conv_out + b * conv_dim + num_heads_ * key_dim_ + k_dim,
                d_conv_out + b * conv_dim,
                d_a + b * num_heads_,
                d_b_raw + b * num_heads_,
                d_a_log_, d_dt_bias_, state.d_recurrent_state,
                d_attn_out + b * z_dim,
                num_heads_, key_dim_, value_dim_);
        } else {
            size_t shared_mem = (2 + 2 * value_dim_) * sizeof(float);
            gated_delta_kernel<<<update_grid, value_dim_, shared_mem>>>(
                d_conv_out + b * conv_dim + num_heads_ * key_dim_,
                d_conv_out + b * conv_dim + num_heads_ * key_dim_ + k_dim,
                d_conv_out + b * conv_dim,
                d_a + b * num_heads_,
                d_b_raw + b * num_heads_,
                d_a_log_, d_dt_bias_, state.d_recurrent_state,
                d_attn_out + b * z_dim,
                num_heads_, key_dim_, value_dim_);
        }
        CUDA_CHECK_LAST_KERNEL();
    }

    dim3 ng_grid(num_heads_, batch_size);
    size_t ng_shmem = sizeof(float);
    norm_gate_fused_batch_kernel<<<ng_grid, 128, ng_shmem>>>(
        d_attn_out, d_norm_weight_, d_z, num_heads_, value_dim_, l2norm_eps, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        launch_la_output_proj(d_attn_out + b * z_dim, output + b * hidden_size_,
                              d_out_proj_weight_, z_dim, hidden_size_);
    }
    CUDA_CHECK_LAST_KERNEL();

    cudaMemcpy(state.d_conv_state,
               d_batch_conv_state + (batch_size - 1) * conv_state_per_token,
               conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_batch_conv_state);
}

} // namespace cuda
} // namespace qwen
