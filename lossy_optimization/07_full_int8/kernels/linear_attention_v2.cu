#include "linear_attention_v2.cuh"
#include "cuda_utils.cuh"
#include "cuda_error_handling.cuh"
#include "cutlass_gemm_wrapper.cuh"
#include <cmath>
#include <cstdio>

namespace qwen {
namespace cuda {

namespace {

// Optimized linear projection v2 using shared memory tiling
template <int TILE_M, int TILE_K, int THREADS>
__global__ void linear_proj_v2_kernel_optimized(const float* __restrict__ input,
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
__global__ void linear_proj_v2_kernel_simple(const float* __restrict__ input, float* __restrict__ output,
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

static void launch_linear_proj_v2(const float* input, float* output, const float* weight,
                                  int hidden_size, int out_size, cudaStream_t stream = 0) {
    if (out_size < 128 || hidden_size < 128) {
        int block = 256;
        int grid = (out_size + block - 1) / block;
        linear_proj_v2_kernel_simple<<<grid, block, 0, stream>>>(input, output, weight, hidden_size, out_size);
        return;
    }

    // Use CUTLASS-style warp-level GEMV
    bool aligned = ((uintptr_t)weight % 16 == 0) &&
                   ((uintptr_t)input % 16 == 0) &&
                   (hidden_size % 4 == 0);

    if (aligned) {
        launch_cutlass_gemv(weight, input, output, out_size, hidden_size, stream);
    } else {
        const int TILE_M = 32;
        const int TILE_K = 256;
        const int THREADS = 256;
        int grid = (out_size + TILE_M - 1) / TILE_M;
        linear_proj_v2_kernel_optimized<TILE_M, TILE_K, THREADS>
            <<<grid, THREADS, 0, stream>>>(input, output, weight, hidden_size, out_size);
    }
}

// Keep old name for backward compatibility - device function wrapper
__device__ void linear_proj_v2_kernel_device(const float* __restrict__ input, float* __restrict__ output,
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

__global__ void linear_proj_v2_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      const float* __restrict__ weight, int hidden_size,
                                      int out_size) {
    linear_proj_v2_kernel_device(input, output, weight, hidden_size, out_size);
}

__global__ void conv1d_update_fused_v2_kernel(const float* __restrict__ mixed_qkv,
                                               float* __restrict__ conv_out,
                                               float* __restrict__ conv_state,
                                               const float* __restrict__ conv_weight, int conv_dim,
                                               int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;

    // Cache conv_weight for this channel in registers
    float w_reg[8];
    #pragma unroll
    for (int k = 0; k < conv_kernel; ++k) {
        w_reg[k] = conv_weight[d * conv_kernel + k];
    }

    float sum = w_reg[conv_kernel - 1] * mixed_qkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += w_reg[k] * conv_state[d * (conv_kernel - 1) + k];
    }
    conv_out[d] = silu(sum);
    for (int k = (conv_kernel - 2); k > 0; --k) {
        conv_state[d * (conv_kernel - 1) + k] = conv_state[d * (conv_kernel - 1) + k - 1];
    }
    conv_state[d * (conv_kernel - 1)] = mixed_qkv[d];
}

__global__ void l2norm_qk_fused_v2_kernel(float* __restrict__ d_q, float* __restrict__ d_k,
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

__global__ void norm_gate_fused_v2_kernel(float* __restrict__ data,
                                          const float* __restrict__ norm_weight,
                                          const float* __restrict__ z, int num_heads,
                                          int value_dim, float eps) {
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
        smem[0] = 1.0f / sqrtf(variance + eps);
    }
    __syncthreads();
    float inv_rms = smem[0];

    // Process elements with register caching to reduce global memory access
    for (int d = tid; d < value_dim; d += blockDim.x) {
        float v_data = data[h * value_dim + d];
        float v_normed = v_data * inv_rms * norm_weight[d];
        float zv = z[h * value_dim + d];
        float s = zv / (1.0f + expf(-zv));
        data[h * value_dim + d] = v_normed * s;
    }
}

// Optimized output projection v2 using shared memory tiling
template <int TILE_M, int TILE_K, int THREADS>
__global__ void la_output_proj_v2_kernel_optimized(const float* __restrict__ attn_out,
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
__global__ void la_output_proj_v2_kernel_simple(const float* __restrict__ attn_out, float* __restrict__ output,
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

static void launch_la_output_proj_v2(const float* attn_out, float* output, const float* out_weight,
                                     int input_dim, int output_dim, cudaStream_t stream = 0) {
    if (output_dim < 128 || input_dim < 128) {
        int block = 256;
        int grid = (output_dim + block - 1) / block;
        la_output_proj_v2_kernel_simple<<<grid, block, 0, stream>>>(attn_out, output, out_weight, input_dim, output_dim);
        return;
    }

    // Use CUTLASS-style warp-level GEMV
    bool aligned = ((uintptr_t)out_weight % 16 == 0) &&
                   ((uintptr_t)attn_out % 16 == 0) &&
                   (input_dim % 4 == 0);

    if (aligned) {
        launch_cutlass_gemv(out_weight, attn_out, output, output_dim, input_dim, stream);
    } else {
        const int TILE_M = 32;
        const int TILE_K = 256;
        const int THREADS = 256;
        int grid = (output_dim + TILE_M - 1) / TILE_M;
        la_output_proj_v2_kernel_optimized<TILE_M, TILE_K, THREADS>
            <<<grid, THREADS, 0, stream>>>(attn_out, output, out_weight, input_dim, output_dim);
    }
}

// Keep old name for backward compatibility - device function wrapper
__device__ void la_output_proj_v2_kernel_device(const float* __restrict__ attn_out, float* __restrict__ output,
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

__global__ void la_output_proj_v2_kernel(const float* __restrict__ attn_out, float* __restrict__ output,
                                         const float* __restrict__ out_weight, int input_dim,
                                         int output_dim) {
    la_output_proj_v2_kernel_device(attn_out, output, out_weight, input_dim, output_dim);
}

__global__ void conv1d_update_fused_batch_v2_kernel(
    const float* __restrict__ mixed_qkv, float* __restrict__ conv_out,
    float* __restrict__ conv_state, const float* __restrict__ conv_weight, int conv_dim,
    int conv_kernel, int batch_size) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= conv_dim || b >= batch_size)
        return;

    const float* mqkv = mixed_qkv + b * conv_dim;
    float* cout = conv_out + b * conv_dim;
    float* cstate = conv_state + b * conv_dim * (conv_kernel - 1);

    // Cache conv_weight for this channel in registers (small kernel size, e.g. 4)
    float w_reg[8];
    #pragma unroll
    for (int k = 0; k < conv_kernel; ++k) {
        w_reg[k] = conv_weight[d * conv_kernel + k];
    }

    float sum = w_reg[conv_kernel - 1] * mqkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += w_reg[k] * cstate[d * (conv_kernel - 1) + k];
    }
    cout[d] = silu(sum);
    for (int k = (conv_kernel - 2); k > 0; --k) {
        cstate[d * (conv_kernel - 1) + k] = cstate[d * (conv_kernel - 1) + k - 1];
    }
    cstate[d * (conv_kernel - 1)] = mqkv[d];
}

__global__ void l2norm_qk_fused_batch_v2_kernel(float* __restrict__ d_q, float* __restrict__ d_k,
                                                 int num_heads, int key_dim, float q_scale,
                                                 float eps, int batch_size) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= num_heads || b >= batch_size)
        return;

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

__global__ void norm_gate_fused_batch_v2_kernel(float* __restrict__ data,
                                                 const float* __restrict__ norm_weight,
                                                 const float* __restrict__ z, int num_heads,
                                                 int value_dim, float eps, int batch_size) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    if (h >= num_heads || b >= batch_size)
        return;

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

    // Process elements with register caching to reduce global memory access
    for (int d = tid; d < value_dim; d += blockDim.x) {
        float v_data = data[offset + d];
        float v_normed = v_data * inv_rms * norm_weight[d];
        float zv = z[offset + d];
        float s = zv / (1.0f + expf(-zv));
        data[offset + d] = v_normed * s;
    }
}

__global__ void broadcast_conv_state_v2_kernel(float* __restrict__ batch_conv_state,
                                                const float* __restrict__ base_conv_state,
                                                int conv_state_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_state_size)
        return;
    float val = base_conv_state[idx];
    for (int b = 0; b < batch_size; ++b) {
        batch_conv_state[b * conv_state_size + idx] = val;
    }
}

__global__ void gated_delta_single_v2_kernel(const float* __restrict__ k, const float* __restrict__ v,
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
        s_params[0] = 1.0f / (1.0f + expf(-b[head_idx]));
        float sp = a[head_idx] + dt_bias[head_idx];
        sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));
        s_params[1] = expf(-expf(a_log[head_idx]) * sp);
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

__global__ void gated_delta_batch_fused_v2_kernel(
    const float* __restrict__ conv_out, const float* __restrict__ a,
    const float* __restrict__ b_raw, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ recurrent_state,
    float* __restrict__ attn_out, int num_heads, int key_dim, int value_dim, int k_dim,
    int conv_dim, int z_dim, int batch_size) {

    int head_idx = blockIdx.x;
    int tidx = threadIdx.x;
    if (head_idx >= num_heads || tidx >= value_dim)
        return;

    int state_base = head_idx * key_dim * value_dim;

    #pragma unroll 4
    for (int b = 0; b < batch_size; ++b) {
        float b_sig = 1.0f / (1.0f + expf(-b_raw[b * num_heads + head_idx]));
        float sp = a[b * num_heads + head_idx] + dt_bias[head_idx];
        sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));
        float g_t = expf(-expf(a_log[head_idx]) * sp);

        const float* k_ptr = conv_out + b * conv_dim + num_heads * key_dim + head_idx * key_dim;
        const float* q_ptr = conv_out + b * conv_dim + head_idx * key_dim;
        const float* v_ptr = conv_out + b * conv_dim + 2 * num_heads * key_dim + head_idx * value_dim;

        float kv_mem = 0.0f;
        #pragma unroll 4
        for (int kd = 0; kd < key_dim; ++kd) {
            float state_val = recurrent_state[state_base + kd * value_dim + tidx];
            state_val *= g_t;
            recurrent_state[state_base + kd * value_dim + tidx] = state_val;
            kv_mem += state_val * k_ptr[kd];
        }

        float v_val = v_ptr[tidx];
        float delta = (v_val - kv_mem) * b_sig;

        float sum = 0.0f;
        #pragma unroll 4
        for (int kd = 0; kd < key_dim; ++kd) {
            float state_val = recurrent_state[state_base + kd * value_dim + tidx];
            state_val += k_ptr[kd] * delta;
            recurrent_state[state_base + kd * value_dim + tidx] = state_val;
            sum += state_val * q_ptr[kd];
        }

        attn_out[b * z_dim + head_idx * value_dim + tidx] = sum;
    }
}

__global__ void gated_delta_batch_fused_v2_reg_kernel(
    const float* __restrict__ conv_out, const float* __restrict__ a,
    const float* __restrict__ b_raw, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, float* __restrict__ recurrent_state,
    float* __restrict__ attn_out, int num_heads, int key_dim, int value_dim, int k_dim,
    int conv_dim, int z_dim, int batch_size) {

    int head_idx = blockIdx.x;
    int tidx = threadIdx.x;
    if (head_idx >= num_heads || tidx >= value_dim)
        return;

    int state_base = head_idx * key_dim * value_dim;

    float state_reg[128];
    for (int kd = 0; kd < key_dim; ++kd) {
        state_reg[kd] = recurrent_state[state_base + kd * value_dim + tidx];
    }

    for (int b = 0; b < batch_size; ++b) {
        // Use __ldg for read-only data to leverage L2 cache
        float b_sig = 1.0f / (1.0f + expf(-__ldg(&b_raw[b * num_heads + head_idx])));
        float sp = __ldg(&a[b * num_heads + head_idx]) + __ldg(&dt_bias[head_idx]);
        sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));
        float g_t = expf(-expf(__ldg(&a_log[head_idx])) * sp);

        const float* k_ptr = conv_out + b * conv_dim + num_heads * key_dim + head_idx * key_dim;
        const float* q_ptr = conv_out + b * conv_dim + head_idx * key_dim;
        const float* v_ptr = conv_out + b * conv_dim + 2 * num_heads * key_dim + head_idx * value_dim;

        float kv_mem = 0.0f;
        #pragma unroll 4
        for (int kd = 0; kd < key_dim; ++kd) {
            state_reg[kd] *= g_t;
            kv_mem += state_reg[kd] * __ldg(&k_ptr[kd]);
        }

        float v_val = __ldg(&v_ptr[tidx]);
        float delta = (v_val - kv_mem) * b_sig;

        float sum = 0.0f;
        #pragma unroll 4
        for (int kd = 0; kd < key_dim; ++kd) {
            state_reg[kd] += __ldg(&k_ptr[kd]) * delta;
            sum += state_reg[kd] * __ldg(&q_ptr[kd]);
        }

        attn_out[b * z_dim + head_idx * value_dim + tidx] = sum;
    }

    for (int kd = 0; kd < key_dim; ++kd) {
        recurrent_state[state_base + kd * value_dim + tidx] = state_reg[kd];
    }
}

} // anonymous namespace

CudaLinearAttentionV2::CudaLinearAttentionV2(int hidden_size, int num_heads, int key_dim,
                                             int value_dim, int conv_kernel)
    : hidden_size_(hidden_size), num_heads_(num_heads), key_dim_(key_dim), value_dim_(value_dim),
      conv_kernel_(conv_kernel), d_batch_mixed_qkv_buf_(nullptr),
      d_batch_conv_out_buf_(nullptr), d_batch_a_buf_(nullptr), d_batch_b_raw_buf_(nullptr),
      d_batch_z_buf_(nullptr), d_batch_attn_out_buf_(nullptr), d_batch_conv_state_buf_(nullptr),
      d_scan_g_buf_(nullptr), d_scan_b_vec_buf_(nullptr), max_batch_size_(0) {
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    cudaMalloc(&d_in_proj_qkv_weight_, static_cast<size_t>(conv_dim) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_in_proj_a_weight_, static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_in_proj_b_weight_, static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_in_proj_z_weight_, static_cast<size_t>(z_dim) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_conv1d_weight_, static_cast<size_t>(conv_dim) * conv_kernel_ * sizeof(float));
    cudaMalloc(&d_out_proj_weight_, static_cast<size_t>(hidden_size_) * z_dim * sizeof(float));
    cudaMalloc(&d_a_log_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_dt_bias_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_norm_weight_, static_cast<size_t>(value_dim_) * sizeof(float));
    cudaMemset(d_norm_weight_, 0, static_cast<size_t>(value_dim_) * sizeof(float));

    cudaMalloc(&d_mixed_qkv_buf_, static_cast<size_t>(conv_dim) * sizeof(float));
    cudaMalloc(&d_conv_out_buf_, static_cast<size_t>(conv_dim) * sizeof(float));
    cudaMalloc(&d_a_buf_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_b_raw_buf_, static_cast<size_t>(num_heads_) * sizeof(float));
    cudaMalloc(&d_attn_out_buf_, static_cast<size_t>(z_dim) * sizeof(float));
    cudaMalloc(&d_z_buf_, static_cast<size_t>(z_dim) * sizeof(float));

    const int PREALLOC_BATCH_SIZE = 128;
    ensure_batch_buffers(PREALLOC_BATCH_SIZE);
}

CudaLinearAttentionV2::~CudaLinearAttentionV2() {
    if (d_in_proj_qkv_weight_) cudaFree(d_in_proj_qkv_weight_);
    if (d_in_proj_a_weight_) cudaFree(d_in_proj_a_weight_);
    if (d_in_proj_b_weight_) cudaFree(d_in_proj_b_weight_);
    if (d_in_proj_z_weight_) cudaFree(d_in_proj_z_weight_);
    if (d_conv1d_weight_) cudaFree(d_conv1d_weight_);
    if (d_out_proj_weight_) cudaFree(d_out_proj_weight_);
    if (d_a_log_) cudaFree(d_a_log_);
    if (d_dt_bias_) cudaFree(d_dt_bias_);
    if (d_norm_weight_) cudaFree(d_norm_weight_);

    if (d_mixed_qkv_buf_) cudaFree(d_mixed_qkv_buf_);
    if (d_conv_out_buf_) cudaFree(d_conv_out_buf_);
    if (d_a_buf_) cudaFree(d_a_buf_);
    if (d_b_raw_buf_) cudaFree(d_b_raw_buf_);
    if (d_attn_out_buf_) cudaFree(d_attn_out_buf_);
    if (d_z_buf_) cudaFree(d_z_buf_);
}

void CudaLinearAttentionV2::ensure_batch_buffers(int batch_size) const {
    if (batch_size <= max_batch_size_ && d_batch_mixed_qkv_buf_ != nullptr) {
        return;
    }
    if (d_batch_mixed_qkv_buf_) cudaFree(d_batch_mixed_qkv_buf_);
    if (d_batch_conv_out_buf_) cudaFree(d_batch_conv_out_buf_);
    if (d_batch_a_buf_) cudaFree(d_batch_a_buf_);
    if (d_batch_b_raw_buf_) cudaFree(d_batch_b_raw_buf_);
    if (d_batch_z_buf_) cudaFree(d_batch_z_buf_);
    if (d_batch_attn_out_buf_) cudaFree(d_batch_attn_out_buf_);
    if (d_batch_conv_state_buf_) cudaFree(d_batch_conv_state_buf_);
    if (d_scan_g_buf_) cudaFree(d_scan_g_buf_);
    if (d_scan_b_vec_buf_) cudaFree(d_scan_b_vec_buf_);

    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;
    size_t conv_state_per_token = static_cast<size_t>(conv_dim) * (conv_kernel_ - 1);

    cudaMalloc(&d_batch_mixed_qkv_buf_, static_cast<size_t>(batch_size) * conv_dim * sizeof(float));
    cudaMalloc(&d_batch_conv_out_buf_, static_cast<size_t>(batch_size) * conv_dim * sizeof(float));
    cudaMalloc(&d_batch_a_buf_, static_cast<size_t>(batch_size) * num_heads_ * sizeof(float));
    cudaMalloc(&d_batch_b_raw_buf_, static_cast<size_t>(batch_size) * num_heads_ * sizeof(float));
    cudaMalloc(&d_batch_z_buf_, static_cast<size_t>(batch_size) * z_dim * sizeof(float));
    cudaMalloc(&d_batch_attn_out_buf_, static_cast<size_t>(batch_size) * z_dim * sizeof(float));
    cudaMalloc(&d_batch_conv_state_buf_,
               conv_state_per_token * batch_size * sizeof(float));
    cudaMalloc(&d_scan_g_buf_, static_cast<size_t>(batch_size) * num_heads_ * sizeof(float));
    cudaMalloc(&d_scan_b_vec_buf_, static_cast<size_t>(batch_size) * num_heads_ * value_dim_ * sizeof(float));
    max_batch_size_ = batch_size;
}

void CudaLinearAttentionV2::set_weights(
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

void CudaLinearAttentionV2::forward(const float* input, float* output,
                                    CudaLinearAttnState& state) const {
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    float* d_mixed_qkv = d_mixed_qkv_buf_;
    float* d_conv_out = d_conv_out_buf_;
    float* d_a = d_a_buf_;
    float* d_b_raw = d_b_raw_buf_;

    launch_linear_proj_v2(input, d_mixed_qkv, d_in_proj_qkv_weight_, hidden_size_, conv_dim);
    CUDA_CHECK_LAST_KERNEL();

    dim3 block(256);
    dim3 conv_grid((conv_dim + 255) / 256);
    conv1d_update_fused_v2_kernel<<<conv_grid, block>>>(d_mixed_qkv, d_conv_out, state.d_conv_state,
                                                        d_conv1d_weight_, conv_dim, conv_kernel_);
    CUDA_CHECK_LAST_KERNEL();

    const float l2norm_eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    l2norm_qk_fused_v2_kernel<<<num_heads_, 1>>>(d_conv_out, d_conv_out + num_heads_ * key_dim_,
                                                  num_heads_, key_dim_, q_scale, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    launch_linear_proj_v2(input, d_a, d_in_proj_a_weight_, hidden_size_, num_heads_);
    CUDA_CHECK_LAST_KERNEL();
    launch_linear_proj_v2(input, d_b_raw, d_in_proj_b_weight_, hidden_size_, num_heads_);
    CUDA_CHECK_LAST_KERNEL();

    launch_linear_proj_v2(input, d_z_buf_, d_in_proj_z_weight_, hidden_size_, z_dim);
    CUDA_CHECK_LAST_KERNEL();

    float* d_attn_out = d_attn_out_buf_;

    dim3 update_grid(num_heads_);
    size_t shared_mem = (2 + 2 * value_dim_) * sizeof(float);
    gated_delta_single_v2_kernel<<<update_grid, value_dim_, shared_mem>>>(
        d_conv_out + num_heads_ * key_dim_,
        d_conv_out + num_heads_ * key_dim_ + k_dim,
        d_conv_out,
        d_a, d_b_raw, d_a_log_, d_dt_bias_, state.d_recurrent_state, d_attn_out,
        num_heads_, key_dim_, value_dim_);
    CUDA_CHECK_LAST_KERNEL();

    size_t norm_gate_shmem = sizeof(float);
    norm_gate_fused_v2_kernel<<<num_heads_, 128, norm_gate_shmem>>>(
        d_attn_out, d_norm_weight_, d_z_buf_, num_heads_, value_dim_, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    launch_la_output_proj_v2(d_attn_out, output, d_out_proj_weight_, z_dim, hidden_size_);
    CUDA_CHECK_LAST_KERNEL();
}

void CudaLinearAttentionV2::forward_batch(const float* input, float* output,
                                          CudaLinearAttnState& state, int batch_size,
                                          cudaStream_t stream) const {
    if (batch_size == 1) {
        forward(input, output, state);
        return;
    }

    ensure_batch_buffers(batch_size);

    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;
    size_t conv_state_per_token = static_cast<size_t>(conv_dim) * (conv_kernel_ - 1);

    float* d_mixed_qkv = d_batch_mixed_qkv_buf_;
    float* d_conv_out = d_batch_conv_out_buf_;
    float* d_a = d_batch_a_buf_;
    float* d_b_raw = d_batch_b_raw_buf_;
    float* d_z = d_batch_z_buf_;
    float* d_attn_out = d_batch_attn_out_buf_;
    float* d_batch_conv_state = d_batch_conv_state_buf_;

    int broadcast_block = 256;
    int broadcast_grid = (static_cast<int>(conv_state_per_token) + 255) / 256;
    broadcast_conv_state_v2_kernel<<<broadcast_grid, broadcast_block, 0, stream>>>(
        d_batch_conv_state, state.d_conv_state, static_cast<int>(conv_state_per_token), batch_size);
    CUDA_CHECK_LAST_KERNEL();

    dim3 block(256);

    for (int b = 0; b < batch_size; ++b) {
        launch_linear_proj_v2(
            input + b * hidden_size_, d_mixed_qkv + b * conv_dim,
            d_in_proj_qkv_weight_, hidden_size_, conv_dim, stream);
    }
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        launch_linear_proj_v2(
            input + b * hidden_size_, d_a + b * num_heads_,
            d_in_proj_a_weight_, hidden_size_, num_heads_, stream);
        launch_linear_proj_v2(
            input + b * hidden_size_, d_b_raw + b * num_heads_,
            d_in_proj_b_weight_, hidden_size_, num_heads_, stream);
    }
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        launch_linear_proj_v2(
            input + b * hidden_size_, d_z + b * z_dim,
            d_in_proj_z_weight_, hidden_size_, z_dim, stream);
    }
    CUDA_CHECK_LAST_KERNEL();

    dim3 conv_block(256);
    dim3 conv_grid((conv_dim + 255) / 256, batch_size);
    conv1d_update_fused_batch_v2_kernel<<<conv_grid, conv_block, 0, stream>>>(
        d_mixed_qkv, d_conv_out, d_batch_conv_state, d_conv1d_weight_,
        conv_dim, conv_kernel_, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    dim3 l2_grid(num_heads_, batch_size);
    const float l2norm_eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    l2norm_qk_fused_batch_v2_kernel<<<l2_grid, 1, 0, stream>>>(
        d_conv_out, d_conv_out + num_heads_ * key_dim_,
        num_heads_, key_dim_, q_scale, l2norm_eps, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    dim3 update_grid(num_heads_);
    gated_delta_batch_fused_v2_reg_kernel<<<update_grid, value_dim_, 0, stream>>>(
        d_conv_out, d_a, d_b_raw, d_a_log_, d_dt_bias_, state.d_recurrent_state, d_attn_out,
        num_heads_, key_dim_, value_dim_, k_dim, conv_dim, z_dim, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    dim3 ng_grid(num_heads_, batch_size);
    size_t ng_shmem = sizeof(float);
    norm_gate_fused_batch_v2_kernel<<<ng_grid, 128, ng_shmem, stream>>>(
        d_attn_out, d_norm_weight_, d_z, num_heads_, value_dim_, l2norm_eps, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    for (int b = 0; b < batch_size; ++b) {
        launch_la_output_proj_v2(
            d_attn_out + b * z_dim, output + b * hidden_size_,
            d_out_proj_weight_, z_dim, hidden_size_, stream);
    }
    CUDA_CHECK_LAST_KERNEL();

    cudaMemcpyAsync(state.d_conv_state,
                    d_batch_conv_state + (batch_size - 1) * conv_state_per_token,
                    conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

} // namespace cuda
} // namespace qwen
