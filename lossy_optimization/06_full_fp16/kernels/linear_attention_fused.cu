#include "linear_attention_cuda.hpp"
#include "cuda_utils.cuh"
#include "cuda_error_handling.cuh"
#include <cmath>

namespace qwen {
namespace cuda {

__global__ void fused_linear_attention_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ in_proj_qkv_weight,
    const float* __restrict__ in_proj_a_weight,
    const float* __restrict__ in_proj_b_weight,
    const float* __restrict__ in_proj_z_weight,
    const float* __restrict__ conv1d_weight,
    const float* __restrict__ out_proj_weight,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_weight,
    float* __restrict__ recurrent_state,
    float* __restrict__ conv_state,
    int hidden_size,
    int num_heads,
    int key_dim,
    int value_dim,
    int conv_kernel) {

    int k_dim = num_heads * key_dim;
    int v_dim = num_heads * value_dim;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads * value_dim;

    // Use shared memory for intermediate results
    extern __shared__ float smem[];
    float* s_mixed_qkv = smem;
    float* s_conv_out = s_mixed_qkv + conv_dim;
    float* s_q = s_conv_out + conv_dim;
    float* s_k = s_q + num_heads * key_dim;
    float* s_v = s_k + k_dim;
    float* s_a = s_v + v_dim;
    float* s_b = s_a + num_heads;
    float* s_z = s_b + num_heads;
    float* s_attn_out = s_z + z_dim;

    // Step 1: Linear projection for QKV
    for (int i = threadIdx.x; i < conv_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += in_proj_qkv_weight[i * hidden_size + j] * input[j];
        }
        s_mixed_qkv[i] = sum;
    }
    __syncthreads();

    // Step 2: Conv1D + SiLU + state update
    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        float sum = conv1d_weight[d * conv_kernel + (conv_kernel - 1)] * s_mixed_qkv[d];
        for (int k = 0; k < conv_kernel - 1; ++k) {
            sum += conv1d_weight[d * conv_kernel + k] * conv_state[d * (conv_kernel - 1) + k];
        }
        s_conv_out[d] = silu(sum);

        // Update conv state
        for (int k = (conv_kernel - 2); k > 0; --k) {
            conv_state[d * (conv_kernel - 1) + k] = conv_state[d * (conv_kernel - 1) + k - 1];
        }
        conv_state[d * (conv_kernel - 1)] = s_mixed_qkv[d];
    }
    __syncthreads();

    // Step 3: Extract Q, K, V
    for (int i = threadIdx.x; i < num_heads * key_dim; i += blockDim.x) {
        s_q[i] = s_conv_out[i];
    }
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
        s_k[i] = s_conv_out[num_heads * key_dim + i];
    }
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
        s_v[i] = s_conv_out[num_heads * key_dim + k_dim + i];
    }
    __syncthreads();

    // Step 4: L2 normalize Q and K
    const float q_scale = 1.0f / sqrtf(static_cast<float>(key_dim));
    for (int h = threadIdx.x; h < num_heads; h += blockDim.x) {
        float l2q = 0.0f, l2k = 0.0f;
        for (int d = 0; d < key_dim; ++d) {
            float vq = s_q[h * key_dim + d];
            float vk = s_k[h * key_dim + d];
            l2q += vq * vq;
            l2k += vk * vk;
        }
        l2q = sqrtf(l2q + 1e-6f);
        l2k = sqrtf(l2k + 1e-6f);
        for (int d = 0; d < key_dim; ++d) {
            s_q[h * key_dim + d] = s_q[h * key_dim + d] / l2q * q_scale;
            s_k[h * key_dim + d] /= l2k;
        }
    }
    __syncthreads();

    // Step 5: Linear projections for a, b, z
    for (int i = threadIdx.x; i < num_heads; i += blockDim.x) {
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum_a += in_proj_a_weight[i * hidden_size + j] * input[j];
            sum_b += in_proj_b_weight[i * hidden_size + j] * input[j];
        }
        s_a[i] = sum_a;
        s_b[i] = sum_b;
    }
    for (int i = threadIdx.x; i < z_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += in_proj_z_weight[i * hidden_size + j] * input[j];
        }
        s_z[i] = sum;
    }
    __syncthreads();

    // Step 6: Gated DeltaNet update
    for (int h = 0; h < num_heads; ++h) {
        float b_sig = 1.0f / (1.0f + expf(-s_b[h]));
        float sp = s_a[h] + dt_bias[h];
        sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));
        float g_t = expf(-expf(a_log[h]) * sp);

        int state_base = h * key_dim * value_dim;

        // Apply g_t to recurrent state
        for (int kd = 0; kd < key_dim; ++kd) {
            for (int vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
                recurrent_state[state_base + kd * value_dim + vd] *= g_t;
            }
        }
        __syncthreads();

        // Compute KV memory and delta
        for (int vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
            float kv_mem = 0.0f;
            for (int kd = 0; kd < key_dim; ++kd) {
                kv_mem += recurrent_state[state_base + kd * value_dim + vd] * s_k[h * key_dim + kd];
            }
            float delta = (s_v[h * value_dim + vd] - kv_mem) * b_sig;

            // Update recurrent state
            for (int kd = 0; kd < key_dim; ++kd) {
                recurrent_state[state_base + kd * value_dim + vd] += s_k[h * key_dim + kd] * delta;
            }

            // Compute attention output
            float sum = 0.0f;
            for (int kd = 0; kd < key_dim; ++kd) {
                sum += recurrent_state[state_base + kd * value_dim + vd] * s_q[h * key_dim + kd];
            }
            s_attn_out[h * value_dim + vd] = sum;
        }
        __syncthreads();
    }

    // Step 7: RMSNorm + Gate (SiLU)
    for (int h = threadIdx.x; h < num_heads; h += blockDim.x) {
        float variance = 0.0f;
        for (int d = 0; d < value_dim; ++d) {
            float v = s_attn_out[h * value_dim + d];
            variance += v * v;
        }
        variance /= value_dim;
        float inv_rms = 1.0f / sqrtf(variance + 1e-6f);
        for (int d = 0; d < value_dim; ++d) {
            float normed = s_attn_out[h * value_dim + d] * inv_rms * norm_weight[d];
            float zv = s_z[h * value_dim + d];
            float s = zv / (1.0f + expf(-zv));
            s_attn_out[h * value_dim + d] = normed * s;
        }
    }
    __syncthreads();

    // Step 8: Output projection
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < z_dim; ++j) {
            sum += out_proj_weight[i * z_dim + j] * s_attn_out[j];
        }
        output[i] = sum;
    }
}

void CudaLinearAttention::forward_fused(const float* input, float* output,
                                        CudaLinearAttnState& state) const {
    int k_dim = num_heads_ * key_dim_;
    int v_dim = num_heads_ * value_dim_;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads_ * value_dim_;

    // Calculate shared memory size
    size_t shared_mem = (conv_dim * 2 + num_heads_ * key_dim_ + k_dim + v_dim + num_heads_ * 2 +
                         z_dim * 2) *
                        sizeof(float);

    // Launch fused kernel with single block
    int block_size = 256;
    fused_linear_attention_kernel<<<1, block_size, shared_mem>>>(
        input, output, d_in_proj_qkv_weight_, d_in_proj_a_weight_, d_in_proj_b_weight_,
        d_in_proj_z_weight_, d_conv1d_weight_, d_out_proj_weight_, d_a_log_, d_dt_bias_,
        d_norm_weight_, state.d_recurrent_state, state.d_conv_state, hidden_size_, num_heads_,
        key_dim_, value_dim_, conv_kernel_);
    CUDA_CHECK_LAST_KERNEL();
}

} // namespace cuda
} // namespace qwen
