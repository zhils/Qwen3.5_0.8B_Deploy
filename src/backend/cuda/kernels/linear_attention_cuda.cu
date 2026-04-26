#include "linear_attention_cuda.hpp"
#include "cuda_utils.cuh"
#include "cuda_error_handling.cuh"
#include <cmath>
#include <stdexcept>

namespace qwen {
namespace cuda {

__global__ void linear_proj_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   const float* __restrict__ weight, int hidden_size,
                                   int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size)
        return;

    float sum = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        sum += weight[idx * hidden_size + j] * input[j];
    }
    output[idx] = sum;
}

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

__global__ void la_output_proj_kernel(const float* __restrict__ attn_out,
                                      float* __restrict__ output,
                                      const float* __restrict__ out_weight, int input_dim,
                                      int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_dim)
        return;

    float sum = 0.0f;
    for (int j = 0; j < input_dim; ++j) {
        sum += out_weight[idx * input_dim + j] * attn_out[j];
    }
    output[idx] = sum;
}

__global__ void conv1d_update_fused_kernel(const float* __restrict__ mixed_qkv,
                                           float* __restrict__ conv_out,
                                           float* __restrict__ conv_state,
                                           const float* __restrict__ conv_weight, int conv_dim,
                                           int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;
    float sum = conv_weight[d * conv_kernel + (conv_kernel - 1)] * mixed_qkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += conv_weight[d * conv_kernel + k] * conv_state[d * (conv_kernel - 1) + k];
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
      conv_kernel_(conv_kernel) {
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
    float* d_q = d_q_buf_;
    float* d_k = d_k_buf_;
    float* d_v = d_v_buf_;
    float* d_a = d_a_buf_;
    float* d_b_raw = d_b_raw_buf_;

    dim3 block(256);
    dim3 grid((conv_dim + 255) / 256);
    linear_proj_kernel<<<grid, block>>>(input, d_mixed_qkv, d_in_proj_qkv_weight_, hidden_size_,
                                        conv_dim);
    CUDA_CHECK_LAST_KERNEL();

    // Opt-5: Fused conv1d + state update — 1 kernel instead of 2
    dim3 conv_grid((conv_dim + 255) / 256);
    conv1d_update_fused_kernel<<<conv_grid, block>>>(d_mixed_qkv, d_conv_out, state.d_conv_state,
                                                     d_conv1d_weight_, conv_dim, conv_kernel_);
    CUDA_CHECK_LAST_KERNEL();

    cudaMemcpy(d_q, d_conv_out, num_heads_ * key_dim_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_k, d_conv_out + num_heads_ * key_dim_, k_dim * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_v, d_conv_out + num_heads_ * key_dim_ + k_dim, v_dim * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Opt-5: Fused L2 norm Q+K — 1 kernel instead of 2
    const float l2norm_eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    l2norm_qk_fused_kernel<<<num_heads_, 1>>>(d_q, d_k, num_heads_, key_dim_, q_scale, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    dim3 small_grid((num_heads_ + 255) / 256);
    linear_proj_kernel<<<small_grid, block>>>(input, d_a, d_in_proj_a_weight_, hidden_size_,
                                              num_heads_);
    CUDA_CHECK_LAST_KERNEL();
    linear_proj_kernel<<<small_grid, block>>>(input, d_b_raw, d_in_proj_b_weight_, hidden_size_,
                                              num_heads_);
    CUDA_CHECK_LAST_KERNEL();

    dim3 z_grid((z_dim + 255) / 256);
    linear_proj_kernel<<<z_grid, block>>>(input, d_z_buf_, d_in_proj_z_weight_, hidden_size_,
                                          z_dim);
    CUDA_CHECK_LAST_KERNEL();

    float* d_attn_out = d_attn_out_buf_;

    dim3 update_grid(num_heads_);
    size_t shared_mem = (2 + 2 * value_dim_) * sizeof(float);
    gated_delta_kernel<<<update_grid, value_dim_, shared_mem>>>(
        d_k, d_v, d_q, d_a, d_b_raw, d_a_log_, d_dt_bias_, state.d_recurrent_state, d_attn_out,
        num_heads_, key_dim_, value_dim_);
    CUDA_CHECK_LAST_KERNEL();

    // Opt-5: Fused norm + gate — 1 kernel instead of 2
    size_t norm_gate_shmem = sizeof(float);
    norm_gate_fused_kernel<<<num_heads_, 128, norm_gate_shmem>>>(
        d_attn_out, d_norm_weight_, d_z_buf_, num_heads_, value_dim_, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    dim3 proj_block(256);
    dim3 proj_grid((hidden_size_ + 255) / 256);
    la_output_proj_kernel<<<proj_grid, proj_block>>>(d_attn_out, output, d_out_proj_weight_, z_dim,
                                                     hidden_size_);
    CUDA_CHECK_LAST_KERNEL();
}

void CudaLinearAttention::forward_batch(const float* input, float* output, CudaLinearAttnState& state,
                                        int batch_size) const {
    for (int b = 0; b < batch_size; ++b) {
        forward(input + b * hidden_size_, output + b * hidden_size_, state);
    }
}

void CudaLinearAttention::forward_batch_streamed(const float* input, float* output,
                                                  CudaLinearAttnState& state, int batch_size,
                                                  cudaStream_t stream) const {
    for (int b = 0; b < batch_size; ++b) {
        forward(input + b * hidden_size_, output + b * hidden_size_, state);
    }
}

} // namespace cuda
} // namespace qwen
