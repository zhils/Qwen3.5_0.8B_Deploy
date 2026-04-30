#include "linear_attention_cuda.hpp"
#include "cuda_utils.cuh"
#include "cuda_error_handling.cuh"
#include "cublas_handle_pool.hpp"
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cuda_bf16.h>

namespace qwen {
namespace cuda {

#define LA_CUBLAS_CHECK(call)                                                                      \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(_err), __FILE__,        \
                    __LINE__);                                                                     \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static __global__ void fp32_to_bf16_kernel(const float* __restrict__ fp32, __nv_bfloat16* __restrict__ bf16, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) bf16[i] = __float2bfloat16(fp32[i]);
}

static __global__ void bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ bf16, float* __restrict__ fp32, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fp32[i] = __bfloat162float(bf16[i]);
}

__global__ void linear_proj_bf16_kernel(const float* __restrict__ input, float* __restrict__ output,
                                        const __nv_bfloat16* __restrict__ weight, int hidden_size,
                                        int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size)
        return;

    float sum = 0.0f;
    for (int j = 0; j < hidden_size; ++j) {
        sum += __bfloat162float(weight[idx * hidden_size + j]) * input[j];
    }
    output[idx] = sum;
}

__global__ void conv1d_update_fused_bf16_kernel(const float* __restrict__ mixed_qkv,
                                                float* __restrict__ conv_out,
                                                float* __restrict__ conv_state,
                                                const __nv_bfloat16* __restrict__ conv_weight,
                                                int conv_dim, int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;
    float sum = __bfloat162float(conv_weight[d * conv_kernel + (conv_kernel - 1)]) * mixed_qkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += __bfloat162float(conv_weight[d * conv_kernel + k]) * conv_state[d * (conv_kernel - 1) + k];
    }
    conv_out[d] = silu(sum);
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

__global__ void la_output_proj_bf16_kernel(const float* __restrict__ attn_out,
                                           float* __restrict__ output,
                                           const __nv_bfloat16* __restrict__ out_weight,
                                           int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_dim)
        return;

    float sum = 0.0f;
    for (int j = 0; j < input_dim; ++j) {
        sum += __bfloat162float(out_weight[idx * input_dim + j]) * attn_out[j];
    }
    output[idx] = sum;
}

__global__ void conv1d_update_fused_batch_bf16_kernel(const float* __restrict__ mixed_qkv,
                                                      float* __restrict__ conv_out,
                                                      float* __restrict__ conv_state,
                                                      const __nv_bfloat16* __restrict__ conv_weight,
                                                      int conv_dim, int conv_kernel, int batch_size) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (d >= conv_dim || b >= batch_size) return;

    const float* mqkv = mixed_qkv + b * conv_dim;
    float* cout = conv_out + b * conv_dim;
    float* cstate = conv_state + b * conv_dim * (conv_kernel - 1);
    float sum = __bfloat162float(conv_weight[d * conv_kernel + (conv_kernel - 1)]) * mqkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += __bfloat162float(conv_weight[d * conv_kernel + k]) * cstate[d * (conv_kernel - 1) + k];
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
               static_cast<size_t>(conv_dim) * hidden_size_ * sizeof(__nv_bfloat16));
    cudaMalloc(&d_in_proj_a_weight_,
               static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(__nv_bfloat16));
    cudaMalloc(&d_in_proj_b_weight_,
               static_cast<size_t>(num_heads_) * hidden_size_ * sizeof(__nv_bfloat16));
    cudaMalloc(&d_in_proj_z_weight_, static_cast<size_t>(z_dim) * hidden_size_ * sizeof(__nv_bfloat16));
    cudaMalloc(&d_conv1d_weight_, static_cast<size_t>(conv_dim) * conv_kernel_ * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_proj_weight_, static_cast<size_t>(hidden_size_) * z_dim * sizeof(__nv_bfloat16));
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

    size_t qkv_size = static_cast<size_t>(conv_dim) * hidden_size_;
    size_t a_size = static_cast<size_t>(num_heads_) * hidden_size_;
    size_t z_size = static_cast<size_t>(z_dim) * hidden_size_;
    size_t conv_size = static_cast<size_t>(conv_dim) * conv_kernel_;
    size_t out_size = static_cast<size_t>(hidden_size_) * z_dim;

    std::vector<__nv_bfloat16> h_qkv_bf16(qkv_size);
    std::vector<__nv_bfloat16> h_a_bf16(a_size);
    std::vector<__nv_bfloat16> h_b_bf16(a_size);
    std::vector<__nv_bfloat16> h_z_bf16(z_size);
    std::vector<__nv_bfloat16> h_conv_bf16(conv_size);
    std::vector<__nv_bfloat16> h_out_bf16(out_size);

    for (size_t i = 0; i < qkv_size; ++i) h_qkv_bf16[i] = __float2bfloat16(in_proj_qkv_weight[i]);
    for (size_t i = 0; i < a_size; ++i) h_a_bf16[i] = __float2bfloat16(in_proj_a_weight[i]);
    for (size_t i = 0; i < a_size; ++i) h_b_bf16[i] = __float2bfloat16(in_proj_b_weight[i]);
    for (size_t i = 0; i < z_size; ++i) h_z_bf16[i] = __float2bfloat16(in_proj_z_weight[i]);
    for (size_t i = 0; i < conv_size; ++i) h_conv_bf16[i] = __float2bfloat16(conv1d_weight[i]);
    for (size_t i = 0; i < out_size; ++i) h_out_bf16[i] = __float2bfloat16(out_proj_weight[i]);

    cudaMemcpy(d_in_proj_qkv_weight_, h_qkv_bf16.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_proj_a_weight_, h_a_bf16.data(), a_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_proj_b_weight_, h_b_bf16.data(), a_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_proj_z_weight_, h_z_bf16.data(), z_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1d_weight_, h_conv_bf16.data(), conv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_proj_weight_, h_out_bf16.data(), out_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    if (!a_log.empty()) {
        cudaMemcpy(d_a_log_, a_log.data(), static_cast<size_t>(num_heads_) * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_a_log_, 0, static_cast<size_t>(num_heads_) * sizeof(float));
    }
    if (!dt_bias.empty()) {
        cudaMemcpy(d_dt_bias_, dt_bias.data(), static_cast<size_t>(num_heads_) * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_dt_bias_, 0, static_cast<size_t>(num_heads_) * sizeof(float));
    }
    if (!norm_weight.empty()) {
        cudaMemcpy(d_norm_weight_, norm_weight.data(), static_cast<size_t>(value_dim_) * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        std::vector<float> ones(value_dim_, 1.0f);
        cudaMemcpy(d_norm_weight_, ones.data(), static_cast<size_t>(value_dim_) * sizeof(float), cudaMemcpyHostToDevice);
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
    float* d_a = d_a_buf_;
    float* d_b_raw = d_b_raw_buf_;

    dim3 block(256);
    dim3 grid((conv_dim + 255) / 256);
    linear_proj_bf16_kernel<<<grid, block>>>(input, d_mixed_qkv, d_in_proj_qkv_weight_, hidden_size_, conv_dim);
    CUDA_CHECK_LAST_KERNEL();

    dim3 conv_grid((conv_dim + 255) / 256);
    conv1d_update_fused_bf16_kernel<<<conv_grid, block>>>(d_mixed_qkv, d_conv_out, state.d_conv_state,
                                                          d_conv1d_weight_, conv_dim, conv_kernel_);
    CUDA_CHECK_LAST_KERNEL();

    const float l2norm_eps = 1e-6f;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim_));
    l2norm_qk_fused_kernel<<<num_heads_, 1>>>(d_conv_out, d_conv_out + num_heads_ * key_dim_,
                                               num_heads_, key_dim_, q_scale, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    dim3 small_grid((num_heads_ + 255) / 256);
    linear_proj_bf16_kernel<<<small_grid, block>>>(input, d_a, d_in_proj_a_weight_, hidden_size_, num_heads_);
    CUDA_CHECK_LAST_KERNEL();
    linear_proj_bf16_kernel<<<small_grid, block>>>(input, d_b_raw, d_in_proj_b_weight_, hidden_size_, num_heads_);
    CUDA_CHECK_LAST_KERNEL();

    dim3 z_grid((z_dim + 255) / 256);
    linear_proj_bf16_kernel<<<z_grid, block>>>(input, d_z_buf_, d_in_proj_z_weight_, hidden_size_, z_dim);
    CUDA_CHECK_LAST_KERNEL();

    float* d_attn_out = d_attn_out_buf_;

    dim3 update_grid(num_heads_);
    size_t shared_mem = (2 + 2 * value_dim_) * sizeof(float);
    gated_delta_kernel<<<update_grid, value_dim_, shared_mem>>>(
        d_conv_out + num_heads_ * key_dim_,
        d_conv_out + num_heads_ * key_dim_ + k_dim,
        d_conv_out,
        d_a, d_b_raw, d_a_log_, d_dt_bias_, state.d_recurrent_state, d_attn_out,
        num_heads_, key_dim_, value_dim_);
    CUDA_CHECK_LAST_KERNEL();

    size_t norm_gate_shmem = sizeof(float);
    norm_gate_fused_kernel<<<num_heads_, 128, norm_gate_shmem>>>(
        d_attn_out, d_norm_weight_, d_z_buf_, num_heads_, value_dim_, l2norm_eps);
    CUDA_CHECK_LAST_KERNEL();

    dim3 proj_block(256);
    dim3 proj_grid((hidden_size_ + 255) / 256);
    la_output_proj_bf16_kernel<<<proj_grid, proj_block>>>(d_attn_out, output, d_out_proj_weight_, z_dim, hidden_size_);
    CUDA_CHECK_LAST_KERNEL();
}

void CudaLinearAttention::forward_batch(const float* input, float* output,
                                        CudaLinearAttnState& state, int batch_size) const {
    cublasHandle_t handle = CublasHandlePool::instance().get();

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
    cudaMemcpy(d_batch_conv_state, state.d_conv_state, conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice);
    for (int b = 1; b < batch_size; ++b) {
        cudaMemcpy(d_batch_conv_state + b * conv_state_per_token, state.d_conv_state,
                   conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    __nv_bfloat16* d_input_bf16;
    __nv_bfloat16* d_mixed_qkv_bf16;
    __nv_bfloat16* d_a_bf16;
    __nv_bfloat16* d_b_raw_bf16;
    __nv_bfloat16* d_z_bf16;
    __nv_bfloat16* d_attn_out_bf16;
    __nv_bfloat16* d_output_bf16;
    
    size_t input_size = static_cast<size_t>(batch_size) * hidden_size_;
    size_t mixed_qkv_size = static_cast<size_t>(batch_size) * conv_dim;
    size_t a_size = static_cast<size_t>(batch_size) * num_heads_;
    size_t z_size = static_cast<size_t>(batch_size) * z_dim;
    size_t output_size = static_cast<size_t>(batch_size) * hidden_size_;
    
    cudaMalloc(&d_input_bf16, input_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_mixed_qkv_bf16, mixed_qkv_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_a_bf16, a_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b_raw_bf16, a_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_z_bf16, z_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_attn_out_bf16, z_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_bf16, output_size * sizeof(__nv_bfloat16));
    
    int block = 256;
    int grid = (input_size + 255) / 256;
    fp32_to_bf16_kernel<<<grid, block>>>(input, d_input_bf16, input_size);

    const float alpha = 1.0f, beta = 0.0f;

    LA_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  conv_dim, batch_size, hidden_size_,
                                  &alpha, d_in_proj_qkv_weight_, CUDA_R_16BF, hidden_size_,
                                  d_input_bf16, CUDA_R_16BF, hidden_size_,
                                  &beta, d_mixed_qkv_bf16, CUDA_R_16BF, conv_dim,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    LA_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  num_heads_, batch_size, hidden_size_,
                                  &alpha, d_in_proj_a_weight_, CUDA_R_16BF, hidden_size_,
                                  d_input_bf16, CUDA_R_16BF, hidden_size_,
                                  &beta, d_a_bf16, CUDA_R_16BF, num_heads_,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    LA_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  num_heads_, batch_size, hidden_size_,
                                  &alpha, d_in_proj_b_weight_, CUDA_R_16BF, hidden_size_,
                                  d_input_bf16, CUDA_R_16BF, hidden_size_,
                                  &beta, d_b_raw_bf16, CUDA_R_16BF, num_heads_,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    LA_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  z_dim, batch_size, hidden_size_,
                                  &alpha, d_in_proj_z_weight_, CUDA_R_16BF, hidden_size_,
                                  d_input_bf16, CUDA_R_16BF, hidden_size_,
                                  &beta, d_z_bf16, CUDA_R_16BF, z_dim,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    grid = (mixed_qkv_size + 255) / 256;
    bf16_to_fp32_kernel<<<grid, block>>>(d_mixed_qkv_bf16, d_mixed_qkv, mixed_qkv_size);
    grid = (a_size + 255) / 256;
    bf16_to_fp32_kernel<<<grid, block>>>(d_a_bf16, d_a, a_size);
    bf16_to_fp32_kernel<<<grid, block>>>(d_b_raw_bf16, d_b_raw, a_size);
    grid = (z_size + 255) / 256;
    bf16_to_fp32_kernel<<<grid, block>>>(d_z_bf16, d_z, z_size);

    dim3 conv_block(256);
    dim3 conv_grid((conv_dim + 255) / 256, batch_size);
    conv1d_update_fused_batch_bf16_kernel<<<conv_grid, conv_block>>>(
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
        CUDA_CHECK_LAST_KERNEL();
    }

    dim3 ng_grid(num_heads_, batch_size);
    size_t ng_shmem = sizeof(float);
    norm_gate_fused_batch_kernel<<<ng_grid, 128, ng_shmem>>>(
        d_attn_out, d_norm_weight_, d_z, num_heads_, value_dim_, l2norm_eps, batch_size);
    CUDA_CHECK_LAST_KERNEL();

    grid = (z_size + 255) / 256;
    fp32_to_bf16_kernel<<<grid, block>>>(d_attn_out, d_attn_out_bf16, z_size);

    LA_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  hidden_size_, batch_size, z_dim,
                                  &alpha, d_out_proj_weight_, CUDA_R_16BF, z_dim,
                                  d_attn_out_bf16, CUDA_R_16BF, z_dim,
                                  &beta, d_output_bf16, CUDA_R_16BF, hidden_size_,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    grid = (output_size + 255) / 256;
    bf16_to_fp32_kernel<<<grid, block>>>(d_output_bf16, output, output_size);

    cudaMemcpy(state.d_conv_state, d_batch_conv_state + (batch_size - 1) * conv_state_per_token,
               conv_state_per_token * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_batch_conv_state);
    cudaFree(d_input_bf16);
    cudaFree(d_mixed_qkv_bf16);
    cudaFree(d_a_bf16);
    cudaFree(d_b_raw_bf16);
    cudaFree(d_z_bf16);
    cudaFree(d_attn_out_bf16);
    cudaFree(d_output_bf16);
}

void CudaLinearAttention::forward_fused(const float* input, float* output,
                                        CudaLinearAttnState& state) const {
    forward(input, output, state);
}

} // namespace cuda
} // namespace qwen
