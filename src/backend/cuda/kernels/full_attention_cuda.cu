#include "full_attention_cuda.hpp"
#include "flash_attention.cuh"
#include "fused_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#define FA_CUDA_CHECK()                                                                            \
    {                                                                                              \
        cudaError_t _err = cudaGetLastError();                                                     \
        if (_err != cudaSuccess) {                                                                 \
            printf("FA Kernel Error: %s\n", cudaGetErrorString(_err));                             \
        }                                                                                          \
    }

#define FA_CUBLAS_CHECK(call)                                                                      \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(_err), __FILE__, __LINE__); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

namespace qwen {
namespace cuda {

__global__ void fused_q_path_kernel(const float* __restrict__ input,
                                    const float* __restrict__ q_weight,
                                    const float* __restrict__ q_norm_weight,
                                    float* __restrict__ d_q, float* __restrict__ d_gate,
                                    int hidden_size, int num_heads, int q_head_dim, int kv_head_dim,
                                    int rotary_dim, float rope_base, int position, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    int tid = threadIdx.x;
    if (tid < q_head_dim) {
        int q_row = h * (q_head_dim * 2) + tid;
        int g_row = h * (q_head_dim * 2) + q_head_dim + tid;
        float sq = 0.0f, sg = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            float x = input[j];
            sq += q_weight[q_row * hidden_size + j] * x;
            sg += q_weight[g_row * hidden_size + j] * x;
        }
        d_q[h * q_head_dim + tid] = sq;
        d_gate[h * q_head_dim + tid] = sg;
    }
    __syncthreads();
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int d = 0; d < kv_head_dim; ++d) {
            float v = d_q[h * q_head_dim + d];
            sum_sq += v * v;
        }
        float inv_rms = rsqrtf(sum_sq / kv_head_dim + eps);
        for (int d = 0; d < kv_head_dim; ++d) {
            d_q[h * q_head_dim + d] *= inv_rms * (1.0f + q_norm_weight[d]);
        }
    }
    __syncthreads();
    if (tid < rotary_dim / 2) {
        float freq =
            1.0f / powf(rope_base, static_cast<float>(tid * 2) / static_cast<float>(rotary_dim));
        float angle = position * freq;
        float co = cosf(angle), si = sinf(angle);
        int i0 = h * q_head_dim + tid * 2;
        int i1 = i0 + 1;
        float q0 = d_q[i0], q1 = d_q[i1];
        d_q[i0] = q0 * co - q1 * si;
        d_q[i1] = q0 * si + q1 * co;
    }
}

__global__ void
fused_kv_cache_kernel(const float* __restrict__ input, const float* __restrict__ k_weight,
                      const float* __restrict__ v_weight, const float* __restrict__ k_norm_weight,
                      float* __restrict__ k_cache_slot, float* __restrict__ v_cache_slot,
                      int hidden_size, int num_kv_heads, int kv_head_dim, int rotary_dim,
                      float rope_base, int position, float eps) {
    int h = blockIdx.x;
    if (h >= num_kv_heads)
        return;
    int tid = threadIdx.x;
    extern __shared__ float skv[];
    float* sk = skv;
    float* sv = skv + kv_head_dim;
    if (tid < kv_head_dim) {
        float ak = 0.0f, av = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            float x = input[j];
            ak += k_weight[(h * kv_head_dim + tid) * hidden_size + j] * x;
            av += v_weight[(h * kv_head_dim + tid) * hidden_size + j] * x;
        }
        sk[tid] = ak;
        sv[tid] = av;
    }
    __syncthreads();
    if (tid == 0) {
        float sq = 0.0f;
        for (int d = 0; d < kv_head_dim; ++d) {
            float v = sk[d];
            sq += v * v;
        }
        float inv_rms = rsqrtf(sq / kv_head_dim + eps);
        for (int d = 0; d < kv_head_dim; ++d) {
            sk[d] *= inv_rms * (1.0f + k_norm_weight[d]);
        }
    }
    __syncthreads();
    if (tid < rotary_dim / 2) {
        float freq =
            1.0f / powf(rope_base, static_cast<float>(tid * 2) / static_cast<float>(rotary_dim));
        float angle = position * freq;
        float co = cosf(angle), si = sinf(angle);
        int i0 = tid * 2, i1 = tid * 2 + 1;
        float k0 = sk[i0], k1 = sk[i1];
        sk[i0] = k0 * co - k1 * si;
        sk[i1] = k0 * si + k1 * co;
    }
    __syncthreads();
    if (tid < kv_head_dim) {
        k_cache_slot[h * kv_head_dim + tid] = sk[tid];
        v_cache_slot[h * kv_head_dim + tid] = sv[tid];
    }
}

__global__ void gate_sigmoid_kernel(float* __restrict__ d_attn_out,
                                    const float* __restrict__ d_gate, int num_heads,
                                    int kv_head_dim, int q_head_dim) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= num_heads || d >= kv_head_dim)
        return;
    float g = d_gate[h * q_head_dim + d];
    d_attn_out[h * kv_head_dim + d] *= 1.0f / (1.0f + expf(-g));
}

__global__ void output_proj_kernel(const float* __restrict__ attn_out, float* __restrict__ output,
                                   const float* __restrict__ weight, int total_out,
                                   int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size)
        return;
    float sum = 0.0f;
    for (int j = 0; j < total_out; ++j) {
        sum += weight[idx * total_out + j] * attn_out[j];
    }
    output[idx] = sum;
}

CudaFullAttention::CudaFullAttention(int hidden_size, int num_heads, int num_kv_heads,
                                     int q_head_dim, int kv_head_dim)
    : hidden_size_(hidden_size), num_heads_(num_heads), num_kv_heads_(num_kv_heads),
      q_head_dim_(q_head_dim), kv_head_dim_(kv_head_dim), d_q_proj_weight_(nullptr),
      d_k_proj_weight_(nullptr), d_v_proj_weight_(nullptr), d_o_proj_weight_(nullptr),
      d_q_norm_weight_(nullptr), d_k_norm_weight_(nullptr), d_q_buf_(nullptr),
      d_gate_buf_(nullptr), d_k_buf_(nullptr), d_v_buf_(nullptr), d_attn_out_buf_(nullptr),
      d_attn_scores_buf_(nullptr), max_seq_len_(8192), d_batch_q_buf_(nullptr),
      d_batch_gate_buf_(nullptr), d_batch_k_buf_(nullptr), d_batch_v_buf_(nullptr),
      d_batch_attn_out_buf_(nullptr), max_batch_size_(0), cublas_handle_(nullptr) {
    int total_q = num_heads_ * q_head_dim_;
    int total_kv = num_kv_heads_ * kv_head_dim_;
    int total_out = num_heads_ * kv_head_dim_;

    cudaMalloc(&d_q_proj_weight_, static_cast<size_t>(total_q * 2) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_k_proj_weight_, static_cast<size_t>(total_kv) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_v_proj_weight_, static_cast<size_t>(total_kv) * hidden_size_ * sizeof(float));
    cudaMalloc(&d_o_proj_weight_, static_cast<size_t>(hidden_size_) * total_out * sizeof(float));
    cudaMalloc(&d_q_norm_weight_, static_cast<size_t>(kv_head_dim_) * sizeof(float));
    cudaMalloc(&d_k_norm_weight_, static_cast<size_t>(kv_head_dim_) * sizeof(float));

    cudaMalloc(&d_q_buf_, static_cast<size_t>(total_q) * sizeof(float));
    cudaMalloc(&d_gate_buf_, static_cast<size_t>(total_q) * sizeof(float));
    cudaMalloc(&d_k_buf_, static_cast<size_t>(total_kv) * sizeof(float));
    cudaMalloc(&d_v_buf_, static_cast<size_t>(total_kv) * sizeof(float));
    cudaMalloc(&d_attn_out_buf_, static_cast<size_t>(total_out) * sizeof(float));
    cudaMalloc(&d_attn_scores_buf_, static_cast<size_t>(num_heads_) * max_seq_len_ * sizeof(float));

    FA_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    FA_CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
}

CudaFullAttention::~CudaFullAttention() {
    if (d_q_proj_weight_)
        cudaFree(d_q_proj_weight_);
    if (d_k_proj_weight_)
        cudaFree(d_k_proj_weight_);
    if (d_v_proj_weight_)
        cudaFree(d_v_proj_weight_);
    if (d_o_proj_weight_)
        cudaFree(d_o_proj_weight_);
    if (d_q_norm_weight_)
        cudaFree(d_q_norm_weight_);
    if (d_k_norm_weight_)
        cudaFree(d_k_norm_weight_);
    if (d_q_buf_)
        cudaFree(d_q_buf_);
    if (d_gate_buf_)
        cudaFree(d_gate_buf_);
    if (d_k_buf_)
        cudaFree(d_k_buf_);
    if (d_v_buf_)
        cudaFree(d_v_buf_);
    if (d_attn_out_buf_)
        cudaFree(d_attn_out_buf_);
    if (d_attn_scores_buf_)
        cudaFree(d_attn_scores_buf_);
    if (cublas_handle_)
        cublasDestroy(cublas_handle_);
}

void CudaFullAttention::set_weights(const std::vector<float>& q_proj_weight,
                                    const std::vector<float>& k_proj_weight,
                                    const std::vector<float>& v_proj_weight,
                                    const std::vector<float>& q_norm_weight,
                                    const std::vector<float>& k_norm_weight,
                                    const std::vector<float>& o_proj_weight) {
    int total_q = num_heads_ * q_head_dim_;
    int total_kv = num_kv_heads_ * kv_head_dim_;
    int total_out = num_heads_ * kv_head_dim_;

    cudaMemcpy(d_q_proj_weight_, q_proj_weight.data(),
               static_cast<size_t>(total_q * 2) * hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_proj_weight_, k_proj_weight.data(),
               static_cast<size_t>(total_kv) * hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_proj_weight_, v_proj_weight.data(),
               static_cast<size_t>(total_kv) * hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_proj_weight_, o_proj_weight.data(),
               static_cast<size_t>(hidden_size_) * total_out * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_norm_weight_, q_norm_weight.data(), kv_head_dim_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_norm_weight_, k_norm_weight.data(), kv_head_dim_ * sizeof(float),
               cudaMemcpyHostToDevice);
}

void CudaKVCache::reset(int nl, int nkh, int hd, int max_len) {
    num_layers = nl;
    num_kv_heads = nkh;
    head_dim = hd;
    max_seq_len = max_len;
    layer_lengths.assign(num_layers, 0);

    size_t total = static_cast<size_t>(nl) * max_len * nkh * hd;
    cudaMalloc(&d_k_cache, total * sizeof(float));
    cudaMalloc(&d_v_cache, total * sizeof(float));
    cudaMemset(d_k_cache, 0, total * sizeof(float));
    cudaMemset(d_v_cache, 0, total * sizeof(float));
}

void CudaKVCache::clear() {
    if (d_k_cache) {
        size_t total = static_cast<size_t>(num_layers) * max_seq_len * num_kv_heads * head_dim;
        cudaMemset(d_k_cache, 0, total * sizeof(float));
    }
    if (d_v_cache) {
        size_t total = static_cast<size_t>(num_layers) * max_seq_len * num_kv_heads * head_dim;
        cudaMemset(d_v_cache, 0, total * sizeof(float));
    }
    std::fill(layer_lengths.begin(), layer_lengths.end(), 0);
}

void CudaFullAttention::forward(const float* input, float* output, CudaKVCache& kv_cache,
                                int layer_idx, int position) const {
    int total_q = num_heads_ * q_head_dim_;
    int total_kv = num_kv_heads_ * kv_head_dim_;
    int total_out = num_heads_ * kv_head_dim_;
    int seq_len = kv_cache.length(layer_idx) + 1;
    if (seq_len > max_seq_len_) {
        seq_len = max_seq_len_;
    }

    float* d_q = d_q_buf_;
    float* d_gate = d_gate_buf_;
    float* d_attn_out = d_attn_out_buf_;

    int rotary_dim = static_cast<int>(kv_head_dim_ * 0.25f);

    fused_q_path_kernel<<<num_heads_, 256>>>(input, d_q_proj_weight_, d_q_norm_weight_, d_q, d_gate,
                                             hidden_size_, num_heads_, q_head_dim_, kv_head_dim_,
                                             rotary_dim, 10000000.0f, position, 1e-6f);
    FA_CUDA_CHECK();

    size_t k_offset =
        static_cast<size_t>(layer_idx) * kv_cache.max_seq_len * num_kv_heads_ * kv_head_dim_ +
        static_cast<size_t>(position) * num_kv_heads_ * kv_head_dim_;
    size_t shmem_kv = 2 * kv_head_dim_ * sizeof(float);
    fused_kv_cache_kernel<<<num_kv_heads_, 256, shmem_kv>>>(
        input, d_k_proj_weight_, d_v_proj_weight_, d_k_norm_weight_, kv_cache.d_k_cache + k_offset,
        kv_cache.d_v_cache + k_offset, hidden_size_, num_kv_heads_, kv_head_dim_, rotary_dim,
        10000000.0f, position, 1e-6f);
    FA_CUDA_CHECK();
    kv_cache.layer_lengths[layer_idx] += 1;

    const float* k_ptr = kv_cache.d_k_cache + static_cast<size_t>(layer_idx) *
                                                  kv_cache.max_seq_len * num_kv_heads_ *
                                                  kv_head_dim_;
    const float* v_ptr = kv_cache.d_v_cache + static_cast<size_t>(layer_idx) *
                                                  kv_cache.max_seq_len * num_kv_heads_ *
                                                  kv_head_dim_;

    float attn_scale = 1.0f / sqrtf(static_cast<float>(q_head_dim_));
    flash_attention_decode(d_q, k_ptr, v_ptr, d_attn_out, num_heads_, num_kv_heads_, q_head_dim_,
                           kv_head_dim_, seq_len, attn_scale);
    FA_CUDA_CHECK();

    gate_sigmoid_kernel<<<num_heads_, kv_head_dim_>>>(d_attn_out, d_gate, num_heads_, kv_head_dim_,
                                                      q_head_dim_);
    FA_CUDA_CHECK();

    dim3 block_proj(256);
    dim3 grid_proj((hidden_size_ + 255) / 256);
    output_proj_kernel<<<grid_proj, block_proj>>>(d_attn_out, output, d_o_proj_weight_, total_out,
                                                  hidden_size_);
    FA_CUDA_CHECK();
}

__global__ void batch_fused_q_path_kernel(const float* __restrict__ input,
                                          const float* __restrict__ q_weight,
                                          const float* __restrict__ q_norm_weight,
                                          float* __restrict__ d_q, float* __restrict__ d_gate,
                                          int hidden_size, int num_heads, int q_head_dim,
                                          int kv_head_dim, int rotary_dim, float rope_base,
                                          const int* positions, float eps, int batch_size) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= num_heads || b >= batch_size)
        return;

    int tid = threadIdx.x;
    const float* in_ptr = input + b * hidden_size;
    float* q_ptr = d_q + b * num_heads * q_head_dim;
    float* g_ptr = d_gate + b * num_heads * q_head_dim;

    if (tid < q_head_dim) {
        int q_row = h * (q_head_dim * 2) + tid;
        int g_row = h * (q_head_dim * 2) + q_head_dim + tid;
        float sq = 0.0f, sg = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            float x = in_ptr[j];
            sq += q_weight[q_row * hidden_size + j] * x;
            sg += q_weight[g_row * hidden_size + j] * x;
        }
        q_ptr[h * q_head_dim + tid] = sq;
        g_ptr[h * q_head_dim + tid] = sg;
    }
    __syncthreads();

    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int d = 0; d < kv_head_dim; ++d) {
            float v = q_ptr[h * q_head_dim + d];
            sum_sq += v * v;
        }
        float inv_rms = rsqrtf(sum_sq / kv_head_dim + eps);
        for (int d = 0; d < kv_head_dim; ++d) {
            q_ptr[h * q_head_dim + d] *= inv_rms * (1.0f + q_norm_weight[d]);
        }
    }
    __syncthreads();

    if (tid < rotary_dim / 2) {
        float freq =
            1.0f / powf(rope_base, static_cast<float>(tid * 2) / static_cast<float>(rotary_dim));
        float angle = positions[b] * freq;
        float co = cosf(angle), si = sinf(angle);
        int i0 = h * q_head_dim + tid * 2;
        int i1 = i0 + 1;
        float q0 = q_ptr[i0], q1 = q_ptr[i1];
        q_ptr[i0] = q0 * co - q1 * si;
        q_ptr[i1] = q0 * si + q1 * co;
    }
}

__global__ void batch_fused_kv_cache_kernel(const float* __restrict__ input,
                                            const float* __restrict__ k_weight,
                                            const float* __restrict__ v_weight,
                                            const float* __restrict__ k_norm_weight,
                                            float* __restrict__ k_cache,
                                            float* __restrict__ v_cache, int hidden_size,
                                            int num_kv_heads, int kv_head_dim, int rotary_dim,
                                            float rope_base, const int* positions, float eps,
                                            int batch_size, int layer_idx, int max_seq_len) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= num_kv_heads || b >= batch_size)
        return;

    int tid = threadIdx.x;
    const float* in_ptr = input + b * hidden_size;
    int pos = positions[b];

    extern __shared__ float skv[];
    float* sk = skv;
    float* sv = skv + kv_head_dim;

    if (tid < kv_head_dim) {
        float ak = 0.0f, av = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            float x = in_ptr[j];
            ak += k_weight[(h * kv_head_dim + tid) * hidden_size + j] * x;
            av += v_weight[(h * kv_head_dim + tid) * hidden_size + j] * x;
        }
        sk[tid] = ak;
        sv[tid] = av;
    }
    __syncthreads();

    if (tid == 0) {
        float sq = 0.0f;
        for (int d = 0; d < kv_head_dim; ++d) {
            float v = sk[d];
            sq += v * v;
        }
        float inv_rms = rsqrtf(sq / kv_head_dim + eps);
        for (int d = 0; d < kv_head_dim; ++d) {
            sk[d] *= inv_rms * (1.0f + k_norm_weight[d]);
        }
    }
    __syncthreads();

    if (tid < rotary_dim / 2) {
        float freq =
            1.0f / powf(rope_base, static_cast<float>(tid * 2) / static_cast<float>(rotary_dim));
        float angle = pos * freq;
        float co = cosf(angle), si = sinf(angle);
        int i0 = tid * 2, i1 = tid * 2 + 1;
        float k0 = sk[i0], k1 = sk[i1];
        sk[i0] = k0 * co - k1 * si;
        sk[i1] = k0 * si + k1 * co;
    }
    __syncthreads();

    if (tid < kv_head_dim) {
        size_t cache_offset = static_cast<size_t>(layer_idx) * max_seq_len * num_kv_heads *
                                  kv_head_dim +
                              static_cast<size_t>(pos) * num_kv_heads * kv_head_dim +
                              h * kv_head_dim + tid;
        k_cache[cache_offset] = sk[tid];
        v_cache[cache_offset] = sv[tid];
    }
}

__global__ void batch_gate_sigmoid_kernel(float* __restrict__ d_attn_out,
                                          const float* __restrict__ d_gate, int num_heads,
                                          int kv_head_dim, int q_head_dim, int batch_size) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int d = threadIdx.x;
    if (h >= num_heads || b >= batch_size || d >= kv_head_dim)
        return;

    int idx = b * num_heads * kv_head_dim + h * kv_head_dim + d;
    float g = d_gate[b * num_heads * q_head_dim + h * q_head_dim + d];
    d_attn_out[idx] *= 1.0f / (1.0f + expf(-g));
}

__global__ void batch_output_proj_kernel(const float* __restrict__ attn_out,
                                         float* __restrict__ output,
                                         const float* __restrict__ weight, int total_out,
                                         int hidden_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (idx >= hidden_size || b >= batch_size)
        return;

    float sum = 0.0f;
    const float* a_ptr = attn_out + b * total_out;
    for (int j = 0; j < total_out; ++j) {
        sum += weight[idx * total_out + j] * a_ptr[j];
    }
    output[b * hidden_size + idx] = sum;
}

void CudaFullAttention::ensure_batch_buffers(int batch_size) const {
    if (batch_size <= max_batch_size_)
        return;

    if (d_batch_q_buf_)
        cudaFree(d_batch_q_buf_);
    if (d_batch_gate_buf_)
        cudaFree(d_batch_gate_buf_);
    if (d_batch_k_buf_)
        cudaFree(d_batch_k_buf_);
    if (d_batch_v_buf_)
        cudaFree(d_batch_v_buf_);
    if (d_batch_attn_out_buf_)
        cudaFree(d_batch_attn_out_buf_);

    size_t total_q = static_cast<size_t>(batch_size) * num_heads_ * q_head_dim_;
    size_t total_kv = static_cast<size_t>(batch_size) * num_kv_heads_ * kv_head_dim_;
    size_t total_out = static_cast<size_t>(batch_size) * num_heads_ * kv_head_dim_;

    cudaMalloc(&d_batch_q_buf_, total_q * sizeof(float));
    cudaMalloc(&d_batch_gate_buf_, total_q * sizeof(float));
    cudaMalloc(&d_batch_k_buf_, total_kv * sizeof(float));
    cudaMalloc(&d_batch_v_buf_, total_kv * sizeof(float));
    cudaMalloc(&d_batch_attn_out_buf_, total_out * sizeof(float));

    max_batch_size_ = batch_size;
}

void CudaFullAttention::forward_batch_prefill(const float* input, float* output,
                                              CudaKVCache& kv_cache, int layer_idx,
                                              const int* positions, int batch_size) const {
    ensure_batch_buffers(batch_size);

    int rotary_dim = static_cast<int>(kv_head_dim_ * 0.25f);
    int total_out = num_heads_ * kv_head_dim_;

    dim3 q_grid(num_heads_, batch_size);
    batch_fused_q_path_kernel<<<q_grid, 256>>>(
        input, d_q_proj_weight_, d_q_norm_weight_, d_batch_q_buf_, d_batch_gate_buf_,
        hidden_size_, num_heads_, q_head_dim_, kv_head_dim_, rotary_dim, 10000000.0f,
        positions, 1e-6f, batch_size);
    FA_CUDA_CHECK();

    size_t shmem_kv = 2 * kv_head_dim_ * sizeof(float);
    dim3 kv_grid(num_kv_heads_, batch_size);
    batch_fused_kv_cache_kernel<<<kv_grid, 256, shmem_kv>>>(
        input, d_k_proj_weight_, d_v_proj_weight_, d_k_norm_weight_,
        kv_cache.d_k_cache, kv_cache.d_v_cache, hidden_size_, num_kv_heads_, kv_head_dim_,
        rotary_dim, 10000000.0f, positions, 1e-6f, batch_size, layer_idx, kv_cache.max_seq_len);
    FA_CUDA_CHECK();

    std::vector<int> h_positions(batch_size);
    cudaMemcpy(h_positions.data(), positions, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int b = 0; b < batch_size; ++b) {
        kv_cache.layer_lengths[layer_idx] = std::max(kv_cache.layer_lengths[layer_idx], 
                                                      h_positions[b] + 1);
    }

    int max_seq = 0;
    for (int b = 0; b < batch_size; ++b) {
        max_seq = std::max(max_seq, h_positions[b] + 1);
    }

    float attn_scale = 1.0f / sqrtf(static_cast<float>(q_head_dim_));
    launch_flash_attn_v2_prefill(
        d_batch_q_buf_, kv_cache.d_k_cache, kv_cache.d_v_cache,
        d_batch_attn_out_buf_, num_heads_, num_kv_heads_, kv_head_dim_,
        max_seq, batch_size, layer_idx, kv_cache.max_seq_len, attn_scale);
    FA_CUDA_CHECK();

    dim3 gate_grid(num_heads_, batch_size);
    batch_gate_sigmoid_kernel<<<gate_grid, kv_head_dim_>>>(
        d_batch_attn_out_buf_, d_batch_gate_buf_, num_heads_, kv_head_dim_, q_head_dim_, batch_size);
    FA_CUDA_CHECK();

    // Use cuBLAS GEMM for output projection: output = attn_out × o_weight^T
    // attn_out: [batch_size, total_out], o_weight: [hidden_size, total_out]
    // output = attn_out × o_weight^T  =>  [batch_size, total_out] × [total_out, hidden_size]
    const float alpha = 1.0f, beta = 0.0f;
    FA_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                hidden_size_, batch_size, total_out,
                                &alpha, d_o_proj_weight_, total_out,
                                d_batch_attn_out_buf_, total_out,
                                &beta, output, hidden_size_));
}

} // namespace cuda
} // namespace qwen
