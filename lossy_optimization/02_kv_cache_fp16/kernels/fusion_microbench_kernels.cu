#include "fusion_microbench_kernels.cuh"
#include "flash_attention.cuh"
#include <cmath>
#include <cstring>

namespace qwen {
namespace cuda {
namespace fusion_bench {

// ---------------------------------------------------------------------------
// Fusion #1: Q proj + Q norm + RoPE(Q)
// ---------------------------------------------------------------------------
__global__ void fb_q_proj_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                 float* __restrict__ d_q, float* __restrict__ d_gate,
                                 int hidden_size, int num_heads, int q_head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = num_heads * q_head_dim;
    if (idx >= total_q)
        return;
    int head = idx / q_head_dim;
    int dim = idx % q_head_dim;
    int q_row = head * (q_head_dim * 2) + dim;
    int g_row = head * (q_head_dim * 2) + q_head_dim + dim;
    float sum_q = 0.f, sum_g = 0.f;
    for (int j = 0; j < hidden_size; ++j) {
        float x = input[j];
        sum_q += weight[q_row * hidden_size + j] * x;
        sum_g += weight[g_row * hidden_size + j] * x;
    }
    d_q[idx] = sum_q;
    d_gate[idx] = sum_g;
}

__global__ void fb_q_norm_kernel(float* __restrict__ d_q, const float* __restrict__ norm_weight,
                                 int num_heads, int kv_head_dim, int q_head_dim, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    float sq_sum = 0.f;
    for (int d = 0; d < kv_head_dim; ++d) {
        float v = d_q[h * q_head_dim + d];
        sq_sum += v * v;
    }
    float inv_rms = rsqrtf(sq_sum / kv_head_dim + eps);
    for (int d = 0; d < kv_head_dim; ++d) {
        d_q[h * q_head_dim + d] *= inv_rms * (1.0f + norm_weight[d]);
    }
}

__global__ void fb_rope_q_only_kernel(float* __restrict__ d_q, int num_heads, int q_head_dim,
                                      int rotary_dim, float base, int position) {
    int h = blockIdx.x;
    int pair = threadIdx.x;
    if (pair >= rotary_dim / 2)
        return;
    float freq = 1.0f / powf(base, static_cast<float>(pair * 2) / static_cast<float>(rotary_dim));
    float angle = position * freq;
    float co = cosf(angle), si = sinf(angle);
    int idx0 = h * q_head_dim + pair * 2;
    int idx1 = h * q_head_dim + pair * 2 + 1;
    float q0 = d_q[idx0], q1 = d_q[idx1];
    d_q[idx0] = q0 * co - q1 * si;
    d_q[idx1] = q0 * si + q1 * co;
}

__global__ void fb_fused1_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                 const float* __restrict__ q_norm_weight, float* __restrict__ d_q,
                                 float* __restrict__ d_gate, int hidden_size, int num_heads,
                                 int q_head_dim, int kv_head_dim, int rotary_dim, float rope_base,
                                 int position, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    int tid = threadIdx.x;
    if (tid < q_head_dim) {
        int q_row = h * (q_head_dim * 2) + tid;
        int g_row = h * (q_head_dim * 2) + q_head_dim + tid;
        float sq = 0.f, sg = 0.f;
        for (int j = 0; j < hidden_size; ++j) {
            float x = input[j];
            sq += weight[q_row * hidden_size + j] * x;
            sg += weight[g_row * hidden_size + j] * x;
        }
        d_q[h * q_head_dim + tid] = sq;
        d_gate[h * q_head_dim + tid] = sg;
    }
    __syncthreads();
    if (tid == 0) {
        float sum_sq = 0.f;
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
    int pair = tid;
    if (pair < rotary_dim / 2) {
        float freq =
            1.0f / powf(rope_base, static_cast<float>(pair * 2) / static_cast<float>(rotary_dim));
        float angle = position * freq;
        float co = cosf(angle), si = sinf(angle);
        int i0 = h * q_head_dim + pair * 2;
        int i1 = i0 + 1;
        float q0 = d_q[i0], q1 = d_q[i1];
        d_q[i0] = q0 * co - q1 * si;
        d_q[i1] = q0 * si + q1 * co;
    }
}

void run_fusion1_baseline_q_path(const float* d_input, const float* d_q_weight,
                                 const float* d_q_norm_w, float* d_q, float* d_gate,
                                 int hidden_size, int num_heads, int q_head_dim, int kv_head_dim,
                                 int rotary_dim, float rope_base, int position) {
    dim3 b(256);
    dim3 g((num_heads * q_head_dim + 255) / 256);
    fb_q_proj_kernel<<<g, b>>>(d_input, d_q_weight, d_q, d_gate, hidden_size, num_heads,
                               q_head_dim);
    fb_q_norm_kernel<<<num_heads, 1>>>(d_q, d_q_norm_w, num_heads, kv_head_dim, q_head_dim, 1e-6f);
    fb_rope_q_only_kernel<<<num_heads, rotary_dim / 2>>>(d_q, num_heads, q_head_dim, rotary_dim,
                                                         rope_base, position);
}

void run_fusion1_fused_q_path(const float* d_input, const float* d_q_weight,
                              const float* d_q_norm_w, float* d_q, float* d_gate, int hidden_size,
                              int num_heads, int q_head_dim, int kv_head_dim, int rotary_dim,
                              float rope_base, int position) {
    fb_fused1_kernel<<<num_heads, 256>>>(d_input, d_q_weight, d_q_norm_w, d_q, d_gate, hidden_size,
                                         num_heads, q_head_dim, kv_head_dim, rotary_dim, rope_base,
                                         position, 1e-6f);
}

// ---------------------------------------------------------------------------
// Fusion #2: KV proj + K norm + RoPE(K) + write cache
// ---------------------------------------------------------------------------
__global__ void fb_kv_proj_kernel(const float* __restrict__ input,
                                  const float* __restrict__ k_weight,
                                  const float* __restrict__ v_weight, float* __restrict__ d_k,
                                  float* __restrict__ d_v, int hidden_size, int total_kv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_kv)
        return;
    float sk = 0.f, sv = 0.f;
    for (int j = 0; j < hidden_size; ++j) {
        float x = input[j];
        sk += k_weight[idx * hidden_size + j] * x;
        sv += v_weight[idx * hidden_size + j] * x;
    }
    d_k[idx] = sk;
    d_v[idx] = sv;
}

__global__ void fb_k_norm_kernel(float* __restrict__ d_k, const float* __restrict__ norm_weight,
                                 int num_kv_heads, int kv_head_dim, float eps) {
    int h = blockIdx.x;
    if (h >= num_kv_heads)
        return;
    float sq_sum = 0.f;
    for (int d = 0; d < kv_head_dim; ++d) {
        float v = d_k[h * kv_head_dim + d];
        sq_sum += v * v;
    }
    float inv_rms = rsqrtf(sq_sum / kv_head_dim + eps);
    for (int d = 0; d < kv_head_dim; ++d) {
        d_k[h * kv_head_dim + d] *= inv_rms * (1.0f + norm_weight[d]);
    }
}

__global__ void fb_rope_k_only_kernel(float* __restrict__ d_k, int num_kv_heads, int kv_head_dim,
                                      int rotary_dim, float base, int position) {
    int h = blockIdx.x;
    int pair = threadIdx.x;
    if (pair >= rotary_dim / 2)
        return;
    float freq = 1.0f / powf(base, static_cast<float>(pair * 2) / static_cast<float>(rotary_dim));
    float angle = position * freq;
    float co = cosf(angle), si = sinf(angle);
    int i0 = h * kv_head_dim + pair * 2;
    int i1 = i0 + 1;
    float k0 = d_k[i0], k1 = d_k[i1];
    d_k[i0] = k0 * co - k1 * si;
    d_k[i1] = k0 * si + k1 * co;
}

__global__ void fb_fused2_kernel(const float* __restrict__ input,
                                 const float* __restrict__ k_weight,
                                 const float* __restrict__ v_weight,
                                 const float* __restrict__ k_norm_weight,
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
        float ak = 0.f, av = 0.f;
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
        float sq = 0.f;
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
    int pair = tid;
    if (pair < rotary_dim / 2) {
        float freq =
            1.0f / powf(rope_base, static_cast<float>(pair * 2) / static_cast<float>(rotary_dim));
        float angle = position * freq;
        float co = cosf(angle), si = sinf(angle);
        int i0 = pair * 2, i1 = pair * 2 + 1;
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

void run_fusion2_baseline_kv_cache(const float* d_input, const float* d_k_w, const float* d_v_w,
                                   const float* d_k_norm_w, float* d_k, float* d_v,
                                   float* d_k_cache, float* d_v_cache, size_t k_offset_elems,
                                   int hidden_size, int num_kv_heads, int kv_head_dim,
                                   int rotary_dim, float rope_base, int position) {
    int total_kv = num_kv_heads * kv_head_dim;
    dim3 b(256);
    dim3 g((total_kv + 255) / 256);
    fb_kv_proj_kernel<<<g, b>>>(d_input, d_k_w, d_v_w, d_k, d_v, hidden_size, total_kv);
    fb_k_norm_kernel<<<num_kv_heads, 1>>>(d_k, d_k_norm_w, num_kv_heads, kv_head_dim, 1e-6f);
    fb_rope_k_only_kernel<<<num_kv_heads, rotary_dim / 2>>>(d_k, num_kv_heads, kv_head_dim,
                                                            rotary_dim, rope_base, position);
    cudaMemcpy(d_k_cache + k_offset_elems, d_k, total_kv * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_v_cache + k_offset_elems, d_v, total_kv * sizeof(float), cudaMemcpyDeviceToDevice);
}

void run_fusion2_fused_kv_cache(const float* d_input, const float* d_k_w, const float* d_v_w,
                                const float* d_k_norm_w, float* /*d_k*/, float* /*d_v*/,
                                float* d_k_cache, float* d_v_cache, size_t k_offset_elems,
                                int hidden_size, int num_kv_heads, int kv_head_dim, int rotary_dim,
                                float rope_base, int position) {
    size_t shmem = 2 * kv_head_dim * sizeof(float);
    fb_fused2_kernel<<<num_kv_heads, 256, shmem>>>(
        d_input, d_k_w, d_v_w, d_k_norm_w, d_k_cache + k_offset_elems, d_v_cache + k_offset_elems,
        hidden_size, num_kv_heads, kv_head_dim, rotary_dim, rope_base, position, 1e-6f);
}

// ---------------------------------------------------------------------------
// Fusion #3: (score+softmax+attn) vs flash; gate+o_proj fused vs 2 kernels
// ---------------------------------------------------------------------------
__global__ void fb_attn_score_kernel(const float* __restrict__ d_q, const float* __restrict__ k_ptr,
                                     float* __restrict__ d_scores, int num_heads, int num_kv_heads,
                                     int kv_head_dim, int q_head_dim, int seq_len) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    int kv_h = h * num_kv_heads / num_heads;
    float scale = 1.0f / sqrtf(static_cast<float>(kv_head_dim));
    for (int p = 0; p < seq_len; ++p) {
        float score = 0.f;
        for (int d = 0; d < kv_head_dim; ++d) {
            score += d_q[h * q_head_dim + d] *
                     k_ptr[p * num_kv_heads * kv_head_dim + kv_h * kv_head_dim + d];
        }
        d_scores[h * seq_len + p] = score * scale;
    }
}

__global__ void fb_softmax_kernel(float* scores, int num_heads, int seq_len) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    float* s = scores + h * seq_len;
    float max_s = s[0];
    for (int i = 1; i < seq_len; ++i)
        max_s = fmaxf(max_s, s[i]);
    float sum_exp = 0.f;
    for (int i = 0; i < seq_len; ++i) {
        s[i] = expf(s[i] - max_s);
        sum_exp += s[i];
    }
    for (int i = 0; i < seq_len; ++i)
        s[i] /= sum_exp;
}

__global__ void fb_attn_out_kernel(const float* __restrict__ attn_scores,
                                   const float* __restrict__ v_ptr, float* __restrict__ d_attn_out,
                                   int num_heads, int num_kv_heads, int kv_head_dim, int q_head_dim,
                                   int seq_len) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= num_heads || d >= kv_head_dim)
        return;
    int kv_h = h * num_kv_heads / num_heads;
    float val = 0.f;
    for (int p = 0; p < seq_len; ++p) {
        val += attn_scores[h * seq_len + p] *
               v_ptr[p * num_kv_heads * kv_head_dim + kv_h * kv_head_dim + d];
    }
    d_attn_out[h * kv_head_dim + d] = val;
}

void run_fusion3_baseline_attn_core(const float* d_q, const float* k_ptr, const float* v_ptr,
                                    float* d_scores, float* d_attn_out, int num_heads,
                                    int num_kv_heads, int kv_head_dim, int q_head_dim,
                                    int seq_len) {
    fb_attn_score_kernel<<<num_heads, 1>>>(d_q, k_ptr, d_scores, num_heads, num_kv_heads,
                                           kv_head_dim, q_head_dim, seq_len);
    fb_softmax_kernel<<<num_heads, 1>>>(d_scores, num_heads, seq_len);
    fb_attn_out_kernel<<<num_heads, kv_head_dim>>>(d_scores, v_ptr, d_attn_out, num_heads,
                                                   num_kv_heads, kv_head_dim, q_head_dim, seq_len);
}

void run_fusion3_flash_attn_core(const float* d_q, const float* k_ptr, const float* v_ptr,
                                 float* d_attn_out, int num_heads, int num_kv_heads, int head_dim,
                                 int seq_len) {
    flash_attention_decode(d_q, k_ptr, v_ptr, d_attn_out, num_heads, num_kv_heads, head_dim,
                           head_dim, seq_len);
}

__global__ void fb_gate_kernel(float* d_attn, const float* d_gate, int num_heads, int kv_head_dim,
                               int q_head_dim) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= num_heads || d >= kv_head_dim)
        return;
    float g = d_gate[h * q_head_dim + d];
    d_attn[h * kv_head_dim + d] *= 1.0f / (1.0f + expf(-g));
}

__global__ void fb_out_proj_kernel(const float* attn, float* out, const float* weight,
                                   int total_out, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size)
        return;
    float sum = 0.f;
    for (int j = 0; j < total_out; ++j)
        sum += weight[idx * total_out + j] * attn[j];
    out[idx] = sum;
}

__global__ void fb_gate_o_fused_kernel(const float* __restrict__ d_attn_pre_gate,
                                       const float* __restrict__ d_gate,
                                       const float* __restrict__ o_weight,
                                       float* __restrict__ output, int num_heads, int kv_head_dim,
                                       int q_head_dim, int total_out, int hidden_size) {
    extern __shared__ float s_gated[];
    for (int j = threadIdx.x; j < total_out; j += blockDim.x) {
        int h = j / kv_head_dim;
        int dd = j - h * kv_head_dim;
        float a = d_attn_pre_gate[j];
        float g = d_gate[h * q_head_dim + dd];
        s_gated[j] = a * (1.0f / (1.0f + expf(-g)));
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size)
        return;
    float sum = 0.f;
    for (int j = 0; j < total_out; ++j) {
        sum += o_weight[idx * total_out + j] * s_gated[j];
    }
    output[idx] = sum;
}

void run_fusion3_gate_o_baseline(float* d_attn_out, const float* d_gate, const float* d_o_weight,
                                 float* d_output, int num_heads, int kv_head_dim, int q_head_dim,
                                 int total_out, int hidden_size) {
    fb_gate_kernel<<<num_heads, kv_head_dim>>>(d_attn_out, d_gate, num_heads, kv_head_dim,
                                               q_head_dim);
    dim3 b(256);
    dim3 g((hidden_size + 255) / 256);
    fb_out_proj_kernel<<<g, b>>>(d_attn_out, d_output, d_o_weight, total_out, hidden_size);
}

void run_fusion3_gate_o_fused(float* d_attn_pre_gate, const float* d_gate, const float* d_o_weight,
                              float* d_output, int num_heads, int kv_head_dim, int q_head_dim,
                              int total_out, int hidden_size) {
    dim3 b(256);
    dim3 g((hidden_size + 255) / 256);
    size_t sh = static_cast<size_t>(total_out) * sizeof(float);
    fb_gate_o_fused_kernel<<<g, b, sh>>>(d_attn_pre_gate, d_gate, d_o_weight, d_output, num_heads,
                                         kv_head_dim, q_head_dim, total_out, hidden_size);
}

// ---------------------------------------------------------------------------
// Fusion #4: RMSNorm + first rows of linear (Input norm + LA prologue)
// ---------------------------------------------------------------------------
__global__ void fb_rms_part(const float* in, const float* nw, float* normed, int hidden_size,
                            float eps) {
    int tid = threadIdx.x;
    extern __shared__ float s[];
    float p = 0.f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = in[i];
        p += v * v;
    }
    s[tid] = p;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s[tid] += s[tid + stride];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        normed[i] = in[i] * inv * (1.f + nw[i]);
    }
}

__global__ void fb_matvec_rows(const float* normed, const float* W, float* out, int hidden_size,
                               int out_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim)
        return;
    float acc = 0.f;
    for (int j = 0; j < hidden_size; ++j)
        acc += W[row * hidden_size + j] * normed[j];
    out[row] = acc;
}

__global__ void fb_fused_rms_linear_rows_tiled(const float* in, const float* nw, const float* W,
                                               float* out, int hidden_size, int out_dim,
                                               float eps) {
    extern __shared__ float sbuf[];
    float* s_part = sbuf;                // blockDim.x
    float* s_normed = sbuf + blockDim.x; // hidden_size

    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = in[i];
        sum_sq += v * v;
    }
    s_part[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_part[threadIdx.x] += s_part[threadIdx.x + stride];
        __syncthreads();
    }

    float inv = rsqrtf(s_part[0] / static_cast<float>(hidden_size) + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        s_normed[i] = in[i] * inv * (1.f + nw[i]);
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim)
        return;
    float acc = 0.f;
    for (int j = 0; j < hidden_size; ++j) {
        acc += W[row * hidden_size + j] * s_normed[j];
    }
    out[row] = acc;
}

void run_fusion4_rmsnorm_then_linear_head(const float* d_input, const float* d_norm_w,
                                          const float* d_W, float* d_out_partial,
                                          float* d_tmp_normed, int hidden_size,
                                          int out_dim_partial) {
    size_t sh = 256 * sizeof(float);
    fb_rms_part<<<1, 256, sh>>>(d_input, d_norm_w, d_tmp_normed, hidden_size, 1e-6f);
    dim3 g((out_dim_partial + 255) / 256);
    dim3 b(256);
    fb_matvec_rows<<<g, b>>>(d_tmp_normed, d_W, d_out_partial, hidden_size, out_dim_partial);
}

void run_fusion4_fused_rmsnorm_linear_head(const float* d_input, const float* d_norm_w,
                                           const float* d_W, float* d_out_partial, int hidden_size,
                                           int out_dim_partial) {
    dim3 b(256);
    dim3 g((out_dim_partial + b.x - 1) / b.x);
    size_t sh = static_cast<size_t>(b.x + hidden_size) * sizeof(float);
    fb_fused_rms_linear_rows_tiled<<<g, b, sh>>>(d_input, d_norm_w, d_W, d_out_partial, hidden_size,
                                                 out_dim_partial, 1e-6f);
}

// ---------------------------------------------------------------------------
// Fusion #5: MLP gate+up+SiLU* (3 kernels -> 2)
// ---------------------------------------------------------------------------
__global__ void fb_mlp_gu(const float* in, float* gate, float* up, const float* gw, const float* uw,
                          int hs, int isz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= isz)
        return;
    float g = 0.f, u = 0.f;
    for (int j = 0; j < hs; ++j) {
        float x = in[j];
        g += gw[i * hs + j] * x;
        u += uw[i * hs + j] * x;
    }
    gate[i] = g / (1.f + expf(-g));
    up[i] = u;
}

__global__ void fb_mlp_hid(const float* g, const float* u, float* h, int isz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= isz)
        return;
    h[i] = g[i] * u[i];
}

__global__ void fb_mlp_dn(const float* h, float* out, const float* dw, int isz, int hs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hs)
        return;
    float s = 0.f;
    for (int j = 0; j < isz; ++j)
        s += dw[i * isz + j] * h[j];
    out[i] = s;
}

__global__ void fb_mlp_f5_fused_gsilu(const float* in, float* hidden, const float* gw,
                                      const float* uw, int hs, int isz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= isz)
        return;
    float g = 0.f, u = 0.f;
    for (int j = 0; j < hs; ++j) {
        float x = in[j];
        g += gw[i * hs + j] * x;
        u += uw[i * hs + j] * x;
    }
    hidden[i] = (g / (1.f + expf(-g))) * u;
}

void run_fusion5_mlp_baseline_chain(const float* d_input, const float* d_gate_w,
                                    const float* d_up_w, const float* d_down_w, float* d_gate,
                                    float* d_up, float* d_hidden, float* d_output, int hidden_size,
                                    int intermediate_size) {
    dim3 b(256);
    dim3 g((intermediate_size + 255) / 256);
    fb_mlp_gu<<<g, b>>>(d_input, d_gate, d_up, d_gate_w, d_up_w, hidden_size, intermediate_size);
    fb_mlp_hid<<<g, b>>>(d_gate, d_up, d_hidden, intermediate_size);
    dim3 g2((hidden_size + 255) / 256);
    fb_mlp_dn<<<g2, b>>>(d_hidden, d_output, d_down_w, intermediate_size, hidden_size);
}

void run_fusion5_mlp_fused_gate_silu(const float* d_input, const float* d_gate_w,
                                     const float* d_up_w, const float* d_down_w, float* d_hidden,
                                     float* d_output, int hidden_size, int intermediate_size) {
    dim3 b(256);
    dim3 g((intermediate_size + 255) / 256);
    fb_mlp_f5_fused_gsilu<<<g, b>>>(d_input, d_hidden, d_gate_w, d_up_w, hidden_size,
                                    intermediate_size);
    dim3 g2((hidden_size + 255) / 256);
    fb_mlp_dn<<<g2, b>>>(d_hidden, d_output, d_down_w, intermediate_size, hidden_size);
}

// ---------------------------------------------------------------------------
// Fusion #6: post RMSNorm + MLP + residual (chain vs fused MLP tail)
// ---------------------------------------------------------------------------
__global__ void fb_add(const float* a, const float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    o[i] = a[i] + b[i];
}

void run_fusion6_chain_postnorm_mlp_residual(const float* d_residual_in, const float* d_post_norm_w,
                                             const float* d_gate_w, const float* d_up_w,
                                             const float* d_down_w, float* d_tmp_normed,
                                             float* d_tmp_gate, float* d_tmp_up,
                                             float* d_tmp_hidden, float* d_mlp_out, int hidden_size,
                                             int intermediate_size) {
    size_t sh = 256 * sizeof(float);
    fb_rms_part<<<1, 256, sh>>>(d_residual_in, d_post_norm_w, d_tmp_normed, hidden_size, 1e-6f);
    run_fusion5_mlp_baseline_chain(d_tmp_normed, d_gate_w, d_up_w, d_down_w, d_tmp_gate, d_tmp_up,
                                   d_tmp_hidden, d_mlp_out, hidden_size, intermediate_size);
    dim3 b(256), g((hidden_size + 255) / 256);
    fb_add<<<g, b>>>(d_residual_in, d_mlp_out, d_mlp_out, hidden_size);
}

void run_fusion6_fused_postnorm_mlp_residual(const float* d_residual_in, const float* d_post_norm_w,
                                             const float* d_gate_w, const float* d_up_w,
                                             const float* d_down_w, float* d_tmp_normed,
                                             float* d_hidden, float* d_mlp_out, int hidden_size,
                                             int intermediate_size) {
    size_t sh = 256 * sizeof(float);
    fb_rms_part<<<1, 256, sh>>>(d_residual_in, d_post_norm_w, d_tmp_normed, hidden_size, 1e-6f);
    run_fusion5_mlp_fused_gate_silu(d_tmp_normed, d_gate_w, d_up_w, d_down_w, d_hidden, d_mlp_out,
                                    hidden_size, intermediate_size);
    dim3 b(256), g((hidden_size + 255) / 256);
    fb_add<<<g, b>>>(d_residual_in, d_mlp_out, d_mlp_out, hidden_size);
}

// ---------------------------------------------------------------------------
// Fusion #7: LA conv1d + state update; L2 Q+K single launch
// ---------------------------------------------------------------------------
__global__ void fb_conv1d_kernel(const float* mixed_qkv, float* conv_out, const float* conv_weight,
                                 const float* conv_state, int conv_dim, int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;
    float sum = conv_weight[d * conv_kernel + (conv_kernel - 1)] * mixed_qkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += conv_weight[d * conv_kernel + k] * conv_state[d * (conv_kernel - 1) + k];
    }
    conv_out[d] = sum / (1.f + expf(-sum));
}

__global__ void fb_update_conv_state_kernel(const float* mixed_qkv, float* conv_state, int conv_dim,
                                            int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;
    for (int k = conv_kernel - 2; k > 0; --k) {
        conv_state[d * (conv_kernel - 1) + k] = conv_state[d * (conv_kernel - 1) + k - 1];
    }
    conv_state[d * (conv_kernel - 1)] = mixed_qkv[d];
}

__global__ void fb_conv1d_update_fused(const float* mixed_qkv, float* conv_out, float* conv_state,
                                       const float* conv_weight, int conv_dim, int conv_kernel) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim)
        return;
    float sum = conv_weight[d * conv_kernel + (conv_kernel - 1)] * mixed_qkv[d];
    for (int k = 0; k < conv_kernel - 1; ++k) {
        sum += conv_weight[d * conv_kernel + k] * conv_state[d * (conv_kernel - 1) + k];
    }
    conv_out[d] = sum / (1.f + expf(-sum));
    for (int k = conv_kernel - 2; k > 0; --k) {
        conv_state[d * (conv_kernel - 1) + k] = conv_state[d * (conv_kernel - 1) + k - 1];
    }
    conv_state[d * (conv_kernel - 1)] = mixed_qkv[d];
}

__global__ void fb_l2_q(float* data, int num_heads, int head_dim, float scale, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    float l2 = 0.f;
    for (int d = 0; d < head_dim; ++d) {
        float v = data[h * head_dim + d];
        l2 += v * v;
    }
    l2 = sqrtf(l2 + eps);
    for (int d = 0; d < head_dim; ++d) {
        data[h * head_dim + d] = data[h * head_dim + d] / l2 * scale;
    }
}

__global__ void fb_l2_k(float* data, int num_heads, int head_dim, float /*scale*/, float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    float l2 = 0.f;
    for (int d = 0; d < head_dim; ++d) {
        float v = data[h * head_dim + d];
        l2 += v * v;
    }
    l2 = sqrtf(l2 + eps);
    for (int d = 0; d < head_dim; ++d) {
        data[h * head_dim + d] /= l2;
    }
}

__global__ void fb_l2_qk_fused(float* d_q, float* d_k, int num_heads, int key_dim, float q_scale,
                               float eps) {
    int h = blockIdx.x;
    if (h >= num_heads)
        return;
    float l2q = 0.f, l2k = 0.f;
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

void run_fusion7_conv1d_update_baseline(const float* d_mixed, const float* d_conv_w,
                                        float* d_conv_state, float* d_conv_out, int conv_dim,
                                        int conv_kernel) {
    dim3 b(256);
    dim3 g((conv_dim + 255) / 256);
    fb_conv1d_kernel<<<g, b>>>(d_mixed, d_conv_out, d_conv_w, d_conv_state, conv_dim, conv_kernel);
    fb_update_conv_state_kernel<<<g, b>>>(d_mixed, d_conv_state, conv_dim, conv_kernel);
}

void run_fusion7_conv1d_update_fused(const float* d_mixed, const float* d_conv_w,
                                     float* d_conv_state, float* d_conv_out, int conv_dim,
                                     int conv_kernel) {
    dim3 b(256);
    dim3 g((conv_dim + 255) / 256);
    fb_conv1d_update_fused<<<g, b>>>(d_mixed, d_conv_out, d_conv_state, d_conv_w, conv_dim,
                                     conv_kernel);
}

void run_fusion7_l2norm_qk_baseline(float* d_q, float* d_k, int num_heads, int key_dim,
                                    float q_scale) {
    const float eps = 1e-6f;
    fb_l2_q<<<num_heads, 1>>>(d_q, num_heads, key_dim, q_scale, eps);
    fb_l2_k<<<num_heads, 1>>>(d_k, num_heads, key_dim, 1.f, eps);
}

void run_fusion7_l2norm_qk_fused(float* d_q, float* d_k, int num_heads, int key_dim,
                                 float q_scale) {
    fb_l2_qk_fused<<<num_heads, 1>>>(d_q, d_k, num_heads, key_dim, q_scale, 1e-6f);
}

} // namespace fusion_bench
} // namespace cuda
} // namespace qwen
