#include "cuda_engine_v3.hpp"
#include "cuda_ops.cuh"
#include "cuda_error_handling.cuh"
#include "flash_attention.cuh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen {
namespace cuda {

static void checkCudaV3(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(result) + " at " +
                                 file + ":" + std::to_string(line));
    }
}
#define CHECK_CUDA_V3(call) checkCudaV3(call, __FILE__, __LINE__)

static __global__ void fp32_to_bf16_kernel(const float* __restrict__ fp32_data,
                                    __nv_bfloat16* __restrict__ bf16_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    bf16_data[i] = __float2bfloat16(fp32_data[i]);
}

// Qwen3.5-0.8B layer types: repeating pattern [Linear, Linear, Linear, Full]
// Full attention at layers: 3, 7, 11, 15, 19, 23 (every 4th layer, starting from layer 3)
static bool is_full_attention_layer(int layer_idx) {
    return (layer_idx % 4) == 3;
}

CudaLayerV3::CudaLayerV3(int layer_idx, const CudaLayerConfigV3& config, bool use_full_attention)
    : layer_idx_(layer_idx), config_(config), use_full_attention_(use_full_attention) {
    input_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    post_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    mlp_ = std::make_unique<CudaMLP>(config.hidden_size, config.intermediate_size);
    
    if (use_full_attention_) {
        full_attn_ = std::make_unique<CudaFullAttention>(config.hidden_size, config.num_heads,
                                                         config.num_kv_heads, config.q_head_dim,
                                                         config.kv_head_dim);
    } else {
        linear_attn_ = std::make_unique<CudaLinearAttention>(
            config.hidden_size, config.linear_num_heads, config.linear_key_dim,
            config.linear_value_dim, config.linear_conv_kernel);
    }
}

CudaLayerV3::~CudaLayerV3() = default;

void CudaLayerV3::set_weights(
    const std::vector<float>& input_norm_weight, const std::vector<float>& post_norm_weight,
    const std::vector<float>& mlp_gate_w, const std::vector<float>& mlp_up_w,
    const std::vector<float>& mlp_down_w, const std::vector<float>& full_q_w,
    const std::vector<float>& full_k_w, const std::vector<float>& full_v_w,
    const std::vector<float>& full_qn_w, const std::vector<float>& full_kn_w,
    const std::vector<float>& full_o_w) {
    input_norm_->set_weights(input_norm_weight);
    post_norm_->set_weights(post_norm_weight);
    mlp_->set_weights(mlp_gate_w, mlp_up_w, mlp_down_w);
    if (full_attn_) {
        full_attn_->set_weights(full_q_w, full_k_w, full_v_w, full_qn_w, full_kn_w, full_o_w);
    }
}

void CudaLayerV3::set_linear_attention_weights(
    const std::vector<float>& in_proj_qkv_weight,
    const std::vector<float>& in_proj_a_weight,
    const std::vector<float>& in_proj_b_weight,
    const std::vector<float>& in_proj_z_weight,
    const std::vector<float>& conv1d_weight,
    const std::vector<float>& out_proj_weight,
    const std::vector<float>& a_log,
    const std::vector<float>& dt_bias,
    const std::vector<float>& norm_weight) {
    if (linear_attn_) {
        linear_attn_->set_weights(in_proj_qkv_weight, in_proj_a_weight, in_proj_b_weight,
                                   in_proj_z_weight, conv1d_weight, out_proj_weight,
                                   a_log, dt_bias, norm_weight);
    }
}

void CudaLayerV3::forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                          float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                          CudaKVCache& kv_cache, CudaLinearAttnState& linear_state, int position) const {
    int hs = config_.hidden_size;

    input_norm_->forward(d_input, d_normed_input_buf, 1);

    if (use_full_attention_ && full_attn_) {
        full_attn_->forward(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_, position);
    } else if (linear_attn_) {
        linear_attn_->forward(d_normed_input_buf, d_attn_out_buf, linear_state);
    }

    post_norm_->forward_with_residual(d_input, d_attn_out_buf, d_output, d_post_normed_buf, 1);

    mlp_->forward_add_residual(d_post_normed_buf, d_output, 1);
}

void CudaLayerV3::forward_batch_prefill(const float* d_input, float* d_output,
                                        float* d_normed_input_buf, float* d_attn_out_buf,
                                        float* d_post_normed_buf, float* d_mlp_out_buf,
                                        CudaKVCache& kv_cache, const int* positions,
                                        int batch_size) const {
    input_norm_->forward(d_input, d_normed_input_buf, batch_size);

    if (use_full_attention_ && full_attn_) {
        full_attn_->forward_batch_prefill(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_,
                                          positions, batch_size);
    } else if (linear_attn_) {
        // Linear attention prefill: process each token in batch
        for (int b = 0; b < batch_size; ++b) {
            const float* input_b = d_input + b * config_.hidden_size;
            float* output_b = d_attn_out_buf + b * config_.hidden_size;
            // For simplicity, use position 0 for all in batch prefill (not ideal but works)
            CudaLinearAttnState tmp_state;
            tmp_state.reset(config_.linear_num_heads, config_.linear_key_dim, config_.linear_value_dim, config_.linear_conv_kernel);
            linear_attn_->forward(input_b, output_b, tmp_state);
        }
    }

    post_norm_->forward_with_residual(d_input, d_attn_out_buf, d_output, d_post_normed_buf,
                                      batch_size);

    mlp_->forward_add_residual(d_post_normed_buf, d_output, batch_size);
}

CudaEngineV3::CudaEngineV3(int num_layers, int hidden_size, int intermediate_size, int vocab_size,
                           int max_seq_len)
    : num_layers_(num_layers), hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      vocab_size_(vocab_size), max_seq_len_(max_seq_len), d_input_buf_(nullptr),
      d_normed_input_(nullptr), d_attn_out_(nullptr), d_post_normed_(nullptr), d_mlp_out_(nullptr),
      d_residual_(nullptr), d_output_buf_(nullptr), d_lmhead_out_(nullptr),
      d_shared_embedding_lmhead_weight_(nullptr), gpu_memory_bytes_(0), ready_(false) {
    CudaLayerConfigV3 default_config;
    default_config.hidden_size = hidden_size;
    default_config.intermediate_size = intermediate_size;

    for (int i = 0; i < num_layers_; ++i) {
        bool use_full = is_full_attention_layer(i);
        layers_.push_back(std::make_unique<CudaLayerV3>(i, default_config, use_full));
    }

    final_norm_ = std::make_unique<CudaRMSNorm>(hidden_size);
    lm_head_ = std::make_unique<CudaLMHead>(hidden_size, vocab_size);
    embedding_ = std::make_unique<CudaTokenEmbedding>(vocab_size, hidden_size);

    // Initialize linear states for non-full-attention layers
    for (int i = 0; i < num_layers_; ++i) {
        if (!is_full_attention_layer(i)) {
            CudaLinearAttnState state;
            state.reset(default_config.linear_num_heads, default_config.linear_key_dim,
                        default_config.linear_value_dim, default_config.linear_conv_kernel);
            linear_states_.push_back(std::move(state));
        }
    }

    allocate_buffers();
    ready_ = true;
}

CudaEngineV3::~CudaEngineV3() {
    free_buffers();
    if (d_shared_embedding_lmhead_weight_) {
        cudaFree(d_shared_embedding_lmhead_weight_);
        d_shared_embedding_lmhead_weight_ = nullptr;
    }
}

void CudaEngineV3::allocate_buffers() {
    size_t total = 0;
    size_t hs = static_cast<size_t>(hidden_size_);

    CHECK_CUDA_V3(cudaMalloc(&d_input_buf_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_normed_input_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_attn_out_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_post_normed_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_mlp_out_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_residual_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_output_buf_, hs * sizeof(float)));
    total += hs * sizeof(float);

    CHECK_CUDA_V3(cudaMalloc(&d_lmhead_out_, static_cast<size_t>(vocab_size_) * sizeof(float)));
    total += static_cast<size_t>(vocab_size_) * sizeof(float);

    kv_cache_.reset(num_layers_, 2, 256, max_seq_len_);

    gpu_memory_bytes_ = total;
}

void CudaEngineV3::free_buffers() {
#define FREE_IF_V3(p)                                                                              \
    if (p) {                                                                                       \
        cudaFree(p);                                                                               \
        p = nullptr;                                                                               \
    }
    FREE_IF_V3(d_input_buf_);
    FREE_IF_V3(d_normed_input_);
    FREE_IF_V3(d_attn_out_);
    FREE_IF_V3(d_post_normed_);
    FREE_IF_V3(d_mlp_out_);
    FREE_IF_V3(d_residual_);
    FREE_IF_V3(d_output_buf_);
    FREE_IF_V3(d_lmhead_out_);
    FREE_IF_V3(d_batch_input_buf_);
    FREE_IF_V3(d_batch_output_buf_);
    FREE_IF_V3(d_positions_buf_);
    FREE_IF_V3(d_batch_normed_input_);
    FREE_IF_V3(d_batch_attn_out_);
    FREE_IF_V3(d_batch_post_normed_);
    FREE_IF_V3(d_batch_mlp_out_);
#undef FREE_IF_V3
}

void CudaEngineV3::ensure_batch_intermediate_buffers(int batch_size) const {
    if (batch_size <= max_batch_intermediate_size_ && d_batch_normed_input_ != nullptr) {
        return;
    }

    if (d_batch_normed_input_) cudaFree(d_batch_normed_input_);
    if (d_batch_attn_out_) cudaFree(d_batch_attn_out_);
    if (d_batch_post_normed_) cudaFree(d_batch_post_normed_);
    if (d_batch_mlp_out_) cudaFree(d_batch_mlp_out_);

    size_t bytes = static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float);
    int num_heads = 8;
    int kv_head_dim = 256;
    size_t attn_bytes = static_cast<size_t>(batch_size) * num_heads * kv_head_dim * sizeof(float);

    cudaMalloc(const_cast<float**>(&d_batch_normed_input_), bytes);
    cudaMalloc(const_cast<float**>(&d_batch_attn_out_), attn_bytes);
    cudaMalloc(const_cast<float**>(&d_batch_post_normed_), bytes);
    cudaMalloc(const_cast<float**>(&d_batch_mlp_out_), bytes);

    const_cast<int&>(max_batch_intermediate_size_) = batch_size;
}

void CudaEngineV3::set_layer_weights(int layer_idx, const std::vector<float>& weights_flat) {
    if (layer_idx < 0 || layer_idx >= num_layers_)
        return;

    int hs = hidden_size_;
    int isz = intermediate_size_;

    int offset = 0;
    auto slice = [&](int n) -> std::vector<float> {
        if (offset + n > static_cast<int>(weights_flat.size())) {
            throw std::out_of_range(
                "set_layer_weights: slice overflow at layer " + std::to_string(layer_idx) +
                ", offset=" + std::to_string(offset) + " n=" + std::to_string(n) +
                " total=" + std::to_string(weights_flat.size()));
        }
        std::vector<float> v(weights_flat.begin() + offset, weights_flat.begin() + offset + n);
        offset += n;
        return v;
    };

    auto input_norm_w = slice(hs);
    auto post_norm_w = slice(hs);
    auto gate_w = slice(isz * hs);
    auto up_w = slice(isz * hs);
    auto down_w = slice(hs * isz);

    if (layers_[layer_idx]->is_full_attention()) {
        // Full attention weights
        // Qwen3.5-0.8B config:
        // num_heads = 8, num_kv_heads = 2, head_dim = 256
        // Q proj: [num_heads * head_dim, hidden_size] = [2048, 1024]
        // K/V proj: [num_kv_heads * head_dim, hidden_size] = [512, 1024]
        int fnh = 8, qhd = 256, khd = 256;
        int num_kv_heads = 2;
        int kv_size = num_kv_heads * khd;  // 512
        
        auto fq = slice(fnh * qhd * hs);  // 8 * 256 * 1024 = 2,097,152
        auto fk = slice(kv_size * hs);     // 512 * 1024 = 524,288
        auto fv = slice(kv_size * hs);     // 512 * 1024 = 524,288
        auto fqn = slice(khd);             // 256
        auto fkn = slice(khd);             // 256
        auto fo = slice(hs * fnh * khd);   // 1024 * 8 * 256 = 2,097,152

        layers_[layer_idx]->set_weights(input_norm_w, post_norm_w, gate_w, up_w, down_w, 
                                        fq, fk, fv, fqn, fkn, fo);
    } else {
        // Linear attention weights
        int linear_num_heads = 16;
        int linear_key_dim = 128;
        int linear_value_dim = 128;
        int conv_kernel = 4;

        int k_dim = linear_num_heads * linear_key_dim;
        int v_dim = linear_num_heads * linear_value_dim;
        int conv_dim = k_dim * 2 + v_dim;
        int z_dim = linear_num_heads * linear_value_dim;

        auto in_proj_qkv = slice(conv_dim * hs);
        auto in_proj_a = slice(linear_num_heads * hs);
        auto in_proj_b = slice(linear_num_heads * hs);
        auto in_proj_z = slice(z_dim * hs);
        auto conv1d_w = slice(conv_dim * conv_kernel);
        auto out_proj = slice(hs * z_dim);
        auto a_log = slice(linear_num_heads);
        auto dt_bias = slice(linear_num_heads);
        auto norm_w = slice(linear_value_dim);

        layers_[layer_idx]->set_weights(input_norm_w, post_norm_w, gate_w, up_w, down_w,
                                        {}, {}, {}, {}, {}, {});
        layers_[layer_idx]->set_linear_attention_weights(
            in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, conv1d_w, out_proj,
            a_log, dt_bias, norm_w);
    }
}

void CudaEngineV3::set_final_norm_weight(const std::vector<float>& weight) {
    final_norm_->set_weights(weight);
}

void CudaEngineV3::set_lm_head_weight(const std::vector<float>& weight) {
    lm_head_->set_weight(weight);
}

void CudaEngineV3::set_shared_embedding_lmhead_weight(const std::vector<float>& weight) {
    size_t weight_size = static_cast<size_t>(vocab_size_) * hidden_size_;
    if (weight.size() != weight_size) {
        throw std::invalid_argument("set_shared_embedding_lmhead_weight: size mismatch, expected " +
                                    std::to_string(weight_size) + ", got " +
                                    std::to_string(weight.size()));
    }

    if (d_shared_embedding_lmhead_weight_) {
        cudaFree(d_shared_embedding_lmhead_weight_);
    }

    CHECK_CUDA_V3(cudaMalloc(&d_shared_embedding_lmhead_weight_, 
                             weight_size * sizeof(__nv_bfloat16)));

    float* d_temp_fp32 = nullptr;
    CHECK_CUDA_V3(cudaMalloc(&d_temp_fp32, weight_size * sizeof(float)));
    CHECK_CUDA_V3(cudaMemcpy(d_temp_fp32, weight.data(), weight_size * sizeof(float),
                             cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((weight_size + 255) / 256);
    fp32_to_bf16_kernel<<<grid, block>>>(d_temp_fp32, d_shared_embedding_lmhead_weight_, weight_size);
    CHECK_CUDA_V3(cudaGetLastError());
    CHECK_CUDA_V3(cudaFree(d_temp_fp32));

    embedding_->set_weight_bf16_ptr(d_shared_embedding_lmhead_weight_);
    lm_head_->set_weight_bf16_ptr(d_shared_embedding_lmhead_weight_);
}

void CudaEngineV3::set_embedding_weight(const std::vector<float>& weight) {
    embedding_->set_weight(weight);
}

void CudaEngineV3::forward(const float* d_input, float* d_output, int position) {
    CHECK_CUDA_V3(
        cudaMemcpy(d_input_buf_, d_input, hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice));

    float* ping = d_input_buf_;
    float* pong = d_residual_;

    int linear_state_idx = 0;
    for (int i = 0; i < num_layers_; ++i) {
        float* layer_out = (i % 2 == 0) ? pong : ping;
        float* layer_in = (i % 2 == 0) ? ping : pong;

        if (layers_[i]->is_full_attention()) {
            layers_[i]->forward(layer_in, layer_out, d_normed_input_, d_attn_out_, d_post_normed_,
                                d_mlp_out_, kv_cache_, linear_states_[0], position);
        } else {
            layers_[i]->forward(layer_in, layer_out, d_normed_input_, d_attn_out_, d_post_normed_,
                                d_mlp_out_, kv_cache_, linear_states_[linear_state_idx++], position);
        }
    }

    float* final_in = (num_layers_ % 2 == 0) ? ping : pong;
    final_norm_->forward(final_in, d_output, 1);
}

void CudaEngineV3::forward_batch_prefill(const float* d_input, float* d_output,
                                         const int* positions, int batch_size) {
    if (batch_size == 1) {
        forward(d_input, d_output, positions[0]);
        return;
    }

    ensure_batch_buffers(batch_size);
    ensure_batch_intermediate_buffers(batch_size);

    CHECK_CUDA_V3(cudaMemcpy(d_batch_input_buf_, d_input,
                             static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    CHECK_CUDA_V3(cudaMemcpy(d_positions_buf_, positions, batch_size * sizeof(int),
                             cudaMemcpyHostToDevice));

    float* ping = d_batch_input_buf_;
    float* pong = d_batch_output_buf_;

    int linear_state_idx = 0;
    for (int i = 0; i < num_layers_; ++i) {
        float* layer_out = (i % 2 == 0) ? pong : ping;
        float* layer_in = (i % 2 == 0) ? ping : pong;

        if (layers_[i]->is_full_attention()) {
            layers_[i]->forward_batch_prefill(layer_in, layer_out, d_batch_normed_input_,
                                              d_batch_attn_out_, d_batch_post_normed_,
                                              d_batch_mlp_out_, kv_cache_, d_positions_buf_,
                                              batch_size);
        } else {
            layers_[i]->forward_batch_prefill(layer_in, layer_out, d_batch_normed_input_,
                                              d_batch_attn_out_, d_batch_post_normed_,
                                              d_batch_mlp_out_, kv_cache_, d_positions_buf_,
                                              batch_size);
        }
    }

    float* final_in = (num_layers_ % 2 == 0) ? ping : pong;
    final_norm_->forward(final_in, d_output, batch_size);
}

void CudaEngineV3::forward_token(int token_id, float* d_output, int position) {
    embedding_->forward(token_id, d_input_buf_);
    forward(d_input_buf_, d_output, position);
}

void CudaEngineV3::forward_tokens(const std::vector<int>& token_ids, float* d_output, 
                                   const int* positions) {
    // Prefill: process tokens sequentially (non-parallel)
    for (int i = 0; i < static_cast<int>(token_ids.size()); ++i) {
        embedding_->forward(token_ids[i], d_input_buf_);
        forward(d_input_buf_, d_output, positions[i]);
    }
}

void CudaEngineV3::ensure_batch_buffers(int batch_size) {
    if (batch_size <= max_batch_size_ && d_batch_input_buf_ != nullptr) {
        return;
    }

    if (d_batch_input_buf_) cudaFree(d_batch_input_buf_);
    if (d_batch_output_buf_) cudaFree(d_batch_output_buf_);
    if (d_positions_buf_) cudaFree(d_positions_buf_);

    size_t batch_bytes = static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float);
    CHECK_CUDA_V3(cudaMalloc(&d_batch_input_buf_, batch_bytes));
    CHECK_CUDA_V3(cudaMalloc(&d_batch_output_buf_, batch_bytes));
    CHECK_CUDA_V3(cudaMalloc(&d_positions_buf_, batch_size * sizeof(int)));
    max_batch_size_ = batch_size;
}

void CudaEngineV3::forward_host(const std::vector<float>& input, std::vector<float>& output,
                                int position) {
    output.resize(hidden_size_);
    CHECK_CUDA_V3(cudaMemcpy(d_input_buf_, input.data(), hidden_size_ * sizeof(float),
                             cudaMemcpyHostToDevice));
    forward(d_input_buf_, d_output_buf_, position);
    CHECK_CUDA_V3(cudaMemcpy(output.data(), d_output_buf_, hidden_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost));
}

std::vector<float> CudaEngineV3::get_output() const {
    std::vector<float> result(hidden_size_);
    CHECK_CUDA_V3(cudaMemcpy(result.data(), d_output_buf_, hidden_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost));
    return result;
}

void CudaEngineV3::reset_cache() {
    kv_cache_.clear();
    for (auto& state : linear_states_) {
        state.clear();
    }
}

void CudaEngineV3::forward_batch_decode(const std::vector<int>& token_ids, float* d_output,
                                        int start_position) {
    int batch_size = static_cast<int>(token_ids.size());
    if (batch_size == 0) return;

    ensure_batch_buffers(batch_size);

    for (int t = 0; t < batch_size; ++t) {
        int position = start_position + t;
        embedding_->forward(token_ids[t], d_input_buf_);

        float* ping = d_input_buf_;
        float* pong = d_residual_;

        int linear_state_idx = 0;
        for (int i = 0; i < num_layers_; ++i) {
            float* layer_out = (i % 2 == 0) ? pong : ping;
            float* layer_in = (i % 2 == 0) ? ping : pong;

            if (layers_[i]->is_full_attention()) {
                layers_[i]->forward(layer_in, layer_out, d_normed_input_, d_attn_out_,
                                   d_post_normed_, d_mlp_out_, kv_cache_, linear_states_[0], position);
            } else {
                layers_[i]->forward(layer_in, layer_out, d_normed_input_, d_attn_out_,
                                   d_post_normed_, d_mlp_out_, kv_cache_, 
                                   linear_states_[linear_state_idx++], position);
            }
        }

        float* final_in = (num_layers_ % 2 == 0) ? pong : ping;
        float* output_t = d_output + t * hidden_size_;
        CHECK_CUDA_V3(cudaMemcpy(output_t, final_in, hidden_size_ * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }
}

std::string CudaEngineV3::get_device_info() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s | CC=%d.%d | Mem=%lluMB", prop.name, prop.major, prop.minor,
             (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));
    return std::string(buf);
}

} // namespace cuda
} // namespace qwen
