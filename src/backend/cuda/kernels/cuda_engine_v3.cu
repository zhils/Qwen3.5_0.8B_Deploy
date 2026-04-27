#include "cuda_engine_v3.hpp"
#include "cuda_ops.cuh"
#include "cuda_error_handling.cuh"
#include "flash_attention.cuh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace qwen {
namespace cuda {

static void checkCudaV3(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(result) + " at " +
                                 file + ":" + std::to_string(line));
    }
}
#define CHECK_CUDA_V3(call) checkCudaV3(call, __FILE__, __LINE__)

CudaLayerV3::CudaLayerV3(int layer_idx, const CudaLayerConfigV3& config)
    : layer_idx_(layer_idx), config_(config) {
    input_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    post_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    mlp_ = std::make_unique<CudaMLP>(config.hidden_size, config.intermediate_size);
    full_attn_ = std::make_unique<CudaFullAttention>(config.hidden_size, config.num_heads,
                                                     config.num_kv_heads, config.q_head_dim,
                                                     config.kv_head_dim);
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
    full_attn_->set_weights(full_q_w, full_k_w, full_v_w, full_qn_w, full_kn_w, full_o_w);
}

void CudaLayerV3::forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                          float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                          CudaKVCache& kv_cache, int position) const {
    int hs = config_.hidden_size;

    input_norm_->forward(d_input, d_normed_input_buf, 1);

    full_attn_->forward(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_, position);

    post_norm_->forward_with_residual(d_input, d_attn_out_buf, d_output, d_post_normed_buf, 1);

    mlp_->forward_add_residual(d_post_normed_buf, d_output, 1);
}

void CudaLayerV3::forward_batch_prefill(const float* d_input, float* d_output,
                                        float* d_normed_input_buf, float* d_attn_out_buf,
                                        float* d_post_normed_buf, float* d_mlp_out_buf,
                                        CudaKVCache& kv_cache, const int* positions,
                                        int batch_size) const {
    input_norm_->forward(d_input, d_normed_input_buf, batch_size);

    full_attn_->forward_batch_prefill(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_,
                                      positions, batch_size);

    post_norm_->forward_with_residual(d_input, d_attn_out_buf, d_output, d_post_normed_buf,
                                      batch_size);

    mlp_->forward_add_residual(d_post_normed_buf, d_output, batch_size);
}

CudaEngineV3::CudaEngineV3(int num_layers, int hidden_size, int intermediate_size, int vocab_size,
                           int max_seq_len)
    : num_layers_(num_layers), hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      vocab_size_(vocab_size), max_seq_len_(max_seq_len), d_input_buf_(nullptr),
      d_normed_input_(nullptr), d_attn_out_(nullptr), d_post_normed_(nullptr), d_mlp_out_(nullptr),
      d_residual_(nullptr), d_output_buf_(nullptr), d_lmhead_out_(nullptr), gpu_memory_bytes_(0),
      ready_(false) {
    CudaLayerConfigV3 default_config;
    default_config.hidden_size = hidden_size;
    default_config.intermediate_size = intermediate_size;

    // v3.0: All layers use Flash Attention (Full Attention)
    for (int i = 0; i < num_layers_; ++i) {
        layers_.push_back(std::make_unique<CudaLayerV3>(i, default_config));
    }

    final_norm_ = std::make_unique<CudaRMSNorm>(hidden_size);
    lm_head_ = std::make_unique<CudaLMHead>(hidden_size, vocab_size);

    allocate_buffers();
    ready_ = true;
}

CudaEngineV3::~CudaEngineV3() {
    free_buffers();
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

    // v3.0: All layers use Full Attention, so all need KV cache
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
    // v3.0: All Flash Attention, attn_out needs batch_size * num_heads * kv_head_dim
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
    int fnh = 8, qhd = 256, khd = 256;

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

    // v3.0: Only Full Attention weights (no Linear Attention weights)
    std::vector<float> fq, fk, fv, fqn, fkn, fo;

    fq = slice(fnh * qhd * 2 * hs);
    fk = slice(2 * khd * hs);
    fv = slice(2 * khd * hs);
    fqn = slice(khd);
    fkn = slice(khd);
    fo = slice(hs * fnh * khd);

    layers_[layer_idx]->set_weights(input_norm_w, post_norm_w, gate_w, up_w, down_w, fq, fk, fv,
                                    fqn, fkn, fo);
}

void CudaEngineV3::set_final_norm_weight(const std::vector<float>& weight) {
    final_norm_->set_weights(weight);
}

void CudaEngineV3::set_lm_head_weight(const std::vector<float>& weight) {
    lm_head_->set_weight(weight);
}

void CudaEngineV3::forward(const float* d_input, float* d_output, int position) {
    CHECK_CUDA_V3(
        cudaMemcpy(d_input_buf_, d_input, hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice));

    float* ping = d_input_buf_;
    float* pong = d_residual_;

    for (int i = 0; i < num_layers_; ++i) {
        float* layer_out = (i % 2 == 0) ? pong : ping;
        float* layer_in = (i % 2 == 0) ? ping : pong;

        layers_[i]->forward(layer_in, layer_out, d_normed_input_, d_attn_out_, d_post_normed_,
                            d_mlp_out_, kv_cache_, position);
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

    for (int i = 0; i < num_layers_; ++i) {
        float* layer_out = (i % 2 == 0) ? pong : ping;
        float* layer_in = (i % 2 == 0) ? ping : pong;

        layers_[i]->forward_batch_prefill(layer_in, layer_out, d_batch_normed_input_,
                                          d_batch_attn_out_, d_batch_post_normed_,
                                          d_batch_mlp_out_, kv_cache_, d_positions_buf_,
                                          batch_size);
    }

    float* final_in = (num_layers_ % 2 == 0) ? ping : pong;
    final_norm_->forward(final_in, d_output, batch_size);
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
