#include "cuda_engine.hpp"
#include "cuda_ops.cuh"
#include "cuda_error_handling.cuh"
#include "flash_attention.cuh"
#include "mlp_cuda.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace qwen {
namespace cuda {

static void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(result) + " at " +
                                 file + ":" + std::to_string(line));
    }
}
#define CHECK_CUDA(call) checkCuda(call, __FILE__, __LINE__)

class StreamManager {
public:
    static constexpr int NUM_STREAMS = 3;

    enum StreamId {
        COMPUTE_DEFAULT = 0,
        COMPUTE_MLP = 1,
        DATA_TRANSFER = 2,
    };

    StreamManager() {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamCreate(&streams_[i]);
        }
    }

    ~StreamManager() {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            if (streams_[i]) {
                cudaStreamDestroy(streams_[i]);
            }
        }
    }

    cudaStream_t get(StreamId id) const {
        return streams_[static_cast<int>(id)];
    }

    void synchronize_all() const {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
    }

private:
    cudaStream_t streams_[NUM_STREAMS];
};

CudaLayer::CudaLayer(int layer_idx, const CudaLayerConfig& config)
    : layer_idx_(layer_idx), config_(config), is_linear_(config.is_linear) {
    input_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    post_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    mlp_ = std::make_unique<CudaMLP>(config.hidden_size, config.intermediate_size);

    if (config.is_linear) {
        linear_attn_ = std::make_unique<CudaLinearAttention>(
            config.hidden_size, config.linear_num_heads, config.key_dim, config.value_dim,
            config.conv_kernel);
    } else {
        full_attn_ = std::make_unique<CudaFullAttention>(config.hidden_size, config.full_num_heads,
                                                         config.num_kv_heads, config.q_head_dim,
                                                         config.kv_head_dim);
    }
}

CudaLayer::~CudaLayer() = default;

void CudaLayer::set_weights(
    const std::vector<float>& input_norm_weight, const std::vector<float>& post_norm_weight,
    const std::vector<float>& mlp_gate_w, const std::vector<float>& mlp_up_w,
    const std::vector<float>& mlp_down_w, bool is_linear, const std::vector<float>& lin_qkv_w,
    const std::vector<float>& lin_a_w, const std::vector<float>& lin_b_w,
    const std::vector<float>& lin_z_w, const std::vector<float>& lin_conv_w,
    const std::vector<float>& lin_out_w, const std::vector<float>& lin_a_log,
    const std::vector<float>& lin_dt_bias, const std::vector<float>& lin_norm_w,
    const std::vector<float>& full_q_w, const std::vector<float>& full_k_w,
    const std::vector<float>& full_v_w, const std::vector<float>& full_qn_w,
    const std::vector<float>& full_kn_w, const std::vector<float>& full_o_w) {
    is_linear_ = is_linear;
    input_norm_->set_weights(input_norm_weight);
    post_norm_->set_weights(post_norm_weight);
    mlp_->set_weights(mlp_gate_w, mlp_up_w, mlp_down_w);

    if (is_linear_) {
        linear_attn_->set_weights(lin_qkv_w, lin_a_w, lin_b_w, lin_z_w, lin_conv_w, lin_out_w,
                                  lin_a_log, lin_dt_bias, lin_norm_w);
    } else {
        if (!full_attn_) {
            full_attn_ = std::make_unique<CudaFullAttention>(
                config_.hidden_size, config_.full_num_heads, config_.num_kv_heads,
                config_.q_head_dim, config_.kv_head_dim);
        }
        set_fa_weights(full_attn_.get(), full_q_w, full_k_w, full_v_w, full_qn_w, full_kn_w,
                       full_o_w, layer_idx_);
    }
}

void CudaLayer::forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                        float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                        CudaKVCache& kv_cache, CudaLinearAttnState& lin_state, int position) const {
    int hs = config_.hidden_size;

    input_norm_->forward(d_input, d_normed_input_buf, 1);

    if (is_linear_) {
        linear_attn_->forward(d_normed_input_buf, d_attn_out_buf, lin_state);
    } else {
        full_attn_->forward(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_, position);
    }

    post_norm_->forward_with_residual(d_input, d_attn_out_buf, d_output, d_post_normed_buf, 1);

    mlp_->forward_add_residual(d_post_normed_buf, d_output, 1);
}

void CudaLayer::forward_batch_prefill(const float* d_input, float* d_output,
                                      float* d_normed_input_buf, float* d_attn_out_buf,
                                      float* d_post_normed_buf, float* d_mlp_out_buf,
                                      CudaKVCache& kv_cache, CudaLinearAttnState& lin_state,
                                      const int* positions, int batch_size) const {
    input_norm_->forward(d_input, d_normed_input_buf, batch_size);

    if (is_linear_) {
        for (int b = 0; b < batch_size; ++b) {
            linear_attn_->forward(d_normed_input_buf + b * config_.hidden_size,
                                  d_attn_out_buf + b * config_.hidden_size, lin_state);
        }
    } else {
        full_attn_->forward_batch_prefill(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_,
                                          positions, batch_size);
    }

    post_norm_->forward_with_residual(d_input, d_attn_out_buf, d_output, d_post_normed_buf,
                                      batch_size);

    mlp_->forward_add_residual(d_post_normed_buf, d_output, batch_size);
}

CudaEngine::CudaEngine(int num_layers, int hidden_size, int intermediate_size, int vocab_size,
                       int max_seq_len)
    : num_layers_(num_layers), hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      vocab_size_(vocab_size), max_seq_len_(max_seq_len), d_input_buf_(nullptr),
      d_normed_input_(nullptr), d_attn_out_(nullptr), d_post_normed_(nullptr), d_mlp_out_(nullptr),
      d_residual_(nullptr), d_output_buf_(nullptr), d_lmhead_out_(nullptr), gpu_memory_bytes_(0),
      ready_(false), stream_manager_(std::make_unique<StreamManager>()) {
    CudaLayerConfig default_config;
    default_config.hidden_size = hidden_size;
    default_config.intermediate_size = intermediate_size;

    for (int i = 0; i < num_layers_; ++i) {
        default_config.is_linear = (i % 4 != 3);
        layers_.push_back(std::make_unique<CudaLayer>(i, default_config));
    }

    final_norm_ = std::make_unique<CudaRMSNorm>(hidden_size);
    lm_head_ = std::make_unique<CudaLMHead>(hidden_size, vocab_size);

    allocate_buffers();
    ready_ = true;
}

CudaEngine::~CudaEngine() {
    free_buffers();
}

void CudaEngine::allocate_buffers() {
    size_t total = 0;
    size_t hs = static_cast<size_t>(hidden_size_);

    CHECK_CUDA(cudaMalloc(&d_input_buf_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_normed_input_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_attn_out_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_post_normed_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_mlp_out_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_residual_, hs * sizeof(float)));
    total += hs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_output_buf_, hs * sizeof(float)));
    total += hs * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_lmhead_out_, static_cast<size_t>(vocab_size_) * sizeof(float)));
    total += static_cast<size_t>(vocab_size_) * sizeof(float);

    kv_cache_.reset(num_layers_, 2, 256, max_seq_len_);
    linear_states_.resize(num_layers_);
    for (auto& s : linear_states_) {
        s.reset(16, 128, 128, 4);
    }

    gpu_memory_bytes_ = total;
}

void CudaEngine::free_buffers() {
#define FREE_IF(p)                                                                                 \
    if (p) {                                                                                       \
        cudaFree(p);                                                                               \
        p = nullptr;                                                                               \
    }
    FREE_IF(d_input_buf_);
    FREE_IF(d_normed_input_);
    FREE_IF(d_attn_out_);
    FREE_IF(d_post_normed_);
    FREE_IF(d_mlp_out_);
    FREE_IF(d_residual_);
    FREE_IF(d_output_buf_);
    FREE_IF(d_lmhead_out_);
#undef FREE_IF
}

void CudaEngine::set_layer_weights(int layer_idx, const std::vector<float>& weights_flat) {
    if (layer_idx < 0 || layer_idx >= num_layers_)
        return;

    int hs = hidden_size_;
    int isz = intermediate_size_;
    int lnh = 16, lnkh = 2, kd = 128, vd = 128, ck = 4;
    int fnh = 8, qhd = 256, khd = 256;

    bool is_linear = (layer_idx % 4 != 3);

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

    std::vector<float> lqkv, la, lb, lz, lconv, lout, l_a_log, l_dt_bias, l_norm_w;
    std::vector<float> fq, fk, fv, fqn, fkn, fo;

    lqkv = slice(lnh * (kd * 2 + vd) * hs);
    la = slice(lnh * hs);
    lb = slice(lnh * hs);
    lz = slice(lnh * vd * hs);
    lconv = slice(lnh * (kd * 2 + vd) * ck);
    lout = slice(hs * lnh * vd);
    l_a_log = slice(lnh);
    l_dt_bias = slice(lnh);
    l_norm_w = slice(vd);

    fq = slice(fnh * qhd * 2 * hs);
    fk = slice(lnkh * khd * hs);
    fv = slice(lnkh * khd * hs);
    fqn = slice(khd);
    fkn = slice(khd);
    fo = slice(hs * fnh * khd);

    layers_[layer_idx]->set_weights(input_norm_w, post_norm_w, gate_w, up_w, down_w, is_linear,
                                    lqkv, la, lb, lz, lconv, lout, l_a_log, l_dt_bias, l_norm_w, fq,
                                    fk, fv, fqn, fkn, fo);
}

void CudaEngine::set_final_norm_weight(const std::vector<float>& weight) {
    final_norm_->set_weights(weight);
}

void CudaEngine::set_lm_head_weight(const std::vector<float>& weight) {
    lm_head_->set_weight(weight);
}

void CudaEngine::forward(const float* d_input, float* d_output, int position) {
    CHECK_CUDA(
        cudaMemcpy(d_input_buf_, d_input, hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice));

    float* ping = d_input_buf_;
    float* pong = d_residual_;

    for (int i = 0; i < num_layers_; ++i) {
        float* layer_out = (i % 2 == 0) ? pong : ping;
        float* layer_in = (i % 2 == 0) ? ping : pong;

        layers_[i]->forward(layer_in, layer_out, d_normed_input_, d_attn_out_, d_post_normed_,
                            d_mlp_out_, kv_cache_, linear_states_[i], position);
    }

    float* final_in = (num_layers_ % 2 == 0) ? ping : pong;
    final_norm_->forward(final_in, d_output, 1);
}

void CudaEngine::forward_batch_prefill(const float* d_input, float* d_output, const int* positions,
                                       int batch_size) {
    if (batch_size == 1) {
        forward(d_input, d_output, positions[0]);
        return;
    }

    ensure_batch_buffers(batch_size);

    CHECK_CUDA(cudaMemcpy(d_batch_input_buf_, d_input,
                          static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_positions_buf_, positions, batch_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    float* ping = d_batch_input_buf_;
    float* pong = d_batch_output_buf_;

    for (int i = 0; i < num_layers_; ++i) {
        float* layer_out = (i % 2 == 0) ? pong : ping;
        float* layer_in = (i % 2 == 0) ? ping : pong;

        layers_[i]->forward_batch_prefill(layer_in, layer_out, d_normed_input_, d_attn_out_,
                                          d_post_normed_, d_mlp_out_, kv_cache_, linear_states_[i],
                                          d_positions_buf_, batch_size);
    }

    float* final_in = (num_layers_ % 2 == 0) ? ping : pong;
    final_norm_->forward(final_in, d_output, batch_size);
}

void CudaEngine::ensure_batch_buffers(int batch_size) {
    if (batch_size <= max_batch_size_ && d_batch_input_buf_ != nullptr) {
        return;
    }

    if (d_batch_input_buf_) cudaFree(d_batch_input_buf_);
    if (d_batch_output_buf_) cudaFree(d_batch_output_buf_);
    if (d_positions_buf_) cudaFree(d_positions_buf_);

    size_t batch_bytes = static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_batch_input_buf_, batch_bytes));
    CHECK_CUDA(cudaMalloc(&d_batch_output_buf_, batch_bytes));
    CHECK_CUDA(cudaMalloc(&d_positions_buf_, batch_size * sizeof(int)));
    max_batch_size_ = batch_size;
}

void CudaEngine::forward_host(const std::vector<float>& input, std::vector<float>& output,
                              int position) {
    output.resize(hidden_size_);
    CHECK_CUDA(cudaMemcpy(d_input_buf_, input.data(), hidden_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    forward(d_input_buf_, d_output_buf_, position);
    CHECK_CUDA(cudaMemcpy(output.data(), d_output_buf_, hidden_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

std::vector<float> CudaEngine::get_output() const {
    std::vector<float> result(hidden_size_);
    CHECK_CUDA(cudaMemcpy(result.data(), d_output_buf_, hidden_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

void CudaEngine::reset_cache() {
    kv_cache_.clear();
    for (auto& s : linear_states_) {
        s.clear();
    }
}

std::string CudaEngine::get_device_info() const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s | CC=%d.%d | Mem=%lluMB", prop.name, prop.major, prop.minor,
             (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));
    return std::string(buf);
}

} // namespace cuda
} // namespace qwen
