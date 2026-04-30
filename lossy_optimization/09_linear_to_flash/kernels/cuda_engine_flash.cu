#include "cuda_engine_flash.hpp"
#include "cuda_error_handling.cuh"
#include <stdexcept>
#include <cstring>

namespace qwen {
namespace cuda {

CudaLayerFlash::CudaLayerFlash(int layer_idx, const CudaLayerConfigFlash& config)
    : layer_idx_(layer_idx), config_(config) {
    input_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    post_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    mlp_ = std::make_unique<CudaMLP>(config.hidden_size, config.intermediate_size);
    flash_attn_ = std::make_unique<CudaFullAttention>(
        config.hidden_size, config.num_heads, config.num_kv_heads,
        config.q_head_dim, config.kv_head_dim);
}

CudaLayerFlash::~CudaLayerFlash() = default;

void CudaLayerFlash::set_weights(const std::vector<float>& input_norm_weight,
                                  const std::vector<float>& post_norm_weight,
                                  const std::vector<float>& mlp_gate_w,
                                  const std::vector<float>& mlp_up_w,
                                  const std::vector<float>& mlp_down_w,
                                  const std::vector<float>& q_proj_w,
                                  const std::vector<float>& k_proj_w,
                                  const std::vector<float>& v_proj_w,
                                  const std::vector<float>& q_norm_w,
                                  const std::vector<float>& k_norm_w,
                                  const std::vector<float>& o_proj_w) {
    input_norm_->set_weight(input_norm_weight);
    post_norm_->set_weight(post_norm_weight);
    mlp_->set_weights(mlp_gate_w, mlp_up_w, mlp_down_w);
    flash_attn_->set_weights(q_proj_w, k_proj_w, v_proj_w, q_norm_w, k_norm_w, o_proj_w);
}

void CudaLayerFlash::forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                              float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                              CudaKVCache& kv_cache, int position) const {
    input_norm_->forward(d_input, d_normed_input_buf);
    flash_attn_->forward(d_normed_input_buf, d_attn_out_buf, kv_cache, layer_idx_, position);
    post_norm_->forward(d_attn_out_buf, d_post_normed_buf);
    mlp_->forward_add_residual(d_post_normed_buf, d_output);
}

void CudaLayerFlash::forward_batch_prefill(const float* d_input, float* d_output,
                                            float* d_normed_input_buf, float* d_attn_out_buf,
                                            float* d_post_normed_buf, float* d_mlp_out_buf,
                                            CudaKVCache& kv_cache, const int* positions,
                                            int batch_size) const {
    input_norm_->forward(d_input, d_normed_input_buf, batch_size);
    flash_attn_->forward_batch_prefill(d_normed_input_buf, d_attn_out_buf, kv_cache,
                                        layer_idx_, positions, batch_size);
    post_norm_->forward(d_attn_out_buf, d_post_normed_buf, batch_size);
    mlp_->forward_add_residual(d_post_normed_buf, d_output, batch_size);
}

CudaEngineFlash::CudaEngineFlash(int num_layers, int hidden_size, int intermediate_size,
                                 int vocab_size, int max_seq_len)
    : num_layers_(num_layers), hidden_size_(hidden_size),
      intermediate_size_(intermediate_size), vocab_size_(vocab_size),
      max_seq_len_(max_seq_len), gpu_memory_bytes_(0), ready_(false) {
    
    CudaLayerConfigFlash config;
    config.hidden_size = hidden_size;
    config.intermediate_size = intermediate_size;
    config.num_heads = 8;
    config.num_kv_heads = 2;
    config.q_head_dim = 256;
    config.kv_head_dim = 256;

    for (int i = 0; i < num_layers_; ++i) {
        layers_.push_back(std::make_unique<CudaLayerFlash>(i, config));
    }

    final_norm_ = std::make_unique<CudaRMSNorm>(hidden_size_);
    lm_head_ = std::make_unique<CudaLMHead>(hidden_size_, vocab_size_);
    embedding_ = std::make_unique<CudaTokenEmbedding>(vocab_size_, hidden_size_);

    kv_cache_.reset(num_layers_, config.num_kv_heads, config.kv_head_dim, 1024);

    allocate_buffers();
    ready_ = true;
}

CudaEngineFlash::~CudaEngineFlash() {
    free_buffers();
}

void CudaEngineFlash::allocate_buffers() {
    size_t bytes = hidden_size_ * sizeof(float);
    cudaMalloc(&d_input_buf_, bytes);
    cudaMalloc(&d_normed_input_, bytes);
    cudaMalloc(&d_attn_out_, bytes);
    cudaMalloc(&d_post_normed_, bytes);
    cudaMalloc(&d_mlp_out_, bytes);
    cudaMalloc(&d_output_buf_, bytes);

    gpu_memory_bytes_ = 6 * bytes;
}

void CudaEngineFlash::free_buffers() {
    if (d_input_buf_) cudaFree(d_input_buf_);
    if (d_normed_input_) cudaFree(d_normed_input_);
    if (d_attn_out_) cudaFree(d_attn_out_);
    if (d_post_normed_) cudaFree(d_post_normed_);
    if (d_mlp_out_) cudaFree(d_mlp_out_);
    if (d_output_buf_) cudaFree(d_output_buf_);
    if (d_batch_input_buf_) cudaFree(d_batch_input_buf_);
    if (d_batch_output_buf_) cudaFree(d_batch_output_buf_);
    if (d_positions_buf_) cudaFree(d_positions_buf_);
}

void CudaEngineFlash::ensure_batch_buffers(int batch_size) {
    if (batch_size <= max_batch_size_ && d_batch_input_buf_ != nullptr) {
        return;
    }
    if (d_batch_input_buf_) cudaFree(d_batch_input_buf_);
    if (d_batch_output_buf_) cudaFree(d_batch_output_buf_);
    if (d_positions_buf_) cudaFree(d_positions_buf_);

    size_t bytes = static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float);
    cudaMalloc(&d_batch_input_buf_, bytes);
    cudaMalloc(&d_batch_output_buf_, bytes);
    cudaMalloc(&d_positions_buf_, batch_size * sizeof(int));
    max_batch_size_ = batch_size;
}

void CudaEngineFlash::set_layer_weights(int layer_idx, const std::vector<float>& weights_flat) {
    if (layer_idx < 0 || layer_idx >= num_layers_) return;

    size_t offset = 0;
    std::vector<float> input_norm_w(hidden_size_);
    std::memcpy(input_norm_w.data(), weights_flat.data() + offset, hidden_size_ * sizeof(float));
    offset += hidden_size_;

    std::vector<float> post_norm_w(hidden_size_);
    std::memcpy(post_norm_w.data(), weights_flat.data() + offset, hidden_size_ * sizeof(float));
    offset += hidden_size_;

    int inter_size = intermediate_size_;
    std::vector<float> mlp_gate(inter_size * hidden_size_);
    std::memcpy(mlp_gate.data(), weights_flat.data() + offset, inter_size * hidden_size_ * sizeof(float));
    offset += inter_size * hidden_size_;

    std::vector<float> mlp_up(inter_size * hidden_size_);
    std::memcpy(mlp_up.data(), weights_flat.data() + offset, inter_size * hidden_size_ * sizeof(float));
    offset += inter_size * hidden_size_;

    std::vector<float> mlp_down(hidden_size_ * inter_size);
    std::memcpy(mlp_down.data(), weights_flat.data() + offset, hidden_size_ * inter_size * sizeof(float));
    offset += hidden_size_ * inter_size;

    int num_heads = 8, num_kv_heads = 2, q_head_dim = 256, kv_head_dim = 256;
    int total_q = num_heads * q_head_dim;
    int total_kv = num_kv_heads * kv_head_dim;

    std::vector<float> q_proj(total_q * 2 * hidden_size_);
    std::memcpy(q_proj.data(), weights_flat.data() + offset, total_q * 2 * hidden_size_ * sizeof(float));
    offset += total_q * 2 * hidden_size_;

    std::vector<float> k_proj(total_kv * hidden_size_);
    std::memcpy(k_proj.data(), weights_flat.data() + offset, total_kv * hidden_size_ * sizeof(float));
    offset += total_kv * hidden_size_;

    std::vector<float> v_proj(total_kv * hidden_size_);
    std::memcpy(v_proj.data(), weights_flat.data() + offset, total_kv * hidden_size_ * sizeof(float));
    offset += total_kv * hidden_size_;

    std::vector<float> q_norm(kv_head_dim);
    std::memcpy(q_norm.data(), weights_flat.data() + offset, kv_head_dim * sizeof(float));
    offset += kv_head_dim;

    std::vector<float> k_norm(kv_head_dim);
    std::memcpy(k_norm.data(), weights_flat.data() + offset, kv_head_dim * sizeof(float));
    offset += kv_head_dim;

    int total_out = num_heads * kv_head_dim;
    std::vector<float> o_proj(hidden_size_ * total_out);
    std::memcpy(o_proj.data(), weights_flat.data() + offset, hidden_size_ * total_out * sizeof(float));

    layers_[layer_idx]->set_weights(input_norm_w, post_norm_w, mlp_gate, mlp_up, mlp_down,
                                     q_proj, k_proj, v_proj, q_norm, k_norm, o_proj);
}

void CudaEngineFlash::set_final_norm_weight(const std::vector<float>& weight) {
    final_norm_->set_weight(weight);
}

void CudaEngineFlash::set_lm_head_weight(const std::vector<float>& weight) {
    lm_head_->set_weight(weight);
}

void CudaEngineFlash::set_embedding_weight(const std::vector<float>& weight) {
    embedding_->set_weight(weight);
}

void CudaEngineFlash::forward(const float* d_input, float* d_output, int position) {
    const float* current_input = d_input;
    float* current_output = d_output;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(current_input, current_output, d_normed_input_, d_attn_out_,
                            d_post_normed_, d_mlp_out_, kv_cache_, position);
        current_input = current_output;
    }

    final_norm_->forward(current_input, d_output);
}

void CudaEngineFlash::forward_batch_prefill(const float* d_input, float* d_output,
                                             const int* positions, int batch_size) {
    ensure_batch_buffers(batch_size);

    cudaMemcpy(d_batch_input_buf_, d_input,
               static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_positions_buf_, positions, batch_size * sizeof(int), cudaMemcpyDeviceToDevice);

    const float* current_input = d_batch_input_buf_;
    float* current_output = d_batch_output_buf_;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward_batch_prefill(current_input, current_output, d_normed_input_,
                                           d_attn_out_, d_post_normed_, d_mlp_out_,
                                           kv_cache_, d_positions_buf_, batch_size);
        current_input = current_output;
    }

    final_norm_->forward(current_input, d_output, batch_size);
}

void CudaEngineFlash::forward_host(const std::vector<float>& input, std::vector<float>& output, int position) {
    cudaMemcpy(d_input_buf_, input.data(), hidden_size_ * sizeof(float), cudaMemcpyHostToDevice);
    forward(d_input_buf_, d_output_buf_, position);
    output.resize(hidden_size_);
    cudaMemcpy(output.data(), d_output_buf_, hidden_size_ * sizeof(float), cudaMemcpyDeviceToHost);
}

std::vector<float> CudaEngineFlash::get_output() const {
    std::vector<float> output(hidden_size_);
    cudaMemcpy(output.data(), d_output_buf_, hidden_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}

void CudaEngineFlash::reset_cache() {
    kv_cache_.clear();
}

std::string CudaEngineFlash::get_device_info() const {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return std::string(prop.name);
}

} // namespace cuda
} // namespace qwen
