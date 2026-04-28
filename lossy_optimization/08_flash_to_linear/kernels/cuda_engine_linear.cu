#include "cuda_engine_linear.hpp"
#include "cuda_error_handling.cuh"
#include <stdexcept>
#include <cstring>

namespace qwen {
namespace cuda {

CudaLayerLinear::CudaLayerLinear(int layer_idx, const CudaLayerConfigLinear& config)
    : layer_idx_(layer_idx), config_(config) {
    input_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    post_norm_ = std::make_unique<CudaRMSNorm>(config.hidden_size);
    mlp_ = std::make_unique<CudaMLP>(config.hidden_size, config.intermediate_size);
    linear_attn_ = std::make_unique<CudaLinearAttentionV2>(
        config.hidden_size, config.num_heads, config.key_dim, config.value_dim, config.conv_kernel);
}

CudaLayerLinear::~CudaLayerLinear() = default;

void CudaLayerLinear::set_weights(const std::vector<float>& input_norm_weight,
                                   const std::vector<float>& post_norm_weight,
                                   const std::vector<float>& mlp_gate_w,
                                   const std::vector<float>& mlp_up_w,
                                   const std::vector<float>& mlp_down_w,
                                   const std::vector<float>& linear_qkv_w,
                                   const std::vector<float>& linear_a_w,
                                   const std::vector<float>& linear_b_w,
                                   const std::vector<float>& linear_z_w,
                                   const std::vector<float>& linear_conv_w,
                                   const std::vector<float>& linear_out_w) {
    input_norm_->set_weight(input_norm_weight);
    post_norm_->set_weight(post_norm_weight);
    mlp_->set_weights(mlp_gate_w, mlp_up_w, mlp_down_w);
    linear_attn_->set_weights(linear_qkv_w, linear_a_w, linear_b_w, linear_z_w,
                              linear_conv_w, linear_out_w);
}

void CudaLayerLinear::forward(const float* d_input, float* d_output, float* d_normed_input_buf,
                               float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                               CudaLinearAttnState& state) const {
    input_norm_->forward(d_input, d_normed_input_buf);
    linear_attn_->forward(d_normed_input_buf, d_attn_out_buf, state);
    post_norm_->forward(d_attn_out_buf, d_post_normed_buf);
    mlp_->forward_add_residual(d_post_normed_buf, d_output);
}

void CudaLayerLinear::forward_batch(const float* d_input, float* d_output, float* d_normed_input_buf,
                                     float* d_attn_out_buf, float* d_post_normed_buf, float* d_mlp_out_buf,
                                     CudaLinearAttnState& state, int batch_size, cudaStream_t stream) const {
    input_norm_->forward(d_input, d_normed_input_buf, batch_size);
    linear_attn_->forward_batch(d_normed_input_buf, d_attn_out_buf, state, batch_size, stream);
    post_norm_->forward(d_attn_out_buf, d_post_normed_buf, batch_size);
    mlp_->forward_add_residual(d_post_normed_buf, d_output, batch_size);
}

CudaEngineLinear::CudaEngineLinear(int num_layers, int hidden_size, int intermediate_size, int vocab_size)
    : num_layers_(num_layers), hidden_size_(hidden_size),
      intermediate_size_(intermediate_size), vocab_size_(vocab_size), gpu_memory_bytes_(0), ready_(false) {
    
    CudaLayerConfigLinear config;
    config.hidden_size = hidden_size;
    config.intermediate_size = intermediate_size;
    config.num_heads = 16;
    config.key_dim = 128;
    config.value_dim = 128;
    config.conv_kernel = 4;

    for (int i = 0; i < num_layers_; ++i) {
        layers_.push_back(std::make_unique<CudaLayerLinear>(i, config));
    }

    final_norm_ = std::make_unique<CudaRMSNorm>(hidden_size_);
    lm_head_ = std::make_unique<CudaLMHead>(hidden_size_, vocab_size_);
    embedding_ = std::make_unique<CudaTokenEmbedding>(vocab_size_, hidden_size_);

    linear_states_.resize(num_layers_);
    for (auto& state : linear_states_) {
        state.reset(config.num_heads, config.key_dim, config.value_dim, config.conv_kernel);
    }

    allocate_buffers();
    ready_ = true;
}

CudaEngineLinear::~CudaEngineLinear() {
    free_buffers();
}

void CudaEngineLinear::allocate_buffers() {
    size_t bytes = hidden_size_ * sizeof(float);
    cudaMalloc(&d_input_buf_, bytes);
    cudaMalloc(&d_normed_input_, bytes);
    cudaMalloc(&d_attn_out_, bytes);
    cudaMalloc(&d_post_normed_, bytes);
    cudaMalloc(&d_mlp_out_, bytes);
    cudaMalloc(&d_output_buf_, bytes);

    gpu_memory_bytes_ = 6 * bytes;
}

void CudaEngineLinear::free_buffers() {
    if (d_input_buf_) cudaFree(d_input_buf_);
    if (d_normed_input_) cudaFree(d_normed_input_);
    if (d_attn_out_) cudaFree(d_attn_out_);
    if (d_post_normed_) cudaFree(d_post_normed_);
    if (d_mlp_out_) cudaFree(d_mlp_out_);
    if (d_output_buf_) cudaFree(d_output_buf_);
    if (d_batch_input_buf_) cudaFree(d_batch_input_buf_);
    if (d_batch_output_buf_) cudaFree(d_batch_output_buf_);
}

void CudaEngineLinear::ensure_batch_buffers(int batch_size) {
    if (batch_size <= max_batch_size_ && d_batch_input_buf_ != nullptr) {
        return;
    }
    if (d_batch_input_buf_) cudaFree(d_batch_input_buf_);
    if (d_batch_output_buf_) cudaFree(d_batch_output_buf_);

    size_t bytes = static_cast<size_t>(batch_size) * hidden_size_ * sizeof(float);
    cudaMalloc(&d_batch_input_buf_, bytes);
    cudaMalloc(&d_batch_output_buf_, bytes);
    max_batch_size_ = batch_size;
}

void CudaEngineLinear::set_layer_weights(int layer_idx, const std::vector<float>& weights_flat) {
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

    int num_heads = 16, key_dim = 128, value_dim = 128;
    int conv_dim = num_heads * (key_dim + value_dim + key_dim);
    int z_dim = num_heads * value_dim;

    std::vector<float> linear_qkv(conv_dim * hidden_size_);
    std::memcpy(linear_qkv.data(), weights_flat.data() + offset, conv_dim * hidden_size_ * sizeof(float));
    offset += conv_dim * hidden_size_;

    std::vector<float> linear_a(num_heads * hidden_size_);
    std::memcpy(linear_a.data(), weights_flat.data() + offset, num_heads * hidden_size_ * sizeof(float));
    offset += num_heads * hidden_size_;

    std::vector<float> linear_b(num_heads * hidden_size_);
    std::memcpy(linear_b.data(), weights_flat.data() + offset, num_heads * hidden_size_ * sizeof(float));
    offset += num_heads * hidden_size_;

    std::vector<float> linear_z(z_dim * hidden_size_);
    std::memcpy(linear_z.data(), weights_flat.data() + offset, z_dim * hidden_size_ * sizeof(float));
    offset += z_dim * hidden_size_;

    std::vector<float> linear_conv(conv_dim * 4);
    std::memcpy(linear_conv.data(), weights_flat.data() + offset, conv_dim * 4 * sizeof(float));
    offset += conv_dim * 4;

    std::vector<float> linear_out(hidden_size_ * z_dim);
    std::memcpy(linear_out.data(), weights_flat.data() + offset, hidden_size_ * z_dim * sizeof(float));

    layers_[layer_idx]->set_weights(input_norm_w, post_norm_w, mlp_gate, mlp_up, mlp_down,
                                     linear_qkv, linear_a, linear_b, linear_z, linear_conv, linear_out);
}

void CudaEngineLinear::set_final_norm_weight(const std::vector<float>& weight) {
    final_norm_->set_weight(weight);
}

void CudaEngineLinear::set_lm_head_weight(const std::vector<float>& weight) {
    lm_head_->set_weight(weight);
}

void CudaEngineLinear::set_embedding_weight(const std::vector<float>& weight) {
    embedding_->set_weight(weight);
}

void CudaEngineLinear::forward(const float* d_input, float* d_output) {
    const float* current_input = d_input;
    float* current_output = d_output;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward(current_input, current_output, d_normed_input_, d_attn_out_,
                            d_post_normed_, d_mlp_out_, linear_states_[i]);
        current_input = current_output;
    }

    final_norm_->forward(current_input, d_output);
}

void CudaEngineLinear::forward_batch(const float* d_input, float* d_output, int batch_size, cudaStream_t stream) {
    ensure_batch_buffers(batch_size);

    const float* current_input = d_input;
    float* current_output = d_batch_output_buf_;

    for (int i = 0; i < num_layers_; ++i) {
        layers_[i]->forward_batch(current_input, current_output, d_normed_input_, d_attn_out_,
                                   d_post_normed_, d_mlp_out_, linear_states_[i], batch_size, stream);
        current_input = current_output;
    }

    final_norm_->forward(current_input, d_output, batch_size);
}

void CudaEngineLinear::forward_host(const std::vector<float>& input, std::vector<float>& output) {
    cudaMemcpy(d_input_buf_, input.data(), hidden_size_ * sizeof(float), cudaMemcpyHostToDevice);
    forward(d_input_buf_, d_output_buf_);
    output.resize(hidden_size_);
    cudaMemcpy(output.data(), d_output_buf_, hidden_size_ * sizeof(float), cudaMemcpyDeviceToHost);
}

std::vector<float> CudaEngineLinear::get_output() const {
    std::vector<float> output(hidden_size_);
    cudaMemcpy(output.data(), d_output_buf_, hidden_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}

void CudaEngineLinear::reset_state() {
    for (auto& state : linear_states_) {
        state.clear();
    }
}

std::string CudaEngineLinear::get_device_info() const {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return std::string(prop.name);
}

} // namespace cuda
} // namespace qwen
