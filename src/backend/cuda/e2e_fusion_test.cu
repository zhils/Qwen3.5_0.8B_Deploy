/**
 * End-to-End Fusion Performance Comparison Test
 *
 * Compares performance of:
 * - Path A: No fusion (all operators as separate kernels)
 * - Path B: Current fused implementation
 *
 * Tests a complete Transformer Layer forward pass
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <chrono>
#include <functional>
#include <memory>

#include "fusion_microbench_kernels.cuh"

using namespace qwen::cuda::fusion_bench;

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":"   \
                      << __LINE__ << std::endl;                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

struct LayerConfig {
    int hidden_size = 1024;
    int intermediate_size = 3584;
    int num_heads = 8;
    int num_kv_heads = 2;
    int q_head_dim = 256;
    int kv_head_dim = 256;
    int head_dim = 256;
    int rotary_dim = 64;
    float rope_base = 10000000.0f;
    int max_seq_len = 2048;
    int num_rounds = 100;
    int warmup_rounds = 10;
    int batch_size = 1;
    int seq_len = 512;
    int conv_dim = 128;
    int conv_kernel = 4;
};

struct TimingResult {
    double path_a_ms = 0;
    double path_b_ms = 0;
    double speedup = 0;
};

static std::vector<float> generate_random_weights(size_t count, float min_val = -0.02f,
                                                   float max_val = 0.02f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    std::vector<float> weights(count);
    for (auto& w : weights) w = dis(gen);
    return weights;
}

static double measure_time_ms(const std::function<void()>& kernel_launch, int warmup, int rounds) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        kernel_launch();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < rounds; ++i) {
        kernel_launch();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return static_cast<double>(ms) / rounds;
}

class FusionTestLayer {
public:
    LayerConfig cfg_;

    float *d_input_, *d_output_;
    float *d_q_weight_, *d_k_weight_, *d_v_weight_;
    float *d_q_norm_w_, *d_k_norm_w_;
    float *d_o_weight_, *d_gate_weight_, *d_up_weight_, *d_down_weight_;
    float *d_post_norm_w_, *d_final_norm_w_;
    float *d_q_, *d_k_, *d_v_, *d_attn_out_, *d_scores_, *d_mlp_out_, *d_residual_;
    float *d_k_cache_, *d_v_cache_;
    float *d_conv_state_, *d_conv_out_;
    float *d_tmp_normed_, *d_tmp_gate_, *d_tmp_up_, *d_tmp_hidden_;
    float *d_hidden_;

    std::vector<float> h_q_weight_, h_k_weight_, h_v_weight_;
    std::vector<float> h_o_weight_, h_gate_weight_, h_up_weight_, h_down_weight_;
    std::vector<float> h_q_norm_w_, h_k_norm_w_;
    std::vector<float> h_post_norm_w_, h_final_norm_w_;

    FusionTestLayer(const LayerConfig& cfg) : cfg_(cfg) {
        CHECK_CUDA(cudaSetDevice(0));

        h_q_weight_ = generate_random_weights(cfg.num_heads * cfg.q_head_dim * 2 * cfg.hidden_size);
        h_k_weight_ = generate_random_weights(cfg.num_kv_heads * cfg.kv_head_dim * 2 * cfg.hidden_size);
        h_v_weight_ = generate_random_weights(cfg.num_kv_heads * cfg.kv_head_dim * 2 * cfg.hidden_size);
        h_o_weight_ = generate_random_weights(cfg.hidden_size * cfg.num_heads * cfg.kv_head_dim);
        h_gate_weight_ = generate_random_weights(cfg.intermediate_size * cfg.hidden_size);
        h_up_weight_ = generate_random_weights(cfg.intermediate_size * cfg.hidden_size);
        h_down_weight_ = generate_random_weights(cfg.hidden_size * cfg.intermediate_size);
        h_q_norm_w_ = generate_random_weights(cfg.kv_head_dim, 0.9f, 1.1f);
        h_k_norm_w_ = generate_random_weights(cfg.kv_head_dim, 0.9f, 1.1f);
        h_post_norm_w_ = generate_random_weights(cfg.hidden_size, 0.9f, 1.1f);
        h_final_norm_w_ = generate_random_weights(cfg.hidden_size, 0.9f, 1.1f);

        CHECK_CUDA(cudaMalloc(&d_input_, cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output_, cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_q_weight_, h_q_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_k_weight_, h_k_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_v_weight_, h_v_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_q_norm_w_, h_q_norm_w_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_k_norm_w_, h_k_norm_w_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_o_weight_, h_o_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gate_weight_, h_gate_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_up_weight_, h_up_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_down_weight_, h_down_weight_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_post_norm_w_, h_post_norm_w_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_final_norm_w_, h_final_norm_w_.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_q_, cfg.num_heads * cfg.q_head_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_k_, cfg.num_kv_heads * cfg.kv_head_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_v_, cfg.num_kv_heads * cfg.kv_head_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_attn_out_, cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_scores_, cfg.num_heads * cfg.max_seq_len * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mlp_out_, cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_residual_, cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_k_cache_, cfg.num_kv_heads * cfg.kv_head_dim * cfg.max_seq_len * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_v_cache_, cfg.num_kv_heads * cfg.kv_head_dim * cfg.max_seq_len * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv_state_, cfg.conv_dim * (cfg.conv_kernel - 1) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv_out_, cfg.conv_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp_normed_, cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp_gate_, cfg.intermediate_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp_up_, cfg.intermediate_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp_hidden_, cfg.intermediate_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hidden_, cfg.hidden_size * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_q_weight_, h_q_weight_.data(), h_q_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_k_weight_, h_k_weight_.data(), h_k_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_v_weight_, h_v_weight_.data(), h_v_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_q_norm_w_, h_q_norm_w_.data(), h_q_norm_w_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_k_norm_w_, h_k_norm_w_.data(), h_k_norm_w_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_o_weight_, h_o_weight_.data(), h_o_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_gate_weight_, h_gate_weight_.data(), h_gate_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_up_weight_, h_up_weight_.data(), h_up_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_down_weight_, h_down_weight_.data(), h_down_weight_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_post_norm_w_, h_post_norm_w_.data(), h_post_norm_w_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_final_norm_w_, h_final_norm_w_.data(), h_final_norm_w_.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~FusionTestLayer() {
        cudaFree(d_input_);
        cudaFree(d_output_);
        cudaFree(d_q_weight_);
        cudaFree(d_k_weight_);
        cudaFree(d_v_weight_);
        cudaFree(d_q_norm_w_);
        cudaFree(d_k_norm_w_);
        cudaFree(d_o_weight_);
        cudaFree(d_gate_weight_);
        cudaFree(d_up_weight_);
        cudaFree(d_down_weight_);
        cudaFree(d_post_norm_w_);
        cudaFree(d_final_norm_w_);
        cudaFree(d_q_);
        cudaFree(d_k_);
        cudaFree(d_v_);
        cudaFree(d_attn_out_);
        cudaFree(d_scores_);
        cudaFree(d_mlp_out_);
        cudaFree(d_residual_);
        cudaFree(d_k_cache_);
        cudaFree(d_v_cache_);
        cudaFree(d_conv_state_);
        cudaFree(d_conv_out_);
        cudaFree(d_tmp_normed_);
        cudaFree(d_tmp_gate_);
        cudaFree(d_tmp_up_);
        cudaFree(d_tmp_hidden_);
        cudaFree(d_hidden_);
    }

    void run_path_a_baseline(int position) {
        run_fusion1_baseline_q_path(d_input_, d_q_weight_, d_q_norm_w_, d_q_, d_hidden_,
                                    cfg_.hidden_size, cfg_.num_heads, cfg_.q_head_dim, cfg_.kv_head_dim,
                                    cfg_.rotary_dim, cfg_.rope_base, position);

        size_t k_offset = static_cast<size_t>(position) * cfg_.num_kv_heads * cfg_.kv_head_dim;
        run_fusion2_baseline_kv_cache(d_input_, d_k_weight_, d_v_weight_, d_k_norm_w_, d_k_, d_v_,
                                       d_k_cache_, d_v_cache_, k_offset,
                                       cfg_.hidden_size, cfg_.num_kv_heads,
                                       cfg_.kv_head_dim, cfg_.rotary_dim, cfg_.rope_base, position);

        int seq_len = position + 1;
        run_fusion3_baseline_attn_core(d_q_, d_k_cache_, d_v_cache_, d_scores_, d_attn_out_,
                                        cfg_.num_heads, cfg_.num_kv_heads, cfg_.kv_head_dim,
                                        cfg_.q_head_dim, seq_len);

        run_fusion3_gate_o_baseline(d_attn_out_, d_hidden_, d_o_weight_, d_residual_,
                                    cfg_.num_heads, cfg_.num_kv_heads, cfg_.q_head_dim,
                                    cfg_.num_heads * cfg_.kv_head_dim, cfg_.hidden_size);

        run_fusion6_chain_postnorm_mlp_residual(d_residual_, d_post_norm_w_, d_gate_weight_,
                                                 d_up_weight_, d_down_weight_, d_tmp_normed_,
                                                 d_tmp_gate_, d_tmp_up_, d_tmp_hidden_, d_mlp_out_,
                                                 cfg_.hidden_size, cfg_.intermediate_size);
    }

    void run_path_b_fused(int position) {
        run_fusion1_fused_q_path(d_input_, d_q_weight_, d_q_norm_w_, d_q_, d_hidden_,
                                 cfg_.hidden_size, cfg_.num_heads, cfg_.q_head_dim, cfg_.kv_head_dim,
                                 cfg_.rotary_dim, cfg_.rope_base, position);

        size_t k_offset = static_cast<size_t>(position) * cfg_.num_kv_heads * cfg_.kv_head_dim;
        run_fusion2_fused_kv_cache(d_input_, d_k_weight_, d_v_weight_, d_k_norm_w_, d_k_, d_v_,
                                   d_k_cache_, d_v_cache_, k_offset,
                                   cfg_.hidden_size, cfg_.num_kv_heads,
                                   cfg_.kv_head_dim, cfg_.rotary_dim, cfg_.rope_base, position);

        int seq_len = position + 1;
        run_fusion3_flash_attn_core(d_q_, d_k_cache_, d_v_cache_, d_attn_out_,
                                     cfg_.num_heads, cfg_.num_kv_heads, cfg_.q_head_dim, seq_len);

        run_fusion3_gate_o_fused(d_attn_out_, d_hidden_, d_o_weight_, d_residual_,
                                 cfg_.num_heads, cfg_.num_kv_heads, cfg_.q_head_dim,
                                 cfg_.num_heads * cfg_.kv_head_dim, cfg_.hidden_size);

        run_fusion6_fused_postnorm_mlp_residual(d_residual_, d_post_norm_w_, d_gate_weight_,
                                                d_up_weight_, d_down_weight_, d_tmp_normed_,
                                                d_hidden_, d_mlp_out_, cfg_.hidden_size,
                                                cfg_.intermediate_size);
    }
};

int main(int argc, char** argv) {
    LayerConfig cfg;
    if (argc > 1) cfg.num_rounds = std::atoi(argv[1]);
    if (argc > 2) cfg.warmup_rounds = std::atoi(argv[2]);
    if (argc > 3) cfg.seq_len = std::atoi(argv[3]);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "  End-to-End Fusion Performance Comparison" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "CC:  " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Test rounds: " << cfg.num_rounds << " (warmup: " << cfg.warmup_rounds << ")" << std::endl;
    std::cout << "Hidden size: " << cfg.hidden_size << std::endl;
    std::cout << "Num heads: " << cfg.num_heads << std::endl;
    std::cout << "Q head dim: " << cfg.q_head_dim << std::endl;
    std::cout << std::endl;

    auto layer = std::make_unique<FusionTestLayer>(cfg);

    auto h_input = generate_random_weights(cfg.hidden_size);
    CHECK_CUDA(cudaMemcpy(layer->d_input_, h_input.data(), cfg.hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(50) << "Test Case"
              << std::right << std::setw(15) << "Path A (ms)"
              << std::setw(15) << "Path B (ms)"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(94, '-') << std::endl;

    auto test_and_print = [&](const std::string& name, int seq_len) {
        std::cout << std::left << std::setw(50) << name << std::flush;

        auto baseline_launch = [&]() { layer->run_path_a_baseline(seq_len); };
        auto fused_launch = [&]() { layer->run_path_b_fused(seq_len); };

        double path_a_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
        double path_b_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
        double speedup = path_a_ms / path_b_ms;

        std::cout << std::right << std::setw(15) << path_a_ms
                  << std::setw(15) << path_b_ms
                  << std::setw(11) << speedup << "x" << std::endl;

        return std::make_tuple(path_a_ms, path_b_ms, speedup);
    };

    std::vector<int> seq_lengths = {128, 256, 512, 1024};
    if (cfg.seq_len > 0) {
        seq_lengths = {cfg.seq_len};
    }

    double total_a = 0, total_b = 0;
    for (int seq_len : seq_lengths) {
        auto [pa, pb, sa] = test_and_print("Seq Len " + std::to_string(seq_len), seq_len);
        total_a += pa;
        total_b += pb;
    }

    std::cout << std::string(94, '-') << std::endl;
    double avg_speedup = (total_a / seq_lengths.size()) / (total_b / seq_lengths.size());
    std::cout << std::left << std::setw(50) << "AVERAGE SPEEDUP"
              << std::right << std::setw(15) << ""
              << std::setw(15) << ""
              << std::setw(11) << avg_speedup << "x" << std::endl;

    std::cout << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "  Test Complete" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\nPath A = No fusion (all separate kernels)" << std::endl;
    std::cout << "Path B = Current fused implementation" << std::endl;

    return 0;
}