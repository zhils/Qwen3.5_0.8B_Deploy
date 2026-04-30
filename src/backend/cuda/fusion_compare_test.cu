/**
 * Fusion Performance Comparison Test
 *
 * Compares performance before and after operator fusion
 * Tests all 7 fusion points in the codebase
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

struct FusionConfig {
    int hidden_size = 1024;
    int intermediate_size = 3584;
    int num_heads = 8;
    int num_kv_heads = 2;
    int q_head_dim = 256;
    int kv_head_dim = 256;
    int rotary_dim = 256;
    int max_seq_len = 2048;
    int rope_base = 10000;
    int num_rounds = 100;
    int warmup_rounds = 10;
};

struct TimingResult {
    double baseline_ms = 0;
    double fused_ms = 0;
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

// ============================================================================
// Fusion #1: Q proj + Q norm + RoPE(Q)
// ============================================================================
static TimingResult test_fusion1_q_path(const FusionConfig& cfg) {
    size_t input_size = cfg.hidden_size;
    size_t q_size = cfg.num_heads * cfg.q_head_dim;
    size_t gate_size = cfg.num_heads * cfg.q_head_dim;
    size_t q_weight_size = cfg.num_heads * cfg.q_head_dim * 2 * cfg.hidden_size;
    size_t q_norm_size = cfg.kv_head_dim;

    float *d_input, *d_q_weight, *d_q_norm_w, *d_q, *d_gate;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_weight, q_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_norm_w, q_norm_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q, q_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate, gate_size * sizeof(float)));

    auto h_input = generate_random_weights(input_size);
    auto h_q_weight = generate_random_weights(q_weight_size);
    auto h_q_norm_w = generate_random_weights(q_norm_size, 0.9f, 1.1f);

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_weight, h_q_weight.data(), q_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_norm_w, h_q_norm_w.data(), q_norm_size * sizeof(float), cudaMemcpyHostToDevice));

    int position = 512;

    auto baseline_launch = [&]() {
        run_fusion1_baseline_q_path(d_input, d_q_weight, d_q_norm_w, d_q, d_gate,
                                    cfg.hidden_size, cfg.num_heads, cfg.q_head_dim, cfg.kv_head_dim,
                                    cfg.rotary_dim, cfg.rope_base, position);
    };

    auto fused_launch = [&]() {
        run_fusion1_fused_q_path(d_input, d_q_weight, d_q_norm_w, d_q, d_gate,
                                 cfg.hidden_size, cfg.num_heads, cfg.q_head_dim, cfg.kv_head_dim,
                                 cfg.rotary_dim, cfg.rope_base, position);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_input);
    cudaFree(d_q_weight);
    cudaFree(d_q_norm_w);
    cudaFree(d_q);
    cudaFree(d_gate);

    return result;
}

// ============================================================================
// Fusion #2: KV proj + K norm + RoPE(K) + write cache
// ============================================================================
static TimingResult test_fusion2_kv_cache(const FusionConfig& cfg) {
    size_t input_size = cfg.hidden_size;
    size_t kv_size = cfg.num_kv_heads * cfg.kv_head_dim;
    size_t k_weight_size = cfg.num_kv_heads * cfg.kv_head_dim * cfg.hidden_size;
    size_t v_weight_size = cfg.num_kv_heads * cfg.kv_head_dim * cfg.hidden_size;
    size_t k_norm_size = cfg.kv_head_dim;
    size_t cache_size = static_cast<size_t>(cfg.max_seq_len) * kv_size;

    float *d_input, *d_k_w, *d_v_w, *d_k_norm_w, *d_k, *d_v, *d_k_cache, *d_v_cache;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_w, k_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_w, v_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_norm_w, k_norm_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, kv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, kv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_cache, cache_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_cache, cache_size * sizeof(float)));

    auto h_input = generate_random_weights(input_size);
    auto h_k_w = generate_random_weights(k_weight_size);
    auto h_v_w = generate_random_weights(v_weight_size);
    auto h_k_norm_w = generate_random_weights(k_norm_size, 0.9f, 1.1f);

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_w, h_k_w.data(), k_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_w, h_v_w.data(), v_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_norm_w, h_k_norm_w.data(), k_norm_size * sizeof(float), cudaMemcpyHostToDevice));

    int position = 512;
    size_t k_offset = position * kv_size;

    auto baseline_launch = [&]() {
        run_fusion2_baseline_kv_cache(d_input, d_k_w, d_v_w, d_k_norm_w, d_k, d_v,
                                      d_k_cache, d_v_cache, k_offset,
                                      cfg.hidden_size, cfg.num_kv_heads, cfg.kv_head_dim,
                                      cfg.rotary_dim, cfg.rope_base, position);
    };

    auto fused_launch = [&]() {
        run_fusion2_fused_kv_cache(d_input, d_k_w, d_v_w, d_k_norm_w, d_k, d_v,
                                   d_k_cache, d_v_cache, k_offset,
                                   cfg.hidden_size, cfg.num_kv_heads, cfg.kv_head_dim,
                                   cfg.rotary_dim, cfg.rope_base, position);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_input);
    cudaFree(d_k_w);
    cudaFree(d_v_w);
    cudaFree(d_k_norm_w);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);

    return result;
}

// ============================================================================
// Fusion #3: Attention core + Gate + O_proj
// ============================================================================
static TimingResult test_fusion3_attn_gate_o(const FusionConfig& cfg) {
    int seq_len = 1024;
    size_t q_size = cfg.num_heads * cfg.q_head_dim;
    size_t kv_cache_size = static_cast<size_t>(seq_len) * cfg.num_kv_heads * cfg.kv_head_dim;
    size_t scores_size = cfg.num_heads * seq_len;
    size_t attn_out_size = cfg.num_heads * cfg.kv_head_dim;
    size_t total_out = cfg.num_heads * cfg.kv_head_dim;
    size_t o_weight_size = cfg.hidden_size * total_out;

    float *d_q, *d_k_cache, *d_v_cache, *d_scores, *d_attn_out, *d_gate, *d_o_weight, *d_output;
    CHECK_CUDA(cudaMalloc(&d_q, q_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_cache, kv_cache_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_cache, kv_cache_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, scores_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_out, attn_out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate, q_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_o_weight, o_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, cfg.hidden_size * sizeof(float)));

    auto h_q = generate_random_weights(q_size);
    auto h_k_cache = generate_random_weights(kv_cache_size);
    auto h_v_cache = generate_random_weights(kv_cache_size);
    auto h_gate = generate_random_weights(q_size);
    auto h_o_weight = generate_random_weights(o_weight_size);

    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_cache, h_k_cache.data(), kv_cache_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_cache, h_v_cache.data(), kv_cache_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), q_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_o_weight, h_o_weight.data(), o_weight_size * sizeof(float), cudaMemcpyHostToDevice));

    auto baseline_launch = [&]() {
        run_fusion3_baseline_attn_core(d_q, d_k_cache, d_v_cache, d_scores, d_attn_out,
                                       cfg.num_heads, cfg.num_kv_heads, cfg.kv_head_dim,
                                       cfg.q_head_dim, seq_len);
        run_fusion3_gate_o_baseline(d_attn_out, d_gate, d_o_weight, d_output,
                                    cfg.num_heads, cfg.kv_head_dim, cfg.q_head_dim,
                                    total_out, cfg.hidden_size);
    };

    auto fused_launch = [&]() {
        run_fusion3_baseline_attn_core(d_q, d_k_cache, d_v_cache, d_scores, d_attn_out,
                                       cfg.num_heads, cfg.num_kv_heads, cfg.kv_head_dim,
                                       cfg.q_head_dim, seq_len);
        run_fusion3_gate_o_fused(d_attn_out, d_gate, d_o_weight, d_output,
                                 cfg.num_heads, cfg.kv_head_dim, cfg.q_head_dim,
                                 total_out, cfg.hidden_size);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_scores);
    cudaFree(d_attn_out);
    cudaFree(d_gate);
    cudaFree(d_o_weight);
    cudaFree(d_output);

    return result;
}

// ============================================================================
// Fusion #4: RMSNorm + Linear projection
// ============================================================================
static TimingResult test_fusion4_rmsnorm_linear(const FusionConfig& cfg) {
    int out_dim = cfg.num_heads * cfg.q_head_dim * 2; // Q+G projection

    size_t input_size = cfg.hidden_size;
    size_t norm_w_size = cfg.hidden_size;
    size_t weight_size = out_dim * cfg.hidden_size;
    size_t out_size = out_dim;

    float *d_input, *d_norm_w, *d_weight, *d_out, *d_tmp_normed;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_norm_w, norm_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp_normed, input_size * sizeof(float)));

    auto h_input = generate_random_weights(input_size);
    auto h_norm_w = generate_random_weights(norm_w_size, 0.9f, 1.1f);
    auto h_weight = generate_random_weights(weight_size);

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_norm_w, h_norm_w.data(), norm_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

    auto baseline_launch = [&]() {
        run_fusion4_rmsnorm_then_linear_head(d_input, d_norm_w, d_weight, d_out, d_tmp_normed,
                                             cfg.hidden_size, out_dim);
    };

    auto fused_launch = [&]() {
        run_fusion4_fused_rmsnorm_linear_head(d_input, d_norm_w, d_weight, d_out,
                                              cfg.hidden_size, out_dim);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_input);
    cudaFree(d_norm_w);
    cudaFree(d_weight);
    cudaFree(d_out);
    cudaFree(d_tmp_normed);

    return result;
}

// ============================================================================
// Fusion #5: MLP Gate+Up+SiLU
// ============================================================================
static TimingResult test_fusion5_mlp(const FusionConfig& cfg) {
    size_t input_size = cfg.hidden_size;
    size_t gate_w_size = cfg.intermediate_size * cfg.hidden_size;
    size_t up_w_size = cfg.intermediate_size * cfg.hidden_size;
    size_t down_w_size = cfg.hidden_size * cfg.intermediate_size;
    size_t intermediate_size = cfg.intermediate_size;

    float *d_input, *d_gate_w, *d_up_w, *d_down_w, *d_gate, *d_up, *d_hidden, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate_w, gate_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up_w, up_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_down_w, down_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, cfg.hidden_size * sizeof(float)));

    auto h_input = generate_random_weights(input_size);
    auto h_gate_w = generate_random_weights(gate_w_size);
    auto h_up_w = generate_random_weights(up_w_size);
    auto h_down_w = generate_random_weights(down_w_size);

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate_w, h_gate_w.data(), gate_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_up_w, h_up_w.data(), up_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_down_w, h_down_w.data(), down_w_size * sizeof(float), cudaMemcpyHostToDevice));

    auto baseline_launch = [&]() {
        run_fusion5_mlp_baseline_chain(d_input, d_gate_w, d_up_w, d_down_w,
                                       d_gate, d_up, d_hidden, d_output,
                                       cfg.hidden_size, cfg.intermediate_size);
    };

    auto fused_launch = [&]() {
        run_fusion5_mlp_fused_gate_silu(d_input, d_gate_w, d_up_w, d_down_w,
                                        d_hidden, d_output,
                                        cfg.hidden_size, cfg.intermediate_size);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_input);
    cudaFree(d_gate_w);
    cudaFree(d_up_w);
    cudaFree(d_down_w);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_hidden);
    cudaFree(d_output);

    return result;
}

// ============================================================================
// Fusion #6: Post-RMSNorm + MLP + Residual
// ============================================================================
static TimingResult test_fusion6_postnorm_mlp_residual(const FusionConfig& cfg) {
    size_t hidden_size = cfg.hidden_size;
    size_t intermediate_size = cfg.intermediate_size;

    size_t gate_w_size = intermediate_size * hidden_size;
    size_t up_w_size = intermediate_size * hidden_size;
    size_t down_w_size = hidden_size * intermediate_size;

    float *d_residual_in, *d_post_norm_w, *d_gate_w, *d_up_w, *d_down_w;
    float *d_tmp_normed, *d_tmp_gate, *d_tmp_up, *d_tmp_hidden, *d_mlp_out;
    float *d_hidden_fused;

    CHECK_CUDA(cudaMalloc(&d_residual_in, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_post_norm_w, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate_w, gate_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up_w, up_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_down_w, down_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp_normed, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp_gate, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp_up, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp_hidden, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_out, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hidden_fused, intermediate_size * sizeof(float)));

    auto h_residual = generate_random_weights(hidden_size);
    auto h_post_norm_w = generate_random_weights(hidden_size, 0.9f, 1.1f);
    auto h_gate_w = generate_random_weights(gate_w_size);
    auto h_up_w = generate_random_weights(up_w_size);
    auto h_down_w = generate_random_weights(down_w_size);

    CHECK_CUDA(cudaMemcpy(d_residual_in, h_residual.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_post_norm_w, h_post_norm_w.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gate_w, h_gate_w.data(), gate_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_up_w, h_up_w.data(), up_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_down_w, h_down_w.data(), down_w_size * sizeof(float), cudaMemcpyHostToDevice));

    auto baseline_launch = [&]() {
        run_fusion6_chain_postnorm_mlp_residual(d_residual_in, d_post_norm_w, d_gate_w, d_up_w, d_down_w,
                                                d_tmp_normed, d_tmp_gate, d_tmp_up, d_tmp_hidden, d_mlp_out,
                                                cfg.hidden_size, cfg.intermediate_size);
    };

    auto fused_launch = [&]() {
        run_fusion6_fused_postnorm_mlp_residual(d_residual_in, d_post_norm_w, d_gate_w, d_up_w, d_down_w,
                                                d_tmp_normed, d_hidden_fused, d_mlp_out,
                                                cfg.hidden_size, cfg.intermediate_size);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_residual_in);
    cudaFree(d_post_norm_w);
    cudaFree(d_gate_w);
    cudaFree(d_up_w);
    cudaFree(d_down_w);
    cudaFree(d_tmp_normed);
    cudaFree(d_tmp_gate);
    cudaFree(d_tmp_up);
    cudaFree(d_tmp_hidden);
    cudaFree(d_mlp_out);
    cudaFree(d_hidden_fused);

    return result;
}

// ============================================================================
// Fusion #7: LA Conv1d + State Update + L2 Norm
// ============================================================================
static TimingResult test_fusion7_conv1d_l2norm(const FusionConfig& cfg) {
    int conv_dim = 16 * (128 * 2 + 128); // lnh * (kd*2 + vd)
    int conv_kernel = 4;
    int num_heads = 16;
    int key_dim = 128;
    float q_scale = 1.0f;

    size_t mixed_size = conv_dim;
    size_t conv_w_size = conv_dim * conv_kernel;
    size_t conv_state_size = conv_dim * (conv_kernel - 1);
    size_t qk_size = num_heads * key_dim;

    float *d_mixed, *d_conv_w, *d_conv_state, *d_conv_out;
    float *d_q, *d_k;

    CHECK_CUDA(cudaMalloc(&d_mixed, mixed_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv_w, conv_w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv_state, conv_state_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv_out, conv_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q, qk_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, qk_size * sizeof(float)));

    auto h_mixed = generate_random_weights(mixed_size);
    auto h_conv_w = generate_random_weights(conv_w_size);
    auto h_conv_state = generate_random_weights(conv_state_size);
    auto h_q = generate_random_weights(qk_size);
    auto h_k = generate_random_weights(qk_size);

    CHECK_CUDA(cudaMemcpy(d_mixed, h_mixed.data(), mixed_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_conv_w, h_conv_w.data(), conv_w_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_conv_state, h_conv_state.data(), conv_state_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), qk_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), qk_size * sizeof(float), cudaMemcpyHostToDevice));

    auto baseline_launch = [&]() {
        run_fusion7_conv1d_update_baseline(d_mixed, d_conv_w, d_conv_state, d_conv_out,
                                           conv_dim, conv_kernel);
        run_fusion7_l2norm_qk_baseline(d_q, d_k, num_heads, key_dim, q_scale);
    };

    auto fused_launch = [&]() {
        run_fusion7_conv1d_update_fused(d_mixed, d_conv_w, d_conv_state, d_conv_out,
                                        conv_dim, conv_kernel);
        run_fusion7_l2norm_qk_fused(d_q, d_k, num_heads, key_dim, q_scale);
    };

    TimingResult result;
    result.baseline_ms = measure_time_ms(baseline_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.fused_ms = measure_time_ms(fused_launch, cfg.warmup_rounds, cfg.num_rounds);
    result.speedup = result.baseline_ms / result.fused_ms;

    cudaFree(d_mixed);
    cudaFree(d_conv_w);
    cudaFree(d_conv_state);
    cudaFree(d_conv_out);
    cudaFree(d_q);
    cudaFree(d_k);

    return result;
}

// ============================================================================
// Main: Run all fusion comparisons
// ============================================================================
int main(int argc, char** argv) {
    FusionConfig cfg;
    if (argc > 1) cfg.num_rounds = std::atoi(argv[1]);
    if (argc > 2) cfg.warmup_rounds = std::atoi(argv[2]);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "  Operator Fusion Performance Comparison" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "CC:  " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Test rounds: " << cfg.num_rounds << " (warmup: " << cfg.warmup_rounds << ")" << std::endl;
    std::cout << std::endl;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(45) << "Fusion Point"
              << std::right << std::setw(12) << "Baseline(ms)"
              << std::setw(12) << "Fused(ms)"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(81, '-') << std::endl;

    auto test_and_print = [&](const std::string& name, const std::function<TimingResult()>& test_fn) {
        std::cout << std::left << std::setw(45) << name << std::flush;
        TimingResult result = test_fn();
        std::cout << std::right << std::setw(12) << result.baseline_ms
                  << std::setw(12) << result.fused_ms
                  << std::setw(11) << result.speedup << "x" << std::endl;
        return result;
    };

    TimingResult r1 = test_and_print("Fusion #1: Q proj + norm + RoPE", [&]() {
        return test_fusion1_q_path(cfg);
    });

    TimingResult r2 = test_and_print("Fusion #2: KV proj + norm + RoPE + cache", [&]() {
        return test_fusion2_kv_cache(cfg);
    });

    TimingResult r3 = test_and_print("Fusion #3: Attn core + gate + O_proj", [&]() {
        return test_fusion3_attn_gate_o(cfg);
    });

    TimingResult r4 = test_and_print("Fusion #4: RMSNorm + linear projection", [&]() {
        return test_fusion4_rmsnorm_linear(cfg);
    });

    TimingResult r5 = test_and_print("Fusion #5: MLP gate+up+SiLU", [&]() {
        return test_fusion5_mlp(cfg);
    });

    TimingResult r6 = test_and_print("Fusion #6: PostNorm + MLP + residual", [&]() {
        return test_fusion6_postnorm_mlp_residual(cfg);
    });

    TimingResult r7 = test_and_print("Fusion #7: Conv1d + state + L2 norm", [&]() {
        return test_fusion7_conv1d_l2norm(cfg);
    });

    std::cout << std::string(81, '-') << std::endl;

    double avg_speedup = (r1.speedup + r2.speedup + r3.speedup + r4.speedup +
                          r5.speedup + r6.speedup + r7.speedup) / 7.0;
    double total_baseline = r1.baseline_ms + r2.baseline_ms + r3.baseline_ms + r4.baseline_ms +
                            r5.baseline_ms + r6.baseline_ms + r7.baseline_ms;
    double total_fused = r1.fused_ms + r2.fused_ms + r3.fused_ms + r4.fused_ms +
                         r5.fused_ms + r6.fused_ms + r7.fused_ms;

    std::cout << std::left << std::setw(45) << "AVERAGE"
              << std::right << std::setw(12) << ""
              << std::setw(12) << ""
              << std::setw(11) << avg_speedup << "x" << std::endl;
    std::cout << std::left << std::setw(45) << "TOTAL"
              << std::right << std::setw(12) << total_baseline
              << std::setw(12) << total_fused
              << std::setw(11) << (total_baseline / total_fused) << "x" << std::endl;

    std::cout << std::endl << std::string(80, '=') << std::endl;
    std::cout << "  Test Complete" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
