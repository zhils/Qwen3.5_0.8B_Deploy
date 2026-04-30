/**
 * Qwen3.5-0.8B v3.3 Performance Test
 * Hybrid: Full Attention (Prefill) + Linear Attention (Decode)
 * With REAL weights from safetensors export
 *
 * Usage: ./performance_test_v3 [prefill_tokens] [decode_tokens] [rounds] [batch_size]
 * Default: prefill=1024, decode=512, rounds=5, batch=1
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <numeric>
#include <fstream>
#include <iostream>

#include "cuda_engine_v3.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

using namespace qwen::cuda;

size_t get_gpu_vram_used() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

std::vector<float> load_binary_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open weight file: %s\n", filepath.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = size / sizeof(float);
    std::vector<float> weights(num_floats);

    if (!file.read(reinterpret_cast<char*>(weights.data()), size)) {
        fprintf(stderr, "Failed to read weight file: %s\n", filepath.c_str());
        return {};
    }

    return weights;
}

bool load_real_weights(CudaEngineV3& engine, int num_layers, const std::string& weights_dir) {
    printf("[Loading REAL weights from %s]\n", weights_dir.c_str());

    bool all_loaded = true;

    // Load layer weights
    for (int layer = 0; layer < num_layers; ++layer) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/layer_%02d.bin", weights_dir.c_str(), layer);

        std::vector<float> weights = load_binary_weights(filename);
        if (weights.empty()) {
            fprintf(stderr, "Warning: failed to load layer %d weights\n", layer);
            all_loaded = false;
            continue;
        }

        try {
            engine.set_layer_weights(layer, weights);
            printf("  Layer %2d: loaded %zu floats\n", layer, weights.size());
        } catch (const std::exception& e) {
            fprintf(stderr, "Error setting layer %d weights: %s\n", layer, e.what());
            all_loaded = false;
        }
    }

    // Load final norm
    std::string norm_path = weights_dir + "/norm.bin";
    std::vector<float> norm_weights = load_binary_weights(norm_path);
    if (!norm_weights.empty()) {
        engine.set_final_norm_weight(norm_weights);
        printf("  Final norm: loaded %zu floats\n", norm_weights.size());
    } else {
        fprintf(stderr, "Warning: failed to load final norm weights\n");
        all_loaded = false;
    }

    // Load embedding (also used as lm_head since tied)
    std::string embed_path = weights_dir + "/embedding.bin";
    std::vector<float> embed_weights = load_binary_weights(embed_path);
    if (!embed_weights.empty()) {
        engine.set_shared_embedding_lmhead_weight(embed_weights);
        printf("  Embedding+LMHead: loaded %zu floats (shared weights)\n", embed_weights.size());
    } else {
        fprintf(stderr, "Warning: failed to load embedding weights\n");
        all_loaded = false;
    }

    return all_loaded;
}

void load_random_weights(CudaEngineV3& engine, int num_layers, int hidden_size,
                         int intermediate_size, int vocab_size, int num_heads, int num_kv_heads,
                         int q_head_dim, int kv_head_dim) {
    printf("[Loading RANDOM weights]\n\n");

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.02f, 0.02f);

    auto generate_random_weights = [&](int count) {
        std::vector<float> weights(count);
        for (auto& w : weights) {
            w = dis(gen);
        }
        return weights;
    };

    auto is_full_attention_layer = [](int layer_idx) {
        return (layer_idx % 4) == 3;
    };

    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<float> weights;

        weights.insert(weights.end(), hidden_size, 1.0f);
        weights.insert(weights.end(), hidden_size, 1.0f);

        auto mlp_gate = generate_random_weights(intermediate_size * hidden_size);
        auto mlp_up = generate_random_weights(intermediate_size * hidden_size);
        auto mlp_down = generate_random_weights(hidden_size * intermediate_size);
        weights.insert(weights.end(), mlp_gate.begin(), mlp_gate.end());
        weights.insert(weights.end(), mlp_up.begin(), mlp_up.end());
        weights.insert(weights.end(), mlp_down.begin(), mlp_down.end());

        if (is_full_attention_layer(layer)) {
            // Q: [num_heads * head_dim, hidden_size] = [2048, 1024]
            auto fq = generate_random_weights(num_heads * q_head_dim * hidden_size);
            // K/V: [num_kv_heads * head_dim, hidden_size] = [512, 1024]
            auto fk = generate_random_weights(num_kv_heads * kv_head_dim * hidden_size);
            auto fv = generate_random_weights(num_kv_heads * kv_head_dim * hidden_size);
            auto fqn = generate_random_weights(kv_head_dim);
            auto fkn = generate_random_weights(kv_head_dim);
            // O: [hidden_size, num_heads * head_dim] = [1024, 2048]
            auto fo = generate_random_weights(hidden_size * num_heads * q_head_dim);

            weights.insert(weights.end(), fq.begin(), fq.end());
            weights.insert(weights.end(), fk.begin(), fk.end());
            weights.insert(weights.end(), fv.begin(), fv.end());
            weights.insert(weights.end(), fqn.begin(), fqn.end());
            weights.insert(weights.end(), fkn.begin(), fkn.end());
            weights.insert(weights.end(), fo.begin(), fo.end());
        } else {
            int linear_num_heads = 16;
            int linear_key_dim = 128;
            int linear_value_dim = 128;
            int conv_kernel = 4;

            int k_dim = linear_num_heads * linear_key_dim;
            int v_dim = linear_num_heads * linear_value_dim;
            int conv_dim = k_dim * 2 + v_dim;
            int z_dim = linear_num_heads * linear_value_dim;

            auto in_proj_qkv = generate_random_weights(conv_dim * hidden_size);
            auto in_proj_a = generate_random_weights(linear_num_heads * hidden_size);
            auto in_proj_b = generate_random_weights(linear_num_heads * hidden_size);
            auto in_proj_z = generate_random_weights(z_dim * hidden_size);
            auto conv1d_w = generate_random_weights(conv_dim * conv_kernel);
            auto out_proj = generate_random_weights(hidden_size * z_dim);
            auto a_log = generate_random_weights(linear_num_heads);
            auto dt_bias = generate_random_weights(linear_num_heads);
            auto norm_w = generate_random_weights(linear_value_dim);

            weights.insert(weights.end(), in_proj_qkv.begin(), in_proj_qkv.end());
            weights.insert(weights.end(), in_proj_a.begin(), in_proj_a.end());
            weights.insert(weights.end(), in_proj_b.begin(), in_proj_b.end());
            weights.insert(weights.end(), in_proj_z.begin(), in_proj_z.end());
            weights.insert(weights.end(), conv1d_w.begin(), conv1d_w.end());
            weights.insert(weights.end(), out_proj.begin(), out_proj.end());
            weights.insert(weights.end(), a_log.begin(), a_log.end());
            weights.insert(weights.end(), dt_bias.begin(), dt_bias.end());
            weights.insert(weights.end(), norm_w.begin(), norm_w.end());
        }

        engine.set_layer_weights(layer, weights);
    }

    std::vector<float> final_norm(hidden_size, 1.0f);
    engine.set_final_norm_weight(final_norm);

    auto embed_w = generate_random_weights(vocab_size * hidden_size);
    engine.set_shared_embedding_lmhead_weight(embed_w);
}

int main(int argc, char** argv) {
    int prefill_tokens = 1024;
    int decode_tokens = 512;
    int rounds = 5;
    int batch_size = 1;

    if (argc >= 2) prefill_tokens = atoi(argv[1]);
    if (argc >= 3) decode_tokens = atoi(argv[2]);
    if (argc >= 4) rounds = atoi(argv[3]);
    if (argc >= 5) batch_size = atoi(argv[4]);

    printf("========================================================================\n");
    printf("  Qwen3.5-0.8B v3.3 Performance Test (REAL Weights)\n");
    printf("========================================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("CC:  %d.%d\n", prop.major, prop.minor);
    printf("VRAM:%llu MB\n", (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));
    printf("\n");

    printf("========================================================================\n");
    printf("  Test Configuration\n");
    printf("========================================================================\n");
    printf("  Prefill tokens:     %d\n", prefill_tokens);
    printf("  Decode tokens:      %d\n", decode_tokens);
    printf("  Batch size:         %d\n", batch_size);
    printf("  Rounds:             %d\n", rounds);
    printf("  Architecture:       Hybrid (Full Attn Prefill + Linear Attn Decode)\n");
    printf("\n");

    const int NUM_LAYERS = 24;
    const int HIDDEN_SIZE = 1024;
    const int INTERMEDIATE_SIZE = 3584;
    const int VOCAB_SIZE = 248320;
    const int MAX_SEQ_LEN = 8192;
    const int NUM_HEADS = 8;
    const int NUM_KV_HEADS = 2;
    const int Q_HEAD_DIM = 256;
    const int KV_HEAD_DIM = 256;

    // Pre-initialize CUDA context before creating engine
    printf("[Initializing CUDA context...]\n");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    printf("[CUDA context initialized OK]\n\n");

    CudaEngineV3 engine(NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE, MAX_SEQ_LEN);

    // Try to load real weights
    std::string weights_dir = "/mnt/d/deploy/Qwen3.5_0.8B_Deploy/weights";
    bool use_real_weights = load_real_weights(engine, NUM_LAYERS, weights_dir);

    if (!use_real_weights) {
        printf("\n[WARNING: Using RANDOM weights as fallback]\n\n");
        load_random_weights(engine, NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE,
                           NUM_HEADS, NUM_KV_HEADS, Q_HEAD_DIM, KV_HEAD_DIM);
    } else {
        printf("\n[SUCCESS: Using REAL model weights]\n\n");
    }
    
    // Verify weights are loaded by checking a small forward pass
    printf("[Verifying engine with single token...]\n");
    float* d_test_output;
    CUDA_CHECK(cudaMalloc(&d_test_output, HIDDEN_SIZE * sizeof(float)));
    std::vector<int> test_token = {0};
    std::vector<int> test_pos = {0};
    try {
        engine.forward_tokens(test_token, d_test_output, test_pos.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Engine verification OK]\n\n");
    } catch (const std::exception& e) {
        printf("[Engine verification FAILED: %s]\n\n", e.what());
        cudaFree(d_test_output);
        return 1;
    }
    cudaFree(d_test_output);

    size_t vram_before = get_gpu_vram_used();
    printf("[GPU VRAM before test: %.3f MB]\n\n", vram_before / (1024.0f * 1024.0f));

    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, HIDDEN_SIZE * sizeof(float)));

    printf("[Warming up engine...]\n");
    
    // Check GPU memory before reset_cache
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU free memory before reset_cache: %.2f MB\n", free_mem / (1024.0f * 1024.0f));
    
    engine.reset_cache();
    
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU free memory after reset_cache: %.2f MB\n", free_mem / (1024.0f * 1024.0f));
    
    std::vector<int> warmup_ids(prefill_tokens);
    std::iota(warmup_ids.begin(), warmup_ids.end(), 0);
    std::vector<int> warmup_positions(prefill_tokens);
    std::iota(warmup_positions.begin(), warmup_positions.end(), 0);
    
    printf("  Running forward_tokens...\n");
    engine.forward_tokens(warmup_ids, d_output, warmup_positions.data());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Warmup complete]\n\n");

    printf("========================================================================\n");
    printf("  Running Prefill Benchmark (%d tokens, %d rounds)\n", prefill_tokens, rounds);
    printf("========================================================================\n");

    std::vector<int> prefill_ids(prefill_tokens);
    std::iota(prefill_ids.begin(), prefill_ids.end(), 0);

    std::vector<int> positions(prefill_tokens);
    std::iota(positions.begin(), positions.end(), 0);

    float total_prefill_ms = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        engine.reset_cache();

        auto start = std::chrono::high_resolution_clock::now();
        engine.forward_tokens(prefill_ids, d_output, positions.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(end - start).count();
        total_prefill_ms += ms;
        printf("  Round %d: %.3f ms\n", r + 1, ms);
    }

    float avg_prefill_ms = total_prefill_ms / rounds;
    float prefill_throughput = (prefill_tokens * batch_size) / (avg_prefill_ms / 1000.0f);

    printf("\n  Average Prefill: %.3f ms\n", avg_prefill_ms);
    printf("  Prefill Throughput: %.3f tokens/sec\n", prefill_throughput);

    printf("\n========================================================================\n");
    printf("  Running Decode Benchmark (%d tokens, %d rounds)\n", decode_tokens, rounds);
    printf("========================================================================\n");

    engine.reset_cache();
    engine.forward_tokens(prefill_ids, d_output, positions.data());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[Prefill done, starting decode...]\n");

    float total_decode_ms = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        engine.reset_cache();
        engine.forward_tokens(prefill_ids, d_output, positions.data());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < decode_tokens; ++t) {
            int token_id = (t % 100) + 1;
            int pos = prefill_tokens + t;
            engine.forward_token(token_id, d_output, pos);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(end - start).count();
        total_decode_ms += ms;
        printf("  Round %d: %.3f ms (%.3f ms/token)\n", r + 1, ms, ms / decode_tokens);
    }

    float avg_decode_ms = total_decode_ms / rounds;
    float decode_tpot = avg_decode_ms / decode_tokens;
    float decode_throughput = (decode_tokens * batch_size) / (avg_decode_ms / 1000.0f);

    printf("\n  Average Decode: %.3f ms total, %.3f ms/token\n", avg_decode_ms, decode_tpot);
    printf("  Decode Throughput: %.3f tokens/sec\n", decode_throughput);

    size_t vram_after = get_gpu_vram_used();

    float total_time = avg_prefill_ms + avg_decode_ms;
    float e2e_throughput = ((prefill_tokens + decode_tokens) * batch_size) / (total_time / 1000.0f);

    CUDA_CHECK(cudaFree(d_output));

    printf("\n");
    printf("========================================================================\n");
    printf("  RESULTS SUMMARY\n");
    printf("========================================================================\n");
    printf("\n");

    printf("--- Prefill (%d tokens, batch=%d) ---\n", prefill_tokens, batch_size);
    printf("  TTFT:            %.3f ms\n", avg_prefill_ms);
    printf("  Throughput:      %.3f tokens/sec\n", prefill_throughput);
    printf("\n");

    printf("--- Decode (%d tokens, batch=%d) ---\n", decode_tokens, batch_size);
    printf("  TPOT:            %.3f ms/token\n", decode_tpot);
    printf("  Throughput:       %.3f tokens/sec\n", decode_throughput);
    printf("\n");

    printf("--- End-to-End (%d+%d tokens) ---\n", prefill_tokens, decode_tokens);
    printf("  Total time:      %.3f ms\n", total_time);
    printf("  E2E throughput:  %.3f tokens/sec\n", e2e_throughput);
    printf("\n");

    printf("--- Memory ---\n");
    printf("  GPU VRAM used:   %.3f MB\n", vram_after / (1024.0f * 1024.0f));
    printf("  GPU VRAM total:  %.3f MB\n", prop.totalGlobalMem / (1024.0f * 1024.0f));
    printf("\n");

    printf("========================================================================\n");
    printf("  Test Complete\n");
    printf("========================================================================\n");

    return 0;
}
