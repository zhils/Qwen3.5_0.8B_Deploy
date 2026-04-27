/**
 * Qwen3.5-0.8B v3.0 Performance Test
 * All layers use Flash Attention (Full Attention)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <string>

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

// Qwen3.5-0.8B config
static const int NUM_LAYERS = 36;
static const int HIDDEN_SIZE = 1024;
static const int INTERMEDIATE_SIZE = 3584;
static const int VOCAB_SIZE = 151936;
static const int MAX_SEQ_LEN = 8192;

// v3.0: All Flash Attention
static const int NUM_HEADS = 8;
static const int NUM_KV_HEADS = 2;
static const int Q_HEAD_DIM = 256;
static const int KV_HEAD_DIM = 256;

static const int PREFILL_TOKENS = 1024;
static const int DECODE_TOKENS = 32;
static const int BATCH_SIZE = 16;
static const int ROUNDS = 1;

std::vector<float> generate_random_weights(int count) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.02f, 0.02f);
    std::vector<float> weights(count);
    for (auto& w : weights) {
        w = dis(gen);
    }
    return weights;
}

void load_random_weights(CudaEngineV3& engine) {
    int hs = HIDDEN_SIZE;
    int isz = INTERMEDIATE_SIZE;
    int fnh = NUM_HEADS;
    int qhd = Q_HEAD_DIM;
    int khd = KV_HEAD_DIM;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        std::vector<float> weights;

        // input_norm + post_norm
        weights.insert(weights.end(), hs, 1.0f);
        weights.insert(weights.end(), hs, 1.0f);

        // MLP gate + up + down
        auto mlp_gate = generate_random_weights(isz * hs);
        auto mlp_up = generate_random_weights(isz * hs);
        auto mlp_down = generate_random_weights(hs * isz);
        weights.insert(weights.end(), mlp_gate.begin(), mlp_gate.end());
        weights.insert(weights.end(), mlp_up.begin(), mlp_up.end());
        weights.insert(weights.end(), mlp_down.begin(), mlp_down.end());

        // v3.0: Full Attention weights only
        auto fq = generate_random_weights(fnh * qhd * 2 * hs);
        auto fk = generate_random_weights(NUM_KV_HEADS * khd * hs);
        auto fv = generate_random_weights(NUM_KV_HEADS * khd * hs);
        auto fqn = generate_random_weights(khd);
        auto fkn = generate_random_weights(khd);
        auto fo = generate_random_weights(hs * fnh * khd);

        weights.insert(weights.end(), fq.begin(), fq.end());
        weights.insert(weights.end(), fk.begin(), fk.end());
        weights.insert(weights.end(), fv.begin(), fv.end());
        weights.insert(weights.end(), fqn.begin(), fqn.end());
        weights.insert(weights.end(), fkn.begin(), fkn.end());
        weights.insert(weights.end(), fo.begin(), fo.end());

        engine.set_layer_weights(layer, weights);
    }

    std::vector<float> final_norm(hs, 1.0f);
    engine.set_final_norm_weight(final_norm);

    auto lm_head_w = generate_random_weights(VOCAB_SIZE * hs);
    engine.set_lm_head_weight(lm_head_w);
}

float benchmark_prefill(CudaEngineV3& engine, int tokens, int batch_size, int rounds) {
    std::vector<float> h_input(batch_size * HIDDEN_SIZE, 0.01f);
    std::vector<int> positions(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        positions[b] = tokens - 1;
    }

    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * HIDDEN_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Warm-up
    engine.reset_cache();
    engine.forward_batch_prefill(d_input, d_output, positions.data(), batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; ++r) {
        engine.reset_cache();
        engine.forward_batch_prefill(d_input, d_output, positions.data(), batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();

    float total_ms = std::chrono::duration<float, std::milli>(end - start).count();

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return total_ms / rounds;
}

float benchmark_decode(CudaEngineV3& engine, int tokens, int batch_size, int rounds) {
    std::vector<float> h_input(HIDDEN_SIZE, 0.01f);

    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), HIDDEN_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Warm-up: prefill first
    engine.reset_cache();
    std::vector<int> prefill_pos(batch_size, tokens - 1);
    std::vector<float> batch_input(batch_size * HIDDEN_SIZE, 0.01f);
    float* d_batch_input;
    float* d_batch_output;
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_output, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_batch_input, batch_input.data(), batch_size * HIDDEN_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));
    engine.forward_batch_prefill(d_batch_input, d_batch_output, prefill_pos.data(), batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < rounds; ++r) {
        for (int t = 0; t < tokens; ++t) {
            engine.forward(d_input, d_output, tokens + t);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();

    float total_ms = std::chrono::duration<float, std::milli>(end - start).count();

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_batch_input));
    CUDA_CHECK(cudaFree(d_batch_output));

    return total_ms / rounds;
}

size_t get_gpu_vram_used() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

int main() {
    printf("========================================================================\n");
    printf("  Qwen3.5-0.8B v3.0 Performance Test (All Flash Attention)\n");
    printf("========================================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("CC:  %d.%d\n", prop.major, prop.minor);
    printf("VRAM:%llu MB\n", (unsigned long long)(prop.totalGlobalMem / (1024 * 1024)));
    printf("\n");

    printf("========================================================================\n");
    printf("  v3.0 Performance Test (All Flash Attention)\n");
    printf("========================================================================\n");
    printf("  Prefill tokens:     %d\n", PREFILL_TOKENS);
    printf("  Decode tokens:      %d\n", DECODE_TOKENS);
    printf("  Batch size:         %d\n", BATCH_SIZE);
    printf("  Rounds:             %d\n", ROUNDS);
    printf("  Layers:             %d (ALL Flash Attention)\n", NUM_LAYERS);
    printf("\n");

    CudaEngineV3 engine(NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE, MAX_SEQ_LEN);

    printf("[Loading random weights...]\n\n");
    load_random_weights(engine);

    size_t vram_before = get_gpu_vram_used();

    printf("[Running %d round(s)...]\n", ROUNDS);

    float prefill_ms = benchmark_prefill(engine, PREFILL_TOKENS, BATCH_SIZE, ROUNDS);
    printf("  Prefill benchmark completed\n");

    float decode_ms = benchmark_decode(engine, PREFILL_TOKENS, BATCH_SIZE, ROUNDS);
    printf("  Decode benchmark completed\n");

    size_t vram_after = get_gpu_vram_used();

    printf("\n");
    printf("========================================================================\n");
    printf("  RESULTS\n");
    printf("========================================================================\n");
    printf("\n");

    printf("--- Prefill (%d tokens) ---\n", PREFILL_TOKENS);
    printf("  Total time:      %.3f ms\n", prefill_ms);
    printf("  TTFT:            %.3f ms\n", prefill_ms);
    printf("  Throughput:       %.3f tokens/sec\n",
           (PREFILL_TOKENS * BATCH_SIZE) / (prefill_ms / 1000.0f));
    printf("\n");

    printf("--- Decode (%d tokens) ---\n", DECODE_TOKENS);
    printf("  TPOT:               %.3f ms/token\n", decode_ms / DECODE_TOKENS);
    printf("  Throughput:     %.3f tokens/sec\n",
           (DECODE_TOKENS * BATCH_SIZE) / (decode_ms / 1000.0f));
    printf("\n");

    printf("--- Memory ---\n");
    printf("  GPU VRAM used:  %.3f MB\n", vram_after / (1024.0f * 1024.0f));
    printf("  GPU VRAM total: %.3f MB\n", prop.totalGlobalMem / (1024.0f * 1024.0f));
    printf("\n");

    printf("========================================================================\n");
    printf("  Test Complete\n");
    printf("========================================================================\n");

    return 0;
}
