/**
 * Qwen3.5-0.8B Memory Analysis Tool
 * Detailed memory consumption breakdown for deployment
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_engine_v3.hpp"

#define CUDA_CHECK(call)                                                                             \
    do {                                                                                             \
        cudaError_t err = call;                                                                      \
        if (err != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                         \
                    cudaGetErrorString(err));                                                        \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

using namespace qwen::cuda;

static const int NUM_LAYERS = 36;
static const int HIDDEN_SIZE = 1024;
static const int INTERMEDIATE_SIZE = 3584;
static const int VOCAB_SIZE = 151936;
static const int MAX_SEQ_LEN = 8192;
static const int NUM_HEADS = 8;
static const int NUM_KV_HEADS = 2;
static const int Q_HEAD_DIM = 256;
static const int KV_HEAD_DIM = 256;

std::vector<float> generate_random_weights(int count) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.02f, 0.02f);
    std::vector<float> weights(count);
    for (auto& w : weights) {
        w = dis(gen);
    }
    return weights;
}

size_t get_gpu_memory_used() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

void print_memory_snapshot(const char* label, size_t baseline = 0) {
    size_t used = get_gpu_memory_used();
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);

    printf("  [%-30s] Used: %7.2f MB", label, used / (1024.0 * 1024.0));
    if (baseline > 0) {
        printf("  (Delta: %+7.2f MB)", (double)(used - baseline) / (1024.0 * 1024.0));
    }
    printf("\n");
}

int main() {
    printf("\n");
    printf("========================================================================\n");
    printf("  Qwen3.5-0.8B Memory Analysis Tool\n");
    printf("========================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total VRAM: %.2f MB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0));

    size_t baseline = get_gpu_memory_used();
    print_memory_snapshot("Baseline (empty)", baseline);

    printf("\n--- Creating Engine ---\n");
    CudaEngineV3 engine(NUM_LAYERS, HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE, MAX_SEQ_LEN);
    size_t after_engine = get_gpu_memory_used();
    print_memory_snapshot("After engine creation", baseline);

    printf("\n--- Loading Weights ---\n");

    size_t before_weights = get_gpu_memory_used();

    int hs = HIDDEN_SIZE;
    int isz = INTERMEDIATE_SIZE;
    int fnh = NUM_HEADS;
    int qhd = Q_HEAD_DIM;
    int khd = KV_HEAD_DIM;

    size_t total_weight_bytes = 0;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        std::vector<float> weights;

        weights.insert(weights.end(), hs, 1.0f);
        weights.insert(weights.end(), hs, 1.0f);

        auto mlp_gate = generate_random_weights(isz * hs);
        auto mlp_up = generate_random_weights(isz * hs);
        auto mlp_down = generate_random_weights(hs * isz);
        weights.insert(weights.end(), mlp_gate.begin(), mlp_gate.end());
        weights.insert(weights.end(), mlp_up.begin(), mlp_up.end());
        weights.insert(weights.end(), mlp_down.begin(), mlp_down.end());

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

        total_weight_bytes += weights.size() * sizeof(float);

        engine.set_layer_weights(layer, weights);
    }

    size_t after_layer_weights = get_gpu_memory_used();
    print_memory_snapshot("After layer weights", before_weights);

    std::vector<float> final_norm(hs, 1.0f);
    engine.set_final_norm_weight(final_norm);

    size_t after_final_norm = get_gpu_memory_used();
    print_memory_snapshot("After final norm", after_layer_weights);

    auto lm_head_w = generate_random_weights(VOCAB_SIZE * hs);
    total_weight_bytes += VOCAB_SIZE * hs * sizeof(float);
    engine.set_lm_head_weight(lm_head_w);

    size_t after_lm_head = get_gpu_memory_used();
    print_memory_snapshot("After LM Head", after_final_norm);

    printf("\n--- Memory Breakdown ---\n\n");

    size_t engine_overhead = after_engine - baseline;
    size_t layer_weights_mem = after_layer_weights - before_weights;
    size_t final_norm_mem = after_final_norm - after_layer_weights;
    size_t lm_head_mem = after_lm_head - after_final_norm;
    size_t total_used = after_lm_head - baseline;

    printf("  %-30s %10.2f MB  (%5.1f%%)\n", "Engine buffers & KV Cache", 
           engine_overhead / (1024.0 * 1024.0), 100.0 * engine_overhead / total_used);
    printf("  %-30s %10.2f MB  (%5.1f%%)\n", "Layer weights (FP32)", 
           layer_weights_mem / (1024.0 * 1024.0), 100.0 * layer_weights_mem / total_used);
    printf("  %-30s %10.2f MB  (%5.1f%%)\n", "Final norm", 
           final_norm_mem / (1024.0 * 1024.0), 100.0 * final_norm_mem / total_used);
    printf("  %-30s %10.2f MB  (%5.1f%%)\n", "LM Head (BF16)", 
           lm_head_mem / (1024.0 * 1024.0), 100.0 * lm_head_mem / total_used);
    printf("  %s\n", "----------------------------------------");
    printf("  %-30s %10.2f MB  (100.0%%)\n", "TOTAL", total_used / (1024.0 * 1024.0));

    printf("\n--- Theoretical Memory Calculation ---\n\n");

    double mlp_per_layer = (isz * hs + isz * hs + hs * isz) * sizeof(float);
    double attn_per_layer = (fnh * qhd * 2 * hs + NUM_KV_HEADS * khd * hs * 2 + khd * 2 +
                             hs * fnh * khd) * sizeof(float);
    double norm_per_layer = hs * 2 * sizeof(float);
    double layer_total = mlp_per_layer + attn_per_layer + norm_per_layer;

    printf("  Per-layer breakdown:\n");
    printf("    MLP weights:       %8.2f MB\n", mlp_per_layer / (1024.0 * 1024.0));
    printf("    Attention weights: %8.2f MB\n", attn_per_layer / (1024.0 * 1024.0));
    printf("    Norm weights:      %8.2f MB\n", norm_per_layer / (1024.0 * 1024.0));
    printf("    Layer total:       %8.2f MB\n", layer_total / (1024.0 * 1024.0));

    double all_layers = layer_total * NUM_LAYERS;
    double final_norm_theory = hs * sizeof(float);
    double lm_head_theory = VOCAB_SIZE * hs * sizeof(float) / 2;  // BF16
    double kv_cache_theory = NUM_LAYERS * MAX_SEQ_LEN * NUM_KV_HEADS * KV_HEAD_DIM * sizeof(float) * 2;

    printf("\n  Total theoretical:\n");
    printf("    All layers:        %8.2f MB\n", all_layers / (1024.0 * 1024.0));
    printf("    Final norm:        %8.2f MB\n", final_norm_theory / (1024.0 * 1024.0));
    printf("    LM Head (BF16):    %8.2f MB\n", lm_head_theory / (1024.0 * 1024.0));
    printf("    KV Cache (max):    %8.2f MB\n", kv_cache_theory / (1024.0 * 1024.0));

    double grand_total_theory = all_layers + final_norm_theory + lm_head_theory + kv_cache_theory;
    printf("    %s\n", "----------------------------------------");
    printf("    Grand total:       %8.2f MB\n", grand_total_theory / (1024.0 * 1024.0));

    printf("\n--- KV Cache Analysis ---\n\n");

    double kv_per_token = NUM_LAYERS * NUM_KV_HEADS * KV_HEAD_DIM * sizeof(float) * 2;
    printf("  KV Cache per token: %.4f MB\n", kv_per_token / (1024.0 * 1024.0));
    printf("  KV Cache at 512 tokens:  %8.2f MB\n", kv_per_token * 512 / (1024.0 * 1024.0));
    printf("  KV Cache at 1024 tokens: %8.2f MB\n", kv_per_token * 1024 / (1024.0 * 1024.0));
    printf("  KV Cache at 2048 tokens: %8.2f MB\n", kv_per_token * 2048 / (1024.0 * 1024.0));
    printf("  KV Cache at 4096 tokens: %8.2f MB\n", kv_per_token * 4096 / (1024.0 * 1024.0));
    printf("  KV Cache at 8192 tokens: %8.2f MB\n", kv_per_token * 8192 / (1024.0 * 1024.0));

    printf("\n--- Optimization Potential ---\n\n");

    double weights_fp32 = all_layers;
    double weights_bf16 = weights_fp32 / 2;
    double weights_int8 = weights_fp32 / 4;

    printf("  Current (FP32 weights + BF16 LM Head):\n");
    printf("    Total: %.2f MB\n", (weights_fp32 + lm_head_theory + kv_cache_theory) / (1024.0 * 1024.0));

    printf("\n  With BF16 weights:\n");
    printf("    Weights: %.2f MB -> %.2f MB (save %.2f MB)\n", 
           weights_fp32 / (1024.0 * 1024.0), weights_bf16 / (1024.0 * 1024.0),
           (weights_fp32 - weights_bf16) / (1024.0 * 1024.0));
    printf("    Total: %.2f MB\n", (weights_bf16 + lm_head_theory + kv_cache_theory) / (1024.0 * 1024.0));

    printf("\n  With INT8 weights:\n");
    printf("    Weights: %.2f MB -> %.2f MB (save %.2f MB)\n", 
           weights_fp32 / (1024.0 * 1024.0), weights_int8 / (1024.0 * 1024.0),
           (weights_fp32 - weights_int8) / (1024.0 * 1024.0));
    printf("    Total: %.2f MB\n", (weights_int8 + lm_head_theory + kv_cache_theory) / (1024.0 * 1024.0));

    printf("\n  With Embedding/LM Head sharing (tied):\n");
    printf("    Save: %.2f MB (one BF16 embedding)\n", lm_head_theory / (1024.0 * 1024.0));

    printf("\n========================================================================\n");
    printf("  Memory Analysis Complete\n");
    printf("========================================================================\n\n");

    return 0;
}
