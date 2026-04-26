/**
 * v2.0 Performance Test
 *
 * Tests all v2.0 optimizations enabled:
 * - Flash Attention v2
 * - Tensor Core (TF32)
 * - Async Data Transfer
 * - Multi-Stream Parallelism
 *
 * Test: 1024 prefill tokens / 512 decode tokens, 5 rounds (default)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>

#include "cuda_engine.hpp"
#include "token_embedding_cuda.hpp"
#include "lm_head_cuda.hpp"
#include "gpu_sampler_argmax.hpp"
#include "flash_attention.cuh"
#include "mlp_cuda.hpp"

using namespace qwen;
using namespace qwen::cuda;

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":"   \
                      << __LINE__ << std::endl;                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

struct TestConfig {
    int num_layers = 24;
    int hidden_size = 1024;
    int intermediate_size = 3584;
    int vocab_size = 248320;
    int num_heads = 8;
    int num_kv_heads = 2;
    int q_head_dim = 256;
    int kv_head_dim = 256;
    int max_seq_len = 2048;
    int prefill_tokens = 1024;
    int decode_tokens = 512;
    int num_rounds = 5;
    int batch_size = 128;
};

struct LatencyStats {
    std::vector<double> samples;

    void add(double ms) {
        samples.push_back(ms);
    }

    double avg() const {
        if (samples.empty()) return 0;
        double sum = 0;
        for (auto s : samples) sum += s;
        return sum / samples.size();
    }

    double p50() const {
        if (samples.empty()) return 0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() * 50 / 100];
    }

    double p95() const {
        if (samples.empty()) return 0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        return sorted[sorted.size() * 95 / 100];
    }

    double min_ms() const {
        if (samples.empty()) return 0;
        return *std::min_element(samples.begin(), samples.end());
    }

    double max_ms() const {
        if (samples.empty()) return 0;
        return *std::max_element(samples.begin(), samples.end());
    }
};

static std::vector<float> generate_random_weights(size_t count, float min_val = -0.02f,
                                                   float max_val = 0.02f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    std::vector<float> weights(count);
    for (auto& w : weights) w = dis(gen);
    return weights;
}

static void append_weights(std::vector<float>& flat, size_t count, float min_val = -0.02f,
                            float max_val = 0.02f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    flat.reserve(flat.size() + count);
    for (size_t i = 0; i < count; ++i) {
        flat.push_back(dis(gen));
    }
}

static std::vector<float> build_flat_weights(bool is_full, const TestConfig& cfg) {
    std::vector<float> flat;
    int hs = cfg.hidden_size;
    int isz = cfg.intermediate_size;
    int fnh = cfg.num_heads;
    int nkh = cfg.num_kv_heads;
    int qhd = cfg.q_head_dim;
    int khd = cfg.kv_head_dim;
    int lnh = 16, kd = 128, vd = 128, ck = 4;

    flat.insert(flat.end(), hs, 1.0f);
    flat.insert(flat.end(), hs, 1.0f);
    append_weights(flat, static_cast<size_t>(isz) * hs);
    append_weights(flat, static_cast<size_t>(isz) * hs);
    append_weights(flat, static_cast<size_t>(hs) * isz);

    if (!is_full) {
        int conv_dim = lnh * (kd * 2 + vd);
        append_weights(flat, static_cast<size_t>(conv_dim) * hs);
        append_weights(flat, static_cast<size_t>(lnh) * hs);
        append_weights(flat, static_cast<size_t>(lnh) * hs);
        append_weights(flat, static_cast<size_t>(lnh) * vd * hs);
        append_weights(flat, static_cast<size_t>(conv_dim) * ck);
        append_weights(flat, static_cast<size_t>(hs) * lnh * vd);
        flat.insert(flat.end(), lnh, 0.0f);
        flat.insert(flat.end(), lnh, 0.0f);
        flat.insert(flat.end(), vd, 1.0f);
    } else {
        int conv_dim = lnh * (kd * 2 + vd);
        flat.insert(flat.end(), static_cast<size_t>(conv_dim) * hs, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(lnh) * hs, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(lnh) * hs, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(lnh) * vd * hs, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(conv_dim) * ck, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(hs) * lnh * vd, 0.0f);
        flat.insert(flat.end(), lnh, 0.0f);
        flat.insert(flat.end(), lnh, 0.0f);
        flat.insert(flat.end(), vd, 1.0f);
    }

    if (is_full) {
        append_weights(flat, static_cast<size_t>(fnh) * qhd * 2 * hs);
        append_weights(flat, static_cast<size_t>(nkh) * khd * hs);
        append_weights(flat, static_cast<size_t>(nkh) * khd * hs);
        flat.insert(flat.end(), khd, 1.0f);
        flat.insert(flat.end(), khd, 1.0f);
        append_weights(flat, static_cast<size_t>(hs) * fnh * khd);
    } else {
        flat.insert(flat.end(), static_cast<size_t>(fnh) * qhd * 2 * hs, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(nkh) * khd * hs, 0.0f);
        flat.insert(flat.end(), static_cast<size_t>(nkh) * khd * hs, 0.0f);
        flat.insert(flat.end(), khd, 1.0f);
        flat.insert(flat.end(), khd, 1.0f);
        flat.insert(flat.end(), static_cast<size_t>(hs) * fnh * khd, 0.0f);
    }

    return flat;
}

static void run_v2_benchmark(const TestConfig& cfg) {
    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  v2.0 Performance Test (All Optimizations Enabled)" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  Prefill tokens:     " << cfg.prefill_tokens << std::endl;
    std::cout << "  Decode tokens:      " << cfg.decode_tokens << std::endl;
    std::cout << "  Batch size:         " << cfg.batch_size << std::endl;
    std::cout << "  Rounds:             " << cfg.num_rounds << std::endl;

    CudaEngine engine(cfg.num_layers, cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size,
                      cfg.max_seq_len);

    std::cout << "\n[Loading random weights...]" << std::endl;
    for (int i = 0; i < cfg.num_layers; ++i) {
        bool is_full = ((i % 4) == 3);
        auto flat = build_flat_weights(is_full, cfg);
        engine.set_layer_weights(i, flat);
    }
    engine.set_final_norm_weight(generate_random_weights(cfg.hidden_size, 0.9f, 1.1f));
    engine.set_lm_head_weight(generate_random_weights(cfg.vocab_size * cfg.hidden_size));

    CudaTokenEmbedding emb(cfg.vocab_size, cfg.hidden_size);
    emb.set_weight(generate_random_weights(cfg.vocab_size * cfg.hidden_size));
    CudaLMHead lmhead(cfg.hidden_size, cfg.vocab_size);
    lmhead.set_weight(generate_random_weights(cfg.vocab_size * cfg.hidden_size));
    GpuGreedyArgmaxSampler sampler(cfg.vocab_size);

    float *d_emb, *d_backbone_out, *d_logits;
    CHECK_CUDA(cudaMalloc(&d_emb, cfg.hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_backbone_out, cfg.hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, static_cast<size_t>(cfg.vocab_size) * sizeof(float)));

    LatencyStats prefill_stats;
    LatencyStats decode_stats;

    std::cout << "\n[Running " << cfg.num_rounds << " round(s)...]" << std::endl;

    for (int round = 0; round < cfg.num_rounds; ++round) {
        engine.reset_cache();
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        CHECK_CUDA(cudaEventRecord(ev_start));

        const int BATCH_SIZE = cfg.batch_size;
        int* d_batch_positions;
        CHECK_CUDA(cudaMalloc(&d_batch_positions, BATCH_SIZE * sizeof(int)));
        float* d_batch_input;
        float* d_batch_output;
        CHECK_CUDA(cudaMalloc(&d_batch_input, BATCH_SIZE * cfg.hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_batch_output, BATCH_SIZE * cfg.hidden_size * sizeof(float)));

        int prefill_pos = 0;
        while (prefill_pos < cfg.prefill_tokens) {
            int current_batch = std::min(BATCH_SIZE, cfg.prefill_tokens - prefill_pos);

            std::vector<int> token_ids(current_batch);
            std::vector<int> positions(current_batch);
            for (int b = 0; b < current_batch; ++b) {
                token_ids[b] = 151644 + ((prefill_pos + b) % 100);
                positions[b] = prefill_pos + b;
            }

            emb.forward(token_ids, d_batch_input);

            CHECK_CUDA(cudaMemcpy(d_batch_positions, positions.data(),
                                  current_batch * sizeof(int), cudaMemcpyHostToDevice));

            engine.forward_batch_prefill(d_batch_input, d_batch_output, positions.data(),
                                         current_batch);

            if (prefill_pos + current_batch >= cfg.prefill_tokens || current_batch < BATCH_SIZE) {
                CHECK_CUDA(cudaMemcpy(d_backbone_out,
                                      d_batch_output + (current_batch - 1) * cfg.hidden_size,
                                      cfg.hidden_size * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            prefill_pos += current_batch;
        }

        lmhead.forward(d_backbone_out, d_logits);
        int next_token = sampler.sample(d_logits);
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float prefill_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&prefill_ms, ev_start, ev_stop));
        prefill_stats.add(prefill_ms);

        cudaFree(d_batch_input);
        cudaFree(d_batch_output);
        cudaFree(d_batch_positions);

        int position = cfg.prefill_tokens;
        for (int step = 0; step < cfg.decode_tokens; ++step) {
            CHECK_CUDA(cudaEventRecord(ev_start));
            emb.forward(next_token, d_emb);
            engine.forward(d_emb, d_backbone_out, position);
            lmhead.forward(d_backbone_out, d_logits);
            next_token = sampler.sample(d_logits);
            position++;
            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));

            float step_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&step_ms, ev_start, ev_stop));
            decode_stats.add(step_ms);
        }

        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        std::cout << "  Round " << (round + 1) << "/" << cfg.num_rounds << " completed" << std::endl;
    }

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  RESULTS" << std::endl;
    std::cout << std::string(72, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- Prefill (" << cfg.prefill_tokens << " tokens) ---" << std::endl;
    std::cout << "  Total time:    " << std::setw(10) << prefill_stats.avg() << " ms" << std::endl;
    std::cout << "  TTFT:          " << std::setw(10) << prefill_stats.avg() << " ms" << std::endl;
    std::cout << "  Throughput:    " << std::setw(10)
              << (cfg.prefill_tokens / (prefill_stats.avg() / 1000.0)) << " tokens/sec" << std::endl;

    std::cout << "\n--- Decode (" << cfg.decode_tokens << " tokens) ---" << std::endl;
    std::cout << "  TPOT:          " << std::setw(10)
              << (decode_stats.avg() / cfg.decode_tokens) << " ms/token" << std::endl;
    std::cout << "  Throughput:    " << std::setw(10)
              << (cfg.decode_tokens / (decode_stats.avg() / 1000.0)) << " tokens/sec" << std::endl;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    std::cout << "\n--- Memory ---" << std::endl;
    std::cout << "  GPU VRAM used: " << std::setw(10) << (used_mem / (1024.0 * 1024.0)) << " MB"
              << std::endl;
    std::cout << "  GPU VRAM total:" << std::setw(10) << (total_mem / (1024.0 * 1024.0)) << " MB"
              << std::endl;

    cudaFree(d_emb);
    cudaFree(d_backbone_out);
    cudaFree(d_logits);
}

int main(int argc, char** argv) {
    TestConfig cfg;

    if (argc > 1) cfg.prefill_tokens = std::atoi(argv[1]);
    if (argc > 2) cfg.decode_tokens = std::atoi(argv[2]);
    if (argc > 3) cfg.num_rounds = std::atoi(argv[3]);
    if (argc > 4) cfg.batch_size = std::atoi(argv[4]);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  Qwen3.5-0.8B v2.0 Performance Test" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "CC:  " << prop.major << "." << prop.minor << std::endl;
    std::cout << "VRAM:" << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    run_v2_benchmark(cfg);

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  Test Complete" << std::endl;
    std::cout << std::string(72, '=') << std::endl;

    return 0;
}
