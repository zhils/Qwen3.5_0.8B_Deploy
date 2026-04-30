/**
 * Stage 2 Performance Baseline: CPU vs GPU (naive CUDA) full-pipeline benchmark.
 *
 * Measures per-component and end-to-end:
 *   - Latency: prefill (ms), per-token decode (ms), total generation (ms)
 *   - Throughput: tokens/sec (prefill), tokens/sec (decode)
 *   - Memory: peak CPU RSS (bytes), peak GPU VRAM (bytes)
 *
 * All numbers are intended as the "before optimization" baseline for Stage 3.
 */
#include "token_embedding.hpp"
#include "language_backbone.hpp"
#include "lm_head.hpp"
#include "sampler.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

// ============================================================
// Helpers
// ============================================================

static size_t get_peak_rss_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize;
    return 0;
#else
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<size_t>(ru.ru_maxrss) * 1024;
#endif
}

static std::vector<float> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

struct LatencyStats {
    double min_ms = 1e30, max_ms = 0, sum_ms = 0;
    int count = 0;
    std::vector<double> samples;

    void add(double ms) {
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
        count++;
        samples.push_back(ms);
    }

    double avg() const {
        return count > 0 ? sum_ms / count : 0;
    }

    double p50() const {
        return percentile(0.50);
    }
    double p95() const {
        return percentile(0.95);
    }
    double p99() const {
        return percentile(0.99);
    }

    double percentile(double p) const {
        if (samples.empty())
            return 0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        int idx = static_cast<int>(p * (sorted.size() - 1));
        return sorted[idx];
    }
};

static void print_latency(const std::string& label, const LatencyStats& s) {
    std::cout << std::left << std::setw(28) << label << "avg=" << std::fixed << std::setprecision(3)
              << std::setw(10) << s.avg() << " p50=" << std::setw(10) << s.p50()
              << " p95=" << std::setw(10) << s.p95() << " min=" << std::setw(10) << s.min_ms
              << " max=" << std::setw(10) << s.max_ms << " (ms, n=" << s.count << ")" << std::endl;
}

// ============================================================
// CPU end-to-end benchmark
// ============================================================

struct BenchmarkResult {
    double prefill_latency_ms = 0;
    LatencyStats decode_latency;
    double total_time_ms = 0;
    size_t peak_memory_bytes = 0;
    int prompt_tokens = 0;
    int decode_tokens = 0;

    double prefill_throughput() const {
        return prompt_tokens / (prefill_latency_ms / 1000.0);
    }
    double decode_throughput() const {
        return decode_latency.count / (decode_latency.sum_ms / 1000.0);
    }
};

static BenchmarkResult run_cpu_benchmark(const std::string& weights_dir,
                                         const std::vector<int>& prompt_tokens, int num_generate,
                                         int warmup_runs = 0) {
    const int hidden_size = 1024;
    const int intermediate_size = 3584;
    const int vocab_size = 248320;

    qwen::TokenEmbedding embedding(vocab_size, hidden_size);
    embedding.set_weights(load_binary(weights_dir + "/language/embed_tokens.bin"));

    qwen::LanguageBackbone backbone(24, hidden_size, intermediate_size);
    std::string layers_dir = weights_dir + "/language_backbone";
    for (int i = 0; i < 24; ++i) {
        bool is_full = ((i % 4) == 3);
        std::string prefix = layers_dir + "/layer_" + std::to_string(i);
        qwen::LanguageLayerWeights lw;
        lw.input_layernorm_weight = load_binary(prefix + "/input_layernorm.bin");
        lw.is_linear = !is_full;
        if (is_full) {
            lw.full_q_proj_weight = load_binary(prefix + "/full_q.bin");
            lw.full_k_proj_weight = load_binary(prefix + "/full_k.bin");
            lw.full_v_proj_weight = load_binary(prefix + "/full_v.bin");
            lw.full_o_proj_weight = load_binary(prefix + "/full_o.bin");
            lw.full_q_norm_weight = load_binary(prefix + "/full_q_norm.bin");
            lw.full_k_norm_weight = load_binary(prefix + "/full_k_norm.bin");
        } else {
            lw.linear_in_proj_qkv_weight = load_binary(prefix + "/linear_qkv.bin");
            lw.linear_in_proj_a_weight = load_binary(prefix + "/linear_a.bin");
            lw.linear_in_proj_b_weight = load_binary(prefix + "/linear_b.bin");
            lw.linear_in_proj_z_weight = load_binary(prefix + "/linear_z.bin");
            lw.linear_conv1d_weight = load_binary(prefix + "/linear_conv1d.bin");
            lw.linear_A_log = load_binary(prefix + "/linear_A_log.bin");
            lw.linear_dt_bias = load_binary(prefix + "/linear_dt_bias.bin");
            lw.linear_norm_weight = load_binary(prefix + "/linear_norm.bin");
            lw.linear_out_proj_weight = load_binary(prefix + "/linear_out.bin");
        }
        lw.post_attention_layernorm_weight = load_binary(prefix + "/post_layernorm.bin");
        lw.mlp_gate_proj_weight = load_binary(prefix + "/mlp_gate.bin");
        lw.mlp_up_proj_weight = load_binary(prefix + "/mlp_up.bin");
        lw.mlp_down_proj_weight = load_binary(prefix + "/mlp_down.bin");
        backbone.set_layer_weights(i, lw);
    }
    backbone.set_final_norm_weight(load_binary(weights_dir + "/language/final_norm.bin"));

    qwen::LMHead lm_head(hidden_size, vocab_size);
    lm_head.set_weight(load_binary(weights_dir + "/language/embed_tokens.bin"));

    qwen::Sampler sampler(vocab_size, qwen::SamplingStrategy::GREEDY);

    // Warmup
    for (int w = 0; w < warmup_runs; ++w) {
        std::vector<qwen::LinearAttnState> ls(24);
        for (auto& s : ls)
            s.reset(16, 128, 128, 4);
        qwen::KVCache kv;
        kv.reset(24, 2, 256, 4096);
        auto emb = embedding.forward(prompt_tokens);
        auto out = backbone.forward_sequence(emb, static_cast<int>(prompt_tokens.size()), ls, kv);
        (void)out;
    }

    // Actual run
    BenchmarkResult result;
    result.prompt_tokens = static_cast<int>(prompt_tokens.size());

    std::vector<qwen::LinearAttnState> linear_states(24);
    for (auto& s : linear_states)
        s.reset(16, 128, 128, 4);
    qwen::KVCache kv_cache;
    kv_cache.reset(24, 2, 256, 4096);

    auto total_start = std::chrono::high_resolution_clock::now();

    // Prefill
    auto pf_start = std::chrono::high_resolution_clock::now();
    auto prompt_embedded = embedding.forward(prompt_tokens);
    auto prefill_out = backbone.forward_sequence(
        prompt_embedded, static_cast<int>(prompt_tokens.size()), linear_states, kv_cache);
    std::vector<float> last_hidden(hidden_size);
    std::copy(prefill_out.end() - hidden_size, prefill_out.end(), last_hidden.begin());
    auto logits = lm_head.forward(last_hidden);
    int next_token = sampler.sample(logits);
    auto pf_end = std::chrono::high_resolution_clock::now();
    result.prefill_latency_ms =
        std::chrono::duration<double, std::milli>(pf_end - pf_start).count();

    int current_position = static_cast<int>(prompt_tokens.size());

    // Decode
    for (int step = 0; step < num_generate - 1; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        auto token_emb = embedding.get_embedding(next_token);
        auto hidden_out = backbone.forward(token_emb, linear_states, kv_cache, current_position);
        logits = lm_head.forward(hidden_out);
        next_token = sampler.sample(logits);
        current_position++;
        auto step_end = std::chrono::high_resolution_clock::now();
        result.decode_latency.add(
            std::chrono::duration<double, std::milli>(step_end - step_start).count());
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    result.decode_tokens = result.decode_latency.count;
    result.peak_memory_bytes = get_peak_rss_bytes();

    return result;
}

// ============================================================
// Report printer
// ============================================================

static void print_benchmark_report(const std::string& title, const BenchmarkResult& r) {
    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(72, '=') << "\n" << std::endl;

    std::cout << "--- Latency ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Prefill latency:         " << std::setw(10) << r.prefill_latency_ms << " ms  ("
              << r.prompt_tokens << " prompt tokens)" << std::endl;
    print_latency("  Decode per-token:", r.decode_latency);
    std::cout << "  Total generation time:   " << std::setw(10) << r.total_time_ms << " ms  ("
              << (r.decode_tokens + 1) << " tokens generated)" << std::endl;

    std::cout << "\n--- Throughput ---" << std::endl;
    if (r.prefill_latency_ms > 0) {
        std::cout << "  Prefill throughput:      " << std::setw(10) << std::setprecision(1)
                  << r.prefill_throughput() << " tokens/sec" << std::endl;
    }
    if (r.decode_latency.sum_ms > 0) {
        std::cout << "  Decode throughput:       " << std::setw(10) << std::setprecision(1)
                  << r.decode_throughput() << " tokens/sec" << std::endl;
    }

    std::cout << "\n--- Memory ---" << std::endl;
    double mem_mb = r.peak_memory_bytes / (1024.0 * 1024.0);
    std::cout << "  Peak RSS:                " << std::setw(10) << std::setprecision(1) << mem_mb
              << " MB" << std::endl;
}

// ============================================================
// CSV export (for later comparison with Stage 3)
// ============================================================

static void export_csv(const std::string& path, const BenchmarkResult& cpu,
                       const std::string& gpu_info = "") {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "Cannot write CSV: " << path << std::endl;
        return;
    }

    f << "metric,cpu_value,unit,notes\n";
    f << std::fixed << std::setprecision(4);
    f << "prefill_latency," << cpu.prefill_latency_ms << ",ms," << cpu.prompt_tokens
      << " prompt tokens\n";
    f << "decode_avg_latency," << cpu.decode_latency.avg() << ",ms,per token\n";
    f << "decode_p50_latency," << cpu.decode_latency.p50() << ",ms,per token\n";
    f << "decode_p95_latency," << cpu.decode_latency.p95() << ",ms,per token\n";
    f << "decode_p99_latency," << cpu.decode_latency.p99() << ",ms,per token\n";
    f << "decode_min_latency," << cpu.decode_latency.min_ms << ",ms,per token\n";
    f << "decode_max_latency," << cpu.decode_latency.max_ms << ",ms,per token\n";
    f << "total_time," << cpu.total_time_ms << ",ms," << (cpu.decode_tokens + 1) << " tokens\n";
    f << "prefill_throughput," << cpu.prefill_throughput() << ",tokens/sec,\n";
    f << "decode_throughput," << cpu.decode_throughput() << ",tokens/sec,\n";
    f << "peak_rss," << (cpu.peak_memory_bytes / (1024.0 * 1024.0)) << ",MB,\n";
    if (!gpu_info.empty())
        f << "gpu_info," << gpu_info << ",,\n";

    std::cout << "\nBenchmark CSV exported: " << path << std::endl;
}

// ============================================================
// Per-module breakdown (CPU)
// ============================================================

static void run_module_breakdown(const std::string& weights_dir) {
    const int H = 1024, ISZ = 3584, V = 248320;

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  Per-Module Latency Breakdown (CPU, decode single-token)" << std::endl;
    std::cout << std::string(72, '=') << "\n" << std::endl;

    auto embed_w = load_binary(weights_dir + "/language/embed_tokens.bin");
    auto fn_w = load_binary(weights_dir + "/language/final_norm.bin");

    // Token Embedding (single lookup)
    {
        qwen::TokenEmbedding emb(V, H);
        emb.set_weights(embed_w);
        LatencyStats st;
        for (int i = 0; i < 100; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = emb.get_embedding(151644);
            auto t1 = std::chrono::high_resolution_clock::now();
            st.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        print_latency("TokenEmbedding (lookup)", st);
    }

    // RMSNorm
    {
        qwen::RMSNorm norm(H);
        norm.set_weight(fn_w);
        std::vector<float> inp(H, 0.5f);
        LatencyStats st;
        for (int i = 0; i < 1000; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = norm.forward(inp);
            auto t1 = std::chrono::high_resolution_clock::now();
            st.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        print_latency("RMSNorm (1024)", st);
    }

    // MLP
    {
        std::string p = weights_dir + "/language_backbone/layer_0";
        qwen::MLP mlp(H, ISZ);
        mlp.set_weights(load_binary(p + "/mlp_gate.bin"), load_binary(p + "/mlp_up.bin"),
                        load_binary(p + "/mlp_down.bin"));
        std::vector<float> inp(H, 0.1f);
        LatencyStats st;
        for (int i = 0; i < 50; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = mlp.forward(inp);
            auto t1 = std::chrono::high_resolution_clock::now();
            st.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        print_latency("MLP/SwiGLU (1024->3584)", st);
    }

    // LMHead
    {
        qwen::LMHead lmhead(H, V);
        lmhead.set_weight(embed_w);
        std::vector<float> inp(H, 0.1f);
        LatencyStats st;
        for (int i = 0; i < 10; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = lmhead.forward(inp);
            auto t1 = std::chrono::high_resolution_clock::now();
            st.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        print_latency("LMHead (1024->248320)", st);
    }

    // FullAttention (layer 3)
    {
        std::string p = weights_dir + "/language_backbone/layer_3";
        qwen::FullAttention fa(H, 8, 2, 256, 256, 10000000.0f, 0.25f);
        fa.set_weights(load_binary(p + "/full_q.bin"), load_binary(p + "/full_k.bin"),
                       load_binary(p + "/full_v.bin"), load_binary(p + "/full_q_norm.bin"),
                       load_binary(p + "/full_k_norm.bin"), load_binary(p + "/full_o.bin"));
        qwen::KVCache kv;
        kv.reset(1, 2, 256, 128);
        std::vector<float> inp(H, 0.1f);
        LatencyStats st;
        for (int i = 0; i < 50; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = fa.forward(inp, kv, 0, i);
            auto t1 = std::chrono::high_resolution_clock::now();
            st.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        print_latency("FullAttention (decode)", st);
    }

    // LinearAttention (layer 0)
    {
        std::string p = weights_dir + "/language_backbone/layer_0";
        qwen::LinearAttention la(H, 16, 128, 128, 4);
        la.set_weights(load_binary(p + "/linear_qkv.bin"), load_binary(p + "/linear_a.bin"),
                       load_binary(p + "/linear_b.bin"), load_binary(p + "/linear_z.bin"),
                       load_binary(p + "/linear_conv1d.bin"), load_binary(p + "/linear_A_log.bin"),
                       load_binary(p + "/linear_dt_bias.bin"), load_binary(p + "/linear_norm.bin"),
                       load_binary(p + "/linear_out.bin"));
        qwen::LinearAttnState state;
        state.reset(16, 128, 128, 4);
        std::vector<float> inp(H, 0.1f);
        LatencyStats st;
        for (int i = 0; i < 50; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto out = la.forward(inp, state);
            auto t1 = std::chrono::high_resolution_clock::now();
            st.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        print_latency("LinearAttention (decode)", st);
    }
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    std::string weights_dir = "../weights";
    int num_generate = 20;
    bool export_to_csv = true;

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--help" || arg1 == "-h") {
            std::cout << "Usage: stage2_cpu_benchmark.exe [weights_dir] [num_generate]\n"
                      << "\n"
                      << "Arguments:\n"
                      << "  weights_dir   Path to model weights directory (default: ../weights)\n"
                      << "  num_generate  Number of tokens to generate per test (default: 20)\n"
                      << "\n"
                      << "Outputs:\n"
                      << "  - stage2_cpu_single_token.csv\n"
                      << "  - stage2_cpu_multi_token.csv\n"
                      << "\n"
                      << "Example:\n"
                      << "  stage2_cpu_benchmark.exe D:/deploy/c++deploy/weights 5\n";
            return 0;
        }
    }

    if (argc > 1)
        weights_dir = argv[1];
    if (argc > 2)
        num_generate = std::stoi(argv[2]);

    std::cout << std::string(72, '=') << std::endl;
    std::cout << "  Qwen3.5-0.8B Stage 2 Performance Baseline" << std::endl;
    std::cout << "  (CPU-only, full-pipeline, real weights)" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "Weights dir: " << weights_dir << std::endl;
    std::cout << "Generate:    " << num_generate << " tokens" << std::endl;

    try {
        // Test case 1: single-token prompt
        {
            std::vector<int> prompt = {151644};
            auto r = run_cpu_benchmark(weights_dir, prompt, num_generate, 0);
            print_benchmark_report("CPU Baseline: single-token prompt -> " +
                                       std::to_string(num_generate) + " tokens",
                                   r);
            if (export_to_csv)
                export_csv("stage2_cpu_single_token.csv", r);
        }

        // Test case 2: multi-token prompt (5 tokens)
        {
            std::vector<int> prompt = {151644, 8948, 198, 2610, 525};
            auto r = run_cpu_benchmark(weights_dir, prompt, num_generate, 0);
            print_benchmark_report(
                "CPU Baseline: 5-token prompt -> " + std::to_string(num_generate) + " tokens", r);
            if (export_to_csv)
                export_csv("stage2_cpu_multi_token.csv", r);
        }

        // Per-module breakdown with real weights
        run_module_breakdown(weights_dir);

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "  Stage 2 CPU Benchmark Complete" << std::endl;
    std::cout << std::string(72, '=') << std::endl;

    return 0;
}
