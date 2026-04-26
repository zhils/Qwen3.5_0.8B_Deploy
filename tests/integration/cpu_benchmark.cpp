#include "language_common.hpp"
#include "language_mlp.hpp"
#include "language_linear_attn.hpp"
#include "language_full_attn.hpp"
#include "language_backbone.hpp"
#include "lm_head.hpp"
#include "token_embedding.hpp"
#include "sampler.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

template <typename Func>
double benchmark(const std::string& name, int warmup, int iterations, Func fn) {
    for (int i = 0; i < warmup; ++i)
        fn();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
        fn();
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;

    std::cout << std::left << std::setw(30) << name << "avg=" << std::fixed << std::setprecision(4)
              << std::setw(10) << avg_ms << " ms"
              << "  (" << iterations << " iters)" << std::endl;

    return avg_ms;
}

int main() {
    const int hidden_size = 1024;
    const int intermediate_size = 3584;
    const int vocab_size = 248320;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    auto random_vec = [&](int size) {
        std::vector<float> v(size);
        for (auto& x : v)
            x = dist(gen);
        return v;
    };

    std::cout << "=================================================\n";
    std::cout << "    Qwen3.5-0.8B CPU Performance Benchmark\n";
    std::cout << "=================================================\n\n";

    // RMSNorm
    {
        qwen::RMSNorm norm(hidden_size);
        norm.set_weight(random_vec(hidden_size));
        auto input = random_vec(hidden_size);
        benchmark("RMSNorm (1024)", 10, 1000, [&]() { norm.forward(input); });
    }

    // MLP
    {
        qwen::MLP mlp(hidden_size, intermediate_size);
        mlp.set_weights(random_vec(intermediate_size * hidden_size),
                        random_vec(intermediate_size * hidden_size),
                        random_vec(hidden_size * intermediate_size));
        auto input = random_vec(hidden_size);
        benchmark("MLP/SwiGLU (1024->3584)", 3, 50, [&]() { mlp.forward(input); });
    }

    // LMHead
    {
        qwen::LMHead lmhead(hidden_size, vocab_size);
        lmhead.set_weight(random_vec(static_cast<size_t>(vocab_size) * hidden_size));
        auto input = random_vec(hidden_size);
        benchmark("LMHead (1024->248320)", 1, 10, [&]() { lmhead.forward(input); });
    }

    // FullAttention single token
    {
        qwen::FullAttention fa(hidden_size, 8, 2, 256, 256, 10000000.0f, 0.25f);
        fa.set_weights(random_vec(8 * 256 * 2 * hidden_size), random_vec(2 * 256 * hidden_size),
                       random_vec(2 * 256 * hidden_size), random_vec(256), random_vec(256),
                       random_vec(hidden_size * 8 * 256));
        qwen::KVCache kv;
        kv.reset(1, 2, 256, 128);
        auto input = random_vec(hidden_size);
        int pos = 0;
        benchmark("FullAttention (decode)", 3, 50, [&]() { fa.forward(input, kv, 0, pos++); });
    }

    // LinearAttention single token
    {
        int num_heads = 16, key_dim = 128, value_dim = 128, conv_kernel = 4;
        int conv_dim = num_heads * (key_dim * 2 + value_dim);
        qwen::LinearAttention la(hidden_size, num_heads, key_dim, value_dim, conv_kernel);
        la.set_weights(
            random_vec(conv_dim * hidden_size), random_vec(num_heads * hidden_size),
            random_vec(num_heads * hidden_size), random_vec(num_heads * value_dim * hidden_size),
            random_vec(conv_dim * conv_kernel), random_vec(num_heads), random_vec(num_heads),
            random_vec(value_dim), random_vec(hidden_size * num_heads * value_dim));
        qwen::LinearAttnState state;
        state.reset(num_heads, key_dim, value_dim, conv_kernel);
        auto input = random_vec(hidden_size);
        benchmark("LinearAttention (decode)", 3, 50, [&]() { la.forward(input, state); });
    }

    std::cout << "\n=================================================\n";
    std::cout << "CPU Benchmark Complete\n";
    std::cout << "=================================================\n";

    return 0;
}
