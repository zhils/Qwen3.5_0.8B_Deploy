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

std::vector<float> load_binary_file(const std::string& path, size_t expected_size = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (expected_size > 0 && file_size != expected_size * sizeof(float)) {
        file.close();
        throw std::runtime_error("File size mismatch: expected " + std::to_string(expected_size) +
                                 " floats (" + std::to_string(expected_size * sizeof(float)) +
                                 " bytes), got " + std::to_string(file_size / sizeof(float)) +
                                 " floats (" + std::to_string(file_size) + " bytes)");
    }

    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    return data;
}

int main() {
    const int hidden_size = 1024;
    const int intermediate_size = 3584;
    const int vocab_size = 248320;

    const std::string weights_dir = "./weights";
    const std::string embed_path = weights_dir + "/language/embed_tokens.bin";
    const std::string layers_dir = weights_dir + "/language_backbone";
    const std::string final_norm_path = weights_dir + "/language/final_norm.bin";
    const std::string lm_head_path = weights_dir + "/language/embed_tokens.bin";

    auto start_total = std::chrono::high_resolution_clock::now();

    std::cout << "\n============================================================" << std::endl;
    std::cout << "End-to-End Inference Test with Performance Measurement" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    try {
        std::cout << "[1] Loading Token Embedding..." << std::endl;
        qwen::TokenEmbedding embedding(vocab_size, hidden_size);
        auto embed_weight = load_binary_file(embed_path);
        embedding.set_weights(embed_weight);
        std::cout << "  OK Token embedding loaded: " << embed_weight.size() << " floats\n"
                  << std::endl;

        std::cout << "[2] Creating Language Backbone (24 layers)..." << std::endl;
        qwen::LanguageBackbone backbone(24, hidden_size, intermediate_size);
        std::cout << "  OK Created 24 layers\n" << std::endl;

        std::cout << "[3] Loading layer weights..." << std::endl;
        for (int i = 0; i < 24; ++i) {
            bool is_full = ((i % 4) == 3);
            std::string prefix = layers_dir + "/layer_" + std::to_string(i);

            qwen::LanguageLayerWeights lw;
            lw.input_layernorm_weight = load_binary_file(prefix + "/input_layernorm.bin");

            if (is_full) {
                lw.is_linear = false;
                lw.full_q_proj_weight = load_binary_file(prefix + "/full_q.bin");
                lw.full_k_proj_weight = load_binary_file(prefix + "/full_k.bin");
                lw.full_v_proj_weight = load_binary_file(prefix + "/full_v.bin");
                lw.full_o_proj_weight = load_binary_file(prefix + "/full_o.bin");
                lw.full_q_norm_weight = load_binary_file(prefix + "/full_q_norm.bin");
                lw.full_k_norm_weight = load_binary_file(prefix + "/full_k_norm.bin");
            } else {
                lw.is_linear = true;
                lw.linear_in_proj_qkv_weight = load_binary_file(prefix + "/linear_qkv.bin");
                lw.linear_in_proj_a_weight = load_binary_file(prefix + "/linear_a.bin");
                lw.linear_in_proj_b_weight = load_binary_file(prefix + "/linear_b.bin");
                lw.linear_in_proj_z_weight = load_binary_file(prefix + "/linear_z.bin");
                lw.linear_conv1d_weight = load_binary_file(prefix + "/linear_conv1d.bin");
                lw.linear_A_log = load_binary_file(prefix + "/linear_A_log.bin");
                lw.linear_dt_bias = load_binary_file(prefix + "/linear_dt_bias.bin");
                lw.linear_norm_weight = load_binary_file(prefix + "/linear_norm.bin");
                lw.linear_out_proj_weight = load_binary_file(prefix + "/linear_out.bin");
            }

            lw.post_attention_layernorm_weight = load_binary_file(prefix + "/post_layernorm.bin");
            lw.mlp_gate_proj_weight = load_binary_file(prefix + "/mlp_gate.bin");
            lw.mlp_up_proj_weight = load_binary_file(prefix + "/mlp_up.bin");
            lw.mlp_down_proj_weight = load_binary_file(prefix + "/mlp_down.bin");

            backbone.set_layer_weights(i, lw);
            std::cout << "  Layer " << std::setw(2) << i << " loaded ("
                      << (is_full ? "Full" : "Linear") << ")" << std::endl;
        }
        std::cout << "  OK All layers loaded\n" << std::endl;

        std::cout << "[4] Loading Final Norm..." << std::endl;
        backbone.set_final_norm_weight(load_binary_file(final_norm_path));
        std::cout << "  OK Final norm loaded\n" << std::endl;

        std::cout << "[5] Creating LM Head..." << std::endl;
        qwen::LMHead lm_head(hidden_size, vocab_size);
        lm_head.set_weight(load_binary_file(lm_head_path));
        std::cout << "  OK LM Head initialized\n" << std::endl;

        auto end_loading = std::chrono::high_resolution_clock::now();
        double loading_time =
            std::chrono::duration<double, std::milli>(end_loading - start_total).count();

        std::cout << "============================================================" << std::endl;
        std::cout << "Model Loading Time: " << std::fixed << std::setprecision(2) << loading_time
                  << " ms\n"
                  << std::endl;

        struct TestConfig {
            std::string name;
            std::vector<int> input_tokens;
            int num_tokens_to_generate;
            qwen::SamplingStrategy strategy;
            float temperature;
            int top_k;
            float top_p;
        };

        std::vector<TestConfig> tests = {
            {"GREEDY - Short", {151644}, 5, qwen::SamplingStrategy::GREEDY, 1.0f, 50, 0.9f},
            {"Temperature(t=0.7)",
             {151644},
             5,
             qwen::SamplingStrategy::TEMPERATURE,
             0.7f,
             50,
             0.9f},
            {"Top-K(k=10)", {151644}, 5, qwen::SamplingStrategy::TOP_K, 1.0f, 10, 0.9f},
            {"Top-P(p=0.9)", {151644}, 5, qwen::SamplingStrategy::TOP_P, 1.0f, 50, 0.9f},
            {"Long Generation",
             {151644, 8948, 198, 2610, 525},
             10,
             qwen::SamplingStrategy::TEMPERATURE,
             0.8f,
             40,
             0.95f}};

        for (const auto& test : tests) {
            std::cout << "\n============================================================"
                      << std::endl;
            std::cout << "Test: " << test.name << std::endl;
            std::cout << "Input tokens: [";
            for (size_t i = 0; i < test.input_tokens.size(); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << test.input_tokens[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "Generate " << test.num_tokens_to_generate << " tokens" << std::endl;
            std::cout << "============================================================\n"
                      << std::endl;

            qwen::Sampler sampler(vocab_size, test.strategy, test.temperature, test.top_k,
                                  test.top_p);

            std::vector<qwen::LinearAttnState> linear_states(24);
            for (auto& state : linear_states) {
                state.reset(16, 128, 128, 4);
            }

            qwen::KVCache kv_cache;
            kv_cache.reset(24, 2, 256, 4096);

            std::vector<int> generated_tokens = test.input_tokens;

            double prefill_time_ms = 0.0;
            double decode_time_ms = 0.0;
            int total_tokens_generated = 0;

            auto gen_start = std::chrono::high_resolution_clock::now();

            // --- Prefill phase: process the entire prompt at once ---
            auto prefill_start = std::chrono::high_resolution_clock::now();

            std::vector<float> prompt_embedded = embedding.forward(test.input_tokens);
            std::vector<float> prefill_out = backbone.forward_sequence(
                prompt_embedded, static_cast<int>(test.input_tokens.size()), linear_states,
                kv_cache);

            std::vector<float> last_hidden(hidden_size);
            std::copy(prefill_out.end() - hidden_size, prefill_out.end(), last_hidden.begin());

            std::vector<float> logits = lm_head.forward(last_hidden);
            int next_token = sampler.sample(logits);
            generated_tokens.push_back(next_token);

            auto prefill_end = std::chrono::high_resolution_clock::now();
            prefill_time_ms =
                std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
            total_tokens_generated++;

            int current_position = static_cast<int>(test.input_tokens.size());

            std::cout << "[Prefill] " << test.input_tokens.size()
                      << " tokens -> token=" << std::setw(6) << next_token
                      << ", time=" << std::fixed << std::setprecision(2) << std::setw(8)
                      << prefill_time_ms << " ms" << std::endl;

            // --- Decode phase: process one new token at a time ---
            for (int step = 1; step < test.num_tokens_to_generate; ++step) {
                auto step_start = std::chrono::high_resolution_clock::now();

                std::vector<float> token_embedded = embedding.get_embedding(next_token);

                std::vector<float> hidden_out =
                    backbone.forward(token_embedded, linear_states, kv_cache, current_position);

                logits = lm_head.forward(hidden_out);
                next_token = sampler.sample(logits);

                generated_tokens.push_back(next_token);
                current_position++;

                auto step_end = std::chrono::high_resolution_clock::now();
                double step_time =
                    std::chrono::duration<double, std::milli>(step_end - step_start).count();

                decode_time_ms += step_time;
                total_tokens_generated++;

                std::cout << "Step " << std::setw(3) << step << "/"
                          << (test.num_tokens_to_generate - 1) << ": token=" << std::setw(6)
                          << next_token << ", time=" << std::fixed << std::setprecision(2)
                          << std::setw(8) << step_time << " ms" << std::endl;
            }

            auto gen_end = std::chrono::high_resolution_clock::now();
            double total_gen_time =
                std::chrono::duration<double, std::milli>(gen_end - gen_start).count();

            int decode_tokens = total_tokens_generated - 1;
            double avg_decode_time = (decode_tokens > 0) ? (decode_time_ms / decode_tokens) : 0.0;
            double decode_throughput =
                (decode_time_ms > 0) ? (decode_tokens / (decode_time_ms / 1000.0)) : 0.0;

            std::cout << "\n--- Results ---" << std::endl;
            std::cout << "Generated sequence length: " << generated_tokens.size() << " tokens"
                      << std::endl;
            std::cout << "Full token sequence:" << std::endl;
            std::cout << "[";
            for (size_t i = 0; i < generated_tokens.size(); ++i) {
                if (i > 0 && i % 15 == 0)
                    std::cout << "\n ";
                if (i > 0)
                    std::cout << ", ";
                std::cout << generated_tokens[i];
            }
            std::cout << "]\n" << std::endl;

            std::cout << "--- Performance ---" << std::endl;
            std::cout << "Total generation time:   " << std::fixed << std::setprecision(2)
                      << total_gen_time << " ms" << std::endl;
            std::cout << "Prefill time:            " << prefill_time_ms << " ms ("
                      << test.input_tokens.size() << " prompt tokens)" << std::endl;
            std::cout << "Decode time:             " << decode_time_ms << " ms (" << decode_tokens
                      << " tokens)" << std::endl;
            std::cout << "Avg decode time/token:   " << avg_decode_time << " ms/token" << std::endl;
            std::cout << "Decode throughput:       " << std::fixed << std::setprecision(1)
                      << decode_throughput << " tokens/sec" << std::endl;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        double total_program_time =
            std::chrono::duration<double, std::milli>(end_total - start_total).count();

        std::cout << "\n============================================================" << std::endl;
        std::cout << "Total Program Execution Time: " << std::fixed << std::setprecision(2)
                  << total_program_time << " ms (" << (total_program_time / 1000.0) << " sec)"
                  << std::endl;
        std::cout << "============================================================\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
