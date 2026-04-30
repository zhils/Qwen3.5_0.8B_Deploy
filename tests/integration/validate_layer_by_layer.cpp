#include "language_linear_attn.hpp"
#include "language_full_attn.hpp"
#include "token_embedding.hpp"
#include "language_common.hpp"
#include "language_mlp.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + path);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

void print_stats(const std::string& name, const std::vector<float>& v) {
    double sum = 0, sum_sq = 0;
    for (auto x : v) {
        sum += x;
        sum_sq += x * x;
    }
    double mean = sum / v.size();
    double std_dev = sqrt(sum_sq / v.size() - mean * mean);
    std::cout << name << ": mean=" << mean << ", std=" << std_dev << ", first=" << v[0]
              << std::endl;
}

int main() {
    try {
        std::cout << "=== Layer-by-Layer Validation ===" << std::endl;

        const int hidden_size = 1024;

        auto embed_w = load_binary_file("weights/language/embed_tokens.bin");
        qwen::TokenEmbedding embedding(248320, hidden_size);
        embedding.set_weights(embed_w);
        auto hidden = embedding.forward({151644});
        print_stats("Embedding", hidden);

        auto norm_w = load_binary_file("weights/language_backbone/layer_0/input_layernorm.bin");
        qwen::RMSNorm norm(hidden_size);
        norm.set_weight(norm_w);

        qwen::LinearAttention linear_attn(hidden_size, 16, 128, 128, 4);
        qwen::FullAttention full_attn(hidden_size, 8, 2, 256, 256, 10000000.0f, 0.25f);
        qwen::MLP mlp(hidden_size, 3584);

        auto post_norm_w = load_binary_file("weights/language_backbone/layer_0/post_layernorm.bin");
        qwen::RMSNorm post_norm(hidden_size);
        post_norm.set_weight(post_norm_w);

        for (int layer = 0; layer < 24; ++layer) {
            std::string prefix = "weights/language_backbone/layer_" + std::to_string(layer);

            norm_w = load_binary_file(prefix + "/input_layernorm.bin");
            norm.set_weight(norm_w);

            post_norm_w = load_binary_file(prefix + "/post_layernorm.bin");
            post_norm.set_weight(post_norm_w);

            bool is_linear = (layer % 4 != 3);

            if (is_linear) {
                linear_attn.set_weights(load_binary_file(prefix + "/linear_qkv.bin"),
                                        load_binary_file(prefix + "/linear_a.bin"),
                                        load_binary_file(prefix + "/linear_b.bin"),
                                        load_binary_file(prefix + "/linear_z.bin"),
                                        load_binary_file(prefix + "/linear_conv1d.bin"),
                                        load_binary_file(prefix + "/linear_A_log.bin"),
                                        load_binary_file(prefix + "/linear_dt_bias.bin"),
                                        load_binary_file(prefix + "/linear_norm.bin"),
                                        load_binary_file(prefix + "/linear_out.bin"));
            } else {
                full_attn.set_weights(load_binary_file(prefix + "/full_q.bin"),
                                      load_binary_file(prefix + "/full_k.bin"),
                                      load_binary_file(prefix + "/full_v.bin"),
                                      load_binary_file(prefix + "/full_q_norm.bin"),
                                      load_binary_file(prefix + "/full_k_norm.bin"),
                                      load_binary_file(prefix + "/full_o.bin"));
            }

            mlp.set_weights(load_binary_file(prefix + "/mlp_gate.bin"),
                            load_binary_file(prefix + "/mlp_up.bin"),
                            load_binary_file(prefix + "/mlp_down.bin"));

            qwen::LinearAttnState lin_state;
            lin_state.reset(16, 128, 128, 4);
            qwen::KVCache kv_cache;
            kv_cache.reset(1, 2, 256, 4096);

            auto h_normed = norm.forward(hidden);
            std::vector<float> attn_out;

            if (is_linear) {
                attn_out = linear_attn.forward(h_normed, lin_state);
            } else {
                attn_out = full_attn.forward(h_normed, kv_cache, layer, 0);
            }

            auto residual = hidden;
            for (size_t i = 0; i < hidden.size(); ++i)
                residual[i] += attn_out[i];

            auto h_post = post_norm.forward(residual);
            auto mlp_out = mlp.forward(h_post);

            for (size_t i = 0; i < hidden.size(); ++i)
                hidden[i] = residual[i] + mlp_out[i];

            if (layer < 5 || layer == 23) {
                std::cout << "\nLayer " << layer << " (" << (is_linear ? "Linear" : "Full")
                          << "):" << std::endl;
                print_stats("  hidden", hidden);
            }
        }

        print_stats("\nFinal hidden", hidden);

        auto final_norm_w = load_binary_file("weights/language_backbone/final_norm.bin");
        qwen::RMSNorm final_norm(hidden_size);
        final_norm.set_weight(final_norm_w);
        auto normed = final_norm.forward(hidden);
        print_stats("After final RMSNorm", normed);

        auto lm_head_w = load_binary_file("weights/lm_head.bin");
        std::vector<float> logits(248320);
        for (int i = 0; i < 248320; ++i) {
            float sum = 0;
            for (int j = 0; j < hidden_size; ++j) {
                sum += lm_head_w[i * hidden_size + j] * normed[j];
            }
            logits[i] = sum;
        }

        int pred_token = 0;
        float max_val = logits[0];
        for (int i = 1; i < 248320; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                pred_token = i;
            }
        }
        std::cout << "\nPredicted token: " << pred_token << " (expected: 10748)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
