#include "src/backend/cpu/core/attention/language_linear_attn.hpp"
#include "src/backend/cpu/core/attention/language_full_attn.hpp"
#include "src/backend/cpu/core/embedding/token_embedding.hpp"
#include "src/backend/cpu/core/common/language_common.hpp"
#include "src/backend/cpu/core/mlp/language_mlp.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return {};
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

double compare(const std::vector<float>& a, const std::vector<float>& b) {
    double max_diff = 0;
    double sum_diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    return max_diff;
}

int main() {
    try {
        const int hidden_size = 1024;

        auto embed_w = load_binary_file("weights/language/embed_tokens.bin");
        qwen::TokenEmbedding embedding(248320, hidden_size);
        embedding.set_weights(embed_w);
        auto hidden = embedding.forward({151644});

        qwen::LinearAttention linear_attn(hidden_size, 16, 128, 128, 4);
        qwen::FullAttention full_attn(hidden_size, 8, 2, 256, 256, 10000000.0f, 0.25f);
        qwen::MLP mlp(hidden_size, 3584);
        qwen::RMSNorm norm(hidden_size);
        qwen::RMSNorm post_norm(hidden_size);

        printf("\n=== 逐层精度对比 (C++ vs PyTorch) ===\n");
        printf("%-8s %-8s %-12s %-12s %-12s %-8s\n", "Layer", "Type", "MaxDiff", "MeanDiff",
               "RefStd", "Status");
        printf("%-8s %-8s %-12s %-12s %-12s %-8s\n", "-----", "----", "-------", "--------",
               "-------", "------");

        for (int layer = 0; layer < 24; ++layer) {
            std::string prefix = "weights/language_backbone/layer_" + std::to_string(layer);
            bool is_linear = (layer % 4 != 3);

            norm.set_weight(load_binary_file(prefix + "/input_layernorm.bin"));
            post_norm.set_weight(load_binary_file(prefix + "/post_layernorm.bin"));

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

            auto h_normed = norm.forward(hidden);
            std::vector<float> attn_out;

            if (is_linear) {
                qwen::LinearAttnState lin_state;
                lin_state.reset(16, 128, 128, 4);
                attn_out = linear_attn.forward(h_normed, lin_state);
            } else {
                qwen::KVCache kv_cache;
                kv_cache.reset(24, 2, 256, 4096);
                attn_out = full_attn.forward(h_normed, kv_cache, layer, 0);
            }

            auto residual = hidden;
            for (size_t i = 0; i < hidden.size(); ++i)
                residual[i] += attn_out[i];

            auto h_post = post_norm.forward(residual);
            auto mlp_out = mlp.forward(h_post);

            for (size_t i = 0; i < hidden.size(); ++i)
                hidden[i] = residual[i] + mlp_out[i];

            std::string ref_path =
                "validation_data_corrected/hidden_" + std::to_string(layer + 1) + ".bin";
            auto ref = load_binary_file(ref_path);

            if (!ref.empty() && ref.size() == hidden.size()) {
                double max_diff = 0, sum_diff = 0;
                for (size_t i = 0; i < hidden.size(); ++i) {
                    double diff = std::abs(hidden[i] - ref[i]);
                    max_diff = std::max(max_diff, diff);
                    sum_diff += diff;
                }
                double mean_diff = sum_diff / hidden.size();
                double ref_std = 0;
                for (auto x : ref)
                    ref_std += x * x;
                ref_std = std::sqrt(ref_std / ref.size());

                const char* status = (max_diff < 0.05)  ? "PASS"
                                     : (max_diff < 0.1) ? "WARN"
                                                        : "FAIL";
                printf("%-8d %-8s %-12.6f %-12.8f %-12.6f %-8s\n", layer,
                       is_linear ? "Linear" : "Full", max_diff, mean_diff, ref_std, status);
            } else {
                printf("%-8d %-8s (no reference data)\n", layer, is_linear ? "Linear" : "Full");
            }
        }

        auto final_norm_w = load_binary_file("weights/language_backbone/final_norm.bin");
        qwen::RMSNorm final_norm(hidden_size);
        final_norm.set_weight(final_norm_w);
        auto normed = final_norm.forward(hidden);

        auto lm_head_w = load_binary_file("weights/language/embed_tokens.bin");
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

        printf("\n=== 最终结�?===\n");
        printf("Predicted token: %d\n", pred_token);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
