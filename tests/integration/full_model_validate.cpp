#include "token_embedding.hpp"
#include "language_backbone.hpp"
#include "lm_head.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

std::vector<float> load_binary_file(const std::string& path, size_t expected_size = 0) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = size / sizeof(float);
    if (expected_size > 0 && num_floats != expected_size) {
        std::cerr << "Warning: file size mismatch for " << path << ". Expected " << expected_size
                  << " floats, got " << num_floats << std::endl;
    }

    std::vector<float> data(num_floats);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return data;
}

void print_stats(const std::vector<float>& data, const std::string& name) {
    if (data.empty()) {
        std::cout << name << " is empty!" << std::endl;
        return;
    }

    float min_val = data[0];
    float max_val = data[0];
    float sum = 0.0f;

    for (float v : data) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
    }

    float mean = sum / data.size();

    std::cout << name << ":" << std::endl;
    std::cout << "  Size: " << data.size() << std::endl;
    std::cout << "  Min:  " << std::fixed << std::setprecision(6) << min_val << std::endl;
    std::cout << "  Max:  " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
}

qwen::LanguageLayerWeights load_layer_weights(int layer_idx, const std::string& base_dir) {
    qwen::LanguageLayerWeights weights;
    std::string layer_dir = base_dir + "/layer_" + std::to_string(layer_idx);

    weights.input_layernorm_weight = load_binary_file(layer_dir + "/input_layernorm.bin");
    weights.post_attention_layernorm_weight = load_binary_file(layer_dir + "/post_layernorm.bin");

    weights.mlp_gate_proj_weight = load_binary_file(layer_dir + "/mlp_gate.bin");
    weights.mlp_up_proj_weight = load_binary_file(layer_dir + "/mlp_up.bin");
    weights.mlp_down_proj_weight = load_binary_file(layer_dir + "/mlp_down.bin");

    weights.is_linear = qwen::is_linear_layer(layer_idx);

    if (weights.is_linear) {
        weights.linear_in_proj_qkv_weight = load_binary_file(layer_dir + "/linear_qkv.bin");
        weights.linear_in_proj_a_weight = load_binary_file(layer_dir + "/linear_a.bin");
        weights.linear_in_proj_b_weight = load_binary_file(layer_dir + "/linear_b.bin");
        weights.linear_in_proj_z_weight = load_binary_file(layer_dir + "/linear_z.bin");
        weights.linear_conv1d_weight = load_binary_file(layer_dir + "/linear_conv1d.bin");
        weights.linear_A_log = load_binary_file(layer_dir + "/linear_A_log.bin");
        weights.linear_dt_bias = load_binary_file(layer_dir + "/linear_dt_bias.bin");
        weights.linear_norm_weight = load_binary_file(layer_dir + "/linear_norm.bin");
        weights.linear_out_proj_weight = load_binary_file(layer_dir + "/linear_out.bin");
    } else {
        weights.full_q_proj_weight = load_binary_file(layer_dir + "/full_q.bin");
        weights.full_k_proj_weight = load_binary_file(layer_dir + "/full_k.bin");
        weights.full_v_proj_weight = load_binary_file(layer_dir + "/full_v.bin");
        weights.full_q_norm_weight = load_binary_file(layer_dir + "/full_q_norm.bin");
        weights.full_k_norm_weight = load_binary_file(layer_dir + "/full_k_norm.bin");
        weights.full_o_proj_weight = load_binary_file(layer_dir + "/full_o.bin");
    }

    return weights;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "Full Language Model Validation" << std::endl;
    std::cout << "============================================================" << std::endl;

    const std::string weights_dir = "weights/language_backbone";
    const std::string embed_dir = "weights/language";
    const int hidden_size = 1024;
    const int intermediate_size = 3584;
    const int num_layers = 24;
    const int vocab_size = 248320;

    std::cout << "\n[1] Loading Token Embedding..." << std::endl;
    qwen::TokenEmbedding embedding(vocab_size, hidden_size);
    auto embed_weight = load_binary_file(embed_dir + "/embed_tokens.bin");
    embedding.set_weights(embed_weight);
    std::cout << "  Token embedding loaded: " << embed_weight.size() << " floats" << std::endl;

    std::cout << "\n[2] Creating Language Backbone..." << std::endl;
    qwen::LanguageBackbone backbone(num_layers, hidden_size, intermediate_size);
    std::cout << "  Created " << num_layers << " layers" << std::endl;

    std::cout << "\n[3] Loading layer weights..." << std::endl;
    for (int i = 0; i < num_layers; ++i) {
        auto weights = load_layer_weights(i, weights_dir);
        backbone.set_layer_weights(i, weights);
        std::cout << "  Layer " << i << " loaded ("
                  << (qwen::is_linear_layer(i) ? "Linear" : "Full") << ")" << std::endl;
    }

    std::cout << "\n[4] Loading final norm..." << std::endl;
    auto final_norm = load_binary_file(weights_dir + "/final_norm.bin");
    backbone.set_final_norm_weight(std::move(final_norm));
    std::cout << "  Final norm loaded: " << final_norm.size() << " floats" << std::endl;

    std::cout << "\n[5] Creating LM Head (sharing weights with embedding)..." << std::endl;
    qwen::LMHead lm_head(hidden_size, vocab_size);
    lm_head.set_weight(embed_weight);
    std::cout << "  LM Head initialized with shared embedding weights" << std::endl;

    std::cout << "\n[6] Initializing states..." << std::endl;
    std::vector<qwen::LinearAttnState> linear_states(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        if (qwen::is_linear_layer(i)) {
            linear_states[i].reset(16, 128, 128, 4);
        }
    }

    qwen::KVCache kv_cache;
    kv_cache.reset(num_layers, 2, 256, 4096);
    std::cout << "  States initialized" << std::endl;

    std::cout << "\n[7] Testing full forward pass..." << std::endl;
    std::vector<int> input_tokens = {151644, 8948, 198, 2610, 525};

    std::cout << "  Input tokens: [";
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << input_tokens[i];
    }
    std::cout << "]" << std::endl;

    std::vector<float> hidden = embedding.forward(input_tokens);
    print_stats(hidden, "\n  Embedded hidden");

    std::vector<float> backbone_out =
        backbone.forward_sequence(hidden, input_tokens.size(), linear_states, kv_cache);
    print_stats(backbone_out, "\n  Backbone output");

    std::vector<float> last_hidden(hidden_size);
    std::copy(backbone_out.end() - hidden_size, backbone_out.end(), last_hidden.begin());

    std::vector<float> logits = lm_head.forward(last_hidden);
    print_stats(logits, "\n  Logits");

    auto top_tokens = lm_head.get_top_tokens(logits, 10);
    std::cout << "\n  Top 10 tokens:" << std::endl;
    for (const auto& [token_id, score] : top_tokens) {
        std::cout << "    Token " << std::setw(8) << token_id << ": " << std::fixed
                  << std::setprecision(6) << score << std::endl;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Full Language Model Validation completed!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
