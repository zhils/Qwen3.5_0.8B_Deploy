#include "src/backend/cpu/core/embedding/token_embedding.hpp"
#include "src/backend/cpu/core/common/language_backbone.hpp"
#include "src/backend/cpu/core/heads/lm_head.hpp"
#include "src/backend/cpu/core/heads/sampler.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
    return data;
}

int main() {
    std::cout << "=== Language Backbone Test (1 layer) ===" << std::endl;

    const int hidden_size = 1024;
    const int vocab_size = 248320;

    std::cout << "[1] Loading token embedding..." << std::endl;
    qwen::TokenEmbedding embedding(vocab_size, hidden_size);
    auto embed_weight = load_binary_file("weights/language/embed_tokens.bin");
    embedding.set_weights(embed_weight);
    std::cout << "  OK: " << embed_weight.size() << " floats" << std::endl;

    std::cout << "[2] Creating Language Backbone with 1 layer..." << std::endl;
    qwen::LanguageBackbone backbone(1, hidden_size, 3584);

    std::cout << "[3] Loading layer 0 weights..." << std::endl;
    std::string layer_dir = "weights/language_backbone/layer_0";
    qwen::LanguageLayerWeights lw;
    lw.input_layernorm_weight = load_binary_file(layer_dir + "/input_layernorm.bin");
    lw.post_attention_layernorm_weight = load_binary_file(layer_dir + "/post_layernorm.bin");
    lw.mlp_gate_proj_weight = load_binary_file(layer_dir + "/mlp_gate.bin");
    lw.mlp_up_proj_weight = load_binary_file(layer_dir + "/mlp_up.bin");
    lw.mlp_down_proj_weight = load_binary_file(layer_dir + "/mlp_down.bin");

    lw.is_linear = true;
    lw.linear_in_proj_qkv_weight = load_binary_file(layer_dir + "/linear_qkv.bin");
    lw.linear_in_proj_a_weight = load_binary_file(layer_dir + "/linear_a.bin");
    lw.linear_in_proj_b_weight = load_binary_file(layer_dir + "/linear_b.bin");
    lw.linear_in_proj_z_weight = load_binary_file(layer_dir + "/linear_z.bin");
    lw.linear_conv1d_weight = load_binary_file(layer_dir + "/linear_conv1d.bin");
    lw.linear_A_log = load_binary_file(layer_dir + "/linear_A_log.bin");
    lw.linear_dt_bias = load_binary_file(layer_dir + "/linear_dt_bias.bin");
    lw.linear_norm_weight = load_binary_file(layer_dir + "/linear_norm.bin");
    lw.linear_out_proj_weight = load_binary_file(layer_dir + "/linear_out.bin");

    backbone.set_layer_weights(0, lw);
    backbone.set_final_norm_weight(load_binary_file("weights/language/final_norm.bin"));
    std::cout << "  OK layer loaded" << std::endl;

    std::cout << "[4] Creating LM Head and Sampler..." << std::endl;
    qwen::LMHead lm_head(hidden_size, vocab_size);
    lm_head.set_weight(embed_weight);
    qwen::Sampler sampler(vocab_size, qwen::SamplingStrategy::GREEDY, 1.0f, 50, 0.9f);
    std::cout << "  OK" << std::endl;

    std::cout << "[5] Running inference..." << std::endl;
    std::vector<int> tokens = {151644};
    std::vector<float> embedded = embedding.forward(tokens);
    std::cout << "  Embedded size: " << embedded.size() << std::endl;

    std::vector<qwen::LinearAttnState> linear_states(1);
    linear_states[0].reset(16, 128, 128, 4);
    qwen::KVCache kv_cache;
    kv_cache.reset(1, 2, 256, 4096);

    std::cout << "  Calling backbone.forward_sequence..." << std::endl;
    auto backbone_out = backbone.forward_sequence(embedded, 1, linear_states, kv_cache);
    std::cout << "  Backbone output size: " << backbone_out.size() << std::endl;

    std::vector<float> last_hidden(hidden_size);
    std::copy(backbone_out.end() - hidden_size, backbone_out.end(), last_hidden.begin());

    std::vector<float> logits = lm_head.forward(last_hidden);
    std::cout << "  Logits size: " << logits.size() << std::endl;

    int sampled = sampler.sample(logits);
    std::cout << "  Sampled token: " << sampled << std::endl;

    std::cout << "\n=== SUCCESS: Language backbone works! ===" << std::endl;
    return 0;
}
