#include "token_embedding.hpp"
#include "language_backbone.hpp"
#include "lm_head.hpp"
#include "sampler.hpp"
#include "mtp_head.hpp"
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

qwen::MTPWeights load_mtp_weights(const std::string& mtp_dir) {
    qwen::MTPWeights weights;

    weights.pre_fc_norm_hidden_weight = load_binary_file(mtp_dir + "/pre_fc_norm_hidden.bin");
    weights.pre_fc_norm_embedding_weight = load_binary_file(mtp_dir + "/pre_fc_norm_embedding.bin");

    weights.layer_input_layernorm_weight = load_binary_file(mtp_dir + "/layer_input_layernorm.bin");
    weights.layer_post_attention_layernorm_weight =
        load_binary_file(mtp_dir + "/layer_post_attention_layernorm.bin");

    weights.attn_q_proj_weight = load_binary_file(mtp_dir + "/attn_q.bin");
    weights.attn_k_proj_weight = load_binary_file(mtp_dir + "/attn_k.bin");
    weights.attn_v_proj_weight = load_binary_file(mtp_dir + "/attn_v.bin");
    weights.attn_q_norm_weight = load_binary_file(mtp_dir + "/attn_q_norm.bin");
    weights.attn_k_norm_weight = load_binary_file(mtp_dir + "/attn_k_norm.bin");
    weights.attn_o_proj_weight = load_binary_file(mtp_dir + "/attn_o.bin");

    weights.mlp_gate_proj_weight = load_binary_file(mtp_dir + "/mlp_gate.bin");
    weights.mlp_up_proj_weight = load_binary_file(mtp_dir + "/mlp_up.bin");
    weights.mlp_down_proj_weight = load_binary_file(mtp_dir + "/mlp_down.bin");

    weights.norm_weight = load_binary_file(mtp_dir + "/mtp_norm.bin");
    weights.fc_weight = load_binary_file(mtp_dir + "/mtp_fc.bin");

    return weights;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "Complete Inference Pipeline Validation" << std::endl;
    std::cout << "============================================================" << std::endl;

    const std::string weights_dir = "weights/language_backbone";
    const std::string embed_dir = "weights/language";
    const std::string mtp_dir = "weights/mtp";
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
    std::cout << "  Final norm loaded" << std::endl;

    std::cout << "\n[5] Creating LM Head..." << std::endl;
    qwen::LMHead lm_head(hidden_size, vocab_size);
    lm_head.set_weight(embed_weight);
    std::cout << "  LM Head initialized with shared weights" << std::endl;

    std::cout << "\n[6] Creating Sampler..." << std::endl;
    qwen::Sampler sampler(vocab_size, qwen::SamplingStrategy::GREEDY, 1.0f, 50, 0.9f, 42);
    std::cout << "  Sampler created (GREEDY mode)" << std::endl;

    std::cout << "\n[7] Creating MTP Head..." << std::endl;
    qwen::MTPHead mtp_head(hidden_size, intermediate_size, 2048);
    try {
        auto mtp_weights = load_mtp_weights(mtp_dir);
        mtp_head.set_weights(mtp_weights);
        std::cout << "  MTP Head loaded and initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  MTP Head not loaded (optional): " << e.what() << std::endl;
    }

    std::cout << "\n[8] Initializing states..." << std::endl;
    std::vector<qwen::LinearAttnState> linear_states(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        if (qwen::is_linear_layer(i)) {
            linear_states[i].reset(16, 128, 128, 4);
        }
    }

    qwen::KVCache kv_cache;
    kv_cache.reset(num_layers, 2, 256, 4096);
    std::cout << "  States initialized" << std::endl;

    std::cout << "\n[9] Testing complete inference pipeline..." << std::endl;

    std::vector<int> input_tokens = {151644, 8948, 198, 2610, 525};

    std::cout << "  Input tokens: [";
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << input_tokens[i];
    }
    std::cout << "]" << std::endl;

    std::vector<float> embedded = embedding.forward(input_tokens);
    print_stats(embedded, "\n  Embedded input");

    std::vector<float> backbone_out =
        backbone.forward_sequence(embedded, input_tokens.size(), linear_states, kv_cache);
    print_stats(backbone_out, "\n  Backbone output");

    std::vector<float> last_hidden(hidden_size);
    std::copy(backbone_out.end() - hidden_size, backbone_out.end(), last_hidden.begin());

    std::vector<float> logits = lm_head.forward(last_hidden);
    print_stats(logits, "\n  Main Logits");

    int generated_token = sampler.sample(logits);
    std::cout << "\n  Generated token (GREEDY): " << generated_token << std::endl;

    auto top_tokens = lm_head.get_top_tokens(logits, 5);
    std::cout << "\n  Top 5 predictions:" << std::endl;
    for (const auto& [token_id, score] : top_tokens) {
        std::cout << "    Token " << std::setw(8) << token_id << ": score=" << std::fixed
                  << std::setprecision(6) << score << std::endl;
    }

    std::cout << "\n[10] Testing different sampling strategies..." << std::endl;

    sampler.set_strategy(qwen::SamplingStrategy::TEMPERATURE);
    sampler.set_temperature(0.8f);
    int temp_token = sampler.sample(logits);
    std::cout << "  Temperature sampling (t=0.8): token=" << temp_token << std::endl;

    sampler.set_strategy(qwen::SamplingStrategy::TOP_K);
    sampler.set_top_k(10);
    int topk_token = sampler.sample(logits);
    std::cout << "  Top-K sampling (k=10): token=" << topk_token << std::endl;

    sampler.set_strategy(qwen::SamplingStrategy::TOP_P);
    sampler.set_top_p(0.95f);
    int topp_token = sampler.sample(logits);
    std::cout << "  Top-P sampling (p=0.95): token=" << topp_token << std::endl;

    std::cout << "\n[11] Testing MTP branch (if available)..." << std::endl;
    try {
        qwen::KVCache mtp_kv_cache;
        mtp_kv_cache.reset(1, 2, 256, 4096);

        std::vector<float> mtp_output = mtp_head.forward(last_hidden, last_hidden, mtp_kv_cache,
                                                         static_cast<int>(input_tokens.size()) - 1);
        print_stats(mtp_output, "\n  MTP Output");

        std::cout << "\n  MTP output size: " << mtp_output.size() << " dims" << std::endl;
        std::cout << "\n  MTP output (first 8): [";
        for (int i = 0; i < 8 && i < static_cast<int>(mtp_output.size()); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << std::fixed << std::setprecision(6) << mtp_output[i];
        }
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  MTP test skipped: " << e.what() << std::endl;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Complete Inference Pipeline Validation completed!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
