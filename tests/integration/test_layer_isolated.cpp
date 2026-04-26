#include "language_common.hpp"
#include "language_linear_attn.hpp"
#include "language_full_attn.hpp"
#include "language_backbone.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <sstream>

std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

double compare(const std::vector<float>& a, const std::vector<float>& b) {
    double max_diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

std::string make_path(const std::string& dir, const std::string& name, int idx) {
    std::ostringstream oss;
    oss << dir << "/" << name << idx << ".bin";
    return oss.str();
}

int main() {
    const int H = 1024, NL = 24;
    std::string val_dir = "../data/validation_data_corrected";
    std::string weight_dir = "../weights/language_backbone";

    qwen::LanguageBackbone backbone(NL, H, 3584);

    for (int i = 0; i < NL; ++i) {
        std::string prefix = weight_dir + "/layer_" + std::to_string(i);
        qwen::LanguageLayerWeights lw;
        lw.input_layernorm_weight = load_binary_file(prefix + "/input_layernorm.bin");
        lw.post_attention_layernorm_weight = load_binary_file(prefix + "/post_layernorm.bin");
        lw.mlp_gate_proj_weight = load_binary_file(prefix + "/mlp_gate.bin");
        lw.mlp_up_proj_weight = load_binary_file(prefix + "/mlp_up.bin");
        lw.mlp_down_proj_weight = load_binary_file(prefix + "/mlp_down.bin");

        bool is_linear = (i % 4) != 3;
        lw.is_linear = is_linear;

        if (is_linear) {
            lw.linear_in_proj_qkv_weight = load_binary_file(prefix + "/linear_qkv.bin");
            lw.linear_in_proj_a_weight = load_binary_file(prefix + "/linear_a.bin");
            lw.linear_in_proj_b_weight = load_binary_file(prefix + "/linear_b.bin");
            lw.linear_in_proj_z_weight = load_binary_file(prefix + "/linear_z.bin");
            lw.linear_conv1d_weight = load_binary_file(prefix + "/linear_conv1d.bin");
            lw.linear_A_log = load_binary_file(prefix + "/linear_A_log.bin");
            lw.linear_dt_bias = load_binary_file(prefix + "/linear_dt_bias.bin");
            lw.linear_norm_weight = load_binary_file(prefix + "/linear_norm.bin");
            lw.linear_out_proj_weight = load_binary_file(prefix + "/linear_out.bin");
        } else {
            lw.full_q_proj_weight = load_binary_file(prefix + "/full_q.bin");
            lw.full_k_proj_weight = load_binary_file(prefix + "/full_k.bin");
            lw.full_v_proj_weight = load_binary_file(prefix + "/full_v.bin");
            lw.full_o_proj_weight = load_binary_file(prefix + "/full_o.bin");
            lw.full_q_norm_weight = load_binary_file(prefix + "/full_q_norm.bin");
            lw.full_k_norm_weight = load_binary_file(prefix + "/full_k_norm.bin");
        }

        backbone.set_layer_weights(i, lw);
    }

    std::vector<qwen::LinearAttnState> linear_states(NL);
    for (auto& s : linear_states)
        s.reset(16, 128, 128, 4);
    qwen::KVCache kv_cache;
    kv_cache.reset(NL, 2, 256, 4096);

    std::cout << "=== Layer-by-layer validation with PyTorch input ===" << std::endl;

    for (int i = 0; i < NL; ++i) {
        kv_cache.clear();

        std::string input_path = make_path(val_dir, "hidden_", i);
        std::string output_path = make_path(val_dir, "hidden_", i + 1);
        auto py_input = load_binary_file(input_path);
        auto py_output = load_binary_file(output_path);

        auto cpp_output = backbone.layers()[i]->forward(py_input, linear_states[i], kv_cache, i);

        double diff = compare(cpp_output, py_output);
        std::string type = ((i % 4) == 3) ? "Full" : "Linear";
        std::cout << "Layer " << i << " (" << type << "): diff=" << diff << std::endl;
    }

    return 0;
}
