#include "language_linear_attn.hpp"
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
    std::cout << "=== LinearAttention Minimal Test ===" << std::endl;

    const int hidden_size = 1024;
    const int num_heads = 16;
    const int key_dim = 128;
    const int value_dim = 128;
    const int conv_kernel = 4;

    std::cout << "[1] Creating LinearAttention..." << std::endl;
    qwen::LinearAttention attn(hidden_size, num_heads, key_dim, value_dim, conv_kernel);
    std::cout << "  OK" << std::endl;

    std::cout << "[2] Loading weights from layer_0..." << std::endl;
    std::string layer_dir = "weights/language_backbone/layer_0";

    auto in_proj_qkv = load_binary_file(layer_dir + "/linear_qkv.bin");
    auto in_proj_a = load_binary_file(layer_dir + "/linear_a.bin");
    auto in_proj_b = load_binary_file(layer_dir + "/linear_b.bin");
    auto in_proj_z = load_binary_file(layer_dir + "/linear_z.bin");
    auto conv1d = load_binary_file(layer_dir + "/linear_conv1d.bin");
    auto A_log = load_binary_file(layer_dir + "/linear_A_log.bin");
    auto dt_bias = load_binary_file(layer_dir + "/linear_dt_bias.bin");
    auto norm_w = load_binary_file(layer_dir + "/linear_norm.bin");
    auto out_proj = load_binary_file(layer_dir + "/linear_out.bin");

    std::cout << "  Weights loaded:" << std::endl;
    std::cout << "    in_proj_qkv: " << in_proj_qkv.size() << std::endl;
    std::cout << "    in_proj_a: " << in_proj_a.size() << std::endl;
    std::cout << "    in_proj_b: " << in_proj_b.size() << std::endl;
    std::cout << "    in_proj_z: " << in_proj_z.size() << std::endl;
    std::cout << "    conv1d: " << conv1d.size() << std::endl;
    std::cout << "    A_log: " << A_log.size() << std::endl;
    std::cout << "    dt_bias: " << dt_bias.size() << std::endl;
    std::cout << "    norm_w: " << norm_w.size() << std::endl;
    std::cout << "    out_proj: " << out_proj.size() << std::endl;

    attn.set_weights(in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, conv1d, A_log, dt_bias, norm_w,
                     out_proj);
    std::cout << "  OK weights set" << std::endl;

    std::cout << "[3] Creating input..." << std::endl;
    std::vector<float> input(hidden_size, 0.1f);
    std::cout << "  Input size: " << input.size() << std::endl;

    std::cout << "[4] Creating state with correct dimensions (16, 128, 128)..." << std::endl;
    qwen::LinearAttnState state;
    state.reset(16, 128, 128, 4);
    std::cout << "  State reset: heads=" << state.num_heads << ", key_dim=" << state.key_dim
              << ", value_dim=" << state.value_dim << std::endl;

    std::cout << "[5] Calling forward..." << std::endl;
    auto output = attn.forward(input, state);
    std::cout << "  Output size: " << output.size() << std::endl;

    std::cout << "\n=== SUCCESS ===" << std::endl;
    return 0;
}
