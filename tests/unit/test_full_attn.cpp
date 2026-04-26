#include "language_full_attn.hpp"
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
    std::cout << "=== FullAttention Minimal Test ===" << std::endl;

    const int hidden_size = 1024;
    const int num_heads = 8;
    const int num_kv_heads = 2;
    const int q_head_dim = 512;
    const int kv_head_dim = 256;

    std::cout << "[1] Creating FullAttention with correct params..." << std::endl;
    qwen::FullAttention attn(hidden_size, num_heads, num_kv_heads, q_head_dim, kv_head_dim,
                             10000000.0f, 0.25f);
    std::cout << "  OK" << std::endl;

    std::cout << "[2] Loading weights from layer_3..." << std::endl;
    std::string layer_dir = "weights/language_backbone/layer_3";

    auto q_proj = load_binary_file(layer_dir + "/full_q.bin");
    auto k_proj = load_binary_file(layer_dir + "/full_k.bin");
    auto v_proj = load_binary_file(layer_dir + "/full_v.bin");
    auto q_norm = load_binary_file(layer_dir + "/full_q_norm.bin");
    auto k_norm = load_binary_file(layer_dir + "/full_k_norm.bin");
    auto o_proj = load_binary_file(layer_dir + "/full_o.bin");

    std::cout << "  Weights loaded:" << std::endl;
    std::cout << "    q_proj: " << q_proj.size() << std::endl;
    std::cout << "    k_proj: " << k_proj.size() << std::endl;
    std::cout << "    v_proj: " << v_proj.size() << std::endl;
    std::cout << "    q_norm: " << q_norm.size() << std::endl;
    std::cout << "    k_norm: " << k_norm.size() << std::endl;
    std::cout << "    o_proj: " << o_proj.size() << std::endl;

    std::cout << "[3] Setting weights..." << std::endl;
    attn.set_weights(q_proj, k_proj, v_proj, q_norm, k_norm, o_proj);
    std::cout << "  OK weights set" << std::endl;

    std::cout << "[4] Creating input..." << std::endl;
    std::vector<float> input(hidden_size, 0.1f);
    std::cout << "  Input size: " << input.size() << std::endl;

    std::cout << "[5] Creating KVCache..." << std::endl;
    qwen::KVCache kv_cache;
    kv_cache.reset(1, 2, 256, 4096);
    std::cout << "  KVCache reset" << std::endl;

    std::cout << "[6] Calling forward..." << std::endl;
    auto output = attn.forward(input, kv_cache, 3, 0);
    std::cout << "  Output size: " << output.size() << std::endl;

    std::cout << "\n=== SUCCESS ===" << std::endl;
    return 0;
}
