#include "language_full_attn.hpp"
#include <iostream>
#include <vector>
#include <fstream>

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

int main() {
    try {
        std::cout << "=== KVCache Test ===" << std::endl;

        qwen::KVCache kv_cache;
        kv_cache.reset(1, 2, 256, 4096);
        std::cout << "[1] KVCache reset OK" << std::endl;

        std::vector<float> k(512, 0.01f); // 2 * 256
        std::vector<float> v(512, 0.01f);
        std::cout << "[2] k, v vectors created" << std::endl;

        kv_cache.append(0, k.data(), v.data());
        std::cout << "[3] append OK, layer0_len=" << kv_cache.length(0) << std::endl;

        auto* k_ptr = kv_cache.get_k(0);
        auto* v_ptr = kv_cache.get_v(0);
        std::cout << "[4] get_k/get_v OK" << std::endl;
        std::cout << "    k[0]=" << k_ptr[0] << ", v[0]=" << v_ptr[0] << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
