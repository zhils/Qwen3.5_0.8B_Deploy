#include "language_common.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

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
    auto embed = load_binary_file("validation_data_detailed/embed.bin");
    auto weight = load_binary_file("validation_data_detailed/layer0_ln_weight.bin");
    auto py_out = load_binary_file("validation_data_detailed/layer0_after_input_norm.bin");

    std::cout << "Embed[0:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << embed[i] << " ";
    std::cout << std::endl;

    std::cout << "Weight[0:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << weight[i] << " ";
    std::cout << std::endl;

    float sum_sq = 0;
    for (auto x : embed)
        sum_sq += x * x;
    float rms = std::sqrt(sum_sq / embed.size() + 1e-6f);
    std::cout << "\nRMS: " << rms << std::endl;

    std::vector<float> manual_out(embed.size());
    for (size_t i = 0; i < embed.size(); ++i) {
        manual_out[i] = (embed[i] / rms) * (1.0f + weight[i]);
    }

    std::cout << "\nManual out[0:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << manual_out[i] << " ";
    std::cout << std::endl;

    std::cout << "PyTorch out[0:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << py_out[i] << " ";
    std::cout << std::endl;

    qwen::RMSNorm ln(1024);
    ln.set_weight(weight);
    auto cpp_out = ln.forward(embed);

    std::cout << "\nC++ RMSNorm out[0:5]: ";
    for (int i = 0; i < 5; ++i)
        std::cout << cpp_out[i] << " ";
    std::cout << std::endl;

    double max_diff = 0;
    for (size_t i = 0; i < cpp_out.size(); ++i) {
        double diff = std::abs(cpp_out[i] - py_out[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    std::cout << "\nMax diff: " << max_diff << std::endl;

    return 0;
}
