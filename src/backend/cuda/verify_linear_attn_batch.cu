#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "linear_attention_cuda.hpp"

using namespace qwen::cuda;

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":"   \
                      << __LINE__ << std::endl;                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

int main() {
    const int hidden_size = 1024;
    const int num_heads = 16;
    const int key_dim = 128;
    const int value_dim = 128;
    const int conv_kernel = 4;
    const int batch_size = 4;

    std::cout << "=== Linear Attention Batch Verification ===" << std::endl;

    CudaLinearAttention attn(hidden_size, num_heads, key_dim, value_dim, conv_kernel);

    // Generate random weights
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    int k_dim = num_heads * key_dim;
    int v_dim = num_heads * value_dim;
    int conv_dim = k_dim * 2 + v_dim;
    int z_dim = num_heads * value_dim;

    auto rand_vec = [&](int size) {
        std::vector<float> v(size);
        for (auto& x : v) x = dist(gen);
        return v;
    };

    attn.set_weights(rand_vec(conv_dim * hidden_size),
                     rand_vec(num_heads * hidden_size),
                     rand_vec(num_heads * hidden_size),
                     rand_vec(z_dim * hidden_size),
                     rand_vec(conv_dim * conv_kernel),
                     rand_vec(hidden_size * z_dim),
                     rand_vec(num_heads),
                     rand_vec(num_heads),
                     rand_vec(value_dim));

    // Generate random input
    std::vector<float> h_input(batch_size * hidden_size);
    for (auto& x : h_input) x = dist(gen);

    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), batch_size * hidden_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Test 1: Serial forward (token by token)
    CudaLinearAttnState state1;
    state1.reset(num_heads, key_dim, value_dim, conv_kernel);

    std::vector<float> h_output_serial(batch_size * hidden_size);
    float* d_output_serial;
    CHECK_CUDA(cudaMalloc(&d_output_serial, batch_size * hidden_size * sizeof(float)));

    for (int b = 0; b < batch_size; ++b) {
        attn.forward(d_input + b * hidden_size, d_output_serial + b * hidden_size, state1);
    }
    CHECK_CUDA(cudaMemcpy(h_output_serial.data(), d_output_serial,
                          batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Test 2: Batch forward
    CudaLinearAttnState state2;
    state2.reset(num_heads, key_dim, value_dim, conv_kernel);

    std::vector<float> h_output_batch(batch_size * hidden_size);
    float* d_output_batch;
    CHECK_CUDA(cudaMalloc(&d_output_batch, batch_size * hidden_size * sizeof(float)));

    attn.forward_batch(d_input, d_output_batch, state2, batch_size);
    CHECK_CUDA(cudaMemcpy(h_output_batch.data(), d_output_batch,
                          batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int diff_count = 0;
    for (int i = 0; i < batch_size * hidden_size; ++i) {
        float diff = std::abs(h_output_serial[i] - h_output_batch[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
        if (diff > 1e-4f) diff_count++;
    }
    avg_diff /= (batch_size * hidden_size);

    std::cout << "Max diff: " << max_diff << std::endl;
    std::cout << "Avg diff: " << avg_diff << std::endl;
    std::cout << "Diff count (>1e-4): " << diff_count << "/" << batch_size * hidden_size
              << std::endl;

    if (max_diff < 1e-3f) {
        std::cout << "PASS: Batch output matches serial output" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: Batch output differs from serial output" << std::endl;
        return 1;
    }
}
