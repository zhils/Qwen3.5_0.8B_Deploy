#include "mlp_cuda.hpp"
#include "cuda_error_handling.cuh"
#include "fused_kernels.cuh"
#include "cublas_handle_pool.hpp"
#include <cmath>
#include <stdexcept>
#include <cublas_v2.h>

namespace qwen {
namespace cuda {

__device__ int8_t unpack_int4(uint8_t packed, int idx) {
    int8_t val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    if (val & 0x08) val |= 0xF0;
    return val;
}

__global__ void gemv_int4_kernel(const uint8_t* __restrict__ weight_packed,
                                 const float* __restrict__ weight_scale,
                                 const float* __restrict__ input, float* __restrict__ output,
                                 int out_size, int in_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_size) return;

    float scale = weight_scale[row];
    float sum = 0.0f;
    int packed_in_size = (in_size + 1) / 2;
    for (int col = 0; col < in_size; ++col) {
        int packed_idx = col / 2;
        int sub_idx = col % 2;
        int8_t w = unpack_int4(weight_packed[row * packed_in_size + packed_idx], sub_idx);
        sum += static_cast<float>(w) * input[col];
    }
    output[row] = sum * scale;
}

__global__ void gemm_int4_kernel(const uint8_t* __restrict__ weight_packed,
                                 const float* __restrict__ weight_scale,
                                 const float* __restrict__ input, float* __restrict__ output,
                                 int out_size, int in_size, int batch_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (row >= out_size || b >= batch_size) return;

    float scale = weight_scale[row];
    float sum = 0.0f;
    const float* in_ptr = input + b * in_size;
    float* out_ptr = output + b * out_size;

    int packed_in_size = (in_size + 1) / 2;
    for (int col = 0; col < in_size; ++col) {
        int packed_idx = col / 2;
        int sub_idx = col % 2;
        int8_t w = unpack_int4(weight_packed[row * packed_in_size + packed_idx], sub_idx);
        sum += static_cast<float>(w) * in_ptr[col];
    }
    out_ptr[row] = sum * scale;
}

CudaMLP::CudaMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      d_gate_proj_weight_packed_(nullptr), d_up_proj_weight_packed_(nullptr),
      d_down_proj_weight_packed_(nullptr),
      d_gate_proj_scale_(nullptr), d_up_proj_scale_(nullptr), d_down_proj_scale_(nullptr),
      d_hidden_buf_(nullptr), max_hidden_batch_(0) {
    size_t gate_packed_size = static_cast<size_t>(intermediate_size_) * ((hidden_size_ + 1) / 2);
    size_t down_packed_size = static_cast<size_t>(hidden_size_) * ((intermediate_size_ + 1) / 2);

    cudaMalloc(&d_gate_proj_weight_packed_, gate_packed_size * sizeof(uint8_t));
    cudaMalloc(&d_up_proj_weight_packed_, gate_packed_size * sizeof(uint8_t));
    cudaMalloc(&d_down_proj_weight_packed_, down_packed_size * sizeof(uint8_t));
    cudaMalloc(&d_gate_proj_scale_, intermediate_size_ * sizeof(float));
    cudaMalloc(&d_up_proj_scale_, intermediate_size_ * sizeof(float));
    cudaMalloc(&d_down_proj_scale_, hidden_size_ * sizeof(float));
}

CudaMLP::~CudaMLP() {
    if (d_gate_proj_weight_packed_) cudaFree(d_gate_proj_weight_packed_);
    if (d_up_proj_weight_packed_) cudaFree(d_up_proj_weight_packed_);
    if (d_down_proj_weight_packed_) cudaFree(d_down_proj_weight_packed_);
    if (d_gate_proj_scale_) cudaFree(d_gate_proj_scale_);
    if (d_up_proj_scale_) cudaFree(d_up_proj_scale_);
    if (d_down_proj_scale_) cudaFree(d_down_proj_scale_);
    if (d_hidden_buf_) cudaFree(d_hidden_buf_);
}

void CudaMLP::ensure_hidden_buffer(int batch_size) const {
    if (batch_size <= max_hidden_batch_ && d_hidden_buf_ != nullptr) {
        return;
    }
    if (d_hidden_buf_) {
        cudaFree(d_hidden_buf_);
    }
    size_t bytes = static_cast<size_t>(batch_size) * intermediate_size_ * sizeof(float) * 2;
    cudaMalloc(&d_hidden_buf_, bytes);
    max_hidden_batch_ = batch_size;
}

void CudaMLP::set_weights(const std::vector<float>& gate_proj_weight,
                          const std::vector<float>& up_proj_weight,
                          const std::vector<float>& down_proj_weight) {
    size_t gate_packed_size = static_cast<size_t>(intermediate_size_) * ((hidden_size_ + 1) / 2);
    size_t down_packed_size = static_cast<size_t>(hidden_size_) * ((intermediate_size_ + 1) / 2);

    auto pack_int4 = [](const std::vector<float>& weights, int out_size, int in_size,
                        std::vector<uint8_t>& packed, std::vector<float>& scale) {
        packed.resize(out_size * ((in_size + 1) / 2));
        scale.resize(out_size);

        for (int row = 0; row < out_size; ++row) {
            float max_val = 0.0f;
            for (int col = 0; col < in_size; ++col) {
                max_val = fmaxf(max_val, fabsf(weights[row * in_size + col]));
            }
            scale[row] = max_val / 7.0f;

            for (int col = 0; col < in_size; col += 2) {
                int8_t v0 = static_cast<int8_t>(roundf(weights[row * in_size + col] / scale[row]));
                v0 = max(-8, min(7, v0));
                int8_t v1 = 0;
                if (col + 1 < in_size) {
                    v1 = static_cast<int8_t>(roundf(weights[row * in_size + col + 1] / scale[row]));
                    v1 = max(-8, min(7, v1));
                }
                packed[row * ((in_size + 1) / 2) + col / 2] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
            }
        }
    };

    std::vector<uint8_t> h_gate_packed, h_up_packed, h_down_packed;
    std::vector<float> h_gate_scale, h_up_scale, h_down_scale;

    pack_int4(gate_proj_weight, intermediate_size_, hidden_size_, h_gate_packed, h_gate_scale);
    pack_int4(up_proj_weight, intermediate_size_, hidden_size_, h_up_packed, h_up_scale);
    pack_int4(down_proj_weight, hidden_size_, intermediate_size_, h_down_packed, h_down_scale);

    cudaMemcpy(d_gate_proj_weight_packed_, h_gate_packed.data(), gate_packed_size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_weight_packed_, h_up_packed.data(), gate_packed_size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_weight_packed_, h_down_packed.data(), down_packed_size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate_proj_scale_, h_gate_scale.data(), intermediate_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_proj_scale_, h_up_scale.data(), intermediate_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_proj_scale_, h_down_scale.data(), hidden_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
}

void CudaMLP::forward(const float* input, float* output, int batch_size) const {
    if (batch_size == 1) {
        float* d_gate_buf;
        float* d_up_buf;
        cudaMalloc(&d_gate_buf, intermediate_size_ * sizeof(float));
        cudaMalloc(&d_up_buf, intermediate_size_ * sizeof(float));

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256);
        gemv_int4_kernel<<<grid_gate, block>>>(d_gate_proj_weight_packed_, d_gate_proj_scale_,
                                                input, d_gate_buf, intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        gemv_int4_kernel<<<grid_gate, block>>>(d_up_proj_weight_packed_, d_up_proj_scale_,
                                                input, d_up_buf, intermediate_size_, hidden_size_);
        CUDA_CHECK_LAST_KERNEL();

        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        dim3 grid_down((hidden_size_ + 255) / 256);
        gemv_int4_kernel<<<grid_down, block>>>(d_down_proj_weight_packed_, d_down_proj_scale_,
                                                d_gate_buf, output, hidden_size_, intermediate_size_);
        CUDA_CHECK_LAST_KERNEL();

        cudaFree(d_gate_buf);
        cudaFree(d_up_buf);
    } else {
        ensure_hidden_buffer(batch_size);

        float* d_gate_buf = d_hidden_buf_;
        float* d_up_buf = d_hidden_buf_ + static_cast<size_t>(batch_size) * intermediate_size_;

        dim3 block(256);
        dim3 grid_gate((intermediate_size_ + 255) / 256, batch_size);
        gemm_int4_kernel<<<grid_gate, block>>>(d_gate_proj_weight_packed_, d_gate_proj_scale_,
                                                input, d_gate_buf, intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        gemm_int4_kernel<<<grid_gate, block>>>(d_up_proj_weight_packed_, d_up_proj_scale_,
                                                input, d_up_buf, intermediate_size_, hidden_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();

        int total_elements = batch_size * intermediate_size_;
        launch_silu_mul_batch(d_gate_buf, d_up_buf, d_gate_buf, total_elements);

        dim3 grid_down((hidden_size_ + 255) / 256, batch_size);
        gemm_int4_kernel<<<grid_down, block>>>(d_down_proj_weight_packed_, d_down_proj_scale_,
                                                d_gate_buf, output, hidden_size_, intermediate_size_, batch_size);
        CUDA_CHECK_LAST_KERNEL();
    }
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    forward(input, residual, batch_size);
}

} // namespace cuda
} // namespace qwen
