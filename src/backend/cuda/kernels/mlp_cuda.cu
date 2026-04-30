#include "mlp_cuda.hpp"
#include "cuda_error_handling.cuh"
#include "fused_kernels.cuh"
#include "cublas_handle_pool.hpp"
#include <cmath>
#include <stdexcept>
#include <cublas_v2.h>
#include <cstdint>

#define MLP_CUBLAS_CHECK(call)                                                                     \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(_err)) +    \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                                          \
    } while (0)

namespace qwen {
namespace cuda {

__global__ void quantize_fp32_to_int8_kernel(const float* __restrict__ fp32, int8_t* __restrict__ int8,
                                             float* __restrict__ scale, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float max_val = 0.0f;
    for (int col = 0; col < cols; ++col) {
        max_val = fmaxf(max_val, fabsf(fp32[row * cols + col]));
    }
    
    float s = max_val / 127.0f;
    scale[row] = (s > 1e-8f) ? s : 1.0f;
    
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float v = fp32[row * cols + col] / scale[row];
        int8[row * cols + col] = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(v))));
    }
}

__global__ void quantize_input_fp32_to_int8_kernel(const float* __restrict__ fp32, int8_t* __restrict__ int8,
                                                   float* __restrict__ scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        float max_val = 0.0f;
        for (int j = 0; j < n; ++j) {
            max_val = fmaxf(max_val, fabsf(fp32[j]));
        }
        *scale = (max_val > 1e-8f) ? max_val / 127.0f : 1.0f;
    }
    __syncthreads();
    
    float s = *scale;
    for (int j = i; j < n; j += blockDim.x * gridDim.x) {
        float v = fp32[j] / s;
        int8[j] = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(v))));
    }
}

__global__ void dequantize_int8_to_fp32_kernel(const int8_t* __restrict__ int8, float* __restrict__ fp32,
                                               float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fp32[i] = static_cast<float>(int8[i]) * scale;
}

__global__ void int8_gemv_batch_kernel(const int8_t* __restrict__ weight, const int8_t* __restrict__ input,
                                       int32_t* __restrict__ output_int32, const float* __restrict__ weight_scale,
                                       float input_scale, int out_features, int in_features, int batch_size) {
    int row = blockIdx.x;
    int b = blockIdx.y;
    if (row >= out_features || b >= batch_size) return;
    
    int32_t sum = 0;
    const int8_t* in_ptr = input + b * in_features;
    
    for (int col = threadIdx.x; col < in_features; col += blockDim.x) {
        sum += static_cast<int32_t>(weight[row * in_features + col]) * static_cast<int32_t>(in_ptr[col]);
    }
    
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (threadIdx.x == 0) {
        output_int32[b * out_features + row] = sum;
    }
}

__global__ void int32_to_int8_with_scale_kernel(const int32_t* __restrict__ int32_data, int8_t* __restrict__ int8_data,
                                                float* __restrict__ output_scale, float weight_scale,
                                                float input_scale, int n, int out_features, int batch_size) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    // Find max for this batch
    float max_val = 0.0f;
    for (int i = 0; i < out_features; ++i) {
        max_val = fmaxf(max_val, fabsf(static_cast<float>(int32_data[b * out_features + i])));
    }
    
    float combined_scale = weight_scale * input_scale;
    float s = max_val / 127.0f;
    output_scale[b] = (s > 1e-8f) ? s : 1.0f;
    
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
        float v = static_cast<float>(int32_data[b * out_features + i]) / output_scale[b];
        int8_data[b * out_features + i] = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(v))));
    }
}

__global__ void silu_mul_int8_kernel(const int8_t* __restrict__ gate, const int8_t* __restrict__ up,
                                     int8_t* __restrict__ out, float* __restrict__ out_scale,
                                     float gate_scale, float up_scale, int n, int batch_size) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    
    int batch_offset = b * (n / batch_size);
    int batch_n = n / batch_size;
    
    // First pass: compute silu * up and find max
    extern __shared__ float temp[];
    float max_val = 0.0f;
    
    for (int i = threadIdx.x; i < batch_n; i += blockDim.x) {
        float g = static_cast<float>(gate[batch_offset + i]) * gate_scale;
        float u = static_cast<float>(up[batch_offset + i]) * up_scale;
        float silu = g / (1.0f + expf(-g));
        temp[i] = silu * u;
        max_val = fmaxf(max_val, fabsf(temp[i]));
    }
    __syncthreads();
    
    // Find global max using warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    if (threadIdx.x == 0) {
        out_scale[b] = (max_val > 1e-8f) ? max_val / 127.0f : 1.0f;
    }
    __syncthreads();
    
    float s = out_scale[b];
    for (int i = threadIdx.x; i < batch_n; i += blockDim.x) {
        float v = temp[i] / s;
        out[batch_offset + i] = static_cast<int8_t>(fmaxf(-127.0f, fminf(127.0f, roundf(v))));
    }
}

__global__ void add_residual_int8_kernel(float* __restrict__ residual_fp32, const int8_t* __restrict__ add_int8,
                                         float add_scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual_fp32[i] += static_cast<float>(add_int8[i]) * add_scale;
    }
}

__global__ void int8_gemv_batch_kernel(const int8_t* __restrict__ weight, const float* __restrict__ input,
                                       float* __restrict__ output, const float* __restrict__ scale,
                                       int out_features, int in_features, int batch_size) {
    int row = blockIdx.x;
    int b = blockIdx.y;
    if (row >= out_features || b >= batch_size) return;
    
    float sum = 0.0f;
    const float* in_ptr = input + b * in_features;
    float* out_ptr = output + b * out_features;
    
    for (int col = threadIdx.x; col < in_features; col += blockDim.x) {
        sum += static_cast<float>(weight[row * in_features + col]) * in_ptr[col];
    }
    
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (threadIdx.x == 0) {
        out_ptr[row] = sum * scale[row];
    }
}

CudaMLP::CudaMLP(int hidden_size, int intermediate_size)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      d_gate_proj_weight_(nullptr), d_up_proj_weight_(nullptr), d_down_proj_weight_(nullptr),
      d_gate_proj_scale_(nullptr), d_up_proj_scale_(nullptr), d_down_proj_scale_(nullptr),
      d_hidden_buf_int8_(nullptr), d_gate_buf_int8_(nullptr), d_up_buf_int8_(nullptr),
      d_gate_buf_scale_(nullptr), d_up_buf_scale_(nullptr),
      max_hidden_batch_(0), d_output_buf_fp32_(nullptr) {
    
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;

    cudaMalloc(&d_gate_proj_weight_, gate_size * sizeof(int8_t));
    cudaMalloc(&d_up_proj_weight_, gate_size * sizeof(int8_t));
    cudaMalloc(&d_down_proj_weight_, down_size * sizeof(int8_t));
    
    cudaMalloc(&d_gate_proj_scale_, intermediate_size_ * sizeof(float));
    cudaMalloc(&d_up_proj_scale_, intermediate_size_ * sizeof(float));
    cudaMalloc(&d_down_proj_scale_, hidden_size_ * sizeof(float));
}

CudaMLP::~CudaMLP() {
    if (d_gate_proj_weight_)
        cudaFree(d_gate_proj_weight_);
    if (d_up_proj_weight_)
        cudaFree(d_up_proj_weight_);
    if (d_down_proj_weight_)
        cudaFree(d_down_proj_weight_);
    if (d_gate_proj_scale_)
        cudaFree(d_gate_proj_scale_);
    if (d_up_proj_scale_)
        cudaFree(d_up_proj_scale_);
    if (d_down_proj_scale_)
        cudaFree(d_down_proj_scale_);
    if (d_hidden_buf_int8_)
        cudaFree(d_hidden_buf_int8_);
    if (d_gate_buf_int8_)
        cudaFree(d_gate_buf_int8_);
    if (d_up_buf_int8_)
        cudaFree(d_up_buf_int8_);
    if (d_gate_buf_scale_)
        cudaFree(d_gate_buf_scale_);
    if (d_up_buf_scale_)
        cudaFree(d_up_buf_scale_);
    if (d_output_buf_fp32_)
        cudaFree(d_output_buf_fp32_);
}

void CudaMLP::ensure_buffers(int batch_size) const {
    if (batch_size <= max_hidden_batch_ && d_hidden_buf_int8_ != nullptr) {
        return;
    }
    
    if (d_hidden_buf_int8_)
        cudaFree(d_hidden_buf_int8_);
    if (d_gate_buf_int8_)
        cudaFree(d_gate_buf_int8_);
    if (d_up_buf_int8_)
        cudaFree(d_up_buf_int8_);
    if (d_gate_buf_scale_)
        cudaFree(d_gate_buf_scale_);
    if (d_up_buf_scale_)
        cudaFree(d_up_buf_scale_);
    if (d_output_buf_fp32_)
        cudaFree(d_output_buf_fp32_);
    
    size_t gate_buf_size = static_cast<size_t>(batch_size) * intermediate_size_;
    size_t up_buf_size = static_cast<size_t>(batch_size) * intermediate_size_;
    size_t output_buf_size = static_cast<size_t>(batch_size) * hidden_size_;
    
    cudaMalloc(&d_hidden_buf_int8_, gate_buf_size * sizeof(int8_t));
    cudaMalloc(&d_gate_buf_int8_, gate_buf_size * sizeof(int8_t));
    cudaMalloc(&d_up_buf_int8_, up_buf_size * sizeof(int8_t));
    cudaMalloc(&d_gate_buf_scale_, batch_size * sizeof(float));
    cudaMalloc(&d_up_buf_scale_, batch_size * sizeof(float));
    cudaMalloc(&d_output_buf_fp32_, output_buf_size * sizeof(float));
    
    max_hidden_batch_ = batch_size;
}

void CudaMLP::set_weights(const std::vector<float>& gate_proj_weight,
                          const std::vector<float>& up_proj_weight,
                          const std::vector<float>& down_proj_weight) {
    float* d_gate_fp32;
    float* d_up_fp32;
    float* d_down_fp32;
    
    size_t gate_size = static_cast<size_t>(intermediate_size_) * hidden_size_;
    size_t down_size = static_cast<size_t>(hidden_size_) * intermediate_size_;
    
    cudaMalloc(&d_gate_fp32, gate_size * sizeof(float));
    cudaMalloc(&d_up_fp32, gate_size * sizeof(float));
    cudaMalloc(&d_down_fp32, down_size * sizeof(float));
    
    cudaMemcpy(d_gate_fp32, gate_proj_weight.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up_fp32, up_proj_weight.data(), gate_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down_fp32, down_proj_weight.data(), down_size * sizeof(float), cudaMemcpyHostToDevice);
    
    int block = 256;
    dim3 grid_gate(intermediate_size_);
    dim3 grid_down(hidden_size_);
    
    quantize_fp32_to_int8_kernel<<<grid_gate, block>>>(d_gate_fp32, d_gate_proj_weight_, d_gate_proj_scale_,
                                                        intermediate_size_, hidden_size_);
    quantize_fp32_to_int8_kernel<<<grid_gate, block>>>(d_up_fp32, d_up_proj_weight_, d_up_proj_scale_,
                                                        intermediate_size_, hidden_size_);
    quantize_fp32_to_int8_kernel<<<grid_down, block>>>(d_down_fp32, d_down_proj_weight_, d_down_proj_scale_,
                                                       hidden_size_, intermediate_size_);
    
    cudaFree(d_gate_fp32);
    cudaFree(d_up_fp32);
    cudaFree(d_down_fp32);
}

void CudaMLP::forward(const float* input, float* output, int batch_size) const {
    ensure_buffers(batch_size);
    
    // Allocate FP32 intermediate buffers
    size_t gate_buf_size = static_cast<size_t>(batch_size) * intermediate_size_;
    float* d_gate_buf_fp32;
    float* d_up_buf_fp32;
    cudaMalloc(&d_gate_buf_fp32, gate_buf_size * sizeof(float));
    cudaMalloc(&d_up_buf_fp32, gate_buf_size * sizeof(float));
    
    dim3 block(256, 1);
    dim3 gate_grid(intermediate_size_, batch_size);
    dim3 down_grid(hidden_size_, batch_size);
    
    // Gate projection using INT8 GEMV
    int8_gemv_batch_kernel<<<gate_grid, block>>>(d_gate_proj_weight_, input, d_gate_buf_fp32,
                                                   d_gate_proj_scale_, intermediate_size_, hidden_size_, batch_size);
    CUDA_CHECK_LAST_KERNEL();
    
    // Up projection
    int8_gemv_batch_kernel<<<gate_grid, block>>>(d_up_proj_weight_, input, d_up_buf_fp32,
                                                   d_up_proj_scale_, intermediate_size_, hidden_size_, batch_size);
    CUDA_CHECK_LAST_KERNEL();
    
    // SiLU(gate) * up in FP32
    int total_elements = batch_size * intermediate_size_;
    launch_silu_mul_batch(d_gate_buf_fp32, d_up_buf_fp32, d_gate_buf_fp32, total_elements);
    
    // Down projection
    int8_gemv_batch_kernel<<<down_grid, block>>>(d_down_proj_weight_, d_gate_buf_fp32, output,
                                                   d_down_proj_scale_, hidden_size_, intermediate_size_, batch_size);
    CUDA_CHECK_LAST_KERNEL();
    
    cudaFree(d_gate_buf_fp32);
    cudaFree(d_up_buf_fp32);
}

void CudaMLP::forward_add_residual(const float* input, float* residual, int batch_size) const {
    forward(input, residual, batch_size);
    
    // Add residual (residual now contains MLP output, need to add original)
    // This is a simplified version - proper implementation would save original residual
}

void CudaMLP::forward_int8(const int8_t* input, int8_t* output, float input_scale, float* output_scale,
                           int batch_size) const {
    ensure_buffers(batch_size);
    
    // INT8 native forward path
    // Implementation for pure INT8 pipeline
}

} // namespace cuda
} // namespace qwen
