#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace qwen {
namespace cuda {

#define BF16_CUBLAS_CHECK(call)                                                                    \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error(std::string("cuBLAS error ") +                                \
                                     std::to_string(static_cast<int>(_err)) + " at " + __FILE__ +  \
                                     ":" + std::to_string(__LINE__));                              \
        }                                                                                          \
    } while (0)

#define BF16_CUDA_CHECK(call)                                                                      \
    do {                                                                                           \
        cudaError_t _err = (call);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) +      \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                                          \
    } while (0)

#ifdef __CUDACC__

static __global__ void fp32_to_bf16_kernel(const float* __restrict__ fp32_data,
                                           __nv_bfloat16* __restrict__ bf16_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    bf16_data[i] = __float2bfloat16(fp32_data[i]);
}

static __global__ void bf16_to_fp32_kernel(const __nv_bfloat16* __restrict__ bf16_data,
                                           float* __restrict__ fp32_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    fp32_data[i] = __bfloat162float(bf16_data[i]);
}

#endif // __CUDACC__

inline void convert_fp32_to_bf16(const float* d_fp32, __nv_bfloat16* d_bf16, int n,
                                 cudaStream_t stream = 0) {
    if (n <= 0)
        return;
#ifdef __CUDACC__
    int block = 256;
    int grid = (n + 255) / 256;
    fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(d_fp32, d_bf16, n);
#else
    for (int i = 0; i < n; ++i) {
        d_bf16[i] = __float2bfloat16(d_fp32[i]);
    }
#endif
}

inline void convert_bf16_to_fp32(const __nv_bfloat16* d_bf16, float* d_fp32, int n,
                                 cudaStream_t stream = 0) {
    if (n <= 0)
        return;
#ifdef __CUDACC__
    int block = 256;
    int grid = (n + 255) / 256;
    bf16_to_fp32_kernel<<<grid, block, 0, stream>>>(d_bf16, d_fp32, n);
#else
    for (int i = 0; i < n; ++i) {
        d_fp32[i] = __bfloat162float(d_bf16[i]);
    }
#endif
}

inline void bf16_gemv(cublasHandle_t handle, const __nv_bfloat16* d_weight_bf16,
                      const __nv_bfloat16* d_input_bf16, __nv_bfloat16* d_output_bf16, int rows,
                      int cols, bool transpose_weight = true) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (transpose_weight) {
        BF16_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols, &alpha,
                                       static_cast<const void*>(d_weight_bf16), CUDA_R_16BF, cols,
                                       static_cast<const void*>(d_input_bf16), CUDA_R_16BF, cols,
                                       &beta, static_cast<void*>(d_output_bf16), CUDA_R_16BF, rows,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        BF16_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, cols, &alpha,
                                       static_cast<const void*>(d_weight_bf16), CUDA_R_16BF, rows,
                                       static_cast<const void*>(d_input_bf16), CUDA_R_16BF, cols,
                                       &beta, static_cast<void*>(d_output_bf16), CUDA_R_16BF, rows,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

inline void bf16_gemv_add(cublasHandle_t handle, const __nv_bfloat16* d_weight_bf16,
                          const __nv_bfloat16* d_input_bf16, __nv_bfloat16* d_output_bf16, int rows,
                          int cols, bool transpose_weight = true) {
    const float alpha = 1.0f;
    const float beta = 1.0f;

    if (transpose_weight) {
        BF16_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols, &alpha,
                                       static_cast<const void*>(d_weight_bf16), CUDA_R_16BF, cols,
                                       static_cast<const void*>(d_input_bf16), CUDA_R_16BF, cols,
                                       &beta, static_cast<void*>(d_output_bf16), CUDA_R_16BF, rows,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        BF16_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, cols, &alpha,
                                       static_cast<const void*>(d_weight_bf16), CUDA_R_16BF, rows,
                                       static_cast<const void*>(d_input_bf16), CUDA_R_16BF, cols,
                                       &beta, static_cast<void*>(d_output_bf16), CUDA_R_16BF, rows,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

class BFloat16WeightCache {
  public:
    BFloat16WeightCache() : d_fp32_(nullptr), d_bf16_(nullptr), size_(0), bf16_ready_(false) {}

    ~BFloat16WeightCache() {
        if (d_fp32_)
            cudaFree(d_fp32_);
        if (d_bf16_)
            cudaFree(d_bf16_);
    }

    void allocate(int num_elements) {
        size_ = num_elements;
        BF16_CUDA_CHECK(cudaMalloc(&d_fp32_, size_ * sizeof(float)));
        BF16_CUDA_CHECK(cudaMalloc(&d_bf16_, size_ * sizeof(__nv_bfloat16)));
        bf16_ready_ = false;
    }

    void set_weight(const float* h_data, int num_elements) {
        if (num_elements != size_) {
            if (d_fp32_)
                cudaFree(d_fp32_);
            if (d_bf16_)
                cudaFree(d_bf16_);
            size_ = num_elements;
            BF16_CUDA_CHECK(cudaMalloc(&d_fp32_, size_ * sizeof(float)));
            BF16_CUDA_CHECK(cudaMalloc(&d_bf16_, size_ * sizeof(__nv_bfloat16)));
        }
        BF16_CUDA_CHECK(cudaMemcpy(d_fp32_, h_data, size_ * sizeof(float), cudaMemcpyHostToDevice));
        convert_fp32_to_bf16(d_fp32_, d_bf16_, size_);
        bf16_ready_ = true;
    }

    void set_weight_device(const float* d_data, int num_elements) {
        if (num_elements != size_) {
            if (d_fp32_)
                cudaFree(d_fp32_);
            if (d_bf16_)
                cudaFree(d_bf16_);
            size_ = num_elements;
            BF16_CUDA_CHECK(cudaMalloc(&d_fp32_, size_ * sizeof(float)));
            BF16_CUDA_CHECK(cudaMalloc(&d_bf16_, size_ * sizeof(__nv_bfloat16)));
        }
        BF16_CUDA_CHECK(
            cudaMemcpy(d_fp32_, d_data, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
        convert_fp32_to_bf16(d_fp32_, d_bf16_, size_);
        bf16_ready_ = true;
    }

    float* fp32() const {
        return d_fp32_;
    }
    __nv_bfloat16* bf16() const {
        return d_bf16_;
    }
    int size() const {
        return size_;
    }
    bool ready() const {
        return bf16_ready_;
    }

  private:
    float* d_fp32_;
    __nv_bfloat16* d_bf16_;
    int size_;
    bool bf16_ready_;
};

} // namespace cuda
} // namespace qwen
