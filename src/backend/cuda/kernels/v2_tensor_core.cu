/**
 * v2.0 Tensor Core (WMMA) Support
 * 
 * Provides mixed-precision GEMM via cuBLAS Tensor Cores:
 * - TF32: Zero-code-change, near-FP32 precision
 * - BF16: 50% memory, good precision, 1.5-2x speedup
 * - FP16: 50% memory, fastest, slight precision loss
 */

#include "cuda_error_handling.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <stdexcept>

namespace qwen {
namespace cuda {

// ============================================================================
// Precision Mode Enum
// ============================================================================
enum class TensorCoreMode {
    FP32,      // Standard FP32, no Tensor Core
    TF32,      // TF32 Tensor Core (Ampere+, near-FP32 precision)
    BF16,      // BF16 Tensor Core (Ampere+, good precision)
    FP16,      // FP16 Tensor Core (all TC GPUs, fastest)
};

// ============================================================================
// Helper: Convert float to/from BF16
// ============================================================================
__host__ __device__ inline unsigned short float_to_bf16(float f) {
    unsigned int bits;
    memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    unsigned int lsb = (bits >> 16) & 1;
    unsigned int rounding_bias = 0x7fff + lsb;
    bits += rounding_bias;
    return static_cast<unsigned short>(bits >> 16);
}

__host__ __device__ inline float bf16_to_float(unsigned short b) {
    unsigned int bits = static_cast<unsigned int>(b) << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// ============================================================================
// Tensor Core GEMM Wrapper
// ============================================================================
class TensorCoreGemm {
public:
    TensorCoreGemm(TensorCoreMode mode = TensorCoreMode::TF32);
    ~TensorCoreGemm();

    void set_mode(TensorCoreMode mode);
    TensorCoreMode mode() const { return mode_; }

    // GEMV: y = alpha * A^T * x + beta * y
    // A is [M, K] row-major, x is [K], y is [M]
    void gemv(int M, int K, const void* A, const void* x, void* y,
              float alpha = 1.0f, float beta = 0.0f);

    // GEMM: C = alpha * A * B + beta * C
    void gemm(int M, int N, int K, const void* A, const void* B, void* C,
              float alpha = 1.0f, float beta = 0.0f);

    // Batch GEMM: multiple GEMMs in one launch
    void gemm_batched(int M, int N, int K, const void* A, const void* B, void* C,
                      int batch_count, float alpha = 1.0f, float beta = 0.0f);

    // Convert weights from FP32 to current precision
    void convert_weights(const float* h_src, void* d_dst, size_t num_elements);

    // Get element size in bytes for current mode
    size_t element_size() const;

    // Get CUDA data type for current mode
    cudaDataType cuda_data_type() const;

private:
    TensorCoreMode mode_;
    cublasHandle_t handle_;

    void setup_math_mode();
};

#define TC_CHECK(call)                                                                             \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error("TensorCore cublas error " +                                  \
                                     std::to_string(static_cast<int>(_err)) + " at " + __FILE__ +  \
                                     ":" + std::to_string(__LINE__));                             \
        }                                                                                          \
    } while (0)

TensorCoreGemm::TensorCoreGemm(TensorCoreMode mode) : mode_(mode) {
    TC_CHECK(cublasCreate(&handle_));
    setup_math_mode();
}

TensorCoreGemm::~TensorCoreGemm() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

void TensorCoreGemm::set_mode(TensorCoreMode mode) {
    if (mode_ != mode) {
        mode_ = mode;
        setup_math_mode();
    }
}

void TensorCoreGemm::setup_math_mode() {
    switch (mode_) {
    case TensorCoreMode::FP32:
        TC_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
        break;
    case TensorCoreMode::TF32:
        TC_CHECK(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
        break;
    case TensorCoreMode::BF16:
    case TensorCoreMode::FP16:
        TC_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
        break;
    }
}

size_t TensorCoreGemm::element_size() const {
    switch (mode_) {
    case TensorCoreMode::FP32:
    case TensorCoreMode::TF32:
        return sizeof(float);
    case TensorCoreMode::BF16:
    case TensorCoreMode::FP16:
        return sizeof(unsigned short);
    }
    return sizeof(float);
}

cudaDataType TensorCoreGemm::cuda_data_type() const {
    switch (mode_) {
    case TensorCoreMode::FP32:
    case TensorCoreMode::TF32:
        return CUDA_R_32F;
    case TensorCoreMode::BF16:
        return CUDA_R_16BF;
    case TensorCoreMode::FP16:
        return CUDA_R_16F;
    }
    return CUDA_R_32F;
}

void TensorCoreGemm::convert_weights(const float* h_src, void* d_dst, size_t num_elements) {
    if (mode_ == TensorCoreMode::FP32 || mode_ == TensorCoreMode::TF32) {
        cudaMemcpy(d_dst, h_src, num_elements * sizeof(float), cudaMemcpyHostToDevice);
        return;
    }

    std::vector<unsigned short> h_dst(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        if (mode_ == TensorCoreMode::BF16) {
            h_dst[i] = float_to_bf16(h_src[i]);
        } else {
            h_dst[i] = reinterpret_cast<const unsigned short&>(__float2half(h_src[i]));
        }
    }
    cudaMemcpy(d_dst, h_dst.data(), num_elements * sizeof(unsigned short), cudaMemcpyHostToDevice);
}

void TensorCoreGemm::gemv(int M, int K, const void* A, const void* x, void* y,
                          float alpha_f, float beta_f) {
    if (mode_ == TensorCoreMode::FP32) {
        TC_CHECK(cublasSgemv(handle_, CUBLAS_OP_T, K, M, &alpha_f,
                             static_cast<const float*>(A), K,
                             static_cast<const float*>(x), 1, &beta_f,
                             static_cast<float*>(y), 1));
        return;
    }

    if (mode_ == TensorCoreMode::TF32) {
        TC_CHECK(cublasSgemv(handle_, CUBLAS_OP_T, K, M, &alpha_f,
                             static_cast<const float*>(A), K,
                             static_cast<const float*>(x), 1, &beta_f,
                             static_cast<float*>(y), 1));
        return;
    }

    // BF16/FP16 use cublasGemmEx
    const __half alpha_h = __float2half(alpha_f);
    const __half beta_h = __float2half(beta_f);
    cudaDataType type = cuda_data_type();
    cublasComputeType_t compute_type = (mode_ == TensorCoreMode::FP16)
                                           ? CUBLAS_COMPUTE_16F
                                           : CUBLAS_COMPUTE_32F;

    TC_CHECK(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                          M, 1, K,
                          (mode_ == TensorCoreMode::FP16) ? static_cast<const void*>(&alpha_h)
                                                          : static_cast<const void*>(&alpha_f),
                          A, type, K,
                          x, type, K,
                          (mode_ == TensorCoreMode::FP16) ? static_cast<const void*>(&beta_h)
                                                          : static_cast<const void*>(&beta_f),
                          y, type, M,
                          compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void TensorCoreGemm::gemm(int M, int N, int K, const void* A, const void* B, void* C,
                          float alpha_f, float beta_f) {
    if (mode_ == TensorCoreMode::FP32 || mode_ == TensorCoreMode::TF32) {
        TC_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha_f,
                             static_cast<const float*>(A), M,
                             static_cast<const float*>(B), K, &beta_f,
                             static_cast<float*>(C), M));
        return;
    }

    const __half alpha_h = __float2half(alpha_f);
    const __half beta_h = __float2half(beta_f);
    cudaDataType type = cuda_data_type();
    cublasComputeType_t compute_type = (mode_ == TensorCoreMode::FP16)
                                           ? CUBLAS_COMPUTE_16F
                                           : CUBLAS_COMPUTE_32F;

    TC_CHECK(cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                          M, N, K,
                          (mode_ == TensorCoreMode::FP16) ? static_cast<const void*>(&alpha_h)
                                                          : static_cast<const void*>(&alpha_f),
                          A, type, M,
                          B, type, K,
                          (mode_ == TensorCoreMode::FP16) ? static_cast<const void*>(&beta_h)
                                                          : static_cast<const void*>(&beta_f),
                          C, type, M,
                          compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void TensorCoreGemm::gemm_batched(int M, int N, int K, const void* A, const void* B, void* C,
                                  int batch_count, float alpha_f, float beta_f) {
    long long int strideA = static_cast<long long int>(M) * K;
    long long int strideB = static_cast<long long int>(K) * N;
    long long int strideC = static_cast<long long int>(M) * N;

    if (mode_ == TensorCoreMode::FP32 || mode_ == TensorCoreMode::TF32) {
        TC_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                           M, N, K, &alpha_f,
                                           static_cast<const float*>(A), M, strideA,
                                           static_cast<const float*>(B), K, strideB, &beta_f,
                                           static_cast<float*>(C), M, strideC, batch_count));
        return;
    }

    const __half alpha_h = __float2half(alpha_f);
    const __half beta_h = __float2half(beta_f);
    cudaDataType type = cuda_data_type();
    cublasComputeType_t compute_type = (mode_ == TensorCoreMode::FP16)
                                           ? CUBLAS_COMPUTE_16F
                                           : CUBLAS_COMPUTE_32F;

    TC_CHECK(cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                        M, N, K,
                                        (mode_ == TensorCoreMode::FP16)
                                            ? static_cast<const void*>(&alpha_h)
                                            : static_cast<const void*>(&alpha_f),
                                        A, type, M, strideA,
                                        B, type, K, strideB,
                                        (mode_ == TensorCoreMode::FP16)
                                            ? static_cast<const void*>(&beta_h)
                                            : static_cast<const void*>(&beta_f),
                                        C, type, M, strideC,
                                        batch_count, compute_type,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

} // namespace cuda
} // namespace qwen
