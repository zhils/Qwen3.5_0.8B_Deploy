#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace qwen {
namespace cuda {

inline cublasStatus_t checkCublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d at %s:%d\n", static_cast<int>(status), file, line);
    }
    return status;
}

#define CUBLAS_CHECK(call) checkCublas(call, __FILE__, __LINE__)

class CublasHandle {
public:
    static CublasHandle& instance() {
        static CublasHandle inst;
        return inst;
    }

    cublasHandle_t handle() const { return handle_; }

    ~CublasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }

private:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t handle_ = nullptr;
};

inline void cublas_sgemm(
    const float* weight,
    const float* input,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0) {
    cublasHandle_t handle = CublasHandle::instance().handle();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        weight, K,
        input, K,
        &beta,
        output, M));
}

inline void cublas_sgemm_add_residual(
    const float* weight,
    const float* input,
    float* output,
    int M, int N, int K,
    cudaStream_t stream = 0) {
    cublasHandle_t handle = CublasHandle::instance().handle();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        weight, K,
        input, K,
        &beta,
        output, M));
}

}
}
