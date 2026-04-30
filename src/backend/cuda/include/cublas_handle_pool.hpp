#pragma once

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t _err = (call);                                                              \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error(std::string("cuBLAS error ") +                                \
                                     std::to_string(static_cast<int>(_err)) + " at " + __FILE__ +  \
                                     ":" + std::to_string(__LINE__));                             \
        }                                                                                          \
    } while (0)

namespace qwen {
namespace cuda {

class CublasHandlePool {
  public:
    static CublasHandlePool& instance() {
        static CublasHandlePool pool;
        return pool;
    }

    cublasHandle_t get() const {
        return handle_;
    }

  private:
    CublasHandlePool() {
        // Check CUDA context
        cudaError_t cuda_err = cudaFree(0);
        if (cuda_err != cudaSuccess) {
            // No CUDA context available - create one
            cudaSetDevice(0);
            cudaFree(0);
        }
        
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("cuBLAS error ") +
                                     std::to_string(static_cast<int>(status)) +
                                     " during cublasCreate at " + __FILE__ +
                                     ":" + std::to_string(__LINE__));
        }
    }

    ~CublasHandlePool() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CublasHandlePool(const CublasHandlePool&) = delete;
    CublasHandlePool& operator=(const CublasHandlePool&) = delete;

    cublasHandle_t handle_ = nullptr;
};

} // namespace cuda
} // namespace qwen
