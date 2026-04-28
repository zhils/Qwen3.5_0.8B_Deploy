#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace qwen {
namespace cuda {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string msg = std::string("CUDA Error at ") + file + ":" + std::to_string(line) + ": " +
                          cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

#define CUDA_CHECK(call) qwen::cuda::cuda_check((call), __FILE__, __LINE__)

#define CUDA_CHECK_LAST_KERNEL()                                                                   \
    do {                                                                                           \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            throw std::runtime_error(std::string("CUDA Kernel Error: ") +                          \
                                     cudaGetErrorString(err));                                     \
        }                                                                                          \
    } while (0)

} // namespace cuda
} // namespace qwen
