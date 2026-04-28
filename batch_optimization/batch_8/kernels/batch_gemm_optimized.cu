#include "batch_config.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace qwen {
namespace cuda {

#if OPTIMIZED_BATCH_SIZE == 8

__global__ void __launch_bounds__(256) batch_gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size) {
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        #pragma unroll UNROLL_FACTOR
        for (int k = 0; k < K; ++k) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C_batch[row * N + col] = sum;
    }
}

#elif OPTIMIZED_BATCH_SIZE == 16

__global__ void __launch_bounds__(256) batch_gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size) {
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float sA[64][32];
    __shared__ float sB[32][64];
    
    float sum = 0.0f;
    
    for (int k_tile = 0; k_tile < K; k_tile += 32) {
        if (threadIdx.y < 64 && threadIdx.x < 32) {
            sA[threadIdx.y][threadIdx.x] = A_batch[row * K + k_tile + threadIdx.x];
        }
        if (threadIdx.y < 32 && threadIdx.x < 64) {
            sB[threadIdx.y][threadIdx.x] = B_batch[(k_tile + threadIdx.y) * N + col];
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

#elif OPTIMIZED_BATCH_SIZE == 32

__global__ void __launch_bounds__(512) batch_gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size) {
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;
    
    int row = blockIdx.y * 128 + threadIdx.y;
    int col = blockIdx.x * 128 + threadIdx.x;
    
    __shared__ float sA[128][32];
    __shared__ float sB[32][128];
    
    float sum = 0.0f;
    
    for (int k_tile = 0; k_tile < K; k_tile += 32) {
        if (row < M && (k_tile + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A_batch[row * K + k_tile + threadIdx.x];
        }
        if ((k_tile + threadIdx.y) < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B_batch[(k_tile + threadIdx.y) * N + col];
        }
        __syncthreads();
        
        #pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

#elif OPTIMIZED_BATCH_SIZE == 64

__global__ void __launch_bounds__(512) batch_gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size) {
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;
    
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    
    int row = blockIdx.y * 128 + warp_id * 4 + (lane_id >> 3);
    int col = blockIdx.x * 256 + (lane_id & 7) * 32;
    
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int k_tile = 0; k_tile < K; k_tile += 64) {
        float a_frag[4], b_frag[4];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = row + i / 2;
            int k = k_tile + lane_id;
            if (r < M && k < K) {
                a_frag[i] = A_batch[r * K + k];
            } else {
                a_frag[i] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int k = k_tile + warp_id;
            int c = col + i * 8;
            if (k < K && c < N) {
                b_frag[i] = B_batch[k * N + c];
            } else {
                b_frag[i] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum[i] += a_frag[j] * b_frag[i];
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = row + i / 2;
        int c = col + i * 8;
        if (r < M && c < N) {
            C_batch[r * N + c] = sum[i];
        }
    }
}

#elif OPTIMIZED_BATCH_SIZE == 128

__global__ void __launch_bounds__(1024) batch_gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size) {
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const float* A_batch = A + batch_idx * M * K;
    const float* B_batch = B + batch_idx * K * N;
    float* C_batch = C + batch_idx * M * N;
    
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    
    int row = blockIdx.y * 256 + warp_id * 8 + (lane_id >> 2);
    int col = blockIdx.x * 256 + (lane_id & 3) * 64;
    
    float sum[8] = {0.0f};
    
    for (int k_tile = 0; k_tile < K; k_tile += 64) {
        float a_frag[8], b_frag[8];
        
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            int r = row + i / 4;
            int k = k_tile + lane_id;
            a_frag[i] = (r < M && k < K) ? A_batch[r * K + k] : 0.0f;
        }
        
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            int k = k_tile + warp_id;
            int c = col + i * 8;
            b_frag[i] = (k < K && c < N) ? B_batch[k * N + c] : 0.0f;
        }
        
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            #pragma unroll 8
            for (int j = 0; j < 8; ++j) {
                sum[i] += a_frag[j] * b_frag[i];
            }
        }
    }
    
    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        int r = row + i / 4;
        int c = col + i * 8;
        if (r < M && c < N) {
            C_batch[r * N + c] = sum[i];
        }
    }
}

#endif

void launch_batch_gemm_optimized(const float* A, const float* B, float* C,
                                  int M, int N, int K, int batch_size,
                                  cudaStream_t stream) {
#if OPTIMIZED_BATCH_SIZE == 8
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, batch_size);
#elif OPTIMIZED_BATCH_SIZE == 16
    dim3 block(16, 16);
    dim3 grid((N + 63) / 64, (M + 63) / 64, batch_size);
#elif OPTIMIZED_BATCH_SIZE == 32
    dim3 block(128, 128);
    dim3 grid((N + 127) / 128, (M + 127) / 128, batch_size);
#elif OPTIMIZED_BATCH_SIZE == 64
    dim3 block(512);
    dim3 grid((N + 255) / 256, (M + 127) / 128, batch_size);
#elif OPTIMIZED_BATCH_SIZE == 128
    dim3 block(1024);
    dim3 grid((N + 255) / 256, (M + 255) / 256, batch_size);
#endif
    
    batch_gemm_optimized<<<grid, block, 0, stream>>>(A, B, C, M, N, K, batch_size);
}

}
}
