#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define EPSILON 1e-4f

// RMSNorm kernel for testing
__global__ void rmsnorm_test_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ weight,
                                     int seq_len, int hidden_size, float eps) {
    int pos = blockIdx.x;
    int tid = threadIdx.x;
    if (pos >= seq_len) return;

    const float* in_ptr = input + pos * hidden_size;
    float* out_ptr = output + pos * hidden_size;

    __shared__ float s_sq_sum[256];

    float local_sq_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = in_ptr[i];
        local_sq_sum += val * val;
    }
    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(s_sq_sum[0] / hidden_size + eps);

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_ptr[i] = in_ptr[i] * inv_rms * weight[i];
    }
}

void test_rmsnorm_uniform_input() {
    printf("=== Test 1: RMSNorm with Uniform Input ===\n");

    const int hidden_size = 1024;
    const int seq_len = 8;
    float eps = 1e-6f;

    size_t input_size = seq_len * hidden_size * sizeof(float);
    size_t weight_size = hidden_size * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_weight = (float*)malloc(weight_size);
    float *h_output = (float*)malloc(input_size);

    for (int i = 0; i < seq_len * hidden_size; ++i) h_input[i] = 1.0f;
    for (int i = 0; i < hidden_size; ++i) h_weight[i] = 1.0f;

    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    CUDA_CHECK(cudaMalloc(&d_output, input_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(seq_len);
    rmsnorm_test_kernel<<<grid, block>>>(d_input, d_output, d_weight, seq_len, hidden_size, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost));

    float expected = 1.0f / sqrtf(1.0f + eps);
    bool passed = true;
    float max_diff = 0.0f;

    for (int i = 0; i < seq_len * hidden_size; ++i) {
        float diff = fabsf(h_output[i] - expected);
        if (diff > max_diff) max_diff = diff;
        if (diff > EPSILON) {
            passed = false;
            if (i < 5) {
                printf("  Mismatch at index %d: expected %.6f, got %.6f\n", i, expected, h_output[i]);
            }
        }
    }

    if (passed) {
        printf("  [PASSED] Uniform input produces expected output\n");
        printf("  Expected: %.6f, Max diff: %.6f\n", expected, max_diff);
    } else {
        printf("  [FAILED] Max diff: %.6f\n", max_diff);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input); free(h_weight); free(h_output);
}

void test_rmsnorm_random_input() {
    printf("=== Test 2: RMSNorm with Random Input ===\n");

    const int hidden_size = 1024;
    const int seq_len = 8;
    float eps = 1e-6f;

    size_t input_size = seq_len * hidden_size * sizeof(float);
    size_t weight_size = hidden_size * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_weight = (float*)malloc(weight_size);
    float *h_output = (float*)malloc(input_size);

    srand(42);
    for (int i = 0; i < seq_len * hidden_size; ++i) h_input[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < hidden_size; ++i) h_weight[i] = (float)(rand() % 100) / 100.0f;

    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    CUDA_CHECK(cudaMalloc(&d_output, input_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(seq_len);
    rmsnorm_test_kernel<<<grid, block>>>(d_input, d_output, d_weight, seq_len, hidden_size, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost));

    // Verify with CPU reference
    bool passed = true;
    float max_diff = 0.0f;

    for (int pos = 0; pos < seq_len; ++pos) {
        float sq_sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = h_input[pos * hidden_size + i];
            sq_sum += val * val;
        }
        float inv_rms = 1.0f / sqrtf(sq_sum / hidden_size + eps);

        for (int i = 0; i < hidden_size; ++i) {
            float expected = h_input[pos * hidden_size + i] * inv_rms * h_weight[i];
            float diff = fabsf(h_output[pos * hidden_size + i] - expected);
            if (diff > max_diff) max_diff = diff;
            if (diff > EPSILON) {
                passed = false;
            }
        }
    }

    if (passed) {
        printf("  [PASSED] Random input matches CPU reference\n");
        printf("  Max diff: %.6f\n", max_diff);
    } else {
        printf("  [FAILED] Max diff: %.6f\n", max_diff);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input); free(h_weight); free(h_output);
}

int main() {
    printf("=================================================\n");
    printf("  RMSNorm Unit Tests\n");
    printf("=================================================\n\n");

    test_rmsnorm_uniform_input();
    test_rmsnorm_random_input();

    printf("\n=================================================\n");
    printf("  All tests completed\n");
    printf("=================================================\n");

    return 0;
}
