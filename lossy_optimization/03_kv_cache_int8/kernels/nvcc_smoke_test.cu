#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_one(float* x) {
    int i = threadIdx.x;
    if (i < 1)
        x[i] += 1.0f;
}

int main() {
    float h = 1.0f;
    float* d = nullptr;
    cudaMalloc(&d, sizeof(float));
    cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice);
    add_one<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    std::printf("smoke=%0.1f\n", h);
    return (h > 1.5f) ? 0 : 1;
}
