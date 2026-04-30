// Vision Patch Embedding (CUDA). Matches src/vision/vision_patch_embedding.cpp layout.
// One thread per output scalar out[batch, token, od].

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

__device__ __forceinline__ size_t dev_weight_index(int od, int ic, int kt, int ky, int kx,
                                                   int in_channels, int temporal_patch,
                                                   int patch_size) {
    const size_t c1 = static_cast<size_t>(in_channels);
    const size_t t1 = static_cast<size_t>(temporal_patch);
    const size_t p1 = static_cast<size_t>(patch_size);
    const size_t s0 = static_cast<size_t>(od) * c1 + static_cast<size_t>(ic);
    const size_t s1 = s0 * t1 + static_cast<size_t>(kt);
    const size_t s2 = s1 * p1 + static_cast<size_t>(ky);
    const size_t s3 = s2 * p1 + static_cast<size_t>(kx);
    return s3;
}

__device__ __forceinline__ size_t dev_input_index(int bi, int ti, int ci, int yi, int xi, int T,
                                                  int C, int H, int W) {
    const size_t t = static_cast<size_t>(T);
    const size_t c = static_cast<size_t>(C);
    const size_t h = static_cast<size_t>(H);
    const size_t w = static_cast<size_t>(W);
    const size_t s0 = static_cast<size_t>(bi) * t + static_cast<size_t>(ti);
    const size_t s1 = s0 * c + static_cast<size_t>(ci);
    const size_t s2 = s1 * h + static_cast<size_t>(yi);
    const size_t s3 = s2 * w + static_cast<size_t>(xi);
    return s3;
}

__global__ void
vision_patch_embed_forward_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                  const float* __restrict__ bias, float* __restrict__ output, int B,
                                  int T_in, int C, int H, int W, int temporal_patch, int patch_size,
                                  int nh, int nw, int /*nt*/, int N, int embed_dim, int use_bias) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                        static_cast<int64_t>(threadIdx.x);
    const int64_t total =
        static_cast<int64_t>(B) * static_cast<int64_t>(N) * static_cast<int64_t>(embed_dim);
    if (idx >= total) {
        return;
    }

    const int od = static_cast<int>(idx % static_cast<int64_t>(embed_dim));
    const int64_t tmp = idx / static_cast<int64_t>(embed_dim);
    const int token_idx = static_cast<int>(tmp % static_cast<int64_t>(N));
    const int bi = static_cast<int>(tmp / static_cast<int64_t>(N));

    const int patch_plane = nh * nw;
    const int tt = token_idx / patch_plane;
    const int rem = token_idx % patch_plane;
    const int yy = rem / nw;
    const int xx = rem % nw;

    float acc = 0.0f;
    if (use_bias) {
        acc = bias[od];
    }
    for (int ic = 0; ic < C; ++ic) {
        for (int kt = 0; kt < temporal_patch; ++kt) {
            const int t_inner = tt * temporal_patch + kt;
            for (int ky = 0; ky < patch_size; ++ky) {
                const int y_in = yy * patch_size + ky;
                for (int kx = 0; kx < patch_size; ++kx) {
                    const int x_in = xx * patch_size + kx;
                    const size_t xi = dev_input_index(bi, t_inner, ic, y_in, x_in, T_in, C, H, W);
                    const size_t wi =
                        dev_weight_index(od, ic, kt, ky, kx, C, temporal_patch, patch_size);
                    acc += input[xi] * weight[wi];
                }
            }
        }
    }

    const size_t out_i =
        (static_cast<size_t>(bi) * static_cast<size_t>(N) + static_cast<size_t>(token_idx)) *
            static_cast<size_t>(embed_dim) +
        static_cast<size_t>(od);
    output[out_i] = acc;
}

} // namespace

extern "C" cudaError_t vision_patch_embed_cuda_launch(const float* d_input, const float* d_weight,
                                                      const float* d_bias, float* d_output, int B,
                                                      int T_in, int C, int H, int W,
                                                      int temporal_patch, int patch_size, int nh,
                                                      int nw, int nt, int N, int embed_dim,
                                                      bool use_bias, cudaStream_t stream) {
    (void)nt;
    const int64_t total =
        static_cast<int64_t>(B) * static_cast<int64_t>(N) * static_cast<int64_t>(embed_dim);
    if (total <= 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    const int64_t blocks64 =
        (total + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
    const int blocks =
        static_cast<int>(blocks64 > static_cast<int64_t>(2147483647) ? 2147483647 : blocks64);

    vision_patch_embed_forward_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_weight, use_bias ? d_bias : nullptr, d_output, B, T_in, C, H, W, temporal_patch,
        patch_size, nh, nw, nt, N, embed_dim, use_bias ? 1 : 0);

    return cudaGetLastError();
}
