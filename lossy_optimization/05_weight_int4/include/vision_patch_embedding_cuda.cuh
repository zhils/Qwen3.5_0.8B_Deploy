#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 与 CPU VisionPatchEmbedding::forward 一致的 GPU 前向。
 * 所有设备指针由调用方分配；布局与 Tensor5D / 权重向量行主序一致。
 */
cudaError_t vision_patch_embed_cuda_launch(const float* d_input, const float* d_weight,
                                           const float* d_bias, float* d_output, int B, int T_in,
                                           int C, int H, int W, int temporal_patch, int patch_size,
                                           int nh, int nw, int nt, int N, int embed_dim,
                                           bool use_bias, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
