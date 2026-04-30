#pragma once

#include "vision_patch_embedding.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qwen {

// Matches transformers `smart_resize` in qwen2_vl/image_processing_qwen2_vl.py
// factor is typically patch_size * merge_size (Qwen3.5-0.8B: 16*2=32).
std::pair<int, int> smart_resize_qwen_vl(int height, int width, int factor, std::int64_t min_pixels,
                                         std::int64_t max_pixels);

// Parse HF preprocessor_config.json (regex, no JSON library).
struct QwenVLPreprocessorConfig {
    std::int64_t shortest_edge = 65536;   // min_pixels (Qwen3.5-0.8B default)
    std::int64_t longest_edge = 16777216; // max_pixels
    int patch_size = 16;
    int temporal_patch_size = 2;
    int merge_size = 2;
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3] = {0.5f, 0.5f, 0.5f};
};

// Returns false if file missing or parse failed; caller keeps defaults.
bool load_preprocessor_config_json(const std::string& path, QwenVLPreprocessorConfig& out);

// Resize RGB uint8 HWC with stbir (sRGB path, bicubic-style).
void resize_rgb_uint8_bicubic(const unsigned char* src, int src_w, int src_h, unsigned char* dst,
                              int dst_w, int dst_h);

// rescale 1/255 then (x-mean)/std into [1,T,C,H,W]; duplicate frame along T when needed.
void build_tensor5d_qwen_vl_norm(const unsigned char* rgb_hwc, int H, int W,
                                 int temporal_patch_size, const float mean[3], const float stdv[3],
                                 Tensor5D& out);

} // namespace qwen
