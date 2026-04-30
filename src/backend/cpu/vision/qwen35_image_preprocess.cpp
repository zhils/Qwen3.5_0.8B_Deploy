#include "qwen35_image_preprocess.hpp"
#include "vision_patch_embedding.hpp"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace qwen {

std::pair<int, int> smart_resize_qwen_vl(int height, int width, int factor, std::int64_t min_pixels,
                                         std::int64_t max_pixels) {

    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("smart_resize: height/width must be > 0");
    }
    const double ar =
        static_cast<double>(std::max(height, width)) / static_cast<double>(std::min(height, width));
    if (ar > 200.0) {
        throw std::invalid_argument("smart_resize: aspect ratio must be <= 200");
    }

    long long h_bar =
        static_cast<long long>(std::llround(static_cast<double>(height) / factor)) * factor;
    long long w_bar =
        static_cast<long long>(std::llround(static_cast<double>(width) / factor)) * factor;

    const double h_w = static_cast<double>(height) * static_cast<double>(width);
    const long long area = h_bar * w_bar;

    if (area > max_pixels) {
        const double beta = std::sqrt(h_w / static_cast<double>(max_pixels));
        h_bar = std::max<long long>(
            factor, static_cast<long long>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max<long long>(
            factor, static_cast<long long>(std::floor(width / beta / factor)) * factor);
    } else if (area < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / h_w);
        h_bar = static_cast<long long>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<long long>(std::ceil(width * beta / factor)) * factor;
    }

    return {static_cast<int>(h_bar), static_cast<int>(w_bar)};
}

static bool parse_int64(const std::string& json, const char* key, std::int64_t& v) {
    std::string pat = std::string("\"") + key + "\"\\s*:\\s*([0-9]+)";
    std::smatch m;
    if (std::regex_search(json, m, std::regex(pat))) {
        v = std::stoll(m[1].str());
        return true;
    }
    return false;
}

static bool parse_int(const std::string& json, const char* key, int& v) {
    std::string pat = std::string("\"") + key + "\"\\s*:\\s*([0-9]+)";
    std::smatch m;
    if (std::regex_search(json, m, std::regex(pat))) {
        v = std::stoi(m[1].str());
        return true;
    }
    return false;
}

static bool parse_float_triplet(const std::string& json, const char* key, float out[3]) {
    std::string pat =
        std::string("\"") + key +
        "\"\\s*:\\s*\\[\\s*([0-9.+-eE]+)\\s*,\\s*([0-9.+-eE]+)\\s*,\\s*([0-9.+-eE]+)\\s*\\]";
    std::smatch m;
    if (std::regex_search(json, m, std::regex(pat))) {
        out[0] = std::stof(m[1].str());
        out[1] = std::stof(m[2].str());
        out[2] = std::stof(m[3].str());
        return true;
    }
    return false;
}

bool load_preprocessor_config_json(const std::string& path, QwenVLPreprocessorConfig& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string json = buf.str();

    std::int64_t se = out.shortest_edge, le = out.longest_edge;
    if (!parse_int64(json, "shortest_edge", se)) {
        // nested under "size"
        std::smatch m;
        if (std::regex_search(
                json, m,
                std::regex("\"size\"\\s*:\\s*\\{[^}]*\"shortest_edge\"\\s*:\\s*([0-9]+)"))) {
            se = std::stoll(m[1].str());
        } else {
            return false;
        }
    }
    if (!parse_int64(json, "longest_edge", le)) {
        std::smatch m;
        if (std::regex_search(
                json, m,
                std::regex("\"size\"\\s*:\\s*\\{[^}]*\"longest_edge\"\\s*:\\s*([0-9]+)"))) {
            le = std::stoll(m[1].str());
        }
    }

    out.shortest_edge = se;
    out.longest_edge = le;
    parse_int(json, "patch_size", out.patch_size);
    parse_int(json, "temporal_patch_size", out.temporal_patch_size);
    parse_int(json, "merge_size", out.merge_size);
    parse_float_triplet(json, "image_mean", out.image_mean);
    parse_float_triplet(json, "image_std", out.image_std);
    return true;
}

void resize_rgb_uint8_bicubic(const unsigned char* src, int src_w, int src_h, unsigned char* dst,
                              int dst_w, int dst_h) {

    stbir_resize_uint8_srgb(src, src_w, src_h, 0, dst, dst_w, dst_h, 0, STBIR_RGB);
}

void build_tensor5d_qwen_vl_norm(const unsigned char* rgb_hwc, int H, int W,
                                 int temporal_patch_size, const float mean[3], const float stdv[3],
                                 Tensor5D& out) {

    if (out.b != 1 || out.c != 3 || out.h != H || out.w != W || out.t % temporal_patch_size != 0) {
        throw std::invalid_argument("build_tensor5d_qwen_vl_norm: Tensor5D shape mismatch");
    }
    for (int t = 0; t < out.t; ++t) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int base = (y * W + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    float v = static_cast<float>(rgb_hwc[static_cast<size_t>(base + c)]) / 255.0f;
                    v = (v - mean[c]) / stdv[c];
                    out.at(0, t, c, y, x) = v;
                }
            }
        }
    }
}

} // namespace qwen
