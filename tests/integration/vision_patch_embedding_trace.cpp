// Trace Vision Patch Embedding (C++): preprocess + VisionPatchEmbedding::forward
// Prints machine-readable blocks for embedding in docs HTML.
// Usage: vision_patch_embedding_trace [image.jpg] [weights/vit or empty]
// If patch_embed_proj_*.bin missing, uses deterministic demo weights (documented).

#include "vision_patch_embedding.hpp"
#include "qwen35_image_preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<float> load_bin(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        throw std::runtime_error("open: " + p.string());
    }
    f.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> v(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sz));
    return v;
}

static void stats_print(const std::string& tag, const std::vector<float>& v) {
    if (v.empty()) {
        std::cout << "STAT|" << tag << "|n=0\n";
        return;
    }
    double s = 0, s2 = 0;
    float mn = v[0], mx = v[0];
    for (float x : v) {
        mn = std::min(mn, x);
        mx = std::max(mx, x);
        s += x;
        s2 += static_cast<double>(x) * x;
    }
    const double n = static_cast<double>(v.size());
    const double mean = s / n;
    const double var = std::max(0.0, s2 / n - mean * mean);
    const double stdv = std::sqrt(var);
    std::cout << "STAT|" << tag << "|n=" << v.size() << "|min=" << std::setprecision(8) << mn
              << "|max=" << mx << "|mean=" << mean << "|std=" << stdv << "\n";
}

static void sample_print(const std::string& tag, const std::vector<float>& v, size_t k) {
    std::cout << "SAMPLE|" << tag << "|";
    const size_t n = std::min(k, v.size());
    for (size_t i = 0; i < n; ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << std::setprecision(8) << v[i];
    }
    std::cout << "\n";
}

static std::vector<float> make_demo_patch_weights(int embed_dim) {
    const size_t total = static_cast<size_t>(embed_dim) * 3 * 2 * 16 * 16;
    std::vector<float> w(total);
    for (size_t i = 0; i < total; ++i) {
        w[i] = 0.001f * std::sin(static_cast<float>(i) * 0.01f);
    }
    return w;
}

int main(int argc, char** argv) {
    const fs::path img = (argc >= 2) ? fs::path(argv[1])
                                     : fs::path(R"(D:\deploy\c++deploy\data\images\cat_dog.jpg)");
    const fs::path wvit = (argc >= 3) ? fs::path(argv[2]) : fs::path("../weights/vit");

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "META|image|" << img.string() << "\n";
    std::cout << "META|weights_dir|" << wvit.string() << "\n";

    qwen::QwenVLPreprocessorConfig pcfg;
    qwen::load_preprocessor_config_json((wvit / "preprocessor_config.json").string(), pcfg);
    const int factor = pcfg.patch_size * pcfg.merge_size;
    const int kT = pcfg.temporal_patch_size;

    int iw = 0, ih = 0, ic = 0;
    unsigned char* pixels = stbi_load(img.string().c_str(), &iw, &ih, &ic, 3);
    if (!pixels) {
        std::cerr << "stbi_load failed\n";
        return 1;
    }
    std::cout << "STEP|1_decode|iw=" << iw << "|ih=" << ih << "|ic=" << ic << "\n";
    std::vector<float> dec_sample(24);
    for (int i = 0; i < 24 && i < iw * ih * 3; ++i) {
        dec_sample[static_cast<size_t>(i)] = static_cast<float>(pixels[i]);
    }
    sample_print("decode_rgb_first24_uint8_as_float", dec_sample, 24);

    auto [rh, rw] =
        qwen::smart_resize_qwen_vl(ih, iw, factor, pcfg.shortest_edge, pcfg.longest_edge);
    std::cout << "STEP|2_smart_resize|H=" << rh << "|W=" << rw << "|factor=" << factor << "\n";

    std::vector<unsigned char> resized(static_cast<size_t>(rw * rh * 3));
    qwen::resize_rgb_uint8_bicubic(pixels, iw, ih, resized.data(), rw, rh);
    stbi_image_free(pixels);

    std::vector<float> rsample(24);
    for (int i = 0; i < 24; ++i) {
        rsample[static_cast<size_t>(i)] = static_cast<float>(resized[static_cast<size_t>(i)]);
    }
    sample_print("resize_rgb_first24_uint8_as_float", rsample, 24);

    qwen::Tensor5D input(1, kT, 3, rh, rw);
    qwen::build_tensor5d_qwen_vl_norm(resized.data(), rh, rw, kT, pcfg.image_mean, pcfg.image_std,
                                      input);
    stats_print("tensor5d_all", input.data);
    sample_print("tensor5d_first32", input.data, 32);

    const int grid_h = rh / pcfg.patch_size;
    const int grid_w = rw / pcfg.patch_size;
    const int nt = kT / pcfg.temporal_patch_size;
    const int N = nt * grid_h * grid_w;
    std::cout << "STEP|3_tensor5d_shape|B=1|T=" << kT << "|C=3|H=" << rh << "|W=" << rw << "\n";
    std::cout << "STEP|3_grid|Nt=" << nt << "|Nh=" << grid_h << "|Nw=" << grid_w
              << "|N_patches=" << N << "\n";

    const int embed_dim = 768;
    std::vector<float> pw;
    std::vector<float> pb;
    bool real_w = false;
    try {
        pw = load_bin(wvit / "patch_embed_proj_weight.bin");
        pb = load_bin(wvit / "patch_embed_proj_bias.bin");
        real_w = (pw.size() == static_cast<size_t>(embed_dim) * 3 * 2 * 16 * 16) &&
                 (pb.size() == static_cast<size_t>(embed_dim));
    } catch (...) {
        pw = make_demo_patch_weights(embed_dim);
        pb.assign(static_cast<size_t>(embed_dim), 0.0f);
    }
    std::cout << "META|patch_weights|" << (real_w ? "real" : "demo_sin") << "\n";

    qwen::VisionPatchEmbedding pe(embed_dim, 3, pcfg.patch_size, pcfg.temporal_patch_size, true);
    pe.set_weights(std::move(pw), std::move(pb));

    qwen::Tensor3D out = pe.forward(input);
    std::cout << "STEP|4_patch_embed_out|B=1|N=" << out.n << "|D=" << out.d << "\n";
    stats_print("tensor3d_patch_all", out.data);
    {
        const size_t take = std::min(static_cast<size_t>(24), static_cast<size_t>(out.d));
        std::vector<float> t0(take);
        for (size_t i = 0; i < take; ++i) {
            t0[i] = out.at(0, 0, static_cast<int>(i));
        }
        sample_print("tensor3d_token0_first24", t0, take);
    }
    if (out.n > 1 && out.d > 0) {
        const size_t take = std::min(static_cast<size_t>(24), static_cast<size_t>(out.d));
        std::vector<float> last(take);
        for (size_t i = 0; i < take; ++i) {
            last[i] = out.at(0, out.n - 1, static_cast<int>(i));
        }
        sample_print("tensor3d_last_token_first24", last, take);
    }

    return 0;
}
