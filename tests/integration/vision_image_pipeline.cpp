// JPEG -> Qwen3.5-aligned preprocess (smart_resize + CLIP-style rescale/norm) -> VisionPatchEmbedding -> ViT -> Merger
// Requires: third_party/stb_image.h (decode only); resize via qwen_core qwen35_image_preprocess (stbir)
// Weights: scripts/weights/export_vit_weights.py -> weights/vit/*.bin
// Optional: preprocessor_config.json from HF repo (Qwen/Qwen3.5-0.8B) for exact min/max pixels.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "qwen35_image_preprocess.hpp"
#include "vision_patch_embedding.hpp"
#include "vision_transformer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::vector<float> load_binary_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    file.seekg(0, std::ios::end);
    const std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(file_size / sizeof(float)));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

void print_stats(const std::string& name, const std::vector<float>& v) {
    if (v.empty()) {
        std::cout << "  [" << name << "] empty\n";
        return;
    }
    float mn = v[0], mx = v[0], s = 0.0f;
    for (float x : v) {
        mn = std::min(mn, x);
        mx = std::max(mx, x);
        s += x;
    }
    std::cout << "  [" << name << "] n=" << v.size() << " min=" << mn << " max=" << mx
              << " mean=" << (s / static_cast<float>(v.size())) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const fs::path image_path = (argc >= 2)
                                    ? fs::path(argv[1])
                                    : fs::path(R"(D:\deploy\c++deploy\data\images\cat_dog.jpg)");
    const fs::path weights_vit = (argc >= 3) ? fs::path(argv[2]) : fs::path("../weights/vit");
    const fs::path preprocessor_json =
        (argc >= 4) ? fs::path(argv[3]) : (weights_vit / "preprocessor_config.json");

    constexpr int vit_hidden = 768;
    constexpr int vit_heads = 12;
    constexpr int vit_mlp = 3072;
    constexpr int vit_depth = 12;
    constexpr int merger_out = 1024;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "======== Vision image pipeline (C++, Qwen3.5-style preprocess) ========\n";
    std::cout << "Image: " << image_path.string() << "\n";
    std::cout << "Weights: " << weights_vit.string() << "\n";
    std::cout << "Preprocessor JSON (optional): " << preprocessor_json.string() << "\n\n";

    qwen::QwenVLPreprocessorConfig pcfg;
    if (!qwen::load_preprocessor_config_json(preprocessor_json.string(), pcfg)) {
        std::cout << "Using built-in Qwen3.5-0.8B defaults (see HF preprocessor_config.json).\n";
    } else {
        std::cout << "Loaded preprocessor config: min_pixels=" << pcfg.shortest_edge
                  << " max_pixels=" << pcfg.longest_edge << " patch=" << pcfg.patch_size
                  << " temporal=" << pcfg.temporal_patch_size << " merge=" << pcfg.merge_size
                  << "\n";
    }

    const int factor = pcfg.patch_size * pcfg.merge_size;
    const int kT = pcfg.temporal_patch_size;

    try {
        int iw = 0, ih = 0, ic = 0;
        unsigned char* pixels = stbi_load(image_path.string().c_str(), &iw, &ih, &ic, 3);
        if (!pixels) {
            std::cerr << "stbi_load failed: " << image_path.string() << "\n";
            return 1;
        }
        std::cout << "[1] Decoded JPEG: " << iw << "x" << ih << " (RGB)\n";

        auto [rh, rw] =
            qwen::smart_resize_qwen_vl(ih, iw, factor, pcfg.shortest_edge, pcfg.longest_edge);
        std::cout << "[2] smart_resize (factor=" << factor << "): target " << rw << "x" << rh
                  << " (H x W)\n";

        std::vector<unsigned char> resized(static_cast<size_t>(rw * rh * 3));
        qwen::resize_rgb_uint8_bicubic(pixels, iw, ih, resized.data(), rw, rh);
        stbi_image_free(pixels);

        const int grid_h = rh / pcfg.patch_size;
        const int grid_w = rw / pcfg.patch_size;
        std::cout << "[3] Bicubic resize + patch grid: " << grid_h << "x" << grid_w
                  << " (patch_size=" << pcfg.patch_size << ")\n";

        qwen::Tensor5D input(1, kT, 3, rh, rw);
        qwen::build_tensor5d_qwen_vl_norm(resized.data(), rh, rw, kT, pcfg.image_mean,
                                          pcfg.image_std, input);

        std::cout << "[4] Tensor5D [1," << kT << ",3," << rh << "," << rw
                  << "] rescale 1/255 + (x-mean)/std (Qwen3.5: mean=std=0.5)\n";
        print_stats(
            "input_sample",
            std::vector<float>(input.data.begin(),
                               input.data.begin() + std::min<size_t>(512, input.data.size())));

        auto patch_w = load_binary_file(weights_vit / "patch_embed_proj_weight.bin");
        auto patch_b = load_binary_file(weights_vit / "patch_embed_proj_bias.bin");
        qwen::VisionPatchEmbedding patch_embed(vit_hidden, 3, pcfg.patch_size,
                                               pcfg.temporal_patch_size, true);
        patch_embed.set_weights(std::move(patch_w), std::move(patch_b));

        qwen::Tensor3D after_patch = patch_embed.forward(input);
        const int n_tokens = after_patch.n;
        std::cout << "[5] PatchEmbedding -> Tensor3D [1, N=" << n_tokens << ", D=" << after_patch.d
                  << "]\n";
        print_stats("after_patch", after_patch.data);

        qwen::Tensor2D vit_in(n_tokens, vit_hidden);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < vit_hidden; ++j) {
                vit_in.at(i, j) = after_patch.at(0, i, j);
            }
        }

        auto pos_full = load_binary_file(weights_vit / "pos_embed.bin");
        if (pos_full.size() < static_cast<size_t>(n_tokens) * vit_hidden) {
            std::ostringstream oss;
            oss << "pos_embed.bin too small: need " << (static_cast<size_t>(n_tokens) * vit_hidden)
                << " floats, got " << pos_full.size();
            throw std::runtime_error(oss.str());
        }
        pos_full.resize(static_cast<size_t>(n_tokens) * vit_hidden);

        qwen::VisionTransformer vit(vit_hidden, vit_heads, vit_mlp, vit_depth);
        vit.set_pos_embed(std::move(pos_full));

        for (int bi = 0; bi < vit_depth; ++bi) {
            const fs::path bd = weights_vit / ("block_" + std::to_string(bi));
            vit.set_block_weights(
                bi, load_binary_file(bd / "norm1_weight.bin"),
                load_binary_file(bd / "norm1_bias.bin"), load_binary_file(bd / "qkv_weight.bin"),
                load_binary_file(bd / "qkv_bias.bin"), load_binary_file(bd / "proj_weight.bin"),
                load_binary_file(bd / "proj_bias.bin"), load_binary_file(bd / "norm2_weight.bin"),
                load_binary_file(bd / "norm2_bias.bin"), load_binary_file(bd / "fc1_weight.bin"),
                load_binary_file(bd / "fc1_bias.bin"), load_binary_file(bd / "fc2_weight.bin"),
                load_binary_file(bd / "fc2_bias.bin"));
        }

        qwen::Tensor2D after_vit = vit.forward(vit_in);
        std::cout << "[6] VisionTransformer -> [N=" << after_vit.n << ", D=" << after_vit.d
                  << "]\n";
        print_stats("after_vit", after_vit.data);

        const fs::path md = weights_vit / "merger";
        qwen::VisualMerger merger(vit_hidden, merger_out, vit_mlp, pcfg.merge_size);
        merger.set_weights(
            load_binary_file(md / "norm_weight.bin"), load_binary_file(md / "norm_bias.bin"),
            load_binary_file(md / "fc1_weight.bin"), load_binary_file(md / "fc1_bias.bin"),
            load_binary_file(md / "fc2_weight.bin"), load_binary_file(md / "fc2_bias.bin"));

        qwen::Tensor2D after_merger = merger.forward(after_vit, grid_h, grid_w);
        std::cout << "[7] VisualMerger -> [N=" << after_merger.n << ", D=" << after_merger.d
                  << "]\n";
        print_stats("after_merger", after_merger.data);

        std::cout << "\n======== Summary ========\n";
        std::cout << "Vision token dim: " << after_merger.d << "\n";
        std::cout << "Num vision tokens: " << after_merger.n << "\n";
        std::cout << "Total floats: " << after_merger.data.size() << "\n";
        std::cout << "First token [0:8]: ";
        for (int i = 0; i < 8 && i < after_merger.d; ++i) {
            std::cout << after_merger.at(0, i) << (i < 7 ? ", " : "");
        }
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "\nEnsure weights/vit/ has patch_embed and ViT bins; copy "
                     "preprocessor_config.json from HF.\n";
        std::cerr << "Python reference: scripts/preprocess/export_qwen35_image_processor_ref.py\n";
        return 1;
    }
}
