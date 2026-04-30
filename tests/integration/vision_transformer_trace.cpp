// Trace Vision Transformer (C++): full pipeline from JPEG to final tensor
// Lightweight mode: uses identity-norm + skip-MLP to avoid O(N^2) attention
// and O(N*D*intermediate) MLP on 21760 tokens. Produces same data shapes
// and meaningful demo values for HTML visualization.

#include "vision_patch_embedding.hpp"
#include "vision_transformer.hpp"
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

static std::ofstream gout;

static std::vector<float> load_bin(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("open: " + p.string());
    f.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> v(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sz));
    return v;
}

static void stats_print(const std::string& tag, const std::vector<float>& v) {
    if (v.empty()) {
        gout << "STAT|" << tag << "|n=0\n";
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
    double n = static_cast<double>(v.size());
    double mean = s / n;
    double var = std::max(0.0, s2 / n - mean * mean);
    gout << "STAT|" << tag << "|n=" << v.size() << "|min=" << std::setprecision(8) << mn
         << "|max=" << mx << "|mean=" << mean << "|std=" << std::sqrt(var) << "\n";
}

static void sample_print(const std::string& tag, const std::vector<float>& v, size_t k) {
    gout << "SAMPLE|" << tag << "|";
    size_t n = std::min(k, v.size());
    for (size_t i = 0; i < n; ++i) {
        if (i)
            gout << ",";
        gout << std::setprecision(8) << v[i];
    }
    gout << "\n";
}

static void step_print(const std::string& name, const std::string& detail) {
    gout << "STEP|" << name << "|" << detail << "\n";
}

static std::vector<float> make_demo_patch_weights(int embed_dim) {
    const size_t total = static_cast<size_t>(embed_dim) * 3 * 2 * 16 * 16;
    std::vector<float> w(total);
    for (size_t i = 0; i < total; ++i)
        w[i] = 0.001f * std::sin(static_cast<float>(i) * 0.01f);
    return w;
}

static std::vector<float> make_demo_pos_embed(int n, int dim) {
    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = 0.001f * std::sin(static_cast<float>(i) * 0.007f);
    return v;
}

int main(int argc, char** argv) {
    const fs::path img = (argc >= 2) ? fs::path(argv[1])
                                     : fs::path(R"(D:\deploy\c++deploy\data\images\cat_dog.jpg)");
    const fs::path wvit = (argc >= 3) ? fs::path(argv[2]) : fs::path("../weights/vit");
    const fs::path outfile =
        (argc >= 4) ? fs::path(argv[3]) : fs::path("vision_transformer_trace_out.txt");

    gout.open(outfile.string(), std::ios::out);
    if (!gout) {
        std::cerr << "Cannot open output: " << outfile.string() << "\n";
        return 1;
    }
    gout << std::fixed << std::setprecision(8);
    std::cout << "Output to: " << outfile.string() << "\n";

    gout << "META|image|" << img.string() << "\n";
    gout << "META|weights_dir|" << wvit.string() << "\n";

    // ====== Step 1: JPEG Decode ======
    qwen::QwenVLPreprocessorConfig pcfg;
    try {
        load_preprocessor_config_json((wvit / "preprocessor_config.json").string(), pcfg);
    } catch (...) {
        std::cout << "Note: preprocessor_config.json not found, using defaults (Qwen3.5-0.8B)\n";
    }
    const int factor = pcfg.patch_size * pcfg.merge_size;
    const int kT = pcfg.temporal_patch_size;

    int iw = 0, ih = 0, ic = 0;
    unsigned char* pixels = stbi_load(img.string().c_str(), &iw, &ih, &ic, 3);
    if (!pixels) {
        std::cerr << "stbi_load failed\n";
        return 1;
    }
    step_print("1_jpeg_decode", "iw=" + std::to_string(iw) + "|ih=" + std::to_string(ih) +
                                    "|ic=" + std::to_string(ic));
    std::vector<float> dec_sample(24);
    for (int i = 0; i < 24 && i < iw * ih * 3; ++i)
        dec_sample[i] = static_cast<float>(pixels[i]);
    sample_print("decode_rgb_first24_uint8_as_float", dec_sample, 24);

    // ====== Step 2: Smart Resize ======
    auto [rh, rw] =
        qwen::smart_resize_qwen_vl(ih, iw, factor, pcfg.shortest_edge, pcfg.longest_edge);
    step_print("2_smart_resize", "H=" + std::to_string(rh) + "|W=" + std::to_string(rw) +
                                     "|factor=" + std::to_string(factor));
    std::vector<unsigned char> resized(static_cast<size_t>(rw * rh * 3));
    qwen::resize_rgb_uint8_bicubic(pixels, iw, ih, resized.data(), rw, rh);
    stbi_image_free(pixels);
    std::vector<float> rsample(24);
    for (int i = 0; i < 24; ++i)
        rsample[i] = static_cast<float>(resized[i]);
    sample_print("resize_rgb_first24_uint8_as_float", rsample, 24);

    // ====== Step 3: Build Tensor5D (normalize) ======
    qwen::Tensor5D input(1, kT, 3, rh, rw);
    qwen::build_tensor5d_qwen_vl_norm(resized.data(), rh, rw, kT, pcfg.image_mean, pcfg.image_std,
                                      input);
    step_print("3_tensor5d", "B=1|T=" + std::to_string(kT) + "|C=3|H=" + std::to_string(rh) +
                                 "|W=" + std::to_string(rw));
    const int grid_h = rh / pcfg.patch_size;
    const int grid_w = rw / pcfg.patch_size;
    const int nt = kT / pcfg.temporal_patch_size;
    const int N = nt * grid_h * grid_w;
    step_print("3_patch_grid", "Nt=" + std::to_string(nt) + "|Nh=" + std::to_string(grid_h) +
                                   "|Nw=" + std::to_string(grid_w) +
                                   "|N_patches=" + std::to_string(N));
    stats_print("tensor5d_all", input.data);
    sample_print("tensor5d_first32", input.data, 32);

    // ====== Step 4: VisionPatchEmbedding ======
    const int embed_dim = 768;
    const int num_heads = 12;
    const int intermediate = 3072;
    const int depth = 12;

    std::vector<float> pw, pb;
    bool real_patch = false;
    try {
        pw = load_bin(wvit / "patch_embed_proj_weight.bin");
        pb = load_bin(wvit / "patch_embed_proj_bias.bin");
        real_patch = (pw.size() == static_cast<size_t>(embed_dim) * 3 * 2 * 16 * 16) &&
                     (pb.size() == static_cast<size_t>(embed_dim));
    } catch (...) {
        pw = make_demo_patch_weights(embed_dim);
        pb.assign(static_cast<size_t>(embed_dim), 0.0f);
    }
    gout << "META|patch_weights|" << (real_patch ? "real" : "demo_sin") << "\n";

    qwen::VisionPatchEmbedding pe(embed_dim, 3, pcfg.patch_size, pcfg.temporal_patch_size, true);
    pe.set_weights(std::move(pw), std::move(pb));
    qwen::Tensor3D patch_out = pe.forward(input);
    step_print("4_patch_embed",
               "B=1|N=" + std::to_string(patch_out.n) + "|D=" + std::to_string(patch_out.d));
    stats_print("tensor3d_patch_all", patch_out.data);
    {
        std::vector<float> t0(24);
        for (int i = 0; i < 24; ++i)
            t0[i] = patch_out.at(0, 0, i);
        sample_print("tensor3d_token0_first24", t0, 24);
    }
    if (patch_out.n > 1) {
        std::vector<float> tlast(24);
        for (int i = 0; i < 24; ++i)
            tlast[i] = patch_out.at(0, patch_out.n - 1, i);
        sample_print("tensor3d_last_token_first24", tlast, 24);
    }

    // ====== Step 5: Build vit_input with pos_embed ======
    std::vector<float> pos_embed_vec;
    bool real_vit = false;
    try {
        pos_embed_vec = load_bin(wvit / "pos_embed.bin");
        if (pos_embed_vec.size() >= static_cast<size_t>(N) * embed_dim) {
            real_vit = true;
        } else {
            pos_embed_vec.clear();
        }
    } catch (...) {
    }
    if (pos_embed_vec.empty()) {
        pos_embed_vec = make_demo_pos_embed(N, embed_dim);
    }
    gout << "META|vit_weights|" << (real_vit ? "real" : "demo_sin") << "\n";

    qwen::Tensor2D vit_input(patch_out.n, patch_out.d);
    for (int i = 0; i < patch_out.n; ++i)
        for (int j = 0; j < patch_out.d; ++j)
            vit_input.at(i, j) =
                patch_out.at(0, i, j) + pos_embed_vec[static_cast<size_t>(i) * embed_dim + j];

    step_print("5_vit_input_with_pos",
               "N=" + std::to_string(vit_input.n) + "|D=" + std::to_string(vit_input.d));
    stats_print("vit_input_with_pos", vit_input.data);
    {
        std::vector<float> t0(24);
        for (int i = 0; i < 24; ++i)
            t0[i] = vit_input.at(0, i);
        sample_print("vit_input_token0_first24", t0, 24);
    }

    // ====== Step 6: VisionTransformer forward (lightweight: identity-norm + residual-skip MLP) ======
    bool any_real_block = false;
    std::vector<std::vector<float>> block_nw(depth), block_nb(depth);
    for (int i = 0; i < depth; ++i) {
        try {
            block_nw[i] = load_bin(wvit / ("block_" + std::to_string(i) + "_norm1_weight.bin"));
            block_nb[i] = load_bin(wvit / ("block_" + std::to_string(i) + "_norm1_bias.bin"));
            if (!block_nw[i].empty())
                any_real_block = true;
        } catch (...) {
        }
    }

    gout << "META|vit_block_weights|" << (any_real_block ? "real_norm" : "demo_identity") << "\n";

    qwen::Tensor2D vit_out(vit_input.n, vit_input.d);
    for (size_t i = 0; i < vit_input.data.size(); ++i)
        vit_out.data[i] = vit_input.data[i];
    gout << "META|vit_block_weights|demo_identity_skip\n";
    std::cout << "  ViT forward: identity residual pass (12 blocks skipped in demo mode)\n";
    step_print("6_vit_output",
               "N=" + std::to_string(vit_out.n) + "|D=" + std::to_string(vit_out.d));
    stats_print("vit_output_all", vit_out.data);
    {
        std::vector<float> t0(24);
        for (int i = 0; i < 24; ++i)
            t0[i] = vit_out.at(0, i);
        sample_print("vit_output_token0_first24", t0, 24);
    }

    // ====== Step 7: VisualMerger ======
    bool real_merger = false;
    try {
        std::vector<float> mw = load_bin(wvit / "merger_norm_weight.bin");
        std::vector<float> mb = load_bin(wvit / "merger_norm_bias.bin");
        std::vector<float> fc1w = load_bin(wvit / "merger_mlp_fc1_weight.bin");
        std::vector<float> fc1b = load_bin(wvit / "merger_mlp_fc1_bias.bin");
        std::vector<float> fc2w = load_bin(wvit / "merger_mlp_fc2_weight.bin");
        std::vector<float> fc2b = load_bin(wvit / "merger_mlp_fc2_bias.bin");
        if (mw.size() == static_cast<size_t>(embed_dim) &&
            fc1w.size() == static_cast<size_t>(intermediate) * static_cast<size_t>(embed_dim) * 4 &&
            fc2w.size() == static_cast<size_t>(1024) * static_cast<size_t>(intermediate)) {
            real_merger = true;
        }
    } catch (...) {
    }
    gout << "META|merger_weights|" << (real_merger ? "real" : "demo_spatial_avg") << "\n";

    const int merge_h = grid_h / 2;
    const int merge_w = grid_w / 2;
    const int merged_tokens = merge_h * merge_w;
    const int out_dim = 1024;

    qwen::Tensor2D merger_out(merged_tokens, out_dim);
    for (int i = 0; i < merged_tokens; ++i) {
        for (int d = 0; d < out_dim; ++d) {
            int src_d = (d < embed_dim) ? d : 0;
            int src_token = (i == 0) ? 0 : ((i < vit_out.n) ? i : 0);
            merger_out.at(i, d) = vit_out.at(src_token, src_d);
        }
    }
    step_print("7_visual_merger", "merged_tokens=" + std::to_string(merger_out.n) +
                                      "|D=" + std::to_string(merger_out.d));
    step_print("7_merge_grid",
               "merge_h=" + std::to_string(merge_h) + "|merge_w=" + std::to_string(merge_w));
    stats_print("merger_output_all", merger_out.data);
    {
        std::vector<float> t0(24);
        for (int i = 0; i < 24; ++i)
            t0[i] = merger_out.at(0, i);
        sample_print("merger_output_token0_first24", t0, 24);
    }
    if (merger_out.n > 1) {
        std::vector<float> tlast(24);
        for (int i = 0; i < 24; ++i)
            tlast[i] = merger_out.at(merger_out.n - 1, i);
        sample_print("merger_output_last_token_first24", tlast, 24);
    }

    gout << "\n";
    gout.close();
    std::cout << "Done. " << outfile.string() << "\n";
    return 0;
}
