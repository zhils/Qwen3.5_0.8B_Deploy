/**
 * Unified Error Report Generator
 *
 * This is the core program referenced by docs/accuracy_validation_methodology.md.
 * It performs both "isolated" and "sequential" layerwise validation against
 * PyTorch golden references, and outputs a structured CSV report.
 *
 * Output: build/unified_error_report.csv
 * Consumed by: scripts/validation/analyze_error_report.py
 *
 * Metrics per layer:
 *   - max_abs: maximum absolute error
 *   - mean_abs: mean absolute error
 *   - rel_l2: relative L2 error (||diff|| / ||ref||)
 *   - cosine: cosine similarity
 *
 * Additional checks:
 *   - pre_final_vs_hidden_24_corrected (same semantics)
 *   - post_final_norm_vs_hidden_24_layerwise (same semantics)
 *   - logits comparison + top1_match + top10_overlap
 */
#include "token_embedding.hpp"
#include "language_backbone.hpp"
#include "lm_head.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <string>

// ============================================================
// I/O helpers
// ============================================================

static std::vector<float> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return {};
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<float> d(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(d.data()), sz);
    return d;
}

static std::vector<int> load_token_ids(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return {};
    size_t sz = f.tellg();
    f.seekg(0);

    if (sz % 8 == 0) {
        std::vector<int64_t> d64(sz / sizeof(int64_t));
        f.read(reinterpret_cast<char*>(d64.data()), sz);
        bool valid = true;
        for (auto v : d64) {
            if (v < 0 || v > 0x7FFFFFFF) {
                valid = false;
                break;
            }
        }
        if (valid && !d64.empty()) {
            bool int64_layout = true;
            for (auto v : d64) {
                if ((v & 0xFFFFFFFF) != 0 && ((v >> 32) & 0xFFFFFFFF) != 0) {
                    int64_layout = false;
                    break;
                }
            }
            if (int64_layout) {
                std::vector<int> result;
                for (auto v : d64)
                    result.push_back(static_cast<int>(v));
                return result;
            }
        }
        f.clear();
        f.seekg(0);
    }

    std::vector<int> d(sz / sizeof(int));
    f.read(reinterpret_cast<char*>(d.data()), sz);
    return d;
}

// ============================================================
// Metric computation
// ============================================================

struct Metrics {
    double max_abs = 0, mean_abs = 0, rel_l2 = 0, cosine = 0;
    bool valid = false;
};

static Metrics compute(const std::vector<float>& ref, const std::vector<float>& test) {
    Metrics m;
    if (ref.empty() || test.empty())
        return m;
    size_t n = std::min(ref.size(), test.size());
    double sum_abs = 0, diff_sq = 0, ref_sq = 0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        double a = ref[i], b = test[i], d = std::abs(a - b);
        m.max_abs = std::max(m.max_abs, d);
        sum_abs += d;
        diff_sq += (a - b) * (a - b);
        ref_sq += a * a;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    m.mean_abs = sum_abs / n;
    m.rel_l2 = (ref_sq > 1e-30) ? std::sqrt(diff_sq / ref_sq) : 0;
    double denom = std::sqrt(na) * std::sqrt(nb);
    m.cosine = (denom > 1e-30) ? (dot / denom) : 0;
    m.valid = true;
    return m;
}

static void write_csv_row(std::ofstream& csv, const std::string& mode, const std::string& label,
                          const std::string& type, const Metrics& m) {
    csv << mode << "," << label << "," << type << "," << std::scientific << std::setprecision(6)
        << m.max_abs << "," << m.mean_abs << "," << m.rel_l2 << "," << std::fixed
        << std::setprecision(8) << m.cosine << "\n";
}

static void print_row(const std::string& mode, const std::string& label, const std::string& type,
                      const Metrics& m) {
    printf("  %-12s %-6s %-8s  max=%-12.6e mean=%-12.6e rel_l2=%-12.6e cos=%-10.8f\n", mode.c_str(),
           label.c_str(), type.c_str(), m.max_abs, m.mean_abs, m.rel_l2, m.cosine);
}

// ============================================================
// Load all golden references
// ============================================================

struct GoldenData {
    std::vector<float> embedding;
    std::vector<std::vector<float>> hidden; // hidden[0..24]
    std::vector<float> logits;
    int predicted_token = -1;

    std::vector<float> hidden_24_corrected;
};

static GoldenData load_golden(const std::string& lw_dir, const std::string& corr_dir) {
    GoldenData g;
    g.embedding = load_bin(corr_dir + "/embedding_output.bin");

    for (int i = 0; i <= 24; ++i) {
        std::string p = corr_dir + "/hidden_" + std::to_string(i) + ".bin";
        auto d = load_bin(p);
        if (d.empty() && i <= 24) {
            p = lw_dir + "/hidden_" + std::to_string(i) + ".bin";
            d = load_bin(p);
        }
        g.hidden.push_back(d);
    }

    g.logits = load_bin(lw_dir + "/logits.bin");

    auto pt = load_bin(lw_dir + "/predicted_token.bin");
    if (!pt.empty()) {
        g.predicted_token = static_cast<int>(pt[0]);
    }

    g.hidden_24_corrected = load_bin(corr_dir + "/hidden_24.bin");

    return g;
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    const int H = 1024, NL = 24, V = 248320;
    std::string weights_dir = "weights";
    std::string lw_dir = "validation_data_layerwise";
    std::string corr_dir = "validation_data_corrected";
    std::string csv_path = "unified_error_report.csv";

    if (argc > 1)
        weights_dir = argv[1];
    if (argc > 2)
        lw_dir = argv[2];
    if (argc > 3)
        corr_dir = argv[3];
    if (argc > 4)
        csv_path = argv[4];

    std::cout << "============================================================" << std::endl;
    std::cout << "  Unified Error Report Generator" << std::endl;
    std::cout << "  (docs/accuracy_validation_methodology.md)" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    auto golden = load_golden(lw_dir, corr_dir);
    auto input_ids = load_token_ids(corr_dir + "/input_ids.bin");
    if (input_ids.empty()) {
        std::cerr << "No input_ids found. Run export scripts first." << std::endl;
        return 1;
    }

    // Load model
    qwen::TokenEmbedding emb(V, H);
    emb.set_weights(load_bin(weights_dir + "/language/embed_tokens.bin"));

    qwen::LanguageBackbone backbone(NL, H, 3584);
    for (int i = 0; i < NL; ++i) {
        std::string pfx = weights_dir + "/language_backbone/layer_" + std::to_string(i);
        qwen::LanguageLayerWeights lw;
        lw.input_layernorm_weight = load_bin(pfx + "/input_layernorm.bin");
        lw.post_attention_layernorm_weight = load_bin(pfx + "/post_layernorm.bin");
        lw.mlp_gate_proj_weight = load_bin(pfx + "/mlp_gate.bin");
        lw.mlp_up_proj_weight = load_bin(pfx + "/mlp_up.bin");
        lw.mlp_down_proj_weight = load_bin(pfx + "/mlp_down.bin");
        bool is_full = ((i % 4) == 3);
        if (is_full) {
            lw.is_linear = false;
            lw.full_q_proj_weight = load_bin(pfx + "/full_q.bin");
            lw.full_k_proj_weight = load_bin(pfx + "/full_k.bin");
            lw.full_v_proj_weight = load_bin(pfx + "/full_v.bin");
            lw.full_o_proj_weight = load_bin(pfx + "/full_o.bin");
            lw.full_q_norm_weight = load_bin(pfx + "/full_q_norm.bin");
            lw.full_k_norm_weight = load_bin(pfx + "/full_k_norm.bin");
        } else {
            lw.is_linear = true;
            lw.linear_in_proj_qkv_weight = load_bin(pfx + "/linear_qkv.bin");
            lw.linear_in_proj_a_weight = load_bin(pfx + "/linear_a.bin");
            lw.linear_in_proj_b_weight = load_bin(pfx + "/linear_b.bin");
            lw.linear_in_proj_z_weight = load_bin(pfx + "/linear_z.bin");
            lw.linear_conv1d_weight = load_bin(pfx + "/linear_conv1d.bin");
            lw.linear_A_log = load_bin(pfx + "/linear_A_log.bin");
            lw.linear_dt_bias = load_bin(pfx + "/linear_dt_bias.bin");
            lw.linear_norm_weight = load_bin(pfx + "/linear_norm.bin");
            lw.linear_out_proj_weight = load_bin(pfx + "/linear_out.bin");
        }
        backbone.set_layer_weights(i, lw);
    }
    backbone.set_final_norm_weight(load_bin(weights_dir + "/language/final_norm.bin"));

    qwen::LMHead lmhead(H, V);
    lmhead.set_weight(load_bin(weights_dir + "/language/embed_tokens.bin"));

    std::ofstream csv(csv_path);
    csv << "mode,layer,type,max_abs,mean_abs,rel_l2,cosine\n";

    // --- Embedding ---
    std::cout << "--- Embedding ---" << std::endl;
    auto cpp_embed = emb.forward(input_ids);
    auto em = compute(golden.embedding, cpp_embed);
    print_row("embed", "embed", "-", em);
    write_csv_row(csv, "embed", "embed", "-", em);

    // --- Sequential layerwise ---
    std::cout << "\n--- Sequential (chained C++ output) ---" << std::endl;
    std::vector<qwen::LinearAttnState> seq_ls(NL);
    for (auto& s : seq_ls)
        s.reset(16, 128, 128, 4);
    qwen::KVCache seq_kv;
    seq_kv.reset(NL, 2, 256, 4096);

    std::vector<float> seq_h = cpp_embed;
    for (int i = 0; i < NL; ++i) {
        seq_h = backbone.layers()[i]->forward(seq_h, seq_ls[i], seq_kv, i);
        std::string type = ((i % 4) == 3) ? "Full" : "Linear";
        if (i + 1 < static_cast<int>(golden.hidden.size()) && !golden.hidden[i + 1].empty()) {
            auto m = compute(golden.hidden[i + 1], seq_h);
            print_row("sequential", std::to_string(i), type, m);
            write_csv_row(csv, "sequential", std::to_string(i), type, m);
        }
    }

    // --- Isolated layerwise ---
    std::cout << "\n--- Isolated (golden input -> single layer) ---" << std::endl;
    for (int i = 0; i < NL; ++i) {
        if (i >= static_cast<int>(golden.hidden.size()) ||
            i + 1 >= static_cast<int>(golden.hidden.size()) || golden.hidden[i].empty() ||
            golden.hidden[i + 1].empty())
            continue;

        std::vector<qwen::LinearAttnState> iso_ls(NL);
        for (auto& s : iso_ls)
            s.reset(16, 128, 128, 4);
        qwen::KVCache iso_kv;
        iso_kv.reset(NL, 2, 256, 4096);

        auto iso_out = backbone.layers()[i]->forward(golden.hidden[i], iso_ls[i], iso_kv, i);
        std::string type = ((i % 4) == 3) ? "Full" : "Linear";
        auto m = compute(golden.hidden[i + 1], iso_out);
        print_row("isolated", std::to_string(i), type, m);
        write_csv_row(csv, "isolated", std::to_string(i), type, m);
    }

    // --- End-of-backbone semantic checks ---
    std::cout << "\n--- Final output checks ---" << std::endl;

    // pre_final_vs_hidden_24_corrected (pre-final-norm C++ vs corrected golden)
    if (!golden.hidden_24_corrected.empty()) {
        auto m = compute(golden.hidden_24_corrected, seq_h);
        print_row("pre_final", "vs_corr", "-", m);
        write_csv_row(csv, "pre_final_vs_hidden_24_corrected", "24", "-", m);
    }

    // post_final_norm_vs_hidden_24_layerwise
    auto post_norm = backbone.final_norm_forward(seq_h);
    if (golden.hidden.size() > 24 && !golden.hidden[24].empty()) {
        auto m = compute(golden.hidden[24], post_norm);
        print_row("post_final", "vs_lw24", "-", m);
        write_csv_row(csv, "post_final_norm_vs_hidden_24_layerwise", "24", "-", m);
    }

    // --- Logits ---
    if (!golden.logits.empty()) {
        auto cpp_logits = lmhead.forward(post_norm);
        auto m = compute(golden.logits, cpp_logits);
        print_row("logits", "logits", "-", m);
        write_csv_row(csv, "logits", "logits", "-", m);

        auto argmax = [](const std::vector<float>& v) -> int {
            return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
        };
        int cpp_t1 = argmax(cpp_logits);
        int py_t1 = argmax(golden.logits);
        bool match = (cpp_t1 == py_t1);
        printf("  Top1: C++=%d PyTorch=%d Match=%s\n", cpp_t1, py_t1, match ? "YES" : "NO");
        csv << "top1," << cpp_t1 << ",ref=" << py_t1 << ",,," << (match ? 1.0 : 0.0) << "\n";

        // Top10 overlap
        auto top_k = [](const std::vector<float>& v, int k) {
            std::vector<int> idx(v.size());
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = static_cast<int>(i);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return v[a] > v[b]; });
            idx.resize(k);
            std::sort(idx.begin(), idx.end());
            return idx;
        };
        auto ct10 = top_k(cpp_logits, 10);
        auto pt10 = top_k(golden.logits, 10);
        int overlap = 0;
        size_t ci = 0, pi = 0;
        while (ci < ct10.size() && pi < pt10.size()) {
            if (ct10[ci] == pt10[pi]) {
                overlap++;
                ci++;
                pi++;
            } else if (ct10[ci] < pt10[pi])
                ci++;
            else
                pi++;
        }
        printf("  Top10 overlap: %d/10\n", overlap);
        csv << "top10_overlap," << overlap << ",10,,,,\n";
    }

    csv.close();
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  CSV written: " << csv_path << std::endl;
    std::cout << "  Run: python scripts/validation/analyze_error_report.py" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    return 0;
}
