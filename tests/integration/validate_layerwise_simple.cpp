/**
 * Layerwise validation: C++ vs PyTorch golden reference.
 *
 * For each layer computes 4 metrics + pass/fail:
 *   - max_abs: maximum absolute error
 *   - mean_abs: mean absolute error
 *   - rel_l2: relative L2 error  (||diff|| / ||ref||)
 *   - cosine: cosine similarity
 *
 * Supports both "isolated" (golden input -> single layer -> compare golden output)
 * and "sequential" (chained C++ output -> compare golden output) modes.
 *
 * Exit code: 0 if all layers pass thresholds, 1 otherwise.
 * Outputs CSV to build/layerwise_error_report.csv for downstream analysis.
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

// ============================================================
// File I/O
// ============================================================

static std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static std::vector<int> load_token_ids(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);

    // Auto-detect int32 vs int64: PyTorch default saves as int64
    if (size % 8 == 0 && size % 4 == 0) {
        // Try int64 first: check if high 32 bits are zero (valid for token IDs < 2^31)
        std::vector<int64_t> data64(size / sizeof(int64_t));
        file.read(reinterpret_cast<char*>(data64.data()), size);
        bool looks_like_int64 = true;
        for (auto v : data64) {
            if (v < 0 || v > 0x7FFFFFFF) {
                looks_like_int64 = false;
                break;
            }
        }
        if (looks_like_int64 && !data64.empty()) {
            // Verify: if reading as int32, would we get zeros in odd positions?
            bool has_zero_padding = true;
            for (size_t i = 0; i < data64.size(); ++i) {
                int32_t low = static_cast<int32_t>(data64[i] & 0xFFFFFFFF);
                int32_t high = static_cast<int32_t>((data64[i] >> 32) & 0xFFFFFFFF);
                if (low != 0 && high != 0) {
                    has_zero_padding = false;
                    break;
                }
            }
            if (has_zero_padding) {
                std::vector<int> result;
                result.reserve(data64.size());
                for (auto v : data64)
                    result.push_back(static_cast<int>(v));
                std::cout << "  (loaded " << result.size() << " token IDs as int64)" << std::endl;
                return result;
            }
        }
        // Fall through to int32
        file.clear();
        file.seekg(0);
    }

    std::vector<int> data(size / sizeof(int));
    file.read(reinterpret_cast<char*>(data.data()), size);
    std::cout << "  (loaded " << data.size() << " token IDs as int32)" << std::endl;
    return data;
}

// ============================================================
// Metric computation
// ============================================================

struct ErrorMetrics {
    double max_abs = 0;
    double mean_abs = 0;
    double rel_l2 = 0;
    double cosine = 0;
    bool valid = false;
};

static ErrorMetrics compute_metrics(const std::vector<float>& ref, const std::vector<float>& test) {
    ErrorMetrics m;
    if (ref.empty() || test.empty())
        return m;
    size_t n = std::min(ref.size(), test.size());

    double sum_abs = 0, sum_diff_sq = 0, sum_ref_sq = 0;
    double dot = 0, norm_a = 0, norm_b = 0;

    for (size_t i = 0; i < n; ++i) {
        double a = ref[i], b = test[i];
        double d = std::abs(a - b);
        m.max_abs = std::max(m.max_abs, d);
        sum_abs += d;
        sum_diff_sq += (a - b) * (a - b);
        sum_ref_sq += a * a;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    m.mean_abs = sum_abs / n;
    m.rel_l2 = (sum_ref_sq > 1e-30) ? std::sqrt(sum_diff_sq / sum_ref_sq) : 0;
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    m.cosine = (denom > 1e-30) ? (dot / denom) : 0;
    m.valid = true;
    return m;
}

// Pass/fail thresholds
static const double THRESH_MAX_ABS = 0.1; // generous for float32 24-layer chain
static const double THRESH_COSINE = 0.99;

static bool is_pass(const ErrorMetrics& m) {
    return m.valid && m.max_abs < THRESH_MAX_ABS && m.cosine > THRESH_COSINE;
}

static void print_header() {
    printf("%-8s %-8s | %-12s %-12s %-12s %-10s | %s\n", "Layer", "Type", "max_abs", "mean_abs",
           "rel_l2", "cosine", "Status");
    std::cout << std::string(90, '-') << std::endl;
}

static void print_row(const std::string& label, const std::string& type, const ErrorMetrics& m) {
    const char* status = !m.valid ? "SKIP" : (is_pass(m) ? "PASS" : "FAIL");
    printf("%-8s %-8s | %-12.6e %-12.6e %-12.6e %-10.6f | %s\n", label.c_str(), type.c_str(),
           m.max_abs, m.mean_abs, m.rel_l2, m.cosine, status);
}

static std::string make_path(const std::string& dir, const std::string& name, int idx) {
    std::ostringstream oss;
    oss << dir << "/" << name << idx << ".bin";
    return oss.str();
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    const int H = 1024, NL = 24, V = 248320;
    std::string val_dir = "validation_data_corrected";
    std::string weights_dir = "weights";
    std::string csv_path = "layerwise_error_report.csv";

    if (argc > 1)
        val_dir = argv[1];
    if (argc > 2)
        weights_dir = argv[2];

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Layerwise Validation: C++ vs PyTorch Golden Reference" << std::endl;
    std::cout << "  Metrics: max_abs, mean_abs, rel_l2, cosine" << std::endl;
    std::cout << "  Thresholds: max_abs<" << THRESH_MAX_ABS << ", cosine>" << THRESH_COSINE
              << std::endl;
    std::cout << "============================================================\n" << std::endl;
    std::cout << "Validation data: " << val_dir << std::endl;
    std::cout << "Weights:         " << weights_dir << "\n" << std::endl;

    auto input_ids = load_token_ids(val_dir + "/input_ids.bin");
    if (input_ids.empty()) {
        std::cerr << "No input_ids found. Run export_corrected_hidden.py first." << std::endl;
        return 1;
    }
    std::cout << "Input IDs: " << input_ids.size() << " tokens" << std::endl;

    // Load model
    qwen::TokenEmbedding embedding(V, H);
    embedding.set_weights(load_binary_file(weights_dir + "/language/embed_tokens.bin"));

    qwen::LanguageBackbone backbone(NL, H, 3584);
    for (int i = 0; i < NL; ++i) {
        std::string prefix = weights_dir + "/language_backbone/layer_" + std::to_string(i);
        qwen::LanguageLayerWeights lw;
        lw.input_layernorm_weight = load_binary_file(prefix + "/input_layernorm.bin");
        lw.post_attention_layernorm_weight = load_binary_file(prefix + "/post_layernorm.bin");
        lw.mlp_gate_proj_weight = load_binary_file(prefix + "/mlp_gate.bin");
        lw.mlp_up_proj_weight = load_binary_file(prefix + "/mlp_up.bin");
        lw.mlp_down_proj_weight = load_binary_file(prefix + "/mlp_down.bin");
        bool is_full = ((i % 4) == 3);
        if (is_full) {
            lw.is_linear = false;
            lw.full_q_proj_weight = load_binary_file(prefix + "/full_q.bin");
            lw.full_k_proj_weight = load_binary_file(prefix + "/full_k.bin");
            lw.full_v_proj_weight = load_binary_file(prefix + "/full_v.bin");
            lw.full_o_proj_weight = load_binary_file(prefix + "/full_o.bin");
            lw.full_q_norm_weight = load_binary_file(prefix + "/full_q_norm.bin");
            lw.full_k_norm_weight = load_binary_file(prefix + "/full_k_norm.bin");
        } else {
            lw.is_linear = true;
            lw.linear_in_proj_qkv_weight = load_binary_file(prefix + "/linear_qkv.bin");
            lw.linear_in_proj_a_weight = load_binary_file(prefix + "/linear_a.bin");
            lw.linear_in_proj_b_weight = load_binary_file(prefix + "/linear_b.bin");
            lw.linear_in_proj_z_weight = load_binary_file(prefix + "/linear_z.bin");
            lw.linear_conv1d_weight = load_binary_file(prefix + "/linear_conv1d.bin");
            lw.linear_A_log = load_binary_file(prefix + "/linear_A_log.bin");
            lw.linear_dt_bias = load_binary_file(prefix + "/linear_dt_bias.bin");
            lw.linear_norm_weight = load_binary_file(prefix + "/linear_norm.bin");
            lw.linear_out_proj_weight = load_binary_file(prefix + "/linear_out.bin");
        }
        backbone.set_layer_weights(i, lw);
    }
    backbone.set_final_norm_weight(load_binary_file(weights_dir + "/language/final_norm.bin"));

    // CSV output
    std::ofstream csv(csv_path);
    csv << "stage,layer,type,max_abs,mean_abs,rel_l2,cosine,pass\n";

    int total_checks = 0, total_pass = 0;

    // --- Embedding check ---
    auto cpp_embed = embedding.forward(input_ids);
    auto py_embed = load_binary_file(val_dir + "/embedding_output.bin");

    std::cout << "\n=== Embedding ===" << std::endl;
    print_header();
    auto em = compute_metrics(py_embed, cpp_embed);
    print_row("Embed", "-", em);
    csv << "sequential,embed,-," << em.max_abs << "," << em.mean_abs << "," << em.rel_l2 << ","
        << em.cosine << "," << (is_pass(em) ? 1 : 0) << "\n";
    total_checks++;
    if (is_pass(em))
        total_pass++;

    // --- Sequential layerwise ---
    std::cout << "\n=== Sequential Layerwise (chained C++ output vs golden) ===" << std::endl;
    print_header();

    std::vector<qwen::LinearAttnState> seq_lin_states(NL);
    for (auto& s : seq_lin_states)
        s.reset(16, 128, 128, 4);
    qwen::KVCache seq_kv;
    seq_kv.reset(NL, 2, 256, 4096);

    std::vector<float> seq_hidden = cpp_embed;
    int seq_position = 0; // single-token input at position 0
    for (int i = 0; i < NL; ++i) {
        auto py_ref = load_binary_file(make_path(val_dir, "hidden_", i + 1));
        seq_hidden =
            backbone.layers()[i]->forward(seq_hidden, seq_lin_states[i], seq_kv, seq_position);

        std::string type = ((i % 4) == 3) ? "Full" : "Linear";
        auto m = compute_metrics(py_ref, seq_hidden);
        print_row("L" + std::to_string(i), type, m);
        csv << "sequential," << i << "," << type << "," << m.max_abs << "," << m.mean_abs << ","
            << m.rel_l2 << "," << m.cosine << "," << (is_pass(m) ? 1 : 0) << "\n";
        total_checks++;
        if (is_pass(m))
            total_pass++;
    }

    // --- Isolated layerwise (golden input -> single layer -> compare golden output) ---
    std::cout
        << "\n=== Isolated Layerwise (golden input -> single layer -> compare golden output) ==="
        << std::endl;
    print_header();

    for (int i = 0; i < NL; ++i) {
        auto golden_input = load_binary_file(make_path(val_dir, "hidden_", i));
        auto golden_output = load_binary_file(make_path(val_dir, "hidden_", i + 1));

        if (golden_input.empty() || golden_output.empty()) {
            print_row("L" + std::to_string(i), "?", ErrorMetrics{});
            continue;
        }

        std::vector<qwen::LinearAttnState> iso_lin(NL);
        for (auto& s : iso_lin)
            s.reset(16, 128, 128, 4);
        qwen::KVCache iso_kv;
        iso_kv.reset(NL, 2, 256, 4096);

        auto iso_out = backbone.layers()[i]->forward(golden_input, iso_lin[i], iso_kv, 0);

        std::string type = ((i % 4) == 3) ? "Full" : "Linear";
        auto m = compute_metrics(golden_output, iso_out);
        print_row("L" + std::to_string(i), type, m);
        csv << "isolated," << i << "," << type << "," << m.max_abs << "," << m.mean_abs << ","
            << m.rel_l2 << "," << m.cosine << "," << (is_pass(m) ? 1 : 0) << "\n";
        total_checks++;
        if (is_pass(m))
            total_pass++;
    }

    // --- Logits / Top1 check ---
    auto py_logits = load_binary_file(val_dir + "/../validation_data_layerwise/logits.bin");
    if (!py_logits.empty() && !seq_hidden.empty()) {
        std::cout << "\n=== Logits & Top1 ===" << std::endl;
        print_header();

        auto final_normed = backbone.final_norm_forward(seq_hidden);
        qwen::LMHead lmhead(H, V);
        lmhead.set_weight(load_binary_file(weights_dir + "/language/embed_tokens.bin"));
        auto cpp_logits = lmhead.forward(final_normed);

        auto lm = compute_metrics(py_logits, cpp_logits);
        print_row("Logits", "-", lm);
        csv << "logits,logits,-," << lm.max_abs << "," << lm.mean_abs << "," << lm.rel_l2 << ","
            << lm.cosine << "," << (is_pass(lm) ? 1 : 0) << "\n";
        total_checks++;
        if (is_pass(lm))
            total_pass++;

        // Top1
        auto argmax = [](const std::vector<float>& v) -> int {
            return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
        };
        int cpp_top1 = argmax(cpp_logits);
        int py_top1 = argmax(py_logits);
        bool top1_match = (cpp_top1 == py_top1);
        printf("  Top1: C++=%d  PyTorch=%d  Match=%s\n", cpp_top1, py_top1,
               top1_match ? "YES" : "NO");
        csv << "top1," << cpp_top1 << "," << py_top1 << ",,,," << (top1_match ? 1 : 0) << "\n";

        // Top10 overlap
        auto top_k_indices = [](const std::vector<float>& v, int k) -> std::vector<int> {
            std::vector<int> idx(v.size());
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = static_cast<int>(i);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return v[a] > v[b]; });
            idx.resize(k);
            std::sort(idx.begin(), idx.end());
            return idx;
        };
        auto cpp_top10 = top_k_indices(cpp_logits, 10);
        auto py_top10 = top_k_indices(py_logits, 10);
        int overlap = 0;
        size_t ci = 0, pi = 0;
        while (ci < cpp_top10.size() && pi < py_top10.size()) {
            if (cpp_top10[ci] == py_top10[pi]) {
                overlap++;
                ci++;
                pi++;
            } else if (cpp_top10[ci] < py_top10[pi])
                ci++;
            else
                pi++;
        }
        printf("  Top10 overlap: %d/10\n", overlap);
        csv << "top10_overlap," << overlap << ",10,,,," << (overlap >= 8 ? 1 : 0) << "\n";
    }

    csv.close();
    std::cout << "\nCSV report: " << csv_path << std::endl;

    // --- Summary ---
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Summary: " << total_pass << "/" << total_checks << " checks passed"
              << std::endl;
    if (total_pass == total_checks) {
        std::cout << "  OVERALL: PASS" << std::endl;
    } else {
        std::cout << "  OVERALL: FAIL (" << (total_checks - total_pass) << " checks failed)"
                  << std::endl;
    }
    std::cout << "============================================================\n" << std::endl;

    return (total_pass == total_checks) ? 0 : 1;
}
