#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <ostream>
#include <fstream>

namespace qwen {
namespace cuda {
namespace tools {

enum class DeviceLocation {
    CPU,
    CUDA,
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE
};

enum class OpOptimization {
    NONE,
    FUSED_KERNEL,
    CUDA_GRAPH,
    CUBLAS_GEMM,
    FLASH_ATTENTION,
    TENSOR_CORE,
    WARP_LEVEL,
    SHARED_MEMORY,
    REGISTER_CACHE
};

struct TensorShape {
    int batch;
    int seq_len;
    int channels;
    int head_dim;
    int num_heads;
    int num_kv_heads;

    std::string to_string() const {
        if (num_heads > 0 && head_dim > 0) {
            return "B=" + std::to_string(batch) +
                   ", S=" + std::to_string(seq_len) +
                   ", H=" + std::to_string(num_heads) +
                   ", D=" + std::to_string(head_dim);
        }
        return "B=" + std::to_string(batch) +
               ", S=" + std::to_string(seq_len) +
               ", C=" + std::to_string(channels);
    }

    size_t total_elements() const {
        if (num_heads > 0 && head_dim > 0) {
            return static_cast<size_t>(batch) * num_heads * head_dim;
        }
        return static_cast<size_t>(batch) * channels;
    }

    std::string memory_size() const {
        size_t bytes = total_elements() * sizeof(float);
        if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + " KB";
        }
        return std::to_string(bytes) + " B";
    }

    static TensorShape make_1d(int batch, int channels) {
        return {batch, 1, channels, 0, 0, 0};
    }

    static TensorShape make_2d(int batch, int seq_len, int channels) {
        return {batch, seq_len, channels, 0, 0, 0};
    }

    static TensorShape make_attn(int batch, int seq_len, int num_heads, int head_dim) {
        return {batch, seq_len, 0, head_dim, num_heads, 0};
    }

    static TensorShape make_attn_gqa(int batch, int seq_len, int num_heads,
                                     int num_kv_heads, int head_dim) {
        return {batch, seq_len, 0, head_dim, num_heads, num_kv_heads};
    }
};

struct OpEdge {
    std::string from_op;
    std::string to_op;
    std::string tensor_name;
    TensorShape shape;
    DeviceLocation location;
};

class OpNode {
public:
    std::string name;
    std::string kernel_name;
    std::string description;
    DeviceLocation location;
    OpOptimization optimization;
    std::vector<TensorShape> inputs;
    std::vector<TensorShape> outputs;
    std::vector<std::string> notes;
    float estimated_flops;
    float estimated_time_ms;

    OpNode(const std::string& name, const std::string& kernel_name,
           DeviceLocation loc, OpOptimization opt = OpOptimization::NONE)
        : name(name), kernel_name(kernel_name), location(loc),
          optimization(opt), estimated_flops(0), estimated_time_ms(0) {}

    OpNode& add_input(const TensorShape& shape, const std::string& name = "") {
        inputs.push_back(shape);
        input_names.push_back(name);
        return *this;
    }

    OpNode& add_output(const TensorShape& shape, const std::string& name = "") {
        outputs.push_back(shape);
        output_names.push_back(name);
        return *this;
    }

    OpNode& add_note(const std::string& note) {
        notes.push_back(note);
        return *this;
    }

    OpNode& set_description(const std::string& desc) {
        description = desc;
        return *this;
    }

    OpNode& set_estimation(float flops, float time_ms) {
        estimated_flops = flops;
        estimated_time_ms = time_ms;
        return *this;
    }

    std::string location_string() const {
        switch (location) {
            case DeviceLocation::CPU: return "CPU";
            case DeviceLocation::CUDA: return "GPU";
            case DeviceLocation::HOST_TO_DEVICE: return "H2D";
            case DeviceLocation::DEVICE_TO_HOST: return "D2H";
            case DeviceLocation::DEVICE_TO_DEVICE: return "D2D";
        }
        return "Unknown";
    }

    std::string optimization_string() const {
        switch (optimization) {
            case OpOptimization::NONE: return "";
            case OpOptimization::FUSED_KERNEL: return "Fused";
            case OpOptimization::CUDA_GRAPH: return "Graph";
            case OpOptimization::CUBLAS_GEMM: return "cuBLAS";
            case OpOptimization::FLASH_ATTENTION: return "FlashAttn";
            case OpOptimization::TENSOR_CORE: return "TensorCore";
            case OpOptimization::WARP_LEVEL: return "WarpLevel";
            case OpOptimization::SHARED_MEMORY: return "SharedMem";
            case OpOptimization::REGISTER_CACHE: return "RegCache";
        }
        return "";
    }

    std::string shape_summary() const {
        if (inputs.empty() && outputs.empty()) return "";
        std::string s = "IN:[";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) s += ",";
            s += inputs[i].to_string();
        }
        s += "] OUT:[";
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0) s += ",";
            s += outputs[i].to_string();
        }
        s += "]";
        return s;
    }

private:
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

class CudaOpFlowchart {
public:
    std::string model_name;
    std::string device_info;
    std::vector<OpNode> nodes;
    std::vector<OpEdge> edges;
    std::unordered_map<std::string, int> node_index;

    CudaOpFlowchart(const std::string& name = "CUDA Flow",
                    const std::string& device = "RTX 5060 Ti")
        : model_name(name), device_info(device) {}

    CudaOpFlowchart& add_node(const OpNode& node) {
        node_index[node.name] = nodes.size();
        nodes.push_back(node);
        return *this;
    }

    CudaOpFlowchart& add_edge(const std::string& from, const std::string& to,
                              const std::string& tensor_name,
                              const TensorShape& shape,
                              DeviceLocation loc = DeviceLocation::CUDA) {
        edges.push_back({from, to, tensor_name, shape, loc});
        return *this;
    }

    CudaOpFlowchart& add_data_transfer(const std::string& from, const std::string& to,
                                       const std::string& tensor_name,
                                       const TensorShape& shape,
                                       DeviceLocation direction) {
        edges.push_back({from, to, tensor_name, shape, direction});
        return *this;
    }

    std::string to_mermaid() const {
        std::string s = "flowchart TB\n";

        for (const auto& node : nodes) {
            std::string color = get_location_color(node.location);
            std::string opt = node.optimization_string();
            std::string opt_badge = opt.empty() ? "" : " **[" + opt + "]**";
            std::string desc = node.description.empty() ? "" : "\n*" + node.description + "*";
            std::string shape_info = node.shape_summary();
            std::string time_info = node.estimated_time_ms > 0
                ? "\n`" + std::to_string(static_cast<int>(node.estimated_time_ms)) + " ms`"
                : "";

            s += "    " + sanitize_id(node.name) + "((\"" + node.name + "\\n" +
                 node.kernel_name + opt_badge + desc + shape_info + time_info + "\"))\n";
            s += "    style " + sanitize_id(node.name) + " fill:" + color + "\n";
        }

        for (const auto& edge : edges) {
            std::string arrow = get_edge_style(edge.location);
            std::string edge_label = edge.tensor_name + "\\n" + edge.shape.memory_size();
            s += "    " + sanitize_id(edge.from_op) + arrow + "|" + edge_label + "|" +
                 sanitize_id(edge.to_op) + "\n";
        }

        s += "\n    subgraph legend[\"**图例**\"]\n";
        s += "        leg_cpu((\"CPU\")) style leg_cpu fill:#f8d7da,stroke:#dc3545\n";
        s += "        leg_gpu((\"GPU\")) style leg_gpu fill:#d1e7dd,stroke:#198754\n";
        s += "        leg_h2d{{\"H2D\"}} style leg_h2d fill:#fff3cd,stroke:#ffc107\n";
        s += "        leg_d2h{{\"D2H\"}} style leg_d2h fill:#cfe2ff,stroke:#0d6efd\n";
        s += "    end\n";

        return s;
    }

    std::string to_ascii() const {
        std::vector<std::string> lines;
        lines.push_back("+==============================================================================+");
        lines.push_back("|  " + model_name + " - CUDA Operator Flowchart");
        lines.push_back("|  Device: " + device_info);
        lines.push_back("+==============================================================================+");
        lines.push_back("");

        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            std::string loc_str = "[" + node.location_string() + "]";
            std::string opt_str = node.optimization_string();
            if (!opt_str.empty()) opt_str = " <" + opt_str + ">";

            lines.push_back("  +-------------------------------------------------------------------------+");
            lines.push_back("  | Node #" + std::to_string(i + 1) + ": " + node.name);
            lines.push_back("  | Kernel: " + node.kernel_name + opt_str);
            lines.push_back("  | Location: " + loc_str);
            if (!node.description.empty()) {
                lines.push_back("  | Desc: " + node.description);
            }

            if (!node.inputs.empty()) {
                lines.push_back("  | Inputs:");
                for (size_t j = 0; j < node.inputs.size(); ++j) {
                    lines.push_back("  |   - " + node.inputs[j].to_string() +
                                   " (" + node.inputs[j].memory_size() + ")");
                }
            }

            if (!node.outputs.empty()) {
                lines.push_back("  | Outputs:");
                for (size_t j = 0; j < node.outputs.size(); ++j) {
                    lines.push_back("  |   - " + node.outputs[j].to_string() +
                                   " (" + node.outputs[j].memory_size() + ")");
                }
            }

            if (!node.notes.empty()) {
                lines.push_back("  | Notes:");
                for (const auto& note : node.notes) {
                    lines.push_back("  |   * " + note);
                }
            }

            if (node.estimated_time_ms > 0) {
                lines.push_back("  | Est. Time: " + std::to_string(node.estimated_time_ms) + " ms");
            }

            lines.push_back("  +-------------------------------------------------------------------------+");
            lines.push_back("");

            if (i < edges.size()) {
                const auto& edge = edges[i];
                std::string arrow = edge.location == DeviceLocation::CUDA ? "-->" :
                                   edge.location == DeviceLocation::HOST_TO_DEVICE ? "-H2D->" :
                                   edge.location == DeviceLocation::DEVICE_TO_HOST ? "-D2H->" : "-->";
                lines.push_back("          " + arrow + " [" + edge.tensor_name + ": " +
                               edge.shape.memory_size() + "]");
                lines.push_back("");
            }
        }

        lines.push_back("+==============================================================================+");
        lines.push_back("  Legend:");
        lines.push_back("  [CPU] = CPU computation, [GPU] = CUDA kernel, [H2D] = Host to Device,");
        lines.push_back("  [D2H] = Device to Host, Fused = Multiple ops merged into one kernel");
        lines.push_back("+==============================================================================+");

        std::string result;
        for (const auto& line : lines) {
            result += line + "\n";
        }
        return result;
    }

    std::string to_json() const {
        std::string s = "{\n";
        s += "  \"model\": \"" + model_name + "\",\n";
        s += "  \"device\": \"" + device_info + "\",\n";
        s += "  \"nodes\": [\n";

        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            s += "    {\n";
            s += "      \"id\": \"" + node.name + "\",\n";
            s += "      \"kernel\": \"" + node.kernel_name + "\",\n";
            s += "      \"location\": \"" + node.location_string() + "\",\n";
            s += "      \"optimization\": \"" + node.optimization_string() + "\",\n";
            s += "      \"description\": \"" + node.description + "\",\n";

            s += "      \"inputs\": [";
            for (size_t j = 0; j < node.inputs.size(); ++j) {
                if (j > 0) s += ", ";
                s += "\"" + node.inputs[j].to_string() + "\"";
            }
            s += "],\n";

            s += "      \"outputs\": [";
            for (size_t j = 0; j < node.outputs.size(); ++j) {
                if (j > 0) s += ", ";
                s += "\"" + node.outputs[j].to_string() + "\"";
            }
            s += "]\n";

            s += "    }";
            if (i < nodes.size() - 1) s += ",";
            s += "\n";
        }

        s += "  ],\n";
        s += "  \"edges\": [\n";

        for (size_t i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            s += "    {\n";
            s += "      \"from\": \"" + edge.from_op + "\",\n";
            s += "      \"to\": \"" + edge.to_op + "\",\n";
            s += "      \"tensor\": \"" + edge.tensor_name + "\",\n";
            s += "      \"shape\": \"" + edge.shape.to_string() + "\",\n";
            std::string loc_str = (edge.location == DeviceLocation::CUDA ? "GPU" :
                                   edge.location == DeviceLocation::HOST_TO_DEVICE ? "H2D" :
                                   edge.location == DeviceLocation::DEVICE_TO_HOST ? "D2H" : "D2D");
            s += "      \"location\": \"" + loc_str + "\"\n";
            s += "    }";
            if (i < edges.size() - 1) s += ",";
            s += "\n";
        }

        s += "  ]\n";
        s += "}\n";

        return s;
    }

    void print_mermaid(std::ostream& os) const {
        os << to_mermaid();
    }

    void print_ascii(std::ostream& os) const {
        os << to_ascii();
    }

    void print_json(std::ostream& os) const {
        os << to_json();
    }

    void save_to_file(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (filename.find(".json") != std::string::npos) {
            print_json(ofs);
        } else if (filename.find(".mmd") != std::string::npos) {
            ofs << to_mermaid();
        } else {
            print_ascii(ofs);
        }
    }

    static CudaOpFlowchart create_linear_attention_flow() {
        CudaOpFlowchart flow("Linear Attention (GatedDeltaNet)", "RTX 5060 Ti");

        auto input_shape = TensorShape::make_2d(32, 512, 2048);
        auto hidden_shape = TensorShape::make_2d(32, 512, 2048);
        auto state_shape = TensorShape::make_attn(32, 1, 8, 256);

        flow.add_node(OpNode("InputEmbedding", "CPU", DeviceLocation::CPU)
            .add_input(input_shape, "token_ids")
            .add_output(TensorShape::make_2d(32, 512, 2048), "embeddings")
            .set_description("Token embedding lookup"));

        flow.add_node(OpNode("RMSNorm", "batch_rmsnorm_kernel", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(hidden_shape, "x")
            .add_output(hidden_shape, "x_norm")
            .set_estimation(32 * 512 * 2048 * 2, 0.1f)
            .add_note("Fused: normalization + residual add"));

        flow.add_node(OpNode("QKVProjection", "cublasSgemm_batch", DeviceLocation::CUDA,
                           OpOptimization::CUBLAS_GEMM)
            .add_input(hidden_shape, "x_norm")
            .add_output(TensorShape::make_attn_gqa(32, 512, 8, 2, 256), "qkv")
            .set_estimation(32 * 512 * 2048 * 2048 * 2, 1.5f)
            .add_note("cuBLAS batched GEMM for Q/K/V projection"));

        flow.add_node(OpNode("Conv1D + StateUpdate", "conv_state_update_fused", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(state_shape, "prev_state")
            .add_input(TensorShape::make_attn(32, 512, 8, 256), "qkv")
            .add_output(state_shape, "new_state")
            .add_output(TensorShape::make_attn(32, 512, 8, 256), "qkv_out")
            .set_estimation(32 * 512 * 8 * 256 * 10, 2.0f)
            .add_note("Register-cached recurrent state, no __syncthreads"));

        flow.add_node(OpNode("Gate + OutputProj", "gate_output_fused", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(TensorShape::make_attn(32, 512, 8, 256), "qkv_out")
            .add_output(hidden_shape, "attn_out")
            .set_estimation(32 * 512 * 2048 * 2048, 1.5f)
            .add_note("SiLU gating merged with output projection"));

        flow.add_node(OpNode("MLP", "cublasSgemm + silu_mul_kernel", DeviceLocation::CUDA,
                           OpOptimization::CUBLAS_GEMM)
            .add_input(hidden_shape, "x")
            .add_output(hidden_shape, "mlp_out")
            .set_estimation(32 * 512 * 2048 * 8192 * 2 * 2, 4.0f)
            .add_note("Gate/Silu fused with GEMM"));

        flow.add_node(OpNode("H2D_Memcpy", "cudaMemcpyAsync", DeviceLocation::HOST_TO_DEVICE)
            .add_input(hidden_shape, "h_input")
            .add_output(hidden_shape, "d_input")
            .set_estimation(0, 0.5f));

        flow.add_node(OpNode("D2H_Memcpy", "cudaMemcpyAsync", DeviceLocation::DEVICE_TO_HOST)
            .add_input(hidden_shape, "d_output")
            .add_output(hidden_shape, "h_output")
            .set_estimation(0, 0.3f));

        flow.add_edge("InputEmbedding", "RMSNorm", "hidden_states", hidden_shape);
        flow.add_edge("RMSNorm", "QKVProjection", "x_norm", hidden_shape);
        flow.add_edge("QKVProjection", "Conv1D + StateUpdate", "qkv", TensorShape::make_attn_gqa(32, 512, 8, 2, 256));
        flow.add_edge("Conv1D + StateUpdate", "Gate + OutputProj", "qkv_out", TensorShape::make_attn(32, 512, 8, 256));
        flow.add_edge("Gate + OutputProj", "MLP", "attn_out", hidden_shape);
        flow.add_edge("MLP", "D2H_Memcpy", "mlp_out", hidden_shape);

        return flow;
    }

    static CudaOpFlowchart create_full_attention_flow() {
        CudaOpFlowchart flow("Full Attention (GQA + RoPE)", "RTX 5060 Ti");

        auto hidden_shape = TensorShape::make_2d(32, 512, 2048);

        flow.add_node(OpNode("InputEmbedding", "CPU", DeviceLocation::CPU)
            .add_input(TensorShape::make_1d(32, 0), "token_ids")
            .add_output(hidden_shape, "embeddings")
            .set_description("Token embedding"));

        flow.add_node(OpNode("RMSNorm", "batch_rmsnorm_kernel", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(hidden_shape, "x")
            .add_output(hidden_shape, "x_norm"));

        flow.add_node(OpNode("QProjection + RoPE", "batch_fused_q_path_kernel", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(hidden_shape, "x_norm")
            .add_output(TensorShape::make_attn_gqa(32, 512, 8, 2, 256), "Q")
            .add_output(TensorShape::make_attn(32, 512, 8, 256), "gate")
            .add_note("RoPE applied within kernel"));

        flow.add_node(OpNode("KVCacheUpdate", "batch_fused_kv_cache_kernel", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(hidden_shape, "x_norm")
            .add_output(TensorShape::make_2d(32, 512, 512), "kv_cache")
            .add_note("K/V RMSNorm + RoPE fused"));

        flow.add_node(OpNode("FlashAttention", "flash_attention_kernel", DeviceLocation::CUDA,
                           OpOptimization::FLASH_ATTENTION)
            .add_input(TensorShape::make_attn_gqa(32, 512, 8, 2, 256), "Q")
            .add_input(TensorShape::make_2d(32, 512, 512), "K_cache")
            .add_input(TensorShape::make_2d(32, 512, 512), "V_cache")
            .add_output(TensorShape::make_attn(32, 512, 8, 256), "attn_out")
            .set_estimation(32 * 512 * 512 * 256 * 2, 3.0f)
            .add_note("Flash Attention v2: warp-level parallelism, tiled computation"));

        flow.add_node(OpNode("GateSigmoid", "batch_gate_sigmoid_kernel", DeviceLocation::CUDA,
                           OpOptimization::FUSED_KERNEL)
            .add_input(TensorShape::make_attn(32, 512, 8, 256), "attn_out")
            .add_input(TensorShape::make_attn(32, 512, 8, 256), "gate")
            .add_output(TensorShape::make_attn(32, 512, 8, 256), "gated_attn"));

        flow.add_node(OpNode("OutputProjection", "cublasSgemm", DeviceLocation::CUDA,
                           OpOptimization::CUBLAS_GEMM)
            .add_input(TensorShape::make_attn(32, 512, 8, 256), "gated_attn")
            .add_output(hidden_shape, "output")
            .add_note("cuBLAS GEMM for output projection"));

        flow.add_edge("InputEmbedding", "RMSNorm", "embeddings", hidden_shape);
        flow.add_edge("RMSNorm", "QProjection + RoPE", "x_norm", hidden_shape);
        flow.add_edge("RMSNorm", "KVCacheUpdate", "x_norm", hidden_shape);
        flow.add_edge("QProjection + RoPE", "FlashAttention", "Q", TensorShape::make_attn_gqa(32, 512, 8, 2, 256));
        flow.add_edge("KVCacheUpdate", "FlashAttention", "KV_cache", TensorShape::make_2d(32, 512, 512));
        flow.add_edge("FlashAttention", "GateSigmoid", "attn_out", TensorShape::make_attn(32, 512, 8, 256));
        flow.add_edge("QProjection + RoPE", "GateSigmoid", "gate", TensorShape::make_attn(32, 512, 8, 256));
        flow.add_edge("GateSigmoid", "OutputProjection", "gated_attn", TensorShape::make_attn(32, 512, 8, 256));
        flow.add_edge("OutputProjection", "D2H_Memcpy", "output", hidden_shape);

        return flow;
    }

private:
    static std::string sanitize_id(const std::string& name) {
        std::string id;
        for (char c : name) {
            if (c == ' ' || c == '+' || c == '-' || c == '(' || c == ')') {
                id += '_';
            } else {
                id += c;
            }
        }
        return id;
    }

    static std::string get_location_color(DeviceLocation loc) {
        switch (loc) {
            case DeviceLocation::CPU: return "#f8d7da";
            case DeviceLocation::CUDA: return "#d1e7dd";
            case DeviceLocation::HOST_TO_DEVICE: return "#fff3cd";
            case DeviceLocation::DEVICE_TO_HOST: return "#cfe2ff";
            case DeviceLocation::DEVICE_TO_DEVICE: return "#e2d9f3";
        }
        return "#ffffff";
    }

    static std::string get_edge_style(DeviceLocation loc) {
        switch (loc) {
            case DeviceLocation::CPU: return "-->";
            case DeviceLocation::CUDA: return "==>";
            case DeviceLocation::HOST_TO_DEVICE: return "-.->";
            case DeviceLocation::DEVICE_TO_HOST: return "-.->";
            case DeviceLocation::DEVICE_TO_DEVICE: return "~~>";
        }
        return "-->";
    }
};

} // namespace tools
} // namespace cuda
} // namespace qwen
