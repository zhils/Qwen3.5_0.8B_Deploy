#include "cuda_op_flowchart.hpp"
#include <iostream>
#include <fstream>

using namespace qwen::cuda::tools;

void demo_linear_attention() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Linear Attention (GatedDeltaNet) CUDA Flow\n";
    std::cout << std::string(80, '=') << "\n";

    auto flow = CudaOpFlowchart::create_linear_attention_flow();

    std::cout << "\n--- Mermaid Format ---\n";
    flow.print_mermaid(std::cout);

    std::cout << "\n--- ASCII Format ---\n";
    flow.print_ascii(std::cout);
}

void demo_full_attention() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Full Attention (GQA + RoPE) CUDA Flow\n";
    std::cout << std::string(80, '=') << "\n";

    auto flow = CudaOpFlowchart::create_full_attention_flow();

    std::cout << "\n--- ASCII Format ---\n";
    flow.print_ascii(std::cout);
}

void demo_custom_flow() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Custom Flow: MLP Layer\n";
    std::cout << std::string(80, '=') << "\n";

    CudaOpFlowchart flow("MLP Layer", "RTX 5060 Ti");

    auto hidden = TensorShape::make_2d(32, 512, 2048);
    auto gate_shape = TensorShape::make_2d(32, 512, 8192);
    auto intermediate_shape = TensorShape::make_2d(32, 512, 8192);

    flow.add_node(OpNode("Input", "CPU", DeviceLocation::CPU)
        .add_output(hidden, "hidden")
        .set_description("Hidden states from previous layer"));

    flow.add_node(OpNode("UpProjection", "cublasSgemm", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden, "hidden")
        .add_output(intermediate_shape, "intermediate")
        .add_note("GATE = X @ W_gate, 32x2048 @ 2048x8192"));

    flow.add_node(OpNode("SiLU + Mul", "silu_mul_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(intermediate_shape, "intermediate")
        .add_output(intermediate_shape, "gated")
        .add_note("silu(x) * x, element-wise"));

    flow.add_node(OpNode("DownProjection", "cublasSgemm", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(intermediate_shape, "gated")
        .add_output(hidden, "output")
        .add_note("OUTPUT = gated @ W_down, 32x8192 @ 8192x2048"));

    flow.add_node(OpNode("H2D_Transfer", "cudaMemcpyAsync", DeviceLocation::HOST_TO_DEVICE)
        .add_input(hidden, "h_input")
        .add_output(hidden, "d_input")
        .set_estimation(0, 0.5f));

    flow.add_node(OpNode("D2H_Transfer", "cudaMemcpyAsync", DeviceLocation::DEVICE_TO_HOST)
        .add_input(hidden, "d_output")
        .add_output(hidden, "h_output")
        .set_estimation(0, 0.3f));

    flow.add_edge("Input", "H2D_Transfer", "hidden", hidden);
    flow.add_edge("H2D_Transfer", "UpProjection", "hidden", hidden);
    flow.add_edge("UpProjection", "SiLU + Mul", "intermediate", intermediate_shape);
    flow.add_edge("SiLU + Mul", "DownProjection", "gated", intermediate_shape);
    flow.add_edge("DownProjection", "D2H_Transfer", "output", hidden);

    std::cout << "\n--- Mermaid Format ---\n";
    flow.print_mermaid(std::cout);

    std::cout << "\n--- JSON Format ---\n";
    flow.print_json(std::cout);

    flow.save_to_file("mlp_flow.mmd");
    std::cout << "\n[Mermaid diagram saved to mlp_flow.mmd]\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         CUDA Operator Flowchart Visualization Tool - Demo                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

    demo_linear_attention();
    demo_full_attention();
    demo_custom_flow();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Tool Features:\n";
    std::cout << "  - TensorShape: B=batch, S=seq_len, H=heads, D=head_dim, C=channels\n";
    std::cout << "  - Memory size auto-calculation (MB/KB/B)\n";
    std::cout << "  - Location tags: [CPU], [GPU], [H2D], [D2H], [D2D]\n";
    std::cout << "  - Optimization badges: Fused, cuBLAS, FlashAttn, TensorCore, etc.\n";
    std::cout << "  - Output formats: Mermaid, ASCII, JSON\n";
    std::cout << std::string(80, '=') << "\n";

    return 0;
}
