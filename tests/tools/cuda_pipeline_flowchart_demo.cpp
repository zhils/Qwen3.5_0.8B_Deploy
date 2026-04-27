#include "cuda_op_flowchart.hpp"
#include <iostream>
#include <fstream>

using namespace qwen::cuda::tools;

// 构建完整的端到端 Linear Attention Layer 流程图
CudaOpFlowchart build_linear_attn_end_to_end() {
    CudaOpFlowchart flow("End-to-End: Linear Attention Layer (Decode)", "RTX 5060 Ti");

    int B = 1, S = 1, H = 1024;
    int NH = 16, KD = 128, VD = 128;
    int CONV_DIM = NH * (KD * 2 + VD);  // 16 * 384 = 6144
    int Z_DIM = NH * VD;                // 16 * 128 = 2048
    int ISZ = 3584;

    auto hidden_shape = TensorShape::make_1d(B, H);
    auto conv_shape = TensorShape::make_1d(B, CONV_DIM);
    auto qkv_shape = TensorShape::make_attn(B, S, NH, KD);
    auto v_shape = TensorShape::make_attn(B, S, NH, VD);
    auto z_shape = TensorShape::make_attn(B, S, NH, VD);
    auto inter_shape = TensorShape::make_1d(B, ISZ);

    // ====== Attention Branch ======
    flow.add_node(OpNode("InputNorm", "rmsnorm_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(hidden_shape, "x")
        .add_output(hidden_shape, "x_norm")
        .set_estimation(H * 2, 0.02f)
        .add_note("Shared memory reduction, block_size=256"));

    flow.add_node(OpNode("QKV_Proj", "cublasSgemv", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden_shape, "x_norm")
        .add_output(conv_shape, "mixed_qkv")
        .set_estimation(H * CONV_DIM * 2, 0.15f)
        .add_note("[1,1024] x [1024,6144], TF32 Tensor Core"));

    flow.add_node(OpNode("Conv1D+State", "conv1d_update_fused_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(conv_shape, "mixed_qkv")
        .add_output(conv_shape, "conv_out")
        .set_estimation(CONV_DIM * 4 * 2, 0.03f)
        .add_note("FUSED: Conv1D + SiLU + state update (was 2 kernels)"));

    flow.add_node(OpNode("L2Norm_QK", "l2norm_qk_fused_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(qkv_shape, "Q")
        .add_input(qkv_shape, "K")
        .add_output(qkv_shape, "Q_norm")
        .add_output(qkv_shape, "K_norm")
        .set_estimation(NH * KD * 2, 0.02f)
        .add_note("FUSED: L2 norm Q + K in one kernel (was 2 kernels)"));

    flow.add_node(OpNode("A_Proj", "linear_proj_kernel", DeviceLocation::CUDA)
        .add_input(hidden_shape, "x_norm")
        .add_output(TensorShape::make_1d(B, NH), "a")
        .set_estimation(H * NH * 2, 0.01f));

    flow.add_node(OpNode("B_Proj", "linear_proj_kernel", DeviceLocation::CUDA)
        .add_input(hidden_shape, "x_norm")
        .add_output(TensorShape::make_1d(B, NH), "b")
        .set_estimation(H * NH * 2, 0.01f));

    flow.add_node(OpNode("Z_Proj", "linear_proj_kernel", DeviceLocation::CUDA)
        .add_input(hidden_shape, "x_norm")
        .add_output(z_shape, "z")
        .set_estimation(H * Z_DIM * 2, 0.01f));

    flow.add_node(OpNode("GatedDelta", "gated_delta_kernel", DeviceLocation::CUDA,
                        OpOptimization::SHARED_MEMORY)
        .add_input(qkv_shape, "Q")
        .add_input(qkv_shape, "K")
        .add_input(v_shape, "V")
        .add_input(TensorShape::make_1d(B, NH), "a")
        .add_input(TensorShape::make_1d(B, NH), "b")
        .add_output(v_shape, "attn_out")
        .set_estimation(NH * KD * VD * 4, 0.20f)
        .add_note("Shared memory for params, sequential per head")
        .add_note("**FUSION OPPORTUNITY**: Merge with Norm+Gate below"));

    flow.add_node(OpNode("Norm+Gate", "norm_gate_fused_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(v_shape, "attn_out")
        .add_input(z_shape, "z")
        .add_output(v_shape, "gated_attn")
        .set_estimation(NH * VD * 2, 0.02f)
        .add_note("FUSED: RMSNorm + SiLU gate (was 2 kernels)"));

    flow.add_node(OpNode("Out_Proj", "la_output_proj_kernel", DeviceLocation::CUDA)
        .add_input(v_shape, "gated_attn")
        .add_output(hidden_shape, "attn_result")
        .set_estimation(Z_DIM * H * 2, 0.05f));

    // ====== Residual + PostNorm ======
    flow.add_node(OpNode("ResAdd+PostNorm", "rmsnorm_add_residual_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(hidden_shape, "x")
        .add_input(hidden_shape, "attn_result")
        .add_output(hidden_shape, "residual")
        .add_output(hidden_shape, "post_norm")
        .set_estimation(H * 2, 0.02f)
        .add_note("FUSED: residual add + RMSNorm (was 2 kernels)"));

    // ====== MLP Branch ======
    flow.add_node(OpNode("Gate_Proj", "cublasSgemv", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden_shape, "post_norm")
        .add_output(inter_shape, "gate")
        .set_estimation(H * ISZ * 2, 0.10f));

    flow.add_node(OpNode("Up_Proj", "cublasSgemv", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden_shape, "post_norm")
        .add_output(inter_shape, "up")
        .set_estimation(H * ISZ * 2, 0.10f));

    flow.add_node(OpNode("SiLU+Mul", "launch_fused_gate_silu_mul", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(inter_shape, "gate")
        .add_input(inter_shape, "up")
        .add_output(inter_shape, "hidden")
        .set_estimation(ISZ * 2, 0.01f)
        .add_note("FUSED: SiLU(gate) * up element-wise"));

    flow.add_node(OpNode("Down_Proj", "cublasSgemv", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(inter_shape, "hidden")
        .add_output(hidden_shape, "mlp_out")
        .set_estimation(ISZ * H * 2, 0.10f));

    flow.add_node(OpNode("MLP_Residual", "cublasSgemv(beta=1)", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden_shape, "mlp_out")
        .add_input(hidden_shape, "residual")
        .add_output(hidden_shape, "layer_out")
        .set_estimation(H, 0.01f)
        .add_note("Residual add fused into GEMM (beta=1)"));

    // ====== Edges ======
    flow.add_edge("InputNorm", "QKV_Proj", "x_norm", hidden_shape);
    flow.add_edge("QKV_Proj", "Conv1D+State", "mixed_qkv", conv_shape);
    flow.add_edge("Conv1D+State", "L2Norm_QK", "conv_out", conv_shape);
    flow.add_edge("L2Norm_QK", "GatedDelta", "Q/K_norm", qkv_shape);
    flow.add_edge("A_Proj", "GatedDelta", "a", TensorShape::make_1d(B, NH));
    flow.add_edge("B_Proj", "GatedDelta", "b", TensorShape::make_1d(B, NH));
    flow.add_edge("GatedDelta", "Norm+Gate", "attn_out", v_shape);
    flow.add_edge("Z_Proj", "Norm+Gate", "z", z_shape);
    flow.add_edge("Norm+Gate", "Out_Proj", "gated_attn", v_shape);
    flow.add_edge("Out_Proj", "ResAdd+PostNorm", "attn_result", hidden_shape);
    flow.add_edge("ResAdd+PostNorm", "Gate_Proj", "post_norm", hidden_shape);
    flow.add_edge("ResAdd+PostNorm", "Up_Proj", "post_norm", hidden_shape);
    flow.add_edge("Gate_Proj", "SiLU+Mul", "gate", inter_shape);
    flow.add_edge("Up_Proj", "SiLU+Mul", "up", inter_shape);
    flow.add_edge("SiLU+Mul", "Down_Proj", "hidden", inter_shape);
    flow.add_edge("Down_Proj", "MLP_Residual", "mlp_out", hidden_shape);

    return flow;
}

// 构建 Batch Prefill 流程图
CudaOpFlowchart build_prefill_end_to_end() {
    CudaOpFlowchart flow("End-to-End: Batch Prefill (128 tokens)", "RTX 5060 Ti");

    int B = 128, H = 1024, ISZ = 3584;
    int NH = 16, KD = 128, VD = 128;
    int CONV_DIM = NH * (KD * 2 + VD);
    int Z_DIM = NH * VD;

    auto hidden_shape = TensorShape::make_2d(B, 1, H);
    auto conv_shape = TensorShape::make_2d(B, 1, CONV_DIM);
    auto qkv_shape = TensorShape::make_attn(B, 1, NH, KD);
    auto v_shape = TensorShape::make_attn(B, 1, NH, VD);
    auto z_shape = TensorShape::make_attn(B, 1, NH, VD);
    auto inter_shape = TensorShape::make_2d(B, 1, ISZ);

    flow.add_node(OpNode("H2D_Input", "cudaMemcpyAsync", DeviceLocation::HOST_TO_DEVICE)
        .add_input(hidden_shape, "h_input")
        .add_output(hidden_shape, "d_input")
        .set_estimation(0, 0.5f)
        .add_note("**BOTTLENECK**: H2D transfer for positions array"));

    flow.add_node(OpNode("InputNorm", "rmsnorm_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(hidden_shape, "d_input")
        .add_output(hidden_shape, "x_norm")
        .set_estimation(B * H * 2, 0.05f));

    flow.add_node(OpNode("QKV_Proj_Batch", "cublasSgemm", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden_shape, "x_norm")
        .add_output(conv_shape, "mixed_qkv")
        .set_estimation(B * H * CONV_DIM * 2, 0.30f)
        .add_note("[128,1024] x [1024,6144] -> [128,6144]"));

    flow.add_node(OpNode("Conv1D_Batch", "conv1d_update_fused_batch_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(conv_shape, "mixed_qkv")
        .add_output(conv_shape, "conv_out")
        .set_estimation(B * CONV_DIM * 4, 0.10f));

    flow.add_node(OpNode("L2Norm_QK_Batch", "l2norm_qk_fused_batch_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(qkv_shape, "Q")
        .add_input(qkv_shape, "K")
        .add_output(qkv_shape, "Q_norm")
        .add_output(qkv_shape, "K_norm")
        .set_estimation(B * NH * KD * 2, 0.05f));

    flow.add_node(OpNode("GatedDelta_Seq", "gated_delta_kernel (loop)", DeviceLocation::CUDA)
        .add_input(qkv_shape, "Q")
        .add_input(qkv_shape, "K")
        .add_input(v_shape, "V")
        .add_output(v_shape, "attn_out")
        .set_estimation(B * NH * KD * VD * 4, 2.0f)
        .add_note("**BOTTLENECK**: Sequential loop over batch, no parallelism")
        .add_note("**FUSION OPPORTUNITY**: Batch-parallel gated delta"));

    flow.add_node(OpNode("Norm+Gate_Batch", "norm_gate_fused_batch_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(v_shape, "attn_out")
        .add_input(z_shape, "z")
        .add_output(v_shape, "gated_attn")
        .set_estimation(B * NH * VD * 2, 0.05f));

    flow.add_node(OpNode("Out_Proj_Batch", "cublasSgemm", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(v_shape, "gated_attn")
        .add_output(hidden_shape, "attn_result")
        .set_estimation(B * Z_DIM * H * 2, 0.15f));

    flow.add_node(OpNode("MLP_Batch", "cublasSgemm x3 + silu_mul", DeviceLocation::CUDA,
                        OpOptimization::CUBLAS_GEMM)
        .add_input(hidden_shape, "post_norm")
        .add_output(hidden_shape, "mlp_out")
        .set_estimation(B * (H * ISZ * 2 + ISZ * H * 2), 0.50f)
        .add_note("Gate/Up/Down projections all via cuBLAS GEMM"));

    flow.add_edge("H2D_Input", "InputNorm", "d_input", hidden_shape, DeviceLocation::HOST_TO_DEVICE);
    flow.add_edge("InputNorm", "QKV_Proj_Batch", "x_norm", hidden_shape);
    flow.add_edge("QKV_Proj_Batch", "Conv1D_Batch", "mixed_qkv", conv_shape);
    flow.add_edge("Conv1D_Batch", "L2Norm_QK_Batch", "conv_out", conv_shape);
    flow.add_edge("L2Norm_QK_Batch", "GatedDelta_Seq", "Q/K", qkv_shape);
    flow.add_edge("GatedDelta_Seq", "Norm+Gate_Batch", "attn_out", v_shape);
    flow.add_edge("Norm+Gate_Batch", "Out_Proj_Batch", "gated_attn", v_shape);
    flow.add_edge("Out_Proj_Batch", "MLP_Batch", "attn_result", hidden_shape);

    return flow;
}

// 构建融合机会分析图
CudaOpFlowchart build_fusion_opportunities() {
    CudaOpFlowchart flow("Fusion Opportunities Analysis", "RTX 5060 Ti");

    auto hidden = TensorShape::make_1d(1, 1024);
    auto attn = TensorShape::make_attn(1, 1, 16, 128);

    // Current state
    flow.add_node(OpNode("Current: GatedDelta", "gated_delta_kernel", DeviceLocation::CUDA)
        .add_input(attn, "Q/K/V/a/b")
        .add_output(attn, "attn_out")
        .set_estimation(16*128*128*4, 0.20f));

    flow.add_node(OpNode("Current: Norm+Gate", "norm_gate_fused_kernel", DeviceLocation::CUDA,
                        OpOptimization::FUSED_KERNEL)
        .add_input(attn, "attn_out")
        .add_output(attn, "gated")
        .set_estimation(16*128*2, 0.02f));

    // Proposed fusion
    flow.add_node(OpNode("PROPOSED: GatedDelta+Norm+Gate", "gated_delta_norm_gate_fused",
                        DeviceLocation::CUDA, OpOptimization::FUSED_KERNEL)
        .add_input(attn, "Q/K/V/a/b/z")
        .add_output(attn, "gated")
        .set_estimation(16*128*128*4, 0.18f)
        .add_note("**SAVE**: 1 kernel launch (~5-10us overhead)")
        .add_note("**SAVE**: Eliminate intermediate attn_out buffer")
        .add_note("**RISK**: Increased register pressure, may reduce occupancy"));

    flow.add_edge("Current: GatedDelta", "Current: Norm+Gate", "attn_out", attn);

    return flow;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     CUDA End-to-End Pipeline Flowchart - Fusion & Bottleneck Analysis        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

    // 1. Decode path (single token)
    auto decode_flow = build_linear_attn_end_to_end();
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  1. DECODE PATH: Single Token Forward (1 layer)\n";
    std::cout << std::string(80, '=') << "\n";
    decode_flow.print_ascii(std::cout);
    decode_flow.save_to_file("decode_flow.mmd");

    // 2. Prefill path (batch)
    auto prefill_flow = build_prefill_end_to_end();
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  2. PREFILL PATH: Batch 128 Tokens (1 layer)\n";
    std::cout << std::string(80, '=') << "\n";
    prefill_flow.print_ascii(std::cout);
    prefill_flow.save_to_file("prefill_flow.mmd");

    // 3. Fusion opportunities
    auto fusion_flow = build_fusion_opportunities();
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  3. FUSION OPPORTUNITIES\n";
    std::cout << std::string(80, '=') << "\n";
    fusion_flow.print_ascii(std::cout);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  Summary:\n";
    std::cout << "  - decode_flow.mmd: Mermaid diagram for decode path\n";
    std::cout << "  - prefill_flow.mmd: Mermaid diagram for prefill path\n";
    std::cout << "  - Green nodes: Already fused kernels\n";
    std::cout << "  - Yellow nodes: Data transfer (H2D/D2H)\n";
    std::cout << "  - Red text: Bottlenecks and fusion opportunities\n";
    std::cout << std::string(80, '=') << "\n";

    return 0;
}
