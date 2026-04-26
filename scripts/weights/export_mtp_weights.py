import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open

MODEL_PATH = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"
OUTPUT_DIR = "weights/mtp"

def export_mtp_weights(f, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Exporting pre-fc norm weights...")
    pre_fc_norm_hidden = f.get_tensor("mtp.pre_fc_norm_hidden.weight").float().numpy()
    pre_fc_norm_embedding = f.get_tensor("mtp.pre_fc_norm_embedding.weight").float().numpy()

    pre_fc_norm_hidden.tofile(os.path.join(output_dir, "pre_fc_norm_hidden.bin"))
    pre_fc_norm_embedding.tofile(os.path.join(output_dir, "pre_fc_norm_embedding.bin"))
    print(f"  pre_fc_norm_hidden: {pre_fc_norm_hidden.shape}")
    print(f"  pre_fc_norm_embedding: {pre_fc_norm_embedding.shape}")

    print("\n[2] Exporting MTP layer 0 weights (FullAttention + MLP)...")
    layer_idx = 0
    prefix = f"mtp.layers.{layer_idx}"

    input_ln = f.get_tensor(f"{prefix}.input_layernorm.weight").float().numpy()
    post_ln = f.get_tensor(f"{prefix}.post_attention_layernorm.weight").float().numpy()

    input_ln.tofile(os.path.join(output_dir, "layer_input_layernorm.bin"))
    post_ln.tofile(os.path.join(output_dir, "layer_post_attention_layernorm.bin"))
    print(f"  layer_input_layernorm: {input_ln.shape}")
    print(f"  layer_post_attention_layernorm: {post_ln.shape}")

    attn_prefix = f"{prefix}.self_attn"
    q_w = f.get_tensor(f"{attn_prefix}.q_proj.weight").float().numpy()
    k_w = f.get_tensor(f"{attn_prefix}.k_proj.weight").float().numpy()
    v_w = f.get_tensor(f"{attn_prefix}.v_proj.weight").float().numpy()
    q_n = f.get_tensor(f"{attn_prefix}.q_norm.weight").float().numpy()
    k_n = f.get_tensor(f"{attn_prefix}.k_norm.weight").float().numpy()
    o_w = f.get_tensor(f"{attn_prefix}.o_proj.weight").float().numpy()

    q_w.tofile(os.path.join(output_dir, "attn_q.bin"))
    k_w.tofile(os.path.join(output_dir, "attn_k.bin"))
    v_w.tofile(os.path.join(output_dir, "attn_v.bin"))
    q_n.tofile(os.path.join(output_dir, "attn_q_norm.bin"))
    k_n.tofile(os.path.join(output_dir, "attn_k_norm.bin"))
    o_w.tofile(os.path.join(output_dir, "attn_o.bin"))

    print(f"  attn_q: {q_w.shape}, attn_k: {k_w.shape}, attn_v: {v_w.shape}")
    print(f"  attn_q_norm: {q_n.shape}, attn_k_norm: {k_n.shape}")
    print(f"  attn_o: {o_w.shape}")

    mlp_prefix = f"{prefix}.mlp"
    gate_w = f.get_tensor(f"{mlp_prefix}.gate_proj.weight").float().numpy()
    up_w = f.get_tensor(f"{mlp_prefix}.up_proj.weight").float().numpy()
    down_w = f.get_tensor(f"{mlp_prefix}.down_proj.weight").float().numpy()

    gate_w.tofile(os.path.join(output_dir, "mlp_gate.bin"))
    up_w.tofile(os.path.join(output_dir, "mlp_up.bin"))
    down_w.tofile(os.path.join(output_dir, "mlp_down.bin"))

    print(f"  mlp_gate: {gate_w.shape}, mlp_up: {up_w.shape}, mlp_down: {down_w.shape}")

    print("\n[3] Exporting MTP final norm and FC weights...")
    mtp_norm = f.get_tensor("mtp.norm.weight").float().numpy()
    mtp_fc = f.get_tensor("mtp.fc.weight").float().numpy()

    mtp_norm.tofile(os.path.join(output_dir, "mtp_norm.bin"))
    mtp_fc.tofile(os.path.join(output_dir, "mtp_fc.bin"))

    print(f"  mtp_norm: {mtp_norm.shape}")
    print(f"  mtp_fc: {mtp_fc.shape}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Exporting MTP Weights")
    print("=" * 60)
    print(f"\nLoading: {MODEL_PATH}")

    with safe_open(MODEL_PATH, framework="pt") as f:
        export_mtp_weights(f, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("MTP Weights Export completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
