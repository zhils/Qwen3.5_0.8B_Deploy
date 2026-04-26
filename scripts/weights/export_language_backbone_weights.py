import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open

MODEL_PATH = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"
OUTPUT_DIR = "weights/language_backbone"

LINEAR_LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
FULL_LAYERS = [3,7,11,15,19,23]


def export_layer_weights(f, layer_idx: int, output_dir: str):
    layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)

    is_linear = layer_idx in LINEAR_LAYERS

    print(f"\n  Layer {layer_idx} ({'Linear' if is_linear else 'Full'}):")

    norm1_key = f"model.language_model.layers.{layer_idx}.input_layernorm.weight"
    norm2_key = f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight"

    norm1 = f.get_tensor(norm1_key).float().numpy()
    norm2 = f.get_tensor(norm2_key).float().numpy()

    norm1.tofile(os.path.join(layer_dir, "input_layernorm.bin"))
    norm2.tofile(os.path.join(layer_dir, "post_layernorm.bin"))
    print(f"    input_layernorm: {norm1.shape}")
    print(f"    post_layernorm: {norm2.shape}")

    gate_key = f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"
    up_key = f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight"
    down_key = f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight"

    gate_w = f.get_tensor(gate_key).float().numpy()
    up_w = f.get_tensor(up_key).float().numpy()
    down_w = f.get_tensor(down_key).float().numpy()

    gate_w.tofile(os.path.join(layer_dir, "mlp_gate.bin"))
    up_w.tofile(os.path.join(layer_dir, "mlp_up.bin"))
    down_w.tofile(os.path.join(layer_dir, "mlp_down.bin"))
    print(f"    mlp_gate: {gate_w.shape}, mlp_up: {up_w.shape}, mlp_down: {down_w.shape}")

    if is_linear:
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"

        qkv_key = f"{prefix}.in_proj_qkv.weight"
        a_key = f"{prefix}.in_proj_a.weight"
        b_key = f"{prefix}.in_proj_b.weight"
        z_key = f"{prefix}.in_proj_z.weight"
        conv_key = f"{prefix}.conv1d.weight"
        A_key = f"{prefix}.A_log"
        dt_key = f"{prefix}.dt_bias"
        norm_key = f"{prefix}.norm.weight"
        out_key = f"{prefix}.out_proj.weight"

        qkv_w = f.get_tensor(qkv_key).float().numpy()
        a_w = f.get_tensor(a_key).float().numpy()
        b_w = f.get_tensor(b_key).float().numpy()
        z_w = f.get_tensor(z_key).float().numpy()
        conv_w = f.get_tensor(conv_key).float().numpy()
        A = f.get_tensor(A_key).float().numpy()
        dt = f.get_tensor(dt_key).float().numpy()
        norm_w = f.get_tensor(norm_key).float().numpy()
        out_w = f.get_tensor(out_key).float().numpy()

        qkv_w.tofile(os.path.join(layer_dir, "linear_qkv.bin"))
        a_w.tofile(os.path.join(layer_dir, "linear_a.bin"))
        b_w.tofile(os.path.join(layer_dir, "linear_b.bin"))
        z_w.tofile(os.path.join(layer_dir, "linear_z.bin"))
        conv_w.tofile(os.path.join(layer_dir, "linear_conv1d.bin"))
        A.tofile(os.path.join(layer_dir, "linear_A_log.bin"))
        dt.tofile(os.path.join(layer_dir, "linear_dt_bias.bin"))
        norm_w.tofile(os.path.join(layer_dir, "linear_norm.bin"))
        out_w.tofile(os.path.join(layer_dir, "linear_out.bin"))

        print(f"    linear_qkv: {qkv_w.shape}, linear_a: {a_w.shape}")
        print(f"    linear_b: {b_w.shape}, linear_z: {z_w.shape}")
        print(f"    linear_conv1d: {conv_w.shape}, A_log: {A.shape}")
        print(f"    dt_bias: {dt.shape}, linear_norm: {norm_w.shape}")
        print(f"    linear_out: {out_w.shape}")

    else:
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"

        q_key = f"{prefix}.q_proj.weight"
        k_key = f"{prefix}.k_proj.weight"
        v_key = f"{prefix}.v_proj.weight"
        qn_key = f"{prefix}.q_norm.weight"
        kn_key = f"{prefix}.k_norm.weight"
        o_key = f"{prefix}.o_proj.weight"

        q_w = f.get_tensor(q_key).float().numpy()
        k_w = f.get_tensor(k_key).float().numpy()
        v_w = f.get_tensor(v_key).float().numpy()
        qn_w = f.get_tensor(qn_key).float().numpy()
        kn_w = f.get_tensor(kn_key).float().numpy()
        o_w = f.get_tensor(o_key).float().numpy()

        q_w.tofile(os.path.join(layer_dir, "full_q.bin"))
        k_w.tofile(os.path.join(layer_dir, "full_k.bin"))
        v_w.tofile(os.path.join(layer_dir, "full_v.bin"))
        qn_w.tofile(os.path.join(layer_dir, "full_q_norm.bin"))
        kn_w.tofile(os.path.join(layer_dir, "full_k_norm.bin"))
        o_w.tofile(os.path.join(layer_dir, "full_o.bin"))

        print(f"    full_q: {q_w.shape}, full_k: {k_w.shape}, full_v: {v_w.shape}")
        print(f"    full_q_norm: {qn_w.shape}, full_k_norm: {kn_w.shape}")
        print(f"    full_o: {o_w.shape}")


def export_final_norm(f, output_dir: str):
    norm_key = "model.language_model.norm.weight"
    norm_w = f.get_tensor(norm_key).float().numpy()

    output_path = os.path.join(output_dir, "final_norm.bin")
    norm_w.tofile(output_path)
    print(f"\n  final_norm: {norm_w.shape} -> {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Exporting Language Backbone Weights")
    print("=" * 60)
    print(f"\nLoading: {MODEL_PATH}")

    with safe_open(MODEL_PATH, framework="pt") as f:
        print("\n[1] Exporting 24 layer weights...")
        for i in range(24):
            export_layer_weights(f, i, OUTPUT_DIR)

        print("\n[2] Exporting final norm...")
        export_final_norm(f, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Export completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
