#!/usr/bin/env python3
"""
Export Qwen3.5-0.8B real weights from safetensors to binary files for C++ engine.

Usage:
    python3 export_real_weights.py

Output:
    weights/layer_XX.bin  - per-layer weights
    weights/norm.bin      - final norm weight
    weights/embedding.bin - embedding weight
"""

import os
import sys
import json
import struct
import numpy as np
from safetensors import safe_open

MODEL_PATH = "/mnt/d/deploy/modules/modles/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
CONFIG_PATH = "/mnt/d/deploy/modules/modles/Qwen3.5-0.8B/config.json"
OUTPUT_DIR = "/mnt/d/deploy/Qwen3.5_0.8B_Deploy/weights"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_floats(filename, data):
    """Write numpy array as float32 binary"""
    data = np.asarray(data, dtype=np.float32)
    with open(filename, 'wb') as f:
        f.write(data.tobytes())
    return data.size


def transpose_if_needed(weight, expected_in, expected_out):
    """Safetensors stores as [out, in], we need [in, out] for some cases"""
    if weight.shape[0] == expected_out and weight.shape[1] == expected_in:
        return weight.T
    return weight


def export_layer_weights(f, layer_idx, config, output_dir):
    """Export weights for a single layer"""
    text_config = config.get('text_config', config)
    hs = text_config['hidden_size']
    isz = text_config['intermediate_size']
    layer_types = text_config['layer_types']

    prefix = f"model.language_model.layers.{layer_idx}"
    is_full = layer_types[layer_idx] == 'full_attention'

    weights = []

    # 1. Input layernorm [hidden_size]
    w = f.get_tensor(f"{prefix}.input_layernorm.weight").float().numpy()
    weights.extend(w.flatten().tolist())

    # 2. Post attention layernorm [hidden_size]
    w = f.get_tensor(f"{prefix}.post_attention_layernorm.weight").float().numpy()
    weights.extend(w.flatten().tolist())

    # 3. MLP gate_proj [hidden_size, intermediate_size] -> transpose to [intermediate_size, hidden_size]
    w = f.get_tensor(f"{prefix}.mlp.gate_proj.weight").float().numpy()
    w = transpose_if_needed(w, hs, isz)
    weights.extend(w.flatten().tolist())

    # 4. MLP up_proj [hidden_size, intermediate_size] -> transpose
    w = f.get_tensor(f"{prefix}.mlp.up_proj.weight").float().numpy()
    w = transpose_if_needed(w, hs, isz)
    weights.extend(w.flatten().tolist())

    # 5. MLP down_proj [intermediate_size, hidden_size] -> transpose
    w = f.get_tensor(f"{prefix}.mlp.down_proj.weight").float().numpy()
    w = transpose_if_needed(w, isz, hs)
    weights.extend(w.flatten().tolist())

    if is_full:
        # Full attention weights
        num_heads = text_config['num_attention_heads']
        num_kv_heads = text_config['num_key_value_heads']
        head_dim = text_config['head_dim']
        q_size = num_heads * head_dim

        # Q proj: [num_heads * GQA_factor * head_dim, hidden_size] = [4096, 1024]
        # GQA_factor = num_kv_heads / num_heads = 2/8 = 0.25? No, actually K/V heads are 2, Q heads are 8
        # So Q has 8*256=2048 dim per head, but with GQA it becomes 4096 total
        # C++ expects [hidden_size, total_q * 2] = [1024, 4096]
        # Model Q is already [4096, 1024] = [total_q * 2, hidden_size], just transpose it
        q_proj = f.get_tensor(f"{prefix}.self_attn.q_proj.weight").float().numpy()  # [4096, 1024]
        q_proj_t = q_proj.T  # [1024, 4096]
        weights.extend(q_proj_t.flatten().tolist())

        # K proj: [num_kv_heads * head_dim, hidden_size] -> transpose
        kv_size = num_kv_heads * head_dim
        w = f.get_tensor(f"{prefix}.self_attn.k_proj.weight").float().numpy()
        w = transpose_if_needed(w, hs, kv_size)
        weights.extend(w.flatten().tolist())

        # V proj: [hidden_size, num_kv_heads * head_dim] -> transpose
        w = f.get_tensor(f"{prefix}.self_attn.v_proj.weight").float().numpy()
        w = transpose_if_needed(w, hs, kv_size)
        weights.extend(w.flatten().tolist())

        # Q norm [head_dim]
        w = f.get_tensor(f"{prefix}.self_attn.q_norm.weight").float().numpy()
        weights.extend(w.flatten().tolist())

        # K norm [head_dim]
        w = f.get_tensor(f"{prefix}.self_attn.k_norm.weight").float().numpy()
        weights.extend(w.flatten().tolist())

        # O proj: [num_heads * head_dim, hidden_size] -> transpose
        w = f.get_tensor(f"{prefix}.self_attn.o_proj.weight").float().numpy()
        w = transpose_if_needed(w, q_size, hs)
        weights.extend(w.flatten().tolist())
    else:
        # Linear attention weights
        linear_num_heads = text_config['linear_num_key_heads']
        linear_key_dim = text_config['linear_key_head_dim']
        linear_value_dim = text_config['linear_value_head_dim']
        conv_kernel = text_config['linear_conv_kernel_dim']
        
        k_dim = linear_num_heads * linear_key_dim
        v_dim = linear_num_heads * linear_value_dim
        conv_dim = k_dim * 2 + v_dim
        z_dim = linear_num_heads * linear_value_dim

        # in_proj_qkv_weight: [hidden_size, conv_dim] -> transpose
        w = f.get_tensor(f"{prefix}.linear_attn.in_proj_qkv.weight").float().numpy()
        w = transpose_if_needed(w, hs, conv_dim)
        weights.extend(w.flatten().tolist())

        # in_proj_a_weight: [num_heads, hidden_size] -> transpose to [hidden_size, num_heads]
        w = f.get_tensor(f"{prefix}.linear_attn.in_proj_a.weight").float().numpy()
        w = transpose_if_needed(w, hs, linear_num_heads)
        weights.extend(w.flatten().tolist())

        # in_proj_b_weight: [num_heads, hidden_size] -> transpose
        w = f.get_tensor(f"{prefix}.linear_attn.in_proj_b.weight").float().numpy()
        w = transpose_if_needed(w, hs, linear_num_heads)
        weights.extend(w.flatten().tolist())

        # in_proj_z_weight: [hidden_size, z_dim] -> transpose
        w = f.get_tensor(f"{prefix}.linear_attn.in_proj_z.weight").float().numpy()
        w = transpose_if_needed(w, hs, z_dim)
        weights.extend(w.flatten().tolist())

        # conv1d_weight: [conv_dim, 1, conv_kernel] -> flatten
        w = f.get_tensor(f"{prefix}.linear_attn.conv1d.weight").float().numpy()
        weights.extend(w.flatten().tolist())

        # out_proj_weight: [z_dim, hidden_size] -> transpose
        w = f.get_tensor(f"{prefix}.linear_attn.out_proj.weight").float().numpy()
        w = transpose_if_needed(w, z_dim, hs)
        weights.extend(w.flatten().tolist())

        # a_log: [num_heads]
        w = f.get_tensor(f"{prefix}.linear_attn.A_log").float().numpy()
        weights.extend(w.flatten().tolist())

        # dt_bias: [num_heads]
        w = f.get_tensor(f"{prefix}.linear_attn.dt_bias").float().numpy()
        weights.extend(w.flatten().tolist())

        # norm_weight: [value_dim] - stored as float32 in model
        w = f.get_tensor(f"{prefix}.linear_attn.norm.weight").float().numpy()
        weights.extend(w.flatten().tolist())

    filename = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    count = write_floats(filename, weights)
    print(f"  Layer {layer_idx:2d} ({'Full' if is_full else 'Linear'}): {count} floats -> {filename}")
    return count


def export_weights():
    print("=" * 70)
    print("Exporting Qwen3.5-0.8B weights to binary files")
    print("=" * 70)

    config = load_config()
    text_config = config.get('text_config', config)
    num_layers = text_config['num_hidden_layers']

    ensure_dir(OUTPUT_DIR)

    total_floats = 0

    with safe_open(MODEL_PATH, framework="pt") as f:
        # Export each layer
        for layer_idx in range(num_layers):
            count = export_layer_weights(f, layer_idx, config, OUTPUT_DIR)
            total_floats += count

        # Export final norm
        w = f.get_tensor("model.language_model.norm.weight").float().numpy()
        norm_file = os.path.join(OUTPUT_DIR, "norm.bin")
        count = write_floats(norm_file, w)
        total_floats += count
        print(f"\n  Final norm: {count} floats -> {norm_file}")

        # Export embedding (also used as lm_head since tied)
        w = f.get_tensor("model.language_model.embed_tokens.weight").float().numpy()
        embed_file = os.path.join(OUTPUT_DIR, "embedding.bin")
        count = write_floats(embed_file, w)
        total_floats += count
        print(f"  Embedding: {count} floats -> {embed_file}")

    # Write metadata
    meta = {
        "model": "Qwen3.5-0.8B",
        "num_layers": num_layers,
        "hidden_size": text_config['hidden_size'],
        "intermediate_size": text_config['intermediate_size'],
        "vocab_size": text_config.get('vocab_size', 248320),
        "num_attention_heads": text_config['num_attention_heads'],
        "num_key_value_heads": text_config['num_key_value_heads'],
        "head_dim": text_config['head_dim'],
        "linear_num_key_heads": text_config['linear_num_key_heads'],
        "linear_key_head_dim": text_config['linear_key_head_dim'],
        "linear_value_head_dim": text_config['linear_value_head_dim'],
        "linear_conv_kernel_dim": text_config['linear_conv_kernel_dim'],
        "layer_types": text_config['layer_types'],
        "total_floats": total_floats,
        "weight_files": [f"layer_{i:02d}.bin" for i in range(num_layers)] + ["norm.bin", "embedding.bin"]
    }

    meta_file = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Metadata: {meta_file}")
    print(f"\n  Total floats exported: {total_floats:,}")
    print(f"  Total size: {total_floats * 4 / 1024 / 1024:.1f} MB")
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    export_weights()
