import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open


def export_vit_weights(safetensors_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading weights from: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt") as f:
        keys = f.keys()
        vit_keys = [k for k in keys if k.startswith("model.visual.")]
        print(f"Found {len(vit_keys)} vision-related keys")

        patch_w = f.get_tensor("model.visual.patch_embed.proj.weight").float().numpy()
        patch_b = f.get_tensor("model.visual.patch_embed.proj.bias").float().numpy()
        patch_w.tofile(os.path.join(output_dir, "patch_embed_proj_weight.bin"))
        patch_b.tofile(os.path.join(output_dir, "patch_embed_proj_bias.bin"))
        print(f"  patch_embed proj: weight{patch_w.shape}, bias{patch_b.shape} -> patch_embed_proj_*.bin")

        pos_embed_key = "model.visual.pos_embed.weight"
        pos_embed = f.get_tensor(pos_embed_key).float().numpy()
        pos_embed_path = os.path.join(output_dir, "pos_embed.bin")
        pos_embed.tofile(pos_embed_path)
        print(f"  pos_embed: {pos_embed.shape} -> {pos_embed_path}")

        for block_idx in range(12):
            prefix = f"model.visual.blocks.{block_idx}"

            norm1_w = f.get_tensor(f"{prefix}.norm1.weight").float().numpy()
            norm1_b = f.get_tensor(f"{prefix}.norm1.bias").float().numpy()
            qkv_w = f.get_tensor(f"{prefix}.attn.qkv.weight").float().numpy()
            qkv_b = f.get_tensor(f"{prefix}.attn.qkv.bias").float().numpy()
            proj_w = f.get_tensor(f"{prefix}.attn.proj.weight").float().numpy()
            proj_b = f.get_tensor(f"{prefix}.attn.proj.bias").float().numpy()
            norm2_w = f.get_tensor(f"{prefix}.norm2.weight").float().numpy()
            norm2_b = f.get_tensor(f"{prefix}.norm2.bias").float().numpy()
            fc1_w = f.get_tensor(f"{prefix}.mlp.linear_fc1.weight").float().numpy()
            fc1_b = f.get_tensor(f"{prefix}.mlp.linear_fc1.bias").float().numpy()
            fc2_w = f.get_tensor(f"{prefix}.mlp.linear_fc2.weight").float().numpy()
            fc2_b = f.get_tensor(f"{prefix}.mlp.linear_fc2.bias").float().numpy()

            block_dir = os.path.join(output_dir, f"block_{block_idx}")
            os.makedirs(block_dir, exist_ok=True)

            norm1_w.tofile(os.path.join(block_dir, "norm1_weight.bin"))
            norm1_b.tofile(os.path.join(block_dir, "norm1_bias.bin"))
            qkv_w.tofile(os.path.join(block_dir, "qkv_weight.bin"))
            qkv_b.tofile(os.path.join(block_dir, "qkv_bias.bin"))
            proj_w.tofile(os.path.join(block_dir, "proj_weight.bin"))
            proj_b.tofile(os.path.join(block_dir, "proj_bias.bin"))
            norm2_w.tofile(os.path.join(block_dir, "norm2_weight.bin"))
            norm2_b.tofile(os.path.join(block_dir, "norm2_bias.bin"))
            fc1_w.tofile(os.path.join(block_dir, "fc1_weight.bin"))
            fc1_b.tofile(os.path.join(block_dir, "fc1_bias.bin"))
            fc2_w.tofile(os.path.join(block_dir, "fc2_weight.bin"))
            fc2_b.tofile(os.path.join(block_dir, "fc2_bias.bin"))

            print(f"  block_{block_idx}: norm1({norm1_w.shape}), qkv({qkv_w.shape}), "
                  f"proj({proj_w.shape}), norm2({norm2_w.shape}), "
                  f"fc1({fc1_w.shape}), fc2({fc2_w.shape})")

        merger_dir = os.path.join(output_dir, "merger")
        os.makedirs(merger_dir, exist_ok=True)

        merger_norm_w = f.get_tensor("model.visual.merger.norm.weight").float().numpy()
        merger_norm_b = f.get_tensor("model.visual.merger.norm.bias").float().numpy()
        merger_fc1_w = f.get_tensor("model.visual.merger.linear_fc1.weight").float().numpy()
        merger_fc1_b = f.get_tensor("model.visual.merger.linear_fc1.bias").float().numpy()
        merger_fc2_w = f.get_tensor("model.visual.merger.linear_fc2.weight").float().numpy()
        merger_fc2_b = f.get_tensor("model.visual.merger.linear_fc2.bias").float().numpy()

        merger_norm_w.tofile(os.path.join(merger_dir, "norm_weight.bin"))
        merger_norm_b.tofile(os.path.join(merger_dir, "norm_bias.bin"))
        merger_fc1_w.tofile(os.path.join(merger_dir, "fc1_weight.bin"))
        merger_fc1_b.tofile(os.path.join(merger_dir, "fc1_bias.bin"))
        merger_fc2_w.tofile(os.path.join(merger_dir, "fc2_weight.bin"))
        merger_fc2_b.tofile(os.path.join(merger_dir, "fc2_bias.bin"))

        print(f"  merger: norm({merger_norm_w.shape}), fc1({merger_fc1_w.shape}), fc2({merger_fc2_w.shape})")

    print(f"\nAll weights exported to: {output_dir}")


def export_reference_vit_output(safetensors_path: str, input_tensor: np.ndarray, output_dir: str):
    import torch.nn as nn
    import torch.nn.functional as F

    print("\nComputing reference output with PyTorch...")

    with safe_open(safetensors_path, framework="pt") as f:
        pos_embed = f.get_tensor("model.visual.pos_embed.weight").float()

        blocks_weights = []
        for block_idx in range(12):
            prefix = f"model.visual.blocks.{block_idx}"
            block_w = {
                'norm1_w': f.get_tensor(f"{prefix}.norm1.weight").float(),
                'norm1_b': f.get_tensor(f"{prefix}.norm1.bias").float(),
                'qkv_w': f.get_tensor(f"{prefix}.attn.qkv.weight").float(),
                'qkv_b': f.get_tensor(f"{prefix}.attn.qkv.bias").float(),
                'proj_w': f.get_tensor(f"{prefix}.attn.proj.weight").float(),
                'proj_b': f.get_tensor(f"{prefix}.attn.proj.bias").float(),
                'norm2_w': f.get_tensor(f"{prefix}.norm2.weight").float(),
                'norm2_b': f.get_tensor(f"{prefix}.norm2.bias").float(),
                'fc1_w': f.get_tensor(f"{prefix}.mlp.linear_fc1.weight").float(),
                'fc1_b': f.get_tensor(f"{prefix}.mlp.linear_fc1.bias").float(),
                'fc2_w': f.get_tensor(f"{prefix}.mlp.linear_fc2.weight").float(),
                'fc2_b': f.get_tensor(f"{prefix}.mlp.linear_fc2.bias").float(),
            }
            blocks_weights.append(block_w)

    x = torch.from_numpy(input_tensor)
    n, d = x.shape
    h = 12
    head_dim = d // h

    x = x + pos_embed[:n]

    for bw in blocks_weights:
        norm1 = F.layer_norm(x, [d], bw['norm1_w'], bw['norm1_b'])

        qkv = F.linear(norm1, bw['qkv_w'], bw['qkv_b'])
        qkv = qkv.reshape(n, 3, h, head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / (head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.permute(1, 0, 2).reshape(n, d)
        attn_out = F.linear(attn_out, bw['proj_w'], bw['proj_b'])

        x = x + attn_out

        norm2 = F.layer_norm(x, [d], bw['norm2_w'], bw['norm2_b'])

        hidden = F.linear(norm2, bw['fc1_w'], bw['fc1_b'])
        hidden = F.gelu(hidden, approximate='tanh')
        mlp_out = F.linear(hidden, bw['fc2_w'], bw['fc2_b'])

        x = x + mlp_out

    output = x.numpy()
    output_path = os.path.join(output_dir, "vit_reference_output.bin")
    output.tofile(output_path)

    print(f"Reference output shape: {output.shape}")
    print(f"Reference output saved to: {output_path}")
    print(f"Stats: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}")

    return output


def export_reference_merger_output(safetensors_path: str, vit_output: np.ndarray,
                                    grid_h: int, grid_w: int, output_dir: str):
    import torch.nn.functional as F

    print("\nComputing Merger reference output with PyTorch...")

    spatial_merge_size = 2
    in_hidden_size = 768
    intermediate_size = 3072
    out_hidden_size = 1024

    with safe_open(safetensors_path, framework="pt") as f:
        merger_norm_w = f.get_tensor("model.visual.merger.norm.weight").float()
        merger_norm_b = f.get_tensor("model.visual.merger.norm.bias").float()
        merger_fc1_w = f.get_tensor("model.visual.merger.linear_fc1.weight").float()
        merger_fc1_b = f.get_tensor("model.visual.merger.linear_fc1.bias").float()
        merger_fc2_w = f.get_tensor("model.visual.merger.linear_fc2.weight").float()
        merger_fc2_b = f.get_tensor("model.visual.merger.linear_fc2.bias").float()

    x = torch.from_numpy(vit_output)

    x = F.layer_norm(x, [in_hidden_size], merger_norm_w, merger_norm_b)

    merge_h = grid_h // spatial_merge_size
    merge_w = grid_w // spatial_merge_size

    x_grid = x.view(1, grid_h, grid_w, in_hidden_size)

    merged = []
    for mh in range(merge_h):
        for mw in range(merge_w):
            patches = []
            for sh in range(spatial_merge_size):
                for sw in range(spatial_merge_size):
                    orig_h = mh * spatial_merge_size + sh
                    orig_w = mw * spatial_merge_size + sw
                    patches.append(x_grid[0, orig_h, orig_w, :])
            merged_token = torch.cat(patches, dim=0)
            merged.append(merged_token)

    merged = torch.stack(merged, dim=0)
    print(f"  After spatial merge: {merged.shape}")

    hidden = F.linear(merged, merger_fc1_w, merger_fc1_b)
    hidden = F.gelu(hidden, approximate='tanh')

    output = F.linear(hidden, merger_fc2_w, merger_fc2_b)

    output_np = output.numpy()
    output_path = os.path.join(output_dir, "merger_reference_output.bin")
    output_np.tofile(output_path)

    print(f"  Merger output shape: {output_np.shape}")
    print(f"  Merger output saved to: {output_path}")
    print(f"  Stats: min={output_np.min():.6f}, max={output_np.max():.6f}, mean={output_np.mean():.6f}")

    return output_np


def main():
    model_path = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"
    output_dir = r"d:\deploy\c++deploy\weights\vit"

    print("=" * 60)
    print("Exporting Vision Transformer weights for C++ validation")
    print("=" * 60)

    export_vit_weights(model_path, output_dir)

    patch_embed_output = np.fromfile(
        r"d:\deploy\c++deploy\weights\reference_output.bin",
        dtype=np.float32
    ).reshape(196, 768)

    export_reference_vit_output(model_path, patch_embed_output, output_dir)

    vit_output = np.fromfile(
        os.path.join(output_dir, "vit_reference_output.bin"),
        dtype=np.float32
    ).reshape(196, 768)

    grid_h, grid_w = 14, 14
    export_reference_merger_output(model_path, vit_output, grid_h, grid_w, output_dir)

    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
