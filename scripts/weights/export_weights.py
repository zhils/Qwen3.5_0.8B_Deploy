import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open
from PIL import Image


def export_weights(safetensors_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    weight_key = "model.visual.patch_embed.proj.weight"
    bias_key = "model.visual.patch_embed.proj.bias"

    print(f"Loading weights from: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt") as f:
        weight_tensor = f.get_tensor(weight_key)
        weight = weight_tensor.float().numpy()

        try:
            bias_tensor = f.get_tensor(bias_key)
            bias = bias_tensor.float().numpy()
        except KeyError:
            bias = np.zeros(weight.shape[0], dtype=np.float32)

    print(f"Weight shape: {weight.shape}")
    print(f"Bias shape: {bias.shape}")

    weight_path = os.path.join(output_dir, "patch_embed_weight.bin")
    bias_path = os.path.join(output_dir, "patch_embed_bias.bin")

    weight.tofile(weight_path)
    bias.tofile(bias_path)

    print(f"\nExported files:")
    print(f"  Weight: {weight_path} ({weight.nbytes} bytes)")
    print(f"  Bias: {bias_path} ({bias.nbytes} bytes)")

    return weight, bias


def export_test_image(image_path: str, output_dir: str, target_size: int = 224):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading image: {image_path}")

    img = Image.open(image_path).convert("RGB")
    print(f"Original size: {img.size}")

    img_resized = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img_normalized = (img_array - mean) / std

    img_chw = img_normalized.transpose(2, 0, 1)

    temporal_patch = 2
    img_5d = np.zeros((temporal_patch, 3, target_size, target_size), dtype=np.float32)
    for t in range(temporal_patch):
        img_5d[t] = img_chw

    image_bin_path = os.path.join(output_dir, "test_image.bin")
    img_5d.tofile(image_bin_path)

    print(f"Preprocessed shape: {img_5d.shape} [T, C, H, W]")
    print(f"Exported image: {image_bin_path} ({img_5d.nbytes} bytes)")

    return img_5d


def export_reference_output(weight: np.ndarray, bias: np.ndarray, img_5d: np.ndarray, output_dir: str):
    temporal_patch = 2
    patch_size = 16
    embed_dim = weight.shape[0]

    T, C, H, W = img_5d.shape

    Nt = T // temporal_patch
    Nh = H // patch_size
    Nw = W // patch_size
    N = Nt * Nh * Nw

    patches = np.zeros((Nt, Nh, Nw, C, temporal_patch, patch_size, patch_size), dtype=np.float32)

    for tt in range(Nt):
        for yy in range(Nh):
            for xx in range(Nw):
                t_start = tt * temporal_patch
                y_start = yy * patch_size
                x_start = xx * patch_size
                patches[tt, yy, xx] = img_5d[
                    t_start:t_start + temporal_patch,
                    :,
                    y_start:y_start + patch_size,
                    x_start:x_start + patch_size
                ].transpose(1, 0, 2, 3)

    patches = patches.reshape(N, -1)

    weight_flat = weight.reshape(embed_dim, -1)
    output = patches @ weight_flat.T + bias

    output_path = os.path.join(output_dir, "reference_output.bin")
    output.tofile(output_path)

    print(f"\nReference output shape: {output.shape} [N, embed_dim]")
    print(f"Exported reference: {output_path} ({output.nbytes} bytes)")
    print(f"\nReference output statistics:")
    print(f"  Min: {output.min():.6f}")
    print(f"  Max: {output.max():.6f}")
    print(f"  Mean: {output.mean():.6f}")
    print(f"  First 10 values: {output[0, :10]}")

    return output


def main():
    model_path = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"
    image_path = r"d:\deploy\c++deploy\cat_dog.jpg"
    output_dir = r"d:\deploy\c++deploy\weights"

    print("=" * 60)
    print("Exporting weights and test data for C++ validation")
    print("=" * 60)

    weight, bias = export_weights(model_path, output_dir)
    img_5d = export_test_image(image_path, output_dir)
    export_reference_output(weight, bias, img_5d, output_dir)

    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
