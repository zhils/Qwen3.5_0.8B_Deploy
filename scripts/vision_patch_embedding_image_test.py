import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open
from PIL import Image
from typing import Tuple, Optional


class VisionPatchEmbedding:
    def __init__(
        self,
        embed_dim: int = 768,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch: int = 2,
        use_bias: bool = True
    ):
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_patch = temporal_patch
        self.use_bias = use_bias

        self.weight = None
        self.bias = None

    def load_weights_from_safetensors(self, safetensors_path: str):
        weight_key = "model.visual.patch_embed.proj.weight"
        bias_key = "model.visual.patch_embed.proj.bias"

        with safe_open(safetensors_path, framework="pt") as f:
            weight_tensor = f.get_tensor(weight_key)
            self.weight = weight_tensor.float().numpy()
            print(f"Loaded weight: shape={self.weight.shape}, original dtype={weight_tensor.dtype}")

            if self.use_bias:
                try:
                    bias_tensor = f.get_tensor(bias_key)
                    self.bias = bias_tensor.float().numpy()
                    print(f"Loaded bias: shape={self.bias.shape}, original dtype={bias_tensor.dtype}")
                except KeyError:
                    print("No bias found, initializing to zeros")
                    self.bias = np.zeros(self.embed_dim, dtype=np.float32)

        expected_shape = (
            self.embed_dim,
            self.in_channels,
            self.temporal_patch,
            self.patch_size,
            self.patch_size
        )
        if self.weight.shape != expected_shape:
            raise ValueError(
                f"Weight shape mismatch. Expected {expected_shape}, got {self.weight.shape}"
            )

        print("Weights loaded successfully!")

    def forward_fast(self, x: np.ndarray) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights_from_safetensors first.")

        B, T, C, H, W = x.shape

        Nt = T // self.temporal_patch
        Nh = H // self.patch_size
        Nw = W // self.patch_size
        N = Nt * Nh * Nw

        patches = np.zeros((B, Nt, Nh, Nw, C, self.temporal_patch, self.patch_size, self.patch_size), dtype=np.float32)

        for tt in range(Nt):
            for yy in range(Nh):
                for xx in range(Nw):
                    t_start = tt * self.temporal_patch
                    y_start = yy * self.patch_size
                    x_start = xx * self.patch_size
                    patches[:, tt, yy, xx] = x[
                        :,
                        t_start:t_start + self.temporal_patch,
                        :,
                        y_start:y_start + self.patch_size,
                        x_start:x_start + self.patch_size
                    ].transpose(0, 2, 1, 3, 4)

        patches = patches.reshape(B, N, -1)

        weight_flat = self.weight.reshape(self.embed_dim, -1)

        output = patches @ weight_flat.T

        if self.use_bias:
            output = output + self.bias

        return output


def load_and_preprocess_image(
    image_path: str,
    target_height: int = 224,
    target_width: int = 224,
    mean: list = [0.5, 0.5, 0.5],
    std: list = [0.5, 0.5, 0.5],
    temporal_patch: int = 2
) -> np.ndarray:
    print(f"\n[Image Loading]")
    print(f"  Path: {image_path}")

    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    print(f"  Original size: {original_size}")

    img_resized = img.resize((target_width, target_height), Image.Resampling.BICUBIC)
    print(f"  Resized to: {target_width} x {target_height}")

    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    print(f"  Array shape after resize: {img_array.shape}")
    print(f"  Value range before norm: [{img_array.min():.4f}, {img_array.max():.4f}]")

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img_normalized = (img_array - mean) / std
    print(f"  Value range after norm: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")

    img_chw = img_normalized.transpose(2, 0, 1)
    print(f"  Shape after CHW transpose: {img_chw.shape}")

    img_5d = np.zeros((1, temporal_patch, 3, target_height, target_width), dtype=np.float32)
    for t in range(temporal_patch):
        img_5d[0, t] = img_chw

    print(f"  Final 5D tensor shape: {img_5d.shape} [B, T, C, H, W]")

    return img_5d


def main():
    model_path = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"
    image_path = r"d:\deploy\c++deploy\cat_dog.jpg"

    print("=" * 70)
    print("Vision Patch Embedding Test with Real Image (cat_dog.jpg)")
    print("=" * 70)

    patch_embed = VisionPatchEmbedding(
        embed_dim=768,
        in_channels=3,
        patch_size=16,
        temporal_patch=2,
        use_bias=True
    )

    print("\n[1] Loading weights from safetensors...")
    patch_embed.load_weights_from_safetensors(model_path)

    print("\n[2] Loading and preprocessing image...")
    test_input = load_and_preprocess_image(
        image_path=image_path,
        target_height=224,
        target_width=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        temporal_patch=2
    )

    print("\n[3] Running forward pass...")
    output = patch_embed.forward_fast(test_input)

    print("\n" + "=" * 70)
    print("[4] Output Results:")
    print("=" * 70)
    print(f"\nOutput 3D Tensor Shape: {output.shape}")
    print(f"  - B (batch):      {output.shape[0]}")
    print(f"  - N (num patches): {output.shape[1]}")
    print(f"  - embed_dim:       {output.shape[2]}")

    Nt = test_input.shape[1] // patch_embed.temporal_patch
    Nh = test_input.shape[3] // patch_embed.patch_size
    Nw = test_input.shape[4] // patch_embed.patch_size
    print(f"\nPatch Breakdown:")
    print(f"  - Nt (temporal patches): {Nt}")
    print(f"  - Nh (height patches):   {Nh}")
    print(f"  - Nw (width patches):    {Nw}")
    print(f"  - Total N = Nt x Nh x Nw = {Nt} x {Nh} x {Nw} = {Nt * Nh * Nw}")

    print(f"\nOutput Statistics:")
    print(f"  - Min:  {output.min():.6f}")
    print(f"  - Max:  {output.max():.6f}")
    print(f"  - Mean: {output.mean():.6f}")
    print(f"  - Std:  {output.std():.6f}")

    print(f"\nFirst Patch Token (first 16 dims):")
    print(f"  {output[0, 0, :16]}")

    print(f"\nCenter Patch Token (patch index {output.shape[1]//2}, first 16 dims):")
    center_idx = output.shape[1] // 2
    print(f"  {output[0, center_idx, :16]}")

    print(f"\nLast Patch Token (first 16 dims):")
    print(f"  {output[0, -1, :16]}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("Output is a valid 3D tensor [B, N, embed_dim] = [1, 196, 768]")
    print("=" * 70)

    return output


if __name__ == "__main__":
    output_tensor = main()
