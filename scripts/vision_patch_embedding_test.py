import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open
from typing import Tuple


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

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("Weights not loaded. Call load_weights_from_safetensors first.")

        if x.ndim != 5:
            raise ValueError(f"Input must be 5D [B, T, C, H, W], got {x.ndim}D")

        B, T, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(f"Input channels mismatch. Expected {self.in_channels}, got {C}")
        if T % self.temporal_patch != 0:
            raise ValueError(f"T must be divisible by temporal_patch ({self.temporal_patch})")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"H and W must be divisible by patch_size ({self.patch_size})")

        Nt = T // self.temporal_patch
        Nh = H // self.patch_size
        Nw = W // self.patch_size
        N = Nt * Nh * Nw

        output = np.zeros((B, N, self.embed_dim), dtype=np.float32)

        for bi in range(B):
            for tt in range(Nt):
                for yy in range(Nh):
                    for xx in range(Nw):
                        token_idx = (tt * Nh + yy) * Nw + xx

                        for od in range(self.embed_dim):
                            acc = self.bias[od] if self.use_bias else 0.0

                            for ic in range(self.in_channels):
                                for kt in range(self.temporal_patch):
                                    t_in = tt * self.temporal_patch + kt
                                    for ky in range(self.patch_size):
                                        y_in = yy * self.patch_size + ky
                                        for kx in range(self.patch_size):
                                            x_in = xx * self.patch_size + kx
                                            xv = x[bi, t_in, ic, y_in, x_in]
                                            wv = self.weight[od, ic, kt, ky, kx]
                                            acc += xv * wv

                            output[bi, token_idx, od] = acc

        return output

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


def create_test_input(batch: int = 1, frames: int = 2, channels: int = 3,
                      height: int = 224, width: int = 224) -> np.ndarray:
    return np.random.randn(batch, frames, channels, height, width).astype(np.float32)


def main():
    model_path = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"

    print("=" * 60)
    print("Vision Patch Embedding Test")
    print("=" * 60)

    patch_embed = VisionPatchEmbedding(
        embed_dim=768,
        in_channels=3,
        patch_size=16,
        temporal_patch=2,
        use_bias=True
    )

    print("\n[1] Loading weights from safetensors...")
    patch_embed.load_weights_from_safetensors(model_path)

    print("\n[2] Creating test input...")
    test_input = create_test_input(batch=1, frames=2, channels=3, height=224, width=224)
    print(f"Input shape: {test_input.shape} [B, T, C, H, W]")

    print("\n[3] Running forward pass (optimized)...")
    output = patch_embed.forward_fast(test_input)

    print("\n[4] Output Results:")
    print(f"Output shape: {output.shape} [B, N, embed_dim]")
    print(f"  - B (batch): {output.shape[0]}")
    print(f"  - N (num patches): {output.shape[1]}")
    print(f"  - embed_dim: {output.shape[2]}")

    Nt = test_input.shape[1] // patch_embed.temporal_patch
    Nh = test_input.shape[3] // patch_embed.patch_size
    Nw = test_input.shape[4] // patch_embed.patch_size
    print(f"\nPatch breakdown:")
    print(f"  - Nt (temporal patches): {Nt}")
    print(f"  - Nh (height patches): {Nh}")
    print(f"  - Nw (width patches): {Nw}")
    print(f"  - Total N = Nt x Nh x Nw = {Nt} x {Nh} x {Nw} = {Nt * Nh * Nw}")

    print(f"\nOutput statistics:")
    print(f"  - Min: {output.min():.6f}")
    print(f"  - Max: {output.max():.6f}")
    print(f"  - Mean: {output.mean():.6f}")
    print(f"  - Std: {output.std():.6f}")

    print(f"\nFirst token (first 10 dims):")
    print(f"  {output[0, 0, :10]}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return output


if __name__ == "__main__":
    output_tensor = main()
