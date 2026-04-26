import numpy as np
import os

d = "debug_pytorch"
out_d = "debug_pytorch_bin"
os.makedirs(out_d, exist_ok=True)

files = [
    "mixed_qkv", "conv_out", "q", "k", "v",
    "a", "beta", "g", "q_scaled", "k_normed",
    "core_attn_out", "z", "final_output"
]

for f in files:
    npy_path = os.path.join(d, f + ".npy")
    bin_path = os.path.join(out_d, f + ".bin")
    data = np.load(npy_path).astype(np.float32)
    data.tofile(bin_path)
    print(f"{f}: {data.shape} -> {bin_path} ({os.path.getsize(bin_path)} bytes)")

print(f"\nAll files converted to {out_d}/")
