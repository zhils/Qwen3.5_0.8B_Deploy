#!/usr/bin/env python3
"""
使用官方 HuggingFace Qwen3.5 的 image_processor 对图像做与推理完全一致的处理，
并导出二进制供 C++ 侧对比（pixel_values / image_grid_thw）。

依赖:
  pip install transformers torch pillow safetensors accelerate

用法:
  set QWEN_MODEL=Qwen/Qwen3.5-0.8B
  python export_qwen35_image_processor_ref.py path/to/image.jpg output_dir/

若本地已有模型目录，可:
  set QWEN_MODEL=D:/path/to/Qwen3.5-0.8B
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: export_qwen35_image_processor_ref.py <image.jpg> <output_dir>")
        return 1

    image_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    model_id = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-0.8B")

    from transformers import AutoImageProcessor
    from PIL import Image

    proc = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    img = Image.open(image_path).convert("RGB")
    batch = proc(images=img, return_tensors="np")

    pv = np.asarray(batch["pixel_values"], dtype=np.float32)
    thw = np.asarray(batch["image_grid_thw"])

    pv_path = os.path.join(out_dir, "pixel_values_hf_ref.bin")
    thw_path = os.path.join(out_dir, "image_grid_thw_hf_ref.bin")
    meta_path = os.path.join(out_dir, "image_processor_meta.json")

    pv.tofile(pv_path)
    thw.tofile(thw_path)

    meta = {
        "model_id": model_id,
        "pixel_values_shape": list(pv.shape),
        "image_grid_thw": thw.tolist(),
        "pixel_values_min": float(pv.min()),
        "pixel_values_max": float(pv.max()),
        "pixel_values_mean": float(pv.mean()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"pixel_values: shape={pv.shape} -> {pv_path}")
    print(f"image_grid_thw: {thw} -> {thw_path}")
    print(f"meta -> {meta_path}")
    print("\n说明: transformers 在 _preprocess 中对张量做了 view/permute/flatten，")
    print("与手写 Conv3d patch 的像素遍历顺序可能不同；若要对齐到比特级，")
    print("请用本脚本导出的 pixel_values 与 C++ patch_embed 权重按 Linear 关系校验，")
    print("或以 PyTorch 视觉前向输出为金标准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
