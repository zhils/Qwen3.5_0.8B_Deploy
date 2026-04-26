import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from safetensors import safe_open


def export_embedding_weights(safetensors_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading weights from: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt") as f:
        embed_key = "model.language_model.embed_tokens.weight"
        embed_weight = f.get_tensor(embed_key).float().numpy()

        print(f"  embed_tokens.weight: {embed_weight.shape}")

        output_path = os.path.join(output_dir, "embed_tokens.bin")
        embed_weight.tofile(output_path)
        print(f"  Saved to: {output_path}")

        print(f"\n  Stats:")
        print(f"    vocab_size: {embed_weight.shape[0]}")
        print(f"    hidden_size: {embed_weight.shape[1]}")
        print(f"    min: {embed_weight.min():.6f}")
        print(f"    max: {embed_weight.max():.6f}")
        print(f"    mean: {embed_weight.mean():.6f}")

    return embed_weight


def export_lm_head_weights(safetensors_path: str, output_dir: str, embed_weight: np.ndarray):
    with safe_open(safetensors_path, framework="pt") as f:
        lm_head_key = "model.language_model.lm_head.weight"
        keys_list = list(f.keys())
        if lm_head_key in keys_list:
            lm_head_weight = f.get_tensor(lm_head_key).float().numpy()
            print(f"\n  lm_head.weight: {lm_head_weight.shape}")

            output_path = os.path.join(output_dir, "lm_head.bin")
            lm_head_weight.tofile(output_path)
            print(f"  Saved to: {output_path}")

            is_tied = np.allclose(lm_head_weight, embed_weight)
            print(f"  Tied with embed_tokens: {is_tied}")
        else:
            print(f"\n  lm_head.weight not found (tied with embed_tokens)")

    return embed_weight


def export_final_norm_weights(safetensors_path: str, output_dir: str):
    with safe_open(safetensors_path, framework="pt") as f:
        norm_key = "model.language_model.norm.weight"
        norm_weight = f.get_tensor(norm_key).float().numpy()

        print(f"\n  final_norm.weight: {norm_weight.shape}")

        output_path = os.path.join(output_dir, "final_norm.bin")
        norm_weight.tofile(output_path)
        print(f"  Saved to: {output_path}")

    return norm_weight


def test_embedding_lookup(embed_weight: np.ndarray):
    print("\n" + "=" * 60)
    print("Testing embedding lookup")
    print("=" * 60)

    test_token_ids = [0, 1, 100, 1000, 248319]

    for token_id in test_token_ids:
        embedding = embed_weight[token_id]
        print(f"\n  Token {token_id}:")
        print(f"    First 8 values: {embedding[:8]}")
        print(f"    Norm: {np.linalg.norm(embedding):.6f}")


def main():
    model_path = r"D:\deploy\modules\modles\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors"
    output_dir = r"d:\deploy\c++deploy\weights\language"

    print("=" * 60)
    print("Exporting Token Embedding weights for C++ validation")
    print("=" * 60)

    embed_weight = export_embedding_weights(model_path, output_dir)
    export_lm_head_weights(model_path, output_dir, embed_weight)
    export_final_norm_weights(model_path, output_dir)

    test_embedding_lookup(embed_weight)

    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
