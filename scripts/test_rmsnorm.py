import torch
import numpy as np
import sys

sys.path.insert(0, 'D:/deploy/modules/modles/Qwen3.5-0.8B')

from transformers import AutoModelForCausalLM

model_path = "D:/deploy/modules/modles/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)

input_ids = torch.tensor([[151644]])

with torch.no_grad():
    embed = model.model.embed_tokens(input_ids)
    print(f"Embed: mean={embed.mean():.6f}, std={embed.std():.6f}")
    
    layer0 = model.model.layers[0]
    
    ln_weight = layer0.input_layernorm.weight
    print(f"\nLayerNorm weight: mean={ln_weight.mean():.6f}, std={ln_weight.std():.6f}")
    print(f"LayerNorm weight[0:5]: {ln_weight[0:5].tolist()}")
    
    ln_out = layer0.input_layernorm(embed)
    print(f"\nAfter input_layernorm: mean={ln_out.mean():.6f}, std={ln_out.std():.6f}")
    print(f"ln_out[0,0,0:5]: {ln_out[0,0,0:5].tolist()}")
    
    embed_np = embed.float().numpy().flatten()
    weight_np = ln_weight.float().numpy()
    
    sum_sq = np.sum(embed_np ** 2)
    rms = np.sqrt(sum_sq / len(embed_np) + 1e-6)
    print(f"\nManual RMS: {rms:.6f}")
    
    manual_out = (embed_np / rms) * (1.0 + weight_np)
    print(f"Manual ln_out[0:5]: {manual_out[0:5].tolist()}")
    
    ln_weight.float().numpy().tofile("validation_data_detailed/layer0_ln_weight.bin")
