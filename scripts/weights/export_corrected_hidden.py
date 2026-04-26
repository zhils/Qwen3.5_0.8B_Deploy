import torch
import numpy as np
import sys
import os

sys.path.insert(0, 'D:/deploy/modules/modles/Qwen3.5-0.8B')

from transformers import AutoModelForCausalLM

model_path = "D:/deploy/modules/modles/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)

input_ids = torch.tensor([[151644]])

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    
    os.makedirs("validation_data_corrected", exist_ok=True)
    
    embed_out = outputs.hidden_states[0]
    embed_out.float().numpy().tofile("validation_data_corrected/embedding_output.bin")
    print(f"embedding_output: mean={embed_out.mean():.6f}, std={embed_out.std():.6f}")
    
    for i, h in enumerate(outputs.hidden_states):
        if i == 0:
            h.float().numpy().tofile(f"validation_data_corrected/hidden_{i}.bin")
            print(f"hidden_{i} (embedding): mean={h.mean():.6f}, std={h.std():.6f}")
        elif i < 24:
            h.float().numpy().tofile(f"validation_data_corrected/hidden_{i}.bin")
            print(f"hidden_{i} (after layer {i-1}): mean={h.mean():.6f}, std={h.std():.6f}")
        else:
            print(f"hidden_{i} (after final norm): mean={h.mean():.6f}, std={h.std():.6f}")
    
    layer23 = model.model.layers[23]
    hidden_23 = outputs.hidden_states[23]
    
    ln_out = layer23.input_layernorm(hidden_23)
    position_ids = torch.tensor([[0]])
    position_embeddings = model.model.rotary_emb(hidden_23, position_ids)
    attn_out, _ = layer23.self_attn(ln_out, position_embeddings=position_embeddings, attention_mask=None, past_key_values=None)
    residual1 = hidden_23 + attn_out
    post_ln_out = layer23.post_attention_layernorm(residual1)
    mlp_out = layer23.mlp(post_ln_out)
    layer23_output = residual1 + mlp_out
    
    layer23_output.float().numpy().tofile("validation_data_corrected/hidden_24.bin")
    print(f"hidden_24 (layer 23 output, before final norm): mean={layer23_output.mean():.6f}, std={layer23_output.std():.6f}")
    
    input_ids.numpy().tofile("validation_data_corrected/input_ids.bin")
    
    final_norm_weight = model.model.norm.weight
    final_norm_weight.float().numpy().tofile("validation_data_corrected/final_norm_weight.bin")
    print(f"\nFinal norm weight: shape={final_norm_weight.shape}")
    
    print("\nDone!")
