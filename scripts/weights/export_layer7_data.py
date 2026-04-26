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
    
    hidden_before_layer7 = outputs.hidden_states[7]
    hidden_after_layer7 = outputs.hidden_states[8]
    
    print(f"Hidden before Layer 7: mean={hidden_before_layer7.mean():.6f}, std={hidden_before_layer7.std():.6f}")
    print(f"Hidden after Layer 7: mean={hidden_after_layer7.mean():.6f}, std={hidden_after_layer7.std():.6f}")
    
    os.makedirs("validation_data_layer7", exist_ok=True)
    
    hidden_before_layer7.float().numpy().tofile("validation_data_layer7/hidden_before.bin")
    hidden_after_layer7.float().numpy().tofile("validation_data_layer7/hidden_after.bin")
    
    layer7 = model.model.layers[7]
    
    ln_weight = layer7.input_layernorm.weight
    ln_weight.float().numpy().tofile("validation_data_layer7/input_ln_weight.bin")
    
    with torch.no_grad():
        ln_out = layer7.input_layernorm(hidden_before_layer7)
        ln_out.float().numpy().tofile("validation_data_layer7/after_ln.bin")
        print(f"After input LN: mean={ln_out.mean():.6f}, std={ln_out.std():.6f}")
    
    attn = layer7.self_attn
    
    q_proj_w = attn.q_proj.weight
    k_proj_w = attn.k_proj.weight
    v_proj_w = attn.v_proj.weight
    o_proj_w = attn.o_proj.weight
    q_norm_w = attn.q_norm.weight
    k_norm_w = attn.k_norm.weight
    
    q_proj_w.float().numpy().tofile("validation_data_layer7/q_proj.bin")
    k_proj_w.float().numpy().tofile("validation_data_layer7/k_proj.bin")
    v_proj_w.float().numpy().tofile("validation_data_layer7/v_proj.bin")
    o_proj_w.float().numpy().tofile("validation_data_layer7/o_proj.bin")
    q_norm_w.float().numpy().tofile("validation_data_layer7/q_norm.bin")
    k_norm_w.float().numpy().tofile("validation_data_layer7/k_norm.bin")
    
    print(f"\nWeight shapes:")
    print(f"  q_proj: {q_proj_w.shape}")
    print(f"  k_proj: {k_proj_w.shape}")
    print(f"  v_proj: {v_proj_w.shape}")
    print(f"  o_proj: {o_proj_w.shape}")
    print(f"  q_norm: {q_norm_w.shape}")
    print(f"  k_norm: {k_norm_w.shape}")
    
    print("\nDone! Exported to validation_data_layer7/")
