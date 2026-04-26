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
    
    os.makedirs("validation_data_all_layers", exist_ok=True)
    
    for i, h in enumerate(outputs.hidden_states):
        h.float().numpy().tofile(f"validation_data_all_layers/hidden_{i}.bin")
        if i == 0:
            print(f"hidden_{i} (embedding): mean={h.mean():.6f}, std={h.std():.6f}")
        else:
            print(f"hidden_{i} (after layer {i-1}): mean={h.mean():.6f}, std={h.std():.6f}")
    
    input_ids.numpy().tofile("validation_data_all_layers/input_ids.bin")
    
    print("\nDone!")
