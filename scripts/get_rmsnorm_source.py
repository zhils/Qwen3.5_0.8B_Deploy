import torch
import sys

sys.path.insert(0, 'D:/deploy/modules/modles/Qwen3.5-0.8B')

from transformers import AutoModelForCausalLM

model_path = "D:/deploy/modules/modles/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)

layer0 = model.model.layers[0]
ln = layer0.input_layernorm

print("Qwen3_5RMSNorm source code:")
import inspect
print(inspect.getsource(ln.__class__))
