import torch
import numpy as np
import os
import sys

sys.path.insert(0, 'D:/deploy/modules/modles/Qwen3.5-0.8B')

from transformers import AutoModelForCausalLM, AutoTokenizer

def export_detailed_layer0():
    model_path = "D:/deploy/modules/modles/Qwen3.5-0.8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    
    input_ids = torch.tensor([[151644]])
    
    output_dir = "validation_data_detailed"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        embed = model.model.embed_tokens(input_ids)
        embed.float().numpy().tofile(f"{output_dir}/embed.bin")
        print(f"Embed: shape={embed.shape}, mean={embed.mean():.6f}")
        
        layer0 = model.model.layers[0]
        
        ln_out = layer0.input_layernorm(embed)
        ln_out.float().numpy().tofile(f"{output_dir}/layer0_after_input_norm.bin")
        print(f"After input_layernorm: mean={ln_out.mean():.6f}, std={ln_out.std():.6f}")
        
        attn_out = layer0.linear_attn(ln_out)
        attn_hidden = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        attn_hidden.float().numpy().tofile(f"{output_dir}/layer0_after_attn.bin")
        print(f"After attention: mean={attn_hidden.mean():.6f}, std={attn_hidden.std():.6f}")
        
        residual1 = embed + attn_hidden
        residual1.float().numpy().tofile(f"{output_dir}/layer0_residual1.bin")
        print(f"Residual1: mean={residual1.mean():.6f}")
        
        post_ln_out = layer0.post_attention_layernorm(residual1)
        post_ln_out.float().numpy().tofile(f"{output_dir}/layer0_after_post_norm.bin")
        print(f"After post_layernorm: mean={post_ln_out.mean():.6f}")
        
        mlp_out = layer0.mlp(post_ln_out)
        mlp_out.float().numpy().tofile(f"{output_dir}/layer0_after_mlp.bin")
        print(f"After MLP: mean={mlp_out.mean():.6f}, std={mlp_out.std():.6f}")
        
        final_out = residual1 + mlp_out
        final_out.float().numpy().tofile(f"{output_dir}/layer0_final.bin")
        print(f"Final output: mean={final_out.mean():.6f}, std={final_out.std():.6f}")

if __name__ == "__main__":
    export_detailed_layer0()
