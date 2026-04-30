import os
import sys
import time
import json
import numpy as np
import torch
from safetensors import safe_open

# Add build directory to path for importing the C++ module
sys.path.insert(0, '/mnt/d/deploy/Qwen3.5_0.8B_Deploy/build')

try:
    import qwen_cuda
    print("Successfully imported qwen_cuda module")
except ImportError as e:
    print(f"Failed to import qwen_cuda: {e}")
    print("Please build the project first: cd build && cmake --build .")
    sys.exit(1)

MODEL_PATH = "/mnt/d/deploy/modules/modles/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
CONFIG_PATH = "/mnt/d/deploy/modules/modles/Qwen3.5-0.8B/config.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def load_weights():
    """Load weights from safetensors file"""
    config = load_config()
    text_config = config.get('text_config', config)
    
    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith("model.language_model."):
                weights[key] = f.get_tensor(key).float().numpy()
    
    return weights, text_config

def create_engine(config):
    """Create CUDA engine with model config"""
    text_config = config.get('text_config', config)
    
    num_layers = text_config['num_hidden_layers']
    hidden_size = text_config['hidden_size']
    intermediate_size = text_config['intermediate_size']
    vocab_size = text_config.get('vocab_size', 151936)
    max_seq_len = text_config.get('max_position_embeddings', 32768)
    
    engine = qwen_cuda.CudaEngineV3(num_layers, hidden_size, intermediate_size, vocab_size, max_seq_len)
    print(f"Created engine: layers={num_layers}, hidden={hidden_size}, intermediate={intermediate_size}, vocab={vocab_size}")
    return engine

def set_layer_weights_from_real(engine, weights, layer_idx, config):
    """Set layer weights from real model weights"""
    text_config = config.get('text_config', config)
    hidden_size = text_config['hidden_size']
    intermediate_size = text_config['intermediate_size']
    
    prefix = f"model.language_model.layers.{layer_idx}"
    
    # Input norm
    input_norm_w = weights.get(f"{prefix}.input_layernorm.weight")
    if input_norm_w is None:
        print(f"Warning: missing input_layernorm for layer {layer_idx}")
        return False
    
    # Post attention norm
    post_norm_w = weights.get(f"{prefix}.post_attention_layernorm.weight")
    if post_norm_w is None:
        print(f"Warning: missing post_attention_layernorm for layer {layer_idx}")
        return False
    
    # MLP weights
    gate_w = weights.get(f"{prefix}.mlp.gate_proj.weight")
    up_w = weights.get(f"{prefix}.mlp.up_proj.weight")
    down_w = weights.get(f"{prefix}.mlp.down_proj.weight")
    
    if gate_w is None or up_w is None or down_w is None:
        print(f"Warning: missing MLP weights for layer {layer_idx}")
        return False
    
    # Transpose MLP weights (safetensors stores as [out, in])
    gate_w = gate_w.T.flatten()
    up_w = up_w.T.flatten()
    down_w = down_w.T.flatten()
    
    # Attention weights - check if full or linear attention
    layer_types = text_config.get('layer_types', [])
    if layer_idx < len(layer_types) and layer_types[layer_idx] == 'full_attention':
        # Full attention weights
        q_w = weights.get(f"{prefix}.self_attn.q_proj.weight")
        k_w = weights.get(f"{prefix}.self_attn.k_proj.weight")
        v_w = weights.get(f"{prefix}.self_attn.v_proj.weight")
        o_w = weights.get(f"{prefix}.self_attn.o_proj.weight")
        q_norm_w = weights.get(f"{prefix}.self_attn.q_norm.weight")
        k_norm_w = weights.get(f"{prefix}.self_attn.k_norm.weight")
        
        if q_w is None or k_w is None or v_w is None or o_w is None:
            print(f"Warning: missing full attention weights for layer {layer_idx}")
            return False
        
        # Transpose and flatten
        q_w = q_w.T.flatten()
        k_w = k_w.T.flatten()
        v_w = v_w.T.flatten()
        o_w = o_w.T.flatten()
        
        # Create flat weights array for set_layer_weights
        # Order: input_norm, post_norm, gate, up, down, q, k, v, q_norm, k_norm, o
        flat_weights = []
        flat_weights.extend(input_norm_w.flatten())
        flat_weights.extend(post_norm_w.flatten())
        flat_weights.extend(gate_w.flatten())
        flat_weights.extend(up_w.flatten())
        flat_weights.extend(down_w.flatten())
        flat_weights.extend(q_w.flatten())
        flat_weights.extend(k_w.flatten())
        flat_weights.extend(v_w.flatten())
        flat_weights.extend(q_norm_w.flatten() if q_norm_w is not None else np.zeros(256, dtype=np.float32))
        flat_weights.extend(k_norm_w.flatten() if k_norm_w is not None else np.zeros(256, dtype=np.float32))
        flat_weights.extend(o_w.flatten())
        
        engine.set_layer_weights(layer_idx, flat_weights)
        print(f"Layer {layer_idx}: Set full attention weights ({len(flat_weights)} floats)")
    else:
        # Linear attention weights
        # For now, use random weights for linear attention (not implemented yet)
        print(f"Layer {layer_idx}: Linear attention - using placeholder weights")
        
        # Create minimal flat weights for the layer (just norms and MLP)
        # The engine expects full attention weights even for linear layers currently
        flat_weights = []
        flat_weights.extend(input_norm_w.flatten())
        flat_weights.extend(post_norm_w.flatten())
        flat_weights.extend(gate_w.flatten())
        flat_weights.extend(up_w.flatten())
        flat_weights.extend(down_w.flatten())
        
        # Add dummy attention weights
        fnh = 8
        qhd = 256
        khd = 256
        hs = hidden_size
        
        flat_weights.extend(np.zeros(fnh * qhd * 2 * hs, dtype=np.float32).flatten())
        flat_weights.extend(np.zeros(2 * khd * hs, dtype=np.float32).flatten())
        flat_weights.extend(np.zeros(2 * khd * hs, dtype=np.float32).flatten())
        flat_weights.extend(np.zeros(khd, dtype=np.float32).flatten())
        flat_weights.extend(np.zeros(khd, dtype=np.float32).flatten())
        flat_weights.extend(np.zeros(hs * fnh * khd, dtype=np.float32).flatten())
        
        engine.set_layer_weights(layer_idx, flat_weights)
    
    return True

def benchmark_with_real_weights():
    """Benchmark with real model weights"""
    print("=" * 70)
    print("Loading real Qwen3.5-0.8B weights...")
    print("=" * 70)
    
    config = load_config()
    weights, text_config = load_weights()
    
    print(f"Loaded {len(weights)} weight tensors")
    print(f"Model config: {text_config['num_hidden_layers']} layers, hidden={text_config['hidden_size']}")
    
    # Create engine
    engine = create_engine(config)
    
    # Set weights for all layers
    num_layers = text_config['num_hidden_layers']
    for i in range(num_layers):
        success = set_layer_weights_from_real(engine, weights, i, config)
        if not success:
            print(f"Failed to set weights for layer {i}")
    
    # Set final norm and lm_head
    final_norm_w = weights.get("model.language_model.norm.weight")
    if final_norm_w is not None:
        engine.set_final_norm_weight(final_norm_w.flatten().tolist())
        print("Set final norm weights")
    
    embed_w = weights.get("model.language_model.embed_tokens.weight")
    if embed_w is not None:
        engine.set_embedding_weight(embed_w.flatten().tolist())
        print("Set embedding weights")
    
    # Benchmark
    print("\n" + "=" * 70)
    print("Starting benchmark with real weights...")
    print("=" * 70)
    
    prefill_tokens = 1024
    decode_tokens = 512
    batch_size = 1
    rounds = 1
    
    # Prefill benchmark
    prefill_ids = list(range(1, prefill_tokens + 1))
    positions = list(range(prefill_tokens))
    
    import numpy as np
    d_output = np.zeros(batch_size * text_config['hidden_size'], dtype=np.float32)
    
    # Warmup
    print("[Warmup...]")
    engine.reset_cache()
    engine.forward_tokens(prefill_ids, d_output, positions)
    
    # Prefill test
    print(f"\n[Prefill benchmark: {prefill_tokens} tokens]")
    engine.reset_cache()
    
    start = time.time()
    engine.forward_tokens(prefill_ids, d_output, positions)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    prefill_ms = (end - start) * 1000
    prefill_tput = prefill_tokens / (prefill_ms / 1000)
    print(f"Prefill time: {prefill_ms:.2f} ms")
    print(f"Prefill throughput: {prefill_tput:.2f} tok/s")
    
    # Decode benchmark
    print(f"\n[Decode benchmark: {decode_tokens} tokens]")
    
    decode_times = []
    for t in range(decode_tokens):
        token_id = (t % 100) + 1
        pos = prefill_tokens + t
        
        start = time.time()
        engine.forward_token(token_id, d_output, pos)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        decode_times.append((end - start) * 1000)
    
    avg_decode_ms = np.mean(decode_times)
    decode_tput = 1.0 / (avg_decode_ms / 1000)
    print(f"Average decode time: {avg_decode_ms:.2f} ms/token")
    print(f"Decode throughput: {decode_tput:.2f} tok/s")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)

if __name__ == "__main__":
    benchmark_with_real_weights()
