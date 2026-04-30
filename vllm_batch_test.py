#!/usr/bin/env python3
"""
vLLM Batch Performance Test - Simplified Version
Usage: python vllm_batch_test.py <batch_size> [batch_size ...]
"""

import sys
import time
import argparse
import subprocess
from vllm import LLM, SamplingParams

MODEL_PATH = "/mnt/d/deploy/modules/modles/Qwen3.5-0.8B"

def get_gpu_memory():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(',')
        return float(used) / 1024, float(total) / 1024
    except:
        return 0, 0

def test_vllm_batch(batch_size, prefill_tokens=1024, decode_tokens=512):
    print(f"\n{'='*70}")
    print(f"  vLLM Batch Performance Test (batch_size={batch_size})")
    print(f"{'='*70}")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=batch_size,
        gpu_memory_utilization=0.80,
        block_size=32,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    prompts = ["hello world " * 32] * batch_size

    sampling_params = SamplingParams(
        max_tokens=decode_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    print(f"\n[Testing with {batch_size} parallel sequences]")
    print(f"  Prefill tokens per seq: {prefill_tokens}")
    print(f"  Decode tokens per seq: {decode_tokens}")

    mem_before = get_gpu_memory()

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start

    total_tokens = prefill_tokens * batch_size + decode_tokens * batch_size
    total_prefill = prefill_tokens * batch_size
    total_decode = decode_tokens * batch_size

    mem_after = get_gpu_memory()
    mem_used = mem_after[0]

    print(f"\n--- Results ---")
    print(f"  Total time:           {total_time*1000:.1f} ms")
    print(f"  E2E throughput:       {total_tokens/total_time:.1f} tokens/sec")
    print(f"  GPU Memory used:     {mem_used:.1f} GiB")

    return {
        'batch_size': batch_size,
        'total_time_ms': total_time * 1000,
        'e2e_throughput': total_tokens / total_time,
        'memory_gb': mem_used,
    }

def main():
    parser = argparse.ArgumentParser(description='vLLM Batch Performance Test')
    parser.add_argument('batch_sizes', type=int, nargs='+',
                        help='Batch sizes to test (e.g., 1 8 16 32 64 128)')
    parser.add_argument('--prefill', type=int, default=1024,
                        help='Prefill tokens per sequence')
    parser.add_argument('--decode', type=int, default=512,
                        help='Decode tokens per sequence')
    args = parser.parse_args()

    results = []
    for bs in args.batch_sizes:
        try:
            result = test_vllm_batch(bs, args.prefill, args.decode)
            results.append(result)
        except Exception as e:
            print(f"Error testing batch_size={bs}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    print(f"{'Batch':>6} | {'Total (ms)':>12} | {'E2E (tok/s)':>14} | {'Mem (GiB)':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['batch_size']:>6} | {r['total_time_ms']:>12.1f} | {r['e2e_throughput']:>14.1f} | {r['memory_gb']:>10.1f}")

if __name__ == '__main__':
    main()
