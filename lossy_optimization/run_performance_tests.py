#!/usr/bin/env python3
"""
Lossy Optimization Performance Test Runner
Tests all variants across different batch sizes and saves results to CSV and MD files.
"""

import subprocess
import os
import re
import csv
from pathlib import Path

PREFILL_TOKENS = 1024
DECODE_TOKENS = 512
ROUNDS = 5
BATCH_SIZES = [1, 8, 16, 32, 64, 128]

VARIANTS = [
    "01_weight_bf16",
    "02_kv_cache_fp16",
    "03_kv_cache_int8",
    "04_weight_int8",
    "05_weight_int4",
    "06_full_fp16",
    "07_full_int8",
    "08_flash_to_linear",
    "09_linear_to_flash",
]

BASE_DIR = Path(__file__).parent

def parse_performance_output(output: str) -> dict:
    """Parse performance_test output and extract metrics."""
    result = {
        "prefill_time_ms": 0,
        "prefill_single_thrpt": 0,
        "prefill_batch_thrpt": 0,
        "decode_tpot_ms": 0,
        "decode_single_thrpt": 0,
        "decode_batch_thrpt": 0,
        "e2e_time_ms": 0,
        "e2e_thrpt": 0,
        "vram_mb": 0,
    }
    
    patterns = {
        "prefill_time_ms": r"Total time:\s+([\d.]+)\s+ms",
        "prefill_single_thrpt": r"Single thrpt:\s+([\d.]+)\s+tokens/sec",
        "prefill_batch_thrpt": r"Batch thrpt:\s+([\d.]+)\s+tokens/sec",
        "decode_tpot_ms": r"TPOT:\s+([\d.]+)\s+ms/token",
        "e2e_time_ms": r"Total time:\s+([\d.]+)\s+ms",
        "e2e_thrpt": r"E2E thrpt:\s+([\d.]+)\s+tokens/sec",
        "vram_mb": r"GPU VRAM used:\s+([\d.]+)\s+MB",
    }
    
    lines = output.split("\n")
    in_prefill = False
    in_decode = False
    in_e2e = False
    
    for line in lines:
        if "--- Prefill" in line:
            in_prefill = True
            in_decode = False
            in_e2e = False
        elif "--- Decode" in line:
            in_prefill = False
            in_decode = True
            in_e2e = False
        elif "--- E2E" in line:
            in_prefill = False
            in_decode = False
            in_e2e = True
        elif "--- Memory" in line:
            in_prefill = False
            in_decode = False
            in_e2e = False
        
        if in_prefill and "Total time:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["prefill_time_ms"] = float(match.group(1))
        elif in_prefill and "Single thrpt:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["prefill_single_thrpt"] = float(match.group(1))
        elif in_prefill and "Batch thrpt:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["prefill_batch_thrpt"] = float(match.group(1))
        elif in_decode and "TPOT:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["decode_tpot_ms"] = float(match.group(1))
        elif in_decode and "Single thrpt:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["decode_single_thrpt"] = float(match.group(1))
        elif in_decode and "Batch thrpt:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["decode_batch_thrpt"] = float(match.group(1))
        elif in_e2e and "Total time:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["e2e_time_ms"] = float(match.group(1))
        elif in_e2e and "E2E thrpt:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["e2e_thrpt"] = float(match.group(1))
        elif "GPU VRAM used:" in line:
            match = re.search(r"([\d.]+)", line)
            if match:
                result["vram_mb"] = float(match.group(1))
    
    return result

def run_test(variant: str, batch_size: int) -> dict:
    """Run performance test for a variant and batch size."""
    build_dir = BASE_DIR / variant / "build"
    executable = build_dir / "performance_test"
    
    if not executable.exists():
        print(f"  [ERROR] Executable not found: {executable}")
        return None
    
    cmd = [str(executable), str(PREFILL_TOKENS), str(DECODE_TOKENS), str(ROUNDS), str(batch_size)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        if result.returncode != 0:
            print(f"  [ERROR] Test failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            return None
        
        return parse_performance_output(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Test timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return None

def save_results_csv(variant: str, results: list):
    """Save results to CSV file."""
    csv_path = BASE_DIR / variant / "performance_results.csv"
    
    fieldnames = [
        "batch_size", "prefill_time_ms", "prefill_single_thrpt", "prefill_batch_thrpt",
        "decode_tpot_ms", "decode_single_thrpt", "decode_batch_thrpt",
        "e2e_time_ms", "e2e_thrpt", "vram_mb"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r:
                writer.writerow(r)
    
    print(f"  Saved CSV: {csv_path}")

def save_results_md(variant: str, results: list):
    """Save results to Markdown file."""
    md_path = BASE_DIR / variant / "PERFORMANCE.md"
    
    with open(md_path, "w") as f:
        f.write(f"# {variant} Performance Results\n\n")
        f.write(f"**Test Configuration:**\n")
        f.write(f"- Prefill tokens: {PREFILL_TOKENS}\n")
        f.write(f"- Decode tokens: {DECODE_TOKENS}\n")
        f.write(f"- Rounds per test: {ROUNDS}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Batch | Prefill (tok/s) | Decode (tok/s) | E2E (tok/s) | VRAM (MB) |\n")
        f.write("|-------|-----------------|----------------|-------------|----------|\n")
        
        for r in results:
            if r:
                f.write(f"| {r['batch_size']} | {r['prefill_single_thrpt']:.1f} | "
                       f"{r['decode_single_thrpt']:.1f} | {r['e2e_thrpt']:.1f} | "
                       f"{r['vram_mb']:.1f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        f.write("| Batch | Prefill Time (ms) | Prefill Single | Prefill Batch | "
               f"Decode TPOT (ms) | Decode Single | Decode Batch | E2E Time (ms) | E2E Thrpt |\n")
        f.write("|-------|-------------------|----------------|---------------|"
               "|-----------------|---------------|--------------|---------------|----------|\n")
        
        for r in results:
            if r:
                f.write(f"| {r['batch_size']} | {r['prefill_time_ms']:.1f} | "
                       f"{r['prefill_single_thrpt']:.1f} | {r['prefill_batch_thrpt']:.1f} | "
                       f"{r['decode_tpot_ms']:.3f} | {r['decode_single_thrpt']:.1f} | "
                       f"{r['decode_batch_thrpt']:.1f} | {r['e2e_time_ms']:.1f} | "
                       f"{r['e2e_thrpt']:.1f} |\n")
    
    print(f"  Saved MD: {md_path}")

def main():
    print("=" * 70)
    print("Lossy Optimization Performance Test Runner")
    print("=" * 70)
    print(f"Prefill tokens: {PREFILL_TOKENS}")
    print(f"Decode tokens: {DECODE_TOKENS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print("=" * 70)
    
    all_results = {}
    
    for variant in VARIANTS:
        print(f"\n[{variant}]")
        variant_results = []
        
        for batch_size in BATCH_SIZES:
            print(f"  Testing batch_size={batch_size}...", end=" ", flush=True)
            result = run_test(variant, batch_size)
            
            if result:
                result["batch_size"] = batch_size
                variant_results.append(result)
                print(f"Prefill: {result['prefill_single_thrpt']:.1f} tok/s, "
                      f"Decode: {result['decode_single_thrpt']:.1f} tok/s")
            else:
                print("FAILED")
                variant_results.append({"batch_size": batch_size})
        
        all_results[variant] = variant_results
        
        if any(r for r in variant_results if r and len(r) > 1):
            save_results_csv(variant, variant_results)
            save_results_md(variant, variant_results)
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
