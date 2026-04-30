#!/bin/bash
# Lossy Optimization Performance Test Script
# Tests all variants with batch_size = 1, 8, 16, 32, 64, 128
# Metrics: Prefill throughput, Decode throughput, TTFT, TPOT, Memory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOSSY_DIR="$SCRIPT_DIR"

BATCH_SIZES=(1 8 16 32 64 128)
PREFILL_TOKENS=1024
DECODE_TOKENS=512
NUM_ROUNDS=5

echo "========================================"
echo "Lossy Optimization Performance Test"
echo "========================================"
echo ""
echo "Test Configuration:"
echo "  Prefill tokens: $PREFILL_TOKENS"
echo "  Decode tokens: $DECODE_TOKENS"
echo "  Rounds: $NUM_ROUNDS"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo ""

VARIANTS=(
    "01_weight_bf16"
    "02_kv_cache_fp16"
    "03_kv_cache_int8"
    "04_weight_int8"
    "05_weight_int4"
    "06_full_fp16"
    "07_full_int8"
    "08_flash_to_linear"
    "09_linear_to_flash"
)

run_variant_test() {
    local variant_name=$1
    local variant_dir="$LOSSY_DIR/$variant_name"
    local build_dir="$variant_dir/build"

    echo "========================================"
    echo "Testing: $variant_name"
    echo "========================================"

    if [ ! -d "$build_dir" ]; then
        echo "Build directory not found: $build_dir"
        echo "Skipping..."
        return 1
    fi

    if [ ! -f "$build_dir/performance_test" ]; then
        echo "performance_test not found in $build_dir"
        echo "Skipping..."
        return 1
    fi

    cd "$build_dir"

    for batch_size in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "--- Batch size: $batch_size ---"
        ./performance_test $PREFILL_TOKENS $DECODE_TOKENS $NUM_ROUNDS $batch_size 2>&1 | tee "test_bs${batch_size}.log"

        if [ -f "test_bs${batch_size}.log" ]; then
            grep -E "(Prefill|Decode|E2E|Memory|TTFT|TPOT)" "test_bs${batch_size}.log" || true
        fi
    done

    cd "$LOSSY_DIR"
    echo ""
}

echo "Starting tests at $(date)"
echo ""

for variant in "${VARIANTS[@]}"; do
    run_variant_test "$variant" || true
    echo ""
done

echo ""
echo "========================================"
echo "All tests completed at $(date)"
echo "========================================"
