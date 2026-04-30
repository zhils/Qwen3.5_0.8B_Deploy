#!/bin/bash
# Lossy Optimization Performance Test Runner
# Tests all variants across different batch sizes

PREFILL_TOKENS=1024
DECODE_TOKENS=512
ROUNDS=5
BATCH_SIZES="1 8 16 32 64 128"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Testing: $variant"
    echo "========================================================================"
    
    BUILD_DIR="$SCRIPT_DIR/$variant/build"
    EXECUTABLE="$BUILD_DIR/performance_test"
    CSV_FILE="$SCRIPT_DIR/$variant/performance_results.csv"
    MD_FILE="$SCRIPT_DIR/$variant/PERFORMANCE.md"
    
    if [ ! -f "$EXECUTABLE" ]; then
        echo "[ERROR] Executable not found: $EXECUTABLE"
        continue
    fi
    
    # Initialize CSV
    echo "batch_size,prefill_time_ms,prefill_single_thrpt,prefill_batch_thrpt,decode_tpot_ms,decode_single_thrpt,decode_batch_thrpt,e2e_time_ms,e2e_thrpt,vram_mb" > "$CSV_FILE"
    
    # Initialize MD
    echo "# $variant Performance Results" > "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo "**Test Configuration:**" >> "$MD_FILE"
    echo "- Prefill tokens: $PREFILL_TOKENS" >> "$MD_FILE"
    echo "- Decode tokens: $DECODE_TOKENS" >> "$MD_FILE"
    echo "- Rounds per test: $ROUNDS" >> "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo "## Performance Summary" >> "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo "| Batch | Prefill (tok/s) | Decode (tok/s) | E2E (tok/s) | VRAM (MB) |" >> "$MD_FILE"
    echo "|-------|-----------------|----------------|-------------|----------|" >> "$MD_FILE"
    
    for batch_size in $BATCH_SIZES; do
        echo ""
        echo "  Testing batch_size=$batch_size..."
        
        OUTPUT=$("$EXECUTABLE" $PREFILL_TOKENS $DECODE_TOKENS $ROUNDS $batch_size 2>&1)
        
        if [ $? -ne 0 ]; then
            echo "  [ERROR] Test failed"
            echo "$batch_size,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$CSV_FILE"
            continue
        fi
        
        # Parse output
        PREFILL_TIME=$(echo "$OUTPUT" | grep -A3 "Prefill" | grep "Total time" | awk '{print $3}')
        PREFILL_SINGLE=$(echo "$OUTPUT" | grep -A5 "Prefill" | grep "Single thrpt" | awk '{print $3}')
        PREFILL_BATCH=$(echo "$OUTPUT" | grep -A5 "Prefill" | grep "Batch thrpt" | awk '{print $3}')
        DECODE_TPOT=$(echo "$OUTPUT" | grep -A3 "Decode" | grep "TPOT" | awk '{print $2}')
        DECODE_SINGLE=$(echo "$OUTPUT" | grep -A5 "Decode" | grep "Single thrpt" | awk '{print $3}')
        DECODE_BATCH=$(echo "$OUTPUT" | grep -A5 "Decode" | grep "Batch thrpt" | awk '{print $3}')
        E2E_TIME=$(echo "$OUTPUT" | grep -A3 "E2E" | grep "Total time" | awk '{print $3}')
        E2E_THRPT=$(echo "$OUTPUT" | grep -A3 "E2E" | grep "E2E thrpt" | awk '{print $3}')
        VRAM=$(echo "$OUTPUT" | grep "GPU VRAM used" | awk '{print $4}')
        
        echo "  Prefill: ${PREFILL_SINGLE} tok/s, Decode: ${DECODE_SINGLE} tok/s, E2E: ${E2E_THRPT} tok/s"
        
        # Append to CSV
        echo "$batch_size,$PREFILL_TIME,$PREFILL_SINGLE,$PREFILL_BATCH,$DECODE_TPOT,$DECODE_SINGLE,$DECODE_BATCH,$E2E_TIME,$E2E_THRPT,$VRAM" >> "$CSV_FILE"
        
        # Append to MD
        echo "| $batch_size | ${PREFILL_SINGLE} | ${DECODE_SINGLE} | ${E2E_THRPT} | ${VRAM} |" >> "$MD_FILE"
    done
    
    echo ""
    echo "  Saved: $CSV_FILE"
    echo "  Saved: $MD_FILE"
done

echo ""
echo "========================================================================"
echo "All tests completed!"
echo "========================================================================"
