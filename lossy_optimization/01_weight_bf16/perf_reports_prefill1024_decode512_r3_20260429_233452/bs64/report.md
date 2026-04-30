# 01_weight_bf16 性能报告 (batch=64)

- Prefill tokens: 1024
- Decode tokens: 512
- Rounds per run: 3
- Repeats: 3

## 平均结果 (3次)

- Prefill throughput: 28215.545 tok/s (std 179.679)
- Decode throughput: 123145.656 tok/s (std 11910.997)
- E2E throughput: 37945.573 tok/s (std 582.034)
- TTFT: 2322.785 ms (std 14.740)
- TPOT: 0.524 ms/token (std 0.049)
- Memory: 4428.562 MB (std 0.000)

## 单次结果

- run1: prefill 28043.621, decode 111776.646, e2e 37376.715, ttft 2336.931, tpot 0.573, mem 4428.562
- run2: prefill 28139.464, decode 118066.007, e2e 37714.788, ttft 2328.971, tpot 0.542, mem 4428.562
- run3: prefill 28463.551, decode 139594.314, e2e 38745.216, ttft 2302.453, tpot 0.458, mem 4428.562
