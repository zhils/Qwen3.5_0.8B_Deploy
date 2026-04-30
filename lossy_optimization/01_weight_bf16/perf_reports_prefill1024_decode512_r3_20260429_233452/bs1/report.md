# 01_weight_bf16 性能报告 (batch=1)

- Prefill tokens: 1024
- Decode tokens: 512
- Rounds per run: 3
- Repeats: 3

## 平均结果 (3次)

- Prefill throughput: 419.443 tok/s (std 8.524)
- Decode throughput: 8809.800 tok/s (std 62.824)
- E2E throughput: 614.531 tok/s (std 12.242)
- TTFT: 2442.327 ms (std 48.931)
- TPOT: 0.114 ms/token (std 0.001)
- Memory: 4370.562 MB (std 0.000)

## 单次结果

- run1: prefill 413.469, decode 8858.914, e2e 606.060, ttft 2476.609, tpot 0.113, mem 4370.562
- run2: prefill 413.363, decode 8721.124, e2e 605.690, ttft 2477.242, tpot 0.115, mem 4370.562
- run3: prefill 431.498, decode 8849.361, e2e 631.842, ttft 2373.129, tpot 0.113, mem 4370.562
