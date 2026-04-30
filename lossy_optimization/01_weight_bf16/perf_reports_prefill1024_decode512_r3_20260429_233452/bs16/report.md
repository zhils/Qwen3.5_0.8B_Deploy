# 01_weight_bf16 性能报告 (batch=16)

- Prefill tokens: 1024
- Decode tokens: 512
- Rounds per run: 3
- Repeats: 3

## 平均结果 (3次)

- Prefill throughput: 5791.401 tok/s (std 1381.364)
- Decode throughput: 81262.912 tok/s (std 14061.065)
- E2E throughput: 8386.783 tok/s (std 1983.085)
- TTFT: 3036.762 ms (std 871.002)
- TPOT: 0.204 ms/token (std 0.040)
- Memory: 4370.562 MB (std 0.000)

## 单次结果

- run1: prefill 3838.411, decode 61378.354, e2e 5583.043, ttft 4268.433, tpot 0.261, mem 4370.562
- run2: prefill 6808.184, decode 91358.806, e2e 9845.429, ttft 2406.515, tpot 0.175, mem 4370.562
- run3: prefill 6727.607, decode 91051.577, e2e 9731.876, ttft 2435.339, tpot 0.176, mem 4370.562
