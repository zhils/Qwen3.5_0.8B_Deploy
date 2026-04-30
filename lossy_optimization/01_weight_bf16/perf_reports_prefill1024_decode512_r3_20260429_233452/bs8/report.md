# 01_weight_bf16 性能报告 (batch=8)

- Prefill tokens: 1024
- Decode tokens: 512
- Rounds per run: 3
- Repeats: 3

## 平均结果 (3次)

- Prefill throughput: 3204.365 tok/s (std 354.902)
- Decode throughput: 54021.117 tok/s (std 6947.004)
- E2E throughput: 4668.039 tok/s (std 519.400)
- TTFT: 2590.752 ms (std 309.289)
- TPOT: 0.151 ms/token (std 0.021)
- Memory: 4370.562 MB (std 0.000)

## 单次结果

- run1: prefill 3516.566, decode 59468.562, e2e 5123.368, ttft 2329.546, tpot 0.135, mem 4370.562
- run2: prefill 3388.603, decode 58378.022, e2e 4939.544, ttft 2417.516, tpot 0.137, mem 4370.562
- run3: prefill 2707.925, decode 44216.766, e2e 3941.204, ttft 3025.194, tpot 0.181, mem 4370.562
