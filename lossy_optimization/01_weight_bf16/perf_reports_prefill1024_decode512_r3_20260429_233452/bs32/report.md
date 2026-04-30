# 01_weight_bf16 性能报告 (batch=32)

- Prefill tokens: 1024
- Decode tokens: 512
- Rounds per run: 3
- Repeats: 3

## 平均结果 (3次)

- Prefill throughput: 9754.780 tok/s (std 2677.932)
- Decode throughput: 89483.072 tok/s (std 8733.605)
- E2E throughput: 13853.342 tok/s (std 3663.684)
- TTFT: 3586.437 ms (std 828.634)
- TPOT: 0.361 ms/token (std 0.035)
- Memory: 4370.562 MB (std 0.000)

## 单次结果

- run1: prefill 7664.423, decode 79282.357, e2e 10966.553, ttft 4275.338, tpot 0.404, mem 4370.562
- run2: prefill 13534.879, decode 100614.419, e2e 19022.822, ttft 2421.004, tpot 0.318, mem 4370.562
- run3: prefill 8065.039, decode 88552.441, e2e 11570.652, ttft 4062.969, tpot 0.361, mem 4370.562
