# 01_weight_bf16 性能报告 (batch=128)

- Prefill tokens: 1024
- Decode tokens: 512
- Rounds per run: 3
- Repeats: 3

## 平均结果 (3次)

- Prefill throughput: 57257.515 tok/s (std 11264.796)
- Decode throughput: 132492.735 tok/s (std 22108.708)
- E2E throughput: 70613.565 tok/s (std 13525.042)
- TTFT: 2386.181 ms (std 497.383)
- TPOT: 0.994 ms/token (std 0.170)
- Memory: 4572.562 MB (std 0.000)

## 单次结果

- run1: prefill 58413.382, decode 132754.877, e2e 71819.461, ttft 2243.869, tpot 0.964, mem 4572.562
- run2: prefill 70439.721, decode 159438.239, e2e 86542.390, ttft 1860.768, tpot 0.803, mem 4572.562
- run3: prefill 42919.443, decode 105285.088, e2e 53478.845, ttft 3053.907, tpot 1.216, mem 4572.562
