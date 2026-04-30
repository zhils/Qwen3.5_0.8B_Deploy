# Latest Auto Eval Summary

- Generated at: 2026-04-29 22:21:17
- Gate pass: **False**
- Config: prefill=128, decode=32, rounds=1, batch=1

## Accuracy

| Check | Result | Value |
|---|---|---|
| verify_linear_attn_batch | True | max_diff=2.94917E-08, avg_diff=2.47039E-09 |
| v2_kernel_accuracy_validate | False | ALL TESTS PASSED required |

## Performance

| Metric | Value |
|---|---:|
| TTFT (ms) | 316.308 |
| Prefill throughput (tok/s) | 404.669 |
| TPOT (ms/token) | 0.808 |
| Decode throughput (tok/s) | 1237.477 |
| E2E throughput (tok/s) | 467.608 |
| VRAM used (MB) | 5212.562 |

## Raw Logs (truncated)

### verify_linear_attn_batch
```text
=== Linear Attention Batch Verification ===
Max diff: 2.94917e-08
Avg diff: 2.47039e-09
Diff count (>1e-4): 0/4096
PASS: Batch output matches serial output

```

### v2_kernel_accuracy_validate
```text
=================================================
  v2.0 Kernel Accuracy Validation
  (conv1d_update + norm_gate_fused)
=================================================

=== Test: conv1d_update + norm_gate_fused (Register caching) ===
  [PASSED] Single forward deterministic, max diff: 0.00000000
  Testing batch forward determinism (batch_size=8)...
  [FAILED] Batch forward non-deterministic, max diff: 0.02995464
  Testing multi-step forward consistency...
  [PASSED] State updates correctly (step diff: 0.00051871)

=================================================
  SOME TESTS FAILED
=================================================

```
