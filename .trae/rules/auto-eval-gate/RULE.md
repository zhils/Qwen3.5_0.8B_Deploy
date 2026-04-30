# Auto Eval Gate Rule

## Purpose

Ensure every inference-related code change ships with reproducible accuracy + performance evidence.

## When This Rule Applies

Apply this rule if changes touch any of:

- `src/backend/cuda/**`
- `src/backend/cpu/**`
- `lossy_optimization/**`
- `tests/integration/**`
- `tests/hallucination_harness/**`

## Required Actions Before Marking Work Complete

1. Run:
   - `powershell -ExecutionPolicy Bypass -File scripts/eval/run_auto_eval.ps1`
2. Confirm generated files exist:
   - `docs/latest_eval_summary.md`
   - `docs/latest_eval_summary.json`
3. Confirm gate status is pass:
   - `gate_pass = true`

## Mandatory Reporting in Final Update

Always include:

- Accuracy checks:
  - `verify_linear_attn_batch` pass/fail and max/avg diff
  - `v2_kernel_accuracy_validate` pass/fail
- Performance checks:
  - TTFT, TPOT, Prefill tok/s, Decode tok/s, E2E tok/s, VRAM
- Compare against prior baseline if available.

## No-Claim Policy

Do not claim "optimization complete" or "performance improved" unless the above report is attached and reproducible.
