---
name: "qwen-accuracy-validation"
description: "Qwen3.5-0.8B CUDA推理引擎精度验证与 Verify-First（先验证后输出）方法。Invoke when user asks for accuracy validation, precision testing, numerical correctness verification, or hallucination-risk reduction."
---

# Qwen3.5-0.8B CUDA推理引擎 - 精度验证与 Verify-First 方法

## 1. 验证目标与原则

### 三层验证目标

1. **算子/层正确性**：每层前向计算与参考实现一致
2. **链路稳定性**：误差在多层串联中无异常放大
3. **任务一致性**：最终 logits 与采样结果与参考一致

### Verify-First 核心原则（Asking LLMs to Verify First）

- **先验证，后输出**：对高风险回答（事实性/数字/时序/实体）先运行验证器，再决定是否直接返回。
- **失败即降级**：验证未通过时，不直接给最终结论，优先触发重生成、补证据或显式不确定输出。
- **低成本增益**：优先选择轻量校验（结构化规则 + 幻觉检测器），在小额延迟开销下显著降低错误输出风险。

### 评估原则

- 先做**语义对齐**，再做数值对齐
- 同时看**绝对误差**与**相对结构相似度**
- 同时做 **isolated（单层）**与 **sequential（串联）**
- Verify-First 评估必须同时报告：**质量收益**（通过率/幻觉率）+ **时延成本**（validation_time）

## 2. 验证物料

### 核心文件

| 物料 | 路径 | 说明 |
|------|------|------|
| 模型权重 | `weights/language_backbone` | 主干权重 |
| 参考激活（逐层） | `validation_data_layerwise/hidden_*.bin` | 每层输入输出 |
| 语义修正参考 | `validation_data_corrected/hidden_24.bin` | layer23 输出（pre-final-norm） |
| 参考 logits | `validation_data_layerwise/logits.bin` | 最终 logits |
| 参考预测 token | `validation_data_layerwise/predicted_token.bin` | Top-1 token |

### 重要语义说明

- `validation_data_corrected/hidden_24.bin` = **layer23 输出（pre-final-norm）**
- HF 的 `hidden_states[24]` = **final RMSNorm 之后（post-final-norm）**

## 3. 指标体系

### 4 个数值指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| `max_abs` | 最大绝对误差 | < 1e-3 |
| `mean_abs` | 平均绝对误差 | < 1e-4 |
| `max_rel` | 最大相对误差 | < 1e-3 |
| `cosine_sim` | 余弦相似度 | > 0.999 |

### 2 个任务指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| `top1_match_rate` | Top-1 token 一致率 | > 99% |
| `top5_match_rate` | Top-5 token 一致率 | > 99.9% |

### Verify-First 线上指标（建议新增）

| 指标 | 说明 | 目标 |
|------|------|------|
| `pass_rate_after_verify` | 验证后可直接返回比例 | 越高越好 |
| `hallucination_rate` | 输出中幻觉占比 | 相对基线下降 |
| `validation_overhead_ms` | 单次验证耗时 | 控制在可接受范围 |
| `overhead_ratio` | 验证耗时 / 生成耗时 | 建议 < 10% |

### Verify-First 验收门槛（Go/No-Go）

默认门槛（可按业务风险调整）：

| 类别 | 指标 | 默认门槛 | 判定 |
|------|------|----------|------|
| 质量收益 | `hallucination_rate_drop` | >= 20% | 未达标则 No-Go |
| 质量收益 | `high_risk_error_drop`（事实/数字/实体） | >= 30% | 未达标则 No-Go |
| 可用性 | `pass_rate_after_verify` | >= 85% | 过低需优化策略 |
| 成本 | `overhead_ratio` | <= 10% | 超标需降级 checker |
| 成本 | `p95_validation_overhead_ms` | <= 50 ms（在线） | 超标需优化路径 |

说明：
- `hallucination_rate_drop = (baseline - verify_first) / baseline`
- `high_risk_error_drop` 仅统计 FACTUAL / NUMERIC / ENTITY 三类
- 高风险业务（医疗/金融）建议将质量门槛提高到 30%/40%

## 4. 验证流程

### 4.0 一键自动评估（推荐默认入口）

优先使用统一脚本，一次产出精度 + 性能结论：

```bash
powershell -ExecutionPolicy Bypass -File scripts/eval/run_auto_eval.ps1
```

可选参数：

```bash
powershell -ExecutionPolicy Bypass -File scripts/eval/run_auto_eval.ps1 `
  -PrefillTokens 1024 -DecodeTokens 512 -Rounds 3 -BatchSize 1 -BuildConfig Release
```

脚本输出：
- `docs/latest_eval_summary.md`
- `docs/latest_eval_summary.json`

若 `gate_pass=false`，视为本轮优化未通过验收，必须先修复再继续。

### 4.1 算子级验证（开发阶段）

每实现一个新 kernel，先做单层验证：

```cpp
// 1. 加载参考输入和权重
float* ref_input = load_bin("validation_data_layerwise/hidden_00.bin");
float* ref_output = load_bin("validation_data_layerwise/hidden_01.bin");

// 2. 运行你的 kernel
float* my_output = my_kernel_forward(ref_input, weights);

// 3. 对比
ErrorMetrics err = compare(my_output, ref_output, size);
```

### 4.2 链路级验证（集成阶段）

整模型端到端验证：

```bash
# 验证 Batch Linear Attention 精度
./verify_linear_attn_batch

# V2 Kernel 精度验证
./v2_kernel_accuracy_validate
```

### 4.3 Verify-First 验证（上线前必跑）

利用项目内置幻觉检测 harness 对“先验证”策略做 A/B：

```bash
# 1) 构建（CPU 即可）
cmake -S . -B build -DENABLE_CUDA=OFF
cmake --build build --config Release --target harness_runner

# 2) 运行批量验证
./build/harness_runner
```

关注输出中的以下字段：
- `Passed` / `Failed`
- `Average Hallucination Score`
- `Total Hallucinations Detected`
- `Validation Time (ms)`

建议对照两组输出：
1. **Baseline**：直接输出，不触发验证
2. **Verify-First**：先 `harness.validate(prompt, response)`，未通过则重生成/降级

对外报告至少包含：
- 幻觉率变化（Verify-First vs Baseline）
- 平均验证时延与开销占比
- 失败样本类型分布（FACTUAL / NUMERIC / ENTITY / CONSISTENCY）

### 4.5 结论生成模板（直接复用）

#### A) 实验结果表（最小集）

| 指标 | Baseline | Verify-First | 变化 |
|------|----------|--------------|------|
| hallucination_rate | x.xx | y.yy | z.zz% |
| high_risk_error_rate | x.xx | y.yy | z.zz% |
| pass_rate_after_verify | - | x.xx | - |
| avg_validation_overhead_ms | - | x.xx | - |
| p95_validation_overhead_ms | - | x.xx | - |
| overhead_ratio | - | x.xx% | - |

#### B) 自动判定规则

```text
IF hallucination_rate_drop >= 20%
AND high_risk_error_drop >= 30%
AND pass_rate_after_verify >= 85%
AND overhead_ratio <= 10%
AND p95_validation_overhead_ms <= 50
THEN = GO (recommend rollout)
ELSE = NO-GO (optimize validators/policy first)
```

### 4.6 与 Rule 联动（强约束）

若改动涉及推理代码（`src/backend/cuda/**`、`lossy_optimization/**` 等），必须遵循：
- `/.trae/rules/auto-eval-gate/RULE.md`
- 未生成 `latest_eval_summary` 或 `gate_pass=false` 时，不得宣称“优化完成”

#### C) 对外一句话结论模板

```text
启用 Verify-First 后，幻觉率从 {baseline} 降至 {verify}（下降 {drop}%），
高风险错误下降 {high_risk_drop}%；
验证开销 p95 为 {p95_ms} ms，占生成时延 {overhead_ratio}%。
在满足质量门槛与时延预算前提下，建议 {灰度/全量} 上线。
```

### 4.4 单元测试

```bash
# RMSNorm 测试
./test_rmsnorm

# Flash Attention 测试
./test_flash_attention

# Linear Attention 测试
./test_linear_attn

# Full Attention 测试
./test_full_attn

# KV Cache 测试
./test_kvcache

# 主干测试
./test_backbone
```

## 5. 关键测试文件

| 文件 | 位置 | 用途 |
|------|------|------|
| `verify_linear_attn_batch.cu` | `src/backend/cuda/` | Batch Linear Attn 精度验证 |
| `v2_kernel_accuracy_validate.cu` | `tests/integration/` | V2 Kernel 精度验证 |
| `test_linear_attn.cpp` | `tests/unit/` | Linear Attn 单元测试 |
| `test_flash_attention.cu` | `tests/unit/` | Flash Attn 单元测试 |
| `harness_runner.cpp` | `tests/hallucination_harness/` | Verify-First 幻觉检测批测 |
| `hallucination_harness.hpp` | `tests/hallucination_harness/core/` | 验证器接口与策略配置 |

## 6. 推理集成模板（先验证后输出）

```cpp
auto response = engine.generate(prompt);
auto report = harness.validate(prompt, response);

if (report.passed) {
    return response;
}

// Verify-First 失败策略：可按业务选一种
// A) 重生成一次
// B) 返回带不确定性声明的安全答案
// C) 命中高风险类型时要求外部检索
return regenerate_with_constraints(prompt, report);
```

## 7. 常见问题

### Q: max_abs 满足但 cosine_sim 低？
A: 可能是尺度不一致，检查是否有溢出或下溢

### Q: 单层 pass 但串联后误差累积？
A: 检查残差连接和 LayerNorm 的数值稳定性

### Q: Top-1 一致但 Top-5 很低？
A: 说明 logits 排序正确但置信度有偏差，检查 softmax 精度

### Q: Verify-First 会不会太慢？
A: 先看 `validation_time_ms` 与 `overhead_ratio`。若开销偏大，可先启用 factual/numeric/consistency 三类轻量验证器，再按风险逐步加 checker。

### Q: 指标达不到 Go 门槛怎么办？
A: 按顺序优化：1) 先降成本（关闭 semantic/entity 重 checker）；2) 再提质量（补知识库、提升 factual/numeric 召回）；3) 最后调整阈值（高风险场景只调严不调松）。
