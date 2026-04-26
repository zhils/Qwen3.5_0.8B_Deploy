# Qwen3.5-0.8B C++/CUDA 精度评估方法论

本文档沉淀本项目的精度验证方法，用于工程复现。

适用范围：
- C++ + CUDA 混合推理链路
- 层级算子验证与端到端输出一致性验证

---

## 1. 目标与原则

精度评估目标分三层：

1. **算子/层正确性**：确认每层前向计算在数值上与参考实现一致。
2. **链路稳定性**：确认误差不会在多层串联中出现异常放大。
3. **任务一致性**：确认最终 logits 与采样结果（Top1/TopK）与参考实现一致。

评估原则：
- 先做**语义对齐**，再做数值对齐。
- 同时看**绝对误差**与**相对结构相似度**（仅看 max_abs 不够）。
- 同时做**isolated（单层）**与**sequential（串联）**，用于区分"局部错误"与"累计误差"。

---

## 2. 验证物料（输入/权重/参考）

核心物料：
- 模型权重：`weights/language_backbone`、`weights/language`
- 参考激活（逐层）：`validation_data_layerwise/hidden_*.bin`
- 语义修正参考：`validation_data_corrected/hidden_24.bin`
- 参考 logits：`validation_data_layerwise/logits.bin`
- 参考预测 token：`validation_data_layerwise/predicted_token.bin`

关键语义说明（必须先讲清）：
- `validation_data_corrected/hidden_24.bin`：**layer23 输出（pre-final-norm）**
- HF 的 `hidden_states[24]`：**final RMSNorm 之后（post-final-norm）**

如果不先做这个语义对齐，会把"定义不一致"误判成"实现错误"。

---

## 3. 指标体系

本项目使用 4 个数值指标 + 2 个任务指标：

数值指标（逐层/全局）：
- `max_abs`：最大绝对误差（发现极端偏差）
- `mean_abs`：平均绝对误差（总体偏差）
- `max_rel`：最大相对误差（关注小数值区域的相对偏差）
- `cosine_sim`：余弦相似度（结构相似性，对尺度不敏感）

任务指标（端到端）：
- `top1_match_rate`：Top-1 token 一致率（模型行为一致性）
- `top5_match_rate`：Top-5 token 一致率（候选集一致性）

---

## 4. 验证流程

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
// 要求: max_abs < 1e-3, mean_abs < 1e-4, cosine_sim > 0.999
```

### 4.2 链路级验证（集成阶段）

整模型端到端验证：

```cpp
// 1. 运行完整推理
std::vector<int> tokens = tokenize(prompt);
auto result = engine.generate(tokens, max_length);

// 2. 对比 logits
float* ref_logits = load_bin("validation_data_layerwise/logits.bin");
ErrorMetrics err = compare(result.logits, ref_logits, vocab_size);

// 3. 对比采样结果
assert(result.tokens[0] == ref_token);  // Top-1 必须一致
```

### 4.3 Batch 精度验证

针对 Batch Linear Attention 的精度验证：

```cpp
// 验证 batch 输出与串行输出一致
CudaLinearAttnState state1, state2;
state1.reset(...); state2.reset(...);

// 串行方式
for (int b = 0; b < batch_size; ++b) {
    attn.forward(input + b * hidden_size, output_serial + b * hidden_size, state1);
}

// Batch 方式
attn.forward_batch(input, output_batch, state2, batch_size);

// 对比
ErrorMetrics err = compare(output_serial, output_batch, batch_size * hidden_size);
// 当前实现: max_diff < 1e-7 (浮点精度级别)
```

---

## 5. 精度标准

| 验证阶段 | max_abs | mean_abs | cosine_sim | top1_match |
|---------|---------|----------|------------|------------|
| 算子级 | < 1e-3 | < 1e-4 | > 0.999 | - |
| 链路级 | < 1e-2 | < 1e-3 | > 0.995 | > 99% |
| Batch 对比 | < 1e-6 | < 1e-7 | > 0.999999 | 100% |

---

## 6. 当前精度状态

### Batch Linear Attention 精度验证

- **测试**: `verify_linear_attn_batch.exe`
- **结果**: max diff = 9.6e-08
- **结论**: Batch 输出与串行输出在浮点精度级别完全一致

### 端到端精度

- 与 PyTorch 参考实现对比：max diff < 0.04
- Top-1 token 一致率：100%
