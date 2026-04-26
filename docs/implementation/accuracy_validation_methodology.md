# Qwen3.5-0.8B C++/CUDA 精度评估方法论

本文档沉淀本项目的精度验证方法，用于工程复现与面试讲解。

适用范围：
- 纯 C++ 推理链路
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
- 同时做**isolated（单层）**与**sequential（串联）**，用于区分“局部错误”与“累计误差”。

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

如果不先做这个语义对齐，会把“定义不一致”误判成“实现错误”。

---

## 3. 指标体系

本项目使用 4 个数值指标 + 2 个任务指标：

数值指标（逐层/全局）：
- `max_abs`：最大绝对误差（发现极端偏差）
- `mean_abs`：平均绝对误差（总体偏差）
- `rel_l2`：相对 L2 误差（考虑参考值尺度）
- `cosine`：余弦相似度（向量方向一致性）

任务指标（输出层）：
- `top1_match`：Top1 是否一致（0/1）
- `top10_overlap`：Top10 重叠数（0~10）

为何这样组合：
- `max_abs` 对“局部爆点”敏感；
- `rel_l2` 与 `cosine` 可避免仅靠单点误差下结论；
- TopK 指标直接回答“会不会影响生成决策”。

---

## 4. 验证流程（标准 SOP）

## 4.1 全局入口

统一脚本（程序）：
- `unified_error_report.cpp`
- 输出：`build/unified_error_report.csv`

## 4.2 分层验证（双路径）

对每一层 `i` 同时计算：

1. **isolated_i**：输入使用参考 `hidden_i`，只跑当前层，和 `hidden_{i+1}` 对比。  
2. **sequential_i**：从 embedding 开始真实串行跑到当前层，再和 `hidden_{i+1}` 对比。  

解释：
- isolated 高误差 -> 当前层实现/权重映射可疑；  
- isolated 低而 sequential 高 -> 多层累计误差或状态管理问题（如缓存长度、数值漂移）。

## 4.3 末端语义对齐验证（关键）

统一报告里显式区分三类比较：
- `pre_final_vs_hidden_24_corrected`（同语义，核心判据）
- `pre_final_vs_hidden_24_layerwise`（跨语义，仅用于提醒）
- `post_final_norm_vs_hidden_24_layerwise`（同语义）

## 4.4 最终输出一致性

final norm 后接 lm head，比较：
- `logits` 四指标
- `top1_match`
- `top10_overlap`

---

## 5. 当前结果解读模板

基于 `build/unified_error_report.csv` 与 `build/error_heatmap_and_ranking.md`：

- pre-final 语义对齐后：`max_abs` 约 3e-2，`cosine` 约 0.9997，说明主干输出一致性较好。  
- post-final 与参考对齐后：误差维持在可解释范围。  
- `top1_match=1`、`top10_overlap=10`，说明最终决策层面一致。  
- sequential 相比 isolated 在中后层有增长，属于 float32 误差累积特征，未观察到“某层局部实现崩坏”证据。  

一句话结论：
**实现正确性已达到工程可用，当前差异主要来自语义口径与多层累计数值误差，而非单点灾难性 bug。**

---

## 6. 未采用的精度验证策略：有哪些、为什么暂不使用

下面是常见可选策略，以及本项目当前阶段未采用（或未作为主判据）的原因。

### 6.1 ULP/bitwise 严格一致（逐元素位级比较）

- **策略**：要求与参考实现逐元素 bitwise 一致或 ULP 很小。  
- **暂不采用原因**：
  - CPU/CUDA 执行顺序、FMA、并行归约顺序不同，bitwise 一致通常不现实；
  - 容易把“可接受浮点差异”误判为错误，工程价值不高。  

### 6.2 大规模数据集任务指标（如 MMLU/C-Eval 全量评测）

- **策略**：跑标准 benchmark，以任务分数判断精度。  
- **暂不采用原因**：
  - 成本高、周期长，不适合当前“内核正确性与加速迭代”阶段；
  - 无法快速定位是哪个层/算子导致偏差。  

### 6.3 分布统计检验（KL/JS/Wasserstein 全层分布对齐）

- **策略**：比较激活或 logits 的概率分布距离。  
- **暂不采用原因**：
  - 实现与解释复杂度高，且对工程定位不如 max_abs + rel_l2 + cosine 直接；
  - 现阶段收益不足以覆盖额外复杂度。  

### 6.4 梯度级校验（反向传播一致性）

- **策略**：做前后向联合校验（training-style check）。  
- **暂不采用原因**：
  - 项目是推理引擎，不涉及训练反向图；
  - 额外工作量大，和当前目标（推理正确+高性能）不匹配。  

### 6.5 随机压力覆盖（海量随机输入 Monte-Carlo）

- **策略**：随机生成大量输入，统计误差分布尾部。  
- **暂不采用原因**：
  - 可补充鲁棒性，但不能替代真实模型轨迹的语义对齐验证；
  - 当前优先级低于端到端 profile 与性能优化。  

### 6.6 多精度混合一致性（FP16/BF16/INT8 全链路对照）

- **策略**：不同精度路径与 FP32 全量对照。  
- **暂不采用原因**：
  - 当前主链路以 FP32 正确性基线为先；
  - 量化/混精会引入额外变量，不适合作为当前阶段主判据。  

---

## 7. 为什么当前方案足够“工程扎实”

当前方法覆盖了三类核心风险：
- **语义风险**：通过 pre/post-final 双口径报告消除“定义不一致”误判；
- **实现风险**：通过 isolated 验证定位单层/算子正确性；
- **系统风险**：通过 sequential 验证识别累计误差与状态传播问题。

同时，`logits + TopK` 将数值偏差映射到任务决策，保证评估不是“只看数学、不看结果”。

---

## 8. 下一步可增强项（可选）

若后续你希望把方法论升级成“面试高阶版本”，建议按优先级增加：

1. 增加固定 prompt 集的端到端统计（P50/P95 latency + top1 consistency）。  
2. 增加 GPU 路径统一报告（与 CPU 同口径并列输出）。  
3. 增加误差预算表（Embedding/Attention/MLP/LMHead 的贡献拆分）。  
4. 增加版本化回归门禁（每次改 kernel 自动比对 `unified_error_report.csv` 阈值）。  

