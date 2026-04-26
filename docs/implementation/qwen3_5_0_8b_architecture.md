# Qwen3.5-0.8B 详细架构图

本文档基于本地模型目录 `D:\deploy\modules\modles\Qwen3.5-0.8B` 的 `config.json` 与权重索引自动整理，给出可视化的“逐过程连接”架构图。

## 1) 总体架构（多模态到生成）

```mermaid
flowchart TD
    A0["文本输入 tokens"] --> A1["Token Embedding<br/>vocab_size=248320, hidden=1024"]
    A2["图像/视频输入"] --> A3["Vision Patch Embedding<br/>patch=16, temporal_patch=2, in_ch=3"]
    A3 --> A4["Vision Transformer x12<br/>hidden=768, heads=12, mlp=3072"]
    A4 --> A5["Visual Merger<br/>fc1 -> act -> fc2 -> out_hidden=1024"]
    A5 --> A6["映射为视觉token序列"]

    A1 --> B0["多模态序列拼接<br/>[vision tokens + text tokens]"]
    A6 --> B0

    B0 --> B1["Qwen3.5 Language Backbone x24<br/>hidden=1024, pre-norm RMSNorm"]

    B1 --> C0["Final RMSNorm"]
    C0 --> C1["LM Head (与Embedding权重共享)"]
    C1 --> C2["Logits -> 采样/解码 -> 输出token"]

    B1 --> D0["MTP分支（训练/推测解码）<br/>mtp_num_hidden_layers=1"]
    D0 --> D1["额外1层(Attn+MLP)+fc"]
    D1 --> C2
```

---

## 2) 语言主干层级调度图（24层）

配置中 `full_attention_interval=4`，并给出显式 `layer_types`：
- 0,1,2 线性注意力；3 全注意力
- 4,5,6 线性注意力；7 全注意力
- ...
- 20,21,22 线性注意力；23 全注意力

等价为：`6 × (3 × LinearAttention + 1 × FullAttention)`。

```mermaid
flowchart LR
    L0["L0 Linear"] --> L1["L1 Linear"] --> L2["L2 Linear"] --> L3["L3 Full"]
    L3 --> L4["L4 Linear"] --> L5["L5 Linear"] --> L6["L6 Linear"] --> L7["L7 Full"]
    L7 --> L8["L8 Linear"] --> L9["L9 Linear"] --> L10["L10 Linear"] --> L11["L11 Full"]
    L11 --> L12["L12 Linear"] --> L13["L13 Linear"] --> L14["L14 Linear"] --> L15["L15 Full"]
    L15 --> L16["L16 Linear"] --> L17["L17 Linear"] --> L18["L18 Linear"] --> L19["L19 Full"]
    L19 --> L20["L20 Linear"] --> L21["L21 Linear"] --> L22["L22 Linear"] --> L23["L23 Full"]
```

---

## 3) 单个语言层内部连接（通用骨架）

每层均为 pre-norm + residual 结构：

```mermaid
flowchart TD
    X0["输入隐藏状态 h"] --> X1["RMSNorm(input_layernorm)"]
    X1 --> X2{"注意力类型?"}

    X2 -->|LinearAttention层| X3["Gated DeltaNet / Linear Attn"]
    X2 -->|FullAttention层| X4["Gated Full Attention"]

    X3 --> X5["注意力输出投影"]
    X4 --> X5
    X5 --> X6["残差相加: h = h + attn_out"]

    X6 --> X7["RMSNorm(post_attention_layernorm)"]
    X7 --> X8["MLP: gate_proj & up_proj -> SiLU -> down_proj"]
    X8 --> X9["残差相加: h = h + mlp_out"]
    X9 --> X10["输出到下一层"]
```

---

## 4) LinearAttention（Gated DeltaNet）细节连接

根据权重命名可见该分支包含：
- `in_proj_qkv`
- `in_proj_a`, `in_proj_b`, `in_proj_z`（门控/状态相关投影）
- `conv1d`（kernel_dim=4）
- `A_log`, `dt_bias`（状态更新参数）
- `norm`, `out_proj`

头部参数：
- `linear_num_key_heads=16`
- `linear_num_value_heads=16`
- `linear_key_head_dim=128`
- `linear_value_head_dim=128`

```mermaid
flowchart TD
    LA0["h_norm"] --> LA1["in_proj_qkv"]
    LA0 --> LA2["in_proj_a"]
    LA0 --> LA3["in_proj_b"]
    LA0 --> LA4["in_proj_z (gate)"]

    LA1 --> LA5["Q/K/V (linear heads)"]
    LA5 --> LA6["conv1d temporal mixing<br/>kernel_dim=4"]
    LA2 --> LA7["state update params a"]
    LA3 --> LA8["state update params b"]
    LA6 --> LA9["delta-state recurrence<br/>A_log + dt_bias + (a,b)"]
    LA7 --> LA9
    LA8 --> LA9

    LA9 --> LA10["norm"]
    LA4 --> LA11["gate(z)"]
    LA10 --> LA12["gated combine"]
    LA11 --> LA12
    LA12 --> LA13["out_proj -> attn_out"]
```

---

## 5) FullAttention（Gated Attention）细节连接

配置参数：
- `num_attention_heads=8`
- `num_key_value_heads=2`（GQA）
- `head_dim=256`
- RoPE: `partial_rotary_factor=0.25`, `rope_theta=1e7`, `mrope_interleaved=true`

```mermaid
flowchart TD
    FA0["h_norm"] --> FA1["q_proj"]
    FA0 --> FA2["k_proj"]
    FA0 --> FA3["v_proj"]

    FA1 --> FA4["q_norm"]
    FA2 --> FA5["k_norm"]

    FA4 --> FA6["RoPE on Q<br/>partial rotary"]
    FA5 --> FA7["RoPE on K<br/>partial rotary"]

    FA6 --> FA8["GQA Attention Score<br/>Q(8 heads) x K(2 kv-heads)"]
    FA7 --> FA8
    FA3 --> FA9["V (2 kv-heads, broadcast to 8)"]
    FA8 --> FA10["softmax(score) * V"]
    FA9 --> FA10

    FA10 --> FA11["attn_output_gate (config=true)"]
    FA11 --> FA12["o_proj -> attn_out"]
```

---

## 6) MLP 细节连接（每层）

```mermaid
flowchart LR
    M0["h_norm"] --> M1["gate_proj: 1024 -> 3584"]
    M0 --> M2["up_proj: 1024 -> 3584"]
    M1 --> M3["SiLU(gate)"]
    M2 --> M4["linear path"]
    M3 --> M5["逐元素乘"]
    M4 --> M5
    M5 --> M6["down_proj: 3584 -> 1024"]
```

---

## 7) 视觉编码器内部连接

配置参数：
- `depth=12`
- `hidden_size=768`
- `num_heads=12`
- `intermediate_size=3072`
- `spatial_merge_size=2`
- 输出维度经 merger 到 `1024`

```mermaid
flowchart TD
    V0["图像/视频帧"] --> V1["PatchEmbed<br/>patch=16, temporal_patch=2"]
    V1 --> V2["+ 位置编码(pos_embed)"]
    V2 --> V3["ViT Block x12"]

    subgraph VB["单个 ViT Block"]
        VB0["x"] --> VB1["Norm1"]
        VB1 --> VB2["MHSA(qkv -> attn -> proj)"]
        VB2 --> VB3["Residual Add"]
        VB3 --> VB4["Norm2"]
        VB4 --> VB5["MLP(fc1 -> act -> fc2)"]
        VB5 --> VB6["Residual Add"]
    end

    V3 --> V4["Spatial/Temporal merge"]
    V4 --> V5["Merger Norm"]
    V5 --> V6["Merger fc1 -> act -> fc2"]
    V6 --> V7["视觉token输出(hidden=1024)"]
```

---

## 8) 关键超参总表（来自本地配置）

- 语言主干：`num_hidden_layers=24`, `hidden_size=1024`
- 布局：`6 × (3×LinearAttention + 1×FullAttention)`
- 线性注意力头：`16(K)` / `16(V)`, head dim `128`
- 全注意力头：`8(Q)` / `2(KV)`, head dim `256`
- MLP中间层：`3584`
- 激活：`SiLU`
- 归一化：`RMSNorm`, `eps=1e-6`
- 最大上下文：`262144`
- 词表：`248320`, 且 `tie_word_embeddings=true`
- 视觉编码器：`depth=12`, `hidden=768`, `heads=12`, `mlp=3072`
- MTP：`mtp_num_hidden_layers=1`

---

## 9) 说明

1. 上图基于你本地权重结构与配置文件，不是通用“猜测图”。
2. 对 LinearAttention 的内部数学细节，官方代码中可能还有实现级优化（如张量重排、缓存策略）；本图按权重命名与结构配置还原主要连接过程。

---

## 10) 实现级细分文档

已生成更细的模块拆解文档（用于直接写代码和加载参数）：

- `qwen3_5_0_8b_details/00_总览与索引.md`
- `qwen3_5_0_8b_details/01_总体架构_实现级流程.md`
- `qwen3_5_0_8b_details/02_文本主干_24层调度与参数映射.md`
- `qwen3_5_0_8b_details/03_LinearAttention_GatedDeltaNet_细节.md`
- `qwen3_5_0_8b_details/04_FullAttention_GQA_RoPE_细节.md`
- `qwen3_5_0_8b_details/05_MLP_层归一化_输出头.md`
- `qwen3_5_0_8b_details/06_视觉编码器与Merger_细节.md`
- `qwen3_5_0_8b_details/07_MTP分支与推测解码_参数加载.md`

