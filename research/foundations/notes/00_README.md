# `research/foundations/` — Substrate 起源论文

本目录收录 **VZ R2「冻结基底」所冻结的那一类对象的"奠基论文"**。

与 `research/openai-frontier-2026/` 等"frontier"目录的差异：
- **frontier**：当下/近 2 年的论文，分析对象是"VZ 借鉴它们什么"
- **foundations**：5–12 年前的奠基论文，分析对象是"**VZ 的 R-ID 体系是如何对它们的回答 / 修正 / 延伸**"

本目录的论文不是用来"借鉴"的（它们的具体方法早被改写），而是用来**理解 VZ 的 R-IDs 站在什么基础上，又在补哪些洞**。

---

## 当前收录

| # | 笔记 | 论文 | 谱系定位 |
|---|---|---|---|
| 01 | [Seq2Seq Sutskever 2014](01_seq2seq_sutskever_2014.md) | Sutskever/Vinyals/Le, **Sequence to Sequence Learning with Neural Networks**, NIPS 2014 (arXiv 1409.3215) | substrate **架构血统**起源 |
| 02 | [Scaling Laws Kaplan 2020](02_scaling_laws_kaplan_2020.md) | Kaplan et al. (OpenAI), **Scaling Laws for Neural Language Models**, arXiv 2001.08361, 2020 | substrate **行为契约**起源（含 Chinchilla 修正） |

PDF 在 [`../papers/`](../papers/)。

---

## 这两篇为什么放在一起

它们共同回答了一个问题：「**VZ R2 所冻结的"substrate"到底是什么、为什么可以被冻结**」

### Seq2Seq 回答「是什么」

- 定义了 substrate 的**架构形状**：sequence-in / sequence-out + autoregressive next-token + EOS-controlled variable length + fixed-dim latent compression
- 这条接口契约从 2014 至今**未被改变**——只是中间填充物从 LSTM 变成 Transformer，从 384M 变成数千亿参数
- VZ 不能选择"不接受这个接口"——这就是 substrate 的**给定形状**

### Scaling Laws 回答「为什么可以被冻结」

- 定义了 substrate 的**行为契约**：给定 (N, D, C)，substrate 能力可以被定量预测（power law）
- 一旦你为 substrate 付了 (N, D, C) 的代价，它的能力**已经被 amortize 完毕**
- 在线 tweak 这个 substrate = 撕毁这份契约（噪声主导，不可能在 turn-level 数据量上 dominate 已 amortize 的投资）
- 这就是 R2「冻结 substrate」的**经济学**根据
- ⚠ Kaplan 的具体 exponent **被 Chinchilla 2022 修正**——但「**substrate 行为可被定量预测**」这条**元事实**没有变

### 二者结合 → R2 的完整论证链

```
Seq2Seq 2014:           "substrate 是 sequence-to-sequence 架构"
+
Scaling Laws 2020:      "该 substrate 的行为是 (N, D, C) 的可预测函数"
+
ETA 2025 信息论:        "joint train substrate + controller → degenerate"
+
NL 2025 系统论:         "不同时间尺度必须有清晰 update boundary"
=
R2 决策:                "把 substrate 冻结在 rare-heavy 离线轨道，
                         所有在线适应放在 controller / memory / reflection 层"
```

详见 `docs/next_gen_emogpt.md` Part 2 R2 / Part 3 NL Bridge / Part 4 ETA Bridge。

---

## 读这两篇的"VZ 视角"

这两篇都是**教科书级**论文，有数百份解读。本目录笔记**不试图**重做"科普"——而是回答 3 个 VZ 特有的问题：

### 问题 1：VZ R2 选择冻结的"那个东西"，最早什么时候出现？

- **答**：Seq2Seq 2014。所有当代 LLM 是它的直系后裔。VZ R2 实际上是对**这条 11 年血统的接口契约**的明确尊重。

### 问题 2：VZ R2 为什么经济上合理？

- **答**：Scaling Laws 2020 + Chinchilla 2022。substrate 能力 = (N, D, C) 投资的 power-law 函数；online turn-level 数据量**不可能 dominate** 这个 amortize 完毕的投资。

### 问题 3：这两篇**没说**的事，VZ 用什么补？

- **答**：见各自笔记的 §「不能告诉 VZ 的事」表。两篇加起来的"洞"覆盖：跨调用记忆、多时间尺度、relationship/self 轨道、regime 持久身份、社会认知、PE 多层化、可命名内部状态、修改的 gate / 回滚、评估覆盖"存在"——**这正是 R1 / R3-R7 / R9-R20 / R-PE 整套 R-IDs 要补的**。

---

## 写笔记的格式约定

本目录与 `openai-frontier-2026/` 等保持兼容的元数据头：

- arXiv ID + 作者 + 本地路径
- 谱系定位（一句话说清楚它在 VZ 的哪个层面有意义）
- R-ID 引用统一指向 `docs/next_gen_emogpt.md`

但 foundations 笔记额外有：

- 「**核心贡献还原**」（不被后人重述污染地复述当时实际证明了什么）
- 「**历史地位 / 谱系**」（在整个发展链中的精确位置）
- 「**对 VZ 不能告诉的事**」（必须靠后续 R-IDs 补的洞）

这是因为 foundations 论文的**当时**与**现在**视角差异远大于 frontier 论文，必须显式区分。

---

## 未来可能加入

下面是潜在候选（按"VZ R-ID 能否找到祖先"排序）。**目前未加入**，等需要时再补：

| 候选 | 是哪个 R-ID 的祖先 | 是否值得加入 |
|---|---|---|
| Vaswani 2017 Transformer (1706.03762) | R2 当代 substrate 形态 | 高（substrate 当代基线） |
| Hochreiter & Schmidhuber 1997 LSTM | seq2seq 的 building block | 中（已被 Transformer 取代） |
| Hoffmann 2022 Chinchilla (2203.15556) | Scaling Laws 修正 | **强烈建议**（已在 02 笔记里详述，但单独 PDF 值得收） |
| Brown 2020 GPT-3 (2005.14165) | scaling law 的 first major 实战验证 | 高（in-context learning 涌现的实证） |
| Bahdanau 2014 Attention (1409.0473) | seq2seq → Transformer 的过渡 | 中 |
| Ouyang 2022 InstructGPT (2203.02155) | 现代 substrate "alignment" 起源 | 高（与 R10/R12 直接相关） |

如果未来要加，遵循同样的 "foundations 视角"格式，重点在**血统位置**与**VZ 不能继承的部分**。
