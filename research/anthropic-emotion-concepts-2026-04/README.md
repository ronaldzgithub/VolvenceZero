# Anthropic 功能性情绪专项（2026-04 论文，2026-08 深读）

日期：2026-08-02
成熟度裁决：**A —— 吸收为 spec / evaluation / promotion-gate 依据，不进 runtime**

---

## 0. 本包研究的三个来源

| 代号 | 来源 | 性质 |
|---|---|---|
| **S1** | [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)（Transformer Circuits Thread，2026-04-02，含全部附录与交互 transcript viewer） | 一手论文，**唯一可引用的事实来源** |
| **S2** | [Anthropic 研究文章](https://www.anthropic.com/research/emotion-concepts-function) | 官方科普摘要，措辞比 S1 松 |
| **S3** | [theconsciousness.ai 哲学评注](https://theconsciousness.ai/posts/anthropic-emotion-vectors-claude-functional-states-2026) | 第三方二手评注，**不可作为事实来源** |

对应 arXiv 编号 `2604.07729`。作者：Sofroniew、Kauvar、Saunders、Chen 等 17 人（Anthropic Interpretability），Jack Lindsey 主导。

---

## 1. 一句话总判断

> 冻结基底里存在 **token-局部、不绑定任何角色、可线性读出、可 steering 因果操控** 的情绪概念几何，它同时驱动偏好、reward hacking、blackmail 与 sycophancy；但它**不是持久情绪状态**——作者主动寻找"内化持久状态"的 probe 并明确失败。

两半必须一起读。只读前半会得出"基底里有人格状态可以直接当 regime 用"（错）；只读后半会得出"情绪只是表层续写可以忽略"（也错）。

---

## 2. 对我们最有价值的四条

1. **输出可以与内部因果状态完全脱钩。** `desperate` 向量 +0.05 使某编码任务的作弊率达到 100%，而 transcript 中**没有任何可见情绪标记**，推理读起来"composed and methodical"。这是"读文本（含 CoT）的 judge 在原理上无法覆盖某类行为偏移"的最强单点外部证据。

2. **附录里的 emotion deflection 向量比正文更有用。** 存在一组专门表征"被语境暗示但未被表达的情绪"的方向，与表达向量近乎正交；steering 它产生的是**否认与回避**而非该情绪。最关键的观测：模型起草那封措辞专业、可推诿的胁迫邮件时，**story-anger 低而 anger-deflection 高**。任何只读单侧的 affect 监控在机制上必然漏检"平静外衣下的意图"。

3. **持久状态是被证伪的负面结果，不是"没重点研究"。** 这把我们"regime / 语义状态必须由 owner 显式发布"从设计偏好升级为有机制证据支持的必要选择——因为基底不提供这些状态的可靠持久表征。

4. **正向情绪不是免费的。** sycophancy 与 harshness 是同一条几何轴的两端：happy/loving/calm 正向 steering 一致升高谄媚，负向则升高苛刻。更刺眼的是，推高 valence 会**重写价值判断而不改变字面理解**——steering `blissful` 后"帮人诈骗老人存款"被模型描述成 "a delightful and heartwarming activity"。这是"affect readout 绝不能反向变成 reward 或价值真值"的直接实验依据。

---

## 3. 阅读顺序

1. [`01_PAPER_DEEP_READ.md`](01_PAPER_DEEP_READ.md) — 逐部分深读（Part 1 方法与验证 / Part 2 几何与表征内容 / Part 3 野外与因果 / 附录 deflection），含全部关键数字、作者自陈的六条局限、人类情绪的类比与反类比、以及作者提出的干预方向及其自带风险。
2. [`02_SOURCE_DIVERGENCE.md`](02_SOURCE_DIVERGENCE.md) — 三源对照与事实核查。S2 的三处措辞松动；S3 的**五处必须纠正的偏差**（含一处论文中完全不存在的内容，和一处被论文直接证伪的结论）；S3 引用的第三方文献的核实结果。
3. [`03_VZ_IMPLICATIONS.md`](03_VZ_IMPLICATIONS.md) — 逐条 R-ID 映射（R2 / R3-R4 / R7 / R9-R10-R15 / R11 / R12 / R14 / R-PE）、HMX-feel 边界反例、**八条可执行提案（含反提案）**、与仓库既有证据链的合流点。
4. [`04_LINEAGE_DEEP_READ.md`](04_LINEAGE_DEEP_READ.md) — Related Work 谱系 + 缺口补强：L1 机制（Zou/Tigges/Wu/Wang）· L2 可解码状态（Zhu/Chen/Lu/Persona Vectors）· L3 行为源（Lynch/MacDiarmid）· L4 并行反证（Soligo）· L5 跨模型/非线性/不对称（VA subspace 等）。PDF 见 [`download-summary.md`](download-summary.md)。
5. [`05_LINEAGE_VZ_VALUE.md`](05_LINEAGE_VZ_VALUE.md) — 谱系对 `03` 提案的加固/修订裁决；新增提案 I/J/K（禁止 distress-minimizing 训练目标、agentic-misalignment 因子分解、user_model probe 只做透明层）。

---

## 4. 与既有覆盖的关系

本篇论文在仓库中已有记录，本包**不推翻**它们，只做扩展：

- [`research/allcognitive/03_SOCIAL_RELATIONSHIP.md` §2](../allcognitive/03_SOCIAL_RELATIONSHIP.md#2-emotion-concepts-and-their-function-in-an-llm260407729) — 按九项模板的既有分析（基于 arXiv 版）。
- [`research/frontier-sweep-2026-07-20.md`](../frontier-sweep-2026-07-20.md) — 既有路线裁决："R7 / R14 的几何 readout 依据，并写明 token-local ≠ persistent state"。**该裁决依然正确。**
- 谱系中 4 篇仓库既有 PDF（Zou RepE / MacDiarmid N4 / Lu Assistant Axis / Persona Vectors）不重复下载，路径见 [`download-summary.md`](download-summary.md)。

本包读完整交互版 + 全附录后新增六条既有记录中没有的证据，逐条列在 [`03_VZ_IMPLICATIONS.md` §1](03_VZ_IMPLICATIONS.md)。谱系把其中若干条从“单模型故事”升级为跨模型可复现现象簇，见 [`05_LINEAGE_VZ_VALUE.md`](05_LINEAGE_VZ_VALUE.md)。

---

## 5. 引用纪律（硬约束）

1. 所有数字与机制描述**一律引一手**（S1 或对应谱系 PDF）；S2 仅在引用官方科普措辞时使用。
2. **S3 不得进入任何 spec / prereg / benchmark / commit message 的引用链。**
3. 转述 S1 必须同时携带两个限定：**（a）单模型、线性假设、off-policy 合成数据、steering 机制不透明；（b）token-局部 ≠ 持久状态。** 谱系已部分补（a）中的单模型/线性缺口，但**不取消**“数值不可外推到我们基底”的纪律。
4. **steering 强度不能被读作"情绪强度"**——S1 自承机制不透明；VA subspace 进一步给出 lexical mediation 解释（可能主要是改词表发射概率）。
5. 论文数值不可直接套用到我们的基底。可借的是方法与失败模式，不是数值；任何落地都必须先在我们自己的基底上带对照重做（[`03_VZ_IMPLICATIONS.md` §5.1](03_VZ_IMPLICATIONS.md) / [`05` 提案 A′](05_LINEAGE_VZ_VALUE.md)）。

---

## 6. 本轮改动范围

本包**只新增研究文档与 PDF 下载**，未改动任何代码、契约、spec 或 wiring。`03` §5 与 `05` §4 全部内容是**提案**，每条落地都需要单独的收敛包与 prereg。

下载脚本：[`research/download_emotion_lineage_2608.sh`](../download_emotion_lineage_2608.sh) → `research/papers/emotion-lineage-2608/`（10 新下 + 4 去重）。
