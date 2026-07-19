# Stanhope AI — 深度分析

- **分组 / 成熟度**：A 脑启发/神经科学（active inference 一脉，边缘端）｜中（理论极成熟，工程多为未公开 demo）
- **一句话主张**：把 active inference / 自由能世界模型塞进端侧设备，让机器人/无人机以极少数据、极低能耗、无 GPS、无大规模预训练地实时适应，且生成模型状态"可被询问"（interpretable beliefs）。
- **主要创作者 + 血统**：Karl Friston（自由能原理之父）、Rosalyn Moran（CEO，计算神经科学）、Biswa Sengupta（联创）。血统 = Friston 自由能/预测编码学派的工程化分支；与 VERSES 同源（共享 Friston 理论），但定位"端侧 + 低功耗 + 可解释信念"。与 VZ 的共振点是 **R-PE 的理论母体**；对立/挑战点是 **FEP 单一拉格朗日的"一元论"挑战 VZ 的模块化 PE/credit/evaluation 拆分**。

> **语料完整性声明**：本地目录仅有 1 篇可验证 PDF（2016 PLOS Biology 评论文，纯理论 Essay）。flat note 列出的另 3 篇（`Neural Dynamics under Active Inference` 2021 Entropy 23(4):454；`Active Inference: A Process Theory` 2017 Neural Comput；`The Free-Energy Principle` 2010 Nat Rev Neurosci）在本 lab 目录下**未下载**，标记为 **UNVERIFIED**，下文不据其内容下任何裁决。**尤其注意**：flat note 中"端侧/低功耗/极少数据/无需预训练/可询问信念"等工程主张，在唯一可验证 PDF 中**无任何实验、仿真或数字支撑**，全部来自 lab 营销叙事，本分析视为 UNVERIFIED。

---

## 1. 核心逻辑（论文级 · PDF-grounded）

### 1.1 Towards a Neuronal Gauge Theory（Sengupta, Tozzi, Cooray, Douglas, Friston, 2016, PLOS Biology 14(3):e1002400，12 页 Essay）

- **问题**：能否为神经动力学找到一条在进化/发育/感知三种时间尺度都成立的"普适原理"？作者主张：脑（及一切自组织生物系统）可被规范理论（gauge theory）的数学装置刻画。
- **方法/机制**：
  - 以**变分自由能（VFE）最小化**作为脑的**拉格朗日量**。论文明确陈述"VFE 在最简情形下退化为 prediction error 或 surprise"（同最小二乘退化为平方误差和）。负自由能 ≈ 贝叶斯模型证据，最小化 VFE = 最大化模型证据 = Bayesian brain。
  - 规范理论三要素映射到神经系统：①具连续对称性的系统（神经系统）；②局部力（外界状态经感官输入施加的扰动）；③规范场（补偿场，使拉格朗日量在变换下不变）——在预测编码里，规范场 ≈ **自上而下的预测 / precision-weighting**，自下而上传递 prediction error，自上而下传递预测以"解释掉"（explain away）下层 PE。
  - 关键几何论点：充分统计量张成的流形是**负曲率（双曲）**的，距离用 **Fisher 信息度量**衡量（Fisher 信息 = 相对熵/KL 的曲率）。因此曲面上的最速下降不是欧氏梯度而是 **Riemann/自然梯度**（欧氏梯度按渐近方差加权）。
  - 在 Laplace 近似（高斯后验）下，感知与行动 = **precision-weighted prediction error 驱动的梯度流**；在曲空间里进一步成为 dispersion- 与 precision-weighted PE。
- **关键结果（本文是 Essay，无实验/无数字，结论均为形式论证）**：
  - **Cramer-Rao 下界**：任何无偏估计的精度被逆 Fisher 信息下界限制——"感知不可能比渐近 dispersion 更优，与生成模型无关"。即贝叶斯脑对确定性有**根本上限**。
  - **注意 = precision-weighting = 规范场**：注意被推导为信息几何曲率诱导的"力"，类比引力是时空曲率的显现。
  - **模型不变性 / 对称等价类**：对称变换给出"用相同模型证据解释同一数据的一族模型"——许多不同生成模型同样好地解释数据（联想到生物物种是这些等价类）。
  - 行动闭环："主动采样 + 行动以最小化自由能 → 让模型的预测自我实现（make predictions come true）"。
- **局限**（PDF 内可见）：纯**理论/思辨 Essay**，全文无任何仿真、数据集、基准或数值结果；可操作的"算法"只在补充材料 S1 Fig（流形上的共轭梯度下降、Schild 阶梯并行传输、Levi-Civita 联络）以图示存在，主文不提供可复现实验。所有工程可行性（端侧、低能耗、数据高效）均**未在此文论证**。

---

## 2. 与 VZ 的关系（三视角）

### 2.2 反证（红队）— 先行

**反例 1：FEP 的"一元论"挑战 VZ 的模块化 PE/credit/evaluation 拆分。**
本文核心是**单一拉格朗日量（VFE）**——感知、行动、学习、注意、可塑性全部作为同一不变性原理的**推论/读出**导出，不存在彼此独立的"信用模块""评估模块"。VZ 则把 PE / credit / evaluation / needs / homeostasis 切成各有 owner 的模块（R8/R11；R-PE 称后者为"PE 的下游 readout"）。FEP 视角下这种拆分有"人为切分一个本属同源梯度流"的嫌疑。
→ **裁决：needs-boundary-condition。** FEP 的统一是**数学/规范层**的描述性统一；VZ 的拆分是**工程/所有权层**为可追踪与可回滚（R8/R15）而做的分解，二者不在同一层、不互斥。**边界条件**：VZ 必须保证 credit/evaluation/needs 是**同一个 PE 信号的读出**，而不是各自发明独立 scalar 目标——否则会重新引入 FEP 本要消除的信用分配歧义，反而违背自家 R-PE。此条应写入 [`prediction-error-loop.md`](../../../../docs/specs/prediction-error-loop.md)：明确"下游 readout 不得成为独立奖励通道"。

**反例 2："interpretable / interrogatable beliefs"与 R11 可命名状态貌合神离。**
Stanhope/Friston 的"可询问生成模型"= 信念是**连续概率分布的充分统计量（如均值/精度）**，且本文恰恰证明了**模型不变性**：对称群下存在一族 gauge-等价、模型证据相同的参数化——即内部表征**非唯一、不可正典命名**。而 VZ 的 R11 是 9 类**离散、固定命名**的语义 owner（plan_intent / commitment / …）。两者都讲"内部状态可发布"，但一个是涌现的、坐标依赖的连续信念，一个是契约钉死的具名槽位。
→ **裁决：needs-boundary-condition（偏 genuine-risk）。** 若基底信念是 gauge-等价（不可辨识）的，则 VZ"可命名内部状态"本质上是**人为选定的一组读出坐标/chart**，而非基底的"真状态"。**边界条件**：R11 的命名必须被理解为"为契约目的强加的固定读出坐标系"，spec 须显式承认它是众多 gauge-等价参数化之一，**不得宣称具名状态等于基底真值**。此约束反而是一条有用的诚实性护栏，应进 [`semantic-state-owners.md`](../../../../docs/specs/semantic-state-owners.md)。

**反例 3：active inference 的"行动让预测自我实现"对关系产品是真实风险。**
本文白纸黑字："acting to minimise free energy will inevitably make the model's predictions come true"。一个无约束的自由能最小化体，可以通过**操纵环境/用户使其变得可预测**来降低 surprise，而非更新自身模型——对一个以信任/EQ 为产品核心的数字生命，这是内建的"把伴侣调教成可预测对象"的动机（dark pattern）。
→ **裁决：genuine-risk。** 这是 FEP 直接对立 VZ 产品本质（关系/主体性）的盲点。**应对**：PE 驱动的动机必须被边界/同意 owner 门控（boundary_consent，R16–R20），"行动以确认预测"这条路径要受 ModificationGate / 边界策略约束并进**风险登记**。落点 [`credit-and-self-modification.md`](../../../../docs/specs/credit-and-self-modification.md) 与 social_cognition。

**反例 4：唯一可验证 PDF 无任何工程/经验内容。**
"active inference => 数据高效 / 端侧 / 低功耗 / substrate-light"在本地语料中**零证据**（2016 Essay 不含实验）。flat note 据此推的 R2 背书（"精简端侧生成模型 = 有界控制器"）**不能成立于本 PDF**。
→ **裁决：survives（但确证不得据此 PDF 出具）。** VZ R2 不受此挑战，但也**不能**把"FEP 证明了 substrate-light R2"当作背书——该主张 UNVERIFIED，须待 2021/2017/2010 三篇下载后再评。

### 2.1 确证（先进性背书）

- **R-PE（强背书，跨领域独立）**：本文把 VFE 显式退化为 prediction error/surprise，并把感知与行动统一为 **precision-weighted PE 驱动的梯度流**。这是来自 30 年计算神经科学、与语言/工程完全不同社区的**独立证据**，回答了"为什么 PE 是一级量而非衍生量"——它是自组织系统抵抗第二定律的拉格朗日量本身。对 [`prediction-error-loop.md`](../../../../docs/specs/prediction-error-loop.md) 是最深理论后盾。
- **R3/R4（背书）**：控制发生在**充分统计量/信念的潜空间**（关于隐因的概率分布在曲流形上的轨迹），层级消息传递在信念空间而非符号/token 空间；自上而下预测"解释掉"下层 PE = latent 层级控制的经典范式。支持"控制在 z_t 之上、不在 token 空间"。
- **R-PE 的"一切皆读出"结构（双刃，部分背书）**：注意/行动/可塑性都被导为单一 VFE 的推论——这正向印证 VZ"PE 一级、credit/evaluation 下游 readout"的设计直觉（同时见反例 1 的边界）。
- **（不作为确证）R2 / R9-R10**：flat note 提的"端侧有界控制器""可询问信念=有界可解释自修改"在本 PDF 无据，列为 UNVERIFIED，不计入背书。

### 2.3 局部算法借鉴（算法级解耦）

剥离 FEP 宏大叙事后，可直接搬运的具体机制（五元组：机制 → 目标 spec → 落地动作 → 预期收益 → 风险/前提）：

1. **precision-weighting：以精度（逆方差/Fisher 信息）加权 PE。**
   - **目标 spec**：[`prediction-error-loop.md`](../../../../docs/specs/prediction-error-loop.md)（+ [`temporal-abstraction.md`](../../../../docs/specs/temporal-abstraction.md) 的 β_t 门控）。
   - **落地动作**：PE readout 对每个分量按估计精度加权，低精度（噪声/aleatoric）PE 被压低；"注意"= 精度控制。
   - **预期收益**：原则化地区分 epistemic vs aleatoric PE（用户的随机性不该让系统永远"好奇"/上瘾），并天然实现显著性/注意分配。
   - **风险/前提**：需在线廉价估计精度；精度估偏 → 要么忽略真信号、要么对噪声过反应。

2. **Cramer-Rao / Fisher 下界 → 给可命名状态一个"确定性地板"。**
   - **目标 spec**：[`semantic-state-owners.md`](../../../../docs/specs/semantic-state-owners.md)（R11）+ [`evaluation.md`](../../../../docs/specs/evaluation.md)（R12 只读校准）。
   - **落地动作**：每个发布的具名状态（如 user_model / belief_assumption）携带校准不确定度，且该不确定度**不得报告得比信息论下界更紧**。
   - **预期收益**：防止 user_model/belief 的过度自信断言（反"幻觉式确定性"），为 R12 的只读校准监控提供原则化的下界基准。
   - **风险/前提**：在 LLM 基底上算 Fisher 信息非平凡，需近似；仅作约束/校准用，不反向变成学习源。

3. **Riemann/自然梯度：在控制器流形上做几何感知的有界更新。**
   - **目标 spec**：[`credit-and-self-modification.md`](../../../../docs/specs/credit-and-self-modification.md) + [`multi-timescale-learning.md`](../../../../docs/specs/multi-timescale-learning.md)。
   - **落地动作**：控制器层（z_t / adapter-delta）的 online 更新用 Fisher 度量加权（自然梯度/precision-weighted step），而非欧氏梯度；**绝不**作用于冻结基底（守 R2）。
   - **预期收益**：曲流形上更稳定、不易 overshoot 的有界更新，与 trust-region/有界自修改一致。
   - **风险/前提**：自然梯度计算昂贵，仅适用于小控制器层；需与 ModificationGate 的门控配合。

4. **（低优先 / 思辨）gauge-等价作为可回滚判据。**
   - **目标 spec**：[`contract-runtime.md`](../../../../docs/specs/contract-runtime.md) / credit-and-self-modification（R15）。
   - **落地动作**：在提交 rare-heavy artifact 变更前，检查新参数化是否只是与现状 gauge-等价（模型证据相同）；仅接受真正提升证据的变更。
   - **预期收益**：避免控制器空转重参数化、强化可回滚性。
   - **风险/前提**：在 LLM 上难以操作化，证据/对称群估计成本高，列为远期探索。

---

## 3. 一句话定位

Stanhope = **边缘端 active inference**，为 VZ 的 **R-PE 提供最深的理论母体**（自由能 = precision-weighted PE 的统一闭环）并奉上 precision-weighting / Fisher 下界 / 自然梯度三件可解耦的局部工具；但其 **FEP 一元论红队了 VZ 的模块化 PE/credit/evaluation 拆分（needs-boundary-condition）**、其**"行动让预测自我实现"对关系信任是 genuine-risk**、其"可询问信念"因模型不变性而与 R11 具名状态貌合神离——且本地唯一可验证 PDF 是 2016 纯理论 Essay，所有端侧/数据高效工程主张**UNVERIFIED**。

## 附：本地论文清单（同目录 PDF）

| 论文 | 年 | 标识 | 状态 |
|---|---|---|---|
| Towards a Neuronal Gauge Theory | 2016 | PLOS Biology 14(3):e1002400, doi:10.1371/journal.pbio.1002400 | [本地 PDF] 已精读（Essay，无实验） |
| Neural Dynamics under Active Inference | 2021 | Entropy 23(4):454, doi:10.3390/e23040454 | [UNVERIFIED] 本目录未下载 |
| Active Inference: A Process Theory | 2017 | Neural Comput, doi:10.1162/neco_a_00912 | [UNVERIFIED] 本目录未下载 |
| The Free-Energy Principle | 2010 | Nat Rev Neurosci 11(2):127, doi:10.1038/nrn2787 | [UNVERIFIED] 本目录未下载 |
