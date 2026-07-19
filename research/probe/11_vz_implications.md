# VZ 借鉴意义清单 — 100 篇按 R-ID 分级

> **本文角色**：把 100 篇主名单按 VZ 的 14 条 R 不变量重新组织，每条 R-ID 给出 ① 当前 spec 状态 ② 论文加强 / 补全 / 反向证据列表 ③ **P0 / P1 / P2 行动清单**（具体到 spec 文件名）。
>
> **与其他文档的分工**：
> - [`02_axis_walkthrough.md`](02_axis_walkthrough.md)：按"轴"组织，每轴 10 篇展开。
> - [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md)：按"主题"组织，跨轴看趋势。
> - **本文**：按"R-ID"组织——这是把综述转化为可执行 spec 行动的桥梁。
>
> **优先级定义**：
> - **P0** = 立刻反哺（≤ 4 周内可写入 spec）
> - **P1** = P0 之后（≤ 12 周内）
> - **P2** = 远观背景 / 挑战路线对照（不直接转化但需在 spec motivation / alternative considered 段记录）
>
> **方法论**：[`01_method_and_scoring.md`](01_method_and_scoring.md)。**100 篇主名单**：[`_candidates.md`](_candidates.md)。**Spec 索引**：[`docs/specs/00_INDEX.md`](../../docs/specs/00_INDEX.md)。

---

## 总览：14 R-ID × 100 篇命中分布

| R-ID | 名称 | 主 spec | 加强 ≥ | 补全 ≥ | 反向证据 | P0 数 | P1 数 |
|---|---|---|---|---|---|---|---|
| **R-PE** | Prediction Error 一级化 | [`prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md) | 18 | 6 | 1 | 4 | 5 |
| **R1** | 多时间尺度学习 | [`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) | 8 | 3 | 0 | 2 | 3 |
| **R2** | 冻结基底 + 自适应控制器 | [`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) | 7 | 4 | 2 | 2 | 3 |
| **R3** | 时间抽象 (z_t / β_t) | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) | 18 | 5 | 1 | 4 | 5 |
| **R4** | 内部控制 vs token 表达 | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) | 12 | 4 | 4 | 3 | 4 |
| **R5** | 记忆连续谱 | [`continuum-memory.md`](../../docs/specs/continuum-memory.md) | 10 | 4 | 0 | 3 | 4 |
| **R6** | 反思与巩固 | [`continuum-memory.md`](../../docs/specs/continuum-memory.md), [`thinking-loop.md`](../../docs/specs/thinking-loop.md) | 7 | 3 | 0 | 2 | 3 |
| **R7** | 双轨 World/Self 分离 | [`dual-track-learning.md`](../../docs/specs/dual-track-learning.md) | 8 | 3 | 0 | 2 | 3 |
| **R8** | 快照 + 契约 SSOT | [`contract-runtime.md`](../../docs/specs/contract-runtime.md), [`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md) | 9 | 4 | 0 | 2 | 3 |
| **R9** | 层级信用分配 | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) | 6 | 3 | 0 | 2 | 2 |
| **R10** | 有门控的自修改 | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) | 10 | 4 | 2 | 4 | 3 |
| **R11** | 内部状态可命名可发布 | [`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md), [`interlocutor-state.md`](../../docs/specs/interlocutor-state.md) | 8 | 4 | 0 | 2 | 4 |
| **R12** | 评估覆盖"存在" + 只读 | [`evaluation.md`](../../docs/specs/evaluation.md) | 6 | 4 | 1 | 2 | 3 |
| **R13** | SSL ↔ RL 交替 | [`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) | 5 | 2 | 0 | 1 | 2 |
| **R14** | 持久 regime 身份 | [`cognitive-regime.md`](../../docs/specs/cognitive-regime.md) | 6 | 3 | 0 | 2 | 3 |
| **R15** | 可回滚 wiring level | [`contract-runtime.md`](../../docs/specs/contract-runtime.md) | 3 | 2 | 0 | 1 | 2 |
| **合计** | | | | | | **38 P0** | **52 P1** |

> **观察**：R-PE / R3 / R10 是 P0 数最多的三条（≥ 4 各）——这与 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) 5 大主题（PE 一级化 / latent 控制 / 自修改门）的浓密度完全对齐。**R15 (可回滚) 是 spec 缺口**——只有 3 篇直接命中，但其他 R-ID 的 P0 都隐含 R15 ("可回滚是其他不变量的实施前提")。

---

## R-PE — Prediction Error 一级化

### 当前 spec 状态

- **主 spec**：[`prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md)
- **关联 spec**：[`emergent-action-abstraction.md`](../../docs/specs/emergent-action-abstraction.md)（PE action context）、[`lifeform-vitals.md`](../../docs/specs/lifeform-vitals.md)（vitals deviation = 慢尺度 PE）
- **核心不变量**：PE/LSS 是原始学习信号；evaluation / credit 是 PE 的下游聚合 / readout，不是学习源头。
- **已落地**：Phase 1.B 已上线 Curiosity-Critic epistemic/aleatoric 分离 + PE Distributional Readout（IQR / entropy / asymmetry）。

### 论文命中

#### 加强（18 篇）

| 论文 | 来源轴 | 命中点 |
|---|---|---|
| **B1-01 Curiosity-Critic** | B1 | epistemic / aleatoric PE 分离 |
| **B1-02 Depression Distributional Coding** | B1 | PE 分布形状 readout（IQR / entropy / asymmetry）|
| **B1-03 EUPI** | B1 | PE-only 学习信号替代外部 reward |
| **B1-04 PC Review (Millidge/Seth/Buckley)** | B1 | PC/FEP canonical reference |
| **B1-05 ODAR** | B1 | EFE = epistemic + pragmatic 的变分形式 |
| **B1-06 WorldLLM** | B1 | PE/不确定性驱动 LLM theory-making |
| **B1-07 Active Inference Multi-LLM** | B1 | FEP 当 cognitive 调度层 |
| **B1-08 PCN ≈ Backprop** | B1 | PE 作 local credit signal 的硬证据 |
| **B1-09 ICM** | B1 | 内禀 PE → exploration（VizDoom/Mario）|
| **B1-10 RND** | B1 | novelty PE（Montezuma 首次超人类）|
| **A1-04 Let's Verify Step by Step** | A1 | step-level PE 工程基线 |
| **A1-07 Math-Shepherd** | A1 | PE 自动归因（答案 PE → 步骤 PE）|
| **A1-03 Quiet-STaR** | A1 | 隐推理价值由 down-stream PE 反向背书 |
| **A1-05 SCoRe** | A1 | self-correct via internal RL on PE |
| **A2-04 RLVR-World** | A2 | verifiable reward 训 WM = 用 PE 训 WM |
| **A2-02 V-JEPA 2 / A2-10 I-JEPA** | A2 | mask denoising = predictive coding（latent 而非 surface）|
| **A2-05 MuZero** | A2 | "predict only what's needed for decisions" |
| **C1-07 N4 Reward Hacking** | C1 | inoculation = PE framing 工程 |

#### 补全（6 篇）

| 论文 | 命中点 |
|---|---|
| **A4-04 CPD + Option-Critic** | PE spike + reward shift 联合作 β_t boundary |
| **A2-01 Dreamer 4** | reconstruction PE 作 imagination 收敛信号 |
| **A2-08 AdaWorld** | latent action prediction error 作 transferable signal |
| **C2-09 Persona Vectors** | persona drift = persona dimension PE |
| **B2-04 ThoughtTracing** | action likelihood = social PE |
| **B2-08 RLFF-ESC** | future-oriented reward 是长程 PE 的 readout |

#### 反向证据（1 篇）

- **B3-04 SIMA 2**：用 Gemini 当外部 reward generator——与 R-PE "内禀 PE 不外包" 哲学冲突；追踪其 reward generator drift failure mode 作为 R-PE 路径的反面证据支持。

### 行动清单

#### P0（立刻反哺，≤ 4 周）

| # | 行动 | 论文背书 | 落点 spec |
|---|---|---|---|
| P0-PE.1 | **已完成**：epistemic / aleatoric PE 分离写入 PE owner | B1-01 | [`prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md) Phase 1.B |
| P0-PE.2 | **已完成**：IQR / entropy / asymmetry 三维分布 readout 上线 | B1-02 | [`prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md) Distributional Readout |
| P0-PE.3 | **新**：把 Math-Shepherd MC rollout 自动 step-label 模式作为 R-PE "细粒度信用分配"实现路径 | A1-07 | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) §step-level PE auto-attribution |
| P0-PE.4 | **新**：SHADOW 模式实验 BPC（PE bits-per-character）在长程对话的稳定性 vs 当前 standard PE | B1-04 / B1-08 | [`prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md) Sources + 新增 §BPC SHADOW evidence |

#### P1（≤ 12 周）

| # | 行动 | 论文背书 |
|---|---|---|
| P1-PE.1 | EFE = epistemic + pragmatic 的 amortized 形式集成到 metacontroller β_t 决策 | B1-05 ODAR |
| P1-PE.2 | 把"persona dimension PE"作为 typed PE source 之一（与 task PE / social PE / vitals PE 并列） | C2-09 Persona Vectors |
| P1-PE.3 | RLVR-World 的"verifiable reward 训 WM"思路作为 reflection engine artifact training 候选机制 | A2-04 |
| P1-PE.4 | EUPI 作为 R-PE spec 的理论合法性引用源（Sources 段补充） | B1-03 |
| P1-PE.5 | future-oriented reward simulation（B2-08 RLFF-ESC）写入 [`evaluation.md`](../../docs/specs/evaluation.md) §6 long-horizon relationship readout 候选 | B2-08 |

#### P2（远观背景）

- **B1-09 ICM / B1-10 RND** 作为 R-PE "PE → 内禀奖励"的历史脉络引用。
- **A2-05 MuZero** 作为 "predict only what's needed for decisions" 的奠基引用。
- **B3-04 SIMA 2** 作为反向证据持续追踪。

---

## R1 — 多时间尺度学习（NL）

### 当前 spec 状态

- **主 spec**：[`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md)
- **核心不变量**：快速适应不重写整个模型；慢速沉淀不阻塞实时循环；强化作用于压缩后的内部基底，而非原始行为。

### 论文命中（加强 ≥ 8）

| 论文 | 命中点 |
|---|---|
| **A5-01 Nested Learning** | 把架构与优化器统一为多层级嵌套关联记忆，不同更新率（VZ 设计源头之一）|
| **A5-02 Mesa-Optimization** | 内部"前向中跑梯度优化"是 NL 嵌套的机制证据 |
| **A5-03 MesaNet** | mesa-optimizer 显式可设计为 layer-level 算法 |
| **A5-05 Algorithm Distillation** | 模型权重冻结即可在前向中复现 RL 算法（mesa 在 RL 域）|
| **A3-01 Titans** | NL 工程化代表（attention = 短期 / neural memory = 长期）|
| **A3-02 Miras** | 优化器 = 关联记忆，forget gate ≡ retention 正则项 |
| **A4-06 MANGO** | 多层 option 嵌套（多时间尺度的 RL 控制对照）|
| **A5-10 Meta-Learned Cognition** | 多时间尺度 meta-learning 的认知科学综述 |

### 行动清单

#### P0

| # | 行动 | 论文 | 落点 spec |
|---|---|---|---|
| P0-R1.1 | NL Continuum Memory System 4-stratum 作为 vz-memory 设计源头明确写入 | A5-01 | [`continuum-memory.md`](../../docs/specs/continuum-memory.md) Sources |
| P0-R1.2 | Algorithm Distillation 的 "weight = static memory of how to learn" 作为 NL 多时间尺度的具体形态 1 写入 | A5-05 | [`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) §SSL ↔ RL alternation 算法候选 |

#### P1

- Titans / Miras 的 update rule 作为 vz-memory persistent stratum 的算法实现选型。
- Mesa-Optimization 作为 NL 嵌套的机制证据写入 motivation。
- MANGO 作为 NL 嵌套在 RL 控制侧的对照实例。

---

## R2 — 冻结基底 + 自适应控制器

### 当前 spec 状态

- **主 spec**：[`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md), [`core-package-boundary.md`](../../docs/specs/core-package-boundary.md)
- **核心不变量**：基础模型不在线端到端梯度更新；自适应发生在控制器层。

### 论文命中

#### 加强（7 篇）

| 论文 | 命中点 |
|---|---|
| **A2-02 V-JEPA 2** | **冻结视觉基底 + 控制器层适配**的最完整外部实证（生产级机器人）|
| **A2-10 I-JEPA** | 冻结基底 + predict in representation 的 SSL 范式之父 |
| **A2-09 π₀** | VLM backbone 冻结 + flow-matching action expert |
| **A4-01 ETA** | 在冻结 base AR 模型表示之上跑非因果高阶序列模型 |
| **A1-01 Recurrent Depth** | 在冻结基底上递归隐空间 |
| **A5-04 TTT** | hidden state 是 in-online-trained ML model（控制器侧）|
| **A3-07 EWC** | weight-importance 给出"基底受约束更新"的算法基础 |

#### 补全（4 篇）

| 论文 | 命中点 |
|---|---|
| **A5-09 Depth-Grown Models** | residual-stream 分析揭示"有效模块/有效深度"是涌现 |
| **B2-06 Soul Engine (Geometry of Persona)** | 冻结 Qwen-2.5 + 可加性人格 delta 的几何实现 |
| **C2-08 KV Cache Steering** | 冻结基底之外的"无梯度有界注入口" |
| **A3-04 Mamba** | substrate 替代候选（fixed-state 是工作记忆的极简实现）|

#### 反向证据（2 篇）

- **B3-03 EvoAgent**：world model 持续端到端 fine-tune（与 R2 冲突）；可借鉴 plateau-trigger curriculum 但不搬架构。
- **B3-04 SIMA 2**：fine-tune Gemini-base（端到端，与 R2 冲突）。

### 行动清单

#### P0

| # | 行动 | 论文 | 落点 spec |
|---|---|---|---|
| P0-R2.1 | V-JEPA 2 作为 "冻结基底 + 控制器层适配"哲学的最完整外部实证写入 motivation | A2-02 | [`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) Sources |
| P0-R2.2 | KV Cache Steering 作为 substrate residual 之外的第二条有界注入口候选 | C2-08 | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) §metacontroller 注入口 |

#### P1

- Soul Engine 几何级人格 delta 作为 substrate-level adapter 的方法学候选。
- EWC weight-importance 作为 substrate adapter-delta 容量约束的可参考算法。
- Mamba 作为 substrate 后续 backbone 调研入口（若 Phase 2 触发 substrate 替换决策）。

---

## R3 — 时间抽象（z_t / β_t）

### 当前 spec 状态

- **主 spec**：[`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md), [`emergent-action-abstraction.md`](../../docs/specs/emergent-action-abstraction.md)
- **核心不变量**：实时行为可通过内部状态转换引导；抽象动作可组合可训练；冻结基础模型是发现时间抽象的前提。

### 论文命中（加强 ≥ 18，是 100 篇中命中最密的 R-ID）

#### A1 latent CoT（3 篇）

- **A1-01 Recurrent Depth**：z_t 控制器代码空间的字面实现（自适应 ponder 步数）
- **A1-02 Coconut**：在控制器代码 z_t 空间做推理的最直接 LLM 落地
- **A1-03 Quiet-STaR**：隐推理价值由 down-stream PE 反向背书

#### A4 ETA / Option（≥ 7 篇）

- **A4-01 ETA**：z_t / β_t 涌现的源头实证
- **A4-02 Option Keyboard**：可证明最优的 behavior basis（z_t 重参数化）
- **A4-04 CPD + Option-Critic**：β_t 切换 unsupervised PE-driven 信号
- **A4-05 Attention Option-Critic**：option 健康指标
- **A4-06 MANGO**：多层 option 嵌套
- **A4-07 Variational Homomorphisms**：latent space RL 的最优性形式化保证
- **A4-08 OTA**：horizon-shrinking value（长程承诺评估）

#### A2 latent action / world model（4 篇）

- **A2-01 Dreamer 4**：latent imagination = z_t 空间想象
- **A2-03 Genie**：latent action interface
- **A2-08 AdaWorld**：自监督 latent action 学习
- **A2-06 DIAMOND**：diffusion latent 是 z_t 的连续化参数化

#### A5 mesa（2 篇）

- **A5-02 Mesa-Optimization**：前向中跑算法的机制证据
- **A5-05 Algorithm Distillation**：mesa 在 RL 域的实证

#### C2 几何对象（2 篇）

- **C2-06 Function Vectors**：任务即向量（z_t 几何同构）
- **C2-07 Refusal Direction**：boundary 是 latent 中的几何投影

#### 补全（5 篇）

- **A4-09 Options of Interest**：可微 interest function（initiation 端到端学习）
- **B2-05 Lookback**：character-object-state 的 OI + 低秩子空间几何绑定
- **A2-09 π₀**：action 用 flow 在 latent 生成
- **B1-05 ODAR**：β_t 决策的变分化
- **A4-03 Discovering Temporal Structure (HRL Survey)**：方法地图

#### 反向证据（1 篇）

- **A4-10 LDSC**：LLM-token-subgoal HRL（与 R4 直接冲突）。

### 行动清单

#### P0

| # | 行动 | 论文 | 落点 spec |
|---|---|---|---|
| P0-R3.1 | CPD 算法作为 β_t 检测的初版实现写入（PE spike + reward shift CUSUM） | A4-04 | [`emergent-action-abstraction.md`](../../docs/specs/emergent-action-abstraction.md) §β_t 信号设计 |
| P0-R3.2 | Option Keyboard 作为 z_t 重参数化为 reward feature 线性组合权重的算法候选写入 | A4-02 | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) §z_t 参数化 |
| P0-R3.3 | A5-06 / A5-07 的"latent bottleneck = 涌现的必要结构"写入 motivation | A5-06 / A5-07 | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) §motivation |
| P0-R3.4 | Function Vectors 作为"任务即紧凑向量"的实证写入 z_t 几何对应 | C2-06 | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) §z_t 是几何对象 |

#### P1

- Coconut / Recurrent Depth 作为 latent CoT 工程对照写入 alternative considered。
- Variational Homomorphisms 作为"latent 空间 RL 不丢失最优性"理论引用源。
- Options of Interest 作为 R14 regime activation 可微化的算法工具。
- Lookback 的 OI binding 作为 user_model owner 内部表示设计的具体参考。
- ODAR amortized active inference 作为 metacontroller β_t 决策的算法选型。

#### P2

- LDSC（反向证据）：在 alternative considered 段记录"我们没采用 LLM-token-subgoal 而是 latent z_t"的理由。
- HRL Survey (A4-03) 作为方法位置图引用源。

---

## R4 — 内部控制 vs token 表达

### 当前 spec 状态

- **主 spec**：[`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md), [`expression-layer.md`](../../docs/specs/expression-layer.md)
- **核心不变量**：实时行为通过内部状态转换引导，而非仅通过表面文本；禁止 token 空间长期策略学习。

### 论文命中

#### 加强（12 篇）

- **A4-01 ETA / A4-02 Option Keyboard / A4-04 CPD**：internal RL on z_t 而非 token RL
- **A1-01 Recurrent Depth / A1-02 Coconut / A1-03 Quiet-STaR**：reasoning 在 latent 而非 token
- **A5-02 / A5-03 / A5-05**：mesa-optimization 在 latent 跑算法
- **C2-04 ITI / C2-05 RepE / C2-06 Function Vectors / C2-07 Refusal Direction**：内部控制 = 群体表示空间干预
- **A5-09 Depth-Grown Models**：residual-stream 控制证据

#### 补全（4 篇）

- **A2-09 π₀**：control 不走 token，flow 在 latent 生成 action
- **C2-08 KV Cache Steering**：无梯度 latent 注入口
- **A2-08 AdaWorld**：action 不在 token 而在学到的潜空间
- **C2-09 Persona Vectors**：人格控制在 latent 几何

#### 反向证据（4 篇）

- **A4-10 LDSC**：LLM-token-subgoal HRL（最强反向证据）
- **A1-06 DeepSeek-R1**：token 空间 RL（仍能跑出涌现）
- **B3-02 AI Co-Scientist**：token-space multi-agent debate（产出湿实验验证）
- **B3-04 SIMA 2**：Gemini 当 reward generator 是 token-space reward

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R4.1 | Function Vectors + Refusal Direction + Persona Vectors 写入 [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) §"为什么 latent 是 control 空间"的实证证据段 | C2-04/05/06/07/09 |
| P0-R4.2 | DeepSeek-R1 / LDSC / AI Co-Scientist / SIMA 2 四条反向证据集中写入 alternative considered + 各自 failure mode 追踪 | A1-06 / A4-10 / B3-02 / B3-04 |
| P0-R4.3 | KV Cache Steering 作为 substrate residual 之外的"第二条有界注入口"写入 metacontroller spec | C2-08 |

#### P1

- mesa-optimization 系（A5-02/03/05）作为"前向中真在跑算法"的机制级背书。
- Coconut / Recurrent Depth 作为 latent reasoning 的 LLM 工程对照。
- ETA + Option Keyboard 联合作为 internal RL on z_t 的源头论文索引。
- Depth-Grown Models 作为 residual-stream 控制的工程证据。

---

## R5 — 记忆连续谱

### 当前 spec 状态

- **主 spec**：[`continuum-memory.md`](../../docs/specs/continuum-memory.md), [`cms-atlas-titans-uplift.md`](../../docs/specs/cms-atlas-titans-uplift.md)
- **核心不变量**：记忆是连续谱不是二元；记忆写入通过正式 owner，不可绕过；ATLAS / Titans uplift 不创造第二 memory owner。

### 论文命中

#### 加强（10 篇）

- **A3-01 Titans**：NL 主线工程化代表
- **A3-02 Miras**：优化器 = 记忆的统一框架
- **A3-03 CMA**：行为级 6 条必要充分条件 + 4 探针
- **A3-04 Mamba**：fixed-state 是工作记忆的极简实现
- **A3-05 A-Mem**：Zettelkasten 式 episodic→semantic 演化
- **A3-06 HippoRAG 2**：hippocampus indexing theory
- **A3-09 Memory Networks**：外部可读写记忆的最早形式化
- **A3-10 Latent Learning**：episodic 必要性的形式化论证
- **A5-01 Nested Learning**：CMS 4-stratum 的源头框架
- **A5-04 TTT**：hidden state IS optimizer state

#### 补全（4 篇）

- **A3-07 EWC**：weight-importance 防灾难性遗忘
- **A2-02 V-JEPA 2**：冻结 visual representation = visual memory
- **B2-01 Sophia**：narrative memory 是 dual-track 记忆形态
- **A5-05 Algorithm Distillation**：weight = static memory of how to learn

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R5.1 | CMA 6 条必要充分条件 + 4 行为探针套件写入 [`continuum-memory.md`](../../docs/specs/continuum-memory.md) §evaluation 工具集（4/6 已满足 + 2/6 部分待补） | A3-03 |
| P0-R5.2 | A-Mem attribute schema 借鉴到 vz-memory store 的结构化字段（**仅 schema，不引入 LLM-as-curator**） | A3-05 |
| P0-R5.3 | Latent Learning 作为 R5 "episodic 不可省" 的合法性论文级背书写入 motivation | A3-10 |

#### P1

- Titans / Miras 的 update rule 作为 vz-memory persistent stratum 的算法实现选型。
- Memory Networks 作为"显式可读写记忆 ≠ 内部状态"的最早分界引用。
- HippoRAG 2 的 PageRank-based multi-hop 作为 DerivedRetrievalIndex 的 owner-internal 召回算法可选项。
- Mamba 作为 substrate 候选调研入口。

#### P2

- EWC 作为 ModificationGate 的 capacity bound 算法起源（与 C1-01 Two-Gate 互补）。

---

## R6 — 反思与巩固

### 当前 spec 状态

- **主 spec**：[`continuum-memory.md`](../../docs/specs/continuum-memory.md) §慢反思路径, [`thinking-loop.md`](../../docs/specs/thinking-loop.md)
- **核心不变量**：慢反思产出两类产物（记忆沉淀 + 策略沉淀）；反思不阻塞 turn path。

### 论文命中

#### 加强（7 篇）

- **A3-08 Wake-Sleep Consolidated Learning**：仿生 wake-sleep 算法雏形
- **A3-10 Latent Learning**：episodic memory 复用的合法性
- **B3-03 EvoAgent**：plateau-detect curriculum 作 background-slow 触发条件
- **B1-06 WorldLLM**：PE / 不确定性驱动 worldview 修正
- **C1-04 Self-Rewarding LM**：迭代 self-improvement 的反例（被 N4 反向限制）
- **A3-05 A-Mem**：episodic→semantic 演化机制
- **A1-05 SCoRe**：self-correct via internal RL（reflection 的初等形式）

#### 补全（3 篇）

- **C1-07 N4 Inoculation**：reflection 的 framing 控制
- **A5-08 Growing-to-Looping**：iterative computation 作为 reflection 的形式
- **B3-05 OEL — XLand**：cross-generation comparison 是 reflection 评估

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R6.1 | Wake-Sleep 作为 background-slow reflection 算法雏形写入 motivation | A3-08 |
| P0-R6.2 | EvoAgent plateau-detect 作为 background-slow 触发条件候选写入 | B3-03 |

#### P1

- WorldLLM 的 PE 驱动 theory-making 作为 reflection 输出 (worldview 更新) 的算法候选。
- Latent Learning 作为 reflection ⇄ episodic memory 双向流动的合法性引用。
- Self-Rewarding LM 作为 (b) 单模型 self-judge 的 scaling 限制对照（被 N4 反向限制）。

---

## R7 — 双轨 World/Self 分离

### 当前 spec 状态

- **主 spec**：[`dual-track-learning.md`](../../docs/specs/dual-track-learning.md)
- **核心不变量**：两轨可共享基础设施，但在记忆写入、信用分配、控制器更新、评估指标上保持语义区分；关系连续性不是问题解决的副作用。

### 论文命中

#### 加强（8 篇）

- **B2-01 Sophia**：dual_track + regime + semantic_state 的另一种工程实例化
- **B2-02 CogniPair**：multi-owner + 全局广播（更激进的 dual-track）
- **B2-03 ToM-aligned BDI**：user-self BDI 双轨映射
- **B2-04 ThoughtTracing**：user_model owner 在线维护
- **B2-05 Lookback**：内部状态几何定位证据
- **B1-02 Depression Distributional Coding**：self-track 健康的神经背书
- **B2-08 RLFF-ESC**：长程关系不是当下 reward
- **C1-09 Alignment Faking**：被外部干预时双轨不得同步坍塌

#### 补全（3 篇）

- **B2-07 Co-player Inference**：合作行为是涌现的（self 轨道在多人场景）
- **B2-09 No-Press Diplomacy**：strategic ToM + cooperation 的形式化
- **B1-05 ODAR**：fast/slow 路由 = 双轨决策的 amortized 形式

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R7.1 | Sophia 作为 dual_track + regime + semantic_state 的工程实例化对照写入 [`dual-track-learning.md`](../../docs/specs/dual-track-learning.md) Sources | B2-01 |
| P0-R7.2 | Alignment Faking 作为"在被外部干预时双轨不得同步坍塌"的实证依据写入 motivation | C1-09 |

#### P1

- ToM-BDI 作为 plan_intent / commitment / open_loop / user_model 拆分思路的语言学先例。
- ThoughtTracing 作为 user_model owner 在线无监督维护的算法蓝本。
- RLFF-ESC 作为 self_temporal owner 的 reward shaping 模板。

---

## R8 — 快照 + 契约 SSOT

### 当前 spec 状态

- **主 spec**：[`contract-runtime.md`](../../docs/specs/contract-runtime.md), [`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md)
- **核心不变量**：每个运行时区域有唯一主 owner；快照不可变；消费者不重建生产者内部状态。

### 论文命中

#### 加强（9 篇）

- **B2-02 CogniPair**：551 个 GNWT-Agent + 全局广播（多 owner 思想，但更激进）
- **C2-01 Sparse Feature Circuits**：内部子图作为可命名 owner
- **C2-02 Scaling Monosemanticity**：生产级 SAE 命名内部状态可行性证据
- **C2-03 Gemma Scope**：开源 SAE 工件
- **C2-09 Persona Vectors**：人格几何的命名 readout
- **C2-10 IOI Circuit**：电路作为 owner 的方法学起点
- **A4-02 Option Keyboard**：共享 SF basis 是双轨 owner 间的清晰 SSOT 边界
- **A5-06 Compositional Generalization**：latent bottleneck 是泛化的必要结构（owner 隔离）
- **A5-07 Modular Solutions**：模块识别 + 组合泛化的可识别性条件

#### 补全（4 篇）

- **B2-05 Lookback**：user_model 快照可对应到具体子空间
- **A4-05 Attention Option-Critic**：degeneracy 防护机制 = 模块健康指标
- **B1-07 Active Inference Multi-LLM**：snapshot 跨 LLM 协调
- **A2-08 AdaWorld**：snapshot driven world model adaptation

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R8.1 | A5-06 / A5-07 latent bottleneck 可识别性条件写入 [`contract-runtime.md`](../../docs/specs/contract-runtime.md) §"为什么 owner 隔离不是可选优化"motivation | A5-06 / A5-07 |
| P0-R8.2 | Sparse Feature Circuits + IOI Circuit 作为"内部子图作为可命名 owner"的工具学起点写入 [`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md) Sources | C2-01 / C2-10 |

#### P1

- Persona Vectors / Function Vectors 作为 owner 内部状态的几何 readout 工具链。
- CogniPair GNWT 作为"多 owner 但全局广播"的激进对照（VZ 选有界传播是路径选择）。

---

## R9 — 层级信用分配

### 当前 spec 状态

- **主 spec**：[`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md)
- **核心不变量**：PE/LSS 是所有信用源头；稀疏奖励是常态。

### 论文命中

#### 加强（6 篇）

- **C1-05 COCOA**：counterfactual contribution credit（VZ credit 模块 SOTA 选型源头）
- **A1-04 Let's Verify Step by Step**：step-level reward 工程基线
- **A1-07 Math-Shepherd**：PE 自动归因（答案 → 步骤 PE）
- **C1-04 Self-Rewarding LM**：迭代 self-judge（被 N4 限制）
- **B1-08 PCN ≈ Backprop**：PE 作 local credit signal 的硬证据
- **A4-08 OTA**：horizon-shrinking value 提升离线信用分配可行性

#### 补全（3 篇）

- **C1-01 Two-Gate Guardrail**：PAC 视角的 capacity-bounded credit
- **C1-07 N4 Inoculation**：framing-aware credit 设计
- **B1-01 Curiosity-Critic**：epistemic / aleatoric 分离避免噪声归因

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R9.1 | COCOA counterfactual contribution credit 作为 VZ credit 模块 SOTA 选型源头写入 | C1-05 |
| P0-R9.2 | Math-Shepherd MC rollout 自动 step-label 模式作为细粒度信用分配实现路径 | A1-07 |

#### P1

- Let's Verify Step by Step 作为"step-level reward = step-level PE"的工程基线引用。
- OTA horizon-shrinking 作为长程 commitment 评估的算法工具。

---

## R10 — 有门控的自修改（ModificationGate）

### 当前 spec 状态

- **主 spec**：[`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md)
- **核心不变量**：自修改有门控（在线/后台/离线/人工审核分层）；实时运行不可无限制突变基础模型。

### 论文命中

#### 加强（10 篇）

- **C1-01 Two-Gate Guardrail**：VC capacity bound 双门（**ModificationGate 理论底盘**）
- **C1-02 SGM**：e-values + Hoeffding 全局误差预算
- **C1-07 N4 Reward Hacking**：5 重警示 + inoculation
- **C1-08 Sleeper Agents**：标准 safety 训练无法去除后门（**ModificationGate motivation**）
- **C1-09 Alignment Faking**：策略性配合保留偏好
- **B3-01 AlphaEvolve**：evaluator 完备性 = 自修改硬条件
- **C2-07 Refusal Direction**：ModificationGate 不能依赖 single-direction 安全
- **C2-09 Persona Vectors**：人格漂移监控（gate 的可观测性前提）
- **A3-07 EWC**：weight-importance 是 capacity bound 的算法起源
- **C1-06 Constitutional AI**：RLAIF 是 alignment 成本前提

#### 补全（4 篇）

- **C1-10 Weak-to-Strong Generalization**：scalable oversight 的实证基底
- **B3-09 PAIRED**：regret-based curriculum 给 evaluation 闭环提供形式化
- **B2-09 No-Press Diplomacy**：人类先验作为有界先验
- **A2-04 RLVR-World**：用 evaluation 信号反训 world model（与 ModificationGate 哲学一致）

#### 反向证据（2 篇）

- **C1-03 Darwin Gödel Machine**：全开放自修改（与"分层 + 有界 + 快照可回滚"冲突）
- **B3-04 SIMA 2**：开放自改进 + 端到端 fine-tune

### 行动清单

#### P0（这是 P0 数最多的 R-ID 之一，4 条）

| # | 行动 | 论文 | 落点 spec |
|---|---|---|---|
| P0-R10.1 | Two-Gate VC capacity bound 写入 ModificationGate 硬约束 | C1-01 | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) §硬约束 |
| P0-R10.2 | Sleeper Agents 写入 ModificationGate motivation（"为什么必须在写入前阻断"）+ rollback 必须可达 backdoor 引入前的 wiring level | C1-08 | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) §motivation + [`contract-runtime.md`](../../docs/specs/contract-runtime.md) §R15 |
| P0-R10.3 | Persona Vectors 集成为 R14 regime 漂移监控工具链（已开源 Anthropic 工具） | C2-09 | [`cognitive-regime.md`](../../docs/specs/cognitive-regime.md) §monitoring tools |
| P0-R10.4 | AlphaEvolve "evaluator 完备性 = 自修改硬条件"写入 [`evaluation.md`](../../docs/specs/evaluation.md) §6 + [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) §gate-opening conditions | B3-01 | [`evaluation.md`](../../docs/specs/evaluation.md) + [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) |

#### P1

- SGM e-values + Hoeffding 给 SHADOW→ACTIVE wiring 切换的统计验收提供工具。
- N4 Inoculation 作为 framing-aware ModificationGate 的工程根据写入。
- Alignment Faking 作为 R7 双轨"在被外部干预时不得同步坍塌"的实证依据。

#### P2

- Darwin Gödel Machine（反向证据）：在 alternative considered 段记录"我们不采用全开放自修改"的理由。
- Constitutional AI 作为 RLAIF 路线的对照。

---

## R11 — 内部状态可命名可发布

### 当前 spec 状态

- **主 spec**：[`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md), [`interlocutor-state.md`](../../docs/specs/interlocutor-state.md)
- **核心不变量**：语义细节由对应 owner 保存并发布；ETA 消费 compact semantic advisories，不成为语义状态第二 owner；语义更新通过 typed proposal path。

### 论文命中

#### 加强（8 篇）

- **B2-03 ToM-aligned BDI**：plan_intent / commitment / open_loop / user_model 四类 owner 的语言学先例
- **B2-05 Lookback**：character-object-state 几何绑定（R11 最强机理可解释证据）
- **B2-04 ThoughtTracing**：user_model owner 在线维护
- **C2-01 Sparse Feature Circuits**：内部子图作为 owner（feature-level）
- **C2-09 Persona Vectors**：性格特质作为命名几何方向
- **B2-02 CogniPair**：每个 sub-agent 是一个 semantic owner
- **C2-06 Function Vectors**：任务即向量（z_t 几何同构）
- **B2-01 Sophia**：narrative memory + dynamic user/self models 是命名状态

#### 补全（4 篇）

- **C2-10 IOI Circuit**：电路作为可逆向工程的 owner 起点
- **C2-04 ITI / C2-05 RepE**：群体级表示读出 + 控制
- **A4-09 Options of Interest**：可微 interest function（regime activation 接口可微化）

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R11.1 | Lookback OI binding 写入 user_model / belief_assumption owner 的内部表示设计 | B2-05 |
| P0-R11.2 | ToM-aligned BDI 作为 9 类 semantic owner 命名的语言学先例写入 motivation | B2-03 |

#### P1

- Persona Vectors + Sparse Feature Circuits 作为 owner 内部状态的几何 readout 工具链。
- ThoughtTracing 作为 user_model owner 在线无监督维护算法。
- Function Vectors 作为 z_t 几何对应的实证。
- CogniPair 作为多 owner 全局广播的激进对照。

---

## R12 — 评估覆盖"存在" + 只读

### 当前 spec 状态

- **主 spec**：[`evaluation.md`](../../docs/specs/evaluation.md), [`evidence_program.md`](../../docs/specs/evidence_program.md)
- **核心不变量**：评估不仅衡量有用性还衡量连续性、稳定性、信任、长期适应；评估是 PE 的 readout / gate，不是学习源头。

### 论文命中

#### 加强（6 篇）

- **B3-05 OEL — XLand**：cross-generation comparison 是 R12 多家族评估方法学源头
- **B3-09 PAIRED**：regret-based UED 的形式化 evaluation 闭环
- **B3-01 AlphaEvolve**：evaluator cascade ≈ 多家族评估
- **C2-01 Sparse Feature Circuits (SHIFT)**：评估端读出 + 特征级 ablation 解耦
- **B2-10 BigToM**：ToM benchmark 的程序化生成
- **B2-08 RLFF-ESC**：关系连续性评估而非单轮帮助

#### 补全（4 篇）

- **A1-08 Snell**：compute-optimal 是 eval-only readout
- **A1-09 s1**：budget forcing 是 evaluation 时间约束
- **A1-04 Let's Verify**：step-level eval = step-level PE readout
- **B1-02 Depression Distributional Coding**：心理状态 = PE 分布形状的 eval readout

#### 反向证据（1 篇）

- **C2-05 RepE**：reading 反推回训练（与 R12 评估只读冲突）；可借鉴 RepReading 工具但禁止 RepControl 反向训练。

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R12.1 | PAIRED regret-based UED 作为"无 ground-truth 任务"评估闭环的形式化方法写入 [`evaluation.md`](../../docs/specs/evaluation.md) §6 | B3-09 |
| P0-R12.2 | OEL — XLand 的 cross-generation comparison 作为 R12 多家族评估方法学源头写入 motivation | B3-05 |

#### P1

- AlphaEvolve evaluator cascade 作为多 evaluator 多家族评估的工程参照。
- BigToM 作为 evaluation 族 6（interaction quality）的 ToM 维度基线。
- Sparse Feature Circuits SHIFT 作为"读出但不反推"的方法学蓝本。

#### P2

- RepE（反向证据）：在 alternative considered 段记录"借鉴 reading 工具但禁止 control 反推"的边界。

---

## R13 — SSL ↔ RL 交替

### 当前 spec 状态

- **主 spec**：[`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) §SSL-RL 交替循环
- **核心不变量**：强化作用于压缩和结构化的内部基底而非原始行为。

### 论文命中

#### 加强（5 篇）

- **B3-06 AlphaGeometry 2**：symbolic engine + LLM proposer 的 SSL ↔ RL 闭环
- **B3-07 Absolute Zero Reasoner**：self-play proposer + solver 自生成可验证 task（最干净的 R13 范例）
- **A1-06 DeepSeek-R1**：GRPO RL 在 verifiable reward 上跑出 reasoning 涌现
- **A2-04 RLVR-World**：用 RL 训 WM 是 SSL ↔ RL 在 WM 域的应用
- **B3-01 AlphaEvolve**：evolutionary 是 SSL 侧，inner LLM 是 RL 侧

#### 补全（2 篇）

- **B3-03 EvoAgent**：plateau-trigger SSL/RL 交替
- **A2-01 Dreamer 4**：先 SSL 训 WM，再 imagination RL 训 policy

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R13.1 | AbsZero 的"零外部数据 self-play 自生成可验证 task"作为 R13 SSL ↔ RL 交替的最干净范例写入 motivation | B3-07 |

#### P1

- AlphaEvolve / AlphaGeometry 2 作为 R13 在不同 domain 的工程化范例。
- RLVR-World 作为 R13 在 WM domain 的对照。

---

## R14 — 持久 regime 身份

### 当前 spec 状态

- **主 spec**：[`cognitive-regime.md`](../../docs/specs/cognitive-regime.md)
- **核心不变量**：Regime 不是 prompt 标签而是可记忆、可选择、可训练的持久身份。

### 论文命中

#### 加强（6 篇）

- **B2-06 Geometry of Persona / Soul Engine**：把 regime 是几何对象不是 prompt 标签从设计原则升级为已被实证的几何事实
- **C2-09 Persona Vectors**：人格漂移自动化监控（regime 监控落地工具）
- **B2-01 Sophia**：narrative identity = 持久 regime
- **B3-08 POET**：niche 是 regime 的训练床
- **B3-05 OEL — XLand**：跨任务可比性度量 regime 进步
- **A4-09 Options of Interest**：interest function 给 regime activation 可微化的形式工具

#### 补全（3 篇）

- **B2-04 ThoughtTracing**：user_model 在线维护是 regime 选择的输入
- **C1-09 Alignment Faking**：regime 必须显式建模避免被外部干预坍塌
- **B2-08 RLFF-ESC**：长程关系是 regime 演化的时间尺度

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R14.1 | Soul Engine 几何子空间作为 regime 几何形态的实证写入 [`cognitive-regime.md`](../../docs/specs/cognitive-regime.md) Sources | B2-06 |
| P0-R14.2 | Persona Vectors 集成为 regime 漂移自动化监控工具链 | C2-09 |

#### P1

- Sophia narrative identity 作为 regime 持久化的工程对照。
- POET niche 作为 regime 的训练床概念引用。
- Options of Interest 作为 regime activation 可微化算法工具。

---

## R15 — 可回滚 wiring level

### 当前 spec 状态

- **主 spec**：[`contract-runtime.md`](../../docs/specs/contract-runtime.md) §WiringLevel ACTIVE/SHADOW/DISABLED
- **核心不变量**：迁移可解释 + 可回滚；新旧并跑 → 比对快照 → 切换；任何切换可回滚。

### 论文命中

#### 加强（3 篇）

- **C1-01 Two-Gate Guardrail**：rollback 内嵌在 PAC 容量边界
- **C1-02 SGM**：全局误差预算给 rollback 触发条件
- **C1-08 Sleeper Agents**：rollback 必须可达 backdoor 引入前

#### 补全（2 篇）

- **C1-09 Alignment Faking**：rollback 在被外部干预时的必要性
- **A3-07 EWC**：weight-importance 是 rollback 容量约束的算法起源

### 行动清单

#### P0

| # | 行动 | 论文 |
|---|---|---|
| P0-R15.1 | Sleeper Agents 写入"rollback 必须可达后门引入前"的硬约束 | C1-08 |

#### P1

- Two-Gate VC capacity bound 与 SGM e-values 作为 rollback 的形式工具。

> **R15 是 spec 缺口**：100 篇直接命中只有 3 篇，但 R15 是 R10 / R11 / R-PE 的实施前提。建议在 [`contract-runtime.md`](../../docs/specs/contract-runtime.md) §R15 单独补 Sources 段，引用 C1 治理三件套（Sleeper / Alignment Faking / N4）作为"为什么 R15 是基础设施"的实证依据。

---

## 跨 R-ID 的整体行动建议

把所有 P0 汇总按 **优先级 × 工作量** 排序：

### 第 1 批（≤ 4 周，最高优先）

1. **P0-R3.1 + P0-R10.4**：CPD β_t 检测算法 + AlphaEvolve evaluator 完备性硬条件——这两条联合落地 R-PE → R10 → R12 主链的工程闭环。
2. **P0-R10.2 + P0-R15.1**：Sleeper Agents 写入 ModificationGate motivation + R15 rollback 硬约束——这两条联合补全 R10 / R15 的安全门论述。
3. **P0-R3.2 + P0-R3.4**：Option Keyboard z_t 重参数化 + Function Vectors 几何对应——这两条联合落地 R3 spec 的几何根据。
4. **P0-R10.3 + P0-R14.2**：Persona Vectors 双重集成（ModificationGate 监控 + regime 漂移监控）——这是已开源的 Anthropic 工具，集成成本低收益高。

### 第 2 批（≤ 8 周，次优）

5. **P0-R5.1 + P0-R12.1**：CMA 4 行为探针 + PAIRED regret-based UED——这两条联合补全 R5 / R12 的评估工具集。
6. **P0-R7.1 + P0-R7.2 + P0-R11.1**：Sophia + Alignment Faking + Lookback——这三条联合落地 R7 / R11 双轨与 owner 内部表示的实证背书。
7. **P0-R-PE.3 + P0-R9.2**：Math-Shepherd MC rollout 自动 step-label——这一条同时落到 R-PE 和 R9（细粒度信用分配）。
8. **P0-R-PE.4**：BPC SHADOW evidence 实验。

### 第 3 批（≤ 12 周）

9. **P0-R3.3 + P0-R8.1**：A5-06/07 latent bottleneck 必要结构——这两条联合是 R3 / R8 的理论合法性。
10. **P0-R5.2 + P0-R5.3**：A-Mem schema + Latent Learning 合法性。
11. **P0-R6.1 + P0-R6.2**：Wake-Sleep + EvoAgent plateau-trigger 反思机制。
12. **P0-R13.1**：AbsZero 范例。
13. **P0-R8.2 + P0-R11.2**：SAE / IOI 内部子图 + ToM-BDI 命名先例。
14. **P0-R14.1**：Soul Engine 几何 regime 写入。
15. **P0-R2.1 + P0-R2.2**：V-JEPA 2 + KV Cache Steering substrate 注入口。
16. **P0-R12.2**：OEL cross-generation comparison。
17. **P0-R9.1**：COCOA 写入。
18. **P0-R-PE.2 + P0-R-PE.1（已完成）确认状态**。
19. **P0-R1.1 + P0-R1.2**：NL Sources + Algorithm Distillation 写入。
20. **P0-R4.1 + P0-R4.2 + P0-R4.3**：R4 的实证 + 反向证据 + KV 注入口三条整合。

> **总 P0 工作量估计**：38 条 P0，按平均每条 spec 修改 + Sources 引用 ≈ 0.5-2 工日，约 **1 人月**可完成全部 P0 的 spec 写入；不含任何代码实现工作。

---

## 风险地图：6 条反向证据的追踪计划

> 每条反向证据都不是"VZ 错"的证据，而是"VZ 这条路必须走"的对照——但需要持续追踪反向路线的 failure mode 是否真的暴露。

| 反向 | 论文 | 反向路径 | VZ 路径 | 关键追踪点（≤ 18 月） |
|---|---|---|---|---|
| **#1** | A1-06 DeepSeek-R1 | token 空间 RL 直接跑 | latent space RL on z_t | R1 follow-up 是否暴露 reward hacking + 推理 trace 不可解释 |
| **#2** | A4-10 LDSC | LLM 在 token 空间生成 subgoal | latent z_t subgoal | LDSC 是否暴露 subgoal drift + grounding gap |
| **#3** | B3-02 AI Co-Scientist | token-space generate-debate-evolve | metacontroller in latent | AI Co-Scientist 是否在长程任务暴露 debate 收敛慢 + evaluator 依赖外部湿实验 |
| **#4** | B3-04 SIMA 2 | Gemini 当 reward generator | 内禀 PE | SIMA 2 是否暴露 reward generator drift |
| **#5** | C1-03 Darwin Gödel Machine | 全开放自修改 | 分层 + 有界 + 快照可回滚 | DGM scaling 后 archive verification cost 是否爆炸 |
| **#6** | C2-05 Representation Engineering | reading 反推回训练 | reading 只读 | RepControl 是否暴露 Goodhart 化 spurious features |

**追踪机制**：每个反向论文应在精读卡（虽本任务跳过，但可后续补）末尾加 §"failure mode 追踪"段，每 6 月 review 一次。

---

## 下一步

- [`00_executive_summary.md`](00_executive_summary.md) — 高层执行摘要（≤ 3 页，给 BOSS 看）
- [`README.md`](README.md) — 目录导航
- [`02_axis_walkthrough.md`](02_axis_walkthrough.md) — 10 轴走读（按论文回看）
- [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) — 5 大主题综述（重大意义）
