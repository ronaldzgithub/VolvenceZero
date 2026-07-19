# NL / ETA 核心作者线增量调研

调研日期：2026-07-19

基线文档：

- `research/core-author-paper-assessment-2026-05.md`
- `research/arxiv-survey-2026-05.md`
- `research/probe/README.md`

本次目标不是重新综述 NL / ETA，而是找出旧研究里**没有分析过**、且与 NL / ETA 核心作者或直接扩散路线相关的新增论文，并判断它们是否改变 VolvenceZero 的技术路线。

## 0. 结论先行

本次真正需要补进研究库的是两篇 NL 作者线新作：

1. `2606.03979` **Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories**
2. `2606.23670` **Tapered Language Models**

另外有两篇应记录为“扩散 / 邻近证据”，但不应误记为 NL / ETA 核心作者线：

1. `2605.16350` **Federated Nested Learning**：外部团队把 NL/Titans 用到联邦学习，证明 NL 视角正在外溢。
2. `2603.09221` **Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning**：不是 ETA 主作者线，但把“test-time memory”推进到“test-time control / value function”，是 ETA 的重要邻域进展。

ETA 核心作者线本次没有发现比 `2512.20605` ETA 主论文、`2506.05233` MesaNet、`2602.16490` Growing-to-Looping 更新且未分析过的同作者主线论文。ETA 的新进展更像是**扩散与工程实现**，不是主作者又提出了新范式。

## 1. 已覆盖集合

旧文档已经覆盖了这些核心项：

- NL 主论文：`2512.24695` Nested Learning
- NL 作者线：Titans、Miras、ATLAS、TNT、Trellis、MS-SSM、Memory Caching、CoDE-Stop
- ETA 主论文：`2512.20605` Emergent Temporal Abstractions
- ETA 作者线：Uncovering mesa-optimization、MesaNet、Growing-to-Looping、Depth-Grown Models、modular compositionality、in-context compositional generalization
- 信用 / 控制 / 多主体：COCOA、least-control、Deep Feedback Control、learning-aware policy gradients、in-context co-player inference、Embedded UPI

因此本次只分析旧文档没有出现的条目。

## 2. 新增论文分析

### 2.1 Language Models Need Sleep (`2606.03979`)

作者线：Ali Behrouz / Farnoosh Hashemi / Adel Javanmard / Vahab Mirrokni。属于 NL 核心作者线，且直接接续 Nested Learning。

核心贡献：

- 把 continual learner 的生命周期从传统 train/test 切换，改成 **wake / sleep** 两相。
- Wake 阶段负责在线吸收输入，NL / Hope / CMS 属于在线 consolidation。
- Sleep 阶段负责 offline consolidation：把高频、脆弱、短期的 in-context memory 蒸馏到更低频、更稳定的参数或模块。
- Sleep 分两步：
  - **Knowledge Seeding / upward distillation**：把 smaller-self 或高频模块的知识蒸馏到更大 / 更稳定的低频模块。
  - **Dreaming**：用自生成数据和 RL / imitation learning 做无人工监督的自改进。
- 提出 periodic parameter activation / deactivation：逐步解锁新参数、停用旧高频参数，避免所有模块更新周期撞在一起造成 catastrophic forgetting。

对 VZ 的意义：

- 这是对 R1/R2/R5/R10/R15 的强补充。旧 NL 更偏在线多频率记忆，Sleep 明确补上 **background-slow / rare-heavy** 的离线整合协议。
- `vz-memory` 的 ReflectionEngine 不应只是“总结过去会话”；更准确地说，它应该是 sleep lifecycle 的一个 owner，负责把高频 episodic / session evidence 转成低频 stable artifact。
- `ModificationGate` 的重要性上升：Dreaming 是自生成训练 + 自改进，如果没有 gate、容量上限、回滚和外部评估，最容易变成自指强化。
- 这篇论文支持“运行时不直接改 frozen substrate，但允许离线 rare-heavy artifact refresh”的边界。它不是让 turn path 里在线训 LLM，而是给 sleep path 一个明确机制。

风险与边界：

- Dreaming 阶段用 RL 生成 curriculum，容易与 VZ 的“禁止 token 空间长期 RL”冲突。VZ 只能吸收其 **sleep lifecycle / consolidation protocol**，不能把 token-level RL 直接并入产品主链。
- Upward distillation 如果直接改 base LLM，就违反 R2；在 VZ 中应落在 `SubstrateRareHeavyCheckpoint`、CMS 低频 band、playbook artifact、semantic owner artifact 等可回滚层。
- Sleep 阶段必须异步，不应阻塞 turn path。

建议：

- 优先级：高。
- 落点：`vz-memory` / `vz-cognition/modification` / rare-heavy offline pipeline。
- 行动：把 “sleep phase = offline consolidation owner” 写进相关 spec；把 Dreaming 标为 gated experiment，先 report-only，不直接回写主链。

### 2.2 Tapered Language Models (`2606.23670`)

作者线：Reza Bayat / Ali Behrouz / Aaron Courville。属于 NL 作者参与的架构设计论文，但不是 NL 主范式论文。

核心贡献：

- 反对 decoder-only LM 中“每层等宽、等容量”的默认设定。
- 在固定总参数和 FLOPs 下，把 MLP capacity 前置：早层更宽，晚层更窄。
- 在 Transformer、Gated Attention、Hope-attention、Titans 四类架构上，cosine taper 的 MLP width 都改善 perplexity 和下游指标。
- 机制解释：越往后层，MLP 输出越接近已有 residual stream，更多是在 refine，而不是写入新特征。

对 VZ 的意义：

- 它给 `vz-substrate` 的 rare-heavy / offline backbone refresh 一个低风险启发：容量不必在深度上均匀分配。
- 它也间接支持 ETA 的 mid-depth controller 结论：越靠中前层，残差流更可塑、可控；晚层更像 refinement。
- 对现有运行时代码影响很小。它不是要改 snapshot、owner、memory store，而是未来训练 artifact 时的 architecture prior。

风险与边界：

- 不应把 TLM 当成当前季度工程任务。它要求重新训练或至少重新构造 backbone，属于 Tier 3 / rare-heavy 研究，而不是 `vz-runtime` 变更。
- 不应用它来手工调 runtime 模块权重。它是训练期容量分配原则。

建议：

- 优先级：中。
- 落点：`vz-substrate` 的未来 open-weight backbone / artifact refresh 研究；也可作为 ETA controller 插入层位的背景证据。
- 当前不进代码路线，只进入 research / future substrate notes。

### 2.3 Federated Nested Learning (`2605.16350`)

作者线：Hong Chen / Pengcheng Wu / Yuanguo Lin / Peilin Zhao / Xiuze Zhou / Fan Lin / Han Yu。不是 NL 核心作者线。

核心贡献：

- 把 Federated Learning 重写成三层 nested optimization：
  - L2：客户端 test-time memory state 在线更新。
  - L1：客户端学习 memory update rule / meta-parameters。
  - L0：服务器聚合规则，而不是聚合私有 memory content。
- 使用 Titans / Delta Rule 作为 test-time adaptation 的机制。
- 核心思想是“共享如何学习，不共享学到的私有记忆”。

对 VZ 的意义：

- 这篇不是主作者新作，但它是 NL 思想外溢的好证据。
- 它和 VZ 的 snapshot / owner 边界高度同构：跨边界不传内部状态，只传契约化规则或快照。
- 对多设备、多用户、端侧人格连续性很有启发：用户私有记忆不出端，只共享可验证的 update rule 或 artifact schema。

风险与边界：

- 它是联邦学习应用，不应直接进入伴侣主链。
- Server 聚合 meta-rule 的设计若落到 VZ，必须受 `ModificationGate` 和用户隐私边界控制。

建议：

- 优先级：低到中。
- 落点：未来多端 / privacy-preserving learning 研究。
- 当前只作为 “NL 扩散证据”，不作为核心作者线成果。

### 2.4 Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning (`2603.09221`)

作者线：Peihao Wang / Shan Yang / Xijun Wang / Tesi Xiao / Xin Liu / Changlong Yu / Yu Lou / Pan Li / Zhangyang Wang / Ming Lin / René Vidal。不是 ETA 核心作者线。

核心贡献：

- 指出 test-time memory 主要解决过去上下文的估计 / 检索，但 reasoning 需要面向未来的 planning。
- 提出 TTC layer：在 latent state 上做 finite-horizon LQR planning，把 value function 内化到神经网络层里。
- 从 memory objective 推进到 control objective：inference 不只是“估计过去”，而是“选择未来动作”。
- 作为 adapter 插入 LLM，在数学 / Sudoku 等 reasoning benchmark 上提升。

对 VZ 的意义：

- 这是 ETA 邻域很重要的“下一步问题”：内部控制器不仅要有 `z_t / beta_t`，还要有 value / cost / horizon 的可学习结构。
- 它支持 `vz-temporal` 不应退化为抽象动作标签路由，而应长期走 latent dynamics + value-aware planning。
- 它也提醒：单纯 memory compression 不等于 agency。NL 解决“如何记住 / 压缩上下文”，ETA/TTC 解决“如何在内部状态上行动和规划”。

风险与边界：

- TTC 是显式 LQR / optimal-control layer，不是 ETA 的 emergent temporal abstraction。VZ 不能把它误读为“应该硬编码一个控制器公式”。
- 如果引入，必须作为 `vz-temporal` owner 内部算法，不能让 runtime 或 expression layer 直接拼装 value function。

建议：

- 优先级：中，作为 ETA 邻接研究。
- 落点：`vz-temporal` future benchmark / internal planning notes。
- 当前不进代码；先用作理论对照：ETA 的 learned `z_t/beta_t` vs TTC 的 explicit latent control。

### 2.5 LETO: Modeling Multivariate Time Series with Memorizing at Test Time

作者线：Ali Behrouz 等。ICML 2025 Oral，旧文档未列。

核心贡献：

- 针对 multivariate time series 设计 2D memory：time memory + variate memory。
- 在时间维用 recurrent / delta-rule memory，在变量维保留 permutation equivariance。

判断：

- 这是 test-time memorization 的有用应用，但主要是时序预测架构，不直接改变 VZ 的 NL/ETA 主线。
- 可作为“多维 memory owner 如何处理不同轴的不变量”的远距参考。

建议：

- 优先级：低。
- 不进入近期路线。

## 3. 这批论文带来的真正进展

### 3.1 NL 从“在线多频记忆”推进到“生命周期协议”

旧 NL 的核心是：模型是多频率 nested optimization；optimizer / memory / architecture 是同一类东西。新 Sleep 论文把这个视角推进了一步：持续学习系统不只有在线 update，还必须有 sleep phase。

这对 VZ 很关键。我们原先的 R1/R2/R5 已经有 online-fast / session-medium / background-slow / rare-heavy，但 Sleep 提供了更清晰的因果链：

- online-fast：吸收新输入，形成脆弱记忆。
- online consolidation：通过 CMS / Hope 把高频信息往低频模块传。
- sleep consolidation：离线抽象、蒸馏、重组。
- dreaming：受 gate 控制的自生成练习和自改进。

因此，VZ 的慢环不应只是“总结器”，而应是一个明确的 sleep owner。

### 3.2 NL 的重点从“更多上下文”转向“写入哪里、何时写、如何合并”

Tapered LM 和 Sleep 放在一起看，方向很清楚：

- Memory Caching / Trellis / Titans：解决 memory capacity 与 retrieval。
- Sleep：解决短期记忆如何变成长期参数或低频 artifact。
- Tapered LM：解决容量在深度上的非均匀分配。

这说明 NL 作者线正在从“长上下文 / test-time memory”走向更完整的“容量、频率、深度、生命周期”四维设计。

### 3.3 ETA 主线暂时没有新范式，但邻域在补“value / planning”

ETA 主论文已经证明：抽象动作可以在 residual stream 中涌现，并且 internal RL 可以在 `z_t` 空间上做稀疏奖励学习。

本次没有发现 ETA 核心作者的新范式论文；但 TTC 这类邻域工作说明 community 正在补 ETA 的下一块：内部控制不仅要有 temporally abstract action，还要有 value function / planning horizon。

对 VZ 来说，路线不变：

- 不做 token-space RL。
- 不用 prompt label 冒充 regime / option。
- 在 `vz-temporal` 内部维护 latent controller、termination、value / credit readout。

但中长期应该给 `vz-temporal` 增加一个 benchmark 问题：abstract action 是否真的减少了 effective horizon 和 credit variance，而不只是生成看起来更连贯的行为。

## 4. 对 VolvenceZero 的路线裁决

### 立刻吸收为研究依据

- `2606.03979` Language Models Need Sleep
  - 加强 R1/R2/R5/R10/R15。
  - 把 background-slow / rare-heavy 定义成 sleep lifecycle，而不是普通 batch job。
  - Dreaming 必须受 `ModificationGate` 管控。

### 进入中期观察

- `2606.23670` Tapered Language Models
  - 作为 future substrate / backbone artifact 的容量分配原则。
  - 当前不动运行时。
- `2603.09221` TTC
  - 作为 ETA 邻接理论：memory 之外还需要 internal value / control。
  - 不把 LQR 公式硬塞进 runtime。

### 记录为外部扩散证据

- `2605.16350` FedNL
  - 说明 NL 视角已经被外部团队用于“共享规则，不共享私有 memory”。
  - 对未来端侧 / 联邦 / privacy-preserving companion 有价值。

### 明确降级

- LETO
  - test-time memorization 在 2D time-series 的应用。
  - 与 VZ 当前伴侣主链距离较远。

## 5. Spec / 代码建议

本次不建议直接改代码。

建议后续 spec 同步：

1. 在 memory / reflection 相关 spec 中补上 **sleep phase**：wake 在线吸收，sleep 离线 consolidation。
2. 在 `ModificationGate` 相关 spec 中补上 Dreaming 风险：自生成训练必须有 capacity cap、validation margin、rollback、只读评估。
3. 在 `vz-temporal` 研究 notes 中补一个 future benchmark：比较 token RL、ETA internal RL、explicit TTC-style control 在 sparse reward / long horizon / credit variance 上的差异。
4. 在 substrate future notes 中记录 Tapered LM：depth-aware capacity allocation 只属于 rare-heavy artifact training，不属于 runtime patch。

## 6. PDF 下载状态

已下载到 `research/papers/`：

- `research/papers/language-models-need-sleep-self-modify-consolidate-memories-2606.03979.pdf`
- `research/papers/tapered-language-models-2606.23670.pdf`
- `research/papers/federated-nested-learning-self-referential-memories-2605.16350.pdf`
- `research/papers/beyond-test-time-memory-state-space-optimal-control-llm-reasoning-2603.09221.pdf`
- `research/papers/leto-modeling-multivariate-time-series-memorizing-test-time.pdf`

## 7. Dreaming 阶段深读与算法提升（`2606.03979` §3.4）

这一节专门拆 Sleep 论文里的 **Dreaming（REM 对应）** 阶段，并给出算法层面的改进方向。它是 VZ 想借鉴 Sleep 时风险最高、也最需要改造的部分。

### 7.1 Dreaming 现在到底是什么

Dreaming 本质是 SEAL 式自改写循环 + 三个补丁。给定采样任务 `(C, τ)`（`C` 是相关上下文，`τ(·)` 是下游评估度量）：

1. **生成**：以 `C` 为上下文，从当前模型采样 `m` 个 dream。MoE 路由在采样时额外随机选一个专家，注入无关知识以制造新颖性。
2. **选择**：对每个 dream 用 SFT 目标的梯度做重要性打分 `g = ∇L_SFT(dream, θ)`，取 Top-k，再加 `b` 个随机样本保多样性。
3. **评估 + 更新**：每个 dream 用一个隔离的模型实例做 LoRA SFT，奖励是二值的（式 5：性能提升记 1，否则 0）。
4. **优化**：用 ReST^EM 优化上述过程。

论文自己的消融证明它脆：知识融合任务里去掉 Dreaming，准确率从 48.1 掉到 35.7；去掉梯度选择或随机专家也都下降。

### 7.2 六个算法弱点

1. **二值奖励（式 5）**：只区分“有没有变好”，丢掉幅度与方向，信号稀疏、方差高，且不做逐 dream 的信用分配。
2. **依赖下游 verifier `τ(·)`**：奖励绑定任务成功率（数学 / ARC / SQuAD）。一是 reward hacking（dream 钻 verifier 漏洞），二是 VZ 的关系 / 主体性域根本没有干净的 `τ`。
3. **梯度范数选梦混淆 epistemic 与 aleatoric**：高梯度既可能是可学习的惊奇，也可能是纯噪声 / 离群。当前打分会优先选到不可约噪声。
4. **随机专家 = 无方向探索**：靠随机性造新颖 dream 是 undirected exploration，样本效率低，不受学习信号引导。
5. **迭代自改的遗忘风险无硬门控**：靠“两阶段 + 隔离实例”缓解，但没有容量上限、验证边界、可回滚这类形式化保证。
6. **模仿奖励 `r_abs`（式 4）用 Levenshtein 编辑距离**：token 表面相似度，成本高，且与“语义优先、不做表面匹配”原则相悖。

### 7.3 改进方向（每条对应研究库已有论文）

| # | 弱点 | 改进 | 参考论文 |
|---|---|---|---|
| ① | 二值 / 稀疏奖励 | 换成 PE 驱动的低方差奖励 + 逐 dream 反事实信用分配 | `curiosity-critic-2604.18701`、`cocoa-2306.16803` |
| ② | 梯度范数选梦 | 改为“对留出集可约 PE 的降低量”+ 与 retention 集梯度的对齐度 | `curiosity-critic-2604.18701` |
| ③ | 随机专家探索 | dream 探索放到 `z_t` 潜空间做 internal RL，并让 curiosity 定向降低世界模型 PE | `emergent-temporal-abstractions-2512.20605`、`worldllm-2506.06725` |
| ④ | 自改无门控 | 每个 dream 更新变成 gate 候选 artifact，过门才 commit、可回滚 | `two-gate-guardrail-2510.04399`、`statistical-godel-machine-2510.10232` |
| ⑤ | 遗忘风险 | 加 replay-based quiet-rest 保留检查，dream 更新须先过旧知识 replay | `wake-sleep-consolidated-learning-2401.08623`、`seslr-2507.02901`、`semi-parametric-memory-consolidation-2504.14727` |
| ⑥ | Levenshtein 距离 | 换成 substrate feature 空间的嵌入距离 | 对齐 `no-keyword-matching-hacks` 规则 |

关键点：③ 让 dreaming 的探索发生在潜空间而非 token 空间，正好守住 VZ “禁止 token 空间长期 RL” 的铁律。

重要更正（2026-07-20）：①②所依赖的底层组件**在仓库里已经实现且在跑**，不是待办（详见 §7.6）。因此 dreaming 若要改奖励，是复用现有 infra，而不是从零造。

### 7.4 对 VZ 最关键的改造：重定义 `τ`

Sleep 的 Dreaming 奖励是 **IQ / 任务成功导向**；VZ 的产品是 **关系与主体性**。因此对我们最大的算法改动不是照搬 SEAL，而是重定义 dreaming 的 `τ`：

- 奖励应是“对用户 / 关系的预测误差下降”，而不是任务通过率。
- 该评估必须 **read-only**，经 `ModificationGate` 后才回写；绝不能让 evaluation 直接写回 credit / regime / memory（VZ 红线）。
- 因双轨（R7），dreaming 必须有分开的 world-track 与 self-track 奖励，不能塌缩成单一任务目标。

### 7.5 落点与优先级

- 优先级：中高（在 Sleep 生命周期之后的第二步）。
- 落点：`vz-cognition/prediction`（PE 奖励）、`vz-cognition/credit`（反事实信用）、`vz-cognition/modification`（gate）、`vz-temporal`（潜空间 dream 探索）、rare-heavy offline pipeline（隔离实例 + replay 复核）。
- 现在不进代码：先把上述改造写成 dreaming 的 spec 草案，跑 shadow / report-only，再决定是否进 rare-heavy 主链。

### 7.6 更正：Curiosity-Critic 与 COCOA 已落地（不是待办）

早前讨论里我把 “先做 Curiosity-Critic 和 COCOA” 当成待办，这是**错的**。代码核查（2026-07-20）确认两者都已在生产库 `vz-cognition` 实现，且不只是 `research/labs` 的 probe：

**Curiosity-Critic（PE epistemic / aleatoric 分解）** — `packages/vz-cognition/src/volvence_zero/prediction/error.py`：

- `PEDecomposition`（`aleatoric_magnitude` / `epistemic_magnitude`）+ owner-internal running-stats critic。
- 已推进到 Phase 1.B → Phase 2.B：epistemic = improvement-PE 分量（`epistemic = value - learned_prediction`）。
- 通过 `PredictionErrorSnapshot.pe_decomposition` 发布；在 `evaluation/backbone.py` 作为 `pe_aleatoric_magnitude` / `pe_epistemic_magnitude` 读出。
- 状态：**report-only，不被任何 acceptance gate 消费**。

**COCOA（反事实贡献信用）** — `packages/vz-cognition/src/volvence_zero/credit/gate.py`：

- `derive_counterfactual_contribution_records`（Phase 1.A 历史基线）+ `CreditLedger.derive_learned_counterfactual_contribution_records`（Phase 2.A owner-internal learned baseline / rewarding-state head，带 gate decision）。
- 有 N-step ledger、`CounterfactualContributionReadout`、least-control 读出。
- 在 `evaluation/mid_layer.py` 聚合消费。

据此修正结论：

1. 当前**没有**“必须先做的高收益前置项”在等待——Curiosity-Critic / COCOA 已完成（Phase 1 全做完，Phase 2 部分做完），处于 shadow / report-only。
2. 真正剩下的独立小决策是：要不要让这两块从 report-only 升级到被 gate 消费。这与 Sleep 无关。
3. Sleep / Dreaming 更**不必要也不划算**：VZ 现有 PE-first + 反事实信用主链已经把 Sleep 想借鉴的最有价值部分落地了，Sleep 只是在上层补了“生命周期”叙事。

## 8. 一句话更新

NL 的最新进展是从“记忆模块”升级为“持续学习生命周期”：wake 中在线压缩，sleep 中离线蒸馏和自改进。ETA 主作者线暂时没有新范式论文，但邻域正在把 latent temporal abstraction 推向 value-aware planning。VZ 现在最应该吸收的是 Sleep 的生命周期叙事，而不是急着改 runtime——因为 Sleep 想借鉴的最有价值部分（PE 一级信号、反事实信用）已经由 `vz-cognition` 的 Curiosity-Critic 与 COCOA 落地（report-only）；真正剩下的只是要不要把它们从 shadow 升级到被 gate 消费，这与 Sleep 无关。
