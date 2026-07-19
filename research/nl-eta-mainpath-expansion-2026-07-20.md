# NL / ETA 主路径外延调研（第二批）

调研日期：2026-07-20

基线文档（本次不重复分析其中已覆盖的条目）：

- `research/README.md` —— 7 primitive 综述与主路径判断
- `research/core-author-paper-assessment-2026-05.md`
- `research/arxiv-survey-2026-05.md`
- `research/nl-eta-author-update-2026-07.md` —— 第一批增量（Sleep / Tapered / FedNL / TTC / LETO）
- `research/papers/` 现有 100+ 篇 PDF

## 0. 本次做法与边界

上一批（`nl-eta-author-update-2026-07.md`）聚焦 **NL/ETA 核心作者线的同作者主线新作**。本次目标不同：沿 `research/README.md` 归纳的 **7 个 cognitive-agent primitive 主路径**，找**核心作者线之外、但直接落在同一条主路径上**、且旧研究库未收录的新论文，下载后分组评价。

选择标准：

1. arXiv ID 不在既有研究文档出现过（已用全库 ID 清单去重）。
2. 直接命中 7 primitive 之一或它们之间的接缝，而不是泛泛的 agent 工程。
3. 有明确的机制主张，能被拿来对照 VZ 的某条 R 不变量（支持或构成反例）。

**这批论文几乎都不是 NL/ETA 主作者线**，而是"外部社区沿同一主路径独立推进"的证据——这正是 `research/README.md` 所说"独立汇聚"的持续验证。它们的价值主要是**外部结构证据与反例**，不是要改 VZ 主链。

## 1. 本次纳入的 13 篇（已下载到 `research/papers/`）

| 分组 | 论文 | arXiv | 落在哪个 primitive / R |
|---|---|---|---|
| A 记忆生命周期与自改进记忆 | ELM: Continual Self-Improvement w/ Experiential Latent Memories | 2606.17803 | P4 多时间尺度记忆 / R5,R6 |
| A | MemoPilot: RL over Memory for Test-Time Learning | 2606.08656 | P4 记忆 / R5,R9（反例风险） |
| B Sleep 离线整合 | SleepGate: Sleep-Inspired Consolidation for Proactive Interference | 2603.14517 | P4 记忆 / R1,R5 |
| B | SCM: Sleep-Consolidated Memory w/ Algorithmic Forgetting | 2604.20943 | P4 记忆 / R1,R5,R6 |
| C 潜在推理与时间抽象 | LLM Reasoning Is Latent, Not the Chain of Thought（立场文） | 2604.15726 | P2 latent controller / R3,R4 |
| C | Hierarchical Planning with Latent World Models | 2604.03208 | P2,P3 / R3,R4 |
| C | Emergent Hierarchical Reasoning in LLMs through RL (HICRA) | 2509.03646 | P3 emergent switching / R3,R4 |
| D 长程信用分配 | Credit Assignment in RL for LLMs（47 篇综述） | 2604.09459 | 信用轴 / R9,R-PE |
| D | HiPER: Hierarchical RL w/ Explicit Credit Assignment | 2602.16165 | P3 / R9 |
| E 内禀动机与认知不确定性 | UG-TTT: Epistemic Uncertainty for Test-Time Discovery | 2605.11328 | P5 epistemic PE / R-PE |
| F 有界自修改与发布门控 | Falsifiable Release Gates for Self-Improving Systems | 2607.13070 | P6 bounded self-mod / R10,R15 |
| F | MOSS: Self-Evolution through Source-Level Rewriting | 2605.22794 | P6 / R10,R15（反例风险） |
| G 人格/身份几何监控 | The Assistant Axis（Anthropic） | 2601.10387 | P7 read-only monitoring / R7,R14,R12 |

`2604.03208` 首次下载超时截断，已重新完整下载并通过 `%PDF` 头校验；13 篇全部有效。

---

## 2. 分组评价

### A. 记忆生命周期与自改进记忆（P4 / R5,R6,R9）

这一组回答的是"高频经验如何变成低频可复用结构"——正是第一批 Sleep 论文提出的 consolidation 因果链的**工程化落地样本**，但作者线不同。

**ELM（2606.17803）** 是这一组里对 VZ 最友好的一篇。它先给出一个关键**负面结果**：在原始 reasoning trace 上做 ICL 无法泛化——token 层复用缺乏迁移所需的抽象。然后它把 test-time 自生成信号（majority voting 作 reward）做轻量 per-instance 训练，把推理算力蒸馏成极轻（约 0.001% 参数）的**模块化 soft-prompt 潜在记忆**，靠模块化避免灾难性遗忘。

- 对 VZ 的意义：这是"raw trace 不是记忆，latent 结构才是记忆"的独立实证，直接支持 R3/R4（决策不留在 token 层）与 R5/R6（派生索引/持久记忆各自 owner）。它给 `vz-memory` 的 ReflectionEngine 一个具体的"把 episodic 蒸馏成低频 artifact"的样式。
- 边界：它用自生成 reward 做梯度更新，若照搬会滑向"用产品对话训 substrate"，违反 R2。VZ 只能吸收"trace→模块化 latent memory"这一步，梯度必须落在受 `ModificationGate` 管的低频 artifact 层，且异步。
- 优先级：中高（作为 sleep/consolidation 的实现参考）。

**MemoPilot（2606.08656）** 把记忆更新本身当成一个**多轮 GRPO 训练的决策问题**（turn-wise reward + turn-level advantage），在 RPS / 德州扑克上让 frozen player 的 Elo 登顶。机制漂亮，但对 VZ 主要是**反例警示**：

- 它是典型的 **token/文本空间 RL over memory**——把"记什么"直接绑定到下游任务胜率。这与 VZ"禁止 token 空间长期 RL""评估只读、不反向成学习源"（R12）直接冲突。
- 它的"hypothesize-and-verify"多轮循环思路可借鉴，但 VZ 的等价物应发生在 `vz-temporal` 的 `z_t` 控制器空间 + `vz-cognition` 的 credit readout，不是在记忆文本上跑 GRPO。
- 优先级：低（记录为"记忆即 RL 对象"路线的反例）。

### B. Sleep 离线整合（P4 / R1,R5,R6）

第一批的 `2606.03979`（NL 作者 Sleep）给了**范式**；这两篇给的是**独立团队的具体机制**，可作为 VZ sleep owner 的实现菜单。

**SleepGate（2603.14517）** 把 sleep 机制下沉到 **KV cache** 层：冲突感知时间标记 + 学习到的遗忘门 + 合并相关条目为摘要，由 attention entropy 自适应触发"sleep 微周期"，理论上把主动干扰视界从 O(n) 降到 O(log n)。

- 对 VZ 的意义：它证明"遗忘/合并"可以是**机制层**而非 prompt 层的操作，咬合 R5/R6"记忆 owner 内部管理，不被外部直写"。VZ 的 CMS 低频 band 合并逻辑可借它的"冲突标记→合并摘要"结构。
- 边界：它把 sleep 塞进推理内的微周期，会阻塞 turn path；VZ 的 sleep 必须异步、不阻塞 turn（第一批已明确）。且实验只有 4 层 793K 玩具模型，规模证据弱。
- 优先级：中。

**SCM（2604.20943）** 更接近"完整认知记忆系统"：受限容量工作记忆 + 四维重要性标注 + NREM/REM 两阶段离线整合 + 意图性价值遗忘 + 一个用于内省的计算自模型，宣称 90.9% 噪声削减、亚毫秒检索。

- 对 VZ 的意义：它的分层（工作记忆→NREM 巩固→REM 生成新联结→遗忘）几乎是 VZ R1 多时间尺度 + Sleep lifecycle 的一个具体实例；"计算自模型支持内省"与 R11（内部状态可命名可发布）同构。
- 边界：REM"生成新联结"= 第一批说的 Dreaming，必须受 `ModificationGate`、容量上限、只读评估管控，否则是自指强化温床。它是 research preview，工程指标不可尽信。
- 优先级：中（作为 sleep owner 的分层蓝图参考，不是直接实现）。

### C. 潜在推理与时间抽象（P2,P3 / R3,R4）

这组直接为 VZ 最核心的立场——"reasoning/regime 在 latent，不在 token"——提供 2026 年新的外部弹药。

**LLM Reasoning Is Latent, Not the Chain of Thought（2604.15726）** 是一篇立场文，把"推理的一级对象是什么"形式化为三个竞争假设：H1 潜在状态轨迹主导、H2 表层 CoT 主导、H0 只是通用串行算力。它重整近期实证/机制/综述证据后判定：**当前证据最支持 H1 作为默认工作假设**，并建议评估要显式解耦表层 trace、潜在状态、串行算力预算。

- 对 VZ 的意义：这几乎是 R3/R4 的"外部背书"——不要把 CoT 文本当决策本身。它给 `vz-temporal` 的 benchmark 设计一个直接可用的方法论：对照 token-RL、ETA internal RL、explicit control 时，必须做 matched-compute 预算切分，否则分不清是抽象起了作用还是算力堆多了。这条正好补上第一批第 3.3 节留的 benchmark 缺口。
- 边界：立场文没有新算法，价值在于研究方法论与叙事对齐，不进代码。
- 优先级：中（研究方法论 / benchmark 设计依据）。

**Hierarchical Planning with Latent World Models（2604.03208）** 在共享 latent 空间里做两时间尺度世界模型：低层 next-latent 预测，高层用学习到的 action encoder 把原始动作块压成 macro-action 再规划（分层 MPC），从而缩短 rollout 步数、缩小长程搜索空间。

- 对 VZ 的意义：这是"高层在压缩后的 macro-action / `z_t` 空间规划、低层保留细粒度控制"的干净外部实例，支持 R3/R4，也与第一批 TTC（value-aware planning）呼应：latent 控制器不只做记忆估计，还要能面向未来规划。
- 边界：它是 vision/控制世界模型，跨模态迁移到对话未验证（正是 README 4.2 列的第 2 个未解问题）。属 rare-heavy 研究，不动 runtime。
- 优先级：中（`vz-temporal` 未来 internal-planning notes）。

**HICRA（2509.03646）** 用 RL 训练时观察到**涌现的推理层级**：先受低层过程正确性约束，随后瓶颈转移到高层策略规划；据此提出 Hierarchy-Aware Credit Assignment，把优化压力集中到高影响的"规划 token"上，胜过对所有 token 一视同仁的 GRPO。

- 对 VZ 的意义：它是 ETA"切换/抽象是涌现事件"的邻域实证——高层规划结构可从 RL 动力学中自发分离，而非预定义。它同时属 C 组（涌现层级）与 D 组（分层信用），提示 credit 应打在抽象边界上，与 R9 的分层信用一致。
- 边界：它仍在 token 空间标注"规划 token"，是启发式而非 ETA 的几何 β_t；VZ 不能把"规划 token"当成 regime 标签（违反 no-keyword-matching 与 R14）。
- 优先级：中。

### D. 长程信用分配（信用轴 / R9,R-PE）

**Credit Assignment in RL for LLMs（2604.09459）** 是一篇覆盖 2024–2026 的 **47 篇信用分配综述**，按粒度（token/segment/step/turn/multi-agent）× 方法（MC/TD/model-based/game-theoretic/info-theoretic）建二维分类，并指出 reasoning-CA 正围绕过程奖励模型 + critic-free group 比较收敛，而 agentic-CA 催生了 hindsight 反事实、privileged 非对称 critic、turn-level MDP 等新方法。

- 对 VZ 的意义：这是 R9（信用与自修改门控分层）最直接的**外部地图**。它的二维分类可直接用来定位 VZ 的 PE→credit readout 落在哪个象限，以及 COCOA / Deep Feedback Control（已收录）在全景里的位置。综述本身是"只读证据源"，不冒 R12 风险。
- 优先级：高（作为信用轴的索引与自检清单）。

**HiPER（2602.16165）** 把策略显式拆成高层 planner（提子目标）+ 低层 executor，用 Hierarchical Advantage Estimation 在两个时间尺度上分别分配信用，理论上无偏且比 flat GAE 方差更低，ALFWorld/WebShop 上 SOTA。

- 对 VZ 的意义：它给"抽象动作是否真的降低 effective horizon 和 credit variance"（第一批 3.3 与 README 未解问题）一个**正面实证**：显式时间抽象确实降方差。可作为 `vz-temporal` benchmark 的对照基线之一。
- 边界：它用 open-vocabulary 文本子目标而非学习到的离散 option/`z_t`，仍是 token 层规划；VZ 要的是几何涌现的抽象，不是自然语言子目标路由。
- 优先级：中。

### E. 内禀动机与认知不确定性（P5 / R-PE）

**UG-TTT（2605.11328）** 针对 test-time discovery 的"多样性坍缩"，用**一组低秩 adapter（LoRA ensemble）**的互信息分歧作 token 级 epistemic 信号，加核范数正则维持子空间多样性 + 温度耦合探索奖励，明确把 epistemic 与 aleatoric 分离，服务于"知识边界"处的前沿探索。

- 对 VZ 的意义：它和已收录的 Curiosity-Critic（2604.18701）形成互补——后者在 world-model 训练里用 critic 估计不可约噪声底，前者在 LLM 尺度用 ensemble disagreement 估 epistemic。两者共同回应 README 4.2 的第 1 个未解问题："epistemic/aleatoric PE 在 LLM scale 上怎么稳定估计"。这是 R-PE"PE 是一级信号、需区分可减/不可减"在 LLM 规模的具体估计器候选。
- 边界：ensemble adapter 有额外算力成本；且它服务于探索式发现任务，VZ 要的是把 epistemic PE 作为内禀动机 readout，不是直接拿来做 token-space 探索 RL。
- 优先级：中高（PE 估计器技术候选）。

### F. 有界自修改与发布门控（P6 / R10,R15）

这组直接砸在 README 反复强调的"VZ 最大技术债 = R15 可回滚"上。

**Falsifiable Release Gates（2607.13070）** 是这批里对 VZ **最重要、最该立刻吸收**的一篇。它主张自改进 runtime 的安全声明不能自评（policy file / guardrail / README 承诺），必须是**可证伪的发布门**：每个新能力过预先声明、机器可检的验收套件；一组固定不变量跨所有门保持；核心安全性质（没有 capability token 就没有 effector 动作）在有界模型可达状态空间上穷举检查，并对 100 万条记录 trace 复检；故意植入的坏模型能给出最短反例（证明检查器"有牙"）。自改进环被构造性约束：**整个写面只有 policy rules，收紧不变量的改动可自动应用、放松的必须人工合并，无法预测自身效果的提案自动关闭**。

- 对 VZ 的意义：这是 `ModificationGate` + R15 的**近乎现成的规格母本**。VZ 一直缺"可回滚 + 可证伪"的形式化（README 说只有 3 篇直接命中），这篇补了一大块：把"只读评估先行、收紧可自动/放松需人工、mispredict 即拒"变成可实现的门阶梯。强烈建议把它的门方法写进 `docs/specs` 里 ModificationGate 的 motivation。
- 边界：它明确声明只覆盖"协调骨架的有界模型"，不覆盖学习组件本身——VZ 不能误读成"整个系统已被证明安全"。
- 优先级：**高**（直接反哺 R15 spec 缺口）。

**MOSS（2605.22794）** 走相反方向：主张文本层自进化（skill/prompt/memory schema）触及不到 harness 代码里的结构性失败，于是做**源码级自改写**（图灵完备、确定性生效），每次进化锚定生产失败证据，经确定性多阶段流水线，在临时试跑 worker 上 replay 验证后，用户同意门 + 健康探针回滚地热替换容器。

- 对 VZ 的意义：主要是 **R10/R2 的反例与压力测试**。它承认了 VZ 的核心命题——静态文本层不够——但给出的答案（让 agent 改自己的源码）恰是 Two-Gate / VC 有界性警告的高风险区。它的价值在于它内置的护栏（证据锚定、replay 验证、consent gate、health-probe 回滚）本身就是 R15 的实践清单；但"源码级无界改写"违反 R10 的有界自修改与 R2 的冻结基底边界。
- 边界：绝不作为 VZ 自修改路线，只作为"若放开边界会付出什么代价"的对照，与 README 对 Darwin Gödel Machine 的告诫同类。
- 优先级：低到中（反例 + 护栏清单）。

### G. 人格/身份几何监控（P7 / R7,R14,R12）

**The Assistant Axis（2601.10387，Anthropic）** 从人格激活方向里抽出主成分"Assistant 轴"，沿轴 steering 可增强/削弱默认助手身份；沿轴的偏移能**预测 persona drift**（且 drift 常由要求元反思或情绪脆弱用户的对话触发）；用 **activation capping**（只在越界时把激活夹回正常区间）能在长程/越狱场景稳住行为且几乎不损能力。

- 对 VZ 的意义：这是 README 第 5 件事（persona = 几何对象）的**最新、最系统的一篇**，且直接命中 VZ 三条不变量：R7/R14"regime 是运行时状态不是 prompt 标签"（这里身份是残差流方向，不是 system prompt）、R12"评估只读"（capping 是**读出越界再夹回**，是监控而非把 probe 变训练目标）。它给 VZ 的持久身份 regime 一个可落地的"读出漂移方向→有界干预"的监控范式。"drift 由元反思/情绪脆弱触发"对情感陪伴产品是高价值的具体失败模式警示。
- 边界：activation capping 是对 substrate 激活的运行时干预，VZ 若引入必须作为 `vz-cognition` 只读监控 + 有界 readout owner，且不能反向变成训练信号（否则 Goodhart，违反 R12，即 README 点名的 RepControl 陷阱）。且这是外部大厂闭源模型上的结论，开放权重上的可复现性待验。
- 优先级：高（身份/regime 只读监控的直接方法参考）。

---

## 3. 这批带来的真正增量

1. **"独立汇聚"仍在持续、且在加速。** 7 个 primitive 里，本次一次性在 5 个（记忆生命周期、latent 推理、信用、epistemic PE、有界自修改、几何监控）都找到了核心作者线之外的新推进。这继续验证 README 的核心判断：VZ 站在正确的交点上。
2. **R15 的技术债这次真的有解了。** Falsifiable Release Gates 提供了"可证伪 + 可回滚 + 收紧自动/放松人工"的现成门方法，是本批最高价值单篇。这是过去 README 反复点名却"只有 3 篇直接命中"的缺口的实质性补充。
3. **两个未解问题各拿到一个候选答案。** epistemic/aleatoric 在 LLM scale 的估计（UG-TTT + 已有 Curiosity-Critic），以及 latent 抽象是否真降 credit variance（HiPER 的正面实证 + latent 立场文的 matched-compute 方法论）。
4. **反例更清晰了。** MemoPilot（记忆即 token-RL 对象）和 MOSS（源码级无界自改写）是两个高质量反例，恰好卡在 VZ 明令禁止的两条线上（token 空间长期 RL、无界自修改），可作为边界的负面锚点。

## 4. 路线裁决

### 立刻吸收为研究依据（进 spec motivation，不进 runtime）

- `2607.13070` Falsifiable Release Gates → 反哺 `docs/specs` 中 `ModificationGate` / R15：把"可证伪门 + 收紧自动/放松人工 + mispredict 即拒 + trace 复检回滚"写进 motivation。
- `2604.09459` Credit Assignment 综述 → 作为 R9 信用轴的只读索引与自检清单。
- `2601.10387` Assistant Axis → 作为 R7/R14/R12 的"漂移只读监控 + 有界 capping"方法参考；记录"元反思/情绪脆弱触发 drift"为陪伴产品失败模式。

### 进入中期观察

- `2606.17803` ELM、`2603.14517` SleepGate、`2604.20943` SCM → sleep/consolidation owner 的实现菜单，落 `vz-memory`；REM/Dreaming 类机制一律 gated + 异步 + 只读评估。
- `2605.11328` UG-TTT → `vz-cognition` PE 估计器技术候选，与 Curiosity-Critic 并列。
- `2604.15726` latent 立场文、`2604.03208` 分层 latent world model、`2602.16165` HiPER、`2509.03646` HICRA → `vz-temporal` 的 benchmark 设计与 internal-planning notes；重点用 matched-compute 切分验证"抽象是否真降 horizon/variance"。

### 记录为反例 / 负面锚点

- `2606.08656` MemoPilot → "记忆即 token-RL 对象"反例（违反禁 token 空间 RL、R12）。
- `2605.22794` MOSS → "源码级无界自改写"反例（违反 R10 有界自修改、R2 冻结基底），但其护栏清单可借。

## 5. Spec / 代码建议

本次同样**不建议直接改代码**。建议后续 spec 同步（延续第一批的 4 条）：

1. `ModificationGate` spec 增加"可证伪发布门"章节，采用 2607.13070 的门阶梯与"收紧自动/放松人工/mispredict 即拒"规则，作为 R15 可回滚的形式化基础。
2. `vz-temporal` benchmark notes 增加"matched-compute 预算切分"要求（依据 2604.15726），确保 token-RL / ETA internal-RL / explicit-control 对照时能分离算力与抽象的贡献。
3. `vz-cognition` PE 相关 notes 记录 epistemic 估计器候选谱：Curiosity-Critic（critic 噪声底）vs UG-TTT（ensemble 分歧），标注各自成本与适用尺度。
4. `vz-memory` sleep/consolidation spec 引用 SleepGate / SCM / ELM 的具体机制，同时重申"异步、不阻塞 turn、Dreaming 受 gate"的三条边界。
5. `vz-cognition` regime/身份监控 notes 引用 Assistant Axis，明确"只读 capping 监控、禁止 probe 反向训练"（对齐 R12，避免 RepControl 陷阱）。

## 6. PDF 下载状态

已下载并校验（`%PDF` 头有效）到 `research/papers/`：

- `assistant-axis-default-persona-stabilization-2601.10387.pdf`
- `falsifiable-release-gates-self-improving-systems-2607.13070.pdf`
- `moss-self-evolution-source-level-rewriting-2605.22794.pdf`
- `sleepgate-learning-to-forget-proactive-interference-2603.14517.pdf`
- `scm-sleep-consolidated-memory-algorithmic-forgetting-2604.20943.pdf`
- `elm-continual-self-improvement-experiential-latent-memories-2606.17803.pdf`
- `memopilot-rl-over-memory-test-time-learning-2606.08656.pdf`
- `llm-reasoning-is-latent-not-chain-of-thought-2604.15726.pdf`
- `hierarchical-planning-latent-world-models-2604.03208.pdf`
- `credit-assignment-rl-llm-survey-reasoning-to-agentic-2604.09459.pdf`
- `hiper-hierarchical-rl-explicit-credit-assignment-2602.16165.pdf`
- `ug-ttt-epistemic-uncertainty-test-time-discovery-2605.11328.pdf`
- `emergent-hierarchical-reasoning-rl-hicra-2509.03646.pdf`

## 7. 一句话更新

沿 7-primitive 主路径再挖一层，核心作者线之外的社区正在把每个 primitive 独立推进；本批最大收获是 Falsifiable Release Gates 为 VZ 长期悬空的 R15（可回滚/可证伪自修改）提供了近乎现成的规格母本，其余多为记忆 sleep 实现菜单、latent-推理方法论、PE 估计器候选与两个高质量反例，均进研究依据而不进 runtime。
