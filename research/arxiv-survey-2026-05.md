# Arxiv 论文调研：与 VolvenceZero 主方向相关的重要工作

**调研日期**：2026-05-04
**主方向**：NL（嵌套学习）+ ETA（涌现时间抽象）、冻结基底+自适应控制器、token 空间 vs 潜在控制器空间、多时间尺度记忆、预测误差作为一级信号、双轨+体制身份、关系与主体性优先于智力

下面把搜到的论文按架构里的 **R1–R15 + R-PE 切分轴** 和 **产品方向** 整理。每条都标了 arxiv id 与"为什么对你重要"。重点已用粗体标出。

---

## 一、最直接对齐设计哲学的核心论文（必读）

这两篇基本就是 `docs/next_gen_emogpt.md` 引用的源头，建议**反复研读**作为底盘。

| 论文 | arxiv | 对应 R-ID | 为什么重要 |
|---|---|---|---|
| **Nested Learning: The Illusion of Deep Learning Architectures** (Behrouz et al., NeurIPS 2025) | `2512.24695` | **R1, R13, R5** | NL 框架的源头。把架构与优化器都写成"嵌套多层级、不同更新率"的同一对象；优化器 = 关联记忆模块，把梯度信息压缩进去；提出 **Hope** 模块和 **Continuum Memory System** —— vz-memory 的 4-stratum 直接是它的一种实例化。 |
| **Emergent temporal abstractions in autoregressive models enable hierarchical RL** | `2512.20605` | **R3, R4, R10** | ETA 框架的源头。"在 base AR 模型的内部表示之上跑一个**非因果**高阶序列模型"发现切换单元 β_t 与控制器代码 z_t；明确提出 **internal RL**（在 z_t 空间而非 token 空间做 RL）—— 完全对应"禁止 token 空间 RL"那条铁律。 |

---

## 二、按切分轴分组的高价值论文

### 1. R2 / R4 —— 冻结基底 + 控制器层（最贴近 vz-substrate / vz-temporal）

| 论文 | arxiv | 看点 |
|---|---|---|
| **CoLA: Controlling LLMs with Latent Actions** | `2503.21383` | 在冻结 LLM 上嵌一个潜动作空间，**RL 在 latent action 上跑**而不是 token 上；与"内部 RL on z_t"思路一致；性价比也漂亮（math500: 38.2→42.4，速度×2）。 |
| **Learning to Ponder: Adaptive Reasoning in Latent Space (FR-Ponder)** | `2509.24238` | <1M 参数轻量控制器观察冻结 LLM 隐状态、自适应分配推理算力；**示范了"寄居在冻结基底之上"的最小可行控制器** —— metacontroller MVP 的良好对照。 |
| **RL for Latent-Space Thinking in LLMs** | `2512.11816` | 直接把 GRPO 等 RL 用到 latent thinking，不是 token；和"控制器空间强化"对齐。 |
| **KV Cache Steering for Frozen LLMs** | `2507.08799` | one-shot 的 KV 干预触发 CoT —— 一种比 activation steering 更稳定的"无梯度控制器"通道，可以作为 substrate residual 之外的另一个有界注入口。 |

### 2. R-PE —— 预测误差作为一级信号

| 论文 | arxiv | 看点 |
|---|---|---|
| **Curiosity-Critic: Cumulative Prediction Error Improvement** | `2604.18701` | **关键**：把瞬时 PE 替换为"PE 的可改进部分"，**在线分离 epistemic vs aleatoric**。这正是 PE readout 想要的：噪声不应触发 credit/needs 反应。强烈建议照搬这个 critic 思路。 |
| **WorldLLM: Curiosity-driven theory-making** | `2506.06725` | 用 PE/不确定性驱动 LLM 主动找证据；可作为反思层 background-slow 的 worldview 修正信号源。 |
| **RLVR-World: Training World Models with RL** | `2505.13934` | 用可验证奖励训世界模型，绕开 MLE 偏差 —— 给 substrate 慢更新（rare-heavy）提供可复用方案。 |
| **3M-Progress (zebrafish)** | `2506.00138` | 神经科学验证：仅靠 model-prior divergence（=PE）就能涌现自然探索行为和全脑动力学 —— 给"PE 是足够丰富的内禀信号"提供 in-vivo 证据。 |

### 3. R3 / R10 —— 时间抽象 / 选项框架（除了 ETA 主论文外）

| 论文 | arxiv | 看点 |
|---|---|---|
| **MANGO: Multi-layer Abstraction for Nested Generation of Options** | `2508.17751` | 多层 option 嵌套，与 NL 的多时间尺度天然契合；可作为 metacontroller 的多尺度扩展参考。 |
| **Option-aware Temporally Abstracted Value (OTA)** | `2505.12737` | 离线 goal-conditioned RL 中把 horizon"压短" —— 做长程关系/承诺评估时直接用得上。 |
| **Variational Homomorphisms in Option-Induced Abstract MDPs** | `2507.16473` | 给"在抽象 latent 空间上学策略保持最优性"提供形式化保证 —— z_t 空间 RL 的理论后盾。 |
| **Change Point Detection + Option-Critic** | `2510.24988` | 用 **PE spike + reward shift** 自动检测 option 边界（不用外部监督）—— β_t 切换信号最自然的实现路径。**强烈推荐**。 |
| **LDSC: LLM-guided Semantic HRL** | `2503.19007` | LLM 做 subgoal 生成 + RL 做 option execution —— 一个混合方案，但注意它仍走"语言 subgoal"的路，不是纯潜空间。 |

### 4. R5 / R6 —— 记忆连续谱、CMS、反思

| 论文 | arxiv | 看点 |
|---|---|---|
| **HippoRAG 2 (From RAG to Memory)** | `2502.14802` | 个性化 PageRank + 深度 passage 整合；给 vz-memory **派生索引层**（4th stratum）一个可复制的方案。 |
| **A-Mem: Agentic Memory** | `2502.12110` | Zettelkasten 式动态卡片系统，新记忆自动生成属性、关联、演化已有记忆 —— **几乎就是 vz-memory 的 episodic→semantic 演化路径**，但他们做得更系统化。 |
| **Memory-R1** | `2508.19828` | 用 PPO/GRPO 训"记忆管理 agent"（add/update/delete）—— 和 ReflectionEngine 的 writeback 是同一类问题；他们已经把它 RL 化。 |
| **Synapse: Spreading Activation 记忆图** | `2601.02744` | 把记忆建成动态图 + 时间衰减 + 横向抑制，三种检索混合 —— 给 retrieval.py 升级的明确方向。 |
| **Latent Learning: Episodic Memory Complements Parametric** | `2509.16189` | 形式化论证"参数学习不够，必须要 episodic 灵活复用" —— 给 R5/R6 的存在合法性。 |
| **Memento 2: Stateful Reflective Memory** | `2512.22716` | 把 episodic memory 当作有读写操作的控制对象，**不更新主模型参数即可持续适应** —— 完美对应"冻结基底 + 记忆侧适应"。 |

### 5. R7 / R14 —— 双轨 + 持久体制身份（产品独特性，论文较少但都要看）

| 论文 | arxiv | 看点 |
|---|---|---|
| **DMWM: Dual-Mind World Model** | `2502.07591` | 双过程世界模型：RSSM-S1 + 逻辑-S2 互反馈 —— 双轨的一种实现形态（虽然他们不是 World/Self 而是 fast/slow，可参考双轨切分方法）。 |
| **Sophia: System-3 Persistent Agent** | `2512.18202` | **强相关**：明确做"narrative identity + 长程适应"的 meta-layer，4 个机制（thought curation, narrative memory, dynamic user/self models, hybrid reward）—— 和"双轨 + 持久 regime + semantic_state"思路高度并行。 |
| **ID-RAG: Identity Retrieval-Augmented Generation** | `2509.25299` | 把人格固定成结构化 KG（核心信念/特质/价值）—— 给 regime 的"持久身份基底"提供具体落地方式。 |
| **The Geometry of Persona / Soul Engine** | `2512.07092` | 把人格视为 LLM latent 中的**正交线性子空间** —— 这条思路非常重要：人格不是 prompt 标签而是几何对象，与"regime 不是 prompt 标签"一致。 |
| **EvoAgent: Continual World Model** | `2502.05907` | 长程任务的自演化 agent + 课程式 reflector —— 给 background-slow 反思层的 long-horizon 验证。 |

### 6. R8 / R9 / R10 / R15 —— 契约式自修改、信用门控、可回滚

| 论文 | arxiv | 看点 |
|---|---|---|
| **Two-Gate Guardrail for Self-Modifying Agents** | `2510.04399` | **极重要**。形式化证明：自修改 agent 要保留 PAC 学习保证，需要 policy-reachable 模型族 VC 维有界。提出"validation margin + capacity cap"双门 —— **几乎是 ModificationGate 的理论底盘**，强烈建议引用并加固 gate 设计。 |
| **Statistical Gödel Machine (SGM)** | `2510.10232` | 用 e-values + Hoeffding 边界做"统计安全层"代替形式证明，全局误差预算 —— 给 rare-heavy artifact training 一个实际可行的"保守批准"机制。 |
| **Darwin Gödel Machine** | `2505.22954` | open-ended 进化 + archive，可作为参考但要注意他们是**全开放自修改**，与"分层 + 有界 + 快照可回滚"是不同哲学。 |

### 7. R11 —— 内部状态可命名可发布（9 个 semantic owner 的设计依据）

| 论文 | arxiv | 看点 |
|---|---|---|
| **Lookback Mechanism for Belief Tracking** | `2505.14685` | 实证 LLM 内部用 OI（Ordering ID）+ low-rank subspace 绑定 character-object-state —— 给 user_model / belief_assumption 的"几何可定位"提供证据。 |
| **ThoughtTracing (Hypothesis-Driven ToM)** | `2502.11881` | 类 SMC 的多假设 belief tracking，无需 ground truth —— 适合 user_model 的在线维护。 |
| **ToM-aligned Conversational Agents (BDI)** | `2502.14171` | 显式建模 Belief/Desire/Intention 来对齐 —— 直接对应 plan_intent / commitment / open_loop / user_model 的命名拆分思路。 |
| **CoDA: Context-Decoupled Hierarchical Agent** | `2512.12716` | 单 LLM 同时演两角色（Planner/Executor）+ context 隔离 + PECO 联训 —— 给"同一基底，多 owner，快照隔离"提供工程方案。 |

### 8. R12 —— 评估覆盖"存在"而非任务

| 论文 | arxiv | 看点 |
|---|---|---|
| **Survey on Evaluation of LLM-based Agents** | `2503.16416` | 列出当前 agent eval 的 4 维度（capability / domain / generalist / framework）+ 缺口（cost / safety / robustness）—— 给 6 族 evaluation 一个对照地图。 |
| **Gaia2 / ARE platform** | `2509.17158` | **异步运行**评估，揭示静态评估看不到的失败模式（如时间约束、多 agent 协作、噪声）—— evaluation/backbone 的 readout 应该照这个方向走。 |
| **AgencyBench** | `2601.11044` | 6 类 agentic 能力 × 32 真实场景 + **资源效率/反馈自纠** —— 比 task-success 更接近"主体性"。 |
| **A2Perf** | `2503.03056` | OOD 泛化 + 资源效率 + 数据成本 —— 强调"能力之外的代价"，匹配"评估覆盖存在不是堆任务"。 |

### 9. 关系优先 / EQ —— 产品独特性（其他人在做但角度不同，要看看避免重复发明）

| 论文 | arxiv | 看点 |
|---|---|---|
| **CogniPair (GNWT-based Multi-Agent)** | `2506.03543` | **特别有意思**：实现 Global Workspace Theory 的多 sub-agent（emotion / memory / social norms / planning / goal）+ 全局广播 —— 与"多 owner + 快照总线"惊人地像（虽然他们更激进）。 |
| **SocialSim** | `2506.16756` | 情感支持对话生成 —— 数据集与基线，可作为预训练语料源。 |
| **COMPEER** | `2508.09521` | 把心理学步骤当 RL 过程奖励 —— 一种**以心理学结构化输出为内部信号**的范式，做 EQ 评估时可以照抄打分维度。 |
| **RLFF-ESC: Future-Oriented Reward** | `2508.12935` | 多 agent 模拟未来轨迹 → 收 future-oriented reward —— 跟"长程关系不是当下 reward"完全合拍。 |
| **Personalized Long-term Interactions (Westhäußer)** | `2510.07925` | 持久记忆 + 动态协调 + 演化用户档案 —— 与 user_model + relationship_state 同构。 |
| **PersonaMem-v2** | `2512.06688` | 隐式人格学习 + agentic memory，128k 多轮，16× 上下文压缩；评估方法值得借鉴。 |

### 10. 其他重要"基础设施类"参考

| 论文 | arxiv | 看点 |
|---|---|---|
| **Wake-Sleep Consolidated Learning** | `2401.08623` | 仿生 wake-sleep + replay 防灾难性遗忘 —— background-slow 反思阶段的算法雏形。 |
| **Semi-parametric Memory Consolidation (Brain-like)** | `2504.14727` | 半参数记忆 + wake-sleep —— 给 CMS stratum 间迁移的具体算法。 |
| **Sleep Enhanced Latent Replay (SESLR)** | `2507.02901` | 噪声增强的 sleep replay；class-incremental +30% 准确率，1/3 内存。 |
| **Active Inference: ODAR** | `2602.23681` | 用 amortized active inference + free-energy 路由 fast/slow —— 给 metacontroller 的 β_t 决策提供变分原理形式化路径。 |
| **Active Inference Multi-LLM Self-Organizing** | `2412.10425` | FEP 当 cognitive layer 调度 LLM agents —— 又一次验证 PE/free-energy 是统一信号。 |
| **Dualformer** | `2410.09918` | 同一模型显式 fast/slow + auto 三模式 —— 给"β_t 在切换"一个工程化对照。 |
| **ACE: Agentic Context Engineering** | `2510.04618` | 把 context 当**演化的 playbook**，generation/reflection/curation 三阶段防 collapse —— 与 vz-application 的 playbook 直接同名。 |

---

## 三、阅读优先级建议（如果时间有限）

**第一档（必读，占用一周）**：
1. `2512.24695` Nested Learning
2. `2512.20605` ETA / Internal RL
3. `2510.04399` Two-Gate Guardrail（VC bounded self-modification）
4. `2604.18701` Curiosity-Critic（PE epistemic vs aleatoric）
5. `2512.18202` Sophia（System-3 narrative identity）

**第二档（强相关，再花一周）**：

6. `2503.21383` CoLA（latent action RL on frozen LLM）
7. `2510.24988` CPD + Option-Critic（β_t 自动检测）
8. `2502.12110` A-Mem（agentic memory 演化）
9. `2512.22716` Memento 2（stateful reflective memory）
10. `2506.03543` CogniPair（GNWT 多 sub-agent，思考差异）
11. `2509.25299` ID-RAG + `2512.07092` Soul Engine（人格的几何 vs 几何识别）

**第三档（按需）**：其余论文，按当前在做哪个 R-ID 模块时再展开。

---

## 四、可以立刻反哺到代码的几条建议

1. **Curiosity-Critic** 的 epistemic/aleatoric 分离思路，应直接进 `vz-cognition/prediction` —— 现在的 PE readout 大概率把噪声当信号。
2. **Two-Gate Guardrail 的 VC 容量上限**应作为 `ModificationGate` 的硬约束写进 spec —— 现在 gate 还偏经验式。
3. **CPD on PE-spikes** 是 β_t 检测最自然的初版实现，比硬编码切换条件更对齐"切换是涌现的"。
4. **A-Mem 的卡片演化协议**和 `vz-memory/store.py` 现有结构很接近，建议对齐它的 attribute schema。
5. **Sophia 的 narrative memory + dynamic user/self model** 几乎就是 dual_track + semantic_state 的另一个表达，可以借它的评估题目验证双轨。

---

## 论文 PDF

所有论文已下载到 `research/papers/` 目录，命名格式为 `<title>-<arxiv_id>.pdf`。
