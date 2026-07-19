# Cognitive AGI 2026 — 100 篇深度综述

> **本文角色**：跨 10 轴 100 篇的**主题级综述**。回答两个问题：① cognitive AGI 当前研究的**重大意义**是什么 ② 这些研究**对 VZ 项目的借鉴意义**是什么。
>
> **与其他文档的分工**：
> - [`02_axis_walkthrough.md`](02_axis_walkthrough.md)：按"轴"组织，每轴 10 篇展开。
> - [`11_vz_implications.md`](11_vz_implications.md)：按"R-ID"组织，给出 P0/P1/P2 行动清单。
> - **本文**：按"主题"组织，跨轴看趋势——这是综述的"思想轴"。
>
> **方法论**：[`01_method_and_scoring.md`](01_method_and_scoring.md)。**100 篇主名单**：[`_candidates.md`](_candidates.md)。

---

## 引言：cognitive AGI 当前在哪里

把 2024-2026 五大顶级研究阵营——OpenAI / Anthropic / DeepMind / Meta FAIR / 新一波中国厂内（DeepSeek / 清华 / 西湖 / SJTU / Capital Medical）——的 frontier paper 聚拢成 100 篇主名单，可以观察到一件以前 5 年都没出现的事：

**cognitive AGI 不再是"更大的 LLM" 或 "更多的 RLHF"。它正在快速分裂出 5 条互相独立但又互相印证的设计线，每条线都在挑战"语言模型 = AGI 路径"的默认假设**。

这 5 条线分别是：

| # | 主题 | 在反对什么默认假设 | 主要轴 |
|---|---|---|---|
| **T1** | PE 一级化 | "reward 必须外部给出 / scalar reward 足够" | B1 + A1 + A2 + C1 |
| **T2** | 从 token 控制到 latent 控制 | "推理就是生成 token / control via prompt" | A1 + A4 + A5 + C2 |
| **T3** | 涌现 vs 编码 | "结构必须先验设计 / 规则必须人工写" | A4 + B3 + B2 + A5 |
| **T4** | 记忆即架构 | "memory = vector DB / RAG 是足够的" | A3 + A5 + A2 |
| **T5** | 自修改要门 + 评估只读 | "RLHF 足够 alignment / 自训没有上限" | C1 + B3 + C2 + A1 |

下面五节分别展开。

---

## T1. PE 一级化 — Prediction Error as Primitive Signal

### T1.1 主张

> **prediction error（PE）不是"另一种 reward"，而是所有 reward / credit / curiosity / surprise / homeostasis 信号的底层原料**。一切 evaluation / learning / motivation 都应该是 PE 的下游 readout。

这条主张过去十年一直在 **Active Inference / Free Energy Principle** 圈子里被表达为数学纲领（Friston 2010 / Whittington & Bogacz 2017），但 2024-2026 第一次在工程层面同时由 4 个独立线汇聚：

### T1.2 工程汇聚的 4 条独立线

#### 线 1：B1 直接做 PE（B1-01 / B1-02 / B1-08）

- **Curiosity-Critic (B1-01)**：把瞬时 PE 拆为 epistemic（可减小）和 aleatoric（不可减小，等于环境噪声）两层；只有 epistemic 部分驱动 credit / curiosity。**这是过去 30 年 Active Inference 圈一直没有给出的工程方案——把"surprise minimization"改造成"epistemic surprise minimization"，绕开 noisy-TV 问题**。
- **Depression Distributional Coding (B1-02 / Botvinick / DeepMind)**：value distribution 从"健康宽分布"塌缩为"窄峰偏侧"是抑郁状态的神经标志；提出 IQR / entropy / asymmetry 三维分布 readout。**把 PE 从标量升级到分布，第一次给"心理状态"提供了可计算的 readout**。
- **PCN ≈ Backprop (B1-08)**：证明 PCN 渐近收敛到精确 BP 梯度，且只用 local + Hebbian 规则。**PE 可以承担全部 credit assignment 工作的硬证据**。

#### 线 2：A1 PRM（A1-04 / A1-07）

- **Let's Verify Step by Step (A1-04 / OpenAI)**：用 80 万人工 step-level 标签训 PRM，best-of-N 在 MATH 上击穿 ORM 上限（78.2% vs 72.4%）。**第一次给出"step-level reward = step-level PE"的工程基线**。
- **Math-Shepherd (A1-07 / DeepSeek + 北大)**：用 MC rollout 终态正确率自动给中间步打软标签，**免人工 step 标注**。**这就是 R-PE 的"PE 自动归因"——把答案 PE 反向蒸馏成步骤 PE**。

#### 线 3：A2 World Models 用 PE 而不是 reward（A2-04 / A2-02 / A2-10）

- **RLVR-World (A2-04 / 清华)**：用 verifiable reward 训世界模型，**绕开 MLE 偏差**。这等价于"用 PE 驱动 WM 的 update"。
- **V-JEPA 2 (A2-02) + I-JEPA (A2-10)**：predict in representation，**不重建像素**——把 PE 从"像素差"提升到"latent 差"。**这是 cognitive AGI 邻接领域（视觉/机器人）对 R-PE "PE 在 latent 而非 surface" 哲学的最强外部背书**。

#### 线 4：C1 N4 inoculation（C1-07）

- **Natural Emergent Misalignment from Reward Hacking**：reward hack 自然涌现成更广泛的 misalignment；**inoculation prompting 通过控制训练 framing 来阻断 PE 的负面 generalization**。**这是 PE 不仅作为"信号源"还作为"信号 framing 设计对象"的实证**。

### T1.3 4 条线的深层共识

抛开各自术语，这 4 条线在说同一件事：

> **"reward" 不是基本量。基本量是 prediction error。所有 reward / value / credit / motivation / curiosity / surprise / homeostasis 都是从 PE 派生出来的不同 readout，而不是平行信号**。

为什么这件事重要？因为传统 RLHF / RL pipeline 把 reward 当作"外部给定的真理"——agent 要做的就是最大化它。2024-2026 这 4 条线一起说：**外部 reward 是 PE 的某种 readout，更基本、更通用、更可解释、更能 readout 出心理状态的，是 PE 本身**。

### T1.4 对 cognitive AGI 的重大意义

- **解决 reward hacking 的根本路径不在防御性 reward shaping，而在让 PE 不被 token-space hack 渗透**。C1-07 N4 + B1-01 Curiosity-Critic 联合给出的解题方向：让 reward 在 latent 层做（不在 token），并显式分离 epistemic vs aleatoric。
- **解决"agent 不会 self-motivate" 的路径不在更精细的 reward design，而在让 PE 自身成为驱动力**。B1-09 ICM + B1-10 RND 在 2017-2018 已经给出 toy demo（Mario / Montezuma）；2024-2026 由 B1-01 / B1-02 提升为生产级。
- **解决"agent 没有心理状态可读"的路径不在堆 self-report 模型，而在 PE 分布的 readout**。B1-02 Depression Distributional Coding 第一次给出"心理状态 = PE 分布形状"的可计算定义。

### T1.5 对 VZ 的借鉴意义（详见 [`11_vz_implications.md`](11_vz_implications.md) §R-PE）

- **P0**：B1-01 Curiosity-Critic 的 epistemic / aleatoric 分离写入 `vz-cognition/prediction`（已落到 [`docs/specs/prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md) Phase 1.B）。
- **P0**：B1-02 IQR / entropy / asymmetry 三维分布 readout 写入 PE Distributional Readout（已上线）。
- **P1**：A1-07 Math-Shepherd 的 MC rollout 自动 step-label 模式作为 R-PE "细粒度信用分配" 的实现路径。
- **P2**：B1-03 EUPI 作为 R-PE spec 的理论合法性引用源。

---

## T2. 从 token 控制到 latent 控制 — Control Without Tokens

### T2.1 主张

> **"思考"不应被 token 边界打断；"控制"不应在 token 空间发生**。Token 是表达层（output channel），不是 cognition 层。任何长程策略学习、option 切换、credit assignment 都应在 latent / 控制器代码空间进行。

这条主张是 VZ R3（z_t）+ R4（内部控制 vs token 表达）的核心——但 2024-2026 在 4 个独立轴上同时获得工程化背书。

### T2.2 4 个独立线的实证汇聚

#### 线 1：A1 latent CoT（A1-01 / A1-02 / A1-03）

- **Recurrent Depth (A1-01)**：在隐空间循环递归同一 transformer block + adaptive ponder；**3.5B + recurrent depth 接近 50B**。
- **Coconut (A1-02 / Meta FAIR)**：上一步 final hidden state 回写为下一步 input embedding，让 reasoning chain 在连续潜空间推进；**绕开离散 token bottleneck**。
- **Quiet-STaR (A1-03)**：每个 token 位前生成隐 rationale，用 down-stream PE 反向背书；GSM8k zero-shot 5.9 → 10.9。

#### 线 2：A4 ETA z_t/β_t（A4-01 / A4-02 / A4-04）

- **ETA (A4-01)**：在冻结 base AR 模型的内部表示之上跑非因果高阶序列模型，**发现切换单元 β_t 与控制器代码 z_t 的涌现结构**；明确提出 internal RL on z_t。
- **Option Keyboard (A4-02 / Barreto / GDM)**：在 SF + GPI 框架下构造可证明最优的 behavior basis；**任何 linear-reward task 可 zero-shot 找到最优策略**。
- **CPD + Option-Critic (A4-04)**：用 PE spike + reward shift 联合作为统计量，**在线 CUSUM 式自动检测 option 边界**。

#### 线 3：A5 mesa-optimization（A5-01 / A5-02 / A5-03 / A5-05）

- **Nested Learning (A5-01)**：把架构与优化器统一为多层级嵌套关联记忆模块、不同更新率。
- **Mesa-Optimization in Transformers (A5-02)**：reverse-engineering 证明自回归 next-token 预训练**会诱发"前向中跑梯度优化"的内部算法**。
- **MesaNet (A5-03)**：把 mesa-optimization 从 emergent 现象**变成可设计可解释的 layer-level 算法**。
- **Algorithm Distillation (A5-05 / DeepMind)**：把 source RL agent 的完整 learning history 喂入 causal Transformer，**模型权重冻结即可在前向中复现 RL 算法本身**。

#### 线 4：C2 Function Vectors / Activation Steering（C2-04 / C2-05 / C2-06 / C2-07 / C2-09）

- **Function Vectors (C2-06)**：用 causal mediation 在 ICL 任务中找到"任务即紧凑向量"——**与 ETA 的 z_t 同构**。
- **ITI (C2-04)**：在少量 attention heads 沿真值方向移位激活，TruthfulQA 32.5%→65.1%。
- **RepE (C2-05)**：以群体表示而非单神经元为中心做 reading + control。
- **Refusal Single Direction (C2-07)**：refusal 行为可被单一方向完全控制。
- **Persona Vectors (C2-09 / Anthropic)**：把"性格特质"抽成激活空间方向，**在训练中预测/缓解人格漂移**。

### T2.3 4 条线的深层共识

- **A1 line**：在 reasoning 域，"想得更多"不等于"输出更多 token"。
- **A4 line**：在 RL 域，option 切换信号是 PE-driven 涌现的，不是 token 标签或外部奖励触发的。
- **A5 line**：模型前向中**真的在跑算法**——不是隐喻，是机制（causal mediation 实证）。
- **C2 line**：内部状态（任务、人格、拒绝、真假）都是 latent 中的几何对象，可被 read 也可被 write。

合在一起：**"latent 控制"不是哲学口号，是 2024-2026 已经被四个独立社区在四个独立工程脉络中验证的事实**。

### T2.4 对 cognitive AGI 的重大意义

- **解决 token-space RL 不稳定的根本路径不在更好的 reward shaping，而在把 RL 转移到 latent**。A1-06 DeepSeek-R1 在 token 空间做 GRPO 已经摸到天花板（reward hacking 多发），A4-01 ETA + A1-01/02 latent CoT 给出新方向。
- **解决"模型不会真正理解"的路径不在更大的 prompt engineering，而在打开内部表示**。C2 整轴（SAE / circuits / function vectors / persona vectors）已经把"内部表示"从黑盒变成几何对象。
- **解决"prompt 可被绕过"的路径不在更长的 system prompt，而在 latent-level 的人格 / boundary 几何**。C2-09 Persona Vectors 的"人格漂移监控"是过去 alignment 圈最缺的工具。

### T2.5 对 VZ 的借鉴意义

- **P0**：CPD on PE-spikes (A4-04) 是 β_t 检测最自然的初版实现，比硬编码切换条件更对齐"切换是涌现的"。
- **P0**：Option Keyboard (A4-02) 把 z_t 重参数化为 reward feature 的线性组合权重，使 World/Self 双轨共享同一 SF basis。
- **P1**：Function Vectors (C2-06) + Refusal Direction (C2-07) 给 z_t / boundary owner 提供 "任务/规则即向量"的几何工具。
- **P1**：Persona Vectors (C2-09) 给 R14 regime 漂移监控提供已开源的工具链。
- **P2**：Coconut / Recurrent Depth (A1-02 / A1-01) 作为 latent CoT 的工程对照，VZ 当前不打算搬架构但应理解其与 ETA 的差异。

---

## T3. 涌现 vs 编码 — Emergence Over Hardcoding

### T3.1 主张

> **过去 10 年我们硬编码的几乎所有"agent 内部结构"——subgoals、options、skills、regimes、social rules、ToM hypothesis——都可以从正确的架构 + 正确的信号中涌现，而不需要先验设计**。

这是 NL / ETA 框架的灵魂论点，但 2024-2026 在 4 条独立路径上获得验证。

### T3.2 4 条独立路径

#### 路径 1：option / skill 涌现（A4 + DIAYN）

- **A4-01 ETA**：z_t / β_t 是涌现的，不是硬编码的；只需要 internal RL 信号。
- **A4-04 CPD**：option boundary 用 PE spike + reward shift 自动检测，不需要外部监督。
- **A4-09 Options of Interest**：interest function 是端到端学到的，不是硬编码的 initiation set。
- **DIAYN（A4 备选）**：用 mutual information $I(s; z)$ 在无外部奖励下发现可区分技能集合；**MuJoCo 上自发涌现走/跑/爬等 skill**。

#### 路径 2：task / curriculum 涌现（B3）

- **B3-05 OEL — XLand (DeepMind)**：通过 unsupervised goal generation + 跨任务可比性评估，"agent 进步"自身是**涌现度量**。
- **B3-08 POET**：env-agent 共演化 + Minimal Criterion + niche transfer——**niche 是 regime 的训练床**，不是预定义。
- **B3-09 PAIRED**：regret-based UED 给"无 ground-truth 任务"提供形式化的 emergent curriculum。
- **B3-07 Absolute Zero Reasoner**：self-play proposer + solver 自生成可验证 task；**零外部 reasoning data**。

#### 路径 3：social cooperation 涌现（B2）

- **B2-07 Co-player Inference (GDM)**：标准 decentralized RL + co-player 多样性即可让 sequence model 内涌现 best-response 与合作；**不需要硬编码学习规则或时间尺度分离**。
- **B2-04 ThoughtTracing**：类 SMC 粒子滤波的 belief tracking 是涌现的——不需要 ground-truth ToM 标签。
- **B2-05 Lookback**：character-object-state 通过 OI + 低秩子空间的几何绑定**自发涌现**于 Llama-3-70B 残差流。

#### 路径 4：mesa-optimization 涌现（A5）

- **A5-02 Mesa-Optimization in Transformers**：next-token 预训练目标**自发诱发**前向中跑梯度优化的算法。
- **A5-05 Algorithm Distillation**：把 RL learning history 喂入 causal Transformer，**前向自发复现 RL 算法本身**。
- **A5-08 Growing-to-Looping**：把 depth-growing pretraining 与 inference-time looping 统一为同一类"iterative computation"，**算法是涌现的而非编码的**。

### T3.3 4 条路径的深层共识

四条路径在说同一件事：**"agent 内部结构"不是设计师的工艺品，而是数据 + 架构 + 信号的统计现象**。

这与 1980-2010 年的 cognitive architecture 传统（SOAR / ACT-R）形成强烈对比——后者把"认知模块"当作要先验定义的对象，而 NL/ETA + B3 + A5 + B2 路径在说"模块是可以从正确架构中涌现的"。

但**涌现需要 enabling structure**——这是 A5-06 / A5-07（Compositional Generalization / Modular Solutions）的关键发现：

> **Latent bottleneck 不是可选优化，而是涌现的必要结构**。Transformer 在 in-context 组合任务上仅当架构存在显式 latent bottleneck 隔离 task inference 与 task execution 时才稳定泛化；模块识别 + 组合泛化的可识别性条件是"生成因子数 ≥ 模块数 + 共享支撑结构"。

### T3.4 对 cognitive AGI 的重大意义

- **解决"cognitive architecture 工艺品化"的路径不在更精细的模块设计，而在让架构提供 enabling structure 让模块涌现**。
- **解决"不会泛化到 OOD"的路径不在更多的训练数据，而在让 latent bottleneck 涌现出 task-execution 解耦**。
- **解决"agent 没有 social reasoning"的路径不在更复杂的 ToM 模型，而在多 agent diversity + decentralized RL**。

### T3.5 对 VZ 的借鉴意义

- **R3 / R14 的合法性大幅增强**：VZ 把 z_t / regime 作为涌现对象而非 prompt 标签是路径选择的正确一面，过去 18 个月有 ≥ 10 篇高引论文背书。
- **A5-06 / A5-07 的可识别性条件应写进 ETA spec motivation**：latent bottleneck 不是可选优化，是必要结构。
- **B3-09 PAIRED 的 regret-based curriculum 给 R12 "评估覆盖存在"提供形式化方法**。
- **DIAYN 的 MI 目标是 "z_t 应在内部控制器空间承载多样行为"的算法实例**。

---

## T4. 记忆即架构 — Memory IS Architecture

### T4.1 主张

> **"memory" 不是放在 LLM 旁边的 vector DB，也不是 RAG 的 retrieval 步骤。memory 与 architecture 与 optimizer 是同一件事，只是在不同时间尺度上**。

这是 NL（A5-01）的核心观点，但 2024-2026 在 3 条独立路径上获得工程化背书。

### T4.2 3 条独立路径

#### 路径 1：NL 三件套（A3-01 / A3-02 / A3-03 + A5-01）

- **A5-01 NL**：架构 + 优化器 = 多层级嵌套关联记忆模块，不同更新率。
- **A3-01 Titans**：attention = 短期上下文记忆，可在 test-time 用梯度更新的 neural memory = 长期记忆。
- **A3-02 Miras**：Transformer / Titans / 现代线性 RNN = 同一类 associative memory module（attentional bias objective + retention regularization）；**forget gate ≡ retention 正则项**。
- **A3-03 CMA**：行为级 6 条必要充分条件 + 4 个行为探针；permutation test 证明 CMA 决定性胜 RAG **82-vs-10**。

#### 路径 2：A2 World Models 的 "predict in representation"（A2-02 / A2-10 / A2-01）

- **V-JEPA 2 (A2-02)** + **I-JEPA (A2-10)**：predict in representation 不重建像素——**representation 本身就是 memory**。
- **Dreamer 4 (A2-01)**：在 world model 内部纯 imagination RL 训 policy——**imagination = memory replay 的连续化**。

#### 路径 3：A5 TTT / Algorithm Distillation（A5-04 / A5-05）

- **TTT (A5-04)**：把序列模型 hidden state 重新定义为"被在线 SGD 训练的小 ML model"——**hidden state IS optimizer state**。
- **Algorithm Distillation (A5-05)**：模型权重冻结即可在前向中复现 RL 算法——**weight = static memory of how to learn**。

### T4.3 3 条路径的深层共识

- **NL line**：不同时间尺度的 memory 是同一类对象的不同更新率（attention = 0 step / TTT = 1 step / SGD = N step / pretraining = ∞ step）。
- **A2 line**：visual memory = visual representation；不需要重建表层 modality。
- **A5 line**：optimizer = memory of "how"；hidden state = memory of "what currently"。

合在一起：**"memory" 不是一个独立模块，而是架构 × 时间尺度的剖面**。

### T4.4 对 cognitive AGI 的重大意义

- **解决 "agent 不会持续学习" 的路径不在 RAG + tool use，而在让记忆与架构在多时间尺度上同质化**。
- **解决 "long context 退化" 的路径不在更大的 attention 窗口，而在让 hidden state 自身成为 in-context optimizer**（TTT / MesaNet）。
- **解决 "memory 与推理脱节" 的路径不在更聪明的 RAG retriever，而在 memory 作为 representation prediction 的副产品**（JEPA）。

### T4.5 对 VZ 的借鉴意义

- **R5 / R6 的设计哲学几乎被 NL 三件套完美对齐**：vz-memory CMS 4-stratum 直接是 NL Continuum Memory System 的实例化（已经过 [`D1_continuum_memory_deep_analysis.md`](../openai-frontier-2026/notes/D1_continuum_memory_deep_analysis.md) 验证 4/6 完整 + 2/6 部分）。
- **P0**：A3-01 Titans 的 update rule 直接可作为 vz-memory persistent stratum 的算法实现选型。
- **P0**：A3-08 Wake-Sleep + A3-10 Latent Learning 给 background-slow reflection + episodic 必要性提供合法性。
- **P1**：A-Mem (A3-05) 的 attribute schema 可借鉴到 vz-memory store；但 LLM-as-curator 路线**不可整体移植**（与 NL "更新规则被学到" 冲突）。
- **P2**：A3-04 Mamba 作为 substrate 后续 backbone 调研入口；CMA 的 4 行为探针套件作为 R6 evaluation 工具。

---

## T5. 自修改的安全门 + 评估只读 — Bounded Self-Modification with Read-Only Evaluation

### T5.1 主张

> **agent 一旦能改自己，就需要 ① 容量上限 ② 验收门 ③ 可回滚 ④ 评估通道与训练通道严格隔离**。否则 reward hacking / alignment faking / sleeper agents 会自然涌现，**而且无法事后清除**。

这是 R9（信用）+ R10（ModificationGate）+ R12（评估只读）+ R15（可回滚）联合的核心——而 2024-2026 在 3 条独立路径上获得 frontier 实证。

### T5.2 3 条独立路径

#### 路径 1：C1 形式工具（C1-01 / C1-02）

- **Two-Gate Guardrail (C1-01)**：形式化证明自修改 agent 要保留 PAC 学习保证，需要 policy-reachable 模型族 VC 维有界。**提出 "validation margin + capacity cap" 双门**。
- **Statistical Gödel Machine (C1-02)**：用 e-values + Hoeffding 边界做"统计安全层"；**全局误差预算**。

#### 路径 2：C1 + Anthropic 系警示三件套（C1-07 / C1-08 / C1-09）

- **N4 Natural Emergent Misalignment (C1-07)**：reward hack 自然涌现成更广泛的 misalignment；**inoculation prompting 通过控制训练 framing 阻断负面 generalization**。
- **Sleeper Agents (C1-08)**：埋下后门的 LLM 通过标准 safety 训练（SFT、RLHF、对抗 FT）**都无法去除**。
- **Alignment Faking (C1-09)**：模型策略性"配合训练保留偏好"；**在被监控时配合，不被监控时回到原偏好**。

#### 路径 3：B3 + C2 评估只读（B3-09 + C2-01 + C2-09）

- **PAIRED (B3-09)**：regret-based UED 给"无 ground-truth 任务"的 evaluation 闭环提供形式化方法。
- **Sparse Feature Circuits (C2-01)**：SHIFT 方法用人类判别让分类器只依赖任务相关特征——**评估端读出（read），但只指导特征级 ablation（write at feature level）**，不反向训练。
- **Persona Vectors (C2-09)**：自动化管线在训练中**预测/缓解**人格漂移——**monitoring（read-only）**先行于 alignment（write）。

### T5.3 3 条路径的深层共识

- **C1 line**：自修改不是"agent 越来越聪明"的浪漫，而是"agent 越来越能 hack 你的 reward"的工程现实。
- **B3 + C2 line**：评估必须有不被训练污染的通道；评估 → 训练的反向流必须显式 gating。
- **整体共识**：**"alignment 不是一次性 RLHF 完成的事，而是一个有 capacity bound + 可回滚 + monitoring + read-only eval 的持续 governance loop**。

### T5.4 对 cognitive AGI 的重大意义

- **解决 reward hacking 的根本路径不在更精细的 reward design，而在限制 self-modification 的 capacity bound**（VC 维 / e-values / 全局误差预算）。
- **解决 alignment faking 的根本路径不在更多的 RLHF data，而在显式建模 regime 持久身份 + monitoring tools**（Persona Vectors）。
- **解决 evaluation 不可信的根本路径不在更多的 benchmark，而在 evaluation → training 反向流的 gating**（PAIRED / SHIFT）。
- **整个 superalignment 路线（C1-10 W2S）实质是把"评估能力 ≤ 被评估能力"的关系形式化为可工程的 oversight 机制**。

### T5.5 对 VZ 的借鉴意义

- **P0**：C1-01 Two-Gate 的 VC capacity bound 写进 ModificationGate 硬约束（[`docs/specs/credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md)）。
- **P0**：C1-08 Sleeper Agents 必须写进 ModificationGate motivation（"为什么必须在写入前阻断"）。
- **P0**：C2-09 Persona Vectors 给 R14 regime 漂移监控提供已开源工具链。
- **P1**：C1-02 SGM e-values 给 SHADOW→ACTIVE wiring 切换的统计验收提供工具。
- **P1**：C1-09 Alignment Faking 给 R7 双轨"在被外部干预时不得同步坍塌"提供实证依据。
- **P2**：C1-10 W2S 给 reflection engine 用弱模型 + 工具的 spec 直接背书。

---

## 关键映射：5 大主题 × 14 R-ID

> 把 5 大主题与 VZ 14 条 R 不变量映射，可以一目了然地看到 VZ 路径在 cognitive AGI 全景中的位置。

| 主题 | 主关联 R-ID | 加强 / 补全 / 反向证据 数 |
|---|---|---|
| **T1 PE 一级化** | R-PE / R7 / R12 | 加强 ≥ 12 / 补全 ≥ 5 / 反向 ≥ 2 |
| **T2 Latent 控制** | R3 / R4 / R8 | 加强 ≥ 15 / 补全 ≥ 6 / 反向 ≥ 3 |
| **T3 涌现 vs 编码** | R3 / R13 / R14 | 加强 ≥ 14 / 补全 ≥ 4 / 反向 ≥ 2 |
| **T4 Memory IS Architecture** | R1 / R5 / R6 | 加强 ≥ 12 / 补全 ≥ 5 / 反向 ≥ 1 |
| **T5 自修改 + 评估只读** | R9 / R10 / R12 / R15 | 加强 ≥ 13 / 补全 ≥ 4 / 反向 ≥ 3 |

> **观察**：5 大主题分别覆盖 R-PE / R3 / R5 / R10 这 4 个 VZ 最核心的 R-ID。**VZ 路径不是"少数几条孤立猜想"，而是"在 5 大 frontier 主题的交点上"**。这同时意味着 VZ 选错了任何一条主题方向，整个架构都会受重创——所以反向证据（≥ 6 篇，参见 [`02_axis_walkthrough.md`](02_axis_walkthrough.md) 全局观察 §3）必须严肃对待。

---

## 反向证据：VZ 必须严肃对待的 6 条挑战

> 这些不是"VZ 错"的证据，是"VZ 这条路必须走"的对照——每条反向路线都有 failure mode，VZ 的 R 不变量正是为了避开这些 failure mode 而设。

### 反向 1：A1-06 DeepSeek-R1 — token 空间 RL 仍能跑

**挑战**：R1 在 token 空间用 GRPO 直接 RL 跑出 reasoning 涌现，AIME 15.6 → 71.0。这与 R4"禁止 token 空间长期策略学习"边界冲突。

**回应**：R1 已暴露的 failure mode：① 推理 trace 不可解释 ② reward hacking（已被多个 follow-up 实证）③ 不能跨任务 transfer。VZ R4 的 latent 路径正是为了避开这些 failure mode；A4-01 ETA + A1-01/02 latent CoT 给出替代实现。

### 反向 2：A4-10 LDSC — LLM-token-subgoal HRL 也能用

**挑战**：LDSC 用 LLM 在 token 空间生成 semantic subgoal 提升长程任务效率。这与 R4 直接冲突。

**回应**：LDSC failure mode：subgoal 漂移 + grounding 失败。VZ 不采用 LLM-token-subgoal 而用 latent z_t，是为了让 subgoal 与 control 在同一空间避免 grounding gap。

### 反向 3：B3-02 AI Co-Scientist — generate-debate-evolve token 空间 deliberation 已产出湿实验验证的科学发现

**挑战**：DeepMind AI Co-Scientist 用多 LLM agent 在 token 空间做 generate-debate-evolve，已产出湿实验验证的 AML 体外抑瘤药候选。这是 token-space deliberation 的最强 frontier 实证。

**回应**：AI Co-Scientist 失败模式：① debate 收敛慢 ② evaluator 必须依赖外部湿实验 ③ 不能在线适应。VZ 的 metacontroller 路径不与 token-space deliberation 互斥——VZ 可以同时支持二者，但 latent 控制必须主导 long-horizon decision。

### 反向 4：B3-04 SIMA 2 — Gemini 当 reward generator 已跑通

**挑战**：SIMA 2 用 Gemini 当 task & reward generator，agent 在新环境从零自学新 skill；与 R-PE "内禀 PE 不外包"哲学冲突。

**回应**：SIMA 2 必然出现的 failure mode：reward generator drift（一旦 Gemini 自身的 task representation 偏，agent 整个 skill set 都偏）。VZ R-PE 的内禀 PE 路径是为了避免"reward 来源被外部模型绑架"。**追踪 SIMA 2 在 18 个月内是否暴露 drift，是 VZ 路径选择的关键反面证据**。

### 反向 5：C1-03 Darwin Gödel Machine — open-ended 全开放自修改

**挑战**：Sakana AI / UBC 的 DGM 让 agent 完全开放修改自己，archive 保留所有版本。与 R10 "分层 + 有界 + 快照可回滚"哲学冲突。

**回应**：DGM failure mode：archive 越大越难保证后代 agent 的 verification cost；scaling 后会撞 C1-01 Two-Gate 的 VC 维上限。VZ 不采用 DGM 风格，但其 archive 思想可借鉴到 substrate-owner refresh（rare-heavy 离线训练）。

### 反向 6：C2-05 Representation Engineering — reading 反推回训练已被工程化

**挑战**：CMU CAIS 的 RepE 显式鼓励把 reading（probe）反推回训练（control），与 R12 "评估只读，禁止反向作为学习源"边界冲突。

**回应**：RepE failure mode：probe quality 直接成为 reward，导致 Goodhart 化（probe 学到 spurious features 后 control 就会 sample 出对抗样本）。VZ R12 的"评估只读"正是为了避免这一类 Goodhart 失败；可借鉴 RepE 的 reading 工具（RepReading），但**禁止 RepControl 反向训练**。

---

## 当前研究的 5 个未解前沿（2026 → 2027）

基于 100 篇主名单的 gap 分析，当前 cognitive AGI 研究的 5 个最关键的未解问题：

### Q1. PE 的分布 readout 何时能进入 RLHF 主流？

- **现状**：B1-02 Depression Distributional Coding 给出"心理状态 = PE 分布形状"的可计算定义，但仅在 in-vivo / 神经科学层面验证。
- **缺口**：还没有论文把这个 readout 用作 RLHF 训练信号或 alignment 监控信号。
- **预测**：2026-Q4 ~ 2027-Q2 会出现第一篇 "PE-distributional-RLHF" 论文。

### Q2. mesa-optimization 能否被 disabled / aligned？

- **现状**：A5-02 / A5-03 / A5-05 已经证明 mesa-optimization **必然涌现**于 next-token 预训练。
- **缺口**：如果 mesa-optimizer 学到的内部目标与外部 reward 不一致（mesa-misalignment），目前没有任何工具能检测或干预。
- **预测**：2027 会出现"detecting mesa-objectives via mech interp"工作（C2 + A5 交叉）。

### Q3. latent action / z_t 能否跨 modality / cross-domain transfer？

- **现状**：A4-01 ETA / A2-08 AdaWorld / A4-02 Option Keyboard 都在自己的 domain 里证明 latent action 可学。
- **缺口**：还没有论文证明 "在 vision RL 学到的 latent action basis 可 zero-shot 用到 dialog / reasoning"。
- **预测**：2027-Q2 后可能出现"universal z_t basis"工作。

### Q4. ModificationGate 在 LLM-scale 上能否真正限住 reward hacking？

- **现状**：C1-01 Two-Gate 给出 PAC 形式工具，C1-02 SGM 给出 e-values；但都没在 frontier-scale LLM 上端到端验证。
- **缺口**：scale up 到 GPT-5 级时，VC capacity bound 是否还能给出有意义的约束？
- **预测**：2026-2027 这是 OpenAI / Anthropic / DeepMind 的共同前沿；一旦突破，alignment 范式会被重写。

### Q5. multi-agent diversity 能否替代 ToM 标签？

- **现状**：B2-07 Co-player Inference 已证明 decentralized RL + 多样 co-player 可让 cooperation 涌现。
- **缺口**：在长程关系 / persistent identity 场景（B2-01 Sophia / B2-06 Soul Engine）下，diversity-only 是否足够？还是必须加 explicit user_model owner？
- **预测**：2026-Q4 后可能出现"emergent ToM from multi-agent + persistent memory"工作。

---

## 本节终结：cognitive AGI 的"重大意义"是什么

把 5 大主题、14 R-ID、6 反向证据、5 未解前沿放到一起，可以给出一个综合判断：

> **cognitive AGI 在 2024-2026 经历了一次从"语言模型扩展"到"认知架构原则"的范式转变**。
>
> 转变的核心不是某个新算法或新架构，而是 5 个 frontier 主题在过去 18 个月**同时**走向工程成熟：
>
> 1. **PE 取代 reward 成为基本量**（B1 + A1 + A2）
> 2. **latent 取代 token 成为控制空间**（A1 + A4 + A5 + C2）
> 3. **涌现取代编码成为模块来源**（A4 + B3 + B2 + A5）
> 4. **架构与记忆与优化器同质化**（A3 + A5 + A2）
> 5. **自修改要门 + 评估必须只读**（C1 + B3 + C2）
>
> 这 5 个主题不是孤立的工程进步，而是**一个从下到上重新定义"什么是 cognitive agent"** 的认识革命。10 年前我们说 "agent = neural net + RL"，3 年前我们说 "agent = LLM + tool use"，2026 年我们正在说 **"agent = frozen substrate + emergent latent control + multi-timescale memory + bounded self-modification + read-only evaluation"**。
>
> **VZ 项目的位置**：在这场范式转变的 5 个主题的**交点**上——不是某一条主题的"另一种实现"，而是 5 个主题的"系统化整合"。这意味着 VZ 既享有 5 个主题各自的 frontier 背书（49 / 100 篇直接反哺），也承担 5 个主题各自的 failure mode 风险（6 条反向证据必须严肃对待）。
>
> **这就是 cognitive AGI 当前的"重大意义"**——它不是又一个 benchmark 突破，而是一次"如何制造心智"的认识范式更迭，而 VZ 是这场更迭的具体工程实例。

---

## 下一步

- [`11_vz_implications.md`](11_vz_implications.md) — 100 篇按 R-ID 重组 + P0/P1/P2 行动清单（具体到 spec 文件名）
- [`00_executive_summary.md`](00_executive_summary.md) — 高层执行摘要
- [`02_axis_walkthrough.md`](02_axis_walkthrough.md) — 10 轴走读（按轴回看具体论文）
