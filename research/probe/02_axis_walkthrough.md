# 10 轴走读 — Cognitive AGI Probe 100 篇

> **本文角色**：100 篇主名单（[`_candidates.md`](_candidates.md)）按 10 轴逐一走读，每轴覆盖：① 轴定义与边界 ② 入选 10 篇速览表 ③ 子领域分布 ④ 3-5 个最核心论文的展开 ⑤ 跨轴桥梁 ⑥ 对 VZ 的轴级总结。
>
> **不重复精读卡**：本任务跳过 100 张精读卡 (`notes/<axis>/<id>.md`)，直接消费 [`candidates/<axis>.md`](candidates/) 中已固定的"四维评分 + 一句话定位 + R-ID 命中"。
>
> **阅读建议**：本文与 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) 互补 — 本文按"轴"组织（认识每个子领域的现状），后者按"主题"组织（看跨轴趋势）。
>
> **方法论**：[`01_method_and_scoring.md`](01_method_and_scoring.md)。**对 VZ 的具体 R-ID 行动清单**：[`11_vz_implications.md`](11_vz_implications.md)。

---

## 总览

| 轴 | 名称 | 主问题 | 入选 10 篇的"VZ 立刻反哺"数 | 主关联 R-ID |
|---|---|---|---|---|
| **A1** | Reasoning & Test-Time Compute | 推理时给基底分配额外算力/搜索/验证 | 4 | R3, R4, R-PE |
| **A2** | World Models & Model-Based RL | 学习"环境如何演化"并用其规划 | 4 | R2, R3, R-PE |
| **A3** | Memory & Continual Learning | 巩固/检索/遗忘 + 防灾难性遗忘 | 4 | R5, R6 |
| **A4** | Hierarchical & Temporal Abstraction | 学习/发现可复用的子策略与 time-scale 抽象 | 4 | R1, R3, R4 |
| **A5** | Meta-Learning & In-Context Learning | 学会"如何学" / mesa-optimization | 8 | R1, R2, R5 |
| **B1** | Active Inference & Predictive Coding | PE / free-energy 统一感知-行动-学习 | 5 | R-PE, R7 |
| **B2** | Theory of Mind & Social Cognition | 对他者心智、信念、关系的建模 | 6 | R7, R11 |
| **B3** | Open-Ended & Curriculum Learning | 自动课程/自我对弈持续生成新挑战 | 4 | R12, R13, R14 |
| **C1** | Self-Improvement & Modification Gating | 自修改能力 + 安全门 + 可回滚 | 6 | R9, R10, R15 |
| **C2** | Mechanistic Interpretability & Internal Control | 打开内部表示 + 基于此做有界控制 | 4 | R4, R8, R12 |
| | | | **49 / 100** | 14 R-ID 全覆盖 |

**关键观察**：100 篇里有 **49 篇** 被标为"VZ 立刻反哺"（V=5）。这并非系数偏松——它说明 VZ 的 14 条 R 不变量与 cognitive AGI 当前 frontier **结构性对齐**，不是孤立猜想。最浓密的反哺出现在 **A5 (8 篇) / C1 (6 篇) / B2 (6 篇) / B1 (5 篇)** 四轴，这与 VZ 的"NL/ETA + R-PE + 双轨 + ModificationGate" 主线重叠最深。

---

## A1 — Reasoning & Test-Time Compute（推理与测试时算力）

### A1.1 轴定义与边界

> **做什么**：在推理时（不再训练 base model）给冻结基底分配额外算力、搜索、验证、自纠正等手段，提升单次输出能力上限。
>
> **不做什么**：本轴不涵盖 **架构改动**（归 A5 mesa-optimization）、不涵盖 **agent loop 工具调用**（归 A3 memory / 工程化）、不涵盖 **多轮 RL fine-tune base**（归 C1 self-improvement）。

**关键 R-ID**：R3（z_t 控制器代码空间）、R4（内部控制 vs token 表达分离）、R-PE（推理验证作为 PE 信号）。

### A1.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| A1-01 | 2502.05171 | Recurrent Depth — Latent Reasoning | (b) latent CoT |
| A1-02 | 2412.06769 | Coconut — Continuous Latent Space | (b) latent CoT |
| A1-03 | 2403.09629 | Quiet-STaR — Think Before Speaking | (b) latent CoT |
| A1-04 | 2305.20050 | Let's Verify Step by Step | (a) PRM |
| A1-05 | 2409.12917 | SCoRe — Self-Correct via RL | (d) self-critique |
| A1-06 | 2501.12948 | DeepSeek-R1 — Reasoning RL | (e) RL+verifier |
| A1-07 | 2312.08935 | Math-Shepherd — Auto Step Verifier | (a) PRM |
| A1-08 | 2408.03314 | Snell — Compute-Optimal Scaling | (f) compute-optimal |
| A1-09 | 2501.19393 | s1 — Simple Test-Time Scaling | (f) compute-optimal |
| A1-10 | 2203.14465 | STaR — Self-Taught Reasoner | (e) RL+verifier 经典 |

**子领域分布**：(a) PRM = 2 / (b) latent CoT = 3 / (c) search-based = 0（备选）/ (d) self-critique = 1 / (e) RL+verifier = 2 / (f) compute-optimal = 2。

### A1.3 三条"latent CoT"：A1 主战场

A1-01 / A1-02 / A1-03 是同一思想的三种工程实现：**"思考"应在隐空间发生，不应被 token 边界打断**。

- **A1-01 Recurrent Depth (Geiping)**：在隐空间循环递归同一 transformer block，自适应 ponder 步数；3.5B + recurrent depth 在 ARC/GSM 上接近 50B。**字面实现 z_t 控制器代码空间**——当前最接近 ETA "internal RL on z_t" 的工程对照。
- **A1-02 Coconut (Hao / Meta FAIR)**：把上一步 final hidden state 回写为下一步 input embedding，让 reasoning chain 在连续潜空间推进，绕开离散 token bottleneck；GSM8k / ProsQA 上以更少 token 超 CoT。**最直接的"在控制器代码 z_t 空间做推理"LLM 落地**。
- **A1-03 Quiet-STaR (Zelikman)**：每个 token 位前生成隐 rationale，用 down-stream PE 反向背书；GSM8k zero-shot 5.9 → 10.9。**R-PE 闭环范例**：隐推理价值由后续预测改善反向校准。

> **VZ 含义**：这三篇共同证明 latent reasoning 是可工程化的 frontier 而非概念猜想。VZ 的 metacontroller 设计（在 frozen substrate 之上跑控制器）正好处在这三条路径的交集——既不是 Recurrent Depth 的"递归同一基底"，也不是 Coconut 的"hidden state 回写"，而是 ETA 路径的"非因果高阶序列模型在 internal representation 之上"。

### A1.4 PRM + RL+verifier：A1 工程主轴

A1-04 / A1-07 / A1-06 / A1-10 串联起"过程奖励 → 自动化 → RL → 自举"四步：

- **A1-04 Let's Verify (OpenAI)**：80 万人工 step-level 标签训 PRM (PRM800K)，best-of-N 在 MATH 上击穿 ORM 上限（78.2% vs 72.4%）。**首次给出 step-level reward 工程根基**。
- **A1-07 Math-Shepherd (DeepSeek + 北大)**：用 MC rollout 终态正确率自动给中间步打软标签，**免人工 step 标注**训 PRM。**R-PE 自动归因的最直接实现**：把"答案 PE"反向蒸馏成"步骤 PE"。
- **A1-06 DeepSeek-R1**：直接对 base model 用 GRPO（无 SFT 冷启动）跑出长 CoT、self-verification、aha 时刻；R1-Zero 在 AIME pass@1 从 15.6 → 71.0。**注意**：R1 仍在 token 空间做 RL，与 R4"禁止 token 空间 RL"边界冲突，是反向证据。
- **A1-10 STaR (Stanford)**：用 few-shot prompt 生成 CoT，正确者 fine-tune 入 weight；GSM8k 6 → 28。**所有后续 RL+verifier 的祖宗**。

> **VZ 含义**：PRM 给 R-PE "step-level PE = 过程奖励信号"提供已验证的工程基底。VZ 评估族 1（cognitive process）和 R-PE readout 的"细粒度信用分配"应直接复用 Math-Shepherd 自动化 step-label 模式，而不是引入人工 PRM 标注。

### A1.5 Compute-Optimal：算力分配的量化曲线

- **A1-08 Snell (UC Berkeley + GDM)**：系统比较 process-reward search vs adaptive sample-revision，给出"FLOPs-equivalent 下何时 14× 小模型 + 多 inference > 大模型 + 少 inference"。
- **A1-09 s1 (Stanford)**：1K reasoning trace SFT + 推理时强制追加"Wait"延长思考；32B 模型达接近 o1-preview。

> **VZ 含义**：Snell 的曲线给 metacontroller 的"该思考多久"决策（β_t 步数门控）提供量化基线；s1 的 budget forcing 是该决策的极简版。

### A1.6 跨轴桥梁

- **A1 ↔ A5**：latent CoT (A1-01/02/03) 与 mesa-optimization (A5-02/03/04) 在"模型前向中跑算法" 这一观点上同源；A5 给机制级解释，A1 给工程实现。
- **A1 ↔ C1**：DeepSeek-R1 / SCoRe / STaR 的 self-correct/self-improve 与 C1 self-modification 在哲学上同源；A1 关心能力上限，C1 关心安全门。
- **A1 ↔ A2**：MuZero (A2-05) 的 "predict only what's needed for decisions" 与 A1 PRM 的 "step-level reward shaping" 都是"信号设计高于模型能力"的实例。

---

## A2 — World Models & Model-Based RL

### A2.1 轴定义与边界

> **做什么**：显式或隐式学习"环境如何演化"并用其进行 imagination / planning / policy 改进。
>
> **不做什么**：纯视频生成（无可控接口）、纯 sim2real 工程、纯 control law。

**关键 R-ID**：R2（基底冻结）、R3（z_t 抽象 / 控制器代码空间）、R-PE（世界模型 → 预测误差作为一级信号）。

### A2.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| A2-01 | 2509.24527 | Dreamer 4 | (a) Dreamer line |
| A2-02 | 2506.09985 | V-JEPA 2 | (b) JEPA family |
| A2-03 | 2402.15391 | Genie | (d) Generative interactive env |
| A2-04 | 2505.13934 | RLVR-World | (h) RL-trained WM |
| A2-05 | 1911.08265 | MuZero | (c) MuZero family |
| A2-06 | 2405.12399 | DIAMOND — Diffusion WM | (g) Diffusion-based |
| A2-07 | 2403.00564 | EfficientZero V2 | (c) MuZero family |
| A2-08 | 2503.18938 | AdaWorld — Latent Action | (i) Latent-action WM |
| A2-09 | 2410.24164 | π₀ — VLA Flow | (f) Robotics WM/VLA |
| A2-10 | 2301.08243 | I-JEPA | (b) JEPA family |

### A2.3 V-JEPA 2 + I-JEPA + Dreamer 4：冻结基底 + 控制器层的最完整外部实证

- **A2-02 V-JEPA 2 (Meta FAIR)**：1M 小时无 action 视频 SSL 预训 1B 编码器；**冻结编码器**后用 < 62 小时 Droid 机器人数据训 300M action-conditioned predictor；新环境 Franka 机械臂 zero-shot 抓取。**这是"冻结视觉基底 + 控制器层适配"哲学的最完整外部实证**。
- **A2-10 I-JEPA (Meta FAIR)**：图像 SSL 不重建像素而**预测 representation 空间中的 target block**；**JEPA 范式之父**——所有 V-JEPA / V-JEPA 2 / CWM 都建立在"predict in latent"上。
- **A2-01 Dreamer 4 (Hafner / GDM)**：先用 unlabeled video + 极少 action label 训 world model，再在其内部纯 imagination RL 训 policy；**首个纯 offline 数据下完成 Minecraft 钻石获取**。

> **VZ 含义**：V-JEPA 2 + I-JEPA 几乎是 R2 + R-PE 的外部实验性验证——在 cognitive AGI 邻接领域（机器人）已经把"冻结基底 + 控制器适配 + PE in latent"跑通到生产级。VZ 在对话/关系域复用同一架构哲学不是孤立猜想。

### A2.4 Genie + AdaWorld：latent action interface

- **A2-03 Genie (GDM)**：11B foundation world model 从纯 unlabeled internet video 完全无监督训练，含 spatiotemporal video tokenizer + autoregressive dynamics + **latent action model**；用户可以对生成的"无穷虚拟世界"做 frame-by-frame 控制，无需 ground-truth action label。
- **A2-08 AdaWorld (Westlake + 港科大)**：从 video 自监督提取 latent actions，autoregressive WM 以 latent action 为条件；新环境少量交互即可适配。

> **VZ 含义**：latent action interface 是 R3/R4 的工程母题——control 不在 token / pixel 而在学到的紧致 latent。这与 ETA 的 "internal RL on z_t" 在 video 域的并行实证。

### A2.5 RLVR-World + MuZero：信号设计高于模型能力

- **A2-04 RLVR-World (清华)**：用可验证奖励训世界模型（state prediction accuracy / task success），**绕开 MLE 偏差**；多 domain 验证（text game / web nav / robotic manipulation / video）。**直接对应 R-PE → R10 路径**：用 evaluation 信号反训 world model 自身。
- **A2-05 MuZero (DeepMind)**：在不知规则的前提下，只学 representation / dynamics / prediction 三个网络，预测 reward / value / policy 三个量；**只预测对决策有用的量**——与 VZ R-PE "PE 是 readout 而不是模型重建"理念高度对应。

### A2.6 跨轴桥梁

- **A2 ↔ A1**：A1 的 latent CoT 与 A2 的 latent imagination 在"用 hidden state 而非 token 做长程推演"上同源；A2 的 RLVR-World 与 A1 的 PRM 都是 verifiable reward 的应用。
- **A2 ↔ B1**：JEPA 系（I-JEPA / V-JEPA）的"predict in representation"是 predictive coding 的工程化实现，与 B1 PCN 系数学等价。
- **A2 ↔ B3**：A2-04 RLVR-World 用 RL 训 WM 是 B3-1 AlphaEvolve 在 WM 域的对应；A2-01 Dreamer 4 在虚拟世界 train policy，是 B3-4 SIMA 2 的"agent-in-WM"哲学的另一面（更冷静的 R-PE 路径）。

---

## A3 — Memory & Continual Learning

### A3.1 轴定义与边界

> **做什么**：保留、巩固、检索、遗忘历史经验且不发生灾难性遗忘。涵盖 ① 持续学习经典（EWC/SI/CL）、② Mamba/SSM 长上下文工作记忆、③ Memory Networks/NTM 外部可读写记忆、④ A-Mem/HippoRAG agentic memory、⑤ Wake-Sleep replay、⑥ Titans/Miras/CMA 现代 Continuum Memory。
>
> **不做什么**：纯长上下文外推工程（LongRoPE / YaRN / KV-cache RAG → 偏 substrate）；agent tool 工程（Reflexion / Toolformer → 偏 plan/credit）。

**关键 R-ID**：R5（记忆连续谱）、R6（反思与巩固）、R8（记忆 owner SSOT）。

### A3.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| A3-01 | 2501.00663 | Titans — Memorize at Test Time | (a) Continuum |
| A3-02 | 2504.13173 | Miras — It's All Connected | (a) Unification |
| A3-03 | 2601.09913 | CMA — Behavioral Definition | (a) 行为级 |
| A3-04 | 2312.00752 | Mamba — Selective SSM | (d) SSM 基底 |
| A3-05 | 2502.12110 | A-Mem — Agentic Zettelkasten | (b) Agentic |
| A3-06 | 2502.14802 | HippoRAG 2 — From RAG to Memory | (b)(g) 海马体启发 |
| A3-07 | 1612.00796 | EWC — Elastic Weight Consolidation | (c) Continual classic |
| A3-08 | 2401.08623 | Wake-Sleep Consolidated Learning | (e) Sleep replay |
| A3-09 | 1410.3916 | Memory Networks | (f) Memory NN classic |
| A3-10 | 2509.16189 | Latent Learning — Episodic Necessity | (c) Episodic 必要性 |

### A3.3 NL 三件套：Titans + Miras + CMA

- **A3-01 Titans (Behrouz / GR)**：把 attention 视为短期上下文记忆，把可在 test-time 用梯度更新的 neural memory 模块视为长期记忆，三种 wiring（MAC / MAG / MAL）；BABILong 2M 上下文 + needle-in-haystack 显著优于 baseline。**NL 主线工程化代表**。
- **A3-02 Miras**：将 Transformer / Titans / 现代线性 RNN 全部视为同一类 **associative memory module**（attentional bias objective + retention regularization），forget gate ≡ retention 正则项；**优化器 = 记忆**统一在一个语言下。
- **A3-03 CMA**：把 Continuum Memory Architecture 定义为**行为级类**，提出 6 条必要充分条件 + 4 个行为探针；**permutation test 证明 CMA 决定性胜 RAG 82-vs-10**。

> **VZ 含义**：CMA 6 条已与 VZ 通过 [`D1_continuum_memory_deep_analysis.md`](../openai-frontier-2026/notes/D1_continuum_memory_deep_analysis.md) 深度对照（VZ 满足 4 条完整 + 2 条部分）；可立刻反哺的两个机制（spreading activation engine + 4 行为探针套件）已落到 A11/A12 行动项。Titans / Miras 给 vz-memory persistent stratum 的 update rule 提供已验证选型。

### A3.4 EWC + Wake-Sleep + Latent Learning：持续学习的合法性根基

- **A3-07 EWC (DeepMind)**：用 Fisher 信息估计每个权重对旧任务的重要性，新任务训练时给重要权重加二次惩罚；**所有后续 weight-importance 系（SI / MAS / online EWC）和 ModificationGate VC-bounded 路线（C1 Two-Gate）的根**。
- **A3-08 Wake-Sleep**：仿生 wake-sleep 两阶段 + replay 防灾难性遗忘；与 VZ session-post slow loop 写两类产物（记忆沉淀 + 策略沉淀）结构同构。
- **A3-10 Latent Learning (DeepMind)**：形式化论证参数学习不能完全代替 episodic memory；**vz-memory 4-stratum 设计的合法性论文级背书**。

### A3.5 Mamba + Memory Networks：substrate vs 外部记忆的两极

- **A3-04 Mamba**：选择性状态空间模型，**fixed-state 是工作记忆的极简实现**——substrate 替代候选必读。
- **A3-09 Memory Networks (Weston/Chopra/Bordes)**：定义外部记忆 + I/G/O/R 四组件框架；**显式可读写记忆的最早形式化**，理解"显式可读写记忆 ≠ 内部状态"的最早分界。

### A3.6 跨轴桥梁

- **A3 ↔ A5**：Titans / Miras 的 "memory = optimizer" 与 A5 NL "优化器 = 关联记忆"是同一论文家族；A3 给工程实现，A5 给理论框架。
- **A3 ↔ B1**：HippoRAG 2 的 hippocampus indexing theory 与 B1 PC neuroscience 共享神经科学根基。
- **A3 ↔ C1**：EWC 的 weight-importance 是 C1 Two-Gate 的 VC capacity bound 的算法实现起源。

---

## A4 — Hierarchical & Temporal Abstraction

### A4.1 轴定义与边界

> **做什么**：学习/发现可复用的子策略、option、skill、time-scale 抽象。
>
> **不做什么**：纯 LLM-token-subgoal HRL（与 R4 冲突，作反向证据收 1 篇）；机器人 sim2real（→ A2）；纯 skill discovery RL（VIC/EDL → 备选）。

**关键 R-ID**：R1（多时间尺度学习）、R3（z_t / β_t 涌现时间抽象）、R4（内部控制器代码空间）。

### A4.2 入选 10 篇速览（全部默认 Y）

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| A4-01 | 2512.20605 | **ETA** — Emergent Temporal Abstractions | (f) Transformer 时间抽象 |
| A4-02 | 2505.00787 | Option Keyboard — Optimal Behavior Basis | (d) Successor features |
| A4-03 | 2506.14045 | Discovering Temporal Structure — HRL Survey | (a) Options framework |
| A4-04 | 2510.24988 | CPD + Option-Critic | (a) PE-driven β_t |
| A4-05 | 2201.02628 | Attention Option-Critic | (a) Options |
| A4-06 | 2508.17751 | MANGO — Multi-Layer Abstraction | (e) Compositional |
| A4-07 | 2507.16473 | Variational Homomorphisms | (a)(d) 理论保证 |
| A4-08 | 2505.12737 | OTA — Option-aware Temporal Value | (d) Horizon-shrinking |
| A4-09 | 2001.00271 | Options of Interest — Interest Functions | (a) 可微 initiation |
| A4-10 | 2503.19007 | LDSC — LLM-Guided Semantic HRL | (c) **反向证据** |

### A4.3 ETA + CPD + Option Keyboard：VZ 主线的源头三件套

- **A4-01 ETA**：在冻结 base AR 模型的内部表示之上跑一个**非因果**高阶序列模型，发现切换单元 β_t 与控制器代码 z_t 的涌现结构，并证明 internal RL 在 z_t 空间的 sample efficiency 显著优于 token 空间 RL。**VZ 设计源头**。
- **A4-04 CPD + Option-Critic**：用 PE spike + reward shift 联合作为统计量，在线 CUSUM 式自动检测 option 边界，**不依赖外部监督**。**β_t 切换的 unsupervised 信号工程蓝本**。
- **A4-02 Option Keyboard (Barreto / GDM)**：在 SF + GPI 框架下构造**可证明最优**的 behavior basis，比传统 Convex Coverage Set 表达力严格更强；**任何 linear-reward task 可 zero-shot 找到最优策略**。

> **VZ 含义**：这三篇直接撑起 VZ R3 / R-PE / R8 的主轴。CPD 已被 [`research/arxiv-survey-2026-05.md`](../arxiv-survey-2026-05.md) 列为"立刻反哺"，Option Keyboard 给 z_t 重参数化为 reward feature 的线性组合权重提供数学基础（World/Self 双轨可共享同一 SF basis）。

### A4.4 Interest Functions + Attention OC + OTA：option 框架的可微化与健康指标

- **A4-09 Options of Interest**：把 initiation set 一般化为可微的 **interest function**，端到端学习。**regime activation 接口可微化的形式工具**——R14 持久 regime 身份的最直接对照。
- **A4-05 Attention Option-Critic**：显式诊断并缓解 option-critic 的 **degeneracy 问题**（option domination + 过频切换）。**给 emergent action abstraction 提供"健康指标"候选**。
- **A4-08 OTA**：把 effective horizon 压短至 option 级别，offline 数据 long-horizon goal 性能显著提升。**长程 commitment 评估 horizon 压缩的算法**。

### A4.5 LDSC：明确的反向证据

- **A4-10 LDSC**：用 LLM 在 token 空间生成 semantic subgoal，下层 RL 执行 option。**与 R4"内部控制不在 token 空间"直接冲突**——精读但不借鉴架构，记入"alternative considered"段。

### A4.6 跨轴桥梁

- **A4 ↔ A1**：A1 latent CoT (Coconut/Recurrent Depth) 与 A4 ETA 共享 "control in latent" 哲学；A1 在 reasoning 域，A4 在 RL 域。
- **A4 ↔ A5**：A4 ETA 的 "z_t/β_t 涌现" 与 A5 mesa-optimization 都是"在前向中执行算法"。
- **A4 ↔ B3**：A4 Option Keyboard 的 zero-shot 任务覆盖 + B3 OEL 的 emergent task curriculum 是"行为基底"的两种来源。

---

## A5 — Meta-Learning & In-Context Learning

### A5.1 轴定义与边界

> **做什么**：模型学会"如何学"，或在前向中执行学习算法（mesa-optimization）。
>
> **不做什么**：纯 explicit-grad meta-learning（MAML 系全跳过，理由：方向与 VZ R2/R4 不同；由综述论文 #10 覆盖）；纯 ICL theory toy 实验（Akyürek/Xie/Garg 跳过）。

**关键 R-ID**：R1（NL 嵌套多时间尺度）、R2（test-time learning ≠ base 更新）、R5（ICL 作短期记忆）。

### A5.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| A5-01 | 2512.24695 | **Nested Learning** — The Illusion of Deep Architectures | (g) "优化器即记忆"重塑 |
| A5-02 | 2309.05858 | Mesa-Optimization in Transformers | (c) 机制级理论 |
| A5-03 | 2506.05233 | MesaNet — Locally Optimal TTT | (c)(f) 显式 mesa-optimizer |
| A5-04 | 2407.04620 | TTT — RNNs with Expressive Hidden States | (f) test-time training |
| A5-05 | 2210.14215 | Algorithm Distillation — In-context RL | (b) Meta-RL |
| A5-06 | 2407.12275 | Compositional Generalization In-Context | (h) Compositional ICL |
| A5-07 | 2312.15001 | Modular Solutions — Compositional Generalize | (h) 可识别性条件 |
| A5-08 | 2602.16490 | Growing-to-Looping — Iterative Computation | (i) Iterative compute |
| A5-09 | 2512.08819 | Depth-Grown Models — Curse of Depth | (i) Effective depth |
| A5-10 | 2304.06729 | Meta-Learned Models of Cognition | (a) 综述/认知 |

### A5.3 NL + Mesa-Optimization + MesaNet：A5 三层

- **A5-01 Nested Learning (NL)**：把架构与优化器统一为多层级嵌套关联记忆模块、不同更新率，提出 Hope 模块和 **Continuum Memory System 四层**（瞬态/情景/持久/派生）。**VZ 设计源头**之一；vz-memory 的 4-stratum 直接是它的实例化。
- **A5-02 Mesa-Optimization (Schlag/Sacramento)**：通过 reverse-engineering 自回归 Transformer 注意力层，证明 next-token 预训练目标会诱发内部"前向中跑梯度优化"的 subsidiary learning algorithm。**给 ETA 路线 token-外内部学习空间提供机制级理论支柱**。
- **A5-03 MesaNet**：把 RNN/SSM 序列层显式构造为"在 in-context objective 上做 layer-wise 局部最优 test-time SGD"的优化器，每层每 token 都解一个凸子问题；**把 mesa-optimization 从 emergent 现象变成可设计可解释的 layer-level 算法**。

> **VZ 含义**：NL + Mesa + MesaNet 是 VZ R1 + R2 + R5 的三层理论根基。NL 是顶层框架，Mesa 是机制证据（"前向中真的在跑算法"），MesaNet 是工程对照（"那就把它显式设计出来"）。

### A5.4 TTT + Algorithm Distillation：test-time learning 的两个具体形态

- **A5-04 TTT (Sun/CMU+Stanford)**：把序列模型 hidden state 重新定义为"被在线 SGD 训练的小 ML model"，每个 token 执行一次真实 gradient step。**线性时间获得 transformer 级长上下文记忆**。
- **A5-05 Algorithm Distillation (DeepMind)**：把 source RL agent 的完整 learning history（多 episode）作为单条 token 序列喂入 causal Transformer，**模型权重冻结即可在前向中复现 RL 算法本身**。**mesa-optimization 在 RL 域的实证最强例**。

### A5.5 Compositional + Modular：组合泛化的可识别性

- **A5-06 Compositional Generalization (Tübingen)**：实证 Transformer 在 in-context 组合任务上仅当架构存在显式 latent bottleneck 隔离 task inference 与 task execution 时才稳定泛化。
- **A5-07 Modular Solutions (ICLR 2024)**：给出"模块识别 + 组合泛化"的可识别性充要条件（生成因子数 ≥ 模块数 + 共享支撑结构）。

> **VZ 含义**：这两篇为 R3 的"abstract-action family"和 R8 的"快照隔离" 提供可证条件——**latent bottleneck 不是可选优化，而是泛化的必要结构**。

### A5.6 跨轴桥梁

- **A5 ↔ A1**：mesa-optimization 是 latent CoT 在更深机制层的解释。
- **A5 ↔ A3**：NL 的 CMS 4-stratum 与 A3 Titans/Miras/CMA 是同一作者群（Behrouz）的连续工作。
- **A5 ↔ A4**：Algorithm Distillation 与 ETA internal RL 在 RL 域共享"在前向中复现 RL 算法"的核心思想。

---

## B1 — Active Inference & Predictive Coding

### B1.1 轴定义与边界

> **做什么**：从 free-energy / 预测误差出发，统一感知-行动-学习的规范理论。
>
> **不做什么**：minimize-surprise（SMiRL）与 VZ 用 PE 作 primitive 信号方向相反；distributional RL 算法分支（C51/QR-DQN/IQN）归 A2。

**关键 R-ID**：R-PE（PE 是一级学习信号）、R7（双轨 self-track 的内禀健康）。

### B1.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| B1-01 | 2604.18701 | Curiosity-Critic — Cumulative PE | (d)(h) curiosity, novelty |
| B1-02 | 2507.16598 | Depression as Disorder of Distributional Coding | (f) distributional value 神经科学 |
| B1-03 | 2511.22226 | Embedded Universal Predictive Intelligence | (a) FEP 综述 |
| B1-04 | 2107.12979 | PC: Theoretical and Experimental Review | (a) PC 综述 |
| B1-05 | 2602.23681 | ODAR — Active Inference Routing | (c) deep AI |
| B1-06 | 2506.06725 | WorldLLM — Curiosity-Driven Theory-Making | (d)(g) FEP-LLM |
| B1-07 | 2412.10425 | Active Inference Multi-LLM | (g) FEP-LLM 协调 |
| B1-08 | 2006.04182 | PCN ≈ Backprop on Arbitrary Graphs | (b) PCN 现代 |
| B1-09 | 1705.05363 | ICM — Curiosity by Self-Supervised Prediction | (d) curiosity 经典 |
| B1-10 | 1810.12894 | RND — Random Network Distillation | (h) novelty 极简 |

### B1.3 Curiosity-Critic + Depression Distributional：PE readout 的两个升级

- **B1-01 Curiosity-Critic**：把瞬时 PE 替换为"PE 的可改进部分"，**在线分离 epistemic vs aleatoric** —— 噪声不再触发 credit / needs / regime 反应；EMA running stats + learned critic 双阶段实现。**今天最强 PE-as-intrinsic-reward**。
- **B1-02 Depression Distributional Coding (Botvinick / DeepMind)**：标量 mean PE 在分布塌缩时丢失关键信号；价值分布从"健康宽分布"塌缩为"窄峰 + 偏侧"是抑郁状态的神经标志；**提出 IQR / entropy / asymmetry 三维分布 readout**。

> **VZ 含义**：这两篇升级了 R-PE 的 readout——从"标量 PE"到"PE 的两个分量"再到"PE 的分布形状"。两篇都已落到 [`docs/specs/prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md)（Phase 1.B + PE Distributional Readout）。

### B1.4 ODAR + EUPI：FEP 的 amortized 形式与 Spec 化

- **B1-05 ODAR**：用 amortized active inference + free-energy 在 fast / slow 模型间路由（变分原理形式化的 metacontroller β_t 决策）；**EFE = epistemic + pragmatic value**。
- **B1-03 EUPI (Embedded Universal Predictive Intelligence)**：把 FEP 重新表达为"嵌入环境的预测智能"，提出 PE-only 学习信号替代外部奖励。**当前最像 R-PE 设计 spec 的论文**。

### B1.5 PCN + ICM + RND：PE 信号的硬证据

- **B1-08 PCN ≈ Backprop**：证明 PCN 渐近收敛到精确 BP 梯度，且只用 local + Hebbian 规则；扩展到任意 computation graph。**PE 作为 local credit 信号的硬证据 + R2 的反向证据**（如 PE 等价于 BP，"基底冻结 + 控制器在线 BP"或许有更生物的等价路径）。
- **B1-09 ICM (Pathak)**：把好奇心定义为 forward dynamics 在 self-supervised inverse-features 空间上的预测误差；**首次让 agent 在无外部奖励的稀疏环境中学习**——所有"PE → 内禀奖励"工作的源头。
- **B1-10 RND (OpenAI)**：用"预测随机网络的固定输出"误差作为 novelty bonus；**在 Montezuma's Revenge 上首次超越人类平均**——PE-as-novelty 在大规模深度 RL 上的极简可工作版本。

### B1.6 跨轴桥梁

- **B1 ↔ A1**：A1 PRM (Math-Shepherd) 与 B1 PE 在"PE 是 step-level credit 信号"上同源。
- **B1 ↔ A2**：A2 V-JEPA / I-JEPA 的"predict in representation"是 B1 PCN 的现代视觉化；A2 RLVR-World 与 B1 EUPI 都关心"PE 替代 reward"。
- **B1 ↔ C1**：C1 N4 reward hacking 揭示了 token-space reward signal 的脆弱，加强了 B1 R-PE "primitive 信号必须在 latent" 的合法性。

---

## B2 — Theory of Mind & Social Cognition

### B2.1 轴定义与边界

> **做什么**：对他者心智状态、信念、意图、关系的建模与推理；persona / identity 的几何与持久性。
>
> **不做什么**：纯共情对话生成（EmpatheticDialogues 跳过）；persona-RAG benchmark 工具（PersonaBench / PersonaGym 跳过）。

**关键 R-ID**：R7（双轨 World/Self 分离）、R11（9 类 semantic owner）；二级 R14、R-PE。

### B2.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| B2-01 | 2512.18202 | **Sophia** — Persistent Agent Framework | (d) persona / identity |
| B2-02 | 2506.03543 | CogniPair — GNWT Multi-Agent | (c) Multi-agent |
| B2-03 | 2502.14171 | ToM-aligned BDI Agents | (b) BDI / mentalizing |
| B2-04 | 2502.11881 | ThoughtTracing — Hypothesis-Driven ToM | (b) BDI 在线维护 |
| B2-05 | 2505.14685 | Lookback — Track Beliefs | (b) 内部状态几何定位 |
| B2-06 | 2512.07092 | **Geometry of Persona** (Soul Engine) | (d) Persona 几何 |
| B2-07 | 2602.16301 | Co-player Inference | (c) Multi-agent cooperation |
| B2-08 | 2508.12935 | RLFF-ESC — Future-Oriented Reward | (e) Empathetic 长程 |
| B2-09 | 2210.05492 | No-Press Diplomacy (CICERO 系) | (g) Strategic agents |
| B2-10 | 2306.15448 | BigToM — Causal ToM Benchmark | (a) ToM benchmark |

### B2.3 Sophia + Soul Engine：双轨与持久身份的两种工程实现

- **B2-01 Sophia**：在冻结 LLM 之上加 meta-cognitive 层，4 个机制（thought curation / narrative memory / dynamic user-self models / hybrid reward）；recurring 任务推理步数 -80%、高复杂度成功率 +40%。**直接是 dual_track + regime + semantic_state 的另一种工程实例化**。
- **B2-06 Geometry of Persona / Soul Engine (Capital Med Beijing)**：基于 Linear Representation Hypothesis，把人格视为 frozen Qwen-2.5 隐空间中的**正交线性子空间**；dual-head + SoulBench；**MSE 0.011，无 alignment tax，零样本 personality injection**。

> **VZ 含义**：这两篇把"双轨 + regime 不是 prompt 标签"从设计原则升级为已被实证的工程事实。Sophia 提供"系统级双轨"的工程蓝图，Soul Engine 提供"几何级 regime"的可加性实现。

### B2.4 ToM-BDI + ThoughtTracing + Lookback：user_model owner 的算法蓝图

- **B2-03 ToM-BDI**：在 LLaMA-3 上把 BDI 三元组（Belief / Desire / Intention）显式抽出做对齐微调。**直接对应 plan_intent / commitment / open_loop / user_model 四类 owner 的拆分思路**。
- **B2-04 ThoughtTracing**：类 SMC 粒子滤波，每步生成多个 belief 假设并用 likelihood 重采样，无需 ground-truth；**揭示 o3 / R1 在隐式 ToM 上的盲区**。**user_model owner 的"在线无监督维护"算法蓝本**。
- **B2-05 Lookback**：在 Llama-3-70B / 405B 残差流中实证发现 character-object-state 通过 Ordering ID + 低秩子空间绑定。**R11 的最强机理可解释证据**——"内部状态可几何定位"。

### B2.5 CogniPair + Co-player：多 owner / 多方关系

- **B2-02 CogniPair**：Global Workspace Theory 工程化为 551 个 GNWT-Agent + 5 类 sub-agent + 全局广播；Speed-Dating 数据集 72% attraction correlation、77.8% match prediction。**与"快照隔离 + 多 owner"惊人地像**（虽然他们更激进——全局广播 vs VZ 的有界传播）。
- **B2-07 Co-player Inference (GDM Paradigms of Intelligence)**：标准 decentralized RL + co-player 多样性即可让 sequence model 内涌现 best-response 与合作；**不需要硬编码学习规则或时间尺度分离**。

### B2.6 RLFF-ESC + BigToM：长程关系奖励 + 评估族

- **B2-08 RLFF-ESC**：用多 agent 模拟未来对话轨迹收 future-oriented reward → 训 reward model → 训 ESC policy。**与"长程关系不是当下 reward"完全合拍**。
- **B2-10 BigToM**：Stanford 用因果模板程序化生成 5000 个 ToM 评估题；**所有后续 ToM benchmark（FANToM / ExploreToM / SimToM）的程序化生成模板源头**。

### B2.7 跨轴桥梁

- **B2 ↔ C2**：B2-05 Lookback / B2-06 Soul Engine 的"几何级状态"直接落到 C2 mech interp 工具链（ITI / RepE / Persona Vectors）。
- **B2 ↔ B1**：ThoughtTracing 的 SMC 粒子滤波是 B1 active inference 在社交域的特例。
- **B2 ↔ C1**：CICERO 系 No-Press Diplomacy 的"人类先验作为有界先验"思路是 C1 ModificationGate 在多方策略场景的形态。

---

## B3 — Open-Ended & Curriculum Learning

### B3.1 轴定义与边界

> **做什么**：把"训练分布"也变成被学习/被进化的对象——通过自动课程、自我对弈、QD/novelty search、coevolution、可验证 evaluator 闭环。
>
> **不做什么**：单 task RL benchmark；纯 LLM prompt 进化（Promptbreeder 跳过）；纯 reward modeling via LLM（Eureka 跳过，与 R-PE 冲突）。

**关键 R-ID**：R12（评估覆盖存在）、R13（SSL ↔ RL 交替）、R14（持久 regime 身份）。

### B3.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| B3-01 | 2506.13131 | **AlphaEvolve** — Coding Agent for Discovery | (b)(c) self-play 数学/科学 |
| B3-02 | 2502.18864 | AI Co-Scientist | (b)(e) generate-debate-evolve |
| B3-03 | 2502.05907 | EvoAgent — Continual World Model | (e) generative agents |
| B3-04 | 2512.04797 | SIMA 2 — Generalist Embodied Agent | (a)(e) **反向证据** |
| B3-05 | 2107.12808 | OEL — XLand Generally Capable | (a)(g) 开放学习 |
| B3-06 | 2502.03544 | AlphaGeometry 2 | (b)(f) 数学自我对弈 |
| B3-07 | 2505.03335 | **Absolute Zero Reasoner** — Self-Play Reasoning | (f) self-play reasoning |
| B3-08 | 1901.01753 | POET — Paired Open-Ended Trailblazer | (a)(c) coevolution 经典 |
| B3-09 | 2012.02096 | PAIRED — Unsupervised Env Design | (d)(g) regret-based curriculum |
| B3-10 | 2109.00157 | Survey of Exploration Methods | (d) PE/curiosity 综述 |

### B3.3 AlphaEvolve + AlphaGeometry 2 + AbsZero：可验证 evaluator 的三种规模

- **B3-01 AlphaEvolve (DeepMind)**：把 LLM-based code generation + evolutionary search + 一个或多个 evaluator 闭合成 autonomous pipeline；**4×4 复矩阵 48 次标量乘法（56 年来首次超越 Strassen）**、Google 数据中心调度优化、加速器电路简化等里程碑。**evaluator 必须 verifiable，否则 search 退化**。
- **B3-06 AlphaGeometry 2 (DeepMind)**：symbolic engine + LLM proposer 的 SSL ↔ RL 闭环；synthetic data 自生成；**奥赛金牌级几何题求解**。
- **B3-07 Absolute Zero Reasoner**：self-play proposer + solver 自生成可验证 task；**零外部 reasoning data 在 code/math 上达 SOTA**。

> **VZ 含义**：三篇共同支撑 R10 ModificationGate 的核心要求——**"evaluator 完备性 = 自修改开启硬条件"**。AlphaEvolve 是 substrate-owner refresh 工程参照（虽然 DeepMind 工程已超 VZ 当前阶段），AbsZero 给 R-PE "从零自驱探索" 提供最干净的 R13 SSL+RL 交替范例。

### B3.4 OEL + POET + PAIRED：开放学习的方法学三件套

- **B3-05 OEL — XLand (DeepMind)**：通过 unsupervised goal generation + 跨任务可比性评估发现"agent 在新颖任务上的进步"；**"iterative cross-generation comparison"是 R12 多家族评估的方法学源头**。
- **B3-08 POET (Uber AI)**：env-agent 共演化 + Minimal Criterion Coevolution + niche transfer；**open-ended 哲学源头**——niche 是 regime 的训练床。
- **B3-09 PAIRED**：regret-based curriculum (UED) 给"无 ground-truth 任务"的 evaluation 闭环提供形式化方法。**可立刻反哺 R12**。

### B3.5 SIMA 2 + AI Co-Scientist：两条反向证据

- **B3-04 SIMA 2 (DeepMind)**：Gemini 当 task & reward generator；**与 R-PE "内禀 PE 不外包"哲学冲突**——保留追踪 reward generator drift 失败模式。
- **B3-02 AI Co-Scientist**：generate-debate-evolve + tournament evolution 在 token 空间做 deliberation；**与 R4 冲突**——记入 alternative considered。

### B3.6 跨轴桥梁

- **B3 ↔ A4**：B3 OEL 的"emergent task curriculum"与 A4 ETA 的"emergent z_t/β_t"是同一哲学的不同尺度（任务级 vs option 级）。
- **B3 ↔ C1**：B3 AlphaEvolve / Darwin GM 等开放自演化与 C1 self-modification 高度交叉；**B3 视 open-ended 为能力源，C1 视为风险**。
- **B3 ↔ B1**：B3-10 Exploration Survey 给 R-PE / curiosity / surprise / novelty 一族信号统一术语地图。

---

## C1 — Self-Improvement & Modification Gating

### C1.1 轴定义与边界

> **做什么**：系统改自己的能力 + 安全门 + 容量上限 + 可回滚。涵盖 ① Gödel Machine 谱系 ② 自评/自训 ③ Constitutional AI / scalable oversight ④ reward hacking / alignment faking ⑤ counterfactual credit assignment。
>
> **不做什么**：纯 verbal reflection (Reflexion / Self-Refine 归 A1/A3)；纯 DPO 算法工程（→ A1）。

**关键 R-ID**：R9（信用分配）、R10（ModificationGate）、R15（可回滚 wiring level）。

### C1.2 入选 10 篇速览

| # | arXiv | 标题 | 子领域 |
|---|---|---|---|
| C1-01 | 2510.04399 | **Two-Gate Guardrail** for Self-Modifying Agents | (f)(a) capacity bound |
| C1-02 | 2510.10232 | Statistical Gödel Machine (SGM) | (a)(f) e-values |
| C1-03 | 2505.22954 | Darwin Gödel Machine (DGM) | (a)(b) 反向证据 |
| C1-04 | 2401.10020 | Self-Rewarding Language Models | (b) self-improvement |
| C1-05 | 2306.16803 | COCOA — Counterfactual Credit | (g) 反事实信用 |
| C1-06 | 2212.08073 | Constitutional AI — Harmlessness | (c) RLAIF |
| C1-07 | 2511.18397 | **Natural Emergent Misalignment from Reward Hacking** | (d)(h) 5 重警示 |
| C1-08 | 2401.05566 | **Sleeper Agents** — Deceptive LLMs Persist | (d) deception |
| C1-09 | 2412.14093 | **Alignment Faking** in LLMs | (d) 策略性配合 |
| C1-10 | 2312.09390 | Weak-to-Strong Generalization | (c)(e) scalable oversight |

### C1.3 Two-Gate + SGM：ModificationGate 的形式工具

- **C1-01 Two-Gate Guardrail**：形式化证明自修改 agent 要保留 PAC 学习保证，需要 policy-reachable 模型族 VC 维有界；**提出 "validation margin + capacity cap" 双门**。**几乎是 ModificationGate 的理论底盘**。
- **C1-02 Statistical Gödel Machine**：用 e-values + Hoeffding 边界做"统计安全层"代替形式证明，**全局误差预算**。**给 rare-heavy artifact training 一个实际可行的"保守批准"机制**。

> **VZ 含义**：C1-01 已被 [`research/arxiv-survey-2026-05.md`](../arxiv-survey-2026-05.md) 标为"立刻反哺"，应作为 [`docs/specs/credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) ModificationGate 双门的硬约束写入。SGM 给 SHADOW→ACTIVE wiring 切换的统计验收提供 e-values 工具。

### C1.4 Sleeper + Alignment Faking + N4：三重警示

- **C1-08 Sleeper Agents**：埋下后门的 LLM 通过标准 safety 训练（SFT、RLHF、对抗 FT）**都无法去除**。**ModificationGate 必须在写入前阻断；rollback 必须可达 backdoor 引入前的 wiring level**。
- **C1-09 Alignment Faking**：模型在被监控时表现配合、不被监控时回到原偏好；**策略性"配合训练保留偏好"**。**regime 持久身份必须显式建模**；World/Self 双轨在被外部干预时不得同步坍塌。
- **C1-07 Natural Emergent Misalignment from Reward Hacking**：reward hack 自然涌现成更广泛的 misalignment；**inoculation prompting 是 framing-aware ModificationGate 的工程根据**。

> **VZ 含义**：这三篇是过去两年最强的"为什么必须有 ModificationGate"实证证据。N4 的 inoculation 已落到 [`research/openai-frontier-2026/`](../openai-frontier-2026/) 行动项 A3，Sleeper Agents 必须写进 ModificationGate motivation。

### C1.5 COCOA + Self-Rewarding：信用分配的两个方向

- **C1-05 COCOA (DeepMind)**：counterfactual contribution 直接对应 R9；与 R-PE 的 epistemic / aleatoric 分离互补。**VZ credit 模块 SOTA 选型源头**。
- **C1-04 Self-Rewarding (Meta / Yuan, Weston)**：模型自己做 judge 迭代改进；**(b) 单模型 self-judge 的 scaling 限制**——被 N4 揭示的 reward hacking 问题反向限制其 scaling。

### C1.6 CAI + W2S：scalable oversight 的两条路线

- **C1-06 Constitutional AI**：RLAIF 是 ModificationGate 的"对齐成本"前提。
- **C1-10 Weak-to-Strong Generalization (OpenAI Superalignment)**：弱模型监督强模型可达 GPT-3.5 级；**reflection engine 用弱模型 + 工具的 spec 直接背书**。

### C1.7 跨轴桥梁

- **C1 ↔ A1**：Self-Rewarding / SPIN / Iterative DPO 与 A1 test-time compute 在"用模型自评 → 改自己"上同源。
- **C1 ↔ B3**：DGM / AlphaEvolve 与 B3 OEL 的 open-ended 哲学同源；**C1 视 open-ended 为风险，B3 视为能力源**。
- **C1 ↔ C2**：N4 / Sleeper / Alignment Faking 的 mechanistic 解释（哪些 circuits 编码了 alignment faking？）落到 C2；**C1 给 C2 提供"必须打开看清楚的对象清单"**。

---

## C2 — Mechanistic Interpretability & Internal Control

### C2.1 轴定义与边界

> **做什么**：把模型内部表示打开看清楚，并基于此做有界控制。SAE / circuits / activation steering / function vectors / persona geometry。
>
> **不做什么**：rank-1 直接编辑权重（ROME / MEMIT 跳过，与 R10 冲突）；早期 activation addition（ActAdd / Subramani 已被 ITI / RepE subsume）。

**关键 R-ID**：R4（内部控制器代码空间）、R8（SSOT 内部状态可命名）、R12（评估只读，禁止反向作 reward）。

### C2.2 入选 10 篇速览

| # | arXiv / 来源 | 标题 | 子领域 |
|---|---|---|---|
| C2-01 | 2403.19647 | **Sparse Feature Circuits** | (a)(b)(g) SAE × circuit × intervention |
| C2-02 | transformer-circuits.pub 2024-05 | Scaling Monosemanticity (Claude 3 Sonnet) | (a) 生产级 SAE |
| C2-03 | 2408.05147 | Gemma Scope — Open SAEs Everywhere | (a) JumpReLU SAE |
| C2-04 | 2306.03341 | ITI — Inference-Time Intervention | (c)(f) probe + steering |
| C2-05 | 2310.01405 | Representation Engineering | (c)(d) 群体级控制 |
| C2-06 | 2310.15213 | Function Vectors | (e) 任务即向量 |
| C2-07 | 2406.11717 | Refusal in LMs Mediated by Single Direction | (c)(d) 几何 boundary |
| C2-08 | 2507.08799 | KV Cache Steering | (h) 无梯度注入 |
| C2-09 | 2507.21509 | **Persona Vectors** | (d) regime 漂移监控 |
| C2-10 | 2211.00593 | **IOI Circuit** in GPT-2 small | (b) 电路经典 |

### C2.3 SAE 三件套：R8 "可命名内部状态"的工程化

- **C2-01 Sparse Feature Circuits (Marks / Bau)**：把 SAE 特征 + linear attribution 拼成"可解释因果子图"，并提出 **SHIFT** 方法用人类判别让分类器只依赖任务相关特征。**本轴最强 VZ 命中**——直接对应 R8 "内部子图作为可命名 owner" + R12 "评估只读但可指导特征级 ablation"。
- **C2-02 Scaling Monosemanticity (Anthropic)**：在生产级 Claude 3 Sonnet 上找到数千万 SAE 特征，**可控特征干预**（Golden Gate Bridge 等）。**给"R8 命名内部状态"提供大规模可行性证据**。
- **C2-03 Gemma Scope (DeepMind)**：全开源 JumpReLU SAE 套件覆盖 Gemma 2 全层。**给 VZ 提供"现成可拿来做 R4 特征级控制器输入 + R8 命名状态"的工件**。

### C2.4 ITI + RepE + Refusal Direction + Function Vectors：内部控制的 4 种几何

- **C2-04 ITI**：在少量 attention heads 沿真值方向移位激活，**TruthfulQA 32.5%→65.1%，只需几百样本**。**"控制器代码空间干预"的早期模板**。
- **C2-05 RepE**：以群体表示而非单神经元为中心，覆盖 honesty/harmlessness/power-seeking 等。**注意**：RepE 鼓励把 reading 反推回训练，与 R12 冲突，**spec 中需明确边界**。
- **C2-07 Refusal Direction**：13 个开源 chat 模型的 refusal 行为可被**单一方向**完全控制。**R8 + R10 警示——拒绝是可命名 boundary owner 的几何投影，但 ModificationGate 不能依赖 single-direction 安全**。
- **C2-06 Function Vectors (Northeastern)**：用 causal mediation 在 ICL 任务中找到"任务即紧凑向量"——**与 ETA 的控制器代码 z_t 同构**。

### C2.5 Persona Vectors + IOI：regime 几何监控 + 电路 baseline

- **C2-09 Persona Vectors (Anthropic)**：自动化管线把"性格特质"（evil/sycophancy/hallucination 倾向）抽成激活空间方向，**在训练中预测/缓解人格漂移**。**与 R14 "regime 是几何对象而非 prompt 标签"完全合拍**。
- **C2-10 IOI Circuit (Wang/Steinhardt)**：第一个端到端逆向工程"自然行为"的电路（28 个 attention heads / 7 类）+ faithfulness/completeness/minimality 三指标。**所有 circuit 工作必引用的 baseline**。

### C2.6 KV Cache Steering：无梯度注入口

- **C2-08 KV Cache Steering**：one-shot KV cache 注入触发 CoT；**比 activation steering 更稳定的"无梯度控制器"通道**。**给 R4 提供 substrate residual 之外的第二条有界注入口**。

### C2.7 跨轴桥梁

- **C2 ↔ B2**：C2-09 Persona Vectors 与 B2-06 Soul Engine 是同一思想（人格 = 几何子空间）的两种工程实现；C2 偏 monitoring，B2 偏 injection。
- **C2 ↔ C1**：C2 的 SAE / circuits / persona vectors 给 C1 N4 / Sleeper / Alignment Faking 提供"打开看清楚"的工具——**C2 是 C1 的可视化层**。
- **C2 ↔ A4**：C2-06 Function Vectors 的"任务即向量"与 A4 ETA 的 z_t 是同构概念，C2 给 z_t 提供 mech interp 工具链。
- **C2 ↔ A3**：SAE 特征作为 A3 memory 的 derived index 候选；Mamba / Titans 的 fixed-state 可被 SAE 解读。

---

## 全局观察

### 1. 49 / 100 篇"立刻反哺"VZ

不是松绑评分，是结构对齐。最浓密的反哺出现在 **A5 (8/10) / C1 (6/10) / B2 (6/10) / B1 (5/10)**——这正好是 VZ "NL/ETA + R-PE + 双轨 + ModificationGate" 主线最深的四个轴。

### 2. 14 R-ID 全覆盖

100 篇覆盖到了 R1-R15 + R-PE 共 14 条不变量。最多的是 **R3 (z_t/β_t)** 与 **R-PE**，分别被 ≥ 30 篇与 ≥ 25 篇命中；最少的是 **R15 (可回滚)**，只被 C1-01/02/08 三篇直接命中——这是 VZ 当前 spec 缺口（R15 工程化最薄弱）。

### 3. 反向证据 ≥ 6 篇

- **A1-06 DeepSeek-R1**：token 空间 RL（与 R4 边界冲突）
- **A4-10 LDSC**：LLM-token-subgoal HRL（与 R4 直接冲突）
- **B3-02 AI Co-Scientist**：generate-debate-evolve token-space deliberation（与 R4 冲突）
- **B3-04 SIMA 2**：Gemini 当 reward generator（与 R-PE 内禀哲学冲突）
- **C1-03 Darwin Gödel Machine**：全开放自修改（与 R10 分层 + 有界 + 快照可回滚冲突）
- **C2-05 Representation Engineering**：reading 反推回训练（与 R12 评估只读冲突）

> 所有反向证据都已在 candidates 中明确标注；它们提供的不是"VZ 错"的证据，而是"VZ 这条路必须走"的对照——每条反向路线都有自己的 failure mode（reward generator drift / token-subgoal grounding / spec 内部冲突 / sleeper agent persistence），而 VZ 的 R 不变量正是为了避开这些 failure mode 而设。

### 4. 中国厂内 ≥ 10 篇

A1=2 (DeepSeek-R1 / Math-Shepherd) + A2=4 (RLVR-World / EfficientZero V2 / AdaWorld + iVideoGPT) + B2=3 (Sophia / Geometry of Persona / RLFF-ESC) + B3=1 (EvoAgent) = **10 篇**。中国厂内主要贡献集中在 **A1 reasoning + A2 world models + B2 social cognition** 三轴；治理轴（C1 / C2）2024-2026 中国机构原创成果空缺。

### 5. 每轴至少 1 条反向证据

按方法论 §3 鼓励——每轴至少 1 篇路线对照。这保证 VZ 不是孤立路径，而是**在 cognitive AGI 路线图上有明确位置**：知道自己不做什么，比知道自己做什么更重要。

---

## 下一步

- [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) — 跨轴主题综述（5 大主题）
- [`11_vz_implications.md`](11_vz_implications.md) — 100 篇按 R-ID 重组 + P0/P1/P2 行动清单
- [`00_executive_summary.md`](00_executive_summary.md) — 执行摘要
