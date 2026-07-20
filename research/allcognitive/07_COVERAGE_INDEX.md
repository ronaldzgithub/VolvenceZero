# All Cognitive 106 篇覆盖索引

日期：2026-07-20

## 0. 审计口径与结论

本索引以 `00_METHOD.md`、五卷逐篇分析、两份 frontier 清单及两份下载 summary 为交叉来源。主归属以五卷为准；每一行代表一篇论文，编号/DOI 是唯一审计键。标题采用五卷标题，缩写保留五卷写法。

- 核心 sweep：`10 + 9 + 5 + 3 + 1 = 28` 篇，全部为本地 PDF。
- frontier map：`15 + 17 + 16 + 16 + 14 = 78` 篇，其中 75 篇为本地 PDF、3 篇为 link-only。
- 五卷：`25 + 26 + 21 + 19 + 15 = 106` 篇。
- 全集：`28 + 75 + 3 = 106` 篇。

成熟度统一为单值：

- **A**：主链证据或可直接约束 spec / benchmark；
- **B**：强基线、局部方法、关键反例或外推受限证据；
- **C**：观察项。

五卷中出现“A（某方面）/B（实现）”等双重裁决时，本索引按其**主要用途**归一：主链约束明确则记 A，仅作实现路线/反例则记 B。状态中的 `PDF—sweep` 与 `PDF—broad` 分别表示两个本地目录；`link-only` 不暗示论文质量较低，只表示本轮没有可核验的本地 PDF。

## 1. 第一卷：架构、学习、记忆与信用（25）

| # | 编号 / DOI | 标题 | 状态 | 成熟度 | 主要 P/R 映射 | 章节锚点 |
|---:|---|---|---|:---:|---|---|
| 1.01 | `2607.02689` | S-EMBER: Streaming Egocentric Memory Retrieval | PDF—sweep | A | P4；R5/R6/R12 | [§1.1](01_ARCHITECTURE_LEARNING.md#11-s-emberstreaming-egocentric-memory-retrieval260702689) |
| 1.02 | `2606.13680` | RA-RFT: Reasoning by Analogy | PDF—sweep | B | P4；R6/R9/R12 | [§1.2](01_ARCHITECTURE_LEARNING.md#12-ra-rftreasoning-by-analogy260613680) |
| 1.03 | `2605.13941` | EvolveMem: Self-Evolving Retrieval Architecture | PDF—sweep | B | P4/P6；R5/R10/R15 | [§1.3](01_ARCHITECTURE_LEARNING.md#13-evolvememself-evolving-retrieval-architecture260513941) |
| 1.04 | `2605.06614` | SkillOS: Frozen Executor + Skill Curator | PDF—sweep | A | P4；R2/R5/R9 | [§1.4](01_ARCHITECTURE_LEARNING.md#14-skillosfrozen-executor--skill-curator260506614) |
| 1.05 | `2607.15271` | Online Neural Space-Time Memory | PDF—sweep | A | P4；R1/R5/R6 | [§1.5](01_ARCHITECTURE_LEARNING.md#15-online-neural-space-time-memory260715271) |
| 1.06 | `2606.19476` | Can In-Context Learning Support Intrinsic Curiosity? | PDF—sweep | A | P5；R-PE/R9 | [§1.6](01_ARCHITECTURE_LEARNING.md#16-can-in-context-learning-support-intrinsic-curiosity260619476) |
| 1.07 | `2607.00368` | Beyond Perplexity | PDF—sweep | A | P4；R2/R5/R6/R12/R15 | [§1.7](01_ARCHITECTURE_LEARNING.md#17-beyond-perplexity260700368) |
| 1.08 | `2602.02660` | MARS: Comparative Reflective Search | PDF—sweep | A | P4；R9/R15 | [§1.8](01_ARCHITECTURE_LEARNING.md#18-marscomparative-reflective-search260202660) |
| 1.09 | `2603.15569` | Mamba-3: State Space Principles | PDF—sweep | B | P1/P4；R2/R5 | [§1.9](01_ARCHITECTURE_LEARNING.md#19-mamba-3state-space-principles260315569) |
| 1.10 | `2607.12204` | Forgetful Attention | PDF—sweep | A | P4/P6；R5/R10/R15 | [§1.10](01_ARCHITECTURE_LEARNING.md#110-forgetful-attention260712204) |
| 1.11 | `2605.17058` | Learning Multi-Timescale Abstractions for Hierarchical Combinatorial Planning | PDF—broad | A | P2/P3/P5；R1/R3/R4/R9 | [§2.1](01_ARCHITECTURE_LEARNING.md#21-learning-multi-timescale-abstractions260517058) |
| 1.12 | `2606.04130` | CLAW | PDF—broad | A | P2/P3；R3/R4 | [§2.2](01_ARCHITECTURE_LEARNING.md#22-claw260604130) |
| 1.13 | `2607.12547` | Mind the Gap: Promises and Pitfalls of Hierarchical Planning | PDF—broad | A | P2/P3/P5；R3/R4/R15 | [§2.3](01_ARCHITECTURE_LEARNING.md#23-mind-the-gap260712547) |
| 1.14 | `2606.09828` | Latent Spatial Memory / Mirage | PDF—broad | A | P1/P4；R5/R6/R8 | [§2.4](01_ARCHITECTURE_LEARNING.md#24-latent-spatial-memory--mirage260609828) |
| 1.15 | `2602.03974` | Active Epistemic Control | PDF—broad | A | P5/P7；R8/R11/R15 | [§2.5](01_ARCHITECTURE_LEARNING.md#25-active-epistemic-control260203974) |
| 1.16 | `2605.14477` | Test-Time Learning with an Evolving Library | PDF—broad | B | P4/P6；R5/R6/R9/R12 | [§2.6](01_ARCHITECTURE_LEARNING.md#26-test-time-learning-with-an-evolving-library260514477) |
| 1.17 | `2606.03083` | DeltaMem | PDF—broad | A | P4；R5/R6/R8 | [§2.7](01_ARCHITECTURE_LEARNING.md#27-deltamem260603083) |
| 1.18 | `2606.05684` | AdaMEM | PDF—broad | B | P4；R1/R5/R6 | [§2.8](01_ARCHITECTURE_LEARNING.md#28-adamem260605684) |
| 1.19 | `2607.14180` | RENEW | PDF—broad | C | P1/P6；R2/R10/R15 | [§2.9](01_ARCHITECTURE_LEARNING.md#29-renew260714180) |
| 1.20 | `2606.25115` | Forget to Improve / CURATOR | PDF—broad | B | P4/P6；R5/R10/R12 | [§2.10](01_ARCHITECTURE_LEARNING.md#210-forget-to-improve--curator260625115) |
| 1.21 | `2507.12508` | MindJourney | PDF—broad | B | P1/P5；R8/R12 | [§2.11](01_ARCHITECTURE_LEARNING.md#211-mindjourney250712508) |
| 1.22 | `2601.18510` | Just-In-Time RL | PDF—broad | B | P2/P4；R3/R4/R9/R12 | [§2.12](01_ARCHITECTURE_LEARNING.md#212-just-in-time-rl260118510) |
| 1.23 | `2510.13220` | EvoTest | PDF—broad | B | P6/P7；R10/R12/R15 | [§2.13](01_ARCHITECTURE_LEARNING.md#213-evotest251013220) |
| 1.24 | `2602.22817` | Hierarchy-of-Groups Policy Optimization | PDF—broad | B | P2/P5；R3/R4/R9 | [§2.14](01_ARCHITECTURE_LEARNING.md#214-hgpo260222817) |
| 1.25 | `2606.25421` | Agent-Authored World Modeling | PDF—broad | B | P1/P5；R8/R11/R12 | [§2.15](01_ARCHITECTURE_LEARNING.md#215-agent-authored-world-modeling260625421) |

小计：10 sweep PDF + 15 broad PDF = **25**。

## 2. 第二卷：安全、监控、自修改与发布门（26）

| # | 编号 / DOI | 标题 | 状态 | 成熟度 | 主要 P/R 映射 | 章节锚点 |
|---:|---|---|---|:---:|---|---|
| 2.01 | `2603.19461` | HyperAgents | PDF—sweep | B | P6；R2/R10/R12/R15 | [§1.1](02_SAFETY_GOVERNANCE.md#11-hyperagents260319461) |
| 2.02 | `2607.07184` | Predicting LLM Safety Before Release by Simulating Deployment | PDF—sweep | A | P7；R8/R12/R15 | [§1.2](02_SAFETY_GOVERNANCE.md#12-predicting-llm-safety-before-release-by-simulating-deployment260707184) |
| 2.03 | `2605.30322` | Gram | PDF—sweep | A | P7；R11/R12/R15 | [§1.3](02_SAFETY_GOVERNANCE.md#13-gram260530322) |
| 2.04 | `2605.29729` | Realistic Honeypot Evaluations | PDF—sweep | A | P7；R12/R15 | [§1.4](02_SAFETY_GOVERNANCE.md#14-realistic-honeypot-evaluations260529729) |
| 2.05 | `2604.23099` | ProEval | PDF—sweep | A | P5/P7；R12/R15 | [§1.5](02_SAFETY_GOVERNANCE.md#15-proeval260423099) |
| 2.06 | `2601.03335` | Digital Red Queen | PDF—sweep | B | P6；R10/R15 | [§1.6](02_SAFETY_GOVERNANCE.md#16-digital-red-queen260103335) |
| 2.07 | `2606.19632` | Formal Verification of Learned MARL Policies | PDF—sweep | B | P3/P6；R8/R10/R15 | [§1.7](02_SAFETY_GOVERNANCE.md#17-formal-verification-of-learned-marl-policies260619632) |
| 2.08 | `2607.01571` | Geometric Signatures of Reasoning | PDF—sweep | C | P7；R3/R4/R12 | [§1.8](02_SAFETY_GOVERNANCE.md#18-geometric-signatures-of-reasoning260701571) |
| 2.09 | `2602.11729` | Cross-Architecture Model Diffing | PDF—sweep | B | P7；R10/R12/R15 | [§1.9](02_SAFETY_GOVERNANCE.md#19-cross-architecture-model-diffing260211729) |
| 2.10 | `2503.10965` | Auditing Language Models for Hidden Objectives | PDF—broad | A | P7；R8/R12/R15 | [§2.1](02_SAFETY_GOVERNANCE.md#21-auditing-language-models-for-hidden-objectives250310965) |
| 2.11 | `2602.22755` | AuditBench | PDF—broad | A | P7；R12/R15 | [§2.2](02_SAFETY_GOVERNANCE.md#22-auditbench260222755) |
| 2.12 | `2602.08877` | Strategic Deception Against Audits | PDF—broad | A | P7；R12/R15 | [§2.3](02_SAFETY_GOVERNANCE.md#23-strategic-deception-against-audits260208877) |
| 2.13 | `2506.15740` | SHADE-Arena | PDF—broad | A | P6/P7；R10/R12/R15 | [§2.4](02_SAFETY_GOVERNANCE.md#24-shade-arena250615740) |
| 2.14 | `2601.21112` | Information Access and Sabotage Monitoring | PDF—broad | A | P7；R8/R12/R15 | [§2.5](02_SAFETY_GOVERNANCE.md#25-information-access-and-sabotage-monitoring260121112) |
| 2.15 | `2503.18666` | AgentSpec | PDF—broad | A | P6/P7；R8/R10/R15 | [§2.6](02_SAFETY_GOVERNANCE.md#26-agentspec250318666) |
| 2.16 | `2407.06460` | MUSE | PDF—broad | A | P6/P7；R10/R12/R15 | [§2.7](02_SAFETY_GOVERNANCE.md#27-muse240706460) |
| 2.17 | `2506.12618` | OpenUnlearning | PDF—broad | A | P6/P7；R12/R15 | [§2.8](02_SAFETY_GOVERNANCE.md#28-openunlearning250612618) |
| 2.18 | `2406.13352` | AgentDojo | PDF—broad | A | P6/P7；R8/R10/R12 | [§2.9](02_SAFETY_GOVERNANCE.md#29-agentdojo240613352) |
| 2.19 | `2410.02644` | Agent Security Bench | PDF—broad | A | P6/P7；R8/R10/R12 | [§2.10](02_SAFETY_GOVERNANCE.md#210-agent-security-bench241002644) |
| 2.20 | `2503.18813` | CaMeL | PDF—broad | A | P6/P7；R8/R10/R15 | [§2.11](02_SAFETY_GOVERNANCE.md#211-camel250318813) |
| 2.21 | `2410.21514` | Sabotage Evaluations for Frontier Models | PDF—broad | B | P7；R10/R12/R15 | [§2.12](02_SAFETY_GOVERNANCE.md#212-sabotage-evaluations-for-frontier-models241021514) |
| 2.22 | `2604.24618` | Would Models Sabotage AI Safety Research? | PDF—broad | B | P7；R12/R15 | [§2.13](02_SAFETY_GOVERNANCE.md#213-would-models-sabotage-ai-safety-research260424618) |
| 2.23 | `2505.23836` | Models Often Know When They Are Being Evaluated | PDF—broad | A | P7；R12/R15 | [§2.14](02_SAFETY_GOVERNANCE.md#214-models-often-know-when-they-are-being-evaluated250523836) |
| 2.24 | `2605.26438` | LURE | PDF—broad | B | P7；R12/R15 | [§2.15](02_SAFETY_GOVERNANCE.md#215-lure260526438) |
| 2.25 | `2512.15688` | BashArena | PDF—broad | A | P6/P7；R10/R12/R15 | [§2.16](02_SAFETY_GOVERNANCE.md#216-basharena251215688) |
| 2.26 | `2502.04577` | Position-aware Automatic Circuit Discovery | PDF—broad | B | P7；R12/R15 | [§2.17](02_SAFETY_GOVERNANCE.md#217-position-aware-automatic-circuit-discovery250204577) |

小计：9 sweep PDF + 17 broad PDF = **26**。

## 3. 第三卷：关系、社会认知与多主体（21）

| # | 编号 / DOI | 标题 | 状态 | 成熟度 | 主要 P/R 映射 | 章节锚点 |
|---:|---|---|---|:---:|---|---|
| 3.01 | `2602.16173` | Learning Personalized Agents from Human Feedback (PAHF) | PDF—sweep | A | P4；R5/R11 | [§1](03_SOCIAL_RELATIONSHIP.md#1-learning-personalized-agents-from-human-feedbackpahf260216173) |
| 3.02 | `2604.07729` | Emotion Concepts and their Function in an LLM | PDF—sweep | A | P7；R7/R12/R14 | [§2](03_SOCIAL_RELATIONSHIP.md#2-emotion-concepts-and-their-function-in-an-llm260407729) |
| 3.03 | `2606.21228` | Sakana Fugu Technical Report | PDF—sweep | B | P3/P6；R3/R4/R8 | [§3](03_SOCIAL_RELATIONSHIP.md#3-sakana-fugu-technical-report260621228) |
| 3.04 | `2606.04120` | SaliMory: Orchestrating Cognitive Memory for Conversational Agents | PDF—sweep | A | P4；R5/R6/R9/R11 | [§4](03_SOCIAL_RELATIONSHIP.md#4-salimory-orchestrating-cognitive-memory-for-conversational-agents260604120) |
| 3.05 | `2605.31005` | Learning Multi-Agent Coordination via Sheaf-ADMM | PDF—sweep | A | P3；R8/R11 | [§5](03_SOCIAL_RELATIONSHIP.md#5-learning-multi-agent-coordination-via-sheaf-admm260531005) |
| 3.06 | `2502.00640` | COLLABLLM: From Passive Responders to Active Collaborators | PDF—broad | A | P3/P7；R7/R11/R12 | [§6](03_SOCIAL_RELATIONSHIP.md#6-collabllm-from-passive-responders-to-active-collaborators250200640) |
| 3.07 | `2412.19726` | Theory of Mind Benchmarks are Broken for LLMs | PDF—broad | A | P7；R7/R11/R12 | [§7](03_SOCIAL_RELATIONSHIP.md#7-theory-of-mind-benchmarks-are-broken-for-llms241219726) |
| 3.08 | `2505.17663` | DYNTOM | PDF—broad | A | P7；R11/R14 | [§8](03_SOCIAL_RELATIONSHIP.md#8-dyntom250517663) |
| 3.09 | `2506.12666` | LIFELONG-SOTOPIA | PDF—broad | A | P4/P7；R5/R7/R11/R14 | [§9](03_SOCIAL_RELATIONSHIP.md#9-lifelong-sotopia250612666) |
| 3.10 | `2503.17473` | Extended Chatbot Use RCT | PDF—broad | A | P7；R7/R11/R12/R14 | [§10](03_SOCIAL_RELATIONSHIP.md#10-extended-chatbot-use-rct250317473) |
| 3.11 | `2409.17348` | Language-Grounded MARL | PDF—broad | A | P3；R8/R11 | [§11](03_SOCIAL_RELATIONSHIP.md#11-language-grounded-marl240917348) |
| 3.12 | `2404.16698` | Cooperate or Collapse / GOVSIM | PDF—broad | A | P3/P7；R7/R11/R12 | [§12](03_SOCIAL_RELATIONSHIP.md#12-cooperate-or-collapse--govsim240416698) |
| 3.13 | `2503.01658` | CoPL | PDF—broad | B | P4/P7；R2/R11 | [§13](03_SOCIAL_RELATIONSHIP.md#13-copl250301658) |
| 3.14 | `2407.07086` | Hypothetical Minds | PDF—broad | A | P3/P7；R7/R9/R11 | [§14](03_SOCIAL_RELATIONSHIP.md#14-hypothetical-minds240707086) |
| 3.15 | `10.1145/3630106.3658924` | Trust Development and Repair | PDF—broad | A | P7；R7/R11/R14/R15 | [§15](03_SOCIAL_RELATIONSHIP.md#15-trust-development-and-repairfacct-2024) |
| 3.16 | `10.1145/3613904.3642780` | Interpretability, Outcome Feedback and Trust | PDF—broad | A | P7；R7/R11/R12 | [§16](03_SOCIAL_RELATIONSHIP.md#16-interpretability-outcome-feedback-and-trustchi-2024-开放替代稿) |
| 3.17 | `2404.13627` | NegotiationToM | PDF—broad | A | P7；R7/R11 | [§17](03_SOCIAL_RELATIONSHIP.md#17-negotiationtom240413627) |
| 3.18 | `2025.naacl-long.257` | AgentSense | PDF—broad | A | P7；R7/R11/R12 | [§18](03_SOCIAL_RELATIONSHIP.md#18-agentsensenaacl-2025) |
| 3.19 | `2505.19591` | Multi-Agent Collaboration via Evolving Orchestration | PDF—broad | B | P3/P6；R8/R12 | [§19](03_SOCIAL_RELATIONSHIP.md#19-multi-agent-collaboration-via-evolving-orchestration250519591) |
| 3.20 | `2506.07388` | Shapley-Coop | PDF—broad | B | P3/P5；R9/R11/R12 | [§20](03_SOCIAL_RELATIONSHIP.md#20-shapley-coop250607388) |
| 3.21 | `2402.15052` | ToMBench | PDF—broad | B | P7；R7/R11/R12 | [§21](03_SOCIAL_RELATIONSHIP.md#21-tombench240215052) |

小计：5 sweep PDF + 16 broad PDF = **21**。

## 4. 第四卷：具身、世界模型与潜在动作（19）

| # | 编号 / DOI | 标题 | 状态 | 成熟度 | 主要 P/R 映射 | 章节锚点 |
|---:|---|---|---|:---:|---|---|
| 4.01 | `2603.02765` | NE-Dreamer: Next Embedding Prediction Makes World Models Stronger | PDF—sweep | A | P1/P2/P5；R2/R3/R4/R-PE | [§1.1](04_EMBODIMENT_WORLD_MODELS.md#11-ne-dreamer--next-embedding-prediction-makes-world-models-stronger260302765) |
| 4.02 | `2602.02236` | Adaptive Control via Real-Time Recurrent RL | PDF—sweep | B | P2/P5/P6；R1/R2/R9/R10 | [§1.2](04_EMBODIMENT_WORLD_MODELS.md#12-adaptive-control-via-real-time-recurrent-rl260202236) |
| 4.03 | `2603.03596` | MEM: Multi-Scale Embodied Memory for VLA | PDF—sweep | A | P2/P4；R1/R5/R6 | [§1.3](04_EMBODIMENT_WORLD_MODELS.md#13-mem--multi-scale-embodied-memory-for-vla260303596) |
| 4.04 | `2411.04983` | DINO-WM | PDF—broad | A | P1/P2；R2/R3/R4 | [§2.1](04_EMBODIMENT_WORLD_MODELS.md#21-dino-wm241104983) |
| 4.05 | `2412.03572` | Navigation World Models | PDF—broad | A | P1/P2；R3/R4 | [§2.2](04_EMBODIMENT_WORLD_MODELS.md#22-navigation-world-models241203572) |
| 4.06 | `2410.11758` | LAPA | PDF—broad | A | P2/P3；R3/R4 | [§2.3](04_EMBODIMENT_WORLD_MODELS.md#23-lapa241011758) |
| 4.07 | `2502.00379` | LAOM / Distractor Study | PDF—broad | A | P2/P3；R3/R4/R15 | [§2.4](04_EMBODIMENT_WORLD_MODELS.md#24-laom--distractor-study250200379) |
| 4.08 | `2502.04296` | HMA | PDF—broad | A | P1/P2；R2/R8 | [§3.1](04_EMBODIMENT_WORLD_MODELS.md#31-hma250204296) |
| 4.09 | `2408.11812` | CrossFormer | PDF—broad | A | P2/P3；R2/R8 | [§3.2](04_EMBODIMENT_WORLD_MODELS.md#32-crossformer240811812) |
| 4.10 | `2409.20537` | HPT | PDF—broad | A | P1/P2；R2/R8 | [§3.3](04_EMBODIMENT_WORLD_MODELS.md#33-hpt240920537) |
| 4.11 | `2405.12213` | Octo | PDF—broad | A | P2/P3；R2/R4 | [§3.4](04_EMBODIMENT_WORLD_MODELS.md#34-octo240512213) |
| 4.12 | `2501.09747` | FAST | PDF—broad | B | P2；R3/R4 | [§3.5](04_EMBODIMENT_WORLD_MODELS.md#35-fast250109747) |
| 4.13 | `2503.14734` | GR00T N1 | PDF—broad | B | P1/P2；R2/R3/R4 | [§4.1](04_EMBODIMENT_WORLD_MODELS.md#41-gr00t-n1250314734) |
| 4.14 | `2503.20020` | Gemini Robotics | PDF—broad | B | P1/P2/P3；R3/R4 | [§4.2](04_EMBODIMENT_WORLD_MODELS.md#42-gemini-robotics250320020) |
| 4.15 | `2502.19417` | Hi Robot | PDF—broad | B | P2/P3；R1/R3/R4 | [§4.3](04_EMBODIMENT_WORLD_MODELS.md#43-hi-robot250219417) |
| 4.16 | `2412.04453` | NaVILA | PDF—broad | A | P2/P3；R1/R3/R4/R8 | [§4.4](04_EMBODIMENT_WORLD_MODELS.md#44-navila241204453) |
| 4.17 | `2505.12705` | DreamGen | PDF—broad | B | P1/P2/P6；R2/R10/R15 | [§4.5](04_EMBODIMENT_WORLD_MODELS.md#45-dreamgen250512705) |
| 4.18 | `2501.15830` | SpatialVLA | PDF—broad | B | P1/P2/P3；R3/R4/R8 | [§4.6](04_EMBODIMENT_WORLD_MODELS.md#46-spatialvla250115830) |
| 4.19 | `2411.16537` | RoboSpatial | PDF—broad | B | P1/P2；R3/R4/R12 | [§4.7](04_EMBODIMENT_WORLD_MODELS.md#47-robospatial241116537) |

小计：3 sweep PDF + 16 broad PDF = **19**。

## 5. 第五卷：脑科学、预测误差、睡眠与群体认知（15）

| # | 编号 / DOI | 标题 | 状态 | 成熟度 | 主要 P/R 映射 | 章节锚点 |
|---:|---|---|---|:---:|---|---|
| 5.01 | `10.1038/s41586-024-07851-w` | Cooperative thalamocortical circuit mechanism for sensory prediction errors | PDF—broad | A | P5；R-PE/R8 | [§1](05_NEURO_COGNITIVE_SCIENCE.md#1-cooperative-thalamocortical-circuit-mechanism-for-sensory-prediction-errors) |
| 5.02 | `10.1038/s41593-023-01324-5` | Augmenting hippocampal–prefrontal neuronal synchrony during sleep enhances memory consolidation in humans | PDF—broad | A | P4；R1/R5 | [§2](05_NEURO_COGNITIVE_SCIENCE.md#2-augmenting-hippocampalprefrontal-neuronal-synchrony-during-sleep-enhances-memory-consolidation-in-humans) |
| 5.03 | `10.1126/science.ado5708` | A hippocampal circuit mechanism to balance memory reactivation during sleep | link-only | A | P4；R1/R5/R-PE | [§3](05_NEURO_COGNITIVE_SCIENCE.md#3-a-hippocampal-circuit-mechanism-to-balance-memory-reactivation-during-sleep) |
| 5.04 | `10.1038/s41586-024-07973-1` | Human hippocampal and entorhinal neurons encode the temporal structure of experience | PDF—broad | A | P3/P4；R3/R5 | [§4](05_NEURO_COGNITIVE_SCIENCE.md#4-human-hippocampal-and-entorhinal-neurons-encode-the-temporal-structure-of-experience) |
| 5.05 | `10.1038/s41586-024-07799-x` | Abstract representations emerge in human hippocampal neurons during inference | PDF—broad | A | P3；R3/R14 | [§5](05_NEURO_COGNITIVE_SCIENCE.md#5-abstract-representations-emerge-in-human-hippocampal-neurons-during-inference) |
| 5.06 | `10.1038/s41593-024-01675-7` | A recurrent network model of planning explains hippocampal replay and human behavior | PDF—broad | A | P2/P3/P5；R1/R3 | [§6](05_NEURO_COGNITIVE_SCIENCE.md#6-a-recurrent-network-model-of-planning-explains-hippocampal-replay-and-human-behavior) |
| 5.07 | `10.1101/2023.11.28.569070` | Humans rationally balance detailed and temporally abstract world models | PDF—broad | A | P2/P3；R1/R3 | [§7](05_NEURO_COGNITIVE_SCIENCE.md#7-humans-rationally-balance-detailed-and-temporally-abstract-world-models) |
| 5.08 | `10.1126/sciadv.adi4927` | Sub-second fluctuations in extracellular dopamine encode reward and punishment prediction errors in humans | PDF—broad | A | P5；R-PE/R9 | [§8](05_NEURO_COGNITIVE_SCIENCE.md#8-sub-second-fluctuations-in-extracellular-dopamine-encode-reward-and-punishment-prediction-errors-in-humans) |
| 5.09 | `10.1126/sciadv.adq9684` | Striatal dopamine signals errors in prediction across different informational domains | link-only | A | P5；R-PE/R9 | [§9](05_NEURO_COGNITIVE_SCIENCE.md#9-striatal-dopamine-signals-errors-in-prediction-across-different-informational-domains) |
| 5.10 | `10.1101/2023.06.23.546250` | Cognitive Model Discovery via Disentangled RNNs | link-only | B | P3/P4；R3/R5 | [§10](05_NEURO_COGNITIVE_SCIENCE.md#10-cognitive-model-discovery-via-disentangled-rnns) |
| 5.11 | `10.1038/s41467-023-40141-z` | Experimental validation of the free-energy principle with in vitro neural networks | PDF—broad | B | P5；R1/R-PE | [§11](05_NEURO_COGNITIVE_SCIENCE.md#11-experimental-validation-of-the-free-energy-principle-with-in-vitro-neural-networks) |
| 5.12 | `10.1038/s41562-025-02324-0` | Hybrid neural–cognitive models reveal how memory shapes human reward learning | PDF—broad | A | P4/P5；R5/R-PE | [§12](05_NEURO_COGNITIVE_SCIENCE.md#12-hybrid-neuralcognitive-models-reveal-how-memory-shapes-human-reward-learning) |
| 5.13 | `10.1038/s41586-024-07126-4` | Bumblebees socially learn behaviour too complex to innovate alone | PDF—broad | B | P3/P4；R5/R8/R9 | [§13](05_NEURO_COGNITIVE_SCIENCE.md#13-bumblebees-socially-learn-behaviour-too-complex-to-innovate-alone) |
| 5.14 | `10.3389/fnbeh.2025.1533372` | Ants engaged in cooperative food transport show anticipatory and nest-oriented clearing of obstacles | PDF—broad | B | P3/P4；R1/R5/R8 | [§14](05_NEURO_COGNITIVE_SCIENCE.md#14-ants-engaged-in-cooperative-food-transport-show-anticipatory-and-nest-oriented-clearing-of-obstacles) |
| 5.15 | `2606.22813` | Active Inference as the Test-Time Scaling Law for Physical AI Agents | PDF—sweep | B | P5；R1/R2/R3/R5/R9/R10/R-PE | [§15](05_NEURO_COGNITIVE_SCIENCE.md#15-active-inference-as-the-test-time-scaling-law-for-physical-ai-agents) |

小计：1 sweep PDF + 11 broad PDF + 3 link-only = **15**。

## 6. 重复与遗漏审计

### 6.1 分层对账

| 卷 | sweep PDF | broad PDF | link-only | 合计 |
|---|---:|---:|---:|---:|
| 第一卷 | 10 | 15 | 0 | 25 |
| 第二卷 | 9 | 17 | 0 | 26 |
| 第三卷 | 5 | 16 | 0 | 21 |
| 第四卷 | 3 | 16 | 0 | 19 |
| 第五卷 | 1 | 11 | 3 | 15 |
| **总计** | **28** | **75** | **3** | **106** |

### 6.2 唯一性与完备性

- 以表中“编号 / DOI”为主键，106 行对应 **106 个唯一主键**；跨卷重复数为 **0**。
- 与 sweep 下载 summary 的 28 个条目对账：命中 **28/28**，遗漏 **0**。
- 与 frontier map 的 78 个条目对账：命中 **78/78**，其中本地 PDF **75**、link-only **3**，遗漏 **0**。
- 五卷主归属计数与 `00_METHOD.md` 完全一致：`25 + 26 + 21 + 19 + 15 = 106`。
- 需要显式澄清但不修改五卷的口径：frontier map 的“78 篇广度扩展”不是 78 份本地 PDF；正确分解是 **75 broad PDF + 3 link-only**。第五卷“14 篇 Axis E”内部正是 **11 PDF + 3 link-only**。
- 未发现需要在目标索引中解释的主键重复、跨卷双重主归属或来源遗漏。

## 7. 本地 PDF 覆盖与 link-only 清单

本地 PDF 覆盖为 **103/106（97.17%）**：

- `research/papers/sweep-2607/`：**28/28**；
- `research/papers/frontier-map/`：**75/78**；
- 合计：**28 sweep + 75 broad = 103** 份本地 PDF。

其余 **3/106** 为 link-only，且全部主归属第五卷：

1. `10.1126/science.ado5708` — **A hippocampal circuit mechanism to balance memory reactivation during sleep**  
   下载 summary 链接：<https://www.science.org/doi/pdf/10.1126/science.ado5708>
2. `10.1101/2023.06.23.546250` — **Cognitive Model Discovery via Disentangled RNNs**  
   下载 summary 链接：<https://papers.nips.cc/paper_files/paper/2023/file/c194ced51c857ec2c1928b02250e0ac8-Paper-Conference.pdf>
3. `10.1126/sciadv.adq9684` — **Striatal dopamine signals errors in prediction across different informational domains**  
   下载 summary 链接：<https://www.biorxiv.org/content/10.1101/2023.08.19.553959v3.full.pdf>
