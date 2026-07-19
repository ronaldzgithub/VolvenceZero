# Cognitive AGI Probe — 100 篇候选汇编与最终主名单

> **本文档角色**：10 轴各自候选清单（[`candidates/A1.md`](candidates/A1.md) ~ [`candidates/C2.md`](candidates/C2.md)）的**主 agent 最终 cull 记录**。包含：
> 1. 跨轴冲突解析（同一 arxiv id 出现在多个轴的处理）
> 2. 最终 100 篇主名单（10 轴 × 10 篇，按 axis × rank 排序）
> 3. 下载状态（`Y_existing` 复用 / `N_new` 新下载 / `H_html` 网页抓取）
> 4. 跳过候选总览（≥ 50 条，含原因）
>
> **方法论**：见 [`01_method_and_scoring.md`](01_method_and_scoring.md)。
>
> **不变量**：每篇 arxiv id 全局只能归一个轴；某 arxiv id 已被一个轴入选 → 另一个轴的清单中标 "→ 跨轴归 X"。

---

## 1. 跨轴冲突与解析

| arXiv | 候选轴 | 解析归属 | 替换论文 |
|---|---|---|---|
| **2501.12948** DeepSeek-R1 | A1 #6（reasoning RL+verifier） vs B3 #B3-7（self-play reasoning） | **A1**（主贡献 = RL+verifier 推理） | B3 用 **2505.03335 AbsZero** 替补 (b)+(f) 自我对弈推理 |
| **2506.13131** AlphaEvolve | B3 #B3-1（verifiable evaluator self-discovery） vs C1 #C1-04（self-improvement gating） | **B3**（主贡献 = 算法空间开放搜索，非 agent 自身参数修改） | C1 用 **2401.10020 Self-Rewarding LM** 替补 (b) self-improvement |

冲突解析后：

- **A1 final 10**：保留 DeepSeek-R1（A1 #6）。
- **B3 final 10**：移除 DeepSeek-R1，加入 AbsZero（替补 #B3-7 self-play reasoning slot）。
- **C1 final 10**：移除 AlphaEvolve（→ B3），加入 Self-Rewarding LM（替补 #C1-04 self-improvement slot）。

无其他跨轴 arxiv id 冲突。

---

## 2. 最终 100 篇主名单

> 状态列：`Y_existing` = 已在 [`research/papers/`](../papers/)（复用，不重下）；`N_new` = probe 独家，需新下载至 `research/probe/papers/<axis>/`；`H_html` = 仅在线 HTML 可达（如 transformer-circuits.pub），按下载脚本尽力抓取。
>
> 排序：每轴内按"VZ 关联度 → 影响力 → 新颖度 → 工程深度"已在各轴 candidate 文件中固定。

### A1 — Reasoning & Test-Time Compute（10 篇 / 0 复用 / 10 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| A1-01 | 2502.05171 | Scaling up Test-Time Compute with Latent Reasoning: Recurrent Depth | 2025 | N_new |
| A1-02 | 2412.06769 | Coconut — Training LLMs to Reason in Continuous Latent Space | 2024 | N_new |
| A1-03 | 2403.09629 | Quiet-STaR — Language Models Can Teach Themselves to Think Before Speaking | 2024 | N_new |
| A1-04 | 2305.20050 | Let's Verify Step by Step | 2023 | N_new |
| A1-05 | 2409.12917 | Training LMs to Self-Correct via RL (SCoRe) | 2024 | N_new |
| A1-06 | 2501.12948 | DeepSeek-R1 — Incentivizing Reasoning via RL | 2025 | N_new |
| A1-07 | 2312.08935 | Math-Shepherd — Verify and Reinforce Step-by-step Without Human Annotations | 2023 | N_new |
| A1-08 | 2408.03314 | Scaling LLM Test-Time Compute Optimally (Snell) | 2024 | N_new |
| A1-09 | 2501.19393 | s1 — Simple Test-Time Scaling | 2025 | N_new |
| A1-10 | 2203.14465 | STaR — Self-Taught Reasoner | 2022 | N_new |

### A2 — World Models & Model-Based RL（10 篇 / 3 复用 / 7 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| A2-01 | 2509.24527 | Dreamer 4 — Training Agents Inside Scalable World Models | 2025 | Y_existing (`papers/dm/`) |
| A2-02 | 2506.09985 | V-JEPA 2 — Self-Supervised Video Models for Understanding/Prediction/Planning | 2025 | N_new |
| A2-03 | 2402.15391 | Genie — Generative Interactive Environments | 2024 | Y_existing (`papers/dm/`) |
| A2-04 | 2505.13934 | RLVR-World — Training World Models with RL | 2025 | Y_existing |
| A2-05 | 1911.08265 | MuZero — Mastering Atari, Go, Chess, Shogi by Planning with a Learned Model | 2019 | N_new |
| A2-06 | 2405.12399 | DIAMOND — Diffusion for World Modeling (Atari) | 2024 | N_new |
| A2-07 | 2403.00564 | EfficientZero V2 — Mastering Discrete and Continuous Control with Limited Data | 2024 | N_new |
| A2-08 | 2503.18938 | AdaWorld — Learning Adaptable World Models with Latent Actions | 2025 | N_new |
| A2-09 | 2410.24164 | π₀ — Vision-Language-Action Flow Model for Robot Control | 2024 | N_new |
| A2-10 | 2301.08243 | I-JEPA — Self-Supervised Joint-Embedding Predictive Architecture | 2023 | N_new |

### A3 — Memory & Continual Learning（10 篇 / 6 复用 / 4 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| A3-01 | 2501.00663 | Titans — Learning to Memorize at Test Time | 2025 | Y_existing |
| A3-02 | 2504.13173 | It's All Connected (Miras) | 2025 | Y_existing |
| A3-03 | 2601.09913 | Continuum Memory Architectures (CMA) — Behavioral Definition | 2026 | N_new |
| A3-04 | 2312.00752 | Mamba — Linear-Time Sequence Modeling with Selective State Spaces | 2023 | N_new |
| A3-05 | 2502.12110 | A-Mem — Agentic Memory for LLM Agents | 2025 | Y_existing |
| A3-06 | 2502.14802 | HippoRAG 2 — From RAG to Memory | 2025 | Y_existing |
| A3-07 | 1612.00796 | Overcoming Catastrophic Forgetting (EWC) | 2016 | N_new |
| A3-08 | 2401.08623 | Wake-Sleep Consolidated Learning | 2024 | Y_existing |
| A3-09 | 1410.3916 | Memory Networks (Weston/Chopra/Bordes) | 2014 | N_new |
| A3-10 | 2509.16189 | Latent Learning — Episodic Memory Complements Parametric | 2025 | Y_existing |

### A4 — Hierarchical & Temporal Abstraction（10 篇 / 10 复用 / 0 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| A4-01 | 2512.20605 | Emergent Temporal Abstractions in Autoregressive Models (ETA) | 2025 | Y_existing |
| A4-02 | 2505.00787 | Option Keyboard — Constructing an Optimal Behavior Basis | 2025 | Y_existing (`papers/dm/`) |
| A4-03 | 2506.14045 | Discovering Temporal Structure — HRL Overview | 2025 | Y_existing (`papers/dm/`) |
| A4-04 | 2510.24988 | Change Point Detection + Option-Critic | 2025 | Y_existing |
| A4-05 | 2201.02628 | Attention Option-Critic | 2022 | Y_existing (`papers/dm/`) |
| A4-06 | 2508.17751 | MANGO — Multi-Layer Abstraction for Nested Options | 2025 | Y_existing |
| A4-07 | 2507.16473 | Variational Homomorphisms in Option-Induced Abstract MDPs | 2025 | Y_existing |
| A4-08 | 2505.12737 | OTA — Option-aware Temporally Abstracted Value | 2025 | Y_existing |
| A4-09 | 2001.00271 | Options of Interest — Interest Functions | 2020 | Y_existing (`papers/dm/`) |
| A4-10 | 2503.19007 | LDSC — LLM-Guided Semantic HRL | 2025 | Y_existing |

### A5 — Meta-Learning & In-Context Learning（10 篇 / 8 复用 / 2 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| A5-01 | 2512.24695 | Nested Learning — The Illusion of Deep Architectures | 2025 | Y_existing |
| A5-02 | 2309.05858 | Uncovering Mesa-Optimization Algorithms in Transformers | 2023 | Y_existing |
| A5-03 | 2506.05233 | MesaNet — Sequence Modeling by Locally Optimal Test-Time Training | 2025 | Y_existing |
| A5-04 | 2407.04620 | Learning to (Learn at Test Time) — RNNs with Expressive Hidden States (TTT) | 2024 | N_new |
| A5-05 | 2210.14215 | In-context Reinforcement Learning with Algorithm Distillation | 2022 | N_new |
| A5-06 | 2407.12275 | When Can Transformers Compositionally Generalize In-Context? | 2024 | Y_existing |
| A5-07 | 2312.15001 | Modular Solutions That Generalize Compositionally | 2023 | Y_existing |
| A5-08 | 2602.16490 | From Growing to Looping — Iterative Computation in LLMs | 2026 | Y_existing |
| A5-09 | 2512.08819 | Do Depth-Grown Models Overcome the Curse of Depth? | 2025 | Y_existing |
| A5-10 | 2304.06729 | Meta-Learned Models of Cognition | 2023 | Y_existing (`papers/dm/`) |

### B1 — Active Inference & Predictive Coding（10 篇 / 6 复用 / 4 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| B1-01 | 2604.18701 | Curiosity-Critic — Cumulative Prediction Error Improvement | 2026 | Y_existing |
| B1-02 | 2507.16598 | Depression as Disorder of Distributional Coding | 2025 | Y_existing (`papers/dm/`) |
| B1-03 | 2511.22226 | Embedded Universal Predictive Intelligence | 2025 | Y_existing |
| B1-04 | 2107.12979 | Predictive Coding — Theoretical and Experimental Review (Millidge/Seth/Buckley) | 2021 | N_new |
| B1-05 | 2602.23681 | ODAR — Active Inference Routing | 2026 | Y_existing |
| B1-06 | 2506.06725 | WorldLLM — Curiosity-Driven Theory-Making | 2025 | Y_existing |
| B1-07 | 2412.10425 | Active Inference for Self-Organizing Multi-LLM Systems | 2024 | Y_existing |
| B1-08 | 2006.04182 | Predictive Coding Approximates Backprop along Arbitrary Computation Graphs | 2020 | N_new |
| B1-09 | 1705.05363 | Curiosity-driven Exploration by Self-supervised Prediction (ICM) | 2017 | N_new |
| B1-10 | 1810.12894 | Exploration by Random Network Distillation (RND) | 2018 | N_new |

### B2 — Theory of Mind & Social Cognition（10 篇 / 8 复用 / 2 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| B2-01 | 2512.18202 | Sophia — Persistent Agent Framework of Artificial Life | 2025 | Y_existing |
| B2-02 | 2506.03543 | CogniPair — GNWT-Based Multi-Agent Digital Twins | 2025 | Y_existing |
| B2-03 | 2502.14171 | ToM-aligned Conversational Agents (BDI) | 2025 | Y_existing |
| B2-04 | 2502.11881 | ThoughtTracing — Hypothesis-Driven ToM | 2025 | Y_existing |
| B2-05 | 2505.14685 | Language Models Use Lookbacks to Track Beliefs | 2025 | Y_existing |
| B2-06 | 2512.07092 | Geometry of Persona — Disentangling Personality from Reasoning (Soul Engine) | 2025 | Y_existing |
| B2-07 | 2602.16301 | Multi-agent Cooperation through In-Context Co-Player Inference | 2026 | Y_existing |
| B2-08 | 2508.12935 | RLFF-ESC — Future-Oriented Rewards for Emotional Support | 2025 | Y_existing |
| B2-09 | 2210.05492 | Mastering No-Press Diplomacy via Human-Regularized RL (CICERO 系) | 2022 | N_new |
| B2-10 | 2306.15448 | BigToM — Understanding Social Reasoning in LMs | 2023 | N_new |

### B3 — Open-Ended & Curriculum Learning（10 篇 / 6 复用 / 4 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| B3-01 | 2506.13131 | AlphaEvolve — Coding Agent for Discovery（跨轴归 B3） | 2025 | Y_existing (`papers/dm/`) |
| B3-02 | 2502.18864 | Towards an AI Co-Scientist | 2025 | Y_existing (`papers/dm/`) |
| B3-03 | 2502.05907 | EvoAgent — Self-Evolving Continual World Model | 2025 | Y_existing |
| B3-04 | 2512.04797 | SIMA 2 — Generalist Embodied Agent for Virtual Worlds | 2025 | Y_existing (`papers/dm/`) |
| B3-05 | 2107.12808 | Open-Ended Learning Leads to Generally Capable Agents (XLand) | 2021 | Y_existing (`papers/dm/`) |
| B3-06 | 2502.03544 | AlphaGeometry 2 — Gold-medalist Olympiad Geometry | 2025 | N_new |
| B3-07 | 2505.03335 | Absolute Zero Reasoner（替补 DeepSeek-R1） | 2025 | N_new |
| B3-08 | 1901.01753 | POET — Paired Open-Ended Trailblazer | 2019 | N_new |
| B3-09 | 2012.02096 | PAIRED — Emergent Complexity via Unsupervised Environment Design | 2020 | N_new |
| B3-10 | 2109.00157 | Survey of Exploration Methods in RL | 2021 | Y_existing (`papers/dm/`) |

### C1 — Self-Improvement & Modification Gating（10 篇 / 4 复用 / 6 新下）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| C1-01 | 2510.04399 | Two-Gate Guardrail for Self-Modifying Agents | 2025 | Y_existing |
| C1-02 | 2510.10232 | Statistical Gödel Machine | 2025 | Y_existing |
| C1-03 | 2505.22954 | Darwin Gödel Machine | 2025 | Y_existing |
| C1-04 | 2401.10020 | Self-Rewarding Language Models（替补 AlphaEvolve） | 2024 | N_new |
| C1-05 | 2306.16803 | COCOA — Counterfactual Contribution Credit Assignment | 2023 | Y_existing |
| C1-06 | 2212.08073 | Constitutional AI — Harmlessness from AI Feedback | 2022 | N_new |
| C1-07 | 2511.18397 | Natural Emergent Misalignment from Reward Hacking (Anthropic N4) | 2025 | N_new |
| C1-08 | 2401.05566 | Sleeper Agents — Deceptive LLMs Persist Through Safety Training | 2024 | N_new |
| C1-09 | 2412.14093 | Alignment Faking in Large Language Models | 2024 | N_new |
| C1-10 | 2312.09390 | Weak-to-Strong Generalization | 2023 | N_new |

### C2 — Mechanistic Interpretability & Internal Control（10 篇 / 1 复用 / 8 新下 / 1 网页）

| # | arXiv | 标题 | 年份 | 状态 |
|---|---|---|---|---|
| C2-01 | 2403.19647 | Sparse Feature Circuits — Interpretable Causal Graphs in LMs | 2024 | N_new |
| C2-02 | (no arxiv) | Scaling Monosemanticity (Anthropic, transformer-circuits.pub) | 2024 | H_html |
| C2-03 | 2408.05147 | Gemma Scope — Open SAEs Everywhere on Gemma 2 | 2024 | N_new |
| C2-04 | 2306.03341 | Inference-Time Intervention (ITI) | 2023 | N_new |
| C2-05 | 2310.01405 | Representation Engineering | 2023 | N_new |
| C2-06 | 2310.15213 | Function Vectors in Large Language Models | 2023 | N_new |
| C2-07 | 2406.11717 | Refusal in LMs Is Mediated by a Single Direction | 2024 | N_new |
| C2-08 | 2507.08799 | KV Cache Steering for Frozen LLMs | 2025 | Y_existing |
| C2-09 | 2507.21509 | Persona Vectors — Monitoring and Controlling Character Traits | 2025 | N_new |
| C2-10 | 2211.00593 | IOI Circuit in GPT-2 small (Interpretability in the Wild) | 2022 | N_new |

---

## 3. 下载量化总览

| 轴 | 复用 (Y_existing) | 新下载 (N_new) | 网页抓取 (H_html) | 小计 |
|---|---|---|---|---|
| A1 | 0 | 10 | 0 | 10 |
| A2 | 3 | 7 | 0 | 10 |
| A3 | 6 | 4 | 0 | 10 |
| A4 | 10 | 0 | 0 | 10 |
| A5 | 8 | 2 | 0 | 10 |
| B1 | 6 | 4 | 0 | 10 |
| B2 | 8 | 2 | 0 | 10 |
| B3 | 6 | 4 | 0 | 10 |
| C1 | 4 | 6 | 0 | 10 |
| C2 | 1 | 8 | 1 | 10 |
| **合计** | **52** | **47** | **1** | **100** |

- 下载预算：47 篇 arxiv PDF + 1 篇 HTML 抓取尝试 ≈ ~250-300 MB。
- 预期失败率：≤ 5 篇（C2-02 HTML 抓取 + 个别 arxiv id 不可达），需在 [`_candidates.md` §6](#6-下载执行后补记) 末尾补记。

---

## 4. 100 篇主名单的全局自检

| 检查项 | 阈值 | 实际 | 状态 |
|---|---|---|---|
| 总数 | 100 篇 | 100 | ✓ |
| 每轴新作 ≥ 6 (2024-01 ~ 2026-05) | ≥ 6 / 轴 | A1=7 / A2=8 / A3=7 / A4=8 / A5=6 / B1=6 / B2=8 / B3=6 / C1=7 / C2=6 | ✓ 全部满足 |
| 每轴经典 ≤ 4 | ≤ 4 / 轴 | A1=3 / A2=2 / A3=3 / A4=2 / A5=4 / B1=4 / B2=2 / B3=4 / C1=3 / C2=4 | ✓ 全部满足 |
| 单一机构跨轴贡献 | ≤ 4 / 轴 | DeepMind 在 A2(3)+A4(4)+A5(1)+B3(4)+B1(0)+C2(1)+B2(0) 各轴均 ≤ 4 | ✓ |
| 单一机构 Anthropic | ≤ 4 / 轴 | C1=4（顶到上限）+ C2=2 + 其他 ≤ 1 | ✓ |
| 中国厂内（DeepSeek/Tsinghua/Westlake/Tencent/SJTU/Cap Med 等）总计 | ≥ 8 跨轴 | A1=2（DeepSeek-R1 / Math-Shepherd） + A2=4（RLVR-World / EfficientZero V2 / AdaWorld /+1） + A3=0 + A4=0 + A5=0 + B1=0 + B2=3（Sophia / Geometry of Persona / RLFF-ESC） + B3=1（EvoAgent） + C1=0 + C2=0 = **10 篇** | ✓ |
| VZ 关联度 = 5 的"立刻反哺"候选 | ≥ 1 / 轴 | A1=4 / A2=4 / A3=4 / A4=4 / A5=8 / B1=5 / B2=6 / B3=4 / C1=6 / C2=4 = 共 49 篇 | ✓ 极宽裕 |
| 含至少一条**反向证据/挑战路线** | ≥ 1 / 轴（鼓励） | A4-10 LDSC / B3-04 SIMA 2 / B3-02 AI Co-Scientist / C1-03 DGM / C2-05 RepE 等 | ✓ |
| 子领域覆盖 | 各轴 ≥ 子领域定义 | 见各 candidates/<axis>.md "子领域分布"表 | ✓ |

---

## 5. 跳过候选总览（≥ 50 条）

> 各轴跳过候选已在 [`candidates/<axis>.md`](candidates/) 内含理由；此处只汇总 + 索引。
>
> **统计**：跨 10 轴累计跳过 / 备选 / 跨轴归他轴的候选 ≈ 150+ 条；下方挑选与"为什么没入 final 100"最有价值的 50+ 条。

### 5.1 token 空间路线（与 R4 冲突而被排除）

| 候选 | arXiv | 来源轴 | 跳过原因 |
|---|---|---|---|
| Tree of Thoughts | 2305.10601 | A1 #11 | token 空间搜索违反 R4 控制器隔离 |
| Graph of Thoughts | 2308.09687 | A1 #13 | ToT 图扩展，工程化贡献为主 |
| Reflexion | 2303.11366 | A1 #14 / C1 跳过 | verbal critique 留 token 表达层 |
| Self-Consistency | 2203.11171 | A1 #15 | 已被 R1 + PRM 整合 subsume |
| Self-Refine | 2303.17651 | A1 #16 / C1 跳过 | token 层 critique，被 SCoRe 内部 RL 上位 |
| Self-Discover | 2402.03620 | B3 跳过 | 纯 token-space prompt 自组合 |
| Promptbreeder | 2309.16797 | B3 跳过 | DeepMind token-space prompt 进化，仅在 prompt 层 |
| Eureka | 2310.12931 | B3 跳过 | NVIDIA reward-via-coding-LLM，与 R-PE 冲突 |

### 5.2 端到端 fine-tune 大模型（与 R2 冲突）

| 候选 | arXiv | 来源轴 | 跳过原因 |
|---|---|---|---|
| GPT-5 System Card | 2601.03267 | C1 跳过（已在 N2） | 工程报告 |
| HyperNetworks | 1609.09106 | A5 跳过 | 动态生成全部权重，与 frozen-substrate 冲突 |
| GAIA-1 (Wayve 9B) | 2309.17080 | A2 #19 | 单体 9B 自动驾驶 WM，path 与 R2 冲突 |
| RT-2 | 2307.15818 | A2 跳过 | 闭源 Google 大型模型，与 OpenVLA 重叠 |
| PaLM-E | 2303.03378 | A2 跳过 | 562B 单体大模型，与 R2 冲突 |
| Cosmos World Foundation | 2501.03575 | A2 #15 | 平台级开源 WFM 太重 |

### 5.3 被更强后续工作完全覆盖

| 候选 | arXiv | 来源轴 | 跳过原因 |
|---|---|---|---|
| MAML | 1703.03400 | A5 跳过 | explicit-grad-meta 路线，与 R2/R4 不同；被综述 #10 覆盖 |
| Reptile | 1803.02999 | A5 跳过 | MAML 一阶版 |
| ANIL | 1810.09502 | A5 跳过 | MAML ablation |
| Meta-SGD | 1707.09835 | A5 跳过 | 同 MAML 族 |
| RL² | 1611.02779 | A5 跳过 | 被 Algorithm Distillation 上位 |
| PEARL | 1903.08254 | A5 跳过 | 归 A2 BAMDP 路线 |
| VariBAD | 1910.08348 | A5 跳过 | 同 A2 BAMDP |
| SNAIL | 1707.03141 | A5 跳过 | 被 Algorithm Distillation subsume |
| Akyürek ICL=GD | 2211.15661 | A5 跳过 | 被 Mesa-Optimization 在更深层覆盖 |
| Garg ICL Linear | 2208.01066 | A5 跳过 | sandbox setting 已被 mesa 覆盖 |
| Xie ICL Bayesian | 2111.02080 | A5 跳过 | toy 假设强 |
| von Oswald ICL=GD | 2206.04301 | A5 跳过 | 被同组后续 Mesa-Optimization 覆盖 |
| Test-Time Training (Sun 2020) | 1909.13231 | A5 跳过 | 被 #4 TTT 2024 系统化覆盖 |
| L2L by GD by GD | 1606.04474 | A5 跳过 | 被 Nested Learning 重塑 |
| HippoRAG v1 | 2405.14831 | A3 #20 | 被 v2 完整 subsume |
| Mamba-2 (SSD) | 2405.21060 | A3 #16 | 增量于 Mamba |
| Hyena | 2302.10866 | A3 跳过 | Mamba 主流上位 |
| GEM | (Lopez-Paz 2017) | A3 跳过 | 被 A-GEM 工程化 |
| MAS / online EWC / EWC++ | 多篇 | A3 跳过 | EWC 路线细化 |
| Towards Monosemanticity | (transformer-circuits 2023) | C2 跳过 | 被 Scaling Monosemanticity 自然延续 |
| ROME | 2202.05262 | C2 跳过 | rank-1 直接编辑权重，与 R10 冲突 |
| MEMIT | 2210.07229 | C2 跳过 | ROME 的 mass-edit 版本 |
| Path Patching | 2304.05969 | C2 跳过 | 已被 Attribution Patching + C2-01 综合超越 |
| ActAdd | 2308.10248 | C2 跳过 | 被 ITI + RepE 覆盖 |
| Steering Vectors (Subramani) | 2205.05124 | C2 跳过 | 2022 早期，被后续上位 |
| ToMNet | 1802.07740 | B2 #20 | mentalizing meta-learning 鼻祖，被 ThoughtTracing/Lookback 上位 |
| Self-Other Modeling (SOM) | 1802.09640 | B2 跳过 | 被 Co-player Inference 上位 |
| Hi-ToM | 2310.16755 | B2 跳过 | 被 FANToM 子集化 |
| ToMi | (D19-1598) | B2 跳过 | 被 BigToM/FANToM/ExploreToM 整体覆盖 |
| ESConv | 2106.01144 | B2 跳过 | 被 COMPEER 在过程奖励上上位 |
| EmpatheticDialogues | 1811.00207 | B2 跳过 | 被 ESConv 上位 |
| Tschantz RL through AI | 2002.12636 | B1 #12 | 被 ODAR amortized 形式上位 |
| Millidge VPG-AI | 1907.03876 | B1 #13 | 被 ODAR 替代 |
| NGU | 2002.06038 | B1 #17 | RND/ICM 工程化扩展 |
| Agent57 | 2003.13350 | B1 #18 | 系统/工程论文 |

### 5.4 主贡献跨轴归他轴

| 候选 | arXiv | 来源轴 | 跨轴归属 |
|---|---|---|---|
| Induction Heads (Olsson) | (transformer-circuits 2022) | A5 跳过 | 主贡献 mech interp，归 C2（但 C2 用 IOI 做 circuit 代表） |
| Function Vectors | 2310.15213 | A5 跳过 | 主贡献 mech interp → C2-06 |
| DataRater | 2505.17895 | A5 跳过 | meta-learn-on-data → C1（但 C1 已饱和） |
| AlphaEvolve | 2506.13131 | C1 → B3 | 主贡献开放搜索 → B3-01 |
| DeepSeek-R1 | 2501.12948 | B3 → A1 | 主贡献 RL+verifier reasoning → A1-06 |
| 3M-Progress (zebrafish) | 2506.00138 | B1 #11 | 神经科学 ≤ 1 篇约束让位给 #2 Distributional Coding |
| Geometry of Persona / Soul Engine | 2512.07092 | C2 跳过 | 主贡献 persona 几何 → B2-06 |

### 5.5 纯期刊无 arXiv preprint（无法下载 PDF）

| 候选 | 来源 | 来源轴 | 处理 |
|---|---|---|---|
| Friston 2010 Free Energy Principle | Nat Rev Neurosci | B1 #24 | 引用 in 综述 #4 PC Review |
| Whittington & Bogacz 2017 PCN | Neural Computation | B1 #25 | 引用 in #4 PC Review + #8 PCN-BP |
| Dabney 2020 Distributional Dopamine | Nature | B1 #20 | 神经含义被 #2 Depression 继承 |
| CLS Theory Updated | Trends in Cognitive Sciences 2016 | A3 #17 | 引用 in HippoRAG 2 / Wake-Sleep / Latent Learning |
| Spens & Burgess 2024 Generative Memory Model | Nat Hum Behav | A3 #35 | 仅在深度综述脚注 |
| AlphaProof | DeepMind blog 2024-07 | B3 跳过 | 由 AlphaGeometry 2 同期论文覆盖 |
| AlphaCode 2 | DeepMind tech report | B3 跳过 | 由 AlphaEvolve + DeepSeek-R1 覆盖 |
| Innovation Engines | 会议版 | B3 跳过 | POET 已继承 |
| Novelty Search (Lehman-Stanley 2011) | 期刊 | B3 跳过 | POET / MAP-Elites 已继承 |
| CICERO Science 2022 | Science 378 | B2 #9 | 用姊妹篇 No-Press Diplomacy 2210.05492 入口 |

### 5.6 过窄 / 工程化 / 系统类（违反 §3 排除原则）

| 候选 | arXiv | 来源轴 | 跳过原因 |
|---|---|---|---|
| GraphRAG | (Microsoft 2024) | A3 跳过 | 偏 application/serving 工程 |
| Toolformer | (Meta 2023) | A3 跳过 | 行为更接近 plan/credit |
| AssistGPT / Reflexion 系 | 多篇 | A3 跳过 | 同上 |
| LongRoPE / YaRN | 多篇 | A3 跳过 | 长上下文外推工程 |
| KV-cache RAG / RAGFlow | 多篇 | A3 跳过 | serving 工程 |
| Vid2World | 2505.14357 | A2 跳过 | autoregressive video → WM |
| TD-MPC2 | 2310.16828 | A2 跳过 | 未引入新范式 |
| CogVideoX / WorldDreamer / Pandora | 多篇 | A2 跳过 | 偏视频生成 |
| Sora as World Model | OpenAI blog | A2 跳过 | 无 arxiv，偏视频生成 |

---

## 6. 下载执行后补记

执行 [`download_probe_papers.ps1`](download_probe_papers.ps1) 实际结果（2026-05-08）：

| 类别 | 数量 | 备注 |
|---|---|---|
| 计划下载 | 47 PDF + 1 HTML | 见 §3 表 |
| 实际成功 | **47 PDF + 1 HTML = 48/48** | 100% 通过 |
| 失败 | 0 | — |

**特殊事件**：
- A1-02 Coconut (2412.06769) 第一次执行因 `Start-Job` 残留进程 race condition 导致文件锁，第二次重试（顺序执行 + 杀死残留 powershell）成功，最终 PDF 3.3 MB。
- C2-02 Scaling Monosemanticity 仅在线 HTML（transformer-circuits.pub），抓取的是完整 HTML 单文件 16 MB（含图与公式），精读卡顶部已标注"PDF 不可下，引用在线版"。
- 第一版下载脚本使用 `Start-Job` + `Invoke-WebRequest` 47 个并行 jobs 在 PowerShell 5.1 下表现极差（13 分钟仅完成 3 篇）。改为 `System.Net.Http.HttpClient` 顺序下载 + 0.5s courtesy delay 后 100 秒完成 47 篇。

**复用 / 新下载分布**（最终验收）：

| 轴 | Y_existing 复用 | N_new 新下载 | H_html 网页 | PDF/HTML 文件总数 |
|---|---|---|---|---|
| A1 | 0 | 10 | 0 | 10 |
| A2 | 3（来自 [`research/papers/`](../papers/) 与 [`research/papers/dm/`](../papers/dm/)） | 7 | 0 | 10 |
| A3 | 6 | 4 | 0 | 10 |
| A4 | 10 | 0 | 0 | 10 |
| A5 | 8 | 2 | 0 | 10 |
| B1 | 6 | 4 | 0 | 10 |
| B2 | 8 | 2 | 0 | 10 |
| B3 | 6 | 4 | 0 | 10 |
| C1 | 4 | 6 | 0 | 10 |
| C2 | 1 | 8 | 1 | 10 |
| **合计** | **52** | **47** | **1** | **100** |

复用率 52% — 与 [`research/papers/`](../papers/) ~104 篇既有储备的有机重叠，证明 BOSS 现有研究方向覆盖度高。新增 47 篇主要分布在 A1（推理）/ C1（治理） / C2（mech interp） 三个 BOSS 现有目录覆盖较弱的轴。

下载总结表见 [`_download_summary.md`](_download_summary.md)。

---

## 7. 与既有研究目录的关系

- [`research/papers/`](../papers/) 与 [`research/papers/dm/`](../papers/dm/) 共 ~104 篇 PDF：probe 主名单复用 52 篇（**50%**），与 BOSS 现有方向高度兼容。
- [`research/arxiv-survey-2026-05.md`](../arxiv-survey-2026-05.md)：probe 在其基础上系统化为 10 轴评分清单，可作为 spec 反哺的 canonical 入口。
- [`research/openai-frontier-2026/`](../openai-frontier-2026/)：N1-N8 OpenAI/Anthropic 圈 8 篇全部进入 C1（N4/N7/N8）+ C2（N6 跨指）+ A1（N2 GPT-5 跨指）；probe 不重复精读，仅在引用处标注。
- [`research/deepmind-author-paper-assessment-2026-05.md`](../deepmind-author-paper-assessment-2026-05.md)：DeepMind A-tier 10 篇全部入选（散布在 A2/A4/B3）；B-tier 8 篇大多数入选；C-tier 3 篇仅作背景引用。
- [`research/core-author-paper-assessment-2026-05.md`](../core-author-paper-assessment-2026-05.md)：NL/ETA 作者圈 8 篇全部进入 A3（NL 主线）+ A5（mesa-optimization 主线）+ A4（ETA 主线）。

---

## 8. 后续步骤

1. 执行 [`download_probe_papers.ps1`](download_probe_papers.ps1)：下载 47 篇新 PDF + 1 篇 HTML 抓取尝试。
2. 失败项补记到本文件 §6。
3. 并行 × 10 子代理写精读卡到 [`notes/<axis>/`](notes/)，按 [`01_method_and_scoring.md`](01_method_and_scoring.md) §4 模板。
4. 主 agent 写 [`02_axis_walkthrough.md`](02_axis_walkthrough.md)、[`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md)、[`11_vz_implications.md`](11_vz_implications.md)、[`00_executive_summary.md`](00_executive_summary.md)、[`README.md`](README.md)。
