# DeepMind 顶尖学者 cognitive AGI 论文延伸阅读与水平评估

调研日期：2026-05-07

调研口径：

- **范围**：Google DeepMind（含原 Google Brain 合并入 GDM 的部分）+ 关键 alumni（Bellemare 已转 Mila、Pieter Abbeel 等思想路线一致者）。**不包含** Google Research 纯学术部门（如 Behrouz/Mirrokni 团队 NL 主线），后者已在 [`core-author-paper-assessment-2026-05.md`](core-author-paper-assessment-2026-05.md) 中评估。
- **时间窗**：以 2024-2026 为主，必要时引用 2018-2023 经典文献作为基础参考。
- **聚焦三轴**（与 [`docs/next_gen_emogpt.md`](../docs/next_gen_emogpt.md) R-ID 对齐）：
  - **R3/R4** — 时间抽象 + latent z_t 内部控制（不在 token 空间做长期决策）
  - **R-PE** — 预测误差作为一级原始信号
  - **R10** — 信用分配 + 自修改门控
- **方法**：候选论文清单经一轮 web 检索 + 既有 70 篇 PDF 去重 + 三档对齐度筛选；最终产出 **A 档 10 篇详评**、**B 档 10 篇简评**（8 篇 arxiv 已下载 + 2 篇 Nature-only 引用 DOI）、**C 档 3 篇 arxiv 背景登记 + 4 篇经典文献 DOI 引用**。新下载 21 个 PDF（A 10 + B 8 + C 3）至 [`research/papers/dm/`](papers/dm/)，总约 230 MB。
- **质量记录**：本次 web 检索给出的 arxiv ID 中 5 个错位（指向无关论文如统计学/物理/化学），全部经下载-PDF 文首验证后发现并修正。两条经验：(1) **LLM 生成的 arxiv ID 必须 PDF 校验**；(2) **DeepMind 部分高影响成果（AlphaDev / AlphaTensor / Wang 2018 / Lake 2017）只发 Nature/BBS 无 arxiv 预印**，须通过 DOI 引用。

成熟度标尺（与既有评估一致）：

- **高**：顶会/期刊已接收，或理论与实验链条完整，可直接进入 spec / benchmark 设计。
- **中**：arXiv 新作但实验面较完整，适合进入技术路线或 shadow prototype。
- **低**：立意重要但仍偏 proposal / workshop / 单点实验，适合观察，不宜直接改运行时主链。

文件路径：[`research/papers/dm/`](papers/dm/)（本次新下载）；与 [`research/papers/`](papers/) 物理隔离便于对照。

---

## 一、总体结论

### 1.1 与已有 NL/ETA 调研的关系

| 调研 | 学者基础 | 主轴 | 输出 |
|---|---|---|---|
| [`arxiv-survey-2026-05.md`](arxiv-survey-2026-05.md) | 全网 | R1-R20 横扫 | 70 篇候选 |
| [`core-author-paper-assessment-2026-05.md`](core-author-paper-assessment-2026-05.md) | Google Research（Behrouz, Mirrokni）+ ETH/Sacramento 团队 | NL + ETA + 信用 | 22 篇精读评估 |
| **本调研** | **Google DeepMind + alumni** | R3/R4 + R-PE + R10 | 20 篇精+简评 + 3 篇 C 档背景 + 4 篇经典 DOI 引用 |

三份调研形成完整的"我们的设计哲学的学术外圈映射"：NL 来自 Google Research、ETA 来自 ETH+Sacramento、而 **DeepMind 主线给的是另一类东西** —— 不是学习算法本身，而是**把学习算法放进可运行的、可评估的、可治理的智能体系统**。

### 1.2 DeepMind 在 cognitive AGI 上的整体路径（与我们对照）

DeepMind 2024-2026 的核心走势可以归纳为四条主线：

1. **大世界模型 + 想象训练**（Hafner/Lillicrap 主导的 Dreamer 4、SIMA 2、Genie）：把 R3/R-PE 的"潜在空间想象 + 预测误差驱动"做到工程规模化；与我们 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md)、[`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 同源，但他们走"巨型 latent 模型 + 想象 rollout"路径，我们走"冻结基底 + 控制器层"路径。**两条路径技术不同，目标一致**。
2. **结构化时序抽象与组合 skill**（Precup 团队 + Barreto 团队 Option Keyboard）：把 options 框架升级为可组合、可发现、可迁移的 skill 库；正中我们 R3 的"不要硬编码 option，让它从数据涌现"。**这是我们当前 [`vz-temporal`](../packages/) metacontroller 最直接的对照面**。
3. **自改进与程序搜索**（AlphaEvolve、AI co-scientist、AlphaDev/AlphaTensor）：把 R10 自修改具象化为"在可验证目标下做大规模程序/算法搜索 + 严格回路"。这个路径**比我们的 ModificationGate 走得更远**，但他们的可验证目标（数学/算法正确性）天然存在，我们的目标（关系质量）必须自己定义评估面。
4. **AGI 路径的认知科学化**（Botvinick/Wang/Kurth-Nelson + Lampinen + 经典 prefrontal meta-RL 一脉）：把神经科学的多巴胺、抑郁、分布价值、prefrontal meta-RL 与 RL 主流缝合，给 R-PE 提供**最强的生物学合法性**。这是我们 [`docs/next_gen_emogpt.md`](../docs/next_gen_emogpt.md) R-PE 设计的最深远学术后盾。

我们与他们的核心差异：

- **基底姿态**：DeepMind 仍在大量训练自己的 backbone（Gemini、Dreamer、SIMA），我们 R2 严格冻结基底，只在控制器层适应。这意味着 DeepMind 论文里**任何"端到端训练"的方法对我们都要做"冻结基底版"的转写**。
- **学习目标**：DeepMind 主流 RL 任务有可验证奖励（数学/游戏/算法），我们的核心目标（关系/EQ/regime）没有可验证奖励，必须依赖 R-PE + 多家族 evaluation 间接刻画。
- **多智能体姿态**：DeepMind 多智能体偏 game theory 与 social dilemma，我们的"用户也在适应系统"是**单 user 的非对称双轨**，更接近 Sacramento 的 in-context co-player inference 但仍不同。

### 1.3 最值得立刻吸收的 3 个方向

按 ROI 排序：

1. **Option Keyboard + Precup HRL Overview** → 对应我们 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) 的"可组合 skill 库 + β_t 切换发现"。建议本月内把"successor-feature-style 组合接口"作为 `world_temporal` / `self_temporal` 控制器的设计候选记入 spec。
2. **Dreamer 4 想象训练机制** → 对应我们 [`docs/specs/multi-timescale-learning.md`](../docs/specs/multi-timescale-learning.md) 的 background-slow 反思。Dreamer 4 把 imagination rollouts 当作主训练源，我们可以把它作为 ReflectionEngine 的"背景慢循环用想象 rollout 而非纯日志重放"的设计参照。
3. **Botvinick 抑郁=分布编码异常** → 给 R-PE 多巴胺式 readout 提供首个"病理学验证窗口"。建议把"分布价值漂移"作为 [`docs/specs/lifeform-vitals.md`](../docs/specs/lifeform-vitals.md) 的内禀健康指标候选，不是行为指标。

详细借鉴见 §四。

---

## 二、按学者/团队聚类

### 2.1 Hafner / Lillicrap 团队 — Dreamer & 大世界模型

主线人物：Danijar Hafner（DeepMind，Dreamer 系列长期一作）、Timothy Lillicrap（DeepMind，从 DDPG 到 Dreamer/SIMA 的合作核心）、Jake Bruce（Genie 一作）。

代表新作：Dreamer 4（2509.24527）、Genie（2402.15391）、SIMA 2（2505.08960）。

共同主题：**在 latent imagination 空间里做策略学习**，把环境交互的样本压力转移到模型内部 rollout。这条路径与我们 R3/R4 高度共振，**但他们靠"训练巨型 world model"实现想象，我们靠"冻结基底 + 控制器侧 rollout"**。

### 2.2 Botvinick / Kurth-Nelson / Dabney 团队 — 神经科学+RL

主线人物：Matt Botvinick（DeepMind Director of Neuroscience Research）、Jane X. Wang（DeepMind，meta-RL 主力）、Zeb Kurth-Nelson（DeepMind，前 UCL）、Will Dabney（DeepMind，distributional RL）。

代表新作：Depression as a disorder of distributional coding（2507.16598）、Meta-Learned Models of Cognition（2304.06729）。背景文献：Wang et al. 2018 *Prefrontal cortex as a meta-RL system*（Nature Neuroscience）。

共同主题：**多巴胺信号 ↔ TD error ↔ distributional RL ↔ 精神病理**的缝合。给 R-PE 提供"为什么 PE 是一级信号、不是衍生信号"的深层生物学论据。

### 2.3 Precup / Klissarov / Khetarpal 团队 — Options & HRL

主线人物：Doina Precup（DeepMind Montreal + McGill）、Martin Klissarov（DeepMind Montreal）、Khimya Khetarpal（DeepMind）、Kunsh Khetarpal、Marlos C. Machado。

代表作：Discovering Temporal Structure: HRL Overview（2506.14045）、Attention Option-Critic（2201.02628）、Options of Interest（2001.00271）、A Survey of Exploration Methods in RL（2109.00157）。

共同主题：**option/skill 的发现、迁移、可读性**。Precup 是 options 框架的奠基人之一，2025 那篇 HRL overview 是当前最权威的"如何从数据里发现时序结构"综述。

### 2.4 Silver / Schaul / van Hasselt — 核心 RL

主线人物：David Silver（DeepMind 首席 RL 科学家）、Hado van Hasselt（DeepMind，DQN/Rainbow）、Tom Schaul（DeepMind，预测/价值网络）、Yunhao Tang。

代表新作：DataRater（2505.17895）。背景：Silver et al. *Era of Experience*（2024-2025）、*Reward is Enough*（2021，AIJ）。

共同主题：**奖励/数据/信用是构建一切学习的原始货币**，2024 之后他们公开转向"经验时代"（Era of Experience）—— 模型必须能从自己的经验中学习，而不是只从人类数据里。

### 2.5 Hassabis / Legg / Kavukcuoglu — AGI 路径与治理

主线人物：Demis Hassabis（CEO）、Shane Legg（Chief AGI Scientist）、Koray Kavukcuoglu（CTO）。

代表作：Levels of AGI（2311.02462）、Gemini 2.5 tech report（2507.06261）。

共同主题：**给 AGI 能力分级、可操作化路径、对外治理语言**。这条线对我们的工程价值在于"用什么语言对外/对自己描述能力进展"。

### 2.6 Barreto / Filos / Kapturowski 团队 — 后继表示与组合 skill

主线人物：André Barreto（DeepMind，successor features 系列长期主力）、Angelos Filos（DeepMind）、Steven Kapturowski（DeepMind，R2D2 等 RL infrastructure）。

代表新作：Option Keyboard（2505.00787）。背景：Barreto et al. 2017/2020 successor features 系列。

共同主题：**用 successor features / generalized policy improvement 把 skills 组织成可线性组合的"键盘"接口**。这一系是我们 R3/R4 latent skill 接口设计最直接对照。

### 2.7 AlphaCode / AlphaEvolve / 程序搜索系

主线人物：Pushmeet Kohli（DeepMind VP, Science）、Alexander Novikov、Matej Balog、Petar Veličković、Daniel Mankowitz、Bernardino Romera-Paredes。

代表新作：AlphaEvolve（2506.13131）、Towards an AI co-scientist（2502.18864）、Amplifying human in competitive programming（2411.19744）、AlphaDev（2306.08631）、AlphaTensor（2210.07401）。

共同主题：**在程序/算法/科学假设空间做大规模搜索，搭配可验证的目标和严格的评估回路**。是 R10（自修改 + 评估证据先行）最完整的工程范例。

---

## 三、按三轴交叉评估

### 3.1 R3/R4 — 时间抽象与内部控制

**关键论文**：Option Keyboard、Discovering Temporal Structure（HRL Overview）、Attention Option-Critic、Options of Interest、Dreamer 4、Genie、SIMA 2、RecurrentGemma。

DeepMind 在这一轴上**已经把 ETA 论文里的"涌现 β_t / z_t"做到了工程化**，但路径有两条：

- **结构化路径**（Precup/Barreto 系）：以 options 框架为底，显式定义 initiation set / termination / interest function，端到端学习；优点是可解释、可读、可组合；缺点是仍需先验定义结构。
- **涌现路径**（Hafner/Lillicrap 系）：训练大 world model，让结构在 latent dynamics 里自动出现；优点是 scale；缺点是不可读、控制接口弱。

**我们的位置**：当前 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) 走的是"在冻结基底之上做小型 metacontroller，让 z_t 涌现 + β_t 切换从数据里学到"，介于两者之间。**借鉴方向**：

- 从结构化路径吸收 Option Keyboard 的"linear combination of skills as control interface"
- 从涌现路径吸收 Dreamer 4 的"latent rollout 作为训练源"

### 3.2 R-PE — 预测误差作为一级信号

**关键论文**：Depression as distributional coding、Meta-Learned Models of Cognition、A Survey of Exploration Methods in RL、3M-Progress（已下载，cross-ref）。

DeepMind 在这一轴上的核心贡献是 **"PE 不是工程 trick，而是有神经科学合法性的一级量"**：

- Botvinick/Dabney 路径：dopamine = TD error，distributional dopamine = distributional value coding，抑郁 = 分布编码异常
- Lampinen / Binz / Wang 路径：用 LLM 当"认知模型仿真器"重现人类预测误差行为

**我们的位置**：[`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 已经把 PE 作为一级 owner。**借鉴方向**：

- 引入 distributional value 表示（不只是 scalar PE），让 needs / regime 触发条件可基于"分布漂移"而非"均值漂移"
- 引入 Curiosity-Critic（已在 [`arxiv-survey-2026-05.md`](arxiv-survey-2026-05.md)）+ Botvinick distributional 框架，区分 epistemic vs aleatoric

### 3.3 R10 — 信用分配与自修改

**关键论文**：AlphaEvolve、AI co-scientist、AlphaDev、AlphaTensor、Levels of AGI、DataRater、Mesa-optimization（cross-ref）。

DeepMind 在 R10 上**走得最远**，但走法是"在可验证目标 + 严格 eval 回路下做大规模程序搜索"。这种姿态对我们的 [`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md) 的启示：

- **门控的核心是可验证 eval，不是 confidence**：AlphaEvolve / AlphaDev 都依赖"程序正确性 / 算法性能"作为硬验证，不需要 RLHF。我们的 ModificationGate 必须先把 evaluation 做硬，否则任何自修改都是赌博。
- **信用必须 counterfactual + 长尾**：AlphaEvolve 的 eval 是"在大规模 hidden test 上的提升"，不是单 turn 反馈。我们对应的"长程信用 + counterfactual"在 COCOA（已下载）已有方法学准备。
- **DataRater**：把"数据 = 训练原料"上升为可学习对象。这与我们 [`docs/specs/continuum-memory.md`](../docs/specs/continuum-memory.md) 的"记忆质量 = 学习质量"思路同源。

---

## 四、对 VolvenceZero 的具体借鉴清单

### 4.1 高优先级转化（建议本月-本季度反映到 spec 或 prototype）

按预估 ROI 排序：

1. **PE 输出从 scalar 升级为 distribution**（来源：A5 Botvinick *Depression as distributional coding*）
   - **现状盲点**：当前 [`packages/vz-cognition/`](../packages/) 的 PE 输出是否包含价值分布信息？需要核查；如只输出 scalar surprise，则错失"分布塌缩 = 病理状态"这一根本健康信号。
   - **可落地动作**：在 [`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 加入"PE distributional readout"小节，定义最简实现（输出 quantile/categorical distribution 而非 mean），并在 [`docs/specs/lifeform-vitals.md`](../docs/specs/lifeform-vitals.md) 把"价值分布方差/熵/asymmetry"列为新增 vitals。
   - **预期收益**：让 R7 self-track 的"心理健康"有原则定义，不再依赖行为代理指标。

2. **z_t 接口采用 successor-feature behavior basis**（来源：A3 Option Keyboard）
   - **现状盲点**：[`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) 的 controller code z_t 接口是否定义了"组合性"？还是只是 K 个独立 controller？
   - **可落地动作**：在 spec 加入"behavior basis as z_t parameterization"段落，把 z_t 设计为 reward feature 的线性组合权重（successor feature 系），保证组合后策略至少不弱于任何基策略（GPI 性质）。
   - **预期收益**：dual-track（world/self）可共享同一 SF basis，每条轨道只学自己的 reward feature，大幅减少参数 + 增加 R8 SSOT 边界清晰度。

3. **regime trigger 用 interest function 替代 hard-coded condition**（来源：B9 Options of Interest）
   - **现状盲点**：[`docs/specs/cognitive-regime.md`](../docs/specs/cognitive-regime.md) 的 regime 激活条件是 hard-coded 还是可学习的？
   - **可落地动作**：把 regime 的 initiation condition 改为可微 interest function，端到端学习；并把 Attention Option-Critic（A9）的 degeneracy 防护机制（option domination 检测、switching frequency 上限）作为健康指标。
   - **预期收益**：R14 持久 regime 身份的 activation 决策从规则驱动变为数据驱动，与"涌现优于硬编码"原则一致。

4. **ModificationGate 必须 anchor 在 verifiable evaluation**（来源：A2 AlphaEvolve、B6 AlphaDev、B7 AlphaTensor）
   - **现状盲点**：[`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md) 的 `claim_rare_heavy_net_benefit` 是否要求 evaluation 已经能给出 counterfactual 提升信号？
   - **可落地动作**：在 spec 加入硬约束："ModificationGate 不能开启，除非 [`docs/specs/evaluation.md`](../docs/specs/evaluation.md) 多家族评估能给出可重复、可对照、可 counterfactual 的提升量"。这与 AlphaEvolve / AlphaDev / AlphaTensor 的成功路径完全一致。
   - **预期收益**：避免我们重蹈"自修改没有可验证目标 → 漂移"的坑；把 evaluation 完备性作为 R10 的前置门槛，而不是事后补丁。

5. **background-slow 反思引入 imagination rollout**（来源：A1 Dreamer 4）
   - **现状盲点**：[`packages/vz-cognition/`](../packages/) 的 ReflectionEngine 当前是否只重放历史 log？imagination 模式（基于 world model 内部 rollout 假设性场景）是否被讨论？
   - **可落地动作**：在 [`docs/specs/multi-timescale-learning.md`](../docs/specs/multi-timescale-learning.md) 的 background-slow 章节加入"imagination-based reflection"候选，参考 Dreamer 4 的 shortcut forcing objective 防止想象漂移；落地路径是用冻结基底 + 控制器内部 rollout（而非训练巨型 latent world model）。
   - **预期收益**：让反思层不只是"过去日志的总结"，而是"假设性未来的探索"——支持 R10 的 counterfactual credit 与 R-PE 的反事实 PE 估计。

6. **episodic→persistent 晋升用 meta-learned data value**（来源：A7 DataRater）
   - **现状盲点**：[`docs/specs/continuum-memory.md`](../docs/specs/continuum-memory.md) 的 episodic→persistent 晋升规则当前是 hard-coded heuristic 还是可学习的？
   - **可落地动作**：把 ReflectionEngine 的 writeback 决策（哪些 episodic 卡片晋升、哪些 forget）作为"data value estimation"问题，借鉴 DataRater 的"meta-learned filtering criterion"思路，但用 R2 兼容的非端到端 meta-gradient 路径（如 bandit / 离线 RL on memory traces）。
   - **预期收益**：让记忆质量随系统经验自动调整，而非依赖人工 heuristic。

7. **evaluation 引入"代际比较"作为开放任务的进步度量**（来源：B3 Open-Ended Learning）
   - **现状盲点**：[`docs/specs/evaluation.md`](../docs/specs/evaluation.md) 的多家族评估是否处理了"开放对话/关系任务奖励不可比"的情形？
   - **可落地动作**：加入 iterative cross-generation comparison metric——不要求每个 session 都有同质 reward，但要求"新一代 agent 在 held-out scenario 上对老一代 agent 的偏好胜率"作为长期进步度量。这与 R-PE 的"PE 减少 = 学习"是兼容的。
   - **预期收益**：解决"无单一 reward 但仍要量化进步"的根本难题。

### 4.2 观察项（不直接吸收但需追踪）

- **Mesa-optimization & MesaNet**（已下载，[`research/papers/mesa-optimization-algorithms-in-transformers-2309.05858.pdf`](papers/mesa-optimization-algorithms-in-transformers-2309.05858.pdf) 与 [`research/papers/mesanet-locally-optimal-test-time-training-2506.05233.pdf`](papers/mesanet-locally-optimal-test-time-training-2506.05233.pdf)）—— Sacramento 团队（GDM 周边）的工作，与我们 R3/R4"内部存在控制器空间"路线一致，但属外部团队，跟 DeepMind 主流 RL 路线不同步；持续追踪是否会被 GDM 主线吸收。
- **Embedded Universal Predictive Intelligence**（已下载）—— 与我们 dual-track + user_model 高度共振的理论蓝图，但 203 页且工程化程度低，作为长期理论参考。
- **AI co-scientist 的 multi-agent generate-debate-evolve**（A10）—— 内部 deliberation 多 agent 模式，对我们 [`docs/specs/thinking-loop.md`](../docs/specs/thinking-loop.md) 是有意思的 alternative，但他们在 token 空间做，违反我们 R4。追踪其是否会出现 latent-space 版本。
- **AlphaEvolve 的进化式 self-improvement**（A2）—— rare-heavy artifact 的提案产生方式 alternative，与当前 RL-based ModificationGate 是不同范式。如果未来 evaluation 完备，可考虑作为 self-modification 的 alternative search method。
- **DeepMind 后续 *Era of Experience* 落地工作**（Silver/Sutton）—— 关注 GDM 是否会把"系统从自己的经验中学习"做成具体方法学，给我们 R5/R6 与 R-PE 提供更多对照。
- **SIMA 2 的开放自学习**（B2）—— 用大模型当 task & reward generator 的范式与我们 R-PE 哲学冲突，但工程上是当前最完整的"open-ended self-improvement"案例之一。追踪其是否暴露"reward generator drift"等失败模式，给我们 R-PE 路线提供反面证据支持。

### 4.3 不借鉴（与我们路线哲学冲突的）

按冲突原因聚类：

- **冲突 R2（冻结基底）的方法**：
  - Dreamer 4（A1）的端到端训练巨型 latent world model — 我们走"冻结基底 + 控制器侧 rollout"
  - Genie（B1）的 11B foundation world model 端到端训练 — 同上
  - DataRater（A7）的端到端 meta-gradient — 我们采纳思想但不采纳实现路径
  - SIMA 2（B2）的端到端 fine-tuning Gemini base — 同上
- **冲突 R4（内部控制不在 token 空间）的方法**：
  - AI co-scientist（A10）的 token-level multi-agent debate — 我们的内部 deliberation 必须发生在 latent metacontroller 上
  - 任何"用 LLM 生成 reasoning chain 作为决策主路径"的设计
- **冲突 R-PE（PE 是原始信号）的方法**：
  - SIMA 2（B2）用 Gemini 生成 task & reward 作为 RL 信号 — 我们 reward 是 PE 的 readout，不是外部模型生成
  - 任何 RLHF / RLAIF 中"用更强模型当 reward 模型"的范式（在 substrate 训练阶段例外，运行时禁止）
- **冲突 R3（涌现优于硬编码）的方法**：
  - 任何把 option / regime / persona 写成 prompt 标签或 router rule 的做法 — 我们让它从 PE 与代际选择压力中涌现

### 4.4 落到现有 spec 的指引

> **本调研不实际改动 spec**。下面只列出"如果要落地 §4.1 的借鉴，应该改动哪些 spec、改哪部分"，作为后续 spec evolution 的参考清单。

| 借鉴项 | 涉及 spec | 改动方向（提示） |
|---|---|---|
| §4.1.1 distributional PE | [`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) + [`docs/specs/lifeform-vitals.md`](../docs/specs/lifeform-vitals.md) | PE owner 输出契约扩展为 (mean, distribution_summary)；vitals 加 distributional moments |
| §4.1.2 SF behavior basis | [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) + [`docs/specs/dual-track-learning.md`](../docs/specs/dual-track-learning.md) | z_t 参数化定义；world/self temporal owner 共享 SF basis 接口 |
| §4.1.3 interest function regime activation | [`docs/specs/cognitive-regime.md`](../docs/specs/cognitive-regime.md) + [`docs/specs/emergent-action-abstraction.md`](../docs/specs/emergent-action-abstraction.md) | regime activation 接口可微化；加 degeneracy 防护指标 |
| §4.1.4 ModificationGate eval 前置门槛 | [`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md) + [`docs/specs/evaluation.md`](../docs/specs/evaluation.md) | gate 开启硬条件加"evaluation 多家族能给出 counterfactual 信号"前置 |
| §4.1.5 imagination-based reflection | [`docs/specs/multi-timescale-learning.md`](../docs/specs/multi-timescale-learning.md) | background-slow 章节加 imagination rollout 候选；明确 R2 兼容路径 |
| §4.1.6 meta-learned data value | [`docs/specs/continuum-memory.md`](../docs/specs/continuum-memory.md) | episodic→persistent 晋升规则可学习化 |
| §4.1.7 cross-generation evaluation | [`docs/specs/evaluation.md`](../docs/specs/evaluation.md) | 加 iterative-improvement metric 族 |

具体 spec 改动应作为后续单独的 convergence packet（参考 [`.cursor/rules/cursor-convergence-workflow.mdc`](../.cursor/rules/cursor-convergence-workflow.mdc)）执行，每条改动单独评估、单独 spec 同步、单独评估证据先行（参考 [`.cursor/rules/first-principles-not-patches.mdc`](../.cursor/rules/first-principles-not-patches.mdc) §"Spec 同步协议"）。

---

## 五、附录：单篇详评

### A 档（10 篇）

#### A1. Dreamer 4: Training Agents Inside of Scalable World Models (`2509.24527`)

文件：[`research/papers/dm/dreamer4-training-agents-inside-scalable-world-models-2509.24527.pdf`](papers/dm/dreamer4-training-agents-inside-scalable-world-models-2509.24527.pdf)

- 作者线索：Danijar Hafner（一作，DeepMind SF）/ Wilson Yan / Timothy Lillicrap。Dreamer 系列正统延续。
- 核心价值：在 world model 内部用 RL 做 imagination training 的工程化里程碑。提出 **shortcut forcing objective + efficient transformer** 双件套，实现 single-GPU 实时 interactive inference + 准确预测 Minecraft 物体交互。**首个纯 offline 数据下完成 Minecraft 钻石获取的 agent**（>20000 步 mouse/keyboard action 序列）。world model 用 small amount of action data 学 general action conditioning，主要知识来自 unlabeled video。
- 对我们意义：直接对应 R3/R4/R-PE 与 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md)、[`docs/specs/multi-timescale-learning.md`](../docs/specs/multi-timescale-learning.md)。**关键启示**：(1) world model 与 policy 可解耦——他们先训 world model，再在其内部做 imagination RL，避免 PE 反向污染基底；这与我们 R2 冻结基底完全相容。(2) shortcut forcing 让 imagination rollout 不会发散——给我们 ReflectionEngine 的"想象式 background-slow 反思"提供工程参照（不要让模拟轨迹漂走）。(3) action conditioning 可从极少标签学到——支持我们 [`docs/specs/affordance.md`](../docs/specs/affordance.md) 用少量监督信号训练 affordance 表征。
- 成熟度：高。GDM 旗舰技术报告，结果 reproducible（diamond achievement 是硬指标）。
- 建议：**最高优先级精读**。把"world model 训练 + 内部 imagination RL"这一对位拆解写入 [`docs/specs/multi-timescale-learning.md`](../docs/specs/multi-timescale-learning.md) 的 background-slow 反思机制候选；不直接搬架构（巨型 latent backbone 与我们 R2 冻结路线不同）。

#### A2. AlphaEvolve: A coding agent for scientific and algorithmic discovery (`2506.13131`)

文件：[`research/papers/dm/alphaevolve-coding-agent-scientific-engineering-discovery-2506.13131.pdf`](papers/dm/alphaevolve-coding-agent-scientific-engineering-discovery-2506.13131.pdf)

- 作者线索：Alexander Novikov / Matej Balog / Pushmeet Kohli 等大型 GDM 团队，FunSearch（Romera-Paredes 2023）的工程化升级版。
- 核心价值：把 LLM-based code generation + evolutionary search + 一个或多个 evaluator 闭合成 autonomous pipeline。硬结果：(1) 4×4 复矩阵乘法 48 次标量乘法——56 年来首次超越 Strassen 算法；(2) Google 数据中心调度算法显著优化；(3) 硬件加速器电路简化（functionally equivalent）；(4) 加速 AlphaEvolve 自身的 LLM 训练。**evolutionary loop 接 verifiable evaluator 是核心**。
- 对我们意义：直接对应 R10 与 [`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md)。**关键启示**：(1) self-improvement 必须 anchor 在 verifiable eval 上——AlphaEvolve 的所有成果都来自"答案可验证"的领域（数学/算法/工程指标）；这反推我们 ModificationGate 的硬约束：**没有可验证 eval，就不能开自修改门**。(2) evolutionary search 比 RL 更适合"目标稀疏 + 解空间结构化"的搜索——给 [`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md) 的 rare-heavy artifact 提案选拔机制提供 alternative 设计点。(3) "改自己的训练 LLM"是非常激进的自指环——值得对比 Darwin/Gödel Machine（已下载）来定位风险边界。
- 成熟度：高。White paper + 公开影响（Strassen 突破已被广泛报道）。
- 建议：**最高优先级**。作为 R10 的"成功范式"参照写入 spec 背景；提示我们必须先把 evaluation 做硬（[`docs/specs/evaluation.md`](../docs/specs/evaluation.md) 的多家族评估必须能给出 counterfactual 的提升信号），否则任何自修改都是空中楼阁。

#### A3. Constructing an Optimal Behavior Basis for the Option Keyboard (`2505.00787`)

文件：[`research/papers/dm/option-keyboard-controllable-world-models-small-2505.00787.pdf`](papers/dm/option-keyboard-controllable-world-models-small-2505.00787.pdf)

- 作者线索：Lucas N. Alegre（UFRGS）/ Ana L. C. Bazzan / **André Barreto（GDM London）** / Bruno C. da Silva（UMass）。NeurIPS 2025。**注意标题与最初 subagent 给的不一样**——这不是 LLM 控制论文，而是 Barreto 系正统的 successor features + GPI 延续。
- 核心价值：在 successor features (SF) + Generalized Policy Improvement (GPI) 框架下，构造一个**可证明最优的 behavior basis**，让任何 linear-reward task 能 zero-shot 找到最优策略。比传统 Convex Coverage Set (CCS) 表达力**严格更强**，能解决某些 non-linear tasks。empirically beat SOTA on hard domains，复杂度越高优势越大。
- 对我们意义：直接对应 R3/R4 与 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md)。**关键启示**：(1) 可组合 skill 不需要 "many skills"，只需要 "right skill basis"——我们的 controller code z_t 设计应思考"basis 的最优性"而非"skill 数量"。(2) successor features 给 reward-conditional 控制提供了一种**线性可组合的接口**——非常适合作为 dual-track（world/self）的统一控制层语言：每条轨道的 reward 不同，但可共享同一组 SF basis。(3) GPI 的 "at least as good" 保证给我们的"控制器切换不能比单一控制器更糟"提供形式工具。
- 成熟度：高。NeurIPS 2025 + Barreto 体系完整。
- 建议：**最高优先级**。把 "successor-feature-style behavior basis" 作为 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) 的 z_t 接口候选写入设计讨论；同时检查我们当前 metacontroller 是否在"训练更多 skill"而非"找到最优 basis"——后者更符合 R8 SSOT 原则。

#### A4. Meta-Learned Models of Cognition (`2304.06729`)

文件：[`research/papers/dm/meta-learned-models-of-cognition-2304.06729.pdf`](papers/dm/meta-learned-models-of-cognition-2304.06729.pdf)

- 作者线索：Marcel Binz（MPI Tübingen）/ Ishita Dasgupta（DeepMind）/ Akshay Jagadish / **Matthew Botvinick（DeepMind）/ Jane X. Wang（DeepMind）** / Eric Schulz。64 页长文。
- 核心价值：把 meta-learning 与 Bayesian rational analysis of cognition 桥接：证明 meta-learning 可以构造 Bayes-optimal learning algorithms，使任何能用 Bayesian model 解释的行为现象都能用 meta-learned model 解释；并讨论 meta-learning 相对 Bayesian 的优势（可处理 Bayesian inference intractable 的情形、可纳入 limited compute 与 neuroscience 约束）。系统综述 + 研究纲领。
- 对我们意义：对 R-PE 与 R3 的**认知科学合法性最强支撑**之一，直接对应 [`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 与 [`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md)。**关键启示**：(1) meta-learner = "用一个 base agent 在大量分布上学到一个 learner"——这正是我们对 metacontroller 的定位。(2) 论文给出"如何用 meta-learning 解释 cognitive bias、heuristic、resource-rational behavior"的方法学清单，给我们 evaluation 中的"行为像人"族评估提供形式语言。(3) Bayes-optimal 视角让我们能定义**最优 PE 处理器是什么样的**——任何小于 Bayes-optimal 的处理都可被解释为"resource-bounded approximation"。
- 成熟度：高。综述性长文，理论扎实。
- 建议：**精读，作为 R-PE 的认知科学外圈参考**。不直接落地实现，但应在 [`docs/next_gen_emogpt.md`](../docs/next_gen_emogpt.md) 的 R-PE 章节加引用，并启发 evaluation 中"模型行为是否符合 Bayes-rational 偏离"这一族 metric 的设计。

#### A5. Depression as a disorder of distributional coding (`2507.16598`)

文件：[`research/papers/dm/depression-as-disorder-of-distributional-coding-2507.16598.pdf`](papers/dm/depression-as-disorder-of-distributional-coding-2507.16598.pdf)

- 作者线索：**Matthew Botvinick / Zeb Kurth-Nelson / Will Dabney（全部 DeepMind）** + Timothy Muller（Oxford）。Perspective paper（不是实验论文）。
- 核心价值：把三个原本独立的研究线缝合成一个 depression 的统一计算理论：(1) **VTA dopamine 功能异常** 的病理学证据（Grace 等的 tonic vs phasic 假说）；(2) **computational psychiatry** 中"depression = 特殊形式的 RL"的工作；(3) **distributional coding** ——大脑用价值分布（而非标量均值）编码奖励的近期发现（Dabney 等的 distributional dopamine）。三者拼起来：stress → VTA 抑制 → 价值分布编码塌缩 → depressive phenomenology。
- 对我们意义：对 R-PE 与 [`docs/specs/lifeform-vitals.md`](../docs/specs/lifeform-vitals.md) 的**最深远启示**。**关键启示**：(1) **PE 不应是 scalar，应是 distribution**——这是我们当前 PE readout 设计的最大盲点：[`packages/vz-cognition/.../prediction/`](../packages/) 是否输出价值分布？还是只输出标量 surprise？(2) "病理状态 = 价值分布编码塌缩"——给 lifeform vitals 一个**有原则的 health metric**：分布的方差/熵/asymmetry，而不是行为指标（如"低活跃"）。(3) 给我们的 R7 self-track（情绪/关系健康）提供从 distributional value 角度的 readout 设计：低 depression-like state = 价值分布 well-calibrated。
- 成熟度：中（Perspective paper，待经验验证），但**作者权重极高**（distributional dopamine 的发明者就是 Dabney）。
- 建议：**最高优先级精读**。作为 [`docs/specs/lifeform-vitals.md`](../docs/specs/lifeform-vitals.md) 与 [`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 的 distributional-PE 升级触发点。建议把"PE 输出是否包含分布信息"列入下一轮 PE spec review 的硬问题。

#### A6. Levels of AGI for Operationalizing Progress on the Path to AGI (`2311.02462`)

文件：[`research/papers/dm/levels-of-agi-operationalizing-progress-2311.02462.pdf`](papers/dm/levels-of-agi-operationalizing-progress-2311.02462.pdf)

- 作者线索：Meredith Ringel Morris / Jascha Sohl-Dickstein / Noah Fiedel / Tris Wartkentin / Allan Dafoe / Aleksandra Faust / Clement Farbaret / **Shane Legg（DeepMind Chief AGI Scientist）**。Position paper / ICML 2024。
- 核心价值：提出 AGI 能力分级框架（Levels of AGI），按**深度（performance） × 广度（generality）**两轴分级（Level 0 No AI → Level 5 Superhuman），并提出 6 条"好的 AGI ontology 应满足的原则"。同时讨论 autonomy 与 deployment risk 的交叉，强调 Human-AI Interaction paradigm 选择。
- 对我们意义：与 R10、R12、R15 直接相关，对应 [`docs/specs/evaluation.md`](../docs/specs/evaluation.md) 与 [`docs/prd.md`](../docs/prd.md)。**关键启示**：(1) **能力 ≠ 自主性**——这两轴必须分开评估。我们的 [`docs/specs/credit-and-self-modification.md`](../docs/specs/credit-and-self-modification.md) ModificationGate 的"自主程度"决策应参考此框架。(2) Level 分级给我们对外（投资人/合作方）和对内（路线图）一种共同语言。(3) Legg 的"AGI 是 ML 系统的能力问题，不是是否有意识"立场与我们"产品是关系不是智力"是**有意义的对比**——他们衡量的是 capability，我们衡量的是 relationship continuity。
- 成熟度：高。已被广泛引用，对外讨论标准之一。
- 建议：**精读，作为 evaluation 与对外沟通的 reference**。建议在 [`docs/specs/evaluation.md`](../docs/specs/evaluation.md) 加入"我们刻意不追求 Level 5 generality，而是追求关系/EQ 的 reliability"的立场陈述，避免被外部话语裹挟。

#### A7. DataRater: Meta-Learned Dataset Curation (`2505.17895`)

文件：[`research/papers/dm/datarater-meta-learned-dataset-curation-2505.17895.pdf`](papers/dm/datarater-meta-learned-dataset-curation-2505.17895.pdf)

- 作者线索：Dan A. Calian / Gregory Farquhar / Iurii Kemaev / Luisa M. Zintgraf / Matteo Hessel / Jeremy Shar / **Junhyuk Oh / András György / Tom Schaul / Jeffrey Dean / Hado van Hasselt / David Silver（全部 GDM）**。GDM RL 顶配团队。
- 核心价值：提出 DataRater——通过 meta-gradients 学习"哪些 data 对训练有价值"的 meta-learner，目标是 held-out efficiency。在多 model scale + 多 dataset 上系统验证：用 DataRater 过滤数据可显著提升 compute efficiency。**关键洞察**：data 的价值应该被学出来，不是手工定 heuristic 或粗粒度 mixture。
- 对我们意义：对应 R5/R6/R10 与 [`docs/specs/continuum-memory.md`](../docs/specs/continuum-memory.md) 的"记忆质量 = 学习质量"思路。**关键启示**：(1) 我们的 ReflectionEngine 决定哪些 episodic memory 能晋升到 persistent / 哪些应该 forget——这本质是 DataRater 问题，可以借鉴 meta-gradient 思路。(2) 我们的 [`docs/specs/aac-commitment-lifecycle.md`](../docs/specs/aac-commitment-lifecycle.md) 决定哪些 commitment 值得长期跟踪——同样是"data value estimation"问题。(3) "让数据自己说出价值"是非常强的设计哲学——与我们"涌现优于硬编码"一致。
- 成熟度：高。GDM 旗舰团队 + 多 scale 实验。
- 建议：**精读**。建议把 DataRater 思想映射到 [`docs/specs/continuum-memory.md`](../docs/specs/continuum-memory.md) 的 episodic→persistent 晋升机制，作为下一阶段 ReflectionEngine 升级的设计候选。**注意**：DataRater 是端到端 meta-gradient，我们需要做"meta-gradient-free version"以保持 R2 冻结基底。

#### A8. Discovering Temporal Structure: An Overview of Hierarchical Reinforcement Learning (`2506.14045`)

文件：[`research/papers/dm/discovering-temporal-structure-hrl-overview-2506.14045.pdf`](papers/dm/discovering-temporal-structure-hrl-overview-2506.14045.pdf)

- 作者线索：Martin Klissarov（Mila/McGill）/ Akhil Bagaria / Ziyan Luo / George Konidaris / **Doina Precup（McGill + GDM Montreal）/ Marlos C. Machado**。当前 options/HRL 领域最权威的综述（5000+ 行）。
- 核心价值：从 fundamental RL challenges 角度论证 HRL 的收益（exploration、credit assignment、transfer、planning），并系统梳理"如何发现 temporal structure"的方法家族：online experience based、offline dataset based、LLM-guided。明确指出当前 HRL 的核心未解问题：**什么是 good temporal structure 本身就没有共识**。
- 对我们意义：直接对应 R3 与 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md)、[`docs/specs/emergent-action-abstraction.md`](../docs/specs/emergent-action-abstraction.md)。**关键启示**：(1) 系统梳理给我们 metacontroller 设计的**方法地图**——我们当前在哪条 lane，缺失哪些 alternatives。(2) "good structure 没有共识"是一个**坦率的 admission**——我们 R3 的 ETA 路线（让 z_t/β_t 从数据涌现）正是对此问题的一种工程化回答。(3) LLM-guided HRL 部分（Klissarov 自己的 work line）给我们 prompt 与 metacontroller 的接口提供 cautionary reference：LLM 不应承担 z_t 的角色。
- 成熟度：高。Survey paper from the field's leading authors。
- 建议：**最高优先级精读**。作为 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) 的 background reading 必读项；建议把其中"discovering temporal structure"的方法分类表纳入我们 spec 的"方法位置"小节，明示我们的选择与未选项。

#### A9. Attention Option-Critic (`2201.02628`)

文件：[`research/papers/dm/attention-option-critic-2201.02628.pdf`](papers/dm/attention-option-critic-2201.02628.pdf)

- 作者线索：Raviteja Chunduru（McGill/Mila）/ **Doina Precup（McGill/Mila/GDM）**。
- 核心价值：在 option-critic 框架上加 attention 机制，让每个 option 学会聚焦不同的 observation 子空间，从而：(1) 行为多样性增强（diverse options）；(2) 实现 state abstraction；(3) 缓解 option-critic 经典的 **degeneracy 问题**（option domination + frequent switching）；(4) 在 ALE 上 transfer 表现更好（option 更可重用 + 可解释）。
- 对我们意义：对 R3 与 [`docs/specs/emergent-action-abstraction.md`](../docs/specs/emergent-action-abstraction.md) 有具体技术启示。**关键启示**：(1) **option-critic 会塌缩**——这是我们必须知道的失败模式。如果 metacontroller 的 z_t/β_t 没有适当 inductive bias，会出现少数 option 主导 + 过频切换。(2) **attention 作为 state abstraction 工具**——给我们提供一种让 controller code z_t 自然 attend 到不同 owner snapshot 子集的设计思路。(3) 可读 + 可重用是 option 设计的核心 desiderata，不是事后属性。
- 成熟度：高（IJCAI 2022 接近）。理论方法稳定。
- 建议：精读。**特别关注 degeneracy 防护机制**——把"option domination 检测"与"switching frequency 上限"作为 [`docs/specs/emergent-action-abstraction.md`](../docs/specs/emergent-action-abstraction.md) 的健康指标候选。

#### A10. Towards an AI co-scientist (`2502.18864`)

文件：[`research/papers/dm/towards-an-ai-co-scientist-2502.18864.pdf`](papers/dm/towards-an-ai-co-scientist-2502.18864.pdf)

- 作者线索：Juraj Gottweis / Wei-Hung Weng / Tao Tu / **Pushmeet Kohli / Nenad Tomasev / Ryutaro Tanno（DeepMind）** + 大量 Google Cloud AI Research / Google Research / Stanford / Imperial College 合作者。**注意：实际上 GDM 占比中等，Google Cloud AI 与 Google Research 是主力**。
- 核心价值：基于 Gemini 2.0 的 multi-agent 系统，使用 **generate-debate-evolve** 范式做 hypothesis generation。架构亮点：(1) asynchronous task execution framework 用于 flexible compute scaling；(2) tournament evolution 做 self-improving hypothesis。生物医学场景验证：drug repurposing、novel target discovery、解释 bacterial evolution & AMR 机制；产生了**经过湿实验验证**的新药候选（AML 体外抑瘤）和肝纤维化新表观遗传靶点。
- 对我们意义：对 R10 与 R3 间接相关，但与我们核心方向（关系/EQ）距离较远。**关键启示**：(1) **multi-agent generate-debate-evolve** 是值得借鉴的内部 deliberation 模式——可以映射到我们 [`docs/specs/thinking-loop.md`](../docs/specs/thinking-loop.md) 的内部 reasoning 多轮 critique。(2) tournament 评估给"内部哪个候选 plan/response 更好"提供机制——我们的 prompt_planner 可借鉴。(3) 但**他们仍在 token 空间做 deliberation**，这违背我们 R4"内部控制不在 token 空间"的红线——**不要直接搬运**。
- 成熟度：高（已有湿实验验证），但**与我们路线哲学有冲突**（token-level multi-agent vs latent metacontroller）。
- 建议：精读，但作为**对照而非借鉴**。在 [`docs/specs/thinking-loop.md`](../docs/specs/thinking-loop.md) 的 alternative-considered 段落记入"我们没采用 generate-debate-evolve 而是 metacontroller 路径"的理由。

### B 档（8 篇 arxiv + 2 篇 Nature-only）

> **调研记录**：本档原计划 10 篇，其中 4 个 arxiv ID（subagent 检索给出）经下载验证后发现指向无关论文（统计学 / 物理 / 工业控制等）。重新检索后 SIMA 2、Open-Ended Learning 找到正确 ID 已重新下载；AlphaDev 与 AlphaTensor 仅 Nature 发表无 arxiv 公开版本，本档只引用 DOI 不下载。本批的 ID 错位是一次有用的提醒：**用 LLM 检索 arxiv ID 必须做下载校验**，否则评估对象可能完全错位。

#### B1. Genie: Generative Interactive Environments (`2402.15391`)

文件：[`research/papers/dm/genie-generative-interactive-environments-2402.15391.pdf`](papers/dm/genie-generative-interactive-environments-2402.15391.pdf)

- 作者线索：Jake Bruce / Michael Dennis / Ashley Edwards / Jack Parker-Holder / Jimmy Shi 等大型 GDM 团队 + **Tim Rocktäschel / Edward Hughes / Nando de Freitas / Satinder Singh**。
- 核心价值：11B 参数 foundation world model。从 unlabelled Internet videos 完全无监督训练，通过 latent action interface 让用户对生成的 3D 世界做 frame-by-frame 控制。可由文字、合成图像、照片、手绘 sketch 提示生成 endless variety of action-controllable virtual worlds。**核心创新是 latent action model**——不需要 ground-truth action label。
- 对我们意义：与 R3/R-PE 相关。**latent action interface** 思路与我们 R4 latent z_t 控制空间是同一类设计哲学：control should live in a learned compact latent，not in token/raw observation space。建议作为"无监督学到的 latent control 接口"的工程范例。**不直接借鉴**架构（11B world model 与我们 R2 冻结基底不同），但 latent action 的 unsupervised 学习方式值得纳入 [`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) 的设计讨论。
- 成熟度：高。GDM 大型 paper，2024 年早期发布，已被广泛引用。

#### B2. SIMA 2: A Generalist Embodied Agent for Virtual Worlds (`2512.04797`)

文件：[`research/papers/dm/sima2-generalist-embodied-agent-virtual-worlds-2512.04797.pdf`](papers/dm/sima2-generalist-embodied-agent-virtual-worlds-2512.04797.pdf)

- 作者线索：SIMA Team, GDM。Gemini 之上构建。
- 核心价值：从只能跟简单语言指令的 SIMA 1 升级为：(1) 能 reason about high-level goals；(2) 与用户对话；(3) 处理含图像的复杂指令；(4) 在 unseen environment 上 zero-shot 表现。**关键是 open-ended self-improvement**——用 Gemini 生成任务和 reward，agent 在新环境从零自学新 skill。任务成功率从 31% (SIMA 1) 提升到 65%（人类平均 71%）。
- 对我们意义：与 R10 高度相关。**Gemini-as-task-and-reward-generator** 是值得借鉴的内部 RL 信号源——但要小心：他们用大模型当 reward 模型本质是 RLHF/RLAIF 变种，与我们 R-PE "PE 才是原始信号" 哲学不同。**有意义的对比**：他们用大模型生成 reward，我们用 PE 生成需求；两者都试图回答"agent 内部如何获得训练信号"，但路径不同。
- 成熟度：高（2025-12 最新 GDM 旗舰），但**与我们路线哲学有冲突**（外部 reward generator vs 内禀 PE）。

#### B3. Open-Ended Learning Leads to Generally Capable Agents (`2107.12808`)

文件：[`research/papers/dm/open-ended-learning-generally-capable-agents-2107.12808.pdf`](papers/dm/open-ended-learning-generally-capable-agents-2107.12808.pdf)

- 作者线索：Open-Ended Learning Team, DeepMind London（Adam Stooke / Max Jaderberg / Wojciech Czarnecki 等）。XLand 环境系列代表作。
- 核心价值：在 procedurally generated 3D physical world（XLand）训练 agent 跨极大任务空间。证明 **pure RL on fixed task distribution 失败**——必须用 open-ended learning process（dynamically change training task distribution）。结果：agent 在所有 humanly solvable eval level 都能得分，zero-shot 推广到 Hide and Seek、Capture the Flag、Tag。**emergent behaviors**：trial-and-error、tool use、option switching、cooperation。
- 对我们意义：直接对应 R3 与 R-PE。**关键启示**：(1) "fixed distribution training" 失败 → 我们的 [`docs/specs/multi-timescale-learning.md`](../docs/specs/multi-timescale-learning.md) 必须保证训练数据分布是动态变化的（场景库 + lifeform-vitals 反馈 + open-dialogue session 滚动）。(2) "iterative notion of improvement between successive generations"——给我们 [`docs/specs/evaluation.md`](../docs/specs/evaluation.md) 提供方法学：当任务奖励不可比时，用"代际超越"而非"绝对分数"度量进步。
- 成熟度：高（2021 经典）。

#### B4. Gemini 2.5 Pushing the Frontier (`2507.06261`)

文件：[`research/papers/dm/gemini25-pushing-the-frontier-tech-report-2507.06261.pdf`](papers/dm/gemini25-pushing-the-frontier-tech-report-2507.06261.pdf)

- 作者线索：Gemini Team, Google（团队署名，无具体 PI 列表）。
- 核心价值：Gemini 2.5 Pro / Flash + 早期 2.0 Flash / Flash-Lite 的统一技术报告。Pro 强调 frontier coding/reasoning + 3 小时 video 处理 + thinking model + agentic workflow + native tool use + >1M token 上下文。Flash 在 compute/latency 大幅压缩。
- 对我们意义：作为**基底候选的参考资料**，与我们 R2 冻结基底直接相关。**关键实务**：(1) >1M token long context 让我们 [`docs/specs/continuum-memory.md`](../docs/specs/continuum-memory.md) 的 transient stratum 设计有了硬条件：cache 可以放更多原始上下文，episodic 提取门槛可以更宽松。(2) Native tool use 与 thinking 给我们 [`docs/specs/thinking-loop.md`](../docs/specs/thinking-loop.md) 的 inference-time compute scaling 提供工程参照。**不直接借鉴架构**，但作为基底选型评估时必须了解。
- 成熟度：高（GDM 旗舰 system card）。
- 建议：作为**基底选型背景资料**，在 [`docs/specs/core-package-boundary.md`](../docs/specs/core-package-boundary.md) 的"substrate model 候选"段落引用。

#### B5. Amplifying human performance in combinatorial competitive programming (`2411.19744`)

文件：[`research/papers/dm/amplifying-human-combinatorial-competitive-programming-2411.19744.pdf`](papers/dm/amplifying-human-combinatorial-competitive-programming-2411.19744.pdf)

- 作者线索：**Petar Veličković（GDM）/ Alex Vitvitskyi / Larisa Markeeva / Borja Ibarz / Lars Buesing / Matej Balog / Alexander Novikov（全部 GDM）**。
- 核心价值：人类写 heuristic backbone + AI（FunSearch）演化 scoring function 的 human-AI 协同范式。在 Google Hash Code 历史比赛上突破 top percentile，几次超越 top human teams。**关键在于"分工"**：人类提供 high-level 结构，AI 提供 fine-grained 优化。
- 对我们意义：与 R10 相关。**human-AI 分工模式**对我们 ModificationGate 设计有提示：高层结构（spec、R-ID、owner 边界）是人类设定的稳定边界，自修改只在控制器内部参数 / scoring / weight 级别发生。这与"前沿 LLM 自动写代码改自己"是有原则差异的。
- 成熟度：中-高（应用 + 公开比赛验证）。
- 建议：精读，作为**人机分工边界**的设计参照。提醒我们 ModificationGate 的"人定结构 + AI 优化参数"分工模式是有理论支撑的，不是保守。

#### B6. AlphaDev: Faster Sorting Algorithms Discovered using Deep RL (Nature 2023, no arxiv)

引用：Mankowitz D J et al., 2023, *Faster sorting algorithms discovered using deep reinforcement learning*, **Nature** 618:257-263. DOI: [10.1038/s41586-023-06004-9](https://doi.org/10.1038/s41586-023-06004-9). 代码：[github.com/google-deepmind/alphadev](https://github.com/google-deepmind/alphadev)。

- 作者线索：Daniel Mankowitz / Andrea Michi / Anton Zhernov / Marco Gelmi / Marco Selvi / Cosmin Paduraru / **Edouard Leurent / Shariq Iqbal / Jean-Baptiste Lespiau / Alex Ahern / Thomas Köppe / Kevin Millikin / Stephen Gaffney / Sophie Elster / Jackson Broshear / Chris Gamble / Kieran Milan / Robert Tung / Minjae Hwang / Taylan Cemgil / Mohammadamin Barekatain / Yujia Li / Amol Mandhane / Thomas Hubert / Julian Schrittwieser / Demis Hassabis / Pushmeet Kohli / Martin Riedmiller / Oriol Vinyals / David Silver**。GDM RL 全明星阵容。
- 核心价值：把汇编级 sort 算法发现表述为单人游戏（"AssemblyGame"），用 deep RL 直接生成汇编指令，reward = 算法正确性 + 实测 CPU 指令延迟。结果：**Sort3-Sort8 + VarSort3-VarSort5 算法已被纳入 LLVM 标准 C++ sort 库**——AI 发现的算法首次替换核心 library 组件。
- 对我们意义：与 A2 AlphaEvolve 同一系列，强化"R10 自修改要 anchor 在 verifiable performance metric"的核心论点。
- 成熟度：高（已落地 LLVM）。
- 建议：作为 R10 的另一个成功案例引用，不需 PDF。

#### B7. AlphaTensor: Discovering faster matrix multiplication (Nature 2022, no arxiv)

引用：Fawzi A et al., 2022, *Discovering faster matrix multiplication algorithms with reinforcement learning*, **Nature** 610:47-53. DOI: [10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4)。

- 作者线索：Alhussein Fawzi 一作 + Matej Balog / Aja Huang / Thomas Hubert / Bernardino Romera-Paredes / Mohammadamin Barekatain / Alexander Novikov / Francisco J R Ruiz / Julian Schrittwieser / Grzegorz Swirszcz / **David Silver / Demis Hassabis / Pushmeet Kohli**。
- 核心价值：基于 AlphaZero 把矩阵乘法算法发现表述为 tensor decomposition 单人游戏。结果：在多个矩阵尺寸上找到优于 SOTA 的算法；**4×4 modular arithmetic 上首次超越 Strassen 算法（1969 年以来 50 年）**。
- 对我们意义：与 B6 同源。AI 在数学发现上的另一个里程碑。引用价值在于"硬可验证目标 + 大规模 RL search 可以触达人类几十年没解决的优化问题"——这是 R10 自修改的远端 ceiling 参考。
- 成熟度：高。
- 建议：与 B6 一并作为 R10 ceiling 参照。

#### B8. RecurrentGemma: Moving Past Transformers for Efficient Open Language Models (`2404.07839`)

文件：[`research/papers/dm/recurrentgemma-moving-past-transformers-efficient-2404.07839.pdf`](papers/dm/recurrentgemma-moving-past-transformers-efficient-2404.07839.pdf)

- 作者线索：Griffin / RLHF / Gemma Teams, GDM。基于 Griffin 架构（De et al. 2024）。
- 核心价值：开源 2B/9B 语言模型，使用 Griffin 架构（linear recurrence + local attention）替代 global attention。**fixed-size state**——内存与 sequence length 解耦，长序列推理高效。性能与同尺寸 transformer-based Gemma 相当，token 训练量更少。
- 对我们意义：与 R2/R4 间接相关。**开源固定状态 backbone 候选**——如果未来基底选型需要走"固定隐状态 + 长序列流式推理"路线（更接近"持续运行的认知 agent"形态），Griffin/RecurrentGemma 是公开权重最成熟选项之一。
- 成熟度：高（开源权重 + 工程 benchmark 完整）。
- 建议：作为**基底选型 alternative**记入 [`docs/specs/core-package-boundary.md`](../docs/specs/core-package-boundary.md) 候选。**不强制现在切换**。

#### B9. Options of Interest: Temporal Abstraction with Interest Functions (`2001.00271`)

文件：[`research/papers/dm/options-of-interest-temporal-abstraction-interest-functions-2001.00271.pdf`](papers/dm/options-of-interest-temporal-abstraction-interest-functions-2001.00271.pdf)

- 作者线索：Khimya Khetarpal / Martin Klissarov / Maxime Chevalier-Boisvert / Pierre-Luc Bacon / **Doina Precup（McGill+GDM）**。AAAI 2020。
- 核心价值：把 options framework 中的 initiation set（"哪些状态可以启动这个 option"）一般化为可微的 **interest function**，端到端学习。提出 interest-option-critic 架构，在 discrete + continuous 环境验证可解释性 + 可重用性。
- 对我们意义：直接对应 R3 与 [`docs/specs/emergent-action-abstraction.md`](../docs/specs/emergent-action-abstraction.md)。**关键启示**：interest function 可以替代我们当前 hard-coded "regime trigger condition"——把"什么时候该激活某个 controller code"做成可微学习对象。这与 R14 持久 regime 身份是兼容的：interest function 可以 condition on regime state。
- 成熓度：高（AAAI 2020）。
- 建议：精读。把 **interest function as differentiable initiation set** 作为 [`docs/specs/cognitive-regime.md`](../docs/specs/cognitive-regime.md) 的 regime activation 升级候选。

#### B10. A Survey of Exploration Methods in Reinforcement Learning (`2109.00157`)

文件：[`research/papers/dm/survey-of-exploration-methods-rl-2109.00157.pdf`](papers/dm/survey-of-exploration-methods-rl-2109.00157.pdf)

- 作者线索：Susan Amin / Maziar Gomrokchi / Harsh Satija / Herke van Hoof（UvA）/ **Doina Precup（McGill+GDM）**。
- 核心价值：系统综述 exploration 方法（intrinsic motivation、count-based、curiosity-driven、Bayesian、option-based 等），把 exploration 与 representation learning、complexity 联系起来。给"PE / surprise / intrinsic reward 大家族"一个统一地图。
- 对我们意义：作为 R-PE 与 [`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 的方法地图。**关键启示**：PE / curiosity / surprise 是同一族信号的不同形式，我们的设计应明确"我们用的是哪个 variant"，避免在不同 spec 用不同术语。
- 成熍度：高（综述）。
- 建议：作为 R-PE 设计的**术语对齐参考**——梳理我们 spec 中"prediction error / surprise / curiosity / novelty / intrinsic reward"的用语是否一致。

### C 档背景登记（3 篇 arxiv + 1 篇 ID 错位 + 2 篇经典文献）

> 这一档是**应用方向**或**领域适配**的 DeepMind 重要工作，与 R3/R4/R-PE/R10 三轴对齐度较弱，仅作背景登记，不做详评。它们对我们的价值在于**了解 DeepMind 整体输出形态**，而非直接借鉴技术。

- **C1. RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation** (`2306.11706`, TMLR 2023)
  - 文件：[`research/papers/dm/robocat-self-improving-generalist-agent-robotic-manipulation-2306.11706.pdf`](papers/dm/robocat-self-improving-generalist-agent-robotic-manipulation-2306.11706.pdf)
  - 作者线索：Bousmalis / Vezzani / Rao / Devin / Lee / Bauza 等大型 GDM 团队 + Riedmiller / Hadsell / Heess 资深署名。
  - 一行评价：multi-embodiment + multi-task generalist robotic agent，self-improvement 通过 visual goal-conditioned decision transformer 在新任务上 fine-tune；与我们方向（关系/EQ）距离远，作为"DeepMind 自改进 agent 落地范例"了解即可。

- **C2. GraphCast: Learning skillful medium-range global weather forecasting** (`2212.12794`, Science 2023)
  - 文件：[`research/papers/dm/graphcast-medium-range-global-weather-forecasting-2212.12794.pdf`](papers/dm/graphcast-medium-range-global-weather-forecasting-2212.12794.pdf)
  - 作者线索：Lam / Sanchez-Gonzalez / Willson / Wirnsberger / Vinyals / Mohamed / Battaglia（GDM）。
  - 一行评价：基于 GNN 的全球中期天气预报，0.25° 全球分辨率，10 天预报 1 分钟生成，90% 验证目标超越 ECMWF HRES；与我们无直接借鉴，作为"GDM 大型科学世界模型"代表登记。

- **C3. AMIE: Towards Conversational Diagnostic AI** (`2401.05654`)
  - 文件：[`research/papers/dm/amie-towards-conversational-diagnostic-ai-2401.05654.pdf`](papers/dm/amie-towards-conversational-diagnostic-ai-2401.05654.pdf)
  - 作者线索：Tao Tu / Anil Palepu / Schaekermann / Tanno / Tomasev / Karthikesalingam / Vivek Natarajan，**主体是 Google Research，GDM 占少数**。
  - 一行评价：LLM 诊断对话系统，self-play simulated environment + automated feedback，与 PCP 在 OSCE 风格 RCT 中比较；和我们 [`docs/specs/social_cognition/`](../docs/specs/social_cognition/) 系列**有间接借鉴**——self-play simulated environment 思路可参照（让两个 agent 互演患者-医生，用 automated feedback 训练对话），但 medical domain 与我们不同，不直接搬。

- **C4. ⚠ Med-PaLM** (期望 ID `2312.13120` 实际指向化学论文 NMR study，已删除错位 PDF)
  - 真实 Med-PaLM 系列 arxiv 散落多个 ID（2212.13138 等），本调研未做二次下载。背景了解参考：Singhal K et al., 2023, *Large Language Models Encode Clinical Knowledge*, Nature。与我们方向远，跳过精读。

### 附：经典文献引用（不下载，仅 DOI 引用）

- **Wang J X et al., 2018, *Prefrontal cortex as a meta-reinforcement learning system***, Nature Neuroscience 21:860-868. DOI: [10.1038/s41593-018-0147-8](https://doi.org/10.1038/s41593-018-0147-8). Botvinick 团队代表作之一，把 PFC 解释为 meta-RL 系统，是当前所有 meta-RL/contextual RL 工作的奠基性参考。**对我们价值**：[`docs/specs/temporal-abstraction.md`](../docs/specs/temporal-abstraction.md) metacontroller 的"PFC-like"类比直接来自此论文。
- **Lake B M et al., 2017, *Building machines that learn and think like people***, Behavioral and Brain Sciences 40:e253. DOI: [10.1017/S0140525X16001837](https://doi.org/10.1017/S0140525X16001837). Lake/Botvinick 等的 BBS target article，把"组成性、因果学习、直觉物理与心理"作为 AI 系统工程目标，是我们 dual-track + persistent regime 设计的认知科学远端依据。**对我们价值**：[`docs/next_gen_emogpt.md`](../docs/next_gen_emogpt.md) R7 双轨 + R14 持久 regime 的认知科学语境锚点。
- **Silver D, Singh S, Precup D, Sutton R S, 2021, *Reward is Enough***, Artificial Intelligence 299:103535. DOI: [10.1016/j.artint.2021.103535](https://doi.org/10.1016/j.artint.2021.103535). DeepMind 核心 RL 学者的立场宣言，与我们 R-PE "PE 是足够的原始信号" 形成对比：Silver 主张 reward 足够，我们主张 PE 比 reward 更原始。**有意义的分歧，不是同源**。**对我们价值**：在 [`docs/specs/prediction-error-loop.md`](../docs/specs/prediction-error-loop.md) 引用作为"我们与主流 RL 的方法学差异"立场点。
- **Silver D, Sutton R S, 2024-2025, *Welcome to the Era of Experience***（DeepMind 公开 essay/keynote）。预示 DeepMind 后续主线从 human data 转向 self-experience。我们 R5/R6 的"系统从自己的对话经验中学习"与此哲学一致。

### 附：经典文献引用（不下载，仅 DOI 引用）

- **Wang J X et al., 2018, *Prefrontal cortex as a meta-reinforcement learning system***, Nature Neuroscience 21:860-868. DOI: [10.1038/s41593-018-0147-8](https://doi.org/10.1038/s41593-018-0147-8). Botvinick 团队代表作之一，把 PFC 解释为 meta-RL 系统，是当前所有 meta-RL/contextual RL 工作的奠基性参考。
- **Lake B M et al., 2017, *Building machines that learn and think like people***, Behavioral and Brain Sciences 40:e253. DOI: [10.1017/S0140525X16001837](https://doi.org/10.1017/S0140525X16001837). Lake/Botvinick 等的 BBS target article，把"组成性、因果学习、直觉物理与心理"作为 AI 系统工程目标，是我们 dual-track + persistent regime 设计的认知科学远端依据。
- **Silver D, Singh S, Precup D, Sutton R S, 2021, *Reward is Enough***, Artificial Intelligence 299:103535. DOI: [10.1016/j.artint.2021.103535](https://doi.org/10.1016/j.artint.2021.103535). DeepMind 核心 RL 学者的立场宣言，与我们 R-PE "PE 是足够的原始信号" 形成对比：Silver 主张 reward 足够，我们主张 PE 比 reward 更原始。**有意义的分歧，不是同源**。
- **Silver D, Sutton R S, 2024-2025, *Welcome to the Era of Experience***（DeepMind 公开 essay/keynote）。预示 DeepMind 后续主线从 human data 转向 self-experience。我们 R5/R6 的"系统从自己的对话经验中学习"与此哲学一致。
