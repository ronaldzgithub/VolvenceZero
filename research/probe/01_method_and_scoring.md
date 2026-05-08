# 方法论与评分标尺

> 本目录的目标：建立一份 **cognitive AGI 全景 100 篇 canonical 主名单**，覆盖 [`research/papers/`](../papers/) 既有方向之外的 promising 子领域，并对每篇做精读卡 + 跨轴深度综述。
>
> **本文档的角色**：固定 10 轴定义、四维评分标尺、选材原则。后续 [`_candidates.md`](_candidates.md) 与 [`notes/`](notes/) 都以此为执行依据。
>
> **与既有研究的关系**：现有 [`research/papers/`](../papers/) ~104 篇围绕 NL/ETA/PE/credit/dual-track 五条主轴；本目录是 **fresh_curated** 重新精选，可与既有重叠（重叠者复用不重下），但 `probe/` 自成"广角主名单"。

---

## 1. 10 轴分类法

把 cognitive AGI 切成 **5 能力轴 + 3 基底轴 + 2 治理轴 = 10 轴**，每轴 10 篇 = 100 篇。轴的命名前缀（A/B/C）与现有 `docs/next_gen_emogpt.md` 的 R-ID 不冲突，避免歧义。

### 能力轴（5 条 — Capability axes，"做什么"）

| 编号 | 轴名 | 一句话定义 | 关键 R-ID 关联 |
|---|---|---|---|
| **A1** | Reasoning & Test-Time Compute | 推理时给基底模型分配额外计算/搜索/验证以提升能力的所有手段 | R3, R4, R-PE |
| **A2** | World Models & Model-Based RL | 显式或隐式学习"环境如何演化"并用其进行规划/想象/policy 改进 | R2, R3, R-PE |
| **A3** | Memory & Continual Learning | 保留、巩固、检索、遗忘历史经验且不发生灾难性遗忘 | R5, R6 |
| **A4** | Hierarchical & Temporal Abstraction | 学习/发现可复用的子策略、option、skill、time scale 抽象 | R1, R3, R4 |
| **A5** | Meta-Learning & In-Context Learning | 模型学会"如何学"或在前向中执行学习算法（mesa-optimization） | R1, R2, R5 |

### 基底轴（3 条 — Substrate axes，"系统是什么"）

| 编号 | 轴名 | 一句话定义 | 关键 R-ID 关联 |
|---|---|---|---|
| **B1** | Active Inference & Predictive Coding | 从 free-energy / 预测误差出发，统一感知-行动-学习的规范理论 | R-PE, R7 |
| **B2** | Theory of Mind & Social Cognition | 对他者心智状态、信念、意图、关系的建模与推理 | R7, R11 |
| **B3** | Open-Ended & Curriculum Learning | 不靠固定任务集合，靠自我对弈/自动课程持续生成新挑战 | R12, R13 |

### 治理轴（2 条 — Governance axes，"如何不出事"）

| 编号 | 轴名 | 一句话定义 | 关键 R-ID 关联 |
|---|---|---|---|
| **C1** | Self-Improvement & Modification Gating | 系统改自己的能力 + 安全门/容量上限/可回滚 | R9, R10, R15 |
| **C2** | Mechanistic Interpretability & Internal Control | 把模型内部表示打开看清楚，并基于此做有界控制 | R4, R8, R12 |

### 轴的边界判定（避免重复入选）

每篇论文如可同时归两轴，按"主贡献优先"原则归一轴；在精读卡的"对 VZ 的映射"部分附带提到次要轴。例如 *Two-Gate Guardrail* 既是 self-modification 也是 PAC theory，归 **C1**；*Mesa-Optimization in Transformers* 既是 ICL 也是 interpretability，归 **A5**。

---

## 2. 四维评分标尺（5 分制）

每篇论文按四维独立评分，**总分不做加和**（避免低维互相抵消的失真），而是要求 **任一维 ≥ 4 才能入选**。

| 维度 | 5 分定义 | 1 分定义 |
|---|---|---|
| **理论新颖度** | 提出新框架/新概念，重塑后续研究问题 | 已知技术的小规模改造或工程整合 |
| **工程深度** | 大规模实证 + 多基线 + 消融 + 复现路径清晰 | 仅小规模 toy 实验或纯理论 |
| **影响力** | 已被多篇 follow-up 大量引用 / 进入主流 baseline / 改变实践 | 引用 < 10 或仅小圈子讨论 |
| **VZ 关联度** | 直接补强或挑战 VZ 现有 R-ID（最好附"具体落点 spec"） | 与 VZ 主轴关联弱，仅作为认知参考 |

**入选阈值**（candidate → final 100）：
1. 任一维 ≥ 4；且
2. 不可同时所有维 ≤ 3；且
3. 若 VZ 关联度 = 5，可破格放宽其他维（例如某篇神经科学硬证据虽工程深度低但能给 R-PE 提供 in-vivo 背书）。

**排序优先级**（每轴内排序）：
- VZ 关联度 → 影响力 → 理论新颖度 → 工程深度。
- 这一序保证"对 VZ 最有用且已被验证"的论文排在轴的最前面。

---

## 3. 选材原则

### 时间窗与新鲜度

- **强制比例**：每轴 ≥ 6 篇是 **2024-01 ~ 2026-05** 的新作（保证捕捉 cognitive AGI 当前前沿）；至多 4 篇是 **奠基经典**（不限年份，但每篇要解释"为什么今天还要看")。
- 经典不能凑数：必须是当前研究真实在引用、真实在 build on 的源头（如 MAML 之于 meta-learning）。

### 来源覆盖

- arXiv 主队列：cs.LG、cs.AI、cs.NE、cs.CL、cs.MA、q-bio.NC、stat.ML。
- 顶会原始 proceedings：NeurIPS、ICML、ICLR、AAAI、IJCAI、CoRL（机器人）、COLM、TMLR。
- 旗舰技术报告：DeepMind Tech Reports、OpenAI System Cards、Anthropic Research、Google Research blog（仅当配套 arXiv preprint 存在时收录）。
- 神经科学 / 认知科学：Nature Neuroscience、Neuron、Trends in Cognitive Sciences（当且仅当对 cognitive AGI 有计算落地映射时收录）。

### 跨轴均衡

- 每个 LLM 顶级机构（OpenAI / Anthropic / DeepMind / Meta FAIR / Microsoft Research / Allen AI）单轴贡献 ≤ 4 篇，避免某一轴成为单机构展示。
- "中国厂内"（DeepSeek / Qwen / GLM / Baichuan / Moonshot）整体不少于 8 篇（横跨各轴），保证非美中心视角。
- 神经科学 / 认知心理学 / Predictive Processing 经典每轴 ≤ 1 篇（避免 B1 之外其他轴被神经科学论文挤占）。

### 排除原则

- **不收**纯 application 论文（医疗诊断、金融预测、特定行业 benchmark），即便方法好。
- **不收**纯系统工程论文（serving、推理优化、量化），除非揭示 cognitive 现象（如 Mamba 的状态空间 → 工作记忆解释）。
- **不收**纯理论论文（无任何实证或可验证假设），除非 VZ 关联度 = 5（如 free-energy principle 数学论文）。
- **不收**已被更强的后续工作 100% subsume 的论文（如 GPT-3 paper 已被多篇综述引用，但其本身不再是研究入口）。

---

## 4. 与 VZ 的映射规则

每张精读卡末尾必填三类标签：

### 4.1 R-ID 命中表

| R-ID | 命中点（精确到 spec 文件名） | 强度 |
|---|---|---|
| R3 | `temporal-abstraction.md` 的 z_t 接口 | **加强** / **补全** / **反向证据** |

- **加强**：给现有 R-ID 提供新实证或更紧的理论上限。
- **补全**：揭示现有 R-ID 当前没有覆盖的具体子情况。
- **反向证据**：与现有 R-ID 路线方向相反，需要 BOSS 严肃考虑（不能默默吞掉）。

### 4.2 行动建议级别

| 级别 | 含义 |
|---|---|
| **可立刻反哺** | 论文方法可不动 R 不变量直接进 spec / 实验，记入 [`11_vz_implications.md`](11_vz_implications.md) 的 P0/P1 |
| **远观背景** | 给 BOSS 当前路线提供认知坐标，但当前不直接转化 |
| **挑战路线** | 与 R 不变量冲突，BOSS 决策是否更新 R |

### 4.3 与 [`research/papers/`](../papers/) 的关系

| 状态 | 处理 |
|---|---|
| **已有 PDF** | 在精读卡顶部标 "本地 PDF: ../../papers/xxx.pdf"，下载脚本跳过 |
| **probe 独家** | 在精读卡顶部标 "本地 PDF: papers/<axis>/xxx.pdf"，下载脚本下载 |

---

## 5. 执行检查清单

每张精读卡完成时主 agent 验收：

- [ ] arxiv id、作者、venue、PDF 路径四件套齐全
- [ ] 四维评分齐全且至少一维 ≥ 4
- [ ] 核心贡献 ≤ 2 句、关键洞察 ≤ 5 句
- [ ] R-ID 命中表至少一行
- [ ] 与既有 [`research/papers/`](../papers/) 重叠状态明确标注

每个轴完成时主 agent 验收：

- [ ] 该轴 10 篇 ≥ 6 篇 2024-2026 新作
- [ ] 单一机构贡献不超过 4 篇
- [ ] 至少 1 篇 VZ 关联度 = 5 的"立刻反哺"候选

跨轴最终验收（深度综述前）：

- [ ] 100 篇分布在 10 轴，每轴 10 篇
- [ ] [`_candidates.md`](_candidates.md) 至少 150 候选 + 50 跳过原因
- [ ] [`papers/`](papers/) 至少 90 PDF（≤ 10 不可达需在 [`_candidates.md`](_candidates.md) 列出原因）

---

## 6. 文档间引用约定

- 精读卡内引用其他论文：`[Title (axis-id)](../<axis>/<file>.md)`
- 引用 VZ spec：`[`docs/specs/xxx.md`](../../docs/specs/xxx.md)`（路径相对于 `research/probe/notes/<axis>/`）
- 引用 VZ 设计源头：`[`docs/next_gen_emogpt.md`](../../docs/next_gen_emogpt.md)`
- 引用既有调研：`[`research/arxiv-survey-2026-05.md`](../../arxiv-survey-2026-05.md)` 等
