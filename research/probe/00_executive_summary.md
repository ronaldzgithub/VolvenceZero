# 执行摘要 — Cognitive AGI 100 篇深度调研

> **本文角色**：给 BOSS 看的 ≤ 3 页摘要。3 分钟读完即获得：① 调研规模与方法 ② 100 篇核心发现 ③ 对 VZ 的 10 条立刻反哺动作 ④ 6 条反向证据的风险地图。
>
> **完整文档**：[`README.md`](README.md) 给出全部文档导航。
>
> **方法论**：[`01_method_and_scoring.md`](01_method_and_scoring.md)。**100 篇主名单**：[`_candidates.md`](_candidates.md)。

---

## 1. 调研规模与方法

- **范围**：cognitive AGI 全景，10 轴 × 10 篇 = 100 篇精选；不局限 VZ 已有方向。
- **时间窗**：每轴 ≥ 6 篇 2024-01 ~ 2026-05 新作；≤ 4 篇奠基经典；100 篇覆盖 2014-2026 共 12 年。
- **来源**：arXiv 主队列 + NeurIPS / ICLR / ICML / COLM / AAMAS / Science；OpenAI / Anthropic / DeepMind / Meta FAIR / 中国厂内（DeepSeek / 清华 / 西湖 / SJTU / Cap Med Beijing）。
- **机构均衡**：单机构单轴 ≤ 4 篇；中国厂内累计 10 篇（A1=2 / A2=4 / B2=3 / B3=1）。
- **下载**：47 篇新 PDF（probe 独家）+ 1 篇 HTML（Anthropic Scaling Monosemanticity）+ 复用 [`research/papers/`](../papers/) 既有 52 篇 = 100/100 全部本地化。**复用率 52%**——证明 BOSS 现有研究方向（NL/ETA/PE/credit/dual-track 主线）与 cognitive AGI 当前 frontier **结构性对齐**。
- **跳过精读**：用户决策跳过 100 张精读卡，直接进入综述阶段。
- **产出**：本文 + [`02_axis_walkthrough.md`](02_axis_walkthrough.md) + [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) + [`11_vz_implications.md`](11_vz_implications.md) + [`README.md`](README.md)。

---

## 2. 100 篇的核心发现：cognitive AGI 在 5 个主题上同时走向工程成熟

> 完整论述见 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md)。

| # | 主题 | 在反对什么默认假设 | 主关联 R-ID | 加强论文 ≥ |
|---|---|---|---|---|
| **T1** | **PE 一级化** | "reward 必须外部给出 / scalar reward 足够" | R-PE / R7 / R12 | 18 |
| **T2** | **从 token 控制到 latent 控制** | "推理就是生成 token / control via prompt" | R3 / R4 / R8 | 18 |
| **T3** | **涌现 vs 编码** | "结构必须先验设计 / 规则必须人工写" | R3 / R13 / R14 | 14 |
| **T4** | **记忆即架构** | "memory = vector DB / RAG 是足够的" | R1 / R5 / R6 | 12 |
| **T5** | **自修改要门 + 评估只读** | "RLHF 足够 alignment / 自训没有上限" | R9 / R10 / R12 / R15 | 13 |

**核心判断**：

> cognitive AGI 在 2024-2026 经历了一次从"语言模型扩展"到"认知架构原则"的范式转变。10 年前我们说 "agent = neural net + RL"，3 年前说 "agent = LLM + tool use"，**2026 年我们正在说 "agent = frozen substrate + emergent latent control + multi-timescale memory + bounded self-modification + read-only evaluation"**。
>
> **VZ 项目正好处在这 5 个主题的交点上**——不是某一条主题的"另一种实现"，而是 5 个主题的"系统化整合"。

---

## 3. 对 VZ 的 10 条立刻反哺动作（P0 第 1-2 批）

> 完整 38 条 P0 + 52 条 P1 见 [`11_vz_implications.md`](11_vz_implications.md)。下面 10 条按 **优先级 × 工作量** 排序的最高优先批：

### 第 1 批（≤ 4 周可落地）

| # | 行动 | 论文背书 | 落点 spec |
|---|---|---|---|
| **A1** | CPD 算法（PE spike + reward shift CUSUM）作为 β_t 检测初版实现 | A4-04 CPD + Option-Critic | [`emergent-action-abstraction.md`](../../docs/specs/emergent-action-abstraction.md) §β_t 信号设计 |
| **A2** | AlphaEvolve "evaluator 完备性 = 自修改硬条件"写入 ModificationGate 开启条件 | B3-01 AlphaEvolve | [`evaluation.md`](../../docs/specs/evaluation.md) + [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) |
| **A3** | Sleeper Agents 写入 ModificationGate motivation + R15 rollback 必须可达后门引入前 | C1-08 Sleeper Agents | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) §motivation + [`contract-runtime.md`](../../docs/specs/contract-runtime.md) §R15 |
| **A4** | Persona Vectors 双重集成（ModificationGate 监控 + R14 regime 漂移监控） | C2-09 Persona Vectors | [`cognitive-regime.md`](../../docs/specs/cognitive-regime.md) §monitoring + [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) |
| **A5** | Option Keyboard z_t 重参数化为 reward feature 线性组合权重（World/Self 双轨共享 SF basis） | A4-02 Option Keyboard | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) §z_t 参数化 |

### 第 2 批（≤ 8 周可落地）

| # | 行动 | 论文背书 | 落点 spec |
|---|---|---|---|
| **B1** | CMA 6 条必要充分条件 + 4 行为探针套件写入（VZ 已满足 4/6 + 部分 2/6） | A3-03 CMA | [`continuum-memory.md`](../../docs/specs/continuum-memory.md) §evaluation 工具集 |
| **B2** | PAIRED regret-based UED 作为"无 ground-truth 任务"评估闭环的形式化方法 | B3-09 PAIRED | [`evaluation.md`](../../docs/specs/evaluation.md) §6 |
| **B3** | Sophia 作为 dual_track + regime + semantic_state 工程实例化对照 + Alignment Faking 作为"双轨不得同步坍塌"实证依据 | B2-01 Sophia + C1-09 Alignment Faking | [`dual-track-learning.md`](../../docs/specs/dual-track-learning.md) Sources |
| **B4** | Lookback OI binding 写入 user_model / belief_assumption owner 内部表示设计 | B2-05 Lookback | [`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md) §user_model |
| **B5** | Math-Shepherd MC rollout 自动 step-label 模式作为 R-PE 细粒度信用分配实现路径 | A1-07 Math-Shepherd | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) §step-level PE auto-attribution |

> **总工作量**：38 条 P0 共约 **1 人月** spec 写入工作（不含代码实现）。

---

## 4. 6 条反向证据：VZ 必须严肃追踪的对照路线

> 完整论述见 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) §反向证据 + [`11_vz_implications.md`](11_vz_implications.md) §风险地图。

| # | 反向论文 | 反向路径 | VZ 路径 | 18 月内追踪点 |
|---|---|---|---|---|
| 1 | **A1-06 DeepSeek-R1** | token 空间 RL 直接跑 | latent space RL on z_t | R1 follow-up 是否暴露 reward hacking + 推理 trace 不可解释 |
| 2 | **A4-10 LDSC** | LLM 在 token 空间生成 subgoal | latent z_t subgoal | LDSC 是否暴露 subgoal drift + grounding gap |
| 3 | **B3-02 AI Co-Scientist** | token-space generate-debate-evolve | metacontroller in latent | 是否在长程任务暴露 debate 收敛慢 + evaluator 依赖外部湿实验 |
| 4 | **B3-04 SIMA 2** | Gemini 当 reward generator | 内禀 PE | 是否暴露 reward generator drift |
| 5 | **C1-03 Darwin Gödel Machine** | 全开放自修改 | 分层 + 有界 + 快照可回滚 | DGM scaling 后 archive verification cost 是否爆炸 |
| 6 | **C2-05 Representation Engineering** | reading 反推回训练 | reading 只读 | RepControl 是否暴露 Goodhart 化 spurious features |

> **追踪机制**：每条反向论文每 6 月 review 一次"failure mode 是否真的暴露"。一旦反向路径 ≥ 2 条同时未暴露 failure mode，VZ 的 R 不变量需要 BOSS 严肃考虑是否更新。

---

## 5. 100 篇主名单的 5 个全局自检（已通过）

| 检查项 | 阈值 | 实际 | 状态 |
|---|---|---|---|
| 总数 | 100 篇 | 100 | ✓ |
| 每轴新作 ≥ 6（2024-01 ~ 2026-05）| ≥ 6 / 轴 | A1=7 / A2=8 / A3=7 / A4=8 / A5=6 / B1=6 / B2=8 / B3=6 / C1=7 / C2=6 | ✓ 全部满足 |
| 每轴经典 ≤ 4 | ≤ 4 / 轴 | 全部 ≤ 4 | ✓ |
| 单一机构跨轴贡献 | ≤ 4 / 轴 | DeepMind / Anthropic / Meta / OpenAI 均 ≤ 4 | ✓ |
| VZ 关联度 = 5 的"立刻反哺"候选 | ≥ 1 / 轴 | A1=4 / A2=4 / A3=4 / A4=4 / A5=8 / B1=5 / B2=6 / B3=4 / C1=6 / C2=4 = **49 / 100** | ✓ 极宽裕 |

---

## 6. 关键观察

### 6.1 49/100 的"立刻反哺"率是结构对齐而非松绑

VZ 14 条 R 不变量与 cognitive AGI 当前 frontier 在 5 大主题（PE 一级化 / latent 控制 / 涌现 / 记忆即架构 / 安全门）上的对齐密度 = **49 / 100**。最浓密的反哺出现在 A5 (8 篇) / C1 (6 篇) / B2 (6 篇) / B1 (5 篇) 四轴——**正好是 NL/ETA + R-PE + 双轨 + ModificationGate 主线最深的四轴**。

### 6.2 R15 (可回滚) 是当前 spec 的最大缺口

100 篇直接命中 R15 只有 3 篇（C1-01 / C1-02 / C1-08），但 R15 是 R10 / R11 / R-PE 的实施前提。建议在 [`contract-runtime.md`](../../docs/specs/contract-runtime.md) §R15 单独补 Sources 段，引用 C1 治理三件套（Sleeper / Alignment Faking / N4）作为"为什么 R15 是基础设施"的实证依据。

### 6.3 中国厂内贡献集中在 A1/A2/B2，治理轴空缺

中国厂内 10 篇分布：A1=2（DeepSeek-R1 / Math-Shepherd） / A2=4（RLVR-World / EfficientZero V2 / AdaWorld / iVideoGPT） / B2=3（Sophia / Geometry of Persona / RLFF-ESC） / B3=1（EvoAgent）。**C1 / C2 治理轴 2024-2026 中国机构原创成果空缺**——这意味着 VZ 在治理设计上的国际化对照需以 Anthropic / DeepMind / OpenAI 为主。

### 6.4 复用率 52% = 既有方向高覆盖度

[`research/papers/`](../papers/) 既有 ~104 篇 PDF 中 52 篇与 probe 主名单重叠，**新增 47 篇主要分布在 A1（推理 +10）/ C1（治理 +6）/ C2（mech interp +8）三个 BOSS 现有目录覆盖较弱的轴**。**这 3 个轴是 VZ 接下来 6-12 个月 spec 重点补全的方向**。

---

## 7. 下一步

### 立刻（≤ 1 周）

- BOSS 阅读 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) 5 大主题 + [`11_vz_implications.md`](11_vz_implications.md) 第 1 批 5 条 P0；决策是否启动 P0 spec 写入。

### 短期（≤ 4 周）

- 启动 P0 第 1 批（A1-A5 共 5 条）spec 写入。
- 6 条反向证据的"failure mode 追踪"机制建立（每 6 月 review）。

### 中期（≤ 12 周）

- 启动 P0 第 2、3 批共 33 条 spec 写入。
- 100 张精读卡（本任务跳过）按需补全（建议从 P0 论文优先）。
- 100 篇按 "VZ R-ID 命中表" 的 owner 化重组到 [`docs/specs/`](../../docs/specs/) Sources。

### 长期（6-12 月）

- 6 条反向证据的 18 月窗口追踪报告。
- 5 个未解前沿（见 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) §当前研究的 5 个未解前沿）的进展跟踪。
- probe 100 篇 → 200 篇扩展（覆盖更多 social cognition / embodied / neuro-symbolic 子轴）。

---

## 8. 文档导航

| 文档 | 角色 | 阅读时间 |
|---|---|---|
| **本文 ([`00_executive_summary.md`](00_executive_summary.md))** | 执行摘要（给 BOSS） | 3 min |
| [`README.md`](README.md) | 目录导航 + 三种 persona 阅读路径 | 2 min |
| [`01_method_and_scoring.md`](01_method_and_scoring.md) | 方法论与评分标尺 | 5 min |
| [`_candidates.md`](_candidates.md) | 100 篇主名单 + 跨轴冲突解析 + 跳过原因 | 10 min |
| [`02_axis_walkthrough.md`](02_axis_walkthrough.md) | 10 轴走读（每轴 10 篇展开） | 30 min |
| [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md) | 跨轴深度综述（5 大主题）+ 重大意义 | 25 min |
| [`11_vz_implications.md`](11_vz_implications.md) | 100 篇按 R-ID 重组 + 38 P0 + 52 P1 行动清单 | 20 min |
| [`candidates/<axis>.md`](candidates/) | 10 轴各自 candidate 清单（详细评分 + 一句话定位） | 各 5 min |
| [`papers/<axis>/`](papers/) | 47 篇 probe 独家 PDF（C2-02 为 HTML） | — |
