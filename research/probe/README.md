# Cognitive AGI Probe — 100 篇深度调研

> **目的**：建立一份 cognitive AGI 全景 100 篇 canonical 主名单，覆盖 [`research/papers/`](../papers/) 既有方向之外的 promising 子领域，并对其做跨轴深度综述 + 对 VZ 项目借鉴意义的可执行 spec 行动清单。
>
> **完成时间**：2026-05-08。
>
> **入口建议**：根据你是谁，选下面对应的"阅读路径"。

---

## 三种 persona 阅读路径

### Path A — BOSS / 决策层（≤ 10 分钟）

只读两份：

1. **[`00_executive_summary.md`](00_executive_summary.md)** —— 3 页执行摘要：方法 + 5 大主题 + 10 条立刻反哺动作 + 6 条反向证据风险地图。
2. **[`11_vz_implications.md`](11_vz_implications.md) §"跨 R-ID 的整体行动建议"** —— 38 条 P0 按"优先级 × 工作量"排序的 3 批落地计划。

判断点：
- 如果 5 大主题判断正确 → 启动 P0 第 1 批 5 条 spec 写入
- 如果 6 条反向证据中 ≥ 2 条有疑虑 → 召集 R-ID 专项 review

### Path B — 研究员 / 综述读者（≤ 1 小时）

按推荐顺序读：

1. **[`00_executive_summary.md`](00_executive_summary.md)** (3 min) — 拿到全景
2. **[`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md)** (25 min) — 5 大主题 + cognitive AGI 重大意义
3. **[`02_axis_walkthrough.md`](02_axis_walkthrough.md)** (30 min) — 10 轴依次走读，看具体论文
4. **[`_candidates.md`](_candidates.md)** §跳过原因 (5 min) — 看哪些被排除及为什么

可选深入：
- 任一轴感兴趣 → [`candidates/<axis>.md`](candidates/) 查该轴 candidate 详细评分
- 任一论文感兴趣 → [`papers/<axis>/`](papers/) 读本地 PDF（probe 独家 47 篇）或 [`research/papers/`](../papers/) 读复用 52 篇

### Path C — 工程师 / spec 实施者（≤ 30 分钟）

按推荐顺序读：

1. **[`00_executive_summary.md`](00_executive_summary.md)** §3 (3 min) — 10 条立刻反哺动作概览
2. **[`11_vz_implications.md`](11_vz_implications.md)** (20 min) — 100 篇按 R-ID 重组 + 38 P0 + 52 P1 完整行动清单（含 spec 文件名）
3. **[`02_axis_walkthrough.md`](02_axis_walkthrough.md)** §对应轴 (5 min / 轴) — 看具体论文一句话定位

实施时：
- 每条 P0 都标注了具体落点 spec 文件名（[`docs/specs/*.md`](../../docs/specs/)）
- 每条 P0 都标注了论文 arXiv ID（PDF 在 [`papers/<axis>/`](papers/) 或 [`research/papers/`](../papers/)）

---

## 文档清单

| 文档 | 角色 | 阅读时间 |
|---|---|---|
| **[`README.md`](README.md)** | 目录导航 + persona 路径 | 2 min |
| **[`00_executive_summary.md`](00_executive_summary.md)** | 执行摘要（给 BOSS） | 3 min |
| **[`01_method_and_scoring.md`](01_method_and_scoring.md)** | 方法论与评分标尺 | 5 min |
| **[`_candidates.md`](_candidates.md)** | 100 篇主名单 + 跨轴冲突解析 + 50+ 跳过原因 | 10 min |
| **[`02_axis_walkthrough.md`](02_axis_walkthrough.md)** | 10 轴走读（每轴 10 篇展开） | 30 min |
| **[`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md)** | 跨轴深度综述（5 大主题）+ 重大意义 | 25 min |
| **[`11_vz_implications.md`](11_vz_implications.md)** | 100 篇按 R-ID 重组 + 38 P0 + 52 P1 | 20 min |
| **[`_download_summary.md`](_download_summary.md)** | 下载执行总结（47/47 + HTML 全部成功） | 1 min |
| **[`candidates/<axis>.md`](candidates/)** | 10 轴各自 candidate 详细清单 | 各 5 min |
| **[`papers/<axis>/`](papers/)** | 47 篇 probe 独家 PDF + 1 篇 HTML | — |

### 10 轴 candidate 文件

| 轴 | 文件 | 名称 |
|---|---|---|
| A1 | [`candidates/A1.md`](candidates/A1.md) | Reasoning & Test-Time Compute |
| A2 | [`candidates/A2.md`](candidates/A2.md) | World Models & Model-Based RL |
| A3 | [`candidates/A3.md`](candidates/A3.md) | Memory & Continual Learning |
| A4 | [`candidates/A4.md`](candidates/A4.md) | Hierarchical & Temporal Abstraction |
| A5 | [`candidates/A5.md`](candidates/A5.md) | Meta-Learning & In-Context Learning |
| B1 | [`candidates/B1.md`](candidates/B1.md) | Active Inference & Predictive Coding |
| B2 | [`candidates/B2.md`](candidates/B2.md) | Theory of Mind & Social Cognition |
| B3 | [`candidates/B3.md`](candidates/B3.md) | Open-Ended & Curriculum Learning |
| C1 | [`candidates/C1.md`](candidates/C1.md) | Self-Improvement & Modification Gating |
| C2 | [`candidates/C2.md`](candidates/C2.md) | Mechanistic Interpretability & Internal Control |

---

## 与 [`research/`](../) 既有调研的关系

| 文档 | 作用 | 与 probe 的关系 |
|---|---|---|
| [`research/papers/`](../papers/) + [`research/papers/dm/`](../papers/dm/) | ~104 篇 PDF | probe 主名单复用 52 篇（**50%**），证明覆盖度高 |
| [`research/arxiv-survey-2026-05.md`](../arxiv-survey-2026-05.md) | NL/ETA 主线 R1-R15 + R-PE 综述 | probe 在其基础上扩展为 10 轴评分清单，可作 spec 反哺的 canonical 入口 |
| [`research/openai-frontier-2026/`](../openai-frontier-2026/) | OpenAI/Anthropic 圈 8 篇精读 | N1-N8 全部归入 probe 的 C1（N4/N7/N8）+ C2（N6 跨指）+ A1（N2 跨指）；probe 不重复精读 |
| [`research/deepmind-author-paper-assessment-2026-05.md`](../deepmind-author-paper-assessment-2026-05.md) | DeepMind A/B/C-tier 评估 | A-tier 10 篇全部入选 probe（散布在 A2/A4/B3）；B-tier 大多入选 |
| [`research/core-author-paper-assessment-2026-05.md`](../core-author-paper-assessment-2026-05.md) | NL/ETA 作者圈 8 篇 | 全部入 probe A3 + A5 + A4（NL 主线 / mesa-optimization / ETA 主线）|

---

## 调研规模总览

| 指标 | 数值 |
|---|---|
| 主名单总数 | 100 篇 |
| 时间窗 | 2014-2026（12 年），其中 2024-2026 新作 ≥ 6 / 轴 |
| 复用 [`research/papers/`](../papers/) | 52 篇（52%） |
| Probe 独家新增 | 47 PDF + 1 HTML |
| 下载成功率 | 100%（48 / 48） |
| 跳过候选总览 | ≥ 50 条（含原因） |
| VZ "立刻反哺" (V=5) 候选 | 49 / 100 |
| 反向证据 / 挑战路线 | 6 篇 |
| R-ID 覆盖 | 14 / 14（R-PE + R1-R15 全覆盖） |
| 中国厂内贡献 | 10 篇（A1=2 / A2=4 / B2=3 / B3=1）|
| 总 P0 行动数 | 38 条 |
| 总 P1 行动数 | 52 条 |
| 总 spec 写入估计工作量 | ≈ 1 人月（不含代码实现） |

---

## 关键发现速递

> 完整论述见 [`10_deep_synthesis_2026.md`](10_deep_synthesis_2026.md)。下面是**一句话版本**：

**Cognitive AGI 在 2024-2026 经历了一次范式转变**——从"语言模型扩展"到"认知架构原则"。5 大 frontier 主题同时走向工程成熟：

1. **PE 一级化**：reward 不是基本量，PE 是；所有 reward / value / credit / curiosity 都是 PE 的下游 readout。
2. **从 token 控制到 latent 控制**：思考不在 token，控制不在 prompt；A1 latent CoT + A4 ETA + A5 mesa + C2 function vectors 联合证明 latent 是控制空间。
3. **涌现 vs 编码**：subgoals / options / regimes / social rules 都可以从正确架构 + 正确信号中涌现，但需要 latent bottleneck 作为 enabling structure。
4. **记忆即架构**：memory ≠ vector DB；memory 与 architecture 与 optimizer 是同一件事，只是不同时间尺度。
5. **自修改要门 + 评估只读**：Sleeper Agents / Alignment Faking / N4 三件套证明 alignment 不是一次性 RLHF，而是 capacity bound + 可回滚 + monitoring + read-only eval 的持续 governance loop。

**VZ 项目正好处在这 5 个主题的交点上**——49/100 篇直接反哺 VZ 14 条 R 不变量。但同时承担 5 个主题的 failure mode 风险——6 条反向证据必须严肃追踪。

---

## 联系 / 下一步

- **BOSS 决策点**：是否启动 P0 第 1 批 5 条 spec 写入（详见 [`11_vz_implications.md`](11_vz_implications.md) §"第 1 批"）
- **下一轮调研建议**：probe 100 → 200 篇扩展（覆盖更多 social cognition / embodied / neuro-symbolic 子轴）
- **持续追踪**：6 条反向证据每 6 月 review 一次"failure mode 是否真的暴露"
