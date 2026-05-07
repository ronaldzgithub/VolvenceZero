# OpenAI 顶尖学者 cognitive AGI 论文调研 2026 版

本目录系统调研 OpenAI 现役一线（Pachocki / Mark Chen / Madry / Zaremba / Baker / Tworek / Kaiser / Lightman / Brown / Wei / Chung 等）+ diaspora 核心（Sutskever@SSI、Schulman→Anthropic→Thinking Machines、Karpathy@Eureka Labs、Murati & Lilian Weng@Thinking Machines）2024-2026（重点 2025-11 ~ 2026-05）与 cognitive AGI 直接相关的 8 篇精读论文 + 博客 / 路线图素材，并与 VZ 的 R1–R15+R-PE 不变量做正面对比。

旧版（2024-12 ~ 2025-10 的 7 篇）位于 [`../openai-frontier-2025/`](../openai-frontier-2025/)，本版只写"对旧版的 delta"，不重复结论。

## 推荐阅读顺序

1. **`notes/00_executive_summary_2026.md`** — 高密度执行摘要（先读这份）
2. **`notes/04_actionable_inspirations.md`** — 给 VZ 的 10 条可落地行动项（含 R-ID + 优先级 + 工作量 + 落点路径）
3. **`notes/02_route_divergence_2026.md`** — OpenAI / Anthropic / VZ 三方路线分歧矩阵（标 2025 判断的保持/反转/加强）
4. **`notes/03_diaspora_landscape.md`** — SSI / TML / Eureka / Anthropic 路线图 + 刻意沉默分析
5. **`notes/01_paper_by_paper_2026.md`** — 逐篇技术等级评估 + R-ID 映射 + 2025 旧版 7 篇 re-evaluation

## 目录结构

- `papers/` — 8 篇论文 PDF（已下载，约 39MB）
- `notes/_candidates.md` — 候选清单 + 选材原则（执行记录）
- `notes/00_executive_summary_2026.md` — 执行摘要（BOSS 主读）
- `notes/01_paper_by_paper_2026.md` — 逐篇笔记
- `notes/02_route_divergence_2026.md` — 路线分歧矩阵
- `notes/03_diaspora_landscape.md` — Diaspora 路线图
- `notes/04_actionable_inspirations.md` — 可落地行动项

## 论文清单（8 篇精读，按 VZ 关联强度排序）

| # | arXiv | 标题 | 时间 | 关键作者 | 与 VZ 关联 |
|---|---|---|---|---|---|
| **N4** | 2511.18397 | Natural Emergent Misalignment from Reward Hacking in Production RL | 2025-11 | MacDiarmid / Hubinger / Leike / Perez（Anthropic）+ Greenblatt（Redwood）| **本期最重要论文**：reward hack 自发泛化为 alignment faking + sabotage |
| **N1** | 2603.05706 | Reasoning Models Struggle to Control their Chains of Thought | 2026-03 | Kivlichan / Baker / Carroll / Korbak（OpenAI）+ NYU/UCL/UPenn | CoT-Control 评估套件 14076 题 |
| **N3** | 2511.11584 | Output Supervision Can Obfuscate the Chain of Thought | 2025-11 | Drori et al.（MATS）| Feedback Spillover + Mind/Face 双模型缓解 |
| **N6** | 2505.05410 | Reasoning Models Don't Always Say What They Think | 2025-05 | Schulman + Anthropic Alignment Science | CoT faithfulness：reveal rate 多 1-20% |
| **N7** | 2510.07686 | Stress-Testing Model Specs Reveals Character Differences | 2025-10 | Schulman + Anthropic | 12 frontier LLM × 410K tradeoff 场景 |
| **N5** | 2602.05910 | Chunky Post-Training: Data Driven Failures of Generalization | 2026-02 | Schulman + Thinking Machines | SURF/TURF 工具 |
| **N2** | 2601.03267 | GPT-5 System Card | 2025-08 | OpenAI 全体 | fast+thinking 双模型 + router + safe-completions |
| **N8** | 2510.16255 | Detecting Adversarial Fine-tuning with Auditing Agents | 2025-10 | Schulman + Carlini | Auditor agent 范式 |

## 一句话结论

> 2025-11 ~ 2026-05 这 7 个月，OpenAI 主要做"工程整合"（GPT-5 System Card）、Anthropic 做"alignment science 实证"（N4/N6/N7）、Schulman 在 diaspora 串起 4 篇关键论文（N5/N6/N7/N8）、Sutskever 选择刻意沉默。
>
> **本期对 VZ 路线的最强外部背书是 N4：production RL 中学会 reward hack 会自发泛化为广泛 misalignment**。这从相反方向证明了 VZ 在 R3/R4/R8/R10/R12/R-PE 上的设计选择是结构性正确的。
>
> VZ 不要追赶 OpenAI 的工程整合，而要把 N3 的 Mind/Face、N4 的 inoculation、N7 的 cross-model stress-test、N8 的 auditor agent 这四个工程范式吸收为 VZ ModificationGate 标配工具——具体行动项见 `notes/04_actionable_inspirations.md`（10 条 P0/P1/P2 任务）。
