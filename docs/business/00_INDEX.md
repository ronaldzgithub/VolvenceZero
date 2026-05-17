# Business 文档索引

> 本目录存放 VolvenceZero 的商业评估、产品化路线、商业化策略相关文档。
> 与 `docs/prd.md`（产品需求）、`archetecture.md`（架构边界）、`docs/specs/`（能力域 spec）平行：
> - `docs/specs/` 回答"系统应该怎么造"
> - 本目录 回答"系统造出来后应该怎么卖"

## 当前文档

| 文档 | 内容 | 状态 |
|---|---|---|
| [commercialization-assessment.md](./commercialization-assessment.md) | VZ 全面商业化评估：差异化资产、市场结构、6 条路径备选、12-24 个月排序、单位经济、GTM、风险与 kill criteria、OKR 草案、反目标 | v0.1（2026-05-13） |
| [xfund-technical-credibility-brief.md](./xfund-technical-credibility-brief.md) | 对 Xfund (Patrick Chung) 一次首谈用的技术可信度对外裁短版：thesis / 架构第一性承诺 / 8 件 ship 的事 / 与 Delphi / Open Evidence 对标 / anti-claims / 60 秒中英文口头版 | v0.1（2026-05-13） |
| [xfund-strategic-thesis.md](./xfund-strategic-thesis.md) | Xfund 完整战略叙事书面版：业界深度调研 / 与头部互相支撑 / 已 ship 实物 + 算法证明 / IQ + EQ 涌现路径 / 三件事 evidence cascade / anti-claims / 60 秒口头版 | v0.1（2026-05-14） |
| [xfund-pitch-deck-blueprint.md](./xfund-pitch-deck-blueprint.md) | Xfund 投资理念深度研究 + PPT 设计原则反推 + 27 页 slide-by-slide 蓝图 + 30/45 分钟两套演讲编排 + leave-behind packet 清单 | **v0.1 已 deprecated** — 由 v2 替代，研究底稿保留 |
| [xfund-pitch-deck-v6-zhaojiangbo.md](./xfund-pitch-deck-v6-zhaojiangbo.md) | **当前主用 — V6 thesis-driven 版**：基于创始人 thesis 8 步链条结构，每一步配明确 Proof 块（命名行业专家 / 同行评议论文 / 仓库可复跑实验 / 可核对市场数字）。恢复 V4 PDF 逻辑清晰度 + 保留 V3/V5 cool-tone 纪律 + 整合 strategic-thesis 技术深度。20 主 slide + 5 appendix。Cover 一句话："The runtime for Cognitive AGI."。开篇 8-step thesis（AGI → Cognitive AGI → continual learning → token-RL infeasible → emergent multi-timescale RL → Body/NL+ETA/Active Learning → 已实现 → 独立路径）；中段 IQ/EQ emergence 两 slide；商业 Mobi unit econ + kill criterion + 18-month plan + ask；附录含 Yang Liu 完整论文表 + 全 thesis 引用索引 + anti-claims + 60s 英文版 + 6 题 Q&A。 | **v6.0（2026-05-17）** |
| [xfund-pitch-deck-v5-zhaojiangbo.md](./xfund-pitch-deck-v5-zhaojiangbo.md) | V5 cool-tone 版（V3 骨架 + V4 三处定向加强：Human-as-vertical-data thesis / Mobi unit econ + kill criterion / Ask 页融资条款）。被 V6 取代。 | v5.0（2026-05-17） |
| [xfund-pitch-deck-v4-zhaojiangbo.md](./xfund-pitch-deck-v4-zhaojiangbo.md) | V4 PDF 配套 markdown（Body+Brain / Soul Migration / Cognitive AI Map / Reverse Validation 等情绪化叙事）。被 V6 取代。 | v4（2026-04） |
| [xfund-pitch-deck-v3-zhaojiangbo.md](./xfund-pitch-deck-v3-zhaojiangbo.md) | V3 cool-tone 第一版骨架（5 个降温原则）。被 V6 取代。 | v3.0（2026-05-17） |
| [xfund-pitch-deck-v2-zhaojiangbo.md](./xfund-pitch-deck-v2-zhaojiangbo.md) | V2 完整版 deck（26 页）：v2.7.2 加强 — **Slide 13 加 Niche Map 对比表**（vs MMLU / Chatbot Arena / MT-Bench / EQ-Bench 3 / RP-Bench / AgentBench — 7 个现有 benchmark 详细对比 + Companion Bench 独占长程关系曲线 niche 明示）+ v2.7.1 — **新增 Slide 8 灵魂 thesis "Human Beings Themselves Are the Vertical Data"** + **Slide 25 加入具体融资条款**（$3-5M / $20-30M pre-money / 7-10% equity）+ Slide 1 Elevator pitch 具体化 + Slide 2 "500 万 fully burned" + Slide 13 Companion Bench reference SUT 诚实标注（known-debts #82 5 phase timeline）+ Slide 23 OpenAI substitution defense + Q&A 全套诚实化 + known-debts #82/#83 入档 + Slide 14 Experiment Roadmap + Slide 13 Einstein Case Study + Slide 11 Hard Evidence + Slide 12 Companion Bench 网站 + 25 年 first principle 弧线 + 私域运营 deep dive + Excel 3 年财务全景 + 12 题 Q&A. 已被 V6 取代但保留为长版参考。 | v2.3（2026-05-15） |

## 阅读顺序建议

**第一次读**：
1. 先读 `commercialization-assessment.md` §1（系统的商业本质）和 §10（不应该做的事）—— 5 分钟内拿到方向感
2. 再读 §4（6 条路径备选）和 §5（推荐排序）—— 知道资源应该怎么分
3. 最后看 §8（风险与 kill criteria）—— 知道什么时候应该停

**做商业决策前**：
1. 先读 §10（反目标）—— 第一道过滤器
2. 读对应路径的章节（§4.x）+ 单位经济（§6）+ GTM（§7）
3. 决策完同步更新 §11 的复盘记录

## 与其他文档的边界

| 这里写 | 不写 |
|---|---|
| 商业模式、定价、客户、GTM | 系统能力、内核架构、契约 |
| 路径概率、风险、kill criteria | R-ID 设计原理、SHADOW/ACTIVE 协议 |
| 单位经济、ARR、毛利率假设 | substrate / owner / snapshot 实现 |
| 反目标、不做什么 | 工程任务、bug、refactor |

工程相关写在 `docs/specs/` 与 `docs/moving forward/`；本目录只写商业判断。

## 复盘节奏

每 90 天复盘一次。复盘输出新建 `commercialization-review-YYYY-MM-DD.md`，不直接 overwrite assessment 主文件；主文件版本号同步更新。
