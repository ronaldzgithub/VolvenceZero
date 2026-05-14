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
| [xfund-pitch-deck-v2-zhaojiangbo.md](./xfund-pitch-deck-v2-zhaojiangbo.md) | **当前主用** — 60 分钟版 22 页 deck 蓝图：v2.1 加强 — 赵江波 25 年 first principle 弧线（高二省物理 / 2017 阿里目标 = 自动化编程 / 好牌 30 万 0 投放 / 2022 自费 all-in） + 杨柳博士 active learning + 独立研究者锚点 + **私域运营 deep dive（5 页 + 7 分钟核心 demo + Mobi 单位经济硬数据 + 与微盟/有赞对比）** + 12 题 Q&A 必答 + 设计制作清单 | v2.1（2026-05-15） |

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
