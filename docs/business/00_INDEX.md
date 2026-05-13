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
