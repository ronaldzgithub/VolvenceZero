# Figure Bake Cost Actuals

> Status: scaffold v0.1 (SHADOW — 数字待 Wave K Einstein bundle 实测回填)
> Last updated: 2026-05-13
> Owner: figure-evidence-packet G-F (debt #63)
> 对账目标：[`commercialization-assessment.md`](commercialization-assessment.md) §6.2 P1 单位经济（30-80 万首单 / 5-15 万 COGS 估算）

## 1. 目的

把 P1 单位经济从"估算 5-15 万 COGS"换成"实测 N figure 平均"，给 P1 第一份 quote 提供锚点。

## 2. 数据源

`scripts/figure_cost_summary.py` 从 `data/figure_audit/<figure_id>/*.json` 读所有 audit record，按 bundle_id group，累加 `cost_breakdown` 字段（debt #63 closure 给 audit 加的字段）。

## 3. Einstein Wave K bundle 实测（待回填）

Bundle: `figure-bundle:einstein:29eacd226a7cdfd0`

### 3.1 工程师人天

| 阶段 | 人天 | 备注 |
|---|---|---|
| L0 crawl seed list 撰写 | TBD | 10 reviewer-staged URL |
| L0 crawl 实跑 + 排查 fail | TBD | 6 SUCCESS / 4 FAILED_HTTP |
| L1 cleaning + reviewer curate | TBD | 5 cleaned → reviewer 选 2 |
| L2 verification ledger | TBD | 14 records (3 PASS + 4 NEEDS_REVIEW per anchor) |
| Bundle bake + audit | TBD | `figure-bake bake-bundle --corpus-mode curated` |
| **小计** | TBD | |

### 3.2 reviewer 人小时

| 阶段 | 人小时 | 单价 | 小计 ¥ |
|---|---|---|---|
| L1 review + curate | TBD | 200 | TBD |
| L2 NEEDS_REVIEW resolve | TBD | 200 | TBD |
| Bundle 验证 | TBD | 200 | TBD |

### 3.3 GPU 小时

| 阶段 | 小时 | 单价 | 小计 ¥ |
|---|---|---|---|
| L0 crawl（轻 CPU，几乎 0） | 0 | 0 | 0 |
| L1 cleaning（CPU） | 0 | 0 | 0 |
| L2 verification（CPU） | 0 | 0 | 0 |
| Bundle bake (synthetic) | 0 | 0 | 0 |
| Bundle bake (PEFT real) | TBD | 5 | TBD |

### 3.4 Archive 访问 / rate limit 等待

实跑 wallclock：TBD

### 3.5 Total

¥TBD

## 4. 报价对账（待回填后）

| 项 | 估算（assessment §6.2） | Wave K 实测 | 差距 |
|---|---|---|---|
| Bundle 编译 COGS | 5-15 万 | TBD | TBD |
| 首单报价 | 30-80 万 | （维持） | TBD |
| 毛利率 | 46-60% | TBD | TBD |

## 5. 风险预警

如果 Wave K 实测 > 25 万：
- 30 万首单 → 接近无毛利
- 触发 [`commercialization-assessment.md`](commercialization-assessment.md) §4.1 P1 kill criteria 再评估
- 选项：
  1. 上调首单价到 50 万+
  2. 拆"编译费 vs 托管费"独立报价
  3. 投资 reviewer SOP 自动化降 reviewer 工时

## 6. 第二款 figure 预算上限

基于 Einstein 实测 + 学习曲线（reviewer 工艺更熟，工程师不需要 debug 管线），第二款 figure（推荐苏轼或居里夫人）预算上限：

- 工程师人天：上限 -30%
- reviewer 工时：上限 -20%
- GPU：与 Einstein 同量级
- **总预算上限**：Einstein 实测 × 0.75

## 7. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W5） | 本文档 v0.1 + `scripts/figure_cost_summary.py` --dry-run；Wave K 实测数字字段 TBD 占位 |
| **ACTIVE**（W5+） | Einstein Wave K 实际 cost_breakdown 回填到 audit log；本文档全部 TBD 替换为真数字；P1 报价模板 update |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold；数字待回填。
