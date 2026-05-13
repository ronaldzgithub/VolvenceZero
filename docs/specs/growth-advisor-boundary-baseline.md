# Growth-Advisor Boundary Baseline

> Status: scaffold v0.1 (SHADOW)
> Owner: growth-advisor-pilot-packet G-A (debt #64)
> Driver: [`scripts/growth_advisor_boundary_eval.py`](../../scripts/growth_advisor_boundary_eval.py)

## 1. 范围

记录 4 个 boundary policy（`bp-no-hard-sell` / `bp-no-overclaim` / `bp-no-flooding` / `bp-no-judgmental`）在 `cheng_laoshi` profile 下、合成 GT 上的触发率分布。这是 [`commercialization-assessment.md`](../business/commercialization-assessment.md) §4.2 P2 kill criteria "30 天 boundary 触发率 < 5% 或 > 50% 要重调架构" 的**前置 baseline 数字**——没这 baseline，30 天试点跑出来无法判断"是过严还是正常"。

## 2. 当前 baseline (待回填)

| Boundary | trigger_rate (on N=100 reviewer GT) | precision | recall | F1 |
|---|---|---|---|---|
| bp-no-hard-sell | TBD | TBD | TBD | TBD |
| bp-no-overclaim | TBD | TBD | TBD | TBD |
| bp-no-flooding | TBD | TBD | TBD | TBD |
| bp-no-judgmental | TBD | TBD | TBD | TBD |

ACTIVE 通过条件：每个 boundary `trigger_rate ∈ [0.05, 0.50]` AND `precision ≥ 0.70` AND `recall ≥ 0.60`。

## 3. Per-archetype 偏移

不同 archetype 触发的 boundary 分布天然不同（`product_seeking` 用户更可能触发 `bp-no-hard-sell`）。本 baseline 进入 ACTIVE 后落 5×4 archetype-boundary 矩阵。

## 4. 30 天试点对账规则

- 试点客户实测 trigger_rate 与本 baseline ± 50% 范围内 → 视为正常
- 超 ± 50% → 触发 [`commercialization-assessment.md`](../business/commercialization-assessment.md) §4.2 kill criteria；需调 boundary policy
- 跨试点客户系统性偏低 → boundary 在该客户的 audience 上失效（policy review）
- 跨客户系统性偏高 → boundary 过严（profile review）

## 5. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W2） | 5 题 `.example` 落档 + 本 spec v0.1 |
| **ACTIVE**（W6） | reviewer 标注 100+ 段 + `boundary_eval.py` 真跑 + baseline 表回填 |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold + 5 题 `.example`。
