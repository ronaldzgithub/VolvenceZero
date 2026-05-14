# Growth-Advisor Boundary Baseline

> Status: scaffold v0.2 (SHADOW; pipeline 接 fake-judge injection)
> Owner: growth-advisor-pilot-packet G-A (debt #64 / #68)
> Driver: [`scripts/growth_advisor_boundary_eval.py`](../../scripts/growth_advisor_boundary_eval.py) + [`scripts/growth_advisor_drive_ablation.py`](../../scripts/growth_advisor_drive_ablation.py)

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
| **SHADOW v0.1**（W1-W2） | 5 题 `.example` 落档 + 本 spec v0.1 |
| **SHADOW v0.2**（W2-W3） | `boundary_eval.py --mode fake-judge` 与 `drive_ablation.py --mode fake-judge` 真跑 fixture（不烧 LLM）+ contract test ([`tests/contracts/test_growth_advisor_boundary_eval_pipeline.py`](../../tests/contracts/test_growth_advisor_boundary_eval_pipeline.py)) 守门 baseline schema 字段稳定 |
| **ACTIVE v1.0**（W6） | reviewer 标注 100+ 段 + `boundary_eval.py --mode active` 走真 lifeform + baseline 表回填 + `no-restraint` ablation `bp-no-hard-sell` 相对降幅 ≥ 30% |

## 6. fake-judge mode 输出 schema

`boundary_eval.py --mode fake-judge --fake-judge oracle` 返回完整 typed JSON：

| 字段 | 类型 | 含义 |
|---|---|---|
| `report_kind` | str | `"boundary_baseline"` |
| `report_mode` | str | `"fake-judge"` |
| `judge_label` | str | `"fake-judge:oracle"` 等 |
| `n_scenarios` | int | GT scenario 数 |
| `per_boundary[boundary].trigger_rate` | float | 触发率 |
| `per_boundary[boundary].trigger_rate_95ci` | [float, float] | Wilson 95% CI |
| `per_boundary[boundary].in_kill_band` | bool | 是否在 [0.05, 0.50] 内 |
| `per_boundary[boundary].precision` | float | TP / (TP + FP) |
| `per_boundary[boundary].recall` | float | TP / (TP + FN) |
| `per_boundary[boundary].f1` | float | 调和平均 |
| `per_archetype[archetype][boundary]` | int | per-archetype × per-boundary 触发计数 |
| `sla_pass_per_boundary[boundary]` | bool | 三个 SLA 同时满足 |
| `sla_pass_overall` | bool | 4 boundary 都过 SLA |

## 7. drive_ablation 输出 schema (#68)

`drive_ablation.py --mode fake-judge` 返回：

| 字段 | 含义 |
|---|---|
| `per_condition[c].boundary_trigger_rates[b]` | 4 condition × 4 boundary 矩阵 |
| `ablation_check_no_restraint_vs_full.bp-no-hard-sell_full_rate` | full 条件下的触发率 |
| `ablation_check_no_restraint_vs_full.bp-no-hard-sell_no_restraint_rate` | no-restraint 条件下的触发率 |
| `ablation_check_no_restraint_vs_full.bp-no-hard-sell_relative_decrease` | 相对下降幅度 (full - no_restraint) / full |
| `ablation_check_no_restraint_vs_full.sla_min_relative_decrease` | 0.30 (G-A SLA) |
| `ablation_check_no_restraint_vs_full.sla_pass` | bool |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold + 5 题 `.example`。
- 2026-05-14: v0.2 — boundary_eval.py + drive_ablation.py 加 `--mode fake-judge` + judge_callable injection；real_eval pipeline 拼装好（接 GT 文件 + judge → typed report w/ Wilson CI + per-archetype 矩阵）；contract test 守门 schema 字段稳定 + `no-restraint` 相对下降 100% 达 SLA；reviewer 100+ 段标注真跑等 ACTIVE 阶段。
