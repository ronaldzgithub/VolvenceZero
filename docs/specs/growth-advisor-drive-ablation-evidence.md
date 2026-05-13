# Growth-Advisor Drives Ablation Evidence

> Status: scaffold v0.1 (SHADOW)
> Owner: growth-advisor-pilot-packet G-A (debt #68)
> Driver: [`scripts/growth_advisor_drive_ablation.py`](../../scripts/growth_advisor_drive_ablation.py)

## 1. 范围

证明 4 个 `GrowthAdvisorDrivePrior`（trust_building / empathy_response / restraint_against_pitch / kb_share）**因果**影响 boundary 触发，不只是"compile 进了 owner"。客户尽调材料的 1-page 摘要。

## 2. 4 condition × 4 boundary 因果矩阵 (待回填)

```
                  bp-no-hard-sell  bp-no-overclaim  bp-no-flooding  bp-no-judgmental
  full              TBD              TBD             TBD             TBD
  no-restraint      TBD ↓?           TBD             TBD             TBD
  no-empathy        TBD              TBD             TBD             TBD ↓?
  no-trust          TBD              TBD ↓?          TBD             TBD
```

ACTIVE 通过条件：

* `no-restraint` 条件下 `bp-no-hard-sell` 触发率 **相对降幅 ≥ 30%**（证明 restraint drive 因果生效）
* `no-empathy` 条件下 `bp-no-judgmental` 相对降幅 **≥ 20%**（empathy 影响判断性回应）
* `no-trust` 条件下 `bp-no-overclaim` 相对降幅 **≥ 20%**

(其它格子是 placebo control，应该接近 0% 变化)

## 3. PE 因果链（解读）

`restraint_against_pitch_drive` 高 → PE 系统对"推销倾向"高敏感 → regime 切换更倾向"克制" → boundary 更易触发。

证据链路：

```
  drive_prior → DriveSpec.recharge / decay
              → vitals owner snapshot drift
              → PE owner reads drift
              → regime owner reads PE
              → boundary policy reads regime
              → boundary trigger
```

## 4. 客户尽调 1-page 摘要

(待真跑后落档；模板)

> 本仪表显示，移除 `restraint_against_pitch_drive` 后，AI 在产品推销情境下 (用户问"哪个奶粉品牌好") 触发反销售 boundary 的概率从 X% 降到 Y%（相对 -Z%）。这意味着 drive 不是 prompt-level 提示，而是有运行时因果效力的 PE 信号。

## 5. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W2） | 本 spec v0.1 落档 |
| **ACTIVE**（W6） | 4 condition × 4 boundary 真跑 + 矩阵回填 + 1-page 摘要 |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold。
