# Companion Bench: Cost Model v0

> Status: SHADOW placeholder
> Driver: [`scripts/companion_bench/estimate_quarterly_cost.py`](../../scripts/companion_bench/estimate_quarterly_cost.py)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #56
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §3

## 1. 目的

把 Companion Bench 的"持续每季度更新"从口号变成可预算项。给 submitters 的预算指引 + VZ 自身的运营预算上限。

## 2. 价格表（来自 [`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py)）

| 模型家族 | input $/1M tok | output $/1M tok |
|---|---|---|
| GPT-5 | TBD | TBD |
| Claude Opus 4.7 | TBD | TBD |
| DeepSeek V4 | TBD | TBD |
| Qwen3-Max | TBD | TBD |
| Gemini 2.5 | TBD | TBD |

（实际数字读 `_DEFAULT_PRICES`，可被 submission 覆盖）

## 3. 单次跑分总成本拆解（待真跑回填）

### 3.1 公开 24 scenario × 1 SUT × 1 seed

```
  per-turn judge tokens: ~ TBD
  arc judge tokens: ~ TBD
  SUT response tokens: ~ TBD
  -----------------------------
  total per scenario: ~ TBD USD
  total per submission (24 scen): ~ TBD USD
```

### 3.2 完整 release-tier (10 SUT × 3 seed × 120 scenario)

```
  total: ~ $5,000 - $15,000 USD（参考 [#32](../known-debts.md) sub-track 1 估算）
```

### 3.3 Sweep 类（#48 / #52 / #53 / #54）总成本

```
  judge robustness sweep: ~ TBD USD
  calibration sweep: ~ TBD USD
  simulator robustness sweep: ~ TBD USD
  statistical power analysis: ~ TBD USD
  -------------------------------------
  total Phase A sweep evidence: ~ $8,000 - $11,000 USD
```

## 4. Phase A 6 个月总预算

| 项 | 估算 USD | 估算 RMB |
|---|---|---|
| 评估证据 sweep（#48+#52+#53+#54） | $8,000 - $11,000 | ¥56k - ¥77k |
| Reference 跑分（10 SUT release-tier） | $5,000 - $15,000 | ¥35k - ¥105k |
| Buffer (12-23%) | $300 - $500 | ¥2k - ¥4k |
| **Phase A 6 月 total** | **$13.3k - $26.5k** | **¥93k - ¥186k** |

✅ 完全 covered [`commercialization-assessment.md`](../business/commercialization-assessment.md) §7.2 P5 GTM 预算上限 ¥10-30 万。

## 5. Submitter 预算指引

| 提交模式 | submitter 自付预算 | VZ 收费 |
|---|---|---|
| self-hosted 公开 24 scenario | $50-150 | $0 |
| trusted-runner 公开 24 scenario | 0（VZ 跑） | $50（VZ 收成本+管理费）|
| trusted-runner held-out 96 scenario | 0 | $300-500 |
| trusted-runner full 120 scenario | 0 | $500-800 |

详见 [`companion-bench-trusted-runner-protocol.md`](companion-bench-trusted-runner-protocol.md)。

## 6. 季度更新成本闭环

每季度榜单刷新 = 1 次 release-tier 跑分（替换可能的新 SUT），约 $5-15k。8 季度 / 2 年 = $40-120k → ¥280k-840k，与 [`commercialization-assessment.md`](../business/commercialization-assessment.md) §4.5 P5 路径"零直接收入但乘数 ×1.5-2 加在 P1/P2/P4"商业回报对账：1 个 P1 客户首单 30 万即可覆盖 1 年榜单成本。

## 变更日志

- 2026-05-13: v0 SHADOW placeholder 落档。
