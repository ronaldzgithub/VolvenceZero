# Rollback Drill Cadence

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: cross-cutting-foundation-packet F-D (debt #50)

## 1. 范围

`figure-vertical.md` + `dlaas-platform.md` 多处承诺 `is_reversible=True + rollback evidence`。本 spec 把"何时跑 rollback drill"从隐式工艺变成显式节奏，让 P1 / P4 客户合同里的"提供回滚证据" SLA 在第一次客户事故之前就有真实战记录。

参考：[`commercialization-assessment.md`](../business/commercialization-assessment.md) §8.1.5 + cross-cutting-foundation-packet F-D 子任务 1-3。

## 2. 三档 drill 节奏

| 触发 | drill 类型 | 频率 | 跑哪些 script | SLA 影响 |
|---|---|---|---|---|
| 例行 | figure / growth-advisor 各跑一次 | **每月 1 次** | `scripts/rollback_drill_figure.sh` + `scripts/rollback_drill_growth_advisor.sh` | 客户合同写"每月演练一次回滚" |
| substrate 升级前 | 全 vertical + substrate upgrade drill | **每次 substrate N → N+1 升级** | 上面两个 + `scripts/rollback_drill_substrate_upgrade.sh` | 升级 PR review 必须 attach drill artifact |
| 新 figure bundle 上线前 | figure 单方向 | **每个新 figure_id** | `scripts/rollback_drill_figure.sh FIGURE_ID=<new>` | bundle 进 production-tier 必须 attach drill |

## 3. drill 通过标准

每次 drill 必须满足：

| 维度 | 阈值 | 检测方式 |
|---|---|---|
| logits L1 distance（rollback 前后）| < 1e-6 | `tests/perf/test_production_rollback_drill.py` 真跑 Qwen |
| audit chain append-only | 100% | drill script 后跑 `python -m lifeform_domain_figure.audit verify_chain --figure <id>` |
| rollback 总耗时 | < 30 min | drill script wallclock |
| frozen base ``state_dict_hash`` 不变 | 100% | drill script 前后两次抽 SHA-256 对比 |

## 4. drill 失败处置

| 失败模式 | 立即动作 |
|---|---|
| logits L1 > 1e-6 | 暂停所有 OFFLINE artifact 进 ACTIVE；触发 incident review |
| audit chain 出现 mutation | 暂停 bundle bake；audit module 紧急 review（debt #23 闭合契约破坏）|
| frozen base 字节漂移 | 暂停 PersonaLoRAPool ACTIVE；调查 forward hook 是否泄露副作用 |
| 总耗时 > 30 min | 不阻塞 ACTIVE，但 ops 调优 GPU pipeline + bundle 持久化路径 |

## 5. drill artifact

每次 drill 输出 `artifacts/rollback_drill/<vertical>-<figure_or_profile>-<date>.json`：

```json
{
  "drill_date": "2026-05-13",
  "vertical": "figure",
  "figure_id": "einstein",
  "passed": true,
  "logits_l1_distance": 4.2e-9,
  "rollback_wallclock_sec": 187.3,
  "audit_chain_byte_stable": true,
  "frozen_base_state_dict_hash_pre": "<sha256>",
  "frozen_base_state_dict_hash_post": "<sha256>",
  "rollback_audit_id": "<deterministic id>"
}
```

artifact 留存：每月归档到客户访问的 `evidence_root_dir/rollback_drill/`，供合规审计。

## 6. 与 commercialisation 商业承诺的对账

- **§4.1 P1 kill criteria**："任何客户合同写'提供回滚证据'但首次回滚失败 → 触发 incident，砍" → 本 spec § 4 失败处置覆盖
- **§8.1.5 团队执行风险**："工程纪律和商业 KPI 之间产生张力（'我们急着上线，跳过 SHADOW 验证'）" → drill cadence 是合同 SLO 的一部分，签合同时 ops 必须确认能跑得起
- **§9.2 KR8**："工程纪律不变量：核心 contract test 零回归" → drill 失败属于回归，必须立即 fix

## 7. SSOT 约束

| 不变量 | 守门 |
|---|---|
| drill artifact 是 readout，不喂回 OFFLINE gate / reward / Face | drill script 输出文件不被任何 owner 消费 |
| audit chain append-only | `lifeform_domain_figure.audit` 的现有契约（debt #23 闭合）+ 本 spec § 4 失败处置 |
| rollback 不破 frozen base 字节稳定 | F-D test + drill script step 7 |

## 8. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（Phase B 第 1 周） | 3 个 sh script 落地 + `tests/perf/test_production_rollback_drill.py` 骨架 + 本 spec v0.1 落档 |
| **ACTIVE**（Phase B 第 2-3 周） | 3 个 script 真跑 30 min 出真 artifact；test 真跑 Qwen byte-identical revert；ops dashboard 加 monthly drill status |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold land。
