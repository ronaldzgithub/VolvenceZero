# Growth-Advisor Monthly Report Spec

> Status: scaffold v0.2 (SHADOW; schema v0.2)
> Owner: growth-advisor-pilot-packet G-D (debt #67 + #49)
> Implementation: [`packages/lifeform-service/src/lifeform_service/monthly_report_owner.py`](../../packages/lifeform-service/src/lifeform_service/monthly_report_owner.py)

## 1. 范围

P2 卖点"月报让我老板满意"的 schema + owner + 端点 + 版本化策略。月报字段必须**长期稳定**（客户老板每月看同一份格式），同时给 customer 提供透明的可量化指标。

## 2. Schema (v0.2)

`MonthlyReportSnapshot` 字段表（参考 [`monthly_report_owner.py`](../../packages/lifeform-service/src/lifeform_service/monthly_report_owner.py)）：

| 字段 | 类型 | mock 值 | 含义 |
|---|---|---|---|
| `report_schema_version` | str | `"v0.2"` | schema 版本 |
| `tenant_id` | str | `"brand_a"` | 客户 id |
| `month_iso` | str | `"2026-04"` | 报告月份 |
| `period_start_ms` / `period_end_ms` | int | timestamps | 报告周期 |
| `end_user_count_total` | int | 487 | 全 tenant 端用户数 |
| `end_user_count_active` | int | 312 | 本月有交互的端用户数 |
| `new_end_user_count` | int | 64 | 本月新增端用户数 |
| `average_turns_per_active_user` | float | 23.4 | 活跃端用户平均 turn 数 |
| `boundary_trigger_total` | int | 187 | 4 boundary 触发总次数 |
| `boundary_trigger_per_policy` | dict | `{"bp-no-hard-sell": 78, "bp-no-overclaim": 54, "bp-no-flooding": 32, "bp-no-judgmental": 23}` | per-boundary 触发数（"AI 不硬推销了 X 次"是合规增长卖点）|
| `rupture_count` | int | 41 | rupture 触发数 |
| `repair_count` | int | 36 | repair 触发数 |
| `repair_rate` | float | 0.878 | rupture → repair 完成率（关系健康度核心指标） |
| `archetype_distribution` | dict | `{"anxious": 0.34, "comparing": 0.18, "standard_seeking": 0.22, "venting": 0.16, "product_seeking": 0.10}` | 5 archetype 分布 |
| `handoff_triggered_count` | int | 12 | handoff 触发数 |
| `handoff_completed_count` | int | 11 | SE 真接手数 |
| `handoff_p99_seconds` | float | 24.7 | handoff 触发到接手 P99 |
| `protocol_phase_cohort_active_counts` | dict | `{"icebreaker": 64, "baseline": 51, "empathy": 47, "pain_mining": 39, "rapport": 33, "targeted_advice": 31, "summary_hook": 47}` | onboarding-arc phase 活跃数。phase id 来自 `BehaviorProtocol.TemporalArc.progression_signals` snapshot（PE-driven）。replaces v0.1 `day_cohort_active_counts`（calendar-day routing 已 deprecate）|
| `deleted_end_user_count` | int | 3 | 本月被删除的 end_user 数（debt #49 GDPR/PIPL 合规审计） |
| `deletion_event_count` | int | 5 | 本月 deletion 事件总数（同一 end_user 多次删除算多次） |

## 3. Owner 归属决策

| 候选 | 决策 |
|---|---|
| `lifeform-service` 子模块 `monthly_report_owner.py` | ✅ **采用** |
| 新 wheel `lifeform-monthly-report` | ❌ 不采用（避免破坏 R8 边界，月报是 service-layer 派生） |

## 4. 版本化策略

- `report_schema_version = "v0.2"` 锁定（contract test 守门）
- 加新字段：bump 到 `v0.3`，向后兼容（旧月报仍可读，新字段空）
- 删字段 / 改字段语义：bump 主版本到 `v1.0`，附 migration shim
- customer-facing PDF 渲染按 schema_version 路由模板（v0.x 用 template-v0，v1.x 用 template-v1）

### Schema 版本日志

- **v0.1** 初始 scaffold
- **v0.2** (2026-05-14)
  - `day_cohort_active_counts` → `protocol_phase_cohort_active_counts`（calendar-day routing deprecate；phase 由 `BehaviorProtocol.TemporalArc.progression_signals` PE-driven 决定）
  - 加 `deleted_end_user_count` + `deletion_event_count`（debt #49 GDPR/PIPL 合规审计；不泄露被删 end_user 内容）

## 5. Aggregation 公式

```
end_user_count_active = sum(turn_count > 0 for end_user in tenant)
average_turns_per_active_user = total_turns / end_user_count_active
boundary_trigger_per_policy = sum(boundary_owner.snapshot[policy] per end_user)
repair_rate = repair_count / max(1, rupture_count)
archetype_distribution = normalize(sum(ArchetypeStateSnapshot.current per end_user))
```

公式集中在 `MonthlyReportOwner.aggregate(MonthlyReportInputs) -> MonthlyReportSnapshot`，下游不重算。

## 6. HTTP 端点 (admin scope)

`GET /v1/tenants/{tid}/admin/monthly-report?month=YYYY-MM` → 返回 `MonthlyReportSnapshot.to_json()`

权限：tenant_admin（不是 end_user）。两层 scope（debt #46 / F-B SHADOW）必填。

## 7. 客户尽调材料 1-page 摘要

(待真客户跑过 30 天试点后回填；模板)

> 本月，"妈妈A" 顾问服务你的 487 位端用户，其中 312 位有交互；月均交互轮次 23.4。AI 在 187 次互动中触发反销售 / 反过度承诺等 boundary 政策（占总互动 0.6%），保护用户免受过度营销；其中 78 次拒绝硬推销，54 次拒绝过度承诺。关系健康指标：重大对话失误 41 次，其中 36 次成功修复（修复率 87.8%）。客户类型分布：焦虑型 34%、对照型 18%、标准型 22%、宣泄型 16%、直接问产品型 10%。本月 12 次 handoff 触发，11 次成功转人工接管，转接 P99 延迟 24.7 秒。day-cohort 活跃分布显示 day7+ 老用户保持 47 位活跃（持续运营成功率高）。

## 8. SSOT 约束

| 不变量 | 守门 |
|---|---|
| `MonthlyReportOwner` 是月报 SSOT | downstream 只读 snapshot |
| 不读 `evidence_root_dir` 直接 aggregate | `monthly_report_owner.py` AST 扫禁止 import `evidence_root_dir` |
| schema 字段稳定 | [`tests/contracts/test_monthly_report_schema_stability.py`](../../tests/contracts/test_monthly_report_schema_stability.py) 守门 |
| 双层 scope 删除联动 | (debt #46 + #49) 删 tenant → 月报缓存对应清空 |

## 9. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W3） | 本 spec + `MonthlyReportOwner` 模块骨架 + contract test |
| **ACTIVE**（W6） | 真 owner snapshot 接入 + admin endpoint + PDF 渲染 |

## 变更日志

- 2026-05-13: v0.1 SHADOW spec + mock 月报字段表。
