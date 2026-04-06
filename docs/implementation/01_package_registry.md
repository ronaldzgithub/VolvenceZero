# Next-Gen Adaptive Agent 收敛包注册表

> Status: draft
> Last updated: 2026-04-06
> Scope: package registry for final-state rollout

## 1. 使用方式

本注册表是实施层的总索引，用来回答三个问题：

1. 当前包的主 owner、slot 和主要 consumer 是什么。
2. 当前包处于什么接线级别。
3. 当前包何时可以扩大范围，何时必须回滚。

所有包默认遵循：

- `owner` 唯一
- `slot` 唯一
- `shadow` 先于 `active`
- contract 先于全量功能
- spec 与 `docs/DATA_CONTRACT.md` 同步优先

## 2. 包总表

| Package | 主 owner | 主要 slot | 主要 consumer | 接线初始级别 | 推荐完成级别 | 前置包 |
|---|---|---|---|---|---|---|
| `P00` Runtime Kernel | Orchestrator / Runtime | runtime infra | 全体模块 | `active` | `active` | 无 |
| `P01` Substrate Adapter | SubstrateModule | `substrate` | `temporal_abstraction`, `memory` | `shadow` | `shadow` | `P00` |
| `P02` Memory Continuum | MemoryModule | `memory` | `dual_track`, `regime`, `reflection` | `shadow` | `active` | `P00` |
| `P03` Dual Track Core | DualTrackModule | `dual_track` | `credit`, `evaluation`, `regime` | `shadow` | `active` | `P00` |
| `P04` Regime Identity | RegimeModule | `regime` | `evaluation`, `orchestrator` | `shadow` | `active` | `P00`, `P02`, `P03` |
| `P05` Evaluation Backbone | EvaluationModule | `evaluation` | `credit`, `orchestrator`, rollout gate | `active` | `active` | `P00` |
| `P06` Credit And Gate | CreditModule | `credit` | `reflection`, rollout gate | `shadow` | `active` | `P03`, `P05` |
| `P07` Reflection Writeback | ReflectionModule | `reflection` | `memory`, `credit`, `temporal_abstraction` | `disabled` | `shadow` | `P02`, `P05`, `P06` |
| `P08` Temporal Abstraction Interface | TemporalModule | `temporal_abstraction` | `dual_track`, `regime`, `credit` | `disabled` | `shadow` | `P00`, `P01` |
| `P09` Final Wiring And Rollout | Integration owner | 全链路 | 全系统 | `disabled` | `active` | `P02`-`P08` |

P09 当前实现口径：

- 默认 active: `substrate` / `memory` / `dual_track` / `evaluation` / `regime` / `credit`
- 默认 shadow: `reflection` / `temporal_abstraction`
- 通过 kill switch 将指定模块回退到 `disabled`
- acceptance report 负责统一汇总 active / shadow / disabled 状态与 rollout 建议

## 3. 接线级别定义

| Level | 含义 | 允许行为 | 不允许行为 |
|---|---|---|---|
| `disabled` | 包实现存在但不在主运行链 | standalone、自测、schema 校验 | 驱动主决策、写回生产数据 |
| `shadow` | 包消费正式上游，但输出不主导行为 | 事件记录、评估、对照 | 静默替代当前正式 owner |
| `active` | 包进入正式快照链 | 作为正式 consumer / publisher | 越过门控直接扩大自修改范围 |

## 4. 扩大范围门槛

| Package | 从 `disabled`/`shadow` 提升到更高等级前必须满足 |
|---|---|
| `P01` | substrate contract 稳定，已明确是否支持残差流级观测，未承诺不可实现 hook |
| `P02` | 记忆写入只能走 owner API；检索与摘要都可观测 |
| `P03` | 双轨状态在记忆、评估、信用里都保持显式分离 |
| `P04` | regime 不是 prompt 标签；可回忆、可评估、可解释 |
| `P05` | 基本告警、评估记录、证据链可工作 |
| `P06` | 门控守卫可阻止越权自修改；所有修改都有审计轨迹 |
| `P07` | writeback 可禁用、可审计、可回滚 |
| `P08` | temporal 快照 shape 稳定，standalone 和 shadow 模式可运行 |
| `P09` | 所有核心 slot 都已有 owner 和回滚点，且 shadow 数据表明可接线 |

## 5. 总接线前必须完成的统一检查

- 所有 slot 已在注册表中有唯一 owner。
- 所有包文档都明确了未接线完成态。
- 所有 `shadow` 包都有事件日志和评估证据。
- 所有异步写回路径都有 kill switch。
- 所有自修改路径都有 gate 和 rollback checkpoint。

## 6. 每包最低交付字段

每个收敛包子计划都必须显式填出以下字段：

- `primary_owner`
- `primary_slot`
- `primary_consumer`
- `capability_domains`
- `specs_to_sync`
- `initial_wiring_level`
- `target_wiring_level`
- `exit_conditions`
- `rollback_trigger`
- `rollback_action`

## 7. 风险聚类

### 7.1 高风险共享契约

- `P00`: runtime base class / propagate 语义
- `P01`: `substrate` contract
- `P05`: `evaluation` schema 和 gate feedback
- `P09`: 全链路接线

这些包必须避免与其他 owner 的行为变更混改。

### 7.2 中风险慢回路

- `P06`: credit 与 gate
- `P07`: reflection writeback
- `P08`: temporal abstraction

这些包必须默认先以 `disabled` 或 `shadow` 形态运行。

### 7.3 低至中风险状态 owner

- `P02`: memory
- `P03`: dual-track
- `P04`: regime

这些包可以优先做 contract 和 state publishing，再逐步扩大 consumer。

## 8. 里程碑映射

虽然本实施不按 `M0-M6` 排序推进，但仍保留映射关系，便于和 `docs/prd.md` 对齐：

| Package | 主要对应里程碑 / 能力域来源 |
|---|---|
| `P00` | M0 / 契约式运行时 |
| `P01` | M0 + R2 stable substrate 前置 |
| `P02` | M1 / 连续记忆 |
| `P03` | M2 / 双轨学习 |
| `P04` | M6 / 认知 Regime |
| `P05` | M6 / 评估体系，但前置实施 |
| `P06` | M5 / 信用分配与门控自修改 |
| `P07` | M1 + M5 / 慢反思写回 |
| `P08` | M3 / 时间抽象与内部控制 |
| `P09` | M0-M6 integration / rollout |

## 9. 相关文档

- [00_master_plan.md](./00_master_plan.md)
- [packages/P00_runtime_kernel.md](./packages/P00_runtime_kernel.md)
- [packages/P09_final_wiring_and_rollout.md](./packages/P09_final_wiring_and_rollout.md)
