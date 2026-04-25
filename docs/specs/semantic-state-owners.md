# Semantic State Owners Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R1, R2, R5, R7, R8, R11, R12, R15

## 要解决的问题

多轮交互中存在大量不能靠 prompt residue 临时重建的语义状态：计划、承诺、开放问题、用户模型、执行结果、信念假设、关系状态、目标价值与授权边界。它们必须由正式 owner 持有并通过不可变快照发布。

## 关键不变量

- 语义细节不存入 ETA / NL 本体；ETA 只消费 compact control advisory，NL 只决定更新时间尺度与沉淀路径。
- 九个 owner 都是独立 runtime slot，拥有自己的 frozen snapshot。
- 语义理解通过 typed `SemanticProposal` 进入 owner，禁止关键词规则直接驱动状态更新。
- 默认 package / synthetic path 使用 `NoOpSemanticProposalRuntime`，只发布低置信 observation，不伪造深语义。
- 每个 slot 有独立 wiring level / kill switch，迁移可回滚。

## 接口契约

新增 slots：

| Slot | Owner | Snapshot |
|------|-------|----------|
| `plan_intent` | `PlanIntentModule` | `PlanIntentSnapshot` |
| `commitment` | `CommitmentModule` | `CommitmentSnapshot` |
| `open_loop` | `OpenLoopModule` | `OpenLoopSnapshot` |
| `user_model` | `UserModelModule` | `UserModelSnapshot` |
| `execution_result` | `ExecutionResultModule` | `ExecutionResultSnapshot` |
| `belief_assumption` | `BeliefAssumptionModule` | `BeliefAssumptionSnapshot` |
| `relationship_state` | `RelationshipStateModule` | `RelationshipStateSnapshot` |
| `goal_value` | `GoalValueModule` | `GoalValueSnapshot` |
| `boundary_consent` | `BoundaryConsentModule` | `BoundaryConsentSnapshot` |

Proposal flow:

```text
substrate + memory + user_input
→ SemanticProposalRuntime
→ SemanticProposalBatch
→ owner-side merge in SemanticStateStore
→ public semantic snapshots
→ temporal / boundary / response / evaluation consumers
```

External adapter flow:

```text
tool/profile/task/reviewed-knowledge event
→ SemanticEventAdapter
→ AdapterSemanticProposalRuntime
→ SemanticProposalBatch
→ owner-side merge in SemanticStateStore
```

Adapters are structured-field mappers. They may map `status`, `consent_grants`, `consent_denials`, task state, or reviewed evidence fields to typed operations, but they must not inspect arbitrary text with keyword rules to decide behavior.

## ETA / NL 集成

- `TrackTemporalModule` 直接消费九个 semantic slots，并把它们压成 `semantic_pressure`，作为 control advisory 写入 public temporal description / feedback signal。
- `ResponseAssemblyModule` 消费九个 slots，发布 `semantic_record_counts`、`semantic_control_signal`、`semantic_residue_summary`。
- `BoundaryPolicyModule` 消费 `boundary_consent`，缺失授权或拒绝边界会提升澄清/边界约束。
- `EvaluationBackbone` 记录 semantic readout metrics，但不把 evaluation 变成学习源头。
- session-post request 携带 semantic state descriptions，供 background-slow 层沉淀与审计。
- `AgentSessionRunner` exposes a bounded pending external-event queue. Each turn drains the queue into `AdapterSemanticProposalRuntime`, so external events are consumed exactly once unless resubmitted.
- `BrainSession` exposes package-facing helper methods for tool result, profile/settings, task/calendar, and reviewed-knowledge events. These helpers enqueue structured events only; they do not mutate owner stores.

## 回滚

每个 semantic slot 都由 `FinalRolloutConfig` 暴露 wiring level，并支持 `kill_switches`。禁用某个 slot 时，下游通过 runtime placeholder 退化，不读取 owner 私有状态。

## 变更日志

- 2026-04-25: 初始版本，建立九个 semantic owner、typed proposal path、ETA/NL/response/evaluation/session-post 集成边界。
- 2026-04-25: 新增 external semantic adapters：tool result、profile/settings、task/calendar 与 reviewed knowledge 事件经 adapter runtime 转为 typed proposals 后进入 semantic owners。
