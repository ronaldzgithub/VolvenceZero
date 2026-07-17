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

Character chapter adapter flow（2026-07-13）:

```text
ReviewedChapterExperience / CharacterSemanticEventBundle
→ CharacterChapterSemanticAdapter
→ AdapterSemanticProposalRuntime
→ SemanticProposalBatch
→ SemanticStateStore single-writer merge
```

`CharacterSemanticEvent` 是 reviewed chapter artifact 的 typed proposal source，不是新的 semantic owner。它必须携带 `target_slot`（9 slots 闭集）、`operation`、`summary`、`detail`、`confidence`、`evidence_locator`。adapter 只做结构字段映射，不能读取原文小说、不能用关键词决定 slot 或 operation。角色 vertical 可生成这些 events，但最终状态仍由 `SemanticStateStore` 持有并发布快照。

## ETA / NL 集成

- `TrackTemporalModule` 直接消费九个 semantic slots，并把它们压成 `semantic_pressure`，作为 control advisory 写入 public temporal description / feedback signal。
- `ResponseAssemblyModule` 消费九个 slots，发布 `semantic_record_counts`、`semantic_control_signal`、`semantic_residue_summary`。
- `BoundaryPolicyModule` 消费 `boundary_consent`，缺失授权或拒绝边界会提升澄清/边界约束。
- `EvaluationBackbone` 记录 semantic readout metrics，并发布 `semantic_spine_coverage` / `cognitive_loop_readiness` 作为窄 cognitive loop 的证据读数；evaluation 只消费 owner 快照，不把 evaluation 变成学习源头。
- session-post request 携带 semantic state descriptions，供 background-slow 层沉淀与审计。
- `AgentSessionRunner` exposes a bounded pending external-event queue. Each turn drains the queue into `AdapterSemanticProposalRuntime`, so external events are consumed exactly once unless resubmitted.
- `BrainSession` exposes package-facing helper methods for tool result, profile/settings, task/calendar, and reviewed-knowledge events. These helpers enqueue structured events only; they do not mutate owner stores.

## 回滚

每个 semantic slot 都由 `FinalRolloutConfig` 暴露 wiring level，并支持 `kill_switches`。禁用某个 slot 时，下游通过 runtime placeholder 退化，不读取 owner 私有状态。

## 验收读数

当 `relationship_state`、`goal_value`、`boundary_consent`、`commitment`、`execution_result` 与 `evaluation` 同时 ACTIVE 时，`FinalAcceptanceReport` 必须能看到：

- `semantic_spine_coverage = 1.0`
- `cognitive_loop_readiness` 已发布
- session / cross-session report 中的 `semantic_spine_readiness` 趋势由 `cognitive_loop_readiness` 派生，用于判断地基是否退化；`semantic_spine_coverage` 只作为完整性验收，不混入趋势
- dialogue benchmark case report、emergence dashboard 与 paper-suite metric values 汇总 `mean_semantic_spine_coverage` / `mean_cognitive_loop_readiness`，作为产品对话回归层和证据产物层的地基证据
- NL essence assessment 发布 `semantic-spine-ready` gate；该 gate 先作为审计证据，不进入默认 required gate 列表
- `claim_companion_stateful_relationship` 的当前轻量 verdict 消费 `semantic-spine-ready` 与 dashboard 读数，作为完整 companion 证据前的状态感知地基门
- paper-suite manifest 将 canonical semantic spine 指标列入 secondary metrics；companion verdict 优先消费 repeated-run summary，reference dashboard 仅作 fallback
- `semantic_state.quality` 提供 proposal-level quality harness，先用于 `boundary_consent` / `goal_value` 的 precision / recall / false-positive / fallback 评估；它只评估 proposal runtime，不写 owner store，并发布 shadow-only `would_block` / `would_allow` / gate reason 读数
- dialogue paper-suite export 可将 proposal quality shadow report 作为 `semantic_proposal_quality_shadow.json` sidecar 与 `EvidenceBundle.reference_artifacts` 条目导出；该 artifact 标记为 non-gating，不改变 owner apply 或 claim verdict

该检查只验证 owner 快照是否形成窄 cognitive loop 证据，不把 readiness 当作学习奖励，也不允许 evaluation 重建 owner 内部状态。

## 变更日志

- 2026-07-17: G2 LLM proposal 覆盖 9/9。`_GENERIC_LLM_SLOT_IDS` 从 4 slot 扩到 8（`plan_intent` / `open_loop` / `execution_result` / `belief_assumption` 加入既有 JSON-schema generic 路径；commitment 仍走专用分类器，合计 9/9 全部 semantic owner 具备 typed LLM proposal source）。per-slot 语义说明集中在 `_GENERIC_SLOT_SEMANTIC_HINTS`（llm-prompt-centralization；原四 slot 的 prompt 字节不变）。owner 单写者、`min_proposal_confidence` 过滤、unparseable→NoOp fail-safe 均不变。测试：`tests/test_llm_semantic_runtime.py` 新四 slot 参数化用例 + hint-line 边界用例。
- 2026-07-14: CP-12 第二波 publisher 接线（GAP-05）。`plan_intent`（kind
  `PLAN_INTENT_PROGRESS`, track world）/ `open_loop`（`OPEN_LOOP_CLOSURE`,
  world）/ `belief_assumption`（`BELIEF_ASSUMPTION_STABILITY`, world）/
  `user_model`（`USER_MODEL_PACING`, self）开始在快照发布
  `owner_prediction_signals`，机制与 first wave 完全一致（store-held v2
  learned forecaster + owner 自 settle）。`user_model` 只预测自身 aggregate
  pacing/stability readout，不与 ToM 四 owner 重复拥有对他人的
  belief/intent/feeling/preference。PE settlement 覆盖扩至 9 slot。测试：
  `tests/contracts/test_owner_prediction_signal.py`（ALL_WAVES 参数化）。
- 2026-07-12: CP-12 owner prediction signal contract。`SemanticOwnerModule` 新增
  `owner_prediction_kind` / `owner_prediction_track` 类属性与
  `_owner_prediction_signals(...)` 助手；五个 first-wave owner（commitment /
  relationship_state / goal_value / boundary_consent / execution_result）在快照
  发布 `owner_prediction_signals`：每轮签发一条对自身 compact readout 的
  persistence-prior v1 预测，并对上一轮 pending 预测由 owner 自己 settle
  （observed readout + outcome evidence）。pending 预测与 id 序列由
  `SemanticStateStore` 持有（owner 模块每轮重建，store 是 durable 组件）。
  mismatch 只由 PE owner 计算（见 `prediction-error-loop.md` 同日条目）；
  消费者无需读取 owner 内部字段即可完成 settlement 消费。第二波
  （plan_intent / open_loop / belief_assumption / user_model）kind 已在闭集
  enum 预留，publisher 于 2026-07-14 接线（见上方条目）。
- 2026-07-13: 登记 character chapter adapter flow。逐章主观烘焙的
  `CharacterSemanticEventBundle` 只能作为 typed proposal source，仍由
  `SemanticStateStore` 单写者合并；禁止用原文关键词或角色 vertical 直写 9 个
  semantic owners。
- 2026-05-03: 新增 `semantic_state.quality` proposal quality harness，首批覆盖 `boundary_consent` / `goal_value` scripted LLM cases，用于在 owner 合并前评估 typed proposal 输入质量；shadow gate 只报告 would-block，不阻断 runtime。
- 2026-05-03: dialogue paper-suite export 新增 non-gating `semantic_proposal_quality_shadow.json` sidecar，并把同一 payload 挂入 evidence bundle reference artifacts。
- 2026-05-03: Commitment / OpenLoop / BoundaryConsent / GoalValue / RelationshipState 增加 owner-side lifecycle / continuity readouts；`LLMSemanticProposalRuntime` 最小扩展到 `boundary_consent`、`goal_value` 的 schema-bound typed proposal 路径，非目标 slot 继续 delegate。
- 2026-05-03: `clone_semantic_store` 开始保留 lifecycle / follow-up policy / typed outcome maps，避免跨上下文复制时丢失 owner-side continuity evidence。
- 2026-05-03: paper-suite manifest 将 canonical semantic spine coverage / cognitive loop readiness 纳入 secondary metrics，companion verdict 优先消费 repeated-run summary。
- 2026-05-03: `claim_companion_stateful_relationship` 当前轻量 verdict 接入 `semantic-spine-ready` 与 dashboard 读数；retain 仍需 cross-session gate，避免把单轮读数夸大为完整 companion 证明。
- 2026-05-03: NL essence assessment 新增 `semantic-spine-ready` gate，将 semantic spine 读数提升为 paper-suite 审计门，但暂不加入默认 required gate。
- 2026-05-03: Dialogue benchmark case report、emergence dashboard 与 paper-suite metric values 开始汇总 semantic spine coverage / cognitive loop readiness，让对话回归和证据产物能直接观察认知地基状态。
- 2026-05-03: `cognitive_loop_readiness` 进入 session / cross-session 趋势，`EvolutionJudgement` 在 `semantic_spine_readiness` 明显退化时回滚，避免扩能力掩盖认知地基退化。
- 2026-05-03: `FinalAcceptanceReport` 开始要求 ACTIVE 核心 semantic spine 发布 `semantic_spine_coverage` 与 `cognitive_loop_readiness`，作为继续扩能力前的验收门槛。
- 2026-05-03: `EvaluationBackbone` 新增 `semantic_spine_coverage` 与 `cognitive_loop_readiness`，基于 `relationship_state`、`goal_value`、`boundary_consent`、`commitment`、`execution_result` 五个 owner 的公开快照评估窄 cognitive loop 的地基成熟度。
- 2026-05-03: 四个情绪决策相关 owner 增加 owner-side readout：
  - `relationship_state` 发布 `emotional_load`、`repair_need`、`trust_delta`、`attunement_gap`、`stabilization_need`
  - `goal_value` 发布 `value_conflict`、`decision_readiness`、`active_tradeoff_count`、`reversibility_need`、`goal_shift_pressure`
  - `boundary_consent` 发布 `autonomy_risk`、`consent_clarity`、`professional_scope_pressure`、`overreach_risk`
  - `user_model` 发布 `preferred_support_pacing`、`decision_style`、`overwhelm_pattern_strength`，并开始从 typed profile proposals 沉淀 `durable_goals`
  - `response_assembly.support_before_decision_pressure` 优先消费这些 owner-side readouts，ETA 只消费压缩后的 action-family advisory，不拥有语义事实
- 2026-04-25: 初始版本，建立九个 semantic owner、typed proposal path、ETA/NL/response/evaluation/session-post 集成边界。
- 2026-04-25: 新增 external semantic adapters：tool result、profile/settings、task/calendar 与 reviewed knowledge 事件经 adapter runtime 转为 typed proposals 后进入 semantic owners。
