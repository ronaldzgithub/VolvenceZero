# Joint Entity Spec

> Status: draft
> Last updated: 2026-05-02
> 对应需求: R20, R16, R18, R19, R7, R14, R-PE, R1, R8, R15

## 要解决的问题

家庭、团队、同伴小组、会议不是 individual relationship states 的集合。一个 group 可以有自己的 continuity、norms、joint commitments、open loops、regime 和 repair history。当前 kernel 只能表达"我和某个 user"或"self/world track"，无法表达 group-as-entity。

R20 引入 `GroupModule`，把 group 作为 first-class adaptive owner，而不是从 per-person state 临时聚合。

## 关键不变量

- group state 不是 individual states 的 union。
- group owner 单写 group identity、membership、joint attention、group regime、joint commitments。
- group-level PE 独立于 per-person PE，但可以引用 per-person evidence。
- group commitments borrow from AAC lifecycle but remain group-scoped。
- group regime is persistent identity, not prompt label。
- group activation / membership evidence must come from Environment Event / role / common-ground snapshots or reviewed proposals, not keyword matching over utterance text.

## Owner / Timescale / Prediction Error / ETA Consumption

### Owner

`GroupModule` owns `groups` snapshot. It consumes R16 identity, R18 role, R19 common ground, commitment / open_loop evidence, and relationship summaries. It does not rewrite individual relationship owners.

### Timescale

- `online-fast`: detect active group scope and current joint attention.
- `session-medium`: update group continuity, joint commitments, group open loops after scene boundaries.
- `background-slow`: consolidate durable group norms, repeated coordination patterns, and group repair history.
- `rare-heavy`: offline group-policy artifact refresh only.

### Prediction Error

Group owner predicts:

- whether a joint commitment remains active
- whether current regime fits group context
- whether a response preserves group trust
- whether shared goal progress is recognized by the group

Outcome mismatch creates group-level social PE. This PE is not reducible to Alice/Bob individual PE; it enters credit as a group-scoped evidence stream.

### ETA Consumption

Group snapshot biases latent action selection: group repair, group grounding, individual aside, joint commitment update, or shared-goal progress. The metacontroller learns these switches in `z_t` / `beta_t`; renderer only expresses selected stance / plan.

## 工程挑战

- Need group_id formation without heuristic keyword rules.
- Group commitments must integrate with AAC lifecycle without making `commitment` second owner.
- Need evidence gates where group continuity diverges from any individual's state.
- Need rollback path so group owner can run SHADOW without affecting single-user companion gates.

## 算法候选

- Explicit Environment Event / host-provided group membership when available.
- Common-ground atoms and role snapshots as group activation evidence.
- AAC-style joint commitment lifecycle with group scope.
- PE-weighted group regime priors.
- Session-post reflection for group norm consolidation.

## 接口契约

```python
@dataclass(frozen=True)
class GroupIdentity:
    group_id: str
    member_ids: tuple[str, ...]
    display_name: str | None
    confidence: float
    evidence: tuple[str, ...]

@dataclass(frozen=True)
class GroupSnapshot:
    groups: tuple[GroupIdentity, ...]
    active_group_id: str | None
    joint_attention: tuple[str, ...]
    joint_commitments: tuple[str, ...]
    group_regime_id: str | None
    active_predictions: tuple[SocialPrediction, ...]
    description: str
```

Group-scoped commitments must preserve AAC alignment / advocacy / outcome semantics but add `group_id` and `member_ids` scope.

Implemented Phase 5 scaffold:

- `volvence_zero.social_cognition.GroupIdentity`: frozen shared contract for `group_id`, unique `member_ids`, optional display name, confidence, and evidence.
- `GroupSnapshot`: frozen shared contract for groups, active group id, joint attention, joint commitments, optional group regime, active predictions, and description.
- Contract tests enforce non-empty unique membership, active group id membership in `groups`, unique joint attention / commitments, and unique prediction ids.
- `GroupModule`: SHADOW owner scaffold registered in final wiring, publishing empty groups / joint attention / joint commitments and no active predictions.
- Explicit group state path: `GroupModule` can publish explicitly injected group identities, active group id, joint attention, joint commitments, and group regime id for tests/evidence. Default runtime remains empty and does not infer groups from text.
- Diagnostic downstream visibility: when `groups` is explicitly ACTIVE, `response_assembly.semantic_record_counts` includes `groups` and `group_joint_commitments` counts. Planner and renderer still do not consume group snapshots.
- Evidence report artifact: Social Cognition evidence report includes GROUP1, proving explicit group identity / joint commitment state is visible in response assembly diagnostics without renderer consumption.
- 2026-07-13 social-learning slice 3: `GroupModule` publishes typed
  `GROUP_COMMITMENT_DURABILITY` `active_predictions` for explicit active
  group state with joint attention / joint commitments; `SocialPredictionAggregateModule`
  forwards these predictions from public `GroupSnapshot.active_predictions`
  without reconstructing group state.
- 2026-07-16 G1: the durability predictions now settle (next-turn
  observed joint state, `SocialRecordStore`-held pending window) into
  `GroupSnapshot.settled_errors` plus a learned per-group
  `group_durability_score` that feeds future prediction confidence;
  `SocialPredictionErrorModule` forwards the settled errors verbatim.
  Wiring stays SHADOW pending the CP-18 evidence gate.

## 与其他能力域的关系

- R16 supplies member identities.
- R18 supplies active group audience / role state.
- R19 supplies group common ground.
- R14 supplies persistent regime identity, generalized to group regime.
- R7 self/relationship track becomes multi-agent and group-aware without collapsing into world/task track.
- Environment Interface supplies scene / membership / audience evidence; group owner learns group continuity above that boundary.

## 变更日志

- 2026-07-16: G1 group-level PE settlement 学习闭环。`GroupModule` 复用
  ToM / common-ground 的 `settle_pending_predictions` 结算机制：
  GROUP_COMMITMENT_DURABILITY 预测停放在 `SocialRecordStore`
  （`pending_group_predictions`，单写者 GroupModule），下一 turn 用
  observed joint state 的 typed summary（commitments / attention /
  regime 计数，语义相似度、无关键词）结算。结算结果驱动有界 learned
  per-group durability score（[0,1]，先验 0.5：CONFIRMED +0.10 /
  DISCONFIRMED −0.20，与 ToM confidence 表同一不对称性），该 score 成为
  下一次 durability 预测的 confidence（受 group identity confidence
  上界钳制），并作为 `GroupSnapshot.group_durability_score` 发布。
  settled errors 发布于 `GroupSnapshot.settled_errors`，由
  `SocialPredictionErrorModule` 直接转发（新增 `groups` 依赖，不做下游
  重建）。final wiring 注入 `turn_index`。无 store 时保持无状态 scaffold。
  默认 wiring 仍 SHADOW；这条 settled PE 流即 CP-18 ACTIVE 判据所需的
  group-level PE 证据源。测试：`tests/test_social_group_settlement.py`。
- 2026-07-14: CP-18 frame membership + R14 group regime（GAP-08）。
  `GroupModule` 不再丢弃 upstream：从 `multi_party_identity` owner 发布的
  canonical frame scope（speaker + addressees + audience）派生 group
  identity——≥3 个不同参与者时生成确定性 `frame-group:<sorted members>` id
  （同一成员集永远同 id，R14 持久身份；单方兼容 frame 永不成组，禁止从
  文本猜测）。group regime 经 `SocialRecordStore.record_group_regime /
  group_regime_for` 持久化（单写者 GroupModule）：orchestrator 显式注入的
  regime 被记录，后续 turn 重建的模块为同一 group 自动 rehydrate，不跨
  group 泄漏。final wiring 注入 `record_store=social_record_store`。默认
  wiring 仍 SHADOW；ACTIVE 判据不变（group PE 有不可归约增量）。测试：
  `tests/test_social_group.py`。
- 2026-07-13: social-learning slice 3. Group snapshots now emit typed group
  durability predictions and the social prediction aggregate forwards them.
  Tests: `tests/test_social_group.py` + `tests/test_final_wiring.py`.
- 2026-05-02: R20 Phase 5 slice 5 landed: Social Cognition evidence report GROUP1 gate covers group diagnostic visibility.
- 2026-05-02: R20 Phase 5 slice 4 landed: group identity count and group joint commitment count surface in `response_assembly.semantic_record_counts` as diagnostics only; no planner / renderer consumption.
- 2026-05-02: R20 Phase 5 slice 3 landed: `GroupModule` supports explicit group state injection while default runtime remains empty.
- 2026-05-02: R20 Phase 5 slice 2 landed: `GroupModule` SHADOW owner scaffold registered in final wiring, defaulting to empty group state.
- 2026-05-02: R20 Phase 5 slice 1 landed in `vz-contracts`: `GroupIdentity` / `GroupSnapshot` contracts with membership, joint attention / commitment, active group, and prediction-id validation.
- 2026-05-02: 补充 Environment Interface 依赖：group owner 消费 scene / membership / audience evidence，不从 raw text 关键词形成 group。
- 2026-05-02: 初始 draft，冻结 GroupModule as Social Cognition Learning Layer Phase 5。
