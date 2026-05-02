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

## 与其他能力域的关系

- R16 supplies member identities.
- R18 supplies active group audience / role state.
- R19 supplies group common ground.
- R14 supplies persistent regime identity, generalized to group regime.
- R7 self/relationship track becomes multi-agent and group-aware without collapsing into world/task track.
- Environment Interface supplies scene / membership / audience evidence; group owner learns group continuity above that boundary.

## 变更日志

- 2026-05-02: 补充 Environment Interface 依赖：group owner 消费 scene / membership / audience evidence，不从 raw text 关键词形成 group。
- 2026-05-02: 初始 draft，冻结 GroupModule as Social Cognition Learning Layer Phase 5。
