# Common Ground Spec

> Status: draft
> Last updated: 2026-05-02
> 对应需求: R19, R16, R17, R18, R5, R6, R-PE, R8, R15

## 要解决的问题

系统可以记住事实，但还不能表达"我和 Alice 都知道 X"、"Alice 和 Bob 共享 Y"、"这个团队已经共同承诺 Z"。没有 common ground，deictic references、"we"、"as before"、ellipsis、indirect repair 都只能靠文本表面猜测。

R19 把 dyad / group mutual knowledge 作为独立 learned owner。它不是 individual memory 的派生，也不是 renderer 的上下文拼接。

## 关键不变量

- common ground 由 `CommonGroundModule` 单写，不由 memory / user_model / renderer 重建。
- common ground keyed by dyad_id or group_id，不直接存在 global scope。
- 递归深度默认 bounded at `k=2`，避免无限 ToM。
- reference-resolution failure 必须产生 common-ground PE。
- common-ground snapshot 可被 controller / regime / planner 消费，但不可作为学习源替代 PE。
- common-ground owner 消费 Environment Event / role 提供的 audience 与 scope，不从文本表面或 renderer 输出重建"谁共享了什么"。

## Owner / Timescale / Prediction Error / ETA Consumption

### Owner

`CommonGroundModule` 是唯一 owner。它消费 R16 identity、R18 role、R17 belief summaries、memory retrieval、repair / clarification outcomes，并发布 `common_ground` snapshot。

### Timescale

- `online-fast`: 判断当前 reference 是否有足够 shared context。
- `session-medium`: scene 结束时沉淀 mutual acceptance，例如"我们刚确认了计划 A"。
- `background-slow`: consolidates durable dyad / group common-ground atoms through reflection。

### Prediction Error

Common ground owner 预测：

- 当前 addressee 是否能解析 pronoun / ellipsis / "we"
- 当前 group 是否共享某个 assumption
- 当前 response 是否需要 re-grounding before action

失败的 reference、clarification request、repair signal、或者用户纠正"我不是那个意思"产生 common-ground PE。

### ETA Consumption

Common-ground state should influence latent action choices: elaborate, compress, ask grounding question, repair reference, or proceed. It must not be implemented as phrase matching for "we" or "yesterday". The renderer may only express a continuity / grounding section selected by planner from owner snapshots.

## 工程挑战

- Dyad / group identifiers require R16 / R20.
- Current memory summaries are individual / track based, not mutual-knowledge based.
- Need PE wiring from clarification / repair / reference failure into common-ground owner.
- Common-ground atoms must be immutable, versioned, and auditable to avoid hidden shared-context drift.

## 算法候选

- Bounded mutual-knowledge atoms with `recursion_depth=2`。
- Reference-resolution predictions over role + memory + ToM summaries。
- Session-post reflection to promote accepted atoms。
- Contradiction-aware retirement when a shared assumption is denied.

## 接口契约

```python
@dataclass(frozen=True)
class CommonGroundAtom:
    atom_id: str
    scope_id: str
    scope_kind: str  # dyad | group
    summary: str
    recursion_depth: int
    confidence: float
    accepted_by_ids: tuple[str, ...]
    evidence: tuple[str, ...]

@dataclass(frozen=True)
class CommonGroundSnapshot:
    dyad_atoms: tuple[CommonGroundAtom, ...]
    group_atoms: tuple[CommonGroundAtom, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
```

Implemented Phase 4 scaffold:

- `volvence_zero.social_cognition.CommonGroundAtom`: frozen shared contract for dyad/group mutual-knowledge atoms.
- `MAX_COMMON_GROUND_RECURSION_DEPTH = 2`: hard upper bound for recursion depth.
- `CommonGroundAtom.scope_kind` accepts only `dyad` or `group`, never global / interlocutor-only scope.
- `accepted_by_ids` must be non-empty and unique.
- `CommonGroundSnapshot` validates dyad atoms and group atoms are kept in the correct buckets and active prediction ids are unique.
- `CommonGroundModule`: SHADOW owner scaffold registered in final wiring, publishing empty dyad/group atoms and no active predictions.
- Explicit atom path: `CommonGroundModule` can publish explicitly injected dyad/group atoms for tests and evidence. Default runtime remains empty and does not infer common ground from text.
- Diagnostic downstream visibility: when `common_ground` is explicitly ACTIVE, `response_assembly.semantic_record_counts` includes `common_ground` as dyad + group atom count. Planner and renderer still do not consume common-ground snapshots.
- Evidence report artifact: Social Cognition evidence report includes G1, proving explicit dyad/group common-ground atoms are visible in response assembly diagnostics without renderer consumption.
- Structured proposal path: `LLMCommonGroundProposalRuntime` consumes bounded JSON output and emits validated dyad/group common-ground proposals; low-confidence / malformed output publishes no atoms.
- Owner consumption path: `CommonGroundModule` can consume structured common-ground proposals while preserving explicit atom injection and default empty SHADOW behavior.
- Evidence gates G2/G3: social cognition evidence now covers structured common-ground runtime and reference/repair common-ground probes.

## 与其他能力域的关系

- R16 provides dyad identity keys.
- R17 belief owner supplies individual belief state but does not own mutual knowledge.
- R18 role determines which common-ground scope is active.
- R20 group owner uses group common ground for joint commitment but does not own common-ground atoms.
- R5/R6 memory continuum provides source evidence and slow consolidation path.
- Environment Interface supplies event provenance, audience scope, and outcome evidence for reference-resolution PE.

## 变更日志

- 2026-05-02: R19 Phase 4 slice 6 landed: structured common-ground proposal runtime + owner consumption + final-wiring diagnostics + G2/G3 evidence gates.
- 2026-05-02: R19 Phase 4 slice 4 landed: common-ground atom count surfaces in `response_assembly.semantic_record_counts` as diagnostics only; no planner / renderer consumption.
- 2026-05-02: R19 Phase 4 slice 5 landed: Social Cognition evidence report G1 gate covers common-ground diagnostic visibility.
- 2026-05-02: R19 Phase 4 slice 3 landed: `CommonGroundModule` supports explicit dyad/group atom injection while default runtime remains empty.
- 2026-05-02: R19 Phase 4 slice 2 landed: `CommonGroundModule` SHADOW owner scaffold registered in final wiring, defaulting to empty dyad/group common-ground atoms.
- 2026-05-02: R19 Phase 4 slice 1 landed in `vz-contracts`: `CommonGroundAtom` / `CommonGroundSnapshot` contracts with bounded recursion depth and dyad/group scope validation.
- 2026-05-02: 补充 Environment Interface 依赖：common ground 的 active scope 来自 canonical conversational frame / role snapshot。
- 2026-05-02: 初始 draft，冻结 CommonGroundModule as Social Cognition Learning Layer Phase 4。
