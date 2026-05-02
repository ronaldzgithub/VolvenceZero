# Social Cognition Learning Layer Implementation Plan

> Status: draft
> Last updated: 2026-05-02
> Scope: 分阶段实现 R16-R20，把当前 single-other kernel 扩展为多人社会认知学习层。
> Parent: `docs/next_gen_emogpt.md`
> Specs: `docs/specs/social_cognition/*.md`
> Contract: `docs/DATA_CONTRACT.md` §6.X

---

## 0. 目标与非目标

### 0.1 目标

实现 Social Cognition Learning Layer，而不是多人 CRM schema。每个阶段都必须满足：

- 有唯一 owner 和 immutable snapshot
- 有 online-fast / session-medium / background-slow update path
- 有 pre-action social prediction
- 有 outcome-driven social prediction error
- 有 ETA consumption test，证明 controller / regime / planner 读取 snapshot，而不是 renderer 从文本硬分支
- 有 WiringLevel.SHADOW → ACTIVE → rollback path

### 0.2 非目标

- 不在本轮一次性改代码。
- 不用 renderer 文案规则替代 social cognition owner。
- 不让 LLM classifier 成为 owner。LLM 只产生 typed proposal。
- 不把 evaluation 当学习源。evaluation 只做 readout / gate，PE 仍是学习源头。

---

## 1. Phase 1 — R16 Multi-Party Identity Learning

**预计**：3-4 周，4-6 个 PR。

**当前状态（2026-05-02）**：slices 1-10 已落地为 SHADOW-compatible scaffold，不改变默认 companion 行为。已完成 shared contracts、`MultiPartyIdentityModule` SHADOW owner、`ResponseContext` / `AgentTurnResult` scope readout、memory subject/audience scope、MemoryModule ACTIVE identity-scope consumption path、social prediction / social PE SHADOW scaffolds、manual social PE → credit carry path、Companion Evidence C5 default-social-scope gate 与 CLI JSON artifact 断言。

### 1.1 Scope

把 single-other identity assumption 改为 keyed identity scaffold：

- `multi_party_identity` planned slot
- `interlocutor_models: Mapping[str, UserModelSnapshot]`
- `relationship_states: Mapping[str, RelationshipStateSnapshot]`
- `interlocutor_states: Mapping[str, InterlocutorState]`
- `MemoryEntry.subject_ids`
- `MemoryEntry.audience_ids`
- `ResponseContext.active_speaker_id`
- `ResponseContext.audience_ids`

默认兼容键：`primary`。

### 1.2 WiringLevel

1. `DISABLED`: types / docs only.
2. `SHADOW`: keyed snapshots publish alongside flat snapshots; all consumers still read flat slots.
3. `ACTIVE`: response assembly / planner opt in to keyed snapshot.
4. `RETIRE_FLAT`: flat snapshots become compatibility views over `primary`.

### 1.3 Evidence gates

- Existing companion evidence v2 passes unchanged.
- New per-interlocutor read tests: Alice preference never applies to Bob.
- New privacy scope test: memory with `audience_ids=("alice",)` is not exposed to Bob.
- Cross-session shared memory still works for `primary`.
- C5 companion evidence proves default single-party scope is `primary/self` and social prediction / social PE counts are zero.

### 1.4 Social PE gates

- Misattribution probe: deliberately route Alice preference into Bob context; assert `social_prediction_error` records wrong-person attribution.
- Audience leakage probe: private memory exposed to wrong audience creates social PE.
- Identity merge/split probe: conflicting aliases create stale / conflict evidence instead of silent overwrite.

### 1.5 ETA consumption tests

- Regime / temporal controller receives keyed identity compact summary.
- Renderer output changes only when planner / response assembly consumed keyed snapshot; no text keyword rule in renderer.

### 1.6 Migration map

| Area | Change |
|---|---|
| `vz-memory` | add subject / audience scope to memory entries |
| `vz-cognition.semantic_state` | keyed views for user / relationship owners |
| `vz-runtime.agent.response` | active speaker / audience fields |
| `lifeform-core` | session readouts become keyed / `primary` compatible |
| `lifeform-evolution` | evidence probes for per-interlocutor memory |

### 1.7 Rollback

Set social cognition wiring to `DISABLED`. Flat `user_model` / `relationship_state` / `interlocutor_state` continue to publish. Compatibility `primary` remains.

---

## 2. Phase 2 — R17 Theory of Mind Owner Decomposition

**预计**：3-4 周，4-6 个 PR。

### 2.1 Scope

Split `UserModelModule.stable_preferences` into learned ToM owners:

- `belief_about_other`
- `intent_about_other`
- `feeling_about_other`
- `preference_about_other`

`UserModelSnapshot` becomes compatibility aggregate / read model during migration.

### 2.2 WiringLevel

1. `DISABLED`: ToM owner types exist.
2. `SHADOW`: owners publish but response assembly still reads old `user_model`.
3. `ACTIVE`: response assembly / planner consume ToM owner summaries.
4. `RETIRE_AGGREGATE_TRUTH`: `user_model` no longer owns ToM truth.

### 2.3 Evidence gates

- False-belief scenario: belief owner updates without changing preference owner.
- Intent mismatch scenario: user intends action but later does not follow through; intent PE fires, durable preference untouched.
- Affect misread scenario: feeling owner updates without changing belief.
- Preference conflict scenario: durable preference and temporary override are represented separately.

### 2.4 Social PE gates

- Owner-specific PE records exist for belief / intent / feeling / preference mismatch.
- Outcome mismatch in one ToM owner does not contaminate another owner.

### 2.5 ETA consumption tests

- ToM owner summaries influence latent controller / regime inputs.
- No `if user_text contains ...` path appears in planner / renderer for ToM decisions.

### 2.6 Migration map

| Area | Change |
|---|---|
| `vz-cognition.semantic_state` | add four ToM owner modules and snapshots |
| semantic proposal runtime | target one of four ToM owners |
| `response_assembly` | consume compact ToM summaries |
| tests | add owner-specific PE / disambiguation probes |

### 2.7 Rollback

Set ToM owners to `DISABLED`; continue reading `user_model` aggregate. SHADOW records are discarded unless explicitly migrated.

---

## 3. Phase 3 — R18 Conversational Role Learning

**预计**：2-3 周，3-5 个 PR。

### 3.1 Scope

Add `conversational_role` owner:

- active speaker
- addressees
- subjects
- witnesses
- overhearers
- group audiences

Default single-user turn maps to `active_speaker_id="primary"` and `addressee_ids=("self",)`.

### 3.2 WiringLevel

1. `DISABLED`: snapshot type only.
2. `SHADOW`: role snapshot publishes from host defaults and structured proposals.
3. `ACTIVE`: memory writes / semantic proposals consume role scope.
4. `ROLE_REQUIRED`: multi-party inputs must provide or infer role with confidence.

### 3.3 Evidence gates

- Three-party scenario: agent speaks to Bob about Carol while Alice is witness.
- Memory write scope follows subject / audience fields.
- Response addressee follows role snapshot, not last-mentioned name.

### 3.4 Social PE gates

- Wrong-addressee outcome produces role PE.
- Subject/addressee confusion produces role PE.
- Witness privacy violation produces role PE.

### 3.5 ETA consumption tests

- Role snapshot changes metacontroller / regime inputs.
- Renderer only expresses selected stance; it does not perform role routing.

### 3.6 Migration map

| Area | Change |
|---|---|
| host / CLI input envelope | optional role fields |
| `ResponseContext` | role summary |
| semantic proposals | role-scoped target owner |
| memory writes | subject / audience scope |

### 3.7 Rollback

Disable role owner and use single-user default role. Multi-party tests are skipped in rollback mode; single-user companion gates remain required.

---

## 4. Phase 4 — R19 Common Ground Learning

**预计**：3-4 周，4-6 个 PR。

### 4.1 Scope

Add `CommonGroundModule` with dyad / group common-ground atoms:

- bounded recursion depth `k=2`
- reference-resolution predictions
- session-medium shared-context updates
- background-slow durable common-ground consolidation

### 4.2 WiringLevel

1. `DISABLED`: type only.
2. `SHADOW`: common-ground predictions and PE publish but do not affect response assembly.
3. `ACTIVE`: response assembly / planner consume common-ground summaries.
4. `GROUNDING_REQUIRED`: selected multi-party regimes require common-ground checks before compressed references.

### 4.3 Evidence gates

- Deictic resolution: "we" routes to the correct dyad / group.
- "As before" resolves only when shared atom exists.
- Clarification updates common ground instead of only adding memory.

### 4.4 Social PE gates

- Failed reference resolution produces common-ground PE.
- User repair "that's not what I meant" retires or weakens the relevant atom.
- Successful grounding lowers future reference PE for same dyad / group.

### 4.5 ETA consumption tests

- Controller chooses grounding / repair / proceed mode from common-ground snapshot.
- No keyword branch for "we", "yesterday", or "as before" exists in renderer.

### 4.6 Migration map

| Area | Change |
|---|---|
| `vz-cognition` | add CommonGroundModule |
| memory retrieval | allow shared-scope filters |
| `response_assembly` | consume compact common-ground advisories |
| evidence suite | add dyad / group reference probes |

### 4.7 Rollback

Disable common-ground owner; revert to explicit, less-compressed responses. Existing memory and ToM states remain intact.

---

## 5. Phase 5 — R20 Joint Entity Learning

**预计**：3-4 周，4-6 个 PR。

### 5.1 Scope

Add `GroupModule`:

- group identity / membership
- active group id
- joint attention
- group regime
- group-scoped joint commitments
- group-level PE and credit evidence

Group commitments borrow AAC lifecycle semantics but add `group_id` scope.

### 5.2 WiringLevel

1. `DISABLED`: type only.
2. `SHADOW`: group snapshot and predictions publish; no behavior changes.
3. `ACTIVE`: selected group-aware regimes consume group snapshot.
4. `GROUP_COMMITMENT_ACTIVE`: group commitments participate in commitment / open-loop evidence.

### 5.3 Evidence gates

- Family/team scenario where group continuity differs from any individual continuity.
- Group joint commitment persists even when one member is absent.
- Group repair differs from individual repair.

### 5.4 Social PE gates

- Joint commitment failure creates group-level PE.
- Group-regime mismatch creates group PE without overwriting individual relationship states.
- Shared-goal progress updates group owner, not each member separately.

### 5.5 ETA consumption tests

- Group snapshot changes latent social action mode.
- Group regime is persistent state, not prompt tag.
- Renderer receives group-aware plan from planner only.

### 5.6 Migration map

| Area | Change |
|---|---|
| `vz-cognition` | add GroupModule |
| commitment / open_loop | group scope fields |
| regime | group regime priors / social controller input |
| evidence suite | group-level continuity and joint commitment probes |

### 5.7 Rollback

Disable group owner; individual relationship and common-ground paths remain active. Group commitments are read-only artifacts until re-enabled.

---

## 6. Companion Preference Readout carry-forward

The prior `CompanionPreferenceReadout` plan is paused as a flat single-user feature. Its useful evidence goals carry forward:

- preference memory present
- preference conflict present
- preference override present
- delayed return present
- widening diversity gate

In Phase 1 these become per-interlocutor readouts under R16. In Phase 2 they are backed by `preference_about_other` and no longer rely on `user_model` flat records.

---

## 7. Acceptance checklist

- Each phase has a SHADOW period and rollback switch.
- Existing companion evidence v2 remains green after every phase.
- Every new owner publishes pre-action prediction and post-action PE.
- Every social PE record points to an owner and scope.
- ETA consumption is tested at controller / regime / planner boundary.
- Renderer never owns social inference.
- `docs/DATA_CONTRACT.md` §6.X remains synchronized with actual implementation status.

---

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Schema-only drift | Phase acceptance requires PE and ETA tests, not just fields |
| Renderer patch drift | Contract forbids renderer-owned social inference |
| LLM classifier becomes owner | Structured output only creates proposals; owner reconciles |
| Companion regression | Companion evidence v2 remains a hard gate each phase |
| Multi-party scope explosion | Default `primary` compatibility and bounded recursion depth `k=2` |
| Cross-owner double counting | SHADOW diffs compare flat vs keyed summaries before ACTIVE |
