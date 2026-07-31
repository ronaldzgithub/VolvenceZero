# Relationship Memory Console

> Status: MVP P0-P6 implementation landed; live seven-day human pilot not yet run
> Last updated: 2026-08-01
> Scope: Gate 8/11 productization for the 7-day relationship assistant MVP

## Problem

Gate 8 and Gate 11 prove bounded wake/sleep consolidation and isolated
per-user continuity at the owner level. Product users still need a visible,
correctable surface for what the system believes it should remember. Without
that surface, long-term relationship intelligence becomes opaque memory rather
than trust.

This MVP turns the proven owner loop into a product loop:

```text
typed observation
→ session-post reflection
→ relationship update proposals
→ user-controlled console action
→ owner API write / correction event
→ next-session hydration
→ PE / CP-12 settlement
→ continuity metrics
```

## Invariants

- `reflection` owns relationship update proposals. Console/API consumers read
  `ReflectionSnapshot.relationship_update_proposals`; they do not rebuild
  proposals from memory entries, lessons, raw text or prompt residue.
- Proposals are `SHADOW` readouts by default. A proposal does not become a
  durable relationship fact until the user confirms it or an explicit future
  boundary policy authorizes automatic apply.
- Durable writes keep the existing owners: `memory`, the nine semantic owners,
  and `boundary_consent`. The console is a product actuator, not a second memory
  or relationship owner.
- Corrections, rewrites and deletes must emit typed outcome evidence through the
  existing `dialogue_external_outcome` path so PE and CP-12 owner predictions can
  settle. Evaluation and console metrics stay read-only.
- Gate 11 negative controls (`stateless`, `swapped-user-state`,
  `shuffled-history`) remain offline regression arms and never enter production
  routing.
- User control is part of the trust contract: every item can be kept, scoped to
  the session, deleted, rewritten, marked sensitive, or marked as not eligible
  for proactive mention.

## Proposal Contract

`ReflectionSnapshot.relationship_update_proposals` is a tuple of
`RelationshipUpdateProposal` values:

| Field | Meaning |
|---|---|
| `proposal_id` | Content-addressed stable id over target, operation, evidence and description |
| `target_owner_slot` | One of `memory`, `commitment`, `open_loop`, `user_model`, `belief_assumption`, `relationship_state`, `boundary_consent` |
| `operation` | `remember`, `promote`, `decay`, `reinforce`, or `review` |
| `human_readable_description` | Owner-authored explanation shown by the console |
| `source_evidence` | Machine-readable provenance such as `memory_entry:*`, `belief_update:*`, `tension:*` or `prediction_error:*` |
| `confidence` | Bounded confidence readout from reflection/consolidation evidence |
| `requires_user_confirmation` | Defaults `true` for MVP |
| `shadow_only` | Defaults `true`; prevents silent durable apply |

The description is authored in `vz-cognition.reflection`; service/UI code must
not inspect arbitrary user text or synthesize a second rationale.

## Console Actions

| Action | Owner path |
|---|---|
| `keep` | Apply the proposed operation through the target owner API |
| `session_only` | Keep in the product/session surface but do not durable-write |
| `delete` | Use Memory owner deletion by scope plus semantic lifecycle close where applicable |
| `rewrite` | Submit typed semantic proposal or memory write request with user-provided replacement |
| `mark_sensitive` | Submit typed `boundary_consent` proposal; boundary owner remains sole sensitivity owner |
| `no_proactive_mention` | Submit typed `boundary_consent` proposal constraining proactive recall |

Unsupported actions must fail loudly. A missing target owner snapshot or failed
write must be returned to the console as an explicit rejected action, not a
silent success.

## HTTP Contract

- `GET /v1/users/me/relationship-memory?session_id=<id>` returns the current
  session's unresolved reflection proposals and Memory-owner-scoped durable
  entries. The session must belong to the authenticated alpha user.
- `POST /v1/users/me/relationship-memory/{item_id}/action` accepts
  `{session_id, action, replacement?, correction_kind?}`. `replacement` is
  required only for `rewrite` and rejected for other actions.
- `action` is exactly one of `keep | session_only | delete | rewrite |
  mark_sensitive | no_proactive_mention`; `correction_kind` is exactly one of
  `content_inaccurate | wrong_user_attribution | stale | boundary_preference`
  and is only accepted for corrective actions.
- GET returns `{user_id, session_id, pending_proposals, durable_entries}`. POST
  returns the frozen action record `{action_id, user_id, session_id, item_id,
  action, status, owner_operations, replacement_entry_id, correction_kind,
  dialogue_outcome_evidence_id, dialogue_outcome_kind, created_at_ms}`; exact
  retry returns 200 with the prior record, first apply returns 201.
- Memory keep/delete/rewrite operations run synchronously through the
  `BrainSession` facade and Memory owner API, then require persistence success.
- Semantic and boundary operations enqueue a typed `GenericSemanticEvent` and
  return `status=queued`; the owning semantic module applies it on the next
  canonical turn. The API never reports a queued event as already durable.
- Exact retries are idempotent. A proposal cannot be resolved by two different
  actions; durable entries can receive later actions as their lifecycle changes.
- `RelationshipMemoryActionLedger` owns only product action/idempotency state.
  It does not copy relationship facts or interpret user text. Its internal
  `request_fingerprint = sha256(action + NUL + replacement + NUL +
  correction_kind)` keys exact retries by `(user_id, session_id, item_id,
  fingerprint)`; resolved proposals also bind their first fingerprint so a
  conflicting second action returns 409.
- `create_app()` binds the ledger to `AlphaServiceConfig.memory_scope_root_dir`
  when configured and hydrates `relationship-memory-console-actions.json` at
  startup. Persistence failure rolls back the just-added ledger record and is
  surfaced; this action ledger still never becomes relationship-state SSOT.
- delete/rewrite use Memory owner checkpoint + persistence and restore the full
  checkpoint on failure. Semantic/boundary changes remain `status=queued` until
  the next canonical owner turn.
- Corrective actions submit a typed `dialogue_external_outcome` with
  `USER_EXPLICIT` evidence. This implemented evidence path may settle PE on a
  later turn; it does not mean P5 continuity aggregation exists.

## Continuity Metrics

`RelationshipContinuityEvaluationModule` 是 `relationship_continuity` 的唯一只读
owner，默认 `SHADOW`。immutable exchange shape 位于
`vz-contracts.relationship_continuity`；runtime 通过 Brain facade 注入公开 snapshot
与 typed console outcome，service 不直接 import cognition owner。

| Metric | Source |
|---|---|
| `callback_hit_rate` | CP-12 closure/follow-through settlement and callback adoption evidence |
| `boundary_violation_rate` | `boundary_consent` overreach/violation readouts and correction events |
| `wrong_user_attribution_rate` | Console correction events marked as wrong-person/wrong-user attribution |
| `open_loop_closure_rate` | `open_loop` lifecycle |
| `user_correction_rate` | Console correction/rewrite/delete events over shown items |
| `remembered_item_usefulness` | User keep/useful signals over remembered items |
| `seven_day_trust_delta` | `relationship_state` trust trajectory plus optional L4 human anchor |

These metrics are evaluation readouts and pilot evidence only. They do not
become PE, credit or ModificationGate input by themselves.

- `GET /v1/users/me/continuity-metrics?session_id=<id>` 要求 alpha user 拥有该
  session，返回去标识化 `user_scope_hash`、七项指标、`sample_sizes` 和
  `wiring_level=shadow`。
- 七日窗口按 timestamp 剪裁。没有有效分母时返回 `null`，禁止用零冒充证据。
- readout 持久化于 scoped Memory backend 的
  `evaluation/relationship_continuity` key；多 session 查询前重新 hydrate 后合并。
- `seven_day_trust_delta` 只使用 owner-published cumulative trust trajectory；L4
  人评 anchor 作为独立材料保留，不混入该数值。

## P6 Regression And Pilot Harness

- `evaluate_gate11_continuity_regression()` 只读取现有 Gate 11
  `ablation_results.json`，要求 correct-user-state 对 stateless、swapped-user-state、
  shuffled-history 三臂均有正增益和正 95% CI 下界，并要求隔离、持久化、删除、
  rollback 等 preregistered gates 全部通过。CLI 为
  `scripts/check_gate11_relationship_continuity.py`；它不运行在生产路径。
- `RelationshipAssistantPilotHarness` 只接受邀请 allowlist 与 day 1-7，按
  participant hash 写每日 metrics 和 transcript artifact。调用者传入 structured
  transcript；harness 会替换显式 user id，输出不含原始 user id 的 L4 material。
- harness 落地不等于真人 pilot 已完成；真人结论必须引用实际 7-day artifact。

## Rollback

- P1 rollback: remove or ignore `relationship_update_proposals`; reflection
  still publishes the existing consolidation snapshot.
- P2/P3 rollback: disable console routes/UI; no kernel owner data is lost.
- P4 rollback: stop injecting console correction outcomes; owner writes already
  made remain subject to the explicit user action log.
- P5 rollback: hide continuity metrics; evaluation learning boundaries remain
  unchanged.
- P6 rollback: remove the CI artifact check and stop pilot capture; production
  relationship routing is unchanged because neither path runs in-turn.

## Pilot Exit

The 7-day pilot may proceed only while wrong-user attribution and boundary
violation are attributable, inspectable and correctable. If either appears in a
non-engineering-fault path, default policy tightens to "all proposals require
manual confirmation" and no automatic durable apply is permitted.
