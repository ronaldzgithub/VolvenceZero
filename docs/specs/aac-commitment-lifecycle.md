# Spec: AAC Commitment Lifecycle (Advocacy → Alignment → Commitment → Followup)

> **R-#:** R8 (snapshot-first), R11 (publishable internal state), R14 (regime persistent identity \u2014 commitments belong to the relationship axis)
> **Owner wheel:** `vz-cognition` (`semantic_state.CommitmentModule`)
> **Status:** v1 \u2014 typed lifecycle landed, follow-up surfacing wired in `lifeform-core`.

## Why this exists

Pre-Gap-7 the kernel exposed only a coarse `status` enum on commitment records (`active` / `blocked` / `completed` / `closed` / `deferred`). That can't distinguish:

* The AI **observed** a possible commitment vs. **actively advocated** it to the user.
* The user **rejected** an advocated commitment vs. simply **didn't respond** to it.

EmoGPT v4.0's PRD §5.6 introduces an explicit `advocacy_state` / `alignment_state` lifecycle to fill that gap. We adopt the same axes in VolvenceZero, but **without** copying EmoGPT's keyword-detection rules: lifecycle transitions are derived from typed `SemanticProposalOperation` values produced by the runtime, not from LLM output text.

## Data contract

```python
class AdvocacyState(str, Enum):
    NOT_READY = "not_ready"  # AI observed but did not surface
    READY     = "ready"       # AI decided to surface but has not yet (DEFER)
    PROPOSED  = "proposed"    # AI surfaced the commitment in-conversation

class AlignmentState(str, Enum):
    UNKNOWN = "unknown"
    AGREE   = "agree"
    MODIFY  = "modify"
    REJECT  = "reject"

@dataclass(frozen=True)
class CommitmentLifecycleEntry:
    record_id: str
    advocacy_state: AdvocacyState
    alignment_state: AlignmentState
```

Published on `CommitmentSnapshot` parallel to existing fields:

```python
class CommitmentSnapshot:
    # legacy fields (unchanged)
    active_commitments: tuple[SemanticRecord, ...]
    honored_commitment_refs: tuple[str, ...]
    at_risk_commitments: tuple[SemanticRecord, ...]
    trust_obligation_count: int
    continuity_score: float
    control_signal: float
    description: str

    # AAC additions
    lifecycle_entries: tuple[CommitmentLifecycleEntry, ...] = ()
    advocacy_proposed_count: int = 0
    advocacy_ready_count: int = 0
    alignment_agree_count: int = 0
    alignment_modify_count: int = 0
    alignment_reject_count: int = 0
```

Backwards compat preserved \u2014 every new field has a default.

## Operation → lifecycle truth table

The lifecycle is derived from one source: `SemanticProposalOperation`. Each operation either advances both axes, advances one and **preserves** the other axis's prior value, or leaves both untouched.

| Operation | advocacy | alignment |
|---|---|---|
| `OBSERVE`  | `not_ready` | `unknown` |
| `CREATE`   | `not_ready` | `unknown` |
| `DEFER`    | `ready`     | *(preserved)* |
| `ACTIVATE` | `proposed`  | *(preserved)* |
| `REVISE`   | `proposed`  | `modify` |
| `COMPLETE` | `proposed`  | `agree` |
| `CLOSE`    | `proposed`  | *(preserved)* |
| `BLOCK`    | `proposed`  | `reject` |

The "preserved" rule matters: if a user said `MODIFY` and the AI then re-advocated the modified version (`ACTIVATE`), the user's prior alignment signal is **not** silently overwritten. Public helper `commitment_lifecycle_for_operation(op, *, previous=...)` returns the next state given the prior tuple.

## Owner-side bookkeeping

`SemanticStateStore` tracks per-record lifecycle as `slot \u2192 record_id \u2192 (advocacy, alignment)`. On every `apply()`:

1. Each proposal's operation produces the next lifecycle for its `proposal_id`, using the previous entry as the `previous` argument.
2. After applying, lifecycle entries whose `record_id` has fallen out of the bounded record window (last 12) are garbage-collected.

`CommitmentModule._build_snapshot` reads `store.lifecycle_for("commitment")` and emits the parallel tuple plus aggregate counts.

## Lifeform-side wiring

`LifeformSession.run_turn` (`packages/lifeform-core/src/lifeform_core/lifeform.py`) now reads `lifecycle_entries` and surfaces a `commitment`-source `FollowupItem` for any record whose `alignment_state` is `REJECT`. This is the closure of the AAC pipeline:

```
Advocacy → Alignment → (Commitment | Reject) → Followup
       │           │            │
       │           │            └─→ kept on followup queue, surfaces "rejected commitment to circle back"
       │           └─→ alignment=AGREE / MODIFY recorded for reflection writeback
       └─→ AI's choice to surface vs hold (NOT_READY / READY)
```

The follow-up handler is intentionally minimal: it dedupes on `record_id`, fires through the existing `FollowupManager.ingest_at_risk_commitments` channel so consumers (CLI / service / scenarios) need no API change.

## Hard rules

1. **No keyword detection drives alignment_state.** All transitions go through `SemanticProposalOperation`. If a future runtime wants to flip alignment to `REJECT` based on user text, it must produce a typed `BLOCK` proposal via `SemanticProposalRuntime.propose`, not a free-text rule. (`.cursor/rules/no-keyword-matching-hacks.mdc`.)
2. **Lifecycle never lives outside the owner.** Consumers read `CommitmentSnapshot.lifecycle_entries`; nothing writes lifecycle directly. (`.cursor/rules/ssot-module-boundaries.mdc`.)
3. **`ACTIVATE` after `REVISE` preserves `MODIFY`.** Re-advocating a modified version does not silently overwrite the user's earlier modify signal. The only way to clear a typed alignment is another typed transition (`COMPLETE` / `BLOCK`).

## R-# trace

* **R8** \u2014 `CommitmentSnapshot` is the only public surface for lifecycle. The store is owner-internal.
* **R11** \u2014 every commitment's state is named (record_id, advocacy_state, alignment_state) and inspectable on the snapshot. Reflection / evaluation / rollback can all consume it.
* **R14** \u2014 commitment lifecycle is part of relationship-axis state; the `relationship_state` owner can later cross-reference it without duplicating the truth.

## Open follow-ups

* **Reflection writeback `outcome_kind` enum** (Gap 7 part 2 in `docs/todo`): `commitment_progressed / completed / stalled / rejected / followup_no_response`. Should be derived from per-record lifecycle aggregates over a session, not from new owner state.
* **Per-record evidence pointers**. `CommitmentLifecycleEntry` could carry a `last_operation: SemanticProposalOperation` field so reflection can audit *why* the lifecycle advanced; deferred to keep this round small.
* **`outcome_kind` on `relationship_state`** (Gap 9 / Gap 10 territory): when a commitment lifecycle ends in `REJECT`, the relationship axis should observe a typed delta, not infer it from text.
