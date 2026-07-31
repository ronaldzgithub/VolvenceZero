# Relationship Memory Console

> Status: draft / MVP P0
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

## Metrics

The MVP continuity panel exposes seven read-only metrics:

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

## Rollback

- P1 rollback: remove or ignore `relationship_update_proposals`; reflection
  still publishes the existing consolidation snapshot.
- P2/P3 rollback: disable console routes/UI; no kernel owner data is lost.
- P4 rollback: stop injecting console correction outcomes; owner writes already
  made remain subject to the explicit user action log.
- P5 rollback: hide continuity metrics; evaluation learning boundaries remain
  unchanged.

## Pilot Exit

The 7-day pilot may proceed only while wrong-user attribution and boundary
violation are attributable, inspectable and correctable. If either appears in a
non-engineering-fault path, default policy tightens to "all proposals require
manual confirmation" and no automatic durable apply is permitted.
