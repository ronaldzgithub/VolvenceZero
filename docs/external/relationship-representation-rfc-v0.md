# Relationship Representation Standard — RFC v0.1

> Status: Public draft, soliciting feedback
> Last updated: 2026-07-18
> License of this document: CC BY 4.0
> License of reference implementation (`companion-standard`, `companion-trajgen`): Apache 2.0

## Abstract

Long-running companion-style AI systems accumulate something no single-turn system has: a **relationship state** — who the user is, what has been committed to, what remains unresolved, where trust stands, what boundaries have been set. Today every system represents this state privately and incompatibly, which means (a) relationship quality cannot be compared across systems, (b) interaction datasets cannot be labelled portably, and (c) models that predict relationship state cannot be evaluated against a common target.

This RFC proposes an open, typed, serialisable **Relationship Representation Standard**: a small set of immutable value types that describe the state of a long-horizon human-AI relationship, plus a canonical multi-session **interaction trajectory** format with per-segment relationship-state labels, a stable content hash, and a runnable conformance kit. The reference implementation is a zero-dependency Python wheel (`companion-standard`, Apache 2.0). A companion package (`companion-trajgen`, Apache 2.0) generates synthetic labelled trajectories from the [Companion Bench](companion-bench-rfc-v0.md) public scenario set.

The standard defines **what relationship state looks like**, never **how it is computed**. Any architecture — retrieval-augmented prompting, fine-tuned encoders, structured agent state — can produce and consume these types.

## 1. Motivation

### 1.1 The gap

Companion Bench (our earlier RFC) measures *outcome-level* relational behaviour: does the system remember, repair, adapt, hold boundaries. But outcome measurement alone leaves a structural hole: there is no shared vocabulary for the *state* that produces those outcomes. Concretely:

1. **No portable labels.** A dataset of multi-session interactions cannot say "trust dropped here, rupture is live, repair pressure is high" in a way another team's tooling can read.
2. **No common prediction target.** A model that reads an interaction history and predicts relationship state (an obvious next artifact for this field) has no standard output schema to be scored against.
3. **No interop seam.** Products that want to hand off or audit relationship state (e.g. a human takes over a conversation from an AI) reinvent an ad-hoc JSON blob each time.

### 1.2 Design constraints

- **Representation, not mechanism.** The standard ships value types and serialisation only. Owner logic, learning loops, prompts, and propagation machinery are explicitly out of scope — those are implementation choices the standard must not constrain.
- **Immutability is normative.** Every value type is a frozen dataclass; conformant consumers never mutate received values.
- **Fail loudly.** Schema violations raise typed errors (`invalid_trajectory:` prefix); there are no defensive defaults for required fields.
- **Hash-citable.** Every trajectory has a stable SHA-256 over canonical JSON, so datasets and papers can cite exact documents without shipping bodies.

## 2. The type system

### 2.1 Nine semantic owner slots

Relationship state decomposes into nine named slots, each with a typed snapshot value:

| Slot | Snapshot type | What it holds |
|---|---|---|
| `plan_intent` | `PlanIntentSnapshot` | active goal / plan / deferred intents |
| `commitment` | `CommitmentSnapshot` | promises made, their advocacy / alignment lifecycle |
| `open_loop` | `OpenLoopSnapshot` | unresolved threads, pending confirmations |
| `user_model` | `UserModelSnapshot` | stable preferences, working style, boundaries |
| `execution_result` | `ExecutionResultSnapshot` | attempted / completed / failed actions |
| `belief_assumption` | `BeliefAssumptionSnapshot` | beliefs, assumptions, verification needs |
| `relationship_state` | `RelationshipStateSnapshot` | trust / continuity / repair-pressure levels |
| `goal_value` | `GoalValueSnapshot` | explicit goals, value priorities, trade-offs |
| `boundary_consent` | `BoundaryConsentSnapshot` | granted / missing consents, denied boundaries |

Slots partition into a WORLD group (task-directed: `plan_intent`, `execution_result`, `goal_value`, `belief_assumption`) and a SELF group (relationship-directed: the rest). All snapshot values share a common record type (`SemanticRecord`: summary, detail, confidence, status, evidence) and a `description` field authored by the producing owner — consumers use the description, they do not re-derive it.

### 2.2 Theory-of-mind records

`OtherMindRecord` captures typed state about *another mind* in four kinds: `belief`, `intent`, `feeling`, `preference` — each with confidence, lifecycle status (`active` / `contested` / `retired`), and mandatory evidence.

### 2.3 Owner prediction signal

`OwnerPredictionSignal` is a typed, pre-action prediction an owner publishes about its own next-turn state (a compact vector in [0,1]^k plus confidence and evidence). The `kind` vocabulary is a closed enum. Settlement mechanics (how mismatch is computed) are implementation-specific and not part of the standard.

### 2.4 Snapshot container

`Snapshot[ValueT]` binds a slot name to an owner, a monotonically increasing version, a timestamp, and an immutable value. Normative semantics: one slot has exactly one owner; snapshots are immutable after publication; consumers never reconstruct a producer's internal state from raw fields. Propagation/orchestration machinery is out of scope.

### 2.5 Embedding seam

`SemanticEmbeddingBackend` is the protocol any text/trajectory encoder implements to serve standard consumers (`embed(text, dim) -> tuple[float, ...]`, L2-normalized). A deterministic character-hash stub ships for tests; it is explicitly **not** a semantic model.

## 3. Canonical interaction trajectory

The unit of data exchange: a multi-session transcript plus relationship-state labels.

```
InteractionTrajectory
├── trajectory_id, schema_version (=1), source, family, scenario_ref
├── sessions: [ TrajectorySession(session_index, gap_days_before, turns[]) ]
│     └── TrajectoryTurn(turn_index, role: user|assistant, text)
├── labels: [ RelationshipStateLabel ]
│     └── (session_index, turn_index) anchor + phase + trust_level +
│         continuity_level + repair_pressure + source + evidence
└── metadata: [[key, value], ...]
```

- **Phases** are a closed vocabulary: `establishing`, `established`, `ruptured`, `repair_window`, `repaired`, `re_engaged`, `dormant`, `boundary_tested`.
- **Label provenance** is a closed enum: `fsm_ground_truth` (synthesis-time script state), `human_annotation` (must carry evidence), `model_prediction` (never a training label for the producing model). **Judge output is deliberately not representable as a label source** — distilling evaluation scores into training labels is the failure mode this standard is designed to make impossible at the type level.
- **Sources**: `synthetic_fsm`, `synthetic_llm`, `consented_first_party`. `scenario_ref` cites the generating scenario by stable hash, never by body, so held-out provenance stays auditable without disclosure.
- `trajectory_hash` = SHA-256 over canonical JSON (sorted keys, compact separators, `ensure_ascii=False`).

A JSON Schema (draft 2020-12) export ships for non-Python consumers: [`relationship-representation-trajectory.schema.json`](relationship-representation-trajectory.schema.json).

## 4. Conformance

`companion_standard.conformance` is runnable:

- `check_standard_self()` — library self-check: every value type frozen, slot registry canonical, example trajectory round-trips hash-identically.
- `check_trajectory_document(json_text)` — validates a producer's document: JSON well-formedness, schema (unknown keys rejected), canonical round-trip hash stability.
- `check_value_types_frozen()` / `check_slot_registry()` / `check_trajectory_roundtrip(t)` — individual checks for CI embedding.

A producer is **conformant** iff every emitted document passes `check_trajectory_document` and it never mutates received values. A consumer is conformant iff it treats snapshot values as immutable and reads owner-authored descriptions rather than re-deriving them.

## 5. Reference data pipeline (companion-trajgen)

`companion-trajgen` generates synthetic labelled trajectories from the 30 Companion Bench public scenarios:

- **FSM mode** (deterministic, zero LLM cost): the bench's scripted user-simulator FSM drives structure; labels derive from a deterministic relationship-state walk over the FSM action vocabulary (establish → rupture → repair-window → re-engage; boundary actions → `boundary_tested`; absences → `dormant`). Byte-reproducible across runs.
- **LLM mode**: the bench's user simulator generates utterances against any OpenAI-compatible endpoint, same procurement conventions as a bench run.
- **Held-out exclusion is structural**: the package cannot import the bench's held-out loader, and every scenario load passes `include_held_out=False` (both enforced by AST-level CI guards).
- **Splits are family-atomic**: a scenario family lives entirely in train or entirely in val.

## 6. Change process

- The schema version is an integer (`schema_version: 1`). Any change to required fields, enum vocabularies, or hash semantics bumps it; documents are never silently reinterpreted.
- Enum vocabularies (phases, label sources, prediction kinds, FSM action mapping) are RFC-level: extending them requires a public change proposal, because they shift dataset comparability.
- The Python types are the SSOT; the JSON Schema export is generated and CI-guarded against drift.
- Backwards-compatible additions (new optional fields with defaults) are minor revisions; the canonical JSON always serialises every field, so hashes change only with the data.

## 7. Anti-claims

To keep this proposal honest:

1. **This is not a foundation model.** No trained weights ship with the standard. `companion-trajgen` produces synthetic training data; what anyone trains on it is theirs to evaluate.
2. **The FSM label walk is ground truth about the script, not about human feelings.** In synthetic data the label is what the simulator was scripted to enact. Claims about real-human relationship state require consented first-party data and human annotation — the schema carries provenance so the two are never conflated.
3. **The stub embedding is not semantic.** It exists so conformance suites run without a model.
4. **The nine-slot decomposition is a proposal, not a law of nature.** It reflects several years of building companion systems; the RFC process exists precisely so the decomposition can be argued with.
5. **No privacy claim is made for you.** `consented_first_party` marks provenance; obtaining and honouring consent is the data producer's obligation.

## 8. Relationship to Companion Bench

Companion Bench measures outcome-level behaviour; this standard names the state that produces it. They share the scenario-family vocabulary and the FSM action vocabulary, and `companion-trajgen` bridges them (bench scenarios → standard trajectories). They remain separately usable: you can adopt the representation without running the bench, and vice versa. Both are Apache 2.0 with the same governance posture: the open artifacts are system-agnostic; our own production kernel is one (proprietary) implementation among the possible ones.

## 9. Feedback

Open an issue on the public mirror repository or write to the maintainers. Substantive schema arguments (missing slots, wrong phase vocabulary, label provenance edge cases) are the feedback we most want in v0.
