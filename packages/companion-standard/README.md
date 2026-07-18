# companion-standard

**Relationship Representation Standard** — a typed, immutable, zero-dependency
schema for describing the state of a long-horizon relationship between an AI
and a person.

This wheel is the single source of truth (SSOT) for:

- **Semantic owner snapshot value types** — the nine owner slots
  (`plan_intent`, `commitment`, `open_loop`, `user_model`, `execution_result`,
  `belief_assumption`, `relationship_state`, `goal_value`, `boundary_consent`)
  and their frozen-dataclass snapshot values.
- **Theory-of-mind record types** — typed beliefs / intents / feelings /
  preferences about another mind.
- **Owner prediction signal** — the typed, owner-authored pre-action
  prediction record embedded in snapshot values.
- **Canonical interaction trajectory** — a multi-session interaction
  trajectory format with per-segment relationship-state labels, canonical
  JSON serialisation, and a stable content hash.
- **Minimal snapshot kernel** — the immutable `Snapshot` container and
  dependency-ordered `propagate` step.
- **Semantic embedding seam** — the `SemanticEmbeddingBackend` protocol and
  its deterministic fallback stub.
- **Conformance kit** — assertions a third party can run to check that its
  producers / consumers respect the standard.

## Design rules

- Zero runtime dependencies; pure stdlib. Python 3.11+.
- Every value type is a frozen dataclass. Consumers must not mutate.
- This package defines *what relationship state looks like*, never *how it is
  computed*. No owner logic, no learning mechanisms, no prompts ship here.

## Boundaries

`companion_standard` has no dependencies and imports nothing outside the
stdlib (CI-enforced). The dependency direction is one-way: runtimes that
adopt the standard re-export from this package, never the reverse.

RFC: `docs/external/relationship-representation-rfc-v0.md`.
