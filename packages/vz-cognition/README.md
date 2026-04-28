# vz-cognition

> R-PE · R7 · R9 · R10 · R11 · R12 · R14

The **cognitive state plane**: prediction error, dual-track learning, hierarchical credit + gated self-modification, regime identity, nine semantic owners, evaluation readout, and the slow-reflection writeback engine.

This is currently one wheel because the seven sub-packages it contains are tightly coupled by snapshot consumption. Future migrations may split it once the inter-owner contracts stabilise (see `SPLIT.md`).

## Sub-packages

| Sub-package | Owns | R-ID |
|---|---|---|
| `dual_track` | World/Self pressure split | R7 |
| `prediction` | `prediction_error` snapshot — the primitive learning signal | R-PE |
| `credit` | Hierarchical credit ledger + `ModificationGate` | R9, R10 |
| `regime` | Persistent social/cognitive regime identity (6 archetypes) | R14 |
| `semantic_state` | 9 semantic owners (plan, commitment, open-loop, user-model, etc.) | R11 |
| `evaluation` | Six-family scores, readout-only | R12 |
| `reflection` | Background-slow consolidation; produces writeback proposals | R6 |
| `application_types` | Frozen-dataclass snapshot definitions consumed by `evaluation` and produced by `vz-application` owners — defined here to break the wheel-level cycle | — |

> **Application owner code lives in the separate `vz-application` wheel.** Only the snapshot type **definitions** (frozen dataclasses, no behavior) live here, in `volvence_zero.application_types`. See `vz-application/README.md` for the cycle-break rationale.

## Hard limits

- Every owner produces an immutable, frozen-dataclass snapshot.
- `evaluation` is **readout** of prediction error, not a learning source.
- All self-modification writeback flows through `ModificationGate` in `credit`.
- Regime is runtime state, not a prompt label.
