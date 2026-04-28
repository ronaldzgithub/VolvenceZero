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
| `application` | Vertical adapter owners (domain knowledge / case memory / playbook / boundary policy) | — |

### Why `application` ships here today

`vz-cognition.evaluation.backbone` imports application snapshot types
(`BoundaryPolicySnapshot`, `CaseMemorySnapshot`, `DomainKnowledgeSnapshot`,
`ResponseAssemblySnapshot`, ...) at module load. The clean fix is to extract
those frozen-dataclass types into `vz-contracts` and let `vz-application`
become its own wheel that depends on cognition without cycling. Until then,
they share a wheel.

## Hard limits

- Every owner produces an immutable, frozen-dataclass snapshot.
- `evaluation` is **readout** of prediction error, not a learning source.
- All self-modification writeback flows through `ModificationGate` in `credit`.
- Regime is runtime state, not a prompt label.
