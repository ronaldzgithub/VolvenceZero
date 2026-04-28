# vz-contracts

> R8 (Snapshot-First, Contract-First) · R15 (Migration Preserves Explainability)

Volvence Zero **contract foundation**. Every other `vz-*` and `lifeform-*` package consumes this. Zero runtime dependencies; pure stdlib.

## What it owns

| Surface | Module |
|---|---|
| `Snapshot[T]`, `RuntimeModule[T]`, `WiringLevel`, `RuntimePlaceholderValue` | `volvence_zero.runtime.kernel` |
| `OwnershipGuard`, `ImmutabilityGuard`, `DependencyGuard`, `SchemaGuard` | `volvence_zero.runtime.kernel` |
| `propagate(...)`, `topo_sort_modules`, `detect_dependency_cycle` | `volvence_zero.runtime.kernel` |
| `EventRecorder`, `DebugEvent` (Layer-1 observability) | `volvence_zero.runtime.kernel` |
| `LearnedUpdateRule`, `LearnedUpdateRuleState`, `LearnedUpdateDecision` | `volvence_zero.learned_update` |

## What it does NOT do

- Does not implement any runtime owner. Implementations live in `vz-substrate`, `vz-memory`, `vz-cognition`, `vz-temporal`, etc.
- Does not depend on torch, numpy, or any third-party library.
- Does not import from `lifeform_*` (kernel must not depend on product layer).

## Invariants enforced at runtime

- Every snapshot is a frozen dataclass.
- Each slot has exactly one owner.
- Snapshot versions are monotonically increasing per slot.
- Consumers can only read declared `dependencies`.
- Published snapshots are content-hashed and verified for immutability.

See `docs/next_gen_emogpt.md` for the full R-ID rationale.
