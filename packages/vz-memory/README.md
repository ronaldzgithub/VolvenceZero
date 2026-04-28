# vz-memory

> R5 (Memory Continuum) · R6 (Reflection & Consolidation owned by `vz-cognition`)

Continuous memory spectrum: transient working state → episodic session state → durable semantic memory → derived indexes. Implements NL's CMS bands and is the only legitimate owner of memory writes.

## What it owns

| Slot | Contents |
|---|---|
| `memory` | 4 strata + lifecycle metrics, single-owner write surface |
| (CMS internals) | Multi-frequency MLP chain with anti-forgetting |

## Reflection lives in vz-cognition

`reflection` (slow consolidation, writeback proposals) physically lives in `vz-cognition` because it consumes credit/regime/PE snapshots and produces writeback **proposals**. `vz-memory` is the apply target, not the producer.

## Hard limits

- All writes go through the `MemoryStore` API. External code cannot mutate internal tables.
- Writeback from reflection is gated through `ModificationGate` in `vz-cognition` — bounded by credit evidence.
