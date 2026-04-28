# vz-runtime

> Orchestration only — no business logic

The **only** wheel that imports every other `vz-*`. Provides the `Brain` / `BrainSession` / `AgentSessionRunner` facade, final-wiring composition, and session-post slow loop.

## Public API (stable surface)

```python
from volvence_zero.brain import Brain, BrainConfig, BrainSession
from volvence_zero.integration import FinalRolloutConfig
```

`vz-runtime` is what product code (e.g. `lifeform-core`) consumes. It is also where new owners get composed via `WiringLevel.SHADOW` for double-running migrations.

## Migration tool: WiringLevel

| Level | Effect |
|---|---|
| `ACTIVE` | Module participates in the wave's active snapshot chain |
| `SHADOW` | Module runs and validates contracts, but its output stays in shadow |
| `DISABLED` | A `RuntimePlaceholderValue` is published in its slot |

Use `SHADOW` when migrating EmoGPT modules into `vz-*` owners — both run side by side until the new owner's snapshot proves equivalent, then flip to `ACTIVE`.

## Install

```bash
pip install vz-runtime                # full kernel, synthetic substrate
pip install "vz-runtime[hf]"          # + HF substrate
pip install "vz-runtime[torch]"       # + torch (for vz-temporal training)
pip install "vz-runtime[dev]"         # + pytest / ruff
```
