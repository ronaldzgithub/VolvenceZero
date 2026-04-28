# lifeform-core

Product-layer skeleton: makes the Volvence Zero brain kernel **live** as a continuously-running digital organism.

## What this wheel ships today

| Component | Role | Spec |
|---|---|---|
| `Lifeform` / `LifeformConfig` / `LifeformSession` | Top-level facade composing Brain + tick + scene + followups + vitals | (this README) |
| `TickEngine` / `TickEngineConfig` | Deterministic metabolic clock (SYSTEM / ENERGY / CONTEXT ticks) | (this README) |
| `SceneManager` / `Scene` | Conversational scene lifecycle; closes scenes on idle and fires the kernel boundary | (this README) |
| `FollowupManager` / `FollowupItem` | Advisory follow-up scheduler driven by kernel snapshots and vitals pressure | (this README) |
| `VitalsModule` / `VitalsBootstrap` / `VitalsSnapshot` | **Always-on drive layer** \u2014 slow-scale R-PE source between turns | [`docs/specs/lifeform-vitals.md`](../../docs/specs/lifeform-vitals.md) |

## Always-on organism, not turn-driven assistant

Without `VitalsModule`, the kernel produces prediction error only when the user speaks. With it, drives accumulate deviation between turns, publishing a continuous PE source that downstream consumers (FollowupManager, PromptPlanner, evaluation) can read. This is the architectural difference between "responsive chatbot" and "continuously adapting digital organism".

A vertical ships its drives via a `VitalsBootstrap` (Python factory \u2014 these are configuration, not learned weights). `LifeformConfig.vitals_bootstrap` plumbs them into every session; `LifeformSession.advance_tick(...)` decays drives on every SYSTEM tick and surfaces a proactive `FollowupItem` when slow-scale PE crosses the configured threshold.

```python
from lifeform_domain_emogpt import build_companion_lifeform

life = build_companion_lifeform()        # ships 3 drives by default
session = life.create_session(session_id="alice")

# 30 idle ticks with no user input \u2014 organism notices.
await session.advance_tick(30)
print(session.vitals_snapshot.total_pe)              # > 0.0
print([f.source for f in session.due_followups()])   # ['vitals', ...]
```

The drive set lives in the vertical wheel (`lifeform-domain-emogpt` ships `bond_warmth`, `user_engagement`, `conversation_continuity`); the kernel does not know which drives matter. Different verticals can encode different "what does this lifeform care about" signatures without changing the kernel.

## Boundary rules

- May import from any `vz-*` wheel.
- May NOT be imported from any `vz-*` wheel.
- Other `lifeform-*` wheels MAY import from this one.

## Future migration map (from EmoGPT)

| EmoGPT path | This wheel's home |
|---|---|
| `core/orchestrator.py` + `core/system_initializer.py` + `core/helper/orchestrator/*` | `Lifeform` facade |
| `core/tick_engine.py` (SYSTEM_TICK / ENERGY_TICK / CONTEXT_TICK) | `lifeform_core.tick_engine` |
| `core/context/scene_manager.py` | `lifeform_core.scene_manager` |
| `core/follow_up/*` | `lifeform_core.followup_manager` |
| `body/*` + `needs_engine/*` + `homeostasis/*` | `lifeform_core.vitals` (drives + slow-scale PE) |
| `personality/personality_core.py` (base persona) | `lifeform_core.persona` (TODO) |

The vitals layer takes the place of EmoGPT's Body / NeedsEngine / Homeostasis / RewardComposer trio: instead of a hardcoded physiological model, drives are a small set of named scalars whose deviation is interpreted as prediction error. This is the first-principles equivalent of those modules under R-PE / R1 / R14.
