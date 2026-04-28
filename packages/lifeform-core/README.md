# lifeform-core

Product-layer skeleton: makes the Volvence Zero brain kernel **live** as a continuously-running digital organism.

This wheel will eventually take over EmoGPT's:
- `core/orchestrator.py` + `core/system_initializer.py` + `core/helper/orchestrator/*` → `Lifeform` facade
- `core/tick_engine.py` (SYSTEM_TICK / ENERGY_TICK / CONTEXT_TICK) → `lifeform_core.tick_engine`
- `core/context/scene_manager.py` → `lifeform_core.scene_manager`
- `core/follow_up/*` → `lifeform_core.followup_manager`
- `personality/personality_core.py` (base persona) → `lifeform_core.persona`

Currently empty — populated incrementally by M0 → M6 according to `SPLIT.md` and `docs/next_gen_emogpt.md`.

## Boundary rules

- May import from any `vz-*` wheel.
- May NOT be imported from any `vz-*` wheel.
- Other `lifeform-*` wheels MAY import from this one.
