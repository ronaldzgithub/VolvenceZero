# Spec: Lifeform Vitals (Always-On Drive Layer)

> **R-#:** R-PE, R1, R8, R11, R14
> **Owner wheel:** `lifeform-core`
> **Vertical configuration wheel(s):** any `lifeform-domain-*`
> **Status:** v1 \u2014 first-class slow-scale PE source between turns.

## Why this exists

The kernel produces prediction error per turn. Without a between-turn PE source the lifeform is a turn-driven assistant: silence is invisible to the system. The vitals layer makes the lifeform **always-on** by introducing slow-scale drives whose deviation from a homeostatic band IS the surprise signal at the metabolic timescale. This is what closes the gap between "responsive chatbot" and "continuously adapting digital organism."

## Where it lives

```
                                    advance_tick(N)              run_turn(text)
                                          \u2502                            \u2502
LifeformSession  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
                                          \u2502                            \u2502
                                          \u25bc                            \u25bc
                            VitalsModule.on_tick(ev)     VitalsModule.on_turn(regime=...)
                                          \u2502                            \u2502
                                          \u25bc                            \u25bc
                              decay drive levels                recharge drive levels
                                          \u2502
                                          \u25bc
                  consider_proactive_followup(current_tick=...)
                                          \u2502
                                          \u25bc
                  FollowupManager.ingest_proactive_drive_pressure(...)
```

Drives are configuration, not learned weights. A vertical ships its drive set as a `VitalsBootstrap` (Python factory, no pickle); the kernel does not know which drives matter. This keeps the kernel vertical-agnostic while letting each product imprint a "what does this lifeform care about" signature on the behavior loop.

## Data contract

```python
DriveSpec(
    name: str,
    target: float,                                 # ideal level [0, 1]
    homeostatic_band: tuple[float, float],         # comfort range
    decay_per_tick: float,                         # subtract from level on every SYSTEM tick
    pe_weight: float,                              # contribution to total slow-scale PE
    initial_level: float = 0.5,
    recharge_per_turn: float = 0.0,                # baseline charge on user turns
    recharge_per_regime: dict[str, float] = {},    # extra charge keyed by active regime
)

VitalsBootstrap(
    schema_version: 1,
    drives: tuple[DriveSpec, ...],
    proactive_pe_threshold: float,                 # cross to surface proactive followup
    proactive_followup_priority: float,
    proactive_cooldown_ticks: int,
)

VitalsSnapshot(
    schema_version: 1,
    tick_index: int,
    drive_levels: tuple[DriveLevel, ...],          # current state per drive
    total_pe: float,
    above_proactive_threshold: bool,
    last_proactive_at_tick: int | None,
)
```

## Lifecycle invariants

1. **One owner.** Each `LifeformSession` owns one `VitalsModule` constructed from the bootstrap. Nothing else writes drive levels.
2. **Snapshot-only read path.** Consumers (`FollowupManager`, `PromptPlanner`, scenario benchmarks, evaluation) read `LifeformSession.vitals_snapshot`; they never reach into `VitalsModule._levels`.
3. **Decay only on SYSTEM ticks.** Energy / Context ticks update `tick_index` but do not consume drive level. Different tick frequencies, different physics \u2014 in line with NL frequency-ordering (R1).
4. **Cooldown is owner-enforced.** Calling `consider_proactive_followup()` on every tick is safe; the module tracks the last firing tick and returns False inside the cooldown.
5. **Frozen at runtime.** `DriveSpec` / `VitalsBootstrap` / `VitalsSnapshot` are frozen dataclasses; reconfiguring drives means rebuilding the bootstrap, not mutating in place.

## R-# trace

* **R-PE \u2014 prediction error is the primitive.** Drive deviation outside the homeostatic band IS the slow-scale prediction error. Total PE is the sum of `pe_weight * deviation` across out-of-band drives. In-band drives contribute exactly 0 (homeostasis is silent). When this exceeds `proactive_pe_threshold` the lifeform surfaces a proactive `FollowupItem` \u2014 a behavioral consequence of internal surprise, not of user input.
* **R1 \u2014 multi-timescale.** Vitals run at the `online-fast` (per-tick) layer for decay and at the `session-medium` (per-turn) layer for recharge. They DO NOT share parameters with the kernel's per-turn PE owner; both are PE owners at distinct frequencies, exactly as NL prescribes.
* **R8 \u2014 snapshot-first.** `VitalsSnapshot` is the single public surface. Three planned consumers \u2014 `FollowupManager` (already wired), `PromptPlanner` (already wired through `plan(..., vitals=...)`), and `lifeform-evolution` benchmark metrics (TODO) \u2014 read this snapshot and never reach into the owner.
* **R11 \u2014 internal state must be nameable and publishable.** Every drive has a stable name; its level, deviation, out-of-band flag, and PE contribution are all on the snapshot. Reflection / evaluation / rollback can all operate on this surface.
* **R14 \u2014 regime persistent identity.** Drives interact with regimes through `recharge_per_regime`: an emotional-support turn recharges `bond_warmth` more than a problem-solving turn. The regime is therefore not just a prompt label \u2014 it has measurable consequences on the lifeform's persistent internal state.

## Companion vertical (`lifeform-domain-emogpt`)

`build_companion_vitals_bootstrap()` ships three drives:

| Drive                     | Decay/tick | Per-turn | Regime bonuses                                           |
|---------------------------|-----------:|---------:|----------------------------------------------------------|
| `bond_warmth`             |      0.005 |     0.02 | emotional_support +0.18 / repair +0.20 / exploration +0.10 |
| `user_engagement`         |      0.020 |     0.30 | (none)                                                   |
| `conversation_continuity` |      0.012 |     0.15 | (none)                                                   |

Threshold `1.0`, cooldown `60` ticks. `build_companion_lifeform()` enables this by default; pass `use_vitals_bootstrap=False` for ablation.

## Open follow-ups

* **Synthesizer-time vitals injection.** The kernel's `ResponseSynthesizer` ABC does not pass session context to `synthesize()`, so vitals currently reach the planner via direct `plan(..., vitals=...)` calls in tests but not through the kernel's call stack during `run_turn`. A clean fix is per-session synthesizers; tracked separately.
* **Vitals-aware regime calibration.** `lifeform-evolution.regime_calibrator` could learn `recharge_per_regime` weights from traces. Today they are hand-tuned configuration.
* **Reflection consumes vitals deviation history.** R6 says reflection should consolidate PE into policy. The slow-loop owner could read accumulated vitals PE and shape long-horizon strategy.
