"""Companion vertical's drive set.

Verticals encode their product priors as drives \u2014 what should the
lifeform "feel" between turns? For the relationship-companion archetype
the canonical signature is:

* ``bond_warmth`` \u2014 affective closeness to the user. Decays slowly when
  silent; recharges noticeably during emotional support and guided
  exploration regimes (the regimes where the lifeform actually does
  relational work). Out-of-band when it falls below 0.5: the companion
  should feel "I haven't connected lately" pressure.
* ``user_engagement`` \u2014 how active the user is. Decays the fastest;
  recharges by any user turn. Drops trigger proactive check-ins.
* ``conversation_continuity`` \u2014 whether the current scene has momentum.
  Medium decay; recharges on every turn regardless of regime. Used as a
  scene-staleness signal for ``end_scene`` heuristics.

These drives are configuration, not learned parameters. Diffing this file
across vertical revisions tells you exactly how the companion's "internal
sense of itself" changed. If a future scenario revision proves the drive
shape needs adapting, retraining the bootstraps will not change this; we
edit this file and ship a new release.
"""

from __future__ import annotations

from lifeform_core import DriveSpec, VitalsBootstrap


def build_companion_vitals_bootstrap() -> VitalsBootstrap:
    """Construct the companion vertical's vitals bootstrap.

    Threshold rationale: with three drives at ``pe_weight=1.0``, sitting
    fully idle takes \u224820 SYSTEM ticks for the per-drive deviations to push
    the total above 1.0 (the proactive threshold). At a typical
    ``system_tick_seconds=1.0`` that's a 20-second silence window before
    the first proactive check-in surfaces. ``proactive_cooldown_ticks=60``
    prevents repeated firing within a minute.
    """
    bond_warmth = DriveSpec(
        name="bond_warmth",
        target=0.7,
        homeostatic_band=(0.5, 0.85),
        decay_per_tick=0.005,
        pe_weight=1.0,
        initial_level=0.6,
        recharge_per_turn=0.02,
        recharge_per_regime={
            "emotional_support": 0.18,
            "guided_exploration": 0.10,
            "repair_de_escalation": 0.20,
            "casual_social": 0.06,
        },
    )
    user_engagement = DriveSpec(
        name="user_engagement",
        target=0.7,
        homeostatic_band=(0.4, 0.9),
        decay_per_tick=0.020,
        pe_weight=1.0,
        initial_level=0.5,
        recharge_per_turn=0.30,
        recharge_per_regime={},
    )
    conversation_continuity = DriveSpec(
        name="conversation_continuity",
        target=0.5,
        homeostatic_band=(0.3, 0.8),
        decay_per_tick=0.012,
        pe_weight=0.7,
        initial_level=0.5,
        recharge_per_turn=0.15,
        recharge_per_regime={},
    )
    return VitalsBootstrap(
        schema_version=1,
        drives=(bond_warmth, user_engagement, conversation_continuity),
        proactive_pe_threshold=1.0,
        proactive_followup_priority=0.55,
        proactive_cooldown_ticks=60,
    )
