"""Pair-programmer vertical's drive set.

Drives encode "what this lifeform attends to between turns". For the
pair-programmer archetype the canonical signature is:

* ``solution_clarity`` \u2014 how well the user's intent has been narrowed.
  Decays slowly while exploring; recharges noticeably on
  ``problem_solving`` regime turns. Out-of-band when low: the lifeform
  should feel "we still don't have a sharp question" pressure.
* ``code_freshness`` \u2014 how recently the partner has looked at concrete
  code. Decays the fastest; recharges on every user turn. Modelled as a
  proxy for "have I lost touch with the actual artefact" pressure.
* ``direction_certainty`` \u2014 confidence in the chosen approach. Decays
  during ``guided_exploration`` (we are still unsure); recharges on
  ``problem_solving`` (we have committed to a path).

The deliberate contrast with ``lifeform-domain-emogpt`` is the point: the
companion vertical's drives are about **bond and engagement**, this
vertical's drives are about **intent clarity and direction**. The kernel
does not see any of these names \u2014 they are vertical configuration that
the kernel consumes through the same ``VitalsBootstrap`` shape.
"""

from __future__ import annotations

from lifeform_core import DriveSpec, VitalsBootstrap


def build_coding_vitals_bootstrap() -> VitalsBootstrap:
    """Construct the pair-programmer vertical's vitals bootstrap.

    Threshold is calibrated so 25\u201330 idle SYSTEM ticks push the lifeform
    above the proactive PE threshold \u2014 a partner who has been left alone
    for half a minute should start noticing that the conversation has
    drifted, not just sit silent indefinitely.
    """
    solution_clarity = DriveSpec(
        name="solution_clarity",
        target=0.7,
        homeostatic_band=(0.5, 0.85),
        decay_per_tick=0.004,
        pe_weight=1.0,
        initial_level=0.5,
        recharge_per_turn=0.05,
        recharge_per_regime={
            "problem_solving": 0.20,
            "guided_exploration": 0.08,
        },
    )
    code_freshness = DriveSpec(
        name="code_freshness",
        target=0.7,
        homeostatic_band=(0.4, 0.9),
        decay_per_tick=0.018,
        pe_weight=1.0,
        initial_level=0.55,
        recharge_per_turn=0.25,
        recharge_per_regime={},
    )
    direction_certainty = DriveSpec(
        name="direction_certainty",
        target=0.6,
        homeostatic_band=(0.35, 0.85),
        decay_per_tick=0.010,
        pe_weight=0.7,
        initial_level=0.45,
        recharge_per_turn=0.05,
        recharge_per_regime={
            "problem_solving": 0.18,
            "guided_exploration": -0.05,  # exploration deliberately drains certainty
        },
    )
    return VitalsBootstrap(
        schema_version=1,
        drives=(solution_clarity, code_freshness, direction_certainty),
        proactive_pe_threshold=1.0,
        proactive_followup_priority=0.55,
        proactive_cooldown_ticks=60,
    )
