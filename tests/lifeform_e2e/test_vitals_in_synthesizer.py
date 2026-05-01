"""Vitals reaching the synthesizer through per-session synthesizer cloning.

The contract pinned here:

* ``Lifeform.create_session`` produces a per-session
  ``GroundedResponseSynthesizer`` clone whose vitals provider points at
  THIS session's ``VitalsModule``. The Brain-level synthesizer is left
  alone \u2014 no shared mutable state across sessions.
* When a drive is in-band, no ``vitals_pressure=...`` tag appears in
  the response rationale \u2014 vitals are silent, exactly like homeostasis.
* When a drive is out-of-band, the planner / synthesizer surface a
  ``vitals_pressure=NAME`` rationale tag. This is true on both code paths
  (custom render + base-delegate render via ``_attach_plan_rationale``).
* The vitals provider is bound by closure, so two concurrent sessions of
  the same Lifeform see different drive states without leaking into each
  other.

We use a deliberately fast-decaying vitals bootstrap so the tests cross
the proactive threshold within a handful of idle ticks.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_core import (
    DriveSpec,
    Lifeform,
    VitalsBootstrap,
)
from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_expression import GroundedResponseSynthesizer
from lifeform_expression.response_synthesizer import GroundedResponseSynthesizer as _G


def _fast_vitals() -> VitalsBootstrap:
    return VitalsBootstrap(
        drives=(
            DriveSpec(
                name="user_engagement",
                target=0.7,
                homeostatic_band=(0.4, 0.9),
                decay_per_tick=0.10,
                pe_weight=1.0,
                initial_level=0.5,
                recharge_per_turn=0.30,
            ),
        ),
        proactive_pe_threshold=0.2,
        proactive_cooldown_ticks=5,
    )


@pytest.fixture
def companion_with_grounded_synth_and_vitals() -> Lifeform:
    base = build_companion_lifeform()
    cfg = replace(base.config, vitals_bootstrap=_fast_vitals())
    return Lifeform(cfg, response_synthesizer=GroundedResponseSynthesizer())


# ---------------------------------------------------------------------------
# Per-session synthesizer wiring
# ---------------------------------------------------------------------------


def test_brain_session_uses_a_per_session_synthesizer_clone(
    companion_with_grounded_synth_and_vitals: Lifeform,
):
    life = companion_with_grounded_synth_and_vitals
    brain_synth = life._init_kwargs["response_synthesizer"]
    assert isinstance(brain_synth, _G)
    assert brain_synth.vitals_snapshot_provider is None  # brain default is unbound

    session = life.create_session(session_id="clone-check")
    runner = session.brain_session.runner
    session_synth = runner._response_synthesizer
    assert isinstance(session_synth, _G)
    assert session_synth is not brain_synth, (
        "Lifeform.create_session must clone the brain-level synthesizer; "
        "binding a vitals provider on the shared instance would make all "
        "sessions of the brain race on each other's drive state."
    )
    assert session_synth.vitals_snapshot_provider is not None
    # The clone reuses the brain's planner so per-session synthesis still
    # consults the same plan strategy.
    assert session_synth.planner is brain_synth.planner


def test_brain_synthesizer_is_left_unchanged_after_create_session(
    companion_with_grounded_synth_and_vitals: Lifeform,
):
    life = companion_with_grounded_synth_and_vitals
    brain_synth = life._init_kwargs["response_synthesizer"]
    life.create_session(session_id="s1")
    life.create_session(session_id="s2")
    assert brain_synth.vitals_snapshot_provider is None, (
        "Brain-level synthesizer must NEVER be mutated by per-session wiring."
    )


def test_lifeform_without_vitals_still_clones_for_interlocutor_state(
    companion_with_grounded_synth_and_vitals: Lifeform,
):
    """Even without vitals the synthesizer is cloned per session.

    Pre-Gap-9-slice-2c the no-vitals path skipped cloning entirely
    (no per-session state to bind). Slice 2c adds an
    ``interlocutor_state_provider`` that is ALWAYS per-session
    (it reads ``LifeformSession.interlocutor_state``), so the
    brain-default synthesizer is now ALWAYS cloned to bind that
    closure. Vitals provider stays unbound when no vitals config
    is supplied; that's the only difference from the with-vitals
    path.
    """
    base = companion_with_grounded_synth_and_vitals
    cfg = replace(base.config, vitals_bootstrap=None)
    life = Lifeform(cfg, response_synthesizer=GroundedResponseSynthesizer())
    brain_synth = life._init_kwargs["response_synthesizer"]
    session = life.create_session(session_id="no-vitals")
    runner = session.brain_session.runner
    session_synth = runner._response_synthesizer
    assert session_synth is not brain_synth
    assert session_synth.vitals_snapshot_provider is None
    assert session_synth.interlocutor_state_provider is not None
    # Same planner instance reused across the clone \u2014 only
    # providers are session-bound.
    assert session_synth.planner is brain_synth.planner


# ---------------------------------------------------------------------------
# End-to-end: vitals reach the rationale only when drives are out-of-band
# ---------------------------------------------------------------------------


async def test_in_band_drive_produces_no_vitals_pressure_tag(
    companion_with_grounded_synth_and_vitals: Lifeform,
):
    session = companion_with_grounded_synth_and_vitals.create_session(
        session_id="in-band"
    )
    result = await session.run_turn("just saying hi")
    assert "vitals_pressure=" not in result.response.rationale


async def test_out_of_band_drive_surfaces_vitals_pressure_tag(
    companion_with_grounded_synth_and_vitals: Lifeform,
):
    session = companion_with_grounded_synth_and_vitals.create_session(
        session_id="out-of-band"
    )
    # Drain user_engagement: decay 0.10/tick from initial 0.50 \u2192 below band
    # of 0.4 by tick 2, total_pe past threshold 0.2 by tick 6.
    await session.advance_tick(10)
    snap = session.vitals_snapshot
    assert snap is not None and snap.above_proactive_threshold

    result = await session.run_turn("still here")
    assert "vitals_pressure=user_engagement" in result.response.rationale, (
        f"expected vitals_pressure tag in rationale, got: "
        f"{result.response.rationale[-300:]}"
    )


# ---------------------------------------------------------------------------
# Two concurrent sessions: their vitals don't leak across the closure
# ---------------------------------------------------------------------------


async def test_two_sessions_have_independent_vitals_providers(
    companion_with_grounded_synth_and_vitals: Lifeform,
):
    life = companion_with_grounded_synth_and_vitals
    s_quiet = life.create_session(session_id="quiet")
    s_loud = life.create_session(session_id="loud")

    # Push s_loud into the out-of-band regime; s_quiet stays calm.
    await s_loud.advance_tick(10)
    assert s_loud.vitals_snapshot.above_proactive_threshold is True
    assert s_quiet.vitals_snapshot.above_proactive_threshold is False

    quiet_result = await s_quiet.run_turn("hi")
    loud_result = await s_loud.run_turn("hello again")

    assert "vitals_pressure=" not in quiet_result.response.rationale
    assert "vitals_pressure=user_engagement" in loud_result.response.rationale


# ---------------------------------------------------------------------------
# Planner-level: vitals tag appears in plan.rationale_tags
# ---------------------------------------------------------------------------


def test_grounded_synthesizer_pulls_provider_through_to_planner_tags():
    from lifeform_core import DriveLevel, VitalsSnapshot
    from lifeform_expression import PromptPlanner
    from volvence_zero.agent.response import ResponseContext

    pressing = VitalsSnapshot(
        tick_index=99,
        drive_levels=(
            DriveLevel(
                name="user_engagement",
                level=0.05,
                target=0.7,
                deviation=0.65,
                out_of_band=True,
                pe_contribution=0.65,
            ),
        ),
        total_pe=0.65,
        above_proactive_threshold=True,
    )

    synth = GroundedResponseSynthesizer(
        planner=PromptPlanner(),
        vitals_snapshot_provider=lambda: pressing,
    )

    # Build a minimal ResponseContext that triggers the JUDGMENT_PROCESS
    # delegate path; even on that path the rationale must carry the tag.
    context = ResponseContext(
        regime_id="guided_exploration",
        regime_name="guided_exploration",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.5,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="ssl-only",
        user_input="hi",
    )
    response = synth.synthesize(context=context, assembly=None)
    assert "vitals_pressure=user_engagement" in response.rationale
