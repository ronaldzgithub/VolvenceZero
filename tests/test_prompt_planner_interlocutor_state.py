"""PromptPlanner + InterlocutorState integration (Gap 9 slice 2c).

The planner gains an optional ``interlocutor_state`` parameter; when
present and the readout has confidence >= 0.30 the planner modulates
section selection and question budget along the 12 continuous axes.

These tests pin the policy at the threshold boundaries so future
calibration is "tune one threshold per axis", not "rewrite the
modulation logic".

Importantly: the tests verify that **same context + different
InterlocutorState produces different plans**. That's the closure of
the perception-response loop \u2014 the kernel can sense user state, and
now the prompt acts on it.
"""

from __future__ import annotations

from dataclasses import replace

from lifeform_expression.prompt_planner import (
    PromptPlanner,
    SectionId,
    TurnIntent,
)
from volvence_zero.agent.response import RepairExpressionAdvisory, ResponseContext
from volvence_zero.interlocutor import InterlocutorState


def _context(regime_id: str = "emotional_support") -> ResponseContext:
    return ResponseContext(
        regime_id=regime_id,
        regime_name=regime_id,
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="idle",
        user_input="",
    )


def _repair_advisory(*, confidence: float = 0.67) -> RepairExpressionAdvisory:
    return RepairExpressionAdvisory(
        rupture_kind="over_directive",
        confidence=confidence,
        signal_strength=0.95,
        description="Rupture kind=over_directive strength=0.95 confidence=0.67.",
    )


def _neutral_state(*, confidence: float = 0.50, **overrides) -> InterlocutorState:
    """Mid-point readout with high enough confidence to fire."""
    base = InterlocutorState(readout_confidence=confidence)
    return replace(base, **overrides)


# ---------------------------------------------------------------------------
# Cold-start / low-confidence => no-op (back-compat)
# ---------------------------------------------------------------------------


def test_planner_without_interlocutor_state_matches_baseline() -> None:
    """Same plan with ``None`` and absent kwarg."""
    planner = PromptPlanner()
    baseline = planner.plan(context=_context(), assembly=None)
    explicit_none = planner.plan(
        context=_context(), assembly=None, interlocutor_state=None
    )
    assert baseline.sections == explicit_none.sections
    assert baseline.question_budget == explicit_none.question_budget


def test_low_confidence_readout_is_a_noop() -> None:
    """``readout_confidence < 0.30`` => no section / budget changes.

    This is the cold-start guard: a fresh session with no kernel
    snapshots should not pick up phantom modulation.
    """
    planner = PromptPlanner()
    baseline = planner.plan(context=_context(), assembly=None)

    # Confidence is below the floor. The other axes are deliberately
    # set to "extreme" values to PROVE the gate is what's stopping
    # modulation, not the axis values themselves.
    cold_state = _neutral_state(
        confidence=0.10,
        emotional_weight=0.95,
        resistance_level=0.90,
        rapport_warmth=0.05,
    )
    cold_plan = planner.plan(
        context=_context(), assembly=None, interlocutor_state=cold_state
    )
    assert cold_plan.sections == baseline.sections


# ---------------------------------------------------------------------------
# Repair alpha advisory: typed advisory only, default off
# ---------------------------------------------------------------------------


def test_repair_advisory_is_noop_when_alpha_disabled() -> None:
    planner = PromptPlanner()
    context = replace(_context(regime_id="acquaintance_building"), repair_advisory=_repair_advisory())

    plan = planner.plan(context=context, assembly=None)

    assert plan.intent is TurnIntent.WARMTH_FIRST
    assert not any(tag.startswith("repair_alpha=") for tag in plan.rationale_tags)


def test_repair_advisory_forces_repair_first_when_alpha_enabled() -> None:
    planner = PromptPlanner(repair_alpha_enabled=True)
    context = replace(_context(regime_id="acquaintance_building"), repair_advisory=_repair_advisory())

    plan = planner.plan(context=context, assembly=None)

    assert plan.intent is TurnIntent.REPAIR_FIRST
    assert plan.sections[:2] == (
        SectionId.ACKNOWLEDGE_PRESSURE,
        SectionId.REGIME_FRAME,
    )
    assert plan.question_budget == 0
    assert "repair_alpha=over_directive" in plan.rationale_tags


def test_low_confidence_repair_advisory_is_noop() -> None:
    planner = PromptPlanner(repair_alpha_enabled=True)
    context = replace(
        _context(regime_id="acquaintance_building"),
        repair_advisory=_repair_advisory(confidence=0.20),
    )

    plan = planner.plan(context=context, assembly=None)

    assert plan.intent is TurnIntent.WARMTH_FIRST
    assert not any(tag.startswith("repair_alpha=") for tag in plan.rationale_tags)


def test_repair_alpha_does_not_use_raw_user_text() -> None:
    planner = PromptPlanner(repair_alpha_enabled=True)
    context = replace(
        _context(regime_id="acquaintance_building"),
        user_input="That felt over-directive and optimized.",
        repair_advisory=None,
    )

    plan = planner.plan(context=context, assembly=None)

    assert plan.intent is TurnIntent.WARMTH_FIRST


# ---------------------------------------------------------------------------
# Per-axis modulation: each rule fires the expected change
# ---------------------------------------------------------------------------


def test_high_emotional_weight_inserts_acknowledge_pressure() -> None:
    """Above-threshold ``emotional_weight`` adds ACKNOWLEDGE_PRESSURE first."""
    planner = PromptPlanner()
    state = _neutral_state(emotional_weight=0.70)
    plan = planner.plan(
        context=_context(regime_id="problem_solving"),
        assembly=None,
        interlocutor_state=state,
    )
    assert SectionId.ACKNOWLEDGE_PRESSURE in plan.sections
    assert plan.sections[0] is SectionId.ACKNOWLEDGE_PRESSURE
    assert any(t.startswith("il_add=acknowledge_pressure") for t in plan.rationale_tags)


def test_high_resistance_inserts_acknowledge_pressure() -> None:
    planner = PromptPlanner()
    state = _neutral_state(resistance_level=0.80)
    plan = planner.plan(
        context=_context(regime_id="guided_exploration"),
        assembly=None,
        interlocutor_state=state,
    )
    assert SectionId.ACKNOWLEDGE_PRESSURE in plan.sections


def test_negative_trust_inserts_acknowledge_pressure() -> None:
    planner = PromptPlanner()
    state = _neutral_state(trust_signal=-0.40)
    plan = planner.plan(
        context=_context(regime_id="acquaintance_building"),
        assembly=None,
        interlocutor_state=state,
    )
    assert SectionId.ACKNOWLEDGE_PRESSURE in plan.sections


def test_cold_rapport_with_engagement_adds_continuity_note() -> None:
    """Cold rapport AND engagement => continuity note. Cold + disengaged is a no-op."""
    planner = PromptPlanner()
    cold_engaged = _neutral_state(
        rapport_warmth=0.20, engagement_intensity=0.55
    )
    cold_disengaged = _neutral_state(
        rapport_warmth=0.20, engagement_intensity=0.10
    )

    plan_engaged = planner.plan(
        context=_context(regime_id="problem_solving"),
        assembly=None,
        interlocutor_state=cold_engaged,
    )
    plan_disengaged = planner.plan(
        context=_context(regime_id="problem_solving"),
        assembly=None,
        interlocutor_state=cold_disengaged,
    )
    assert SectionId.CONTINUITY_NOTE in plan_engaged.sections
    assert SectionId.CONTINUITY_NOTE not in plan_disengaged.sections


def test_high_pace_pressure_drops_meta_sections() -> None:
    planner = PromptPlanner()
    state = _neutral_state(pace_pressure=0.85)

    # Use a regime whose default plan includes OPEN_LOOP_HANDOFF so we
    # have something to drop.
    plan = planner.plan(
        context=_context(regime_id="problem_solving"),
        assembly=None,
        interlocutor_state=state,
    )
    assert SectionId.OPEN_LOOP_HANDOFF not in plan.sections
    assert SectionId.REFLECTION_HOOK not in plan.sections


# ---------------------------------------------------------------------------
# Question budget cap
# ---------------------------------------------------------------------------


def test_high_emotional_weight_caps_question_budget() -> None:
    planner = PromptPlanner()
    state = _neutral_state(emotional_weight=0.75)
    # Use a context that would normally allow 1 question
    # (CLARIFY_FIRST). The intent for ``guided_exploration`` is
    # ``JUDGMENT_PROCESS`` whose default budget is 0; we instead
    # use ``acquaintance_building`` which yields ``WARMTH_FIRST``
    # with default 1.
    plan = planner.plan(
        context=_context(regime_id="acquaintance_building"),
        assembly=None,
        interlocutor_state=state,
    )
    assert plan.question_budget == 0


def test_low_directness_caps_question_budget() -> None:
    planner = PromptPlanner()
    state = _neutral_state(directness=0.20)
    plan = planner.plan(
        context=_context(regime_id="acquaintance_building"),
        assembly=None,
        interlocutor_state=state,
    )
    assert plan.question_budget == 0


def test_high_pace_pressure_caps_question_budget() -> None:
    planner = PromptPlanner()
    state = _neutral_state(pace_pressure=0.90)
    plan = planner.plan(
        context=_context(regime_id="acquaintance_building"),
        assembly=None,
        interlocutor_state=state,
    )
    assert plan.question_budget == 0


def test_neutral_state_does_not_cap_question_budget() -> None:
    planner = PromptPlanner()
    state = _neutral_state()
    baseline = planner.plan(
        context=_context(regime_id="acquaintance_building"),
        assembly=None,
    )
    with_state = planner.plan(
        context=_context(regime_id="acquaintance_building"),
        assembly=None,
        interlocutor_state=state,
    )
    assert with_state.question_budget == baseline.question_budget


# ---------------------------------------------------------------------------
# Rationale tags
# ---------------------------------------------------------------------------


def test_rationale_tags_record_confidence_when_modulating() -> None:
    planner = PromptPlanner()
    state = _neutral_state(confidence=0.65, emotional_weight=0.80)
    plan = planner.plan(
        context=_context(regime_id="problem_solving"),
        assembly=None,
        interlocutor_state=state,
    )
    assert any(t.startswith("interlocutor_conf=") for t in plan.rationale_tags)


def test_rationale_tags_omit_confidence_below_floor() -> None:
    planner = PromptPlanner()
    state = _neutral_state(confidence=0.05, emotional_weight=0.80)
    plan = planner.plan(
        context=_context(regime_id="problem_solving"),
        assembly=None,
        interlocutor_state=state,
    )
    assert not any(t.startswith("interlocutor_conf=") for t in plan.rationale_tags)


# ---------------------------------------------------------------------------
# End-to-end: same context, different state => different plan
# ---------------------------------------------------------------------------


def test_different_states_yield_different_plans() -> None:
    """Demonstrates the perception-response loop is closed.

    The key claim of slice 2c: the planner now reacts to user
    state. Two readouts that differ on emotional_weight + pace
    produce visibly different section lists for the SAME regime
    and assembly.
    """
    planner = PromptPlanner()
    calm = _neutral_state(emotional_weight=0.20, pace_pressure=0.30)
    distressed = _neutral_state(emotional_weight=0.85, pace_pressure=0.85)

    ctx = _context(regime_id="problem_solving")
    plan_calm = planner.plan(context=ctx, assembly=None, interlocutor_state=calm)
    plan_distressed = planner.plan(
        context=ctx, assembly=None, interlocutor_state=distressed
    )

    assert plan_calm.sections != plan_distressed.sections
    # And the rationales also differ.
    assert plan_calm.rationale_tags != plan_distressed.rationale_tags
