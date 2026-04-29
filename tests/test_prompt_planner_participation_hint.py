"""PromptPlanner + ParticipationHint integration (Gap 8).

Gap 8 adds an optional ``participation_hint`` parameter to
``PromptPlanner.plan``. When provided, the planner drops sections
marked SILENT by the hint's panorama / method / task levels. When
None, behaviour is identical to pre-Gap-8.

These tests stub ``ResponseContext`` + ``ResponseAssemblySnapshot``
with minimal values so the section-picking path is exercised
directly; no kernel involvement.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_expression.prompt_planner import (
    PromptPlanner,
    SectionId,
    TurnIntent,
)
from volvence_zero.agent.response import ResponseContext
from volvence_zero.regime import (
    ParticipationFlowKind,
    ParticipationHint,
    ParticipationLevel,
)


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


# ---------------------------------------------------------------------------
# No hint => baseline unchanged (back-compat)
# ---------------------------------------------------------------------------


def test_planner_without_hint_produces_baseline_plan() -> None:
    planner = PromptPlanner()
    plan_without = planner.plan(context=_context(), assembly=None)
    plan_with_default = planner.plan(
        context=_context(),
        assembly=None,
        participation_hint=ParticipationHint(),  # all STRUCTURED default
    )
    # Default hint (all STRUCTURED) adds rationale tags but MUST NOT
    # drop any sections \u2014 STRUCTURED is "full inclusion".
    assert plan_without.sections == plan_with_default.sections


# ---------------------------------------------------------------------------
# panorama_level=SILENT => drop CLARIFICATION
# ---------------------------------------------------------------------------


def test_panorama_silent_drops_clarification() -> None:
    """Force a baseline that includes CLARIFICATION and verify
    panorama=SILENT drops it. We subclass PromptPlanner so the
    test does not depend on intent-picking heuristics which are
    exercised elsewhere.
    """

    class _ClarifyPlanner(PromptPlanner):
        def _pick_intent(self, *, context, assembly):
            return TurnIntent.CLARIFY_FIRST

    planner = _ClarifyPlanner()
    ctx = _context(regime_id="guided_exploration")
    baseline = planner.plan(context=ctx, assembly=None)
    assert SectionId.CLARIFICATION in baseline.sections

    hint = ParticipationHint(
        flow_kind=ParticipationFlowKind.INFO,
        panorama_level=ParticipationLevel.SILENT,
        method_level=ParticipationLevel.STRUCTURED,
        task_level=ParticipationLevel.STRUCTURED,
    )
    filtered = planner.plan(context=ctx, assembly=None, participation_hint=hint)
    assert SectionId.CLARIFICATION not in filtered.sections
    assert any("hint_dropped=clarification" in tag for tag in filtered.rationale_tags)


# ---------------------------------------------------------------------------
# method_level=SILENT => drop REGIME_FRAME
# ---------------------------------------------------------------------------


def test_method_silent_drops_regime_frame() -> None:
    planner = PromptPlanner()
    ctx = _context(regime_id="emotional_support")
    baseline = planner.plan(context=ctx, assembly=None)
    assert SectionId.REGIME_FRAME in baseline.sections

    hint = ParticipationHint(
        method_level=ParticipationLevel.SILENT,
    )
    filtered = planner.plan(context=ctx, assembly=None, participation_hint=hint)
    assert SectionId.REGIME_FRAME not in filtered.sections


# ---------------------------------------------------------------------------
# task_level=SILENT => drop NEXT_STEP and OPEN_LOOP_HANDOFF
# ---------------------------------------------------------------------------


def test_task_silent_drops_next_step_and_open_loop_handoff() -> None:
    planner = PromptPlanner()
    # repair_and_deescalation base set includes both NEXT_STEP and
    # OPEN_LOOP_HANDOFF; use it to verify both get dropped.
    ctx = _context(regime_id="repair_and_deescalation")
    baseline = planner.plan(context=ctx, assembly=None)
    assert SectionId.NEXT_STEP in baseline.sections
    assert SectionId.OPEN_LOOP_HANDOFF in baseline.sections

    hint = ParticipationHint(
        task_level=ParticipationLevel.SILENT,
    )
    filtered = planner.plan(context=ctx, assembly=None, participation_hint=hint)
    assert SectionId.NEXT_STEP not in filtered.sections
    assert SectionId.OPEN_LOOP_HANDOFF not in filtered.sections


# ---------------------------------------------------------------------------
# Overapplied hint: all-SILENT falls back to unfiltered
# ---------------------------------------------------------------------------


def test_overapplied_all_silent_hint_falls_back_and_records_rationale() -> None:
    planner = PromptPlanner()
    # casual_social base = (REGIME_FRAME, CONTINUITY_NOTE). Dropping
    # REGIME_FRAME via method=silent leaves just CONTINUITY_NOTE, which
    # is fine. But if we ALSO silence task, it doesn't drop anything
    # more. Instead, pick a regime whose entire base set is droppable
    # if we silenced everything \u2014 emotional_support has
    # (ACKNOWLEDGE_PRESSURE, REGIME_FRAME, NEXT_STEP, CONTINUITY_NOTE).
    # Dropping REGIME_FRAME + NEXT_STEP leaves ACKNOWLEDGE_PRESSURE +
    # CONTINUITY_NOTE, which is non-empty. So over-application is
    # hard to trigger with realistic hints.
    #
    # To test the overapplied path deterministically we pass a tiny
    # assembly + context that would produce a 1-section base, then
    # try to drop that one section.
    #
    # Simpler: construct a PromptPlanner subclass where _pick_sections
    # returns just one section that the hint wants to drop.
    class _MiniPlanner(PromptPlanner):
        def _pick_sections(self, **kwargs):
            return [SectionId.REGIME_FRAME]

    mini = _MiniPlanner()
    hint = ParticipationHint(method_level=ParticipationLevel.SILENT)
    plan = mini.plan(context=_context(), assembly=None, participation_hint=hint)
    # Fallback: sections stay as-is because dropping REGIME_FRAME
    # would leave the plan empty.
    assert SectionId.REGIME_FRAME in plan.sections
    assert any("hint_overapplied_skipped" in tag for tag in plan.rationale_tags)


# ---------------------------------------------------------------------------
# Rationale tags carry hint metadata for audit
# ---------------------------------------------------------------------------


def test_rationale_tags_include_participation_levels() -> None:
    planner = PromptPlanner()
    hint = ParticipationHint(
        flow_kind=ParticipationFlowKind.SOCIAL,
        panorama_level=ParticipationLevel.SILENT,
        method_level=ParticipationLevel.BRIEF,
        task_level=ParticipationLevel.STRUCTURED,
    )
    plan = planner.plan(
        context=_context("casual_social"),
        assembly=None,
        participation_hint=hint,
    )
    tags_str = " | ".join(plan.rationale_tags)
    assert "flow_kind=social" in tags_str
    assert "panorama:silent" in tags_str
    assert "method:brief" in tags_str
    assert "task:structured" in tags_str


def test_rationale_tags_without_hint_do_not_mention_participation() -> None:
    planner = PromptPlanner()
    plan = planner.plan(context=_context(), assembly=None)
    tags_str = " | ".join(plan.rationale_tags)
    assert "participation=" not in tags_str
    assert "flow_kind=" not in tags_str
