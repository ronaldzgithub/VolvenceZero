"""CP-04 (intent-alignment W2.E): prompt planner consumes AffordanceSnapshot.

The planner adds an owner-approved ``AFFORDANCE_OFFER`` section when the
AffordanceModule published a non-None ``selected`` candidate, surfaces typed
audit tags for blocked candidates, and never re-scores or re-gates - the
owner's decision is consumed as-is.
"""

from __future__ import annotations

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceSafety,
)
from lifeform_affordance.snapshot import AffordanceCandidate, AffordanceSnapshot
from lifeform_expression.prompt_planner import PromptPlanner, SectionId
from lifeform_expression.response_synthesizer import GroundedResponseSynthesizer
from volvence_zero.agent.response import ResponseContext

_HINT = (
    "Use this planner probe when validating that owner-approved affordance "
    "selections surface as an AFFORDANCE_OFFER section in the prompt plan."
)


def _context(regime_id: str = "problem_solving") -> ResponseContext:
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


def _descriptor(name: str) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=AffordanceKind.TOOL,
        version="1.0.0",
        display_name=name.replace("_", " ").title(),
        description="Planner affordance probe descriptor.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Do not use outside this test.",
        parameters_schema={"type": "object", "properties": {}},
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.INSTANT),
        safety_model=AffordanceSafety(),
    )


def _candidate(
    name: str, *, score: float, blocked_reason: str = ""
) -> AffordanceCandidate:
    descriptor = _descriptor(name)
    return AffordanceCandidate(
        descriptor_name=name,
        score=score,
        rationale=f"test:score={score:.2f}",
        expected_cost=descriptor.cost_model,
        blocked_reason=blocked_reason,
    )


def _snapshot_with_selection() -> AffordanceSnapshot:
    selected = _candidate("web_lookup", score=0.82)
    return AffordanceSnapshot(
        available=(_descriptor("web_lookup"), _descriptor("run_test")),
        candidates_for_turn=(selected, _candidate("run_test", score=0.41)),
        selected=selected,
        description="planner probe: selection present",
    )


def _snapshot_all_blocked() -> AffordanceSnapshot:
    return AffordanceSnapshot(
        available=(_descriptor("web_lookup"), _descriptor("run_test")),
        candidates_for_turn=(
            _candidate(
                "web_lookup",
                score=0.0,
                blocked_reason="consent_blocked:external_action:kind=tool:consent=denied",
            ),
            _candidate(
                "run_test",
                score=0.0,
                blocked_reason="regime_blocked:emotional_support in ['emotional_support']",
            ),
        ),
        selected=None,
        description="planner probe: all blocked",
    )


def test_selected_affordance_adds_owner_approved_section() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_context(),
        assembly=None,
        affordance_snapshot=_snapshot_with_selection(),
    )
    assert SectionId.AFFORDANCE_OFFER in plan.sections
    assert "affordance=selected(name=web_lookup,score=0.82)" in plan.rationale_tags
    assert "affordance_add=affordance_offer(web_lookup)" in plan.rationale_tags


def test_blocked_candidates_surface_audit_tag_without_section() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_context(),
        assembly=None,
        affordance_snapshot=_snapshot_all_blocked(),
    )
    assert SectionId.AFFORDANCE_OFFER not in plan.sections
    assert "affordance_blocked=2" in plan.rationale_tags


def test_no_affordance_snapshot_is_a_no_op() -> None:
    planner = PromptPlanner()
    baseline = planner.plan(context=_context(), assembly=None)
    with_none = planner.plan(
        context=_context(), assembly=None, affordance_snapshot=None
    )
    assert baseline.sections == with_none.sections
    assert baseline.rationale_tags == with_none.rationale_tags


def test_renderer_offers_selected_display_name_without_auto_invoking() -> None:
    text = GroundedResponseSynthesizer._render_affordance_offer(
        affordance_snapshot=_snapshot_with_selection()
    )
    assert "Web Lookup" in text
    assert "I can use" in text
    # Offer-only phrasing: the renderer proposes, never claims execution.
    assert "say the word" in text

    assert (
        GroundedResponseSynthesizer._render_affordance_offer(
            affordance_snapshot=_snapshot_all_blocked()
        )
        == ""
    )
    assert (
        GroundedResponseSynthesizer._render_affordance_offer(affordance_snapshot=None)
        == ""
    )
