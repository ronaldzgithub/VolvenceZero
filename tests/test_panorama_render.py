"""Contract tests for the decision panorama reaching the user.

Everything before this file decided *whether* to open a panorama, *what*
it contains, and *what may be claimed about it*. None of that was
visible: the workspace owner had no consumer and the three gate tiers
produced identical prompt plans.

What is defended here:

* A panorama actually appears — and only when the gate opened.
* The three tiers are genuinely different. Identical output across tiers
  is the exact state this work started from.
* **The claim licence constrains the text.** There is no path from
  overlapping intervals to a sentence naming a winner. This is asserted
  on the licence's effect on wording rather than on the wording itself,
  so rephrasing a sentence can never silently relax what the system is
  entitled to assert.
* A safety hold suppresses the ranking at the point of speech, not just
  in the data structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lifeform_expression.prompt_planner import (  # noqa: E402
    PromptPlanner,
    SectionId,
)
from lifeform_expression.response_synthesizer import (  # noqa: E402
    GroundedResponseSynthesizer,
)
from test_decision_valuation import _fourth_act  # noqa: E402
from test_prompt_planner_participation_hint import _context  # noqa: E402
from volvence_zero.decision_workspace import (  # noqa: E402
    DecisionOption,
    DecisionUnknown,
    DecisionWorkspaceSnapshot,
)
from volvence_zero.decision_workspace.rendering import (  # noqa: E402
    CLAIM_COMPARATIVE,
    CLAIM_NONE,
    CLAIM_ROBUSTNESS,
    plan_panorama_render,
)
from volvence_zero.regime import ParticipationHint, ParticipationLevel  # noqa: E402


def _workspace(
    level: ParticipationLevel,
    *,
    options: int = 4,
    unknowns: int = 2,
    safety_hold: bool = False,
) -> DecisionWorkspaceSnapshot:
    return DecisionWorkspaceSnapshot(
        engagement=level,
        options=tuple(
            DecisionOption(option_id=f"opt-{i}", plan_ref=f"plan-{i}")
            for i in range(options)
        ),
        dimension_refs=("children", "money", "emotion"),
        unknowns=tuple(
            DecisionUnknown(unknown_id=f"u-{i}", belief_ref=f"belief-{i}")
            for i in range(unknowns)
        ),
        safety_hold=safety_hold,
    )


def _render(plan: object | None) -> str:
    return GroundedResponseSynthesizer._render_decision_panorama(
        panorama_render_plan=plan
    )


# ---------------------------------------------------------------------------
# It comes out at all
# ---------------------------------------------------------------------------


def test_a_panorama_actually_reaches_the_response() -> None:
    text = _render(plan_panorama_render(_workspace(ParticipationLevel.STRUCTURED)))
    assert text.strip() != ""


def test_the_planner_adds_the_section() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_context(regime_id="guided_exploration"),
        assembly=None,
        participation_hint=ParticipationHint(
            panorama_level=ParticipationLevel.STRUCTURED
        ),
        panorama_render_plan=plan_panorama_render(
            _workspace(ParticipationLevel.STRUCTURED)
        ),
    )
    assert plan.has_section(SectionId.DECISION_PANORAMA)


def test_closed_gate_produces_no_section_and_no_text() -> None:
    """The common case by a wide margin.

    A closed gate is a real absence, not an empty section that some
    downstream renderer might decide to fill.
    """
    assert plan_panorama_render(_workspace(ParticipationLevel.SILENT)) is None
    assert _render(None) == ""
    planner = PromptPlanner()
    plan = planner.plan(
        context=_context(regime_id="guided_exploration"),
        assembly=None,
        panorama_render_plan=None,
    )
    assert not plan.has_section(SectionId.DECISION_PANORAMA)


def test_the_three_tiers_are_genuinely_different() -> None:
    """The state this work began from: all three tiers rendered the same.

    If this ever collapses back to one output, the gate has stopped
    mattering downstream however well it decides.
    """
    silent = _render(plan_panorama_render(_workspace(ParticipationLevel.SILENT)))
    brief = _render(plan_panorama_render(_workspace(ParticipationLevel.BRIEF)))
    structured = _render(
        plan_panorama_render(_workspace(ParticipationLevel.STRUCTURED))
    )
    assert silent == ""
    assert brief != ""
    assert structured != ""
    assert brief != structured
    assert len(structured) > len(brief)


def test_brief_offers_rather_than_lays_out() -> None:
    """BRIEF must not preempt the decision to think structurally."""
    brief = _render(plan_panorama_render(_workspace(ParticipationLevel.BRIEF)))
    assert "?" in brief
    # No dimensions, no ranking, no next question at this tier.
    for leaked in ("weighed on", "comes out ahead", "worth settling first"):
        assert leaked not in brief


def test_brief_stays_silent_when_there_is_no_real_choice() -> None:
    assert (
        _render(
            plan_panorama_render(_workspace(ParticipationLevel.BRIEF, options=1))
        )
        == ""
    )


# ---------------------------------------------------------------------------
# The licence constrains the text
# ---------------------------------------------------------------------------


def test_overlapping_intervals_never_produce_a_winner_sentence() -> None:
    """The correction to the fourth act's own wording, at the point of speech.

    "现在收益最高的是先分开三个月" reads as a computed result. With
    overlapping ranges no such result exists, and no branch of the
    renderer can reach a sentence that asserts one.
    """
    valuation = _fourth_act()
    render_plan = plan_panorama_render(
        _workspace(ParticipationLevel.STRUCTURED), valuation
    )
    assert render_plan.claim_kind == CLAIM_ROBUSTNESS
    assert render_plan.may_state_a_winner is False
    text = _render(render_plan)
    assert "comes out ahead" not in text
    assert "nothing wins outright" in text


def test_separation_is_the_only_route_to_a_comparative_sentence() -> None:
    from volvence_zero.decision_workspace.valuation import (
        DimensionEstimate,
        Interval,
        evaluate_options,
    )

    separated = evaluate_options(
        (
            DimensionEstimate(
                option_ref="a", dimension_ref="d", interval=Interval(40, 50, 60),
                evidence_refs=("src",),
            ),
            DimensionEstimate(
                option_ref="b", dimension_ref="d", interval=Interval(0, 10, 20),
                evidence_refs=("src",),
            ),
        )
    )
    render_plan = plan_panorama_render(
        _workspace(ParticipationLevel.STRUCTURED, unknowns=0), separated
    )
    assert render_plan.claim_kind == CLAIM_COMPARATIVE
    assert "comes out ahead" in _render(render_plan)


def test_unverified_figures_are_marked_at_the_point_of_speech() -> None:
    render_plan = plan_panorama_render(
        _workspace(ParticipationLevel.STRUCTURED), _fourth_act()
    )
    text = _render(render_plan)
    assert "unverified" in text
    assert "placeholder" in text


def test_the_open_question_is_named_not_just_counted() -> None:
    render_plan = plan_panorama_render(
        _workspace(ParticipationLevel.STRUCTURED), _fourth_act()
    )
    assert render_plan.next_question_ref is not None
    assert render_plan.next_question_ref in _render(render_plan)


def test_no_valuation_means_no_ranking_claim() -> None:
    """Structure without arithmetic may describe, never rank."""
    render_plan = plan_panorama_render(_workspace(ParticipationLevel.STRUCTURED))
    assert render_plan.claim_kind == CLAIM_NONE
    text = _render(render_plan)
    assert "comes out ahead" not in text
    assert "nothing wins outright" not in text


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_safety_hold_suppresses_the_ranking_in_the_text() -> None:
    """Withheld where the user can see it, not only in the data structure."""
    render_plan = plan_panorama_render(
        _workspace(ParticipationLevel.STRUCTURED, safety_hold=True), _fourth_act()
    )
    assert render_plan.safety_hold is True
    assert render_plan.may_rank_at_all is False
    text = _render(render_plan)
    assert "not going to rank" in text
    assert "comes out ahead" not in text
    assert "nothing wins outright" not in text


def test_safety_hold_beats_a_cleanly_separated_ranking() -> None:
    from volvence_zero.decision_workspace.valuation import (
        DimensionEstimate,
        Interval,
        evaluate_options,
    )

    separated = evaluate_options(
        (
            DimensionEstimate(
                option_ref="a", dimension_ref="d", interval=Interval(40, 50, 60)
            ),
            DimensionEstimate(
                option_ref="b", dimension_ref="d", interval=Interval(0, 10, 20)
            ),
        )
    )
    held = plan_panorama_render(
        _workspace(ParticipationLevel.STRUCTURED, safety_hold=True), separated
    )
    assert held.claim_kind == CLAIM_NONE
    assert "comes out ahead" not in _render(held)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def test_the_plan_records_what_the_turn_was_entitled_to_claim() -> None:
    """An auditor must not have to re-derive the licence from the prose.

    The rendered text and the rationale tag are the only two places the
    claim appears; if they can disagree, the audit is worthless.
    """
    planner = PromptPlanner()
    plan = planner.plan(
        context=_context(regime_id="guided_exploration"),
        assembly=None,
        panorama_render_plan=plan_panorama_render(
            _workspace(ParticipationLevel.STRUCTURED), _fourth_act()
        ),
    )
    tags = [tag for tag in plan.rationale_tags if tag.startswith("panorama=")]
    assert tags
    assert f"claim={CLAIM_ROBUSTNESS}" in tags[0]


def test_safety_hold_is_visible_in_the_audit_tags() -> None:
    planner = PromptPlanner()
    plan = planner.plan(
        context=_context(regime_id="guided_exploration"),
        assembly=None,
        panorama_render_plan=plan_panorama_render(
            _workspace(ParticipationLevel.STRUCTURED, safety_hold=True)
        ),
    )
    assert "panorama_safety_hold=1" in plan.rationale_tags


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_counts_read_as_english(count: int) -> None:
    """Small thing, but "1 things are still unresolved" undoes the tone."""
    text = _render(
        plan_panorama_render(
            _workspace(ParticipationLevel.STRUCTURED, options=count, unknowns=count)
        )
    )
    assert "1 things" not in text
    assert "1 options" not in text
