"""Contract tests for the decision_workspace owner.

The two invariants worth defending, in order of how quietly they would
rot:

1. **The owner does not decide when it exists.** Activation comes from
   ``regime``'s panorama gate. A second activation judgement here would
   reintroduce exactly the per-scenario rules the gate replaced, and it
   would do so invisibly — the tests below vary the gate against a fixed,
   maximally decision-shaped set of owner snapshots, so any local
   "well, this looks important" logic shows up immediately.

2. **It holds references, not copies.** The semantic content belongs to
   ``plan_intent`` / ``goal_value`` / ``open_loop`` /
   ``belief_assumption``. This is enforced structurally — the records
   have nowhere to put prose — and the field-shape test below pins that,
   because "we'll just add a summary field for convenience" is how the
   second writer gets born.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from companion_standard.semantic_state import (
    BeliefAssumptionSnapshot,
    GoalValueSnapshot,
    OpenLoopSnapshot,
    PlanIntentSnapshot,
    SemanticRecord,
)

from volvence_zero.decision_workspace import (
    CONCLUSION_NONE,
    CONCLUSION_PROVISIONAL,
    CONCLUSION_SETTLED,
    DecisionOption,
    DecisionUnknown,
    DecisionWorkspaceModule,
    DecisionWorkspaceSnapshot,
)
from volvence_zero.regime import (
    ParticipationHint,
    ParticipationLevel,
    RegimeSnapshot,
    build_regime_identity,
)
from volvence_zero.runtime import Snapshot


def _record(record_id: str, *, summary: str = "sensitive prose") -> SemanticRecord:
    return SemanticRecord(
        record_id=record_id,
        summary=summary,
        detail="more sensitive prose",
        confidence=0.7,
        status="open",
        source_turn=1,
        evidence="test",
    )


def _snapshot(slot: str, value: object) -> Snapshot[object]:
    return Snapshot(
        slot_name=slot, owner="test", version=1, timestamp_ms=0, value=value
    )


def _upstream(
    level: ParticipationLevel,
    *,
    unknowns: int = 2,
    options: int = 3,
) -> dict[str, Snapshot[object]]:
    regime = RegimeSnapshot(
        active_regime=build_regime_identity(
            regime_id="guided_exploration", historical_effectiveness={}
        ),
        previous_regime=None,
        switch_reason="",
        candidate_regimes=(("guided_exploration", 0.7),),
        turns_in_current_regime=3,
        delayed_outcomes=(),
        delayed_attributions=(),
        identity_hints=(),
        effectiveness_trend=(),
        regime_changed=False,
        description="",
        participation_hint=ParticipationHint(
            panorama_level=level, rationale="test-gate"
        ),
    )
    plan_intent = PlanIntentSnapshot(
        active_plan_id="cand-0",
        active_goal="",
        active_step="",
        active_constraints=(),
        deferred_intents=(_record("deferred-0"),),
        standing_plans=(),
        candidate_plans=tuple(_record(f"cand-{i}") for i in range(options)),
        completed_plan_refs=(),
        plan_revision_count=2,
        continuity_score=0.4,
        control_signal=0.0,
        description="",
    )
    goal_value = GoalValueSnapshot(
        explicit_goals=(),
        value_priorities=(_record("dim-0"), _record("dim-1")),
        tradeoff_notes=(_record("trade-0"),),
        active_goal_id=None,
        alignment_score=0.3,
        control_signal=0.0,
        description="",
        value_conflict=0.7,
    )
    open_loop = OpenLoopSnapshot(
        unresolved_loops=tuple(_record(f"loop-{i}") for i in range(unknowns)),
        pending_confirmations=(),
        closure_refs=(),
        highest_priority_loop_id=None,
        closure_pressure=0.7,
        control_signal=0.0,
        description="",
    )
    belief = BeliefAssumptionSnapshot(
        beliefs=(_record("belief-0"),),
        assumptions=(),
        verification_needs=tuple(_record(f"verify-{i}") for i in range(unknowns)),
        contradiction_refs=(),
        mean_confidence=0.3,
        control_signal=0.0,
        description="",
    )
    return {
        "regime": _snapshot("regime", regime),
        "plan_intent": _snapshot("plan_intent", plan_intent),
        "goal_value": _snapshot("goal_value", goal_value),
        "open_loop": _snapshot("open_loop", open_loop),
        "belief_assumption": _snapshot("belief_assumption", belief),
    }


def _run(level: ParticipationLevel, **kwargs: int) -> DecisionWorkspaceSnapshot:
    module = DecisionWorkspaceModule()
    snapshot = asyncio.run(module.process(_upstream(level, **kwargs)))
    assert isinstance(snapshot.value, DecisionWorkspaceSnapshot)
    return snapshot.value


# ---------------------------------------------------------------------------
# The owner subscribes; it does not decide
# ---------------------------------------------------------------------------


def test_silent_gate_leaves_no_decision_structure() -> None:
    """Richest possible owner state, gate closed: nothing is built.

    If this ever starts returning options, some local heuristic has
    decided the situation was important enough to override the gate.
    """
    workspace = _run(ParticipationLevel.SILENT)
    assert workspace.instantiated is False
    assert workspace.options == ()
    assert workspace.unknowns == ()
    assert workspace.dimension_refs == ()
    assert workspace.conclusion_state == CONCLUSION_NONE


def test_brief_gate_tracks_structure_but_withholds_it() -> None:
    workspace = _run(ParticipationLevel.BRIEF)
    assert workspace.instantiated is True
    assert workspace.options != ()
    assert workspace.unknowns != ()
    # Withheld at this tier: no dimensions, no conclusion, no evidence.
    assert workspace.dimension_refs == ()
    assert workspace.evidence_refs == ()
    assert workspace.conclusion_state == CONCLUSION_NONE


def test_structured_gate_publishes_the_full_structure() -> None:
    workspace = _run(ParticipationLevel.STRUCTURED)
    assert workspace.dimension_refs == ("dim-0", "dim-1")
    assert workspace.evidence_refs == ("belief-0",)
    assert workspace.conclusion_state == CONCLUSION_PROVISIONAL


def test_only_the_gate_varies_across_the_three_tiers() -> None:
    """Same owner snapshots, three gate values, three different outputs.

    This is the subscription contract stated as an experiment: the sole
    input that changed is the enum.
    """
    outputs = {
        level: _run(level)
        for level in (
            ParticipationLevel.SILENT,
            ParticipationLevel.BRIEF,
            ParticipationLevel.STRUCTURED,
        )
    }
    assert len({len(w.options) for w in outputs.values()}) == 2
    assert len({w.conclusion_state for w in outputs.values()}) == 2


def test_missing_regime_snapshot_is_treated_as_silent() -> None:
    module = DecisionWorkspaceModule()
    upstream = _upstream(ParticipationLevel.STRUCTURED)
    del upstream["regime"]
    workspace = asyncio.run(module.process(upstream)).value
    assert isinstance(workspace, DecisionWorkspaceSnapshot)
    assert workspace.instantiated is False


def test_inactive_owner_placeholder_is_not_read_as_content() -> None:
    """A SHADOW/DISABLED owner publishes a placeholder, not a snapshot."""

    class _Placeholder:
        pass

    module = DecisionWorkspaceModule()
    upstream = _upstream(ParticipationLevel.STRUCTURED)
    upstream["plan_intent"] = _snapshot("plan_intent", _Placeholder())
    workspace = asyncio.run(module.process(upstream)).value
    assert isinstance(workspace, DecisionWorkspaceSnapshot)
    assert workspace.options == ()


# ---------------------------------------------------------------------------
# References, not copies
# ---------------------------------------------------------------------------


def _field_names(cls: type) -> set[str]:
    return {field.name for field in dataclasses.fields(cls)}


@pytest.mark.parametrize(
    "record_type", [DecisionOption, DecisionUnknown]
)
def test_records_have_nowhere_to_put_prose(record_type: type) -> None:
    """The ownership contract, enforced by shape rather than by review.

    A ``summary`` / ``detail`` / ``label`` field here would let this
    owner become a second writer of facts that already have one.
    """
    forbidden = {"summary", "detail", "label", "text", "content", "note"}
    assert not (_field_names(record_type) & forbidden)


def test_no_source_prose_appears_anywhere_in_the_snapshot() -> None:
    """Behavioural counterpart to the shape test.

    Every source record carries a distinctive summary; none of it may
    surface in the published workspace.
    """
    workspace = _run(ParticipationLevel.STRUCTURED)
    rendered = repr(workspace)
    assert "sensitive prose" not in rendered


def test_unknown_without_a_source_ref_is_rejected() -> None:
    with pytest.raises(ValueError, match="no source ref"):
        DecisionUnknown(unknown_id="unknown:invented")


def test_option_refs_point_at_plan_intent_records() -> None:
    workspace = _run(ParticipationLevel.STRUCTURED, options=2)
    assert [option.plan_ref for option in workspace.options] == [
        "cand-0",
        "cand-1",
        "deferred-0",
    ]
    assert workspace.options[0].status == "chosen"
    assert workspace.options[2].status == "deferred"


# ---------------------------------------------------------------------------
# Conclusion state
# ---------------------------------------------------------------------------


def test_conclusion_stays_provisional_while_unknowns_are_open() -> None:
    assert _run(ParticipationLevel.STRUCTURED, unknowns=3).conclusion_state == (
        CONCLUSION_PROVISIONAL
    )


def test_conclusion_settles_only_when_no_unknown_remains() -> None:
    assert _run(ParticipationLevel.STRUCTURED, unknowns=0).conclusion_state == (
        CONCLUSION_SETTLED
    )


# ---------------------------------------------------------------------------
# Spine boundary
# ---------------------------------------------------------------------------


def test_decision_workspace_is_not_a_semantic_spine_owner() -> None:
    """It must not enter ``semantic_spine_coverage``'s denominator.

    Adding it there would shift every historical paper-suite and
    companion readout for reasons unrelated to relationship state.
    """
    from companion_standard.semantic_state import SEMANTIC_OWNER_SLOTS

    assert DecisionWorkspaceModule.slot_name not in SEMANTIC_OWNER_SLOTS


def test_default_wiring_is_shadow() -> None:
    from volvence_zero.runtime import WiringLevel

    assert DecisionWorkspaceModule.default_wiring_level is WiringLevel.SHADOW


# ---------------------------------------------------------------------------
# Safety sits above the ranking
# ---------------------------------------------------------------------------


def _boundary_policy(*, risk_band: str, refer_out: bool):
    from volvence_zero.application.types import (
        BoundaryDecision,
        BoundaryPolicySnapshot,
        ProfessionalScope,
        RiskBand,
    )

    return BoundaryPolicySnapshot(
        active_decision=BoundaryDecision(
            decision_id="test",
            risk_band=RiskBand(risk_band),
            professional_scope=ProfessionalScope.GENERAL_SUPPORT,
            answer_depth_limit="standard",
            citation_required=False,
            clarification_required=False,
            refer_out_required=refer_out,
            blocked_topics=(),
            required_disclaimers=(),
            description="",
        ),
        trigger_reasons=(),
        description="",
    )


def _run_with_boundary(**boundary: object) -> DecisionWorkspaceSnapshot:
    module = DecisionWorkspaceModule()
    upstream = _upstream(ParticipationLevel.STRUCTURED)
    upstream["boundary_policy"] = _snapshot(
        "boundary_policy", _boundary_policy(**boundary)  # type: ignore[arg-type]
    )
    workspace = asyncio.run(module.process(upstream)).value
    assert isinstance(workspace, DecisionWorkspaceSnapshot)
    return workspace


def test_critical_band_literal_matches_the_owner_enum() -> None:
    """The safety literal must not drift from the owner's vocabulary.

    ``decision_workspace`` sits below vz-application and names the band
    as a string rather than importing the enum. If ``RiskBand.CRITICAL``
    were ever renamed, the hold would stop firing silently — a safety
    check that fails open and says nothing. This test is the seam.
    """
    from volvence_zero.application.types import RiskBand

    from volvence_zero.decision_workspace import _CRITICAL_RISK_BAND

    assert _CRITICAL_RISK_BAND == RiskBand.CRITICAL.value


def test_critical_risk_band_withholds_the_ranking() -> None:
    workspace = _run_with_boundary(risk_band="critical", refer_out=False)
    assert workspace.safety_hold is True
    assert workspace.conclusion_state == "withheld-safety"
    assert "risk-band-critical" in workspace.safety_reasons


def test_refer_out_withholds_the_ranking() -> None:
    workspace = _run_with_boundary(risk_band="low", refer_out=True)
    assert workspace.safety_hold is True
    assert "refer-out-required" in workspace.safety_reasons


def test_ordinary_risk_does_not_withhold() -> None:
    workspace = _run_with_boundary(risk_band="high", refer_out=False)
    assert workspace.safety_hold is False
    assert workspace.conclusion_state == CONCLUSION_PROVISIONAL


def test_withholding_still_publishes_what_was_held() -> None:
    """Suppressing the ranking is not the same as erasing the decision.

    An audit needs to see which options and unknowns existed at the
    moment the hold fired.
    """
    workspace = _run_with_boundary(risk_band="critical", refer_out=False)
    assert workspace.options != ()
    assert workspace.unknowns != ()


def test_missing_boundary_policy_does_not_invent_a_hold() -> None:
    workspace = _run(ParticipationLevel.STRUCTURED)
    assert workspace.safety_hold is False


def test_safety_hold_is_not_reachable_through_the_panorama_gate() -> None:
    """The two gates are independent, and must stay that way.

    A closed panorama means "do not lay out a decision here". A safety
    hold means "do not state a ranking at all". Collapsing one into the
    other would let a wide-open panorama imply the safety question had
    been answered.
    """
    for level in (ParticipationLevel.BRIEF, ParticipationLevel.STRUCTURED):
        module = DecisionWorkspaceModule()
        upstream = _upstream(level)
        upstream["boundary_policy"] = _snapshot(
            "boundary_policy", _boundary_policy(risk_band="critical", refer_out=False)
        )
        workspace = asyncio.run(module.process(upstream)).value
        assert isinstance(workspace, DecisionWorkspaceSnapshot)
        if level is ParticipationLevel.STRUCTURED:
            assert workspace.safety_hold is True
        else:
            # BRIEF never publishes a conclusion in the first place, so
            # there is nothing for the hold to withhold.
            assert workspace.conclusion_state == CONCLUSION_NONE
