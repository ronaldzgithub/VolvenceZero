"""Packet 6.1: WEIGHT_REINFORCE / BOUNDARY_REFINEMENT /
IDENTITY_CLARIFICATION / PROTOCOL_RETIREMENT apply tests + new rules."""

from __future__ import annotations

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
    ReviewStatus,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.reflection import (
    PROTOCOL_RETIREMENT_MIN_TURNS,
    STRATEGY_REINFORCE_MIN_TURNS,
    propose_protocol_retirement,
    propose_strategy_reinforce,
)


def _evidence() -> ProposalEvidence:
    return ProposalEvidence(
        observation_window_turns=10,
        pe_signature="test",
        summary="test",
    )


def _build_module():
    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    return module


def _pe(turn: int, sr: float) -> PredictionErrorSnapshot:
    ctx = PredictionActionContext()
    pred = PredictedOutcome(
        source_turn_index=turn,
        target_turn_index=turn + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="",
        action_context=ctx,
    )
    actual = ActualOutcome(
        observed_turn_index=turn,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="",
        action_context=ctx,
    )
    pe = PredictionError(
        task_error=sr,
        relationship_error=sr,
        regime_error=sr,
        action_error=sr,
        magnitude=abs(sr),
        signed_reward=sr,
        description="",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=pred,
        actual_outcome=actual,
        next_prediction=pred,
        error=pe,
        turn_index=turn,
        bootstrap=False,
        description="",
        action_context=ctx,
    )


def _mixture(*entries: tuple[str, float]) -> ActiveMixtureSnapshot:
    return ActiveMixtureSnapshot(
        active_protocols=tuple(
            ActiveProtocolEntry(protocol_id=pid, activation_weight=w)
            for pid, w in entries
        ),
        boundary_union_ids=(),
        revision_fingerprint="",
        description="",
    )


# ---------------------------------------------------------------------------
# WEIGHT_REINFORCE
# ---------------------------------------------------------------------------


def test_weight_reinforce_multiplies_strategy_weights() -> None:
    """WEIGHT_REINFORCE multiplies; result clamped to [0, 1]
    schema-valid range."""
    module = _build_module()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    # Pick a strategy with low initial_weight so 1.5x doesn't saturate.
    candidates = [s for s in bp.strategy_priors if s.initial_weight < 0.5]
    if not candidates:
        # Fallback: fixture's weights might all be ≥ 0.5; use index 0
        # and observe the clamp.
        target = bp.strategy_priors[0].rule_id
        pre = bp.strategy_priors[0].initial_weight
    else:
        target = candidates[0].rule_id
        pre = candidates[0].initial_weight

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:reinforce",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_REINFORCE,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 1.5},
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    target_strat = next(
        s for s in bp_after.strategy_priors if s.rule_id == target
    )
    expected = min(pre * 1.5, 1.0)
    assert abs(target_strat.initial_weight - expected) < 1e-9


def test_weight_reinforce_clamps_at_one() -> None:
    """If pre * multiplier > 1, the result is clamped to 1.0."""
    module = _build_module()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    # Pick a strategy with initial_weight close to 1.
    target = max(bp.strategy_priors, key=lambda s: s.initial_weight).rule_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:reinforce-clamp",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_REINFORCE,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 5.0},
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    target_strat = next(
        s for s in bp_after.strategy_priors if s.rule_id == target
    )
    assert target_strat.initial_weight <= 1.0
    assert target_strat.initial_weight > 0.0


# ---------------------------------------------------------------------------
# BOUNDARY_REFINEMENT
# ---------------------------------------------------------------------------


def test_boundary_refinement_updates_boundary_fields() -> None:
    module = _build_module()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    target = bp.boundary_contracts[0].boundary_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:bd-refine",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.BOUNDARY_CONTRACT,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.BOUNDARY_REFINEMENT,
        evidence=_evidence(),
        proposed_payload={
            "blocked_topics": ["promo", "discount", "limited time"],
            "description": "tightened boundary description",
        },
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    refined = next(
        b for b in bp_after.boundary_contracts if b.boundary_id == target
    )
    assert "limited time" in refined.blocked_topics
    assert refined.description == "tightened boundary description"


def test_boundary_refinement_unknown_id_no_op() -> None:
    """Refinement targeting a non-existent boundary leaves protocol untouched."""
    module = _build_module()
    bp_before = module.registry.get("growth_advisor:cheng-laoshi")
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:bd-noop",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.BOUNDARY_CONTRACT,
        target_entry_id="bp-does-not-exist",
        change_kind=ProtocolRevisionChangeKind.BOUNDARY_REFINEMENT,
        evidence=_evidence(),
        proposed_payload={"blocked_topics": ["x"]},
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    assert bp_after.boundary_contracts == bp_before.boundary_contracts


# ---------------------------------------------------------------------------
# IDENTITY_CLARIFICATION
# ---------------------------------------------------------------------------


def test_identity_clarification_updates_identity_assertion() -> None:
    module = _build_module()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:identity",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.IDENTITY_ASSERTION,
        target_entry_id="growth_advisor:cheng-laoshi",
        change_kind=ProtocolRevisionChangeKind.IDENTITY_CLARIFICATION,
        evidence=_evidence(),
        proposed_payload={
            "requires_self_traits": [
                "warm_peer_register",
                "long_horizon",
                "patient_listener",
            ],
            "forbidden_self_traits": ["aggressive_sales"],
        },
        required_review_level=ReviewLevel.L4,
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    assert "patient_listener" in bp_after.identity_assertion.requires_self_traits
    assert "aggressive_sales" in bp_after.identity_assertion.forbidden_self_traits


# ---------------------------------------------------------------------------
# PROTOCOL_RETIREMENT
# ---------------------------------------------------------------------------


def test_protocol_retirement_marks_review_status() -> None:
    module = _build_module()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:retire",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,  # field-agnostic
        target_entry_id="growth_advisor:cheng-laoshi",
        change_kind=ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT,
        evidence=_evidence(),
        required_review_level=ReviewLevel.L4,
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    assert bp_after.review_status is ReviewStatus.RETIRED


# ---------------------------------------------------------------------------
# Reflection rules
# ---------------------------------------------------------------------------


def test_propose_strategy_reinforce_emits_for_consistent_positive_pe() -> None:
    n = STRATEGY_REINFORCE_MIN_TURNS + 2
    pe_history = tuple(_pe(t, 0.7) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_winner", 1.0)) for _ in range(n))
    proposals = propose_strategy_reinforce(
        pe_history=pe_history, active_mixture_history=am_history
    )
    assert proposals
    p = proposals[0]
    assert p.target_protocol_id == "p_winner"
    assert p.change_kind is ProtocolRevisionChangeKind.WEIGHT_REINFORCE
    assert p.required_review_level is ReviewLevel.L1


def test_propose_strategy_reinforce_no_emit_when_negative() -> None:
    n = STRATEGY_REINFORCE_MIN_TURNS + 2
    pe_history = tuple(_pe(t, -0.5) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_loser", 1.0)) for _ in range(n))
    assert (
        propose_strategy_reinforce(
            pe_history=pe_history, active_mixture_history=am_history
        )
        == ()
    )


def test_propose_protocol_retirement_emits_for_catastrophic_failure() -> None:
    n = PROTOCOL_RETIREMENT_MIN_TURNS + 2
    pe_history = tuple(_pe(t, -0.9) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_dead", 1.0)) for _ in range(n))
    proposals = propose_protocol_retirement(
        pe_history=pe_history, active_mixture_history=am_history
    )
    assert any(
        p.change_kind is ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT
        for p in proposals
    )
    p = next(
        p for p in proposals
        if p.change_kind is ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT
    )
    assert p.required_review_level is ReviewLevel.L4


def test_propose_protocol_retirement_no_emit_under_threshold_window() -> None:
    n = PROTOCOL_RETIREMENT_MIN_TURNS - 2
    pe_history = tuple(_pe(t, -0.9) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_dead", 1.0)) for _ in range(n))
    assert (
        propose_protocol_retirement(
            pe_history=pe_history, active_mixture_history=am_history
        )
        == ()
    )
