"""Packet 5.2: NewStrategyPrior path tests.

Asserts the contract for adding a new strategy_prior via
reflection-driven proposal:

* ``ProtocolRevisionChangeKind.ADD_STRATEGY`` accepted by
  ``apply_revision``.
* Payload validation: missing ``problem_pattern`` /
  ``recommended_ordering`` / empty rule_id → ValueError.
* New strategy appended to ``protocol.strategy_priors``;
  recompile path picks it up into application owners.
* Idempotent: same rule_id applied twice → no duplicate.
* ``propose_strategy_addition`` rule emits proposal under the
  documented heuristic (4 consecutive successful low-weight
  turns).
"""

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
    ADD_STRATEGY_MIN_TURNS,
    propose_strategy_addition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence() -> ProposalEvidence:
    return ProposalEvidence(
        observation_window_turns=10,
        pe_signature="test",
        summary="test summary",
    )


def _build_module():
    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    return module, rare


def _pe(turn: int, signed_reward: float) -> PredictionErrorSnapshot:
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
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
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
# apply_revision: ADD_STRATEGY
# ---------------------------------------------------------------------------


def test_add_strategy_appends_to_protocol() -> None:
    module, _ = _build_module()
    bp_before = module.registry.get("growth_advisor:cheng-laoshi")
    pre_count = len(bp_before.strategy_priors)
    pre_ids = {s.rule_id for s in bp_before.strategy_priors}

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:add",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="autosynth-rule-1",
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=_evidence(),
        proposed_payload={
            "rule_id": "autosynth-rule-1",
            "problem_pattern": "user expresses confusion",
            "recommended_ordering": ("acknowledge", "clarify"),
            "recommended_pacing": "slow",
            "applicability_phase": ("day1",),
            "initial_weight": 0.5,
            "description": "test new strategy",
        },
    )
    module.apply_revision(proposal)

    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    assert len(bp_after.strategy_priors) == pre_count + 1
    new_ids = {s.rule_id for s in bp_after.strategy_priors}
    assert new_ids - pre_ids == {"autosynth-rule-1"}


def test_add_strategy_idempotent_on_duplicate_rule_id() -> None:
    module, _ = _build_module()
    bp_before = module.registry.get("growth_advisor:cheng-laoshi")
    target_existing_id = bp_before.strategy_priors[0].rule_id
    pre_count = len(bp_before.strategy_priors)

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:dup",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target_existing_id,
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=_evidence(),
        proposed_payload={
            "rule_id": target_existing_id,
            "problem_pattern": "duplicate",
            "recommended_ordering": ("noop",),
            "recommended_pacing": "moderate",
        },
    )
    module.apply_revision(proposal)
    bp_after = module.registry.get("growth_advisor:cheng-laoshi")
    assert len(bp_after.strategy_priors) == pre_count


def test_add_strategy_missing_problem_pattern_raises() -> None:
    module, _ = _build_module()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:bad",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="rule-no-pattern",
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=_evidence(),
        proposed_payload={
            "rule_id": "rule-no-pattern",
            "recommended_ordering": ("step1",),
        },
    )
    with pytest.raises(ValueError, match="problem_pattern"):
        module.apply_revision(proposal)


def test_add_strategy_missing_ordering_raises() -> None:
    module, _ = _build_module()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:bad-ordering",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="rule-no-order",
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=_evidence(),
        proposed_payload={
            "rule_id": "rule-no-order",
            "problem_pattern": "test",
            "recommended_ordering": (),
        },
    )
    with pytest.raises(ValueError, match="recommended_ordering"):
        module.apply_revision(proposal)


def test_add_strategy_uses_target_entry_id_when_payload_lacks_rule_id() -> None:
    module, _ = _build_module()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:fallback-id",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="fallback-rule-id",
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=_evidence(),
        proposed_payload={
            "problem_pattern": "fallback test",
            "recommended_ordering": ("step1",),
        },
    )
    module.apply_revision(proposal)
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    assert any(s.rule_id == "fallback-rule-id" for s in bp.strategy_priors)


def test_add_strategy_recompile_pushes_to_application_owner() -> None:
    """After ADD_STRATEGY, the strategy_playbook compile output gains
    a protocol-prefixed entry for the new rule."""
    module, rare = _build_module()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:recompile",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="recompile-rule",
        change_kind=ProtocolRevisionChangeKind.ADD_STRATEGY,
        evidence=_evidence(),
        proposed_payload={
            "rule_id": "recompile-rule",
            "problem_pattern": "test recompile",
            "recommended_ordering": ("step1",),
            "recommended_pacing": "moderate",
        },
    )
    module.apply_revision(proposal)
    rule_ids = [r.rule_id for r in rare.distilled_playbook_rules]
    assert any(
        rid.endswith(":playbook:recompile-rule") for rid in rule_ids
    ), rule_ids


# ---------------------------------------------------------------------------
# propose_strategy_addition rule
# ---------------------------------------------------------------------------


def test_propose_strategy_addition_no_history_no_proposal() -> None:
    proposals = propose_strategy_addition(
        pe_history=(),
        active_mixture_history=(),
    )
    assert proposals == ()


def test_propose_strategy_addition_emits_for_low_weight_success_pattern() -> None:
    """ADD_STRATEGY_MIN_TURNS consecutive successful turns where
    protocol weight stays low → proposal emitted."""
    n = ADD_STRATEGY_MIN_TURNS + 2
    pe_history = tuple(_pe(t, 0.5) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_low_weight", 0.2)) for _ in range(n))

    proposals = propose_strategy_addition(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    assert any(
        p.target_protocol_id == "p_low_weight"
        and p.change_kind is ProtocolRevisionChangeKind.ADD_STRATEGY
        for p in proposals
    ), [p.target_protocol_id for p in proposals]
    p = next(
        p for p in proposals if p.target_protocol_id == "p_low_weight"
    )
    assert p.required_review_level is ReviewLevel.L3
    payload = p.proposed_payload or {}
    assert "problem_pattern" in payload
    assert payload["recommended_ordering"]


def test_propose_strategy_addition_no_proposal_when_protocol_dominates() -> None:
    """High weight (≥ 0.5) → not "low weight" → no proposal."""
    n = ADD_STRATEGY_MIN_TURNS + 2
    pe_history = tuple(_pe(t, 0.9) for t in range(1, n + 1))
    am_history = tuple(_mixture(("p_dominant", 0.95)) for _ in range(n))

    proposals = propose_strategy_addition(
        pe_history=pe_history,
        active_mixture_history=am_history,
    )
    assert not proposals
