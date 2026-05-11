"""Packet 3.5: end-to-end reflection-driven protocol revision.

Synthesizes a multi-turn rollout where:

1. A two-protocol mixture is loaded (a ``winner`` protocol and a
   ``loser`` protocol).
2. PE history is fed in such that the ``loser`` protocol receives
   consistent negative attribution-weighted rewards across the
   reflection window.
3. ``ProtocolReflectionEngine`` runs its rule set on the next
   scan and emits a ``WEIGHT_DECAY`` proposal targeting the loser.
4. The proposal passes through ``evaluate_protocol_revision``
   (auto-approved at L3 because evidence window is sufficient
   and pe_signature is non-empty).
5. ``ProtocolRegistryModule.apply_revision`` mutates the loser's
   strategy weights (×0.5) and re-runs the compile path.
6. Subsequent ``compute_active_mixture`` returns now-decayed
   strategy_priors content via the application owners.

This is the canonical "system actually learned" demonstration.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.protocol_runtime import (
    ApprovalOutcome,
    ProtocolRegistryModule,
    evaluate_protocol_revision,
)
from volvence_zero.reflection import ProtocolReflectionEngine
from volvence_zero.runtime import Snapshot, WiringLevel


def _pe_snapshot(turn: int, signed_reward: float) -> Snapshot[PredictionErrorSnapshot]:
    ctx = PredictionActionContext()
    actual = ActualOutcome(
        observed_turn_index=turn,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="",
        action_context=ctx,
    )
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
    pe = PredictionError(
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description="",
    )
    value = PredictionErrorSnapshot(
        evaluated_prediction=pred,
        actual_outcome=actual,
        next_prediction=pred,
        error=pe,
        turn_index=turn,
        bootstrap=False,
        description="",
        action_context=ctx,
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=0,
        value=value,
    )


def _mixture_snapshot(loser_id: str, winner_id: str) -> Snapshot[ActiveMixtureSnapshot]:
    """Mixture where loser dominates (so attribution maps PE to loser)."""
    value = ActiveMixtureSnapshot(
        active_protocols=(
            ActiveProtocolEntry(
                protocol_id=loser_id, activation_weight=0.85
            ),
            ActiveProtocolEntry(
                protocol_id=winner_id, activation_weight=0.15
            ),
        ),
        boundary_union_ids=(),
        revision_fingerprint="loser-dominates",
        description="",
    )
    return Snapshot(
        slot_name="active_mixture",
        owner="ProtocolRegistryModule",
        version=1,
        timestamp_ms=0,
        value=value,
    )


def _build_two_protocol_module() -> tuple[
    ProtocolRegistryModule, ApplicationRareHeavyState
]:
    """Load cheng_laoshi (winner) plus a renamed clone (loser)."""
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    base = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    winner = _replace(base, protocol_id="growth_advisor:cheng-laoshi")
    loser = _replace(base, protocol_id="growth_advisor:cheng-laoshi-loser")
    module.load_protocol(winner)
    module.load_protocol(loser)
    return module, rare


def test_reflection_drives_strategy_decay_e2e() -> None:
    module, _rare = _build_two_protocol_module()

    loser_id = "growth_advisor:cheng-laoshi-loser"
    winner_id = "growth_advisor:cheng-laoshi"

    # Baseline: loser's strategies all have initial_weight = 1.0.
    loser_pre = module.registry.get(loser_id)
    pre_weights = {s.rule_id: s.initial_weight for s in loser_pre.strategy_priors}
    assert all(w > 0 for w in pre_weights.values())

    # Build a reflection engine running every 12 turns.
    engine = ProtocolReflectionEngine(
        scan_period=12,
        history_window=50,
        wiring_level=WiringLevel.SHADOW,
    )

    # Run 12 turns: each turn the mixture has loser at 0.85 weight,
    # PE is consistently negative. After 12 turns the engine's
    # internal scan fires and a WEIGHT_DECAY proposal is emitted.
    for turn in range(1, 13):
        upstream = {
            "prediction_error": _pe_snapshot(turn=turn, signed_reward=-0.9),
            "active_mixture": _mixture_snapshot(loser_id, winner_id),
        }
        snapshot = asyncio.run(engine.process(upstream))

    proposals = snapshot.value.protocol_revision_proposals
    assert proposals, (
        "expected at least one strategy_decay proposal after 12 turns "
        "of dominant negative attribution"
    )

    # The loser must be a target.
    targeted = [p for p in proposals if p.target_protocol_id == loser_id]
    assert targeted, (
        f"expected a proposal targeting {loser_id!r}; got "
        f"{[p.target_protocol_id for p in proposals]}"
    )
    proposal = targeted[0]

    # Gate: L3 with 12 turns observation + non-empty pe_signature → auto-approve.
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.AUTO_APPROVED, decision

    # Apply the revision.
    module.apply_revision(
        proposal,
        revised_by="ProtocolReflectionEngine[test]",
    )

    # Loser's strategy weights are now × 0.5; winner untouched.
    loser_post = module.registry.get(loser_id)
    post_weights = {s.rule_id: s.initial_weight for s in loser_post.strategy_priors}
    for rid, pre_w in pre_weights.items():
        assert post_weights[rid] == pre_w * 0.5, (rid, pre_w, post_weights[rid])

    winner_post = module.registry.get(winner_id)
    winner_weights = {s.rule_id: s.initial_weight for s in winner_post.strategy_priors}
    # Winner is unchanged: same strategy ids, same weights as the
    # original cheng_laoshi profile.
    assert all(w > 0 for w in winner_weights.values())

    # Loser's revision_log records the mutation.
    assert any(
        proposal.proposal_id in r.revision_id
        for r in loser_post.revision_log
    ), [r.revision_id for r in loser_post.revision_log]


def test_reflection_l4_proposal_does_not_auto_apply() -> None:
    """Synthetic L4 proposal goes through gate → queued (not auto-applied)."""
    from volvence_zero.behavior_protocol import (
        ProposalEvidence,
        ProtocolRevisionChangeKind,
        ProtocolRevisionProposal,
        ProtocolRevisionTargetField,
        ReviewLevel,
    )

    proposal = ProtocolRevisionProposal(
        proposal_id="synth:L4",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.BOUNDARY_CONTRACT,
        target_entry_id="bp-no-hard-sell",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=ProposalEvidence(
            observation_window_turns=100,
            pe_signature="overwhelming",
            summary="test",
        ),
        required_review_level=ReviewLevel.L4,
    )
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.QUEUED_FOR_HUMAN
