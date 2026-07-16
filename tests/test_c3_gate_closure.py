"""C3: DualTrackGateLearner promotion closure + ModificationGate learned
risk SHADOW bypass.

1. The session-held dual-track gate learner now runs a genuine SHADOW
   dual-run (learned vs fixed-prior heuristic settled against the same
   realized target), exposes code-level promotion exit conditions and a
   checkpoint/rollback contract. The ACTIVE flip stays evidence-gated.
2. The modification gate keeps its rule cascade as the R9/R10 safety
   floor; a bounded logistic risk head observes every realized decision
   and publishes a report-only readout in the credit snapshot.
"""

from __future__ import annotations

import asyncio

from volvence_zero.credit.gate import (
    CreditModule,
    GateDecision,
    GateRiskLearner,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
    gate_risk_features,
)
from volvence_zero.dual_track import (
    DualTrackGateLearner,
    DualTrackGateLearnerState,
    TrackState,
)
from volvence_zero.dual_track.core import DualTrackSnapshot
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.memory import Track


def _track(*, tension: float, goals: tuple[str, ...] = ()) -> TrackState:
    return TrackState(
        track=Track.WORLD,
        active_goals=goals,
        recent_credits=(),
        controller_code=(tension, 0.0, tension),
        tension_level=tension,
    )


def _empty_evaluation() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        description="test evaluation",
    )


def _passing_proposal() -> ModificationProposal:
    return ModificationProposal(
        target="test.target",
        desired_gate=ModificationGate.ONLINE,
        old_value_hash="old",
        new_value_hash="new",
        justification="test",
        is_reversible=True,
        validation_delta=0.05,
        capacity_cost=0.05,
        rollback_evidence="checkpoint-1",
    )


def _blocked_proposal() -> ModificationProposal:
    return ModificationProposal(
        target="test.target",
        desired_gate=ModificationGate.ONLINE,
        old_value_hash="old",
        new_value_hash="new",
        justification="test",
        is_reversible=False,
        validation_delta=-0.5,
        capacity_cost=0.9,
        rollback_evidence="",
    )


# ---------------------------------------------------------------------------
# DualTrackGateLearner promotion closure
# ---------------------------------------------------------------------------


def _run_settled_turns(learner: DualTrackGateLearner, turns: int) -> None:
    for index in range(turns):
        high = 0.9 if index % 2 == 0 else 0.2
        learner.derive_shadow(
            world_track=_track(tension=high),
            self_track=_track(tension=1.0 - high),
            cross_track_tension=0.3,
        )
        learner.observe_realized_outcome(
            task_progress=high,
            relationship_delta=(1.0 - high) * 2.0 - 1.0,
        )


def test_promotion_readout_blocks_on_insufficient_updates() -> None:
    learner = DualTrackGateLearner()
    readout = learner.promotion_readout()
    assert not readout.ready
    assert not readout.kill_recommended
    assert any("updates" in reason for reason in readout.blocking_reasons)


def test_dual_run_settles_heuristic_candidate_in_lockstep() -> None:
    learner = DualTrackGateLearner()
    _run_settled_turns(learner, 10)
    # First derive_shadow has no settleable window; the following 9 do.
    assert learner.update_count == 9
    readout = learner.promotion_readout()
    assert readout.heuristic_mae > 0.0
    assert readout.update_count == 9


def test_promotion_readout_reports_all_exit_conditions() -> None:
    learner = DualTrackGateLearner()
    _run_settled_turns(learner, 60)
    readout = learner.promotion_readout()
    # Enough updates; readiness now hinges purely on the MAE margin.
    assert all("updates" not in reason for reason in readout.blocking_reasons)
    assert readout.update_count >= 50
    if not readout.ready:
        assert any("MAE improvement" in reason for reason in readout.blocking_reasons)


def test_export_restore_roundtrip_is_rollback_path() -> None:
    learner = DualTrackGateLearner()
    _run_settled_turns(learner, 10)
    checkpoint = learner.export_state()
    _run_settled_turns(learner, 10)
    assert learner.update_count > checkpoint.update_count
    learner.restore_state(checkpoint)
    assert learner.export_state() == checkpoint


def test_restore_rejects_wrong_dimension() -> None:
    learner = DualTrackGateLearner()
    bad = DualTrackGateLearnerState(
        weights=(0.0, 0.5),
        update_count=0,
        abs_error_sum=0.0,
        heuristic_abs_error_sum=0.0,
        settled_comparison_count=0,
    )
    try:
        learner.restore_state(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("restore_state must reject wrong-dim weights")


def test_reset_returns_to_neutral_prior() -> None:
    learner = DualTrackGateLearner()
    _run_settled_turns(learner, 10)
    learner.reset()
    assert learner.update_count == 0
    shadow = learner.derive_shadow(
        world_track=_track(tension=0.5),
        self_track=_track(tension=0.5),
        cross_track_tension=0.0,
    )
    assert abs(shadow.world_weight - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# ModificationGate learned risk SHADOW bypass
# ---------------------------------------------------------------------------


def test_gate_risk_learner_learns_block_vs_allow() -> None:
    learner = GateRiskLearner()
    evaluation = _empty_evaluation()
    blocked_features = gate_risk_features(
        proposal=_blocked_proposal(), evaluation_snapshot=evaluation
    )
    allowed_features = gate_risk_features(
        proposal=_passing_proposal(), evaluation_snapshot=evaluation
    )
    for _ in range(200):
        learner.observe_decision(features=blocked_features, decision=GateDecision.BLOCK)
        learner.observe_decision(features=allowed_features, decision=GateDecision.ALLOW)
    assert learner.predict_risk(blocked_features) > 0.7
    assert learner.predict_risk(allowed_features) < 0.3


def test_gate_risk_readout_never_changes_rule_decision() -> None:
    """The rule cascade is the safety floor: the learner has no input path."""

    evaluation = _empty_evaluation()
    proposal = _blocked_proposal()
    reasons_before = evaluate_gate_reasons(
        proposal=proposal, evaluation_snapshot=evaluation
    )
    learner = GateRiskLearner()
    features = gate_risk_features(proposal=proposal, evaluation_snapshot=evaluation)
    for _ in range(50):
        # Feed the learner the OPPOSITE label; the rules must not budge.
        learner.observe_decision(features=features, decision=GateDecision.ALLOW)
    reasons_after = evaluate_gate_reasons(
        proposal=proposal, evaluation_snapshot=evaluation
    )
    assert reasons_before == reasons_after


def test_credit_module_publishes_gate_risk_readout() -> None:
    module = CreditModule(pending_proposals=(_blocked_proposal(),))
    world = _track(tension=0.6)
    self_track = _track(tension=0.4)
    dual_track = DualTrackSnapshot(
        world_track=world,
        self_track=self_track,
        cross_track_tension=0.2,
        description="test dual track",
    )
    snapshot = asyncio.run(
        module.process_standalone(
            dual_track_snapshot=dual_track,
            evaluation_snapshot=_empty_evaluation(),
            timestamp_ms=1,
        )
    )
    readout = snapshot.value.gate_risk_readout
    assert readout is not None
    assert readout.realized_block is True
    assert readout.update_count == 1
    assert 0.0 <= readout.predicted_risk <= 1.0
