from __future__ import annotations

import pytest

from lifeform_core import (
    BoundedContentCandidate,
    BoundedContentPolicy,
    BoundedContentPolicyCredit,
    BoundedPolicyCandidate,
    apply_bounded_policy_credit,
    default_bounded_content_policy_checkpoint,
    rank_and_gate_bounded_policy,
)


def _candidates() -> tuple[BoundedPolicyCandidate, ...]:
    return (
        BoundedPolicyCandidate(
            candidate_id="candidate-a",
            action_key="act-a",
            feature_values=(1.0, 0.0),
        ),
        BoundedPolicyCandidate(
            candidate_id="candidate-b",
            action_key="act-b",
            feature_values=(0.0, 1.0),
        ),
    )


def test_shared_policy_ranks_gates_and_updates_from_explicit_credit() -> None:
    weights = (("act-a", (1.0, 0.0)), ("act-b", (0.0, 1.0)))
    decision = rank_and_gate_bounded_policy(
        candidates=_candidates(),
        action_weights=weights,
        intervention_weights=(1.0, 0.0),
        intervention_bias=0.0,
        maximum_candidates=2,
    )

    assert tuple(item.candidate_id for item in decision.ranked_candidates) == (
        "candidate-a",
        "candidate-b",
    )
    assert decision.selected_candidate_id == "candidate-a"
    assert decision.intervenes

    update = apply_bounded_policy_credit(
        action_weights=weights,
        intervention_weights=(1.0, 0.0),
        intervention_bias=0.0,
        decision=decision,
        credited_candidate_id="candidate-a",
        noop_candidate_id="no-op",
        signed_credit=0.8,
        learning_rate=0.1,
        max_abs_parameter=4.0,
    )

    assert update.action_weights != weights
    assert update.parameter_delta_l2 > 0.0


def test_shared_policy_noop_credit_changes_only_intervention_timing() -> None:
    weights = (("act-a", (1.0, 0.0)), ("act-b", (0.0, 1.0)))
    decision = rank_and_gate_bounded_policy(
        candidates=_candidates(),
        action_weights=weights,
        intervention_weights=(0.0, 0.0),
        intervention_bias=-4.0,
        maximum_candidates=2,
    )
    assert not decision.intervenes

    update = apply_bounded_policy_credit(
        action_weights=weights,
        intervention_weights=(0.0, 0.0),
        intervention_bias=-4.0,
        decision=decision,
        credited_candidate_id="no-op",
        noop_candidate_id="no-op",
        signed_credit=0.7,
        learning_rate=0.1,
        max_abs_parameter=4.0,
    )

    assert update.action_weights == weights
    assert update.intervention_weights != (0.0, 0.0)


def test_shared_policy_rejects_missing_action_geometry() -> None:
    with pytest.raises(ValueError, match="missing action weights"):
        rank_and_gate_bounded_policy(
            candidates=_candidates(),
            action_weights=(("act-a", (1.0, 0.0)),),
            intervention_weights=(0.0, 0.0),
            intervention_bias=0.0,
            maximum_candidates=2,
        )


def test_bounded_content_policy_preserves_noop_and_learns_exact_credit() -> None:
    feature_order = ("rank", "strength", "recency", "durable", "pe")
    checkpoint = default_bounded_content_policy_checkpoint(
        artifact_id="test-content-policy.v1",
        feature_order=feature_order,
    )
    policy = BoundedContentPolicy()
    candidates = (
        BoundedContentCandidate(
            entry_id="entry-b",
            feature_values=tuple(
                zip(feature_order, (0.5, 0.8, 0.9, 0.0, 0.7), strict=True)
            ),
        ),
        BoundedContentCandidate(
            entry_id="entry-c",
            feature_values=tuple(
                zip(feature_order, (0.3, 0.3, 0.2, 1.0, 0.7), strict=True)
            ),
        ),
    )
    decision = policy.decide(
        owner_order=("entry-a", "entry-b", "entry-c"),
        challengers=candidates,
        source_prediction_id="prediction-1",
        checkpoint=checkpoint,
    )
    assert decision is not None
    if decision.intervened:
        assert decision.output_entry_ids[0] == decision.selected_entry_id
        credited = decision.selected_entry_id
    else:
        assert decision.output_entry_ids == decision.input_entry_ids
        credited = "bounded-content-policy:no-op"
    credit = BoundedContentPolicyCredit.create(
        policy_decision_id=decision.policy_decision_id,
        credited_candidate_id=credited,
        prediction_id="prediction-1",
        settlement_ref="outcome-1",
        signed_prediction_error=0.8,
        source_credit_record_ids=("credit-1", "credit-2", "credit-3", "credit-4"),
        observed_at_ms=100,
    )
    next_checkpoint, receipt = policy.observe_credit(
        checkpoint=checkpoint,
        decision=decision,
        credit=credit,
    )

    assert next_checkpoint.update_count == 1
    assert next_checkpoint.processed_credit_ids == (credit.credit_id,)
    assert receipt.previous_checkpoint_id == checkpoint.checkpoint_id
    assert receipt.next_checkpoint_id == next_checkpoint.checkpoint_id
    assert receipt.parameter_delta_l2 > 0.0
    assert (
        type(checkpoint).from_json(next_checkpoint.to_json())
        == next_checkpoint
    )
    assert type(decision).from_json(decision.to_json()) == decision
    assert type(credit).from_json(credit.to_json()) == credit
    assert type(receipt).from_json(receipt.to_json()) == receipt
