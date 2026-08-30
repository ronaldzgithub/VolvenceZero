from __future__ import annotations

import pytest

from lifeform_core import (
    BoundedPolicyCandidate,
    apply_bounded_policy_credit,
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
