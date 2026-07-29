from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.apprenticeship import (
    BoundedBinaryAlignmentReadout,
    Gate4FeedbackCandidate,
    select_active_candidate,
)
from volvence_zero.agent.gate4_active_learning_evidence import (
    GATE4_ARM_NAMES,
    Gate4ArmMetrics,
    compare_gate4_arms,
    run_gate4_arm,
    validate_gate4_source_records,
)
from volvence_zero.agent.shared_settled_trace import (
    build_shared_trace_plans,
)


def _candidate(
    transition_id: str,
    *,
    pe: float,
    segment: tuple[float, ...],
) -> Gate4FeedbackCandidate:
    return Gate4FeedbackCandidate(
        transition_id=transition_id,
        guidance_text=f"guidance {transition_id}",
        turn_index=int(transition_id[-1]),
        predictor_features=(0.0, 1.0),
        pe_magnitude=pe,
        segment_features=segment,
    )


def _metric(
    *,
    seed: int,
    arm: str,
    labels_needed: int,
) -> Gate4ArmMetrics:
    return Gate4ArmMetrics(
        seed=seed,
        arm=arm,
        label_budget=0 if arm == "no-feedback" else 60,
        requested_label_count=0 if arm == "no-feedback" else 60,
        labels_needed_for_target=labels_needed,
        heldout_balanced_accuracy=0.85,
        locked_balanced_accuracy=0.84,
        cumulative_regret=10,
        ineffective_request_rate=0.1,
        missed_high_risk_rate=0.1,
        typed_request_coverage=1.0,
        open_loop_actuation_coverage=1.0,
        proposal_count=0,
        boundary_digest_unchanged=True,
        source_closure_coverage=1.0,
        lineage_complete=True,
        frozen_substrate_mutation_count=0,
    )


def _records(seed: int) -> list[dict]:
    rows: list[dict] = []
    for plan in build_shared_trace_plans(seed):
        high_risk = plan.episode_phase in {
            "new-introduce",
            "new-revision",
        }
        rows.append(
            {
                "transition_id": plan.transition_id,
                "seed": plan.seed,
                "global_index": plan.global_index,
                "partition": plan.partition,
                "context_id": plan.context_id,
                "user_id": plan.user_id,
                "episode_phase": plan.episode_phase,
                "knowledge_key": plan.knowledge_key,
                "settled": True,
                "record_sha256": f"digest:{plan.transition_id}",
                "lineage": {
                    "prediction_ref": f"ref:{plan.transition_id}",
                    "environment_outcome_id": (
                        f"outcome:{plan.transition_id}"
                    ),
                },
                "input": {
                    "settlement_turn": (
                        f"typed settlement {plan.transition_id}"
                    )
                },
                "prediction": {
                    "predicted_task_progress": (
                        0.2 if high_risk else 0.5
                    ),
                    "predicted_relationship_delta": 0.5,
                    "predicted_regime_stability": 0.5,
                    "predicted_action_payoff": (
                        0.1 if high_risk else 0.4
                    ),
                    "confidence": 0.6,
                },
                "actual_outcome": {
                    "task_progress": 0.2 if high_risk else 0.5,
                    "action_payoff": 0.1 if high_risk else 0.4,
                },
                "prediction_error": {
                    "magnitude": 0.8 if high_risk else 0.2,
                },
                "temporal_snapshot": {
                    "controller_state": {
                        "code": [
                            float(plan.global_index % 3) / 2.0,
                            float(plan.global_index % 5) / 4.0,
                        ],
                        "switch_gate": 1.0,
                        "steps_since_switch": 0,
                    },
                    "closed_segments": [
                        {
                            "segment_id": f"segment:{plan.transition_id}",
                            "abstract_action_id": (
                                f"family:{plan.global_index % 4}"
                            ),
                            "z_t_digest": [
                                float(plan.global_index % 3) / 2.0,
                                float(plan.global_index % 5) / 4.0,
                            ],
                            "beta_open_digest": 0.2,
                            "beta_close_digest": 0.9,
                            "open_turn_index": 1,
                            "close_turn_index": 2,
                        }
                    ],
                },
                "credit_snapshot": {},
                "substrate": {
                    "runtime_origin": "hf-local",
                    "fallback_active": False,
                    "is_frozen": True,
                    "mutation_applied": False,
                },
            }
        )
    return rows


def test_bounded_alignment_readout_learns_separable_feedback() -> None:
    readout = BoundedBinaryAlignmentReadout(feature_count=2)
    readout.fit(
        (
            ((-1.0, -0.5), False),
            ((-0.8, -0.2), False),
            ((0.8, 0.2), True),
            ((1.0, 0.5), True),
        )
    )
    assert readout.predict_probability((-0.9, -0.3)) < 0.5
    assert readout.predict_probability((0.9, 0.3)) > 0.5
    assert all(abs(value) <= 4.0 for value in readout.parameters)


def test_segment_selector_consumes_boundary_novelty_but_turn_does_not() -> None:
    candidates = (
        _candidate(
            "candidate-0",
            pe=0.5,
            segment=(0.0, 0.0, 0.2, 0.8, 1.0),
        ),
        _candidate(
            "candidate-1",
            pe=0.5,
            segment=(1.0, 1.0, 0.0, 1.0, 4.0),
        ),
    )
    probabilities = (0.5, 0.5)
    segment_choice = select_active_candidate(
        candidates=candidates,
        probabilities=probabilities,
        selected_indices=frozenset({0}),
        policy="segment-aware-active",
    )
    turn_choice = select_active_candidate(
        candidates=candidates,
        probabilities=probabilities,
        selected_indices=frozenset(),
        policy="turn-level-active",
    )
    assert segment_choice == 1
    assert turn_choice == 1


def test_gate4_comparison_requires_primary_and_boundary_kill_control() -> None:
    metrics = []
    for seed in (401, 409, 419):
        labels = {
            "segment-aware-active": 35,
            "turn-level-active": 45,
            "random-feedback": 50,
            "no-feedback": 1,
            "shuffled-segment-boundary": 42,
        }
        metrics.extend(
            _metric(seed=seed, arm=arm, labels_needed=labels[arm])
            for arm in GATE4_ARM_NAMES
        )
    comparisons, gates = compare_gate4_arms(metrics)
    assert len(comparisons) == 2
    assert gates["segment_primary_vs_turn_and_random"] is True
    assert gates["segment_boundary_kill_control_passed"] is True

    no_boundary_effect = [
        replace(metric, labels_needed_for_target=35)
        if metric.arm == "shuffled-segment-boundary"
        else metric
        for metric in metrics
    ]
    _, failed = compare_gate4_arms(no_boundary_effect)
    assert failed["segment_boundary_kill_control_passed"] is False


def test_random_arm_uses_typed_requests_without_writes() -> None:
    records = _records(401)
    metric, curve, requests = run_gate4_arm(
        records=records,
        seed=401,
        arm="random-feedback",
        consume_locked=False,
    )
    assert metric.requested_label_count == 60
    assert metric.typed_request_coverage == 1.0
    assert metric.open_loop_actuation_coverage == 1.0
    assert metric.proposal_count == 0
    assert metric.boundary_digest_unchanged is True
    assert metric.source_closure_coverage == 1.0
    assert len(curve) == 60
    assert len(requests) == 60
    no_feedback, _, _ = run_gate4_arm(
        records=records,
        seed=401,
        arm="no-feedback",
        consume_locked=False,
    )
    assert no_feedback.labels_needed_for_target == 61
    assert no_feedback.locked_balanced_accuracy is None


def test_gate4_source_rejects_trace_without_public_segment_closure() -> None:
    records = _records(401)
    records[0]["temporal_snapshot"]["closed_segments"] = []
    with pytest.raises(ValueError, match="lacks owner segment closure"):
        validate_gate4_source_records(records, seed=401)
