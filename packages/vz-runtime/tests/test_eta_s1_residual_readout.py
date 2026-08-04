from __future__ import annotations

import pytest

from volvence_zero.agent.eta_s1_residual_readout import (
    S1ResidualReadoutMetrics,
    S1ResidualReadoutThresholds,
    assess_s1_residual_readout,
    run_eta_s1_residual_readout,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SyntheticOpenWeightResidualRuntime,
)


def _metrics(
    *,
    accuracy: float,
    late_accuracy: float,
) -> S1ResidualReadoutMetrics:
    return S1ResidualReadoutMetrics(
        accuracy=accuracy,
        chance_accuracy=0.125,
        majority_accuracy=0.25,
        early_accuracy=accuracy,
        late_accuracy=late_accuracy,
        mean_score_margin=0.4,
        min_score_margin=0.01,
        support=100,
        early_support=50,
        late_support=50,
    )


def test_s1_admission_requires_accuracy_retention_and_generalization() -> None:
    admitted = assess_s1_residual_readout(
        train_metrics=_metrics(accuracy=0.96, late_accuracy=0.95),
        heldout_metrics=_metrics(accuracy=0.90, late_accuracy=0.82),
        thresholds=S1ResidualReadoutThresholds(),
    )
    assert admitted.admitted is True
    assert admitted.failed_conditions == ()

    blocked = assess_s1_residual_readout(
        train_metrics=_metrics(accuracy=1.0, late_accuracy=1.0),
        heldout_metrics=_metrics(accuracy=0.70, late_accuracy=0.45),
        thresholds=S1ResidualReadoutThresholds(),
    )
    assert blocked.admitted is False
    assert blocked.failed_conditions == (
        "heldout-accuracy",
        "late-retention",
        "generalization-gap",
    )


def test_s1_admission_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="chance_multiple must be positive"):
        assess_s1_residual_readout(
            train_metrics=_metrics(accuracy=0.9, late_accuracy=0.9),
            heldout_metrics=_metrics(accuracy=0.9, late_accuracy=0.9),
            thresholds=S1ResidualReadoutThresholds(chance_multiple=0.0),
        )


def test_s1_evidence_runs_through_public_snapshot_publishers() -> None:
    corpus = generate_eta_proof_corpus(
        seed=20260802,
        objective_count=8,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=64,
        heldout_route_count=24,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-s1")
    report, artifact = run_eta_s1_residual_readout(
        corpus=corpus,
        runtime=runtime,
        model_fingerprint=SubstrateFingerprint(
            model_id="synthetic-s1",
            version="fixture-v1",
            weights_sha256="a" * 64,
        ),
        model_source="synthetic-fixture",
        device="cpu",
        expected_layer_indices=(0, 1, 2),
        expected_activation_widths=(3, 3, 3),
    )

    assert report.train_row_count > report.heldout_row_count > 0
    assert report.artifact_id == artifact.artifact_id
    assert report.production_wiring_changed is False
    assert report.feedback_to_learning is False
    assert artifact.training_snapshot_fingerprint == (
        report.training_snapshot_fingerprint
    )
