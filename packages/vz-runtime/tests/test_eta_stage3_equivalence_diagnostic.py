"""Cheap unit coverage for the Stage-3 P1 attribution packet."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from volvence_zero.agent.eta_stage3_equivalence_diagnostic import (  # noqa: E402
    Stage3EquivalenceThresholds,
    Stage3ExactEntryProbeReport,
    Stage3SteeringControlPoint,
    assess_stage3_equivalence,
    fit_exact_stage3_entry_probe,
    subgoal_transition_labels,
)
from volvence_zero.substrate import (  # noqa: E402
    ResidualActivation,
    TraceStep,
    TrainingTrace,
)
from volvence_zero.temporal.torch_store_ssl import (  # noqa: E402
    STEERED_TRAINING_BIAS_ONLY,
    STEERED_TRAINING_FULL,
)


def _trace(trace_id: str, vectors: tuple[tuple[float, ...], ...]) -> TrainingTrace:
    return TrainingTrace(
        trace_id=trace_id,
        source_text=trace_id,
        steps=tuple(
            TraceStep(
                step=index,
                token=f"t{index}",
                feature_surface=(),
                residual_activations=(
                    ResidualActivation(
                        layer_index=20,
                        activation=vector,
                        step=index,
                    ),
                ),
            )
            for index, vector in enumerate(vectors)
        ),
    )


def _probe(*, readable: bool = True) -> Stage3ExactEntryProbeReport:
    return Stage3ExactEntryProbeReport(
        layer_indices=(20, 21, 22),
        activation_width=8,
        folded_width=16,
        accuracy=0.9 if readable else 0.2,
        chance_accuracy=0.125,
        majority_accuracy=0.2,
        support=100,
        reference_gate2_accuracy=0.944,
        accuracy_retention_from_gate2=0.95 if readable else 0.21,
        readable_at_two_x_chance=readable,
    )


def _point(
    *,
    seed: int,
    mode: str,
    heldout: float,
    zero: float | None = None,
    permuted: float | None = None,
    action_f1: float = 0.2,
    oracle_f1: float = 0.5,
) -> Stage3SteeringControlPoint:
    return Stage3SteeringControlPoint(
        seed=seed,
        training_mode=mode,
        optimizer_steps=10,
        baseline_train_distortion=2.0,
        baseline_heldout_distortion=2.0,
        train_distortion=heldout,
        heldout_distortion=heldout,
        heldout_zero_z_distortion=zero,
        heldout_permuted_z_distortion=permuted,
        action_boundary_f1=action_f1,
        oracle_subgoal_boundary_f1=oracle_f1,
        oracle_boundary_switch_probability=0.6,
        oracle_continuation_switch_probability=0.2,
        mean_switch_probability=0.3,
        hard_switch_frequency=0.2,
        final_total_loss=heldout,
        final_grad_norm=0.1,
        wall_seconds=1.0,
    )


def test_exact_entry_probe_reads_only_the_folded_single_step_surface() -> None:
    train = (
        _trace("train-a", ((2.0, 0.0, 0.0, 0.0),) * 3),
        _trace("train-b", ((0.0, 2.0, 0.0, 0.0),) * 3),
    )
    heldout = (
        _trace("heldout-a", ((1.5, 0.0, 0.0, 0.0),) * 2),
        _trace("heldout-b", ((0.0, 1.5, 0.0, 0.0),) * 2),
    )

    report = fit_exact_stage3_entry_probe(
        torch_module=torch,
        train_traces=train,
        heldout_traces=heldout,
        train_subgoals={
            "train-a": ("a",) * 3,
            "train-b": ("b",) * 3,
        },
        heldout_subgoals={
            "heldout-a": ("a",) * 2,
            "heldout-b": ("b",) * 2,
        },
        objective_ids=("a", "b"),
        layer_indices=(20,),
        activation_width=4,
        folded_width=4,
        reference_gate2_accuracy=0.9,
    )

    assert report.accuracy == pytest.approx(1.0)
    assert report.readable_at_two_x_chance is True
    assert report.folded_width == 4


def test_subgoal_oracle_labels_changes_not_action_changes() -> None:
    assert subgoal_transition_labels(("a", "a", "b", "b", None)) == (
        0.0,
        1.0,
        0.0,
        1.0,
    )


def test_attribution_identifies_the_free_bias_bypass() -> None:
    points = (
        _point(
            seed=0,
            mode=STEERED_TRAINING_FULL,
            heldout=1.0,
            zero=1.1,
            permuted=1.05,
        ),
        _point(
            seed=0,
            mode=STEERED_TRAINING_BIAS_ONLY,
            heldout=1.15,
        ),
    )

    assessment = assess_stage3_equivalence(
        exact_entry_probe=_probe(),
        control_points=points,
        thresholds=Stage3EquivalenceThresholds(),
    )

    assert assessment.free_bias_bypass_open is True
    assert assessment.learned_z_causal is False
    assert assessment.dominant_attribution == "incentive-bypass-via-free-bias"


def test_attribution_can_clear_bias_and_find_causal_z() -> None:
    points = (
        _point(
            seed=0,
            mode=STEERED_TRAINING_FULL,
            heldout=1.0,
            zero=1.8,
            permuted=1.3,
        ),
        _point(
            seed=0,
            mode=STEERED_TRAINING_BIAS_ONLY,
            heldout=1.8,
        ),
    )

    assessment = assess_stage3_equivalence(
        exact_entry_probe=_probe(),
        control_points=points,
        thresholds=Stage3EquivalenceThresholds(),
    )

    assert assessment.free_bias_bypass_open is False
    assert assessment.learned_z_causal is True
    assert assessment.dominant_attribution == "mechanism-shape-mismatch"
    assert assessment.boundary_semantics_materially_different is True


def test_information_dead_entry_takes_precedence_in_attribution() -> None:
    points = (
        _point(
            seed=0,
            mode=STEERED_TRAINING_FULL,
            heldout=1.0,
            zero=1.1,
            permuted=1.05,
        ),
        _point(
            seed=0,
            mode=STEERED_TRAINING_BIAS_ONLY,
            heldout=1.1,
        ),
    )

    assessment = assess_stage3_equivalence(
        exact_entry_probe=_probe(readable=False),
        control_points=points,
        thresholds=Stage3EquivalenceThresholds(),
    )

    assert assessment.dominant_attribution == "information-dead-at-stage3-entry"


def test_attribution_requires_matched_full_and_bias_seeds() -> None:
    with pytest.raises(ValueError, match="matched full and bias-only"):
        assess_stage3_equivalence(
            exact_entry_probe=_probe(),
            control_points=(
                _point(
                    seed=0,
                    mode=STEERED_TRAINING_FULL,
                    heldout=1.0,
                    zero=1.5,
                    permuted=1.3,
                ),
            ),
            thresholds=Stage3EquivalenceThresholds(),
        )
