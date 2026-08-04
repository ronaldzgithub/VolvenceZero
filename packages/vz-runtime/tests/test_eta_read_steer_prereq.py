"""Unit tests for the ETA read->steer S3 prerequisite (no model required)."""

from __future__ import annotations

import numpy as np
import pytest

from volvence_zero.agent.eta_read_steer_prereq import (
    ReadSteerAggregate,
    ReadSteerThresholds,
    _FrozenLinearReader,
    _ReadSteerExample,
    _reader_accuracy,
    _route_bootstrap,
    assess_read_steer,
    fit_condition_reader,
)


def _example(subgoal: int, context: tuple[float, ...]) -> _ReadSteerExample:
    return _ReadSteerExample(
        case_id=f"case-{subgoal}",
        observation_text="obs",
        subgoal_revealed_text="ctx",
        subgoal_index=subgoal,
        action_index=0,
        action_residual=(0.0, 0.0, 0.0),
        context_residual=context,
    )


def _separable_examples() -> tuple[_ReadSteerExample, ...]:
    rng = np.random.default_rng(7)
    centers = np.eye(3) * 6.0
    examples: list[_ReadSteerExample] = []
    for subgoal in range(3):
        for _ in range(24):
            point = centers[subgoal] + rng.normal(scale=0.2, size=3)
            examples.append(_example(subgoal, tuple(float(v) for v in point)))
    return tuple(examples)


def test_condition_reader_reads_separable_context() -> None:
    examples = _separable_examples()
    reader = fit_condition_reader(examples, class_count=3, ridge_lambda=1.0)
    assert _reader_accuracy(reader, examples) >= 0.95


def test_frozen_linear_reader_predicts_by_argmax() -> None:
    reader = _FrozenLinearReader(
        weights=np.array([[3.0, -3.0], [-3.0, 3.0]]),
        feature_mean=np.zeros(2),
        feature_scale=np.ones(2),
    )
    predictions = reader.predict(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert predictions.tolist() == [0, 1]


def test_route_bootstrap_lower_positive_for_constant_gain() -> None:
    case_ids = tuple(f"c{index // 4}" for index in range(20))
    effect = [0.5] * 20
    result = _route_bootstrap(
        case_ids=case_ids,
        per_row_effect=effect,
        seed=1,
        resamples=500,
        confidence=0.95,
    )
    assert result.ci_lower > 0.0
    assert result.route_count == 5
    assert result.mean == pytest.approx(0.5)


def test_route_bootstrap_lower_can_be_nonpositive_for_noisy_zero() -> None:
    case_ids = tuple(f"c{index}" for index in range(20))
    effect = [(-1.0) ** index for index in range(20)]
    result = _route_bootstrap(
        case_ids=case_ids,
        per_row_effect=effect,
        seed=2,
        resamples=500,
        confidence=0.95,
    )
    assert result.ci_lower <= 0.0


def _aggregate(**overrides: float) -> ReadSteerAggregate:
    base = dict(
        seed_count=3,
        reader_heldout_accuracy_mean=0.99,
        noop_nll_mean=2.8,
        conditional_oracle_nll_mean=0.05,
        conditional_online_nll_mean=0.06,
        unconditional_nll_mean=1.3,
        random_condition_nll_mean=7.0,
        subgoal_revealed_ceiling_nll_mean=0.22,
        online_gap_closed_nll_mean=2.74,
        online_conditional_advantage_nll_mean=1.24,
        online_vs_noop_ci_lower_min=2.4,
        online_vs_unconditional_ci_lower_min=0.9,
    )
    base.update(overrides)
    return ReadSteerAggregate(**base)


def _assess(aggregate: ReadSteerAggregate):
    return assess_read_steer(
        aggregate=aggregate,
        thresholds=ReadSteerThresholds(),
        free_bias_present=False,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=True,
    )


def test_admission_passes_when_read_loop_closes() -> None:
    assert _assess(_aggregate()).admitted


def test_admission_blocks_when_reader_is_chance() -> None:
    admission = _assess(_aggregate(reader_heldout_accuracy_mean=0.13))
    assert not admission.admitted
    assert "reader-accuracy" in admission.failed_conditions


def test_admission_blocks_when_online_does_not_beat_unconditional() -> None:
    admission = _assess(
        _aggregate(
            online_conditional_advantage_nll_mean=0.0,
            online_vs_unconditional_ci_lower_min=-0.2,
        )
    )
    assert not admission.admitted
    assert "online-conditional-advantage" in admission.failed_conditions
    assert "bootstrap-lower-positive" in admission.failed_conditions


def test_admission_blocks_when_bootstrap_lower_crosses_zero() -> None:
    admission = _assess(_aggregate(online_vs_noop_ci_lower_min=-0.01))
    assert not admission.admitted
    assert "bootstrap-lower-positive" in admission.failed_conditions


def test_admission_blocks_when_free_bias_present() -> None:
    admission = assess_read_steer(
        aggregate=_aggregate(),
        thresholds=ReadSteerThresholds(),
        free_bias_present=True,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=True,
    )
    assert not admission.admitted
    assert "structural-integrity" in admission.failed_conditions
