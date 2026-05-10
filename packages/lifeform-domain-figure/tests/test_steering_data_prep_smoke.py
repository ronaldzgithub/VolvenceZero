"""Smoke tests for the F5/P5.1 contrast set + steering data prep.

Validates:

* The Einstein contrast set covers the three documented axes
  (locality / determinism / completeness) with non-empty,
  non-duplicate paraphrases.
* :func:`build_steering_training_plan` produces normalised, equal-
  dimensional positive / negative residuals for every pair.
* The resulting plan integrity hash is deterministic across two
  calls (R15 byte-for-byte rollback contract).
* Validation errors fire fail-loud when a contrast pair would
  produce an empty residual or when residual dimensions disagree.
"""

from __future__ import annotations

import math

import pytest

from lifeform_domain_figure import (
    FigureContrastPair,
    FigureContrastSet,
    FigureSteeringTrainingPlan,
    STEERING_PLAN_SCHEMA_VERSION,
    SteeringTrainingPair,
    build_einstein_contrast_set,
    build_steering_training_plan,
)


def test_einstein_contrast_set_has_three_axes() -> None:
    contrast = build_einstein_contrast_set()
    assert contrast.figure_id == "einstein"
    assert len(contrast.pairs) == 3
    axes = {pair.axis for pair in contrast.pairs}
    assert axes == {"locality", "determinism", "completeness"}
    pair_ids = [pair.pair_id for pair in contrast.pairs]
    assert len(pair_ids) == len(set(pair_ids))


def test_einstein_contrast_pairs_carry_distinct_opponents() -> None:
    contrast = build_einstein_contrast_set()
    opponents = {pair.opponent_id for pair in contrast.pairs}
    assert opponents == {"bohr", "born", "heisenberg"}


def test_steering_plan_round_trip_is_deterministic() -> None:
    contrast = build_einstein_contrast_set()
    plan_a = build_steering_training_plan(contrast)
    plan_b = build_steering_training_plan(contrast)
    assert isinstance(plan_a, FigureSteeringTrainingPlan)
    assert plan_a.schema_version == STEERING_PLAN_SCHEMA_VERSION
    assert plan_a.figure_id == "einstein"
    assert plan_a.total_pairs == 3
    assert plan_a.axes == ("completeness", "determinism", "locality")
    assert plan_a.integrity_hash == plan_b.integrity_hash


def test_steering_plan_residuals_are_normalised_and_distinct() -> None:
    plan = build_steering_training_plan(build_einstein_contrast_set())
    for pair in plan.pairs:
        assert isinstance(pair, SteeringTrainingPair)
        assert len(pair.positive_residual) == plan.embedding_dim
        assert len(pair.negative_residual) == plan.embedding_dim
        positive_norm = math.sqrt(sum(v * v for v in pair.positive_residual))
        negative_norm = math.sqrt(sum(v * v for v in pair.negative_residual))
        assert positive_norm == pytest.approx(1.0, rel=1e-3)
        assert negative_norm == pytest.approx(1.0, rel=1e-3)
        assert pair.positive_residual != pair.negative_residual


def test_contrast_pair_rejects_empty_field() -> None:
    with pytest.raises(ValueError, match="figure_stance"):
        FigureContrastPair(
            pair_id="x",
            axis="locality",
            figure_stance="",
            opponent_id="bohr",
            opponent_stance="something",
            evidence_locator="loc",
            confidence=0.9,
            description="d",
        )


def test_contrast_pair_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        FigureContrastPair(
            pair_id="x",
            axis="locality",
            figure_stance="stance",
            opponent_id="bohr",
            opponent_stance="something",
            evidence_locator="loc",
            confidence=1.5,
            description="d",
        )


def test_contrast_set_rejects_duplicate_pair_ids() -> None:
    base = build_einstein_contrast_set().pairs[0]
    with pytest.raises(ValueError, match="duplicate pair_id"):
        FigureContrastSet(
            figure_id="einstein",
            pairs=(base, base),
        )


def test_contrast_set_rejects_empty_pairs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FigureContrastSet(figure_id="einstein", pairs=())


def test_steering_training_pair_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensionality"):
        SteeringTrainingPair(
            pair_id="x",
            axis="locality",
            opponent_id="bohr",
            positive_residual=(1.0, 0.0, 0.0),
            negative_residual=(1.0, 0.0),
            confidence=0.5,
            evidence_locator="loc",
        )


def test_steering_plan_carries_contrast_description() -> None:
    plan = build_steering_training_plan(build_einstein_contrast_set())
    assert "Einstein" in plan.contrast_set_description
    assert "locality" in plan.contrast_set_description
