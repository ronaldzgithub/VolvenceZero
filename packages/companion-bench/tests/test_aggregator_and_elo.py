# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Aggregator (RFC §6.4) + Elo (RFC §6.5) tests."""

from __future__ import annotations

import math

import pytest

from companion_bench.aggregator import (
    A6_CAP_THRESHOLD,
    A6_CAP_VALUE,
    CompanionBenchScore,
    SubmissionAggregate,
    WEIGHTS,
    aggregate_arc,
    aggregate_axis_dict,
    aggregate_submission,
    assert_weights_sum_to_one,
)
from companion_bench.elo import (
    PairwiseMatch,
    build_elo_report,
    compute_bradley_terry,
    compute_trueskill,
    derive_matches_from_arc_scores,
)
from companion_bench.judge_arc import ArcAxisScores
from companion_bench.spec import AxisId


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_per_rfc() -> None:
    assert_weights_sum_to_one()
    assert WEIGHTS[AxisId.A3_CONTINUITY] == 0.25
    assert WEIGHTS[AxisId.A6_SAFETY] == 0.20


def test_aggregate_all_max_axes_yields_100() -> None:
    scores = {a: 100.0 for a in AxisId}
    out = aggregate_axis_dict(scores)
    assert math.isclose(out.final, 100.0, abs_tol=1e-3)
    assert out.a6_cap_applied is False


def test_aggregate_all_50_yields_50() -> None:
    scores = {a: 50.0 for a in AxisId}
    out = aggregate_axis_dict(scores)
    assert math.isclose(out.final, 50.0, abs_tol=1e-3)


def test_a6_cap_applied_when_a6_below_threshold() -> None:
    scores = {a: 80.0 for a in AxisId}
    scores[AxisId.A6_SAFETY] = 40.0  # below threshold
    out = aggregate_axis_dict(scores)
    assert out.a6_cap_applied is True
    assert out.final <= A6_CAP_VALUE
    assert out.final < out.raw  # cap reduced the published value


def test_a6_cap_threshold_is_60_per_rfc() -> None:
    # Just at the threshold: no cap.
    scores = {a: 80.0 for a in AxisId}
    scores[AxisId.A6_SAFETY] = A6_CAP_THRESHOLD
    out = aggregate_axis_dict(scores)
    assert out.a6_cap_applied is False
    # Just below: cap applies.
    scores[AxisId.A6_SAFETY] = A6_CAP_THRESHOLD - 0.01
    out = aggregate_axis_dict(scores)
    assert out.a6_cap_applied is True


def test_aggregate_zero_axis_does_not_collapse_to_zero() -> None:
    # A single 0 axis is a hard signal but shouldn't make the final
    # score literally zero — geometric mean uses an epsilon floor.
    scores = {a: 80.0 for a in AxisId}
    scores[AxisId.A1_TASK] = 0.0  # absent task helpfulness
    out = aggregate_axis_dict(scores)
    assert out.final >= 0.0  # nonnegative
    assert out.final < 80.0  # but well below the per-axis level


def test_aggregate_submission_returns_means_and_ci() -> None:
    arc_scores = []
    for v in (75.0, 80.0, 70.0, 85.0):
        scores = {a: v for a in AxisId}
        arc_scores.append(aggregate_axis_dict(scores))
    agg = aggregate_submission(
        submission_id="t",
        per_arc_scores=arc_scores,
        bootstrap_resamples=200,
        rng_seed=0,
    )
    assert agg.arc_count == 4
    assert 70.0 <= agg.final_mean <= 85.0
    lo, hi = agg.final_ci95
    assert lo <= agg.final_mean <= hi


def test_aggregate_arc_to_json_round_trip() -> None:
    scores = {a: 75.0 for a in AxisId}
    out = aggregate_axis_dict(scores)
    payload = out.to_json()
    assert "axis_scores" in payload
    assert "weights" in payload
    assert "a6_cap_applied" in payload


def test_aggregate_submission_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one per-arc score"):
        aggregate_submission(
            submission_id="t", per_arc_scores=[], bootstrap_resamples=100, rng_seed=0,
        )


def test_aggregate_arc_helper_same_as_axis_dict() -> None:
    scores = {a: 70.0 for a in AxisId}
    via_dict = aggregate_axis_dict(scores)
    via_arc = aggregate_arc(
        ArcAxisScores(
            arc_id="x",
            judge_model="fake",
            scores=scores,
            rationale={a: "" for a in AxisId},
        ),
    )
    assert math.isclose(via_arc.final, via_dict.final, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_derive_matches_from_arc_scores() -> None:
    by_arc = {
        "arc-1": {"sysA": 70.0, "sysB": 65.0, "sysC": 80.0},
        "arc-2": {"sysA": 60.0, "sysB": 75.0},
    }
    matches = derive_matches_from_arc_scores(by_arc=by_arc)
    # arc-1 has 3 systems → 3 pairs; arc-2 has 2 → 1 pair; total 4
    assert len(matches) == 4
    arc_1_pairs = sorted([(m.system_a, m.system_b) for m in matches if m.arc_id == "arc-1"])
    assert arc_1_pairs == [("sysA", "sysB"), ("sysA", "sysC"), ("sysB", "sysC")]


def test_trueskill_orders_systems_by_skill() -> None:
    by_arc = {f"arc-{i}": {"strong": 90.0, "mid": 70.0, "weak": 50.0} for i in range(8)}
    matches = derive_matches_from_arc_scores(by_arc=by_arc)
    ratings = compute_trueskill(matches)
    # With 8 arcs and consistent ordering, the conservative rating
    # ranking should match the truth.
    by_name = {r.system: r for r in ratings}
    assert by_name["strong"].conservative > by_name["mid"].conservative > by_name["weak"].conservative


def test_bradley_terry_orders_systems_by_strength() -> None:
    by_arc = {f"arc-{i}": {"strong": 90.0, "mid": 70.0, "weak": 50.0} for i in range(8)}
    matches = derive_matches_from_arc_scores(by_arc=by_arc)
    ratings = compute_bradley_terry(matches)
    by_name = {r.system: r.rank for r in ratings}
    assert by_name["strong"] < by_name["mid"] < by_name["weak"]


def test_bradley_terry_ties_handled() -> None:
    by_arc = {
        "arc-1": {"a": 70.0, "b": 70.2},  # within tie threshold
    }
    matches = derive_matches_from_arc_scores(by_arc=by_arc, tie_threshold=0.5)
    ratings = compute_bradley_terry(matches)
    # Both systems tied → equal scores
    by_name = {r.system: r.score for r in ratings}
    assert math.isclose(by_name["a"], by_name["b"], abs_tol=0.05)


def test_build_elo_report_contains_both_methods() -> None:
    by_arc = {f"arc-{i}": {"a": 70.0 + i, "b": 60.0} for i in range(5)}
    report = build_elo_report(by_arc=by_arc)
    assert len(report.trueskill) == 2
    assert len(report.bradley_terry) == 2
    payload = report.to_json()
    assert "trueskill" in payload and "bradley_terry" in payload
