from __future__ import annotations

import pytest

from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_s2_causal_steering import (
    S2CausalSteeringThresholds,
    S2EffectEstimate,
    S2ScaleMetrics,
    _bootstrap_effect,
    _heldout_steering_rows,
    assess_s2_causal_steering,
)


def _effect(mean: float, *, lower: float | None = None) -> S2EffectEstimate:
    return S2EffectEstimate(
        mean=mean,
        ci_lower=mean / 2.0 if lower is None else lower,
        ci_upper=mean * 1.5,
        route_count=24,
    )


def _metrics(
    *,
    noop: float = 0.05,
    minus: float = 0.08,
    shuffled: float = 0.05,
    win_rate: float = 0.75,
    lower: float = 0.01,
) -> S2ScaleMetrics:
    return S2ScaleMetrics(
        scale_fraction=0.5,
        control_norm=1.0,
        noop_mean_nll=1.0,
        plus_mean_nll=0.95,
        minus_mean_nll=1.03,
        shuffled_mean_nll=1.0,
        plus_vs_noop=_effect(noop, lower=lower),
        plus_vs_minus=_effect(minus, lower=lower),
        plus_vs_shuffled=_effect(shuffled, lower=lower),
        plus_vs_noop_route_win_rate=win_rate,
        plus_vs_minus_route_win_rate=win_rate,
        plus_vs_shuffled_route_win_rate=win_rate,
        plus_vs_noop_row_win_rate=0.7,
        row_count=299,
    )


def test_s2_admission_requires_effect_specificity_and_route_consistency() -> None:
    admitted = assess_s2_causal_steering(
        primary=_metrics(),
        thresholds=S2CausalSteeringThresholds(),
    )
    assert admitted.admitted is True
    assert admitted.failed_conditions == ()

    blocked = assess_s2_causal_steering(
        primary=_metrics(
            noop=0.0,
            minus=0.01,
            shuffled=-0.01,
            win_rate=0.5,
            lower=-0.02,
        ),
        thresholds=S2CausalSteeringThresholds(),
    )
    assert blocked.admitted is False
    assert blocked.failed_conditions == (
        "plus-vs-noop-effect",
        "plus-vs-minus-effect",
        "plus-vs-shuffled-effect",
        "route-win-rate",
        "bootstrap-lower-positive",
    )


def test_s2_bootstrap_is_route_level_and_deterministic() -> None:
    first = _bootstrap_effect(
        (0.1, 0.2, 0.3, 0.4),
        seed=17,
        resamples=500,
        confidence=0.95,
    )
    second = _bootstrap_effect(
        (0.1, 0.2, 0.3, 0.4),
        seed=17,
        resamples=500,
        confidence=0.95,
    )
    assert first == second
    assert first.mean == pytest.approx(0.25)
    assert first.route_count == 4
    assert first.ci_lower < first.mean < first.ci_upper


def test_s2_heldout_rows_keep_probe_subgoal_and_action_lineage_aligned() -> None:
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
    rows, class_ids = _heldout_steering_rows(corpus)

    assert len(rows) == 299
    assert len(class_ids) == 8
    assert len({row.sample_id for row in rows}) == len(rows)
    assert all(row.active_subgoal in class_ids for row in rows)
    assert all(row.expert_action_id.startswith("move:") for row in rows)
