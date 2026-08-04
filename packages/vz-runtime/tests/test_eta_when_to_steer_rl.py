"""Unit tests for the S3 when-to-steer Internal RL gate (no model required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from volvence_zero.agent.eta_when_to_steer_rl import (
    WhenToSteerAggregate,
    WhenToSteerThresholds,
    _evaluate_policy,
    _feature_matrix,
    _RowRecord,
    _route_bootstrap,
    _run_seed,
    _standardize_columns,
    _train_gate_policy,
    assess_when_to_steer,
)


def _record(
    *,
    case_id: str,
    order: int,
    post_switch: bool,
    agrees: bool,
    steer_nll: float,
    noop_nll: float,
) -> _RowRecord:
    return _RowRecord(
        case_id=case_id,
        order_in_case=order,
        post_switch=post_switch,
        belief_correct=agrees,
        belief_agrees_fresh=agrees,
        belief_margin=0.9 if agrees else 0.8,
        fresh_margin=0.95,
        base_action_entropy=0.4 if agrees else 1.1,
        noop_nll=noop_nll,
        steer_belief_nll=steer_nll,
    )


def _separable_records() -> tuple[_RowRecord, ...]:
    """Fresh rows: steering is great. Post-switch rows: steering is terrible."""

    records: list[_RowRecord] = []
    for case in range(8):
        for order in range(4):
            post = order >= 2
            if post:
                records.append(
                    _record(
                        case_id=f"c{case}",
                        order=order,
                        post_switch=True,
                        agrees=False,
                        steer_nll=7.0,
                        noop_nll=2.5,
                    )
                )
            else:
                records.append(
                    _record(
                        case_id=f"c{case}",
                        order=order,
                        post_switch=False,
                        agrees=True,
                        steer_nll=0.02,
                        noop_nll=2.8,
                    )
                )
    return tuple(records)


def _features(records: tuple[_RowRecord, ...]) -> torch.Tensor:
    matrix = _feature_matrix(records)
    mean, scale = _standardize_columns(matrix, continuous_columns=(0, 1, 3))
    return torch.tensor((matrix - mean) / scale, dtype=torch.float32)


def test_feature_matrix_encodes_disagreement_bit() -> None:
    agree = _record(
        case_id="c",
        order=0,
        post_switch=False,
        agrees=True,
        steer_nll=0.1,
        noop_nll=2.0,
    )
    disagree = _record(
        case_id="c",
        order=1,
        post_switch=True,
        agrees=False,
        steer_nll=7.0,
        noop_nll=2.0,
    )
    matrix = _feature_matrix((agree, disagree))
    assert matrix[0, 2] == 0.0
    assert matrix[1, 2] == 1.0


def test_route_bootstrap_positive_for_constant_gain() -> None:
    case_ids = tuple(f"c{index // 4}" for index in range(16))
    result = _route_bootstrap(
        case_ids=case_ids,
        per_row_effect=[0.7] * 16,
        seed=1,
        resamples=400,
        confidence=0.95,
    )
    assert result.ci_lower > 0.0
    assert result.route_count == 4
    assert result.mean == pytest.approx(0.7)


def test_reinforce_gate_learns_to_steer_only_when_fresh() -> None:
    records = _separable_records()
    features = _features(records)
    policy, trajectory, initial = _train_gate_policy(
        torch=torch,
        seed=0,
        train_records=records,
        train_features=features,
        heldout_records=records,
        heldout_features=features,
        max_online_episodes=400,
        policy_learning_rate=0.1,
        policy_batch_cases=4,
        entropy_coef=0.02,
        init_noop_bias=2.0,
        eval_every=40,
        baseline_beta=0.9,
        progress=None,
    )
    chosen, steer_flags = _evaluate_policy(
        torch=torch, policy=policy, features=features, records=records
    )
    steer_fresh = np.mean(
        [steer_flags[i] for i, r in enumerate(records) if not r.post_switch]
    )
    steer_post = np.mean(
        [steer_flags[i] for i, r in enumerate(records) if r.post_switch]
    )
    # Learned gate must concentrate steering where the belief is fresh.
    assert steer_fresh > 0.8
    assert steer_post < 0.2
    # And its chosen-arm NLL must beat always-on-belief (mean steer NLL).
    always_on = float(np.mean([r.steer_belief_nll for r in records]))
    assert float(np.mean(chosen)) < always_on
    # Learning happened: heldout NLL improved from first to last eval.
    assert trajectory[0] - trajectory[-1] > 0.2
    assert any(
        float((a - b.detach()).abs().max()) > 1e-8
        for a, b in zip(initial, policy.parameters(), strict=True)
    )


def test_run_seed_multi_restart_selects_train_best_and_beats_always_on() -> None:
    records = _separable_records()
    features = _features(records)
    point = _run_seed(
        torch=torch,
        seed=3,
        train_records=records,
        train_features=features,
        heldout_records=records,
        heldout_features=features,
        heldout_fresh_ceiling=[0.02] * len(records),
        max_online_episodes=200,
        policy_learning_rate=0.1,
        policy_batch_cases=4,
        entropy_coef=0.1,
        init_noop_bias=0.0,
        policy_restarts=4,
        eval_every=40,
        convergence_window=3,
        baseline_beta=0.9,
        bootstrap_resamples=200,
        bootstrap_confidence=0.95,
        progress=None,
    )
    # A restart within range was selected and its train-side NLL recorded.
    assert 0 <= point.selected_restart < 4
    assert point.selection_train_nll < point.arms.always_on_belief
    # Robustified selection yields a selective gate that beats always-on-belief.
    assert point.arms.pe_gated_online < point.arms.always_on_belief
    assert point.gate_selectivity > 0.5


def _aggregate(**overrides: float) -> WhenToSteerAggregate:
    base = dict(
        seed_count=3,
        noop_nll_mean=2.8,
        pe_gated_online_nll_mean=1.1,
        always_on_belief_nll_mean=1.79,
        random_gate_nll_mean=1.9,
        oracle_gate_ceiling_nll_mean=1.09,
        pe_hard_gate_ceiling_nll_mean=1.09,
        fresh_ceiling_nll_mean=0.03,
        convergence_improvement_nll_mean=0.6,
        gate_selectivity_mean=0.7,
        gain_vs_noop_ci_lower_min=1.2,
        gain_vs_always_on_ci_lower_min=0.4,
        gain_vs_random_gate_ci_lower_min=0.5,
    )
    base.update(overrides)
    return WhenToSteerAggregate(**base)


def _assess(aggregate: WhenToSteerAggregate, **structural: object):
    kwargs = dict(
        free_bias_present=False,
        zero_code_strict_noop=True,
        substrate_trainable_parameter_count=0,
        reader_parameters_changed=False,
        executor_parameters_changed=False,
        policy_parameters_changed=True,
    )
    kwargs.update(structural)
    return assess_when_to_steer(
        aggregate=aggregate, thresholds=WhenToSteerThresholds(), **kwargs
    )


def test_admission_passes_when_gate_learns() -> None:
    assert _assess(_aggregate()).admitted


def test_admission_blocks_when_not_converged() -> None:
    admission = _assess(_aggregate(convergence_improvement_nll_mean=0.05))
    assert not admission.admitted
    assert "convergence" in admission.failed_conditions


def test_admission_blocks_when_not_beating_always_on() -> None:
    admission = _assess(
        _aggregate(
            pe_gated_online_nll_mean=1.75,
            gain_vs_always_on_ci_lower_min=-0.1,
        )
    )
    assert not admission.admitted
    assert "gain-vs-always-on" in admission.failed_conditions


def test_admission_blocks_when_gate_not_selective() -> None:
    admission = _assess(_aggregate(gate_selectivity_mean=0.05))
    assert not admission.admitted
    assert "gate-selectivity" in admission.failed_conditions


def test_admission_blocks_when_substrate_trained() -> None:
    admission = _assess(_aggregate(), substrate_trainable_parameter_count=4)
    assert not admission.admitted
    assert "structural-integrity" in admission.failed_conditions
