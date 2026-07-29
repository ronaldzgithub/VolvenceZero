from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.agent.gate5_cms_pareto_evidence import (
    GATE5_ARM_NAMES,
    GATE5_FULL_ARM,
    GATE5_SINGLE_TIMESCALE_ARM,
    Gate5ArmMetrics,
    build_gate5_arm_store,
    compare_gate5_arms,
)
from volvence_zero.memory import CMSMemoryCore


def _metric(
    *,
    seed: int,
    arm: str,
    absorption: float,
    retention: float,
) -> Gate5ArmMetrics:
    cadence = (
        (1, 2, 4)
        if arm == GATE5_FULL_ARM
        else (1, 1, 1)
        if arm == GATE5_SINGLE_TIMESCALE_ARM
        else ()
    )
    return Gate5ArmMetrics(
        seed=seed,
        arm=arm,
        settled_transition_count=510,
        locked_transition_count=60,
        new_knowledge_absorption=absorption,
        old_knowledge_retention=retention,
        memory_churn=0.1,
        erroneous_promotion_rate=0.0,
        retrieval_hit_rate=1.0,
        retrieval_weighted_payoff=0.2,
        cms_parameter_count=204 if arm != "memory-only" else 0,
        cadence_intervals=cadence,
        frozen_substrate_mutation_count=0,
        lineage_complete=True,
    )


def test_replay_signal_runs_complete_cms_cadence_and_fails_loudly() -> None:
    core = CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        variant="nested",
        session_cadence=2,
        background_cadence=4,
        pe_features_enabled=True,
        replay_window_sizes={
            "online-fast": 8,
            "session-medium": 4,
            "background-slow": 2,
        },
    )
    for step in range(4):
        core.observe_replay_signal(
            signal=(0.1 + step * 0.01, 0.2, 0.3, 0.4),
            timestamp_ms=step + 1,
        )
    snapshot = core.snapshot()
    assert snapshot.total_observations == 4
    assert snapshot.online_fast.cadence_interval == 1
    assert snapshot.session_medium.cadence_interval == 2
    assert snapshot.background_slow.cadence_interval == 4
    assert snapshot.background_slow.last_update_ms == 4
    assert 0.0 <= snapshot.new_knowledge_absorption <= 1.0
    assert 0.0 <= snapshot.old_knowledge_retention <= 1.0
    with pytest.raises(ValueError, match="non-empty"):
        core.observe_replay_signal(signal=(), timestamp_ms=5)
    with pytest.raises(ValueError, match="finite"):
        core.observe_replay_signal(
            signal=(0.1, float("nan")),
            timestamp_ms=5,
        )


def test_gate5_arm_matrix_freezes_cadence_and_parameter_budget() -> None:
    snapshots = {
        arm: build_gate5_arm_store(arm).snapshot(retrieved_entries=())
        for arm in GATE5_ARM_NAMES
    }
    full = snapshots[GATE5_FULL_ARM].cms_state
    single = snapshots[GATE5_SINGLE_TIMESCALE_ARM].cms_state
    assert full is not None
    assert single is not None
    assert (
        full.online_fast.cadence_interval,
        full.session_medium.cadence_interval,
        full.background_slow.cadence_interval,
    ) == (1, 2, 4)
    assert (
        single.online_fast.cadence_interval,
        single.session_medium.cadence_interval,
        single.background_slow.cadence_interval,
    ) == (1, 1, 1)
    assert sum(
        band.mlp_param_count
        for band in (
            full.online_fast,
            full.session_medium,
            full.background_slow,
        )
    ) == sum(
        band.mlp_param_count
        for band in (
            single.online_fast,
            single.session_medium,
            single.background_slow,
        )
    )
    assert snapshots["memory-only"].cms_state is None


def test_gate5_comparison_requires_pareto_and_preregistered_effect() -> None:
    metrics = []
    for seed in (401, 409, 419):
        metrics.extend(
            (
                _metric(
                    seed=seed,
                    arm=GATE5_FULL_ARM,
                    absorption=0.55,
                    retention=0.90,
                ),
                _metric(
                    seed=seed,
                    arm=GATE5_SINGLE_TIMESCALE_ARM,
                    absorption=0.50,
                    retention=0.89,
                ),
                _metric(
                    seed=seed,
                    arm="no-ATLAS-replay",
                    absorption=0.54,
                    retention=0.90,
                ),
                _metric(
                    seed=seed,
                    arm="no-PE-write-gate",
                    absorption=0.54,
                    retention=0.90,
                ),
                _metric(
                    seed=seed,
                    arm="memory-only",
                    absorption=0.0,
                    retention=0.90,
                ),
            )
        )
    comparisons, gates = compare_gate5_arms(metrics)
    assert len(comparisons) == 4
    assert gates["full_pareto_non_worse_all_controls"] is True
    assert gates["full_significant_vs_single_timescale"] is True

    regressed = [
        replace(
            metric,
            old_knowledge_retention=0.70,
        )
        if metric.arm == GATE5_FULL_ARM and metric.seed == 401
        else metric
        for metric in metrics
    ]
    _, failed_gates = compare_gate5_arms(regressed)
    assert failed_gates["full_pareto_non_worse_all_controls"] is False
