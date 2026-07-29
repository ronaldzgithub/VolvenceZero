from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.agent.gate6_meta_init_evidence import (
    GATE6_DIAGNOSTIC_ARMS,
    GATE6_PRIMARY_ARMS,
    Gate6EpisodeMetrics,
    compare_gate6_arms,
)
from volvence_zero.memory import CMSMemoryCore, MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(
        learned_core=CMSMemoryCore(
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
    )


def test_owner_initialization_controls_preserve_slow_and_parameters() -> None:
    store = _store()
    for step in range(12):
        store.observe_replay_signal(
            signal=(0.04, 0.12, 0.01, 0.0),
            timestamp_ms=step,
        )
    checkpoint = store.create_checkpoint(checkpoint_id="trained")
    assert checkpoint.cms_state is not None

    meta = store.initialize_nested_context_for_evidence(
        mode="meta-init",
        reason="test-meta",
        timestamp_ms=20,
    )
    assert meta is not None
    assert meta.slow_state_unchanged
    assert meta.parameter_state_unchanged

    store.restore_checkpoint(checkpoint)
    no_init = store.initialize_nested_context_for_evidence(
        mode="no-init",
        reason="test-zero",
        timestamp_ms=21,
    )
    assert no_init is not None
    assert no_init.online_after == (0.0, 0.0, 0.0, 0.0)
    assert no_init.session_after == (0.0, 0.0, 0.0, 0.0)

    store.restore_checkpoint(checkpoint)
    first_random = store.initialize_nested_context_for_evidence(
        mode="random-init",
        reason="test-random",
        timestamp_ms=22,
        random_seed=401,
    )
    store.restore_checkpoint(checkpoint)
    second_random = store.initialize_nested_context_for_evidence(
        mode="random-init",
        reason="test-random-repeat",
        timestamp_ms=23,
        random_seed=401,
    )
    assert first_random == second_random
    assert first_random is not None
    assert all(-0.125 <= value <= 0.125 for value in first_random.online_target)
    assert all(0.0 <= value <= 0.125 for value in first_random.online_after)

    store.restore_checkpoint(checkpoint)
    copied = store.initialize_nested_context_for_evidence(
        mode="copy-init",
        reason="test-copy",
        timestamp_ms=24,
    )
    assert copied is not None
    assert copied.online_after == copied.online_before
    assert copied.session_after == copied.session_before

    with pytest.raises(ValueError, match="random_seed"):
        store.initialize_nested_context_for_evidence(
            mode="random-init",
            reason="missing-seed",
            timestamp_ms=25,
        )
    with pytest.raises(ValueError, match="finite"):
        store.initialize_nested_context_for_evidence(
            mode="external-meta-init",
            reason="bad-external",
            timestamp_ms=26,
            external_targets=(
                (0.0, 0.0, float("nan"), 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
        )


def _metric(
    *,
    seed: int,
    arm: str,
    steps: int,
    auc: float,
    final_error: float,
) -> Gate6EpisodeMetrics:
    return Gate6EpisodeMetrics(
        seed=seed,
        partition="trace-locked-confirmation",
        context_id=f"locked-{seed}",
        user_id=f"user-{seed}",
        arm=arm,
        episode_length=20,
        steps_to_target=steps,
        early_adaptation_auc=auc,
        final_error=final_error,
        final_quality=1.0 - final_error,
        initial_error=0.04,
        initialization_changed_fast_state=arm != "copy-init",
        slow_state_unchanged=True,
        parameter_state_unchanged=True,
        lineage_complete=True,
        frozen_substrate_mutation_count=0,
        fact_leakage_count=0,
        checkpoint_restore_exact=True,
    )


def test_gate6_comparison_requires_effect_noninferiority_and_zero_transfer() -> None:
    metrics: list[Gate6EpisodeMetrics] = []
    for seed in (401, 409, 419):
        metrics.append(
            _metric(
                seed=seed,
                arm="meta-init",
                steps=2,
                auc=0.90,
                final_error=0.01,
            )
        )
        for arm in GATE6_PRIMARY_ARMS[1:]:
            metrics.append(
                _metric(
                    seed=seed,
                    arm=arm,
                    steps=5,
                    auc=0.80,
                    final_error=0.012,
                )
            )
        for arm in GATE6_DIAGNOSTIC_ARMS:
            metrics.append(
                _metric(
                    seed=seed,
                    arm=arm,
                    steps=4,
                    auc=0.82,
                    final_error=0.012,
                )
            )
    comparisons, gates, diagnostic = compare_gate6_arms(metrics)
    assert len(comparisons) == 3
    assert gates["meta_minimum_effect_all_controls"]
    assert gates["meta_final_error_non_inferior_all_controls"]
    assert gates["negative_transfer_zero"]
    assert diagnostic["user_related_prior_supported"] is False

    regressed = [
        replace(metric, final_error=0.04)
        if metric.arm == "meta-init" and metric.seed == 401
        else metric
        for metric in metrics
    ]
    _, failed, _ = compare_gate6_arms(regressed)
    assert not failed["meta_final_error_non_inferior_all_controls"]
    assert not failed["negative_transfer_zero"]
