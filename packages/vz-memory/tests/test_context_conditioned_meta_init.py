from __future__ import annotations

import pytest

from volvence_zero.memory import CMSMemoryCore, MemoryStore


def _core(
    *,
    conditioned: bool,
    prototype_count: int,
) -> CMSMemoryCore:
    return CMSMemoryCore(
        mode="mlp",
        d_in=4,
        d_hidden=8,
        variant="nested",
        session_cadence=2,
        background_cadence=4,
        context_conditioned_meta_init=conditioned,
        context_prototype_count=prototype_count,
    )


def _train_context(
    core: CMSMemoryCore,
    *,
    context: tuple[float, ...],
    target: tuple[float, ...],
    start: int,
) -> None:
    for step in range(12):
        core.observe_replay_signal(
            signal=target,
            context_signal=context,
            timestamp_ms=start + step,
        )


def test_conditioned_prototypes_are_stable_and_checkpointed() -> None:
    core = _core(conditioned=True, prototype_count=2)
    context_a = (1.0, 0.0, 0.0, 0.0)
    context_b = (0.0, 1.0, 0.0, 0.0)
    target_a = (0.8, 0.1, 0.1, 0.2)
    target_b = (0.1, 0.8, 0.2, 0.1)
    _train_context(
        core,
        context=context_a,
        target=target_a,
        start=0,
    )
    first_a = core.nested_reset_targets(context_signal=context_a)
    assert first_a is not None
    _train_context(
        core,
        context=context_b,
        target=target_b,
        start=20,
    )
    second_a = core.nested_reset_targets(context_signal=context_a)
    selected_b = core.nested_reset_targets(context_signal=context_b)
    assert second_a is not None
    assert selected_b is not None

    assert second_a[0] == pytest.approx(first_a[0], abs=0.01)
    assert selected_b[0] != pytest.approx(second_a[0], abs=0.05)

    checkpoint = core.export_state()
    restored = _core(conditioned=True, prototype_count=2)
    restored.restore_state(checkpoint)
    restored_a = restored.nested_reset_targets(
        context_signal=context_a
    )
    restored_b = restored.nested_reset_targets(
        context_signal=context_b
    )
    assert restored_a is not None
    assert restored_b is not None
    for actual, expected in zip(restored_a, second_a, strict=True):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(restored_b, selected_b, strict=True):
        assert actual == pytest.approx(expected)


def test_conditioning_off_with_one_slot_matches_legacy_ema() -> None:
    legacy = _core(conditioned=False, prototype_count=8)
    rollback = _core(conditioned=False, prototype_count=1)
    context = (0.2, 0.4, 0.6, 0.8)
    target = (0.7, 0.3, 0.5, 0.1)
    for step in range(8):
        for core in (legacy, rollback):
            core.observe_replay_signal(
                signal=target,
                context_signal=context,
                timestamp_ms=step,
            )

    assert rollback.export_state() == legacy.export_state()


def test_store_publishes_copy_init_loss_advantage() -> None:
    store = MemoryStore(learned_core=_core(
        conditioned=True,
        prototype_count=2,
    ))
    context = (1.0, 0.0, 0.0, 0.0)
    target = (0.9, 0.1, 0.2, 0.1)
    for step in range(12):
        store.observe_replay_signal(
            signal=target,
            context_signal=context,
            timestamp_ms=step,
        )
    store.initialize_nested_context_for_evidence(
        mode="meta-init",
        reason="conditioned-test",
        timestamp_ms=20,
        context_signal=context,
    )
    for step in range(5):
        store.observe_replay_signal(
            signal=target,
            context_signal=context,
            timestamp_ms=21 + step,
        )
    metrics = dict(store.snapshot(
        retrieved_entries=(),
        active_subject_scope=(),
    ).lifecycle_metrics)

    assert metrics["slow_to_fast_init_benefit"] != 0.0
