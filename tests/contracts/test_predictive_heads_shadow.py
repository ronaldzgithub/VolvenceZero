"""CP-11 world/self predictive heads SHADOW tests.

Proves the learned heads dual-run INSIDE the single PE owner (no new slot),
publish a report-only readout with baseline comparison, learn online, and
never alter the live prediction chain.
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.prediction import (
    PredictionErrorModule,
    PredictiveHeadCheckpoint,
    PredictiveHeadCheckpointError,
    PredictiveHeadKillCriteria,
    PredictiveHeadReadout,
)


async def test_shadow_heads_publish_readout_without_touching_live_chain() -> None:
    runner = AgentSessionRunner(rare_heavy_enabled=False)

    first = await runner.run_turn("Let's map out the harbor logistics problem.")
    pe_first = first.active_snapshots["prediction_error"].value
    # Turn 1: heads have queued a forecast but scored nothing yet.
    assert pe_first.predictive_head_readout is None

    second = await runner.run_turn("The schedule slipped; I feel a bit anxious about it.")
    third = await runner.run_turn("Ok, walk me through the recovery options.")

    for result, expected_samples in ((second, 1), (third, 2)):
        pe_value = result.active_snapshots["prediction_error"].value
        readout = pe_value.predictive_head_readout
        assert isinstance(readout, PredictiveHeadReadout)
        assert readout.sample_count == expected_samples
        for mae in (
            readout.world_learned_mae,
            readout.world_baseline_mae,
            readout.self_learned_mae,
            readout.self_baseline_mae,
        ):
            assert 0.0 <= mae <= 1.0
        assert readout.world_improvement == round(
            readout.world_baseline_mae - readout.world_learned_mae, 4
        )
        # Report-only invariant: the live chain fields are still produced by
        # the hand-crafted head — next_prediction / error carry no reference
        # to the shadow heads.
        assert pe_value.next_prediction is not None
        assert "shadow" not in pe_value.error.description.lower()


def test_linear_axis_head_learns_a_constant_target() -> None:
    from volvence_zero.prediction.error import _LinearAxisHead

    head = _LinearAxisHead(feature_dim=3, learning_rate=0.2)
    features = (0.5, 0.5, 1.0)
    for _ in range(200):
        head.update(features=features, target=0.9)
    assert abs(head.predict(features) - 0.9) < 0.05


# ---------------------------------------------------------------------------
# W1.D gate completeness: session-medium checkpoint export/restore and the
# self-reward autocorrelation kill-criteria readout.
# ---------------------------------------------------------------------------


async def test_head_checkpoint_round_trip_after_live_turns() -> None:
    runner = AgentSessionRunner(rare_heavy_enabled=False)
    await runner.run_turn("Let's map out the harbor logistics problem.")
    await runner.run_turn("The schedule slipped; I feel a bit anxious about it.")
    await runner.run_turn("Ok, walk me through the recovery options.")

    checkpoint = runner.prediction_module.export_predictive_head_checkpoint(
        checkpoint_id="test-ckpt"
    )
    assert isinstance(checkpoint, PredictiveHeadCheckpoint)
    assert checkpoint.sample_count == 2  # turns 2 and 3 scored a forecast

    fresh = PredictionErrorModule()
    fresh.restore_predictive_head_checkpoint(checkpoint)
    reexported = fresh.export_predictive_head_checkpoint(checkpoint_id="test-ckpt")
    assert reexported == checkpoint


def test_head_checkpoint_restore_fails_loudly_on_mismatch() -> None:
    module = PredictionErrorModule()
    checkpoint = module.export_predictive_head_checkpoint(checkpoint_id="base")

    with pytest.raises(PredictiveHeadCheckpointError, match="schema_version"):
        module.restore_predictive_head_checkpoint(
            dataclasses.replace(checkpoint, schema_version="bogus.v0")
        )
    with pytest.raises(PredictiveHeadCheckpointError, match="feature_dim"):
        module.restore_predictive_head_checkpoint(
            dataclasses.replace(checkpoint, feature_dim=checkpoint.feature_dim + 1)
        )
    with pytest.raises(PredictiveHeadCheckpointError, match="axis set mismatch"):
        module.restore_predictive_head_checkpoint(
            dataclasses.replace(
                checkpoint,
                world_weights=(("bogus_axis", checkpoint.world_weights[0][1]),),
            )
        )


def test_kill_criteria_cold_start_is_not_triggered() -> None:
    module = PredictionErrorModule()
    kill = module.predictive_head_kill_criteria()
    assert isinstance(kill, PredictiveHeadKillCriteria)
    assert kill.samples_in_window == 0
    assert not kill.window_filled
    assert not kill.kill_triggered


def test_kill_criteria_flags_self_reward_loop() -> None:
    # Frozen prediction series vs moving targets: the head is echoing itself,
    # not tracking realized outcomes. Window must be full to trigger.
    module = PredictionErrorModule()
    for index in range(module.predictive_head_kill_criteria().window_size):
        module._head_recent_pairs.append((0.5, 0.1 + 0.8 * (index % 10) / 10))
    kill = module.predictive_head_kill_criteria()
    assert kill.window_filled
    assert kill.prediction_self_autocorrelation >= 0.98
    assert kill.prediction_target_correlation <= 0.05
    assert kill.kill_triggered


def test_kill_criteria_stays_quiet_when_head_tracks_targets() -> None:
    module = PredictionErrorModule()
    for index in range(module.predictive_head_kill_criteria().window_size):
        target = 0.1 + 0.8 * (index % 10) / 10
        module._head_recent_pairs.append((target + 0.02, target))
    kill = module.predictive_head_kill_criteria()
    assert kill.window_filled
    assert kill.prediction_target_correlation > 0.9
    assert not kill.kill_triggered
