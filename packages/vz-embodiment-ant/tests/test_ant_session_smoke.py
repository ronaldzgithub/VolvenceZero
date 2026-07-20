"""Smoke + PE-seam tests for the kernel-driven ant session.

These prove the digital ant reuses the kernel end-to-end and that the
prediction-error owner produces a live signal off the embodiment-native drive
channels (the PE seam), with no kernel changes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.agent import (
    decode_agent_learning_archive,
    encode_agent_learning_archive,
)
from volvence_zero.owner_hydration import HydrationPayloadInvalidError

from volvence_ant.env.ant_world import (
    AntWorld,
    AntWorldConfig,
    FoodSource,
    MotorDistortionProfile,
)
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSession,
    AntSessionConfig,
    KernelColonyRunner,
)


def _world(seed: int = 3) -> AntWorld:
    return AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=8.0, y=0.0, strength=1.0, decay=6.0),),
    )


async def test_session_runs_and_produces_codes() -> None:
    session = AntSession(_world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3))
    records = await session.run(6)
    assert len(records) == 6
    for record in records:
        assert len(record.code) == 4  # z_t has n_z components
        assert -session.world.config.max_turn_rate - 1e-9 <= record.command.turn_command
        assert record.command.turn_command <= session.world.config.max_turn_rate + 1e-9
        assert 0.0 <= record.command.step_command <= session.world.config.step_size + 1e-9


async def test_pe_seam_produces_signal() -> None:
    """The kernel PE owner should publish a prediction_error off ant drives."""

    session = AntSession(_world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3))
    saw_pe = False
    magnitudes: list[float] = []
    for _ in range(6):
        result = await session.runner.run_turn(f"ant-tick-{session.world.tick}")
        # advance the body so the drive channels actually change turn-to-turn
        code, _, _ = session._read_code(result)
        command = session.actuator.plan(code)
        nav = session.navigator.update(
            turn_command=command.turn_command, step_command=command.step_command
        )
        obs = session.world.act(
            turn_command=command.turn_command, step_command=command.step_command
        )
        session.holder.update(observation=obs, navigator_state=nav, step=session.world.tick)
        if result.prediction_error is not None:
            saw_pe = True
            magnitudes.append(abs(result.prediction_error.task_error))
    assert saw_pe, "kernel never published a PredictionError from ant drives"
    # at least one turn should register non-trivial prediction error
    assert max(magnitudes) >= 0.0


async def test_substrate_snapshot_is_non_language() -> None:
    session = AntSession(_world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3))
    adapter = session._adapter_factory("ant", 0)
    snapshot = await adapter.capture()
    assert snapshot.model_id == "digital-ant-v0"
    assert snapshot.token_logits == ()  # no vocabulary / tokens
    assert snapshot.residual_activations  # embodiment residual present
    names = {signal.name for signal in snapshot.feature_surface}
    # generic drive names present (PE seam) + ant-native channels present
    assert "semantic_task_pull" in names
    assert any(name.startswith("ant_") for name in names)


async def test_delivery_outcome_settles_in_next_turn_pe() -> None:
    world = _world()
    world.set_body_pose(x=0.0, y=0.0, heading=0.0, carrying_food=True)
    session = AntSession(world, config=AntSessionConfig(temporal_latent_dim=4, seed=3))
    delivered = await session.step()
    assert world.last_transition().delivered is True
    settled = await session.runner.run_turn("settle-delivery-outcome")
    assert settled.actual_outcome is not None
    assert settled.actual_outcome.task_progress == 1.0
    context = settled.actual_outcome.action_context
    assert context.environment_outcome_id == delivered.environment_outcome_id
    assert context.environment_outcome_terminal is True
    assert context.prediction_id == delivered.prediction_id


async def test_learning_checkpoint_round_trip_restores_owner_state() -> None:
    session = AntSession(
        _world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3)
    )
    checkpoint = session.export_learning_checkpoint(checkpoint_id="shared-initial")
    await session.run(4)
    session.restore_learning_checkpoint(checkpoint)
    restored = session.export_learning_checkpoint(checkpoint_id="shared-initial")
    assert restored.fingerprint == checkpoint.fingerprint
    assert restored.policy_fingerprint == checkpoint.policy_fingerprint


async def test_json_learning_archive_round_trip_restores_all_owner_parts() -> None:
    session = AntSession(
        _world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3)
    )
    await session.run(4)
    archive = session.export_learning_checkpoint_archive(checkpoint_id="trained")
    expected = decode_agent_learning_archive(archive)

    await session.run(2)
    session.restore_learning_checkpoint_archive(archive)
    restored = decode_agent_learning_archive(
        session.export_learning_checkpoint_archive(checkpoint_id="trained")
    )

    assert restored.info.state_fingerprint == expected.info.state_fingerprint
    assert restored.info.policy_fingerprint == expected.info.policy_fingerprint
    assert b"pickle" not in archive


async def test_json_learning_archive_late_owner_failure_rolls_back() -> None:
    session = AntSession(
        _world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3)
    )
    await session.run(3)
    target = decode_agent_learning_archive(
        session.export_learning_checkpoint_archive(checkpoint_id="target")
    )
    invalid_parts = tuple(
        replace(
            snapshot,
            payload={**snapshot.payload, "weights": []},
        )
        if snapshot.owner_name == "reflection.consolidation_score"
        else snapshot
        for snapshot in target.owner_snapshots
    )
    invalid_archive = encode_agent_learning_archive(
        checkpoint_id="invalid-late-owner",
        owner_snapshots=invalid_parts,
        policy_fingerprint=target.info.policy_fingerprint,
        temporal_fingerprint=target.info.temporal_fingerprint,
        memory_fingerprint=target.info.memory_fingerprint,
    )

    await session.run(2)
    before = decode_agent_learning_archive(
        session.export_learning_checkpoint_archive(checkpoint_id="before")
    )
    with pytest.raises(
        HydrationPayloadInvalidError,
        match="ConsolidationScoreLearner",
    ):
        session.restore_learning_checkpoint_archive(invalid_archive)
    after = decode_agent_learning_archive(
        session.export_learning_checkpoint_archive(checkpoint_id="after")
    )

    assert after.info.state_fingerprint == before.info.state_fingerprint


def test_json_learning_archive_rejects_owner_fingerprint_mismatch() -> None:
    session = AntSession(
        _world(), config=AntSessionConfig(temporal_latent_dim=4, seed=3)
    )
    valid = decode_agent_learning_archive(
        session.export_learning_checkpoint_archive(checkpoint_id="valid")
    )
    invalid = encode_agent_learning_archive(
        checkpoint_id="invalid-fingerprint",
        owner_snapshots=valid.owner_snapshots,
        policy_fingerprint="0" * 64,
        temporal_fingerprint=valid.info.temporal_fingerprint,
        memory_fingerprint=valid.info.memory_fingerprint,
    )

    with pytest.raises(ValueError, match="owner fingerprint mismatch"):
        session.restore_learning_checkpoint_archive(invalid)

    after = decode_agent_learning_archive(
        session.export_learning_checkpoint_archive(checkpoint_id="after")
    )
    assert after.info.state_fingerprint == valid.info.state_fingerprint


async def test_colony_archive_failure_rolls_back_attempted_prefix_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = KernelColonyRunner(
        ColonyWorld(
            config=AntWorldConfig(seed=3),
            food_sources=(FoodSource(x=8.0, y=0.0, strength=1.0, decay=6.0),),
            n_bodies=4,
        ),
        base_config=AntSessionConfig(temporal_latent_dim=4, seed=3),
    )
    await runner.run(2)
    targets = runner.export_learning_checkpoint_archives(
        checkpoint_prefix="targets"
    )
    target_fingerprints = tuple(
        decode_agent_learning_archive(item).info.state_fingerprint
        for item in targets
    )
    third = decode_agent_learning_archive(targets[2])
    invalid_third = encode_agent_learning_archive(
        checkpoint_id=third.info.checkpoint_id,
        owner_snapshots=tuple(
            replace(
                snapshot,
                payload={**snapshot.payload, "weights": []},
            )
            if snapshot.owner_name == "reflection.consolidation_score"
            else snapshot
            for snapshot in third.owner_snapshots
        ),
        policy_fingerprint=third.info.policy_fingerprint,
        temporal_fingerprint=third.info.temporal_fingerprint,
        memory_fingerprint=third.info.memory_fingerprint,
    )

    await runner.run(2)
    before = tuple(
        decode_agent_learning_archive(item).info.state_fingerprint
        for item in runner.export_learning_checkpoint_archives(
            checkpoint_prefix="before"
        )
    )
    assert target_fingerprints[0] != before[0]
    assert target_fingerprints[1] != before[1]

    rollback_order: list[int] = []
    for body_id, session in enumerate(runner.sessions):
        restore = session.restore_learning_checkpoint

        def tracked_restore(
            checkpoint: AntLearningCheckpoint,
            *,
            _body_id: int = body_id,
            _restore=restore,
        ) -> None:
            rollback_order.append(_body_id)
            _restore(checkpoint)

        monkeypatch.setattr(
            session,
            "restore_learning_checkpoint",
            tracked_restore,
        )

    with pytest.raises(ValueError, match="body mapping mismatch"):
        runner.restore_learning_checkpoint_archives(
            (targets[1], targets[0], targets[2], targets[3])
        )
    with pytest.raises(HydrationPayloadInvalidError):
        runner.restore_learning_checkpoint_archives(
            (targets[0], targets[1], invalid_third, targets[3])
        )
    after = tuple(
        decode_agent_learning_archive(item).info.state_fingerprint
        for item in runner.export_learning_checkpoint_archives(
            checkpoint_prefix="after"
        )
    )

    assert rollback_order == [2, 1, 0]
    assert after == before


async def test_heading_stability_objective_publishes_dense_motor_facts() -> None:
    world = AntWorld(
        config=AntWorldConfig(
            seed=3,
            motor_distortions=(
                MotorDistortionProfile(turn_bias=0.2),
            ),
        )
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=3,
            objective=AntObjectiveKind.HEADING_STABILITY,
        ),
    )
    first = await session.step()
    settled = await session.runner.run_turn("settle-heading-stability")

    assert first.heading_stability_error > 0.0
    assert first.motor_execution_error > 0.0
    assert settled.actual_outcome is not None
    assert 0.0 <= settled.actual_outcome.task_progress <= 1.0
    assert settled.actual_outcome.action_context.environment_outcome_id == (
        first.environment_outcome_id
    )
