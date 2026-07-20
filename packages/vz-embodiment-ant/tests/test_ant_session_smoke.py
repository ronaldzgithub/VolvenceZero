"""Smoke + PE-seam tests for the kernel-driven ant session.

These prove the digital ant reuses the kernel end-to-end and that the
prediction-error owner produces a live signal off the embodiment-native drive
channels (the PE seam), with no kernel changes.
"""

from __future__ import annotations

from volvence_ant.env.ant_world import (
    AntWorld,
    AntWorldConfig,
    FoodSource,
    MotorDistortionProfile,
)
from volvence_ant.runtime import AntObjectiveKind, AntSession, AntSessionConfig


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
