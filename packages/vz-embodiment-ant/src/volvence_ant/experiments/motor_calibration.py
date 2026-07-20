"""Hidden motor-distortion adaptation experiment.

The task is deliberately not foraging. A body must preserve its initial
sky-compass heading while its actuator applies an unknown turn bias that flips
sign halfway through the episode. The environment publishes only observable
heading-stability and motor-execution facts through ``EnvironmentOutcome``;
it never tells the controller which compensating command to emit.

The matched arms share substrate, trace budget, PE/SSL schedule, reflection
writeback and the reward->code bridge. ``no-optimize`` differs only by restoring
the post-SSL/pre-RL checkpoint after every optimizer call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from volvence_ant.env import (
    AntWorld,
    AntWorldConfig,
    MotorDistortionProfile,
)
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSession,
    AntSessionConfig,
)


@dataclass(frozen=True)
class MotorCalibrationArm:
    arm: str
    mean_error_before_switch: float
    mean_error_after_switch: float
    mean_error_late: float
    recovery: float
    mean_motor_execution_error: float
    initial_checkpoint_fingerprint: str
    trained_checkpoint_fingerprint: str
    parameters_changed: bool
    heading_errors: tuple[float, ...]


@dataclass(frozen=True)
class MotorCalibrationReport:
    seed: int
    ticks: int
    switch_tick: int
    initial_turn_bias: float
    switched_turn_bias: float
    arms: tuple[MotorCalibrationArm, ...]
    learned_late_error_advantage: float
    learned_recovers_better: bool
    description: str


def _window_mean(values: tuple[float, ...], start: int, end: int) -> float:
    window = values[max(0, start) : min(len(values), end)]
    if not window:
        raise ValueError(
            f"motor-calibration window is empty: start={start}, end={end}, "
            f"n={len(values)}"
        )
    return sum(window) / len(window)


async def _run_arm(
    *,
    arm: str,
    config: AntSessionConfig,
    ticks: int,
    switch_tick: int,
    turn_bias: float,
    switched_turn_bias: float,
    seed: int,
) -> MotorCalibrationArm:
    world = AntWorld(
        config=AntWorldConfig(
            seed=seed,
            motor_distortions=(
                MotorDistortionProfile(
                    turn_bias=turn_bias,
                    switch_tick=switch_tick,
                    switched_turn_bias=switched_turn_bias,
                ),
            ),
        )
    )
    session = AntSession(
        world,
        config=replace(
            config,
            seed=seed,
            session_id=f"motor-calibration:{arm}:{seed}",
            objective=AntObjectiveKind.HEADING_STABILITY,
        ),
    )
    initial = session.export_learning_checkpoint(checkpoint_id=f"{arm}:initial")
    records = await session.run(ticks)
    trained = session.export_learning_checkpoint(checkpoint_id=f"{arm}:trained")
    errors = tuple(record.heading_stability_error for record in records)
    execution_errors = tuple(record.motor_execution_error for record in records)
    window = max(3, min(10, ticks // 5))
    before = _window_mean(errors, switch_tick - window, switch_tick)
    after = _window_mean(errors, switch_tick, switch_tick + window)
    late = _window_mean(errors, ticks - window, ticks)
    return MotorCalibrationArm(
        arm=arm,
        mean_error_before_switch=before,
        mean_error_after_switch=after,
        mean_error_late=late,
        recovery=after - late,
        mean_motor_execution_error=sum(execution_errors) / len(execution_errors),
        initial_checkpoint_fingerprint=initial.fingerprint,
        trained_checkpoint_fingerprint=trained.fingerprint,
        parameters_changed=initial.fingerprint != trained.fingerprint,
        heading_errors=errors,
    )


async def run_motor_calibration_experiment(
    *,
    learned_config: AntSessionConfig,
    no_optimize_config: AntSessionConfig,
    ticks: int = 60,
    switch_tick: int = 30,
    turn_bias: float = 0.18,
    switched_turn_bias: float = -0.18,
    seed: int = 0,
) -> MotorCalibrationReport:
    if ticks < 12:
        raise ValueError("motor calibration requires at least 12 ticks")
    if not 3 <= switch_tick <= ticks - 3:
        raise ValueError(
            f"switch_tick must leave at least 3 ticks per phase, got {switch_tick}"
        )
    learned = await _run_arm(
        arm="learned",
        config=learned_config,
        ticks=ticks,
        switch_tick=switch_tick,
        turn_bias=turn_bias,
        switched_turn_bias=switched_turn_bias,
        seed=seed,
    )
    no_optimize = await _run_arm(
        arm="no_optimize",
        config=no_optimize_config,
        ticks=ticks,
        switch_tick=switch_tick,
        turn_bias=turn_bias,
        switched_turn_bias=switched_turn_bias,
        seed=seed,
    )
    late_advantage = no_optimize.mean_error_late - learned.mean_error_late
    recovers_better = (
        late_advantage > 0.0 and learned.recovery > no_optimize.recovery
    )
    return MotorCalibrationReport(
        seed=seed,
        ticks=ticks,
        switch_tick=switch_tick,
        initial_turn_bias=turn_bias,
        switched_turn_bias=switched_turn_bias,
        arms=(learned, no_optimize),
        learned_late_error_advantage=late_advantage,
        learned_recovers_better=recovers_better,
        description=(
            "hidden motor calibration: "
            f"learned late_error={learned.mean_error_late:.4f}, "
            f"no_optimize late_error={no_optimize.mean_error_late:.4f}, "
            f"advantage={late_advantage:.4f}, "
            f"learned_recovers_better={recovers_better}"
        ),
    )


__all__ = [
    "MotorCalibrationArm",
    "MotorCalibrationReport",
    "run_motor_calibration_experiment",
]
