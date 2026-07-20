"""Contract tests for the hidden motor-calibration experiment."""

from __future__ import annotations

import volvence_ant.experiments.motor_calibration as calibration
from volvence_ant.runtime import AntSessionConfig


def _arm(
    name: str, *, late: float, recovery: float
) -> calibration.MotorCalibrationArm:
    return calibration.MotorCalibrationArm(
        arm=name,
        mean_error_before_switch=0.1,
        mean_error_after_switch=0.3,
        mean_error_late=late,
        recovery=recovery,
        mean_motor_execution_error=0.2,
        initial_checkpoint_fingerprint=f"{name}:initial",
        trained_checkpoint_fingerprint=f"{name}:trained",
        parameters_changed=True,
        heading_errors=(0.1,) * 12,
    )


async def test_motor_calibration_gate_rejects_noise_sized_advantage(
    monkeypatch,
) -> None:
    arms = iter(
        (
            _arm("learned", late=0.4997, recovery=0.0003),
            _arm("no_optimize", late=0.5000, recovery=0.0000),
        )
    )

    async def fake_run_arm(**_kwargs):
        return next(arms)

    monkeypatch.setattr(calibration, "_run_arm", fake_run_arm)
    report = await calibration.run_motor_calibration_experiment(
        learned_config=AntSessionConfig(),
        no_optimize_config=AntSessionConfig(),
        ticks=12,
        switch_tick=6,
    )
    assert report.learned_late_error_advantage > 0.0
    assert report.learned_recovers_better is False


async def test_motor_calibration_gate_accepts_predeclared_effect_size(
    monkeypatch,
) -> None:
    arms = iter(
        (
            _arm("learned", late=0.45, recovery=0.08),
            _arm("no_optimize", late=0.50, recovery=0.02),
        )
    )

    async def fake_run_arm(**_kwargs):
        return next(arms)

    monkeypatch.setattr(calibration, "_run_arm", fake_run_arm)
    report = await calibration.run_motor_calibration_experiment(
        learned_config=AntSessionConfig(),
        no_optimize_config=AntSessionConfig(),
        ticks=12,
        switch_tick=6,
    )
    assert report.learned_late_error_advantage >= 0.02
    assert report.learned_recovery_advantage >= 0.01
    assert report.learned_recovers_better is True
