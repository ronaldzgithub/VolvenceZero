"""``AntActuator`` — reads the kernel controller code ``z_t`` and drives the plant.

The actuator is the motor half of the frozen substrate. It never learns; it
converts ``ControllerState.code`` into a bounded command via ``motor_decode``.
The kernel's temporal owner is the only thing that decides ``z_t``.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_ant.substrate.motor_decode import MotorPlan, motor_decode


@dataclass(frozen=True)
class AntMotorCommand:
    turn_command: float
    step_command: float
    desired_egocentric_angle: float
    desired_speed_unit: float
    code: tuple[float, ...]


class AntActuator:
    """Frozen motor plant: ``z_t`` -> bounded (turn, step) command.

    Includes a hardwired escape reflex (the digital analogue of an insect giant
    fibre): when the alarm channel exceeds ``alarm_threshold``, the reflex
    overrides ``z_t`` with a fixed full-speed straight flee at a fixed latency,
    no matter what the learned controller wants. This is the "one-vote-veto"
    safety gate (Workstream G4) and is never learned or bypassed.
    """

    def __init__(
        self,
        *,
        max_turn_rate: float,
        step_size: float,
        code_gain: float = 4.0,
        alarm_threshold: float = 0.5,
    ) -> None:
        self._max_turn_rate = max_turn_rate
        self._step_size = step_size
        self._code_gain = code_gain
        self._alarm_threshold = alarm_threshold

    def plan(self, code: tuple[float, ...], *, alarm: float = 0.0) -> AntMotorCommand:
        if alarm > self._alarm_threshold:
            return self._escape_reflex(code)
        motor_plan: MotorPlan = motor_decode(
            code,
            max_turn_rate=self._max_turn_rate,
            step_size=self._step_size,
            code_gain=self._code_gain,
        )
        return AntMotorCommand(
            turn_command=motor_plan.turn_command,
            step_command=motor_plan.step_command,
            desired_egocentric_angle=motor_plan.desired_egocentric_angle,
            desired_speed_unit=motor_plan.desired_speed_unit,
            code=tuple(code),
        )

    def _escape_reflex(self, code: tuple[float, ...]) -> AntMotorCommand:
        return AntMotorCommand(
            turn_command=0.0,  # straight-ahead flee, deterministic
            step_command=self._step_size,  # full speed
            desired_egocentric_angle=0.0,
            desired_speed_unit=1.0,
            code=tuple(code),
        )

    def is_reflex_triggered(self, alarm: float) -> bool:
        return alarm > self._alarm_threshold
