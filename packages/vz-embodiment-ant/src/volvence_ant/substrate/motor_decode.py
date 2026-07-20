"""``motor_decode`` — the second frozen substrate function.

Maps the kernel controller code ``z_t`` (``ControllerState.code``, length n_z)
into a physical ``(turn_command, step_command)`` and the efference copy used to
advance path integration. Pure deterministic algebra, no learnable parameters.

Contract on ``z_t`` (egocentric abstract action):

- ``z[0], z[1]`` -> non-negative opponent-coded steering evidence
  (right/forward vs left/forward). The frozen plant converts it to an
  egocentric residual vector ``forward = 1 + z0 + z1``,
  ``left = z1 - z0``. The fixed forward baseline makes near-zero latent codes
  produce near-zero turns instead of ±45° circles. Equal channels mean
  straight; either channel can dominate, so a controller whose latent is
  bounded to ``[0, 1]`` can still turn both left and right. The historical
  direct ``atan2(z1, z0)`` mapping made non-negative controllers structurally
  incapable of right turns.
- ``z[2]`` (if present) -> desired speed via a squashing function.

The controller is free to learn any mapping from sensory features to this
egocentric action; motor_decode only converts it to a bounded command. This
keeps the *policy* in the learnable kernel and the *plant* frozen here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPlan:
    turn_command: float
    step_command: float
    desired_egocentric_angle: float
    desired_speed_unit: float


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return exp_v / (1.0 + exp_v)


def motor_decode(
    code: tuple[float, ...],
    *,
    max_turn_rate: float,
    step_size: float,
    code_gain: float = 4.0,
) -> MotorPlan:
    """Convert an egocentric ``z_t`` into a bounded motor command.

    ``code_gain`` scales the (typically small) controller code so that early,
    near-zero codes still produce meaningful turns; it is a fixed plant
    constant, not a learned parameter.
    """

    if not code:
        return MotorPlan(
            turn_command=0.0,
            step_command=step_size,
            desired_egocentric_angle=0.0,
            desired_speed_unit=1.0,
        )
    right_evidence = code[0] * code_gain
    left_evidence = code[1] * code_gain if len(code) > 1 else right_evidence
    zx = 1.0 + right_evidence + left_evidence
    zy = left_evidence - right_evidence
    norm = math.hypot(zx, zy)
    if norm < 1e-6:
        desired_angle = 0.0
    else:
        desired_angle = math.atan2(zy, zx)
    if len(code) > 2:
        speed_unit = _sigmoid(code[2] * code_gain)
    else:
        speed_unit = 1.0

    turn_command = max(-max_turn_rate, min(max_turn_rate, desired_angle))
    step_command = speed_unit * step_size
    return MotorPlan(
        turn_command=turn_command,
        step_command=step_command,
        desired_egocentric_angle=desired_angle,
        desired_speed_unit=speed_unit,
    )
