"""Digital-ant substrate: two frozen vector functions + adapter + actuator.

``sense_encode`` / ``motor_decode`` are the literal meaning of "frozen
substrate" in this prototype: deterministic numpy algebra with no learnable
parameters, mirroring how a real ant's receptor->glomerulus map and motor
plant are genetically fixed. All learning happens in the kernel controller
sitting between them.
"""

from __future__ import annotations

from volvence_ant.substrate.ant_adapter import AntSubstrateAdapter, AntSenseHolder
from volvence_ant.substrate.ant_actuator import AntActuator, AntMotorCommand
from volvence_ant.substrate.motor_decode import motor_decode
from volvence_ant.substrate.navigator import AntNavigator, NavigatorState
from volvence_ant.substrate.sense_encode import SENSE_CHANNELS, sense_encode

__all__ = [
    "SENSE_CHANNELS",
    "AntActuator",
    "AntMotorCommand",
    "AntNavigator",
    "AntSenseHolder",
    "AntSubstrateAdapter",
    "NavigatorState",
    "motor_decode",
    "sense_encode",
]
