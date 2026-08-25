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
from volvence_ant.substrate.sense_encode import (
    SENSE_CHANNELS,
    SENSE_CHANNELS_ECOLOGY_V2,
    SENSE_CHANNELS_V1,
    AntSenseSchema,
    sense_channels,
    sense_encode,
    sense_mirror_transform,
)

__all__ = [
    "SENSE_CHANNELS",
    "SENSE_CHANNELS_ECOLOGY_V2",
    "SENSE_CHANNELS_V1",
    "AntActuator",
    "AntMotorCommand",
    "AntNavigator",
    "AntSenseSchema",
    "AntSenseHolder",
    "AntSubstrateAdapter",
    "NavigatorState",
    "motor_decode",
    "sense_channels",
    "sense_encode",
    "sense_mirror_transform",
]
