"""``sense_encode`` — the first frozen substrate function.

Maps a :class:`WorldObservation` + body-side :class:`NavigatorState` into a
fixed-width sensory vector. Pure deterministic algebra, no learnable
parameters: this is the literal meaning of "frozen substrate" here, mirroring
a real ant's genetically fixed receptor->glomerulus map.

The same vector is published two ways by :class:`AntSubstrateAdapter`:

- as ``residual_activations`` (layer 0) so the kernel temporal encoder sees a
  low-dimensional embodiment "residual", and
- projected onto the generic ``semantic_*_pull`` feature names so the kernel
  prediction-error owner runs unchanged over embodiment-native drives
  (see ``docs/specs/digital-ant-embodiment.md`` for why we reuse the generic
  names instead of changing a core contract).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from volvence_ant.env.ant_world import WorldObservation
from volvence_ant.substrate.navigator import NavigatorState, wrap_angle


class AntSenseSchema(str, Enum):
    V1 = "ant-sense.v1"
    ECOLOGY_V2 = "ant-sense.ecology-v2"


#: Ordered v1 sensory channels.  Existing evidence lanes remain pinned here.
SENSE_CHANNELS_V1: tuple[str, ...] = (
    "food_left",
    "food_right",
    "food_diff",
    "home_ego_cos",
    "home_ego_sin",
    "home_distance_norm",
    "carrying_food",
    "home_pher_diff",
    "trail_pher_diff",
    "last_turn_command",
    "alarm",
    "obstacle_left",
    "obstacle_right",
    "obstacle_contact",
)

#: Ecology adds local thermal receptors without changing the v1 vector.
SENSE_CHANNELS_ECOLOGY_V2: tuple[str, ...] = (
    *SENSE_CHANNELS_V1,
    "heat_left",
    "heat_right",
    "heat_diff",
    "heat_center",
    "heat_harmful",
)

# Compatibility name used by existing v1 tests and consumers.
SENSE_CHANNELS = SENSE_CHANNELS_V1

_HOME_DISTANCE_SCALE = 10.0


@dataclass(frozen=True)
class AntDrives:
    """Embodiment-native drive readouts, published under the generic
    ``semantic_*_pull`` names so the kernel PE owner runs unchanged."""

    forage_pull: float  # -> semantic_task_pull
    homing_pull: float  # -> semantic_support_pull
    alarm_pull: float  # -> semantic_repair_pull
    explore_pull: float  # -> semantic_exploration_pull
    commit_pull: float  # -> semantic_directive_pull


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def sense_encode(
    observation: WorldObservation,
    navigator_state: NavigatorState,
    *,
    turn_command_scale: float,
    schema: AntSenseSchema = AntSenseSchema.V1,
) -> np.ndarray:
    """Return one versioned, frozen sensory vector."""

    rel_home = wrap_angle(navigator_state.home_bearing - navigator_state.h_hat)
    home_ego_cos = math.cos(rel_home)
    home_ego_sin = math.sin(rel_home)
    home_distance_norm = math.tanh(navigator_state.home_distance / _HOME_DISTANCE_SCALE)
    last_turn_norm = observation.last_turn_command / turn_command_scale if turn_command_scale > 0 else 0.0
    values = [
        observation.food_left,
        observation.food_right,
        observation.food_left - observation.food_right,
        home_ego_cos,
        home_ego_sin,
        home_distance_norm,
        1.0 if observation.carrying_food else 0.0,
        observation.home_pher_left - observation.home_pher_right,
        observation.trail_pher_left - observation.trail_pher_right,
        float(np.clip(last_turn_norm, -1.0, 1.0)),
        _clamp_unit(observation.alarm),
        _clamp_unit(observation.obstacle_left),
        _clamp_unit(observation.obstacle_right),
        1.0 if observation.obstacle_contact else 0.0,
    ]
    if schema is AntSenseSchema.ECOLOGY_V2:
        values.extend(
            (
                observation.heat_left,
                observation.heat_right,
                observation.heat_left - observation.heat_right,
                observation.heat_center,
                1.0 if observation.heat_harmful else 0.0,
            )
        )
    elif schema is not AntSenseSchema.V1:
        raise ValueError(f"unsupported ant sense schema: {schema!r}")
    vector = np.array(values, dtype=float)
    return vector


def sense_channels(schema: AntSenseSchema) -> tuple[str, ...]:
    if schema is AntSenseSchema.V1:
        return SENSE_CHANNELS_V1
    if schema is AntSenseSchema.ECOLOGY_V2:
        return SENSE_CHANNELS_ECOLOGY_V2
    raise ValueError(f"unsupported ant sense schema: {schema!r}")


def sense_mirror_transform(
    schema: AntSenseSchema,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Publish the frozen left/right reflection of one sense schema.

    The permutation maps each mirrored output coordinate to its source input
    coordinate. ``signs`` marks pseudoscalars whose orientation reverses.
    Applying the returned transform twice is exactly the identity.
    """

    channels = sense_channels(schema)
    reflected: dict[str, tuple[str, int]] = {
        "food_left": ("food_right", 1),
        "food_right": ("food_left", 1),
        "food_diff": ("food_diff", -1),
        "home_ego_cos": ("home_ego_cos", 1),
        "home_ego_sin": ("home_ego_sin", -1),
        "home_distance_norm": ("home_distance_norm", 1),
        "carrying_food": ("carrying_food", 1),
        "home_pher_diff": ("home_pher_diff", -1),
        "trail_pher_diff": ("trail_pher_diff", -1),
        "last_turn_command": ("last_turn_command", -1),
        "alarm": ("alarm", 1),
        "obstacle_left": ("obstacle_right", 1),
        "obstacle_right": ("obstacle_left", 1),
        "obstacle_contact": ("obstacle_contact", 1),
        "heat_left": ("heat_right", 1),
        "heat_right": ("heat_left", 1),
        "heat_diff": ("heat_diff", -1),
        "heat_center": ("heat_center", 1),
        "heat_harmful": ("heat_harmful", 1),
    }
    missing = tuple(channel for channel in channels if channel not in reflected)
    if missing:
        raise ValueError(
            f"ant sense mirror transform is missing channels: {missing!r}"
        )
    index_by_name = {name: index for index, name in enumerate(channels)}
    permutation = tuple(
        index_by_name[reflected[channel][0]]
        for channel in channels
    )
    signs = tuple(reflected[channel][1] for channel in channels)
    return permutation, signs


def sense_to_drives(observation: WorldObservation, navigator_state: NavigatorState) -> AntDrives:
    """Project the situation onto embodiment-native drives in [0, 1].

    These are deliberately *not* hardcoded behaviour rules — they are a frozen
    readout of the current situation (how much food is nearby, how far from
    home, whether alarmed). The controller learns what to DO with them; these
    only give the PE owner something to predict turn-to-turn.
    """

    food_center = observation.food_center
    forage_pull = _clamp_unit(food_center)
    # homing urgency rises when carrying food and far from the nest
    homing_pull = _clamp_unit(
        (1.0 if observation.carrying_food else 0.2) * math.tanh(navigator_state.home_distance / _HOME_DISTANCE_SCALE)
    )
    alarm_pull = _clamp_unit(max(observation.alarm, observation.heat_center))
    # exploration drive is high when there is little to sense
    explore_pull = _clamp_unit(1.0 - food_center)
    # commit drive: how sharp the local food gradient is (worth committing to)
    commit_pull = _clamp_unit(abs(observation.food_left - observation.food_right) * 4.0)
    return AntDrives(
        forage_pull=forage_pull,
        homing_pull=homing_pull,
        alarm_pull=alarm_pull,
        explore_pull=explore_pull,
        commit_pull=commit_pull,
    )
