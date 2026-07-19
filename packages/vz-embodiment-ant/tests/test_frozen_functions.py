"""Tests for the two frozen substrate functions + navigator."""

from __future__ import annotations

import math

import numpy as np

from volvence_ant.env.ant_world import WorldObservation
from volvence_ant.substrate.motor_decode import motor_decode
from volvence_ant.substrate.navigator import AntNavigator, wrap_angle
from volvence_ant.substrate.sense_encode import (
    SENSE_CHANNELS,
    sense_encode,
    sense_to_drives,
)


def _obs(**overrides) -> WorldObservation:
    base = dict(
        food_left=0.2,
        food_right=0.1,
        home_pher_left=0.0,
        home_pher_right=0.0,
        trail_pher_left=0.0,
        trail_pher_right=0.0,
        last_turn_command=0.0,
        carrying_food=False,
        at_nest=False,
        at_food=False,
        food_center=0.15,
        alarm=0.0,
        eval_home_bearing=0.0,
        eval_home_distance=3.0,
        eval_true_heading=0.0,
    )
    base.update(overrides)
    return WorldObservation(**base)


def test_sense_encode_shape_and_determinism() -> None:
    nav = AntNavigator(step_size=0.4, heading_noise=0.0, step_noise=0.0)
    nav.reset(initial_heading=0.0)
    state = nav.state
    v1 = sense_encode(_obs(), state, turn_command_scale=0.785)
    v2 = sense_encode(_obs(), state, turn_command_scale=0.785)
    assert v1.shape == (len(SENSE_CHANNELS),)
    assert np.allclose(v1, v2)  # frozen == deterministic


def test_sense_encode_food_diff_channel() -> None:
    nav = AntNavigator(step_size=0.4, heading_noise=0.0, step_noise=0.0)
    nav.reset()
    v = sense_encode(_obs(food_left=0.5, food_right=0.1), nav.state, turn_command_scale=0.785)
    idx = SENSE_CHANNELS.index("food_diff")
    assert math.isclose(v[idx], 0.4, rel_tol=1e-9)


def test_motor_decode_bounds() -> None:
    plan = motor_decode((10.0, 10.0, 5.0), max_turn_rate=0.785, step_size=0.4)
    assert -0.785 <= plan.turn_command <= 0.785
    assert 0.0 <= plan.step_command <= 0.4


def test_motor_decode_empty_code_goes_straight() -> None:
    plan = motor_decode((), max_turn_rate=0.785, step_size=0.4)
    assert plan.turn_command == 0.0
    assert plan.step_command == 0.4


def test_motor_decode_direction_sign() -> None:
    left = motor_decode((0.1, 0.5), max_turn_rate=1.5, step_size=0.4)
    right = motor_decode((0.1, -0.5), max_turn_rate=1.5, step_size=0.4)
    assert left.turn_command > 0  # positive y -> turn left
    assert right.turn_command < 0


def test_navigator_path_integration_roundtrip() -> None:
    """Walk out along +x, the home vector should point back along -x."""

    nav = AntNavigator(step_size=1.0, heading_noise=0.0, step_noise=0.0)
    nav.reset(initial_heading=0.0)
    for _ in range(5):
        nav.update(turn_command=0.0, step_command=1.0)
    state = nav.state
    # moved +5 in x, so home vector (to nest) is about -5 in x
    assert state.home_dx < -4.5
    assert abs(state.home_dy) < 1e-6
    assert math.isclose(state.home_distance, 5.0, rel_tol=1e-6)


def test_wrap_angle_range() -> None:
    for angle in (-10.0, -math.pi, 0.0, math.pi, 10.0):
        wrapped = wrap_angle(angle)
        assert -math.pi <= wrapped <= math.pi


def test_drives_in_unit_range() -> None:
    nav = AntNavigator(step_size=0.4)
    nav.reset()
    drives = sense_to_drives(_obs(carrying_food=True, food_center=0.8), nav.state)
    for value in (
        drives.forage_pull,
        drives.homing_pull,
        drives.alarm_pull,
        drives.explore_pull,
        drives.commit_pull,
    ):
        assert 0.0 <= value <= 1.0
