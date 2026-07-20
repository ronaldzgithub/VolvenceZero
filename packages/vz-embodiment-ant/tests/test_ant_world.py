"""Tests for the 2D ant world (food field + motion + contacts)."""

from __future__ import annotations

import math

from volvence_ant.env.ant_world import (
    AntWorld,
    AntWorldConfig,
    FoodSource,
    MotorDistortionProfile,
)


def test_food_field_peaks_at_source() -> None:
    world = AntWorld(food_sources=(FoodSource(x=5.0, y=0.0, strength=1.0, decay=4.0),))
    at_source = world.food_intensity(5.0, 0.0)
    far = world.food_intensity(-5.0, 0.0)
    assert at_source > far
    assert math.isclose(at_source, 1.0, rel_tol=1e-6)  # exp(0) * strength


def test_food_field_superposition() -> None:
    world = AntWorld(
        food_sources=(
            FoodSource(x=3.0, y=0.0, strength=1.0, decay=4.0),
            FoodSource(x=-3.0, y=0.0, strength=1.0, decay=4.0),
        )
    )
    center = world.food_intensity(0.0, 0.0)
    single = math.exp(-3.0 / 4.0)
    assert math.isclose(center, 2.0 * single, rel_tol=1e-6)


def test_act_moves_body_and_updates_tick() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0, step_size=0.5, max_turn_rate=1.0))
    start = world.body(0)
    world.act(turn_command=0.0, step_command=0.5, body_id=0)
    end = world.body(0)
    assert world.tick == 1
    moved = math.hypot(end.x - start.x, end.y - start.y)
    assert math.isclose(moved, 0.5, rel_tol=1e-6)


def test_pickup_and_deposit_cycle() -> None:
    cfg = AntWorldConfig(
        nest_x=0.0, nest_y=0.0, nest_radius=1.0, step_size=1.0, max_turn_rate=math.pi
    )
    world = AntWorld(
        config=cfg,
        food_sources=(FoodSource(x=2.0, y=0.0, strength=1.0, decay=4.0, radius=1.5),),
    )
    # place body next to food by walking east
    world._bodies[0] = world.body(0).__class__(x=2.0, y=0.0, heading=0.0, carrying_food=False)
    world.act(turn_command=0.0, step_command=0.0, body_id=0)
    assert world.body(0).carrying_food is True
    assert world.food_pickups == 1
    # walk back to nest
    world._bodies[0] = world.body(0).__class__(x=0.0, y=0.0, heading=0.0, carrying_food=True)
    world.act(turn_command=0.0, step_command=0.0, body_id=0)
    assert world.body(0).carrying_food is False
    assert world.food_delivered == 1


def test_alarm_trigger_and_decay() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0))
    world.trigger_alarm(magnitude=1.0)
    obs = world.observe(0)
    assert obs.alarm == 1.0
    for _ in range(5):
        world.act(turn_command=0.0, step_command=0.1, body_id=0)
    assert world.observe(0).alarm == 0.0  # decayed away


def test_observation_eval_fields_present() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0))
    obs = world.observe(0)
    assert obs.eval_home_distance >= 0.0
    assert -math.pi <= obs.eval_home_bearing <= math.pi


def test_motor_distortion_is_hidden_inside_world_transition() -> None:
    profile = MotorDistortionProfile(
        turn_gain=1.0,
        turn_bias=0.2,
        switch_tick=1,
        switched_turn_gain=1.0,
        switched_turn_bias=-0.2,
    )
    world = AntWorld(
        config=AntWorldConfig(
            seed=0,
            max_turn_rate=1.0,
            motor_distortions=(profile,),
        )
    )
    world.act(turn_command=0.0, step_command=0.0)
    first = world.last_transition()
    world.act(turn_command=0.0, step_command=0.0)
    second = world.last_transition()

    assert first.commanded_turn == 0.0
    assert math.isclose(first.applied_turn, 0.2)
    assert math.isclose(second.applied_turn, -0.2)
    assert math.isclose(world.observe().last_turn_command, -0.2)


def test_default_motor_transfer_is_identity() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0, max_turn_rate=1.0))
    world.act(turn_command=0.3, step_command=0.0)
    transition = world.last_transition()
    assert transition.commanded_turn == transition.applied_turn == 0.3


def test_runtime_motor_disturbance_changes_plant_not_observation_schema() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0, max_turn_rate=1.0))
    before_fields = tuple(world.observe().__dataclass_fields__)
    world.set_motor_distortion(
        MotorDistortionProfile(turn_gain=0.5, turn_bias=-0.2)
    )
    world.act(turn_command=0.4, step_command=0.0)
    transition = world.last_transition()

    assert math.isclose(transition.commanded_turn, 0.4)
    assert math.isclose(transition.applied_turn, 0.0)
    assert tuple(world.observe().__dataclass_fields__) == before_fields
    assert "turn_gain" not in before_fields
    assert "turn_bias" not in before_fields
