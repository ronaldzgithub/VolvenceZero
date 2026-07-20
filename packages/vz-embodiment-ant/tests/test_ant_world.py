"""Tests for the 2D ant world (food field + motion + contacts)."""

from __future__ import annotations

import math

import pytest

from volvence_ant.env.ant_world import (
    AntWorld,
    AntWorldConfig,
    AxisAlignedObstacle,
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
    assert math.isclose(world.observe().last_turn_command, 0.0)


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


def test_obstacle_blocks_segment_without_tunneling_and_publishes_evidence() -> None:
    world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=1.0, max_turn_rate=math.pi),
        obstacles=(
            AxisAlignedObstacle(
                obstacle_id="thin-wall",
                min_x=0.4,
                max_x=0.41,
                min_y=-1.0,
                max_y=1.0,
            ),
        ),
    )
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    world.act(turn_command=0.0, step_command=1.0)

    body = world.body()
    transition = world.last_transition()
    assert body.x == pytest.approx(0.4, abs=1e-7)
    assert body.y == pytest.approx(0.0)
    assert transition.commanded_step == 1.0
    assert transition.applied_step == pytest.approx(0.4, abs=1e-7)
    assert transition.blocked_by_obstacle
    assert transition.obstacle_id == "thin-wall"
    assert world.observe().obstacle_contact


def test_obstacle_antennae_are_local_and_directional() -> None:
    world = AntWorld(
        config=AntWorldConfig(
            seed=0,
            antenna_offset_deg=30.0,
            antenna_reach=0.6,
        ),
        obstacles=(
            AxisAlignedObstacle(
                obstacle_id="left-receptor-target",
                min_x=0.45,
                max_x=0.60,
                min_y=0.20,
                max_y=0.40,
            ),
        ),
    )
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    observation = world.observe()

    assert observation.obstacle_left == 1.0
    assert observation.obstacle_right == 0.0
    assert not observation.obstacle_contact


def test_runtime_obstacle_replacement_is_atomic_and_validated() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0))
    barrier = AxisAlignedObstacle(
        obstacle_id="barrier",
        min_x=2.0,
        max_x=3.0,
        min_y=-1.0,
        max_y=1.0,
    )
    world.set_obstacles((barrier,))
    assert world.obstacles() == (barrier,)

    with pytest.raises(ValueError, match="unique"):
        world.set_obstacles((barrier, barrier))


def test_body_inside_new_obstacle_may_exit_but_not_reenter() -> None:
    obstacle = AxisAlignedObstacle(
        obstacle_id="activated-around-body",
        min_x=0.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
    )
    world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=1.0, max_turn_rate=math.pi),
        obstacles=(obstacle,),
    )
    world.set_body_pose(x=0.5, y=0.0, heading=math.pi)
    world.act(turn_command=0.0, step_command=1.0)
    assert world.body().x == pytest.approx(-0.5)
    assert not world.last_transition().blocked_by_obstacle

    world.set_body_pose(x=-0.5, y=0.0, heading=0.0)
    world.act(turn_command=0.0, step_command=1.0)
    assert world.body().x == pytest.approx(0.0, abs=1e-7)
    assert world.last_transition().blocked_by_obstacle


def test_body_inside_new_obstacle_cannot_move_deeper() -> None:
    obstacle = AxisAlignedObstacle(
        obstacle_id="activated-around-body",
        min_x=0.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
    )
    world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=0.2, max_turn_rate=math.pi),
        obstacles=(obstacle,),
    )
    world.set_body_pose(x=0.1, y=0.0, heading=0.0)
    world.act(turn_command=0.0, step_command=0.2)

    assert world.body().x == pytest.approx(0.1)
    assert world.last_transition().blocked_by_obstacle
