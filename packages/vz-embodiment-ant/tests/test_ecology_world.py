"""Ecology-v2 object, sensing and outcome contract tests."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from volvence_ant.env import (
    AntWorld,
    AntWorldConfig,
    BurningMatch,
    ButterSource,
    WoodStick,
    WorldObjectKind,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyProbeKind,
    run_ecology_action_probes,
)
from volvence_ant.runtime.ant_session import (
    AntObjectiveKind,
    AntSession,
    AntSessionConfig,
)
from volvence_ant.substrate.navigator import AntNavigator
from volvence_ant.substrate.sense_encode import (
    SENSE_CHANNELS_ECOLOGY_V2,
    SENSE_CHANNELS_V1,
    AntSenseSchema,
    sense_encode,
)


def test_world_object_snapshots_are_owner_authored_and_frozen() -> None:
    world = AntWorld(
        world_objects=(
            ButterSource(object_id="butter-1", x=2.0, y=1.0),
            WoodStick(
                object_id="stick-1",
                start_x=-1.0,
                start_y=2.0,
                end_x=2.0,
                end_y=3.0,
            ),
            BurningMatch(object_id="match-1", x=-2.0, y=-1.0),
        )
    )
    snapshots = world.world_object_snapshots()

    assert tuple(item.kind for item in snapshots) == (
        WorldObjectKind.BUTTER,
        WorldObjectKind.WOOD_STICK,
        WorldObjectKind.BURNING_MATCH,
    )
    assert snapshots[2].effect_radius > 0.0
    with pytest.raises(FrozenInstanceError):
        snapshots[0].active = False  # type: ignore[misc]


def test_butter_is_the_only_food_when_explicitly_placed() -> None:
    world = AntWorld(
        config=AntWorldConfig(step_size=0.0),
        world_objects=(
            ButterSource(
                object_id="butter",
                x=0.0,
                y=0.0,
                remaining=1.0,
            ),
        ),
    )

    assert world.food_sources() == ()
    world.act(turn_command=0.0, step_command=0.0)
    assert world.body().carrying_food
    assert world.food_pickups == 1
    snapshot = world.world_object_snapshots()[0]
    assert snapshot.remaining == 0.0
    assert not snapshot.active


def test_directional_wood_stick_blocks_without_diagonal_tunnelling() -> None:
    world = AntWorld(
        config=AntWorldConfig(
            seed=0,
            step_size=4.0,
            max_turn_rate=math.pi,
        ),
        world_objects=(
            WoodStick(
                object_id="diagonal-stick",
                start_x=1.0,
                start_y=-1.0,
                end_x=3.0,
                end_y=1.0,
                radius=0.2,
            ),
        ),
    )
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    world.act(turn_command=0.0, step_command=4.0)

    transition = world.last_transition()
    assert transition.blocked_by_obstacle
    assert transition.obstacle_id == "diagonal-stick"
    assert world.body().x < 2.0


def test_match_heat_is_local_directional_and_does_not_trigger_global_alarm() -> None:
    world = AntWorld(
        config=AntWorldConfig(
            seed=0,
            antenna_offset_deg=30.0,
            antenna_reach=0.6,
            step_size=0.5,
        ),
        world_objects=(
            BurningMatch(
                object_id="match",
                x=0.5,
                y=0.3,
                heat_decay=0.8,
                harm_threshold=0.6,
            ),
        ),
    )
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    observation = world.observe()

    assert observation.heat_left > observation.heat_right
    assert observation.alarm == 0.0
    world.act(turn_command=0.0, step_command=0.5)
    transition = world.last_transition()
    assert transition.heat_load_after > transition.heat_load_before
    assert transition.entered_harmful_heat
    world.set_body_pose(x=0.5, y=0.0, heading=math.pi)
    world.act(turn_command=0.0, step_command=0.5)
    assert world.last_transition().escaped_harmful_heat


def test_ecology_sense_schema_appends_heat_without_changing_v1() -> None:
    world = AntWorld(world_objects=(BurningMatch(object_id="match", x=0.5, y=0.3),))
    navigator = AntNavigator(step_size=world.config.step_size)
    navigator.reset(initial_heading=world.body().heading)
    observation = world.observe()
    v1 = sense_encode(
        observation,
        navigator.state,
        turn_command_scale=world.config.max_turn_rate,
        schema=AntSenseSchema.V1,
    )
    v2 = sense_encode(
        observation,
        navigator.state,
        turn_command_scale=world.config.max_turn_rate,
        schema=AntSenseSchema.ECOLOGY_V2,
    )

    assert v1.shape == (len(SENSE_CHANNELS_V1),)
    assert v2.shape == (len(SENSE_CHANNELS_ECOLOGY_V2),)
    assert tuple(v2[: len(v1)]) == pytest.approx(tuple(v1))
    heat_diff = SENSE_CHANNELS_ECOLOGY_V2.index("heat_diff")
    assert v2[heat_diff] == pytest.approx(observation.heat_left - observation.heat_right)


def test_ant_session_declares_full_ecology_input_width_to_runtime() -> None:
    session = AntSession(
        AntWorld(),
        config=AntSessionConfig(
            temporal_latent_dim=4,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )

    assert session.runner.temporal_input_dim == len(SENSE_CHANNELS_ECOLOGY_V2)
    assert (
        session.runner._world_temporal_policy.parameter_store.n_z
        == session.config.temporal_latent_dim
    )


async def test_ecology_step_publishes_owner_authored_diagnostics() -> None:
    world = AntWorld(
        world_objects=(
            ButterSource(object_id="butter", x=1.5, y=0.0),
            WoodStick(
                object_id="stick",
                start_x=0.8,
                start_y=-0.5,
                end_x=0.8,
                end_y=0.5,
            ),
            BurningMatch(object_id="match", x=1.2, y=0.5),
        )
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=8,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )

    record = await session.step()

    assert tuple(name for name, _ in record.sense_activation) == (
        SENSE_CHANNELS_ECOLOGY_V2
    )
    assert record.nearest_food_distance is not None
    assert record.nearest_obstacle_distance is not None
    assert record.nearest_heat_distance is not None


async def test_paired_ecology_channels_reach_code_and_motor_output() -> None:
    """Cold probes prove sensory reachability, not steering.

    Under exclusive steering the state-conditioned head owns the actuator
    contrast axis and its cold parameters are exactly zero, so a cold
    controller emits zero turn by construction. Demanding cold
    ``action_sensitive`` would assert that steering exists before any
    training -- the opposite of the capability the curriculum must produce.
    Turn sensitivity and direction are asserted on trained checkpoints by the
    P1 ``paired_action_sensitivity`` / ``food_steering_alignment`` gates.
    """

    probes = await run_ecology_action_probes(
        temporal_latent_dim=8,
        seed=9,
    )

    assert tuple(item.kind for item in probes) == tuple(EcologyProbeKind)
    assert all(item.input_reachable for item in probes)
    assert all(item.code_l1_delta > 0.0 for item in probes)
    assert not any(item.action_sensitive for item in probes)
    by_kind = {item.kind: item for item in probes}
    assert by_kind[EcologyProbeKind.FOOD].target_aligned is (
        by_kind[EcologyProbeKind.FOOD].left_turn > 0.0
        and by_kind[EcologyProbeKind.FOOD].right_turn < 0.0
    )
    assert by_kind[EcologyProbeKind.HEAT].target_aligned is (
        by_kind[EcologyProbeKind.HEAT].left_turn < 0.0
        and by_kind[EcologyProbeKind.HEAT].right_turn > 0.0
    )
    assert by_kind[EcologyProbeKind.HOME].target_aligned is (
        by_kind[EcologyProbeKind.HOME].right_turn > 0.0
    )


def test_environment_publishes_local_valence_without_target_direction() -> None:
    world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=0.4),
        world_objects=(
            ButterSource(
                object_id="butter",
                x=2.0,
                y=0.0,
                strength=2.2,
                decay=2.4,
                radius=0.2,
            ),
        ),
    )
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    session = AntSession(
        world,
        config=AntSessionConfig(
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
            ecology_local_valence_enabled=True,
        ),
    )

    world.act(turn_command=0.0, step_command=0.4)
    transition = world.last_transition()
    outcome = session._ecology_environment_outcome(
        transition=transition,
        prediction_id="prediction-1",
    )

    assert (
        transition.local_food_signal_after
        > transition.local_food_signal_before
    )
    assert outcome.measurement is not None
    assert outcome.measurement.action_payoff > 0.0
    assert all("direction" not in item for item in outcome.evidence)

    valence_off = AntSession(
        world,
        config=AntSessionConfig(
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
            ecology_local_valence_enabled=False,
        ),
    )._ecology_environment_outcome(
        transition=transition,
        prediction_id="prediction-1",
    )
    assert valence_off.measurement is None


def test_ecology_neutral_stick_contact_is_observable_but_valence_free() -> None:
    """Wood-stick contact stays a physical fact and never enters payoff.

    The body is pinned against the stick so the blocked move keeps every
    local food/home/heat delta exactly zero: the only fact left is the
    neutral contact, which must not produce a measurement.
    """
    world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=0.5),
        world_objects=(
            WoodStick(
                object_id="stick",
                start_x=0.35,
                start_y=-1.0,
                end_x=0.35,
                end_y=1.0,
            ),
        ),
    )
    world.set_body_pose(x=0.25, y=0.0, heading=0.0)
    session = AntSession(
        world,
        config=AntSessionConfig(
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    world.act(turn_command=0.0, step_command=0.05)
    transition = world.last_transition()
    assert transition.blocked_by_obstacle
    assert transition.applied_step == 0.0
    outcome = session._ecology_environment_outcome(
        transition=transition,
        prediction_id="prediction-1",
    )

    assert outcome.status == "obstacle_contact"
    assert outcome.measurement is None
    assert "no coordinates" in outcome.detail
    assert "turn" not in outcome.evidence[0]


def test_carrying_local_valence_uses_path_integration_home_progress() -> None:
    world = AntWorld(config=AntWorldConfig(seed=0, step_size=0.4))
    world.set_body_pose(
        x=2.0,
        y=0.0,
        heading=math.pi,
        carrying_food=True,
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    world.act(turn_command=0.0, step_command=0.4)
    transition = world.last_transition()

    closer = session._ecology_environment_outcome(
        transition=transition,
        prediction_id="closer",
        home_progress=0.4,
    )
    farther = session._ecology_environment_outcome(
        transition=transition,
        prediction_id="farther",
        home_progress=-0.4,
    )

    assert closer.measurement is not None
    assert farther.measurement is not None
    assert closer.measurement.action_payoff > 0.0
    assert farther.measurement.action_payoff < 0.0
