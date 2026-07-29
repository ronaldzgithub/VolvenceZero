"""Ecology-v2 object, sensing and outcome contract tests."""

from __future__ import annotations

import asyncio
import math
from dataclasses import FrozenInstanceError, replace

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
from volvence_ant.substrate.motor_decode import motor_decode
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
    # A cold checkpoint has NO steering authority, so no valenced probe may
    # report direction alignment. Asserting the published truth directly is
    # stronger than restating the probe's own alignment rule inline: a sign
    # comparison against 0.0 calls a denormal 1e-18 turn "aligned", which is
    # exactly the false positive the frozen turn-delta threshold exists to
    # reject. The neutral obstacle probe is alignment-exempt by contract.
    assert all(
        not by_kind[kind].target_aligned
        for kind in (
            EcologyProbeKind.FOOD,
            EcologyProbeKind.HEAT,
            EcologyProbeKind.HOME,
        )
    )
    assert all(item.turn_delta < 1e-8 for item in probes)
    # The probe's OWN alignment expressions, restated against its published
    # turns. On a cold head this pins the rule's shape (nothing may report
    # aligned by default); the direction convention it encodes is proved in
    # ``test_target_alignment_is_toward_food_and_away_from_heat`` below.
    threshold = 1e-4
    food = by_kind[EcologyProbeKind.FOOD]
    heat = by_kind[EcologyProbeKind.HEAT]
    home = by_kind[EcologyProbeKind.HOME]
    assert food.target_aligned is (
        food.left_turn > threshold and food.right_turn < -threshold
    )
    assert heat.target_aligned is (
        heat.left_turn < -threshold and heat.right_turn > threshold
    )
    assert home.target_aligned is (home.right_turn > threshold)
    # Neutral geometry is reachability evidence, never a steering target.
    assert by_kind[EcologyProbeKind.OBSTACLE].target_aligned


async def test_target_alignment_is_toward_food_and_away_from_heat() -> None:
    """Prove the direction convention the three alignment gates consume.

    ``target_aligned`` is the only published statement of WHICH WAY a trained
    ecology policy must steer, and three P1/P2 gates read it.  A cold head
    emits exactly zero turn by design, so no assertion on cold turns can tell
    the convention from its mirror image.  What CAN be proved without a
    trained head is the two halves the convention is built out of:

    1. which lane of a paired probe carries the stimulus on the body's LEFT
       (read off the probe's own published sensor pairs and sense vectors);
    2. which sign of ``turn_command`` rotates the frozen plant to its left.

    Together those fix the meaning of the rule: ``food`` aligned means the
    lane that smells food on the left turns left (toward it), ``heat`` aligned
    means that lane turns right (away from it), and ``home`` aligned means the
    carrying lane turns toward home.  Flip either half and this test fails.
    """

    probes = await run_ecology_action_probes(
        temporal_latent_dim=8,
        seed=9,
    )
    by_kind = {item.kind: item for item in probes}

    # (1) The probe's "left" lane really is the stimulus-on-the-left lane.
    food = by_kind[EcologyProbeKind.FOOD]
    assert food.left_sensor_pair[0] > food.left_sensor_pair[1]
    assert food.right_sensor_pair[1] > food.right_sensor_pair[0]
    left_food = dict(food.left_sense)
    right_food = dict(food.right_sense)
    assert left_food["food_left"] > left_food["food_right"]
    assert right_food["food_right"] > right_food["food_left"]

    heat = by_kind[EcologyProbeKind.HEAT]
    assert heat.left_sensor_pair[0] > heat.left_sensor_pair[1]
    assert heat.right_sensor_pair[1] > heat.right_sensor_pair[0]
    left_heat = dict(heat.left_sense)
    right_heat = dict(heat.right_sense)
    assert left_heat["heat_left"] > left_heat["heat_right"]
    assert right_heat["heat_right"] > right_heat["heat_left"]

    # The HOME pair shares one geometry; only the carrying state differs, and
    # the carrying lane -- the one the convention names -- is the right one.
    home = by_kind[EcologyProbeKind.HOME]
    assert home.right_sensor_pair[0] == pytest.approx(1.0)
    assert home.left_sensor_pair[0] == pytest.approx(0.0)
    right_home = dict(home.right_sense)
    # Pinned at (2, 0) facing +y, so home is a quarter turn to the LEFT:
    # the egocentric sine of the home bearing is positive.
    assert right_home["home_ego_sin"] > 0.0

    # (2) A positive turn_command rotates the frozen plant counter-clockwise,
    # which is the body's left.
    world = AntWorld(
        config=AntWorldConfig(
            seed=0,
            antenna_offset_deg=45.0,
            antenna_reach=0.9,
        )
    )
    world.set_body_pose(body_id=0, x=0.0, y=0.0, heading=0.0)
    world.act(turn_command=0.3, step_command=0.2, body_id=0)
    turned_left = world.body(0)
    assert turned_left.heading == pytest.approx(0.3)
    assert turned_left.y > 0.0
    world.set_body_pose(body_id=0, x=0.0, y=0.0, heading=0.0)
    world.act(turn_command=-0.3, step_command=0.2, body_id=0)
    turned_right = world.body(0)
    assert turned_right.heading == pytest.approx(2.0 * math.pi - 0.3)
    assert turned_right.y < 0.0

    # (1) + (2): the published rules are approach-food / avoid-heat /
    # approach-home. A convention flipped on any kind contradicts one of the
    # two halves above.
    assert food.target_aligned is (
        food.left_turn > 1e-4 and food.right_turn < -1e-4
    )
    assert heat.target_aligned is (
        heat.left_turn < -1e-4 and heat.right_turn > 1e-4
    )
    assert home.target_aligned is (home.right_turn > 1e-4)


def test_graded_motor_decode_cannot_stop_or_slow_below_half_speed() -> None:
    """The graded controller has no stop and no crawl.

    ``motor_decode`` reads ``step_command = sigmoid(4*z2) * step_size`` and
    ``z2`` is bounded to ``LATENT_CODE_BOUNDS = (0.0, 1.0)``, so the plant's
    reachable speed band is [0.5, 0.982] of ``step_size``.  Any solvability
    argument that relies on stopping to turn in place -- the ecology oracle
    diagnostic does exactly that -- proves a manoeuvre the learned policy
    structurally cannot execute, and therefore cannot certify a layout for
    the graded arm.
    """

    step_size = 0.4
    commands = tuple(
        motor_decode(
            (0.0, 0.0, code),
            max_turn_rate=math.radians(45.0),
            step_size=step_size,
        ).step_command
        for code in (0.0, 0.25, 0.5, 0.75, 1.0)
    )

    assert min(commands) == pytest.approx(0.5 * step_size)
    assert max(commands) == pytest.approx(
        step_size / (1.0 + math.exp(-4.0))
    )
    assert all(value > 0.0 for value in commands)


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

    # ``ecology_local_valence_enabled=False`` is the DENSE-LOCAL-SHAPING-OFF
    # arm, not a "no reward" arm: only the dense per-tick local-progress
    # measurement disappears. The sparse milestone reward (pickup/delivery/
    # escape) is published elsewhere and is untouched by this lever.
    dense_local_shaping_off = AntSession(
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
    assert dense_local_shaping_off.measurement is None


def test_ecology_outcome_declares_discrete_milestone_only_on_pickup() -> None:
    """The environment owner types its boundary events; nothing is inferred.

    ``EnvironmentMeasurement.discrete_milestone`` is the owner-declared typed
    boundary event that replaces the refuted PE-magnitude event detector:
    only the discrete pickup/delivery state transitions set it, while dense
    local-valence ticks publish measurements WITHOUT it.
    """

    config = AntSessionConfig(
        objective=AntObjectiveKind.ECOLOGY,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
        ecology_local_valence_enabled=True,
    )
    source = ButterSource(
        object_id="butter",
        x=2.0,
        y=0.0,
        strength=2.2,
        decay=2.4,
        radius=0.2,
    )

    pickup_world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=0.4),
        world_objects=(source,),
    )
    # One step ends inside the ButterSource's own contact radius (0.2).
    pickup_world.set_body_pose(x=1.5, y=0.0, heading=0.0)
    pickup_world.act(turn_command=0.0, step_command=0.4)
    pickup_transition = pickup_world.last_transition()
    assert pickup_transition.picked_up
    pickup_outcome = AntSession(
        pickup_world,
        config=config,
    )._ecology_environment_outcome(
        transition=pickup_transition,
        prediction_id="pickup",
    )
    assert pickup_outcome.measurement is not None
    assert pickup_outcome.measurement.discrete_milestone is True

    routine_world = AntWorld(
        config=AntWorldConfig(seed=0, step_size=0.4),
        world_objects=(source,),
    )
    # Approaches the source (positive dense local valence) without pickup.
    routine_world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    routine_world.act(turn_command=0.0, step_command=0.4)
    routine_transition = routine_world.last_transition()
    assert not routine_transition.picked_up
    routine_outcome = AntSession(
        routine_world,
        config=config,
    )._ecology_environment_outcome(
        transition=routine_transition,
        prediction_id="routine",
    )
    assert routine_outcome.measurement is not None
    assert routine_outcome.measurement.discrete_milestone is False


def test_real_pickup_forces_temporal_switch_on_next_ant_tick() -> None:
    """End-to-end: a live pickup closes the segment and forces the switch.

    This pins the full owner chain inside the REAL ant loop (world transition
    -> owner-declared ``discrete_milestone`` -> buffered outcome -> next
    ``run_turn`` typed signal -> forced ``beta_t`` switch + credit-segment
    closure), under the actual evidence profile. The learned beta threshold
    is pinned prohibitively high after the pickup tick so a natural switch
    cannot explain the outcome; the DISABLED arm is the differential control.
    """

    from volvence_ant.evidence.runtime_profile import (
        ant_runtime_replay_rollout_config,
    )
    from volvence_zero.runtime import WiringLevel

    def run_arm(milestone_wiring: WiringLevel) -> tuple:
        rollout = ant_runtime_replay_rollout_config(
            enable_sparse_exploration=False,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        )
        rollout = replace(
            rollout,
            environment_milestone_temporal_switch=milestone_wiring,
        )
        world = AntWorld(
            config=AntWorldConfig(seed=0, step_size=0.4),
            world_objects=(
                # The body spawns on the source (away from the nest so no
                # immediate delivery), so tick 1 picks up no matter which
                # action the policy chooses (max step 0.4 stays inside the
                # 0.6 contact radius).
                ButterSource(
                    object_id="butter",
                    x=3.0,
                    y=0.0,
                    strength=2.2,
                    decay=2.4,
                    radius=0.6,
                ),
            ),
        )
        world.set_body_pose(x=3.0, y=0.0, heading=0.0)
        session = AntSession(
            world,
            config=AntSessionConfig(
                session_id=f"milestone-{milestone_wiring.value}",
                rollout_config=rollout,
                objective=AntObjectiveKind.ECOLOGY,
                sense_schema=AntSenseSchema.ECOLOGY_V2,
                ecology_local_valence_enabled=True,
            ),
        )

        async def drive() -> tuple:
            first = await session.step()
            assert world.last_transition(0).picked_up
            for store in (
                session.runner._joint_loop.world_temporal_policy
                .parameter_store,
                session.runner._joint_loop.self_temporal_policy
                .parameter_store,
            ):
                store.beta_threshold = 0.99
            second = await session.step()
            # Runtime replay settles one turn late, so the pickup
            # transition (and its milestone segment closure) lands two
            # ticks after the pickup; the carrying transitions then start
            # an independent open segment.
            later = [await session.step() for _ in range(3)]
            return first, second, later

        return asyncio.run(drive())

    _first, second, later = run_arm(WiringLevel.ACTIVE)
    assert second.is_switching is True
    assert second.steps_since_switch == 0
    assert second.switch_gate >= 0.99
    # The settled pickup transition closes the outbound credit segment on
    # its own (length 1: the ant spawned on the source), and the carrying
    # actions accumulate in a NEW open segment instead of mixing back in.
    assert later[-1].runtime_closed_segments == 1
    assert later[-1].runtime_open_segment_transitions >= 1

    _, control_second, _ = run_arm(WiringLevel.DISABLED)
    assert control_second.is_switching is False


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
