"""Tests for the two frozen substrate functions + navigator."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from volvence_ant.env.ant_world import WorldObservation
from volvence_ant.substrate.motor_decode import motor_decode
from volvence_ant.substrate.navigator import TWO_PI, AntNavigator, wrap_angle
from volvence_ant.substrate.sense_encode import (
    SENSE_CHANNELS,
    sense_encode,
    sense_to_drives,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _obs(**overrides) -> WorldObservation:
    base = dict(
        food_left=0.2,
        food_right=0.1,
        home_pher_left=0.0,
        home_pher_right=0.0,
        trail_pher_left=0.0,
        trail_pher_right=0.0,
        obstacle_left=0.0,
        obstacle_right=0.0,
        obstacle_contact=False,
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
    right = motor_decode((0.5, 0.1), max_turn_rate=1.5, step_size=0.4)
    straight = motor_decode((0.3, 0.3), max_turn_rate=1.5, step_size=0.4)
    assert left.turn_command > 0  # left evidence dominates
    assert right.turn_command < 0
    assert straight.turn_command == 0.0


def test_motor_decode_near_zero_code_only_turns_slightly() -> None:
    tiny = motor_decode((0.01, 0.0), max_turn_rate=1.5, step_size=0.4)
    large = motor_decode((0.5, 0.0), max_turn_rate=1.5, step_size=0.4)
    assert tiny.turn_command < 0
    assert abs(tiny.turn_command) < 0.05
    assert abs(large.turn_command) > abs(tiny.turn_command)


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


def test_compass_fusion_reduces_heading_drift() -> None:
    """The sky-compass channel must curb the sqrt(N) heading random walk.

    With identical proprioceptive noise and RNG seed, fusing a noisy absolute
    heading reading keeps the estimate closer to truth than pure dead reckoning.
    """

    import numpy as np

    def final_heading_error(*, compass_gain: float) -> float:
        nav = AntNavigator(
            step_size=0.4,
            heading_noise=0.05,
            step_noise=0.0,
            compass_gain=compass_gain,
            compass_noise=0.007,
            seed=7,
        )
        nav.reset(initial_heading=0.0)
        rng = np.random.default_rng(7 + 10_000)
        true_heading = 0.0
        for _ in range(200):
            turn = float(np.clip(rng.normal(0.0, 0.4), -0.6, 0.6))
            true_heading = (true_heading + turn + float(rng.normal(0.0, 0.05))) % TWO_PI
            nav.update(turn_command=turn, step_command=0.4, true_heading=true_heading)
        return abs(wrap_angle(nav.state.h_hat - true_heading))

    drift_pure = final_heading_error(compass_gain=0.0)
    drift_compass = final_heading_error(compass_gain=0.85)
    assert drift_compass < drift_pure
    # An AntBot-class compass pins heading to roughly its own noise scale.
    assert drift_compass < 0.05


def test_compass_gain_bounds_are_validated() -> None:
    import pytest

    with pytest.raises(ValueError):
        AntNavigator(step_size=0.4, compass_gain=1.5)
    with pytest.raises(ValueError):
        AntNavigator(step_size=0.4, compass_noise=-0.1)


def test_wrap_angle_range() -> None:
    for angle in (-10.0, -math.pi, 0.0, math.pi, 10.0):
        wrapped = wrap_angle(angle)
        assert -math.pi <= wrapped <= math.pi


def _navigator_sensor_parameters(navigator: AntNavigator) -> dict[str, float]:
    """Read the sensor parameters a navigator was actually CONSTRUCTED with.

    ``AntNavigator`` keeps them private and the point of the guard below is to
    compare construction, not behaviour, so the stored fields are read
    directly. A rename raises ``KeyError`` here — the correct loud failure
    rather than a silent ``getattr`` default.
    """

    stored = vars(navigator)
    return {
        "heading_noise": stored["_heading_noise"],
        "step_noise": stored["_step_noise"],
        "compass_gain": stored["_compass_gain"],
        "compass_noise": stored["_compass_noise"],
    }


def _expected_frozen_sensors() -> dict[str, float]:
    """The one frozen body-sensor declaration every arm is required to run."""

    from volvence_ant.controllers.e2e_rl_ant import (
        SHARED_COMPASS_GAIN,
        SHARED_COMPASS_NOISE,
        SHARED_HEADING_NOISE,
        SHARED_STEP_NOISE,
    )

    return {
        "heading_noise": SHARED_HEADING_NOISE,
        "step_noise": SHARED_STEP_NOISE,
        "compass_gain": SHARED_COMPASS_GAIN,
        "compass_noise": SHARED_COMPASS_NOISE,
    }


#: Arms of the P2-B confirmatory matrix
#: (``research/ant/05_ecology_p0_p1_p2_plan.md`` §5.4:
#: "learned、no-optimize、PE-off、ETA-off、FixedRule、E2E-RL、random").
#: Every one of them must run the same frozen body.
_P2B_ARMS: frozenset[str] = frozenset(
    {
        "learned",
        "no_optimize",
        "pe_off",
        "eta_off",
        "fixed_rule",
        "e2e_rl",
        "random",
    }
)


def _record_substrate_calls(monkeypatch):
    """Record navigator construction and the world-act / navigator-update order.

    Returns ``(constructed, events)``. ``events`` is an ordered log of
    ``("world.act", post_move_true_heading)`` and
    ``("navigator.update", true_heading_argument)`` so a caller can prove that
    an arm integrates path integration AFTER the world moved the body, using
    the world's own true heading — the two halves of "the compass is really
    wired in" that a ``compass_gain`` declaration alone does not give.
    """

    from volvence_ant.env.ant_world import AntWorld

    constructed: list[AntNavigator] = []
    events: list[tuple[str, float | None]] = []
    original_init = AntNavigator.__init__
    original_update = AntNavigator.update
    original_act = AntWorld.act

    def init(self, **kwargs):
        original_init(self, **kwargs)
        constructed.append(self)

    def update(self, *, turn_command, step_command, true_heading=None):
        events.append(("navigator.update", true_heading))
        return original_update(
            self,
            turn_command=turn_command,
            step_command=step_command,
            true_heading=true_heading,
        )

    def act(self, *, turn_command, step_command, body_id=0):
        observation = original_act(
            self,
            turn_command=turn_command,
            step_command=step_command,
            body_id=body_id,
        )
        events.append(("world.act", observation.eval_true_heading))
        return observation

    monkeypatch.setattr(AntNavigator, "__init__", init)
    monkeypatch.setattr(AntNavigator, "update", update)
    monkeypatch.setattr(AntWorld, "act", act)
    return constructed, events


def _assert_integrated_after_acting(
    events: list[tuple[str, float | None]], *, ticks: int
) -> None:
    """spec §3: act first, then fuse the POST-move absolute heading."""

    assert [name for name, _ in events] == [
        "world.act",
        "navigator.update",
    ] * ticks, [name for name, _ in events]
    for index in range(0, len(events), 2):
        acted_heading = events[index][1]
        fused_heading = events[index + 1][1]
        # ``true_heading=None`` means the compass channel is dead no matter
        # what ``compass_gain`` claims.
        assert fused_heading is not None, index
        assert fused_heading == acted_heading, (
            index,
            fused_heading,
            acted_heading,
        )


def _distorted_world(*, bias: float, seed: int = 5):
    """A world whose actuator silently adds ``bias`` radians to every turn.

    The efference copy therefore diverges from truth by ``bias`` per tick, so a
    pure dead-reckoning arm accumulates unbounded heading error while a
    compass-fused arm stays pinned — orders of magnitude apart.
    """

    from volvence_ant.env.ant_world import (
        AntWorld,
        AntWorldConfig,
        MotorDistortionProfile,
    )

    return AntWorld(
        config=AntWorldConfig(
            seed=seed,
            motor_distortions=(MotorDistortionProfile(turn_bias=bias),),
        )
    )


@pytest.mark.skipif(
    not _HAS_TORCH, reason="the E2E-RL arm needs the optional torch extra"
)
def test_matched_control_arms_share_the_frozen_navigator_sensors() -> None:
    """spec §3: one frozen substrate across EVERY matched-control arm.

    "罗盘是所有导航共用的 substrate 传感器 ... 同一 frozen substrate 在
    matched-control 各臂间一致". Both the E2E-RL baseline and the ``random``
    floor used to build compass-less navigators, which quietly gave them a
    different (easier) body than the arms they are meant to bound.

    The kernel arms are built from the PUBLISHED factories in
    ``scripts/run_ant_matched_control.py`` rather than from hand-written
    configs, so an arm-specific sensor override in the runner is caught here.
    """

    from scripts.run_ant_matched_control import (
        _learned_config,
        _pe_off_config,
        _schedule_gated_arms,
    )
    from volvence_ant.controllers.e2e_rl_ant import E2ERLAnt
    from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
    from volvence_ant.controllers.random_ant import RandomAnt
    from volvence_ant.env.ant_world import AntWorld, AntWorldConfig
    from volvence_ant.proofs.matched_control import arm_substrate_parameters
    from volvence_ant.runtime.ant_session import AntSession, AntSessionConfig

    expected = _expected_frozen_sensors()
    n_z = 16

    def world() -> AntWorld:
        return AntWorld(config=AntWorldConfig(seed=11))

    schedule_gated = _schedule_gated_arms(seed=0, n_z=n_z)
    kernel_configs = {
        "learned": _learned_config(0, n_z),
        "pe_off": _pe_off_config(0, n_z),
        "no_optimize": schedule_gated["no_optimize"],
        "eta_off": schedule_gated["eta_off"],
    }
    baseline = E2ERLAnt(seed=0, hidden_dim=4)
    baseline.attach(world(), body_id=0, seed=0)
    navigators: dict[str, AntNavigator] = {
        name: AntSession(world(), config=config).navigator
        for name, config in kernel_configs.items()
    }
    navigators["fixed_rule"] = FixedRuleAnt(world()).navigator
    # ``navigator`` is read the SAME way on every arm: a plain attribute, never
    # a call. E2ERLAnt exposes it as a property over its per-body registry so
    # duck-typed cross-arm access does not break on exactly one arm.
    navigators["e2e_rl"] = baseline.navigator
    navigators["random"] = RandomAnt(world(), seed=0).navigator

    assert set(navigators) == set(_P2B_ARMS)
    for arm, navigator in sorted(navigators.items()):
        assert isinstance(navigator, AntNavigator), arm
        assert _navigator_sensor_parameters(navigator) == expected, arm

    # E2ERLAnt keeps a multi-body registry; the uniform attribute must be the
    # body-0 entry, and an unattached body must fail loudly, not return None.
    assert baseline.navigator is baseline.navigator_for(0)
    with pytest.raises(RuntimeError, match="never attached"):
        baseline.navigator_for(3)

    # The arms must agree because they read one frozen declaration, not
    # because several copies happened to be typed identically today.
    assert AntSessionConfig().compass_gain == expected["compass_gain"]
    assert FixedRuleConfig().compass_gain == expected["compass_gain"]

    # The digest the evidence runners bind into provenance must describe the
    # body the arms actually construct, otherwise it certifies nothing.
    declared = arm_substrate_parameters()
    assert set(declared) == {"kernel", "fixed_rule", "e2e_rl", "random"}
    for family, values in sorted(declared.items()):
        assert values == expected, family


@pytest.mark.skipif(
    not _HAS_TORCH, reason="the E2E-RL arm needs the optional torch extra"
)
def test_e2e_rl_arm_actually_fuses_the_sky_compass(monkeypatch) -> None:
    """``step()`` — the path the colony/ecology harnesses drive."""

    from volvence_ant.controllers.e2e_rl_ant import E2ERLAnt

    ticks = 40
    bias = 0.05
    world = _distorted_world(bias=bias)
    ant = E2ERLAnt(seed=0, hidden_dim=4)
    ant.attach(world, body_id=0, seed=0)
    constructed, events = _record_substrate_calls(monkeypatch)
    for _ in range(ticks):
        ant.step(world, body_id=0)

    assert constructed == []
    _assert_integrated_after_acting(events, ticks=ticks)
    error = abs(wrap_angle(ant.navigator.state.h_hat - world.body(0).heading))
    # Pure dead reckoning would sit near ``ticks * bias`` (~2.0 rad) here.
    assert error < 0.1, error


@pytest.mark.skipif(
    not _HAS_TORCH, reason="the E2E-RL arm needs the optional torch extra"
)
def test_e2e_rl_evaluate_path_fuses_the_sky_compass(monkeypatch) -> None:
    """``evaluate()`` is where the PUBLISHED ``e2e_rl`` numbers come from.

    ``proofs/matched_control._run_e2e_arm`` reports this arm entirely from
    ``evaluate()``; pinning ``step()`` alone leaves the evidence-producing path
    free to drift back to a compass-less, integrate-before-acting body with the
    whole suite still green.
    """

    from volvence_ant.controllers.e2e_rl_ant import E2ERLAnt

    ticks = 40
    bias = 0.05
    world = _distorted_world(bias=bias)
    ant = E2ERLAnt(seed=0, hidden_dim=4)
    constructed, events = _record_substrate_calls(monkeypatch)

    evaluation = ant.evaluate(world=world, ticks=ticks, seed=0)

    assert len(evaluation.positions) == ticks
    assert len(constructed) == 1, len(constructed)
    assert _navigator_sensor_parameters(constructed[0]) == _expected_frozen_sensors()
    _assert_integrated_after_acting(events, ticks=ticks)
    error = abs(wrap_angle(constructed[0].state.h_hat - world.body(0).heading))
    assert error < 0.1, error


@pytest.mark.skipif(
    not _HAS_TORCH, reason="the E2E-RL arm needs the optional torch extra"
)
def test_e2e_rl_train_path_fuses_the_sky_compass(monkeypatch) -> None:
    """``train()`` produces the policy the published numbers are evaluated on.

    A trainer that learns on a compass-less body is optimising a different
    control problem than the one it is scored on, so this path carries the same
    frozen-substrate obligation as ``evaluate()``.
    """

    from volvence_ant.controllers.e2e_rl_ant import E2ERLAnt, PPOConfig
    from volvence_ant.env.ant_world import AntWorld

    ticks = 40
    bias = 0.05
    worlds: list[AntWorld] = []

    def world_factory(seed: int) -> AntWorld:
        world = _distorted_world(bias=bias, seed=seed)
        worlds.append(world)
        return world

    ant = E2ERLAnt(seed=0, hidden_dim=4)
    constructed, events = _record_substrate_calls(monkeypatch)

    ant.train(
        world_factory=world_factory,
        seed=3,
        config=PPOConfig(
            episodes=1, ticks_per_episode=ticks, update_epochs=1
        ),
    )

    assert len(worlds) == 1, len(worlds)
    assert len(constructed) == 1, len(constructed)
    assert _navigator_sensor_parameters(constructed[0]) == _expected_frozen_sensors()
    _assert_integrated_after_acting(events, ticks=ticks)
    error = abs(
        wrap_angle(constructed[0].state.h_hat - worlds[0].body(0).heading)
    )
    assert error < 0.1, error


def test_random_floor_arm_actually_fuses_the_sky_compass(monkeypatch) -> None:
    """The ``random`` floor is a first-class P2-B arm, not a scratch baseline.

    It used to build a compass-less navigator and integrate the efference copy
    BEFORE ``world.act`` — exactly the defect that was fixed for E2E-RL.
    """

    from volvence_ant.controllers.random_ant import RandomAnt

    ticks = 40
    bias = 0.05
    world = _distorted_world(bias=bias)
    constructed, events = _record_substrate_calls(monkeypatch)

    ant = RandomAnt(world, seed=3)
    ant.run(ticks)

    assert len(constructed) == 1, len(constructed)
    assert constructed[0] is ant.navigator
    assert _navigator_sensor_parameters(ant.navigator) == _expected_frozen_sensors()
    _assert_integrated_after_acting(events, ticks=ticks)
    assert len(ant.positions) == ticks
    error = abs(wrap_angle(ant.navigator.state.h_hat - world.body(0).heading))
    # Pure dead reckoning would sit near ``ticks * bias`` (~2.0 rad) here.
    assert error < 0.1, error


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
