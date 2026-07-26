"""Behavioural pin for the frozen left/right mirror map of ``sense_encode``.

``docs/specs/digital-ant-embodiment.md`` §5.3 freezes a 19-dim signed
involutive permutation and the temporal owner consumes it blind — its own
validation (``causal_action_projection``) only checks that the permutation is a
self-inverse and the signs square to one, which an *identity* map with all signs
``+1`` also satisfies. So mapping ``obstacle_left -> obstacle_left`` or dropping
the ``trail_pher_diff`` sign would silently kill mirror equivariance for that
channel while every algebraic check still passed.

The pin here is behavioural: mirror the WORLD (swap what is physically on the
ant's left and right, reverse its last turn, reflect its path-integration home
vector about its own body axis), run the frozen ``sense_encode`` on both worlds,
and require element-wise that

    sense(mirrored_world)[i] == signs[i] * sense(world)[permutation[i]]

for every channel. On top of that, every swap and every negation the spec names
is asserted BY CHANNEL NAME, and the named sets are asserted to be exact — an
extra negation is as much a contract break as a missing one.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from volvence_ant.env.ant_world import WorldObservation
from volvence_ant.substrate.navigator import NavigatorState, wrap_angle
from volvence_ant.substrate.sense_encode import (
    SENSE_CHANNELS_ECOLOGY_V2,
    SENSE_CHANNELS_V1,
    AntSenseSchema,
    sense_channels,
    sense_encode,
    sense_mirror_transform,
)

_TURN_COMMAND_SCALE = 0.785

# docs/specs/digital-ant-embodiment.md §5.3: "food/heat/obstacle 左右触角互换".
_SPEC_SWAPPED_PAIRS: tuple[tuple[str, str], ...] = (
    ("food_left", "food_right"),
    ("heat_left", "heat_right"),
    ("obstacle_left", "obstacle_right"),
)

# docs/specs/digital-ant-embodiment.md §5.3: these six are the pseudoscalars.
_SPEC_NEGATED_CHANNELS: tuple[str, ...] = (
    "food_diff",
    "heat_diff",
    "home_ego_sin",
    "home_pher_diff",
    "trail_pher_diff",
    "last_turn_command",
)


def _mirror_observation(observation: WorldObservation) -> WorldObservation:
    """Reflect the physical world about the ant's own body axis.

    Whatever was on the ant's left is now on its right; orientation-free
    scalars (centre intensities, contact flags, distances) are unchanged; the
    efference copy of the last turn, a signed rotation, reverses. The
    ``eval_*`` ground truth is mirrored too so the observation stays physically
    self-consistent, even though ``sense_encode`` never reads it.
    """

    return replace(
        observation,
        food_left=observation.food_right,
        food_right=observation.food_left,
        home_pher_left=observation.home_pher_right,
        home_pher_right=observation.home_pher_left,
        trail_pher_left=observation.trail_pher_right,
        trail_pher_right=observation.trail_pher_left,
        obstacle_left=observation.obstacle_right,
        obstacle_right=observation.obstacle_left,
        heat_left=observation.heat_right,
        heat_right=observation.heat_left,
        last_turn_command=-observation.last_turn_command,
        eval_home_bearing=wrap_angle(
            2.0 * observation.eval_true_heading - observation.eval_home_bearing
        ),
    )


def _mirror_navigator_state(state: NavigatorState) -> NavigatorState:
    """Reflect the path-integration home vector about the heading axis.

    The heading estimate itself is the mirror axis, so it is unchanged; the
    egocentric home bearing ``home_bearing - h_hat`` reverses sign.
    """

    egocentric = wrap_angle(state.home_bearing - state.h_hat)
    mirrored_bearing = state.h_hat - egocentric
    distance = state.home_distance
    return NavigatorState(
        h_hat=state.h_hat,
        home_dx=distance * math.cos(mirrored_bearing),
        home_dy=distance * math.sin(mirrored_bearing),
    )


def _mirror_cases(
    count: int = 24,
) -> list[tuple[WorldObservation, NavigatorState]]:
    """Non-degenerate world states: every mirrored channel actually moves."""

    rng = np.random.default_rng(20260726)
    cases: list[tuple[WorldObservation, NavigatorState]] = []
    for index in range(count):
        # Egocentric home bearing kept away from 0 / pi so sin(rel) != 0.
        egocentric = float(rng.uniform(0.2, math.pi - 0.2)) * (
            1.0 if index % 2 == 0 else -1.0
        )
        heading = float(rng.uniform(0.0, 2.0 * math.pi))
        distance = float(rng.uniform(0.5, 12.0))
        bearing = heading + egocentric
        navigator_state = NavigatorState(
            h_hat=heading,
            home_dx=distance * math.cos(bearing),
            home_dy=distance * math.sin(bearing),
        )
        observation = WorldObservation(
            food_left=float(rng.uniform(0.0, 1.0)),
            food_right=float(rng.uniform(0.0, 1.0)),
            home_pher_left=float(rng.uniform(0.0, 1.0)),
            home_pher_right=float(rng.uniform(0.0, 1.0)),
            trail_pher_left=float(rng.uniform(0.0, 1.0)),
            trail_pher_right=float(rng.uniform(0.0, 1.0)),
            obstacle_left=float(rng.uniform(0.0, 1.0)),
            obstacle_right=float(rng.uniform(0.0, 1.0)),
            obstacle_contact=bool(index % 3 == 0),
            last_turn_command=float(
                rng.uniform(0.1, 0.7) * (1.0 if index % 2 == 0 else -1.0)
            ),
            carrying_food=bool(index % 2 == 0),
            at_nest=bool(index % 5 == 0),
            at_food=bool(index % 7 == 0),
            food_center=float(rng.uniform(0.0, 1.0)),
            alarm=float(rng.uniform(0.0, 1.0)),
            eval_home_bearing=float(rng.uniform(-math.pi, math.pi)),
            eval_home_distance=distance,
            eval_true_heading=heading,
            heat_left=float(rng.uniform(0.0, 1.0)),
            heat_right=float(rng.uniform(0.0, 1.0)),
            heat_center=float(rng.uniform(0.0, 1.0)),
            heat_harmful=bool(index % 4 == 0),
        )
        cases.append((observation, navigator_state))
    return cases


def _encode(
    observation: WorldObservation,
    navigator_state: NavigatorState,
    schema: AntSenseSchema,
) -> np.ndarray:
    return sense_encode(
        observation,
        navigator_state,
        turn_command_scale=_TURN_COMMAND_SCALE,
        schema=schema,
    )


@pytest.mark.parametrize(
    "schema", [AntSenseSchema.V1, AntSenseSchema.ECOLOGY_V2]
)
def test_sense_encode_is_equivariant_under_a_real_world_mirror(
    schema: AntSenseSchema,
) -> None:
    """The published transform must equal what mirroring the WORLD produces."""

    channels = sense_channels(schema)
    permutation, signs = sense_mirror_transform(schema)
    index_of = {name: index for index, name in enumerate(channels)}

    for observation, navigator_state in _mirror_cases():
        original = _encode(observation, navigator_state, schema)
        mirrored = _encode(
            _mirror_observation(observation),
            _mirror_navigator_state(navigator_state),
            schema,
        )
        expected = np.array(
            [
                signs[index] * original[permutation[index]]
                for index in range(len(channels))
            ],
            dtype=float,
        )

        # Guard against a vacuous pass: every channel the spec names must
        # genuinely move under this world state.
        for name in _SPEC_NEGATED_CHANNELS:
            if name not in index_of:
                continue
            assert abs(original[index_of[name]]) > 1e-6, name
        for left, right in _SPEC_SWAPPED_PAIRS:
            if left not in index_of:
                continue
            assert (
                abs(original[index_of[left]] - original[index_of[right]]) > 1e-6
            ), (left, right)

        assert np.allclose(mirrored, expected, rtol=0.0, atol=1e-9), (
            "sense_encode is not equivariant under the published mirror map; "
            f"schema={schema.value} "
            + ", ".join(
                f"{channels[index]}: got {mirrored[index]!r} "
                f"expected {expected[index]!r}"
                for index in range(len(channels))
                if abs(mirrored[index] - expected[index]) > 1e-9
            )
        )


@pytest.mark.parametrize(
    "schema", [AntSenseSchema.V1, AntSenseSchema.ECOLOGY_V2]
)
def test_mirroring_the_world_twice_restores_the_sense_vector(
    schema: AntSenseSchema,
) -> None:
    for observation, navigator_state in _mirror_cases():
        original = _encode(observation, navigator_state, schema)
        twice = _encode(
            _mirror_observation(_mirror_observation(observation)),
            _mirror_navigator_state(_mirror_navigator_state(navigator_state)),
            schema,
        )
        assert np.allclose(original, twice, rtol=0.0, atol=1e-9)


def test_ecology_mirror_map_matches_the_frozen_spec_channel_by_channel() -> None:
    """Every swap and every negation the spec names, by NAME, and exactly those."""

    channels = SENSE_CHANNELS_ECOLOGY_V2
    permutation, signs = sense_mirror_transform(AntSenseSchema.ECOLOGY_V2)
    assert len(channels) == len(permutation) == len(signs) == 19

    source_of = {
        name: channels[permutation[index]] for index, name in enumerate(channels)
    }
    sign_of = {name: signs[index] for index, name in enumerate(channels)}

    for left, right in _SPEC_SWAPPED_PAIRS:
        assert source_of[left] == right, left
        assert source_of[right] == left, right
    swapped = {name for name, source in source_of.items() if source != name}
    assert swapped == {name for pair in _SPEC_SWAPPED_PAIRS for name in pair}

    for name in _SPEC_NEGATED_CHANNELS:
        assert sign_of[name] == -1, name
    negated = {name for name, sign in sign_of.items() if sign == -1}
    assert negated == set(_SPEC_NEGATED_CHANNELS)

    assert set(sign_of.values()) == {-1, 1}
    # "其余标量保持不变": everything not named above is a fixed point with +1.
    untouched = set(channels) - swapped - set(_SPEC_NEGATED_CHANNELS)
    for name in untouched:
        assert source_of[name] == name, name
        assert sign_of[name] == 1, name


def test_v1_mirror_map_is_the_ecology_map_restricted_to_v1_channels() -> None:
    assert SENSE_CHANNELS_ECOLOGY_V2[: len(SENSE_CHANNELS_V1)] == SENSE_CHANNELS_V1
    v1_permutation, v1_signs = sense_mirror_transform(AntSenseSchema.V1)
    eco_permutation, eco_signs = sense_mirror_transform(
        AntSenseSchema.ECOLOGY_V2
    )
    width = len(SENSE_CHANNELS_V1)
    assert v1_permutation == eco_permutation[:width]
    assert v1_signs == eco_signs[:width]
    # The v1 block must be closed under the mirror: no v1 channel may source
    # from an ecology-only channel.
    assert max(v1_permutation) < width


@pytest.mark.parametrize(
    "schema", [AntSenseSchema.V1, AntSenseSchema.ECOLOGY_V2]
)
def test_mirror_transform_is_a_signed_involution_and_not_the_identity(
    schema: AntSenseSchema,
) -> None:
    permutation, signs = sense_mirror_transform(schema)
    width = len(sense_channels(schema))

    assert sorted(permutation) == list(range(width))  # a genuine permutation
    for index, source in enumerate(permutation):
        assert permutation[source] == index
        assert signs[index] * signs[source] == 1

    # The two failure modes the algebraic checks alone cannot see.
    assert permutation != tuple(range(width)), "mirror map degenerated to identity"
    assert -1 in signs, "mirror map lost every pseudoscalar sign"
