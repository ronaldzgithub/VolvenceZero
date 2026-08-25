"""Phase 0 experiments: path-integration homing + route familiarity.

Two benchmarks with published biological ground truth:

- ``homing_precision_experiment`` measures the ring-attractor / path-integration
  home-vector error as a function of outbound journey length. This is a
  *substrate-level* property (the frozen navigator), so it is measured directly
  and fast, and compared against the AntBot aggregate result (0.67% +/- 0.27% of
  journey length over 26 runs; Dupeyroux 2019). The navigator fuses an
  AntBot-class sky-compass (absolute-heading) reading, matching AntBot's own
  celestial-compass + optic-flow configuration; a pure efference-copy integrator
  cannot reach this scale because heading error grows as sqrt(N).

- ``route_learning_experiment`` drives the kernel ant along a FIXED scripted
  route repeatedly and records the prediction-error (novelty) per exposure.
  Familiarity should rise (PE fall) within tens of exposures, mirroring
  Ardin 2016. This exercises the R-PE main chain + memory, so it uses the real
  kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime.ant_session import AntSession, AntSessionConfig
from volvence_ant.substrate.navigator import AntNavigator


@dataclass(frozen=True)
class HomingCurvePoint:
    journey_length: float
    mean_endpoint_error: float
    mean_normalized_error: float
    mean_direction_error: float
    n_trials: int


@dataclass(frozen=True)
class HomingPrecisionResult:
    curve: tuple[HomingCurvePoint, ...]
    antbot_reference_ratio: float  # AntBot aggregate 0.67% (Dupeyroux 2019)
    passes_antbot_scale: bool
    description: str


def _random_outbound_home_error(
    *,
    journey_steps: int,
    step_size: float,
    max_turn_rate: float,
    heading_noise: float,
    step_noise: float,
    compass_gain: float,
    compass_noise: float,
    seed: int,
) -> tuple[float, float, float]:
    """One outbound random walk; returns (endpoint_error, path_length, dir_error)."""

    nav = AntNavigator(
        step_size=step_size,
        heading_noise=heading_noise,
        step_noise=step_noise,
        compass_gain=compass_gain,
        compass_noise=compass_noise,
        seed=seed,
    )
    nav.reset(initial_heading=0.0)
    rng = np.random.default_rng(seed + 10_000)
    true_x = 0.0
    true_y = 0.0
    true_heading = 0.0
    path_length = 0.0
    for _ in range(journey_steps):
        turn = float(np.clip(rng.normal(0.0, 0.4), -max_turn_rate, max_turn_rate))
        # World process noise and estimator noise come from independent RNGs.
        # Advance the world truth first, then let the navigator fuse the noisy
        # sky-compass reading of that post-turn absolute heading.
        true_heading = (
            true_heading + turn + float(rng.normal(0.0, heading_noise))
        ) % (2.0 * math.pi)
        nav.update(
            turn_command=turn, step_command=step_size, true_heading=true_heading
        )
        true_step = max(0.0, step_size + float(rng.normal(0.0, step_noise)))
        true_x += true_step * math.cos(true_heading)
        true_y += true_step * math.sin(true_heading)
        path_length += true_step
    state = nav.state
    true_home_dx = -true_x
    true_home_dy = -true_y
    err_x = state.home_dx - true_home_dx
    err_y = state.home_dy - true_home_dy
    endpoint_error = math.hypot(err_x, err_y)
    est_bearing = math.atan2(state.home_dy, state.home_dx)
    true_bearing = math.atan2(true_home_dy, true_home_dx)
    dir_error = abs((est_bearing - true_bearing + math.pi) % (2.0 * math.pi) - math.pi)
    return endpoint_error, path_length, dir_error


def homing_precision_experiment(
    *,
    journey_step_grid: tuple[int, ...] = (10, 20, 40, 80, 160),
    n_trials: int = 24,
    step_size: float = 0.4,
    max_turn_rate: float = math.radians(45.0),
    heading_noise: float = 0.02,
    step_noise: float = 0.004,
    compass_gain: float = 0.85,
    compass_noise: float = 0.007,
    seed: int = 0,
) -> HomingPrecisionResult:
    curve: list[HomingCurvePoint] = []
    worst_ratio = 0.0
    for journey_steps in journey_step_grid:
        endpoint_errors: list[float] = []
        normalized: list[float] = []
        dir_errors: list[float] = []
        lengths: list[float] = []
        for trial in range(n_trials):
            endpoint_error, path_length, dir_error = _random_outbound_home_error(
                journey_steps=journey_steps,
                step_size=step_size,
                max_turn_rate=max_turn_rate,
                heading_noise=heading_noise,
                step_noise=step_noise,
                compass_gain=compass_gain,
                compass_noise=compass_noise,
                seed=seed + trial + journey_steps * 1000,
            )
            endpoint_errors.append(endpoint_error)
            lengths.append(path_length)
            if path_length > 0:
                normalized.append(endpoint_error / path_length)
            dir_errors.append(dir_error)
        mean_norm = float(np.mean(normalized))
        worst_ratio = max(worst_ratio, mean_norm)
        curve.append(
            HomingCurvePoint(
                journey_length=float(np.mean(lengths)),
                mean_endpoint_error=float(np.mean(endpoint_errors)),
                mean_normalized_error=mean_norm,
                mean_direction_error=float(np.mean(dir_errors)),
                n_trials=n_trials,
            )
        )
    # AntBot aggregate homing error: 0.67% +/- 0.27% of journey length over 26
    # runs (Dupeyroux 2019, PI-Full aggregate). The aggregate mean is the
    # representative AntBot figure; the single 14 m 0.47% run is a best case.
    antbot_ratio = 0.0067
    passes = worst_ratio <= antbot_ratio
    return HomingPrecisionResult(
        curve=tuple(curve),
        antbot_reference_ratio=antbot_ratio,
        passes_antbot_scale=passes,
        description=(
            f"path-integration homing: worst normalized error={worst_ratio:.4f} "
            f"(AntBot aggregate ref={antbot_ratio:.4f}); passes_scale={passes}"
        ),
    )


@dataclass(frozen=True)
class RouteLearningResult:
    pe_by_exposure: tuple[float, ...]
    novelty_by_exposure: tuple[float, ...]
    first_exposure_novelty: float
    last_exposure_novelty: float
    familiarity_improved: bool
    exposures: int
    route_length: int
    novel_route_novelty: float
    shuffled_route_novelty: float
    memory_off_last_novelty: float
    pe_off_last_novelty: float
    description: str


def _scripted_route(
    *, length: int, turn_sign: float = 1.0
) -> tuple[tuple[float, float], ...]:
    """Fixed motor-command route; every point is reached through ``world.act``."""

    route: list[tuple[float, float]] = []
    for i in range(length):
        turn = turn_sign * math.radians(12.0 if i % 2 == 0 else 4.0)
        route.append((turn, 0.4))
    return tuple(route)


async def _walk_route(
    *,
    session: AntSession,
    commands: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    session.world.reset_body()
    initial = session.world.observe()
    session.navigator.reset(initial_heading=initial.eval_true_heading)
    session.holder.update(
        observation=initial,
        navigator_state=session.navigator.state,
        step=session.world.tick,
    )
    errors: list[float] = []
    novelties: list[float] = []
    for turn, step in commands:
        observation = session.world.act(turn_command=turn, step_command=step)
        nav = session.navigator.update(
            turn_command=turn,
            step_command=step,
            true_heading=observation.eval_true_heading,
        )
        session.holder.update(
            observation=observation,
            navigator_state=nav,
            step=session.world.tick,
        )
        result = await session.runner.run_turn(f"physical-route-{session.world.tick}")
        if result.prediction_error is not None:
            errors.append(abs(result.prediction_error.task_error))
        novelties.append(_epistemic_novelty(result))
    return (
        float(np.mean(errors)) if errors else 0.0,
        float(np.mean(novelties)) if novelties else 0.0,
    )


async def route_learning_experiment(
    *,
    exposures: int = 12,
    route_length: int = 6,
    temporal_latent_dim: int = 4,
    seed: int = 0,
) -> RouteLearningResult:
    world = AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=20.0, y=0.0, strength=0.2, decay=8.0),),
    )
    session = AntSession(
        world, config=AntSessionConfig(temporal_latent_dim=temporal_latent_dim, seed=seed)
    )
    route = _scripted_route(length=route_length)
    pe_by_exposure: list[float] = []
    novelty_by_exposure: list[float] = []
    for _ in range(exposures):
        error, novelty = await _walk_route(session=session, commands=route)
        pe_by_exposure.append(error)
        novelty_by_exposure.append(novelty)
    _, novel_route_novelty = await _walk_route(
        session=session,
        commands=_scripted_route(length=route_length, turn_sign=-1.0),
    )
    rng = np.random.default_rng(seed + 73)
    shuffled = tuple(route[index] for index in rng.permutation(len(route)))
    _, shuffled_route_novelty = await _walk_route(
        session=session,
        commands=shuffled,
    )

    pe_off = AntSession(
        AntWorld(
            config=AntWorldConfig(seed=seed),
            food_sources=(FoodSource(x=20.0, y=0.0, strength=0.2, decay=8.0),),
        ),
        config=AntSessionConfig(
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
            external_prediction_error_drive=False,
        ),
    )
    memory_off_last = 0.0
    pe_off_last = 0.0
    for exposure in range(exposures):
        # A fresh session each exposure is the explicit memory-off control:
        # sensing/runtime remain identical but no state can cross exposures.
        memory_off = AntSession(
            AntWorld(
                config=AntWorldConfig(seed=seed + exposure),
                food_sources=(
                    FoodSource(x=20.0, y=0.0, strength=0.2, decay=8.0),
                ),
            ),
            config=AntSessionConfig(
                temporal_latent_dim=temporal_latent_dim,
                seed=seed + exposure,
            ),
        )
        _, memory_off_last = await _walk_route(session=memory_off, commands=route)
        _, pe_off_last = await _walk_route(session=pe_off, commands=route)
    first = novelty_by_exposure[0] if novelty_by_exposure else 0.0
    last = novelty_by_exposure[-1] if novelty_by_exposure else 0.0
    return RouteLearningResult(
        pe_by_exposure=tuple(pe_by_exposure),
        novelty_by_exposure=tuple(novelty_by_exposure),
        first_exposure_novelty=first,
        last_exposure_novelty=last,
        familiarity_improved=last <= first,
        exposures=exposures,
        route_length=route_length,
        novel_route_novelty=novel_route_novelty,
        shuffled_route_novelty=shuffled_route_novelty,
        memory_off_last_novelty=memory_off_last,
        pe_off_last_novelty=pe_off_last,
        description=(
            f"route familiarity: reducible novelty {first:.4f} -> {last:.4f} over "
            f"{exposures} exposures (improved={last <= first})"
        ),
    )


def _epistemic_novelty(result: object) -> float:
    """Reducible-surprise readout (epistemic PE) from the prediction_error snapshot.

    This is the digital-ant analogue of mushroom-body novelty: the part of the
    prediction error the critic can still drive down. It should fall as a fixed
    route becomes familiar.
    """

    snapshot = result.active_snapshots.get("prediction_error")  # type: ignore[attr-defined]
    if snapshot is None:
        return 0.0
    # pe_decomposition is a documented-optional field (None only on bootstrap
    # turns); direct access keeps a schema mismatch loud instead of masking it.
    decomposition = snapshot.value.pe_decomposition
    if decomposition is None:
        return 0.0
    return float(decomposition.epistemic_magnitude)
