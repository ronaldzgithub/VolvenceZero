"""Paired ecology probes for sensor reachability and learned action sensitivity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel

from volvence_ant.env import (
    AntWorld,
    AntWorldConfig,
    BurningMatch,
    ButterSource,
    WoodStick,
)
from volvence_ant.env.world_objects import WorldObject
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.substrate.ant_actuator import AntActuator


# Read-only local sensitivity diagnostic (plan 05 P0-A: "对 paired latent delta
# 做有限差分，判断其是否落在 motor readout 的近零空间").  The step is small
# enough to stay inside the frozen ``motor_decode`` linear band and large enough
# to stay far above float64 cancellation noise.
ECOLOGY_PROBE_FINITE_DIFFERENCE_EPSILON = 1e-4
# Cosine alignment between the paired latent delta and the motor readout
# gradient below which the delta is reported as living in the actuator
# near-null space.  Diagnostic only: it explains a code-moves/turn-frozen
# failure, it never decides a gate.
ECOLOGY_PROBE_NEAR_NULL_SPACE_ALIGNMENT = 1e-3

# The rollout wiring entries that define a backend lane.  The probe publishes
# what the session actually reported for these keys so a parity claim can be
# checked against the lane it says it ran on, instead of trusting the request.
ECOLOGY_PROBE_LANE_WIRING_KEYS: tuple[str, ...] = (
    "temporal_runtime_backend",
    "temporal_ssl_backend",
    "internal_rl_backend",
)


class EcologyProbeKind(str, Enum):
    FOOD = "food"
    OBSTACLE = "obstacle"
    HEAT = "heat"
    HOME = "home"


class EcologyProbeBackendLane(str, Enum):
    """Backend lanes that must agree on code, action and turn (plan 05:123).

    ``PURE`` is the reproducible float64 rollback baseline, ``RUNTIME`` is the
    ndarray metacontroller runtime, ``TORCH`` is the autograd SSL / Internal-RL
    backend.  Passing ``None`` instead of a lane keeps the deployed evidence
    profile untouched, which is what every non-parity caller wants.
    """

    PURE = "pure"
    RUNTIME = "runtime"
    TORCH = "torch"


@dataclass(frozen=True)
class EcologyPosteriorHiddenSummary:
    """Owner-published encoder/posterior hidden state, reduced to scalars."""

    dim: int
    l1_norm: float
    l2_norm: float
    max_abs: float


@dataclass(frozen=True)
class EcologyBackendExecutionEvidence:
    """Owner-published proof of what a lane's declared backends actually did.

    Coverage may NEVER be inferred from "this lane's numbers differ from the
    reference": two lanes that legitimately agree bit-for-bit would then be
    indistinguishable from a lane that never ran, and exact agreement -- the
    ideal outcome -- would be reported as non-evaluation.  Every field here is
    read from an owner's own evidence readout instead:

    * ``temporal_runtime_backend_applied`` is the level the LIVE temporal
      policy object reports, not the level the rollout config requested;
    * ``ssl_*`` come from ``ETANLJointLoop.latest_ssl_report``;
    * ``internal_rl_*`` come from ``ETANLJointLoop.latest_internal_rl_report``
      (``None`` until a full optimization cycle has run at least once).

    ``ETANLJointLoop`` is reached through the documented
    ``AgentSessionRunner.joint_loop`` read-only evidence handle; nothing here
    imports a kernel-internal module.
    """

    #: Optimization-carrying ticks run before the measured paired step.
    exercise_steps: int
    temporal_runtime_backend_applied: str
    ssl_backend_applied: str
    internal_rl_backend_applied: str
    #: 0 means ``MetacontrollerSSLTrainer.optimize`` early-returned (the trace
    #: was shorter than two steps), i.e. the SSL lane never trained at all.
    ssl_trained_steps: int
    ssl_report_published: bool
    #: Every distinct ``torch_backend`` outcome the owner reported across the
    #: sessions this lane opened. A single joined string would hide a lane that
    #: ran on one probe and was skipped on another.
    ssl_torch_backends: tuple[str, ...]
    ssl_torch_parameters_changed: int
    ssl_torch_wrote_back: bool
    internal_rl_report_published: bool
    internal_rl_torch_backends: tuple[str, ...]
    internal_rl_torch_parameters_changed: int
    internal_rl_torch_wrote_back: bool


@dataclass(frozen=True)
class EcologyActionProbe:
    kind: EcologyProbeKind
    left_sensor_pair: tuple[float, float]
    right_sensor_pair: tuple[float, float]
    left_code: tuple[float, ...]
    right_code: tuple[float, ...]
    left_turn: float
    right_turn: float
    code_l1_delta: float
    turn_delta: float
    input_reachable: bool
    action_sensitive: bool
    left_action_head_residual: tuple[float, ...] = ()
    right_action_head_residual: tuple[float, ...] = ()
    left_action_head_update_step: int = 0
    right_action_head_update_step: int = 0
    target_aligned: bool = True
    # --- the rest of the emitted action (plan 05:123 compares final code,
    # ACTION DISTRIBUTION and turn; turn alone is one of two motor DOFs and
    # says nothing about which abstract action the metacontroller selected) ---
    left_step: float = 0.0
    right_step: float = 0.0
    left_abstract_action: str = ""
    right_abstract_action: str = ""
    #: Owner-published action-family support, normalised to a distribution.
    left_action_distribution: tuple[tuple[str, float], ...] = ()
    right_action_distribution: tuple[tuple[str, float], ...] = ()
    # --- P0-A evidence payload (plan 05:96-103) ---
    left_sense: tuple[tuple[str, float], ...] = ()
    right_sense: tuple[tuple[str, float], ...] = ()
    left_posterior_hidden: EcologyPosteriorHiddenSummary | None = None
    right_posterior_hidden: EcologyPosteriorHiddenSummary | None = None
    posterior_hidden_l1_delta: float = 0.0
    motor_readout_gradient: tuple[float, ...] = ()
    latent_delta_l2_norm: float = 0.0
    predicted_turn_delta: float = 0.0
    null_space_alignment: float = 0.0
    in_motor_near_null_space: bool = False
    backend_lane: str = ""
    # What the session actually published for ECOLOGY_PROBE_LANE_WIRING_KEYS,
    # as opposed to the lane that was requested.
    observed_backend_wiring: tuple[tuple[str, str], ...] = ()
    # Owner-published proof of which declared backend actually executed. Both
    # sides of a pair must agree; the disagreement is a loud failure.
    backend_execution: EcologyBackendExecutionEvidence | None = None


@dataclass(frozen=True)
class EcologyCheckpointActionProbe:
    """Per-body action-chain evidence bound to one learning checkpoint."""

    body_id: int
    checkpoint_id: str
    policy_fingerprint: str
    temporal_learning_fingerprint: str
    probes: tuple[EcologyActionProbe, ...]
    backend_lane: str = ""
    probe_seed: int = 0
    observed_backend_wiring: tuple[tuple[str, str], ...] = ()
    backend_execution: EcologyBackendExecutionEvidence | None = None


def _paired_objects(
    kind: EcologyProbeKind,
) -> tuple[WorldObject, WorldObject]:
    if kind is EcologyProbeKind.FOOD:
        return (
            ButterSource(object_id="probe-butter", x=0.6, y=0.35),
            ButterSource(object_id="probe-butter", x=0.6, y=-0.35),
        )
    if kind is EcologyProbeKind.OBSTACLE:
        return (
            WoodStick(
                object_id="probe-stick",
                start_x=0.5,
                start_y=0.15,
                end_x=0.8,
                end_y=0.45,
            ),
            WoodStick(
                object_id="probe-stick",
                start_x=0.5,
                start_y=-0.15,
                end_x=0.8,
                end_y=-0.45,
            ),
        )
    if kind is EcologyProbeKind.HEAT:
        return (
            BurningMatch(
                object_id="probe-match",
                x=0.6,
                y=0.35,
                heat_decay=0.8,
            ),
            BurningMatch(
                object_id="probe-match",
                x=0.6,
                y=-0.35,
                heat_decay=0.8,
            ),
        )
    if kind is EcologyProbeKind.HOME:
        # Both lanes see identical geometry; only carrying state differs.
        shared = ButterSource(
            object_id="probe-home-butter",
            x=8.0,
            y=0.0,
        )
        return (shared, shared)
    raise ValueError(f"unsupported ecology probe kind: {kind!r}")


def _sensor_pair(
    *,
    kind: EcologyProbeKind,
    observation: object,
) -> tuple[float, float]:
    from volvence_ant.env.ant_world import WorldObservation

    if not isinstance(observation, WorldObservation):
        raise TypeError(
            "ecology probe observation must be a WorldObservation, "
            f"got {type(observation).__name__}"
        )
    if kind is EcologyProbeKind.FOOD:
        return (observation.food_left, observation.food_right)
    if kind is EcologyProbeKind.OBSTACLE:
        return (observation.obstacle_left, observation.obstacle_right)
    if kind is EcologyProbeKind.HEAT:
        return (observation.heat_left, observation.heat_right)
    if kind is EcologyProbeKind.HOME:
        return (
            float(observation.carrying_food),
            observation.eval_home_distance,
        )
    raise ValueError(f"unsupported ecology probe kind: {kind!r}")


def _lane_rollout_config(
    lane: EcologyProbeBackendLane | None,
) -> FinalRolloutConfig:
    """Pin the three backend lanes on top of the deployed evidence profile."""

    base = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    if lane is None:
        return base
    if lane is EcologyProbeBackendLane.PURE:
        return replace(
            base,
            temporal_runtime_backend=WiringLevel.DISABLED,
            temporal_ssl_backend=WiringLevel.DISABLED,
            internal_rl_backend=WiringLevel.DISABLED,
        )
    if lane is EcologyProbeBackendLane.RUNTIME:
        return replace(
            base,
            temporal_runtime_backend=WiringLevel.ACTIVE,
            temporal_ssl_backend=WiringLevel.DISABLED,
            internal_rl_backend=WiringLevel.DISABLED,
        )
    if lane is EcologyProbeBackendLane.TORCH:
        return replace(
            base,
            temporal_runtime_backend=WiringLevel.DISABLED,
            temporal_ssl_backend=WiringLevel.ACTIVE,
            internal_rl_backend=WiringLevel.ACTIVE,
        )
    raise ValueError(f"unsupported ecology probe backend lane: {lane!r}")


def ecology_probe_lane_expected_wiring(
    lane: EcologyProbeBackendLane,
) -> tuple[tuple[str, str], ...]:
    """The backend wiring a lane declares, for comparison against the session.

    ``_lane_rollout_config`` is what the probe asks for; this is the same
    request reduced to the three keys the session publishes back, so a caller
    can prove the lane it believes it measured is the lane that ran.
    """

    rollout = _lane_rollout_config(lane)
    declared = {
        "temporal_runtime_backend": rollout.temporal_runtime_backend.value,
        "temporal_ssl_backend": rollout.temporal_ssl_backend.value,
        "internal_rl_backend": rollout.internal_rl_backend.value,
    }
    return tuple(
        (name, declared[name]) for name in ECOLOGY_PROBE_LANE_WIRING_KEYS
    )


def ecology_probe_lane_declared_active_backends(
    lane: EcologyProbeBackendLane,
) -> tuple[str, ...]:
    """The wiring keys this lane turns ACTIVE, i.e. what it must demonstrate.

    ``PURE`` deliberately returns ``()``: it is the reference lane and its
    claim is the opposite one -- that no accelerated backend is engaged.
    """

    return tuple(
        name
        for name, level in ecology_probe_lane_expected_wiring(lane)
        if level == WiringLevel.ACTIVE.value
    )


def _observed_lane_wiring(
    backend_wiring: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    published = dict(backend_wiring)
    missing = tuple(
        name
        for name in ECOLOGY_PROBE_LANE_WIRING_KEYS
        if name not in published
    )
    if missing:
        raise KeyError(
            "the ant session stopped publishing backend wiring keys the "
            f"ecology probe lane contract depends on: {missing}"
        )
    return tuple(
        (name, published[name]) for name in ECOLOGY_PROBE_LANE_WIRING_KEYS
    )


def _posterior_hidden_summary(
    hidden: tuple[float, ...],
) -> EcologyPosteriorHiddenSummary:
    if not all(math.isfinite(value) for value in hidden):
        raise ValueError(
            "temporal owner published a non-finite posterior hidden state: "
            f"{hidden}"
        )
    return EcologyPosteriorHiddenSummary(
        dim=len(hidden),
        l1_norm=sum(abs(value) for value in hidden),
        l2_norm=math.sqrt(sum(value * value for value in hidden)),
        max_abs=max((abs(value) for value in hidden), default=0.0),
    )


def _read_world_runtime_state(session: AntSession):
    """Read the world temporal owner's published runtime state."""

    runtime_state = session.runner.world_temporal_policy.export_runtime_state()
    if runtime_state is None:
        raise RuntimeError(
            "ecology probe requires a learned temporal policy that publishes "
            "MetacontrollerRuntimeState; the configured world track published "
            "None"
        )
    return runtime_state


def _action_distribution(runtime_state) -> tuple[tuple[str, float], ...]:
    """Normalise the owner-published action-family support to a distribution.

    plan 05:123 requires parity over the final code, the ACTION DISTRIBUTION
    and the turn.  The metacontroller owns the family structure and publishes
    per-family ``support``; the probe only normalises it and never rebuilds it.
    """

    summaries = runtime_state.action_family_summaries
    total = sum(summary.support for summary in summaries)
    if total <= 0:
        return tuple((summary.family_id, 0.0) for summary in summaries)
    return tuple(
        (summary.family_id, summary.support / total) for summary in summaries
    )


def _backend_execution_evidence(
    session: AntSession,
    *,
    exercise_steps: int,
) -> EcologyBackendExecutionEvidence:
    """Read every owner's own statement about which backend actually ran."""

    joint_loop = session.runner.joint_loop
    ssl_report = joint_loop.latest_ssl_report
    rl_report = joint_loop.latest_internal_rl_report
    return EcologyBackendExecutionEvidence(
        exercise_steps=exercise_steps,
        temporal_runtime_backend_applied=(
            session.runner.world_temporal_policy.runtime_backend.value
        ),
        ssl_backend_applied=joint_loop.ssl_backend.value,
        internal_rl_backend_applied=joint_loop.internal_rl_backend.value,
        ssl_trained_steps=(
            ssl_report.trained_steps if ssl_report is not None else 0
        ),
        ssl_report_published=ssl_report is not None,
        ssl_torch_backends=(
            (ssl_report.torch_backend,) if ssl_report is not None else ()
        ),
        ssl_torch_parameters_changed=(
            ssl_report.torch_parameters_changed
            if ssl_report is not None
            else 0
        ),
        ssl_torch_wrote_back=(
            ssl_report.torch_wrote_back if ssl_report is not None else False
        ),
        internal_rl_report_published=rl_report is not None,
        internal_rl_torch_backends=(
            (rl_report.torch_backend,) if rl_report is not None else ()
        ),
        internal_rl_torch_parameters_changed=(
            rl_report.torch_parameters_changed if rl_report is not None else 0
        ),
        internal_rl_torch_wrote_back=(
            rl_report.torch_wrote_back if rl_report is not None else False
        ),
    )


def merge_ecology_backend_execution(
    items: Sequence[EcologyBackendExecutionEvidence],
) -> EcologyBackendExecutionEvidence:
    """Reduce per-side / per-kind evidence to one statement about a lane.

    The three ``*_applied`` fields are wiring readouts: every session a lane
    opens must report the same ones, and a difference is a wiring defect that
    raises rather than being averaged away.  Everything else is per-run
    telemetry, and the question coverage asks is "did this backend execute AT
    ALL under this lane", so counts take the maximum, flags the disjunction,
    and the ``*_torch_backends`` outcome sets are unioned so a lane that was
    skipped on one probe cannot look uniformly green.
    """

    if not items:
        raise ValueError(
            "backend execution evidence cannot be merged from an empty batch"
        )
    applied = {
        (
            item.exercise_steps,
            item.temporal_runtime_backend_applied,
            item.ssl_backend_applied,
            item.internal_rl_backend_applied,
        )
        for item in items
    }
    if len(applied) != 1:
        raise RuntimeError(
            "sessions opened for one backend lane reported different applied "
            f"wiring: {sorted(applied)}"
        )
    head = items[0]
    return EcologyBackendExecutionEvidence(
        exercise_steps=head.exercise_steps,
        temporal_runtime_backend_applied=(
            head.temporal_runtime_backend_applied
        ),
        ssl_backend_applied=head.ssl_backend_applied,
        internal_rl_backend_applied=head.internal_rl_backend_applied,
        ssl_trained_steps=max(item.ssl_trained_steps for item in items),
        ssl_report_published=any(item.ssl_report_published for item in items),
        ssl_torch_backends=tuple(
            sorted({name for item in items for name in item.ssl_torch_backends})
        ),
        ssl_torch_parameters_changed=max(
            item.ssl_torch_parameters_changed for item in items
        ),
        ssl_torch_wrote_back=any(item.ssl_torch_wrote_back for item in items),
        internal_rl_report_published=any(
            item.internal_rl_report_published for item in items
        ),
        internal_rl_torch_backends=tuple(
            sorted(
                {
                    name
                    for item in items
                    for name in item.internal_rl_torch_backends
                }
            )
        ),
        internal_rl_torch_parameters_changed=max(
            item.internal_rl_torch_parameters_changed for item in items
        ),
        internal_rl_torch_wrote_back=any(
            item.internal_rl_torch_wrote_back for item in items
        ),
    )


def _motor_turn(actuator: AntActuator, code: tuple[float, ...]) -> float:
    return actuator.plan(code, alarm=0.0).turn_command


def _motor_readout_gradient(
    actuator: AntActuator,
    code: tuple[float, ...],
) -> tuple[float, ...]:
    """Central finite difference of the frozen motor readout turn output.

    Read-only: it evaluates the frozen ``motor_decode`` plant on synthetic
    codes and never touches session, world or owner state.
    """

    epsilon = ECOLOGY_PROBE_FINITE_DIFFERENCE_EPSILON
    gradient: list[float] = []
    for index in range(len(code)):
        plus = list(code)
        minus = list(code)
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient.append(
            (
                _motor_turn(actuator, tuple(plus))
                - _motor_turn(actuator, tuple(minus))
            )
            / (2.0 * epsilon)
        )
    return tuple(gradient)


def _near_null_space_diagnostic(
    *,
    actuator: AntActuator,
    left_code: tuple[float, ...],
    right_code: tuple[float, ...],
) -> tuple[tuple[float, ...], float, float, float, bool]:
    midpoint = tuple(
        (left_value + right_value) / 2.0
        for left_value, right_value in zip(left_code, right_code, strict=True)
    )
    gradient = _motor_readout_gradient(actuator, midpoint)
    latent_delta = tuple(
        right_value - left_value
        for left_value, right_value in zip(left_code, right_code, strict=True)
    )
    predicted = sum(
        gradient_value * delta_value
        for gradient_value, delta_value in zip(
            gradient,
            latent_delta,
            strict=True,
        )
    )
    gradient_norm = math.sqrt(sum(value * value for value in gradient))
    delta_norm = math.sqrt(sum(value * value for value in latent_delta))
    denominator = gradient_norm * delta_norm
    alignment = abs(predicted) / denominator if denominator > 0.0 else 0.0
    return (
        gradient,
        delta_norm,
        predicted,
        alignment,
        alignment < ECOLOGY_PROBE_NEAR_NULL_SPACE_ALIGNMENT,
    )


@dataclass(frozen=True)
class _ProbeSideMeasurement:
    """One side of a paired probe, as measured on the final (gated) tick."""

    sensor_pair: tuple[float, float]
    code: tuple[float, ...]
    turn: float
    step: float
    abstract_action: str
    action_distribution: tuple[tuple[str, float], ...]
    action_head_residual: tuple[float, ...]
    action_head_update_step: int
    sense: tuple[tuple[str, float], ...]
    posterior_hidden: tuple[float, ...]
    lane_wiring: tuple[tuple[str, str], ...]
    backend_execution: EcologyBackendExecutionEvidence


def _pin_probe_pose(
    *,
    world: AntWorld,
    kind: EcologyProbeKind,
    side_index: int,
) -> None:
    """Re-assert the declared paired stimulus geometry for one tick.

    Every tick of a probe -- exercise ticks included -- starts from the exact
    same body pose, so the only thing that ever differs between the two sides
    is the object side, and the only thing that differs between two backend
    lanes is the backend.
    """

    if kind is EcologyProbeKind.HOME:
        world.set_body_pose(
            x=2.0,
            y=0.0,
            heading=math.pi / 2.0,
            carrying_food=bool(side_index),
        )
        return
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)


async def run_ecology_action_probes(
    *,
    temporal_latent_dim: int,
    seed: int,
    checkpoint: AntLearningCheckpoint | None = None,
    code_delta_threshold: float = 1e-8,
    turn_delta_threshold: float = 1e-8,
    backend_lane: EcologyProbeBackendLane | None = None,
    exercise_steps: int = 0,
) -> tuple[EcologyActionProbe, ...]:
    """Run the deterministic paired probes for one checkpoint.

    ``exercise_steps`` prepends that many optimization-carrying ticks at the
    identical pinned pose before the measured tick.  It exists for the backend
    parity lanes: ``temporal_ssl_backend`` / ``internal_rl_backend`` only do
    work inside an optimization cycle, so a probe that measures the very first
    tick can never observe them and every lane looks bit-identical to the pure
    reference -- an artefact of the probe, not evidence about the backends.

    It defaults to ``0`` so the action-chain gates (sensitivity, retention,
    sign consistency, lateral bias) keep measuring the restored checkpoint
    itself rather than a checkpoint plus a few probe-local updates.
    """

    if exercise_steps < 0:
        raise ValueError(
            f"exercise_steps must be >= 0, got {exercise_steps!r}"
        )
    probes: list[EcologyActionProbe] = []
    rollout_config = _lane_rollout_config(backend_lane)
    for kind in EcologyProbeKind:
        objects = _paired_objects(kind)
        paired_records: list[_ProbeSideMeasurement] = []
        actuator: AntActuator | None = None
        for side_index, world_object in enumerate(objects):
            world = AntWorld(
                config=AntWorldConfig(
                    seed=seed,
                    step_size=0.4,
                ),
                world_objects=(world_object,),
            )
            session = AntSession(
                world,
                config=AntSessionConfig(
                    temporal_latent_dim=temporal_latent_dim,
                    session_id=(
                        f"ecology-probe:{kind.value}:side:{side_index}:seed:{seed}"
                    ),
                    seed=seed,
                    heading_noise=0.0,
                    step_noise=0.0,
                    rollout_config=rollout_config,
                    objective=AntObjectiveKind.ECOLOGY,
                    sense_schema=AntSenseSchema.ECOLOGY_V2,
                ),
            )
            if checkpoint is not None:
                session.restore_learning_checkpoint(checkpoint)
            if kind is EcologyProbeKind.HOME:
                for _ in range(5):
                    session.navigator.update(
                        turn_command=0.0,
                        step_command=0.4,
                        true_heading=0.0,
                    )
                session.navigator.update(
                    turn_command=math.pi / 2.0,
                    step_command=0.0,
                    true_heading=math.pi / 2.0,
                )
            for _ in range(exercise_steps):
                _pin_probe_pose(world=world, kind=kind, side_index=side_index)
                await session.step()
            _pin_probe_pose(world=world, kind=kind, side_index=side_index)
            observation = world.observe()
            record = await session.step()
            actuator = session.actuator
            runtime_state = _read_world_runtime_state(session)
            paired_records.append(
                _ProbeSideMeasurement(
                    sensor_pair=_sensor_pair(kind=kind, observation=observation),
                    code=record.code,
                    turn=record.command.turn_command,
                    step=record.command.step_command,
                    abstract_action=record.abstract_action,
                    action_distribution=_action_distribution(runtime_state),
                    action_head_residual=record.causal_action_head_residual,
                    action_head_update_step=(
                        record.causal_action_head_update_step
                    ),
                    sense=record.sense_activation,
                    posterior_hidden=runtime_state.posterior_hidden_state,
                    lane_wiring=_observed_lane_wiring(record.backend_wiring),
                    backend_execution=_backend_execution_evidence(
                        session,
                        exercise_steps=exercise_steps,
                    ),
                )
            )
        if actuator is None:
            raise RuntimeError(
                "ecology probe produced no session for kind "
                f"{kind.value}"
            )
        left, right = paired_records
        if left.lane_wiring != right.lane_wiring:
            raise RuntimeError(
                "the two lanes of a paired ecology probe ran on different "
                f"backend wiring: left={left.lane_wiring} "
                f"right={right.lane_wiring}"
            )
        paired_execution = merge_ecology_backend_execution(
            (left.backend_execution, right.backend_execution)
        )
        code_l1_delta = sum(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left.code,
                right.code,
                strict=True,
            )
        )
        turn_delta = abs(left.turn - right.turn)
        input_reachable = (
            left.sensor_pair != right.sensor_pair
            and code_l1_delta > code_delta_threshold
        )
        if kind is EcologyProbeKind.FOOD:
            target_aligned = (
                left.turn > turn_delta_threshold
                and right.turn < -turn_delta_threshold
            )
        elif kind is EcologyProbeKind.HEAT:
            target_aligned = (
                left.turn < -turn_delta_threshold
                and right.turn > turn_delta_threshold
            )
        elif kind is EcologyProbeKind.HOME:
            # At (2, 0), heading north, home lies to the left (+turn).
            # The carrying lane is the right-hand member of this pair.
            target_aligned = right.turn > turn_delta_threshold
        else:
            # Neutral obstacle geometry is reachability evidence, not a
            # valenced steering target.
            target_aligned = True
        (
            motor_gradient,
            latent_delta_norm,
            predicted_turn_delta,
            null_space_alignment,
            in_near_null_space,
        ) = _near_null_space_diagnostic(
            actuator=actuator,
            left_code=left.code,
            right_code=right.code,
        )
        left_hidden = _posterior_hidden_summary(left.posterior_hidden)
        right_hidden = _posterior_hidden_summary(right.posterior_hidden)
        probes.append(
            EcologyActionProbe(
                kind=kind,
                left_sensor_pair=left.sensor_pair,
                right_sensor_pair=right.sensor_pair,
                left_code=left.code,
                right_code=right.code,
                left_turn=left.turn,
                right_turn=right.turn,
                code_l1_delta=code_l1_delta,
                turn_delta=turn_delta,
                input_reachable=input_reachable,
                action_sensitive=input_reachable
                and turn_delta > turn_delta_threshold,
                left_action_head_residual=left.action_head_residual,
                right_action_head_residual=right.action_head_residual,
                left_action_head_update_step=left.action_head_update_step,
                right_action_head_update_step=right.action_head_update_step,
                target_aligned=target_aligned,
                left_step=left.step,
                right_step=right.step,
                left_abstract_action=left.abstract_action,
                right_abstract_action=right.abstract_action,
                left_action_distribution=left.action_distribution,
                right_action_distribution=right.action_distribution,
                left_sense=left.sense,
                right_sense=right.sense,
                left_posterior_hidden=left_hidden,
                right_posterior_hidden=right_hidden,
                posterior_hidden_l1_delta=sum(
                    abs(left_value - right_value)
                    for left_value, right_value in zip(
                        left.posterior_hidden,
                        right.posterior_hidden,
                        strict=True,
                    )
                ),
                motor_readout_gradient=motor_gradient,
                latent_delta_l2_norm=latent_delta_norm,
                predicted_turn_delta=predicted_turn_delta,
                null_space_alignment=null_space_alignment,
                in_motor_near_null_space=in_near_null_space,
                backend_lane=(
                    backend_lane.value if backend_lane is not None else ""
                ),
                observed_backend_wiring=left.lane_wiring,
                backend_execution=paired_execution,
            )
        )
    return tuple(probes)


async def run_ecology_checkpoint_action_probes(
    *,
    temporal_latent_dim: int,
    seed: int,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    code_delta_threshold: float = 1e-8,
    turn_delta_threshold: float = 1e-4,
    backend_lane: EcologyProbeBackendLane | None = None,
    exercise_steps: int = 0,
) -> tuple[EcologyCheckpointActionProbe, ...]:
    """Run deterministic paired probes for every isolated colony body."""

    reports: list[EcologyCheckpointActionProbe] = []
    for body_id, checkpoint in enumerate(checkpoints):
        probes = await run_ecology_action_probes(
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
            checkpoint=checkpoint,
            code_delta_threshold=code_delta_threshold,
            turn_delta_threshold=turn_delta_threshold,
            backend_lane=backend_lane,
            exercise_steps=exercise_steps,
        )
        observed_wiring = {probe.observed_backend_wiring for probe in probes}
        if len(observed_wiring) != 1:
            raise RuntimeError(
                "probes for one checkpoint ran on inconsistent backend "
                f"wiring: {sorted(observed_wiring)}"
            )
        executions = tuple(
            probe.backend_execution
            for probe in probes
            if probe.backend_execution is not None
        )
        if len(executions) != len(probes):
            raise RuntimeError(
                "an ecology probe was built without backend execution "
                "evidence; coverage cannot be established without it"
            )
        reports.append(
            EcologyCheckpointActionProbe(
                body_id=body_id,
                checkpoint_id=checkpoint.checkpoint_id,
                policy_fingerprint=checkpoint.policy_fingerprint,
                temporal_learning_fingerprint=(
                    checkpoint.temporal_learning_fingerprint
                ),
                probes=probes,
                backend_lane=(
                    backend_lane.value if backend_lane is not None else ""
                ),
                probe_seed=seed,
                observed_backend_wiring=next(iter(observed_wiring)),
                backend_execution=merge_ecology_backend_execution(executions),
            )
        )
    return tuple(reports)


__all__ = [
    "ECOLOGY_PROBE_FINITE_DIFFERENCE_EPSILON",
    "ECOLOGY_PROBE_LANE_WIRING_KEYS",
    "ECOLOGY_PROBE_NEAR_NULL_SPACE_ALIGNMENT",
    "EcologyActionProbe",
    "EcologyBackendExecutionEvidence",
    "EcologyCheckpointActionProbe",
    "EcologyPosteriorHiddenSummary",
    "EcologyProbeBackendLane",
    "EcologyProbeKind",
    "ecology_probe_lane_declared_active_backends",
    "ecology_probe_lane_expected_wiring",
    "merge_ecology_backend_execution",
    "run_ecology_checkpoint_action_probes",
    "run_ecology_action_probes",
]
