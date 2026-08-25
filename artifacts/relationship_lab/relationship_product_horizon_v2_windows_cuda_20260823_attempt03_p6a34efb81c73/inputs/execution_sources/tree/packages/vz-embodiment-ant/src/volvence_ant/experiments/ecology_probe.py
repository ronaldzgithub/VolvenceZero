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

# Frozen sensor geometry of the ecology body.
#
# EVERY curriculum world -- training, validation and held-out alike -- is built
# with ``antenna_offset_deg=45.0, antenna_reach=0.9``.  The paired probe used to
# fall back to the ``AntWorldConfig`` defaults (30 deg / 0.6), so three P1 gates
# and the whole P0 action-chain audit graded the learned policy on a sensory
# body that no world it was trained or evaluated in ever had: an
# off-distribution measurement presented as the frozen sensitivity truth.
#
# The curriculum world owns this geometry; these constants restate it for the
# probe, which cannot import the curriculum (the curriculum imports the probe).
# ``test_ecology_p1.test_probe_world_matches_curriculum_sensor_geometry``
# constructs a real curriculum world and asserts equality, so any future drift
# on either side fails loudly instead of silently reopening the gap.
ECOLOGY_PROBE_ANTENNA_OFFSET_DEG = 45.0
ECOLOGY_PROBE_ANTENNA_REACH = 0.9
ECOLOGY_PROBE_STEP_SIZE = 0.4
ECOLOGY_PROBE_NEST_RADIUS = 0.5

# Frozen post-pickup U-turn gate. Sixteen ticks leave ample room for the
# graded plant to correct a 3*pi/4 heading error (three acts at its pi/4
# ceiling) and then demonstrate more than a one-tick directional coincidence.
ECOLOGY_POST_PICKUP_UTURN_HORIZON = 16
ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET = 3.0 * math.pi / 4.0
ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS = ECOLOGY_PROBE_STEP_SIZE
ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS = 3
# The action that resolves pickup was selected from the pre-pickup
# observation. The first two subsequent actions are the bounded window in
# which the persisted action family must acknowledge carrying_food=True.
ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY = 2
ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS = 3
_ECOLOGY_POST_PICKUP_UTURN_START_DISTANCE = 2.0
_ECOLOGY_POST_PICKUP_UTURN_SOURCE_RADIUS = 0.7


def ecology_probe_world_config(*, seed: int) -> AntWorldConfig:
    """The world config every ecology probe must run in.

    Published so consumers (and the drift test) read the same single statement
    of the probe's sensory body instead of re-deriving it from literals.
    """

    return AntWorldConfig(
        seed=seed,
        step_size=ECOLOGY_PROBE_STEP_SIZE,
        antenna_offset_deg=ECOLOGY_PROBE_ANTENNA_OFFSET_DEG,
        antenna_reach=ECOLOGY_PROBE_ANTENNA_REACH,
        nest_radius=ECOLOGY_PROBE_NEST_RADIUS,
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
class EcologyPostPickupUTurnLane:
    """One frozen large-angle return lane after a real pickup transition."""

    side: str
    heading_offset: float
    picked_up: bool
    pickup_tick: int | None
    delivered: bool
    delivery_tick: int | None
    home_distances_after_pickup: tuple[float, ...]
    turn_commands_after_pickup: tuple[float, ...]
    switch_steps_after_pickup: tuple[int, ...]
    first_post_pickup_switch_step: int | None
    post_pickup_switch_observed: bool
    action_families_after_pickup: tuple[str, ...]
    first_post_pickup_switch_family: str | None
    post_switch_family_survival_actions: int
    post_switch_family_observation_censored: bool
    net_home_progress: float
    max_consecutive_approach_steps: int
    policy_fingerprint_stable: bool
    temporal_learning_fingerprint_stable: bool
    passed: bool


@dataclass(frozen=True)
class EcologyCheckpointPostPickupUTurnProbe:
    """Frozen ±135-degree U-turn evidence for one isolated body checkpoint."""

    body_id: int
    checkpoint_id: str
    lanes: tuple[EcologyPostPickupUTurnLane, ...]
    passed: bool


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


def _antenna_sample_offset() -> tuple[float, float]:
    """Where the left antenna samples for the pinned probe pose.

    ``_pin_probe_pose`` puts every non-HOME probe body at the origin facing
    +x, so the left antenna samples at ``(+dx, +dy)`` and the right one at
    ``(+dx, -dy)`` for the frozen geometry below.
    """

    phi = math.radians(ECOLOGY_PROBE_ANTENNA_OFFSET_DEG)
    return (
        ECOLOGY_PROBE_ANTENNA_REACH * math.cos(phi),
        ECOLOGY_PROBE_ANTENNA_REACH * math.sin(phi),
    )


def _paired_objects(
    kind: EcologyProbeKind,
) -> tuple[WorldObject, WorldObject]:
    if kind is EcologyProbeKind.FOOD:
        return (
            ButterSource(object_id="probe-butter", x=0.6, y=0.35),
            ButterSource(object_id="probe-butter", x=0.6, y=-0.35),
        )
    if kind is EcologyProbeKind.OBSTACLE:
        # ``obstacle_left/right`` is a BINARY containment test at the antenna
        # sample point, so unlike the smooth food/heat fields the stick has to
        # physically cover one antenna and miss the other.  The old literal
        # capsule was placed for the 30 deg / 0.6 default antennae; under the
        # frozen curriculum geometry both antennae fall outside it and the
        # probe reports ``input_reachable=False`` -- a broken instrument, not a
        # broken policy.  Derive the capsule from the antenna sample point so
        # the stimulus can never drift away from the sensor again.
        offset_x, offset_y = _antenna_sample_offset()
        half_length = 0.2
        return (
            WoodStick(
                object_id="probe-stick",
                start_x=offset_x - half_length,
                start_y=offset_y,
                end_x=offset_x + half_length,
                end_y=offset_y,
            ),
            WoodStick(
                object_id="probe-stick",
                start_x=offset_x - half_length,
                start_y=-offset_y,
                end_x=offset_x + half_length,
                end_y=-offset_y,
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
    session: AntSession,
    kind: EcologyProbeKind,
    side_index: int,
) -> None:
    """Re-assert the declared paired sensorimotor state for one tick.

    Every tick of a probe -- exercise ticks included -- starts from the same
    physical pose *and* navigator estimate.  Pinning only ``AntWorld`` leaves
    ``h_hat`` at the seed-dependent random spawn heading, so repeated probes
    silently compare different owner inputs even though the body coordinates
    look identical.
    """

    world = session.world
    if kind is EcologyProbeKind.HOME:
        world.set_body_pose(
            x=2.0,
            y=0.0,
            heading=math.pi / 2.0,
            carrying_food=bool(side_index),
        )
        session.navigator.sync_to(
            x=2.0,
            y=0.0,
            heading=math.pi / 2.0,
            nest=world.nest,
        )
        return
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    session.navigator.sync_to(
        x=0.0,
        y=0.0,
        heading=0.0,
        nest=world.nest,
    )


async def run_ecology_action_probes(
    *,
    temporal_latent_dim: int,
    seed: int,
    checkpoint: AntLearningCheckpoint | None = None,
    code_delta_threshold: float = 1e-8,
    turn_delta_threshold: float = 1e-8,
    backend_lane: EcologyProbeBackendLane | None = None,
    exercise_steps: int = 0,
    rollout_config: FinalRolloutConfig | None = None,
    learning_enabled: bool = True,
) -> tuple[EcologyActionProbe, ...]:
    """Run the deterministic paired probes for one checkpoint.

    ``exercise_steps`` prepends that many optimization-carrying ticks at the
    identical pinned pose before the measured tick. It exists for backend
    coverage: ``temporal_ssl_backend`` / ``internal_rl_backend`` only do work
    inside an optimization cycle. After coverage is captured, checkpointed
    parity lanes restore the same owner state before the measured forward;
    otherwise the probe would compare different learned parameters and call
    optimizer algorithm drift a runtime-forward mismatch.

    It defaults to ``0`` so the action-chain gates (sensitivity, retention,
    sign consistency, lateral bias) keep measuring the restored checkpoint
    itself rather than a checkpoint plus a few probe-local updates.

    ``learning_enabled=False`` propagates the runtime's hard no-write boundary
    to every ephemeral probe session.  Read-only checkpoint prechecks use it;
    backend coverage keeps the default because its purpose is to prove the
    declared optimizers execute and write back.
    """

    if exercise_steps < 0:
        raise ValueError(
            f"exercise_steps must be >= 0, got {exercise_steps!r}"
        )
    if rollout_config is not None and backend_lane is not None:
        raise ValueError(
            "an explicit rollout_config cannot be combined with a backend_lane"
        )
    probes: list[EcologyActionProbe] = []
    resolved_rollout_config = (
        rollout_config
        if rollout_config is not None
        else _lane_rollout_config(backend_lane)
    )
    for kind in EcologyProbeKind:
        objects = _paired_objects(kind)
        paired_records: list[_ProbeSideMeasurement] = []
        actuator: AntActuator | None = None
        for side_index, world_object in enumerate(objects):
            world = AntWorld(
                config=ecology_probe_world_config(seed=seed),
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
                    rollout_config=resolved_rollout_config,
                    joint_apply_policy_optimization=learning_enabled,
                    joint_learning_enabled=learning_enabled,
                    objective=AntObjectiveKind.ECOLOGY,
                    sense_schema=AntSenseSchema.ECOLOGY_V2,
                ),
            )
            if checkpoint is not None:
                session.restore_learning_checkpoint(checkpoint)
            coverage_execution: EcologyBackendExecutionEvidence | None = None
            for _ in range(exercise_steps):
                _pin_probe_pose(
                    session=session,
                    kind=kind,
                    side_index=side_index,
                )
                await session.step()
            if (
                checkpoint is not None
                and backend_lane is not None
                and exercise_steps > 0
            ):
                coverage_execution = _backend_execution_evidence(
                    session,
                    exercise_steps=exercise_steps,
                )
                # Learning checkpoints intentionally contain adaptive owner
                # state, not serving recurrent/session context. Measure the
                # same-checkpoint forward in a fresh session so exercise-local
                # state cannot masquerade as backend numerical drift.
                world = AntWorld(
                    config=ecology_probe_world_config(seed=seed),
                    world_objects=(_paired_objects(kind)[side_index],),
                )
                session = AntSession(
                    world,
                    config=AntSessionConfig(
                        temporal_latent_dim=temporal_latent_dim,
                        session_id=(
                            f"ecology-probe:{kind.value}:side:{side_index}:"
                            f"seed:{seed}:measured"
                        ),
                        seed=seed,
                        heading_noise=0.0,
                        step_noise=0.0,
                        rollout_config=resolved_rollout_config,
                        joint_apply_policy_optimization=learning_enabled,
                        joint_learning_enabled=learning_enabled,
                        objective=AntObjectiveKind.ECOLOGY,
                        sense_schema=AntSenseSchema.ECOLOGY_V2,
                    ),
                )
                session.restore_learning_checkpoint(checkpoint)
            _pin_probe_pose(
                session=session,
                kind=kind,
                side_index=side_index,
            )
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
                    backend_execution=(
                        coverage_execution
                        or _backend_execution_evidence(
                            session,
                            exercise_steps=exercise_steps,
                        )
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
    rollout_config: FinalRolloutConfig | None = None,
    learning_enabled: bool = True,
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
            rollout_config=rollout_config,
            learning_enabled=learning_enabled,
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


def _max_consecutive_approach_steps(
    distances: Sequence[float],
) -> int:
    longest = 0
    current = 0
    for previous, following in zip(
        distances,
        distances[1:],
        strict=False,
    ):
        if following < previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _post_switch_family_survival(
    *,
    action_families: Sequence[str],
    switch_steps: Sequence[int],
) -> tuple[str | None, int, bool]:
    """Count the exact selected family after the first post-pickup switch.

    ``action_families`` starts at the first action selected from the carrying
    observation; switch steps are one-based from the pickup action. The switch
    action itself is the first survival action. A second beta switch ends the
    run even if the owner selects the same family label again: that is the
    t+2 re-selection/back-jump D4 is meant to detect. A run with no second
    switch that keeps the exact family through the frozen lane is
    right-censored rather than pretending it ended at the probe horizon.
    """

    if not switch_steps:
        return None, 0, False
    first_switch_step = switch_steps[0]
    family_index = first_switch_step - 1
    if family_index < 0 or family_index >= len(action_families):
        raise ValueError(
            "post-pickup switch step is outside the published action-family "
            f"trace: step={first_switch_step}, trace={len(action_families)}"
        )
    family = action_families[family_index]
    survival = 0
    next_switch_step = next(
        (step for step in switch_steps[1:] if step > first_switch_step),
        None,
    )
    family_limit = (
        len(action_families)
        if next_switch_step is None
        else next_switch_step - 1
    )
    for selected in action_families[family_index:family_limit]:
        if selected != family:
            break
        survival += 1
    censored = (
        next_switch_step is None
        and family_index + survival == len(action_families)
    )
    return family, survival, censored


async def _run_post_pickup_uturn_lane(
    *,
    checkpoint: AntLearningCheckpoint,
    temporal_latent_dim: int,
    seed: int,
    side: int,
) -> EcologyPostPickupUTurnLane:
    if side not in {-1, 1}:
        raise ValueError(f"U-turn lane side must be -1 or 1, got {side}")
    start_x = _ECOLOGY_POST_PICKUP_UTURN_START_DISTANCE
    start_y = 0.0
    home_bearing = math.pi
    heading = (
        home_bearing
        - side * ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET
    )
    world = AntWorld(
        config=ecology_probe_world_config(seed=seed),
        world_objects=(
            ButterSource(
                object_id="post-pickup-uturn-butter",
                x=start_x,
                y=start_y,
                strength=2.2,
                decay=2.4,
                radius=_ECOLOGY_POST_PICKUP_UTURN_SOURCE_RADIUS,
            ),
        ),
    )
    world.set_body_pose(
        x=start_x,
        y=start_y,
        heading=heading,
        carrying_food=False,
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=temporal_latent_dim,
            session_id=f"post-pickup-uturn:{seed}:{side}",
            seed=seed,
            heading_noise=0.0,
            step_noise=0.0,
            rollout_config=ant_runtime_replay_rollout_config(
                enable_sparse_exploration=False,
                sense_schema=AntSenseSchema.ECOLOGY_V2,
            ),
            joint_apply_policy_optimization=False,
            joint_learning_enabled=False,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    session.restore_learning_checkpoint(checkpoint)
    session.navigator.sync_to(
        x=start_x,
        y=start_y,
        heading=heading,
        nest=world.nest,
    )

    pickup_tick: int | None = None
    pickup_step: int | None = None
    delivery_tick: int | None = None
    distances: list[float] = []
    turns: list[float] = []
    switch_steps: list[int] = []
    action_families: list[str] = []
    for step_index in range(ECOLOGY_POST_PICKUP_UTURN_HORIZON):
        record = await session.step()
        if record.picked_up and pickup_tick is None:
            pickup_tick = record.tick
            pickup_step = step_index
        if pickup_tick is not None:
            distances.append(
                math.hypot(
                    record.x - world.nest[0],
                    record.y - world.nest[1],
                )
            )
            turns.append(record.command.turn_command)
        if pickup_step is not None and step_index > pickup_step:
            action_families.append(record.abstract_action)
        # Do not count the pickup act itself: its action was selected from the
        # unladen observation. Only a later switch can prove that the policy
        # reacted to the carrying-state transition.
        if (
            pickup_step is not None
            and step_index > pickup_step
            and record.is_switching
        ):
            switch_steps.append(step_index - pickup_step)
        if record.delivered:
            delivery_tick = record.tick
            break

    post = session.export_learning_checkpoint(
        checkpoint_id=f"post-pickup-uturn:{seed}:{side}:post",
        include_runtime_replay=False,
    )
    policy_stable = (
        post.policy_fingerprint == checkpoint.policy_fingerprint
    )
    temporal_stable = (
        post.temporal_learning_fingerprint
        == checkpoint.temporal_learning_fingerprint
    )
    net_progress = (
        distances[0] - distances[-1]
        if len(distances) >= 2
        else 0.0
    )
    consecutive = _max_consecutive_approach_steps(distances)
    delivered = delivery_tick is not None
    first_switch_step = switch_steps[0] if switch_steps else None
    (
        first_switch_family,
        family_survival_actions,
        family_observation_censored,
    ) = _post_switch_family_survival(
        action_families=action_families,
        switch_steps=switch_steps,
    )
    prompt_switch = (
        first_switch_step is not None
        and first_switch_step
        <= ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY
    )
    passed = (
        pickup_tick is not None
        and prompt_switch
        and family_survival_actions
        >= ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS
        and policy_stable
        and temporal_stable
        and (
            delivered
            or (
                net_progress
                >= ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS
                and consecutive
                >= ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS
            )
        )
    )
    return EcologyPostPickupUTurnLane(
        side="left" if side > 0 else "right",
        heading_offset=(
            side * ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET
        ),
        picked_up=pickup_tick is not None,
        pickup_tick=pickup_tick,
        delivered=delivered,
        delivery_tick=delivery_tick,
        home_distances_after_pickup=tuple(distances),
        turn_commands_after_pickup=tuple(turns),
        switch_steps_after_pickup=tuple(switch_steps),
        first_post_pickup_switch_step=first_switch_step,
        post_pickup_switch_observed=prompt_switch,
        action_families_after_pickup=tuple(action_families),
        first_post_pickup_switch_family=first_switch_family,
        post_switch_family_survival_actions=family_survival_actions,
        post_switch_family_observation_censored=(
            family_observation_censored
        ),
        net_home_progress=net_progress,
        max_consecutive_approach_steps=consecutive,
        policy_fingerprint_stable=policy_stable,
        temporal_learning_fingerprint_stable=temporal_stable,
        passed=passed,
    )


async def run_ecology_checkpoint_post_pickup_uturn_probes(
    *,
    temporal_latent_dim: int,
    seed: int,
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[EcologyCheckpointPostPickupUTurnProbe, ...]:
    """Grade sustained frozen homing after a real pickup, on both turn sides."""

    reports: list[EcologyCheckpointPostPickupUTurnProbe] = []
    for body_id, checkpoint in enumerate(checkpoints):
        lane_rows: list[EcologyPostPickupUTurnLane] = []
        for side in (1, -1):
            lane_rows.append(
                await _run_post_pickup_uturn_lane(
                    checkpoint=checkpoint,
                    temporal_latent_dim=temporal_latent_dim,
                    seed=seed,
                    side=side,
                )
            )
        lanes = tuple(lane_rows)
        reports.append(
            EcologyCheckpointPostPickupUTurnProbe(
                body_id=body_id,
                checkpoint_id=checkpoint.checkpoint_id,
                lanes=lanes,
                passed=all(lane.passed for lane in lanes),
            )
        )
    return tuple(reports)


__all__ = [
    "ECOLOGY_PROBE_ANTENNA_OFFSET_DEG",
    "ECOLOGY_PROBE_ANTENNA_REACH",
    "ECOLOGY_PROBE_FINITE_DIFFERENCE_EPSILON",
    "ECOLOGY_PROBE_LANE_WIRING_KEYS",
    "ECOLOGY_PROBE_NEAR_NULL_SPACE_ALIGNMENT",
    "ECOLOGY_PROBE_NEST_RADIUS",
    "ECOLOGY_PROBE_STEP_SIZE",
    "ECOLOGY_POST_PICKUP_UTURN_HEADING_OFFSET",
    "ECOLOGY_POST_PICKUP_UTURN_HORIZON",
    "ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS",
    "ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS",
    "ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY",
    "ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS",
    "EcologyActionProbe",
    "EcologyBackendExecutionEvidence",
    "EcologyCheckpointActionProbe",
    "EcologyCheckpointPostPickupUTurnProbe",
    "EcologyPostPickupUTurnLane",
    "EcologyPosteriorHiddenSummary",
    "EcologyProbeBackendLane",
    "EcologyProbeKind",
    "ecology_probe_lane_declared_active_backends",
    "ecology_probe_lane_expected_wiring",
    "ecology_probe_world_config",
    "merge_ecology_backend_execution",
    "run_ecology_checkpoint_action_probes",
    "run_ecology_checkpoint_post_pickup_uturn_probes",
    "run_ecology_action_probes",
]
