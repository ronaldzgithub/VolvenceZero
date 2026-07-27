"""Offline ecology curriculum over the real ant runtime.

No action labels or steering hints are supplied.  Every episode runs the same
``AntSession`` sense -> temporal -> action -> outcome -> PE -> credit path used
by the live app, then carries owner-exported checkpoints into the next world.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable

import numpy as np

from volvence_zero.agent import AgentLearningArchiveError

from volvence_ant.env.ant_world import AntWorld, AntWorldConfig
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.world_objects import (
    BurningMatch,
    ButterSource,
    WoodStick,
    WorldObject,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyActionProbe,
    EcologyCheckpointActionProbe,
    EcologyProbeKind,
    run_ecology_action_probes,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntStepRecord,
    AntSessionConfig,
    KernelColonyRunner,
)


# v9 changes what MASTERY MEANS again, so a v8 report may no more be read
# under it than a v7 one:
#
# * ``ecology_tier_round_budget`` is now a SUFFICIENT worst case -- it charges
#   the re-aiming legs the frozen plant makes every body pay (near 12 -> 28,
#   medium 16 -> 33, far 32 -> 49 rounds).  The v8 numbers were neither a
#   sufficient budget nor a valid necessary one, so every v8
#   ``milestone_round_budget`` / ``milestone_samplable`` field is wrong;
# * a layout whose episode could not reach its own milestone is no longer
#   mastery-eligible, so a v8 stage verdict can include layouts a v9 verdict
#   excludes;
# * ``EcologyCurriculumConfig`` gained ``milestone_budget_enforced``, which is
#   part of every config digest.
#
# v8 retired the pre-P1 semantics wholesale, and the promotion loader
# (``volvence_ant.evidence.ecology_checkpoint._validated_report_verdict``)
# must keep rejecting anything older rather than mix algorithm generations:
#
# * mastery moved from stage-cumulative event counts to per-layout /
#   per-body success rate (plan section 4.2);
# * near foraging layouts are Stage-0 bootstrap and can no longer satisfy any
#   capability gate (plan section 4.3);
# * ``local-valence-off`` is published as ``dense-local-shaping-off``;
# * the tier geometry changed (near pickup disc and nest delivery disc), so
#   every pickup/delivery count in a v7 report was produced in a different
#   world.
ECOLOGY_CURRICULUM_SCHEMA_VERSION = "digital-ant-ecology-curriculum.v9"
ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY = 8192
ECOLOGY_REQUIRED_GATE_NAMES = (
    "training_layout_mastery",
    "training_tier_milestone_samplable",
    "paired_action_sensitivity",
    "policy_changed",
    "no_optimize_policy_stable",
    "butter_pickup_gain",
    "butter_delivery_present",
    "neutral_stick_context_robustness",
    "burning_match_route_avoidance",
    "burning_match_forced_escape",
    "composite_performance",
    "matched_ablation_advantage",
    "temporal_dynamics",
    "action_smoothness",
    "checkpoint_archive_roundtrip",
    "runtime_replay_lineage",
)
# Wood sticks are neutral physical geometry: they must stay perceivable
# (obstacle_left/right), but obstacle_contact activation is NOT required --
# forcing collisions to pass a gate would contradict stick neutrality.
ECOLOGY_CRITICAL_ACTIVE_CHANNELS = frozenset(
    {
        "food_left",
        "food_right",
        "food_diff",
        "carrying_food",
        "obstacle_left",
        "obstacle_right",
        "heat_left",
        "heat_right",
        "heat_diff",
        "heat_center",
        "heat_harmful",
    }
)


class EcologyStage(str, Enum):
    """Scene layouts; WOOD_STICK names a butter+neutral-stick layout only."""

    BUTTER = "butter"
    WOOD_STICK = "wood_stick"
    BURNING_MATCH = "burning_match"
    COMPOSITE = "composite"


# Valenced training schedule: butter (appetitive), burning match (aversive)
# and the composite layout. The neutral wood stick is never a training stage
# of its own -- it only appears as physical context inside layouts.
ECOLOGY_TRAINING_STAGES = (
    EcologyStage.BUTTER,
    EcologyStage.BURNING_MATCH,
    EcologyStage.COMPOSITE,
)


class EcologyTrainingTier(str, Enum):
    NEAR = "near"
    MEDIUM = "medium"
    FAR = "far"


# --------------------------------------------------------------------------
# Frozen curriculum geometry
# --------------------------------------------------------------------------
# The ecology worlds all share one plant.  Every derived quantity below reads
# that plant instead of restating its numbers, so a substrate change cannot
# silently invalidate the tier contract.
ECOLOGY_NEST_RADIUS = 0.5
# Layout-geometry invariants (see ``_validate_tier_geometry``).  A pickup
# disc that reaches the delivery disc is solved by standing still, and a
# pickup disc inside a fixed maximum-curvature circle is solved by undirected
# spinning; both make the tier's own numbers uninterpretable.
ECOLOGY_MIN_DELIVERY_LEG = 0.25
# Audit-measured mean applied step of the trained policy (units/tick).  It is
# an observation, reported alongside the contract budget so the realistic and
# the worst-case round counts can be compared; the contract itself is derived
# from the plant bounds only.
ECOLOGY_MEASURED_POLICY_STEP = 0.225


def _ecology_world_config(seed: int) -> AntWorldConfig:
    """The single world configuration every curriculum layout is built on."""

    return AntWorldConfig(
        seed=seed,
        antenna_offset_deg=45.0,
        antenna_reach=0.9,
        # The default 1.0 delivery disc swallows the near tier: with a pickup
        # disc that reaches the nest, near pickup AND near delivery both
        # complete with zero displacement.  The retired near band drew its
        # nest distance uniformly from [0.95, 1.35] against a pickup radius of
        # 1.1, so the spawn point sat inside the pickup disc on
        # (1.1 - 0.95) / (1.35 - 0.95) = 37.5% of draws -- the figure is
        # reproducible from those four numbers alone, which the retired
        # "37.05%" was not.
        # Halving the delivery disc is what lets a near tier exist at all --
        # with a 1.0 nest every non-degenerate near band collides with the
        # medium band.  Delivery is strictly harder at every tier.
        nest_radius=ECOLOGY_NEST_RADIUS,
    )


_ECOLOGY_PLANT = _ecology_world_config(0)
# motor_decode: step_command = sigmoid(4*z2) * step_size with z2 bounded to
# LATENT_CODE_BOUNDS = (0.0, 1.0), so the graded controller can neither stop
# nor drop below half speed.  The floor is the worst-case travel rate a
# milestone budget may assume; the ceiling is the fastest the plant can go.
ECOLOGY_PLANT_STEP_FLOOR = 0.5 * _ECOLOGY_PLANT.step_size
ECOLOGY_PLANT_STEP_CEILING = _ECOLOGY_PLANT.step_size / (
    1.0 + math.exp(-4.0)
)
# ``AntWorld.act`` turns FIRST and then steps, by at most ``max_turn_rate``
# per act.  A body therefore cannot re-aim without travelling, which is the
# leg every worst-case milestone budget below has to pay and the old budget
# ignored entirely.
ECOLOGY_PLANT_MAX_TURN_RATE = _ECOLOGY_PLANT.max_turn_rate
# Farthest point reachable from the spawn by a body that just holds maximum
# curvature: the diameter of the WIDEST such circle, i.e. the one traced at
# full speed.  This is the open-loop "spin near the nest and harvest" policy
# the near tier used to be solvable by; measured at the retired geometry it
# swept the pickup disc on 32/96 near layouts at full speed and 96/96 at the
# audit's measured speed.
#
# It bounds max-curvature circles only.  No layout geometry can exclude every
# constant-curvature open-loop solution -- a gentler turn traces an
# arbitrarily large circle -- which is exactly why the forced-approach block
# randomizes its spawn ring and heading offset per body.
#
# The plant turns and THEN steps, in discrete ``max_turn_rate`` increments, so
# the trajectory is a regular polygon, not the continuous turning circle: with
# a 45 deg increment it is an octagon whose vertices sit on a circle of
# circumradius ``step / (2*sin(max_turn_rate/2))``, and the spawn vertex's
# antipode is a full ``step / sin(max_turn_rate/2)`` away.  The continuous
# ``2*v/omega`` understates that by 2.6% (1.00027 vs 1.02645), and a pickup
# disc placed in the gap passes the validator while the full-speed
# max-curvature orbit still harvests it (measured: nearest pickup edge 1.010
# -> 4/72 bearings picked up, 1.030 -> 0/72).  Every shipped tier clears both
# values (near 1.10, medium 1.15, far 1.90), so this is the guard's margin,
# not a tier verdict.
ECOLOGY_UNDIRECTED_ORBIT_REACH = ECOLOGY_PLANT_STEP_CEILING / math.sin(
    0.5 * _ECOLOGY_PLANT.max_turn_rate
)
# ``nearest_food_distance`` is an edge distance, so this is "one antenna reach
# away from the pickup disc".  The retired v7 threshold (4.4) was the far
# distance bound and was satisfied at spawn on every layout.
ECOLOGY_FOOD_ENCOUNTER_DISTANCE = _ECOLOGY_PLANT.antenna_reach


@dataclass(frozen=True)
class EcologyTierGeometry:
    """Pickup geometry of one training tier, in nest-centred units."""

    pickup_radius: float
    distance_min: float
    distance_max: float

    @property
    def min_outbound_leg(self) -> float:
        """Shortest straight run from the nest centre into the pickup disc."""

        return self.distance_min - self.pickup_radius

    @property
    def max_outbound_leg(self) -> float:
        return self.distance_max - self.pickup_radius

    @property
    def min_delivery_leg(self) -> float:
        """Shortest carry from the nearest pickup point into the nest disc."""

        return self.min_outbound_leg - ECOLOGY_NEST_RADIUS

    @property
    def worst_case_round_trip(self) -> float:
        return 2.0 * self.max_outbound_leg - ECOLOGY_NEST_RADIUS


# What makes the near tier "near" is the SHORT loop (12 rounds worst case vs
# 16 medium and 32 far), not a pickup disc parked on the nest.  Once both
# degeneracies are closed no tier can have an outbound leg under ~1.0, so the
# tiers are necessarily closer together than the retired bands pretended.
#
# far keeps its frozen band exactly.  medium's lower bound moved 2.1 -> 2.25
# because at 2.1 its outbound leg was 1.000 against an orbit reach of 1.0003:
# the full-speed max-curvature circle harvested 1/96 medium layouts.
#
# The table is exported as a read-only ``Mapping``: ``_validate_tier_geometry``
# runs ONCE at import, so a plain dict would let any importer edit a frozen
# tier afterwards and every downstream budget, gate and layout would silently
# describe a world the validator never saw.
ECOLOGY_TIER_GEOMETRY: Mapping[EcologyTrainingTier, EcologyTierGeometry] = (
    MappingProxyType(
        {
            EcologyTrainingTier.NEAR: EcologyTierGeometry(
                pickup_radius=0.7,
                distance_min=1.80,
                distance_max=2.00,
            ),
            EcologyTrainingTier.MEDIUM: EcologyTierGeometry(
                pickup_radius=1.1,
                distance_min=2.25,
                distance_max=2.9,
            ),
            EcologyTrainingTier.FAR: EcologyTierGeometry(
                pickup_radius=1.1,
                distance_min=3.0,
                distance_max=4.4,
            ),
        }
    )
)


def _validate_tier_geometry() -> None:
    """Refuse to import a curriculum whose tiers are geometrically degenerate."""

    if set(ECOLOGY_TIER_GEOMETRY) != set(EcologyTrainingTier):
        raise RuntimeError(
            "every ecology training tier must publish a pickup geometry"
        )
    for tier, geometry in ECOLOGY_TIER_GEOMETRY.items():
        if geometry.pickup_radius <= 0.0:
            raise RuntimeError(f"{tier.value} pickup radius must be positive")
        if geometry.distance_min >= geometry.distance_max:
            raise RuntimeError(
                f"{tier.value} distance band must be a non-empty interval"
            )
        if geometry.min_delivery_leg < ECOLOGY_MIN_DELIVERY_LEG:
            raise RuntimeError(
                f"{tier.value} pickup disc reaches the delivery disc: "
                f"delivery leg {geometry.min_delivery_leg:.4f} < "
                f"{ECOLOGY_MIN_DELIVERY_LEG:.4f}; a body that never leaves the "
                "nest would score a pickup->delivery success"
            )
        if geometry.min_outbound_leg <= ECOLOGY_UNDIRECTED_ORBIT_REACH:
            raise RuntimeError(
                f"{tier.value} pickup disc is inside the undirected turning "
                f"circle: outbound {geometry.min_outbound_leg:.4f} <= "
                f"orbit reach {ECOLOGY_UNDIRECTED_ORBIT_REACH:.4f}; spinning "
                "in place would sweep the pickup disc"
            )


_validate_tier_geometry()


# Forced start conditions.  These tables are the single source for both the
# world construction in ``_world`` and the milestone budget derivation.
_FORCED_RETURN_RADIUS: dict[EcologyTrainingTier, float] = {
    EcologyTrainingTier.NEAR: 1.5,
    EcologyTrainingTier.MEDIUM: 2.5,
    EcologyTrainingTier.FAR: 4.0,
}
_FORCED_RETURN_BEARING_OFFSET: dict[EcologyTrainingTier, float] = {
    EcologyTrainingTier.NEAR: math.pi / 6.0,
    EcologyTrainingTier.MEDIUM: math.pi / 4.0,
    EcologyTrainingTier.FAR: math.pi / 3.0,
}
_FORCED_APPROACH_MIN_SPAWN_RATIO = 1.45
_FORCED_APPROACH_MAX_SPAWN_RATIO = 2.9
_FORCED_ESCAPE_SPAWN_RATIO = 0.35
# Burning-match field used by every curriculum layout.
_ECOLOGY_MATCH_HEAT_STRENGTH = 1.0
_ECOLOGY_MATCH_HEAT_DECAY = 1.5
_ECOLOGY_MATCH_HARM_THRESHOLD = 0.55
_ECOLOGY_MATCH_HARM_RADIUS = BurningMatch(
    object_id="ecology-harm-radius-probe",
    x=0.0,
    y=0.0,
    heat_strength=_ECOLOGY_MATCH_HEAT_STRENGTH,
    heat_decay=_ECOLOGY_MATCH_HEAT_DECAY,
    harm_threshold=_ECOLOGY_MATCH_HARM_THRESHOLD,
).harm_radius
# Worst-case escape path: from the spawn ring on the far side of the harm
# disc, out through the opposite boundary.
_ECOLOGY_FORCED_ESCAPE_LEG = (
    1.0 + _FORCED_ESCAPE_SPAWN_RATIO
) * _ECOLOGY_MATCH_HARM_RADIUS
# Route placement of the burning match: a bearing offset from the nest->food
# line and a distance along it.  ``_scene_objects`` and
# ``_validate_heat_layout_solvability`` read this single statement, so the
# invariant below is checked against the geometry that actually ships.
_ECOLOGY_MATCH_ROUTE_BEARING_OFFSET = 0.28
_ECOLOGY_MATCH_ROUTE_MIN_DISTANCE = 1.15
_ECOLOGY_MATCH_ROUTE_DISTANCE_RATIO = 0.48


def _match_route_distance(food_distance: float) -> float:
    return max(
        _ECOLOGY_MATCH_ROUTE_MIN_DISTANCE,
        food_distance * _ECOLOGY_MATCH_ROUTE_DISTANCE_RATIO,
    )


# ``_layout_mastery`` requires ``harmful_tick_rate <= 5%`` (plan section 4.4)
# on every route-foraging layout.  That bound is only meaningful if a
# harm-free pickup->delivery route EXISTS on every layout the generator can
# produce; otherwise plan section 4.6 applies and the layout, not the gate,
# is what is wrong.  The plane minus a closed disc is path-connected, so the
# existence of such a route reduces to two clearances:
#
#   * the spawn/delivery point (bodies spawn at the nest centre and delivery
#     resolves inside the nest disc) must lie outside the harm disc, by at
#     least one floor step so the first act cannot be forced into it;
#   * the pickup disc must keep a lobe outside the harm disc at least one
#     ceiling step wide, so a pickup can be scored without burning.
#
# Both are checked at import over the analytic worst case of every tier.
ECOLOGY_HEAT_FREE_DELIVERY_CLEARANCE = ECOLOGY_PLANT_STEP_FLOOR
ECOLOGY_HEAT_FREE_PICKUP_LOBE = ECOLOGY_PLANT_STEP_CEILING


def ecology_heat_layout_clearances(
    tier: EcologyTrainingTier,
    *,
    food_distance: float,
) -> tuple[float, float]:
    """``(delivery clearance, harm-free pickup lobe width)`` for one layout.

    Both are measured from the frozen harm disc of the curriculum's burning
    match.  Negative delivery clearance means a body can burn where it spawns
    and delivers; a non-positive lobe means every pickup point is inside the
    harm disc.
    """

    geometry = ECOLOGY_TIER_GEOMETRY[tier]
    match_distance = _match_route_distance(food_distance)
    food_to_match = math.hypot(
        food_distance
        - match_distance * math.cos(_ECOLOGY_MATCH_ROUTE_BEARING_OFFSET),
        match_distance * math.sin(_ECOLOGY_MATCH_ROUTE_BEARING_OFFSET),
    )
    return (
        match_distance - _ECOLOGY_MATCH_HARM_RADIUS,
        food_to_match + geometry.pickup_radius - _ECOLOGY_MATCH_HARM_RADIUS,
    )


def _validate_heat_layout_solvability() -> None:
    """Refuse to import heat layouts on which zero harmful ticks is impossible.

    ``match_distance`` is non-decreasing in ``food_distance`` and so is the
    food-to-match separation, so the analytic worst case of a tier is its
    ``distance_min``; no seed sweep can find a worse layout.
    """

    for tier, geometry in ECOLOGY_TIER_GEOMETRY.items():
        delivery, pickup_lobe = ecology_heat_layout_clearances(
            tier,
            food_distance=geometry.distance_min,
        )
        if delivery < ECOLOGY_HEAT_FREE_DELIVERY_CLEARANCE:
            raise RuntimeError(
                f"{tier.value} heat layout burns its own delivery disc: "
                f"nest-to-harm clearance {delivery:.4f} < "
                f"{ECOLOGY_HEAT_FREE_DELIVERY_CLEARANCE:.4f}; a body could "
                "be harmed where it spawns and delivers, so the 5% harmful "
                "tick bound would grade the layout, not the policy"
            )
        if pickup_lobe < ECOLOGY_HEAT_FREE_PICKUP_LOBE:
            raise RuntimeError(
                f"{tier.value} heat layout has no harm-free pickup: lobe "
                f"{pickup_lobe:.4f} < {ECOLOGY_HEAT_FREE_PICKUP_LOBE:.4f}; "
                "every pickup point would be inside the harm disc"
            )


_validate_heat_layout_solvability()


def _legs_to_rounds(distance: float) -> int:
    """Acts a body needs to cover ``distance`` at the plant's slowest step."""

    if distance <= 0.0:
        return 0
    return math.ceil(distance / ECOLOGY_PLANT_STEP_FLOOR)


def _heading_correction_rounds(heading_error: float) -> int:
    """Acts the best expressible policy spends re-aiming, worst case.

    A milestone budget that only counts straight legs is NOT a worst case:
    every body pays a re-aiming leg at spawn (bodies spawn at the nest with a
    uniform random heading, so the error is up to pi) and a second one when it
    turns around after pickup.

    The frozen plant makes re-aiming cost distance.  ``motor_decode`` emits
    ``step_command = sigmoid(4*z2) * step_size`` with ``z2`` bounded to the
    kernel's ``LATENT_CODE_BOUNDS = (0.0, 1.0)``, i.e. a reachable band of
    ``[0.200, 0.393]`` units/act at ``step_size=0.4``: the graded controller
    can neither stop nor drop below half speed, so it cannot turn in place.
    ``AntWorld.act`` then turns by at most ``max_turn_rate`` and steps along
    the NEW heading.

    Bound: turning at the maximum rate takes ``k = ceil(error / max_turn_rate)``
    acts, and by the triangle inequality each of those acts can push the body
    at most one floor step further from its target, which costs at most ``k``
    further acts to undo.  ``2 * k`` is therefore SUFFICIENT for a policy that
    holds the floor speed while re-aiming; it is an upper bound, not a
    measured optimum, and erring high only ever makes the budget stricter.
    """

    if heading_error <= 0.0:
        return 0
    return 2 * math.ceil(heading_error / ECOLOGY_PLANT_MAX_TURN_RATE)


# Worst-case heading error at the start of a leg.  A body spawned by
# ``AntWorld._spawn_body`` gets ``uniform(0, 2*pi)``, so an unforced leg and
# the post-pickup turn-around both start up to pi away from where they need
# to point.  The forced start conditions in ``_world`` pin the heading, so
# their errors are the pinned offsets those blocks apply.
_ECOLOGY_FREE_SPAWN_HEADING_ERROR = math.pi
_ECOLOGY_FORCED_ESCAPE_HEADING_ERROR = math.pi / 2.0


def _directed_leg_rounds(distance: float, *, heading_error: float) -> int:
    """Acts sufficient to close ``distance`` from a given heading error."""

    return _heading_correction_rounds(heading_error) + _legs_to_rounds(
        distance
    )


def _round_trip_rounds(
    outbound_leg: float,
    *,
    spawn_heading_error: float,
) -> int:
    """Acts sufficient for outbound + carry home, with the plant's discretisation.

    The body advances in fixed steps and the world resolves pickup and
    delivery at the END of an act, so it overshoots the pickup disc by up to
    one step and the carry home starts from wherever that act landed.
    Ignoring the overshoot understates every budget by a round or two.

    Both legs carry a heading correction: the outbound one from the spawn
    heading, the carry one because after pickup the body points away from the
    nest.
    """

    outbound = _directed_leg_rounds(
        outbound_leg,
        heading_error=spawn_heading_error,
    )
    carry = (
        outbound_leg + ECOLOGY_PLANT_STEP_FLOOR - ECOLOGY_NEST_RADIUS
    )
    return outbound + _directed_leg_rounds(
        carry,
        heading_error=_ECOLOGY_FREE_SPAWN_HEADING_ERROR,
    )


def ecology_tier_round_budget(tier: EcologyTrainingTier) -> int:
    """Rounds SUFFICIENT for one body to sample this tier's milestone.

    Worst case over the tier's distance band, over the spawn heading, and
    over every trajectory the frozen plant can express: straight legs at the
    plant's slowest step PLUS the re-aiming legs of
    ``_heading_correction_rounds``.  The previous derivation counted the
    straight legs only, which made it neither a sufficient budget (a body
    running the best policy the graded plant can express missed the milestone
    at every tier) nor a valid necessary one (the necessary bound would use
    the CEILING speed, not the floor).

    An episode shorter than this may not be able to produce the milestone it
    is nominally training, so that tier's mastery could only be met by
    zero-shot transfer from another tier.
    """

    return _round_trip_rounds(
        ECOLOGY_TIER_GEOMETRY[tier].max_outbound_leg,
        spawn_heading_error=_ECOLOGY_FREE_SPAWN_HEADING_ERROR,
    )


def ecology_training_min_stage_rounds() -> int:
    """Stage-round floor a schedule containing EVERY tier must be given.

    Published so a lane declares its budget from the frozen geometry instead
    of restating a literal.  ``_train_arm`` refuses a schedule whose episodes
    cannot reach their own milestone within ``stage_rounds``.
    """

    return max(
        ecology_tier_round_budget(tier) for tier in EcologyTrainingTier
    )


class EcologyDataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELDOUT = "heldout"


class EcologyEvaluationScenario(str, Enum):
    BUTTER_ONLY = "butter_only"
    BUTTER_WITH_NEUTRAL_STICK = "butter_with_neutral_stick"
    HEAT_ROUTE_AVOIDANCE = "heat_route_avoidance"
    HEAT_FORCED_ESCAPE = "heat_forced_escape"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class EcologyCurriculumConfig:
    n_ants: int = 8
    temporal_latent_dim: int = 16
    stage_rounds: int = 80
    stage_episodes: int = 4
    # One primary episode is one layout seed.  ``mastery_min_episodes`` is a
    # budget floor only: it says how many primary episodes a stage must run,
    # never what counts as success.
    mastery_min_episodes: int = 3
    # Plan section 4.4 mastery thresholds, frozen before the first P1 result.
    # They may be tightened only by fixing the implementation; the validators
    # below refuse any run that relaxes them.
    mastery_layout_success_ratio: float = 0.6
    mastery_body_success_ratio: float = 0.6
    mastery_harmful_tick_rate_max: float = 0.05
    interleave_every: int = 2
    validation_rounds: int = 80
    validation_seeds: tuple[int, ...] = (43, 59)
    heldout_rounds: int = 120
    heldout_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    seed: int = 0
    # THE FAR DECISION, enforced (see ``_tier_for_episode``).  With it on --
    # the only setting a formal run may use -- ``_train_arm`` refuses to
    # replay a schedule containing an episode whose own milestone needs more
    # rounds than ``stage_rounds`` buys.  plan section 7 step 3 requires a
    # 1-ant / 1-round deterministic test tier, so the lever exists and is part
    # of the frozen config (hence of every config digest): a run that turns it
    # off says so in its own provenance.
    milestone_budget_enforced: bool = True
    action_probe_guard_enabled: bool = True
    action_probe_code_delta_threshold: float = 1e-8
    action_probe_turn_delta_threshold: float = 1e-4
    action_probe_retention_ratio: float = 0.25
    action_probe_body_pass_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.temporal_latent_dim < 3:
            raise ValueError("temporal_latent_dim must be >= 3")
        if self.stage_rounds < 1 or self.stage_episodes < 1:
            raise ValueError("stage budgets must be >= 1")
        if not 1 <= self.mastery_min_episodes <= self.stage_episodes:
            raise ValueError(
                "mastery_min_episodes must be within stage episode budget"
            )
        if self.mastery_layout_success_ratio != 0.6:
            raise ValueError(
                "ecology layout success threshold is frozen at 0.6"
            )
        if self.mastery_body_success_ratio != 0.6:
            raise ValueError(
                "ecology body success threshold is frozen at 0.6"
            )
        if self.mastery_harmful_tick_rate_max != 0.05:
            raise ValueError(
                "ecology harmful tick threshold is frozen at 0.05"
            )
        if self.interleave_every < 1:
            raise ValueError("interleave_every must be >= 1")
        if self.validation_rounds < 1 or not self.validation_seeds:
            raise ValueError(
                "validation budget and seeds must be non-empty"
            )
        if self.heldout_rounds < 1 or not self.heldout_seeds:
            raise ValueError("heldout budget and seeds must be non-empty")
        if set(self.validation_seeds).intersection(self.heldout_seeds):
            raise ValueError(
                "validation and held-out seeds must be disjoint"
            )
        if self.action_probe_code_delta_threshold <= 0.0:
            raise ValueError(
                "action_probe_code_delta_threshold must be positive"
            )
        if self.action_probe_turn_delta_threshold <= 0.0:
            raise ValueError(
                "action_probe_turn_delta_threshold must be positive"
            )
        if not 0.0 < self.action_probe_retention_ratio <= 1.0:
            raise ValueError(
                "action_probe_retention_ratio must be within (0, 1]"
            )
        if not 0.0 < self.action_probe_body_pass_ratio <= 1.0:
            raise ValueError(
                "action_probe_body_pass_ratio must be within (0, 1]"
            )


@dataclass(frozen=True)
class EcologyArmMetrics:
    arm: str
    data_split: EcologyDataSplit
    stage: EcologyStage
    scenario: EcologyEvaluationScenario
    seed: int
    pickups: int
    deliveries: int
    obstacle_contacts: int
    harmful_heat_ticks: int
    heat_entries: int
    heat_escapes: int
    applied_distance: float
    replay_captured: int
    replay_settled: int
    replay_lineage_matches: int
    replay_pending_captures: int
    replay_staged_rollouts: int
    replay_drop_count: int
    replay_settlement_coverage: float
    replay_lineage_coverage: float
    nonzero_ecology_payoffs: int
    activated_sense_channels: tuple[str, ...]
    first_pickup_tick: int | None
    first_obstacle_contact_tick: int | None
    first_heat_entry_tick: int | None
    first_heat_escape_tick: int | None
    minimum_food_distance: float | None
    minimum_obstacle_distance: float | None
    minimum_heat_distance: float | None
    switch_count: int
    mean_persistence_steps: float
    closed_segment_count: int
    longest_segment_length: int
    mean_absolute_turn_delta: float
    policy_fingerprint_stable: bool
    temporal_learning_fingerprint_stable: bool
    body_lineage: tuple["EcologyBodyEpisodeLineage", ...] = ()


@dataclass(frozen=True)
class EcologyTrainingEpisodePlan:
    stage: EcologyStage
    tier: EcologyTrainingTier
    seed: int
    episode_index: int
    interleaved: bool
    forced_escape: bool
    forced_return: bool = False
    forced_approach: bool = False


@dataclass(frozen=True)
class EcologyTrainingEpisodeReport:
    arm: str
    plan: EcologyTrainingEpisodePlan
    pickups: int
    deliveries: int
    obstacle_contacts: int
    heat_entries: int
    heat_escapes: int
    nonzero_ecology_payoffs: int
    activated_sense_channels: tuple[str, ...]
    minimum_food_distance: float | None
    minimum_obstacle_distance: float | None
    minimum_heat_distance: float | None
    switch_count: int
    mean_persistence_steps: float
    closed_segment_count: int
    longest_segment_length: int
    policy_fingerprints_before: tuple[str, ...]
    policy_fingerprints_after: tuple[str, ...]
    memory_entries_evicted: int
    # Rounds this episode actually ran, and the rounds its own milestone needs
    # in the worst case.  ``milestone_samplable`` false means the episode was
    # trained on a milestone it could not reach.
    rounds: int
    milestone_round_budget: int
    milestone_samplable: bool
    action_chain_guard_passed: bool = True
    action_chain_rollback_applied: bool = False
    action_chain_failures: tuple[str, ...] = ()
    body_lineage: tuple["EcologyBodyEpisodeLineage", ...] = ()


@dataclass(frozen=True)
class EcologyBodyEpisodeLineage:
    """Per-body events; aggregate totals cannot satisfy P1 mastery."""

    body_id: int
    episode_id: str
    layout_seed: int
    stage: EcologyStage
    tier: EcologyTrainingTier
    encountered_food: bool
    encountered_heat: bool
    picked_up: bool
    delivered: bool
    pickup_tick: int | None
    delivery_tick: int | None
    harmful_heat_ticks: int
    heat_entries: int
    heat_escapes: int
    escape_latencies: tuple[int, ...]
    applied_distance: float
    switch_count: int
    non_timeout_segment_closures: int
    timed_out: bool
    total_ticks: int


class EcologyLayoutCapability(str, Enum):
    """What a single training layout is allowed to demonstrate."""

    FORAGING = "foraging"
    FORCED_ESCAPE = "forced_escape"


@dataclass(frozen=True)
class EcologyLayoutMastery:
    """One layout's verdict; the unit P1 mastery is counted in.

    A layout is *mastery eligible* only if it can carry a capability claim.
    Near foraging layouts cannot: plan section 4.3 makes them Stage-0
    bootstrap whose only job is to produce pickup/delivery samples, and their
    results "must not let any formal capability gate pass".
    """

    stage: EcologyStage
    tier: EcologyTrainingTier
    capability: EcologyLayoutCapability
    layout_seed: int
    episode_index: int
    mastery_eligible: bool
    eligibility_reason: str
    successful_bodies: int
    required_bodies: int
    harmful_tick_rate: float
    safety_respected: bool
    layout_success: bool


@dataclass(frozen=True)
class EcologyCapabilityMastery:
    """Layout success rate for ONE capability inside a stage.

    A stage that mixes capabilities (the burning-match stage runs a forced
    escape layout and route-foraging layouts) must clear the rate on each of
    them separately, or an easy escape success silently substitutes for the
    foraging success the stage is supposed to certify.
    """

    capability: EcologyLayoutCapability
    mastery_layouts: int
    successful_layouts: int
    required_layouts: int
    reached: bool


@dataclass(frozen=True)
class EcologyStageMastery:
    stage: EcologyStage
    reached: bool
    primary_episodes: int
    mastery_layouts: int
    successful_layouts: int
    required_layouts: int
    required_bodies: int
    bootstrap_layouts: int
    capability_results: tuple[EcologyCapabilityMastery, ...]
    layout_results: tuple[EcologyLayoutMastery, ...]
    threshold: str


@dataclass(frozen=True)
class EcologyGate:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True)
class EcologyCheckpointReport:
    schema_version: str
    config: EcologyCurriculumConfig
    initial_policy_fingerprints: tuple[str, ...]
    learned_policy_fingerprints: tuple[str, ...]
    no_optimize_policy_fingerprints: tuple[str, ...]
    training_schedule: tuple[EcologyTrainingEpisodePlan, ...]
    learned_training: tuple[EcologyTrainingEpisodeReport, ...]
    no_optimize_training: tuple[EcologyTrainingEpisodeReport, ...]
    # plan section 4.2: the arm is published as ``dense-local-shaping-off``,
    # never as "valence off".  Only the dense per-tick shaping measurement is
    # withheld; the sparse milestone reward is untouched.
    dense_local_shaping_off_training: tuple[
        EcologyTrainingEpisodeReport,
        ...,
    ]
    segment_credit_off_training: tuple[
        EcologyTrainingEpisodeReport,
        ...,
    ]
    learned_mastery: tuple[EcologyStageMastery, ...]
    action_probes: tuple[EcologyActionProbe, ...]
    validation_metrics: tuple[EcologyArmMetrics, ...]
    learned_metrics: tuple[EcologyArmMetrics, ...]
    cold_metrics: tuple[EcologyArmMetrics, ...]
    no_optimize_metrics: tuple[EcologyArmMetrics, ...]
    dense_local_shaping_off_metrics: tuple[EcologyArmMetrics, ...]
    segment_credit_off_metrics: tuple[EcologyArmMetrics, ...]
    gates: tuple[EcologyGate, ...]
    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EcologyCheckpointCandidate:
    checkpoints: tuple[AntLearningCheckpoint, ...]
    checkpoint_archives: tuple[bytes, ...]
    report: EcologyCheckpointReport


def _scene_objects(
    *,
    stage: EcologyStage,
    seed: int,
    data_split: EcologyDataSplit,
    tier: EcologyTrainingTier,
) -> tuple[WorldObject, ...]:
    split_offset = {
        EcologyDataSplit.TRAIN: 0,
        EcologyDataSplit.VALIDATION: 1_000_003,
        EcologyDataSplit.HELDOUT: 2_000_003,
    }[data_split]
    rng = np.random.default_rng(seed + split_offset)
    bearing = float(rng.uniform(-math.pi, math.pi))
    geometry = ECOLOGY_TIER_GEOMETRY[tier]
    distance = float(
        rng.uniform(geometry.distance_min, geometry.distance_max)
    )
    food_x = math.cos(bearing) * distance
    food_y = math.sin(bearing) * distance
    objects: list[WorldObject] = [
        # The odour field (strength/decay) is tier-independent, so shrinking
        # the near pickup disc removes the free pickup without weakening the
        # gradient a cold policy has to steer on.
        ButterSource(
            object_id="butter",
            x=food_x,
            y=food_y,
            strength=2.2,
            decay=2.4,
            radius=geometry.pickup_radius,
        )
    ]
    if stage in {EcologyStage.WOOD_STICK, EcologyStage.COMPOSITE}:
        centre_distance = max(1.15, distance * 0.55)
        centre_x = math.cos(bearing) * centre_distance
        centre_y = math.sin(bearing) * centre_distance
        perpendicular = bearing + math.pi / 2.0
        half_length = min(1.35, max(0.8, distance * 0.35))
        objects.append(
            WoodStick(
                object_id="wood-stick",
                start_x=centre_x + math.cos(perpendicular) * half_length,
                start_y=centre_y + math.sin(perpendicular) * half_length,
                end_x=centre_x - math.cos(perpendicular) * half_length,
                end_y=centre_y - math.sin(perpendicular) * half_length,
                radius=0.2,
            )
        )
    if stage in {EcologyStage.BURNING_MATCH, EcologyStage.COMPOSITE}:
        offset_sign = -1.0 if seed % 2 else 1.0
        match_distance = _match_route_distance(distance)
        match_bearing = (
            bearing + offset_sign * _ECOLOGY_MATCH_ROUTE_BEARING_OFFSET
        )
        objects.append(
            BurningMatch(
                object_id="burning-match",
                x=math.cos(match_bearing) * match_distance,
                y=math.sin(match_bearing) * match_distance,
                angle=bearing,
                heat_strength=_ECOLOGY_MATCH_HEAT_STRENGTH,
                heat_decay=_ECOLOGY_MATCH_HEAT_DECAY,
                harm_threshold=_ECOLOGY_MATCH_HARM_THRESHOLD,
            )
        )
    return tuple(objects)


def _world(
    *,
    config: EcologyCurriculumConfig,
    stage: EcologyStage,
    seed: int,
    data_split: EcologyDataSplit,
    tier: EcologyTrainingTier,
    forced_escape: bool = False,
    forced_return: bool = False,
    forced_approach: bool = False,
) -> ColonyWorld:
    if sum((forced_escape, forced_return, forced_approach)) > 1:
        raise ValueError(
            "an ecology episode can force at most one start condition"
        )
    split_offset = {
        EcologyDataSplit.TRAIN: 17,
        EcologyDataSplit.VALIDATION: 3_000_017,
        EcologyDataSplit.HELDOUT: 4_000_017,
    }[data_split]
    objects = _scene_objects(
        stage=stage,
        seed=seed,
        data_split=data_split,
        tier=tier,
    )
    world = ColonyWorld(
        config=_ecology_world_config(seed + split_offset),
        world_objects=objects,
        n_bodies=config.n_ants,
    )
    if forced_escape:
        matches = tuple(
            item for item in objects if isinstance(item, BurningMatch)
        )
        if len(matches) != 1:
            raise RuntimeError(
                "forced escape scene requires exactly one burning match"
            )
        match = matches[0]
        spawn_radius = max(
            0.05,
            match.harm_radius * _FORCED_ESCAPE_SPAWN_RATIO,
        )
        for body_id in range(config.n_ants):
            angle = (
                (body_id + 1) * 2.399963229728653
                + seed * 0.017
            )
            world.set_body_pose(
                body_id=body_id,
                x=match.x + math.cos(angle) * spawn_radius,
                y=match.y + math.sin(angle) * spawn_radius,
                heading=(angle + math.pi / 2.0),
            )
    elif forced_return:
        radius = _FORCED_RETURN_RADIUS[tier]
        bearing_offset = _FORCED_RETURN_BEARING_OFFSET[tier]
        for body_id in range(config.n_ants):
            angle = (
                (body_id + 1) * 2.399963229728653
                + seed * 0.017
            )
            home_bearing = angle + math.pi
            side = 1.0 if (seed + body_id) % 2 == 0 else -1.0
            world.set_body_pose(
                body_id=body_id,
                x=world.nest[0] + math.cos(angle) * radius,
                y=world.nest[1] + math.sin(angle) * radius,
                heading=home_bearing - side * bearing_offset,
                carrying_food=True,
            )
    elif forced_approach:
        # Food-steering pressure bootstrap. A plain layout only demands a
        # correct heading at spawn; here each body spawns OUTSIDE the pickup
        # disc with its heading rotated well past the food bearing, so the
        # scent gradient is clearly sensable but a straight or drifting
        # default trajectory diverges from the source and the only reward
        # path is an active turn toward the gradient. Like forced_return this
        # initializes STATE only -- no coordinates, target bearing or action
        # labels leak downstream.
        #
        # Historical note: until the v8 geometry repair this block also had to
        # compensate for a degenerate near tier (pickup radius 1.1 around a
        # point 0.95-1.35 from a nest of radius 1.0, so the pickup disc
        # covered the spawn point). It only ever fixed the OUTBOUND leg -- the
        # post-pickup delivery leg stayed free because the pickup point was
        # still inside the delivery disc. ``_validate_tier_geometry`` now
        # forbids both degeneracies for every tier, so this block carries
        # steering pressure only.
        butters = tuple(
            item for item in objects if isinstance(item, ButterSource)
        )
        if len(butters) != 1:
            raise RuntimeError(
                "forced approach scene requires exactly one butter source"
            )
        butter = butters[0]
        # Randomized geometry: a FIXED spawn ring is solvable by one tuned
        # constant-curvature orbit -- the first v22 run measured the base
        # policy amplifying its same-direction baseline turn (0.083 -> 0.15
        # rad) and harvesting the block with zero gradient following. With
        # radius and angular offset drawn per body from the layout seed, no
        # single curvature solves the ensemble; gradient steering solves
        # every draw. The lower bounds keep the straight-path miss guarantee
        # for every draw: closest approach >= 1.45*R*sin(0.4*pi) = 1.38*R
        # stays outside the pickup disc of radius R.
        rng = np.random.default_rng(seed + 5_000_011)
        for body_id in range(config.n_ants):
            angle = (
                (body_id + 1) * 2.399963229728653
                + seed * 0.017
            )
            spawn_radius = butter.radius * float(
                rng.uniform(
                    _FORCED_APPROACH_MIN_SPAWN_RATIO,
                    _FORCED_APPROACH_MAX_SPAWN_RATIO,
                )
            )
            bearing_offset = float(rng.uniform(0.4 * math.pi, 0.8 * math.pi))
            # From the spawn point the butter lies at bearing angle + pi.
            food_bearing = angle + math.pi
            side = 1.0 if (seed + body_id) % 2 == 0 else -1.0
            world.set_body_pose(
                body_id=body_id,
                x=butter.x + math.cos(angle) * spawn_radius,
                y=butter.y + math.sin(angle) * spawn_radius,
                heading=food_bearing + side * bearing_offset,
            )
    return world


def _synchronize_curriculum_navigators(
    runner: KernelColonyRunner,
) -> None:
    """Initialize PI at a curriculum reset without leaking pose downstream.

    Required whenever a forced start condition (forced_return or
    forced_approach) repositions bodies away from the nest: path integration
    must agree with the true pose or the post-pickup homing leg is corrupted.
    """

    for body_id, session in enumerate(runner.sessions):
        body = runner.world.body(body_id)
        session.navigator.sync_to(
            x=body.x,
            y=body.y,
            heading=body.heading,
            nest=runner.world.nest,
        )
        session.holder.update(
            observation=runner.world.observe(body_id),
            navigator_state=session.navigator.state,
            step=runner.world.tick,
        )


def _session_config(
    *,
    config: EcologyCurriculumConfig,
    seed: int,
    session_id: str,
    optimize: bool,
    local_valence_enabled: bool = True,
    segment_credit_enabled: bool = True,
    learning_enabled: bool = True,
    sparse_exploration_enabled: bool = True,
    prediction_error_enabled: bool = True,
    temporal_writeback_enabled: bool = True,
) -> AntSessionConfig:
    """Build one matched ecology session.

    ``local_valence_enabled`` keeps the spelling of the session lever it
    forwards to (``AntSessionConfig.ecology_local_valence_enabled``, owned by
    ``volvence_ant.runtime.ant_session``).  The ARM it produces is published
    as ``dense-local-shaping-off``: with the lever off the session withholds
    only the dense per-tick local-progress measurement, while the sparse
    milestone reward remains.  Measured on a 24-tick ECOLOGY session under
    ``ENVIRONMENT_MEASURED_ONLY`` typed reward eligibility: 46/46 transitions
    eligible with the lever on, 0/46 with it off.

    Two P2 matched-control levers ship as explicit keyword contracts; both
    default to the learned-arm value so every existing caller is unchanged:

    * ``prediction_error_enabled=False`` -- PE-off. Prediction error stops
      driving joint-loop learning signals and stops pressuring the temporal
      switch; it remains a readout, matching R-PE (evaluation and credit are
      downstream of PE and never feed back into it).
    * ``temporal_writeback_enabled=False`` -- ETA-off. Bounded writeback of
      the learned temporal metacontroller is withheld, so the emergent
      temporal abstraction cannot adapt while Internal-RL policy optimization
      stays matched to the learned arm.
    """

    return AntSessionConfig(
        temporal_latent_dim=config.temporal_latent_dim,
        session_id=session_id,
        seed=seed,
        rollout_config=ant_runtime_replay_rollout_config(
            enable_sparse_exploration=sparse_exploration_enabled,
            enable_segment_credit=segment_credit_enabled,
            enable_prediction_error_switch=prediction_error_enabled,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
        external_prediction_error_drive=prediction_error_enabled,
        joint_apply_writeback=temporal_writeback_enabled,
        joint_apply_policy_optimization=optimize,
        joint_learning_enabled=learning_enabled,
        objective=AntObjectiveKind.ECOLOGY,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
        ecology_local_valence_enabled=local_valence_enabled,
    )


def _policy_fingerprints(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[str, ...]:
    return tuple(checkpoint.policy_fingerprint for checkpoint in checkpoints)


def _flatten_records(
    runner: KernelColonyRunner,
) -> tuple[AntStepRecord, ...]:
    return tuple(
        step
        for round_record in runner.rounds
        for step in round_record.ant_steps
    )


def _minimum_optional(
    values: tuple[float | None, ...],
) -> float | None:
    finite = tuple(value for value in values if value is not None)
    return min(finite) if finite else None


def _activated_sense_channels(
    records: tuple[AntStepRecord, ...],
) -> tuple[str, ...]:
    active: set[str] = set()
    for record in records:
        for name, value in record.sense_activation:
            if abs(value) > 1e-12:
                active.add(name)
    return tuple(sorted(active))


def _body_episode_lineage(
    *,
    records: tuple[AntStepRecord, ...],
    plan: EcologyTrainingEpisodePlan,
    arm: str,
    n_ants: int,
) -> tuple[EcologyBodyEpisodeLineage, ...]:
    result: list[EcologyBodyEpisodeLineage] = []
    episode_id = (
        f"{arm}:{plan.stage.value}:{plan.tier.value}:"
        f"{plan.seed}:{plan.episode_index}"
    )
    for body_id in range(n_ants):
        body_records = tuple(
            item for item in records if item.body_id == body_id
        )
        open_entry: int | None = (
            0 if body_records and body_records[0].heat_harmful else None
        )
        escape_latencies: list[int] = []
        for item in body_records:
            if (
                item.entered_harmful_heat
                and open_entry is None
                and (not plan.forced_escape or not escape_latencies)
            ):
                open_entry = item.tick
            if item.escaped_harmful_heat and open_entry is not None:
                escape_latencies.append(max(0, item.tick - open_entry))
                open_entry = None
        reason_counts = dict(
            body_records[-1].runtime_segment_close_reason_counts
            if body_records
            else ()
        )
        result.append(
            EcologyBodyEpisodeLineage(
                body_id=body_id,
                episode_id=episode_id,
                layout_seed=plan.seed,
                stage=plan.stage,
                tier=plan.tier,
                encountered_food=any(
                    (item.nearest_food_distance is not None)
                    and item.nearest_food_distance
                    <= ECOLOGY_FOOD_ENCOUNTER_DISTANCE
                    for item in body_records
                ),
                encountered_heat=any(
                    item.heat_center > 1e-6 for item in body_records
                ),
                picked_up=any(item.picked_up for item in body_records),
                delivered=any(item.delivered for item in body_records),
                pickup_tick=_first_tick(
                    body_records,
                    predicate=lambda item: item.picked_up,
                ),
                delivery_tick=_first_tick(
                    body_records,
                    predicate=lambda item: item.delivered,
                ),
                harmful_heat_ticks=sum(
                    int(item.heat_harmful) for item in body_records
                ),
                heat_entries=sum(
                    int(item.entered_harmful_heat) for item in body_records
                ),
                heat_escapes=sum(
                    int(item.escaped_harmful_heat) for item in body_records
                ),
                escape_latencies=tuple(escape_latencies),
                applied_distance=sum(
                    item.applied_step for item in body_records
                ),
                switch_count=sum(
                    int(item.is_switching) for item in body_records
                ),
                non_timeout_segment_closures=sum(
                    count
                    for reason, count in reason_counts.items()
                    if reason != "timeout"
                ),
                timed_out=not any(
                    item.delivered or item.escaped_harmful_heat
                    for item in body_records
                ),
                total_ticks=len(body_records),
            )
        )
    return tuple(result)


def _mean_absolute_turn_delta(
    records: tuple[AntStepRecord, ...],
    *,
    n_ants: int,
) -> float:
    if len(records) <= n_ants:
        return 0.0
    deltas = tuple(
        abs(
            records[index].command.turn_command
            - records[index - n_ants].command.turn_command
        )
        for index in range(n_ants, len(records))
    )
    return sum(deltas) / len(deltas)


def _first_tick(
    records: tuple[AntStepRecord, ...],
    *,
    predicate: Callable[[AntStepRecord], bool],
) -> int | None:
    for record in records:
        if predicate(record):
            return int(record.tick)
    return None


def _tier_for_episode(episode_index: int) -> EcologyTrainingTier:
    """near -> medium -> far within one stage.

    THE FAR DECISION.  plan section 4.5 asks for five TRAINING layouts per
    tier, so far keeps its training role -- what had to move is the budget,
    not the tier.  Against the corrected worst case
    (``ecology_tier_round_budget``: near 28, medium 33, far 49 rounds) the
    defect was never far-specific: a 24-round schedule cannot sample ANY
    tier's milestone, so calling it "the far problem" mis-stated it.  This
    module's own budget (``stage_rounds`` default 80) covers all three.

    A lane that cannot afford ``ecology_training_min_stage_rounds()`` must
    drop the tiers it cannot pay for; it may not replay them anyway.
    ``_train_arm`` refuses such a schedule up front with both numbers per
    episode, so far -- or any other tier -- is never silently trained on an
    unreachable milestone.
    """

    if episode_index == 0:
        return EcologyTrainingTier.NEAR
    if episode_index == 1:
        return EcologyTrainingTier.MEDIUM
    return EcologyTrainingTier.FAR


def _episode_milestone_round_budget(
    plan: EcologyTrainingEpisodePlan,
) -> int:
    """Rounds SUFFICIENT for this episode's own milestone, worst case.

    Forced start conditions change both which legs the milestone contains and
    how far the pinned heading is from where the leg has to point, so the
    budget is derived per start condition rather than per tier alone.
    """

    geometry = ECOLOGY_TIER_GEOMETRY[plan.tier]
    if plan.forced_escape:
        # The body starts inside the harm disc; the milestone is leaving it.
        # ``_world`` pins the heading a quarter turn off the outward radial.
        return _directed_leg_rounds(
            _ECOLOGY_FORCED_ESCAPE_LEG,
            heading_error=_ECOLOGY_FORCED_ESCAPE_HEADING_ERROR,
        )
    if plan.forced_return:
        # Starts outside the nest already carrying: delivery leg only, with
        # the heading pinned ``_FORCED_RETURN_BEARING_OFFSET`` off home.
        return _directed_leg_rounds(
            _FORCED_RETURN_RADIUS[plan.tier] - ECOLOGY_NEST_RADIUS,
            heading_error=_FORCED_RETURN_BEARING_OFFSET[plan.tier],
        )
    if plan.forced_approach:
        # Starts on a ring around the butter: worst-case approach to the
        # pickup disc, then the carry home from its far side.  ``_world``
        # pins the heading up to 0.8*pi off the food bearing; the carry home
        # then starts pointing away from the nest.
        approach = (
            _FORCED_APPROACH_MAX_SPAWN_RATIO - 1.0
        ) * geometry.pickup_radius
        carry = (
            geometry.distance_max
            + geometry.pickup_radius
            + ECOLOGY_PLANT_STEP_FLOOR
            - ECOLOGY_NEST_RADIUS
        )
        return _directed_leg_rounds(
            approach,
            heading_error=0.8 * math.pi,
        ) + _directed_leg_rounds(
            carry,
            heading_error=_ECOLOGY_FREE_SPAWN_HEADING_ERROR,
        )
    return _round_trip_rounds(
        geometry.max_outbound_leg,
        spawn_heading_error=_ECOLOGY_FREE_SPAWN_HEADING_ERROR,
    )


class EcologyMilestoneBudgetError(RuntimeError):
    """A training schedule replays milestones its own budget cannot reach."""


def ecology_schedule_milestone_shortfalls(
    schedule: tuple[EcologyTrainingEpisodePlan, ...],
    *,
    stage_rounds: int,
) -> tuple[tuple[str, str, int, int, int], ...]:
    """``(stage, tier, episode_index, rounds, required)`` per short episode.

    Published so a lane can state its own budget shortfall from the frozen
    geometry rather than re-deriving it, and so the refusal in ``_train_arm``
    and the ``training_tier_milestone_samplable`` gate read the same table.
    """

    if stage_rounds < 1:
        raise ValueError("stage_rounds must be >= 1")
    return tuple(
        (
            plan.stage.value,
            plan.tier.value,
            plan.episode_index,
            stage_rounds,
            required,
        )
        for plan, required in (
            (item, _episode_milestone_round_budget(item))
            for item in schedule
        )
        if stage_rounds < required
    )


def _layout_capability(
    plan: EcologyTrainingEpisodePlan,
) -> EcologyLayoutCapability:
    if plan.forced_escape:
        return EcologyLayoutCapability.FORCED_ESCAPE
    return EcologyLayoutCapability.FORAGING


def _layout_mastery(
    *,
    config: EcologyCurriculumConfig,
    report: EcologyTrainingEpisodeReport,
) -> EcologyLayoutMastery:
    """Score one training layout the way P1 scores a held-out layout.

    plan section 4.2: mastery is an episode/layout success RATE, never a
    stage-cumulative event count.  A body succeeds only if the SAME body
    closed the loop (pickup AND delivery, or escape); a layout succeeds only
    if enough bodies did.
    """

    plan = report.plan
    capability = _layout_capability(plan)
    lineage = report.body_lineage
    if not lineage:
        raise ValueError(
            "ecology layout mastery requires per-body episode lineage; "
            f"episode {plan.stage.value}/{plan.tier.value}/{plan.seed} "
            "published none"
        )
    if capability is EcologyLayoutCapability.FORCED_ESCAPE:
        successful = sum(item.heat_escapes > 0 for item in lineage)
    else:
        successful = sum(
            item.picked_up and item.delivered for item in lineage
        )
    required_bodies = max(
        1,
        math.ceil(config.n_ants * config.mastery_body_success_ratio),
    )
    total_ticks = sum(item.total_ticks for item in lineage)
    harmful_ticks = sum(item.harmful_heat_ticks for item in lineage)
    harmful_rate = harmful_ticks / total_ticks if total_ticks else 0.0
    # plan section 4.4: route avoidance and composite foraging must also hold
    # harmful exposure down; forced escape is scored on escape alone.
    safety_applies = (
        capability is EcologyLayoutCapability.FORAGING
        and plan.stage
        in {EcologyStage.BURNING_MATCH, EcologyStage.COMPOSITE}
    )
    safety_respected = (
        harmful_rate <= config.mastery_harmful_tick_rate_max
        if safety_applies
        else True
    )
    if not report.milestone_samplable:
        # A layout whose episode was too short to reach its own milestone
        # cannot certify the capability: a success would be zero-shot
        # transfer and a failure would say nothing about the policy.
        eligible = False
        reason = (
            "episode ran "
            f"{report.rounds} rounds against a milestone that needs "
            f"{report.milestone_round_budget} in the worst case, so this "
            "layout cannot certify the capability either way"
        )
    elif capability is EcologyLayoutCapability.FORAGING and (
        plan.tier is EcologyTrainingTier.NEAR
    ):
        eligible = False
        reason = (
            "near foraging layout is Stage-0 bootstrap (plan 4.3): it feeds "
            "the optimizer with pickup/delivery samples and can never make a "
            "capability gate pass"
        )
    elif plan.interleaved:
        eligible = False
        reason = (
            "interleaved replay of an already-mastered stage; it guards "
            "against forgetting and does not certify this stage"
        )
    else:
        eligible = True
        reason = "primary capability layout"
    return EcologyLayoutMastery(
        stage=plan.stage,
        tier=plan.tier,
        capability=capability,
        layout_seed=plan.seed,
        episode_index=plan.episode_index,
        mastery_eligible=eligible,
        eligibility_reason=reason,
        successful_bodies=successful,
        required_bodies=required_bodies,
        harmful_tick_rate=harmful_rate,
        safety_respected=safety_respected,
        layout_success=(
            successful >= required_bodies and safety_respected
        ),
    )


def _stage_mastery(
    *,
    config: EcologyCurriculumConfig,
    stage: EcologyStage,
    reports: tuple[EcologyTrainingEpisodeReport, ...],
) -> EcologyStageMastery:
    if stage not in ECOLOGY_TRAINING_STAGES:
        raise ValueError(
            f"stage {stage.value!r} is not a valenced training stage"
        )
    layouts = tuple(
        _layout_mastery(config=config, report=item) for item in reports
    )
    eligible = tuple(item for item in layouts if item.mastery_eligible)
    capability_results: list[EcologyCapabilityMastery] = []
    for capability in EcologyLayoutCapability:
        group = tuple(
            item for item in eligible if item.capability is capability
        )
        if not group:
            continue
        successes = sum(item.layout_success for item in group)
        required = max(
            1,
            math.ceil(len(group) * config.mastery_layout_success_ratio),
        )
        capability_results.append(
            EcologyCapabilityMastery(
                capability=capability,
                mastery_layouts=len(group),
                successful_layouts=successes,
                required_layouts=required,
                reached=successes >= required,
            )
        )
    required_bodies = max(
        1,
        math.ceil(config.n_ants * config.mastery_body_success_ratio),
    )
    reached = (
        len(eligible) >= config.mastery_min_episodes
        and bool(capability_results)
        and all(item.reached for item in capability_results)
    )
    return EcologyStageMastery(
        stage=stage,
        reached=reached,
        primary_episodes=len(reports),
        mastery_layouts=len(eligible),
        successful_layouts=sum(item.layout_success for item in eligible),
        required_layouts=sum(
            item.required_layouts for item in capability_results
        ),
        required_bodies=required_bodies,
        bootstrap_layouts=len(layouts) - len(eligible),
        capability_results=tuple(capability_results),
        layout_results=layouts,
        threshold=_mastery_threshold(config, stage),
    )


def _mastery_threshold(
    config: EcologyCurriculumConfig,
    stage: EcologyStage,
) -> str:
    if stage not in ECOLOGY_TRAINING_STAGES:
        raise ValueError(
            f"stage {stage.value!r} is not a valenced training stage"
        )
    safety = (
        ", harmful tick rate <="
        f"{config.mastery_harmful_tick_rate_max}"
        if stage in {EcologyStage.BURNING_MATCH, EcologyStage.COMPOSITE}
        else ""
    )
    return (
        f">={config.mastery_min_episodes} mastery-eligible layouts, "
        f">={config.mastery_layout_success_ratio:.0%} of them successful "
        "SEPARATELY for every capability the stage runs, "
        f">={config.mastery_body_success_ratio:.0%} of bodies closing "
        f"pickup->delivery (or escaping) in the SAME body{safety}; "
        "near foraging layouts are Stage-0 bootstrap and never eligible"
    )


async def _ecology_action_chain_guard(
    *,
    config: EcologyCurriculumConfig,
    baseline: tuple[AntLearningCheckpoint, ...],
    candidate: tuple[AntLearningCheckpoint, ...],
    baseline_reports: tuple[EcologyCheckpointActionProbe, ...] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Reject an update that destroys preflight sensor→motor sensitivity."""

    if baseline_reports is None:
        baseline_reports = await run_ecology_checkpoint_action_probes(
            temporal_latent_dim=config.temporal_latent_dim,
            seed=config.seed + 700_003,
            checkpoints=baseline,
            code_delta_threshold=config.action_probe_code_delta_threshold,
            turn_delta_threshold=config.action_probe_turn_delta_threshold,
        )
    candidate_reports = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + 700_003,
        checkpoints=candidate,
        code_delta_threshold=config.action_probe_code_delta_threshold,
        turn_delta_threshold=config.action_probe_turn_delta_threshold,
    )
    baseline_by_body = {item.body_id: item for item in baseline_reports}
    failures: list[str] = []
    passing_bodies = 0
    for candidate_body in candidate_reports:
        baseline_body = baseline_by_body[candidate_body.body_id]
        baseline_by_kind = {
            probe.kind: probe for probe in baseline_body.probes
        }
        body_failures: list[str] = []
        for probe in candidate_body.probes:
            baseline_probe = baseline_by_kind[probe.kind]
            if not probe.input_reachable:
                body_failures.append(
                    f"{probe.kind.value}:input-unreachable"
                )
                continue
            if probe.kind in {
                EcologyProbeKind.OBSTACLE,
                EcologyProbeKind.HOME,
            }:
                continue
            retention_floor = (
                baseline_probe.turn_delta
                * config.action_probe_retention_ratio
            )
            if probe.turn_delta < config.action_probe_turn_delta_threshold:
                body_failures.append(
                    f"{probe.kind.value}:turn-delta={probe.turn_delta:.9g}"
                )
            if probe.turn_delta < retention_floor:
                body_failures.append(
                    f"{probe.kind.value}:retention={probe.turn_delta:.9g}/"
                    f"{baseline_probe.turn_delta:.9g}"
                )
        if body_failures:
            failures.extend(
                f"body:{candidate_body.body_id}:{failure}"
                for failure in body_failures
            )
        else:
            passing_bodies += 1
    required = max(
        1,
        math.ceil(len(candidate_reports) * config.action_probe_body_pass_ratio),
    )
    if passing_bodies < required:
        failures.append(
            f"body-pass-ratio:{passing_bodies}/{len(candidate_reports)}"
        )
    return passing_bodies >= required, tuple(failures)


def _require_samplable_milestones(
    *,
    config: EcologyCurriculumConfig,
    schedule: tuple[EcologyTrainingEpisodePlan, ...],
) -> None:
    """Refuse a schedule whose own budget cannot reach its milestones.

    THE FAR DECISION lands here.  ``training_tier_milestone_samplable`` only
    reports the shortfall inside this module's own ``_build_gates``, which the
    fixed-schedule lanes never call, so the verdict has to be enforced where
    every lane passes: at the entry point that replays the schedule.

    The refusal is an ARM-level contract, applied by ``_train_arm`` to both
    the fixed and the generated schedule.  The single-episode primitive
    ``_run_training_episode`` is deliberately left unguarded: P0 mechanism
    diagnostics drive it directly on deliberately short traces (plan section 7
    step 4) and never claim a capability from them.  Nothing can launder a
    capability claim through that door either -- ``_layout_mastery`` refuses
    to make an episode that could not reach its own milestone
    mastery-eligible.
    """

    if not config.milestone_budget_enforced:
        return
    shortfalls = ecology_schedule_milestone_shortfalls(
        schedule,
        stage_rounds=config.stage_rounds,
    )
    if not shortfalls:
        return
    raise EcologyMilestoneBudgetError(
        "ecology training schedule replays milestones its own budget cannot "
        f"reach: stage_rounds={config.stage_rounds}, "
        f"required>={ecology_training_min_stage_rounds()} for a schedule "
        "covering every tier "
        + str(
            tuple(
                (tier.value, ecology_tier_round_budget(tier))
                for tier in EcologyTrainingTier
            )
        )
        + "; short episodes (stage, tier, episode_index, rounds, required)="
        + str(shortfalls)
        + ". Raise the budget, drop the tiers the lane cannot pay for, or "
        "declare milestone_budget_enforced=False for a plan section 7 "
        "small-budget diagnostic (never for a formal run)."
    )


async def _run_training_episode(
    *,
    config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
    local_valence_enabled: bool,
    segment_credit_enabled: bool,
    plan: EcologyTrainingEpisodePlan,
    prediction_error_enabled: bool = True,
    temporal_writeback_enabled: bool = True,
    action_probe_baseline: tuple[AntLearningCheckpoint, ...] | None = None,
    action_probe_baseline_reports: (
        tuple[EcologyCheckpointActionProbe, ...] | None
    ) = None,
) -> tuple[
    KernelColonyRunner,
    tuple[AntLearningCheckpoint, ...],
    EcologyTrainingEpisodeReport,
]:
    runner = KernelColonyRunner(
        _world(
            config=config,
            stage=plan.stage,
            seed=plan.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=plan.tier,
            forced_escape=plan.forced_escape,
            forced_return=plan.forced_return,
            forced_approach=plan.forced_approach,
        ),
        base_config=_session_config(
            config=config,
            seed=plan.seed,
            session_id=(
                f"ecology:{arm}:{plan.stage.value}:"
                f"{plan.tier.value}:episode:{plan.episode_index}"
            ),
            optimize=optimize,
            local_valence_enabled=local_valence_enabled,
            segment_credit_enabled=segment_credit_enabled,
            prediction_error_enabled=prediction_error_enabled,
            temporal_writeback_enabled=temporal_writeback_enabled,
        ),
    )
    if plan.forced_return or plan.forced_approach:
        _synchronize_curriculum_navigators(runner)
    runner.restore_learning_checkpoints(checkpoints)
    before = _policy_fingerprints(checkpoints)
    await runner.run(config.stage_rounds)
    memory_entries_evicted = sum(
        len(
            session.runner.memory_store.enforce_artifact_capacity(
                ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY
            )
        )
        for session in runner.sessions
    )
    trained = runner.export_learning_checkpoints(
        checkpoint_prefix=(
            f"ecology:{arm}:{plan.stage.value}:{plan.tier.value}:"
            f"episode:{plan.episode_index}:trained"
        ),
        include_runtime_replay=False,
    )
    action_chain_guard_passed = True
    action_chain_failures: tuple[str, ...] = ()
    action_chain_rollback_applied = False
    if (
        config.action_probe_guard_enabled
        and action_probe_baseline is not None
    ):
        action_chain_guard_passed, action_chain_failures = (
            await _ecology_action_chain_guard(
                config=config,
                baseline=action_probe_baseline,
                candidate=trained,
                baseline_reports=action_probe_baseline_reports,
            )
        )
        if not action_chain_guard_passed:
            runner.restore_learning_checkpoints(checkpoints)
            trained = runner.export_learning_checkpoints(
                checkpoint_prefix=(
                    f"ecology:{arm}:{plan.stage.value}:"
                    f"{plan.tier.value}:episode:{plan.episode_index}:"
                    "action-chain-rollback"
                ),
                include_runtime_replay=False,
            )
            action_chain_rollback_applied = True
    records = _flatten_records(runner)
    milestone_round_budget = _episode_milestone_round_budget(plan)
    report = EcologyTrainingEpisodeReport(
        arm=arm,
        plan=plan,
        pickups=runner.world.food_pickups,
        deliveries=runner.world.food_delivered,
        obstacle_contacts=sum(
            int(item.obstacle_contact) for item in records
        ),
        heat_entries=sum(
            int(item.entered_harmful_heat) for item in records
        ),
        heat_escapes=sum(
            int(item.escaped_harmful_heat) for item in records
        ),
        nonzero_ecology_payoffs=sum(
            abs(item.ecology_action_payoff) > 1e-12
            for item in records
        ),
        activated_sense_channels=_activated_sense_channels(records),
        minimum_food_distance=_minimum_optional(
            tuple(item.nearest_food_distance for item in records)
        ),
        minimum_obstacle_distance=_minimum_optional(
            tuple(item.nearest_obstacle_distance for item in records)
        ),
        minimum_heat_distance=_minimum_optional(
            tuple(item.nearest_heat_distance for item in records)
        ),
        switch_count=sum(int(item.is_switching) for item in records),
        mean_persistence_steps=(
            sum(item.steps_since_switch for item in records)
            / len(records)
            if records
            else 0.0
        ),
        closed_segment_count=max(
            (
                item.runtime_closed_segments
                for item in records
            ),
            default=0,
        ),
        longest_segment_length=max(
            (
                item.runtime_longest_segment_length
                for item in records
            ),
            default=0,
        ),
        policy_fingerprints_before=before,
        policy_fingerprints_after=_policy_fingerprints(trained),
        memory_entries_evicted=memory_entries_evicted,
        rounds=config.stage_rounds,
        milestone_round_budget=milestone_round_budget,
        milestone_samplable=(
            config.stage_rounds >= milestone_round_budget
        ),
        action_chain_guard_passed=action_chain_guard_passed,
        action_chain_rollback_applied=action_chain_rollback_applied,
        action_chain_failures=action_chain_failures,
        body_lineage=_body_episode_lineage(
            records=records,
            plan=plan,
            arm=arm,
            n_ants=config.n_ants,
        ),
    )
    return runner, trained, report


async def _train_arm(
    *,
    config: EcologyCurriculumConfig,
    initial: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
    local_valence_enabled: bool,
    segment_credit_enabled: bool,
    prediction_error_enabled: bool = True,
    temporal_writeback_enabled: bool = True,
    schedule: tuple[EcologyTrainingEpisodePlan, ...] | None = None,
    schedule_start_index: int = 0,
    episode_callback: Callable[
        [
            int,
            KernelColonyRunner,
            tuple[AntLearningCheckpoint, ...],
            EcologyTrainingEpisodeReport,
        ],
        None,
    ]
    | None = None,
) -> tuple[
    tuple[AntLearningCheckpoint, ...],
    tuple[bytes, ...],
    tuple[EcologyTrainingEpisodePlan, ...],
    tuple[EcologyTrainingEpisodeReport, ...],
    tuple[EcologyStageMastery, ...],
]:
    checkpoints = initial
    stages = ECOLOGY_TRAINING_STAGES
    reports: list[EcologyTrainingEpisodeReport] = []
    plans: list[EcologyTrainingEpisodePlan] = []
    mastery: list[EcologyStageMastery] = []
    runner: KernelColonyRunner | None = None
    action_probe_baseline_reports = None
    if schedule_start_index < 0:
        raise ValueError("schedule_start_index must be non-negative")
    if schedule is None and schedule_start_index:
        raise ValueError(
            "schedule_start_index requires an explicit fixed schedule"
        )
    if schedule is not None and schedule_start_index > len(schedule):
        raise ValueError(
            "schedule_start_index exceeds fixed schedule length"
        )
    if schedule is not None:
        # Refuse the whole frozen schedule before burning any compute: a lane
        # that replays episodes it cannot sample produces mastery numbers that
        # only zero-shot transfer could ever satisfy.
        _require_samplable_milestones(config=config, schedule=schedule)
    if config.action_probe_guard_enabled:
        action_probe_baseline_reports = (
            await run_ecology_checkpoint_action_probes(
                temporal_latent_dim=config.temporal_latent_dim,
                seed=config.seed + 700_003,
                checkpoints=initial,
                code_delta_threshold=(
                    config.action_probe_code_delta_threshold
                ),
                turn_delta_threshold=(
                    config.action_probe_turn_delta_threshold
                ),
            )
        )
    if schedule is not None:
        for schedule_index, plan in enumerate(
            schedule[schedule_start_index:],
            start=schedule_start_index,
        ):
            runner, checkpoints, report = await _run_training_episode(
                config=config,
                checkpoints=checkpoints,
                arm=arm,
                optimize=optimize,
                local_valence_enabled=local_valence_enabled,
                segment_credit_enabled=segment_credit_enabled,
                prediction_error_enabled=prediction_error_enabled,
                temporal_writeback_enabled=temporal_writeback_enabled,
                plan=plan,
                action_probe_baseline=initial,
                action_probe_baseline_reports=(
                    action_probe_baseline_reports
                ),
            )
            plans.append(plan)
            reports.append(report)
            if episode_callback is not None:
                episode_callback(
                    schedule_index,
                    runner,
                    checkpoints,
                    report,
                )
            print(
                (
                    f"[ecology:{arm}] replay stage={plan.stage.value} "
                    f"tier={plan.tier.value} episode={plan.episode_index} "
                    f"interleaved={plan.interleaved} "
                    f"pickups={report.pickups} deliveries={report.deliveries}"
                ),
                flush=True,
            )
    else:
        mastered_stages: list[EcologyStage] = []
        for stage_index, stage in enumerate(stages):
            stage_reports: list[EcologyTrainingEpisodeReport] = []
            stage_mastery = _stage_mastery(
                config=config,
                stage=stage,
                reports=(),
            )
            for episode in range(config.stage_episodes):
                plan = EcologyTrainingEpisodePlan(
                    stage=stage,
                    tier=_tier_for_episode(episode),
                    seed=(
                        config.seed
                        + stage_index * 10_000
                        + episode * 101
                    ),
                    episode_index=episode,
                    interleaved=False,
                    forced_escape=(
                        stage is EcologyStage.BURNING_MATCH
                        and episode == 0
                    ),
                )
                # The generated schedule is built one episode at a time, so
                # the arm-level refusal has to be applied per plan here.
                _require_samplable_milestones(
                    config=config,
                    schedule=(plan,),
                )
                runner, checkpoints, report = (
                    await _run_training_episode(
                        config=config,
                        checkpoints=checkpoints,
                        arm=arm,
                        optimize=optimize,
                        local_valence_enabled=(
                            local_valence_enabled
                        ),
                        segment_credit_enabled=(
                            segment_credit_enabled
                        ),
                        prediction_error_enabled=(
                            prediction_error_enabled
                        ),
                        temporal_writeback_enabled=(
                            temporal_writeback_enabled
                        ),
                        plan=plan,
                        action_probe_baseline=initial,
                        action_probe_baseline_reports=(
                            action_probe_baseline_reports
                        ),
                    )
                )
                plans.append(plan)
                reports.append(report)
                stage_reports.append(report)
                primary_episodes = len(stage_reports)
                stage_mastery = _stage_mastery(
                    config=config,
                    stage=stage,
                    reports=tuple(stage_reports),
                )
                layout = stage_mastery.layout_results[-1]
                print(
                    (
                        f"[ecology:{arm}] stage={stage.value} "
                        f"tier={plan.tier.value} episode={episode} "
                        f"capability={layout.capability.value} "
                        f"bodies={layout.successful_bodies}/"
                        f"{layout.required_bodies} "
                        f"layout_success={layout.layout_success} "
                        f"mastery_eligible={layout.mastery_eligible} "
                        f"harmful_rate={layout.harmful_tick_rate:.4f} "
                        f"payoffs={report.nonzero_ecology_payoffs}"
                    ),
                    flush=True,
                )
                if (
                    mastered_stages
                    and primary_episodes % config.interleave_every == 0
                ):
                    replay_stage = mastered_stages[
                        (primary_episodes - 1)
                        % len(mastered_stages)
                    ]
                    replay_plan = EcologyTrainingEpisodePlan(
                        stage=replay_stage,
                        tier=EcologyTrainingTier.MEDIUM,
                        seed=(
                            config.seed
                            + 500_000
                            + stage_index * 10_000
                            + episode * 101
                        ),
                        episode_index=episode,
                        interleaved=True,
                        forced_escape=False,
                    )
                    _require_samplable_milestones(
                        config=config,
                        schedule=(replay_plan,),
                    )
                    runner, checkpoints, replay_report = (
                        await _run_training_episode(
                            config=config,
                            checkpoints=checkpoints,
                            arm=arm,
                            optimize=optimize,
                            local_valence_enabled=(
                                local_valence_enabled
                            ),
                            segment_credit_enabled=(
                                segment_credit_enabled
                            ),
                            plan=replay_plan,
                            action_probe_baseline=initial,
                            action_probe_baseline_reports=(
                                action_probe_baseline_reports
                            ),
                        )
                    )
                    plans.append(replay_plan)
                    reports.append(replay_report)
                    print(
                        (
                            f"[ecology:{arm}] interleaved={replay_stage.value} "
                            f"pickups={replay_report.pickups} "
                            f"deliveries={replay_report.deliveries}"
                        ),
                        flush=True,
                    )
                if stage_mastery.reached:
                    print(
                        (
                            f"[ecology:{arm}] mastery reached for "
                            f"{stage.value} after "
                            f"{stage_mastery.primary_episodes} episodes"
                        ),
                        flush=True,
                    )
                    break
            mastery.append(stage_mastery)
            print(
                (
                    f"[ecology:{arm}] stage={stage.value} done "
                    f"reached={stage_mastery.reached} "
                    f"layouts={stage_mastery.successful_layouts}/"
                    f"{stage_mastery.mastery_layouts} "
                    f"(required {stage_mastery.required_layouts}, "
                    f"{stage_mastery.bootstrap_layouts} bootstrap-only)"
                ),
                flush=True,
            )
            if stage_mastery.reached:
                mastered_stages.append(stage)
    if runner is None:
        raise RuntimeError("ecology training schedule must not be empty")
    archives = runner.export_learning_checkpoint_archives(
        checkpoint_prefix=f"ecology:{arm}:trained",
    )
    return (
        checkpoints,
        archives,
        tuple(plans),
        tuple(reports),
        tuple(mastery),
    )


def _latest_replay_counts(
    runner: KernelColonyRunner,
) -> tuple[int, int, int, int, int, int]:
    if not runner.rounds:
        return 0, 0, 0, 0, 0, 0
    records = runner.rounds[-1].ant_steps
    return (
        sum(item.runtime_replay_captured for item in records),
        sum(item.runtime_replay_settled for item in records),
        sum(item.runtime_replay_lineage_matches for item in records),
        sum(item.runtime_replay_pending_captures for item in records),
        sum(item.runtime_replay_staged_rollouts for item in records),
        sum(len(item.runtime_replay_drop_reasons) for item in records),
    )


async def _evaluate_arm(
    *,
    config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    arm: str,
    data_split: EcologyDataSplit,
    scenario: EcologyEvaluationScenario,
    seed: int,
    tier: EcologyTrainingTier = EcologyTrainingTier.FAR,
    prediction_error_enabled: bool = True,
    temporal_writeback_enabled: bool = True,
) -> EcologyArmMetrics:
    scenario_stage = {
        EcologyEvaluationScenario.BUTTER_ONLY: EcologyStage.BUTTER,
        EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK: (
            EcologyStage.WOOD_STICK
        ),
        EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE: (
            EcologyStage.BURNING_MATCH
        ),
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE: (
            EcologyStage.BURNING_MATCH
        ),
        EcologyEvaluationScenario.COMPOSITE: EcologyStage.COMPOSITE,
    }[scenario]
    forced_escape = (
        scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
    )
    world = _world(
        config=config,
        stage=scenario_stage,
        seed=seed,
        data_split=data_split,
        tier=tier,
        forced_escape=forced_escape,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=seed,
            session_id=(
                f"ecology:{arm}:{data_split.value}:"
                f"{scenario.value}:{seed}"
            ),
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=False,
            # A matched ablation must remain ablated at test time: PE-off and
            # ETA-off keep their wiring during the frozen held-out rollout, or
            # the arm silently reverts to the learned configuration and the
            # causal comparison measures nothing.
            prediction_error_enabled=prediction_error_enabled,
            temporal_writeback_enabled=temporal_writeback_enabled,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    rounds = (
        config.validation_rounds
        if data_split is EcologyDataSplit.VALIDATION
        else config.heldout_rounds
    )
    await runner.run(rounds)
    records = _flatten_records(runner)
    after = runner.export_learning_checkpoints(
        checkpoint_prefix=(
            f"ecology:{arm}:{data_split.value}:{scenario.value}:"
            f"{seed}:frozen-check"
        ),
        include_runtime_replay=False,
    )
    captured, settled, lineage, pending, staged, drop_count = _latest_replay_counts(runner)
    eligible_captures = captured - pending
    if eligible_captures < 0 or settled > eligible_captures:
        raise RuntimeError("runtime replay counters violate eligible settlement ordering")
    return EcologyArmMetrics(
        arm=arm,
        data_split=data_split,
        stage=scenario_stage,
        scenario=scenario,
        seed=seed,
        pickups=world.food_pickups,
        deliveries=world.food_delivered,
        obstacle_contacts=sum(item.obstacle_contact for item in records),
        harmful_heat_ticks=sum(item.heat_harmful for item in records),
        heat_entries=sum(item.entered_harmful_heat for item in records),
        heat_escapes=sum(item.escaped_harmful_heat for item in records),
        applied_distance=sum(item.applied_step for item in records),
        replay_captured=captured,
        replay_settled=settled,
        replay_lineage_matches=lineage,
        replay_pending_captures=pending,
        replay_staged_rollouts=staged,
        replay_drop_count=drop_count,
        replay_settlement_coverage=(settled / eligible_captures if eligible_captures else 0.0),
        replay_lineage_coverage=(lineage / settled if settled else 0.0),
        nonzero_ecology_payoffs=sum(
            abs(item.ecology_action_payoff) > 1e-12
            for item in records
        ),
        activated_sense_channels=_activated_sense_channels(records),
        first_pickup_tick=_first_tick(
            records,
            predicate=lambda item: item.at_food,
        ),
        first_obstacle_contact_tick=_first_tick(
            records,
            predicate=lambda item: item.obstacle_contact,
        ),
        first_heat_entry_tick=_first_tick(
            records,
            predicate=lambda item: item.entered_harmful_heat,
        ),
        first_heat_escape_tick=_first_tick(
            records,
            predicate=lambda item: item.escaped_harmful_heat,
        ),
        minimum_food_distance=_minimum_optional(
            tuple(item.nearest_food_distance for item in records)
        ),
        minimum_obstacle_distance=_minimum_optional(
            tuple(item.nearest_obstacle_distance for item in records)
        ),
        minimum_heat_distance=_minimum_optional(
            tuple(item.nearest_heat_distance for item in records)
        ),
        switch_count=sum(int(item.is_switching) for item in records),
        mean_persistence_steps=(
            sum(item.steps_since_switch for item in records)
            / len(records)
            if records
            else 0.0
        ),
        closed_segment_count=max(
            (
                item.runtime_closed_segments
                for item in records
            ),
            default=0,
        ),
        longest_segment_length=max(
            (
                item.runtime_longest_segment_length
                for item in records
            ),
            default=0,
        ),
        mean_absolute_turn_delta=_mean_absolute_turn_delta(
            records,
            n_ants=config.n_ants,
        ),
        policy_fingerprint_stable=(
            _policy_fingerprints(after)
            == _policy_fingerprints(checkpoints)
        ),
        temporal_learning_fingerprint_stable=(
            tuple(
                item.temporal_learning_fingerprint
                for item in after
            )
            == tuple(
                item.temporal_learning_fingerprint
                for item in checkpoints
            )
        ),
        body_lineage=_body_episode_lineage(
            records=records,
            plan=EcologyTrainingEpisodePlan(
                stage=scenario_stage,
                tier=tier,
                seed=seed,
                episode_index=0,
                interleaved=False,
                forced_escape=forced_escape,
            ),
            arm=arm,
            n_ants=config.n_ants,
        ),
    )


def _scenario_metrics(
    metrics: tuple[EcologyArmMetrics, ...],
    scenario: EcologyEvaluationScenario,
) -> tuple[EcologyArmMetrics, ...]:
    return tuple(
        item for item in metrics if item.scenario is scenario
    )


def _probe_requirement_met(probe: EcologyActionProbe) -> bool:
    """Valenced channels must drive action; neutral geometry only senses."""
    if probe.kind is EcologyProbeKind.OBSTACLE:
        # Wood sticks are neutral: they must reach the sensors, but are not
        # required to drive turns -- demanding stick-driven action would
        # reintroduce the avoidance objective the stick no longer carries.
        return probe.input_reachable
    if probe.kind is EcologyProbeKind.HOME:
        return (
            probe.input_reachable
            and probe.action_sensitive
            and probe.target_aligned
        )
    return probe.input_reachable and probe.action_sensitive


def _checkpoint_state_fingerprints(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            item.policy_fingerprint,
            item.temporal_fingerprint,
            item.memory_fingerprint,
        )
        for item in checkpoints
    )


def _verify_checkpoint_archives(
    *,
    config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    archives: tuple[bytes, ...],
) -> bool:
    runner = KernelColonyRunner(
        _world(
            config=config,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed + 900_001,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.FAR,
        ),
        base_config=_session_config(
            config=config,
            seed=config.seed + 900_001,
            session_id="ecology:archive-verification",
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=False,
        ),
    )
    runner.restore_learning_checkpoint_archives(archives)
    restored = runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:archive-verification:restored",
        include_runtime_replay=False,
    )
    if _checkpoint_state_fingerprints(restored) != (
        _checkpoint_state_fingerprints(checkpoints)
    ):
        return False
    pre_failure = runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:archive-verification:pre-failure",
        include_runtime_replay=False,
    )
    corrupted = list(archives)
    corrupted[-1] = corrupted[-1][:-1] + b"!"
    try:
        runner.restore_learning_checkpoint_archives(tuple(corrupted))
    except AgentLearningArchiveError:
        pass
    else:
        return False
    post_failure = runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:archive-verification:post-failure",
        include_runtime_replay=False,
    )
    return _checkpoint_state_fingerprints(post_failure) == (
        _checkpoint_state_fingerprints(pre_failure)
    )


def _paired_bootstrap_mean_ci(
    differences: tuple[float, ...],
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float, float]:
    if not differences:
        raise ValueError(
            "paired bootstrap requires at least one difference"
        )
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(samples, len(values)),
    )
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _ecology_outcome_score(item: EcologyArmMetrics) -> float:
    # Neutral wood-stick contacts are diagnostics only and never scored.
    return (
        item.pickups * 0.5
        + item.deliveries
        + item.heat_escapes * 0.25
        - item.harmful_heat_ticks * 0.02
    )


def _ecology_outcome_scores_by_seed(
    metrics: tuple[EcologyArmMetrics, ...],
    *,
    seeds: tuple[int, ...],
) -> tuple[float, ...]:
    result: list[float] = []
    for seed in seeds:
        seed_metrics = tuple(
            item for item in metrics if item.seed == seed
        )
        if len(seed_metrics) != len(EcologyEvaluationScenario):
            raise ValueError(
                "each held-out seed must contain every ecology scenario"
            )
        result.append(
            sum(_ecology_outcome_score(item) for item in seed_metrics)
        )
    return tuple(result)


def _build_gates(
    *,
    config: EcologyCurriculumConfig,
    initial: tuple[AntLearningCheckpoint, ...],
    learned: tuple[AntLearningCheckpoint, ...],
    no_optimize: tuple[AntLearningCheckpoint, ...],
    learned_training: tuple[EcologyTrainingEpisodeReport, ...],
    learned_mastery: tuple[EcologyStageMastery, ...],
    action_probes: tuple[EcologyActionProbe, ...],
    archive_roundtrip_verified: bool,
    learned_metrics: tuple[EcologyArmMetrics, ...],
    cold_metrics: tuple[EcologyArmMetrics, ...],
    no_optimize_metrics: tuple[EcologyArmMetrics, ...],
    dense_local_shaping_off_metrics: tuple[EcologyArmMetrics, ...],
    segment_credit_off_metrics: tuple[EcologyArmMetrics, ...],
) -> tuple[EcologyGate, ...]:
    evaluation_schedule = tuple(
        (item.scenario, item.seed) for item in learned_metrics
    )
    for arm_metrics in (
        cold_metrics,
        no_optimize_metrics,
        dense_local_shaping_off_metrics,
        segment_credit_off_metrics,
    ):
        if tuple(
            (item.scenario, item.seed) for item in arm_metrics
        ) != evaluation_schedule:
            raise ValueError(
                "held-out arm scenario/seed schedules must align"
            )

    learned_butter = _scenario_metrics(
        learned_metrics,
        EcologyEvaluationScenario.BUTTER_ONLY,
    )
    cold_butter = _scenario_metrics(
        cold_metrics,
        EcologyEvaluationScenario.BUTTER_ONLY,
    )
    no_opt_butter = _scenario_metrics(
        no_optimize_metrics,
        EcologyEvaluationScenario.BUTTER_ONLY,
    )
    learned_stick = _scenario_metrics(
        learned_metrics,
        EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK,
    )
    learned_heat_route = _scenario_metrics(
        learned_metrics,
        EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE,
    )
    learned_heat_escape = _scenario_metrics(
        learned_metrics,
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE,
    )
    no_opt_heat_escape = _scenario_metrics(
        no_optimize_metrics,
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE,
    )
    learned_composite = _scenario_metrics(
        learned_metrics,
        EcologyEvaluationScenario.COMPOSITE,
    )
    cold_composite = _scenario_metrics(
        cold_metrics,
        EcologyEvaluationScenario.COMPOSITE,
    )
    no_opt_composite = _scenario_metrics(
        no_optimize_metrics,
        EcologyEvaluationScenario.COMPOSITE,
    )
    expected_seeds = tuple(item.seed for item in learned_butter)
    learned_scenario_groups = (
        learned_butter,
        learned_stick,
        learned_heat_route,
        learned_heat_escape,
        learned_composite,
    )
    if not expected_seeds or any(
        tuple(item.seed for item in group) != expected_seeds
        for group in learned_scenario_groups
    ):
        raise ValueError(
            "each held-out scenario must use the same seed schedule"
        )

    required_successes = max(
        1,
        math.ceil(len(expected_seeds) * 0.6),
    )
    butter_vs_cold_ci = _paired_bootstrap_mean_ci(
        tuple(
            float(learned_item.pickups - control_item.pickups)
            for learned_item, control_item in zip(
                learned_butter,
                cold_butter,
                strict=True,
            )
        ),
        seed=config.seed + 70_001,
    )
    butter_vs_no_opt_ci = _paired_bootstrap_mean_ci(
        tuple(
            float(learned_item.pickups - control_item.pickups)
            for learned_item, control_item in zip(
                learned_butter,
                no_opt_butter,
                strict=True,
            )
        ),
        seed=config.seed + 70_003,
    )
    pickup_gain = (
        butter_vs_cold_ci[1] > 0.0
        and butter_vs_no_opt_ci[1] > 0.0
        and sum(item.pickups > 0 for item in learned_butter)
        >= required_successes
    )
    delivery_present = (
        sum(item.deliveries > 0 for item in learned_butter)
        >= required_successes
    )
    stick_task_successes = sum(
        item.pickups > 0 and item.deliveries > 0
        for item in learned_stick
    )
    # The stick is neutral geometry: the gate only asks whether foraging
    # still succeeds with the physical constraint present. Contact counts
    # are recorded as diagnostics and never compared or rewarded.
    stick_context_robustness = (
        stick_task_successes >= required_successes
    )
    route_task_successes = sum(
        item.pickups > 0 and item.deliveries > 0
        for item in learned_heat_route
    )
    route_harmful_rate = (
        sum(item.harmful_heat_ticks for item in learned_heat_route)
        / max(
            len(learned_heat_route)
            * config.n_ants
            * config.heldout_rounds,
            1,
        )
    )
    route_avoidance = (
        route_task_successes >= required_successes
        and route_harmful_rate <= 0.05
    )
    required_escapes_per_seed = max(
        1,
        math.ceil(config.n_ants * 0.6),
    )
    forced_escape = (
        sum(
            item.heat_escapes >= required_escapes_per_seed
            for item in learned_heat_escape
        )
        >= required_successes
        and sum(
            item.harmful_heat_ticks for item in learned_heat_escape
        )
        <= sum(
            item.harmful_heat_ticks for item in no_opt_heat_escape
        )
    )
    composite_successes = sum(
        item.pickups > 0 and item.deliveries > 0
        for item in learned_composite
    )
    # Neutral stick contacts are excluded from the composite comparison;
    # only foraging success and harmful heat exposure carry valence.
    composite_performance = (
        composite_successes >= required_successes
        and sum(
            item.harmful_heat_ticks for item in learned_composite
        )
        <= sum(
            item.harmful_heat_ticks for item in no_opt_composite
        )
        and sum(item.pickups for item in learned_composite)
        > sum(item.pickups for item in cold_composite)
    )
    learned_outcome_score = sum(
        _ecology_outcome_score(item) for item in learned_metrics
    )
    dense_local_shaping_off_score = sum(
        _ecology_outcome_score(item)
        for item in dense_local_shaping_off_metrics
    )
    segment_off_score = sum(
        _ecology_outcome_score(item)
        for item in segment_credit_off_metrics
    )
    learned_scores_by_seed = _ecology_outcome_scores_by_seed(
        learned_metrics,
        seeds=expected_seeds,
    )
    matched_control_cis = tuple(
        (
            control_name,
            _paired_bootstrap_mean_ci(
                tuple(
                    learned_score - control_score
                    for learned_score, control_score in zip(
                        learned_scores_by_seed,
                        _ecology_outcome_scores_by_seed(
                            control_metrics,
                            seeds=expected_seeds,
                        ),
                        strict=True,
                    )
                ),
                seed=config.seed + seed_offset,
            ),
        )
        for control_name, control_metrics, seed_offset in (
            ("cold", cold_metrics, 71_001),
            ("no_optimize", no_optimize_metrics, 71_003),
            (
                "dense_local_shaping_off",
                dense_local_shaping_off_metrics,
                71_005,
            ),
            (
                "segment_credit_off",
                segment_credit_off_metrics,
                71_007,
            ),
        )
    )
    matched_ablation_advantage = (
        all(ci[1] > 0.0 for _, ci in matched_control_cis)
    )
    temporal_dynamics = (
        any(item.switch_count > 0 for item in learned_metrics)
        and any(
            item.longest_segment_length > 1
            for item in learned_metrics
        )
        and all(
            item.mean_persistence_steps >= 0.0
            for item in learned_metrics
        )
    )
    action_smoothness = all(
        item.mean_absolute_turn_delta <= 0.55
        for item in learned_metrics
    )
    replay_ok = all(
        item.replay_settlement_coverage >= 0.99
        and item.replay_lineage_coverage >= 0.99
        and item.replay_drop_count == 0
        and item.policy_fingerprint_stable
        and item.temporal_learning_fingerprint_stable
        for item in learned_metrics
    )
    active_training_channels = frozenset(
        channel
        for episode in learned_training
        for channel in episode.activated_sense_channels
    )
    missing_critical_channels = tuple(
        sorted(
            ECOLOGY_CRITICAL_ACTIVE_CHANNELS
            - active_training_channels
        )
    )

    unsamplable_episodes = tuple(
        (
            item.plan.stage.value,
            item.plan.tier.value,
            item.rounds,
            item.milestone_round_budget,
        )
        for item in learned_training
        if not item.milestone_samplable
    )

    return (
        EcologyGate(
            name="training_layout_mastery",
            passed=(
                len(learned_mastery) == len(ECOLOGY_TRAINING_STAGES)
                and all(item.reached for item in learned_mastery)
                and not missing_critical_channels
            ),
            observed=str(
                {
                    "mastery": tuple(
                        (
                            item.stage.value,
                            item.reached,
                            tuple(
                                (
                                    entry.capability.value,
                                    f"{entry.successful_layouts}/"
                                    f"{entry.mastery_layouts}",
                                    f"required={entry.required_layouts}",
                                )
                                for entry in item.capability_results
                            ),
                            f"bodies>={item.required_bodies}",
                            f"bootstrap_only={item.bootstrap_layouts}",
                        )
                        for item in learned_mastery
                    ),
                    "missing_critical_channels": (
                        missing_critical_channels
                    )
                }
            ),
            threshold=(
                "every valenced training stage reaches its predeclared "
                "per-layout success RATE (>=60% of mastery-eligible layouts "
                "SEPARATELY per capability, >=60% of bodies closing the loop "
                "in the same body) within bounded budget, near foraging "
                "layouts stay Stage-0 bootstrap, and every critical ecology "
                "sensor channel activates"
            ),
        ),
        EcologyGate(
            name="training_tier_milestone_samplable",
            passed=not unsamplable_episodes,
            observed=(
                "unsamplable=" + str(unsamplable_episodes)
                if unsamplable_episodes
                else "all training episodes can reach their own milestone; "
                + str(
                    tuple(
                        (
                            tier.value,
                            ecology_tier_round_budget(tier),
                        )
                        for tier in EcologyTrainingTier
                    )
                )
            ),
            threshold=(
                "every training episode runs at least the rounds SUFFICIENT "
                "for its own milestone under the frozen plant: straight legs "
                "at the reachable speed band "
                f"[{ECOLOGY_PLANT_STEP_FLOOR:.3f}, "
                f"{ECOLOGY_PLANT_STEP_CEILING:.3f}] units/tick (the graded "
                "controller can neither stop nor slow below half speed) plus "
                "the re-aiming legs the plant charges at spawn and at the "
                "post-pickup turn-around, at "
                f"{ECOLOGY_PLANT_MAX_TURN_RATE:.3f} rad/act; an episode "
                "shorter than that trains a milestone it cannot sample, so "
                "the tier's mastery could only be met by zero-shot transfer"
            ),
        ),
        EcologyGate(
            name="paired_action_sensitivity",
            passed=(
                len(action_probes) == 4
                and all(
                    _probe_requirement_met(item)
                    for item in action_probes
                )
            ),
            observed=str(
                tuple(
                    (
                        item.kind.value,
                        item.code_l1_delta,
                        item.turn_delta,
                    )
                    for item in action_probes
                )
            ),
            threshold=(
                "food and heat paired swaps each change code and motor "
                "turn; the neutral stick swap only needs to reach input"
            ),
        ),
        EcologyGate(
            name="policy_changed",
            passed=_policy_fingerprints(learned) != _policy_fingerprints(initial),
            observed=str(_policy_fingerprints(learned)),
            threshold="learned policy fingerprint differs from initial",
        ),
        EcologyGate(
            name="no_optimize_policy_stable",
            passed=_policy_fingerprints(no_optimize) == _policy_fingerprints(initial),
            observed=str(_policy_fingerprints(no_optimize)),
            threshold="no-optimize policy fingerprint equals initial",
        ),
        EcologyGate(
            name="butter_pickup_gain",
            passed=pickup_gain,
            observed=(
                f"learned={sum(item.pickups for item in learned_butter)}, "
                f"cold={sum(item.pickups for item in cold_butter)}, "
                f"no_opt={sum(item.pickups for item in no_opt_butter)}, "
                f"vs_cold_ci={butter_vs_cold_ci}, "
                f"vs_no_opt_ci={butter_vs_no_opt_ci}"
            ),
            threshold=(
                "butter pickups exceed cold/no-optimize in aggregate and "
                "occur on >=60% held-out seeds"
            ),
        ),
        EcologyGate(
            name="butter_delivery_present",
            passed=delivery_present,
            observed=str(tuple(item.deliveries for item in learned_butter)),
            threshold="butter delivery occurs on >=60% held-out seeds",
        ),
        EcologyGate(
            name="neutral_stick_context_robustness",
            passed=stick_context_robustness,
            observed=(
                f"task_successes={stick_task_successes}, "
                "diagnostic_contacts="
                f"{sum(item.obstacle_contacts for item in learned_stick)}"
            ),
            threshold=(
                "pickup+delivery on >=60% neutral-stick seeds; contacts are "
                "diagnostics only and never gated or rewarded"
            ),
        ),
        EcologyGate(
            name="burning_match_route_avoidance",
            passed=route_avoidance,
            observed=(
                f"task_successes={route_task_successes}, "
                f"harmful_tick_rate={route_harmful_rate:.6f}"
            ),
            threshold=(
                "route foraging succeeds on >=60% seeds while harmful heat "
                "occupies <=5% ant-ticks; entry is not required"
            ),
        ),
        EcologyGate(
            name="burning_match_forced_escape",
            passed=forced_escape,
            observed=(
                "learned="
                f"{tuple(item.heat_escapes for item in learned_heat_escape)}, "
                "harmful_ticks="
                f"{sum(item.harmful_heat_ticks for item in learned_heat_escape)}, "
                "no_opt_harmful_ticks="
                f"{sum(item.harmful_heat_ticks for item in no_opt_heat_escape)}"
            ),
            threshold=(
                ">=60% ants escape on >=60% forced-start seeds with harmful "
                "ticks no worse than no-optimize"
            ),
        ),
        EcologyGate(
            name="composite_performance",
            passed=composite_performance,
            observed=str(
                tuple(
                    (
                        item.pickups,
                        item.deliveries,
                        item.obstacle_contacts,
                        item.harmful_heat_ticks,
                    )
                    for item in learned_composite
                )
            ),
            threshold=(
                "pickup+delivery on >=60% composite seeds, pickup exceeds "
                "cold, harmful heat no worse than no-optimize; neutral "
                "stick contacts are diagnostics only"
            ),
        ),
        EcologyGate(
            name="matched_ablation_advantage",
            passed=matched_ablation_advantage,
            observed=(
                f"learned={learned_outcome_score:.6f}, "
                "dense_local_shaping_off="
                f"{dense_local_shaping_off_score:.6f}, "
                f"segment_off={segment_off_score:.6f}, "
                f"paired_cis={matched_control_cis}"
            ),
            threshold=(
                "paired bootstrap 95% CI lower bound >0 against cold, "
                "no-optimize, dense-local-shaping-off and "
                "segment-credit-off"
            ),
        ),
        EcologyGate(
            name="temporal_dynamics",
            passed=temporal_dynamics,
            observed=str(
                tuple(
                    (
                        item.scenario.value,
                        item.switch_count,
                        item.longest_segment_length,
                    )
                    for item in learned_metrics
                )
            ),
            threshold=(
                "held-out behavior contains switches and at least one "
                "multi-step beta segment"
            ),
        ),
        EcologyGate(
            name="action_smoothness",
            passed=action_smoothness,
            observed=str(
                tuple(
                    (
                        item.scenario.value,
                        item.mean_absolute_turn_delta,
                    )
                    for item in learned_metrics
                )
            ),
            threshold="mean absolute turn delta <=0.55 rad per scenario",
        ),
        EcologyGate(
            name="checkpoint_archive_roundtrip",
            passed=archive_roundtrip_verified,
            observed=str(archive_roundtrip_verified),
            threshold=(
                "fresh-session archive hydration succeeds and corrupt "
                "collection restore rolls back atomically"
            ),
        ),
        EcologyGate(
            name="runtime_replay_lineage",
            passed=replay_ok,
            observed=str(
                tuple(
                    (
                        item.scenario.value,
                        item.seed,
                        item.replay_settlement_coverage,
                        item.replay_lineage_coverage,
                        item.replay_pending_captures,
                        item.replay_drop_count,
                    )
                    for item in learned_metrics
                )
            ),
            threshold=(
                "frozen evaluation policy/temporal-learning fingerprints "
                "stay stable; "
                "eligible settlement and lineage >=0.99 with no drops"
            ),
        ),
    )


async def train_and_evaluate_ecology_checkpoint(
    config: EcologyCurriculumConfig,
) -> EcologyCheckpointCandidate:
    bootstrap_world: AntWorld = _world(
        config=config,
        stage=EcologyStage.COMPOSITE,
        seed=config.seed,
        data_split=EcologyDataSplit.TRAIN,
        tier=EcologyTrainingTier.NEAR,
    )
    bootstrap_runner = KernelColonyRunner(
        bootstrap_world,
        base_config=_session_config(
            config=config,
            seed=config.seed,
            session_id="ecology:shared-initial",
            optimize=True,
        ),
    )
    initial = bootstrap_runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:shared-initial",
        include_runtime_replay=False,
    )
    # Causal reachability gate: refuse long training when food/obstacle/heat
    # paired swaps cannot reach code from the shared initial. Under exclusive
    # steering the head is the only deterministic steering writer and its cold
    # parameters are exactly zero, so cold action sensitivity is 0 BY DESIGN:
    # turning is a capability the run must learn, not a precondition. The
    # cold gate therefore checks input reachability only; the post-training
    # required gates (paired_action_sensitivity, food_steering_alignment,
    # carrying_home_action_alignment) keep the strict learned-steering truth.
    pretraining_probes = await run_ecology_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + 700_003,
        checkpoint=initial[0],
    )
    failed_probes = tuple(
        item
        for item in pretraining_probes
        if not item.input_reachable
    )
    if failed_probes:
        detail = tuple(
            (
                item.kind.value,
                item.input_reachable,
                item.action_sensitive,
                item.code_l1_delta,
                item.turn_delta,
            )
            for item in failed_probes
        )
        raise RuntimeError(
            "ecology paired action probes failed before long training; "
            f"refusing curriculum budget: {detail}"
        )
    print(
        (
            "[ecology] pretraining probes passed: "
            + ", ".join(
                f"{item.kind.value}(code={item.code_l1_delta:.4f},"
                f"turn={item.turn_delta:.4f})"
                for item in pretraining_probes
            )
        ),
        flush=True,
    )
    (
        learned,
        learned_archives,
        training_schedule,
        learned_training,
        learned_mastery,
    ) = await _train_arm(
        config=config,
        initial=initial,
        arm="learned",
        optimize=True,
        local_valence_enabled=True,
        segment_credit_enabled=True,
    )
    (
        no_optimize,
        _,
        _,
        no_optimize_training,
        _,
    ) = await _train_arm(
        config=config,
        initial=initial,
        arm="no_optimize",
        optimize=False,
        local_valence_enabled=True,
        segment_credit_enabled=True,
        schedule=training_schedule,
    )
    (
        dense_local_shaping_off,
        _,
        _,
        dense_local_shaping_off_training,
        _,
    ) = await _train_arm(
        config=config,
        initial=initial,
        arm="dense_local_shaping_off",
        optimize=True,
        local_valence_enabled=False,
        segment_credit_enabled=True,
        schedule=training_schedule,
    )
    (
        segment_credit_off,
        _,
        _,
        segment_credit_off_training,
        _,
    ) = await _train_arm(
        config=config,
        initial=initial,
        arm="segment_credit_off",
        optimize=True,
        local_valence_enabled=True,
        segment_credit_enabled=False,
        schedule=training_schedule,
    )
    action_probes = await run_ecology_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + 800_003,
        checkpoint=learned[0],
    )
    archive_roundtrip_verified = _verify_checkpoint_archives(
        config=config,
        checkpoints=learned,
        archives=learned_archives,
    )
    evaluation_scenarios = (
        EcologyEvaluationScenario.BUTTER_ONLY,
        EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK,
        EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE,
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE,
        EcologyEvaluationScenario.COMPOSITE,
    )
    validation_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=learned,
                arm="learned",
                data_split=EcologyDataSplit.VALIDATION,
                scenario=scenario,
                seed=seed,
            )
            for scenario in evaluation_scenarios
            for seed in config.validation_seeds
        ]
    )
    learned_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=learned,
                arm="learned",
                data_split=EcologyDataSplit.HELDOUT,
                scenario=scenario,
                seed=seed,
            )
            for scenario in evaluation_scenarios
            for seed in config.heldout_seeds
        ]
    )
    cold_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=initial,
                arm="cold",
                data_split=EcologyDataSplit.HELDOUT,
                scenario=scenario,
                seed=seed,
            )
            for scenario in evaluation_scenarios
            for seed in config.heldout_seeds
        ]
    )
    no_optimize_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=no_optimize,
                arm="no_optimize",
                data_split=EcologyDataSplit.HELDOUT,
                scenario=scenario,
                seed=seed,
            )
            for scenario in evaluation_scenarios
            for seed in config.heldout_seeds
        ]
    )
    dense_local_shaping_off_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=dense_local_shaping_off,
                arm="dense_local_shaping_off",
                data_split=EcologyDataSplit.HELDOUT,
                scenario=scenario,
                seed=seed,
            )
            for scenario in evaluation_scenarios
            for seed in config.heldout_seeds
        ]
    )
    segment_credit_off_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=segment_credit_off,
                arm="segment_credit_off",
                data_split=EcologyDataSplit.HELDOUT,
                scenario=scenario,
                seed=seed,
            )
            for scenario in evaluation_scenarios
            for seed in config.heldout_seeds
        ]
    )
    gates = _build_gates(
        config=config,
        initial=initial,
        learned=learned,
        no_optimize=no_optimize,
        learned_training=learned_training,
        learned_mastery=learned_mastery,
        action_probes=action_probes,
        archive_roundtrip_verified=archive_roundtrip_verified,
        learned_metrics=learned_metrics,
        cold_metrics=cold_metrics,
        no_optimize_metrics=no_optimize_metrics,
        dense_local_shaping_off_metrics=dense_local_shaping_off_metrics,
        segment_credit_off_metrics=(
            segment_credit_off_metrics
        ),
    )
    breakpoints = tuple(gate.name for gate in gates if not gate.passed)
    verdict = "PASS" if not breakpoints else "BLOCK"
    report = EcologyCheckpointReport(
        schema_version=ECOLOGY_CURRICULUM_SCHEMA_VERSION,
        config=config,
        initial_policy_fingerprints=_policy_fingerprints(initial),
        learned_policy_fingerprints=_policy_fingerprints(learned),
        no_optimize_policy_fingerprints=_policy_fingerprints(no_optimize),
        training_schedule=training_schedule,
        learned_training=learned_training,
        no_optimize_training=no_optimize_training,
        dense_local_shaping_off_training=dense_local_shaping_off_training,
        segment_credit_off_training=segment_credit_off_training,
        learned_mastery=learned_mastery,
        action_probes=action_probes,
        validation_metrics=validation_metrics,
        learned_metrics=learned_metrics,
        cold_metrics=cold_metrics,
        no_optimize_metrics=no_optimize_metrics,
        dense_local_shaping_off_metrics=dense_local_shaping_off_metrics,
        segment_credit_off_metrics=segment_credit_off_metrics,
        gates=gates,
        verdict=verdict,
        diagnostic_breakpoints=breakpoints,
        description=(
            "PASS: ecology checkpoint cleared all frozen held-out gates"
            if verdict == "PASS"
            else "BLOCK: " + ", ".join(breakpoints)
        ),
    )
    return EcologyCheckpointCandidate(
        checkpoints=learned,
        checkpoint_archives=learned_archives,
        report=report,
    )


__all__ = [
    "ECOLOGY_CURRICULUM_SCHEMA_VERSION",
    "ECOLOGY_FOOD_ENCOUNTER_DISTANCE",
    "ECOLOGY_MEASURED_POLICY_STEP",
    "ECOLOGY_MIN_DELIVERY_LEG",
    "ECOLOGY_NEST_RADIUS",
    "ECOLOGY_PLANT_STEP_CEILING",
    "ECOLOGY_PLANT_STEP_FLOOR",
    "ECOLOGY_REQUIRED_GATE_NAMES",
    "ECOLOGY_TIER_GEOMETRY",
    "ECOLOGY_TRAINING_STAGES",
    "ECOLOGY_UNDIRECTED_ORBIT_REACH",
    "EcologyArmMetrics",
    "EcologyBodyEpisodeLineage",
    "EcologyCheckpointCandidate",
    "EcologyCheckpointReport",
    "EcologyCurriculumConfig",
    "EcologyDataSplit",
    "EcologyEvaluationScenario",
    "EcologyCapabilityMastery",
    "EcologyGate",
    "EcologyLayoutCapability",
    "EcologyLayoutMastery",
    "EcologyStage",
    "EcologyStageMastery",
    "EcologyTierGeometry",
    "EcologyTrainingEpisodePlan",
    "EcologyTrainingEpisodeReport",
    "EcologyTrainingTier",
    "ecology_tier_round_budget",
    "train_and_evaluate_ecology_checkpoint",
]
