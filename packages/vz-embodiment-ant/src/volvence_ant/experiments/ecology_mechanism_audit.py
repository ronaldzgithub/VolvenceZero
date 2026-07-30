"""P0 mechanism audit for the digital-ant ecology learning path.

The audit is intentionally diagnostic: it can only report PASS/BLOCK and never
produces a promotable checkpoint.  It binds deterministic paired action probes,
a scripted temporal transition protocol, and owner-scoped frozen-evaluation
fingerprints to the exact checkpoints observed during a short matched training
schedule.

Three properties make the verdict mean what ``research/ant/05`` s3 says it
means, and every one of them was missing from the v1/v2 audit:

* the per-episode action-chain rollback guard is **disabled** here (as P1/P2
  already do), so the gated checkpoint is the trained one rather than a
  restored cold checkpoint;
* the shared-initial checkpoint is gated on **input reachability only**,
  because under exclusive steering a cold action head is exactly zero by
  design (``docs/specs/digital-ant-embodiment.md`` s"冷启探针门"), and a
  retention floor derived from a zero baseline raises instead of silently
  evaluating to ``0``;
* every owner published in ``learning_owner_fingerprints`` is compared **per
  tick**, not endpoint-to-endpoint, and anything outside the declared
  allow-list blocks.

Backend evidence is split into two separately named gates.  Fusing them made
the admissible band ``0 < delta <= 1e-3``: a lane had to DIFFER from the pure
reference to count as evaluated, so a torch lane that ran correctly and agreed
exactly was reported as non-evaluation and blocked.  Now
``backend_lane_coverage`` asks "did this lane's declared backends execute?" and
answers it only from owner-published execution evidence, while
``backend_parity`` asks "do the measured lanes agree on final code, action
distribution and turn?" and lets exact agreement pass.  The parity probe also
carries a frozen number of optimization ticks, because
``temporal_ssl_backend`` / ``internal_rl_backend`` do no work outside an
optimization cycle and a single no-optimize probe step could never observe
them.

Requirements this audit knowingly does not meet are published in every
artifact as ``ECOLOGY_AUDIT_DECLARED_GAPS`` rather than living in a comment,
and every surface that is recorded but gates nothing is declared in
``EcologyMechanismAuditReport.diagnostic_surfaces``.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass
from enum import Enum

from volvence_ant.env import (
    AntWorld,
    AntWorldConfig,
    BurningMatch,
    ButterSource,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.ecology_curriculum import (
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyEvaluationScenario,
    EcologyStage,
    EcologyTrainingEpisodePlan,
    EcologyTrainingEpisodeReport,
    EcologyTrainingTier,
    _run_training_episode,
    _session_config,
    _world,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyActionProbe,
    EcologyBackendExecutionEvidence,
    EcologyCheckpointActionProbe,
    EcologyProbeBackendLane,
    EcologyProbeKind,
    ecology_probe_lane_declared_active_backends,
    ecology_probe_lane_expected_wiring,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
    AntStepRecord,
    KernelColonyRunner,
)


# ---------------------------------------------------------------------------
# Frozen schema and thresholds (plan 05 s2.1: thresholds are written into the
# schema/test before the first new result is read; afterwards only the
# implementation may be fixed, never the threshold relaxed).
# ---------------------------------------------------------------------------

# ``v1`` is the shape of the only committed artifact
# (research/ant/results/ecology_recovery/p0/ecology_mechanism_audit.v1.json);
# ``v2`` was an in-tree bump that never produced an artifact, and ``v3`` was an
# in-tree bump that never produced one either.  ``v4`` adds the backend
# coverage/parity split, the action-distribution parity dimension, the declared
# diagnostic-only surfaces and the beta histogram / SSL switch parameters, so
# the version advances again rather than silently re-using a name whose
# committed artifact has a different shape.  The driver's artifact filename
# carries the same ``v4`` token, and the frozen v1 artifact is never
# overwritten.
ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION = (
    "digital-ant-ecology-mechanism-audit.v4"
)

ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD = 1e-8
ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD = 1e-4
ECOLOGY_AUDIT_RETENTION_RATIO = 0.25
ECOLOGY_AUDIT_BODY_PASS_RATIO = 0.8
# plan 05:120 -- "方向符号在同一 checkpoint 的重复运行中一致".  Three repeats at
# distinct probe seeds; a sign that does not exist (|turn| below the frozen
# turn threshold) is NOT counted as consistent.
ECOLOGY_AUDIT_SIGN_REPEAT_COUNT = 3
ECOLOGY_AUDIT_SIGN_REPEAT_SEED_STRIDE = 1_009
# plan 05:162 -- the negative-control switch-rate ceiling is pre-declared.  A
# steady-state input may still carry the encoder warm-up transient of the first
# few ticks; more than one switch in five ticks is the "高频 alternating"
# failure the control exists to catch.
ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING = 0.2
# plan 05:164 -- "正常 trace 不能全部依赖 24-step timeout".  The ratio is
# bounded-horizon closures over all closures and the comparison is strict, so
# the ceiling encodes exactly the plan's requirement.  P1 may tighten it; it
# must never be raised.
ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING = 1.0
# plan 05:161 -- a positive-control switch must be localizable to a declared
# state change.  Four ticks is the widest window that still excludes the
# neighbouring phase in the shortest scripted phase (4 ticks).
ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW = 4
# plan 05:123 -- pure/runtime/torch agree "在约定容差内".  The ndarray runtime
# lane accumulates in float32 through the encoder recurrence while the pure
# lane is float64; 1e-3 absolute is the declared parity band for code, turn and
# action-head residual.
ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE = 1e-3
# plan 05:123 -- the parity lanes must actually RUN their backends.
# ``temporal_ssl_backend`` and ``internal_rl_backend`` only do work inside an
# optimization cycle, and the joint-loop schedule reaches its first full cycle
# (the tick that publishes ``latest_internal_rl_report``) on the sixth turn of
# a fresh session.  Six exercise ticks is therefore the SMALLEST budget at
# which every declared backend has had one opportunity to execute before the
# measured tick; anything smaller measures the probe, not the backend.  It is
# a probe budget, not an acceptance threshold: raising it cannot turn a BLOCK
# into a PASS, and the audit asserts the opportunity actually materialised
# instead of assuming this number is still right.
ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS = 6
# plan 05:165 -- segment-credit on/off may differ only in credit aggregation.
# Both lanes run the identical float64 forward path, so the only admissible
# difference is summation-order noise.
ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE = 1e-6
# plan 05:197 -- settlement/lineage >= 0.99, drop = 0.
ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR = 0.99

# plan 05:186-188 -- the two prescribed minimal repros.  These are literal
# held-out seeds, not derived offsets; the v1/v2 audit derived ``seed+101`` and
# ``seed+211`` so neither prescribed case could ever occur.
ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES: tuple[
    tuple[EcologyEvaluationScenario, int], ...
] = (
    (EcologyEvaluationScenario.BUTTER_ONLY, 307),
    (EcologyEvaluationScenario.HEAT_FORCED_ESCAPE, 101),
)
ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS: tuple[int, ...] = tuple(
    sorted({seed for _scenario, seed in ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES})
)

# plan 05:185 requires an EXPLICIT "allowed to change" vs "must not change"
# split and plan 05:193 requires per-tick invariance of every learned owner.
# The kernel publishes exactly these eight names under
# ``AgentLearningCheckpoint.learning_owner_fingerprints``; publishing a name
# there is the owner's own assertion that the digest covers learned state, so
# each one is gated and each entry records why.
ECOLOGY_AUDIT_GATED_LEARNING_OWNERS: tuple[tuple[str, str], ...] = (
    (
        "joint-loop/policy",
        "Internal-RL policy parameters in z_t space; the whole point of "
        "learning_enabled=False is that no policy update lands.",
    ),
    (
        "joint-loop/temporal-learning",
        "encoder / beta_t / decoder learned parameters; the embodiment spec "
        "requires policy and temporal-learning fingerprints to stay constant "
        "for the entire evaluation.",
    ),
    (
        "joint-loop/memory",
        "CMS strata are a background-slow adaptive layer. Consolidation is "
        "learning at a lower frequency, not episode bookkeeping, so a frozen "
        "evaluation must not move it.",
    ),
    (
        "prediction",
        "PredictionError is the primary learning signal owner. Per-turn "
        "prediction records are episodic, but the kernel folds them into the "
        "same digest it publishes as a LEARNING owner, so a change here is "
        "either real learning or a publisher contract defect; both block.",
    ),
    (
        "credit",
        "Credit assignment state feeds policy optimization. A frozen "
        "evaluation must not accumulate credit that a later run could learn "
        "from.",
    ),
    (
        "regime",
        "regime is a memorable, selectable, TRAINABLE runtime identity "
        "(AGENTS.md s2), not a prompt label; its learned selection statistics "
        "must be frozen with everything else.",
    ),
    (
        "dual-track-gate",
        "The dual-track gate learner arbitrates world/self writeback and is "
        "explicitly a learner.",
    ),
    (
        "reflection",
        "background-slow ReflectionEngine consolidation scores; evaluation is "
        "a downstream readout and must not drive reflection learning.",
    ),
)
# Intentionally empty. Runtime-scoped state that legitimately changes during a
# frozen evaluation -- replay capture/settlement counters, segment counters,
# episode counters, navigator/path-integration state -- is published OUTSIDE
# ``learning_owner_fingerprints`` and is therefore never compared here (plan
# 05:196: runtime controller and replay counters may change but must not be
# counted into a learning fingerprint).  If a learned-owner digest moves, the
# honest verdict is BLOCK: either the freeze leaks, or the publisher mixes
# episodic state into a learning fingerprint. Both are defects, and neither may
# be silenced by adding an entry here without a written justification.
ECOLOGY_AUDIT_ALLOWED_TO_CHANGE_OWNERS: tuple[tuple[str, str], ...] = ()

# ---------------------------------------------------------------------------
# Scripted, action-label-free transition protocol (plan 05 s3.3).
# ---------------------------------------------------------------------------

ECOLOGY_AUDIT_PROTOCOL_BUTTER_XY = (5.0, 0.0)
ECOLOGY_AUDIT_PROTOCOL_MATCH_XY = (0.0, 6.0)
ECOLOGY_AUDIT_PROTOCOL_CRUISE_XY = (2.0, -3.0)
ECOLOGY_AUDIT_PROTOCOL_STEADY_TICKS = 24


class EcologyTransitionPhase(str, Enum):
    CRUISE = "cruise"
    FOOD_APPROACH = "food_approach"
    PICKUP_CARRYING = "pickup_carrying"
    HOME_APPROACH_DELIVERY = "home_approach_delivery"
    SAFE_TO_HARMFUL = "safe_to_harmful"
    HARMFUL_TO_COOLING = "harmful_to_cooling"


ECOLOGY_AUDIT_PROTOCOL_PHASE_TICKS: tuple[
    tuple[EcologyTransitionPhase, int], ...
] = (
    (EcologyTransitionPhase.CRUISE, 5),
    (EcologyTransitionPhase.FOOD_APPROACH, 5),
    (EcologyTransitionPhase.PICKUP_CARRYING, 4),
    (EcologyTransitionPhase.HOME_APPROACH_DELIVERY, 5),
    (EcologyTransitionPhase.SAFE_TO_HARMFUL, 4),
    (EcologyTransitionPhase.HARMFUL_TO_COOLING, 5),
)


class EcologyMechanismAuditError(RuntimeError):
    """A P0 audit invariant was violated and no verdict may be produced."""


@dataclass(frozen=True)
class EcologyMechanismAuditConfig:
    n_ants: int = 4
    temporal_latent_dim: int = 16
    episode_rounds: int = 12
    episodes_per_stage: int = 3
    evaluation_rounds: int = 24
    seed: int = 0
    code_delta_threshold: float = ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD
    turn_delta_threshold: float = ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD
    retention_ratio_threshold: float = ECOLOGY_AUDIT_RETENTION_RATIO
    body_pass_ratio: float = ECOLOGY_AUDIT_BODY_PASS_RATIO
    negative_control_switch_rate_ceiling: float = (
        ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING
    )
    timeout_closure_ratio_ceiling: float = (
        ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING
    )
    switch_localization_window: int = ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW
    backend_parity_tolerance: float = ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE
    backend_parity_exercise_steps: int = (
        ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS
    )
    segment_credit_parity_tolerance: float = (
        ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE
    )
    sign_repeat_count: int = ECOLOGY_AUDIT_SIGN_REPEAT_COUNT

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.temporal_latent_dim < 3:
            raise ValueError("temporal_latent_dim must be >= 3")
        if self.episode_rounds < 1 or self.episodes_per_stage < 1:
            raise ValueError("training audit budgets must be >= 1")
        if self.evaluation_rounds < 3:
            raise ValueError(
                "evaluation_rounds must be >= 3 so replay can settle"
            )
        # plan 05 s2.1: P0 thresholds are frozen. They live as module
        # constants and the config may not re-open them; a run that wants a
        # different number is a different (documented) schema version.
        if self.code_delta_threshold != ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD:
            raise ValueError("P0 code-delta threshold is frozen at 1e-8")
        if self.turn_delta_threshold != ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD:
            raise ValueError("P0 turn-delta threshold is frozen at 1e-4")
        if self.retention_ratio_threshold != ECOLOGY_AUDIT_RETENTION_RATIO:
            raise ValueError("P0 retention ratio is frozen at 0.25")
        if self.body_pass_ratio != ECOLOGY_AUDIT_BODY_PASS_RATIO:
            raise ValueError("P0 body pass ratio is frozen at 0.8")
        if (
            self.negative_control_switch_rate_ceiling
            != ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING
        ):
            raise ValueError(
                "P0 negative-control switch-rate ceiling is frozen at 0.2"
            )
        if (
            self.timeout_closure_ratio_ceiling
            != ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING
        ):
            raise ValueError(
                "P0 timeout closure ratio ceiling is frozen at 1.0"
            )
        if (
            self.switch_localization_window
            != ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW
        ):
            raise ValueError("P0 switch localization window is frozen at 4")
        if (
            self.backend_parity_tolerance
            != ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE
        ):
            raise ValueError("P0 backend parity tolerance is frozen at 1e-3")
        if self.backend_parity_exercise_steps < (
            ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS
        ):
            # Lowering it un-exercises the backends; raising it only gives them
            # MORE opportunity to run, so it stays open upward.
            raise ValueError(
                "P0 backend-parity exercise budget may not drop below "
                f"{ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS}; a shorter "
                "probe cannot reach the joint-loop's first full optimization "
                "cycle and would measure the probe instead of the backend"
            )
        if (
            self.segment_credit_parity_tolerance
            != ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE
        ):
            raise ValueError(
                "P0 segment-credit parity tolerance is frozen at 1e-6"
            )
        if self.sign_repeat_count != ECOLOGY_AUDIT_SIGN_REPEAT_COUNT:
            raise ValueError("P0 sign-consistency repeat count is frozen at 3")
        if (self.seed + 43) in ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS:
            raise ValueError(
                "audit seed collides with a frozen held-out repro seed "
                f"{ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS}; validation and "
                "held-out namespaces must stay disjoint (plan 05 s2.1)"
            )


# ---------------------------------------------------------------------------
# Report value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EcologyActionChainSnapshot:
    arm: str
    label: str
    stage: str
    tier: str
    episode_index: int
    gate_mode: str
    body_reports: tuple[EcologyCheckpointActionProbe, ...]
    required_body_passes: int
    passing_bodies: int
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class EcologySignConsistency:
    body_id: int
    kind: str
    probe_seeds: tuple[int, ...]
    left_turn_signs: tuple[int, ...]
    right_turn_signs: tuple[int, ...]
    consistent: bool


@dataclass(frozen=True)
class EcologyLateralBias:
    kind: str
    body_count: int
    mean_common_mode: float
    mean_contrast: float
    systematic_same_direction: bool


@dataclass(frozen=True)
class EcologyActionHeadUpdate:
    body_id: int
    update_step: int
    residual_finite: bool
    policy_fingerprint_changed: bool
    passed: bool


@dataclass(frozen=True)
class EcologyBackendParityLane:
    """Coverage and parity evidence for one backend lane.

    The v3 audit fused two different questions into one ``evaluated`` flag and
    then required a non-reference lane to DIFFER from the reference before it
    counted as evaluated.  That made the admissible band ``0 < delta <= 1e-3``:
    a torch lane that ran correctly and agreed exactly was reported as
    "not evaluated" and BLOCKed.  The two questions are now separate:

    * COVERAGE -- ``covered`` -- did this lane's declared backends actually
      execute?  Answered only from owner-published execution evidence
      (``EcologyBackendExecutionEvidence``), never from "its numbers differ".
    * PARITY -- ``within_tolerance`` -- do the lanes that DID run agree on
      final code, action distribution and turn (plan 05:123)?  Exact agreement
      is the ideal outcome and passes.

    ``measured`` is the weaker precondition for parity: the lane's backend is
    importable and the session published the wiring the lane declares, so the
    deltas below are real numbers rather than ``None``.
    """

    lane: str
    measured: bool
    covered: bool
    not_measured_reason: str
    not_covered_reason: str
    declared_active_backends: tuple[str, ...]
    #: ``None`` exactly when the lane was not measured. A JSON artifact must
    #: never carry a non-standard ``NaN`` token, and "no measurement" is a
    #: different statement from "a measurement that happens to be nan".
    max_code_delta: float | None
    max_turn_delta: float | None
    max_step_delta: float | None
    max_action_head_residual_delta: float | None
    #: plan 05:123 also requires the ACTION DISTRIBUTION to agree. This is the
    #: largest per-family probability-mass difference against the reference.
    max_action_distribution_delta: float | None
    abstract_actions_agree: bool
    within_tolerance: bool
    observed_backend_wiring: tuple[tuple[str, str], ...] = ()
    distinguishable_from_reference: bool = False
    backend_execution: EcologyBackendExecutionEvidence | None = None


@dataclass(frozen=True)
class EcologySegmentTelemetry:
    arm: str
    stage: str
    tier: str
    episode_index: int
    switch_count: int
    closed_segment_count: int
    longest_segment_length: int
    close_reason_counts: tuple[tuple[str, int], ...]
    switch_gate_min: float
    switch_gate_mean: float
    switch_gate_max: float
    track_switch_gate_ranges: tuple[tuple[str, float, float], ...]
    action_chain_guard_passed: bool
    action_chain_rollback_applied: bool
    action_chain_failures: tuple[str, ...]


@dataclass(frozen=True)
class EcologyTemporalTick:
    tick: int
    phase: str
    world_beta: float
    self_beta: float
    world_beta_threshold: float
    self_beta_threshold: float
    world_beta_binary: int
    self_beta_binary: int
    track_switch_gates: tuple[tuple[str, float], ...]
    binary_switch: bool
    fast_prior_switch_pressure: float
    prediction_error_switch_pressure: float
    external_switch_pressure: float
    steps_since_switch: int
    open_segment_transitions: int
    closed_segments: int
    segment_closed_this_tick: bool
    last_segment_close_reason: str
    carrying_food: bool
    heat_harmful: bool
    picked_up: bool
    delivered: bool


@dataclass(frozen=True)
class EcologyBoundaryLocalization:
    phase: str
    boundary_tick: int
    nearest_switch_tick: int
    distance: int
    localized: bool


@dataclass(frozen=True)
class EcologySwitchParameterSnapshot:
    """Owner-published switch parameters at one end of a trace.

    plan 05:150 asks the per-tick log to carry "SSL 前后 switch 参数和
    histogram".  These are read from ``MetacontrollerRuntimeState`` before the
    first scripted tick and after the last one, per track.
    """

    label: str
    world_beta_threshold: float
    self_beta_threshold: float
    world_switch_sparsity: float
    self_switch_sparsity: float
    world_binary_switch_rate: float
    self_binary_switch_rate: float
    world_mean_persistence_window: float
    self_mean_persistence_window: float
    world_ssl_loss: float
    world_ssl_kl_loss: float
    self_ssl_loss: float
    self_ssl_kl_loss: float


@dataclass(frozen=True)
class EcologyTransitionTrace:
    label: str
    segment_credit_enabled: bool
    checkpoint_id: str
    ticks: tuple[EcologyTemporalTick, ...]
    boundary_localizations: tuple[EcologyBoundaryLocalization, ...]
    switch_ticks: tuple[int, ...]
    switch_rate: float
    close_reason_counts: tuple[tuple[str, int], ...]
    closed_segment_count: int
    timeout_closure_ratio: float
    pickups: int
    deliveries: int
    harmful_ticks: int
    #: Ten equal bins over [0, 1] of the per-tick continuous beta, per track.
    #: Derived by this audit from its own per-tick log: the owner's
    #: ``SwitchGateStats`` histogram is only published by the SSL trainer, and
    #: the ant's SSL trainer never trains (see ``declared_plan_deviations``).
    world_beta_histogram: tuple[int, ...] = ()
    self_beta_histogram: tuple[int, ...] = ()
    switch_parameters_before: EcologySwitchParameterSnapshot | None = None
    switch_parameters_after: EcologySwitchParameterSnapshot | None = None


@dataclass(frozen=True)
class EcologySegmentCreditParity:
    tolerance: float
    max_sense_delta: float
    max_turn_delta: float
    max_step_delta: float
    lineage_aligned: bool
    lineage_differences: tuple[tuple[str, str, str], ...]
    first_misaligned_tick: int
    passed: bool


@dataclass(frozen=True)
class EcologyTemporalSwitchAudit:
    positive_control: EcologyTransitionTrace
    negative_control: EcologyTransitionTrace
    segment_credit_off_control: EcologyTransitionTrace
    parity: EcologySegmentCreditParity


@dataclass(frozen=True)
class EcologyOwnerDifference:
    body_id: int
    tick: int
    owner_name: str
    field_name: str
    before_fingerprint: str
    after_fingerprint: str


@dataclass(frozen=True)
class EcologyFrozenEvaluationAudit:
    scenario: str
    seed: int
    rounds: int
    gated_owner_names: tuple[str, ...]
    allowed_owner_names: tuple[str, ...]
    policy_stable: bool
    temporal_learning_stable: bool
    unstable_owner_names: tuple[str, ...]
    first_differences: tuple[EcologyOwnerDifference, ...]
    block_reason: str
    replay_settlement_coverage: float
    replay_lineage_coverage: float
    replay_drop_count: int
    passed: bool


@dataclass(frozen=True)
class EcologyMechanismGate:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True)
class EcologyDiagnosticSurface:
    """A published surface, and whether any gate actually reads it.

    Reviewers found the per-episode learned series recorded with
    ``passed=False`` while gating nothing, which reads as a gate but is not
    one.  Every non-gating surface is now declared here explicitly, together
    with the trigger that replaces it.
    """

    name: str
    gated: bool
    reason: str


@dataclass(frozen=True)
class EcologyDeclaredGap:
    """A plan requirement this audit does NOT satisfy, stated in the artifact.

    A gap that lives only in a source comment is invisible to anyone reading
    the evidence, so each one is published with the plan line it belongs to,
    the owner that would have to close it, and whether it blocks the verdict.
    """

    plan_reference: str
    requirement: str
    status: str
    owner: str
    gate_failing: bool


# plan 05 requirements this audit knowingly does not meet.  They are published
# in every artifact rather than living in a code comment.
ECOLOGY_AUDIT_DECLARED_GAPS: tuple[EcologyDeclaredGap, ...] = (
    EcologyDeclaredGap(
        plan_reference="research/ant/05_ecology_p0_p1_p2_plan.md:121",
        requirement=(
            "训练后的 turn sensitivity 不得低于 shared-initial 的 25%，"
            "除非绝对值仍高于预先声明的任务有效阈值。"
        ),
        status=(
            "DEVIATION: the retention floor is taken against the PEAK "
            "ACQUIRED sensitivity of the arm, not against shared-initial, and "
            "the plan's OR-form escape clause is not implemented -- both the "
            "absolute turn threshold AND the retention floor must hold. "
            "shared-initial is unusable as a baseline because a cold "
            "exclusive-steering head is exactly zero by design, so a floor "
            "derived from it evaluates to 0 and the gate becomes vacuous "
            "(_retention_floor raises rather than deriving one). The "
            "implemented rule is strictly stricter than the plan's, but the "
            "plan sentence above must be rewritten before P0 can be signed "
            "off; that edit belongs to the documentation package, not here."
        ),
        owner="research/ant/05_ecology_p0_p1_p2_plan.md",
        gate_failing=False,
    ),
    EcologyDeclaredGap(
        plan_reference="research/ant/05_ecology_p0_p1_p2_plan.md:149",
        requirement=(
            "closure cause：world switch、self switch、milestone、terminal "
            "或 max-step timeout。"
        ),
        status=(
            "GAP: the runtime-replay owner publishes a single 'beta-switch' "
            "close reason for both tracks, so a world-track closure and a "
            "self-track closure are indistinguishable in "
            "close_reason_counts. The per-tick log does carry both tracks' "
            "beta separately, so the split is recoverable by hand but is not "
            "gated. Closing it means changing the close-reason vocabulary in "
            "the vz-temporal runtime-replay owner, which this package does "
            "not own."
        ),
        owner="vz-temporal runtime-replay segment owner",
        gate_failing=False,
    ),
    EcologyDeclaredGap(
        plan_reference="research/ant/05_ecology_p0_p1_p2_plan.md:150",
        requirement="SSL 前后 switch 参数和 histogram。",
        status=(
            "PARTIAL: switch parameters before/after each trace are published "
            "(EcologySwitchParameterSnapshot) and this audit derives its own "
            "ten-bin beta histogram per track from its per-tick log. The "
            "owner-published SwitchGateStats histogram is NOT available: the "
            "ant session's residual trace is shorter than two steps every "
            "turn, so MetacontrollerSSLTrainer.optimize early-returns and the "
            "SSL trainer never trains (trained_steps=0). That same fact is "
            "what fails the temporal_ssl_backend coverage lane."
        ),
        owner="vz-temporal MetacontrollerSSLTrainer / the ant trace length",
        gate_failing=True,
    ),
)


@dataclass(frozen=True)
class EcologyMechanismAuditReport:
    schema_version: str
    config: EcologyMechanismAuditConfig
    action_probe_guard_enabled: bool
    initial_snapshot: EcologyActionChainSnapshot
    final_learned_snapshot: EcologyActionChainSnapshot
    action_chain_snapshots: tuple[EcologyActionChainSnapshot, ...]
    sign_consistency: tuple[EcologySignConsistency, ...]
    lateral_bias: tuple[EcologyLateralBias, ...]
    action_head_updates: tuple[EcologyActionHeadUpdate, ...]
    backend_parity: tuple[EcologyBackendParityLane, ...]
    segment_telemetry: tuple[EcologySegmentTelemetry, ...]
    rollback_episodes: tuple[str, ...]
    temporal_switch: EcologyTemporalSwitchAudit
    frozen_evaluations: tuple[EcologyFrozenEvaluationAudit, ...]
    gates: tuple[EcologyMechanismGate, ...]
    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    description: str
    #: plan 05:130 -- "初始通过、某阶段后失败：在首个失败 episode 内做二分
    #: replay". The per-episode learned series gates nothing (see
    #: ``diagnostic_surfaces``); this label is the trigger for that branch and
    #: is empty when no learned episode failed.
    first_failing_learned_episode: str = ""
    diagnostic_surfaces: tuple[EcologyDiagnosticSurface, ...] = ()
    declared_gaps: tuple[EcologyDeclaredGap, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# The shared-initial checkpoint is gated on input reachability only. Under
# exclusive steering the causal action head owns the actuator contrast axis and
# its cold parameters are exactly zero, so a cold turn_delta of 0 is the
# DESIGN, not a defect (docs/specs/digital-ant-embodiment.md, and
# tests/test_ecology_world.py asserts it by construction).
_GATE_MODE_INPUT_REACHABILITY = "input-reachability"
_GATE_MODE_POST_TRAINING = "post-training"

_STEERING_PROBE_KINDS = (EcologyProbeKind.FOOD, EcologyProbeKind.HEAT)


def _curriculum_config(
    config: EcologyMechanismAuditConfig,
) -> EcologyCurriculumConfig:
    return EcologyCurriculumConfig(
        n_ants=config.n_ants,
        temporal_latent_dim=config.temporal_latent_dim,
        stage_rounds=config.episode_rounds,
        stage_episodes=config.episodes_per_stage,
        mastery_min_episodes=min(3, config.episodes_per_stage),
        validation_rounds=config.evaluation_rounds,
        validation_seeds=(config.seed + 43,),
        heldout_rounds=config.evaluation_rounds,
        heldout_seeds=ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS,
        seed=config.seed,
        # The per-episode rollback guard restores the pre-episode checkpoints
        # whenever a candidate update fails the probe, which makes every
        # downstream "trained" snapshot a copy of the cold checkpoint and the
        # whole gate vacuous. P1/P2 already disable it and enforce the same
        # frozen thresholds once on the final checkpoint; P0 now does the same.
        action_probe_guard_enabled=False,
        action_probe_code_delta_threshold=config.code_delta_threshold,
        action_probe_turn_delta_threshold=config.turn_delta_threshold,
        action_probe_retention_ratio=config.retention_ratio_threshold,
        action_probe_body_pass_ratio=config.body_pass_ratio,
    )


def _probe_by_kind(
    body_report: EcologyCheckpointActionProbe,
    kind: EcologyProbeKind,
) -> EcologyActionProbe:
    return next(probe for probe in body_report.probes if probe.kind is kind)


def _required_body_passes(
    *,
    body_count: int,
    body_pass_ratio: float,
) -> int:
    return max(1, math.ceil(body_count * body_pass_ratio))


def _retention_floor(
    *,
    baseline_turn_delta: float,
    config: EcologyMechanismAuditConfig,
    context: str,
) -> float:
    """Derive a retention floor, refusing to derive one from nothing.

    A floor computed from a zero baseline evaluates to ``0`` and turns the
    retention gate into a tautology.  That is exactly how the v1/v2 audit
    reported a retained sensitivity it had never measured, so this raises.
    """

    if not math.isfinite(baseline_turn_delta):
        raise EcologyMechanismAuditError(
            f"{context}: retention baseline is not finite "
            f"({baseline_turn_delta!r})"
        )
    if baseline_turn_delta < config.turn_delta_threshold:
        raise EcologyMechanismAuditError(
            f"{context}: retention floor requested from baseline turn_delta="
            f"{baseline_turn_delta:.9g}, which is below the frozen turn "
            f"threshold {config.turn_delta_threshold:.9g}. A retention gate "
            "derived from a zero baseline is vacuous; the shared-initial "
            "checkpoint is gated on input reachability only."
        )
    return baseline_turn_delta * config.retention_ratio_threshold


def _turn_sign(turn: float, *, threshold: float) -> int:
    if turn > threshold:
        return 1
    if turn < -threshold:
        return -1
    return 0


def _paired_turn_signs(
    left_turn: float,
    right_turn: float,
    *,
    contrast_threshold: float,
) -> tuple[int, int]:
    """Classify directions after the paired sensitivity floor is met.

    Mirror-equivariant steering splits one opponent-coded contrast across the
    two sides. Requiring each side to exceed the complete paired threshold
    silently doubles the frozen sensitivity gate. The pair first clears that
    unchanged contrast floor; half of it is then the numerical floor used only
    to classify each side's direction.
    """

    if abs(left_turn - right_turn) < contrast_threshold:
        return (0, 0)
    side_threshold = contrast_threshold / 2.0
    return (
        _turn_sign(left_turn, threshold=side_threshold),
        _turn_sign(right_turn, threshold=side_threshold),
    )


def _evaluate_action_snapshot(
    *,
    config: EcologyMechanismAuditConfig,
    arm: str,
    label: str,
    stage: str,
    tier: str,
    episode_index: int,
    gate_mode: str,
    body_reports: tuple[EcologyCheckpointActionProbe, ...],
    retention_baselines: dict[tuple[int, str], float],
) -> EcologyActionChainSnapshot:
    if gate_mode not in {
        _GATE_MODE_INPUT_REACHABILITY,
        _GATE_MODE_POST_TRAINING,
    }:
        raise EcologyMechanismAuditError(
            f"unsupported action-chain gate mode: {gate_mode!r}"
        )
    failures: list[str] = []
    passing_bodies = 0
    for body_report in body_reports:
        body_failures: list[str] = []
        for probe in body_report.probes:
            if not probe.input_reachable:
                body_failures.append(f"{probe.kind.value}:input-unreachable")
        if gate_mode == _GATE_MODE_POST_TRAINING:
            for kind in _STEERING_PROBE_KINDS:
                probe = _probe_by_kind(body_report, kind)
                if probe.turn_delta < config.turn_delta_threshold:
                    body_failures.append(
                        f"{kind.value}:turn-delta={probe.turn_delta:.9g}"
                    )
                    continue
                baseline = retention_baselines.get(
                    (body_report.body_id, kind.value)
                )
                if baseline is None:
                    continue
                floor = _retention_floor(
                    baseline_turn_delta=baseline,
                    config=config,
                    context=(
                        f"{arm}:{label}:body:{body_report.body_id}:"
                        f"{kind.value}"
                    ),
                )
                if probe.turn_delta < floor:
                    body_failures.append(
                        f"{kind.value}:retention={probe.turn_delta:.9g}/"
                        f"{baseline:.9g}"
                    )
        if body_failures:
            failures.extend(
                f"body:{body_report.body_id}:{item}" for item in body_failures
            )
        else:
            passing_bodies += 1
    required = _required_body_passes(
        body_count=len(body_reports),
        body_pass_ratio=config.body_pass_ratio,
    )
    return EcologyActionChainSnapshot(
        arm=arm,
        label=label,
        stage=stage,
        tier=tier,
        episode_index=episode_index,
        gate_mode=gate_mode,
        body_reports=body_reports,
        required_body_passes=required,
        passing_bodies=passing_bodies,
        passed=passing_bodies >= required,
        failures=tuple(failures),
    )


def _update_retention_baselines(
    *,
    baselines: dict[tuple[int, str], float],
    body_reports: tuple[EcologyCheckpointActionProbe, ...],
    config: EcologyMechanismAuditConfig,
) -> None:
    """Track the peak acquired sensitivity per (body, kind).

    Retention is "does an acquired capability survive later training", so the
    baseline is the peak the arm actually reached after training started.  The
    shared-initial checkpoint never enters it: a cold head is zero by design.
    """

    for body_report in body_reports:
        for kind in _STEERING_PROBE_KINDS:
            probe = _probe_by_kind(body_report, kind)
            if probe.turn_delta < config.turn_delta_threshold:
                continue
            key = (body_report.body_id, kind.value)
            baselines[key] = max(baselines.get(key, 0.0), probe.turn_delta)


def _segment_telemetry(
    *,
    arm: str,
    plan: EcologyTrainingEpisodePlan,
    runner: KernelColonyRunner,
    training_report: EcologyTrainingEpisodeReport,
) -> EcologySegmentTelemetry:
    records = tuple(
        step for round_record in runner.rounds for step in round_record.ant_steps
    )
    latest = tuple(
        session.trajectory[-1]
        for session in runner.sessions
        if session.trajectory
    )
    reason_counts: dict[str, int] = {}
    for record in latest:
        for reason, count in record.runtime_segment_close_reason_counts:
            reason_counts[reason] = reason_counts.get(reason, 0) + count
    switch_gates = tuple(record.switch_gate for record in records)
    track_values: dict[str, list[float]] = {}
    for record in records:
        for track_name, value in record.track_switch_gates:
            track_values.setdefault(track_name, []).append(value)
    return EcologySegmentTelemetry(
        arm=arm,
        stage=plan.stage.value,
        tier=plan.tier.value,
        episode_index=plan.episode_index,
        switch_count=sum(int(record.is_switching) for record in records),
        closed_segment_count=sum(
            record.runtime_closed_segments for record in latest
        ),
        longest_segment_length=max(
            (record.runtime_longest_segment_length for record in latest),
            default=0,
        ),
        close_reason_counts=tuple(sorted(reason_counts.items())),
        switch_gate_min=min(switch_gates, default=0.0),
        switch_gate_mean=(
            sum(switch_gates) / len(switch_gates)
            if switch_gates
            else 0.0
        ),
        switch_gate_max=max(switch_gates, default=0.0),
        track_switch_gate_ranges=tuple(
            (
                track_name,
                min(values),
                max(values),
            )
            for track_name, values in sorted(track_values.items())
        ),
        action_chain_guard_passed=(
            training_report.action_chain_guard_passed
        ),
        action_chain_rollback_applied=(
            training_report.action_chain_rollback_applied
        ),
        action_chain_failures=training_report.action_chain_failures,
    )


async def _probe_checkpoints(
    *,
    config: EcologyMechanismAuditConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    seed_offset: int,
    backend_lane: EcologyProbeBackendLane | None = None,
    exercise_steps: int = 0,
) -> tuple[EcologyCheckpointActionProbe, ...]:
    return await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + seed_offset,
        checkpoints=checkpoints,
        code_delta_threshold=config.code_delta_threshold,
        turn_delta_threshold=config.turn_delta_threshold,
        backend_lane=backend_lane,
        exercise_steps=exercise_steps,
    )


# Retention is a within-probe comparison: every checkpoint must see the exact
# same paired worlds and session seeds.
_PRIMARY_PROBE_SEED_OFFSET = 700_003

_AUDIT_TRAINING_STAGES: tuple[EcologyStage, ...] = (
    EcologyStage.BUTTER,
    EcologyStage.BURNING_MATCH,
    EcologyStage.COMPOSITE,
)
_AUDIT_TRAINING_TIERS: tuple[EcologyTrainingTier, ...] = (
    EcologyTrainingTier.NEAR,
    EcologyTrainingTier.MEDIUM,
    EcologyTrainingTier.FAR,
)
# Training lives in a dedicated high namespace. Frozen held-out repros are
# literal seeds 101/307 by contract and validation uses ``config.seed + 43``;
# deriving training directly from ``config.seed`` made the default second
# episode collide with held-out seed 101 after an expensive full audit.
_AUDIT_TRAINING_SEED_NAMESPACE = 1_000_003


def _episode_plan_seed(
    *,
    config: EcologyMechanismAuditConfig,
    stage_index: int,
    episode_index: int,
) -> int:
    return (
        config.seed
        + _AUDIT_TRAINING_SEED_NAMESPACE
        + stage_index * 10_000
        + episode_index * 101
    )


def ecology_mechanism_audit_seed_schedule(
    config: EcologyMechanismAuditConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The ``(training_seeds, layout_seeds)`` this audit will actually use.

    plan 05 s2.1 requires every run to record its training and layout seeds,
    and requires the training / validation / held-out namespaces to stay
    disjoint.  Publishing the schedule from the module that generates it keeps
    the provenance record from drifting away from the run it describes.
    """

    training = tuple(
        _episode_plan_seed(
            config=config,
            stage_index=stage_index,
            episode_index=episode_index,
        )
        for stage_index in range(len(_AUDIT_TRAINING_STAGES))
        for episode_index in range(config.episodes_per_stage)
    )
    layout = (config.seed + 43,) + ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS
    overlap = set(training) & set(layout)
    if overlap:
        raise EcologyMechanismAuditError(
            "training and validation/held-out seed namespaces overlap: "
            f"{sorted(overlap)} (plan 05 s2.1)"
        )
    return training, layout


async def _train_audit_arm(
    *,
    audit_config: EcologyMechanismAuditConfig,
    curriculum_config: EcologyCurriculumConfig,
    initial: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
) -> tuple[
    tuple[AntLearningCheckpoint, ...],
    tuple[EcologyActionChainSnapshot, ...],
    tuple[EcologySegmentTelemetry, ...],
]:
    if curriculum_config.action_probe_guard_enabled:
        raise EcologyMechanismAuditError(
            "the P0 audit must run with the per-episode action-chain rollback "
            "guard disabled; otherwise every gated snapshot is a restored "
            "cold checkpoint"
        )
    checkpoints = initial
    snapshots: list[EcologyActionChainSnapshot] = []
    segment_reports: list[EcologySegmentTelemetry] = []
    retention_baselines: dict[tuple[int, str], float] = {}
    stages = _AUDIT_TRAINING_STAGES
    tiers = _AUDIT_TRAINING_TIERS
    for stage_index, stage in enumerate(stages):
        for episode_index in range(audit_config.episodes_per_stage):
            tier = tiers[min(episode_index, len(tiers) - 1)]
            plan = EcologyTrainingEpisodePlan(
                stage=stage,
                tier=tier,
                seed=_episode_plan_seed(
                    config=audit_config,
                    stage_index=stage_index,
                    episode_index=episode_index,
                ),
                episode_index=episode_index,
                interleaved=False,
                forced_escape=(
                    stage is EcologyStage.BURNING_MATCH
                    and episode_index == 0
                ),
            )
            runner, checkpoints, training_report = (
                await _run_training_episode(
                    config=curriculum_config,
                    checkpoints=checkpoints,
                    arm=arm,
                    optimize=optimize,
                    local_valence_enabled=True,
                    segment_credit_enabled=True,
                    plan=plan,
                    # Guard disabled above: passing no baseline makes that
                    # explicit instead of relying on the flag alone.
                    action_probe_baseline=None,
                    action_probe_baseline_reports=None,
                )
            )
            body_reports = await _probe_checkpoints(
                config=audit_config,
                checkpoints=checkpoints,
                seed_offset=_PRIMARY_PROBE_SEED_OFFSET,
            )
            snapshots.append(
                _evaluate_action_snapshot(
                    config=audit_config,
                    arm=arm,
                    label=(
                        f"{stage.value}:{tier.value}:"
                        f"episode:{episode_index}"
                    ),
                    stage=stage.value,
                    tier=tier.value,
                    episode_index=episode_index,
                    gate_mode=_GATE_MODE_POST_TRAINING,
                    body_reports=body_reports,
                    retention_baselines=retention_baselines,
                )
            )
            _update_retention_baselines(
                baselines=retention_baselines,
                body_reports=body_reports,
                config=audit_config,
            )
            segment_reports.append(
                _segment_telemetry(
                    arm=arm,
                    plan=plan,
                    runner=runner,
                    training_report=training_report,
                )
            )
    return checkpoints, tuple(snapshots), tuple(segment_reports)


# ---------------------------------------------------------------------------
# P0-A extras: sign consistency, lateral bias, head update, backend parity
# ---------------------------------------------------------------------------


async def _sign_consistency(
    *,
    config: EcologyMechanismAuditConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    first_repeat: tuple[EcologyCheckpointActionProbe, ...],
) -> tuple[tuple[EcologySignConsistency, ...], tuple[int, ...]]:
    """Repeat the paired probes at one checkpoint and compare turn signs.

    ``first_repeat`` is the already-measured probe batch at the primary probe
    seed, so the gated snapshot and repeat 0 are provably the same
    measurement; the remaining repeats use distinct probe seeds.
    """

    seeds = tuple(
        _PRIMARY_PROBE_SEED_OFFSET
        + repeat * ECOLOGY_AUDIT_SIGN_REPEAT_SEED_STRIDE
        for repeat in range(config.sign_repeat_count)
    )
    repeats: list[tuple[EcologyCheckpointActionProbe, ...]] = [first_repeat]
    for seed_offset in seeds[1:]:
        repeats.append(
            await _probe_checkpoints(
                config=config,
                checkpoints=checkpoints,
                seed_offset=seed_offset,
            )
        )
    results: list[EcologySignConsistency] = []
    for body_index in range(len(checkpoints)):
        for kind in _STEERING_PROBE_KINDS:
            paired_signs = tuple(
                _paired_turn_signs(
                    _probe_by_kind(repeat[body_index], kind).left_turn,
                    _probe_by_kind(repeat[body_index], kind).right_turn,
                    contrast_threshold=config.turn_delta_threshold,
                )
                for repeat in repeats
            )
            left_signs = tuple(item[0] for item in paired_signs)
            right_signs = tuple(item[1] for item in paired_signs)
            # A sign that does not exist is not a consistent sign: a head that
            # emits zero on every repeat has no direction to be consistent
            # about.
            consistent = (
                len(set(left_signs)) == 1
                and len(set(right_signs)) == 1
                and left_signs[0] != 0
                and right_signs[0] != 0
            )
            results.append(
                EcologySignConsistency(
                    body_id=body_index,
                    kind=kind.value,
                    probe_seeds=tuple(
                        config.seed + offset for offset in seeds
                    ),
                    left_turn_signs=left_signs,
                    right_turn_signs=right_signs,
                    consistent=consistent,
                )
            )
    return tuple(results), tuple(config.seed + offset for offset in seeds)


def _lateral_bias(
    *,
    config: EcologyMechanismAuditConfig,
    body_reports: tuple[EcologyCheckpointActionProbe, ...],
) -> tuple[EcologyLateralBias, ...]:
    """plan 05:124 -- reject a colony-wide same-direction turn.

    ``common_mode`` is the side-independent turn the head emits regardless of
    which antenna is stimulated; ``contrast`` is the part that actually encodes
    the side.  A head whose colony-mean common mode dominates its colony-mean
    contrast is steering by a fixed bias, not by the sensor.
    """

    results: list[EcologyLateralBias] = []
    for kind in _STEERING_PROBE_KINDS:
        probes = tuple(
            _probe_by_kind(body_report, kind) for body_report in body_reports
        )
        if not probes:
            raise EcologyMechanismAuditError(
                "lateral-bias diagnostic requires at least one body report"
            )
        common_modes = tuple(
            (probe.left_turn + probe.right_turn) / 2.0 for probe in probes
        )
        contrasts = tuple(
            (probe.left_turn - probe.right_turn) / 2.0 for probe in probes
        )
        mean_common = sum(common_modes) / len(common_modes)
        mean_contrast = sum(contrasts) / len(contrasts)
        results.append(
            EcologyLateralBias(
                kind=kind.value,
                body_count=len(probes),
                mean_common_mode=mean_common,
                mean_contrast=mean_contrast,
                systematic_same_direction=(
                    abs(mean_common) > config.turn_delta_threshold
                    and abs(mean_common) >= abs(mean_contrast)
                ),
            )
        )
    return tuple(results)


def _action_head_updates(
    *,
    initial: tuple[AntLearningCheckpoint, ...],
    learned: tuple[AntLearningCheckpoint, ...],
    body_reports: tuple[EcologyCheckpointActionProbe, ...],
) -> tuple[EcologyActionHeadUpdate, ...]:
    """plan 05:122 -- the learned action head must have taken a real update.

    Everything here comes from owner-published evidence: the checkpoint policy
    fingerprint and the per-step ``causal_action_head_update_step`` /
    ``causal_action_head_residual`` the metacontroller publishes.  No
    curriculum internals are imported.
    """

    results: list[EcologyActionHeadUpdate] = []
    for body_report in body_reports:
        body_id = body_report.body_id
        update_step = max(
            max(
                probe.left_action_head_update_step,
                probe.right_action_head_update_step,
            )
            for probe in body_report.probes
        )
        residual_finite = all(
            math.isfinite(value)
            for probe in body_report.probes
            for value in (
                probe.left_action_head_residual
                + probe.right_action_head_residual
            )
        )
        fingerprint_changed = (
            initial[body_id].policy_fingerprint
            != learned[body_id].policy_fingerprint
        )
        results.append(
            EcologyActionHeadUpdate(
                body_id=body_id,
                update_step=update_step,
                residual_finite=residual_finite,
                policy_fingerprint_changed=fingerprint_changed,
                passed=(
                    update_step > 0
                    and residual_finite
                    and fingerprint_changed
                ),
            )
        )
    return tuple(results)


def _backend_lane_availability(
    lane: EcologyProbeBackendLane,
) -> tuple[bool, str]:
    if lane is EcologyProbeBackendLane.PURE:
        return True, ""
    if lane is EcologyProbeBackendLane.RUNTIME:
        if importlib.util.find_spec("numpy") is None:
            return False, "numpy is not importable in this process"
        return True, ""
    if lane is EcologyProbeBackendLane.TORCH:
        if importlib.util.find_spec("torch") is None:
            return False, "torch is not importable in this process"
        return True, ""
    raise EcologyMechanismAuditError(f"unknown backend lane {lane!r}")


def _max_sequence_delta(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if len(left) != len(right):
        raise EcologyMechanismAuditError(
            "backend parity comparison requires equal-width sequences: "
            f"{len(left)} vs {len(right)}"
        )
    return max(
        (
            abs(left_value - right_value)
            for left_value, right_value in zip(left, right, strict=True)
        ),
        default=0.0,
    )


def _lane_wiring_mismatch(
    *,
    lane: EcologyProbeBackendLane,
    body_reports: tuple[EcologyCheckpointActionProbe, ...],
) -> str:
    """Reject a lane whose session did not actually run on its declared wiring.

    Without this, ``_lane_rollout_config`` silently failing to reach the
    session would still yield a green parity lane.
    """

    expected = ecology_probe_lane_expected_wiring(lane)
    for body_report in body_reports:
        if body_report.observed_backend_wiring != expected:
            return (
                f"the probe session published backend wiring "
                f"{body_report.observed_backend_wiring} while the "
                f"{lane.value} lane declares {expected}"
            )
    return ""


def _max_distribution_delta(
    left: tuple[tuple[str, float], ...],
    right: tuple[tuple[str, float], ...],
) -> float:
    """Largest per-family probability-mass difference between two lanes.

    A family present on one side only counts at its full mass, so a lane that
    invented or dropped an action family cannot hide behind the families the
    two lanes happen to share.
    """

    left_map = dict(left)
    right_map = dict(right)
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise EcologyMechanismAuditError(
            "action distribution published a duplicated family id: "
            f"{left} vs {right}"
        )
    return max(
        (
            abs(left_map.get(name, 0.0) - right_map.get(name, 0.0))
            for name in left_map.keys() | right_map.keys()
        ),
        default=0.0,
    )


def _lane_coverage_failure(
    *,
    lane: EcologyProbeBackendLane,
    execution: EcologyBackendExecutionEvidence,
) -> str:
    """Why a lane's declared backends cannot be shown to have executed.

    Answered ONLY from owner-published execution evidence.  "This lane's
    numbers differ from the reference" is deliberately not admissible: a
    correct backend that agrees exactly would then be indistinguishable from
    one that never ran, which is the defect this split exists to remove.
    """

    declared = ecology_probe_lane_declared_active_backends(lane)
    applied = {
        "temporal_runtime_backend": (
            execution.temporal_runtime_backend_applied
        ),
        "temporal_ssl_backend": execution.ssl_backend_applied,
        "internal_rl_backend": execution.internal_rl_backend_applied,
    }
    failures: list[str] = []
    for name in declared:
        if applied[name] != "active":
            failures.append(
                f"{name} declared active but the live owner reports "
                f"{applied[name]!r}"
            )
            continue
        if name == "temporal_runtime_backend":
            # The live FullLearnedTemporalPolicy reporting ACTIVE is the
            # owner's own statement that its ndarray forward path is the one
            # running; there is no separate per-turn execution counter.
            continue
        if name == "temporal_ssl_backend":
            if execution.ssl_trained_steps <= 0:
                failures.append(
                    "temporal_ssl_backend is active but the SSL trainer "
                    "reported trained_steps=0 (MetacontrollerSSLTrainer."
                    "optimize early-returns on a trace shorter than two "
                    "steps), so the torch SSL path never ran"
                )
            elif "active" not in execution.ssl_torch_backends:
                failures.append(
                    "temporal_ssl_backend is active but the SSL trainer "
                    f"reported torch backends {execution.ssl_torch_backends}"
                )
            continue
        if name == "internal_rl_backend":
            if not execution.internal_rl_report_published:
                failures.append(
                    "internal_rl_backend is active but the joint loop never "
                    "published an optimization report, so no full cycle ran "
                    f"in {execution.exercise_steps} exercise ticks"
                )
            elif "active" not in execution.internal_rl_torch_backends:
                failures.append(
                    "internal_rl_backend is active but the sandbox reported "
                    f"torch backends {execution.internal_rl_torch_backends}"
                )
            elif not execution.internal_rl_torch_wrote_back:
                failures.append(
                    "internal_rl_backend ran but never wrote back, so the "
                    "measured forward path is still the pure one"
                )
            continue
        raise EcologyMechanismAuditError(
            f"unclassified declared backend key {name!r} for lane {lane.value}"
        )
    if lane is EcologyProbeBackendLane.PURE:
        engaged = tuple(
            name for name, level in applied.items() if level == "active"
        )
        if engaged:
            failures.append(
                "the pure reference lane must engage no accelerated backend, "
                f"but the live owners report {engaged} active"
            )
    return "; ".join(failures)


async def _backend_parity(
    *,
    config: EcologyMechanismAuditConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[EcologyBackendParityLane, ...]:
    """Measure coverage and parity for every declared backend lane.

    Every lane runs the SAME optimization-carrying probe schedule: without it
    ``temporal_ssl_backend`` / ``internal_rl_backend`` never enter an
    optimization cycle and each lane is bit-identical to the reference for a
    reason that says nothing about the backends.
    """

    reference = await _probe_checkpoints(
        config=config,
        checkpoints=checkpoints,
        seed_offset=_PRIMARY_PROBE_SEED_OFFSET,
        backend_lane=EcologyProbeBackendLane.PURE,
        exercise_steps=config.backend_parity_exercise_steps,
    )
    lanes: list[EcologyBackendParityLane] = []
    for lane in EcologyProbeBackendLane:
        declared = ecology_probe_lane_declared_active_backends(lane)
        available, reason = _backend_lane_availability(lane)
        if not available:
            # plan 05:123 -- an unavailable backend is published explicitly and
            # FAILS coverage. Silently omitting it would let a parity claim
            # rest on a lane that never ran.
            lanes.append(
                EcologyBackendParityLane(
                    lane=lane.value,
                    measured=False,
                    covered=False,
                    not_measured_reason=reason,
                    not_covered_reason=reason,
                    declared_active_backends=declared,
                    max_code_delta=None,
                    max_turn_delta=None,
                    max_step_delta=None,
                    max_action_head_residual_delta=None,
                    max_action_distribution_delta=None,
                    abstract_actions_agree=False,
                    within_tolerance=False,
                )
            )
            continue
        candidate = (
            reference
            if lane is EcologyProbeBackendLane.PURE
            else await _probe_checkpoints(
                config=config,
                checkpoints=checkpoints,
                seed_offset=_PRIMARY_PROBE_SEED_OFFSET,
                backend_lane=lane,
                exercise_steps=config.backend_parity_exercise_steps,
            )
        )
        code_delta = 0.0
        turn_delta = 0.0
        step_delta = 0.0
        residual_delta = 0.0
        distribution_delta = 0.0
        actions_agree = True
        for reference_body, candidate_body in zip(
            reference,
            candidate,
            strict=True,
        ):
            for reference_probe, candidate_probe in zip(
                reference_body.probes,
                candidate_body.probes,
                strict=True,
            ):
                code_delta = max(
                    code_delta,
                    _max_sequence_delta(
                        reference_probe.left_code,
                        candidate_probe.left_code,
                    ),
                    _max_sequence_delta(
                        reference_probe.right_code,
                        candidate_probe.right_code,
                    ),
                )
                turn_delta = max(
                    turn_delta,
                    abs(reference_probe.left_turn - candidate_probe.left_turn),
                    abs(
                        reference_probe.right_turn
                        - candidate_probe.right_turn
                    ),
                )
                step_delta = max(
                    step_delta,
                    abs(reference_probe.left_step - candidate_probe.left_step),
                    abs(
                        reference_probe.right_step
                        - candidate_probe.right_step
                    ),
                )
                residual_delta = max(
                    residual_delta,
                    _max_sequence_delta(
                        reference_probe.left_action_head_residual,
                        candidate_probe.left_action_head_residual,
                    ),
                    _max_sequence_delta(
                        reference_probe.right_action_head_residual,
                        candidate_probe.right_action_head_residual,
                    ),
                )
                distribution_delta = max(
                    distribution_delta,
                    _max_distribution_delta(
                        reference_probe.left_action_distribution,
                        candidate_probe.left_action_distribution,
                    ),
                    _max_distribution_delta(
                        reference_probe.right_action_distribution,
                        candidate_probe.right_action_distribution,
                    ),
                )
                actions_agree = (
                    actions_agree
                    and reference_probe.left_abstract_action
                    == candidate_probe.left_abstract_action
                    and reference_probe.right_abstract_action
                    == candidate_probe.right_abstract_action
                )
        observed_wiring = candidate[0].observed_backend_wiring
        worst = max(
            code_delta,
            turn_delta,
            step_delta,
            residual_delta,
            distribution_delta,
        )
        not_measured_reason = _lane_wiring_mismatch(
            lane=lane,
            body_reports=candidate,
        )
        execution = candidate[0].backend_execution
        if execution is None:
            raise EcologyMechanismAuditError(
                f"the {lane.value} lane published no backend execution "
                "evidence; coverage cannot be established"
            )
        not_covered_reason = not_measured_reason or _lane_coverage_failure(
            lane=lane,
            execution=execution,
        )
        # A lane that ran on wiring other than the one it declares did not
        # measure the lane it claims to, so its numbers are published as "no
        # measurement" rather than as a parity result. ``None`` keeps the
        # invariant "a delta is None exactly when the lane was not measured",
        # and keeps a non-standard NaN token out of the canonical artifact.
        measured = not not_measured_reason
        lanes.append(
            EcologyBackendParityLane(
                lane=lane.value,
                measured=measured,
                covered=not not_covered_reason,
                not_measured_reason=not_measured_reason,
                not_covered_reason=not_covered_reason,
                declared_active_backends=declared,
                max_code_delta=code_delta if measured else None,
                max_turn_delta=turn_delta if measured else None,
                max_step_delta=step_delta if measured else None,
                max_action_head_residual_delta=(
                    residual_delta if measured else None
                ),
                max_action_distribution_delta=(
                    distribution_delta if measured else None
                ),
                abstract_actions_agree=measured and actions_agree,
                within_tolerance=(
                    measured
                    and actions_agree
                    and worst <= config.backend_parity_tolerance
                ),
                observed_backend_wiring=observed_wiring,
                distinguishable_from_reference=measured and worst > 0.0,
                backend_execution=execution,
            )
        )
    return tuple(lanes)


# ---------------------------------------------------------------------------
# P0-B: deterministic, action-label-free transition protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScriptedPose:
    phase: EcologyTransitionPhase
    x: float
    y: float
    heading: float
    carrying_food: bool | None


def _transition_protocol() -> tuple[_ScriptedPose, ...]:
    """Script the STATE trajectory; never the action.

    The controller still chooses every turn/step command from its own sense;
    the protocol only teleports the body so the declared state changes happen
    on declared ticks regardless of what the policy does.  ``carrying_food`` is
    forced to ``False`` only while the body is scripted to be outbound and is
    left to the world (``None``) everywhere else, so pickup and delivery are
    real world events rather than scripted labels.
    """

    butter_x, butter_y = ECOLOGY_AUDIT_PROTOCOL_BUTTER_XY
    match_x, match_y = ECOLOGY_AUDIT_PROTOCOL_MATCH_XY
    cruise_x, cruise_y = ECOLOGY_AUDIT_PROTOCOL_CRUISE_XY
    poses: list[_ScriptedPose] = []
    for phase, ticks in ECOLOGY_AUDIT_PROTOCOL_PHASE_TICKS:
        for index in range(ticks):
            fraction = (index + 1) / ticks
            if phase is EcologyTransitionPhase.CRUISE:
                poses.append(
                    _ScriptedPose(phase, cruise_x, cruise_y, 0.0, False)
                )
            elif phase is EcologyTransitionPhase.FOOD_APPROACH:
                poses.append(
                    _ScriptedPose(
                        phase,
                        cruise_x + (butter_x - cruise_x) * fraction,
                        cruise_y * (1.0 - fraction),
                        0.0,
                        False,
                    )
                )
            elif phase is EcologyTransitionPhase.PICKUP_CARRYING:
                poses.append(
                    _ScriptedPose(phase, butter_x, butter_y, math.pi, None)
                )
            elif phase is EcologyTransitionPhase.HOME_APPROACH_DELIVERY:
                poses.append(
                    _ScriptedPose(
                        phase,
                        butter_x * (1.0 - fraction),
                        0.0,
                        math.pi,
                        None,
                    )
                )
            elif phase is EcologyTransitionPhase.SAFE_TO_HARMFUL:
                poses.append(
                    _ScriptedPose(
                        phase,
                        match_x,
                        match_y - 0.2 * index,
                        math.pi / 2.0,
                        None,
                    )
                )
            elif phase is EcologyTransitionPhase.HARMFUL_TO_COOLING:
                poses.append(
                    _ScriptedPose(
                        phase,
                        match_x,
                        match_y - 1.0 - 0.9 * index,
                        -math.pi / 2.0,
                        None,
                    )
                )
            else:
                raise EcologyMechanismAuditError(
                    f"unhandled transition phase {phase!r}"
                )
    return tuple(poses)


def _steady_state_protocol() -> tuple[_ScriptedPose, ...]:
    cruise_x, cruise_y = ECOLOGY_AUDIT_PROTOCOL_CRUISE_XY
    return tuple(
        _ScriptedPose(
            EcologyTransitionPhase.CRUISE,
            cruise_x,
            cruise_y,
            0.0,
            False,
        )
        for _ in range(ECOLOGY_AUDIT_PROTOCOL_STEADY_TICKS)
    )


def _protocol_world(*, seed: int) -> AntWorld:
    butter_x, butter_y = ECOLOGY_AUDIT_PROTOCOL_BUTTER_XY
    match_x, match_y = ECOLOGY_AUDIT_PROTOCOL_MATCH_XY
    return AntWorld(
        config=AntWorldConfig(
            seed=seed,
            step_size=0.4,
            antenna_offset_deg=45.0,
            antenna_reach=0.9,
        ),
        world_objects=(
            ButterSource(object_id="p0b-butter", x=butter_x, y=butter_y),
            BurningMatch(object_id="p0b-match", x=match_x, y=match_y),
        ),
    )


def _protocol_session(
    *,
    world: AntWorld,
    config: EcologyMechanismAuditConfig,
    label: str,
    segment_credit_enabled: bool,
) -> AntSession:
    return AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=config.temporal_latent_dim,
            session_id=f"ecology:p0b:{label}",
            seed=config.seed,
            heading_noise=0.0,
            step_noise=0.0,
            rollout_config=ant_runtime_replay_rollout_config(
                enable_sparse_exploration=False,
                enable_segment_credit=segment_credit_enabled,
                sense_schema=AntSenseSchema.ECOLOGY_V2,
            ),
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
            # The temporal audit measures the live switch dynamics of a fixed
            # controller. Optimization is off so segment-credit on/off cannot
            # change the forward equations through a policy update.
            joint_apply_policy_optimization=False,
            joint_learning_enabled=False,
        ),
    )


def _boundary_ticks() -> tuple[tuple[EcologyTransitionPhase, int], ...]:
    boundaries: list[tuple[EcologyTransitionPhase, int]] = []
    cursor = 0
    for phase, ticks in ECOLOGY_AUDIT_PROTOCOL_PHASE_TICKS:
        if cursor:
            boundaries.append((phase, cursor))
        cursor += ticks
    return tuple(boundaries)


_BETA_HISTOGRAM_BINS = 10


def _beta_histogram(values: tuple[float, ...]) -> tuple[int, ...]:
    """Ten equal bins over [0, 1]; out-of-range betas clamp into the ends."""

    bins = [0] * _BETA_HISTOGRAM_BINS
    for value in values:
        index = int(value * _BETA_HISTOGRAM_BINS)
        bins[min(max(index, 0), _BETA_HISTOGRAM_BINS - 1)] += 1
    return tuple(bins)


def _switch_parameter_snapshot(
    *,
    session: AntSession,
    label: str,
) -> EcologySwitchParameterSnapshot:
    world_state = session.runner.world_temporal_policy.export_runtime_state()
    self_state = session.runner.self_temporal_policy.export_runtime_state()
    if world_state is None or self_state is None:
        raise EcologyMechanismAuditError(
            "the temporal transition protocol requires both tracks to "
            "publish MetacontrollerRuntimeState; "
            f"world={world_state is not None}, self={self_state is not None}"
        )
    return EcologySwitchParameterSnapshot(
        label=label,
        world_beta_threshold=world_state.beta_threshold,
        self_beta_threshold=self_state.beta_threshold,
        world_switch_sparsity=world_state.switch_sparsity,
        self_switch_sparsity=self_state.switch_sparsity,
        world_binary_switch_rate=world_state.binary_switch_rate,
        self_binary_switch_rate=self_state.binary_switch_rate,
        world_mean_persistence_window=world_state.mean_persistence_window,
        self_mean_persistence_window=self_state.mean_persistence_window,
        world_ssl_loss=world_state.latest_ssl_loss,
        world_ssl_kl_loss=world_state.latest_ssl_kl_loss,
        self_ssl_loss=self_state.latest_ssl_loss,
        self_ssl_kl_loss=self_state.latest_ssl_kl_loss,
    )


async def _run_transition_trace(
    *,
    config: EcologyMechanismAuditConfig,
    checkpoint: AntLearningCheckpoint,
    poses: tuple[_ScriptedPose, ...],
    label: str,
    segment_credit_enabled: bool,
    include_boundaries: bool,
) -> tuple[EcologyTransitionTrace, tuple[AntStepRecord, ...]]:
    world = _protocol_world(seed=config.seed + 977)
    session = _protocol_session(
        world=world,
        config=config,
        label=label,
        segment_credit_enabled=segment_credit_enabled,
    )
    session.restore_learning_checkpoint(checkpoint)
    parameters_before = _switch_parameter_snapshot(
        session=session,
        label="before",
    )
    ticks: list[EcologyTemporalTick] = []
    previous_closed = 0
    for index, pose in enumerate(poses):
        world.set_body_pose(
            x=pose.x,
            y=pose.y,
            heading=pose.heading,
            carrying_food=pose.carrying_food,
        )
        record = await session.step()
        world_state = session.runner.world_temporal_policy.export_runtime_state()
        self_state = session.runner.self_temporal_policy.export_runtime_state()
        if world_state is None or self_state is None:
            raise EcologyMechanismAuditError(
                "the temporal transition protocol requires both tracks to "
                "publish MetacontrollerRuntimeState; "
                f"world={world_state is not None}, "
                f"self={self_state is not None}"
            )
        ticks.append(
            EcologyTemporalTick(
                tick=index,
                phase=pose.phase.value,
                world_beta=world_state.latest_switch_gate,
                self_beta=self_state.latest_switch_gate,
                world_beta_threshold=world_state.beta_threshold,
                self_beta_threshold=self_state.beta_threshold,
                world_beta_binary=world_state.beta_binary,
                self_beta_binary=self_state.beta_binary,
                track_switch_gates=record.track_switch_gates,
                binary_switch=record.is_switching,
                fast_prior_switch_pressure=(
                    record.fast_prior_switch_pressure_delta
                ),
                prediction_error_switch_pressure=(
                    record.prediction_error_switch_pressure_delta
                ),
                external_switch_pressure=(
                    record.fast_prior_switch_pressure_delta
                    + record.prediction_error_switch_pressure_delta
                ),
                steps_since_switch=record.steps_since_switch,
                open_segment_transitions=(
                    record.runtime_open_segment_transitions
                ),
                closed_segments=record.runtime_closed_segments,
                segment_closed_this_tick=(
                    record.runtime_closed_segments > previous_closed
                ),
                last_segment_close_reason=(
                    record.runtime_last_segment_close_reason
                ),
                carrying_food=record.carrying_food,
                heat_harmful=record.heat_harmful,
                picked_up=record.picked_up,
                delivered=record.delivered,
            )
        )
        previous_closed = record.runtime_closed_segments
    records = tuple(session.trajectory)
    if not records:
        raise EcologyMechanismAuditError(
            f"transition protocol {label} produced no step records"
        )
    final = records[-1]
    reason_counts = dict(final.runtime_segment_close_reason_counts)
    closed = sum(reason_counts.values())
    timeout_ratio = (
        reason_counts.get("bounded-horizon", 0) / closed if closed else 1.0
    )
    switch_ticks = tuple(item.tick for item in ticks if item.binary_switch)
    localizations: list[EcologyBoundaryLocalization] = []
    if include_boundaries:
        for phase, boundary_tick in _boundary_ticks():
            if switch_ticks:
                nearest = min(
                    switch_ticks,
                    key=lambda value: abs(value - boundary_tick),
                )
                distance = abs(nearest - boundary_tick)
            else:
                nearest = -1
                distance = len(poses)
            localizations.append(
                EcologyBoundaryLocalization(
                    phase=phase.value,
                    boundary_tick=boundary_tick,
                    nearest_switch_tick=nearest,
                    distance=distance,
                    localized=distance <= config.switch_localization_window,
                )
            )
    trace = EcologyTransitionTrace(
        label=label,
        segment_credit_enabled=segment_credit_enabled,
        checkpoint_id=checkpoint.checkpoint_id,
        ticks=tuple(ticks),
        boundary_localizations=tuple(localizations),
        switch_ticks=switch_ticks,
        switch_rate=len(switch_ticks) / len(ticks),
        close_reason_counts=tuple(sorted(reason_counts.items())),
        closed_segment_count=closed,
        timeout_closure_ratio=timeout_ratio,
        pickups=world.food_pickups,
        deliveries=world.food_delivered,
        harmful_ticks=sum(int(item.heat_harmful) for item in ticks),
        world_beta_histogram=_beta_histogram(
            tuple(item.world_beta for item in ticks)
        ),
        self_beta_histogram=_beta_histogram(
            tuple(item.self_beta for item in ticks)
        ),
        switch_parameters_before=parameters_before,
        switch_parameters_after=_switch_parameter_snapshot(
            session=session,
            label="after",
        ),
    )
    return trace, records


def _segment_credit_parity(
    *,
    config: EcologyMechanismAuditConfig,
    credit_on: tuple[AntStepRecord, ...],
    credit_off: tuple[AntStepRecord, ...],
) -> EcologySegmentCreditParity:
    """plan 05:165 -- only credit aggregation may differ.

    Sense, the pre-credit motor command and the published rollout lineage must
    align tick by tick; the segment-credit wiring entry itself is the single
    admissible lineage difference.
    """

    if len(credit_on) != len(credit_off):
        raise EcologyMechanismAuditError(
            "segment-credit parity requires equal-length traces: "
            f"{len(credit_on)} vs {len(credit_off)}"
        )
    max_sense = 0.0
    max_turn = 0.0
    max_step = 0.0
    first_misaligned = -1
    lineage_differences: list[tuple[str, str, str]] = []
    for index, (on_record, off_record) in enumerate(
        zip(credit_on, credit_off, strict=True)
    ):
        on_channels = tuple(name for name, _ in on_record.sense_activation)
        off_channels = tuple(name for name, _ in off_record.sense_activation)
        if on_channels != off_channels:
            raise EcologyMechanismAuditError(
                "segment-credit parity found different sense schemas: "
                f"{on_channels} vs {off_channels}"
            )
        sense_delta = _max_sequence_delta(
            tuple(value for _, value in on_record.sense_activation),
            tuple(value for _, value in off_record.sense_activation),
        )
        turn_delta = abs(
            on_record.command.turn_command - off_record.command.turn_command
        )
        step_delta = abs(
            on_record.command.step_command - off_record.command.step_command
        )
        max_sense = max(max_sense, sense_delta)
        max_turn = max(max_turn, turn_delta)
        max_step = max(max_step, step_delta)
        if (
            first_misaligned < 0
            and max(sense_delta, turn_delta, step_delta)
            > config.segment_credit_parity_tolerance
        ):
            first_misaligned = index
    on_wiring = dict(credit_on[-1].backend_wiring)
    off_wiring = dict(credit_off[-1].backend_wiring)
    for name in sorted(on_wiring.keys() | off_wiring.keys()):
        if name == "internal_rl_runtime_segment_credit":
            continue
        if on_wiring.get(name) != off_wiring.get(name):
            lineage_differences.append(
                (name, on_wiring.get(name, ""), off_wiring.get(name, ""))
            )
    lineage_aligned = not lineage_differences
    return EcologySegmentCreditParity(
        tolerance=config.segment_credit_parity_tolerance,
        max_sense_delta=max_sense,
        max_turn_delta=max_turn,
        max_step_delta=max_step,
        lineage_aligned=lineage_aligned,
        lineage_differences=tuple(lineage_differences),
        first_misaligned_tick=first_misaligned,
        passed=(
            lineage_aligned
            and max(max_sense, max_turn, max_step)
            <= config.segment_credit_parity_tolerance
        ),
    )


async def _temporal_switch_audit(
    *,
    config: EcologyMechanismAuditConfig,
    checkpoint: AntLearningCheckpoint,
) -> EcologyTemporalSwitchAudit:
    protocol = _transition_protocol()
    positive, credit_on_records = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=protocol,
        label="positive-control",
        segment_credit_enabled=True,
        include_boundaries=True,
    )
    negative, _ = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=_steady_state_protocol(),
        label="negative-control",
        segment_credit_enabled=True,
        include_boundaries=False,
    )
    credit_off, credit_off_records = await _run_transition_trace(
        config=config,
        checkpoint=checkpoint,
        poses=protocol,
        label="segment-credit-off",
        segment_credit_enabled=False,
        include_boundaries=True,
    )
    return EcologyTemporalSwitchAudit(
        positive_control=positive,
        negative_control=negative,
        segment_credit_off_control=credit_off,
        parity=_segment_credit_parity(
            config=config,
            credit_on=credit_on_records,
            credit_off=credit_off_records,
        ),
    )


# ---------------------------------------------------------------------------
# P0-C: owner-by-owner frozen evaluation
# ---------------------------------------------------------------------------


def _owner_map(
    checkpoint: AntLearningCheckpoint,
) -> dict[str, str]:
    return dict(checkpoint.learning_owner_fingerprints)


def _classify_owner_names(names: set[str]) -> None:
    gated = {name for name, _ in ECOLOGY_AUDIT_GATED_LEARNING_OWNERS}
    allowed = {name for name, _ in ECOLOGY_AUDIT_ALLOWED_TO_CHANGE_OWNERS}
    overlap = gated & allowed
    if overlap:
        raise EcologyMechanismAuditError(
            f"owners declared both gated and allowed-to-change: {overlap}"
        )
    unknown = names - gated - allowed
    if unknown:
        raise EcologyMechanismAuditError(
            "the kernel published learning owners the P0 audit has not "
            f"classified: {sorted(unknown)}. Classify each one as gated or "
            "allowed-to-change with a written justification before running "
            "the audit; silently ignoring an owner is how a vacuous PASS is "
            "produced."
        )
    missing = gated - names
    if missing:
        raise EcologyMechanismAuditError(
            "the P0 audit gates owners the kernel no longer publishes: "
            f"{sorted(missing)}"
        )


async def _frozen_evaluation_audit(
    *,
    audit_config: EcologyMechanismAuditConfig,
    curriculum_config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    scenario: EcologyEvaluationScenario,
    seed: int,
) -> EcologyFrozenEvaluationAudit:
    stage = {
        EcologyEvaluationScenario.BUTTER_ONLY: EcologyStage.BUTTER,
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE: (
            EcologyStage.BURNING_MATCH
        ),
    }[scenario]
    runner = KernelColonyRunner(
        _world(
            config=curriculum_config,
            stage=stage,
            seed=seed,
            data_split=EcologyDataSplit.HELDOUT,
            tier=EcologyTrainingTier.FAR,
            forced_escape=(
                scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
            ),
        ),
        base_config=_session_config(
            config=curriculum_config,
            seed=seed,
            session_id=f"ecology:p0:frozen:{scenario.value}:{seed}",
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=False,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    previous = runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:p0:frozen:before",
        include_runtime_replay=False,
    )
    allowed = {name for name, _ in ECOLOGY_AUDIT_ALLOWED_TO_CHANGE_OWNERS}
    first_differences: dict[tuple[int, str], EcologyOwnerDifference] = {}
    unstable: set[str] = set()
    for tick in range(audit_config.evaluation_rounds):
        await runner.step_round()
        current = runner.export_learning_checkpoints(
            checkpoint_prefix=f"ecology:p0:frozen:tick:{tick}",
            include_runtime_replay=False,
        )
        for body_id, (before, after) in enumerate(
            zip(previous, current, strict=True)
        ):
            before_map = _owner_map(before)
            after_map = _owner_map(after)
            names = before_map.keys() | after_map.keys()
            _classify_owner_names(names)
            for owner_name in sorted(names):
                if owner_name in allowed:
                    continue
                if before_map.get(owner_name) != after_map.get(owner_name):
                    unstable.add(owner_name)
                    first_differences.setdefault(
                        (body_id, owner_name),
                        EcologyOwnerDifference(
                            body_id=body_id,
                            tick=tick,
                            owner_name=owner_name,
                            field_name=(
                                "learning_owner_fingerprints["
                                f"{owner_name}]"
                            ),
                            before_fingerprint=before_map.get(owner_name, ""),
                            after_fingerprint=after_map.get(owner_name, ""),
                        ),
                    )
        previous = current
    final = previous
    policy_stable = all(
        before.policy_fingerprint == after.policy_fingerprint
        for before, after in zip(checkpoints, final, strict=True)
    )
    temporal_stable = all(
        before.temporal_learning_fingerprint
        == after.temporal_learning_fingerprint
        for before, after in zip(checkpoints, final, strict=True)
    )
    latest = tuple(
        session.trajectory[-1]
        for session in runner.sessions
        if session.trajectory
    )
    captured = sum(item.runtime_replay_captured for item in latest)
    settled = sum(item.runtime_replay_settled for item in latest)
    lineage = sum(item.runtime_replay_lineage_matches for item in latest)
    pending = sum(item.runtime_replay_pending_captures for item in latest)
    eligible = captured - pending
    drops = sum(len(item.runtime_replay_drop_reasons) for item in latest)
    settlement_coverage = settled / eligible if eligible else 0.0
    lineage_coverage = lineage / settled if settled else 0.0
    ordered_differences = tuple(
        sorted(
            first_differences.values(),
            key=lambda item: (item.tick, item.owner_name, item.body_id),
        )
    )
    if ordered_differences:
        head = ordered_differences[0]
        block_reason = (
            f"learned owner changed under learning_enabled=False: "
            f"owner={head.owner_name} field={head.field_name} "
            f"tick={head.tick} body={head.body_id} "
            f"before={head.before_fingerprint[:16]} "
            f"after={head.after_fingerprint[:16]}"
        )
    else:
        block_reason = ""
    passed = (
        not ordered_differences
        and policy_stable
        and temporal_stable
        and settlement_coverage >= ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR
        and lineage_coverage >= ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR
        and drops == 0
    )
    return EcologyFrozenEvaluationAudit(
        scenario=scenario.value,
        seed=seed,
        rounds=audit_config.evaluation_rounds,
        gated_owner_names=tuple(
            name for name, _ in ECOLOGY_AUDIT_GATED_LEARNING_OWNERS
        ),
        allowed_owner_names=tuple(sorted(allowed)),
        policy_stable=policy_stable,
        temporal_learning_stable=temporal_stable,
        unstable_owner_names=tuple(sorted(unstable)),
        first_differences=ordered_differences,
        block_reason=block_reason,
        replay_settlement_coverage=settlement_coverage,
        replay_lineage_coverage=lineage_coverage,
        replay_drop_count=drops,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Gate assembly
# ---------------------------------------------------------------------------


def build_ecology_mechanism_gates(
    *,
    config: EcologyMechanismAuditConfig,
    initial_snapshot: EcologyActionChainSnapshot,
    final_learned_snapshot: EcologyActionChainSnapshot,
    sign_consistency: tuple[EcologySignConsistency, ...],
    lateral_bias: tuple[EcologyLateralBias, ...],
    action_head_updates: tuple[EcologyActionHeadUpdate, ...],
    backend_parity: tuple[EcologyBackendParityLane, ...],
    rollback_episodes: tuple[str, ...],
    action_probe_guard_enabled: bool,
    gated_checkpoint_is_post_training: bool,
    no_optimize_stable: bool,
    temporal_switch: EcologyTemporalSwitchAudit,
    frozen_evaluations: tuple[EcologyFrozenEvaluationAudit, ...],
) -> tuple[EcologyMechanismGate, ...]:
    """Assemble every P0 gate from already-collected evidence.

    Kept separate from the run so a constructed BLOCK input can be gated in a
    unit test without paying for a training schedule.
    """

    required_head_updates = _required_body_passes(
        body_count=len(action_head_updates),
        body_pass_ratio=config.body_pass_ratio,
    )
    passing_head_updates = sum(item.passed for item in action_head_updates)
    positive = temporal_switch.positive_control
    negative = temporal_switch.negative_control
    positive_reasons = dict(positive.close_reason_counts)
    localized = tuple(
        item for item in positive.boundary_localizations if item.localized
    )
    return (
        EcologyMechanismGate(
            name="action_chain_input_reachability",
            passed=initial_snapshot.passed,
            observed=(
                f"passing_bodies={initial_snapshot.passing_bodies}/"
                f"{len(initial_snapshot.body_reports)} "
                f"failures={initial_snapshot.failures}"
            ),
            threshold=(
                "shared-initial food/heat/obstacle/home paired swaps reach "
                "code on the required body ratio; turn is NOT gated here "
                "because a cold exclusive-steering head is zero by design"
            ),
        ),
        EcologyMechanismGate(
            name="action_chain_final_sensitivity",
            passed=final_learned_snapshot.passed,
            observed=(
                f"passing_bodies={final_learned_snapshot.passing_bodies}/"
                f"{len(final_learned_snapshot.body_reports)} "
                f"failures={final_learned_snapshot.failures}"
            ),
            threshold=(
                "final trained checkpoint: food/heat turn_delta >= "
                f"{config.turn_delta_threshold:g} and >= "
                f"{config.retention_ratio_threshold:g} of the peak acquired "
                "sensitivity, on the required body ratio"
            ),
        ),
        EcologyMechanismGate(
            name="action_chain_sign_consistency",
            passed=bool(sign_consistency)
            and all(item.consistent for item in sign_consistency),
            observed=str(
                tuple(
                    (
                        item.body_id,
                        item.kind,
                        item.left_turn_signs,
                        item.right_turn_signs,
                    )
                    for item in sign_consistency
                )
            ),
            threshold=(
                f"{config.sign_repeat_count} repeated probes at the final "
                "checkpoint agree on a non-zero left and right turn sign"
            ),
        ),
        EcologyMechanismGate(
            name="action_chain_lateral_bias",
            passed=not any(
                item.systematic_same_direction for item in lateral_bias
            ),
            observed=str(
                tuple(
                    (
                        item.kind,
                        item.mean_common_mode,
                        item.mean_contrast,
                        item.systematic_same_direction,
                    )
                    for item in lateral_bias
                )
            ),
            threshold=(
                "colony-mean side-independent turn does not dominate the "
                "colony-mean left/right contrast"
            ),
        ),
        EcologyMechanismGate(
            name="action_head_update_applied",
            passed=passing_head_updates >= required_head_updates,
            observed=str(
                tuple(
                    (
                        item.body_id,
                        item.update_step,
                        item.residual_finite,
                        item.policy_fingerprint_changed,
                    )
                    for item in action_head_updates
                )
            ),
            threshold=(
                "learned bodies published a finite, non-NaN action-head "
                "residual, a positive update step and a changed policy "
                f"fingerprint on >= {required_head_updates} bodies"
            ),
        ),
        EcologyMechanismGate(
            name="action_chain_no_rollback",
            passed=(
                not rollback_episodes
                and not action_probe_guard_enabled
                and gated_checkpoint_is_post_training
            ),
            observed=str(
                {
                    "rollback_episodes": rollback_episodes,
                    "action_probe_guard_enabled": action_probe_guard_enabled,
                    "gated_checkpoint_is_post_training": (
                        gated_checkpoint_is_post_training
                    ),
                }
            ),
            threshold=(
                "the gated checkpoint is the TRAINED one. The guard-disabled "
                "precondition is really enforced by a raise in "
                "_train_audit_arm (the curriculum only rolls back when both "
                "the flag and a baseline are supplied, and the audit supplies "
                "neither), so the first two clauses are a belt-and-braces "
                "publication of that fact and cannot fail on their own. The "
                "clause that CAN fail is the third: the final learned "
                "checkpoint's policy fingerprint must differ from the "
                "shared-initial one on every body, which is exactly the state "
                "a silent rollback -- or an optimizer that never moved -- "
                "would produce"
            ),
        ),
        EcologyMechanismGate(
            name="backend_lane_coverage",
            passed=bool(backend_parity)
            and all(item.covered for item in backend_parity),
            observed=str(
                tuple(
                    (
                        item.lane,
                        item.covered,
                        item.declared_active_backends,
                        item.not_covered_reason,
                    )
                    for item in backend_parity
                )
            ),
            threshold=(
                "every declared backend of every lane is shown to have "
                "EXECUTED by its own owner's evidence readout (applied wiring "
                "on the live policy, SSL trained steps / torch backend, "
                "Internal-RL optimization report and write-back). Numeric "
                "difference from the reference is deliberately NOT admissible "
                "evidence of coverage"
            ),
        ),
        EcologyMechanismGate(
            name="backend_parity",
            passed=bool(backend_parity)
            and all(
                item.measured and item.within_tolerance
                for item in backend_parity
            ),
            observed=str(
                tuple(
                    (
                        item.lane,
                        item.measured,
                        item.not_measured_reason,
                        item.max_code_delta,
                        item.max_turn_delta,
                        item.max_step_delta,
                        item.max_action_distribution_delta,
                        item.abstract_actions_agree,
                    )
                    for item in backend_parity
                )
            ),
            threshold=(
                "pure/runtime/torch each ran on its declared wiring and all "
                "agree on final code, turn, step, action-head residual and "
                "the owner-published action distribution within "
                f"{config.backend_parity_tolerance:g}, on the same abstract "
                "action. EXACT agreement passes: agreement is the ideal "
                "outcome, and whether a lane ran at all is the separate "
                "backend_lane_coverage gate"
            ),
        ),
        EcologyMechanismGate(
            name="no_optimize_policy_stable",
            passed=no_optimize_stable,
            observed=str(no_optimize_stable),
            threshold="no-optimize policy fingerprints equal shared initial",
        ),
        EcologyMechanismGate(
            name="temporal_positive_control",
            passed=bool(positive.switch_ticks) and bool(localized),
            observed=str(
                {
                    "switch_ticks": positive.switch_ticks,
                    "localizations": tuple(
                        (item.phase, item.boundary_tick, item.distance)
                        for item in positive.boundary_localizations
                    ),
                }
            ),
            threshold=(
                "the scripted transition trace produces at least one real "
                "beta switch and at least one switch within "
                f"{config.switch_localization_window} ticks of a declared "
                "state change"
            ),
        ),
        EcologyMechanismGate(
            name="temporal_negative_control",
            passed=(
                negative.switch_rate
                <= config.negative_control_switch_rate_ceiling
            ),
            observed=(
                f"switch_rate={negative.switch_rate:.4f} "
                f"switch_ticks={negative.switch_ticks}"
            ),
            threshold=(
                "steady-state input switch rate <= "
                f"{config.negative_control_switch_rate_ceiling:g}"
            ),
        ),
        EcologyMechanismGate(
            name="temporal_segment_closure",
            passed=(
                positive.closed_segment_count > 0
                and positive_reasons.get("beta-switch", 0) > 0
                and positive_reasons.get("environment-milestone", 0) > 0
                and positive.timeout_closure_ratio
                < config.timeout_closure_ratio_ceiling
            ),
            observed=str(
                {
                    "close_reasons": positive.close_reason_counts,
                    "timeout_ratio": positive.timeout_closure_ratio,
                }
            ),
            threshold=(
                "at least one beta-switch closure, at least one "
                "milestone/terminal closure, and a bounded-horizon closure "
                "ratio strictly below "
                f"{config.timeout_closure_ratio_ceiling:g}"
            ),
        ),
        EcologyMechanismGate(
            name="segment_credit_parity",
            passed=temporal_switch.parity.passed,
            observed=str(
                {
                    "sense": temporal_switch.parity.max_sense_delta,
                    "turn": temporal_switch.parity.max_turn_delta,
                    "step": temporal_switch.parity.max_step_delta,
                    "lineage": temporal_switch.parity.lineage_differences,
                    "first_misaligned_tick": (
                        temporal_switch.parity.first_misaligned_tick
                    ),
                }
            ),
            threshold=(
                "segment-credit on/off share sense, pre-credit action and "
                "rollout lineage within "
                f"{config.segment_credit_parity_tolerance:g}"
            ),
        ),
        EcologyMechanismGate(
            name="frozen_evaluation",
            passed=bool(frozen_evaluations)
            and all(item.passed for item in frozen_evaluations),
            observed=str(
                tuple(
                    (
                        item.scenario,
                        item.seed,
                        item.unstable_owner_names,
                        item.block_reason,
                        item.replay_settlement_coverage,
                        item.replay_lineage_coverage,
                        item.replay_drop_count,
                    )
                    for item in frozen_evaluations
                )
            ),
            threshold=(
                "every gated learning owner is fingerprint-identical on every "
                "tick under learning_enabled=False; replay settlement and "
                f"lineage >= {ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR} with no "
                "drops"
            ),
        ),
    )


async def run_ecology_mechanism_audit(
    config: EcologyMechanismAuditConfig,
) -> EcologyMechanismAuditReport:
    """Run the complete P0 audit without emitting a promotion checkpoint."""

    curriculum_config = _curriculum_config(config)
    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum_config,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum_config,
            seed=config.seed,
            session_id="ecology:p0:shared-initial",
            optimize=True,
        ),
    )
    initial = bootstrap.export_learning_checkpoints(
        checkpoint_prefix="ecology:p0:shared-initial",
        include_runtime_replay=False,
    )
    initial_probes = await _probe_checkpoints(
        config=config,
        checkpoints=initial,
        seed_offset=_PRIMARY_PROBE_SEED_OFFSET,
    )
    initial_snapshot = _evaluate_action_snapshot(
        config=config,
        arm="shared-initial",
        label="shared-initial",
        stage="initial",
        tier="initial",
        episode_index=-1,
        gate_mode=_GATE_MODE_INPUT_REACHABILITY,
        body_reports=initial_probes,
        retention_baselines={},
    )
    learned, learned_snapshots, learned_segments = await _train_audit_arm(
        audit_config=config,
        curriculum_config=curriculum_config,
        initial=initial,
        arm="learned",
        optimize=True,
    )
    no_optimize, no_opt_snapshots, no_opt_segments = (
        await _train_audit_arm(
            audit_config=config,
            curriculum_config=curriculum_config,
            initial=initial,
            arm="no_optimize",
            optimize=False,
        )
    )
    final_probes = await _probe_checkpoints(
        config=config,
        checkpoints=learned,
        seed_offset=_PRIMARY_PROBE_SEED_OFFSET,
    )
    final_learned_snapshot = _evaluate_action_snapshot(
        config=config,
        arm="learned",
        label="final",
        stage="final",
        tier="final",
        episode_index=config.episodes_per_stage,
        gate_mode=_GATE_MODE_POST_TRAINING,
        body_reports=final_probes,
        retention_baselines=_peak_baselines(
            snapshots=learned_snapshots,
            config=config,
        ),
    )
    sign_consistency, _sign_seeds = await _sign_consistency(
        config=config,
        checkpoints=learned,
        first_repeat=final_probes,
    )
    lateral_bias = _lateral_bias(config=config, body_reports=final_probes)
    action_head_updates = _action_head_updates(
        initial=initial,
        learned=learned,
        body_reports=final_probes,
    )
    backend_parity = await _backend_parity(config=config, checkpoints=learned)
    temporal_switch = await _temporal_switch_audit(
        config=config,
        checkpoint=learned[0],
    )
    frozen = tuple(
        [
            await _frozen_evaluation_audit(
                audit_config=config,
                curriculum_config=curriculum_config,
                checkpoints=learned,
                scenario=scenario,
                seed=seed,
            )
            for scenario, seed in ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES
        ]
    )
    action_snapshots = (
        (initial_snapshot,)
        + learned_snapshots
        + no_opt_snapshots
        + (final_learned_snapshot,)
    )
    segment_telemetry = learned_segments + no_opt_segments
    rollback_episodes = tuple(
        f"{item.arm}:{item.stage}:{item.tier}:episode:{item.episode_index}"
        for item in segment_telemetry
        if item.action_chain_rollback_applied
    )
    no_optimize_stable = all(
        before.policy_fingerprint == after.policy_fingerprint
        for before, after in zip(initial, no_optimize, strict=True)
    )
    # The failure a silent rollback produces: the gated "trained" checkpoint
    # is byte-identical to the cold one it forked from.
    gated_checkpoint_is_post_training = all(
        before.policy_fingerprint != after.policy_fingerprint
        for before, after in zip(initial, learned, strict=True)
    )
    first_failing_learned_episode = next(
        (
            f"{item.arm}:{item.label}"
            for item in learned_snapshots
            if not item.passed
        ),
        "",
    )
    gates = build_ecology_mechanism_gates(
        config=config,
        initial_snapshot=initial_snapshot,
        final_learned_snapshot=final_learned_snapshot,
        sign_consistency=sign_consistency,
        lateral_bias=lateral_bias,
        action_head_updates=action_head_updates,
        backend_parity=backend_parity,
        rollback_episodes=rollback_episodes,
        action_probe_guard_enabled=(
            curriculum_config.action_probe_guard_enabled
        ),
        gated_checkpoint_is_post_training=gated_checkpoint_is_post_training,
        no_optimize_stable=no_optimize_stable,
        temporal_switch=temporal_switch,
        frozen_evaluations=frozen,
    )
    breakpoints = tuple(gate.name for gate in gates if not gate.passed)
    verdict = "PASS" if not breakpoints else "BLOCK"
    return EcologyMechanismAuditReport(
        schema_version=ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION,
        config=config,
        action_probe_guard_enabled=(
            curriculum_config.action_probe_guard_enabled
        ),
        initial_snapshot=initial_snapshot,
        final_learned_snapshot=final_learned_snapshot,
        action_chain_snapshots=action_snapshots,
        sign_consistency=sign_consistency,
        lateral_bias=lateral_bias,
        action_head_updates=action_head_updates,
        backend_parity=backend_parity,
        segment_telemetry=segment_telemetry,
        rollback_episodes=rollback_episodes,
        temporal_switch=temporal_switch,
        frozen_evaluations=frozen,
        gates=gates,
        verdict=verdict,
        diagnostic_breakpoints=breakpoints,
        description=(
            f"{verdict}: "
            + (", ".join(breakpoints) if breakpoints else "all P0 gates passed")
        ),
        first_failing_learned_episode=first_failing_learned_episode,
        diagnostic_surfaces=(
            EcologyDiagnosticSurface(
                name="action_chain_snapshots[learned:per-episode]",
                gated=False,
                reason=(
                    "The intermediate learned snapshots are EVALUATED and "
                    "recorded (each carries its own passed/failures), but no "
                    "gate reads them: gating a mid-training episode would "
                    "block on a checkpoint that later training is allowed to "
                    "improve. plan 05:130's 'first failing episode -> bisect "
                    "replay' branch is triggered by "
                    "report.first_failing_learned_episode instead, which "
                    "names the first one that failed."
                ),
            ),
            EcologyDiagnosticSurface(
                name="action_chain_snapshots[no_optimize:per-episode]",
                gated=False,
                reason=(
                    "The no-optimize arm's action sensitivity is a control, "
                    "not a requirement: what the arm must prove is that its "
                    "policy fingerprint never moved, and that is the "
                    "no_optimize_policy_stable gate."
                ),
            ),
            EcologyDiagnosticSurface(
                name="lateral_bias / sign_consistency probe seeds",
                gated=True,
                reason=(
                    "Read by action_chain_lateral_bias and "
                    "action_chain_sign_consistency."
                ),
            ),
        ),
        declared_gaps=ECOLOGY_AUDIT_DECLARED_GAPS,
    )


def _peak_baselines(
    *,
    snapshots: tuple[EcologyActionChainSnapshot, ...],
    config: EcologyMechanismAuditConfig,
) -> dict[tuple[int, str], float]:
    baselines: dict[tuple[int, str], float] = {}
    for snapshot in snapshots:
        _update_retention_baselines(
            baselines=baselines,
            body_reports=snapshot.body_reports,
            config=config,
        )
    return baselines


__all__ = [
    "ECOLOGY_AUDIT_ALLOWED_TO_CHANGE_OWNERS",
    "ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS",
    "ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE",
    "ECOLOGY_AUDIT_BODY_PASS_RATIO",
    "ECOLOGY_AUDIT_DECLARED_GAPS",
    "ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD",
    "ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES",
    "ECOLOGY_AUDIT_FROZEN_EVALUATION_SEEDS",
    "ECOLOGY_AUDIT_GATED_LEARNING_OWNERS",
    "ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING",
    "ECOLOGY_AUDIT_PROTOCOL_PHASE_TICKS",
    "ECOLOGY_AUDIT_REPLAY_COVERAGE_FLOOR",
    "ECOLOGY_AUDIT_RETENTION_RATIO",
    "ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE",
    "ECOLOGY_AUDIT_SIGN_REPEAT_COUNT",
    "ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW",
    "ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING",
    "ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD",
    "ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION",
    "EcologyActionChainSnapshot",
    "EcologyActionHeadUpdate",
    "EcologyBackendParityLane",
    "EcologyBoundaryLocalization",
    "EcologyDeclaredGap",
    "EcologyDiagnosticSurface",
    "EcologyFrozenEvaluationAudit",
    "EcologyLateralBias",
    "EcologyMechanismAuditConfig",
    "EcologyMechanismAuditError",
    "EcologyMechanismAuditReport",
    "EcologyMechanismGate",
    "EcologyOwnerDifference",
    "EcologySegmentCreditParity",
    "EcologySegmentTelemetry",
    "EcologySignConsistency",
    "EcologySwitchParameterSnapshot",
    "EcologyTemporalSwitchAudit",
    "EcologyTemporalTick",
    "EcologyTransitionPhase",
    "EcologyTransitionTrace",
    "build_ecology_mechanism_gates",
    "ecology_mechanism_audit_seed_schedule",
    "run_ecology_mechanism_audit",
]
