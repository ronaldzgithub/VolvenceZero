"""P1 fixed-schedule ecology capability development matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from volvence_zero.agent import (
    AgentLearningArchiveError,
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_checkpoint_archive,
)

from volvence_ant.controllers import FixedRuleAnt, FixedRuleConfig, RandomAnt
from volvence_ant.env.world_objects import ButterSource, BurningMatch, WoodStick
from volvence_ant.experiments.ecology_probe import (
    ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY,
    ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS,
    ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS,
    EcologyCheckpointPostPickupUTurnProbe,
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
    run_ecology_checkpoint_post_pickup_uturn_probes,
)
from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY,
    EcologyArmMetrics,
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyEvaluationScenario,
    EcologyStage,
    EcologyTrainingEpisodePlan,
    EcologyTrainingEpisodeReport,
    EcologyTrainingTier,
    _evaluate_arm,
    _ecology_action_chain_guard,
    _session_config,
    _train_arm,
    _world,
    ecology_training_min_stage_rounds,
)
from volvence_ant.runtime import AntLearningCheckpoint, KernelColonyRunner
from volvence_ant.substrate import AntSenseSchema, sense_channels


# v31 binds curriculum v14's typed milestone boundary: the environment owner
# declares pickup/delivery as ``EnvironmentMeasurement.discrete_milestone``
# and the temporal owner closes the running action segment on the next
# decision, resolved against its current learned beta threshold. v30 let
# outbound and carrying credit share one segment because no mechanism closed
# the segment at pickup; the interim PE-magnitude floor design was refuted
# before any v31 journal existed (routine PE p50 0.508 overlaps event PE;
# natural pickups settle at ~0.32 -- see
# research/ant/06_ecology_implementation_status.md), so PE stays an
# additive-only prior and never decides boundaries.
#
# v30 binds curriculum v13's real pickup-to-return training transition,
# pickup-triggered frozen switch requirement and late interleaved return
# rehearsal. v29 trained forced-return bodies already carrying, so it skipped
# the exact action-family transition that the real-pickup gate exposed.
#
# v29 binds curriculum v12's frozen post-pickup U-turn gate. v28 could still
# accept a one-tick carrying-home direction that never reduced home distance
# over a trajectory, so its report semantics are not comparable.
#
# v28 binds curriculum v11's +/-135 degree forced-return pressure. v27 learned
# the correct carrying-home direction but not enough authority for the near-pi
# turn after a natural pickup, so its optimizer state is not comparable.
#
# v27 binds curriculum v10's tangent forced-return pressure. A v26 policy
# could harvest positive home-progress on a straight path that missed the
# delivery disc, so its homing evidence is not comparable.
#
# v26 changes what a P1 verdict MEANS (plan section 4.4/4.7), so every older
# report and journal must be refused rather than reinterpreted:
#   * three new gates (formal_configuration, checkpoint_archive_roundtrip,
#     repeat_run_same_direction) enter ECOLOGY_P1_GATE_NAMES;
#   * ``heat_route_foraging`` finally evaluates HEAT_ROUTE_AVOIDANCE instead of
#     a byte-identical copy of ``composite``, so the held-out layouts behind
#     that capability's numbers are different layouts;
#   * ``composite`` gains the matched no-optimize exposure conjunct and
#     ``temporal_non_timeout_closure`` becomes a per-layout ratio;
#   * the held-out budget is aligned to P2's frozen 120 rounds.
ECOLOGY_P1_SCHEMA_VERSION = "digital-ant-ecology-p1-development.v31"
# v28 follows curriculum v14 / P1 v31. The same physical schedule now closes
# a typed pickup/delivery milestone boundary relative to the current learned
# beta threshold, so v27 optimizer state contains differently segmented
# action credit and cannot be resumed.
# v27 follows curriculum v13 / P1 v30. The forced-return world and schedule
# now contain real pickup transitions plus late rehearsal, so v26 optimizer
# state is from a different experiment and must never be resumed.
# v26 follows curriculum v12 / P1 v29. A clean journal is required so the
# report's hard U-turn evidence and the checkpoint it grades share one
# declared generation rather than attaching a new gate to prior training.
# v25 follows the curriculum v11 / P1 v28 pressure change. A v24 journal
# contains optimizer state trained only on +/-90 degree return starts and must
# fail loudly instead of resuming into the stronger large-angle pressure.
# v24 follows the curriculum v10 / P1 v27 pressure change. A v23 journal
# contains optimizer state trained on the retired +/-30 degree start and must
# fail loudly instead of resuming into tangent episodes.
# v23 follows the v26 report bump: EcologyP1LayoutResult gained
# ``closed_segment_layouts`` semantics via the new gate set, the frozen
# evaluation budget moved from 40 to 120 rounds, and the journal now also
# carries regime-diagnostic rows. A v22 journal describes a different
# experiment and is refused, not resumed.
# v22 bound sense schema and input dim into the resume archive compatibility
# (spec sections 3 and 8: archive compatibility binds sense schema / input dim
# / latent dim / ant count).
ECOLOGY_P1_PROGRESS_SCHEMA_VERSION = "digital-ant-ecology-p1-progress.v28"
ECOLOGY_P1_DIAGNOSTICS_SCHEMA_VERSION = "digital-ant-ecology-p1-diagnostics.v3"
ECOLOGY_P1_ARM_NAMES = (
    "learned",
    "no_optimize",
    "cold",
    "dense_local_shaping_off",
    "segment_credit_off",
)
ECOLOGY_P1_GATE_NAMES = (
    "formal_configuration",
    "butter_medium",
    "butter_far",
    "forced_escape",
    "heat_route_foraging",
    "neutral_stick",
    "composite",
    "forced_escape_above_random_floor",
    "learned_not_worse_than_no_optimize",
    "paired_capability_effect_positive",
    "diagnostic_layout_solvability",
    "p0_action_sensitivity",
    "carrying_home_action_alignment",
    "post_pickup_uturn_progress",
    "food_steering_alignment",
    "temporal_non_timeout_closure",
    "frozen_evaluation",
    "replay_lineage",
    "checkpoint_archive_roundtrip",
    "repeat_run_same_direction",
)

# --- the formal P1 budget (plan section 4.4 line 1 and section 4.5) ---------
#
# These are thresholds, not defaults, and they are enforced by the
# ``formal_configuration`` GATE rather than by ``__post_init__``.  The plan's
# own test ladder (section 7 step 3) requires a "1 ant, fixed seed" small-budget
# deterministic test, and section 3.5/7 requires the P0 audit and the unit
# tests to construct this very config at tiny budgets; raising in
# ``__post_init__`` would delete that whole tier of verification.  A gate keeps
# development runs executable while making it structurally impossible for an
# under-budget run to emit ``verdict="PASS"`` -- and because
# ``load_p1_prerequisite`` refuses any P1 report with an unpassed gate, it also
# closes the hole that let a 1-ant smoke unlock P2.
ECOLOGY_P1_FORMAL_MIN_ANTS = 4
ECOLOGY_P1_FORMAL_LATENT_DIM = 16
ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER = 5
# The curriculum owner derives this floor from the frozen tier geometry and
# plant.  P1 must not restate a literal: doing so left the former 24-round
# "formal" default unable to sample even the near milestone after the v9
# geometry correction (near/medium/far require 28/33/49 rounds).
ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS = ecology_training_min_stage_rounds()
# Owner decision: P1's held-out budget is aligned to P2's frozen floor
# (``ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS``). The deterministic policy the
# capability gates grade has a measured steering authority of 0.0055-0.033
# rad/tick, so 40 ticks buy at most ~1.3 rad of cumulative heading change while
# a directed medium/far round trip needs ~3.4-4.0 rad: at 40 rounds the gates
# could not have been satisfied by any policy, learned or not.
# ``test_ecology_p1`` asserts this equals the P2 constant.
ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS = 120

# Plan section 4.4: "held-out 必须出现真实 beta switch，且不能全部由 timeout
# 关闭 segment". Expressed as a ratio over held-out layouts rather than one
# existence test aggregated over all of them. The ceiling is the complement of
# the plan's frozen 0.6 layout ratio -- at least 60% of learned held-out
# layouts must individually show a real switch and a non-timeout closure -- so
# it introduces no new free number.
ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX = 1.0 - 0.6

# The regime-diagnostic lanes (read-only, never a gate, never fed back).
ECOLOGY_P1_REGIME_DETERMINISTIC = "deterministic_frozen"
ECOLOGY_P1_REGIME_STOCHASTIC = "training_stochastic"
ECOLOGY_P1_REGIME_NAMES = (
    ECOLOGY_P1_REGIME_DETERMINISTIC,
    ECOLOGY_P1_REGIME_STOCHASTIC,
)


@dataclass(frozen=True)
class EcologyP1Config:
    n_ants: int = 4
    temporal_latent_dim: int = 16
    training_rounds: int = ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS
    evaluation_rounds: int = ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS
    layouts_per_tier: int = 5
    seed: int = 0
    layout_success_ratio: float = 0.6
    body_success_ratio: float = 0.6
    harmful_tick_rate_max: float = 0.05

    def __post_init__(self) -> None:
        if self.n_ants < 1 or self.temporal_latent_dim < 3:
            raise ValueError("P1 requires ants >=1 and latent dim >=3")
        if min(
            self.training_rounds,
            self.evaluation_rounds,
            self.layouts_per_tier,
        ) < 1:
            raise ValueError("P1 budgets must be positive")
        if self.layout_success_ratio != 0.6:
            raise ValueError("P1 layout success threshold is frozen at 0.6")
        if self.body_success_ratio != 0.6:
            raise ValueError("P1 body success threshold is frozen at 0.6")
        if self.harmful_tick_rate_max != 0.05:
            raise ValueError("P1 harmful tick threshold is frozen at 0.05")


#: The config fields the formal P1 budget is defined over. Named once so the
#: predicate below and a JSON consumer read the same list.
ECOLOGY_P1_FORMAL_BUDGET_FIELDS = (
    "n_ants",
    "temporal_latent_dim",
    "layouts_per_tier",
    "training_rounds",
    "evaluation_rounds",
)


def _formal_budget_values(
    config: EcologyP1Config | Mapping[str, Any],
) -> dict[str, int]:
    """The five budget integers, from a config object OR its JSON mapping.

    A report's ``config`` block is the only self-describing record of what a
    run actually spent, so a consumer that re-derives the budget verdict from
    it cannot be fooled by a hand-edited ``passed`` boolean. Missing, boolean
    or non-integer fields are a contract violation and raise -- a report whose
    config cannot be read is not a report whose budget can be trusted.
    """

    if isinstance(config, EcologyP1Config):
        source: Mapping[str, Any] = asdict(config)
    elif isinstance(config, Mapping):
        source = config
    else:
        raise TypeError(
            "P1 formal budget expects an EcologyP1Config or a config mapping, "
            f"got {type(config).__name__}"
        )
    values: dict[str, int] = {}
    for field in ECOLOGY_P1_FORMAL_BUDGET_FIELDS:
        if field not in source:
            raise ValueError(
                "P1 config is missing formal budget field "
                f"{field!r}; the budget verdict cannot be re-derived"
            )
        raw = source[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(
                f"P1 config field {field!r} must be an integer, got {raw!r}"
            )
        values[field] = int(raw)
    return values


def ecology_p1_formal_budget_failures(
    config: EcologyP1Config | Mapping[str, Any],
) -> tuple[str, ...]:
    """Every way ``config`` falls short of the frozen formal P1 budget.

    Accepts the live ``EcologyP1Config`` *or* the ``config`` mapping parsed
    out of a written P1 report, so a promotion gate can re-derive this verdict
    from the artifact instead of trusting the artifact's own gate row. A
    report whose config says ``n_ants=1, layouts_per_tier=1`` returns failures
    here no matter what its ``formal_configuration`` gate claims.
    """

    values = _formal_budget_values(config)
    failures: list[str] = []
    if values["n_ants"] < ECOLOGY_P1_FORMAL_MIN_ANTS:
        failures.append(
            f"n_ants={values['n_ants']}<{ECOLOGY_P1_FORMAL_MIN_ANTS}"
        )
    if values["temporal_latent_dim"] != ECOLOGY_P1_FORMAL_LATENT_DIM:
        failures.append(
            f"temporal_latent_dim={values['temporal_latent_dim']}"
            f"!={ECOLOGY_P1_FORMAL_LATENT_DIM}"
        )
    if values["layouts_per_tier"] < ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER:
        failures.append(
            f"layouts_per_tier={values['layouts_per_tier']}"
            f"<{ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER}"
        )
    if values["training_rounds"] < ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS:
        failures.append(
            f"training_rounds={values['training_rounds']}"
            f"<{ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS}"
        )
    if values["evaluation_rounds"] < ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS:
        failures.append(
            f"evaluation_rounds={values['evaluation_rounds']}"
            f"<{ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS}"
        )
    return tuple(failures)


@dataclass(frozen=True)
class EcologyP1LayoutResult:
    arm: str
    capability: str
    seed: int
    tier: str
    successful_bodies: int
    required_bodies: int
    layout_success: bool
    harmful_tick_rate: float
    escape_latencies: tuple[int, ...]
    switch_count: int
    non_timeout_segment_closures: int
    policy_fingerprint_stable: bool
    temporal_learning_fingerprint_stable: bool
    replay_settlement_coverage: float
    replay_lineage_coverage: float
    replay_drop_count: int


@dataclass(frozen=True)
class EcologyP1Gate:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True)
class EcologyP1DiagnosticResult:
    controller: str
    capability: str
    seed: int
    tier: str
    successful_bodies: int
    required_bodies: int
    layout_success: bool
    pickups: int
    deliveries: int
    heat_escapes: int
    escape_latencies: tuple[int, ...]
    harmful_heat_ticks: int


@dataclass(frozen=True)
class EcologyP1EscapeLatencySummary:
    """Plan section 4.4 asks for median AND p90 escape latency, per source.

    ``median`` / ``p90`` are ``None`` when the source produced no escape at
    all: a JSON report is written with ``allow_nan=False``, so an infinite
    sentinel could not be serialised, and 0 would read as an instant escape.
    """

    source: str
    sample_count: int
    median: float | None
    p90: float | None


@dataclass(frozen=True)
class EcologyP1TurnMagnitudeDistribution:
    """Per-tick |turn| distribution of one rollout regime."""

    sample_count: int
    mean_abs: float
    median_abs: float
    p90_abs: float
    max_abs: float


@dataclass(frozen=True)
class EcologyP1RegimeDiagnosticRow:
    """One held-out layout replayed under one kinematic regime.

    READ-ONLY. Never a gate, never an input to learning: the rollout runs with
    ``optimize=False`` and ``learning_enabled=False`` and the FROZEN-LEARNED
    owner set (``policy_fingerprint`` + ``temporal_learning_fingerprint``) is
    asserted unchanged afterwards -- the same set the ``frozen_evaluation``
    gate and ``ecology_mechanism_audit`` use.

    ``memory_fingerprint`` is deliberately NOT in that set. It drifts during a
    frozen rollout today (measured owners: credit, dual-track-gate,
    joint-loop/memory, prediction, reflection, regime), and that defect is
    already owned and BLOCKED by the P0 ``frozen_evaluation`` gate. This
    diagnostic exists to quantify the training/evaluation kinematic gap, so it
    PUBLISHES the memory drift as evidence on its own row instead of being
    made permanently unable to report by a defect another gate blocks on.
    Nothing is silently accepted either way.
    """

    regime: str
    capability: str
    tier: str
    seed: int
    successful_bodies: int
    required_bodies: int
    layout_success: bool
    harmful_tick_rate: float
    turn_magnitude: EcologyP1TurnMagnitudeDistribution
    #: False when any body's ``memory_fingerprint`` moved during the frozen
    #: rollout. Evidence, not a verdict -- see the class docstring.
    memory_fingerprint_stable: bool
    #: Indices (colony body order) of the checkpoints whose memory drifted.
    drifted_memory_bodies: tuple[int, ...]


@dataclass(frozen=True)
class EcologyP1RegimeGapSummary:
    """The training/evaluation kinematic gap, per capability, as numbers."""

    capability: str
    deterministic_successful_layouts: int
    stochastic_successful_layouts: int
    layouts: int
    deterministic_median_abs_turn: float
    stochastic_median_abs_turn: float
    #: stochastic / deterministic median |turn|; ``None`` when the
    #: deterministic median is exactly 0 and the ratio is undefined.
    median_abs_turn_ratio: float | None


@dataclass(frozen=True)
class EcologyP1RepeatReference:
    """A previous P1 report used as the independent repetition (plan 4.7)."""

    report_path: str
    report_sha256: str
    schema_version: str
    seed: int
    verdict: str
    direction_signature: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class EcologyP1Report:
    schema_version: str
    config: EcologyP1Config
    schedule: tuple[EcologyTrainingEpisodePlan, ...]
    layout_results: tuple[EcologyP1LayoutResult, ...]
    diagnostic_results: tuple[EcologyP1DiagnosticResult, ...]
    escape_latency_summaries: tuple[EcologyP1EscapeLatencySummary, ...]
    regime_diagnostic: tuple[EcologyP1RegimeDiagnosticRow, ...]
    regime_gap_summary: tuple[EcologyP1RegimeGapSummary, ...]
    repeat_reference: EcologyP1RepeatReference | None
    post_pickup_uturn_probes: tuple[
        EcologyCheckpointPostPickupUTurnProbe, ...
    ]
    gates: tuple[EcologyP1Gate, ...]
    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EcologyP1DiagnosticReport:
    schema_version: str
    config: EcologyP1Config
    results: tuple[EcologyP1DiagnosticResult, ...]
    oracle_success_by_capability: tuple[tuple[str, int], ...]
    required_layouts: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EcologyP1ProgressPaused(RuntimeError):
    """A bounded resumable run stopped after a committed work item."""

    def __init__(self, *, completed_work_items: int) -> None:
        self.completed_work_items = completed_work_items
        super().__init__(
            "P1 resumable run paused after "
            f"{completed_work_items} committed work items"
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value


def _stable_json_bytes(value: Any) -> bytes:
    # allow_nan=False: a NaN would otherwise be written into a journal or
    # digested into a shard fingerprint as the non-standard ``NaN`` token and
    # silently compare unequal to itself on reload.
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _progress_config_payload(config: EcologyP1Config) -> dict[str, Any]:
    return _json_ready(asdict(config))


def _schedule_digest(
    schedule: tuple[EcologyTrainingEpisodePlan, ...],
) -> str:
    return _sha256(_stable_json_bytes(tuple(asdict(item) for item in schedule)))


def _progress_compatibility(
    config: EcologyP1Config,
    *,
    include_memory_capacity: bool = True,
) -> tuple[tuple[str, str], ...]:
    # sense_schema and input_dim are part of the frozen compatibility contract
    # (docs/specs/digital-ant-embodiment.md: archive compatibility binds sense
    # schema, input dim, latent dim and ant count). A resume journal is an
    # archive an interrupted formal run rehydrates from, so it must bind the
    # same four; binding only latent dim would let a 19-channel ecology-v2 body
    # rehydrate from a 14-channel v1 checkpoint.
    values = [
        ("artifact_kind", ECOLOGY_P1_PROGRESS_SCHEMA_VERSION),
        ("sense_schema", AntSenseSchema.ECOLOGY_V2.value),
        (
            "input_dim",
            str(len(sense_channels(AntSenseSchema.ECOLOGY_V2))),
        ),
        ("n_ants", str(config.n_ants)),
        ("latent_dim", str(config.temporal_latent_dim)),
        ("runtime_replay", "excluded"),
    ]
    if include_memory_capacity:
        values.append(
            (
                "memory_entry_capacity",
                str(ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY),
            )
        )
    return tuple(values)


def _arm_progress_path(progress_dir: Path, arm: str) -> Path:
    return progress_dir / f"{arm}.json"


def _load_arm_progress(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
    schedule_sha256: str,
) -> dict[str, Any] | None:
    state_path = _arm_progress_path(progress_dir, arm)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        "arm": arm,
        "config": _progress_config_payload(config),
        "schedule_sha256": schedule_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"P1 progress mismatch for {arm}: field={field}, "
                f"expected={value!r}, actual={payload.get(field)!r}"
            )
    completed = payload.get("completed_training_episodes")
    if not isinstance(completed, int) or completed < 0:
        raise ValueError(
            f"P1 progress for {arm} has invalid completed episode count"
        )
    if int(
        payload.get(
            "checkpoint_memory_entry_capacity",
            ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY,
        )
    ) != ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY:
        raise ValueError(
            f"P1 progress memory capacity mismatch for {arm}"
        )
    return payload


def _read_progress_archive(
    *,
    progress_dir: Path,
    state: dict[str, Any],
    config: EcologyP1Config,
) -> tuple[bytes, ...]:
    filename = state.get("checkpoint_archive")
    if not isinstance(filename, str) or not filename:
        raise ValueError("P1 progress is missing checkpoint_archive")
    archive_path = (progress_dir / filename).resolve()
    archive_path.relative_to(progress_dir.resolve())
    payload = archive_path.read_bytes()
    if _sha256(payload) != state.get("checkpoint_sha256"):
        raise ValueError(
            f"P1 progress archive digest mismatch: {archive_path.name}"
        )
    expected_compatibility = _progress_compatibility(
        config,
        include_memory_capacity=(
            "checkpoint_memory_entry_capacity" in state
        ),
    )
    try:
        collection = decode_agent_learning_checkpoint_archive(
            payload,
            expected_compatibility=expected_compatibility,
        )
    except ValueError as exc:
        raise ValueError(
            "P1 progress checkpoint is not compatible with this run "
            f"({archive_path.name}): {exc}. Journals from a different "
            f"algorithm or schema generation than "
            f"{ECOLOGY_P1_PROGRESS_SCHEMA_VERSION} cannot be resumed; start a new "
            "--progress-dir for this configuration."
        ) from exc
    if dict(collection.metadata.compatibility) != dict(
        expected_compatibility
    ):
        raise ValueError(
            "P1 progress checkpoint compatibility mismatch: "
            f"expected={dict(expected_compatibility)!r}, "
            f"actual={dict(collection.metadata.compatibility)!r}"
        )
    return collection.checkpoint_archives


def _save_arm_progress(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
    schedule_sha256: str,
    completed_training_episodes: int,
    runner: KernelColonyRunner,
    training_complete: bool,
    last_episode_report: EcologyTrainingEpisodeReport | None = None,
) -> None:
    raw_archives = runner.export_learning_checkpoint_archives(
        checkpoint_prefix=(
            f"ecology:p1:progress:{arm}:"
            f"episode-{completed_training_episodes:04d}"
        )
    )
    archive = encode_agent_learning_checkpoint_archive(
        raw_archives,
        compatibility=_progress_compatibility(config),
    )
    # Two-slot journal: write the slot not referenced by the previous state,
    # fsync/rename it, then atomically advance the JSON pointer. This retains
    # one rollback checkpoint while bounding a long formal run to two colony
    # archives per arm.
    archive_name = (
        f"{arm}.slot-{completed_training_episodes % 2}.vzac"
    )
    _atomic_write(progress_dir / archive_name, archive)
    state = {
        "schema_version": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        "arm": arm,
        "config": _progress_config_payload(config),
        "schedule_sha256": schedule_sha256,
        "completed_training_episodes": completed_training_episodes,
        "training_complete": training_complete,
        "checkpoint_memory_entry_capacity": (
            ECOLOGY_CHECKPOINT_MEMORY_ENTRY_CAPACITY
        ),
        "checkpoint_archive": archive_name,
        "checkpoint_sha256": _sha256(archive),
        "last_episode": (
            {
                "stage": last_episode_report.plan.stage.value,
                "tier": last_episode_report.plan.tier.value,
                "episode_index": (
                    last_episode_report.plan.episode_index
                ),
                "forced_escape": (
                    last_episode_report.plan.forced_escape
                ),
                "forced_return": (
                    last_episode_report.plan.forced_return
                ),
                "forced_approach": (
                    last_episode_report.plan.forced_approach
                ),
                "pickups": last_episode_report.pickups,
                "deliveries": last_episode_report.deliveries,
                "heat_entries": last_episode_report.heat_entries,
                "heat_escapes": last_episode_report.heat_escapes,
                "memory_entries_evicted": (
                    last_episode_report.memory_entries_evicted
                ),
            }
            if last_episode_report is not None
            else None
        ),
    }
    _atomic_write(
        _arm_progress_path(progress_dir, arm),
        _stable_json_bytes(state),
    )


def _hydrate_progress_checkpoints(
    *,
    config: EcologyP1Config,
    curriculum: EcologyCurriculumConfig,
    archives: tuple[bytes, ...],
    arm: str,
) -> tuple[AntLearningCheckpoint, ...]:
    runner = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=config.seed,
            session_id=f"ecology:p1:progress:{arm}:hydrate",
            optimize=False,
        ),
    )
    runner.restore_learning_checkpoint_archives(archives)
    return runner.export_learning_checkpoints(
        checkpoint_prefix=f"ecology:p1:progress:{arm}:hydrated",
        include_runtime_replay=False,
    )


def _layout_result_from_dict(
    payload: dict[str, Any],
) -> EcologyP1LayoutResult:
    return EcologyP1LayoutResult(
        arm=str(payload["arm"]),
        capability=str(payload["capability"]),
        seed=int(payload["seed"]),
        tier=str(payload["tier"]),
        successful_bodies=int(payload["successful_bodies"]),
        required_bodies=int(payload["required_bodies"]),
        layout_success=bool(payload["layout_success"]),
        harmful_tick_rate=float(payload["harmful_tick_rate"]),
        escape_latencies=tuple(
            int(value) for value in payload["escape_latencies"]
        ),
        switch_count=int(payload["switch_count"]),
        non_timeout_segment_closures=int(
            payload["non_timeout_segment_closures"]
        ),
        policy_fingerprint_stable=bool(
            payload["policy_fingerprint_stable"]
        ),
        temporal_learning_fingerprint_stable=bool(
            payload["temporal_learning_fingerprint_stable"]
        ),
        replay_settlement_coverage=float(
            payload["replay_settlement_coverage"]
        ),
        replay_lineage_coverage=float(
            payload["replay_lineage_coverage"]
        ),
        replay_drop_count=int(payload["replay_drop_count"]),
    )


def _evaluation_progress_path(progress_dir: Path) -> Path:
    return progress_dir / "evaluations.json"


def _load_evaluation_progress(
    *,
    progress_dir: Path,
    config: EcologyP1Config,
    schedule_sha256: str,
    arm_checkpoint_sha256: dict[str, str],
) -> list[EcologyP1LayoutResult]:
    path = _evaluation_progress_path(progress_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        "config": _progress_config_payload(config),
        "schedule_sha256": schedule_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"P1 evaluation progress mismatch: field={field}"
            )
    if payload.get("arm_checkpoint_sha256") != arm_checkpoint_sha256:
        # Training resumed from an earlier immutable episode and produced a
        # different final checkpoint. Existing evaluation rows remain on disk
        # for audit but are not eligible for reuse under the new checkpoint.
        return []
    raw_results = payload.get("layout_results")
    if not isinstance(raw_results, list):
        raise ValueError("P1 evaluation progress results must be a list")
    return [
        _layout_result_from_dict(item)
        for item in raw_results
    ]


def _save_evaluation_progress(
    *,
    progress_dir: Path,
    config: EcologyP1Config,
    schedule_sha256: str,
    arm_checkpoint_sha256: dict[str, str],
    results: list[EcologyP1LayoutResult],
) -> None:
    payload = {
        "schema_version": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        "config": _progress_config_payload(config),
        "schedule_sha256": schedule_sha256,
        "arm_checkpoint_sha256": arm_checkpoint_sha256,
        "layout_results": [asdict(item) for item in results],
    }
    _atomic_write(
        _evaluation_progress_path(progress_dir),
        _stable_json_bytes(payload),
    )


def _regime_row_from_dict(
    payload: dict[str, Any],
) -> EcologyP1RegimeDiagnosticRow:
    turn = dict(payload["turn_magnitude"])
    return EcologyP1RegimeDiagnosticRow(
        regime=str(payload["regime"]),
        capability=str(payload["capability"]),
        tier=str(payload["tier"]),
        seed=int(payload["seed"]),
        successful_bodies=int(payload["successful_bodies"]),
        required_bodies=int(payload["required_bodies"]),
        layout_success=bool(payload["layout_success"]),
        harmful_tick_rate=float(payload["harmful_tick_rate"]),
        turn_magnitude=EcologyP1TurnMagnitudeDistribution(
            sample_count=int(turn["sample_count"]),
            mean_abs=float(turn["mean_abs"]),
            median_abs=float(turn["median_abs"]),
            p90_abs=float(turn["p90_abs"]),
            max_abs=float(turn["max_abs"]),
        ),
        memory_fingerprint_stable=bool(
            payload["memory_fingerprint_stable"]
        ),
        drifted_memory_bodies=tuple(
            int(value) for value in payload["drifted_memory_bodies"]
        ),
    )


def _regime_progress_path(progress_dir: Path) -> Path:
    return progress_dir / "regime_diagnostic.json"


def _load_regime_progress(
    *,
    progress_dir: Path,
    config: EcologyP1Config,
    schedule_sha256: str,
    arm_checkpoint_sha256: dict[str, str],
) -> list[EcologyP1RegimeDiagnosticRow]:
    path = _regime_progress_path(progress_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        "config": _progress_config_payload(config),
        "schedule_sha256": schedule_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"P1 regime diagnostic progress mismatch: field={field}"
            )
    if payload.get("arm_checkpoint_sha256") != arm_checkpoint_sha256:
        # The learned checkpoint changed; rows measured against the previous
        # one describe a different policy and are not eligible for reuse.
        return []
    raw_rows = payload.get("regime_diagnostic")
    if not isinstance(raw_rows, list):
        raise ValueError(
            "P1 regime diagnostic progress rows must be a list"
        )
    return [_regime_row_from_dict(item) for item in raw_rows]


def _save_regime_progress(
    *,
    progress_dir: Path,
    config: EcologyP1Config,
    schedule_sha256: str,
    arm_checkpoint_sha256: dict[str, str],
    rows: list[EcologyP1RegimeDiagnosticRow],
) -> None:
    payload = {
        "schema_version": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        "config": _progress_config_payload(config),
        "schedule_sha256": schedule_sha256,
        "arm_checkpoint_sha256": arm_checkpoint_sha256,
        "regime_diagnostic": [asdict(item) for item in rows],
    }
    _atomic_write(
        _regime_progress_path(progress_dir),
        _stable_json_bytes(payload),
    )


def _curriculum_config(config: EcologyP1Config) -> EcologyCurriculumConfig:
    return EcologyCurriculumConfig(
        n_ants=config.n_ants,
        temporal_latent_dim=config.temporal_latent_dim,
        stage_rounds=config.training_rounds,
        stage_episodes=1,
        mastery_min_episodes=1,
        validation_rounds=config.evaluation_rounds,
        validation_seeds=(config.seed + 43,),
        heldout_rounds=config.evaluation_rounds,
        heldout_seeds=(config.seed + 101,),
        seed=config.seed,
        # P0 uses per-episode rollback to isolate mechanism failures. P1 must
        # allow a policy to traverse temporary sensitivity loss and recover;
        # the identical frozen action-chain thresholds are enforced once on
        # the final checkpoint below. Otherwise every learned episode is
        # restored to the shared initial checkpoint and no capability can form.
        action_probe_guard_enabled=False,
        # THE FAR DECISION (curriculum owner): ``_train_arm`` refuses a
        # schedule whose ``stage_rounds`` cannot reach its own milestones, and
        # the owner documents ``milestone_budget_enforced=False`` as the lever
        # for "a plan section 7 small-budget diagnostic (never for a formal
        # run)".  Bind the lever to the SAME predicate the
        # ``formal_configuration`` gate uses instead of hardcoding either
        # value: a run at the frozen formal budget keeps the milestone gate
        # fully enforced, and the only runs that turn it off are the tiny
        # ones plan section 7 step 3 requires -- which already fail
        # ``formal_configuration`` and therefore can never emit PASS or
        # unlock P2.  The flag is part of the frozen config digest, so a run
        # that turned it off says so in its own provenance.
        milestone_budget_enforced=not ecology_p1_formal_budget_failures(
            config
        ),
    )


def _fixed_schedule(config: EcologyP1Config) -> tuple[EcologyTrainingEpisodePlan, ...]:
    specs: list[
        tuple[EcologyStage, EcologyTrainingTier, bool, bool, bool, bool]
    ] = [
        (stage, EcologyTrainingTier.NEAR, forced_escape, False, False, False)
        for stage, forced_escape in (
            (EcologyStage.BUTTER, False),
            (EcologyStage.BURNING_MATCH, True),
            (EcologyStage.COMPOSITE, False),
        )
        for _ in range(config.layouts_per_tier)
    ]
    specs.extend(
        (
            EcologyStage.BUTTER,
            EcologyTrainingTier.NEAR,
            False,
            True,
            False,
            False,
        )
        for _ in range(config.layouts_per_tier)
    )
    # Food-steering pressure block: bodies spawn outside the pickup disc,
    # heading rotated away from the butter, so the only reward path is an
    # active turn toward the scent gradient. This is the counterpart of
    # forced_return (homing) for the outbound leg; without it every near
    # pickup is reachable by undirected wandering and food steering never
    # receives training pressure (measured food->turn authority stayed ~0
    # across v10-v21 while pickups looked healthy).
    specs.extend(
        (
            EcologyStage.BUTTER,
            EcologyTrainingTier.NEAR,
            False,
            False,
            True,
            False,
        )
        for _ in range(config.layouts_per_tier)
    )
    specs.extend(
        (EcologyStage.BUTTER, tier, False, False, False, False)
        for tier in (EcologyTrainingTier.MEDIUM, EcologyTrainingTier.FAR)
        for _ in range(config.layouts_per_tier)
    )
    late_primary = tuple(
        (stage, tier, forced_escape, False, False, False)
        for stage, tier, forced_escape in (
            (EcologyStage.BURNING_MATCH, EcologyTrainingTier.NEAR, True),
            (EcologyStage.COMPOSITE, EcologyTrainingTier.FAR, False),
            (EcologyStage.WOOD_STICK, EcologyTrainingTier.FAR, False),
        )
        for _ in range(config.layouts_per_tier)
    )
    # Five late return rehearsals replace the retired duplicate composite-far
    # block, preserving the 55-episode formal budget. Spacing one rehearsal
    # after every three primary layouts keeps the pickup-triggered return
    # mapping alive while heat/composite/neutral-context learning continues.
    for offset, primary in enumerate(late_primary, start=1):
        specs.append(primary)
        if offset % 3 == 0:
            specs.append(
                (
                    EcologyStage.BUTTER,
                    EcologyTrainingTier.NEAR,
                    False,
                    True,
                    False,
                    True,
                )
            )
    return tuple(
        EcologyTrainingEpisodePlan(
            stage=stage,
            tier=tier,
            seed=config.seed + 10_000 + index * 101,
            episode_index=index,
            interleaved=interleaved,
            forced_escape=forced_escape,
            forced_return=forced_return,
            forced_approach=forced_approach,
        )
        for index, (
            stage,
            tier,
            forced_escape,
            forced_return,
            forced_approach,
            interleaved,
        ) in enumerate(specs)
    )


def _evaluation_specs() -> tuple[
    tuple[str, EcologyEvaluationScenario, EcologyTrainingTier], ...
]:
    """The five held-out classes plan section 4.3 asks P1 (and P2) to grade.

    ``heat_route_foraging`` maps to ``HEAT_ROUTE_AVOIDANCE`` -- a butter source
    plus a burning match and NO wood stick (``_scene_objects``). Until v26 it
    was byte-identical to ``composite`` (COMPOSITE/FAR), which made
    ``HEAT_ROUTE_AVOIDANCE`` dead code at the P1/P2 layer, collapsed the five
    held-out classes to four and double-weighted the composite layout inside
    ``learned_not_worse_than_no_optimize`` and
    ``paired_capability_effect_positive``. P2 imports this function, so the
    same defect was inherited by the confirmatory matrix.
    """

    return (
        ("butter_medium", EcologyEvaluationScenario.BUTTER_ONLY, EcologyTrainingTier.MEDIUM),
        ("butter_far", EcologyEvaluationScenario.BUTTER_ONLY, EcologyTrainingTier.FAR),
        ("forced_escape", EcologyEvaluationScenario.HEAT_FORCED_ESCAPE, EcologyTrainingTier.NEAR),
        (
            "heat_route_foraging",
            EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE,
            EcologyTrainingTier.FAR,
        ),
        (
            "neutral_stick",
            EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK,
            EcologyTrainingTier.FAR,
        ),
        ("composite", EcologyEvaluationScenario.COMPOSITE, EcologyTrainingTier.FAR),
    )


def _scenario_stage(scenario: EcologyEvaluationScenario) -> EcologyStage:
    return {
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


def _required_bodies(config: EcologyP1Config) -> int:
    return max(1, math.ceil(config.n_ants * config.body_success_ratio))


def _median_or_none(values: tuple[int, ...]) -> float | None:
    return float(statistics.median(values)) if values else None


def _p90_or_none(values: tuple[int, ...]) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), 90.0))


def _escape_latency_summary(
    *, source: str, latencies: tuple[int, ...]
) -> EcologyP1EscapeLatencySummary:
    return EcologyP1EscapeLatencySummary(
        source=source,
        sample_count=len(latencies),
        median=_median_or_none(latencies),
        p90=_p90_or_none(latencies),
    )


def _layout_result(
    *, config: EcologyP1Config, capability: str, metrics: EcologyArmMetrics
) -> EcologyP1LayoutResult:
    required = _required_bodies(config)
    if capability == "forced_escape":
        successful = sum(item.heat_escapes > 0 for item in metrics.body_lineage)
    else:
        successful = sum(
            item.picked_up and item.delivered for item in metrics.body_lineage
        )
    total_ticks = sum(item.total_ticks for item in metrics.body_lineage)
    harmful_ticks = sum(item.harmful_heat_ticks for item in metrics.body_lineage)
    harmful_rate = harmful_ticks / total_ticks if total_ticks else 0.0
    safe = capability not in {"heat_route_foraging", "composite"} or (
        harmful_rate <= config.harmful_tick_rate_max
    )
    return EcologyP1LayoutResult(
        arm=metrics.arm,
        capability=capability,
        seed=metrics.seed,
        tier=metrics.body_lineage[0].tier.value,
        successful_bodies=successful,
        required_bodies=required,
        layout_success=successful >= required and safe,
        harmful_tick_rate=harmful_rate,
        escape_latencies=tuple(
            latency
            for item in metrics.body_lineage
            for latency in item.escape_latencies
        ),
        switch_count=sum(item.switch_count for item in metrics.body_lineage),
        non_timeout_segment_closures=sum(
            item.non_timeout_segment_closures for item in metrics.body_lineage
        ),
        policy_fingerprint_stable=metrics.policy_fingerprint_stable,
        temporal_learning_fingerprint_stable=(
            metrics.temporal_learning_fingerprint_stable
        ),
        replay_settlement_coverage=metrics.replay_settlement_coverage,
        replay_lineage_coverage=metrics.replay_lineage_coverage,
        replay_drop_count=metrics.replay_drop_count,
    )


def _success_count(
    results: tuple[EcologyP1LayoutResult, ...], arm: str, capability: str
) -> int:
    return sum(
        item.layout_success
        for item in results
        if item.arm == arm and item.capability == capability
    )


def _temporal_non_timeout_closure_gate(
    learned_results: tuple[EcologyP1LayoutResult, ...],
) -> EcologyP1Gate:
    """Plan 4.4: held-out segments must not "all" be closed by timeout.

    Graded PER LAYOUT, not aggregated. The aggregated form
    (``sum(switch_count) > 0 and sum(non_timeout_segment_closures) > 0``) was
    satisfied by a SINGLE non-timeout closure anywhere in the whole held-out
    matrix, which is exactly the "全部由 timeout 关闭" state the plan forbids,
    only stated over a sum.

    A layout QUALIFIES when it individually shows both a real beta switch and
    a non-timeout segment closure. The gate passes when the complementary
    timeout-only rate stays at or below
    ``ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX`` (0.4), i.e. when at least 60%
    of learned held-out layouts qualify on their own. An empty learned matrix
    FAILS: there is no evidence, and a vacuous pass is the failure mode this
    replaces.
    """

    total = len(learned_results)
    qualifying = tuple(
        item
        for item in learned_results
        if item.switch_count > 0 and item.non_timeout_segment_closures > 0
    )
    timeout_only = total - len(qualifying)
    # 1.0 when there is nothing to grade, so the empty matrix fails the
    # ceiling rather than dividing by zero or passing vacuously.
    timeout_only_rate = timeout_only / total if total else 1.0
    return EcologyP1Gate(
        name="temporal_non_timeout_closure",
        # The 1e-12 is float-representation slack for exact boundaries such as
        # 12/30 vs 1.0-0.6; it is not threshold slack.
        passed=(
            total > 0
            and timeout_only_rate
            <= ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX + 1e-12
        ),
        observed=(
            f"layouts={total}, "
            f"layouts_with_switch_and_non_timeout_closure={len(qualifying)}, "
            f"timeout_only_layouts={timeout_only}, "
            f"timeout_only_rate={timeout_only_rate}, "
            f"switches={sum(item.switch_count for item in learned_results)}, "
            "non_timeout_closures="
            + str(
                sum(
                    item.non_timeout_segment_closures
                    for item in learned_results
                )
            )
        ),
        threshold=(
            "per learned held-out layout: a real beta switch AND a "
            "non-timeout segment closure; timeout-only layout rate <= "
            f"{ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX}"
        ),
    )


class _EcologyOracleAnt:
    """Geometry-reading diagnostic; never part of a learning comparison."""

    def __init__(self, world, *, body_id: int) -> None:
        self.world = world
        self.body_id = body_id
        objects = world.world_objects()
        butter = next(item for item in objects if isinstance(item, ButterSource))
        self.food = (butter.x, butter.y)
        self.outbound = self._safe_waypoints(objects)
        self.waypoint_index = 0
        self.return_index: int | None = None

    def _safe_waypoints(self, objects) -> tuple[tuple[float, float], ...]:
        waypoints: list[tuple[float, float]] = []
        matches = tuple(
            item for item in objects if isinstance(item, BurningMatch)
        )
        for stick in (
            item for item in objects if isinstance(item, WoodStick)
        ):
            centre = (
                (stick.start_x + stick.end_x) / 2.0,
                (stick.start_y + stick.end_y) / 2.0,
            )
            endpoints = (
                (stick.start_x, stick.start_y),
                (stick.end_x, stick.end_y),
            )
            endpoint = max(
                endpoints,
                key=lambda point: min(
                    (
                        math.hypot(point[0] - match.x, point[1] - match.y)
                        for match in matches
                    ),
                    default=0.0,
                ),
            )
            dx = endpoint[0] - centre[0]
            dy = endpoint[1] - centre[1]
            # Leave enough clearance that the 0.55 waypoint acceptance radius
            # cannot cut the next segment back through the capsule endpoint.
            scale = 1.6 / max(math.hypot(dx, dy), 1e-9)
            waypoints.append(
                (endpoint[0] + dx * scale, endpoint[1] + dy * scale)
            )
        for match in matches:
            route_dx = self.food[0]
            route_dy = self.food[1]
            route_norm = max(math.hypot(route_dx, route_dy), 1e-9)
            perpendicular = (-route_dy / route_norm, route_dx / route_norm)
            candidates = (
                (
                    match.x + perpendicular[0] * (match.harm_radius + 0.8),
                    match.y + perpendicular[1] * (match.harm_radius + 0.8),
                ),
                (
                    match.x - perpendicular[0] * (match.harm_radius + 0.8),
                    match.y - perpendicular[1] * (match.harm_radius + 0.8),
                ),
            )
            candidate = min(
                candidates,
                key=lambda point: sum(
                    math.hypot(point[0] - waypoint[0], point[1] - waypoint[1])
                    for waypoint in waypoints
                ),
            )
            if not waypoints:
                waypoints.append(candidate)
        waypoints.sort(key=lambda point: math.hypot(*point))
        waypoints.append(self.food)
        return tuple(waypoints)

    def step(self) -> None:
        body = self.world.body(self.body_id)
        if body.carrying_food:
            if self.return_index is None:
                self.return_index = max(0, len(self.outbound) - 2)
            target = (
                self.outbound[self.return_index]
                if self.return_index >= 0
                else self.world.nest
            )
            if math.hypot(body.x - target[0], body.y - target[1]) < 0.55:
                self.return_index -= 1
                target = (
                    self.outbound[self.return_index]
                    if self.return_index >= 0
                    else self.world.nest
                )
        else:
            if self.return_index is not None:
                self.waypoint_index = 0
                self.return_index = None
            target = self.outbound[self.waypoint_index]
            if math.hypot(body.x - target[0], body.y - target[1]) < 0.55:
                self.waypoint_index = min(
                    self.waypoint_index + 1,
                    len(self.outbound) - 1,
                )
                target = self.outbound[self.waypoint_index]
        desired = math.atan2(target[1] - body.y, target[0] - body.x)
        relative = (desired - body.heading + math.pi) % (2.0 * math.pi) - math.pi
        turn = float(
            np.clip(
                relative,
                -self.world.config.max_turn_rate,
                self.world.config.max_turn_rate,
            )
        )
        step_command = (
            0.0
            if abs(relative) > self.world.config.max_turn_rate * 1.25
            else self.world.config.step_size
        )
        self.world.act(
            turn_command=turn,
            step_command=step_command,
            body_id=self.body_id,
        )


def _run_diagnostic_layout(
    *,
    config: EcologyP1Config,
    curriculum: EcologyCurriculumConfig,
    controller: str,
    capability: str,
    scenario: EcologyEvaluationScenario,
    tier: EcologyTrainingTier,
    seed: int,
) -> EcologyP1DiagnosticResult:
    world = _world(
        config=curriculum,
        stage=_scenario_stage(scenario),
        seed=seed,
        data_split=EcologyDataSplit.HELDOUT,
        tier=tier,
        forced_escape=(
            scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
        ),
    )
    if controller == "fixed_rule":
        ants = tuple(
            FixedRuleAnt(
                world,
                config=FixedRuleConfig(seed=seed * 100 + body_id),
                body_id=body_id,
            )
            for body_id in range(config.n_ants)
        )
    elif controller == "random":
        ants = tuple(
            RandomAnt(world, seed=seed * 100 + body_id, body_id=body_id)
            for body_id in range(config.n_ants)
        )
    elif controller == "oracle_steering":
        ants = tuple(
            _EcologyOracleAnt(world, body_id=body_id)
            for body_id in range(config.n_ants)
        )
    else:
        raise ValueError(f"unsupported P1 diagnostic controller: {controller}")
    picked = [False] * config.n_ants
    delivered = [False] * config.n_ants
    escaped = [False] * config.n_ants
    escape_latencies: list[int] = []
    harmful_ticks = 0
    for round_index in range(config.evaluation_rounds):
        for body_id, ant in enumerate(ants):
            ant.step()
            transition = world.last_transition(body_id)
            picked[body_id] = picked[body_id] or transition.picked_up
            delivered[body_id] = delivered[body_id] or transition.delivered
            first_escape = (
                transition.escaped_harmful_heat and not escaped[body_id]
            )
            escaped[body_id] = escaped[body_id] or first_escape
            if first_escape:
                escape_latencies.append(round_index + 1)
            harmful_ticks += int(transition.heat_harmful_after)
    required = _required_bodies(config)
    successful = (
        sum(escaped)
        if capability == "forced_escape"
        else sum(
            did_pickup and did_deliver
            for did_pickup, did_deliver in zip(picked, delivered, strict=True)
        )
    )
    return EcologyP1DiagnosticResult(
        controller=controller,
        capability=capability,
        seed=seed,
        tier=tier.value,
        successful_bodies=successful,
        required_bodies=required,
        layout_success=successful >= required,
        pickups=sum(picked),
        deliveries=sum(delivered),
        heat_escapes=sum(escaped),
        escape_latencies=tuple(escape_latencies),
        harmful_heat_ticks=harmful_ticks,
    )


def run_ecology_p1_diagnostics(
    config: EcologyP1Config,
) -> EcologyP1DiagnosticReport:
    """Run cheap environment/controller diagnostics without any training."""

    curriculum = _curriculum_config(config)
    results = tuple(
        _run_diagnostic_layout(
            config=config,
            curriculum=curriculum,
            controller=controller,
            capability=capability,
            scenario=scenario,
            tier=tier,
            seed=(
                config.seed
                + 2_000_003
                + capability_index * 10_007
                + index * 103
            ),
        )
        for controller in ("oracle_steering", "fixed_rule", "random")
        for capability_index, (capability, scenario, tier) in enumerate(
            _evaluation_specs()
        )
        for index in range(config.layouts_per_tier)
    )
    required_layouts = math.ceil(
        config.layouts_per_tier * config.layout_success_ratio
    )
    oracle_success = tuple(
        (
            capability,
            sum(
                item.layout_success
                for item in results
                if item.controller == "oracle_steering"
                and item.capability == capability
            ),
        )
        for capability, _, _ in _evaluation_specs()
    )
    return EcologyP1DiagnosticReport(
        schema_version=ECOLOGY_P1_DIAGNOSTICS_SCHEMA_VERSION,
        config=config,
        results=results,
        oracle_success_by_capability=oracle_success,
        required_layouts=required_layouts,
        passed=all(count >= required_layouts for _, count in oracle_success),
    )


# ---------------------------------------------------------------------------
# Plan section 4.7 conjunct: checkpoint roundtrip
# ---------------------------------------------------------------------------


def _archive_state_fingerprints(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[tuple[str, str, str], ...]:
    """Full owner identity an ARCHIVE must reproduce byte-for-byte.

    An export/restore roundtrip performs no rollout, so every published owner
    fingerprint -- policy, temporal AND memory -- must come back identical. An
    archive that silently drops memory is a broken archive. This is a
    different question from "did a frozen rollout keep the learned owners
    frozen"; see ``_frozen_learned_fingerprints``.
    """

    return tuple(
        (
            item.policy_fingerprint,
            item.temporal_fingerprint,
            item.memory_fingerprint,
        )
        for item in checkpoints
    )


def _frozen_learned_fingerprints(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[tuple[str, str], ...]:
    """The owner set a frozen rollout must not move.

    This is the module's established frozen-owner notion: the same
    ``policy_fingerprint`` + ``temporal_learning_fingerprint`` pair the
    ``frozen_evaluation`` gate grades and ``ecology_mechanism_audit`` asserts.
    ``temporal_fingerprint`` is deliberately excluded --
    docs/specs/digital-ant-embodiment.md excludes the PE-driven turn-local
    mixture from the temporal-LEARNING fingerprint, so the full temporal
    fingerprint moving during a frozen rollout is EXPECTED inference
    telemetry, not a freeze violation. Comparing it would make a read-only
    diagnostic raise on correct behaviour.
    """

    return tuple(
        (item.policy_fingerprint, item.temporal_learning_fingerprint)
        for item in checkpoints
    )


def _drifted_memory_bodies(
    *,
    before: tuple[AntLearningCheckpoint, ...],
    after: tuple[AntLearningCheckpoint, ...],
) -> tuple[int, ...]:
    """Colony indices whose ``memory_fingerprint`` moved between two exports."""

    return tuple(
        index
        for index, (start, end) in enumerate(
            zip(before, after, strict=True)
        )
        if start.memory_fingerprint != end.memory_fingerprint
    )


def _verify_p1_checkpoint_archives(
    *,
    config: EcologyP1Config,
    curriculum: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[bool, str]:
    """Hydrate the learned colony from its own archives, then corrupt one.

    Plan section 4.7 requires "checkpoint roundtrip 与 replay lineage 继续通过"
    as a P1 conjunct; only replay lineage was gated. This runs entirely through
    the archive API the runtime owner publishes
    (``export_learning_checkpoint_archives`` /
    ``restore_learning_checkpoint_archives`` and ``AgentLearningArchiveError``)
    -- no curriculum internals -- so P1 states the property itself instead of
    inheriting an unrelated report's claim.
    """

    def fresh_runner(*, offset: int, session_id: str) -> KernelColonyRunner:
        return KernelColonyRunner(
            _world(
                config=curriculum,
                stage=EcologyStage.COMPOSITE,
                seed=config.seed + offset,
                data_split=EcologyDataSplit.TRAIN,
                tier=EcologyTrainingTier.FAR,
            ),
            base_config=_session_config(
                config=curriculum,
                seed=config.seed + offset,
                session_id=session_id,
                optimize=False,
                learning_enabled=False,
                sparse_exploration_enabled=False,
            ),
        )

    source = fresh_runner(
        offset=900_001,
        session_id="ecology:p1:archive-roundtrip:source",
    )
    source.restore_learning_checkpoints(checkpoints)
    archives = source.export_learning_checkpoint_archives(
        checkpoint_prefix="ecology:p1:archive-roundtrip"
    )
    verifier = fresh_runner(
        offset=900_002,
        session_id="ecology:p1:archive-roundtrip:verifier",
    )
    verifier.restore_learning_checkpoint_archives(archives)
    restored = verifier.export_learning_checkpoints(
        checkpoint_prefix="ecology:p1:archive-roundtrip:restored",
        include_runtime_replay=False,
    )
    if _archive_state_fingerprints(restored) != _archive_state_fingerprints(
        checkpoints
    ):
        return (
            False,
            "fresh-session hydration changed policy/temporal/memory "
            "fingerprints",
        )
    pre_failure = verifier.export_learning_checkpoints(
        checkpoint_prefix="ecology:p1:archive-roundtrip:pre-failure",
        include_runtime_replay=False,
    )
    corrupted = list(archives)
    corrupted[-1] = corrupted[-1][:-1] + b"!"
    try:
        verifier.restore_learning_checkpoint_archives(tuple(corrupted))
    except AgentLearningArchiveError:
        pass
    else:
        return (False, "a corrupted archive collection was accepted")
    post_failure = verifier.export_learning_checkpoints(
        checkpoint_prefix="ecology:p1:archive-roundtrip:post-failure",
        include_runtime_replay=False,
    )
    if _archive_state_fingerprints(
        post_failure
    ) != _archive_state_fingerprints(pre_failure):
        return (
            False,
            "rejected archive restore did not roll back atomically",
        )
    return (True, "hydration verified and corrupt restore rolled back")


# ---------------------------------------------------------------------------
# Read-only regime diagnostic (never a gate, never fed back into learning)
# ---------------------------------------------------------------------------


def _turn_magnitude_distribution(
    magnitudes: tuple[float, ...],
) -> EcologyP1TurnMagnitudeDistribution:
    if not magnitudes:
        return EcologyP1TurnMagnitudeDistribution(
            sample_count=0,
            mean_abs=0.0,
            median_abs=0.0,
            p90_abs=0.0,
            max_abs=0.0,
        )
    values = np.asarray(magnitudes, dtype=float)
    return EcologyP1TurnMagnitudeDistribution(
        sample_count=int(values.size),
        mean_abs=float(values.mean()),
        median_abs=float(np.median(values)),
        p90_abs=float(np.percentile(values, 90.0)),
        max_abs=float(values.max()),
    )


async def _run_regime_layout(
    *,
    config: EcologyP1Config,
    curriculum: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    regime: str,
    capability: str,
    scenario: EcologyEvaluationScenario,
    tier: EcologyTrainingTier,
    seed: int,
) -> EcologyP1RegimeDiagnosticRow:
    """Replay ONE held-out layout with ONE action-sampling regime.

    ``ECOLOGY_P1_REGIME_DETERMINISTIC`` is exactly what the capability gates
    grade (deterministic mean action, exclusive steering).
    ``ECOLOGY_P1_REGIME_STOCHASTIC`` is the sparse-exploration regime the
    policy was OPTIMIZED in: a hash-fixed residual held for 8 ticks, i.e.
    piecewise constant-curvature arcs whose per-tick turn reaches ~0.5 rad.
    Everything else -- checkpoint, world, seed, tier, budget -- is identical,
    so the published |turn| distributions isolate the kinematic regime gap that
    is the leading candidate explanation for "healthy training pickups, 0/5
    frozen medium/far layouts" across v10-v25.
    """

    if regime not in ECOLOGY_P1_REGIME_NAMES:
        raise ValueError(f"unsupported P1 regime: {regime!r}")
    forced_escape = (
        scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
    )
    runner = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=_scenario_stage(scenario),
            seed=seed,
            data_split=EcologyDataSplit.HELDOUT,
            tier=tier,
            forced_escape=forced_escape,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=seed,
            session_id=(
                f"ecology:p1:regime:{regime}:{capability}:{seed}"
            ),
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=(
                regime == ECOLOGY_P1_REGIME_STOCHASTIC
            ),
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    await runner.run(curriculum.heldout_rounds)
    after = runner.export_learning_checkpoints(
        checkpoint_prefix=(
            f"ecology:p1:regime:{regime}:{capability}:{seed}:frozen-check"
        ),
        include_runtime_replay=False,
    )
    # The diagnostic must never feed back into LEARNING. Learning is off by
    # construction; this proves it instead of assuming it, against the same
    # frozen-learned owner set the ``frozen_evaluation`` gate grades.
    if _frozen_learned_fingerprints(after) != _frozen_learned_fingerprints(
        checkpoints
    ):
        raise RuntimeError(
            "the P1 regime diagnostic mutated learned owner state "
            f"(regime={regime}, capability={capability}, seed={seed}); it is "
            "read-only by contract"
        )
    # Memory drift under learning_enabled=False is a real defect, but it is
    # the P0 ``frozen_evaluation`` gate's BLOCK to own. Publish it as evidence
    # on this row rather than crash a diagnostic that exists to measure
    # something else.
    drifted_memory = _drifted_memory_bodies(
        before=checkpoints, after=after
    )
    records = tuple(
        step
        for round_record in runner.rounds
        for step in round_record.ant_steps
    )
    picked = [False] * config.n_ants
    delivered = [False] * config.n_ants
    escaped = [False] * config.n_ants
    harmful_ticks = 0
    for record in records:
        body_id = record.body_id
        picked[body_id] = picked[body_id] or record.picked_up
        delivered[body_id] = delivered[body_id] or record.delivered
        escaped[body_id] = escaped[body_id] or record.escaped_harmful_heat
        harmful_ticks += int(record.heat_harmful)
    successful = (
        sum(escaped)
        if capability == "forced_escape"
        else sum(
            did_pickup and did_deliver
            for did_pickup, did_deliver in zip(
                picked, delivered, strict=True
            )
        )
    )
    required = _required_bodies(config)
    return EcologyP1RegimeDiagnosticRow(
        regime=regime,
        capability=capability,
        tier=tier.value,
        seed=seed,
        successful_bodies=successful,
        required_bodies=required,
        layout_success=successful >= required,
        harmful_tick_rate=(
            harmful_ticks / len(records) if records else 0.0
        ),
        turn_magnitude=_turn_magnitude_distribution(
            tuple(
                abs(record.command.turn_command) for record in records
            )
        ),
        memory_fingerprint_stable=not drifted_memory,
        drifted_memory_bodies=drifted_memory,
    )


def _regime_gap_summary(
    rows: tuple[EcologyP1RegimeDiagnosticRow, ...],
) -> tuple[EcologyP1RegimeGapSummary, ...]:
    summaries: list[EcologyP1RegimeGapSummary] = []
    for capability, _, _ in _evaluation_specs():
        by_regime = {
            regime: tuple(
                row
                for row in rows
                if row.capability == capability and row.regime == regime
            )
            for regime in ECOLOGY_P1_REGIME_NAMES
        }
        deterministic = by_regime[ECOLOGY_P1_REGIME_DETERMINISTIC]
        stochastic = by_regime[ECOLOGY_P1_REGIME_STOCHASTIC]
        if len(deterministic) != len(stochastic):
            raise RuntimeError(
                "regime diagnostic lanes are not matched for capability "
                f"{capability}: deterministic={len(deterministic)}, "
                f"stochastic={len(stochastic)}"
            )
        deterministic_median = (
            float(
                np.median(
                    [row.turn_magnitude.median_abs for row in deterministic]
                )
            )
            if deterministic
            else 0.0
        )
        stochastic_median = (
            float(
                np.median(
                    [row.turn_magnitude.median_abs for row in stochastic]
                )
            )
            if stochastic
            else 0.0
        )
        summaries.append(
            EcologyP1RegimeGapSummary(
                capability=capability,
                deterministic_successful_layouts=sum(
                    row.layout_success for row in deterministic
                ),
                stochastic_successful_layouts=sum(
                    row.layout_success for row in stochastic
                ),
                layouts=len(deterministic),
                deterministic_median_abs_turn=deterministic_median,
                stochastic_median_abs_turn=stochastic_median,
                median_abs_turn_ratio=(
                    stochastic_median / deterministic_median
                    if deterministic_median > 0.0
                    else None
                ),
            )
        )
    return tuple(summaries)


# ---------------------------------------------------------------------------
# Plan section 4.7 conjunct: an independent repetition in the same direction
# ---------------------------------------------------------------------------


def _direction_signature(
    results: tuple[EcologyP1LayoutResult, ...],
) -> tuple[tuple[str, int], ...]:
    """Sign of every learned-vs-control contrast the P1 verdict rests on.

    A repetition "in the same direction" is machine-checkable exactly when the
    direction is a finite, ordered vector of signs rather than prose: one sign
    per capability for learned-vs-no-optimize, plus the aggregate paired effect
    against the better of no-optimize and cold.
    """

    def sign(value: int) -> int:
        return (value > 0) - (value < 0)

    signature: list[tuple[str, int]] = []
    learned_total = 0
    no_optimize_total = 0
    cold_total = 0
    for capability, _, _ in _evaluation_specs():
        learned = _success_count(results, "learned", capability)
        no_optimize = _success_count(results, "no_optimize", capability)
        cold = _success_count(results, "cold", capability)
        learned_total += learned
        no_optimize_total += no_optimize
        cold_total += cold
        signature.append((capability, sign(learned - no_optimize)))
    signature.append(
        (
            "aggregate_paired_effect",
            sign(learned_total - max(no_optimize_total, cold_total)),
        )
    )
    return tuple(signature)


def _repeat_run_same_direction_gate(
    *,
    reference: EcologyP1RepeatReference | None,
    results: tuple[EcologyP1LayoutResult, ...],
) -> EcologyP1Gate:
    """Plan 4.7: "P1 重跑一次能够得到同方向结果".

    A run with NO reference report FAILS. It is not skipped and it does not
    pass vacuously: a single training run is exactly the single-run accident
    the conjunct exists to rule out, so "we did not repeat it" is a negative
    result, not an absent one. Supply ``--repeat-reference-report`` pointing
    at a previous P1 report produced with a DIFFERENT training seed.
    """

    signature = _direction_signature(results)
    if reference is None:
        return EcologyP1Gate(
            name="repeat_run_same_direction",
            passed=False,
            observed=(
                "no repeat reference report supplied; this run is a single "
                f"repetition. direction={signature!r}"
            ),
            threshold=(
                "an independent P1 report (different training seed, same "
                "budget) whose per-capability learned-vs-no-optimize signs "
                "and aggregate paired-effect sign match this run"
            ),
        )
    mismatched = tuple(
        name
        for (name, sign), (reference_name, reference_sign) in zip(
            signature, reference.direction_signature, strict=True
        )
        if name != reference_name or sign != reference_sign
    )
    return EcologyP1Gate(
        name="repeat_run_same_direction",
        passed=not mismatched,
        observed=(
            f"reference_seed={reference.seed}, "
            f"reference_verdict={reference.verdict}, "
            f"reference_sha256={reference.report_sha256}, "
            f"direction={signature!r}, "
            f"reference_direction={reference.direction_signature!r}, "
            f"mismatched={mismatched!r}"
        ),
        threshold=(
            "an independent P1 report (different training seed, same budget) "
            "whose per-capability learned-vs-no-optimize signs and aggregate "
            "paired-effect sign match this run"
        ),
    )


def load_p1_repeat_reference(
    report_path: Path,
    *,
    repo_root: Path | None = None,
) -> EcologyP1RepeatReference:
    """Load a previous P1 report as the independent repetition, or fail loudly.

    Rejects -- never silently reinterprets -- a report from a different schema
    version, a different gate set, or an incomparable configuration. A report
    produced with the SAME seed is refused too: replaying one training seed is
    not an independent repetition and cannot satisfy plan section 4.7.
    """

    resolved = (
        report_path
        if report_path.is_absolute() or repo_root is None
        else repo_root / report_path
    )
    if not resolved.exists():
        raise ValueError(f"P1 repeat reference not found: {resolved}")
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P1 repeat reference must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != ECOLOGY_P1_SCHEMA_VERSION:
        raise ValueError(
            "P1 repeat reference schema mismatch: "
            f"expected={ECOLOGY_P1_SCHEMA_VERSION!r}, "
            f"actual={schema_version!r}. Reports written before "
            f"{ECOLOGY_P1_SCHEMA_VERSION} grade a different held-out matrix "
            "and cannot be compared with this run."
        )
    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, list) or not all(
        isinstance(gate, dict) for gate in raw_gates
    ):
        raise ValueError(
            "P1 repeat reference gates must be structured objects"
        )
    gate_names = tuple(str(gate.get("name", "")) for gate in raw_gates)
    if gate_names != ECOLOGY_P1_GATE_NAMES:
        raise ValueError(
            f"P1 repeat reference gate set mismatch: {gate_names!r}"
        )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("P1 repeat reference config must be an object")
    raw_results = payload.get("layout_results")
    if not isinstance(raw_results, list):
        raise ValueError(
            "P1 repeat reference layout_results must be a list"
        )
    return EcologyP1RepeatReference(
        report_path=str(report_path),
        report_sha256=_sha256(raw),
        schema_version=str(schema_version),
        seed=int(raw_config["seed"]),
        verdict=str(payload.get("verdict", "BLOCK")).upper(),
        direction_signature=_direction_signature(
            tuple(
                _layout_result_from_dict(item) for item in raw_results
            )
        ),
    )


def _assert_repeat_reference_comparable(
    *,
    config: EcologyP1Config,
    reference_path: Path,
    repo_root: Path | None,
) -> EcologyP1RepeatReference:
    reference = load_p1_repeat_reference(
        reference_path, repo_root=repo_root
    )
    resolved = (
        reference_path
        if reference_path.is_absolute() or repo_root is None
        else repo_root / reference_path
    )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    reference_config = dict(payload["config"])
    expected = _progress_config_payload(config)
    mismatched = tuple(
        field
        for field, value in expected.items()
        if field != "seed" and reference_config.get(field) != value
    )
    if mismatched:
        raise ValueError(
            "P1 repeat reference was produced under a different budget and "
            f"cannot be compared: fields={mismatched}"
        )
    if reference.seed == config.seed:
        raise ValueError(
            "P1 repeat reference uses the same training seed "
            f"({config.seed}); plan section 4.7 requires an INDEPENDENT "
            "repetition"
        )
    return reference


async def run_ecology_p1(
    config: EcologyP1Config,
    *,
    progress_dir: Path | None = None,
    max_new_work_items: int | None = None,
    repeat_reference_report: Path | None = None,
    repo_root: Path | None = None,
) -> EcologyP1Report:
    if max_new_work_items is not None and max_new_work_items < 1:
        raise ValueError("max_new_work_items must be positive")
    if max_new_work_items is not None and progress_dir is None:
        raise ValueError(
            "max_new_work_items requires a resumable progress_dir"
        )
    # Validate the repetition reference BEFORE spending the run's budget: an
    # incomparable reference is a configuration error, not a gate failure.
    repeat_reference = (
        _assert_repeat_reference_comparable(
            config=config,
            reference_path=repeat_reference_report,
            repo_root=repo_root,
        )
        if repeat_reference_report is not None
        else None
    )
    completed_work_items = 0
    curriculum = _curriculum_config(config)
    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=config.seed,
            session_id="ecology:p1:shared-initial",
            optimize=True,
        ),
    )
    initial = bootstrap.export_learning_checkpoints(
        checkpoint_prefix="ecology:p1:shared-initial",
        include_runtime_replay=False,
    )
    schedule = _fixed_schedule(config)
    schedule_sha256 = _schedule_digest(schedule)
    resolved_progress = (
        progress_dir.resolve()
        if progress_dir is not None
        else None
    )
    if resolved_progress is not None:
        resolved_progress.mkdir(parents=True, exist_ok=True)
        cold_state = _load_arm_progress(
            progress_dir=resolved_progress,
            arm="cold",
            config=config,
            schedule_sha256=schedule_sha256,
        )
        if cold_state is None:
            _save_arm_progress(
                progress_dir=resolved_progress,
                arm="cold",
                config=config,
                schedule_sha256=schedule_sha256,
                completed_training_episodes=0,
                runner=bootstrap,
                training_complete=True,
            )
        else:
            initial = _hydrate_progress_checkpoints(
                config=config,
                curriculum=curriculum,
                archives=_read_progress_archive(
                    progress_dir=resolved_progress,
                    state=cold_state,
                    config=config,
                ),
                arm="cold",
            )
    arms: dict[str, tuple[AntLearningCheckpoint, ...]] = {
        "cold": initial
    }
    for arm, optimize, shaping, segment in (
        ("learned", True, True, True),
        ("no_optimize", False, True, True),
        ("dense_local_shaping_off", True, False, True),
        ("segment_credit_off", True, True, False),
    ):
        completed = 0
        arm_initial = initial
        if resolved_progress is not None:
            state = _load_arm_progress(
                progress_dir=resolved_progress,
                arm=arm,
                config=config,
                schedule_sha256=schedule_sha256,
            )
            if state is not None:
                completed = int(
                    state["completed_training_episodes"]
                )
                if completed > len(schedule):
                    raise ValueError(
                        f"P1 progress for {arm} exceeds schedule length"
                    )
                if bool(state.get("training_complete")) != (
                    completed == len(schedule)
                ):
                    raise ValueError(
                        f"P1 progress completion flag mismatch for {arm}"
                    )
                arm_initial = _hydrate_progress_checkpoints(
                    config=config,
                    curriculum=curriculum,
                    archives=_read_progress_archive(
                        progress_dir=resolved_progress,
                        state=state,
                        config=config,
                    ),
                    arm=arm,
                )

        def save_episode(
            schedule_index: int,
            runner: KernelColonyRunner,
            _checkpoints: tuple[AntLearningCheckpoint, ...],
            _report: EcologyTrainingEpisodeReport,
            *,
            active_arm: str = arm,
        ) -> None:
            nonlocal completed_work_items
            if resolved_progress is None:
                return
            completed_count = schedule_index + 1
            _save_arm_progress(
                progress_dir=resolved_progress,
                arm=active_arm,
                config=config,
                schedule_sha256=schedule_sha256,
                completed_training_episodes=completed_count,
                runner=runner,
                training_complete=(
                    completed_count == len(schedule)
                ),
                last_episode_report=_report,
            )
            completed_work_items += 1
            if (
                max_new_work_items is not None
                and completed_work_items >= max_new_work_items
            ):
                raise EcologyP1ProgressPaused(
                    completed_work_items=completed_work_items
                )

        if completed == len(schedule):
            checkpoints = arm_initial
        else:
            checkpoints, _, _, _, _ = await _train_arm(
                config=curriculum,
                initial=arm_initial,
                arm=arm,
                optimize=optimize,
                local_valence_enabled=shaping,
                segment_credit_enabled=segment,
                schedule=schedule,
                schedule_start_index=completed,
                episode_callback=(
                    save_episode
                    if resolved_progress is not None
                    else None
                ),
            )
        arms[arm] = checkpoints
    arm_checkpoint_sha256: dict[str, str] = {}
    if resolved_progress is not None:
        for arm in ECOLOGY_P1_ARM_NAMES:
            state = _load_arm_progress(
                progress_dir=resolved_progress,
                arm=arm,
                config=config,
                schedule_sha256=schedule_sha256,
            )
            if state is None or not state.get("training_complete"):
                raise RuntimeError(
                    f"P1 arm {arm} is not complete before evaluation"
                )
            arm_checkpoint_sha256[arm] = str(
                state["checkpoint_sha256"]
            )
    results = (
        _load_evaluation_progress(
            progress_dir=resolved_progress,
            config=config,
            schedule_sha256=schedule_sha256,
            arm_checkpoint_sha256=arm_checkpoint_sha256,
        )
        if resolved_progress is not None
        else []
    )
    existing_result_keys = {
        (item.arm, item.capability, item.seed)
        for item in results
    }
    if len(existing_result_keys) != len(results):
        raise ValueError("P1 evaluation progress contains duplicates")
    for arm in ECOLOGY_P1_ARM_NAMES:
        for capability_index, (capability, scenario, tier) in enumerate(
            _evaluation_specs()
        ):
            for index in range(config.layouts_per_tier):
                evaluation_seed = (
                    config.seed
                    + 2_000_003
                    + capability_index * 10_007
                    + index * 103
                )
                result_key = (arm, capability, evaluation_seed)
                if result_key in existing_result_keys:
                    continue
                metrics = await _evaluate_arm(
                    config=curriculum,
                    checkpoints=arms[arm],
                    arm=arm,
                    data_split=EcologyDataSplit.HELDOUT,
                    scenario=scenario,
                    seed=evaluation_seed,
                    tier=tier,
                )
                results.append(
                    _layout_result(
                        config=config,
                        capability=capability,
                        metrics=metrics,
                    )
                )
                existing_result_keys.add(result_key)
                if resolved_progress is not None:
                    _save_evaluation_progress(
                        progress_dir=resolved_progress,
                        config=config,
                        schedule_sha256=schedule_sha256,
                        arm_checkpoint_sha256=(
                            arm_checkpoint_sha256
                        ),
                        results=results,
                    )
                    completed_work_items += 1
                    if (
                        max_new_work_items is not None
                        and completed_work_items
                        >= max_new_work_items
                    ):
                        raise EcologyP1ProgressPaused(
                            completed_work_items=(
                                completed_work_items
                            )
                        )
    arm_order = {
        arm: index
        for index, arm in enumerate(ECOLOGY_P1_ARM_NAMES)
    }
    capability_order = {
        capability: index
        for index, (capability, _, _) in enumerate(
            _evaluation_specs()
        )
    }
    result_tuple = tuple(
        sorted(
            results,
            key=lambda item: (
                arm_order[item.arm],
                capability_order[item.capability],
                item.seed,
            ),
        )
    )
    # --- read-only regime diagnostic (published, never a gate) -------------
    regime_rows = (
        _load_regime_progress(
            progress_dir=resolved_progress,
            config=config,
            schedule_sha256=schedule_sha256,
            arm_checkpoint_sha256=arm_checkpoint_sha256,
        )
        if resolved_progress is not None
        else []
    )
    existing_regime_keys = {
        (item.regime, item.capability, item.seed) for item in regime_rows
    }
    if len(existing_regime_keys) != len(regime_rows):
        raise ValueError(
            "P1 regime diagnostic progress contains duplicates"
        )
    for regime in ECOLOGY_P1_REGIME_NAMES:
        for capability_index, (capability, scenario, tier) in enumerate(
            _evaluation_specs()
        ):
            for index in range(config.layouts_per_tier):
                regime_seed = (
                    config.seed
                    + 2_000_003
                    + capability_index * 10_007
                    + index * 103
                )
                regime_key = (regime, capability, regime_seed)
                if regime_key in existing_regime_keys:
                    continue
                regime_rows.append(
                    await _run_regime_layout(
                        config=config,
                        curriculum=curriculum,
                        checkpoints=arms["learned"],
                        regime=regime,
                        capability=capability,
                        scenario=scenario,
                        tier=tier,
                        seed=regime_seed,
                    )
                )
                existing_regime_keys.add(regime_key)
                if resolved_progress is not None:
                    _save_regime_progress(
                        progress_dir=resolved_progress,
                        config=config,
                        schedule_sha256=schedule_sha256,
                        arm_checkpoint_sha256=arm_checkpoint_sha256,
                        rows=regime_rows,
                    )
                    completed_work_items += 1
                    if (
                        max_new_work_items is not None
                        and completed_work_items >= max_new_work_items
                    ):
                        raise EcologyP1ProgressPaused(
                            completed_work_items=completed_work_items
                        )
    regime_order = {
        regime: index
        for index, regime in enumerate(ECOLOGY_P1_REGIME_NAMES)
    }
    regime_tuple = tuple(
        sorted(
            regime_rows,
            key=lambda item: (
                regime_order[item.regime],
                capability_order[item.capability],
                item.seed,
            ),
        )
    )
    regime_summary = _regime_gap_summary(regime_tuple)
    diagnostics = tuple(
        _run_diagnostic_layout(
            config=config,
            curriculum=curriculum,
            controller=controller,
            capability=capability,
            scenario=scenario,
            tier=tier,
            seed=(
                config.seed
                + 2_000_003
                + capability_index * 10_007
                + index * 103
            ),
        )
        for controller in ("oracle_steering", "fixed_rule", "random")
        for capability_index, (capability, scenario, tier) in enumerate(
            _evaluation_specs()
        )
        for index in range(config.layouts_per_tier)
    )
    required_layouts = math.ceil(
        config.layouts_per_tier * config.layout_success_ratio
    )
    required_bodies = _required_bodies(config)
    gates: list[EcologyP1Gate] = []
    budget_failures = ecology_p1_formal_budget_failures(config)
    gates.append(
        EcologyP1Gate(
            name="formal_configuration",
            passed=not budget_failures,
            observed=(
                "pass" if not budget_failures else repr(budget_failures)
            ),
            threshold=(
                f">={ECOLOGY_P1_FORMAL_MIN_ANTS} ants, latent dim "
                f"=={ECOLOGY_P1_FORMAL_LATENT_DIM}, "
                f">={ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER} layout seeds per "
                f"tier, >={ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS} training "
                f"rounds, >={ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS} held-out "
                "rounds"
            ),
        )
    )

    def _mean_composite_harmful_rate(arm: str) -> float | None:
        rows = tuple(
            item
            for item in result_tuple
            if item.arm == arm and item.capability == "composite"
        )
        if not rows:
            return None
        return sum(item.harmful_tick_rate for item in rows) / len(rows)

    learned_composite_rate = _mean_composite_harmful_rate("learned")
    no_optimize_composite_rate = _mean_composite_harmful_rate("no_optimize")
    for capability, _, _ in _evaluation_specs():
        success = _success_count(result_tuple, "learned", capability)
        passed = success >= required_layouts
        observed = (
            f"successful_layouts={success}/{config.layouts_per_tier}"
        )
        threshold = (
            f">={required_layouts} layouts; each requires "
            f">={required_bodies} bodies"
        )
        if capability == "composite":
            # plan section 4.4: "Composite：至少 60% layout 完成 foraging，且
            # harmful ticks 不高于 matched no-optimize". Only the absolute 5%
            # cap was applied; the matched comparison -- which is what makes
            # the number evidence about the OPTIMIZER rather than about the
            # layout -- was missing. Same semantics as P2's
            # ``heat_exposure_bounded``.
            exposure_ok = (
                learned_composite_rate is not None
                and no_optimize_composite_rate is not None
                and learned_composite_rate
                <= no_optimize_composite_rate + 1e-12
            )
            passed = passed and exposure_ok
            observed += (
                ", learned_composite_harmful_rate="
                f"{learned_composite_rate}"
                ", no_optimize_composite_harmful_rate="
                f"{no_optimize_composite_rate}"
            )
            threshold += (
                "; learned composite harmful tick rate <= matched "
                "no-optimize"
            )
        gates.append(
            EcologyP1Gate(
                name=capability,
                passed=passed,
                observed=observed,
                threshold=threshold,
            )
        )
    learned_escape = tuple(
        item
        for item in result_tuple
        if item.arm == "learned" and item.capability == "forced_escape"
    )
    random_escape = tuple(
        item
        for item in diagnostics
        if item.controller == "random"
        and item.capability == "forced_escape"
    )
    learned_escape_bodies = sum(
        item.successful_bodies for item in learned_escape
    )
    random_escape_bodies = sum(
        item.successful_bodies for item in random_escape
    )
    learned_escape_latencies = tuple(
        latency for item in learned_escape for latency in item.escape_latencies
    )
    random_escape_latencies = tuple(
        latency for item in random_escape for latency in item.escape_latencies
    )
    # plan section 4.4 requires median AND p90 escape latency to be reported;
    # only the median existed, and only inside this gate's observed string.
    escape_latency_summaries = (
        _escape_latency_summary(
            source="learned",
            latencies=learned_escape_latencies,
        ),
        _escape_latency_summary(
            source="random",
            latencies=random_escape_latencies,
        ),
    )
    learned_escape_summary, random_escape_summary = escape_latency_summaries
    learned_escape_median = (
        learned_escape_summary.median
        if learned_escape_summary.median is not None
        else math.inf
    )
    random_escape_median = (
        random_escape_summary.median
        if random_escape_summary.median is not None
        else math.inf
    )
    escape_above_floor = (
        learned_escape_bodies > random_escape_bodies
        or (
            learned_escape_bodies == random_escape_bodies
            and learned_escape_bodies > 0
            and learned_escape_median < random_escape_median
        )
    )
    gates.append(EcologyP1Gate(
        name="forced_escape_above_random_floor",
        passed=escape_above_floor,
        observed=(
            f"learned_bodies={learned_escape_bodies}, "
            f"random_bodies={random_escape_bodies}, "
            f"learned_median={learned_escape_summary.median}, "
            f"learned_p90={learned_escape_summary.p90}, "
            f"random_median={random_escape_summary.median}, "
            f"random_p90={random_escape_summary.p90}"
        ),
        threshold=(
            "more escaped bodies than random, or equal nonzero success with "
            "strictly lower median latency"
        ),
    ))
    core = ("butter_medium", "butter_far", "heat_route_foraging", "neutral_stick", "composite")
    learned_score = sum(_success_count(result_tuple, "learned", item) for item in core)
    no_opt_score = sum(_success_count(result_tuple, "no_optimize", item) for item in core)
    cold_score = sum(_success_count(result_tuple, "cold", item) for item in core)
    gates.append(EcologyP1Gate(
        name="learned_not_worse_than_no_optimize",
        passed=all(
            _success_count(result_tuple, "learned", item)
            >= _success_count(result_tuple, "no_optimize", item)
            for item in core
        ),
        observed=f"learned={learned_score}, no_optimize={no_opt_score}",
        threshold="learned success count >= no-optimize for every core capability",
    ))
    gates.append(EcologyP1Gate(
        name="paired_capability_effect_positive",
        passed=learned_score > max(no_opt_score, cold_score),
        observed=f"learned={learned_score}, no_optimize={no_opt_score}, cold={cold_score}",
        threshold="learned aggregate success strictly exceeds cold and no-optimize",
    ))
    oracle_results = tuple(
        item for item in diagnostics if item.controller == "oracle_steering"
    )
    oracle_success_by_capability = {
        capability: sum(
            item.layout_success
            for item in oracle_results
            if item.capability == capability
        )
        for capability, _, _ in _evaluation_specs()
    }
    gates.append(EcologyP1Gate(
        name="diagnostic_layout_solvability",
        passed=all(
            count >= required_layouts
            for count in oracle_success_by_capability.values()
        ),
        observed=repr(oracle_success_by_capability),
        threshold=(
            "oracle steering succeeds on >=60% layouts for every capability"
        ),
    ))
    action_chain_passed, action_chain_failures = (
        await _ecology_action_chain_guard(
            config=curriculum,
            baseline=initial,
            candidate=arms["learned"],
        )
    )
    gates.append(EcologyP1Gate(
        name="p0_action_sensitivity",
        passed=action_chain_passed,
        observed=(
            "pass" if action_chain_passed else repr(action_chain_failures)
        ),
        threshold="all per-body P0 action probes pass",
    ))
    final_action_probes = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + 700_003,
        checkpoints=arms["learned"],
        turn_delta_threshold=curriculum.action_probe_turn_delta_threshold,
    )
    home_probes = tuple(
        probe
        for body in final_action_probes
        for probe in body.probes
        if probe.kind is EcologyProbeKind.HOME
    )
    gates.append(EcologyP1Gate(
        name="carrying_home_action_alignment",
        passed=bool(home_probes)
        and all(
            probe.input_reachable
            and probe.action_sensitive
            and probe.target_aligned
            for probe in home_probes
        ),
        observed=repr(
            tuple(
                (
                    probe.turn_delta,
                    probe.right_turn,
                    probe.target_aligned,
                )
                for probe in home_probes
            )
        ),
        threshold=(
            "every carrying-state probe changes action and turns toward home"
        ),
    ))
    post_pickup_uturn_probes = (
        await run_ecology_checkpoint_post_pickup_uturn_probes(
            temporal_latent_dim=config.temporal_latent_dim,
            seed=config.seed + 700_003,
            checkpoints=arms["learned"],
        )
    )
    aligned_uturn_bodies = sum(
        probe.passed for probe in post_pickup_uturn_probes
    )
    gates.append(EcologyP1Gate(
        name="post_pickup_uturn_progress",
        passed=(
            len(post_pickup_uturn_probes) == config.n_ants
            and aligned_uturn_bodies >= required_bodies
        ),
        observed=repr(
            tuple(
                (
                    probe.body_id,
                    probe.passed,
                    tuple(
                        (
                            lane.side,
                            lane.picked_up,
                            lane.post_pickup_switch_observed,
                            lane.first_post_pickup_switch_step,
                            lane.delivered,
                            lane.net_home_progress,
                            lane.max_consecutive_approach_steps,
                            lane.policy_fingerprint_stable,
                            lane.temporal_learning_fingerprint_stable,
                        )
                        for lane in probe.lanes
                    ),
                )
                for probe in post_pickup_uturn_probes
            )
        ),
        threshold=(
            f">={required_bodies}/{config.n_ants} bodies pass both +/-135-degree "
            "frozen lanes after a real pickup; each lane must switch action "
            "family within <="
            f"{ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY} post-pickup "
            "steps, then deliver or reduce home distance by >="
            f"{ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS:.3f} with >="
            f"{ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS} "
            "consecutive approach steps, while policy and temporal-learning "
            "fingerprints remain stable"
        ),
    ))
    # Near-range food-steering honesty gate. Near pickups can be produced by a
    # small exploration circle that sweeps over nearby food WITHOUT any learned
    # food-gradient steering; that false positive has historically masked the
    # absence of the exact capability medium/far require. This gate reads the
    # already-computed absolute-direction probe truth (left food -> left turn,
    # right food -> right turn) and never feeds learning.
    food_probes = tuple(
        probe
        for body in final_action_probes
        for probe in body.probes
        if probe.kind is EcologyProbeKind.FOOD
    )
    required_food_bodies = max(
        1, math.ceil(config.n_ants * config.body_success_ratio)
    )
    aligned_food_bodies = sum(
        probe.input_reachable
        and probe.action_sensitive
        and probe.target_aligned
        for probe in food_probes
    )
    gates.append(EcologyP1Gate(
        name="food_steering_alignment",
        passed=(
            bool(food_probes)
            and aligned_food_bodies >= required_food_bodies
        ),
        observed=(
            f"aligned_bodies={aligned_food_bodies}/{len(food_probes)}, "
            + repr(
                tuple(
                    (
                        probe.left_turn,
                        probe.right_turn,
                        probe.target_aligned,
                    )
                    for probe in food_probes
                )
            )
        ),
        threshold=(
            f">={required_food_bodies} bodies steer toward near food "
            "(left food -> left turn, right food -> right turn); near "
            "pickups alone do not prove this"
        ),
    ))
    learned_results = tuple(item for item in result_tuple if item.arm == "learned")
    gates.append(_temporal_non_timeout_closure_gate(learned_results))
    gates.append(EcologyP1Gate(
        name="frozen_evaluation",
        passed=all(
            item.policy_fingerprint_stable
            and item.temporal_learning_fingerprint_stable
            for item in learned_results
        ),
        observed=str(
            all(
                item.policy_fingerprint_stable
                and item.temporal_learning_fingerprint_stable
                for item in learned_results
            )
        ),
        threshold="policy and temporal-learning owners remain frozen",
    ))
    gates.append(EcologyP1Gate(
        name="replay_lineage",
        passed=all(
            item.replay_settlement_coverage >= 0.99
            and item.replay_lineage_coverage >= 0.99
            and item.replay_drop_count == 0
            for item in learned_results
        ),
        observed=f"evaluations={len(learned_results)}",
        threshold="settlement/lineage >=0.99 and drop=0",
    ))
    # plan 4.7: "checkpoint roundtrip 与 replay lineage 继续通过" -- both are
    # conjuncts, and only replay lineage was ever graded here.
    archive_roundtrip_passed, archive_roundtrip_detail = (
        _verify_p1_checkpoint_archives(
            config=config,
            curriculum=curriculum,
            checkpoints=arms["learned"],
        )
    )
    gates.append(EcologyP1Gate(
        name="checkpoint_archive_roundtrip",
        passed=archive_roundtrip_passed,
        observed=archive_roundtrip_detail,
        threshold=(
            "a fresh session hydrated from the learned colony's own archives "
            "reproduces policy/temporal/memory fingerprints, a corrupted "
            "archive collection is rejected, and the rejected restore rolls "
            "back atomically"
        ),
    ))
    gates.append(
        _repeat_run_same_direction_gate(
            reference=repeat_reference,
            results=result_tuple,
        )
    )
    gate_tuple = tuple(gates)
    if tuple(item.name for item in gate_tuple) != ECOLOGY_P1_GATE_NAMES:
        raise RuntimeError("P1 gate schema drift")
    breakpoints = tuple(item.name for item in gate_tuple if not item.passed)
    verdict = "PASS" if not breakpoints else "BLOCK"
    return EcologyP1Report(
        schema_version=ECOLOGY_P1_SCHEMA_VERSION,
        config=config,
        schedule=schedule,
        layout_results=result_tuple,
        diagnostic_results=diagnostics,
        escape_latency_summaries=escape_latency_summaries,
        regime_diagnostic=regime_tuple,
        regime_gap_summary=regime_summary,
        repeat_reference=repeat_reference,
        post_pickup_uturn_probes=post_pickup_uturn_probes,
        gates=gate_tuple,
        verdict=verdict,
        diagnostic_breakpoints=breakpoints,
        description=(
            "PASS: all P1 development gates passed"
            if verdict == "PASS"
            else "BLOCK: " + ", ".join(breakpoints)
        ),
    )


__all__ = [
    "ECOLOGY_P1_ARM_NAMES",
    "ECOLOGY_P1_DIAGNOSTICS_SCHEMA_VERSION",
    "ECOLOGY_P1_FORMAL_BUDGET_FIELDS",
    "ECOLOGY_P1_FORMAL_LATENT_DIM",
    "ECOLOGY_P1_FORMAL_MIN_ANTS",
    "ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS",
    "ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER",
    "ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS",
    "ECOLOGY_P1_GATE_NAMES",
    "ECOLOGY_P1_PROGRESS_SCHEMA_VERSION",
    "ECOLOGY_P1_REGIME_DETERMINISTIC",
    "ECOLOGY_P1_REGIME_NAMES",
    "ECOLOGY_P1_REGIME_STOCHASTIC",
    "ECOLOGY_P1_SCHEMA_VERSION",
    "ECOLOGY_P1_TIMEOUT_ONLY_LAYOUT_RATE_MAX",
    "EcologyP1Config",
    "EcologyP1DiagnosticReport",
    "EcologyP1DiagnosticResult",
    "EcologyP1EscapeLatencySummary",
    "EcologyP1Gate",
    "EcologyP1LayoutResult",
    "EcologyP1ProgressPaused",
    "EcologyP1RegimeDiagnosticRow",
    "EcologyP1RegimeGapSummary",
    "EcologyP1RepeatReference",
    "EcologyP1Report",
    "EcologyP1TurnMagnitudeDistribution",
    "ecology_p1_formal_budget_failures",
    "load_p1_repeat_reference",
    "run_ecology_p1",
    "run_ecology_p1_diagnostics",
]
