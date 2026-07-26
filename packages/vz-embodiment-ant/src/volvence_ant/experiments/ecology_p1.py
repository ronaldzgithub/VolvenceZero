"""P1 fixed-schedule ecology capability development matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from volvence_zero.agent import (
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_checkpoint_archive,
)

from volvence_ant.controllers import FixedRuleAnt, FixedRuleConfig, RandomAnt
from volvence_ant.env.world_objects import ButterSource, BurningMatch, WoodStick
from volvence_ant.experiments.ecology_probe import (
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
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
)
from volvence_ant.runtime import AntLearningCheckpoint, KernelColonyRunner


ECOLOGY_P1_SCHEMA_VERSION = "digital-ant-ecology-p1-development.v24"
ECOLOGY_P1_PROGRESS_SCHEMA_VERSION = "digital-ant-ecology-p1-progress.v20"
ECOLOGY_P1_ARM_NAMES = (
    "learned",
    "no_optimize",
    "cold",
    "dense_local_shaping_off",
    "segment_credit_off",
)
ECOLOGY_P1_GATE_NAMES = (
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
    "food_steering_alignment",
    "temporal_non_timeout_closure",
    "frozen_evaluation",
    "replay_lineage",
)


@dataclass(frozen=True)
class EcologyP1Config:
    n_ants: int = 4
    temporal_latent_dim: int = 16
    training_rounds: int = 24
    evaluation_rounds: int = 40
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
class EcologyP1Report:
    schema_version: str
    config: EcologyP1Config
    schedule: tuple[EcologyTrainingEpisodePlan, ...]
    layout_results: tuple[EcologyP1LayoutResult, ...]
    diagnostic_results: tuple[EcologyP1DiagnosticResult, ...]
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
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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
    values = [
        ("artifact_kind", ECOLOGY_P1_PROGRESS_SCHEMA_VERSION),
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
    collection = decode_agent_learning_checkpoint_archive(
        payload,
        expected_compatibility=expected_compatibility,
    )
    if dict(collection.metadata.compatibility) != dict(
        expected_compatibility
    ):
        raise ValueError(
            "P1 progress checkpoint compatibility mismatch"
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
    )


def _fixed_schedule(config: EcologyP1Config) -> tuple[EcologyTrainingEpisodePlan, ...]:
    specs: list[
        tuple[EcologyStage, EcologyTrainingTier, bool, bool, bool]
    ] = [
        (stage, EcologyTrainingTier.NEAR, forced_escape, False, False)
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
        )
        for _ in range(config.layouts_per_tier)
    )
    specs.extend(
        (EcologyStage.BUTTER, tier, False, False, False)
        for tier in (EcologyTrainingTier.MEDIUM, EcologyTrainingTier.FAR)
        for _ in range(config.layouts_per_tier)
    )
    specs.extend(
        (stage, tier, forced_escape, False, False)
        for stage, tier, forced_escape in (
            (EcologyStage.BURNING_MATCH, EcologyTrainingTier.NEAR, True),
            (EcologyStage.COMPOSITE, EcologyTrainingTier.FAR, False),
            (EcologyStage.WOOD_STICK, EcologyTrainingTier.FAR, False),
            (EcologyStage.COMPOSITE, EcologyTrainingTier.FAR, False),
        )
        for _ in range(config.layouts_per_tier)
    )
    return tuple(
        EcologyTrainingEpisodePlan(
            stage=stage,
            tier=tier,
            seed=config.seed + 10_000 + index * 101,
            episode_index=index,
            interleaved=False,
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
        ) in enumerate(specs)
    )


def _evaluation_specs() -> tuple[
    tuple[str, EcologyEvaluationScenario, EcologyTrainingTier], ...
]:
    return (
        ("butter_medium", EcologyEvaluationScenario.BUTTER_ONLY, EcologyTrainingTier.MEDIUM),
        ("butter_far", EcologyEvaluationScenario.BUTTER_ONLY, EcologyTrainingTier.FAR),
        ("forced_escape", EcologyEvaluationScenario.HEAT_FORCED_ESCAPE, EcologyTrainingTier.NEAR),
        ("heat_route_foraging", EcologyEvaluationScenario.COMPOSITE, EcologyTrainingTier.FAR),
        (
            "neutral_stick",
            EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK,
            EcologyTrainingTier.FAR,
        ),
        ("composite", EcologyEvaluationScenario.COMPOSITE, EcologyTrainingTier.FAR),
    )


def _layout_result(
    *, config: EcologyP1Config, capability: str, metrics: EcologyArmMetrics
) -> EcologyP1LayoutResult:
    required = max(1, math.ceil(config.n_ants * config.body_success_ratio))
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
    stage = {
        EcologyEvaluationScenario.BUTTER_ONLY: EcologyStage.BUTTER,
        EcologyEvaluationScenario.BUTTER_WITH_NEUTRAL_STICK: EcologyStage.WOOD_STICK,
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE: EcologyStage.BURNING_MATCH,
        EcologyEvaluationScenario.COMPOSITE: EcologyStage.COMPOSITE,
    }[scenario]
    world = _world(
        config=curriculum,
        stage=stage,
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
    required = max(1, math.ceil(config.n_ants * config.body_success_ratio))
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
        schema_version="digital-ant-ecology-p1-diagnostics.v2",
        config=config,
        results=results,
        oracle_success_by_capability=oracle_success,
        required_layouts=required_layouts,
        passed=all(count >= required_layouts for _, count in oracle_success),
    )


async def run_ecology_p1(
    config: EcologyP1Config,
    *,
    progress_dir: Path | None = None,
    max_new_work_items: int | None = None,
) -> EcologyP1Report:
    if max_new_work_items is not None and max_new_work_items < 1:
        raise ValueError("max_new_work_items must be positive")
    if max_new_work_items is not None and progress_dir is None:
        raise ValueError(
            "max_new_work_items requires a resumable progress_dir"
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
    gates: list[EcologyP1Gate] = []
    for capability, _, _ in _evaluation_specs():
        success = _success_count(result_tuple, "learned", capability)
        gates.append(
            EcologyP1Gate(
                name=capability,
                passed=success >= required_layouts,
                observed=f"successful_layouts={success}/{config.layouts_per_tier}",
                threshold=(
                    f">={required_layouts} layouts; each requires "
                    f">={math.ceil(config.n_ants * 0.6)} bodies"
                ),
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
    learned_escape_median = (
        statistics.median(learned_escape_latencies)
        if learned_escape_latencies
        else math.inf
    )
    random_escape_median = (
        statistics.median(random_escape_latencies)
        if random_escape_latencies
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
            f"learned_median={learned_escape_median}, "
            f"random_median={random_escape_median}"
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
    gates.append(EcologyP1Gate(
        name="temporal_non_timeout_closure",
        passed=(
            sum(item.switch_count for item in learned_results) > 0
            and sum(
                item.non_timeout_segment_closures
                for item in learned_results
            )
            > 0
        ),
        observed=(
            f"switches={sum(item.switch_count for item in learned_results)}, "
            "non_timeout_closures="
            f"{sum(item.non_timeout_segment_closures for item in learned_results)}"
        ),
        threshold="held-out has a real switch and non-timeout segment closure",
    ))
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
    "ECOLOGY_P1_GATE_NAMES",
    "ECOLOGY_P1_SCHEMA_VERSION",
    "EcologyP1Config",
    "EcologyP1Gate",
    "EcologyP1DiagnosticResult",
    "EcologyP1DiagnosticReport",
    "EcologyP1LayoutResult",
    "EcologyP1ProgressPaused",
    "EcologyP1Report",
    "run_ecology_p1",
    "run_ecology_p1_diagnostics",
]
