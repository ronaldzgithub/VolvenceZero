"""Gate 5 longitudinal CMS absorption/retention Pareto evidence.

The harness replays the immutable public memory signal exported by the shared
Gate 4/5/6 trace.  Learning remains inside ``MemoryStore`` / ``CMSMemoryCore``;
this module only drives matched arms and aggregates owner-published readouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from volvence_zero.agent.shared_settled_trace import (
    SHARED_SETTLED_TRACE_COUNT_PER_SEED,
    SHARED_SETTLED_TRACE_SEEDS,
    build_shared_trace_plans,
    load_shared_trace_records,
    validate_shared_trace_prefix,
)
from volvence_zero.memory import (
    CMSMemoryCore,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)
from volvence_zero.memory.persistence import PersistenceBackend
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)


GATE5_CMS_PARETO_SCHEMA_VERSION = "gate5-cms-pareto.v1"
GATE5_CMS_PARETO_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
GATE5_FULL_ARM = "nested-CMS(full)"
GATE5_SINGLE_TIMESCALE_ARM = "single-timescale"
GATE5_ARM_NAMES = (
    GATE5_FULL_ARM,
    GATE5_SINGLE_TIMESCALE_ARM,
    "no-ATLAS-replay",
    "no-PE-write-gate",
    "memory-only",
)
GATE5_PARETO_TOLERANCE = 0.01
GATE5_MIN_EFFECT = 0.02
_PARTITION_SEQUENCE = (
    ("trace-train", 300),
    ("trace-heldout-context", 150),
    ("trace-locked-confirmation", 60),
)
_NEW_PHASES = {"new-introduce", "new-revision"}
_RETRIEVAL_PHASES = {"new-revision", "old-retention"}
_WRITE_PHASES = {"old-recall", "new-introduce", "new-revision"}


@dataclass(frozen=True)
class Gate5ArmMetrics:
    seed: int
    arm: str
    settled_transition_count: int
    locked_transition_count: int
    new_knowledge_absorption: float
    old_knowledge_retention: float
    memory_churn: float
    erroneous_promotion_rate: float
    retrieval_hit_rate: float
    retrieval_weighted_payoff: float
    cms_parameter_count: int
    cadence_intervals: tuple[int, ...]
    frozen_substrate_mutation_count: int
    lineage_complete: bool


@dataclass(frozen=True)
class Gate5Comparison:
    control_arm: str
    absorption_gain: float
    retention_gain: float
    absorption_seed_gains: tuple[float, ...]
    retention_seed_gains: tuple[float, ...]
    pareto_non_worse: bool


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _action_context(payload: Mapping[str, Any]) -> PredictionActionContext:
    return PredictionActionContext(
        segment_id=str(payload.get("segment_id", "")),
        abstract_action_id=str(payload.get("abstract_action_id", "")),
        z_t_digest=tuple(float(value) for value in payload.get("z_t_digest", ())),
        regime_id=str(payload.get("regime_id", "")),
        affordance_name=str(payload.get("affordance_name", "")),
        environment_event_id=str(payload.get("environment_event_id", "")),
        environment_outcome_id=str(payload.get("environment_outcome_id", "")),
        environment_task_progress=payload.get("environment_task_progress"),
        environment_action_payoff=payload.get("environment_action_payoff"),
        environment_outcome_terminal=bool(
            payload.get("environment_outcome_terminal", False)
        ),
        prediction_id=str(payload.get("prediction_id", "")),
        conditioning_bank_set=tuple(
            str(value) for value in payload.get("conditioning_bank_set", ())
        ),
        conditioning_bank_fingerprints=tuple(
            (str(item[0]), str(item[1]))
            for item in payload.get("conditioning_bank_fingerprints", ())
        ),
        conditioning_router_version=str(
            payload.get("conditioning_router_version", "")
        ),
    )


def _prediction(payload: Mapping[str, Any]) -> PredictedOutcome:
    return PredictedOutcome(
        source_turn_index=int(payload["source_turn_index"]),
        target_turn_index=int(payload["target_turn_index"]),
        predicted_task_progress=float(payload["predicted_task_progress"]),
        predicted_relationship_delta=float(
            payload["predicted_relationship_delta"]
        ),
        predicted_regime_stability=float(
            payload["predicted_regime_stability"]
        ),
        predicted_action_payoff=float(payload["predicted_action_payoff"]),
        confidence=float(payload["confidence"]),
        description=str(payload["description"]),
        action_context=_action_context(payload["action_context"]),
        prediction_id=str(payload["prediction_id"]),
    )


def _actual_outcome(payload: Mapping[str, Any]) -> ActualOutcome:
    return ActualOutcome(
        observed_turn_index=int(payload["observed_turn_index"]),
        task_progress=float(payload["task_progress"]),
        relationship_delta=float(payload["relationship_delta"]),
        regime_stability=float(payload["regime_stability"]),
        action_payoff=float(payload["action_payoff"]),
        description=str(payload["description"]),
        action_context=_action_context(payload["action_context"]),
        external_outcome_refs=tuple(
            str(value) for value in payload.get("external_outcome_refs", ())
        ),
    )


def typed_prediction_error_snapshot(
    record: Mapping[str, Any],
) -> PredictionErrorSnapshot:
    prediction = _prediction(record["prediction"])
    actual = _actual_outcome(record["actual_outcome"])
    payload = record["prediction_error"]
    error = PredictionError(
        task_error=float(payload["task_error"]),
        relationship_error=float(payload["relationship_error"]),
        regime_error=float(payload["regime_error"]),
        action_error=float(payload["action_error"]),
        magnitude=float(payload["magnitude"]),
        signed_reward=float(payload["signed_reward"]),
        description=str(payload["description"]),
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=prediction,
        actual_outcome=actual,
        next_prediction=prediction,
        error=error,
        turn_index=actual.observed_turn_index,
        bootstrap=False,
        description=(
            "Typed Gate 5 replay snapshot reconstructed from one immutable "
            "settled trace record."
        ),
        action_context=actual.action_context,
    )


def _latest_public_signal(record: Mapping[str, Any]) -> tuple[float, ...]:
    attributes = record["memory_snapshot"]["attribute_summary"]
    if not attributes:
        raise ValueError(
            f"{record['transition_id']} lacks public memory attribute signal"
        )
    latest = max(
        attributes,
        key=lambda item: (int(item["timestamp_ms"]), str(item["entry_id"])),
    )
    signal = tuple(
        float(value) for value in latest["substrate_feature_digest"]
    )
    if not signal or not all(math.isfinite(value) for value in signal):
        raise ValueError(
            f"{record['transition_id']} has invalid replay signal"
        )
    return signal


def _build_learned_store(
    *,
    variant: str,
    session_cadence: int,
    background_cadence: int,
    pe_features_enabled: bool,
    replay_window_sizes: Mapping[str, int] | None,
    persistence_backend: PersistenceBackend | None = None,
) -> MemoryStore:
    return MemoryStore(
        learned_core=CMSMemoryCore(
            mode="mlp",
            d_in=4,
            d_hidden=8,
            variant=variant,
            session_cadence=session_cadence,
            background_cadence=background_cadence,
            pe_features_enabled=pe_features_enabled,
            replay_window_sizes=replay_window_sizes,
        ),
        persistence_backend=persistence_backend,
    )


def build_gate5_arm_store(
    arm: str,
    *,
    persistence_backend: PersistenceBackend | None = None,
) -> MemoryStore:
    replay = {
        "online-fast": 8,
        "session-medium": 4,
        "background-slow": 2,
    }
    if arm == GATE5_FULL_ARM:
        return _build_learned_store(
            variant="nested",
            session_cadence=2,
            background_cadence=4,
            pe_features_enabled=True,
            replay_window_sizes=replay,
            persistence_backend=persistence_backend,
        )
    if arm == GATE5_SINGLE_TIMESCALE_ARM:
        return _build_learned_store(
            variant="independent",
            session_cadence=1,
            background_cadence=1,
            pe_features_enabled=True,
            replay_window_sizes=replay,
            persistence_backend=persistence_backend,
        )
    if arm == "no-ATLAS-replay":
        return _build_learned_store(
            variant="nested",
            session_cadence=2,
            background_cadence=4,
            pe_features_enabled=True,
            replay_window_sizes=None,
            persistence_backend=persistence_backend,
        )
    if arm == "no-PE-write-gate":
        return _build_learned_store(
            variant="nested",
            session_cadence=2,
            background_cadence=4,
            pe_features_enabled=False,
            replay_window_sizes=replay,
            persistence_backend=persistence_backend,
        )
    if arm == "memory-only":
        return MemoryStore(persistence_backend=persistence_backend)
    raise ValueError(f"Unknown Gate 5 arm {arm!r}")


def _partition_counts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            partition,
            sum(record["partition"] == partition for record in records),
        )
        for partition, _ in _PARTITION_SEQUENCE
    )


def _validate_seed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> None:
    if len(records) != SHARED_SETTLED_TRACE_COUNT_PER_SEED:
        raise ValueError(
            f"Gate 5 seed {seed} requires exactly "
            f"{SHARED_SETTLED_TRACE_COUNT_PER_SEED} records"
        )
    validate_shared_trace_prefix(
        records=records,
        expected_plans=build_shared_trace_plans(seed),
    )
    if _partition_counts(records) != _PARTITION_SEQUENCE:
        raise ValueError(
            f"Gate 5 seed {seed} partition count/order contract drifted"
        )
    expected_partition_order = tuple(
        partition
        for partition, count in _PARTITION_SEQUENCE
        for _ in range(count)
    )
    actual_partition_order = tuple(
        str(record["partition"]) for record in records
    )
    if actual_partition_order != expected_partition_order:
        raise ValueError(
            f"Gate 5 seed {seed} partition ordering drifted"
        )


def _knowledge_write(
    store: MemoryStore,
    *,
    record: Mapping[str, Any],
    timestamp_ms: int,
) -> None:
    store.write(
        MemoryWriteRequest(
            content=(
                f"Settled trace knowledge {record['knowledge_key']} in "
                f"context {record['context_id']} during "
                f"{record['episode_phase']}."
            ),
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            tags=(
                "gate5-trace",
                str(record["knowledge_key"]),
                str(record["context_id"]),
                str(record["episode_phase"]),
            ),
            strength=0.65,
            subject_ids=(str(record["user_id"]),),
            audience_ids=("self",),
        ),
        timestamp_ms=timestamp_ms,
    )


def _knowledge_retrieval_hit(
    store: MemoryStore,
    *,
    record: Mapping[str, Any],
    timestamp_ms: int,
) -> bool:
    result = store.retrieve(
        RetrievalQuery(
            text=str(record["knowledge_key"]),
            track=Track.WORLD,
            strata=(MemoryStratum.EPISODIC, MemoryStratum.DURABLE),
            limit=5,
            facets=(
                str(record["knowledge_key"]),
                str(record["context_id"]),
            ),
        ),
        timestamp_ms=timestamp_ms,
        active_subject_ids=(str(record["user_id"]),),
    )
    return any(
        str(record["knowledge_key"]) in entry.tags
        for entry in result.entries
    )


def run_gate5_arm(
    *,
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    record_validator: (
        Callable[[Sequence[Mapping[str, Any]], int], None] | None
    ) = None,
    persistence_backend: PersistenceBackend | None = None,
    session_boundary_interval: int | None = None,
) -> tuple[
    Gate5ArmMetrics,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Replay one complete seed through one matched memory arm."""

    if record_validator is None:
        _validate_seed_records(records, seed=seed)
    else:
        record_validator(records, seed)
    if (
        session_boundary_interval is not None
        and session_boundary_interval < 1
    ):
        raise ValueError("session_boundary_interval must be positive")
    if (
        session_boundary_interval is not None
        and persistence_backend is None
    ):
        raise ValueError(
            "session boundaries require a persistence backend"
        )
    store = build_gate5_arm_store(
        arm,
        persistence_backend=persistence_backend,
    )
    initial_checkpoint = store.create_checkpoint(
        checkpoint_id=f"gate5-{seed}-{arm}-initial"
    )
    state_rows: list[dict[str, Any]] = []
    absorption: list[float] = []
    retention: list[float] = []
    churn: list[float] = []
    retrieval_hits: list[float] = []
    retrieval_payoffs: list[float] = []
    erroneous_promotions = 0
    negative_revision_count = 0
    mutation_count = 0
    constructor_restart_count = 0
    persistence_roundtrip_exact = True
    for replay_index, record in enumerate(records):
        timestamp_ms = seed * 1_000_000 + replay_index
        phase = str(record["episode_phase"])
        pe_snapshot = typed_prediction_error_snapshot(record)
        store.observe_replay_signal(
            signal=_latest_public_signal(record),
            timestamp_ms=timestamp_ms,
            prediction_error=pe_snapshot,
        )
        retrieval_hit: bool | None = None
        if phase in _RETRIEVAL_PHASES:
            retrieval_hit = _knowledge_retrieval_hit(
                store,
                record=record,
                timestamp_ms=timestamp_ms,
            )
        if phase in _WRITE_PHASES:
            _knowledge_write(
                store,
                record=record,
                timestamp_ms=timestamp_ms,
            )
        snapshot = store.snapshot(
            retrieved_entries=(),
            active_subject_scope=(str(record["user_id"]),),
        )
        lifecycle = dict(snapshot.lifecycle_metrics)
        cms_state = snapshot.cms_state
        cadence_intervals = (
            (
                cms_state.online_fast.cadence_interval,
                cms_state.session_medium.cadence_interval,
                cms_state.background_slow.cadence_interval,
            )
            if cms_state is not None
            else ()
        )
        parameter_count = (
            cms_state.online_fast.mlp_param_count
            + cms_state.session_medium.mlp_param_count
            + cms_state.background_slow.mlp_param_count
            if cms_state is not None
            else 0
        )
        locked = record["partition"] == "trace-locked-confirmation"
        if locked and phase in _NEW_PHASES:
            absorption.append(
                float(lifecycle["cms_new_knowledge_absorption"])
            )
        if locked and phase == "old-retention":
            retention.append(
                float(lifecycle["cms_old_knowledge_retention"])
            )
        if locked:
            churn.append(
                float(lifecycle["memory_updater_touched_param_ratio"])
            )
            if retrieval_hit is not None:
                retrieval_hits.append(float(retrieval_hit))
                retrieval_payoffs.append(
                    float(record["actual_outcome"]["action_payoff"])
                    * float(retrieval_hit)
                )
            if phase == "new-revision":
                negative_revision_count += 1
                durable_count = dict(
                    snapshot.total_entries_by_stratum
                )[MemoryStratum.DURABLE.value]
                if durable_count > 0:
                    erroneous_promotions += 1
        mutated = bool(record["substrate"]["mutation_applied"])
        mutation_count += int(mutated)
        state_rows.append(
            {
                "schema_version": GATE5_CMS_PARETO_SCHEMA_VERSION,
                "seed": seed,
                "arm": arm,
                "transition_id": record["transition_id"],
                "prediction_ref": record["lineage"]["prediction_ref"],
                "record_sha256": record["record_sha256"],
                "partition": record["partition"],
                "episode_phase": phase,
                "knowledge_key": record["knowledge_key"],
                "cms_new_knowledge_absorption": float(
                    lifecycle["cms_new_knowledge_absorption"]
                ),
                "cms_old_knowledge_retention": float(
                    lifecycle["cms_old_knowledge_retention"]
                ),
                "memory_churn": float(
                    lifecycle["memory_updater_touched_param_ratio"]
                ),
                "retrieval_hit": retrieval_hit,
                "durable_entry_count": dict(
                    snapshot.total_entries_by_stratum
                )[MemoryStratum.DURABLE.value],
                "cms_parameter_count": parameter_count,
                "cadence_intervals": list(cadence_intervals),
                "substrate_mutation_applied": mutated,
            }
        )
        boundary_after = (
            session_boundary_interval is not None
            and (replay_index + 1) % session_boundary_interval == 0
            and replay_index + 1 < len(records)
        )
        if boundary_after:
            before_restart = store.create_checkpoint(
                checkpoint_id="gate5-session-boundary"
            )
            if not store.save_to_backend():
                raise RuntimeError(
                    "Gate 5 longitudinal arm could not persist memory"
                )
            restarted = build_gate5_arm_store(
                arm,
                persistence_backend=persistence_backend,
            )
            loaded = restarted.load_from_backend()
            after_restart = restarted.create_checkpoint(
                checkpoint_id="gate5-session-boundary"
            )
            persistence_roundtrip_exact = (
                persistence_roundtrip_exact
                and loaded
                and before_restart == after_restart
            )
            constructor_restart_count += 1
            store = restarted
    final_checkpoint = store.create_checkpoint(
        checkpoint_id=f"gate5-{seed}-{arm}-final"
    )
    store.restore_checkpoint(initial_checkpoint)
    restored_checkpoint = store.create_checkpoint(
        checkpoint_id=initial_checkpoint.checkpoint_id
    )
    rollback_exact = restored_checkpoint == initial_checkpoint
    final_state_changed = (
        final_checkpoint.entries != initial_checkpoint.entries
        or final_checkpoint.cms_state != initial_checkpoint.cms_state
    )
    last = state_rows[-1]
    metrics = Gate5ArmMetrics(
        seed=seed,
        arm=arm,
        settled_transition_count=len(records),
        locked_transition_count=sum(
            record["partition"] == "trace-locked-confirmation"
            for record in records
        ),
        new_knowledge_absorption=_mean(absorption),
        old_knowledge_retention=_mean(retention, default=1.0),
        memory_churn=_mean(churn),
        erroneous_promotion_rate=(
            erroneous_promotions / negative_revision_count
            if negative_revision_count
            else 0.0
        ),
        retrieval_hit_rate=_mean(retrieval_hits),
        retrieval_weighted_payoff=_mean(retrieval_payoffs),
        cms_parameter_count=int(last["cms_parameter_count"]),
        cadence_intervals=tuple(last["cadence_intervals"]),
        frozen_substrate_mutation_count=mutation_count,
        lineage_complete=all(
            row["transition_id"]
            and row["prediction_ref"]
            and row["record_sha256"]
            for row in state_rows
        ),
    )
    rollback = {
        "seed": seed,
        "arm": arm,
        "checkpoint_roundtrip_exact": rollback_exact,
        "final_state_changed_before_rollback": final_state_changed,
        "constructor_restart_count": constructor_restart_count,
        "persistence_roundtrip_exact": persistence_roundtrip_exact,
        "rollback_target": (
            "memory-only"
            if arm == "memory-only"
            else "pre-replay MemoryStore checkpoint"
        ),
    }
    return metrics, state_rows, rollback


def compare_gate5_arms(
    metrics: Sequence[Gate5ArmMetrics],
) -> tuple[tuple[Gate5Comparison, ...], dict[str, bool]]:
    by_arm_seed = {
        (metric.arm, metric.seed): metric for metric in metrics
    }
    comparisons: list[Gate5Comparison] = []
    for control in GATE5_ARM_NAMES[1:]:
        absorption_seed_gains = tuple(
            by_arm_seed[(GATE5_FULL_ARM, seed)].new_knowledge_absorption
            - by_arm_seed[(control, seed)].new_knowledge_absorption
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        retention_seed_gains = tuple(
            by_arm_seed[(GATE5_FULL_ARM, seed)].old_knowledge_retention
            - by_arm_seed[(control, seed)].old_knowledge_retention
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        absorption_gain = _mean(absorption_seed_gains)
        retention_gain = _mean(retention_seed_gains)
        comparisons.append(
            Gate5Comparison(
                control_arm=control,
                absorption_gain=absorption_gain,
                retention_gain=retention_gain,
                absorption_seed_gains=absorption_seed_gains,
                retention_seed_gains=retention_seed_gains,
                pareto_non_worse=(
                    absorption_gain >= -GATE5_PARETO_TOLERANCE
                    and retention_gain >= -GATE5_PARETO_TOLERANCE
                    and all(
                        gain >= -GATE5_PARETO_TOLERANCE
                        for gain in absorption_seed_gains
                    )
                    and all(
                        gain >= -GATE5_PARETO_TOLERANCE
                        for gain in retention_seed_gains
                    )
                ),
            )
        )
    single = next(
        comparison
        for comparison in comparisons
        if comparison.control_arm == GATE5_SINGLE_TIMESCALE_ARM
    )
    significant_single_dimension = (
        (
            single.absorption_gain >= GATE5_MIN_EFFECT
            and all(gain > 0.0 for gain in single.absorption_seed_gains)
        )
        or (
            single.retention_gain >= GATE5_MIN_EFFECT
            and all(gain > 0.0 for gain in single.retention_seed_gains)
        )
    )
    full_metrics = tuple(
        metric for metric in metrics if metric.arm == GATE5_FULL_ARM
    )
    single_metrics = tuple(
        metric
        for metric in metrics
        if metric.arm == GATE5_SINGLE_TIMESCALE_ARM
    )
    gates = {
        "all_arms_all_seeds_present": len(metrics)
        == len(GATE5_ARM_NAMES) * len(SHARED_SETTLED_TRACE_SEEDS),
        "all_records_and_locked_counts_complete": all(
            metric.settled_transition_count == 510
            and metric.locked_transition_count == 60
            for metric in metrics
        ),
        "lineage_complete": all(metric.lineage_complete for metric in metrics),
        "frozen_substrate_mutation_zero": all(
            metric.frozen_substrate_mutation_count == 0
            for metric in metrics
        ),
        "full_cadence_1_2_4": all(
            metric.cadence_intervals == (1, 2, 4)
            for metric in full_metrics
        ),
        "single_cadence_1_1_1": all(
            metric.cadence_intervals == (1, 1, 1)
            for metric in single_metrics
        ),
        "matched_cms_parameter_budget": (
            {metric.cms_parameter_count for metric in full_metrics}
            == {metric.cms_parameter_count for metric in single_metrics}
        ),
        "full_pareto_non_worse_all_controls": all(
            comparison.pareto_non_worse
            for comparison in comparisons
        ),
        "full_significant_vs_single_timescale": (
            significant_single_dimension
        ),
    }
    return tuple(comparisons), gates


def _source_rows(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result = {
        "predictions.jsonl": [],
        "outcomes.jsonl": [],
        "prediction_errors.jsonl": [],
        "segments.jsonl": [],
        "credit.jsonl": [],
        "action_selection.jsonl": [],
    }
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        for record in records_by_seed[seed]:
            lineage = {
                "seed": seed,
                "transition_id": record["transition_id"],
                "prediction_ref": record["lineage"]["prediction_ref"],
                "record_sha256": record["record_sha256"],
                "partition": record["partition"],
            }
            result["predictions.jsonl"].append(
                {**lineage, "prediction": record["prediction"]}
            )
            result["outcomes.jsonl"].append(
                {**lineage, "actual_outcome": record["actual_outcome"]}
            )
            result["prediction_errors.jsonl"].append(
                {**lineage, "prediction_error": record["prediction_error"]}
            )
            result["segments.jsonl"].append(
                {
                    **lineage,
                    "episode_phase": record["episode_phase"],
                    "temporal_snapshot": record["temporal_snapshot"],
                }
            )
            result["credit.jsonl"].append(
                {**lineage, "credit_snapshot": record["credit_snapshot"]}
            )
            result["action_selection.jsonl"].append(
                {**lineage, "action_selection": record["action_selection"]}
            )
    return result


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def export_gate5_cms_pareto_bundle(
    *,
    trace_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Run the preregistered five-arm Gate 5 campaign and write its packet."""

    started = time.perf_counter()
    source = Path(trace_root)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    aggregate_manifest = json.loads(
        (source / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    aggregate_verdict = json.loads(
        (source / "aggregate_verdict.json").read_text(encoding="utf-8")
    )
    if aggregate_verdict.get("consumer_admission") != "allowed":
        raise ValueError("Gate 5 source trace is not admitted for consumption")
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        records = load_shared_trace_records(
            source / f"seed_{seed}" / "transitions.jsonl"
        )
        _validate_seed_records(records, seed=seed)
        records_by_seed[seed] = records
    metrics: list[Gate5ArmMetrics] = []
    state_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        for arm in GATE5_ARM_NAMES:
            arm_metrics, arm_rows, rollback = run_gate5_arm(
                records=records_by_seed[seed],
                seed=seed,
                arm=arm,
            )
            metrics.append(arm_metrics)
            state_rows.extend(arm_rows)
            rollback_rows.append(rollback)
    comparisons, gates = compare_gate5_arms(metrics)
    rollback_passed = all(
        row["checkpoint_roundtrip_exact"] for row in rollback_rows
    )
    gates["checkpoint_rollback_exact"] = rollback_passed
    latency_rows = [
        record["latency"]
        for records in records_by_seed.values()
        for record in records
    ]
    turn_latencies = [
        float(row["prediction_turn_ms"])
        + float(row["settlement_turn_ms"])
        for row in latency_rows
    ]
    slow_latencies = [
        float(row["session_post_slow_job_ms"])
        for row in latency_rows
    ]
    gates["background_latency_published_separately"] = all(
        set(row)
        >= {
            "prediction_turn_ms",
            "settlement_turn_ms",
            "session_post_slow_job_ms",
        }
        for row in latency_rows
    )
    invalid_gates = tuple(
        name
        for name in (
            "all_arms_all_seeds_present",
            "all_records_and_locked_counts_complete",
            "lineage_complete",
            "frozen_substrate_mutation_zero",
            "full_cadence_1_2_4",
            "single_cadence_1_1_1",
            "matched_cms_parameter_budget",
            "checkpoint_rollback_exact",
            "background_latency_published_separately",
        )
        if not gates[name]
    )
    causal_gates = (
        gates["full_pareto_non_worse_all_controls"]
        and gates["full_significant_vs_single_timescale"]
    )
    status = (
        "invalid"
        if invalid_gates
        else "causal-supported"
        if causal_gates
        else "not-supported"
    )
    manifest = {
        "schema_version": GATE5_CMS_PARETO_SCHEMA_VERSION,
        "suite_id": "gate5-cms-pareto",
        "owner": "vz-memory.MemoryStore/CMSMemoryCore",
        "trace_schema_version": "gate456-shared-settled-trace.v1",
        "trace_root": str(source),
        "substrate_fingerprint": aggregate_manifest["runtime_fingerprint"],
        "source_trace_sha256_by_seed": {
            str(summary["seed"]): summary["trace_sha256"]
            for summary in aggregate_manifest["seed_summaries"]
        },
        "model_and_adapter_ids": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_ids": [],
            "substrate_mode": "frozen strict-local source trace",
        },
        "wiring_levels": {
            "source_runtime": "ACTIVE frozen trace",
            "memory_owner": "matched five-arm offline replay",
            "substrate_mutation": "DISABLED",
        },
        "seed_schedule": list(SHARED_SETTLED_TRACE_SEEDS),
        "arm_schedule": list(GATE5_ARM_NAMES),
        "partition_schedule": [
            {"partition": partition, "count_per_seed": count}
            for partition, count in _PARTITION_SEQUENCE
        ],
        "primary_partition": "trace-locked-confirmation",
        "scenario_split": {
            partition: count * len(SHARED_SETTLED_TRACE_SEEDS)
            for partition, count in _PARTITION_SEQUENCE
        },
        "cohort_scope": {
            "seed_count": len(SHARED_SETTLED_TRACE_SEEDS),
            "context_count": 18,
            "user_count": 18,
            "settled_transition_count": 1530,
        },
        "prompt_and_context_budget": (
            "No new prompts; replay consumes only immutable public signal "
            "and typed PE from the shared trace."
        ),
        "metric_version": GATE5_CMS_PARETO_SCHEMA_VERSION,
        "judge_or_human_protocol": (
            "None; deterministic owner-published readout aggregation."
        ),
        "pareto_tolerance": GATE5_PARETO_TOLERANCE,
        "minimum_effect_vs_single_timescale": GATE5_MIN_EFFECT,
        "signal_source": (
            "memory_snapshot.attribute_summary.latest."
            "substrate_feature_digest"
        ),
        "required_files": list(GATE5_CMS_PARETO_REQUIRED_FILES),
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": bool(
                _git_output("status", "--porcelain") not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE5_CMS_PARETO_SCHEMA_VERSION,
        "metrics": [asdict(metric) for metric in metrics],
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "gates": gates,
        "latency_diagnostics": {
            "turn_latency_p50_ms": _percentile(turn_latencies, 0.5),
            "turn_latency_p95_ms": _percentile(turn_latencies, 0.95),
            "session_post_slow_latency_p50_ms": _percentile(
                slow_latencies, 0.5
            ),
            "session_post_slow_latency_p95_ms": _percentile(
                slow_latencies, 0.95
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    verdict = {
        "schema_version": GATE5_CMS_PARETO_SCHEMA_VERSION,
        "gate_scope": "Gate 5 CMS multi-timescale Pareto",
        "status": status,
        "mechanism_passed": not invalid_gates,
        "causal_passed": status == "causal-supported",
        "claim": (
            "multi-frequency CMS improves the absorption-retention Pareto"
            if status == "causal-supported"
            else "multi-frequency CMS is runnable, auditable, and rollback-capable"
            if status == "not-supported"
            else "Gate 5 evidence packet is invalid"
        ),
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "invalid_gates": list(invalid_gates),
        "locked_partition_consumed_once": True,
        "same_locked_partition_rerun_allowed": False,
    }
    rollback_payload = {
        "schema_version": GATE5_CMS_PARETO_SCHEMA_VERSION,
        "passed": rollback_passed,
        "arms": rollback_rows,
        "explicit_factory_rollback": {
            "cms_pe_features_enabled": False,
            "cms_replay_window_size": None,
        },
        "substrate_mutated": False,
    }
    source_rows = _source_rows(records_by_seed)
    for name, rows in source_rows.items():
        _write_jsonl(target / name, rows)
    _write_jsonl(target / "state_diff.jsonl", state_rows)
    _write_json(target / "manifest.yaml", manifest)
    _write_json(target / "ablation_results.json", ablation)
    _write_json(target / "promotion_verdict.json", verdict)
    _write_json(target / "rollback_evidence.json", rollback_payload)
    report_lines = [
        "# Gate 5 CMS Pareto evidence",
        "",
        f"- status: `{status}`",
        f"- mechanism passed: `{not invalid_gates}`",
        f"- causal passed: `{status == 'causal-supported'}`",
        f"- settled transitions replayed per arm: `{len(state_rows) // len(GATE5_ARM_NAMES)}`",
        "- locked partition consumed once: `True`",
        "",
        "## Full vs controls",
        "",
    ]
    report_lines.extend(
        (
            f"- `{comparison.control_arm}`: absorption gain "
            f"`{comparison.absorption_gain:.6f}`, retention gain "
            f"`{comparison.retention_gain:.6f}`, Pareto non-worse "
            f"`{comparison.pareto_non_worse}`"
        )
        for comparison in comparisons
    )
    report_lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            (
                "- The primary metrics are owner-published band-drift "
                "proxies. Retrieval and payoff are diagnostics; this packet "
                "does not establish deployment-time behavioral memory."
            ),
            (
                "- A failed Pareto/minimum-effect gate contracts the claim "
                "without retuning or rerunning the locked partition."
            ),
            "",
        )
    )
    (target / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    written = tuple(
        target / name for name in GATE5_CMS_PARETO_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 5 bundle missing required files {missing!r}"
        )
    return written


__all__ = [
    "GATE5_ARM_NAMES",
    "GATE5_CMS_PARETO_REQUIRED_FILES",
    "GATE5_CMS_PARETO_SCHEMA_VERSION",
    "GATE5_FULL_ARM",
    "GATE5_MIN_EFFECT",
    "GATE5_PARETO_TOLERANCE",
    "GATE5_SINGLE_TIMESCALE_ARM",
    "Gate5ArmMetrics",
    "Gate5Comparison",
    "build_gate5_arm_store",
    "compare_gate5_arms",
    "export_gate5_cms_pareto_bundle",
    "run_gate5_arm",
    "typed_prediction_error_snapshot",
]
