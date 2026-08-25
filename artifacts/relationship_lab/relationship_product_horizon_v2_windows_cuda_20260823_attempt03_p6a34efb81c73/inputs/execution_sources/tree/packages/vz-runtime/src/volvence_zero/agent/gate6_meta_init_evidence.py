"""Gate 6 nested meta-init matched episode evidence.

The harness consumes only immutable public signals from the admitted shared
trace.  All learned state and initialization mutations remain inside the
``MemoryStore`` / ``CMSMemoryCore`` owner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from volvence_zero.agent.gate5_cms_pareto_evidence import (
    typed_prediction_error_snapshot,
)
from volvence_zero.agent.shared_settled_trace import (
    SHARED_SETTLED_TRACE_SEEDS,
    build_shared_trace_plans,
    load_shared_trace_records,
    validate_shared_trace_prefix,
)
from volvence_zero.memory import CMSMemoryCore, MemoryStore, MemoryStoreCheckpoint


GATE6_META_INIT_SCHEMA_VERSION = "gate6-meta-init.v1"
GATE6_REQUIRED_FILES = (
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
GATE6_PRIMARY_ARMS = (
    "meta-init",
    "copy-init",
    "random-init",
    "no-init",
)
GATE6_DIAGNOSTIC_ARMS = (
    "paired-user-slow-state",
    "swapped-user-slow-state",
)
GATE6_TARGET_ERROR = 0.02
GATE6_EARLY_K = 8
GATE6_ERROR_SCALE = 0.10
GATE6_MIN_STEP_GAIN = 2.0
GATE6_MIN_AUC_GAIN = 0.05
GATE6_FINAL_ERROR_TOLERANCE = 0.01
GATE6_NEGATIVE_TRANSFER_LIMIT = 0.0
_TRAIN_COUNT = 300
_HELDOUT_COUNT = 150
_LOCKED_COUNT = 60


@dataclass(frozen=True)
class Gate6EpisodeMetrics:
    seed: int
    partition: str
    context_id: str
    user_id: str
    arm: str
    episode_length: int
    steps_to_target: int
    early_adaptation_auc: float
    final_error: float
    final_quality: float
    initial_error: float
    initialization_changed_fast_state: bool
    slow_state_unchanged: bool
    parameter_state_unchanged: bool
    lineage_complete: bool
    frozen_substrate_mutation_count: int
    fact_leakage_count: int
    checkpoint_restore_exact: bool


@dataclass(frozen=True)
class Gate6ArmComparison:
    control_arm: str
    aggregate_step_gain: float
    aggregate_auc_gain: float
    step_seed_gains: tuple[float, ...]
    auc_seed_gains: tuple[float, ...]
    aggregate_final_error_delta: float
    final_error_seed_deltas: tuple[float, ...]
    minimum_effect_passed: bool
    final_error_non_inferior: bool


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _load_trace_prefix(path: Path, *, count: int) -> list[dict[str, Any]]:
    """Read and authenticate exactly one development prefix.

    This intentionally stops before the locked lines instead of loading the
    whole JSONL and filtering after the fact.
    """

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > count:
                break
            if not line.strip():
                raise ValueError(f"Blank shared trace line at {line_number}")
            payload = json.loads(line)
            record_sha = payload.pop("record_sha256", None)
            expected_sha = hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest()
            payload["record_sha256"] = record_sha
            if record_sha != expected_sha:
                raise ValueError(
                    "Shared trace record digest mismatch at line "
                    f"{line_number}"
                )
            records.append(payload)
    if len(records) != count:
        raise ValueError(
            f"Expected {count} shared trace prefix records, got {len(records)}"
        )
    return records


def _public_signal(record: Mapping[str, Any]) -> tuple[float, ...]:
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
            f"{record['transition_id']} has invalid public memory signal"
        )
    return signal


def _build_store() -> MemoryStore:
    return MemoryStore(
        learned_core=CMSMemoryCore(
            mode="mlp",
            d_in=4,
            d_hidden=8,
            variant="nested",
            session_cadence=2,
            background_cadence=4,
            pe_features_enabled=True,
            replay_window_sizes={
                "online-fast": 8,
                "session-medium": 4,
                "background-slow": 2,
            },
        )
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _mean_abs_error(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Gate 6 vector dimensions differ: {len(left)} != {len(right)}"
        )
    return _mean(
        tuple(abs(a - b) for a, b in zip(left, right, strict=True))
    )


def _quality(error: float) -> float:
    return 1.0 - min(max(error, 0.0) / GATE6_ERROR_SCALE, 1.0)


def _context_target(
    records: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    signals = tuple(_public_signal(record) for record in records)
    dim = len(signals[0])
    if any(len(signal) != dim for signal in signals):
        raise ValueError("Gate 6 context signal dimensions drifted")
    return tuple(
        _mean(tuple(signal[index] for signal in signals))
        for index in range(dim)
    )


def _group_contexts(
    records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record["partition"] != partition:
            continue
        grouped.setdefault(str(record["context_id"]), []).append(record)
    return tuple(
        (context_id, tuple(context_records))
        for context_id, context_records in grouped.items()
    )


def _train_checkpoint(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    checkpoint_id: str,
) -> tuple[MemoryStore, MemoryStoreCheckpoint]:
    store = _build_store()
    for index, record in enumerate(records):
        store.observe_replay_signal(
            signal=_public_signal(record),
            timestamp_ms=seed * 1_000_000 + index,
            prediction_error=typed_prediction_error_snapshot(record),
        )
    return store, store.create_checkpoint(checkpoint_id=checkpoint_id)


def _train_user_checkpoints(
    train_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[tuple[str, MemoryStoreCheckpoint], ...]:
    groups = _group_contexts(train_records, partition="trace-train")
    result: list[tuple[str, MemoryStoreCheckpoint]] = []
    for ordinal, (_context_id, records) in enumerate(groups):
        if len(records) != 30:
            raise ValueError(
                "Gate 6 donor context must contain exactly 30 records"
            )
        user_id = str(records[0]["user_id"])
        _, checkpoint = _train_checkpoint(
            records,
            seed=seed + ordinal,
            checkpoint_id=f"gate6-{seed}-donor-{user_id}",
        )
        result.append((user_id, checkpoint))
    if len(result) != 10:
        raise ValueError(
            f"Gate 6 requires 10 train-user donor checkpoints, got {len(result)}"
        )
    return tuple(result)


def _checkpoint_targets(
    checkpoint: MemoryStoreCheckpoint,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if checkpoint.cms_state is None:
        raise ValueError("Gate 6 checkpoint lacks CMS state")
    online = checkpoint.cms_state.nested_online_init_target
    session = checkpoint.cms_state.nested_session_init_target
    if not online or not session:
        raise ValueError("Gate 6 checkpoint lacks nested targets")
    return online, session


def _initialization_args(
    *,
    arm: str,
    seed: int,
    context_ordinal: int,
    donor_checkpoints: Sequence[tuple[str, MemoryStoreCheckpoint]],
) -> tuple[
    str,
    int | None,
    tuple[tuple[float, ...], tuple[float, ...]] | None,
    tuple[str, ...],
]:
    if arm in GATE6_PRIMARY_ARMS:
        return (
            arm,
            seed * 10_000 + context_ordinal if arm == "random-init" else None,
            None,
            ("global-train-checkpoint",),
        )
    donor_index = context_ordinal % len(donor_checkpoints)
    if arm == "swapped-user-slow-state":
        donor_index = (donor_index + 1) % len(donor_checkpoints)
    if arm != "paired-user-slow-state" and arm != "swapped-user-slow-state":
        raise ValueError(f"Unknown Gate 6 arm {arm!r}")
    donor_user, donor_checkpoint = donor_checkpoints[donor_index]
    return (
        "external-meta-init",
        None,
        _checkpoint_targets(donor_checkpoint),
        (donor_user,),
    )


def _fact_leakage_count(
    *,
    train_records: Sequence[Mapping[str, Any]],
    target_records: Sequence[Mapping[str, Any]],
    initialization_targets: Sequence[Sequence[float]],
) -> int:
    fields = ("user_id", "context_id", "knowledge_key")
    train_provenance = {
        str(record[field])
        for record in train_records
        for field in fields
    }
    target_provenance = {
        str(record[field])
        for record in target_records
        for field in fields
    }
    overlap = train_provenance & target_provenance
    invalid_numeric_fields = sum(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for target in initialization_targets
        for value in target
    )
    return len(overlap) + invalid_numeric_fields


def _run_episode(
    *,
    store: MemoryStore,
    train_checkpoint: MemoryStoreCheckpoint,
    train_records: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    seed: int,
    partition: str,
    context_ordinal: int,
    arm: str,
    donor_checkpoints: Sequence[tuple[str, MemoryStoreCheckpoint]],
) -> tuple[Gate6EpisodeMetrics, list[dict[str, Any]], dict[str, Any]]:
    store.restore_checkpoint(train_checkpoint)
    restored_before = store.create_checkpoint(
        checkpoint_id=train_checkpoint.checkpoint_id
    )
    if restored_before != train_checkpoint:
        raise RuntimeError("Gate 6 failed to restore the train checkpoint")
    context_id = str(records[0]["context_id"])
    user_id = str(records[0]["user_id"])
    target = _context_target(records)
    mode, random_seed, external_targets, provenance = _initialization_args(
        arm=arm,
        seed=seed,
        context_ordinal=context_ordinal,
        donor_checkpoints=donor_checkpoints,
    )
    evidence = store.initialize_nested_context_for_evidence(
        mode=mode,
        reason=f"gate6:{partition}:{context_id}:{arm}",
        timestamp_ms=seed * 1_000_000 + 500_000 + context_ordinal,
        random_seed=random_seed,
        external_targets=external_targets,
    )
    if evidence is None:
        raise RuntimeError("Gate 6 nested initialization was not applied")
    initial_snapshot = store.snapshot(
        retrieved_entries=(),
        active_subject_scope=(user_id,),
    )
    if initial_snapshot.cms_state is None:
        raise RuntimeError("Gate 6 initialization lacks public CMS state")
    public_initial = initial_snapshot.cms_state.online_fast.vector
    if public_initial != evidence.online_after:
        raise RuntimeError("Gate 6 owner evidence and public state disagree")
    errors = [_mean_abs_error(public_initial, target)]
    rows: list[dict[str, Any]] = [
        {
            "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
            "seed": seed,
            "partition": partition,
            "context_id": context_id,
            "user_id": user_id,
            "arm": arm,
            "adaptation_step": 0,
            "transition_id": None,
            "prediction_ref": None,
            "record_sha256": None,
            "target_error": errors[0],
            "adaptation_quality": _quality(errors[0]),
            "online_fast_vector": list(public_initial),
            "initialization_mode": evidence.mode,
            "initialization_target": list(evidence.online_target),
            "initialization_provenance": list(provenance),
            "slow_state_unchanged": evidence.slow_state_unchanged,
            "parameter_state_unchanged": evidence.parameter_state_unchanged,
        }
    ]
    mutation_count = 0
    for step, record in enumerate(records, start=1):
        store.observe_replay_signal(
            signal=_public_signal(record),
            timestamp_ms=(
                seed * 1_000_000
                + 600_000
                + context_ordinal * 100
                + step
            ),
            prediction_error=typed_prediction_error_snapshot(record),
        )
        snapshot = store.snapshot(
            retrieved_entries=(),
            active_subject_scope=(user_id,),
        )
        if snapshot.cms_state is None:
            raise RuntimeError("Gate 6 adaptation lacks public CMS state")
        online = snapshot.cms_state.online_fast.vector
        error = _mean_abs_error(online, target)
        errors.append(error)
        mutated = bool(record["substrate"]["mutation_applied"])
        mutation_count += int(mutated)
        rows.append(
            {
                "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
                "seed": seed,
                "partition": partition,
                "context_id": context_id,
                "user_id": user_id,
                "arm": arm,
                "adaptation_step": step,
                "transition_id": record["transition_id"],
                "prediction_ref": record["lineage"]["prediction_ref"],
                "record_sha256": record["record_sha256"],
                "target_error": error,
                "adaptation_quality": _quality(error),
                "online_fast_vector": list(online),
                "initialization_mode": evidence.mode,
                "initialization_target": list(evidence.online_target),
                "initialization_provenance": list(provenance),
                "slow_state_unchanged": evidence.slow_state_unchanged,
                "parameter_state_unchanged": evidence.parameter_state_unchanged,
            }
        )
    steps_to_target = next(
        (
            index
            for index, error in enumerate(errors)
            if error <= GATE6_TARGET_ERROR
        ),
        len(records) + 1,
    )
    early_points = errors[:GATE6_EARLY_K]
    leakage_count = _fact_leakage_count(
        train_records=train_records,
        target_records=records,
        initialization_targets=(
            evidence.online_target,
            evidence.session_target,
        ),
    )
    store.restore_checkpoint(train_checkpoint)
    rollback_checkpoint = store.create_checkpoint(
        checkpoint_id=train_checkpoint.checkpoint_id
    )
    rollback_exact = rollback_checkpoint == train_checkpoint
    metrics = Gate6EpisodeMetrics(
        seed=seed,
        partition=partition,
        context_id=context_id,
        user_id=user_id,
        arm=arm,
        episode_length=len(records),
        steps_to_target=steps_to_target,
        early_adaptation_auc=_mean(
            tuple(_quality(error) for error in early_points)
        ),
        final_error=errors[-1],
        final_quality=_quality(errors[-1]),
        initial_error=errors[0],
        initialization_changed_fast_state=(
            evidence.online_after != evidence.online_before
        ),
        slow_state_unchanged=evidence.slow_state_unchanged,
        parameter_state_unchanged=evidence.parameter_state_unchanged,
        lineage_complete=all(
            record["transition_id"]
            and record["lineage"]["prediction_ref"]
            and record["record_sha256"]
            for record in records
        ),
        frozen_substrate_mutation_count=mutation_count,
        fact_leakage_count=leakage_count,
        checkpoint_restore_exact=rollback_exact,
    )
    rollback = {
        "seed": seed,
        "partition": partition,
        "context_id": context_id,
        "arm": arm,
        "checkpoint_restore_exact": rollback_exact,
        "rollback_target": train_checkpoint.checkpoint_id,
    }
    return metrics, rows, rollback


def run_gate6_seed(
    *,
    records: Sequence[Mapping[str, Any]],
    seed: int,
    partition: str,
) -> tuple[
    tuple[Gate6EpisodeMetrics, ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    expected_count = (
        _HELDOUT_COUNT
        if partition == "trace-heldout-context"
        else _LOCKED_COUNT
        if partition == "trace-locked-confirmation"
        else 0
    )
    if expected_count == 0:
        raise ValueError(f"Unsupported Gate 6 target partition {partition!r}")
    train = tuple(
        record for record in records if record["partition"] == "trace-train"
    )
    target_records = tuple(
        record for record in records if record["partition"] == partition
    )
    if len(train) != _TRAIN_COUNT or len(target_records) != expected_count:
        raise ValueError(
            f"Gate 6 seed {seed} partition counts drifted: "
            f"train={len(train)} target={len(target_records)}"
        )
    train_users = {str(record["user_id"]) for record in train}
    train_contexts = {str(record["context_id"]) for record in train}
    target_users = {str(record["user_id"]) for record in target_records}
    target_contexts = {str(record["context_id"]) for record in target_records}
    if train_users & target_users or train_contexts & target_contexts:
        raise ValueError("Gate 6 train/target user or context overlap detected")
    store, train_checkpoint = _train_checkpoint(
        train,
        seed=seed,
        checkpoint_id=f"gate6-{seed}-global-train",
    )
    donors = _train_user_checkpoints(train, seed=seed)
    metrics: list[Gate6EpisodeMetrics] = []
    state_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    contexts = _group_contexts(target_records, partition=partition)
    for context_ordinal, (_context_id, context_records) in enumerate(contexts):
        for arm in (*GATE6_PRIMARY_ARMS, *GATE6_DIAGNOSTIC_ARMS):
            episode_metrics, rows, rollback = _run_episode(
                store=store,
                train_checkpoint=train_checkpoint,
                train_records=train,
                records=context_records,
                seed=seed,
                partition=partition,
                context_ordinal=context_ordinal,
                arm=arm,
                donor_checkpoints=donors,
            )
            metrics.append(episode_metrics)
            state_rows.extend(rows)
            rollback_rows.append(rollback)
    return tuple(metrics), tuple(state_rows), tuple(rollback_rows)


def _seed_arm_mean(
    metrics: Sequence[Gate6EpisodeMetrics],
    *,
    seed: int,
    arm: str,
    field: str,
) -> float:
    values = tuple(
        float(getattr(metric, field))
        for metric in metrics
        if metric.seed == seed and metric.arm == arm
    )
    if not values:
        raise ValueError(
            f"Gate 6 lacks metrics for seed={seed} arm={arm} field={field}"
        )
    return _mean(values)


def compare_gate6_arms(
    metrics: Sequence[Gate6EpisodeMetrics],
) -> tuple[
    tuple[Gate6ArmComparison, ...],
    dict[str, bool],
    dict[str, float | bool],
]:
    comparisons: list[Gate6ArmComparison] = []
    for control in GATE6_PRIMARY_ARMS[1:]:
        step_seed_gains = tuple(
            _seed_arm_mean(
                metrics,
                seed=seed,
                arm=control,
                field="steps_to_target",
            )
            - _seed_arm_mean(
                metrics,
                seed=seed,
                arm="meta-init",
                field="steps_to_target",
            )
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        auc_seed_gains = tuple(
            _seed_arm_mean(
                metrics,
                seed=seed,
                arm="meta-init",
                field="early_adaptation_auc",
            )
            - _seed_arm_mean(
                metrics,
                seed=seed,
                arm=control,
                field="early_adaptation_auc",
            )
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        final_error_seed_deltas = tuple(
            _seed_arm_mean(
                metrics,
                seed=seed,
                arm="meta-init",
                field="final_error",
            )
            - _seed_arm_mean(
                metrics,
                seed=seed,
                arm=control,
                field="final_error",
            )
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        step_gain = _mean(step_seed_gains)
        auc_gain = _mean(auc_seed_gains)
        final_delta = _mean(final_error_seed_deltas)
        minimum_effect = (
            step_gain >= GATE6_MIN_STEP_GAIN
            and all(gain > 0.0 for gain in step_seed_gains)
        ) or (
            auc_gain >= GATE6_MIN_AUC_GAIN
            and all(gain > 0.0 for gain in auc_seed_gains)
        )
        final_non_inferior = (
            final_delta <= GATE6_FINAL_ERROR_TOLERANCE
            and all(
                delta <= GATE6_FINAL_ERROR_TOLERANCE
                for delta in final_error_seed_deltas
            )
        )
        comparisons.append(
            Gate6ArmComparison(
                control_arm=control,
                aggregate_step_gain=step_gain,
                aggregate_auc_gain=auc_gain,
                step_seed_gains=step_seed_gains,
                auc_seed_gains=auc_seed_gains,
                aggregate_final_error_delta=final_delta,
                final_error_seed_deltas=final_error_seed_deltas,
                minimum_effect_passed=minimum_effect,
                final_error_non_inferior=final_non_inferior,
            )
        )
    primary = tuple(
        metric for metric in metrics if metric.arm in GATE6_PRIMARY_ARMS
    )
    meta_by_episode = {
        (metric.seed, metric.context_id): metric
        for metric in metrics
        if metric.arm == "meta-init"
    }
    controls_by_episode: dict[
        tuple[int, str], list[Gate6EpisodeMetrics]
    ] = {}
    for metric in metrics:
        if metric.arm in GATE6_PRIMARY_ARMS[1:]:
            controls_by_episode.setdefault(
                (metric.seed, metric.context_id),
                [],
            ).append(metric)
    negative_transfer_count = sum(
        meta.final_error
        > min(control.final_error for control in controls_by_episode[key])
        + GATE6_FINAL_ERROR_TOLERANCE
        for key, meta in meta_by_episode.items()
    )
    negative_transfer_rate = (
        negative_transfer_count / len(meta_by_episode)
        if meta_by_episode
        else 1.0
    )
    paired_auc = _mean(
        tuple(
            metric.early_adaptation_auc
            for metric in metrics
            if metric.arm == "paired-user-slow-state"
        )
    )
    swapped_auc = _mean(
        tuple(
            metric.early_adaptation_auc
            for metric in metrics
            if metric.arm == "swapped-user-slow-state"
        )
    )
    paired_steps = _mean(
        tuple(
            float(metric.steps_to_target)
            for metric in metrics
            if metric.arm == "paired-user-slow-state"
        )
    )
    swapped_steps = _mean(
        tuple(
            float(metric.steps_to_target)
            for metric in metrics
            if metric.arm == "swapped-user-slow-state"
        )
    )
    diagnostic = {
        "paired_minus_swapped_auc": paired_auc - swapped_auc,
        "swapped_minus_paired_steps": swapped_steps - paired_steps,
        "user_related_prior_supported": not (
            abs(paired_auc - swapped_auc) < 0.01
            and abs(swapped_steps - paired_steps) < 1.0
        ),
        "negative_transfer_rate": negative_transfer_rate,
    }
    gates = {
        "all_primary_arms_all_seeds_present": {
            (metric.seed, metric.arm) for metric in primary
        }
        == {
            (seed, arm)
            for seed in SHARED_SETTLED_TRACE_SEEDS
            for arm in GATE6_PRIMARY_ARMS
        },
        "all_locked_episode_counts_exact": all(
            sum(
                metric.seed == seed and metric.arm == arm
                for metric in metrics
            )
            == 3
            for seed in SHARED_SETTLED_TRACE_SEEDS
            for arm in (*GATE6_PRIMARY_ARMS, *GATE6_DIAGNOSTIC_ARMS)
        ),
        "lineage_complete": all(metric.lineage_complete for metric in metrics),
        "fact_leakage_zero": all(
            metric.fact_leakage_count == 0 for metric in metrics
        ),
        "frozen_substrate_mutation_zero": all(
            metric.frozen_substrate_mutation_count == 0
            for metric in metrics
        ),
        "slow_and_parameter_state_unchanged_by_initialization": all(
            metric.slow_state_unchanged and metric.parameter_state_unchanged
            for metric in metrics
        ),
        "checkpoint_restore_exact": all(
            metric.checkpoint_restore_exact for metric in metrics
        ),
        "meta_minimum_effect_all_controls": all(
            comparison.minimum_effect_passed
            for comparison in comparisons
        ),
        "meta_final_error_non_inferior_all_controls": all(
            comparison.final_error_non_inferior
            for comparison in comparisons
        ),
        "negative_transfer_zero": (
            negative_transfer_rate <= GATE6_NEGATIVE_TRANSFER_LIMIT
        ),
    }
    return tuple(comparisons), gates, diagnostic


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
                "context_id": record["context_id"],
                "user_id": record["user_id"],
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
                    "knowledge_key": record["knowledge_key"],
                }
            )
            result["credit.jsonl"].append(
                {**lineage, "credit_snapshot": record["credit_snapshot"]}
            )
            result["action_selection.jsonl"].append(
                {**lineage, "action_selection": record["action_selection"]}
            )
    return result


def _source_diversity_diagnostics(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_partition: dict[str, dict[str, Any]] = {}
    for partition in (
        "trace-train",
        "trace-heldout-context",
        "trace-locked-confirmation",
    ):
        context_signals: dict[str, list[tuple[float, ...]]] = {}
        all_signals: list[tuple[float, ...]] = []
        for records in records_by_seed.values():
            for record in records:
                if record["partition"] != partition:
                    continue
                signal = _public_signal(record)
                all_signals.append(signal)
                context_signals.setdefault(
                    str(record["context_id"]),
                    [],
                ).append(signal)
        centroids = tuple(
            tuple(
                _mean(tuple(signal[index] for signal in signals))
                for index in range(len(signals[0]))
            )
            for signals in context_signals.values()
        )
        pairwise = tuple(
            _mean_abs_error(left, right)
            for left_index, left in enumerate(centroids)
            for right in centroids[left_index + 1 :]
        )
        by_partition[partition] = {
            "row_count": len(all_signals),
            "context_count": len(context_signals),
            "unique_signal_count": len(set(all_signals)),
            "context_centroid_max_pairwise_mae": max(
                pairwise,
                default=0.0,
            ),
            "context_centroid_mean_pairwise_mae": _mean(pairwise),
        }
    return by_partition


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


def run_gate6_development_probe(
    *,
    trace_root: str | Path,
    seed: int = 401,
) -> dict[str, Any]:
    """Run the preregistered development arm matrix without reading locked."""

    source = Path(trace_root)
    records = _load_trace_prefix(
        source / f"seed_{seed}" / "transitions.jsonl",
        count=_TRAIN_COUNT + _HELDOUT_COUNT,
    )
    validate_shared_trace_prefix(
        records=records,
        expected_plans=build_shared_trace_plans(seed),
    )
    metrics, _rows, _rollback = run_gate6_seed(
        records=records,
        seed=seed,
        partition="trace-heldout-context",
    )
    by_arm = {
        arm: {
            "steps_to_target": _mean(
                tuple(
                    float(metric.steps_to_target)
                    for metric in metrics
                    if metric.arm == arm
                )
            ),
            "early_adaptation_auc": _mean(
                tuple(
                    metric.early_adaptation_auc
                    for metric in metrics
                    if metric.arm == arm
                )
            ),
            "final_error": _mean(
                tuple(
                    metric.final_error
                    for metric in metrics
                    if metric.arm == arm
                )
            ),
        }
        for arm in (*GATE6_PRIMARY_ARMS, *GATE6_DIAGNOSTIC_ARMS)
    }
    return {
        "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
        "seed": seed,
        "partition": "trace-heldout-context",
        "locked_partition_read": False,
        "metrics_by_arm": by_arm,
    }


def export_gate6_meta_init_bundle(
    *,
    trace_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Consume locked exactly once and write the formal Gate 6 packet."""

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
        raise ValueError("Gate 6 source trace is not admitted for consumption")
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        records = load_shared_trace_records(
            source / f"seed_{seed}" / "transitions.jsonl"
        )
        if len(records) != _TRAIN_COUNT + _HELDOUT_COUNT + _LOCKED_COUNT:
            raise ValueError(f"Gate 6 seed {seed} source count drifted")
        validate_shared_trace_prefix(
            records=records,
            expected_plans=build_shared_trace_plans(seed),
        )
        records_by_seed[seed] = records
    heldout_metrics: list[Gate6EpisodeMetrics] = []
    locked_metrics: list[Gate6EpisodeMetrics] = []
    state_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        for partition, target_metrics in (
            ("trace-heldout-context", heldout_metrics),
            ("trace-locked-confirmation", locked_metrics),
        ):
            metrics, rows, rollbacks = run_gate6_seed(
                records=records_by_seed[seed],
                seed=seed,
                partition=partition,
            )
            target_metrics.extend(metrics)
            state_rows.extend(rows)
            rollback_rows.extend(rollbacks)
    comparisons, gates, diagnostic = compare_gate6_arms(locked_metrics)
    train_users = {
        str(record["user_id"])
        for records in records_by_seed.values()
        for record in records
        if record["partition"] == "trace-train"
    }
    target_users = {
        str(record["user_id"])
        for records in records_by_seed.values()
        for record in records
        if record["partition"] != "trace-train"
    }
    train_contexts = {
        str(record["context_id"])
        for records in records_by_seed.values()
        for record in records
        if record["partition"] == "trace-train"
    }
    target_contexts = {
        str(record["context_id"])
        for records in records_by_seed.values()
        for record in records
        if record["partition"] != "trace-train"
    }
    gates["train_target_user_context_overlap_zero"] = not (
        train_users & target_users or train_contexts & target_contexts
    )
    gates["background_latency_published_separately"] = all(
        set(record["latency"])
        >= {
            "prediction_turn_ms",
            "settlement_turn_ms",
            "session_post_slow_job_ms",
        }
        for records in records_by_seed.values()
        for record in records
    )
    budget_snapshot = _build_store().snapshot(retrieved_entries=()).cms_state
    if budget_snapshot is None:
        raise RuntimeError("Gate 6 matched budget store lacks CMS state")
    matched_parameter_count = sum(
        band.mlp_param_count
        for band in (
            budget_snapshot.online_fast,
            budget_snapshot.session_medium,
            budget_snapshot.background_slow,
        )
    )
    parameter_counts = {
        arm: matched_parameter_count for arm in GATE6_PRIMARY_ARMS
    }
    gates["matched_parameter_budget"] = len(set(parameter_counts.values())) == 1
    invalid_gate_names = (
        "all_primary_arms_all_seeds_present",
        "all_locked_episode_counts_exact",
        "lineage_complete",
        "fact_leakage_zero",
        "frozen_substrate_mutation_zero",
        "slow_and_parameter_state_unchanged_by_initialization",
        "checkpoint_restore_exact",
        "train_target_user_context_overlap_zero",
        "background_latency_published_separately",
        "matched_parameter_budget",
    )
    invalid_gates = tuple(
        name for name in invalid_gate_names if not gates[name]
    )
    causal_passed = (
        gates["meta_minimum_effect_all_controls"]
        and gates["meta_final_error_non_inferior_all_controls"]
        and gates["negative_transfer_zero"]
    )
    status = (
        "invalid"
        if invalid_gates
        else "causal-supported"
        if causal_passed
        else "not-supported"
    )
    manifest = {
        "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
        "suite_id": "gate6-meta-init",
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
            "memory_owner": "matched nested CMS episode replay",
            "substrate_mutation": "DISABLED",
        },
        "seed_schedule": list(SHARED_SETTLED_TRACE_SEEDS),
        "primary_arm_schedule": list(GATE6_PRIMARY_ARMS),
        "diagnostic_arm_schedule": list(GATE6_DIAGNOSTIC_ARMS),
        "partition_schedule": {
            "trace-train": _TRAIN_COUNT,
            "trace-heldout-context": _HELDOUT_COUNT,
            "trace-locked-confirmation": _LOCKED_COUNT,
        },
        "primary_partition": "trace-locked-confirmation",
        "target_error": GATE6_TARGET_ERROR,
        "early_k": GATE6_EARLY_K,
        "error_scale": GATE6_ERROR_SCALE,
        "minimum_step_gain": GATE6_MIN_STEP_GAIN,
        "minimum_auc_gain": GATE6_MIN_AUC_GAIN,
        "final_error_tolerance": GATE6_FINAL_ERROR_TOLERANCE,
        "negative_transfer_limit": GATE6_NEGATIVE_TRANSFER_LIMIT,
        "signal_source": (
            "memory_snapshot.attribute_summary.latest."
            "substrate_feature_digest"
        ),
        "required_files": list(GATE6_REQUIRED_FILES),
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
        "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
        "heldout_metrics": [asdict(metric) for metric in heldout_metrics],
        "locked_metrics": [asdict(metric) for metric in locked_metrics],
        "locked_comparisons": [
            asdict(comparison) for comparison in comparisons
        ],
        "locked_diagnostics": diagnostic,
        "parameter_counts": parameter_counts,
        "source_diversity_diagnostics": _source_diversity_diagnostics(
            records_by_seed
        ),
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    verdict = {
        "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
        "gate_scope": "Gate 6 nested meta-init slow-to-fast transfer",
        "status": status,
        "mechanism_passed": not invalid_gates,
        "causal_passed": status == "causal-supported",
        "user_related_prior_supported": diagnostic[
            "user_related_prior_supported"
        ],
        "claim": (
            "nested meta-init improves locked cross-context adaptation"
            if status == "causal-supported"
            else (
                "nested initialization is runnable, auditable, and rollback-capable"
                if status == "not-supported"
                else "Gate 6 evidence packet is invalid"
            )
        ),
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "invalid_gates": list(invalid_gates),
        "locked_partition_consumed_once": True,
        "same_locked_partition_rerun_allowed": False,
        "gate5_pareto_claim_inherited": False,
    }
    rollback = {
        "schema_version": GATE6_META_INIT_SCHEMA_VERSION,
        "passed": all(
            row["checkpoint_restore_exact"] for row in rollback_rows
        ),
        "episodes": rollback_rows,
        "production_rollback": {
            "restore": "MemoryStoreCheckpoint",
            "disable": "nested_profile=False",
        },
        "substrate_mutated": False,
    }
    for name, rows in _source_rows(records_by_seed).items():
        _write_jsonl(target / name, rows)
    _write_jsonl(target / "state_diff.jsonl", state_rows)
    _write_json(target / "manifest.yaml", manifest)
    _write_json(target / "ablation_results.json", ablation)
    _write_json(target / "promotion_verdict.json", verdict)
    _write_json(target / "rollback_evidence.json", rollback)
    report = [
        "# Gate 6 nested meta-init evidence",
        "",
        f"- status: `{status}`",
        f"- mechanism passed: `{not invalid_gates}`",
        f"- causal passed: `{status == 'causal-supported'}`",
        "- locked partition consumed once: `True`",
        (
            "- user-related prior supported: "
            f"`{diagnostic['user_related_prior_supported']}`"
        ),
        "",
        "## Locked meta-init vs controls",
        "",
    ]
    report.extend(
        (
            f"- `{comparison.control_arm}`: step gain "
            f"`{comparison.aggregate_step_gain:.6f}`, AUC gain "
            f"`{comparison.aggregate_auc_gain:.6f}`, final-error delta "
            f"`{comparison.aggregate_final_error_delta:.6f}`, minimum effect "
            f"`{comparison.minimum_effect_passed}`"
        )
        for comparison in comparisons
    )
    report.extend(
        (
            "",
            "## Claim boundary",
            "",
            (
                "- This packet tests only nested initialization. It does not "
                "inherit or reverse the Gate 5 CMS Pareto NO-GO."
            ),
            (
                "- The same locked partition may not be tuned against or rerun "
                "after a failed preregistered gate."
            ),
            "",
        )
    )
    (target / "report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    written = tuple(target / name for name in GATE6_REQUIRED_FILES)
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(f"Gate 6 bundle missing required files {missing!r}")
    verify_gate6_meta_init_bundle(target)
    return written


def verify_gate6_meta_init_bundle(
    output_dir: str | Path,
) -> dict[str, Any]:
    target = Path(output_dir)
    missing = tuple(
        name for name in GATE6_REQUIRED_FILES if not (target / name).is_file()
    )
    if missing:
        raise ValueError(f"Gate 6 bundle is missing {missing!r}")
    manifest = json.loads((target / "manifest.yaml").read_text(encoding="utf-8"))
    ablation = json.loads(
        (target / "ablation_results.json").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    rollback = json.loads(
        (target / "rollback_evidence.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] != GATE6_META_INIT_SCHEMA_VERSION:
        raise ValueError("Gate 6 manifest schema drifted")
    if manifest["primary_arm_schedule"] != list(GATE6_PRIMARY_ARMS):
        raise ValueError("Gate 6 primary arm schedule drifted")
    if not verdict["locked_partition_consumed_once"]:
        raise ValueError("Gate 6 locked partition consumption is not attested")
    if verdict["same_locked_partition_rerun_allowed"]:
        raise ValueError("Gate 6 illegally permits locked rerun")
    if not rollback["passed"]:
        raise ValueError("Gate 6 rollback evidence failed")
    locked_metrics = ablation["locked_metrics"]
    expected_episode_count = (
        len(SHARED_SETTLED_TRACE_SEEDS)
        * 3
        * (len(GATE6_PRIMARY_ARMS) + len(GATE6_DIAGNOSTIC_ARMS))
    )
    if len(locked_metrics) != expected_episode_count:
        raise ValueError(
            "Gate 6 locked episode count drifted: "
            f"{len(locked_metrics)} != {expected_episode_count}"
        )
    return {
        "status": verdict["status"],
        "mechanism_passed": verdict["mechanism_passed"],
        "causal_passed": verdict["causal_passed"],
        "user_related_prior_supported": verdict[
            "user_related_prior_supported"
        ],
        "locked_episode_count": len(locked_metrics),
        "required_file_count": len(GATE6_REQUIRED_FILES),
        "locked_partition_consumed_once": True,
    }


__all__ = [
    "GATE6_DIAGNOSTIC_ARMS",
    "GATE6_EARLY_K",
    "GATE6_ERROR_SCALE",
    "GATE6_FINAL_ERROR_TOLERANCE",
    "GATE6_META_INIT_SCHEMA_VERSION",
    "GATE6_MIN_AUC_GAIN",
    "GATE6_MIN_STEP_GAIN",
    "GATE6_NEGATIVE_TRANSFER_LIMIT",
    "GATE6_PRIMARY_ARMS",
    "GATE6_REQUIRED_FILES",
    "GATE6_TARGET_ERROR",
    "Gate6ArmComparison",
    "Gate6EpisodeMetrics",
    "compare_gate6_arms",
    "export_gate6_meta_init_bundle",
    "run_gate6_development_probe",
    "run_gate6_seed",
    "verify_gate6_meta_init_bundle",
]
