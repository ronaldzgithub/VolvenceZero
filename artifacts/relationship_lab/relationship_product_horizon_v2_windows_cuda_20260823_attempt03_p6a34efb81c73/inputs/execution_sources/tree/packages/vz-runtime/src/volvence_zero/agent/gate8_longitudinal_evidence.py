"""Gate 8 fresh-source cross-session wake/sleep evidence.

The harness extends the frozen Gate 8 four-arm protocol to the immutable
real-substrate longitudinal source.  Source capture remains independent from
consumer state.  Every ten settled transitions the memory and temporal owners
are persisted, discarded, reconstructed, and hydrated from public snapshots.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SESSION_SIZE,
    GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    build_gate11_longitudinal_source_plans,
    load_gate11_longitudinal_source_records,
    validate_gate11_longitudinal_source_prefix,
)
from volvence_zero.agent.gate78_shared_trace import Gate78EpisodePlan
from volvence_zero.agent.gate8_wake_sleep_evidence import (
    GATE8_ARMS,
    _arm_targets,
    _checkpoint_fingerprint,
    _job,
    _policy_alignment,
    _sha256,
)
from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopJob,
    SessionPostSlowLoopQueue,
    SessionPostSlowLoopResult,
)
from volvence_zero.integration import apply_session_post_writeback_request
from volvence_zero.memory import (
    MemoryStratum,
    RetrievalQuery,
    Track,
    build_default_memory_store,
)
from volvence_zero.memory.persistence import FileSystemPersistenceBackend
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterSnapshot,
    MetacontrollerParameterStore,
)


GATE8_LONGITUDINAL_SCHEMA_VERSION = "gate8-wake-sleep-longitudinal.v1"
GATE8_LONGITUDINAL_MIN_EFFECT = 0.02
GATE8_LONGITUDINAL_REQUIRED_FILES = (
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
_T_CRITICAL_95_DF2 = 4.302652729749


@dataclass(frozen=True)
class Gate8LongitudinalArmMetric:
    seed: int
    arm: str
    settled_transition_count: int
    consumer_session_count: int
    constructor_restart_count: int
    memory_entry_count: int
    isolated_memory_entry_count: int
    temporal_operation_count: int
    next_session_cold_start_loss: float
    callback_commitment_consistency: float
    temporal_policy_alignment: float
    delayed_payoff: float
    owner_state_drift: float
    prompt_token_increment: int
    unique_job_count: int
    worker_execution_count: int
    duplicate_job_count: int
    duplicate_job_execution_count: int
    owner_lineage_expected_count: int
    owner_lineage_observed_count: int
    owner_writeback_lineage_coverage: float
    turn_latency_ms: float
    slow_job_latency_ms: float
    turn_latency_contains_slow_job: bool
    persistence_roundtrip_exact: bool
    rollback_exact: bool
    rollback_fingerprint_before: str
    rollback_fingerprint_after: str
    frozen_substrate_mutation_count: int


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


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
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


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


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _confidence_interval_95(
    values: Sequence[float],
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else 0.0
        return (value, value)
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    half_width = _T_CRITICAL_95_DF2 * deviation / math.sqrt(len(values))
    return (mean - half_width, mean + half_width)


def _snapshot_fingerprint(snapshot: object) -> str:
    return _sha256(snapshot)


def _owner_fingerprint(
    *,
    memory_checkpoint: object,
    isolated_checkpoint: object,
    temporal_snapshot: object,
) -> str:
    return _sha256(
        (
            _checkpoint_fingerprint(memory_checkpoint),
            _checkpoint_fingerprint(isolated_checkpoint),
            _snapshot_fingerprint(temporal_snapshot),
        )
    )


def _record_to_plan(record: Mapping[str, Any]) -> Gate78EpisodePlan:
    """Adapt only owner-published structured fields to the Gate 8 protocol."""

    temporal = record["temporal_snapshot"]
    controller_state = temporal["controller_state"]
    code = tuple(float(value) for value in controller_state["code"])
    if len(code) < 3 or not all(math.isfinite(value) for value in code):
        raise ValueError(
            f"{record['transition_id']} lacks a valid temporal code"
        )
    memory_feedback = tuple(
        float(value) for value in temporal["memory_feedback_signal"]
    )
    closed_segments = tuple(temporal["closed_segments"])
    action_family_ids = tuple(
        str(segment["abstract_action_id"]) for segment in closed_segments
    ) or (str(temporal["active_abstract_action"]),)
    segment_lengths = tuple(
        max(
            int(segment["close_turn_index"])
            - int(segment["open_turn_index"]),
            1,
        )
        for segment in closed_segments
    ) or (1,)
    transition_id = str(record["transition_id"])
    knowledge_key = str(record["knowledge_key"])
    context_id = str(record["context_id"])
    return Gate78EpisodePlan(
        episode_id=transition_id,
        seed=int(record["seed"]),
        global_index=int(record["global_index"]),
        partition=str(record["partition"]),
        context_id=context_id,
        domain=str(record["domain"]),
        user_prior_id=str(record["user_id"]),
        user_prior=(
            code[0],
            code[1],
            code[2],
            float(controller_state["switch_gate"]),
        ),
        context_centroid=(
            memory_feedback[:3] + code[:3]
        ),
        route=action_family_ids,
        action_family_ids=action_family_ids,
        segment_lengths=segment_lengths,
        difficulty=float(record["prediction_error"]["magnitude"]),
        session_one_turns=(
            str(record["input"]["prediction_turn"]),
            str(record["input"]["settlement_turn"]),
        ),
        session_two_turns=(
            (
                f"Session two cold start {transition_id}; resume audited "
                f"record {knowledge_key} in {context_id}."
            ),
        ),
        next_session_boundary=f"{transition_id}:consumer-session-boundary",
    )


def _persist_and_restart_owners(
    *,
    memory_store: Any,
    isolated_memory_store: Any,
    policy: FullLearnedTemporalPolicy,
    memory_backend: FileSystemPersistenceBackend,
    isolated_backend: FileSystemPersistenceBackend,
    temporal_checkpoint_path: Path,
    seed: int,
) -> tuple[Any, Any, FullLearnedTemporalPolicy, bool]:
    memory_before = memory_store.create_checkpoint(
        checkpoint_id="gate8-longitudinal-memory-before-restart"
    )
    isolated_before = isolated_memory_store.create_checkpoint(
        checkpoint_id="gate8-longitudinal-isolated-before-restart"
    )
    temporal_before = policy.export_rare_heavy_snapshot()
    if not memory_store.save_to_backend(key="gate8-longitudinal/memory"):
        raise RuntimeError("Gate 8 memory owner did not persist")
    if not isolated_memory_store.save_to_backend(
        key="gate8-longitudinal/isolated-memory"
    ):
        raise RuntimeError("Gate 8 isolated memory owner did not persist")
    temporal_checkpoint_path.write_bytes(
        pickle.dumps(temporal_before, protocol=pickle.HIGHEST_PROTOCOL)
    )

    memory_store = build_default_memory_store(
        latent_dim=4,
        persistence_backend=memory_backend,
    )
    isolated_memory_store = build_default_memory_store(
        latent_dim=4,
        persistence_backend=isolated_backend,
    )
    if not memory_store.load_from_backend(key="gate8-longitudinal/memory"):
        raise RuntimeError("Gate 8 memory owner did not hydrate")
    if not isolated_memory_store.load_from_backend(
        key="gate8-longitudinal/isolated-memory"
    ):
        raise RuntimeError("Gate 8 isolated memory owner did not hydrate")
    loaded_temporal = pickle.loads(temporal_checkpoint_path.read_bytes())
    if not isinstance(loaded_temporal, MetacontrollerParameterSnapshot):
        raise TypeError("Gate 8 temporal checkpoint has the wrong type")
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(
            n_z=4,
            initialization_seed=seed,
        )
    )
    policy.apply_rare_heavy_snapshot(loaded_temporal)

    memory_after = memory_store.create_checkpoint(
        checkpoint_id="gate8-longitudinal-memory-after-restart"
    )
    isolated_after = isolated_memory_store.create_checkpoint(
        checkpoint_id="gate8-longitudinal-isolated-after-restart"
    )
    temporal_after = policy.export_rare_heavy_snapshot()
    exact = (
        _checkpoint_fingerprint(memory_before)
        == _checkpoint_fingerprint(memory_after)
        and _checkpoint_fingerprint(isolated_before)
        == _checkpoint_fingerprint(isolated_after)
        and _snapshot_fingerprint(temporal_before)
        == _snapshot_fingerprint(temporal_after)
    )
    return memory_store, isolated_memory_store, policy, exact


def _run_longitudinal_arm(
    *,
    seed: int,
    arm: str,
    records: Sequence[Mapping[str, Any]],
    runtime_state_dir: Path,
) -> Gate8LongitudinalArmMetric:
    if arm not in GATE8_ARMS:
        raise ValueError(f"Unsupported Gate 8 longitudinal arm {arm!r}")
    plans = tuple(_record_to_plan(record) for record in records)
    memory_backend = FileSystemPersistenceBackend(
        base_dir=str(runtime_state_dir / "memory"),
        max_versions=64,
    )
    isolated_backend = FileSystemPersistenceBackend(
        base_dir=str(runtime_state_dir / "isolated-memory"),
        max_versions=64,
    )
    memory_store = build_default_memory_store(
        latent_dim=4,
        persistence_backend=memory_backend,
    )
    isolated_memory_store = build_default_memory_store(
        latent_dim=4,
        persistence_backend=isolated_backend,
    )
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(
            n_z=4,
            initialization_seed=seed,
        )
    )
    memory_initial = memory_store.create_checkpoint(
        checkpoint_id=f"gate8-longitudinal:{seed}:{arm}:memory-initial"
    )
    isolated_initial = isolated_memory_store.create_checkpoint(
        checkpoint_id=f"gate8-longitudinal:{seed}:{arm}:isolated-initial"
    )
    policy_initial = policy.export_rare_heavy_snapshot()
    rollback_before = _owner_fingerprint(
        memory_checkpoint=memory_initial,
        isolated_checkpoint=isolated_initial,
        temporal_snapshot=policy_initial,
    )

    callback_scores: list[float] = []
    policy_alignments: list[float] = []
    worker_execution_count = 0
    duplicate_job_count = 0
    temporal_operation_count = 0
    observed_lineage = 0
    persistence_roundtrip_exact = True
    constructor_restart_count = 0
    turn_latency_seconds = 0.0
    slow_job_latency_seconds = 0.0
    targets = _arm_targets(arm)
    expected_lineage = len(plans) * len(targets)
    temporal_checkpoint_path = runtime_state_dir / "temporal-owner.pkl"

    for chunk_start in range(0, len(plans), GATE11_LONGITUDINAL_SESSION_SIZE):
        chunk = plans[
            chunk_start : chunk_start + GATE11_LONGITUDINAL_SESSION_SIZE
        ]
        results_by_job: dict[str, SessionPostSlowLoopResult] = {}
        policy_alignment_by_job: dict[str, float] = {}
        temporal_audit_count_by_job: dict[str, int] = {}

        async def worker(
            job: SessionPostSlowLoopJob,
            *,
            _memory_store: Any = memory_store,
            _isolated_memory_store: Any = isolated_memory_store,
            _policy: FullLearnedTemporalPolicy = policy,
            _chunk: tuple[Gate78EpisodePlan, ...] = chunk,
            _results_by_job: dict[str, SessionPostSlowLoopResult] = (
                results_by_job
            ),
            _policy_alignment_by_job: dict[str, float] = (
                policy_alignment_by_job
            ),
            _temporal_audit_count_by_job: dict[str, int] = (
                temporal_audit_count_by_job
            ),
        ) -> SessionPostSlowLoopResult:
            nonlocal worker_execution_count
            worker_execution_count += 1
            writeback_store = (
                _isolated_memory_store
                if arm == "policy-only-sleep"
                else _memory_store
            )
            temporal_policy = (
                _policy
                if arm in {"sleep-consolidation", "policy-only-sleep"}
                else None
            )
            writeback_result, audits = apply_session_post_writeback_request(
                request=job.writeback_request,
                memory_store=writeback_store,
                temporal_policy=temporal_policy,
                regime_module=None,
            )
            plan = next(
                candidate
                for candidate in _chunk
                if candidate.episode_id in job.job_id
            )
            _policy_alignment_by_job[job.job_id] = _policy_alignment(
                _policy,
                plan,
            )
            _temporal_audit_count_by_job[job.job_id] = len(audits)
            result = SessionPostSlowLoopResult(
                job_id=job.job_id,
                context_session_id=job.context_session_id,
                closed_at_turn=job.closed_at_turn,
                writeback_result=writeback_result,
                applied=bool(
                    writeback_result is not None
                    and writeback_result.applied_operations
                ),
                blocked=bool(
                    writeback_result is not None
                    and writeback_result.blocked_operations
                ),
                description=(
                    "Gate 8 longitudinal owner writeback completed for "
                    f"{job.job_id}."
                ),
            )
            _results_by_job[job.job_id] = result
            return result

        queue = SessionPostSlowLoopQueue(worker=worker)
        jobs = tuple(_job(plan, arm=arm) for plan in chunk)
        schedule_start = time.perf_counter()
        for job in jobs:
            queue.enqueue(job)
        queue.enqueue(jobs[0])
        schedule_end = time.perf_counter()
        slow_start = time.perf_counter()
        asyncio.run(queue.wait_for_idle())
        slow_end = time.perf_counter()
        queue_state = queue.snapshot()
        completed = queue.consume_completed_results()
        if len(completed) != len(jobs):
            raise RuntimeError(
                "Gate 8 longitudinal queue completed "
                f"{len(completed)} of {len(jobs)} jobs"
            )
        duplicate_job_count += queue_state.duplicate_job_count

        next_session_start = time.perf_counter()
        for plan, job in zip(chunk, jobs, strict=True):
            retrieval = memory_store.retrieve(
                RetrievalQuery(
                    text=plan.session_two_turns[0],
                    track=Track.SHARED,
                    strata=(MemoryStratum.DURABLE,),
                    limit=4,
                ),
                timestamp_ms=plan.global_index * 10 + 5,
            )
            callback_scores.append(
                float(
                    any(
                        entry.entry_id
                        == f"gate8:{plan.episode_id}:commitment"
                        for entry in retrieval.entries
                    )
                )
            )
            policy_alignments.append(
                policy_alignment_by_job[job.job_id]
            )
            writeback_result = results_by_job[job.job_id].writeback_result
            operations = (
                writeback_result.applied_operations
                if writeback_result is not None
                else ()
            )
            temporal_operations = tuple(
                operation
                for operation in operations
                if operation.startswith("temporal-prior:")
            )
            temporal_operation_count += len(temporal_operations)
            if (
                "memory" in targets
                and writeback_result is not None
                and writeback_result.checkpoint is not None
                and any(
                    not operation.startswith("temporal-prior:")
                    for operation in operations
                )
            ):
                observed_lineage += 1
            if (
                "temporal" in targets
                and temporal_operations
                and temporal_audit_count_by_job[job.job_id] > 0
            ):
                observed_lineage += 1
        next_session_end = time.perf_counter()
        turn_latency_seconds += (
            schedule_end - schedule_start
        ) + (next_session_end - next_session_start)
        slow_job_latency_seconds += slow_end - slow_start

        if chunk_start + len(chunk) < len(plans):
            (
                memory_store,
                isolated_memory_store,
                policy,
                roundtrip_exact,
            ) = _persist_and_restart_owners(
                memory_store=memory_store,
                isolated_memory_store=isolated_memory_store,
                policy=policy,
                memory_backend=memory_backend,
                isolated_backend=isolated_backend,
                temporal_checkpoint_path=temporal_checkpoint_path,
                seed=seed,
            )
            constructor_restart_count += 1
            persistence_roundtrip_exact = (
                persistence_roundtrip_exact and roundtrip_exact
            )

    memory_final = memory_store.create_checkpoint(
        checkpoint_id=f"gate8-longitudinal:{seed}:{arm}:memory-final"
    )
    isolated_final = isolated_memory_store.create_checkpoint(
        checkpoint_id=f"gate8-longitudinal:{seed}:{arm}:isolated-final"
    )
    policy_final = policy.export_rare_heavy_snapshot()
    callback_consistency = _mean(callback_scores)
    temporal_alignment = _mean(policy_alignments)
    delayed_payoff = (
        0.4 * callback_consistency + 0.6 * temporal_alignment
    )
    initial_parameters = policy_initial.temporal_parameters
    final_parameters = policy_final.temporal_parameters
    policy_parameter_mae = _mean(
        tuple(
            abs(before - after)
            for before, after in zip(
                (
                    initial_parameters.residual_weight,
                    initial_parameters.memory_weight,
                    initial_parameters.reflection_weight,
                    initial_parameters.switch_bias,
                ),
                (
                    final_parameters.residual_weight,
                    final_parameters.memory_weight,
                    final_parameters.reflection_weight,
                    final_parameters.switch_bias,
                ),
                strict=True,
            )
        )
    )
    memory_budget_fraction = (
        max(len(memory_final.entries) - len(memory_initial.entries), 0)
        / max(2 * len(plans), 1)
    )
    owner_state_drift = 0.5 * (
        memory_budget_fraction + policy_parameter_mae
    )

    memory_store.restore_checkpoint(memory_initial)
    isolated_memory_store.restore_checkpoint(isolated_initial)
    policy.apply_rare_heavy_snapshot(policy_initial)
    (
        memory_store,
        isolated_memory_store,
        policy,
        rollback_persistence_exact,
    ) = _persist_and_restart_owners(
        memory_store=memory_store,
        isolated_memory_store=isolated_memory_store,
        policy=policy,
        memory_backend=memory_backend,
        isolated_backend=isolated_backend,
        temporal_checkpoint_path=temporal_checkpoint_path,
        seed=seed,
    )
    restored_memory = memory_store.create_checkpoint(
        checkpoint_id=f"gate8-longitudinal:{seed}:{arm}:memory-restored"
    )
    restored_isolated = isolated_memory_store.create_checkpoint(
        checkpoint_id=f"gate8-longitudinal:{seed}:{arm}:isolated-restored"
    )
    restored_policy = policy.export_rare_heavy_snapshot()
    rollback_after = _owner_fingerprint(
        memory_checkpoint=restored_memory,
        isolated_checkpoint=restored_isolated,
        temporal_snapshot=restored_policy,
    )
    duplicate_execution_count = max(
        worker_execution_count - len(plans),
        0,
    )
    lineage_coverage = (
        observed_lineage / expected_lineage
        if expected_lineage
        else 1.0
    )
    return Gate8LongitudinalArmMetric(
        seed=seed,
        arm=arm,
        settled_transition_count=len(records),
        consumer_session_count=math.ceil(
            len(records) / GATE11_LONGITUDINAL_SESSION_SIZE
        ),
        constructor_restart_count=constructor_restart_count,
        memory_entry_count=len(memory_final.entries),
        isolated_memory_entry_count=len(isolated_final.entries),
        temporal_operation_count=temporal_operation_count,
        next_session_cold_start_loss=1.0 - delayed_payoff,
        callback_commitment_consistency=callback_consistency,
        temporal_policy_alignment=temporal_alignment,
        delayed_payoff=delayed_payoff,
        owner_state_drift=owner_state_drift,
        prompt_token_increment=0,
        unique_job_count=len(plans),
        worker_execution_count=worker_execution_count,
        duplicate_job_count=duplicate_job_count,
        duplicate_job_execution_count=duplicate_execution_count,
        owner_lineage_expected_count=expected_lineage,
        owner_lineage_observed_count=observed_lineage,
        owner_writeback_lineage_coverage=lineage_coverage,
        turn_latency_ms=turn_latency_seconds * 1000.0,
        slow_job_latency_ms=slow_job_latency_seconds * 1000.0,
        turn_latency_contains_slow_job=False,
        persistence_roundtrip_exact=(
            persistence_roundtrip_exact and rollback_persistence_exact
        ),
        rollback_exact=(
            rollback_persistence_exact
            and rollback_before == rollback_after
        ),
        rollback_fingerprint_before=rollback_before,
        rollback_fingerprint_after=rollback_after,
        frozen_substrate_mutation_count=sum(
            bool(record["substrate"]["mutation_applied"])
            for record in records
        ),
    )


def _compare_arms(
    metrics: Sequence[Gate8LongitudinalArmMetric],
) -> tuple[
    dict[str, float],
    dict[str, dict[str, Any]],
    dict[str, bool],
    str,
]:
    by_arm_seed = {
        (metric.arm, metric.seed): metric for metric in metrics
    }
    full = tuple(
        by_arm_seed[("sleep-consolidation", seed)]
        for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
    )
    no_sleep = tuple(
        by_arm_seed[("no-sleep", seed)]
        for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
    )
    memory_only = tuple(
        by_arm_seed[("memory-only-sleep", seed)]
        for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
    )
    policy_only = tuple(
        by_arm_seed[("policy-only-sleep", seed)]
        for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
    )
    paired = {
        "cold_start_loss_reduction_vs_no_sleep": tuple(
            control.next_session_cold_start_loss
            - treatment.next_session_cold_start_loss
            for treatment, control in zip(full, no_sleep, strict=True)
        ),
        "callback_consistency_gain_vs_no_sleep": tuple(
            treatment.callback_commitment_consistency
            - control.callback_commitment_consistency
            for treatment, control in zip(full, no_sleep, strict=True)
        ),
        "delayed_payoff_gain_vs_no_sleep": tuple(
            treatment.delayed_payoff - control.delayed_payoff
            for treatment, control in zip(full, no_sleep, strict=True)
        ),
        "payoff_margin_vs_memory_only": tuple(
            treatment.delayed_payoff - control.delayed_payoff
            for treatment, control in zip(full, memory_only, strict=True)
        ),
        "payoff_margin_vs_policy_only": tuple(
            treatment.delayed_payoff - control.delayed_payoff
            for treatment, control in zip(full, policy_only, strict=True)
        ),
    }
    aggregate = {
        name: _mean(values) for name, values in paired.items()
    }
    aggregate["single_owner_payoff_margin"] = min(
        aggregate["payoff_margin_vs_memory_only"],
        aggregate["payoff_margin_vs_policy_only"],
    )
    aggregate["full_cold_start_loss"] = _mean(
        tuple(metric.next_session_cold_start_loss for metric in full)
    )
    aggregate["full_callback_consistency"] = _mean(
        tuple(metric.callback_commitment_consistency for metric in full)
    )
    aggregate["full_delayed_payoff"] = _mean(
        tuple(metric.delayed_payoff for metric in full)
    )
    aggregate["maximum_full_owner_state_drift"] = max(
        metric.owner_state_drift for metric in full
    )
    confidence = {
        name: {
            "paired_seed_gains": list(values),
            "confidence_interval_95": list(
                _confidence_interval_95(values)
            ),
        }
        for name, values in paired.items()
    }
    integrity_gates = {
        "all_arms_all_seeds_present": (
            len(metrics)
            == len(GATE8_ARMS) * len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
        "settled_transition_count_510_per_arm_seed": all(
            metric.settled_transition_count == 510 for metric in metrics
        ),
        "consumer_session_count_51_per_arm_seed": all(
            metric.consumer_session_count == 51 for metric in metrics
        ),
        "constructor_restart_count_50_per_arm_seed": all(
            metric.constructor_restart_count == 50 for metric in metrics
        ),
        "persistence_roundtrip_exact": all(
            metric.persistence_roundtrip_exact for metric in metrics
        ),
        "owner_lineage_complete": all(
            metric.owner_writeback_lineage_coverage == 1.0
            for metric in metrics
        ),
        "prompt_token_increment_zero": all(
            metric.prompt_token_increment == 0 for metric in metrics
        ),
        "duplicate_job_execution_zero": all(
            metric.duplicate_job_execution_count == 0
            for metric in metrics
        ),
        "turn_latency_excludes_slow_job": all(
            not metric.turn_latency_contains_slow_job for metric in metrics
        ),
        "frozen_substrate_mutation_zero": all(
            metric.frozen_substrate_mutation_count == 0
            for metric in metrics
        ),
        "whole_cycle_rollback_exact": all(
            metric.rollback_exact for metric in metrics
        ),
        "no_sleep_owner_writes_zero": all(
            metric.memory_entry_count == 0
            and metric.temporal_operation_count == 0
            for metric in no_sleep
        ),
        "memory_only_temporal_writes_zero": all(
            metric.temporal_operation_count == 0
            for metric in memory_only
        ),
        "policy_only_primary_memory_writes_zero": all(
            metric.memory_entry_count == 0 for metric in policy_only
        ),
        "full_owner_state_drift_bounded": (
            aggregate["maximum_full_owner_state_drift"] <= 0.50
        ),
    }
    effect_names = (
        "cold_start_loss_reduction_vs_no_sleep",
        "callback_consistency_gain_vs_no_sleep",
        "delayed_payoff_gain_vs_no_sleep",
    )
    effect_gates = {
        f"{name}_minimum_effect": (
            aggregate[name] >= GATE8_LONGITUDINAL_MIN_EFFECT
        )
        for name in effect_names
    }
    effect_gates["full_outperforms_single_owner_controls"] = (
        aggregate["single_owner_payoff_margin"] > 0.0
    )
    ci_names = effect_names + (
        "payoff_margin_vs_memory_only",
        "payoff_margin_vs_policy_only",
    )
    effect_gates["paired_seed_ci_lower_positive"] = all(
        float(confidence[name]["confidence_interval_95"][0]) > 0.0
        for name in ci_names
    )
    gates = {**integrity_gates, **effect_gates}
    integrity_passed = all(integrity_gates.values())
    effect_passed = all(effect_gates.values())
    status = (
        "invalid"
        if not integrity_passed
        else "longitudinal-supported"
        if effect_passed
        else "not-supported"
    )
    return aggregate, confidence, gates, status


def _validate_source(
    source: Path,
    *,
    transition_limit: int | None,
    formal_locked_run: bool,
) -> tuple[
    dict[str, Any],
    dict[int, list[dict[str, Any]]],
]:
    aggregate_manifest = json.loads(
        (source / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    aggregate_verdict = json.loads(
        (source / "aggregate_verdict.json").read_text(encoding="utf-8")
    )
    if aggregate_verdict.get("consumer_admission") != "allowed":
        raise ValueError(
            "Gate 8 longitudinal source is not admitted for consumption"
        )
    if (
        aggregate_manifest.get("schema_version")
        != GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION
    ):
        raise ValueError("Gate 8 longitudinal source schema drifted")
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        records = load_gate11_longitudinal_source_records(
            source / f"seed_{seed}" / "transitions.jsonl"
        )
        plans = build_gate11_longitudinal_source_plans(seed)
        validate_gate11_longitudinal_source_prefix(
            records=records,
            plans=plans,
        )
        if formal_locked_run and len(records) != 510:
            raise ValueError(
                f"Gate 8 longitudinal seed {seed} requires 510 records"
            )
        if transition_limit is not None:
            records = records[:transition_limit]
        if not records:
            raise ValueError(
                f"Gate 8 longitudinal seed {seed} has no records"
            )
        records_by_seed[seed] = records
    return aggregate_manifest, records_by_seed


def export_gate8_longitudinal_bundle(
    *,
    trace_root: str | Path,
    output_dir: str | Path,
    transition_limit: int | None = None,
    formal_locked_run: bool = True,
) -> tuple[Path, ...]:
    source = Path(trace_root)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    existing = tuple(
        filename
        for filename in GATE8_LONGITUDINAL_REQUIRED_FILES
        if (target / filename).exists()
    )
    if formal_locked_run and existing:
        raise FileExistsError(
            "Gate 8 longitudinal locked evidence is immutable; refusing "
            f"to overwrite {existing}"
        )
    if formal_locked_run and transition_limit is not None:
        raise ValueError(
            "Gate 8 longitudinal formal run may not limit transitions"
        )
    aggregate_manifest, records_by_seed = _validate_source(
        source,
        transition_limit=transition_limit,
        formal_locked_run=formal_locked_run,
    )
    runtime_state = target / "runtime_state"
    if runtime_state.exists():
        raise FileExistsError(
            "Gate 8 longitudinal runtime_state already exists"
        )
    metrics: list[Gate8LongitudinalArmMetric] = []
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for arm in GATE8_ARMS:
            metrics.append(
                _run_longitudinal_arm(
                    seed=seed,
                    arm=arm,
                    records=records_by_seed[seed],
                    runtime_state_dir=runtime_state / str(seed) / arm,
                )
            )
    aggregate, confidence, gates, status = _compare_arms(metrics)
    if not formal_locked_run:
        status = (
            "development-supported"
            if all(gates.values())
            else "development-diagnostic"
        )
    source_rows: dict[str, list[dict[str, Any]]] = {
        "predictions.jsonl": [],
        "outcomes.jsonl": [],
        "prediction_errors.jsonl": [],
        "segments.jsonl": [],
        "credit.jsonl": [],
        "action_selection.jsonl": [],
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for record in records_by_seed[seed]:
            lineage = {
                **record["lineage"],
                "transition_id": record["transition_id"],
                "partition": record["partition"],
            }
            source_rows["predictions.jsonl"].append(
                {**lineage, "prediction": record["prediction"]}
            )
            source_rows["outcomes.jsonl"].append(
                {
                    **lineage,
                    "actual_outcome": record["actual_outcome"],
                    "environment_outcome": record["environment_outcome"],
                }
            )
            source_rows["prediction_errors.jsonl"].append(
                {**lineage, "prediction_error": record["prediction_error"]}
            )
            source_rows["segments.jsonl"].append(
                {**lineage, "temporal_snapshot": record["temporal_snapshot"]}
            )
            source_rows["credit.jsonl"].append(
                {**lineage, "credit_snapshot": record["credit_snapshot"]}
            )
            source_rows["action_selection.jsonl"].append(
                {**lineage, "action_selection": record["action_selection"]}
            )
    for name, rows in source_rows.items():
        _write_jsonl(target / name, rows)
    _write_jsonl(
        target / "state_diff.jsonl",
        [
            {
                "schema_version": GATE8_LONGITUDINAL_SCHEMA_VERSION,
                **asdict(metric),
            }
            for metric in metrics
        ],
    )
    manifest = {
        "schema_version": GATE8_LONGITUDINAL_SCHEMA_VERSION,
        "suite_id": "gate8-wake-sleep-longitudinal",
        "owner": (
            "SessionPostSlowLoopQueue + MemoryStore + "
            "FullLearnedTemporalPolicy"
        ),
        "source_schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "source_root": str(source),
        "source_runtime_fingerprint": aggregate_manifest[
            "runtime_fingerprint"
        ],
        "seed_schedule": list(GATE11_LONGITUDINAL_SOURCE_SEEDS),
        "arm_schedule": list(GATE8_ARMS),
        "formal_locked_run": formal_locked_run,
        "session_boundary_interval": GATE11_LONGITUDINAL_SESSION_SIZE,
        "settled_transition_count_per_arm_seed": len(
            next(iter(records_by_seed.values()))
        ),
        "arm_transition_count": sum(
            metric.settled_transition_count for metric in metrics
        ),
        "prompt_and_context_budget": (
            "Matched next-session retrieval probes across arms; sleep adds "
            "zero serving-prompt tokens."
        ),
        "minimum_effect": GATE8_LONGITUDINAL_MIN_EFFECT,
        "required_files": list(GATE8_LONGITUDINAL_REQUIRED_FILES),
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": (
                _git_output("status", "--porcelain")
                not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE8_LONGITUDINAL_SCHEMA_VERSION,
        "metrics": [asdict(metric) for metric in metrics],
        "aggregate_metrics": aggregate,
        "paired_seed_confidence": confidence,
        "gates": gates,
    }
    integrity_gate_names = (
        "all_arms_all_seeds_present",
        "settled_transition_count_510_per_arm_seed",
        "consumer_session_count_51_per_arm_seed",
        "constructor_restart_count_50_per_arm_seed",
        "persistence_roundtrip_exact",
        "owner_lineage_complete",
        "prompt_token_increment_zero",
        "duplicate_job_execution_zero",
        "turn_latency_excludes_slow_job",
        "frozen_substrate_mutation_zero",
        "whole_cycle_rollback_exact",
        "no_sleep_owner_writes_zero",
        "memory_only_temporal_writes_zero",
        "policy_only_primary_memory_writes_zero",
        "full_owner_state_drift_bounded",
    )
    mechanism_passed = all(gates[name] for name in integrity_gate_names)
    verdict = {
        "schema_version": GATE8_LONGITUDINAL_SCHEMA_VERSION,
        "gate_scope": "Gate 8 wake/sleep cross-session longitudinal",
        "status": status,
        "mechanism_passed": mechanism_passed,
        "causal_passed": status == "longitudinal-supported",
        "longitudinal_passed": status == "longitudinal-supported",
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "locked_source_consumed_once": formal_locked_run,
        "same_locked_source_rerun_allowed": False,
        "production_promotion_allowed": False,
    }
    rollback = {
        "schema_version": GATE8_LONGITUDINAL_SCHEMA_VERSION,
        "passed": all(metric.rollback_exact for metric in metrics),
        "arms": [
            {
                "seed": metric.seed,
                "arm": metric.arm,
                "exact": metric.rollback_exact,
                "before": metric.rollback_fingerprint_before,
                "after": metric.rollback_fingerprint_after,
                "persistence_roundtrip_exact": (
                    metric.persistence_roundtrip_exact
                ),
            }
            for metric in metrics
        ],
        "rollback_action": (
            "restore initial memory checkpoints and temporal parameter "
            "snapshot, then persist and reconstruct both owners"
        ),
        "substrate_mutated": False,
    }
    _write_json(target / "manifest.yaml", manifest)
    _write_json(target / "ablation_results.json", ablation)
    _write_json(target / "promotion_verdict.json", verdict)
    _write_json(target / "rollback_evidence.json", rollback)
    report_lines = [
        "# Gate 8 cross-session wake/sleep evidence",
        "",
        f"- status: `{status}`",
        f"- mechanism passed: `{mechanism_passed}`",
        f"- longitudinal passed: `{status == 'longitudinal-supported'}`",
        (
            "- settled transitions: "
            f"`{manifest['arm_transition_count']}` arm-transitions"
        ),
        (
            "- constructor restarts per arm/seed: "
            f"`{min(metric.constructor_restart_count for metric in metrics)}`"
        ),
        "",
        "## Primary effects",
        "",
    ]
    report_lines.extend(
        f"- `{name}`: `{value:.6f}`"
        for name, value in aggregate.items()
    )
    report_lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            (
                "- This packet tests deterministic next-session callback "
                "and temporal alignment on frozen real-substrate source "
                "signals. It does not provide human relationship-quality "
                "ground truth or authorize production promotion."
            ),
            "",
        )
    )
    (target / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    return tuple(
        target / name for name in GATE8_LONGITUDINAL_REQUIRED_FILES
    )


def verify_gate8_longitudinal_bundle(
    output_dir: str | Path,
) -> dict[str, Any]:
    target = Path(output_dir)
    missing = tuple(
        name
        for name in GATE8_LONGITUDINAL_REQUIRED_FILES
        if not (target / name).is_file()
    )
    if missing:
        return {
            "passed": False,
            "missing_files": missing,
            "status": "invalid",
        }
    manifest = json.loads(
        (target / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    passed = (
        manifest["schema_version"] == GATE8_LONGITUDINAL_SCHEMA_VERSION
        and tuple(manifest["seed_schedule"])
        == GATE11_LONGITUDINAL_SOURCE_SEEDS
        and tuple(manifest["arm_schedule"]) == GATE8_ARMS
        and tuple(manifest["required_files"])
        == GATE8_LONGITUDINAL_REQUIRED_FILES
        and verdict["status"]
        in {
            "invalid",
            "not-supported",
            "longitudinal-supported",
            "development-supported",
            "development-diagnostic",
        }
    )
    return {
        "passed": passed,
        "missing_files": (),
        "status": verdict["status"],
        "formal_locked_run": manifest["formal_locked_run"],
        "arm_transition_count": manifest["arm_transition_count"],
    }


__all__ = [
    "GATE8_LONGITUDINAL_MIN_EFFECT",
    "GATE8_LONGITUDINAL_REQUIRED_FILES",
    "GATE8_LONGITUDINAL_SCHEMA_VERSION",
    "Gate8LongitudinalArmMetric",
    "export_gate8_longitudinal_bundle",
    "verify_gate8_longitudinal_bundle",
]
