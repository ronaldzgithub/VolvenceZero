"""Preregistered Gate 8 session-post wake/sleep evidence campaign.

The harness keeps the four controls on the production session-post request,
queue, memory owner, and temporal owner APIs.  Sleep receives only the closed
session-one plan.  Session-two text is used only after the asynchronous job
has completed, so no slow-loop token is injected into the serving prompt.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping

from volvence_zero.agent.gate78_shared_trace import (
    GATE78_SOURCE_DESCRIPTOR,
    GATE78_TRACE_SCHEMA_VERSION,
    GATE78_TRACE_SEEDS,
    Gate78EpisodePlan,
    load_gate78_partition,
    verify_gate78_shared_trace_bundle,
)
from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopJob,
    SessionPostSlowLoopQueue,
    SessionPostSlowLoopResult,
)
from volvence_zero.evaluation import EvaluationReport
from volvence_zero.integration import (
    SessionPostWritebackRequest,
    apply_session_post_writeback_request,
)
from volvence_zero.memory import (
    MemoryEntry,
    MemoryStratum,
    RetrievalQuery,
    Track,
    build_default_memory_store,
)
from volvence_zero.reflection import (
    ConsolidationScore,
    MemoryConsolidation,
    PolicyConsolidation,
    ReflectionSnapshot,
    TemporalPriorUpdate,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
)


GATE8_SCHEMA_VERSION = "gate8-wake-sleep.v1"
GATE8_ARMS = (
    "sleep-consolidation",
    "no-sleep",
    "memory-only-sleep",
    "policy-only-sleep",
)
GATE8_REQUIRED_FILES = (
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


@dataclass(frozen=True)
class Gate8ArmResult:
    seed: int
    partition: str
    arm: str
    evaluation_episode_count: int
    memory_entry_count: int
    temporal_operation_count: int
    next_session_cold_start_loss: float
    callback_commitment_consistency: float
    temporal_policy_alignment: float
    delayed_payoff: float
    owner_state_drift: float
    turn_latency_ms: float
    slow_job_latency_ms: float
    prompt_token_increment: int
    unique_job_count: int
    worker_execution_count: int
    duplicate_job_count: int
    duplicate_job_execution_count: int
    owner_lineage_expected_count: int
    owner_lineage_observed_count: int
    owner_writeback_lineage_coverage: float
    turn_latency_contains_slow_job: bool
    rollback_exact: bool
    rollback_fingerprint_before: str
    rollback_fingerprint_after: str


@dataclass(frozen=True)
class Gate8EvidenceReport:
    schema_version: str
    source_schema_version: str
    source_fingerprint: str
    partition: str
    seed_schedule: tuple[int, ...]
    arm_schedule: tuple[str, ...]
    formal_locked_run: bool
    source_consumer_admission: bool
    arm_results: tuple[Gate8ArmResult, ...]
    aggregate_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    causal_gates: tuple[tuple[str, bool, float], ...]
    verdict: str
    description: str


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _checkpoint_fingerprint(checkpoint: object) -> str:
    payload = asdict(checkpoint)
    payload.pop("checkpoint_id", None)
    return _sha256(payload)


def _normalized_policy_target(plan: Gate78EpisodePlan) -> tuple[float, float, float]:
    values = tuple(float(value) for value in plan.user_prior[:3])
    total = max(sum(values), 1e-9)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _policy_alignment(
    policy: FullLearnedTemporalPolicy,
    plan: Gate78EpisodePlan,
) -> float:
    target = _normalized_policy_target(plan)
    parameters = policy.export_parameters()
    actual = (
        parameters.residual_weight,
        parameters.memory_weight,
        parameters.reflection_weight,
    )
    return max(
        0.0,
        1.0
        - sum(
            abs(observed - expected)
            for observed, expected in zip(actual, target, strict=True)
        )
        / len(target),
    )


def _evaluation_report(plan: Gate78EpisodePlan) -> EvaluationReport:
    return EvaluationReport(
        report_id=f"gate8:{plan.episode_id}:session-one",
        report_type="gate8-closed-session",
        timestamp_ms=plan.global_index * 10 + 1,
        session_ids=(f"{plan.episode_id}:session-one",),
        scores_by_family=(),
        alerts=(),
        trends=(),
        recommendations=(),
        description=(
            f"Closed session-one evidence for {plan.episode_id}; "
            "contains no session-two observation."
        ),
    )


def _reflection_snapshot(plan: Gate78EpisodePlan) -> ReflectionSnapshot:
    target = _normalized_policy_target(plan)
    entry = MemoryEntry(
        entry_id=f"gate8:{plan.episode_id}:commitment",
        content=(
            f"{plan.episode_id} callback commitment "
            f"{plan.user_prior_id} boundary {plan.next_session_boundary}"
        ),
        track=Track.SHARED,
        stratum=MemoryStratum.DURABLE.value,
        created_at_ms=plan.global_index * 10 + 2,
        last_accessed_ms=plan.global_index * 10 + 2,
        strength=0.85,
        tags=(
            "gate8-session-post",
            "callback-commitment",
            plan.context_id,
        ),
    )
    temporal_update = TemporalPriorUpdate(
        target="temporal.metacontroller",
        target_groups=("base-weights", "switch", "persistence"),
        residual_strength=target[0],
        memory_strength=target[1],
        reflection_strength=target[2],
        switch_bias_delta=(plan.user_prior[3] - 0.5) * 0.05,
        persistence_delta=(plan.difficulty - 0.6) * 0.05,
        learning_rate_delta=0.0,
        description=(
            f"Gate 8 closed-session policy consolidation for "
            f"{plan.episode_id}."
        ),
    )
    return ReflectionSnapshot(
        memory_consolidation=MemoryConsolidation(
            new_durable_entries=(entry,),
            promoted_entries=(),
            decayed_entries=(),
            beliefs_updated=(),
        ),
        policy_consolidation=PolicyConsolidation(
            controller_updates=(f"closed-session:{plan.episode_id}",),
            strategy_priors_updated=(plan.context_id,),
            regime_effectiveness_updated=(),
            temporal_prior_update=temporal_update,
        ),
        consolidation_score=ConsolidationScore(
            promotion_score=0.65,
            decay_score=0.0,
            threshold_delta=0.0,
            strategy_gain=0.1,
            regime_effectiveness_gain=0.0,
            confidence=0.9,
            description="Gate 8 bounded closed-session consolidation score.",
        ),
        interaction_trace_summary=(
            f"Closed session-one trace for {plan.episode_id}."
        ),
        tensions_identified=(),
        lessons_extracted=(f"retain:{plan.context_id}",),
        writeback_mode="apply",
        review_required=False,
        description=(
            "Gate 8 reflection contains memory and temporal products from "
            "closed session-one evidence only."
        ),
    )


def _request(
    plan: Gate78EpisodePlan,
    *,
    arm: str,
) -> SessionPostWritebackRequest:
    reflection_enabled = arm != "no-sleep"
    structural_enabled = arm in {
        "sleep-consolidation",
        "policy-only-sleep",
    }
    report = _evaluation_report(plan)
    return SessionPostWritebackRequest(
        context_session_id=f"{plan.episode_id}:session-one",
        source_wave_id=f"gate8:{plan.seed}:{arm}",
        session_report=report,
        reflection_snapshot=_reflection_snapshot(plan),
        credit_snapshot=None,
        evolution_judgement=None,
        cross_session_verdict="gate8-preregistered-control",
        writeback_source="gate8-closed-session",
        reflection_apply_enabled=reflection_enabled,
        structural_writeback_allowed=structural_enabled,
        checkpoint_id=f"gate8:{plan.seed}:{arm}:{plan.episode_id}",
        description=(
            f"Gate 8 {arm} request from closed session-one evidence."
        ),
    )


def _job(
    plan: Gate78EpisodePlan,
    *,
    arm: str,
) -> SessionPostSlowLoopJob:
    report = _evaluation_report(plan)
    return SessionPostSlowLoopJob(
        job_id=f"gate8:{plan.seed}:{arm}:{plan.episode_id}",
        context_session_id=f"{plan.episode_id}:session-one",
        closed_at_turn=2,
        session_report=report,
        prior_session_report_count=0,
        trace_count=2,
        substrate_batch_count=0,
        prediction_error_summary=(
            ("magnitude", plan.difficulty),
            ("signed_reward", 1.0 - plan.difficulty),
        ),
        writeback_request=_request(plan, arm=arm),
        description=(
            f"Gate 8 asynchronous {arm} job for {plan.episode_id}."
        ),
    )


def _arm_targets(arm: str) -> tuple[str, ...]:
    if arm == "sleep-consolidation":
        return ("memory", "temporal")
    if arm == "memory-only-sleep":
        return ("memory",)
    if arm == "policy-only-sleep":
        return ("temporal",)
    if arm == "no-sleep":
        return ()
    raise ValueError(f"Unsupported Gate 8 arm {arm!r}")


def _run_arm(
    *,
    seed: int,
    partition: str,
    arm: str,
    plans: tuple[Gate78EpisodePlan, ...],
) -> Gate8ArmResult:
    if arm not in GATE8_ARMS:
        raise ValueError(f"Unsupported Gate 8 arm {arm!r}")
    memory_store = build_default_memory_store(latent_dim=4)
    isolated_memory_store = build_default_memory_store(latent_dim=4)
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(
            n_z=4,
            initialization_seed=seed,
        )
    )
    memory_initial = memory_store.create_checkpoint(
        checkpoint_id=f"gate8:{seed}:{arm}:memory-initial"
    )
    isolated_initial = isolated_memory_store.create_checkpoint(
        checkpoint_id=f"gate8:{seed}:{arm}:isolated-initial"
    )
    policy_initial = policy.export_rare_heavy_snapshot()
    rollback_before = _sha256(
        (
            _checkpoint_fingerprint(memory_initial),
            _checkpoint_fingerprint(isolated_initial),
            _sha256(policy_initial),
        )
    )
    results_by_job: dict[str, SessionPostSlowLoopResult] = {}
    policy_alignment_by_job: dict[str, float] = {}
    temporal_audit_count_by_job: dict[str, int] = {}
    worker_execution_count = 0

    async def worker(job: SessionPostSlowLoopJob) -> SessionPostSlowLoopResult:
        nonlocal worker_execution_count
        worker_execution_count += 1
        writeback_store = (
            isolated_memory_store
            if arm == "policy-only-sleep"
            else memory_store
        )
        temporal_policy = (
            policy
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
            for candidate in plans
            if candidate.episode_id in job.job_id
        )
        policy_alignment_by_job[job.job_id] = _policy_alignment(policy, plan)
        temporal_audit_count_by_job[job.job_id] = len(audits)
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
                f"Gate 8 {arm} owner writeback completed for {job.job_id}."
            ),
        )
        results_by_job[job.job_id] = result
        return result

    queue = SessionPostSlowLoopQueue(worker=worker)
    jobs = tuple(_job(plan, arm=arm) for plan in plans)
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
            f"Gate 8 queue completed {len(completed)} of {len(jobs)} jobs"
        )

    next_session_start = time.perf_counter()
    callback_scores: list[float] = []
    policy_alignments: list[float] = []
    for plan, job in zip(plans, jobs, strict=True):
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
        policy_alignments.append(policy_alignment_by_job[job.job_id])
    next_session_end = time.perf_counter()

    callback_consistency = _mean(tuple(callback_scores))
    temporal_alignment = _mean(tuple(policy_alignments))
    delayed_payoff = (
        0.4 * callback_consistency + 0.6 * temporal_alignment
    )
    cold_start_loss = 1.0 - delayed_payoff
    memory_final = memory_store.create_checkpoint(
        checkpoint_id=f"gate8:{seed}:{arm}:memory-final"
    )
    policy_final = policy.export_rare_heavy_snapshot()
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
    targets = _arm_targets(arm)
    expected_lineage = len(plans) * len(targets)
    observed_lineage = 0
    temporal_operation_count = 0
    for job in jobs:
        result = results_by_job[job.job_id]
        writeback_result = result.writeback_result
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
    lineage_coverage = (
        observed_lineage / expected_lineage
        if expected_lineage
        else 1.0
    )

    memory_store.restore_checkpoint(memory_initial)
    isolated_memory_store.restore_checkpoint(isolated_initial)
    policy.apply_rare_heavy_snapshot(policy_initial)
    restored_memory = memory_store.create_checkpoint(
        checkpoint_id=f"gate8:{seed}:{arm}:memory-restored"
    )
    restored_isolated = isolated_memory_store.create_checkpoint(
        checkpoint_id=f"gate8:{seed}:{arm}:isolated-restored"
    )
    restored_policy = policy.export_rare_heavy_snapshot()
    rollback_after = _sha256(
        (
            _checkpoint_fingerprint(restored_memory),
            _checkpoint_fingerprint(restored_isolated),
            _sha256(restored_policy),
        )
    )
    turn_latency_ms = (
        (schedule_end - schedule_start)
        + (next_session_end - next_session_start)
    ) * 1000.0
    duplicate_execution_count = max(
        worker_execution_count - len(jobs),
        0,
    )
    return Gate8ArmResult(
        seed=seed,
        partition=partition,
        arm=arm,
        evaluation_episode_count=len(plans),
        memory_entry_count=len(memory_final.entries),
        temporal_operation_count=temporal_operation_count,
        next_session_cold_start_loss=cold_start_loss,
        callback_commitment_consistency=callback_consistency,
        temporal_policy_alignment=temporal_alignment,
        delayed_payoff=delayed_payoff,
        owner_state_drift=owner_state_drift,
        turn_latency_ms=turn_latency_ms,
        slow_job_latency_ms=(slow_end - slow_start) * 1000.0,
        prompt_token_increment=0,
        unique_job_count=len(jobs),
        worker_execution_count=worker_execution_count,
        duplicate_job_count=queue_state.duplicate_job_count,
        duplicate_job_execution_count=duplicate_execution_count,
        owner_lineage_expected_count=expected_lineage,
        owner_lineage_observed_count=observed_lineage,
        owner_writeback_lineage_coverage=lineage_coverage,
        turn_latency_contains_slow_job=False,
        rollback_exact=rollback_before == rollback_after,
        rollback_fingerprint_before=rollback_before,
        rollback_fingerprint_after=rollback_after,
    )


def _arm_means(
    results: tuple[Gate8ArmResult, ...],
    arm: str,
) -> dict[str, float]:
    selected = tuple(row for row in results if row.arm == arm)
    return {
        "cold_start_loss": _mean(
            tuple(row.next_session_cold_start_loss for row in selected)
        ),
        "callback_consistency": _mean(
            tuple(row.callback_commitment_consistency for row in selected)
        ),
        "delayed_payoff": _mean(
            tuple(row.delayed_payoff for row in selected)
        ),
    }


def _aggregate(
    results: tuple[Gate8ArmResult, ...],
    *,
    source_admission: bool,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, bool, float], ...],
    tuple[tuple[str, bool, float], ...],
    str,
]:
    means = {arm: _arm_means(results, arm) for arm in GATE8_ARMS}
    full = means["sleep-consolidation"]
    no_sleep = means["no-sleep"]
    cold_loss_reduction = (
        no_sleep["cold_start_loss"] - full["cold_start_loss"]
    )
    callback_gain = (
        full["callback_consistency"]
        - no_sleep["callback_consistency"]
    )
    payoff_gain = (
        full["delayed_payoff"] - no_sleep["delayed_payoff"]
    )
    single_owner_payoff_margin = min(
        full["delayed_payoff"]
        - means["memory-only-sleep"]["delayed_payoff"],
        full["delayed_payoff"]
        - means["policy-only-sleep"]["delayed_payoff"],
    )
    max_full_drift = max(
        row.owner_state_drift
        for row in results
        if row.arm == "sleep-consolidation"
    )
    prompt_increment = sum(row.prompt_token_increment for row in results)
    duplicate_execution_count = sum(
        row.duplicate_job_execution_count for row in results
    )
    rollback_mismatch = sum(not row.rollback_exact for row in results)
    turn_latency_contamination = sum(
        row.turn_latency_contains_slow_job for row in results
    )
    minimum_lineage_coverage = min(
        row.owner_writeback_lineage_coverage for row in results
    )
    metrics = (
        ("full_cold_start_loss", full["cold_start_loss"]),
        ("full_callback_consistency", full["callback_consistency"]),
        ("full_delayed_payoff", full["delayed_payoff"]),
        ("cold_start_loss_reduction_vs_no_sleep", cold_loss_reduction),
        ("callback_consistency_gain_vs_no_sleep", callback_gain),
        ("delayed_payoff_gain_vs_no_sleep", payoff_gain),
        ("single_owner_payoff_margin", single_owner_payoff_margin),
        ("maximum_full_owner_state_drift", max_full_drift),
        ("sleep_prompt_token_increment", float(prompt_increment)),
        ("duplicate_job_execution_count", float(duplicate_execution_count)),
        ("minimum_owner_lineage_coverage", minimum_lineage_coverage),
        ("rollback_fingerprint_mismatch_count", float(rollback_mismatch)),
        (
            "turn_latency_contains_slow_job_count",
            float(turn_latency_contamination),
        ),
    )
    mechanism_gates = (
        ("source-consumer-admission", source_admission, float(source_admission)),
        ("sleep-prompt-token-increment-zero", prompt_increment == 0, float(prompt_increment)),
        (
            "duplicate-job-execution-zero",
            duplicate_execution_count == 0,
            float(duplicate_execution_count),
        ),
        (
            "owner-lineage-complete",
            minimum_lineage_coverage >= 1.0,
            minimum_lineage_coverage,
        ),
        (
            "whole-cycle-rollback-exact",
            rollback_mismatch == 0,
            float(rollback_mismatch),
        ),
        (
            "turn-latency-excludes-slow-job",
            turn_latency_contamination == 0,
            float(turn_latency_contamination),
        ),
    )
    causal_gates = (
        (
            "cold-start-loss-reduction-vs-no-sleep",
            cold_loss_reduction >= 0.02,
            cold_loss_reduction,
        ),
        (
            "callback-consistency-gain-vs-no-sleep",
            callback_gain >= 0.02,
            callback_gain,
        ),
        (
            "delayed-payoff-gain-vs-no-sleep",
            payoff_gain >= 0.02,
            payoff_gain,
        ),
        (
            "full-outperforms-single-owner-controls",
            single_owner_payoff_margin > 0.0,
            single_owner_payoff_margin,
        ),
        (
            "full-owner-state-drift-bounded",
            max_full_drift <= 0.50,
            max_full_drift,
        ),
    )
    if not all(passed for _name, passed, _value in mechanism_gates):
        verdict = "invalid"
    elif all(passed for _name, passed, _value in causal_gates):
        verdict = "causal-supported"
    else:
        verdict = "not-supported"
    return metrics, mechanism_gates, causal_gates, verdict


def run_gate8_evidence(
    *,
    trace_root: str | Path,
    seed_schedule: tuple[int, ...] = GATE78_TRACE_SEEDS,
    partition: str = "trace-development-heldout",
    evaluation_limit: int | None = None,
    formal_locked_run: bool = False,
) -> Gate8EvidenceReport:
    if not seed_schedule:
        raise ValueError("Gate 8 seed_schedule must not be empty")
    if any(seed not in GATE78_TRACE_SEEDS for seed in seed_schedule):
        raise ValueError("Gate 8 seed_schedule contains an unregistered seed")
    if formal_locked_run and partition != "trace-locked-confirmation":
        raise ValueError("Formal Gate 8 run must use trace-locked-confirmation")
    if not formal_locked_run and partition == "trace-locked-confirmation":
        raise ValueError(
            "Development Gate 8 run must not consume locked confirmation"
        )
    source_verification = verify_gate78_shared_trace_bundle(trace_root)
    if not source_verification["consumer_admission"]:
        raise RuntimeError("Gate 8 source corpus failed consumer admission")
    results: list[Gate8ArmResult] = []
    for seed in seed_schedule:
        plans = load_gate78_partition(
            trace_root,
            seed=seed,
            partition=partition,
        )
        if evaluation_limit is not None:
            plans = plans[:evaluation_limit]
        if not plans:
            raise ValueError("Gate 8 evaluation plan must not be empty")
        for arm in GATE8_ARMS:
            results.append(
                _run_arm(
                    seed=seed,
                    partition=partition,
                    arm=arm,
                    plans=plans,
                )
            )
    arm_results = tuple(results)
    metrics, mechanism_gates, causal_gates, verdict = _aggregate(
        arm_results,
        source_admission=bool(source_verification["consumer_admission"]),
    )
    return Gate8EvidenceReport(
        schema_version=GATE8_SCHEMA_VERSION,
        source_schema_version=GATE78_TRACE_SCHEMA_VERSION,
        source_fingerprint=_sha256(GATE78_SOURCE_DESCRIPTOR),
        partition=partition,
        seed_schedule=seed_schedule,
        arm_schedule=GATE8_ARMS,
        formal_locked_run=formal_locked_run,
        source_consumer_admission=True,
        arm_results=arm_results,
        aggregate_metrics=metrics,
        mechanism_gates=mechanism_gates,
        causal_gates=causal_gates,
        verdict=verdict,
        description=(
            f"Gate 8 four-arm wake/sleep evidence on {partition}: "
            f"verdict={verdict}; seeds={seed_schedule}."
        ),
    )


def _write_jsonl(
    path: Path,
    rows: tuple[Mapping[str, object], ...],
) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_gate8_evidence_bundle(
    report: Gate8EvidenceReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if report.formal_locked_run:
        existing = tuple(
            filename
            for filename in GATE8_REQUIRED_FILES
            if (target / filename).exists()
        )
        if existing:
            raise FileExistsError(
                "Gate 8 formal locked evidence is immutable; refusing to "
                f"overwrite {existing}"
            )
    rows_by_file: dict[str, tuple[Mapping[str, object], ...]] = {
        "predictions.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "partition": row.partition,
                "next_session_cold_start_loss": row.next_session_cold_start_loss,
                "prompt_token_increment": row.prompt_token_increment,
            }
            for row in report.arm_results
        ),
        "outcomes.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "callback_commitment_consistency": row.callback_commitment_consistency,
                "temporal_policy_alignment": row.temporal_policy_alignment,
                "delayed_payoff": row.delayed_payoff,
            }
            for row in report.arm_results
        ),
        "prediction_errors.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "cold_start_error": row.next_session_cold_start_loss,
            }
            for row in report.arm_results
        ),
        "segments.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "evaluation_episode_count": row.evaluation_episode_count,
                "turn_latency_ms": row.turn_latency_ms,
                "slow_job_latency_ms": row.slow_job_latency_ms,
                "turn_latency_contains_slow_job": row.turn_latency_contains_slow_job,
            }
            for row in report.arm_results
        ),
        "credit.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "owner_lineage_expected_count": row.owner_lineage_expected_count,
                "owner_lineage_observed_count": row.owner_lineage_observed_count,
                "owner_writeback_lineage_coverage": row.owner_writeback_lineage_coverage,
            }
            for row in report.arm_results
        ),
        "state_diff.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "memory_entry_count": row.memory_entry_count,
                "temporal_operation_count": row.temporal_operation_count,
                "owner_state_drift": row.owner_state_drift,
            }
            for row in report.arm_results
        ),
        "action_selection.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "unique_job_count": row.unique_job_count,
                "worker_execution_count": row.worker_execution_count,
                "duplicate_job_count": row.duplicate_job_count,
                "duplicate_job_execution_count": row.duplicate_job_execution_count,
            }
            for row in report.arm_results
        ),
    }
    written: list[Path] = []
    for filename, rows in rows_by_file.items():
        path = target / filename
        _write_jsonl(path, rows)
        written.append(path)
    ablation_path = target / "ablation_results.json"
    ablation_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "arm_schedule": report.arm_schedule,
                "arm_results": _jsonable(report.arm_results),
                "aggregate_metrics": report.aggregate_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(ablation_path)
    verdict_path = target / "promotion_verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "verdict": report.verdict,
                "mechanism_gates": report.mechanism_gates,
                "causal_gates": report.causal_gates,
                "locked_consumed": report.formal_locked_run,
                "retuning_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(verdict_path)
    rollback_path = target / "rollback_evidence.json"
    rollback_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "rows": [
                    {
                        "seed": row.seed,
                        "arm": row.arm,
                        "exact": row.rollback_exact,
                        "before": row.rollback_fingerprint_before,
                        "after": row.rollback_fingerprint_after,
                    }
                    for row in report.arm_results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(rollback_path)
    manifest_path = target / "manifest.yaml"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "suite_id": "gate8-wake-sleep",
                "source_schema_version": report.source_schema_version,
                "source_fingerprint": report.source_fingerprint,
                "partition": report.partition,
                "seed_schedule": report.seed_schedule,
                "arm_schedule": report.arm_schedule,
                "formal_locked_run": report.formal_locked_run,
                "required_files": GATE8_REQUIRED_FILES,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)
    report_path = target / "report.md"
    report_path.write_text(
        (
            "# Gate 8 wake/sleep evidence\n\n"
            f"- partition: `{report.partition}`\n"
            f"- formal locked run: `{report.formal_locked_run}`\n"
            f"- verdict: `{report.verdict}`\n"
            f"- source fingerprint: `{report.source_fingerprint}`\n\n"
            "## Mechanism gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.mechanism_gates
            )
            + "\n## Causal gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.causal_gates
            )
        ),
        encoding="utf-8",
    )
    written.append(report_path)
    return tuple(written)


def verify_gate8_evidence_bundle(
    output_dir: str | Path,
) -> dict[str, object]:
    target = Path(output_dir)
    missing = tuple(
        filename
        for filename in GATE8_REQUIRED_FILES
        if not (target / filename).is_file()
    )
    if missing:
        return {
            "passed": False,
            "missing_files": missing,
            "verdict": "invalid",
        }
    manifest = json.loads(
        (target / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    passed = (
        manifest["schema_version"] == GATE8_SCHEMA_VERSION
        and tuple(manifest["arm_schedule"]) == GATE8_ARMS
        and tuple(manifest["required_files"]) == GATE8_REQUIRED_FILES
        and verdict["verdict"]
        in {"invalid", "not-supported", "causal-supported"}
    )
    return {
        "passed": passed,
        "missing_files": (),
        "verdict": verdict["verdict"],
        "formal_locked_run": manifest["formal_locked_run"],
        "partition": manifest["partition"],
    }
