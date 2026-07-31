"""One-shot Gate 1 PE-drive causal retest on the Gate 7/8 v2 corpus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from pathlib import Path
import statistics

from volvence_zero.agent.gate78_shared_trace import (
    GATE78_SOURCE_DESCRIPTOR,
    GATE78_TRACE_SCHEMA_VERSION,
    GATE78_TRACE_SEEDS,
    Gate78EpisodePlan,
    load_gate78_partition,
    verify_gate78_shared_trace_bundle,
)
from volvence_zero.agent.gate_v2_retest_common import (
    canonical_json,
    export_gate_v2_bundle,
    verify_gate_v2_bundle,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionErrorModule,
    PredictionErrorSnapshot,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    TemporalModule,
)


GATE1_V2_SCHEMA_VERSION = "gate1-pe-causal-v3-retest.v1"
GATE1_V2_SUITE_ID = "gate1-pe-causal-v3-retest"
GATE1_V2_ARMS = ("pe-eta-v3", "pe-drive-off-v3")


@dataclass(frozen=True)
class Gate1V2ArmResult:
    seed: int
    partition: str
    arm: str
    episode_count: int
    mean_next_session_policy_loss: float
    mean_prediction_error_magnitude: float
    pe_applied_count: int
    temporal_parameter_change_count: int
    lineage_coverage: float
    source_mutation_count: int
    rollback_exact: bool
    rollback_before: str
    rollback_after: str


@dataclass(frozen=True)
class Gate1V2Report:
    partition: str
    formal_locked_run: bool
    results: tuple[Gate1V2ArmResult, ...]
    aggregate_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    causal_gates: tuple[tuple[str, bool, float], ...]
    verdict: str


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _substrate_snapshot(
    *,
    plan: Gate78EpisodePlan,
    session: int,
) -> SubstrateSnapshot:
    turns = (
        plan.session_one_turns
        if session == 1
        else plan.session_two_turns
    )
    trace = build_training_trace(
        trace_id=f"{plan.episode_id}:session-{session}",
        source_text=" ".join(turns),
    )
    step = trace.steps[-1]
    prefix = tuple(
        ResidualSequenceStep(
            step=item.step,
            token=item.token,
            feature_surface=item.feature_surface,
            residual_activations=item.residual_activations,
            description=(
                f"{plan.episode_id} session-{session} causal prefix "
                f"{item.step}"
            ),
        )
        for item in trace.steps
    )
    return SubstrateSnapshot(
        model_id=f"gate78-frozen:{plan.seed}",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=tuple(
            sum(signal.values) / len(signal.values)
            for signal in step.feature_surface
        ),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=prefix,
        unavailable_fields=(),
        description=(
            f"{plan.episode_id} session-{session}; "
            f"boundary={plan.next_session_boundary}"
        ),
    )


def _prediction_error(plan: Gate78EpisodePlan) -> PredictionErrorSnapshot:
    context = PredictionActionContext(
        segment_id=f"{plan.episode_id}:session-one",
        abstract_action_id=plan.action_family_ids[-1],
        z_t_digest=plan.user_prior,
        environment_event_id=f"{plan.episode_id}:event",
        environment_outcome_id=f"{plan.episode_id}:outcome",
        environment_task_progress=1.0 - plan.difficulty,
        environment_action_payoff=plan.user_prior[2],
        environment_outcome_terminal=True,
    )
    predicted = PredictedOutcome(
        source_turn_index=0,
        target_turn_index=1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.7,
        description=f"Gate 1 v2 prediction for {plan.episode_id}.",
        action_context=context,
        prediction_id=f"{plan.episode_id}:prediction",
    )
    actual = ActualOutcome(
        observed_turn_index=1,
        task_progress=1.0 - plan.difficulty,
        relationship_delta=plan.user_prior[0],
        regime_stability=plan.user_prior[1],
        action_payoff=plan.user_prior[2],
        description=f"Gate 1 v2 settled outcome for {plan.episode_id}.",
        action_context=context,
    )
    error = PredictionErrorModule().compute_prediction_error(
        predicted=predicted,
        actual_outcome=actual,
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=actual,
        next_prediction=predicted,
        error=error,
        turn_index=1,
        bootstrap=False,
        description=(
            f"Owner-computed Gate 1 v2 PE for {plan.episode_id}."
        ),
        action_context=context,
    )


def _loss(
    controller_code: tuple[float, ...],
    target: tuple[float, ...],
) -> float:
    return _mean(
        tuple(
            abs(observed - expected)
            for observed, expected in zip(
                controller_code,
                target,
                strict=True,
            )
        )
    )


def _run_arm(
    *,
    seed: int,
    partition: str,
    arm: str,
    plans: tuple[Gate78EpisodePlan, ...],
) -> Gate1V2ArmResult:
    losses: list[float] = []
    magnitudes: list[float] = []
    applied_count = 0
    parameter_change_count = 0
    lineage_count = 0
    mutation_count = 0
    rollback_rows: list[bool] = []
    before_fingerprints: list[str] = []
    after_fingerprints: list[str] = []
    for plan in plans:
        policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(
                n_z=4,
                initialization_seed=seed,
            )
        )
        policy.set_prediction_error_runtime_modulation_enabled(True)
        module = TemporalModule(
            policy=policy,
            wiring_level=WiringLevel.ACTIVE,
        )
        checkpoint = policy.export_rare_heavy_snapshot()
        before = _sha256(checkpoint)
        session_one = _substrate_snapshot(plan=plan, session=1)
        session_two = _substrate_snapshot(plan=plan, session=2)
        source_before = _sha256((session_one, session_two))
        pe_snapshot = _prediction_error(plan)
        signal = pe_snapshot if arm == "pe-eta-v3" else None
        asyncio.run(
            module.process_standalone(
                substrate_snapshot=session_one,
                prediction_error_snapshot=signal,
            )
        )
        parameter_change_count += int(
            _sha256(policy.export_parameters())
            != _sha256(checkpoint.temporal_parameters)
        )
        next_snapshot = asyncio.run(
            module.process_standalone(
                substrate_snapshot=session_two,
                prediction_error_snapshot=None,
            )
        )
        losses.append(
            _loss(
                next_snapshot.value.controller_state.code,
                plan.user_prior,
            )
        )
        magnitudes.append(pe_snapshot.error.magnitude)
        applied_count += int(signal is not None)
        lineage_count += int(
            bool(
                pe_snapshot.evaluated_prediction is not None
                and pe_snapshot.evaluated_prediction.prediction_id
                and pe_snapshot.action_context.environment_outcome_id
            )
        )
        mutation_count += int(
            source_before != _sha256((session_one, session_two))
        )
        policy.apply_rare_heavy_snapshot(checkpoint)
        after = _sha256(policy.export_rare_heavy_snapshot())
        rollback_rows.append(before == after)
        before_fingerprints.append(before)
        after_fingerprints.append(after)
    return Gate1V2ArmResult(
        seed=seed,
        partition=partition,
        arm=arm,
        episode_count=len(plans),
        mean_next_session_policy_loss=_mean(tuple(losses)),
        mean_prediction_error_magnitude=_mean(tuple(magnitudes)),
        pe_applied_count=applied_count,
        temporal_parameter_change_count=parameter_change_count,
        lineage_coverage=lineage_count / len(plans),
        source_mutation_count=mutation_count,
        rollback_exact=all(rollback_rows),
        rollback_before=_sha256(tuple(before_fingerprints)),
        rollback_after=_sha256(tuple(after_fingerprints)),
    )


def run_gate1_v2_retest(
    *,
    trace_root: str | Path,
    seed_schedule: tuple[int, ...] = GATE78_TRACE_SEEDS,
    partition: str = "trace-development-heldout",
    formal_locked_run: bool = False,
    evaluation_limit: int | None = None,
) -> Gate1V2Report:
    if formal_locked_run and partition != "trace-locked-confirmation":
        raise ValueError("Formal Gate 1 v2 run must use locked confirmation")
    if not formal_locked_run and partition == "trace-locked-confirmation":
        raise ValueError("Development Gate 1 v2 run must not consume locked")
    source = verify_gate78_shared_trace_bundle(trace_root)
    if not source["consumer_admission"]:
        raise RuntimeError("Gate 1 v2 source admission failed")
    rows: list[Gate1V2ArmResult] = []
    for seed in seed_schedule:
        if seed not in GATE78_TRACE_SEEDS:
            raise ValueError(f"Unregistered Gate 1 v2 seed {seed}")
        plans = load_gate78_partition(
            trace_root,
            seed=seed,
            partition=partition,
        )
        if evaluation_limit is not None:
            plans = plans[:evaluation_limit]
        for arm in GATE1_V2_ARMS:
            rows.append(
                _run_arm(
                    seed=seed,
                    partition=partition,
                    arm=arm,
                    plans=plans,
                )
            )
    results = tuple(rows)
    seed_gains = tuple(
        next(
            row.mean_next_session_policy_loss
            for row in results
            if row.seed == seed and row.arm == "pe-drive-off-v3"
        )
        - next(
            row.mean_next_session_policy_loss
            for row in results
            if row.seed == seed and row.arm == "pe-eta-v3"
        )
        for seed in seed_schedule
    )
    mean_gain = _mean(seed_gains)
    lineage = min(row.lineage_coverage for row in results)
    mutation = sum(row.source_mutation_count for row in results)
    rollback_mismatch = sum(not row.rollback_exact for row in results)
    metrics = (
        ("mean_policy_loss_reduction", mean_gain),
        ("minimum_seed_policy_loss_reduction", min(seed_gains)),
        ("prediction_error_lineage_coverage", lineage),
        ("source_mutation_count", float(mutation)),
        ("rollback_mismatch_count", float(rollback_mismatch)),
    )
    mechanism_gates = (
        ("source-consumer-admission", True, 1.0),
        ("prediction-error-lineage-complete", lineage >= 1.0, lineage),
        ("frozen-source-mutation-zero", mutation == 0, float(mutation)),
        ("rollback-exact", rollback_mismatch == 0, float(rollback_mismatch)),
    )
    causal_gates = (
        ("mean-policy-loss-reduction", mean_gain >= 0.05, mean_gain),
        (
            "every-seed-direction-positive",
            all(gain > 0.0 for gain in seed_gains),
            min(seed_gains),
        ),
    )
    if not all(passed for _name, passed, _value in mechanism_gates):
        verdict = "invalid"
    elif all(passed for _name, passed, _value in causal_gates):
        verdict = "causal-supported"
    else:
        verdict = "not-supported"
    return Gate1V2Report(
        partition=partition,
        formal_locked_run=formal_locked_run,
        results=results,
        aggregate_metrics=metrics,
        mechanism_gates=mechanism_gates,
        causal_gates=causal_gates,
        verdict=verdict,
    )


def export_gate1_v2_bundle(
    report: Gate1V2Report,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    rows = report.results
    common = tuple(
        {
            "seed": row.seed,
            "arm": row.arm,
            "partition": row.partition,
        }
        for row in rows
    )
    rows_by_file = {
        "predictions.jsonl": tuple(
            {
                **base,
                "pe_applied_count": row.pe_applied_count,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "outcomes.jsonl": tuple(
            {
                **base,
                "next_session_policy_loss": (
                    row.mean_next_session_policy_loss
                ),
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "prediction_errors.jsonl": tuple(
            {
                **base,
                "mean_prediction_error_magnitude": (
                    row.mean_prediction_error_magnitude
                ),
                "lineage_coverage": row.lineage_coverage,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "segments.jsonl": tuple(
            {**base, "episode_count": row.episode_count}
            for base, row in zip(common, rows, strict=True)
        ),
        "credit.jsonl": tuple(
            {**base, "pe_drive_active": row.arm == "pe-eta-v3"}
            for base, row in zip(common, rows, strict=True)
        ),
        "state_diff.jsonl": tuple(
            {
                **base,
                "source_mutation_count": row.source_mutation_count,
                "temporal_parameter_change_count": (
                    row.temporal_parameter_change_count
                ),
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "action_selection.jsonl": tuple(
            {
                **base,
                "target_source": "gate78-numeric-user-prior",
            }
            for base in common
        ),
    }
    rollback = tuple(
        {
            "seed": row.seed,
            "arm": row.arm,
            "exact": row.rollback_exact,
            "before": row.rollback_before,
            "after": row.rollback_after,
        }
        for row in rows
    )
    return export_gate_v2_bundle(
        schema_version=GATE1_V2_SCHEMA_VERSION,
        suite_id=GATE1_V2_SUITE_ID,
        source_schema_version=GATE78_TRACE_SCHEMA_VERSION,
        source_fingerprint=_sha256(GATE78_SOURCE_DESCRIPTOR),
        partition=report.partition,
        seed_schedule=tuple(dict.fromkeys(row.seed for row in rows)),
        arm_schedule=GATE1_V2_ARMS,
        formal_locked_run=report.formal_locked_run,
        rows_by_file=rows_by_file,
        arm_results=rows,
        aggregate_metrics=report.aggregate_metrics,
        mechanism_gates=report.mechanism_gates,
        causal_gates=report.causal_gates,
        verdict=report.verdict,
        rollback_rows=rollback,
        output_dir=output_dir,
    )


def verify_gate1_v2_bundle(output_dir: str | Path) -> dict[str, object]:
    return verify_gate_v2_bundle(
        output_dir,
        schema_version=GATE1_V2_SCHEMA_VERSION,
        suite_id=GATE1_V2_SUITE_ID,
        arm_schedule=GATE1_V2_ARMS,
    )
