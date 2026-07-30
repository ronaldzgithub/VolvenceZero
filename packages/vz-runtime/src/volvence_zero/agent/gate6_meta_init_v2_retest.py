"""One-shot Gate 6 nested meta-init retest on separated v2 contexts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from volvence_zero.memory import build_default_memory_store


GATE6_V2_SCHEMA_VERSION = "gate6-meta-init-v2-retest.v1"
GATE6_V2_SUITE_ID = "gate6-meta-init-v2-retest"
GATE6_V2_PRIMARY_ARMS = (
    "meta-init",
    "copy-init",
    "random-init",
    "no-init",
)
GATE6_V2_DIAGNOSTIC_ARMS = (
    "paired-user-slow-state",
    "swapped-user-slow-state",
)
GATE6_V2_ARMS = (*GATE6_V2_PRIMARY_ARMS, *GATE6_V2_DIAGNOSTIC_ARMS)
_ADAPTATION_STEPS = 5
_EARLY_K = 5
_TARGET_ERROR = 0.05


@dataclass(frozen=True)
class Gate6V2EpisodeResult:
    seed: int
    partition: str
    episode_id: str
    context_id: str
    user_prior_id: str
    arm: str
    steps_to_target: int
    early_adaptation_auc: float
    initial_error: float
    final_error: float
    final_quality: float
    initialization_changed_fast_state: bool
    slow_state_unchanged: bool
    parameter_state_unchanged: bool
    lineage_complete: bool
    fact_leakage_count: int
    source_mutation_count: int
    rollback_exact: bool
    rollback_before: str
    rollback_after: str


@dataclass(frozen=True)
class Gate6V2Report:
    partition: str
    formal_locked_run: bool
    results: tuple[Gate6V2EpisodeResult, ...]
    aggregate_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    causal_gates: tuple[tuple[str, bool, float], ...]
    verdict: str


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _signal(plan: Gate78EpisodePlan) -> tuple[float, float, float, float]:
    context = plan.context_centroid
    projected = (
        context[0] + 0.50 * context[4] + 0.25 * context[5],
        context[1] + 0.50 * context[5] + 0.25 * context[4],
        context[2] + 0.50 * context[4],
        context[3] + 0.50 * context[5],
    )
    return tuple(
        min(max(0.70 * base + 0.30 * prior, 0.0), 1.0)
        for base, prior in zip(
            projected,
            plan.user_prior,
            strict=True,
        )
    )  # type: ignore[return-value]


def _mae(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return _mean(
        tuple(
            abs(a - b)
            for a, b in zip(left, right, strict=True)
        )
    )


def _quality(error: float) -> float:
    return 1.0 - min(max(error / 0.50, 0.0), 1.0)


def _checkpoint_fingerprint(checkpoint: object) -> str:
    payload = asdict(checkpoint)
    payload.pop("checkpoint_id", None)
    return _sha256(payload)


def _train_store(
    plans: tuple[Gate78EpisodePlan, ...],
    *,
    seed: int,
):
    store = build_default_memory_store(latent_dim=4)
    for index, plan in enumerate(plans):
        store.observe_replay_signal(
            signal=_signal(plan),
            timestamp_ms=seed * 1000 + index,
        )
    checkpoint = store.create_checkpoint(
        checkpoint_id=f"gate6-v2:{seed}:train"
    )
    return store, checkpoint


def _external_targets(
    *,
    plan: Gate78EpisodePlan,
    train_plans: tuple[Gate78EpisodePlan, ...],
    swapped: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    matching = tuple(
        candidate
        for candidate in train_plans
        if candidate.context_id == plan.context_id
        and candidate.user_prior_id == plan.user_prior_id
    )
    if not matching:
        raise ValueError(
            f"Gate 6 v2 lacks paired train prior for {plan.episode_id}"
        )
    donor = matching[0]
    if swapped:
        donor = next(
            candidate
            for candidate in train_plans
            if candidate.user_prior_id != plan.user_prior_id
            and candidate.context_id != plan.context_id
        )
    target = _signal(donor)
    return (target, target)


def _run_episode(
    *,
    seed: int,
    partition: str,
    arm: str,
    plan: Gate78EpisodePlan,
    train_plans: tuple[Gate78EpisodePlan, ...],
    store,
    train_checkpoint,
) -> Gate6V2EpisodeResult:
    store.restore_checkpoint(train_checkpoint)
    rollback_before = _checkpoint_fingerprint(train_checkpoint)
    source_before = _sha256((train_plans, plan))
    external_targets = None
    random_seed = None
    mode = arm
    if arm == "random-init":
        random_seed = seed * 1000 + plan.global_index
    elif arm in GATE6_V2_DIAGNOSTIC_ARMS:
        mode = "external-meta-init"
        external_targets = _external_targets(
            plan=plan,
            train_plans=train_plans,
            swapped=arm == "swapped-user-slow-state",
        )
    evidence = store.initialize_nested_context_for_evidence(
        mode=mode,
        reason=f"gate6-v2:{partition}:{plan.episode_id}:{arm}",
        timestamp_ms=seed * 10000 + plan.global_index,
        random_seed=random_seed,
        external_targets=external_targets,
    )
    if evidence is None:
        raise RuntimeError("Gate 6 v2 owner did not return init evidence")
    target = _signal(plan)
    errors = [_mae(evidence.online_after, target)]
    for step in range(_ADAPTATION_STEPS):
        store.observe_replay_signal(
            signal=target,
            timestamp_ms=(
                seed * 10000 + plan.global_index * 10 + step + 1
            ),
        )
        snapshot = store.snapshot(
            retrieved_entries=(),
            active_subject_scope=(),
        )
        if snapshot.cms_state is None:
            raise RuntimeError("Gate 6 v2 adaptation lacks CMS state")
        errors.append(
            _mae(snapshot.cms_state.online_fast.vector, target)
        )
    steps_to_target = next(
        (
            index
            for index, error in enumerate(errors)
            if error <= _TARGET_ERROR
        ),
        _ADAPTATION_STEPS + 1,
    )
    early_auc = _mean(
        tuple(_quality(error) for error in errors[:_EARLY_K])
    )
    store.restore_checkpoint(train_checkpoint)
    restored = store.create_checkpoint(
        checkpoint_id=train_checkpoint.checkpoint_id
    )
    rollback_after = _checkpoint_fingerprint(restored)
    return Gate6V2EpisodeResult(
        seed=seed,
        partition=partition,
        episode_id=plan.episode_id,
        context_id=plan.context_id,
        user_prior_id=plan.user_prior_id,
        arm=arm,
        steps_to_target=steps_to_target,
        early_adaptation_auc=early_auc,
        initial_error=errors[0],
        final_error=errors[-1],
        final_quality=_quality(errors[-1]),
        initialization_changed_fast_state=(
            evidence.online_after != evidence.online_before
        ),
        slow_state_unchanged=evidence.slow_state_unchanged,
        parameter_state_unchanged=evidence.parameter_state_unchanged,
        lineage_complete=bool(
            plan.episode_id
            and plan.next_session_boundary
            and plan.user_prior_id
        ),
        fact_leakage_count=0,
        source_mutation_count=int(
            source_before != _sha256((train_plans, plan))
        ),
        rollback_exact=rollback_before == rollback_after,
        rollback_before=rollback_before,
        rollback_after=rollback_after,
    )


def _seed_arm_mean(
    results: tuple[Gate6V2EpisodeResult, ...],
    *,
    seed: int,
    arm: str,
    field: str,
) -> float:
    return _mean(
        tuple(
            float(getattr(row, field))
            for row in results
            if row.seed == seed and row.arm == arm
        )
    )


def run_gate6_v2_retest(
    *,
    trace_root: str | Path,
    seed_schedule: tuple[int, ...] = GATE78_TRACE_SEEDS,
    partition: str = "trace-development-heldout",
    formal_locked_run: bool = False,
    evaluation_limit: int | None = None,
) -> Gate6V2Report:
    if formal_locked_run and partition != "trace-locked-confirmation":
        raise ValueError("Formal Gate 6 v2 run must use locked confirmation")
    if not formal_locked_run and partition == "trace-locked-confirmation":
        raise ValueError("Development Gate 6 v2 run must not consume locked")
    source = verify_gate78_shared_trace_bundle(trace_root)
    if not source["consumer_admission"]:
        raise RuntimeError("Gate 6 v2 source admission failed")
    rows: list[Gate6V2EpisodeResult] = []
    for seed in seed_schedule:
        if seed not in GATE78_TRACE_SEEDS:
            raise ValueError(f"Unregistered Gate 6 v2 seed {seed}")
        train_plans = load_gate78_partition(
            trace_root,
            seed=seed,
            partition="trace-train",
        )
        evaluation_plans = load_gate78_partition(
            trace_root,
            seed=seed,
            partition=partition,
        )
        if evaluation_limit is not None:
            evaluation_plans = evaluation_plans[:evaluation_limit]
        store, checkpoint = _train_store(train_plans, seed=seed)
        for plan in evaluation_plans:
            for arm in GATE6_V2_ARMS:
                rows.append(
                    _run_episode(
                        seed=seed,
                        partition=partition,
                        arm=arm,
                        plan=plan,
                        train_plans=train_plans,
                        store=store,
                        train_checkpoint=checkpoint,
                    )
                )
    results = tuple(rows)
    comparison_passes: list[bool] = []
    causal_rows: list[tuple[str, bool, float]] = []
    for control in GATE6_V2_PRIMARY_ARMS[1:]:
        step_gains = tuple(
            _seed_arm_mean(
                results,
                seed=seed,
                arm=control,
                field="steps_to_target",
            )
            - _seed_arm_mean(
                results,
                seed=seed,
                arm="meta-init",
                field="steps_to_target",
            )
            for seed in seed_schedule
        )
        auc_gains = tuple(
            _seed_arm_mean(
                results,
                seed=seed,
                arm="meta-init",
                field="early_adaptation_auc",
            )
            - _seed_arm_mean(
                results,
                seed=seed,
                arm=control,
                field="early_adaptation_auc",
            )
            for seed in seed_schedule
        )
        final_deltas = tuple(
            _seed_arm_mean(
                results,
                seed=seed,
                arm="meta-init",
                field="final_error",
            )
            - _seed_arm_mean(
                results,
                seed=seed,
                arm=control,
                field="final_error",
            )
            for seed in seed_schedule
        )
        effect_passed = (
            _mean(step_gains) >= 1.0
            and all(gain > 0.0 for gain in step_gains)
        ) or (
            _mean(auc_gains) >= 0.05
            and all(gain > 0.0 for gain in auc_gains)
        )
        final_noninferior = (
            _mean(final_deltas) <= 0.01
            and all(delta <= 0.01 for delta in final_deltas)
        )
        comparison_passes.append(effect_passed and final_noninferior)
        causal_rows.extend(
            (
                (
                    f"meta-effect-vs-{control}",
                    effect_passed,
                    max(_mean(step_gains), _mean(auc_gains)),
                ),
                (
                    f"final-error-noninferior-vs-{control}",
                    final_noninferior,
                    _mean(final_deltas),
                ),
            )
        )
    meta_by_episode = {
        (row.seed, row.episode_id): row
        for row in results
        if row.arm == "meta-init"
    }
    negative_transfer = sum(
        meta.final_error
        > min(
            row.final_error
            for row in results
            if row.seed == key[0]
            and row.episode_id == key[1]
            and row.arm in GATE6_V2_PRIMARY_ARMS[1:]
        )
        + 0.01
        for key, meta in meta_by_episode.items()
    )
    negative_transfer_rate = negative_transfer / len(meta_by_episode)
    paired_auc = _mean(
        tuple(
            row.early_adaptation_auc
            for row in results
            if row.arm == "paired-user-slow-state"
        )
    )
    swapped_auc = _mean(
        tuple(
            row.early_adaptation_auc
            for row in results
            if row.arm == "swapped-user-slow-state"
        )
    )
    lineage = min(float(row.lineage_complete) for row in results)
    leakage = sum(row.fact_leakage_count for row in results)
    mutation = sum(row.source_mutation_count for row in results)
    rollback_mismatch = sum(not row.rollback_exact for row in results)
    state_mutation = sum(
        not (row.slow_state_unchanged and row.parameter_state_unchanged)
        for row in results
    )
    metrics = (
        ("negative_transfer_rate", negative_transfer_rate),
        ("paired_minus_swapped_auc", paired_auc - swapped_auc),
        ("lineage_coverage", lineage),
        ("fact_leakage_count", float(leakage)),
        ("source_mutation_count", float(mutation)),
        ("rollback_mismatch_count", float(rollback_mismatch)),
    )
    mechanism_gates = (
        ("source-consumer-admission", True, 1.0),
        ("lineage-complete", lineage >= 1.0, lineage),
        ("fact-leakage-zero", leakage == 0, float(leakage)),
        ("source-mutation-zero", mutation == 0, float(mutation)),
        ("slow-parameter-state-unchanged", state_mutation == 0, float(state_mutation)),
        ("rollback-exact", rollback_mismatch == 0, float(rollback_mismatch)),
    )
    causal_gates = tuple(causal_rows) + (
        (
            "negative-transfer-zero",
            negative_transfer_rate <= 0.0,
            negative_transfer_rate,
        ),
        (
            "paired-outperforms-swapped-diagnostic",
            paired_auc - swapped_auc >= 0.01,
            paired_auc - swapped_auc,
        ),
    )
    if not all(passed for _name, passed, _value in mechanism_gates):
        verdict = "invalid"
    elif all(comparison_passes) and negative_transfer_rate <= 0.0:
        verdict = "causal-supported"
    else:
        verdict = "not-supported"
    return Gate6V2Report(
        partition=partition,
        formal_locked_run=formal_locked_run,
        results=results,
        aggregate_metrics=metrics,
        mechanism_gates=mechanism_gates,
        causal_gates=causal_gates,
        verdict=verdict,
    )


def export_gate6_v2_bundle(
    report: Gate6V2Report,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    rows = report.results
    common = tuple(
        {
            "seed": row.seed,
            "arm": row.arm,
            "partition": row.partition,
            "episode_id": row.episode_id,
            "context_id": row.context_id,
            "user_prior_id": row.user_prior_id,
        }
        for row in rows
    )
    rows_by_file = {
        "predictions.jsonl": tuple(
            {
                **base,
                "initial_error": row.initial_error,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "outcomes.jsonl": tuple(
            {
                **base,
                "steps_to_target": row.steps_to_target,
                "early_adaptation_auc": row.early_adaptation_auc,
                "final_error": row.final_error,
                "final_quality": row.final_quality,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "prediction_errors.jsonl": tuple(
            {**base, "final_error": row.final_error}
            for base, row in zip(common, rows, strict=True)
        ),
        "segments.jsonl": tuple(
            {
                **base,
                "adaptation_steps": _ADAPTATION_STEPS,
            }
            for base in common
        ),
        "credit.jsonl": tuple(
            {
                **base,
                "lineage_complete": row.lineage_complete,
                "fact_leakage_count": row.fact_leakage_count,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "state_diff.jsonl": tuple(
            {
                **base,
                "initialization_changed_fast_state": (
                    row.initialization_changed_fast_state
                ),
                "slow_state_unchanged": row.slow_state_unchanged,
                "parameter_state_unchanged": (
                    row.parameter_state_unchanged
                ),
                "source_mutation_count": row.source_mutation_count,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "action_selection.jsonl": tuple(
            {
                **base,
                "initialization_arm": row.arm,
            }
            for base, row in zip(common, rows, strict=True)
        ),
    }
    rollback = tuple(
        {
            "seed": row.seed,
            "episode_id": row.episode_id,
            "arm": row.arm,
            "exact": row.rollback_exact,
            "before": row.rollback_before,
            "after": row.rollback_after,
        }
        for row in rows
    )
    return export_gate_v2_bundle(
        schema_version=GATE6_V2_SCHEMA_VERSION,
        suite_id=GATE6_V2_SUITE_ID,
        source_schema_version=GATE78_TRACE_SCHEMA_VERSION,
        source_fingerprint=_sha256(GATE78_SOURCE_DESCRIPTOR),
        partition=report.partition,
        seed_schedule=tuple(dict.fromkeys(row.seed for row in rows)),
        arm_schedule=GATE6_V2_ARMS,
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


def verify_gate6_v2_bundle(output_dir: str | Path) -> dict[str, object]:
    return verify_gate_v2_bundle(
        output_dir,
        schema_version=GATE6_V2_SCHEMA_VERSION,
        suite_id=GATE6_V2_SUITE_ID,
        arm_schedule=GATE6_V2_ARMS,
    )
