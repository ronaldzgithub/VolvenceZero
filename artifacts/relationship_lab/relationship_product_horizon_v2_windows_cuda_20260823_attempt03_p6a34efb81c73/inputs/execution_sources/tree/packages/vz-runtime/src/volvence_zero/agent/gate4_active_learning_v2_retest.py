"""One-shot Gate 4 segment-aware active-learning retest on v2 traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import statistics

from volvence_zero.apprenticeship import (
    BoundedBinaryAlignmentReadout,
    BoundedLabelUtilityReadout,
    Gate4FeedbackCandidate,
    label_utility_features,
    random_feedback_order,
    select_active_candidate,
)
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


GATE4_V2_SCHEMA_VERSION = "gate4-label-utility-v3-retest.v1"
GATE4_V2_SUITE_ID = "gate4-label-utility-v3-retest"
GATE4_V2_ARMS = (
    "segment-aware-active",
    "turn-level-active",
    "random-feedback",
    "no-feedback",
    "shuffled-segment-boundary",
)
_LABEL_BUDGET = 12
_BOOTSTRAP_LABELS = 4
_TARGET_ACCURACY = 0.80


@dataclass(frozen=True)
class Gate4V2ArmResult:
    seed: int
    partition: str
    arm: str
    train_candidate_count: int
    evaluation_count: int
    requested_label_count: int
    labels_needed_for_target: int
    final_balanced_accuracy: float
    cumulative_regret: int
    selected_positive_rate: float
    typed_candidate_lineage_coverage: float
    source_mutation_count: int
    isolated_reset_exact: bool
    utility_observation_count: int
    learned_utility_selection_count: int
    cold_start_selection_count: int


@dataclass(frozen=True)
class Gate4V2Report:
    partition: str
    formal_locked_run: bool
    results: tuple[Gate4V2ArmResult, ...]
    aggregate_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    causal_gates: tuple[tuple[str, bool, float], ...]
    verdict: str


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _binary_log_loss(probability: float, label: bool) -> float:
    bounded = max(1e-9, min(1.0 - 1e-9, probability))
    return (
        -math.log(bounded)
        if label
        else -math.log(1.0 - bounded)
    )


def _predictor_features(plan: Gate78EpisodePlan) -> tuple[float, ...]:
    return (
        *plan.user_prior,
        *plan.context_centroid,
        plan.difficulty,
        _mean(tuple(float(value) for value in plan.segment_lengths)) / 5.0,
    )


def _segment_features(plan: Gate78EpisodePlan) -> tuple[float, ...]:
    family_presence = tuple(
        float(family in plan.action_family_ids)
        for family in ("alpha", "beta", "gamma", "delta", "epsilon")
    )
    lengths = tuple(float(value) for value in plan.segment_lengths)
    return (
        *plan.context_centroid,
        *family_presence,
        min(lengths) / 5.0,
        max(lengths) / 5.0,
        _mean(lengths),
    )


def _candidate(plan: Gate78EpisodePlan) -> Gate4FeedbackCandidate:
    return Gate4FeedbackCandidate(
        transition_id=f"{plan.episode_id}:closed-segment",
        guidance_text=plan.session_one_turns[-1],
        turn_index=plan.global_index,
        predictor_features=_predictor_features(plan),
        pe_magnitude=plan.difficulty,
        segment_features=_segment_features(plan),
    )


def _label(plan: Gate78EpisodePlan) -> bool:
    return plan.difficulty >= 0.60


def _balanced_accuracy(
    *,
    probabilities: tuple[float, ...],
    labels: tuple[bool, ...],
) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Gate 4 v2 evaluation requires both label classes")
    true_positive = sum(
        probability >= 0.5 and label
        for probability, label in zip(
            probabilities,
            labels,
            strict=True,
        )
    )
    true_negative = sum(
        probability < 0.5 and not label
        for probability, label in zip(
            probabilities,
            labels,
            strict=True,
        )
    )
    return (true_positive / positives + true_negative / negatives) / 2.0


def _shuffled_candidates(
    candidates: tuple[Gate4FeedbackCandidate, ...],
    *,
    seed: int,
) -> tuple[Gate4FeedbackCandidate, ...]:
    order = tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: hashlib.sha256(
                f"gate4-v2:{seed}:{index}".encode("utf-8")
            ).hexdigest(),
        )
    )
    shuffled = tuple(candidates[index].segment_features for index in order)
    return tuple(
        Gate4FeedbackCandidate(
            transition_id=candidate.transition_id,
            guidance_text=candidate.guidance_text,
            turn_index=candidate.turn_index,
            predictor_features=candidate.predictor_features,
            pe_magnitude=candidate.pe_magnitude,
            segment_features=shuffled[index],
        )
        for index, candidate in enumerate(candidates)
    )


def _run_arm(
    *,
    seed: int,
    partition: str,
    arm: str,
    train_plans: tuple[Gate78EpisodePlan, ...],
    evaluation_plans: tuple[Gate78EpisodePlan, ...],
) -> Gate4V2ArmResult:
    candidates = tuple(_candidate(plan) for plan in train_plans)
    selection_candidates = (
        _shuffled_candidates(candidates, seed=seed)
        if arm == "shuffled-segment-boundary"
        else candidates
    )
    labels = tuple(_label(plan) for plan in train_plans)
    evaluation_features = tuple(
        _predictor_features(plan) for plan in evaluation_plans
    )
    evaluation_labels = tuple(_label(plan) for plan in evaluation_plans)
    source_before = _sha256((train_plans, evaluation_plans))
    readout = BoundedBinaryAlignmentReadout(
        feature_count=len(candidates[0].predictor_features)
    )
    utility_readout = BoundedLabelUtilityReadout(
        feature_count=5 + len(candidates[0].segment_features),
        minimum_observations=_BOOTSTRAP_LABELS,
    )
    initial_parameters = readout.parameters
    initial_utility_parameters = utility_readout.parameters
    selected: set[int] = set()
    selected_order: list[int] = []
    labels_needed = _LABEL_BUDGET + 1
    cumulative_regret = 0
    random_order = random_feedback_order(candidates)
    budget = 0 if arm == "no-feedback" else _LABEL_BUDGET
    learned_utility_selection_count = 0
    cold_start_selection_count = 0

    def refit() -> None:
        readout.fit(
            tuple(
                (
                    candidates[index].predictor_features,
                    labels[index],
                )
                for index in selected_order
            )
        )

    def evaluation_accuracy() -> float:
        return _balanced_accuracy(
            probabilities=tuple(
                readout.predict_probability(features)
                for features in evaluation_features
            ),
            labels=evaluation_labels,
        )

    while len(selected_order) < budget:
        probabilities = tuple(
            readout.predict_probability(candidate.predictor_features)
            for candidate in candidates
        )
        cumulative_regret += sum(
            index not in selected
            and labels[index]
            and probability < 0.5
            for index, probability in enumerate(probabilities)
        )
        if len(selected_order) < _BOOTSTRAP_LABELS:
            index = next(
                candidate_index
                for candidate_index in random_order
                if candidate_index not in selected
            )
        elif arm == "random-feedback":
            index = next(
                candidate_index
                for candidate_index in random_order
                if candidate_index not in selected
            )
        else:
            utility_ready = utility_readout.ready
            index = select_active_candidate(
                candidates=selection_candidates,
                probabilities=probabilities,
                selected_indices=frozenset(selected),
                policy=arm,
                utility_readout=utility_readout,
            )
            learned_utility_selection_count += int(utility_ready)
            cold_start_selection_count += int(not utility_ready)
        utility_features = label_utility_features(
            candidate=selection_candidates[index],
            probability=probabilities[index],
            selected_segment_features=tuple(
                selection_candidates[candidate_index].segment_features
                for candidate_index in sorted(selected)
            ),
        )
        utility_indices = (*selected_order, index)
        loss_before = _mean(
            tuple(
                _binary_log_loss(
                    probabilities[candidate_index],
                    labels[candidate_index],
                )
                for candidate_index in utility_indices
            )
        )
        selected.add(index)
        selected_order.append(index)
        refit()
        loss_after = _mean(
            tuple(
                _binary_log_loss(
                    readout.predict_probability(
                        candidates[candidate_index].predictor_features
                    ),
                    labels[candidate_index],
                )
                for candidate_index in utility_indices
            )
        )
        utility_readout.observe_loss_delta(
            features=utility_features,
            loss_before=loss_before,
            loss_after=loss_after,
        )
        if (
            labels_needed == _LABEL_BUDGET + 1
            and evaluation_accuracy() >= _TARGET_ACCURACY
        ):
            labels_needed = len(selected_order)
    final_accuracy = evaluation_accuracy()
    reset = BoundedBinaryAlignmentReadout(
        feature_count=len(candidates[0].predictor_features)
    )
    utility_reset = BoundedLabelUtilityReadout(
        feature_count=5 + len(candidates[0].segment_features),
        minimum_observations=_BOOTSTRAP_LABELS,
    )
    return Gate4V2ArmResult(
        seed=seed,
        partition=partition,
        arm=arm,
        train_candidate_count=len(candidates),
        evaluation_count=len(evaluation_plans),
        requested_label_count=len(selected_order),
        labels_needed_for_target=labels_needed,
        final_balanced_accuracy=final_accuracy,
        cumulative_regret=cumulative_regret,
        selected_positive_rate=(
            _mean(tuple(float(labels[index]) for index in selected_order))
            if selected_order
            else 0.0
        ),
        typed_candidate_lineage_coverage=(
            sum(
                bool(candidate.transition_id)
                and bool(candidate.guidance_text)
                for candidate in candidates
            )
            / len(candidates)
        ),
        source_mutation_count=int(
            source_before != _sha256((train_plans, evaluation_plans))
        ),
        isolated_reset_exact=(
            reset.parameters == initial_parameters
            and utility_reset.parameters == initial_utility_parameters
        ),
        utility_observation_count=utility_readout.observation_count,
        learned_utility_selection_count=learned_utility_selection_count,
        cold_start_selection_count=cold_start_selection_count,
    )


def run_gate4_v2_retest(
    *,
    trace_root: str | Path,
    seed_schedule: tuple[int, ...] = GATE78_TRACE_SEEDS,
    partition: str = "trace-development-heldout",
    formal_locked_run: bool = False,
) -> Gate4V2Report:
    if formal_locked_run and partition != "trace-locked-confirmation":
        raise ValueError("Formal Gate 4 v2 run must use locked confirmation")
    if not formal_locked_run and partition == "trace-locked-confirmation":
        raise ValueError("Development Gate 4 v2 run must not consume locked")
    source = verify_gate78_shared_trace_bundle(trace_root)
    if not source["consumer_admission"]:
        raise RuntimeError("Gate 4 v2 source admission failed")
    rows: list[Gate4V2ArmResult] = []
    for seed in seed_schedule:
        if seed not in GATE78_TRACE_SEEDS:
            raise ValueError(f"Unregistered Gate 4 v2 seed {seed}")
        train = load_gate78_partition(
            trace_root,
            seed=seed,
            partition="trace-train",
        )
        evaluation = load_gate78_partition(
            trace_root,
            seed=seed,
            partition=partition,
        )
        for arm in GATE4_V2_ARMS:
            rows.append(
                _run_arm(
                    seed=seed,
                    partition=partition,
                    arm=arm,
                    train_plans=train,
                    evaluation_plans=evaluation,
                )
            )
    results = tuple(rows)
    controls = (
        "turn-level-active",
        "random-feedback",
        "shuffled-segment-boundary",
    )
    gains_by_control = {
        control: tuple(
            next(
                row.labels_needed_for_target
                for row in results
                if row.seed == seed and row.arm == control
            )
            - next(
                row.labels_needed_for_target
                for row in results
                if row.seed == seed
                and row.arm == "segment-aware-active"
            )
            for seed in seed_schedule
        )
        for control in controls
    }
    accuracy_margins = tuple(
        next(
            row.final_balanced_accuracy
            for row in results
            if row.seed == seed and row.arm == "segment-aware-active"
        )
        - max(
            row.final_balanced_accuracy
            for row in results
            if row.seed == seed and row.arm in controls
        )
        for seed in seed_schedule
    )
    lineage = min(
        row.typed_candidate_lineage_coverage for row in results
    )
    mutation = sum(row.source_mutation_count for row in results)
    reset_mismatch = sum(not row.isolated_reset_exact for row in results)
    learned_rows = tuple(
        row
        for row in results
        if row.arm in {
            "segment-aware-active",
            "shuffled-segment-boundary",
        }
    )
    minimum_learned_selections = min(
        row.learned_utility_selection_count for row in learned_rows
    )
    observation_mismatch = sum(
        row.utility_observation_count != row.requested_label_count
        for row in results
    )
    metrics = tuple(
        (
            f"mean_labels_saved_vs_{control}",
            _mean(gains_by_control[control]),
        )
        for control in controls
    ) + (
        ("minimum_final_accuracy_margin", min(accuracy_margins)),
        ("typed_candidate_lineage_coverage", lineage),
        ("source_mutation_count", float(mutation)),
        ("isolated_reset_mismatch_count", float(reset_mismatch)),
        (
            "minimum_learned_utility_selection_count",
            float(minimum_learned_selections),
        ),
        (
            "utility_observation_mismatch_count",
            float(observation_mismatch),
        ),
    )
    mechanism_gates = (
        ("source-consumer-admission", True, 1.0),
        ("typed-candidate-lineage-complete", lineage >= 1.0, lineage),
        ("source-mutation-zero", mutation == 0, float(mutation)),
        ("readout-isolated-reset-exact", reset_mismatch == 0, float(reset_mismatch)),
        (
            "label-utility-observations-complete",
            observation_mismatch == 0,
            float(observation_mismatch),
        ),
        (
            "learned-utility-selector-active",
            minimum_learned_selections > 0,
            float(minimum_learned_selections),
        ),
    )
    causal_gates = tuple(
        (
            f"labels-saved-vs-{control}",
            _mean(gains_by_control[control]) >= 2.0
            and all(gain > 0 for gain in gains_by_control[control]),
            _mean(gains_by_control[control]),
        )
        for control in controls
    ) + (
        (
            "final-accuracy-noninferior",
            min(accuracy_margins) >= -0.01,
            min(accuracy_margins),
        ),
    )
    if not all(passed for _name, passed, _value in mechanism_gates):
        verdict = "invalid"
    elif all(passed for _name, passed, _value in causal_gates):
        verdict = "causal-supported"
    else:
        verdict = "not-supported"
    return Gate4V2Report(
        partition=partition,
        formal_locked_run=formal_locked_run,
        results=results,
        aggregate_metrics=metrics,
        mechanism_gates=mechanism_gates,
        causal_gates=causal_gates,
        verdict=verdict,
    )


def export_gate4_v2_bundle(
    report: Gate4V2Report,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    rows = report.results
    common = tuple(
        {"seed": row.seed, "arm": row.arm, "partition": row.partition}
        for row in rows
    )
    rows_by_file = {
        "predictions.jsonl": tuple(
            {
                **base,
                "final_balanced_accuracy": row.final_balanced_accuracy,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "outcomes.jsonl": tuple(
            {
                **base,
                "labels_needed_for_target": row.labels_needed_for_target,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "prediction_errors.jsonl": tuple(
            {**base, "cumulative_regret": row.cumulative_regret}
            for base, row in zip(common, rows, strict=True)
        ),
        "segments.jsonl": tuple(
            {
                **base,
                "train_candidate_count": row.train_candidate_count,
                "evaluation_count": row.evaluation_count,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "credit.jsonl": tuple(
            {
                **base,
                "requested_label_count": row.requested_label_count,
                "selected_positive_rate": row.selected_positive_rate,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "state_diff.jsonl": tuple(
            {
                **base,
                "source_mutation_count": row.source_mutation_count,
            }
            for base, row in zip(common, rows, strict=True)
        ),
        "action_selection.jsonl": tuple(
            {
                **base,
                "typed_candidate_lineage_coverage": (
                    row.typed_candidate_lineage_coverage
                ),
                "utility_observation_count": (
                    row.utility_observation_count
                ),
                "learned_utility_selection_count": (
                    row.learned_utility_selection_count
                ),
                "cold_start_selection_count": (
                    row.cold_start_selection_count
                ),
            }
            for base, row in zip(common, rows, strict=True)
        ),
    }
    rollback = tuple(
        {
            "seed": row.seed,
            "arm": row.arm,
            "exact": row.isolated_reset_exact,
            "scope": "bounded-readout-isolation",
        }
        for row in rows
    )
    return export_gate_v2_bundle(
        schema_version=GATE4_V2_SCHEMA_VERSION,
        suite_id=GATE4_V2_SUITE_ID,
        source_schema_version=GATE78_TRACE_SCHEMA_VERSION,
        source_fingerprint=_sha256(GATE78_SOURCE_DESCRIPTOR),
        partition=report.partition,
        seed_schedule=tuple(dict.fromkeys(row.seed for row in rows)),
        arm_schedule=GATE4_V2_ARMS,
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


def verify_gate4_v2_bundle(output_dir: str | Path) -> dict[str, object]:
    return verify_gate_v2_bundle(
        output_dir,
        schema_version=GATE4_V2_SCHEMA_VERSION,
        suite_id=GATE4_V2_SUITE_ID,
        arm_schedule=GATE4_V2_ARMS,
    )
