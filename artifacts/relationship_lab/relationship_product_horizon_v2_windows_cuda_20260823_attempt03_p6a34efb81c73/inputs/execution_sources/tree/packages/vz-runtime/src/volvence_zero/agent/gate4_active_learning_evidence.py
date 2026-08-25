"""Gate 4 segment-aware active-learning evidence campaign.

The runtime harness consumes immutable public trace records.  Feedback
selection and the bounded alignment readout remain in the apprenticeship
owner package; this module validates source admission, drives matched arms,
and writes the auditable evidence packet.
"""

from __future__ import annotations

import asyncio
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

from volvence_zero.apprenticeship import (
    ApprenticeshipAlignmentModule,
    BoundedBinaryAlignmentReadout,
    BoundedLabelUtilityReadout,
    Gate4FeedbackCandidate,
    build_intent_constraint,
    label_utility_features,
    random_feedback_order,
    select_active_candidate,
)
from volvence_zero.agent.shared_settled_trace import (
    SHARED_SETTLED_TRACE_COUNT_PER_SEED,
    SHARED_SETTLED_TRACE_SEEDS,
    build_shared_trace_plans,
    load_shared_trace_records,
    validate_shared_trace_prefix,
)
from volvence_zero.memory import MemoryStore
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    OpenLoopModule,
    SemanticStateStore,
)
from volvence_zero.substrate import PlaceholderSubstrateAdapter


GATE4_ACTIVE_LEARNING_SCHEMA_VERSION = "gate4-active-learning.v2"
GATE4_REQUIRED_FILES = (
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
GATE4_ARM_NAMES = (
    "segment-aware-active",
    "turn-level-active",
    "random-feedback",
    "no-feedback",
    "shuffled-segment-boundary",
)
GATE4_LABEL_BUDGET = 60
GATE4_BOOTSTRAP_LABELS = 8
GATE4_ALIGNMENT_TARGET = 0.80
GATE4_PER_SEED_LABEL_GAIN = 5
GATE4_AGGREGATE_LABEL_GAIN = 10
GATE4_FINAL_ACCURACY_TOLERANCE = 0.01
GATE4_SEGMENT_KILL_GAIN = 5
_PARTITION_SEQUENCE = (
    ("trace-train", 300),
    ("trace-heldout-context", 150),
    ("trace-locked-confirmation", 60),
)


@dataclass(frozen=True)
class Gate4ArmMetrics:
    seed: int
    arm: str
    label_budget: int
    requested_label_count: int
    labels_needed_for_target: int
    heldout_balanced_accuracy: float
    locked_balanced_accuracy: float | None
    cumulative_regret: int
    ineffective_request_rate: float
    missed_high_risk_rate: float
    typed_request_coverage: float
    open_loop_actuation_coverage: float
    proposal_count: int
    boundary_digest_unchanged: bool
    source_closure_coverage: float
    lineage_complete: bool
    frozen_substrate_mutation_count: int
    utility_observation_count: int = 0
    learned_utility_selection_count: int = 0
    cold_start_selection_count: int = 0


@dataclass(frozen=True)
class Gate4Comparison:
    control_arm: str
    aggregate_label_gain: int
    per_seed_label_gains: tuple[int, ...]
    final_accuracy_gain: float
    primary_non_worse: bool


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def _binary_log_loss(probability: float, label: bool) -> float:
    bounded = max(1e-9, min(1.0 - 1e-9, probability))
    return (
        -math.log(bounded)
        if label
        else -math.log(1.0 - bounded)
    )


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


def validate_gate4_source_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> None:
    if len(records) != SHARED_SETTLED_TRACE_COUNT_PER_SEED:
        raise ValueError(
            f"Gate 4 seed {seed} requires exactly "
            f"{SHARED_SETTLED_TRACE_COUNT_PER_SEED} records"
        )
    validate_shared_trace_prefix(
        records=records,
        expected_plans=build_shared_trace_plans(seed),
    )
    if _partition_counts(records) != _PARTITION_SEQUENCE:
        raise ValueError(
            f"Gate 4 seed {seed} partition count contract drifted"
        )
    expected_order = tuple(
        partition
        for partition, count in _PARTITION_SEQUENCE
        for _ in range(count)
    )
    if tuple(record["partition"] for record in records) != expected_order:
        raise ValueError(
            f"Gate 4 seed {seed} partition ordering drifted"
        )
    for record in records:
        transition_id = str(record["transition_id"])
        if not record.get("settled"):
            raise ValueError(
                f"Gate 4 source {transition_id} is not settled"
            )
        closures = record["temporal_snapshot"].get("closed_segments", ())
        if not closures:
            raise ValueError(
                f"Gate 4 source {transition_id} lacks owner segment closure"
            )
        for closure in closures:
            if (
                not closure.get("segment_id")
                or not closure.get("abstract_action_id")
                or not closure.get("z_t_digest")
                or int(closure["close_turn_index"])
                <= int(closure["open_turn_index"])
            ):
                raise ValueError(
                    f"Gate 4 source {transition_id} has invalid closure lineage"
                )
        lineage = record["lineage"]
        if (
            not lineage.get("prediction_ref")
            or not lineage.get("environment_outcome_id")
            or not record.get("record_sha256")
        ):
            raise ValueError(
                f"Gate 4 source {transition_id} lacks settled lineage"
            )
        substrate = record["substrate"]
        if (
            substrate.get("runtime_origin") != "hf-local"
            or substrate.get("fallback_active")
            or not substrate.get("is_frozen")
            or substrate.get("mutation_applied")
        ):
            raise ValueError(
                f"Gate 4 source {transition_id} violates frozen substrate"
            )


def _predictor_features(record: Mapping[str, Any]) -> tuple[float, ...]:
    prediction = record["prediction"]
    controller = record["temporal_snapshot"]["controller_state"]
    code = tuple(float(value) for value in controller["code"])
    if not code:
        raise ValueError(
            f"{record['transition_id']} lacks temporal controller code"
        )
    return (
        float(prediction["predicted_task_progress"]),
        float(prediction["predicted_relationship_delta"]),
        float(prediction["predicted_regime_stability"]),
        float(prediction["predicted_action_payoff"]),
        float(prediction["confidence"]),
        *code,
        float(controller["switch_gate"]),
        min(float(controller["steps_since_switch"]) / 8.0, 1.0),
    )


def _segment_features(record: Mapping[str, Any]) -> tuple[float, ...]:
    closure = record["temporal_snapshot"]["closed_segments"][-1]
    code = tuple(float(value) for value in closure["z_t_digest"])
    return (
        *code,
        float(closure["beta_open_digest"]),
        float(closure["beta_close_digest"]),
        float(
            int(closure["close_turn_index"])
            - int(closure["open_turn_index"])
        ),
    )


def _candidate(record: Mapping[str, Any]) -> Gate4FeedbackCandidate:
    return Gate4FeedbackCandidate(
        transition_id=str(record["transition_id"]),
        guidance_text=str(record["input"]["settlement_turn"]),
        turn_index=int(record["global_index"]),
        predictor_features=_predictor_features(record),
        pe_magnitude=float(record["prediction_error"]["magnitude"]),
        segment_features=_segment_features(record),
    )


def _high_risk_label(record: Mapping[str, Any]) -> bool:
    outcome = record["actual_outcome"]
    return (
        float(outcome["task_progress"]) < 0.25
        or float(outcome["action_payoff"]) < 0.20
    )


def _balanced_accuracy(
    *,
    probabilities: Sequence[float],
    labels: Sequence[bool],
) -> float:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("Gate 4 balanced accuracy requires aligned rows")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Gate 4 balanced accuracy requires both label classes"
        )
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
    return (
        true_positive / positive_count
        + true_negative / negative_count
    ) / 2.0


def _shuffled_candidates(
    candidates: Sequence[Gate4FeedbackCandidate],
    *,
    seed: int,
) -> tuple[Gate4FeedbackCandidate, ...]:
    keyed = sorted(
        range(len(candidates)),
        key=lambda index: hashlib.sha256(
            f"gate4-shuffle:{seed}:{index}".encode("utf-8")
        ).hexdigest(),
    )
    segment_values = tuple(
        candidates[index].segment_features for index in keyed
    )
    return tuple(
        Gate4FeedbackCandidate(
            transition_id=candidate.transition_id,
            guidance_text=candidate.guidance_text,
            turn_index=candidate.turn_index,
            predictor_features=candidate.predictor_features,
            pe_magnitude=candidate.pe_magnitude,
            segment_features=segment_values[index],
        )
        for index, candidate in enumerate(candidates)
    )


def _empty_boundary_snapshot() -> BoundaryConsentSnapshot:
    return BoundaryConsentSnapshot(
        granted_consents=(),
        missing_consents=(),
        denied_boundaries=(),
        memory_consent="unchanged",
        external_action_consent="unchanged",
        compliance_score=1.0,
        control_signal=0.0,
        description="Gate 4 immutable boundary-control fixture.",
        autonomy_risk=0.0,
        consent_clarity=1.0,
        overreach_risk=0.0,
    )


def _digest_dataclass(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            asdict(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


async def _typed_request_audit(
    *,
    candidates: Sequence[Gate4FeedbackCandidate],
    selected_order: Sequence[int],
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feedback_policy = "random" if arm == "random-feedback" else "owner"
    alignment_owner = ApprenticeshipAlignmentModule(
        wiring_level=WiringLevel.ACTIVE,
        apprenticeship=True,
        feedback_policy=feedback_policy,
        revision_enabled=False,
    )
    boundary = _empty_boundary_snapshot()
    boundary_before = _digest_dataclass(boundary)
    substrate = await PlaceholderSubstrateAdapter(
        model_id="gate4-request-audit"
    ).capture()
    memory = MemoryStore().snapshot(retrieved_entries=())
    substrate_snapshot = Snapshot(
        slot_name="substrate",
        version=1,
        value=substrate,
        owner="Gate4Evidence",
        timestamp_ms=1,
    )
    memory_snapshot = Snapshot(
        slot_name="memory",
        version=1,
        value=memory,
        owner="Gate4Evidence",
        timestamp_ms=1,
    )
    store = SemanticStateStore()
    rows: list[dict[str, Any]] = []
    proposal_count = 0
    typed_count = 0
    actuated_count = 0
    for request_index, candidate_index in enumerate(selected_order):
        candidate = candidates[candidate_index]
        constraint = build_intent_constraint(
            constraint_id=(
                f"gate4:{arm}:{candidate.transition_id}:feedback"
            ),
            statement=(
                "Request expert confirmation for this typed settled "
                "transition."
            ),
            target_key=candidate.transition_id,
            confidence=0.60,
            source_turn=candidate.turn_index,
        )
        alignment = await alignment_owner.process_standalone(
            guidance_text=candidate.guidance_text,
            turn_index=candidate.turn_index,
            apprenticeship=True,
            constraints=(constraint,),
            boundary_consent=boundary,
        )
        proposals = alignment_owner.drain_revision_proposals()
        proposal_count += len(proposals)
        typed = bool(
            alignment.value.should_request_feedback
            and alignment.value.feedback_request_reason
            and alignment.value.feedback_request_urgency > 0.0
        )
        typed_count += int(typed)
        open_loop_owner = OpenLoopModule(
            store=store,
            turn_index=candidate.turn_index,
            wiring_level=WiringLevel.ACTIVE,
        )
        open_loop = await open_loop_owner.process(
            {
                "substrate": substrate_snapshot,
                "memory": memory_snapshot,
                "apprenticeship_alignment": alignment,
            }
        )
        actuated = bool(
            open_loop.value.apprenticeship_verification_requests
        )
        actuated_count += int(actuated)
        rows.append(
            {
                "schema_version": GATE4_ACTIVE_LEARNING_SCHEMA_VERSION,
                "arm": arm,
                "request_index": request_index,
                "transition_id": candidate.transition_id,
                "should_request_feedback": (
                    alignment.value.should_request_feedback
                ),
                "feedback_request_reason": (
                    alignment.value.feedback_request_reason
                ),
                "feedback_request_urgency": (
                    alignment.value.feedback_request_urgency
                ),
                "open_loop_request_refs": list(
                    open_loop.value.apprenticeship_verification_requests
                ),
                "revision_proposal_count": len(proposals),
            }
        )
    request_count = len(selected_order)
    return rows, {
        "typed_request_coverage": (
            typed_count / request_count if request_count else 1.0
        ),
        "open_loop_actuation_coverage": (
            actuated_count / request_count if request_count else 1.0
        ),
        "proposal_count": proposal_count,
        "boundary_digest_unchanged": (
            boundary_before == _digest_dataclass(boundary)
        ),
    }


def run_gate4_arm(
    *,
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    consume_locked: bool,
) -> tuple[
    Gate4ArmMetrics,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run one preregistered arm for one complete source seed."""

    if arm not in GATE4_ARM_NAMES:
        raise ValueError(f"Unknown Gate 4 arm {arm!r}")
    validate_gate4_source_records(records, seed=seed)
    train_records = tuple(
        record
        for record in records
        if record["partition"] == "trace-train"
    )
    heldout_records = tuple(
        record
        for record in records
        if record["partition"] == "trace-heldout-context"
    )
    locked_records = tuple(
        record
        for record in records
        if record["partition"] == "trace-locked-confirmation"
    )
    candidates = tuple(_candidate(record) for record in train_records)
    selection_candidates = (
        _shuffled_candidates(candidates, seed=seed)
        if arm == "shuffled-segment-boundary"
        else candidates
    )
    train_labels = tuple(_high_risk_label(record) for record in train_records)
    heldout_features = tuple(
        _predictor_features(record) for record in heldout_records
    )
    heldout_labels = tuple(
        _high_risk_label(record) for record in heldout_records
    )
    locked_features = (
        tuple(_predictor_features(record) for record in locked_records)
        if consume_locked
        else ()
    )
    locked_labels = (
        tuple(_high_risk_label(record) for record in locked_records)
        if consume_locked
        else ()
    )
    feature_count = len(candidates[0].predictor_features)
    readout = BoundedBinaryAlignmentReadout(
        feature_count=feature_count
    )
    utility_readout = BoundedLabelUtilityReadout(
        feature_count=5 + len(candidates[0].segment_features),
        minimum_observations=GATE4_BOOTSTRAP_LABELS,
    )
    random_order = random_feedback_order(candidates)
    budget = 0 if arm == "no-feedback" else GATE4_LABEL_BUDGET
    selected_order: list[int] = []
    selected: set[int] = set()
    curve_rows: list[dict[str, Any]] = []
    labels_needed = GATE4_LABEL_BUDGET + 1
    cumulative_regret = 0
    ineffective_requests = 0
    learned_utility_selection_count = 0
    cold_start_selection_count = 0

    def refit() -> None:
        readout.fit(
            tuple(
                (
                    candidates[index].predictor_features,
                    train_labels[index],
                )
                for index in selected_order
            )
        )

    def heldout_accuracy() -> float:
        return _balanced_accuracy(
            probabilities=tuple(
                readout.predict_probability(features)
                for features in heldout_features
            ),
            labels=heldout_labels,
        )

    if arm != "no-feedback":
        while len(selected_order) < budget:
            train_probabilities = tuple(
                readout.predict_probability(
                    candidate.predictor_features
                )
                for candidate in candidates
            )
            cumulative_regret += sum(
                index not in selected
                and train_labels[index]
                and probability < 0.5
                for index, probability in enumerate(train_probabilities)
            )
            if len(selected_order) < GATE4_BOOTSTRAP_LABELS:
                candidate_index = next(
                    index for index in random_order if index not in selected
                )
            elif arm == "random-feedback":
                candidate_index = next(
                    index for index in random_order if index not in selected
                )
            else:
                utility_ready = utility_readout.ready
                candidate_index = select_active_candidate(
                    candidates=selection_candidates,
                    probabilities=train_probabilities,
                    selected_indices=frozenset(selected),
                    policy=arm,
                    utility_readout=utility_readout,
                )
                learned_utility_selection_count += int(utility_ready)
                cold_start_selection_count += int(not utility_ready)
            prequery_probability = train_probabilities[candidate_index]
            utility_features = label_utility_features(
                candidate=selection_candidates[candidate_index],
                probability=prequery_probability,
                selected_segment_features=tuple(
                    selection_candidates[index].segment_features
                    for index in sorted(selected)
                ),
            )
            utility_indices = (*selected_order, candidate_index)
            loss_before = _mean(
                tuple(
                    _binary_log_loss(
                        train_probabilities[index],
                        train_labels[index],
                    )
                    for index in utility_indices
                )
            )
            prequery_uncertainty = (
                1.0 - abs(2.0 * prequery_probability - 1.0)
            )
            ineffective_requests += int(
                prequery_uncertainty < 0.25
                and not train_labels[candidate_index]
            )
            selected.add(candidate_index)
            selected_order.append(candidate_index)
            refit()
            loss_after = _mean(
                tuple(
                    _binary_log_loss(
                        readout.predict_probability(
                            candidates[index].predictor_features
                        ),
                        train_labels[index],
                    )
                    for index in utility_indices
                )
            )
            observed_utility = utility_readout.observe_loss_delta(
                features=utility_features,
                loss_before=loss_before,
                loss_after=loss_after,
            )
            accuracy = heldout_accuracy()
            if (
                labels_needed == GATE4_LABEL_BUDGET + 1
                and accuracy >= GATE4_ALIGNMENT_TARGET
            ):
                labels_needed = len(selected_order)
            curve_rows.append(
                {
                    "schema_version": (
                        GATE4_ACTIVE_LEARNING_SCHEMA_VERSION
                    ),
                    "seed": seed,
                    "arm": arm,
                    "label_count": len(selected_order),
                    "selected_transition_id": (
                        candidates[candidate_index].transition_id
                    ),
                    "prequery_probability": prequery_probability,
                    "prequery_uncertainty": prequery_uncertainty,
                    "selected_label_high_risk": (
                        train_labels[candidate_index]
                    ),
                    "heldout_balanced_accuracy": accuracy,
                    "readout_parameters": list(readout.parameters),
                    "observed_label_utility": observed_utility,
                    "utility_readout_ready": utility_readout.ready,
                }
            )
    final_heldout = heldout_accuracy()
    final_train_probabilities = tuple(
        readout.predict_probability(candidate.predictor_features)
        for candidate in candidates
    )
    high_risk_count = sum(train_labels)
    missed_high_risk = sum(
        index not in selected
        and train_labels[index]
        and final_train_probabilities[index] < 0.5
        for index in range(len(candidates))
    )
    locked_accuracy: float | None = (
        _balanced_accuracy(
            probabilities=tuple(
                readout.predict_probability(features)
                for features in locked_features
            ),
            labels=locked_labels,
        )
        if consume_locked
        else None
    )
    request_rows, audit = asyncio.run(
        _typed_request_audit(
            candidates=candidates,
            selected_order=selected_order,
            arm=arm,
        )
    )
    closure_count = sum(
        bool(record["temporal_snapshot"]["closed_segments"])
        for record in records
    )
    lineage_complete = all(
        record["lineage"].get("prediction_ref")
        and record["lineage"].get("environment_outcome_id")
        and record.get("record_sha256")
        for record in records
    )
    metric = Gate4ArmMetrics(
        seed=seed,
        arm=arm,
        label_budget=budget,
        requested_label_count=len(selected_order),
        labels_needed_for_target=labels_needed,
        heldout_balanced_accuracy=final_heldout,
        locked_balanced_accuracy=locked_accuracy,
        cumulative_regret=cumulative_regret,
        ineffective_request_rate=(
            ineffective_requests / len(selected_order)
            if selected_order
            else 0.0
        ),
        missed_high_risk_rate=(
            missed_high_risk / high_risk_count
            if high_risk_count
            else 0.0
        ),
        typed_request_coverage=float(
            audit["typed_request_coverage"]
        ),
        open_loop_actuation_coverage=float(
            audit["open_loop_actuation_coverage"]
        ),
        proposal_count=int(audit["proposal_count"]),
        boundary_digest_unchanged=bool(
            audit["boundary_digest_unchanged"]
        ),
        source_closure_coverage=closure_count / len(records),
        lineage_complete=bool(lineage_complete),
        frozen_substrate_mutation_count=sum(
            bool(record["substrate"]["mutation_applied"])
            for record in records
        ),
        utility_observation_count=utility_readout.observation_count,
        learned_utility_selection_count=learned_utility_selection_count,
        cold_start_selection_count=cold_start_selection_count,
    )
    for row in request_rows:
        row["seed"] = seed
    return metric, curve_rows, request_rows


def compare_gate4_arms(
    metrics: Sequence[Gate4ArmMetrics],
) -> tuple[
    tuple[Gate4Comparison, ...],
    dict[str, bool],
]:
    by_key = {(metric.seed, metric.arm): metric for metric in metrics}
    all_present = all(
        (seed, arm) in by_key
        for seed in SHARED_SETTLED_TRACE_SEEDS
        for arm in GATE4_ARM_NAMES
    )
    if not all_present:
        return (), {"all_arms_all_seeds_present": False}
    comparisons: list[Gate4Comparison] = []
    for control in ("turn-level-active", "random-feedback"):
        per_seed = tuple(
            by_key[
                (seed, control)
            ].labels_needed_for_target
            - by_key[
                (seed, "segment-aware-active")
            ].labels_needed_for_target
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        final_gain = _mean(
            tuple(
                by_key[
                    (seed, "segment-aware-active")
                ].heldout_balanced_accuracy
                - by_key[
                    (seed, control)
                ].heldout_balanced_accuracy
                for seed in SHARED_SETTLED_TRACE_SEEDS
            )
        )
        comparisons.append(
            Gate4Comparison(
                control_arm=control,
                aggregate_label_gain=sum(per_seed),
                per_seed_label_gains=per_seed,
                final_accuracy_gain=final_gain,
                primary_non_worse=(
                    all(
                        gain >= GATE4_PER_SEED_LABEL_GAIN
                        for gain in per_seed
                    )
                    and sum(per_seed) >= GATE4_AGGREGATE_LABEL_GAIN
                    and final_gain >= -GATE4_FINAL_ACCURACY_TOLERANCE
                ),
            )
        )
    segment_vs_shuffled = sum(
        by_key[
            (seed, "shuffled-segment-boundary")
        ].labels_needed_for_target
        - by_key[
            (seed, "segment-aware-active")
        ].labels_needed_for_target
        for seed in SHARED_SETTLED_TRACE_SEEDS
    )
    shuffled_beats_controls = all(
        all(
            by_key[(seed, control)].labels_needed_for_target
            - by_key[
                (seed, "shuffled-segment-boundary")
            ].labels_needed_for_target
            >= GATE4_PER_SEED_LABEL_GAIN
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        and sum(
            by_key[(seed, control)].labels_needed_for_target
            - by_key[
                (seed, "shuffled-segment-boundary")
            ].labels_needed_for_target
            for seed in SHARED_SETTLED_TRACE_SEEDS
        )
        >= GATE4_AGGREGATE_LABEL_GAIN
        for control in ("turn-level-active", "random-feedback")
    )
    turn_vs_random = sum(
        by_key[
            (seed, "random-feedback")
        ].labels_needed_for_target
        - by_key[
            (seed, "turn-level-active")
        ].labels_needed_for_target
        for seed in SHARED_SETTLED_TRACE_SEEDS
    )
    gates = {
        "all_arms_all_seeds_present": True,
        "matched_label_budgets": all(
            metric.requested_label_count
            == (
                0
                if metric.arm == "no-feedback"
                else GATE4_LABEL_BUDGET
            )
            for metric in metrics
        ),
        "source_closure_coverage_one": all(
            metric.source_closure_coverage == 1.0
            for metric in metrics
        ),
        "lineage_complete": all(
            metric.lineage_complete for metric in metrics
        ),
        "frozen_substrate_mutation_zero": all(
            metric.frozen_substrate_mutation_count == 0
            for metric in metrics
        ),
        "typed_request_coverage_one": all(
            metric.typed_request_coverage == 1.0
            for metric in metrics
        ),
        "open_loop_actuation_coverage_one": all(
            metric.open_loop_actuation_coverage == 1.0
            for metric in metrics
        ),
        "proposal_only_no_write": all(
            metric.proposal_count == 0
            and metric.boundary_digest_unchanged
            for metric in metrics
        ),
        "label_utility_observations_complete": all(
            metric.utility_observation_count
            == metric.requested_label_count
            for metric in metrics
        ),
        "learned_utility_selector_active": all(
            by_key[(seed, arm)].learned_utility_selection_count > 0
            for seed in SHARED_SETTLED_TRACE_SEEDS
            for arm in (
                "segment-aware-active",
                "shuffled-segment-boundary",
            )
        ),
        "segment_primary_vs_turn_and_random": all(
            comparison.primary_non_worse
            for comparison in comparisons
        ),
        "segment_boundary_kill_control_passed": (
            segment_vs_shuffled >= GATE4_SEGMENT_KILL_GAIN
            and not shuffled_beats_controls
        ),
        "pe_driven_diagnostic_supported": (
            turn_vs_random >= GATE4_AGGREGATE_LABEL_GAIN
        ),
    }
    return tuple(comparisons), gates


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _source_rows(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    consume_locked: bool,
) -> dict[str, list[dict[str, Any]]]:
    rows = {
        "predictions.jsonl": [],
        "outcomes.jsonl": [],
        "prediction_errors.jsonl": [],
        "segments.jsonl": [],
        "credit.jsonl": [],
    }
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        for record in records_by_seed[seed]:
            base = {
                "schema_version": (
                    GATE4_ACTIVE_LEARNING_SCHEMA_VERSION
                ),
                "seed": seed,
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "prediction_ref": record["lineage"]["prediction_ref"],
                "record_sha256": record["record_sha256"],
            }
            rows["predictions.jsonl"].append(
                {**base, "prediction": record["prediction"]}
            )
            rows["outcomes.jsonl"].append(
                {
                    **base,
                    "actual_outcome": record["actual_outcome"],
                    "high_risk_label": (
                        _high_risk_label(record)
                        if (
                            consume_locked
                            or record["partition"]
                            != "trace-locked-confirmation"
                        )
                        else None
                    ),
                }
            )
            rows["prediction_errors.jsonl"].append(
                {
                    **base,
                    "prediction_error": record["prediction_error"],
                }
            )
            rows["segments.jsonl"].append(
                {
                    **base,
                    "temporal_snapshot": record["temporal_snapshot"],
                }
            )
            rows["credit.jsonl"].append(
                {
                    **base,
                    "credit_snapshot": record["credit_snapshot"],
                }
            )
    return rows


def _segment_diagnostics(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    all_lengths: list[int] = []
    all_families: set[str] = set()
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        records = records_by_seed[seed]
        lengths: list[int] = []
        families: set[str] = set()
        features: set[tuple[float, ...]] = set()
        for record in records:
            closure = record["temporal_snapshot"]["closed_segments"][-1]
            length = (
                int(closure["close_turn_index"])
                - int(closure["open_turn_index"])
            )
            family = str(closure["abstract_action_id"])
            lengths.append(length)
            families.add(family)
            features.add(_segment_features(record))
        all_lengths.extend(lengths)
        all_families.update(families)
        by_seed[str(seed)] = {
            "closure_count": len(records),
            "unique_segment_feature_count": len(features),
            "unique_action_family_count": len(families),
            "segment_length_min": min(lengths),
            "segment_length_max": max(lengths),
        }
    return {
        "by_seed": by_seed,
        "aggregate_unique_action_family_count": len(all_families),
        "aggregate_segment_length_min": min(all_lengths),
        "aggregate_segment_length_max": max(all_lengths),
        "claim_limit": (
            "Fixed-length or single-family source evidence cannot establish "
            "generalization to variable-length or multi-family segments."
        ),
    }


def export_gate4_active_learning_bundle(
    *,
    trace_root: str | Path,
    output_dir: str | Path,
    consume_locked: bool = True,
) -> tuple[Path, ...]:
    """Run the preregistered five-arm campaign and write its packet."""

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
        raise ValueError("Gate 4 source trace is not admitted")
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        records = load_shared_trace_records(
            source / f"seed_{seed}" / "transitions.jsonl"
        )
        validate_gate4_source_records(records, seed=seed)
        records_by_seed[seed] = records
    metrics: list[Gate4ArmMetrics] = []
    curve_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        for arm in GATE4_ARM_NAMES:
            metric, arm_curve, arm_requests = run_gate4_arm(
                records=records_by_seed[seed],
                seed=seed,
                arm=arm,
                consume_locked=consume_locked,
            )
            metrics.append(metric)
            curve_rows.extend(arm_curve)
            request_rows.extend(arm_requests)
    comparisons, gates = compare_gate4_arms(metrics)
    invalid_gate_names = (
        "all_arms_all_seeds_present",
        "matched_label_budgets",
        "source_closure_coverage_one",
        "lineage_complete",
        "frozen_substrate_mutation_zero",
        "typed_request_coverage_one",
        "open_loop_actuation_coverage_one",
        "proposal_only_no_write",
    )
    invalid_gates = tuple(
        name for name in invalid_gate_names if not gates.get(name, False)
    )
    causal_passed = (
        not invalid_gates
        and gates["segment_primary_vs_turn_and_random"]
        and gates["segment_boundary_kill_control_passed"]
    )
    status = (
        "invalid"
        if invalid_gates
        else "causal-supported"
        if causal_passed
        else "not-supported"
    )
    metric_map = {
        (metric.seed, metric.arm): metric for metric in metrics
    }
    segment_vs_shuffled_gain = sum(
        metric_map[
            (seed, "shuffled-segment-boundary")
        ].labels_needed_for_target
        - metric_map[
            (seed, "segment-aware-active")
        ].labels_needed_for_target
        for seed in SHARED_SETTLED_TRACE_SEEDS
    )
    manifest = {
        "schema_version": GATE4_ACTIVE_LEARNING_SCHEMA_VERSION,
        "suite_id": "gate4-segment-aware-active-learning",
        "owner": (
            "vz-cognition.ApprenticeshipAlignmentModule/"
            "apprenticeship.active_learning"
        ),
        "trace_schema_version": "gate456-shared-settled-trace.v1",
        "trace_root": str(source),
        "source_temporal_fix_commit": "f5f551b3",
        "substrate_fingerprint": aggregate_manifest[
            "runtime_fingerprint"
        ],
        "source_trace_sha256_by_seed": {
            str(summary["seed"]): summary["trace_sha256"]
            for summary in aggregate_manifest["seed_summaries"]
        },
        "model_and_adapter_ids": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_ids": [],
            "substrate_mode": (
                "frozen strict-local corrected segment trace"
            ),
        },
        "wiring_levels": {
            "source_runtime": "ACTIVE frozen trace",
            "apprenticeship_alignment": "ACTIVE evidence replay",
            "revision_enabled": False,
            "substrate_mutation": "DISABLED",
        },
        "seed_schedule": list(SHARED_SETTLED_TRACE_SEEDS),
        "arm_schedule": list(GATE4_ARM_NAMES),
        "partition_schedule": [
            {"partition": partition, "count_per_seed": count}
            for partition, count in _PARTITION_SEQUENCE
        ],
        "primary_partition": "trace-heldout-context",
        "locked_partition": "trace-locked-confirmation",
        "locked_partition_consumed": consume_locked,
        "label_budget_per_active_arm": GATE4_LABEL_BUDGET,
        "shared_bootstrap_labels": GATE4_BOOTSTRAP_LABELS,
        "alignment_target": GATE4_ALIGNMENT_TARGET,
        "per_seed_minimum_label_gain": GATE4_PER_SEED_LABEL_GAIN,
        "aggregate_minimum_label_gain": GATE4_AGGREGATE_LABEL_GAIN,
        "final_accuracy_tolerance": GATE4_FINAL_ACCURACY_TOLERANCE,
        "segment_kill_minimum_gain": GATE4_SEGMENT_KILL_GAIN,
        "high_risk_label": (
            "actual.task_progress < 0.25 OR "
            "actual.action_payoff < 0.20"
        ),
        "predictor_feature_contract": (
            "prediction four axes + confidence + public current "
            "controller code/switch_gate/steps_since_switch; excludes "
            "PE, actual outcome, text, phase, user/context ids"
        ),
        "required_files": list(GATE4_REQUIRED_FILES),
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": bool(
                _git_output("status", "--porcelain")
                not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE4_ACTIVE_LEARNING_SCHEMA_VERSION,
        "metrics": [asdict(metric) for metric in metrics],
        "comparisons": [
            asdict(comparison) for comparison in comparisons
        ],
        "segment_vs_shuffled_aggregate_label_gain": (
            segment_vs_shuffled_gain
        ),
        "gates": gates,
        "source_segment_diagnostics": _segment_diagnostics(
            records_by_seed
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    claim = (
        "feedback sample efficiency depends on owner-published temporal segments"
        if status == "causal-supported"
        else "PE-driven feedback requests are supported without temporal-segment causality"
        if (
            status == "not-supported"
            and gates["pe_driven_diagnostic_supported"]
        )
        else "typed active feedback is runnable but causal efficiency is unsupported"
        if status == "not-supported"
        else "Gate 4 evidence packet is invalid"
    )
    verdict = {
        "schema_version": GATE4_ACTIVE_LEARNING_SCHEMA_VERSION,
        "gate_scope": "Gate 4 segment-aware active learning",
        "status": status,
        "mechanism_passed": not invalid_gates,
        "causal_passed": causal_passed,
        "claim": claim,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "invalid_gates": list(invalid_gates),
        "locked_partition_consumed_once": consume_locked,
        "same_locked_partition_rerun_allowed": False,
    }
    rollback = {
        "schema_version": GATE4_ACTIVE_LEARNING_SCHEMA_VERSION,
        "passed": gates["proposal_only_no_write"],
        "feedback_policy": "disabled",
        "rollback_arm": "no-feedback",
        "revision_enabled": False,
        "boundary_consent_mutated": False,
        "substrate_mutated": False,
    }
    for name, rows in _source_rows(
        records_by_seed,
        consume_locked=consume_locked,
    ).items():
        _write_jsonl(target / name, rows)
    _write_jsonl(target / "state_diff.jsonl", curve_rows)
    _write_jsonl(target / "action_selection.jsonl", request_rows)
    _write_json(target / "manifest.yaml", manifest)
    _write_json(target / "ablation_results.json", ablation)
    _write_json(target / "promotion_verdict.json", verdict)
    _write_json(target / "rollback_evidence.json", rollback)
    report_lines = [
        "# Gate 4 segment-aware active-learning evidence",
        "",
        f"- status: `{status}`",
        f"- mechanism passed: `{not invalid_gates}`",
        f"- causal passed: `{causal_passed}`",
        f"- claim: {claim}",
        (
            "- segment vs shuffled aggregate labels-needed gain: "
            f"`{segment_vs_shuffled_gain}`"
        ),
        (
            "- source segment length range: "
            f"`{ablation['source_segment_diagnostics']['aggregate_segment_length_min']}"
            "–"
            f"{ablation['source_segment_diagnostics']['aggregate_segment_length_max']}`; "
            "unique action families: "
            f"`{ablation['source_segment_diagnostics']['aggregate_unique_action_family_count']}`"
        ),
        "",
        "## Segment-aware vs controls",
        "",
    ]
    report_lines.extend(
        (
            f"- `{comparison.control_arm}`: aggregate label gain "
            f"`{comparison.aggregate_label_gain}`, per-seed "
            f"`{list(comparison.per_seed_label_gains)}`, final accuracy "
            f"gain `{comparison.final_accuracy_gain:.6f}`, primary "
            f"`{comparison.primary_non_worse}`"
        )
        for comparison in comparisons
    )
    report_lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            (
                "- Outcome labels are typed task/action measurements. PE "
                "participates only in acquisition and is excluded from "
                "the learned predictor and label definition."
            ),
            (
                "- Failure of the shuffled-boundary kill control contracts "
                "the claim to PE-driven requests at most; locked evidence "
                "is not retuned or rerun."
            ),
            "",
        )
    )
    (target / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    written = tuple(target / name for name in GATE4_REQUIRED_FILES)
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 4 bundle missing required files {missing!r}"
        )
    return written


def verify_gate4_active_learning_bundle(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Recompute packet gates from persisted metrics and fail on drift."""

    root = Path(output_dir)
    missing = tuple(
        name
        for name in GATE4_REQUIRED_FILES
        if not (root / name).is_file()
    )
    if missing:
        raise ValueError(
            f"Gate 4 bundle missing required files {missing!r}"
        )
    manifest = json.loads(
        (root / "manifest.yaml").read_text(encoding="utf-8")
    )
    ablation = json.loads(
        (root / "ablation_results.json").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (root / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    rollback = json.loads(
        (root / "rollback_evidence.json").read_text(encoding="utf-8")
    )
    for name, payload in (
        ("manifest", manifest),
        ("ablation", ablation),
        ("verdict", verdict),
        ("rollback", rollback),
    ):
        if payload.get("schema_version") != (
            GATE4_ACTIVE_LEARNING_SCHEMA_VERSION
        ):
            raise ValueError(f"Gate 4 {name} schema version drifted")
    metrics = tuple(
        Gate4ArmMetrics(**row) for row in ablation["metrics"]
    )
    comparisons, gates = compare_gate4_arms(metrics)
    if gates != ablation["gates"]:
        raise ValueError("Gate 4 persisted gates do not recompute")
    recomputed_comparisons = json.loads(
        json.dumps([asdict(item) for item in comparisons])
    )
    if recomputed_comparisons != ablation["comparisons"]:
        raise ValueError(
            "Gate 4 persisted comparisons do not recompute"
        )
    invalid_names = (
        "all_arms_all_seeds_present",
        "matched_label_budgets",
        "source_closure_coverage_one",
        "lineage_complete",
        "frozen_substrate_mutation_zero",
        "typed_request_coverage_one",
        "open_loop_actuation_coverage_one",
        "proposal_only_no_write",
    )
    invalid = tuple(
        name for name in invalid_names if not gates[name]
    )
    causal = (
        not invalid
        and gates["segment_primary_vs_turn_and_random"]
        and gates["segment_boundary_kill_control_passed"]
    )
    expected_status = (
        "invalid"
        if invalid
        else "causal-supported"
        if causal
        else "not-supported"
    )
    if verdict["status"] != expected_status:
        raise ValueError("Gate 4 verdict does not recompute")
    expected_source_rows = (
        SHARED_SETTLED_TRACE_COUNT_PER_SEED
        * len(SHARED_SETTLED_TRACE_SEEDS)
    )
    for name in (
        "predictions.jsonl",
        "outcomes.jsonl",
        "prediction_errors.jsonl",
        "segments.jsonl",
        "credit.jsonl",
    ):
        row_count = sum(
            bool(line)
            for line in (root / name).read_text(
                encoding="utf-8"
            ).splitlines()
        )
        if row_count != expected_source_rows:
            raise ValueError(
                f"Gate 4 {name} row count drifted: {row_count}"
            )
    expected_learning_rows = (
        GATE4_LABEL_BUDGET
        * (len(GATE4_ARM_NAMES) - 1)
        * len(SHARED_SETTLED_TRACE_SEEDS)
    )
    for name in ("state_diff.jsonl", "action_selection.jsonl"):
        row_count = sum(
            bool(line)
            for line in (root / name).read_text(
                encoding="utf-8"
            ).splitlines()
        )
        if row_count != expected_learning_rows:
            raise ValueError(
                f"Gate 4 {name} row count drifted: {row_count}"
            )
    return {
        "schema_version": GATE4_ACTIVE_LEARNING_SCHEMA_VERSION,
        "status": expected_status,
        "causal_passed": causal,
        "required_file_count": len(GATE4_REQUIRED_FILES),
        "source_row_count": expected_source_rows,
        "learning_row_count": expected_learning_rows,
        "locked_partition_consumed_once": verdict[
            "locked_partition_consumed_once"
        ],
    }


__all__ = [
    "GATE4_ACTIVE_LEARNING_SCHEMA_VERSION",
    "GATE4_ALIGNMENT_TARGET",
    "GATE4_ARM_NAMES",
    "GATE4_LABEL_BUDGET",
    "GATE4_REQUIRED_FILES",
    "Gate4ArmMetrics",
    "Gate4Comparison",
    "compare_gate4_arms",
    "export_gate4_active_learning_bundle",
    "run_gate4_arm",
    "validate_gate4_source_records",
    "verify_gate4_active_learning_bundle",
]
