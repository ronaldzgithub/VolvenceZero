from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from volvence_zero.semantic_state import (
    SemanticProposalOperation,
    SemanticProposalRuntime,
)


@dataclass(frozen=True)
class SemanticProposalQualityCase:
    case_id: str
    target_slot: str
    user_input: str
    expected_operations: tuple[SemanticProposalOperation, ...]
    min_confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class SemanticProposalQualityCaseResult:
    case_id: str
    target_slot: str
    expected_operations: tuple[SemanticProposalOperation, ...]
    observed_operations: tuple[SemanticProposalOperation, ...]
    true_positive_count: int
    false_positive_count: int
    missing_count: int
    confidence_floor_passed: bool
    fell_back: bool
    passed: bool
    description: str


@dataclass(frozen=True)
class SemanticProposalQualityReport:
    target_slot: str
    runtime_id: str
    case_results: tuple[SemanticProposalQualityCaseResult, ...]
    passed_case_count: int
    total_case_count: int
    precision: float
    recall: float
    false_positive_count: int
    missing_count: int
    fallback_count: int
    description: str


def evaluate_semantic_proposal_quality(
    *,
    runtime: SemanticProposalRuntime,
    cases: tuple[SemanticProposalQualityCase, ...],
) -> SemanticProposalQualityReport:
    if not cases:
        raise ValueError("Semantic proposal quality evaluation requires at least one case.")
    target_slots = {case.target_slot for case in cases}
    if len(target_slots) != 1:
        raise ValueError("All semantic proposal quality cases must target the same slot.")
    case_results: list[SemanticProposalQualityCaseResult] = []
    true_positive_total = false_positive_total = missing_total = fallback_count = 0
    runtime_id = runtime.runtime_id
    for turn_index, case in enumerate(cases):
        batch = runtime.propose(
            target_slot=case.target_slot,
            user_input=case.user_input,
            substrate_snapshot=None,
            memory_snapshot=None,
            previous_snapshot=None,
            turn_index=turn_index,
        )
        runtime_id = batch.runtime_id
        semantic_proposals = tuple(
            proposal
            for proposal in batch.proposals
            if proposal.operation is not SemanticProposalOperation.OBSERVE
        )
        observed = tuple(proposal.operation for proposal in semantic_proposals)
        expected_counts = Counter(case.expected_operations)
        observed_counts = Counter(observed)
        true_positive = sum(
            min(expected_counts[operation], observed_counts[operation])
            for operation in set(expected_counts) | set(observed_counts)
        )
        false_positive = sum(
            max(observed_counts[operation] - expected_counts[operation], 0)
            for operation in observed_counts
        )
        missing = sum(
            max(expected_counts[operation] - observed_counts[operation], 0)
            for operation in expected_counts
        )
        confidence_floor_passed = all(
            proposal.confidence >= case.min_confidence for proposal in semantic_proposals
        )
        fell_back = "fell back to base" in batch.description
        if fell_back:
            fallback_count += 1
        passed = false_positive == 0 and missing == 0 and confidence_floor_passed
        true_positive_total += true_positive
        false_positive_total += false_positive
        missing_total += missing
        case_results.append(
            SemanticProposalQualityCaseResult(
                case_id=case.case_id,
                target_slot=case.target_slot,
                expected_operations=case.expected_operations,
                observed_operations=observed,
                true_positive_count=true_positive,
                false_positive_count=false_positive,
                missing_count=missing,
                confidence_floor_passed=confidence_floor_passed,
                fell_back=fell_back,
                passed=passed,
                description=(
                    f"Case {case.case_id} expected={tuple(item.value for item in case.expected_operations)} "
                    f"observed={tuple(item.value for item in observed)} fallback={int(fell_back)}."
                ),
            )
        )
    precision_denominator = true_positive_total + false_positive_total
    recall_denominator = true_positive_total + missing_total
    precision = (
        true_positive_total / precision_denominator
        if precision_denominator > 0
        else 1.0
    )
    recall = true_positive_total / recall_denominator if recall_denominator > 0 else 1.0
    passed_case_count = sum(1 for result in case_results if result.passed)
    target_slot = next(iter(target_slots))
    return SemanticProposalQualityReport(
        target_slot=target_slot,
        runtime_id=runtime_id,
        case_results=tuple(case_results),
        passed_case_count=passed_case_count,
        total_case_count=len(case_results),
        precision=precision,
        recall=recall,
        false_positive_count=false_positive_total,
        missing_count=missing_total,
        fallback_count=fallback_count,
        description=(
            f"Semantic proposal quality target={target_slot} passed={passed_case_count}/{len(case_results)} "
            f"precision={precision:.3f} recall={recall:.3f} fallback={fallback_count}."
        ),
    )
