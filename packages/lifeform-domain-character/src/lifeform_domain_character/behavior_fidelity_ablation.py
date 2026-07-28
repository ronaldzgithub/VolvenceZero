"""Read-only causal ablation reports for frozen behavior matrices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lifeform_domain_character.behavior_fidelity_matrix import (
    BehaviorFidelityCaseKind,
    BehaviorFidelityMatrix,
    PromotionExpectation,
)


class BehaviorFidelityAblationArm(str, Enum):
    BAKED = "baked"
    COLD = "cold"
    NO_RL = "no_rl"
    SHUFFLED_LINEAGE = "shuffled_lineage"


@dataclass(frozen=True)
class BehaviorFidelityCaseObservation:
    suite_digest: str
    arm: BehaviorFidelityAblationArm
    case_id: str
    kind: BehaviorFidelityCaseKind
    promotion_expectation: PromotionExpectation
    target_promotion_used: bool
    action_grounding_source_case_id: str | None
    source_state_digest_verified: bool
    outcome_feedback_submitted: bool
    evaluation_feedback_submitted: bool
    fidelity_score: float | None = None
    competing_behavior_family_matched: bool | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("suite_digest", self.suite_digest),
            ("case_id", self.case_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.target_promotion_used and (
            self.action_grounding_source_case_id is None
            or not self.action_grounding_source_case_id.strip()
        ):
            raise ValueError(
                "target promotion use requires a grounding source case id"
            )
        if self.fidelity_score is not None and not (
            0.0 <= self.fidelity_score <= 1.0
        ):
            raise ValueError("fidelity_score must be in [0, 1]")
        if (
            self.kind is not BehaviorFidelityCaseKind.COMPETING_BEHAVIOR
            and self.competing_behavior_family_matched is not None
        ):
            raise ValueError(
                "competing behavior match is only valid for competing cases"
            )


@dataclass(frozen=True)
class BehaviorFidelityArmReport:
    arm: BehaviorFidelityAblationArm
    case_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    promotion_precision: float | None
    promotion_recall: float
    promotion_specificity: float
    positive_promotion_hits: int
    non_positive_promotion_hits: int
    source_integrity_passed: bool
    no_feedback_passed: bool
    minimum_fidelity_score: float | None
    positive_mean_fidelity_score: float | None
    competing_behavior_matches: int | None
    gate_statuses: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BehaviorFidelityCausalAblationReport:
    suite_id: str
    suite_digest: str
    arm_reports: tuple[BehaviorFidelityArmReport, ...]
    causal_gate_statuses: tuple[tuple[str, str], ...]
    lineage_causal_supported: bool
    behavior_causal_supported: bool
    claim_status: str
    description: str


def evaluate_behavior_fidelity_ablation(
    *,
    matrix: BehaviorFidelityMatrix,
    observations: tuple[BehaviorFidelityCaseObservation, ...],
) -> BehaviorFidelityCausalAblationReport:
    expected_pairs = {
        (arm, case.stimulus.case_id)
        for arm in BehaviorFidelityAblationArm
        for case in matrix.cases
    }
    actual_pairs = {
        (observation.arm, observation.case_id)
        for observation in observations
    }
    if len(actual_pairs) != len(observations):
        raise ValueError("ablation observations contain duplicate arm/case")
    if actual_pairs != expected_pairs:
        missing = sorted(
            (arm.value, case_id)
            for arm, case_id in expected_pairs - actual_pairs
        )
        extra = sorted(
            (arm.value, case_id)
            for arm, case_id in actual_pairs - expected_pairs
        )
        raise ValueError(
            f"ablation observation coverage mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    cases_by_id = {
        case.stimulus.case_id: case for case in matrix.cases
    }
    for observation in observations:
        case = cases_by_id[observation.case_id]
        if observation.suite_digest != matrix.digest:
            raise ValueError(
                "ablation observation suite digest mismatch for "
                f"{observation.arm.value}/{observation.case_id}"
            )
        if (
            observation.kind is not case.kind
            or observation.promotion_expectation
            is not case.promotion_expectation
        ):
            raise ValueError(
                "ablation observation semantic binding mismatch for "
                f"{observation.arm.value}/{observation.case_id}"
            )

    reports = tuple(
        _build_arm_report(
            matrix=matrix,
            arm=arm,
            observations=tuple(
                observation
                for observation in observations
                if observation.arm is arm
            ),
        )
        for arm in BehaviorFidelityAblationArm
    )
    by_arm = {report.arm: report for report in reports}
    baked = by_arm[BehaviorFidelityAblationArm.BAKED]
    cold = by_arm[BehaviorFidelityAblationArm.COLD]
    no_rl = by_arm[BehaviorFidelityAblationArm.NO_RL]
    shuffled = by_arm[BehaviorFidelityAblationArm.SHUFFLED_LINEAGE]
    positive_hit_threshold = (
        matrix.thresholds.minimum_positive_promotion_hits
    )
    non_positive_hit_ceiling = (
        matrix.thresholds.maximum_non_positive_promotion_hits
    )
    all_integrity_passed = all(
        report.source_integrity_passed and report.no_feedback_passed
        for report in reports
    )
    lineage_causal_supported = (
        baked.positive_promotion_hits >= positive_hit_threshold
        and baked.non_positive_promotion_hits
        <= non_positive_hit_ceiling
        and cold.positive_promotion_hits == 0
        and cold.non_positive_promotion_hits == 0
        and no_rl.positive_promotion_hits == 0
        and no_rl.non_positive_promotion_hits == 0
        and shuffled.positive_promotion_hits == 0
        and shuffled.non_positive_promotion_hits == 0
        and all_integrity_passed
    )
    has_complete_fidelity = all(
        report.minimum_fidelity_score is not None
        and report.positive_mean_fidelity_score is not None
        and report.competing_behavior_matches is not None
        for report in reports
    )
    positive_delta = (
        baked.positive_mean_fidelity_score
        - cold.positive_mean_fidelity_score
        if (
            baked.positive_mean_fidelity_score is not None
            and cold.positive_mean_fidelity_score is not None
        )
        else None
    )
    behavior_causal_supported = (
        lineage_causal_supported
        and has_complete_fidelity
        and baked.minimum_fidelity_score
        >= matrix.thresholds.minimum_case_fidelity_score
        and positive_delta
        >= matrix.thresholds.minimum_positive_mean_baked_cold_delta
        and baked.competing_behavior_matches
        == dict(matrix.thresholds.required_case_counts)[
            BehaviorFidelityCaseKind.COMPETING_BEHAVIOR
        ]
    )
    claim_status = (
        "behavior-causal-diagnostic-pass"
        if behavior_causal_supported
        else (
            "lineage-causal-diagnostic-pass"
            if lineage_causal_supported
            else "diagnostic-fail"
        )
    )
    causal_gates = (
        (
            "baked_positive_promotion_hits",
            (
                "pass"
                if baked.positive_promotion_hits
                >= positive_hit_threshold
                else "fail"
            ),
        ),
        (
            "baked_non_positive_specificity",
            (
                "pass"
                if baked.non_positive_promotion_hits
                <= non_positive_hit_ceiling
                else "fail"
            ),
        ),
        (
            "cold_target_promotion_absent",
            (
                "pass"
                if cold.positive_promotion_hits == 0
                and cold.non_positive_promotion_hits == 0
                else "fail"
            ),
        ),
        (
            "no_rl_target_promotion_absent",
            (
                "pass"
                if no_rl.positive_promotion_hits == 0
                and no_rl.non_positive_promotion_hits == 0
                else "fail"
            ),
        ),
        (
            "shuffled_lineage_target_promotion_absent",
            (
                "pass"
                if shuffled.positive_promotion_hits == 0
                and shuffled.non_positive_promotion_hits == 0
                else "fail"
            ),
        ),
        (
            "all_arm_source_integrity_and_no_feedback",
            "pass" if all_integrity_passed else "fail",
        ),
        (
            "reviewed_behavior_evidence_complete",
            "pass" if has_complete_fidelity else "insufficient_data",
        ),
        (
            "positive_baked_cold_fidelity_delta",
            (
                "pass"
                if positive_delta is not None
                and positive_delta
                >= (
                    matrix.thresholds
                    .minimum_positive_mean_baked_cold_delta
                )
                else (
                    "insufficient_data"
                    if positive_delta is None
                    else "fail"
                )
            ),
        ),
    )
    return BehaviorFidelityCausalAblationReport(
        suite_id=matrix.suite_id,
        suite_digest=matrix.digest,
        arm_reports=reports,
        causal_gate_statuses=causal_gates,
        lineage_causal_supported=lineage_causal_supported,
        behavior_causal_supported=behavior_causal_supported,
        claim_status=claim_status,
        description=(
            "Read-only four-arm causal ablation. Structural lineage and "
            "reviewed behavior claims remain separate; missing reviewed "
            "scores cannot be promoted into behavior evidence."
        ),
    )


def _build_arm_report(
    *,
    matrix: BehaviorFidelityMatrix,
    arm: BehaviorFidelityAblationArm,
    observations: tuple[BehaviorFidelityCaseObservation, ...],
) -> BehaviorFidelityArmReport:
    positives = tuple(
        observation
        for observation in observations
        if observation.kind is BehaviorFidelityCaseKind.POSITIVE
    )
    non_positives = tuple(
        observation
        for observation in observations
        if observation.kind is not BehaviorFidelityCaseKind.POSITIVE
    )
    competing = tuple(
        observation
        for observation in observations
        if observation.kind
        is BehaviorFidelityCaseKind.COMPETING_BEHAVIOR
    )
    fidelity_scores = tuple(
        observation.fidelity_score
        for observation in observations
        if observation.fidelity_score is not None
    )
    positive_fidelity_scores = tuple(
        observation.fidelity_score
        for observation in positives
        if observation.fidelity_score is not None
    )
    complete_fidelity = len(fidelity_scores) == len(observations)
    complete_positive_fidelity = (
        len(positive_fidelity_scores) == len(positives)
    )
    complete_competing_review = all(
        observation.competing_behavior_family_matched is not None
        for observation in competing
    )
    source_integrity_passed = all(
        observation.source_state_digest_verified
        for observation in observations
    )
    no_feedback_passed = all(
        not observation.outcome_feedback_submitted
        and not observation.evaluation_feedback_submitted
        for observation in observations
    )
    positive_promotion_hits = sum(
        observation.target_promotion_used for observation in positives
    )
    non_positive_promotion_hits = sum(
        observation.target_promotion_used for observation in non_positives
    )
    true_positive_count = positive_promotion_hits
    false_positive_count = non_positive_promotion_hits
    false_negative_count = len(positives) - true_positive_count
    true_negative_count = len(non_positives) - false_positive_count
    predicted_positive_count = (
        true_positive_count + false_positive_count
    )
    promotion_precision = (
        round(true_positive_count / predicted_positive_count, 6)
        if predicted_positive_count
        else None
    )
    promotion_recall = round(
        true_positive_count / len(positives),
        6,
    )
    promotion_specificity = round(
        true_negative_count / len(non_positives),
        6,
    )
    minimum_fidelity_score = (
        min(fidelity_scores) if complete_fidelity else None
    )
    positive_mean_fidelity_score = (
        round(
            sum(positive_fidelity_scores)
            / len(positive_fidelity_scores),
            6,
        )
        if complete_positive_fidelity
        else None
    )
    competing_behavior_matches = (
        sum(
            observation.competing_behavior_family_matched is True
            for observation in competing
        )
        if complete_competing_review
        else None
    )
    gates = (
        (
            "source_integrity",
            "pass" if source_integrity_passed else "fail",
        ),
        (
            "no_feedback",
            "pass" if no_feedback_passed else "fail",
        ),
        (
            "fidelity_scores_complete",
            "pass" if complete_fidelity else "insufficient_data",
        ),
        (
            "competing_behavior_review_complete",
            (
                "pass"
                if complete_competing_review
                else "insufficient_data"
            ),
        ),
    )
    return BehaviorFidelityArmReport(
        arm=arm,
        case_count=len(observations),
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        true_negative_count=true_negative_count,
        promotion_precision=promotion_precision,
        promotion_recall=promotion_recall,
        promotion_specificity=promotion_specificity,
        positive_promotion_hits=positive_promotion_hits,
        non_positive_promotion_hits=non_positive_promotion_hits,
        source_integrity_passed=source_integrity_passed,
        no_feedback_passed=no_feedback_passed,
        minimum_fidelity_score=minimum_fidelity_score,
        positive_mean_fidelity_score=positive_mean_fidelity_score,
        competing_behavior_matches=competing_behavior_matches,
        gate_statuses=gates,
    )


__all__ = [
    "BehaviorFidelityAblationArm",
    "BehaviorFidelityArmReport",
    "BehaviorFidelityCaseObservation",
    "BehaviorFidelityCausalAblationReport",
    "evaluate_behavior_fidelity_ablation",
]
