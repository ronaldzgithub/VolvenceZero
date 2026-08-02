"""Matched long-session evidence for conditioning credit feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

CREDIT_LONGITUDINAL_SCHEMA_VERSION = "state-kv-credit-longitudinal.v2"


@dataclass(frozen=True)
class CreditLongitudinalSample:
    turn_index: int
    shadow_confidence: float
    active_confidence: float
    shadow_credit_delta: float
    active_credit_delta: float
    responses_differ: bool


@dataclass(frozen=True)
class CreditOutcomeJudgeResult:
    judge_model_id: str
    judge_family: str
    shadow_matched_count: int
    active_matched_count: int
    improvement_count: int
    regression_count: int
    sample_count: int

    def __post_init__(self) -> None:
        if not self.judge_model_id or not self.judge_family:
            raise ValueError("outcome judge requires model id and family")
        counts = (
            self.shadow_matched_count,
            self.active_matched_count,
            self.improvement_count,
            self.regression_count,
            self.sample_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("outcome judge counts must be non-negative")
        if self.shadow_matched_count > self.sample_count:
            raise ValueError("shadow matched count exceeds sample count")
        if self.active_matched_count > self.sample_count:
            raise ValueError("active matched count exceeds sample count")
        if self.improvement_count + self.regression_count > self.sample_count:
            raise ValueError("paired outcome changes exceed sample count")

    def as_json_dict(self) -> dict[str, object]:
        return {
            "judge_model_id": self.judge_model_id,
            "judge_family": self.judge_family,
            "shadow_matched_count": self.shadow_matched_count,
            "active_matched_count": self.active_matched_count,
            "improvement_count": self.improvement_count,
            "regression_count": self.regression_count,
            "net_improvement": self.improvement_count - self.regression_count,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class CreditLongitudinalVerdict:
    artifact_id: str
    sample_count: int
    mean_first_half_increment: float | None
    mean_second_half_increment: float | None
    growth: float | None
    response_divergence_rate: float
    mechanism_state: str
    outcome_claim_state: str
    outcome_judges: tuple[CreditOutcomeJudgeResult, ...]
    minimum_net_improvement: int

    @property
    def gate_state(self) -> str:
        if self.mechanism_state != "pass":
            return self.mechanism_state
        return (
            "pass"
            if self.outcome_claim_state == "pass"
            else "mechanism_supported"
        )

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": CREDIT_LONGITUDINAL_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "artifact_id": self.artifact_id,
            "sample_count": self.sample_count,
            "mean_first_half_increment": self.mean_first_half_increment,
            "mean_second_half_increment": self.mean_second_half_increment,
            "growth": self.growth,
            "response_divergence_rate": self.response_divergence_rate,
            "outcome_judges": [
                judge.as_json_dict() for judge in self.outcome_judges
            ],
            "outcome_preregistration": {
                "minimum_distinct_judges": 2,
                "minimum_net_improvement": self.minimum_net_improvement,
                "maximum_regressions": 0,
                "aggregation": "all-judges-pass",
            },
            "claims": [
                {
                    "claim": "claim_credit_feedback_applied_increment_grows",
                    "state": self.mechanism_state,
                    "detail": (
                        f"first_half={self.mean_first_half_increment!r}, "
                        f"second_half={self.mean_second_half_increment!r}, "
                        f"growth={self.growth!r}"
                    ),
                },
                {
                    "claim": "claim_credit_feedback_improves_matched_outcome",
                    "state": self.outcome_claim_state,
                    "detail": (
                        "Frozen dual embedding judges score the same matched "
                        "responses against the preregistered outcome rubric; "
                        f"judges={len(self.outcome_judges)}, "
                        f"minimum_net_improvement={self.minimum_net_improvement}."
                    ),
                },
            ],
            "description": (
                "Matched I/J long-session readout. Confidence and owner-"
                "published credit deltas are mechanism evidence; the verdict "
                "does not promote behavioral quality without an outcome judge."
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def build_credit_longitudinal_verdict(
    *,
    samples: Sequence[CreditLongitudinalSample],
    artifact_id: str,
    minimum_samples: int = 8,
    minimum_growth: float = 0.005,
    outcome_judges: Sequence[CreditOutcomeJudgeResult] = (),
    minimum_net_improvement: int = 1,
) -> CreditLongitudinalVerdict:
    if not artifact_id:
        raise ValueError("credit longitudinal verdict requires artifact_id")
    if minimum_samples < 4 or minimum_growth <= 0.0:
        raise ValueError("credit longitudinal thresholds are invalid")
    if minimum_net_improvement < 1:
        raise ValueError("minimum_net_improvement must be positive")
    increments = [
        sample.active_confidence - sample.shadow_confidence
        for sample in samples
    ]
    split = len(increments) // 2
    first = (
        sum(increments[:split]) / split
        if split
        else None
    )
    second_count = len(increments) - split
    second = (
        sum(increments[split:]) / second_count
        if second_count
        else None
    )
    growth = (
        second - first
        if first is not None and second is not None
        else None
    )
    if len(samples) < minimum_samples:
        mechanism_state = "insufficient_data"
    elif (
        growth is not None
        and growth >= minimum_growth
        and any(sample.active_credit_delta != 0.0 for sample in samples)
    ):
        mechanism_state = "pass"
    else:
        mechanism_state = "fail"
    frozen_judges = tuple(outcome_judges)
    judge_ids = {judge.judge_model_id for judge in frozen_judges}
    judge_families = {judge.judge_family for judge in frozen_judges}
    if frozen_judges and any(
        judge.sample_count != len(samples) for judge in frozen_judges
    ):
        raise ValueError("outcome judge sample count must match longitudinal samples")
    if len(frozen_judges) < 2:
        outcome_state = "insufficient_data"
    elif len(judge_ids) != len(frozen_judges) or len(judge_families) != len(
        frozen_judges
    ):
        raise ValueError("outcome panel requires distinct model ids and families")
    elif all(
        judge.improvement_count - judge.regression_count
        >= minimum_net_improvement
        and judge.regression_count == 0
        for judge in frozen_judges
    ):
        outcome_state = "pass"
    else:
        outcome_state = "fail"
    return CreditLongitudinalVerdict(
        artifact_id=artifact_id,
        sample_count=len(samples),
        mean_first_half_increment=first,
        mean_second_half_increment=second,
        growth=growth,
        response_divergence_rate=(
            sum(sample.responses_differ for sample in samples) / len(samples)
            if samples
            else 0.0
        ),
        mechanism_state=mechanism_state,
        outcome_claim_state=outcome_state,
        outcome_judges=frozen_judges,
        minimum_net_improvement=minimum_net_improvement,
    )


__all__ = [
    "CREDIT_LONGITUDINAL_SCHEMA_VERSION",
    "CreditLongitudinalSample",
    "CreditOutcomeJudgeResult",
    "CreditLongitudinalVerdict",
    "build_credit_longitudinal_verdict",
]
