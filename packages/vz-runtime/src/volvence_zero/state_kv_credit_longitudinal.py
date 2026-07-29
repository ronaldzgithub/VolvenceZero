"""Matched long-session evidence for conditioning credit feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

CREDIT_LONGITUDINAL_SCHEMA_VERSION = "state-kv-credit-longitudinal.v1"


@dataclass(frozen=True)
class CreditLongitudinalSample:
    turn_index: int
    shadow_confidence: float
    active_confidence: float
    shadow_credit_delta: float
    active_credit_delta: float
    responses_differ: bool


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
                        "Requires a frozen matched quality/outcome judge; "
                        "response divergence alone is mechanism evidence."
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
    matched_outcome_improved: bool | None = None,
) -> CreditLongitudinalVerdict:
    if not artifact_id:
        raise ValueError("credit longitudinal verdict requires artifact_id")
    if minimum_samples < 4 or minimum_growth <= 0.0:
        raise ValueError("credit longitudinal thresholds are invalid")
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
    outcome_state = (
        "insufficient_data"
        if matched_outcome_improved is None
        else ("pass" if matched_outcome_improved else "fail")
    )
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
    )


__all__ = [
    "CREDIT_LONGITUDINAL_SCHEMA_VERSION",
    "CreditLongitudinalSample",
    "CreditLongitudinalVerdict",
    "build_credit_longitudinal_verdict",
]
