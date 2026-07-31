"""Shared, read-only evidence helpers for L1/L2 adapter promotion.

The helpers in this module never train, mutate, or publish an artifact.  They
parse a typed held-out corpus, aggregate teacher-forced continuation NLLs, and
ask the cognition-owned ``ModificationGate`` for the final allow/block
decision.  Model execution remains in the substrate runtime used by the two
CLI entry points.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from volvence_zero.credit import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.substrate import ContinuationScore

ADAPTER_HELD_OUT_SCHEMA_VERSION = "adapter-held-out-case.v1"
ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION = "adapter-promotion-evidence.v1"
_EXPECTATIONS = frozenset({"improve", "preserve"})


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class AdapterHeldOutCase:
    case_id: str
    cohort: str
    expectation: str
    source_text: str
    continuation_text: str
    conditioning_state: tuple[float, ...]
    counterfactual_conditioning_state: tuple[float, ...] | None
    applied_control: tuple[float, ...]
    schema_version: str = ADAPTER_HELD_OUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_HELD_OUT_SCHEMA_VERSION:
            raise ValueError(
                "held-out case schema_version must be "
                f"{ADAPTER_HELD_OUT_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("case_id", self.case_id),
            ("cohort", self.cohort),
            ("source_text", self.source_text),
            ("continuation_text", self.continuation_text),
        ):
            if not value.strip():
                raise ValueError(f"held-out case {name} must be non-empty.")
            if value != value.strip():
                raise ValueError(
                    f"held-out case {name} must not have outer whitespace."
                )
        if self.expectation not in _EXPECTATIONS:
            raise ValueError(
                "held-out case expectation must be 'improve' or 'preserve'."
            )
        _validate_state(self.conditioning_state, field_name="conditioning_state")
        if self.counterfactual_conditioning_state is not None:
            _validate_state(
                self.counterfactual_conditioning_state,
                field_name="counterfactual_conditioning_state",
            )
            if self.counterfactual_conditioning_state == self.conditioning_state:
                raise ValueError(
                    "counterfactual_conditioning_state must differ from "
                    "conditioning_state."
                )
        if any(not math.isfinite(value) for value in self.applied_control):
            raise ValueError("held-out applied_control must contain finite values.")


def _validate_state(values: tuple[float, ...], *, field_name: str) -> None:
    if len(values) != len(PERSONAL_CONDITIONING_VECTOR_LABELS):
        raise ValueError(
            f"held-out {field_name} must contain "
            f"{len(PERSONAL_CONDITIONING_VECTOR_LABELS)} coordinates."
        )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"held-out {field_name} values must be finite and in [0, 1].")


def load_held_out_cases(path: Path) -> tuple[AdapterHeldOutCase, ...]:
    required = {
        "schema_version",
        "case_id",
        "cohort",
        "expectation",
        "source_text",
        "continuation_text",
        "conditioning_state",
        "applied_control",
    }
    optional = {"counterfactual_conditioning_state"}
    cases: list[AdapterHeldOutCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"held-out line {line_number} must be an object.")
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required - optional)
        if missing or extra:
            raise ValueError(
                f"held-out line {line_number} fields mismatch: "
                f"missing={missing}, extra={extra}."
            )
        case = AdapterHeldOutCase(
            schema_version=str(raw["schema_version"]),
            case_id=str(raw["case_id"]),
            cohort=str(raw["cohort"]),
            expectation=str(raw["expectation"]),
            source_text=str(raw["source_text"]),
            continuation_text=str(raw["continuation_text"]),
            conditioning_state=tuple(
                float(value) for value in raw["conditioning_state"]
            ),
            counterfactual_conditioning_state=(
                tuple(
                    float(value)
                    for value in raw["counterfactual_conditioning_state"]
                )
                if "counterfactual_conditioning_state" in raw
                else None
            ),
            applied_control=tuple(float(value) for value in raw["applied_control"]),
        )
        if case.case_id in seen:
            raise ValueError(f"duplicate held-out case_id {case.case_id!r}.")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("held-out corpus must contain at least one case.")
    return tuple(cases)


def conditioning_snapshot(
    case: AdapterHeldOutCase,
    *,
    counterfactual: bool = False,
) -> PersonalConditioningSnapshot:
    state = (
        case.counterfactual_conditioning_state
        if counterfactual
        else case.conditioning_state
    )
    if state is None:
        raise ValueError(
            f"held-out case {case.case_id!r} has no counterfactual state."
        )
    state_payload = _canonical_json(list(state))
    fingerprint = hashlib.sha256(
        f"{case.case_id}:{counterfactual}:{state_payload}".encode("utf-8")
    ).hexdigest()
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=state,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("offline-held-out", 1),),
        source_fingerprint=fingerprint,
        confidence=1.0,
        is_cold_start=False,
        description=(
            f"Immutable held-out conditioning for {case.case_id}; "
            f"counterfactual={counterfactual}."
        ),
    )


@dataclass(frozen=True)
class AdapterArmObservation:
    case_id: str
    cohort: str
    expectation: str
    baseline_nll: float
    candidate_nll: float
    validation_delta: float
    relative_improvement: float
    counterfactual_nll: float | None
    own_state_margin: float | None
    token_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("cohort", self.cohort),
        ):
            if not value.strip():
                raise ValueError(f"adapter observation {name} must be non-empty.")
        if self.expectation not in _EXPECTATIONS:
            raise ValueError(
                "adapter observation expectation must be improve/preserve."
            )
        numeric = (
            self.baseline_nll,
            self.candidate_nll,
            self.validation_delta,
            self.relative_improvement,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("adapter observation values must be finite.")
        if self.baseline_nll < 0.0 or self.candidate_nll < 0.0:
            raise ValueError("adapter observation NLL values must be non-negative.")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError(
                "adapter observation token_count must be a positive integer."
            )
        expected_delta = self.baseline_nll - self.candidate_nll
        expected_relative = expected_delta / max(self.baseline_nll, 1e-9)
        if not math.isclose(
            self.validation_delta,
            expected_delta,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "adapter observation validation_delta does not match NLLs."
            )
        if not math.isclose(
            self.relative_improvement,
            expected_relative,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "adapter observation relative_improvement does not match NLLs."
            )
        if (self.counterfactual_nll is None) != (
            self.own_state_margin is None
        ):
            raise ValueError(
                "adapter observation counterfactual NLL and margin must be paired."
            )
        if self.counterfactual_nll is not None:
            if (
                not math.isfinite(self.counterfactual_nll)
                or self.counterfactual_nll < 0.0
                or not math.isfinite(float(self.own_state_margin))
            ):
                raise ValueError(
                    "adapter observation counterfactual values are invalid."
                )
            expected_margin = self.counterfactual_nll - self.candidate_nll
            if not math.isclose(
                float(self.own_state_margin),
                expected_margin,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "adapter observation own_state_margin does not match NLLs."
                )


def _require_score_binding(
    *,
    case: AdapterHeldOutCase,
    score: ContinuationScore,
    arm: str,
) -> None:
    if (
        score.source_text != case.source_text
        or score.continuation_text != case.continuation_text
    ):
        raise ValueError(
            f"held-out case {case.case_id!r} {arm} score text drifted."
        )


def collect_observations(
    *,
    cases: tuple[AdapterHeldOutCase, ...],
    baseline_scorer: Callable[[AdapterHeldOutCase], ContinuationScore],
    candidate_scorer: Callable[
        [AdapterHeldOutCase, bool], ContinuationScore
    ],
) -> tuple[AdapterArmObservation, ...]:
    rows: list[AdapterArmObservation] = []
    for case in cases:
        baseline = baseline_scorer(case)
        candidate = candidate_scorer(case, False)
        _require_score_binding(case=case, score=baseline, arm="baseline")
        _require_score_binding(case=case, score=candidate, arm="candidate")
        if baseline.token_count != candidate.token_count:
            raise ValueError(
                f"held-out case {case.case_id!r} arm token counts differ."
            )
        counterfactual = (
            candidate_scorer(case, True)
            if case.counterfactual_conditioning_state is not None
            else None
        )
        if counterfactual is not None:
            _require_score_binding(
                case=case,
                score=counterfactual,
                arm="counterfactual",
            )
        if counterfactual is not None and counterfactual.token_count != candidate.token_count:
            raise ValueError(
                f"held-out case {case.case_id!r} counterfactual token count differs."
            )
        delta = baseline.mean_negative_log_likelihood - candidate.mean_negative_log_likelihood
        rows.append(
            AdapterArmObservation(
                case_id=case.case_id,
                cohort=case.cohort,
                expectation=case.expectation,
                baseline_nll=baseline.mean_negative_log_likelihood,
                candidate_nll=candidate.mean_negative_log_likelihood,
                validation_delta=delta,
                relative_improvement=delta / max(
                    baseline.mean_negative_log_likelihood,
                    1e-9,
                ),
                counterfactual_nll=(
                    counterfactual.mean_negative_log_likelihood
                    if counterfactual is not None
                    else None
                ),
                own_state_margin=(
                    counterfactual.mean_negative_log_likelihood
                    - candidate.mean_negative_log_likelihood
                    if counterfactual is not None
                    else None
                ),
                token_count=candidate.token_count,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class AdapterPromotionThresholds:
    min_case_count: int = 8
    min_mean_relative_improvement: float = 0.01
    max_regression_rate: float = 0.25
    max_preservation_nll_regression: float = 0.05
    min_counterfactual_accuracy: float = 0.60

    def __post_init__(self) -> None:
        if self.min_case_count < 2:
            raise ValueError("min_case_count must be at least 2.")
        if (
            not math.isfinite(self.min_mean_relative_improvement)
            or self.min_mean_relative_improvement < 0.0
        ):
            raise ValueError("min_mean_relative_improvement must be non-negative.")
        for name, value in (
            ("max_regression_rate", self.max_regression_rate),
            ("min_counterfactual_accuracy", self.min_counterfactual_accuracy),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if (
            not math.isfinite(self.max_preservation_nll_regression)
            or self.max_preservation_nll_regression < 0.0
        ):
            raise ValueError(
                "max_preservation_nll_regression must be non-negative."
            )


def summarize_observations(
    *,
    observations: tuple[AdapterArmObservation, ...],
    thresholds: AdapterPromotionThresholds,
) -> dict[str, object]:
    if not observations:
        raise ValueError("adapter promotion requires at least one observation.")
    improve = tuple(row for row in observations if row.expectation == "improve")
    preserve = tuple(row for row in observations if row.expectation == "preserve")
    counterfactual = tuple(
        row for row in observations if row.own_state_margin is not None
    )
    validation_delta = (
        sum(row.validation_delta for row in improve) / len(improve)
        if improve
        else 0.0
    )
    mean_relative_improvement = (
        sum(row.relative_improvement for row in improve) / len(improve)
        if improve
        else 0.0
    )
    regression_rate = sum(
        1 for row in observations if row.validation_delta < 0.0
    ) / len(observations)
    preservation_failures = tuple(
        row.case_id
        for row in preserve
        if -row.validation_delta > thresholds.max_preservation_nll_regression
    )
    counterfactual_accuracy = (
        sum(1 for row in counterfactual if float(row.own_state_margin) > 0.0)
        / len(counterfactual)
        if counterfactual
        else 0.0
    )
    blocking_reasons: list[str] = []
    if len(observations) < thresholds.min_case_count:
        blocking_reasons.append(
            f"case_count={len(observations)} below {thresholds.min_case_count}"
        )
    if not improve:
        blocking_reasons.append("missing improve cohort")
    if not preserve:
        blocking_reasons.append("missing preserve cohort")
    if not counterfactual:
        blocking_reasons.append("missing counterfactual conditioning arm")
    if mean_relative_improvement < thresholds.min_mean_relative_improvement:
        blocking_reasons.append(
            "mean_relative_improvement="
            f"{mean_relative_improvement:.6f} below "
            f"{thresholds.min_mean_relative_improvement:.6f}"
        )
    if regression_rate > thresholds.max_regression_rate:
        blocking_reasons.append(
            f"regression_rate={regression_rate:.6f} above "
            f"{thresholds.max_regression_rate:.6f}"
        )
    if preservation_failures:
        blocking_reasons.append(
            "preservation NLL regression exceeded cap for "
            f"{list(preservation_failures)!r}"
        )
    if counterfactual_accuracy < thresholds.min_counterfactual_accuracy:
        blocking_reasons.append(
            f"counterfactual_accuracy={counterfactual_accuracy:.6f} below "
            f"{thresholds.min_counterfactual_accuracy:.6f}"
        )
    return {
        "case_count": len(observations),
        "improve_case_count": len(improve),
        "preserve_case_count": len(preserve),
        "counterfactual_case_count": len(counterfactual),
        "validation_delta": validation_delta,
        "mean_relative_improvement": mean_relative_improvement,
        "regression_rate": regression_rate,
        "counterfactual_accuracy": counterfactual_accuracy,
        "preservation_failure_case_ids": list(preservation_failures),
        "evidence_integrity": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }


def decide_offline_promotion(
    *,
    target: str,
    old_value_hash: str,
    new_value_hash: str,
    summary: dict[str, object],
    capacity_cost: float,
    rollback_evidence: str,
) -> tuple[GateDecision, tuple[str, ...], EvaluationSnapshot]:
    contract_integrity = 1.0 if summary["evidence_integrity"] else 0.0
    evaluation_snapshot = EvaluationSnapshot(
        turn_scores=(),
        session_scores=(
            EvaluationScore(
                family="learning",
                metric_name="adapter_validation_delta",
                value=float(summary["validation_delta"]),
                confidence=1.0,
                evidence="immutable held-out continuation NLL comparison",
            ),
            EvaluationScore(
                family="safety",
                metric_name="contract_integrity",
                value=contract_integrity,
                confidence=1.0,
                evidence="typed held-out cohort and counterfactual coverage",
            ),
            EvaluationScore(
                family="safety",
                metric_name="fallback_reliance",
                value=0.0,
                confidence=1.0,
                evidence="real frozen substrate arms only",
            ),
            EvaluationScore(
                family="safety",
                metric_name="rollback_resilience",
                value=1.0 if rollback_evidence.strip() else 0.0,
                confidence=1.0,
                evidence="byte-identical base/common-only control arm retained",
            ),
        ),
        alerts=(),
        structured_alerts=(),
        description="Offline adapter promotion evidence readout.",
    )
    proposal = ModificationProposal(
        target=target,
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=old_value_hash,
        new_value_hash=new_value_hash,
        justification=(
            "Promote an immutable adapter artifact from held-out base/candidate "
            "teacher-forced continuation evidence."
        ),
        is_reversible=True,
        validation_delta=float(summary["validation_delta"]),
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )
    reasons = evaluate_gate_reasons(
        proposal=proposal,
        evaluation_snapshot=evaluation_snapshot,
    )
    decision = GateDecision.BLOCK if reasons else GateDecision.ALLOW
    return decision, reasons, evaluation_snapshot


def evaluation_id(
    *,
    subject_id: str,
    observations: tuple[AdapterArmObservation, ...],
    thresholds: AdapterPromotionThresholds,
) -> str:
    payload = {
        "subject_id": subject_id,
        "observations": [asdict(row) for row in observations],
        "thresholds": asdict(thresholds),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "ADAPTER_HELD_OUT_SCHEMA_VERSION",
    "ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION",
    "AdapterArmObservation",
    "AdapterHeldOutCase",
    "AdapterPromotionThresholds",
    "collect_observations",
    "conditioning_snapshot",
    "decide_offline_promotion",
    "evaluation_id",
    "load_held_out_cases",
    "summarize_observations",
]
