"""Typed eval-gate contracts (Slice 6).

The platform exposes three control-plane surfaces that gate launch:

* Audience analysis — an immutable profile per cohort that
  summarises common questions / communication style / emotion
  triggers from a tenant-supplied corpus.
* Exam questions + runs — typed scenarios with rubric breakdowns
  and run aggregates. Runs can be completed with caller-supplied
  AI responses or executed by the platform against a live
  ``ai_id``.
* Launch license — gating signal for ``template.status →
  published``. The license is granted iff a passing exam run
  exists for the template version.

All three surfaces are **readouts** (R12 / EVO-2 / OA-1):

* No reward / Face gradient ever flows back from these endpoints.
* The LLM judge — when wired — returns a structured rubric score
  set, not a single scalar. Slice 6 ships a deterministic
  fail-closed scorer; the LLM judge backend is plugged in via the
  ``RubricGrader`` protocol when a real provider is configured.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Audience analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudienceProfileSpec:
    """One cohort summary derived from a tenant-supplied corpus."""

    profile_id: str
    template_id: str
    cohort_name: str
    asset_ids: tuple[str, ...]
    common_questions: tuple[str, ...] = ()
    communication_style: str = ""
    emotion_triggers: tuple[str, ...] = ()
    decision_patterns: tuple[str, ...] = ()
    evidence_stats: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "template_id": self.template_id,
            "cohort_name": self.cohort_name,
            "asset_ids": list(self.asset_ids),
            "common_questions": list(self.common_questions),
            "communication_style": self.communication_style,
            "emotion_triggers": list(self.emotion_triggers),
            "decision_patterns": list(self.decision_patterns),
            "evidence_stats": dict(self.evidence_stats),
            "created_at_ms": self.created_at_ms,
        }


# ---------------------------------------------------------------------------
# Exam questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricEntry:
    """One rubric criterion."""

    criterion: str
    description: str = ""
    max_score: float = 10.0
    weight: float = 1.0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RubricEntry":
        criterion = str(data.get("criterion", "") or "")
        if not criterion.strip():
            raise ValueError("RubricEntry.criterion must be non-empty")
        return cls(
            criterion=criterion,
            description=str(data.get("description", "") or ""),
            max_score=float(data.get("max_score", 10.0) or 10.0),
            weight=float(data.get("weight", 1.0) or 1.0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "description": self.description,
            "max_score": self.max_score,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class ExamQuestionSpec:
    """One scenario-tagged exam question."""

    question_id: str
    template_id: str
    scenario_tag: str
    user_prompt: str
    context: Mapping[str, Any] = field(default_factory=dict)
    rubric: tuple[RubricEntry, ...] = ()
    reference_answer: str = ""
    tags: tuple[str, ...] = ()
    difficulty: str = "medium"
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "template_id": self.template_id,
            "scenario_tag": self.scenario_tag,
            "user_prompt": self.user_prompt,
            "context": dict(self.context),
            "rubric": [r.to_json() for r in self.rubric],
            "reference_answer": self.reference_answer,
            "tags": list(self.tags),
            "difficulty": self.difficulty,
            "created_at_ms": self.created_at_ms,
        }


# ---------------------------------------------------------------------------
# Exam runs
# ---------------------------------------------------------------------------


class ExamRunStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ExamSubmissionScore:
    """Per-question score breakdown."""

    question_id: str
    ai_response: str
    weighted_score: float
    rubric_breakdown: tuple[Mapping[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "ai_response": self.ai_response,
            "weighted_score": self.weighted_score,
            "rubric_breakdown": [dict(b) for b in self.rubric_breakdown],
        }


@dataclass(frozen=True)
class ExamRunSpec:
    """Aggregate state of one exam run."""

    run_id: str
    template_id: str
    template_version: int
    run_type: str
    question_ids: tuple[str, ...] = ()
    status: ExamRunStatus = ExamRunStatus.PENDING
    operator_id: str = ""
    operator_name: str = ""
    comment: str = ""
    ai_id: str = ""
    contract_id: str = ""
    session_id: str = ""
    aggregate_score: float = 0.0
    pass_threshold: float = 0.6
    passed: bool = False
    wrong_set: tuple[str, ...] = ()
    submissions: tuple[ExamSubmissionScore, ...] = ()
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "run_type": self.run_type,
            "question_ids": list(self.question_ids),
            "status": self.status.value,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "comment": self.comment,
            "ai_id": self.ai_id,
            "contract_id": self.contract_id,
            "session_id": self.session_id,
            "aggregate_score": self.aggregate_score,
            "pass_threshold": self.pass_threshold,
            "passed": self.passed,
            "wrong_set": list(self.wrong_set),
            "submissions": [s.to_json() for s in self.submissions],
            "created_at_ms": self.created_at_ms,
        }


# ---------------------------------------------------------------------------
# Launch license
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchLicenseSpec:
    """Outcome of evaluating launch readiness for a template version."""

    license_id: str
    template_id: str
    template_version: int
    granted: bool
    reason: str = ""
    granted_by_run_id: str = ""
    issued_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "license_id": self.license_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "granted": self.granted,
            "reason": self.reason,
            "granted_by_run_id": self.granted_by_run_id,
            "issued_at_ms": self.issued_at_ms,
        }


__all__ = [
    "AudienceProfileSpec",
    "ExamQuestionSpec",
    "ExamRunSpec",
    "ExamRunStatus",
    "ExamSubmissionScore",
    "LaunchLicenseSpec",
    "RubricEntry",
]
