"""Typed interview-run contracts (面试 gate).

An interview run is an **interactive** evaluation of one persona
(template + optional live ``ai_id``): an interviewer — an LLM
interviewer following a question plan, a human operator, or both —
asks multi-turn questions, each turn is scored, and the run completes
with an aggregate verdict.

Position in the unified persona training lifecycle::

    ... → exam → **interview** → inducted

The interview is the last gate before induction/publish. Like the exam
gate it is a **readout** (R12): no reward or learning signal ever
flows back from interview scores into the kernel.

Turn scores are normalised to ``[0, 1]``. A run can only complete when
every recorded turn carries a score — there is no silent pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InterviewRunStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class InterviewerKind(str, Enum):
    LLM = "llm"
    OPERATOR = "operator"
    MIXED = "mixed"


@dataclass(frozen=True)
class InterviewTurn:
    """One question/answer exchange inside an interview run."""

    turn_index: int
    question: str
    ai_response: str = ""
    asked_by: str = "operator"  # "llm" | "operator"
    score: float | None = None
    notes: str = ""
    recorded_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "InterviewTurn":
        question = str(data.get("question", "") or "")
        if not question.strip():
            raise ValueError("InterviewTurn.question must be non-empty")
        raw_score = data.get("score")
        score: float | None
        if raw_score is None:
            score = None
        else:
            score = float(raw_score)
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"InterviewTurn.score must be in [0, 1], got {score!r}"
                )
        return cls(
            turn_index=int(data.get("turn_index", 0)),
            question=question,
            ai_response=str(data.get("ai_response", "") or ""),
            asked_by=str(data.get("asked_by", "operator") or "operator"),
            score=score,
            notes=str(data.get("notes", "") or ""),
            recorded_at_ms=int(data.get("recorded_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "question": self.question,
            "ai_response": self.ai_response,
            "asked_by": self.asked_by,
            "score": self.score,
            "notes": self.notes,
            "recorded_at_ms": self.recorded_at_ms,
        }


@dataclass(frozen=True)
class InterviewRunSpec:
    """Aggregate state of one interview run."""

    run_id: str
    template_id: str
    template_version: int = 1
    ai_id: str = ""
    session_id: str = ""
    interviewer_kind: InterviewerKind = InterviewerKind.OPERATOR
    status: InterviewRunStatus = InterviewRunStatus.PENDING
    question_plan: tuple[str, ...] = ()
    turns: tuple[InterviewTurn, ...] = ()
    aggregate_score: float = 0.0
    pass_threshold: float = 0.6
    passed: bool = False
    operator_id: str = ""
    verdict_comment: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "ai_id": self.ai_id,
            "session_id": self.session_id,
            "interviewer_kind": self.interviewer_kind.value,
            "status": self.status.value,
            "question_plan": list(self.question_plan),
            "turns": [t.to_json() for t in self.turns],
            "aggregate_score": self.aggregate_score,
            "pass_threshold": self.pass_threshold,
            "passed": self.passed,
            "operator_id": self.operator_id,
            "verdict_comment": self.verdict_comment,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }


__all__ = [
    "InterviewRunSpec",
    "InterviewRunStatus",
    "InterviewTurn",
    "InterviewerKind",
]
