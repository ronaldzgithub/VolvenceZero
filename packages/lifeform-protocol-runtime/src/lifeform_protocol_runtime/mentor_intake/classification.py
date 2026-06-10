"""Typed mentor-intake classifier.

Mentor intake is a routing step, not a new runtime owner. It turns
human guidance into an explicit owner decision so service code can route
to protocol runtime, knowledge, case, or experience surfaces without
prompt-only side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lifeform_protocol_runtime.document_uptake.extraction import LlmJsonClient
from lifeform_protocol_runtime.mentor_intake.prompts import (
    MENTOR_INTAKE_CLASSIFIER_SYSTEM_PROMPT,
    MENTOR_INTAKE_CLASSIFIER_USER_TEMPLATE,
)


class MentorIntakeKind(str, Enum):
    PROTOCOL = "protocol"
    PROTOCOL_REVISION = "protocol_revision"
    BOUNDARY = "boundary"
    KNOWLEDGE = "knowledge"
    CASE = "case"
    EXPERIENCE = "experience"


class MentorIntakeApplyMode(str, Enum):
    CLASSIFY_ONLY = "classify_only"
    APPLY_TO_SESSION = "apply_to_session"
    SUBMIT_FOR_REVIEW = "submit_for_review"


@dataclass(frozen=True)
class MentorIntakeRequest:
    guidance: str
    mentor_id: str = "anonymous-mentor"
    target_protocol_id: str | None = None
    apply_mode: MentorIntakeApplyMode = MentorIntakeApplyMode.CLASSIFY_ONLY


@dataclass(frozen=True)
class MentorIntakeDecision:
    intake_kind: MentorIntakeKind
    routed_owner: str
    confidence: float
    reason: str
    actionable_summary: str
    applies_to_current_session: bool = False
    unsupported_reason: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "intake_kind": self.intake_kind.value,
            "routed_owner": self.routed_owner,
            "confidence": self.confidence,
            "reason": self.reason,
            "actionable_summary": self.actionable_summary,
            "applies_to_current_session": self.applies_to_current_session,
            "unsupported_reason": self.unsupported_reason,
        }


_KIND_BY_VALUE: dict[str, MentorIntakeKind] = {
    kind.value: kind for kind in MentorIntakeKind
}

_DEFAULT_OWNER_BY_KIND: dict[MentorIntakeKind, str] = {
    MentorIntakeKind.PROTOCOL: "protocol_registry",
    MentorIntakeKind.PROTOCOL_REVISION: "protocol_registry",
    MentorIntakeKind.BOUNDARY: "protocol_registry",
    MentorIntakeKind.KNOWLEDGE: "domain_knowledge",
    MentorIntakeKind.CASE: "case_memory",
    MentorIntakeKind.EXPERIENCE: "experience_consolidation",
}


def classify_mentor_intake(
    request: MentorIntakeRequest,
    *,
    llm_client: LlmJsonClient,
) -> MentorIntakeDecision:
    """Classify mentor guidance into the owner that should receive it."""

    if not request.guidance.strip():
        raise ValueError("classify_mentor_intake: guidance must be non-empty")
    payload = llm_client.complete_json(
        system_prompt=MENTOR_INTAKE_CLASSIFIER_SYSTEM_PROMPT,
        user_prompt=MENTOR_INTAKE_CLASSIFIER_USER_TEMPLATE.format(
            mentor_id=request.mentor_id,
            target_protocol_id=request.target_protocol_id or "",
            guidance=request.guidance.strip(),
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("classify_mentor_intake: LLM returned non-object JSON")

    raw_kind = str(payload.get("intake_kind") or "").strip()
    kind = _KIND_BY_VALUE.get(raw_kind)
    if kind is None:
        raise ValueError(
            f"classify_mentor_intake: unknown intake_kind {raw_kind!r}"
        )

    routed_owner = str(payload.get("routed_owner") or "").strip()
    if not routed_owner:
        routed_owner = _DEFAULT_OWNER_BY_KIND[kind]
    confidence = _clamp_confidence(payload.get("confidence", 0.0))
    reason = str(payload.get("reason") or "").strip()
    actionable_summary = str(payload.get("actionable_summary") or "").strip()
    if not actionable_summary:
        actionable_summary = request.guidance.strip()

    unsupported_reason = ""
    if kind in {
        MentorIntakeKind.KNOWLEDGE,
        MentorIntakeKind.CASE,
        MentorIntakeKind.EXPERIENCE,
        MentorIntakeKind.PROTOCOL_REVISION,
    }:
        unsupported_reason = (
            f"{kind.value} mentor intake is classified but not live-applied "
            "by the session-local protocol path yet."
        )

    return MentorIntakeDecision(
        intake_kind=kind,
        routed_owner=routed_owner,
        confidence=confidence,
        reason=reason,
        actionable_summary=actionable_summary,
        unsupported_reason=unsupported_reason,
    )


def _clamp_confidence(raw: object) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = [
    "MentorIntakeApplyMode",
    "MentorIntakeDecision",
    "MentorIntakeKind",
    "MentorIntakeRequest",
    "classify_mentor_intake",
]
