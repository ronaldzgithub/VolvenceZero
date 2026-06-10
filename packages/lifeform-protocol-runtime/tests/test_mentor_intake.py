from __future__ import annotations

import pytest

from lifeform_protocol_runtime import (
    MentorIntakeKind,
    MentorIntakeRequest,
    classify_mentor_intake,
)


class _ClassifierClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_classify_mentor_intake_returns_typed_decision():
    client = _ClassifierClient(
        {
            "intake_kind": "protocol",
            "routed_owner": "protocol_registry",
            "confidence": 0.91,
            "reason": "Changes the next-turn action posture.",
            "actionable_summary": "Clarify boundary before giving options.",
        }
    )

    decision = classify_mentor_intake(
        MentorIntakeRequest(
            guidance="When planning, confirm boundaries before advice.",
            mentor_id="mentor:alice",
        ),
        llm_client=client,
    )

    assert decision.intake_kind is MentorIntakeKind.PROTOCOL
    assert decision.routed_owner == "protocol_registry"
    assert decision.confidence == 0.91
    assert client.calls


def test_classify_mentor_intake_rejects_unknown_kind():
    client = _ClassifierClient(
        {
            "intake_kind": "prompt_hint",
            "confidence": 0.5,
            "reason": "invalid",
            "actionable_summary": "invalid",
        }
    )

    with pytest.raises(ValueError, match="unknown intake_kind"):
        classify_mentor_intake(
            MentorIntakeRequest(guidance="Do something different."),
            llm_client=client,
        )


def test_non_protocol_kinds_are_explicitly_unsupported_for_live_apply():
    client = _ClassifierClient(
        {
            "intake_kind": "experience",
            "confidence": 0.8,
            "reason": "This is a retrospective outcome note.",
            "actionable_summary": "Record the failed outcome.",
        }
    )

    decision = classify_mentor_intake(
        MentorIntakeRequest(guidance="The last dense reply caused dropout."),
        llm_client=client,
    )

    assert decision.intake_kind is MentorIntakeKind.EXPERIENCE
    assert decision.routed_owner == "experience_consolidation"
    assert "not live-applied" in decision.unsupported_reason
