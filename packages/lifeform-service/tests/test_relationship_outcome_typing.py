from __future__ import annotations

import pytest

from lifeform_service.relationship_outcome_typing import (
    RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION,
    LlmStructuredRelationshipOutcomeTyper,
    RelationshipOutcomeEvidenceBasis,
)


class _JsonClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        return dict(self.response)


def _response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION,
        "outcome_kind": "felt_heard",
        "confidence": 0.91,
        "evidence_basis": "explicit_report",
        "needs_human_review": False,
    }
    payload.update(overrides)
    return payload


def test_structured_llm_typer_uses_frozen_schema_without_text_rules() -> None:
    client = _JsonClient(_response())
    typer = LlmStructuredRelationshipOutcomeTyper(
        client=client,
        runtime_id="qualified-typing-runtime",
    )

    result = typer.classify("I felt understood even though nothing was solved.")

    assert result.outcome_kind == "felt_heard"
    assert result.confidence == 0.91
    assert result.evidence_basis is RelationshipOutcomeEvidenceBasis.EXPLICIT_REPORT
    assert result.runtime_id == "qualified-typing-runtime"
    assert len(client.calls) == 1
    system_prompt, user_prompt = client.calls[0]
    assert "keywords" in system_prompt
    assert "I felt understood" not in system_prompt
    assert "I felt understood" in user_prompt
    assert RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION in user_prompt


def test_structured_llm_typer_fails_closed_on_schema_drift() -> None:
    client = _JsonClient(_response(extra_field="not allowed"))
    typer = LlmStructuredRelationshipOutcomeTyper(
        client=client,
        runtime_id="qualified-typing-runtime",
    )

    with pytest.raises(ValueError, match="frozen schema"):
        typer.classify("An explicit follow-up.")


def test_unknown_typing_requires_human_review() -> None:
    client = _JsonClient(
        _response(
            outcome_kind="unknown",
            confidence=0.3,
            evidence_basis="mixed_or_ambiguous",
            needs_human_review=False,
        )
    )
    typer = LlmStructuredRelationshipOutcomeTyper(
        client=client,
        runtime_id="qualified-typing-runtime",
    )

    with pytest.raises(ValueError, match="must request human review"):
        typer.classify("It was complicated.")
