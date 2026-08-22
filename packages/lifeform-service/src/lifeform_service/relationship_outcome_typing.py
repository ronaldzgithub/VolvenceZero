"""Qualified structured-LLM typing for real-user relationship outcomes.

This module owns only the service-side conversion from one explicit user
follow-up observation to a frozen five-way result.  It never writes PE,
credit, memory, or runtime snapshots.  The P4 controller decides whether the
result is eligible for the canonical ``dialogue_external_outcome`` channel.
"""

from __future__ import annotations

import importlib.resources as resources
import json
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from lifeform_core import LlmJsonClient


RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION = (
    "relationship-outcome-typing-result.v1"
)
RELATIONSHIP_OUTCOME_UNKNOWN = "unknown"
_RELATIONSHIP_OUTCOME_VALUES = frozenset(
    {
        "helped",
        "felt_heard",
        "missed",
        "over_directive",
        RELATIONSHIP_OUTCOME_UNKNOWN,
    }
)


class RelationshipOutcomeEvidenceBasis(str, Enum):
    EXPLICIT_REPORT = "explicit_report"
    BEHAVIORAL_CONSEQUENCE = "behavioral_consequence"
    MIXED_OR_AMBIGUOUS = "mixed_or_ambiguous"


@dataclass(frozen=True)
class RelationshipOutcomeTypingResult:
    outcome_kind: str
    confidence: float
    evidence_basis: RelationshipOutcomeEvidenceBasis
    needs_human_review: bool
    runtime_id: str
    schema_version: str = RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outcome_kind not in _RELATIONSHIP_OUTCOME_VALUES:
            raise ValueError("relationship outcome typing result has unknown kind")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("relationship outcome typing confidence must be in [0, 1]")
        if not self.runtime_id:
            raise ValueError("relationship outcome typing runtime_id must be non-empty")
        if self.schema_version != RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION:
            raise ValueError("relationship outcome typing result schema mismatch")
        if (
            self.outcome_kind == RELATIONSHIP_OUTCOME_UNKNOWN
            and not self.needs_human_review
        ):
            raise ValueError("unknown relationship outcomes must request human review")


class RelationshipOutcomeTypingRuntime(Protocol):
    @property
    def runtime_id(self) -> str: ...

    @property
    def schema_version(self) -> str: ...

    def classify(self, outcome_text: str) -> RelationshipOutcomeTypingResult: ...


class LlmStructuredRelationshipOutcomeTyper:
    """Schema-strict LLM adapter; no keyword or regular-expression path."""

    def __init__(self, *, client: LlmJsonClient, runtime_id: str) -> None:
        if not runtime_id:
            raise ValueError("relationship outcome typer runtime_id must be non-empty")
        package = resources.files("lifeform_service")
        self._system_prompt = (
            package / "prompts" / "relationship_outcome_typing_system.txt"
        ).read_text(encoding="utf-8")
        schema_text = (
            package / "schemas" / "relationship_outcome_typing_v1.json"
        ).read_text(encoding="utf-8")
        try:
            schema = json.loads(schema_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - packaged asset
            raise RuntimeError("relationship outcome typing schema is invalid") from exc
        if not isinstance(schema, dict):  # pragma: no cover - packaged asset
            raise RuntimeError("relationship outcome typing schema must be an object")
        if schema.get("$id") != RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION:
            raise RuntimeError("relationship outcome typing schema id mismatch")
        self._schema = schema
        self._client = client
        self._runtime_id = runtime_id

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def schema_version(self) -> str:
        return RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION

    def classify(self, outcome_text: str) -> RelationshipOutcomeTypingResult:
        if not isinstance(outcome_text, str) or not outcome_text.strip():
            raise ValueError("outcome_text must be a non-empty string")
        response = self._client.complete_json(
            system_prompt=self._system_prompt,
            user_prompt=json.dumps(
                {
                    "output_schema": self._schema,
                    "outcome_text": outcome_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        expected_keys = {
            "schema_version",
            "outcome_kind",
            "confidence",
            "evidence_basis",
            "needs_human_review",
        }
        if set(response) != expected_keys:
            raise ValueError(
                "relationship outcome typer response does not match frozen schema"
            )
        schema_version = response["schema_version"]
        outcome_kind = response["outcome_kind"]
        confidence = response["confidence"]
        evidence_basis = response["evidence_basis"]
        needs_human_review = response["needs_human_review"]
        if not isinstance(schema_version, str):
            raise ValueError("typing schema_version must be a string")
        if not isinstance(outcome_kind, str):
            raise ValueError("typing outcome_kind must be a string")
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("typing confidence must be numeric")
        if not isinstance(evidence_basis, str):
            raise ValueError("typing evidence_basis must be a string")
        if not isinstance(needs_human_review, bool):
            raise ValueError("typing needs_human_review must be a boolean")
        return RelationshipOutcomeTypingResult(
            outcome_kind=outcome_kind,
            confidence=float(confidence),
            evidence_basis=RelationshipOutcomeEvidenceBasis(evidence_basis),
            needs_human_review=needs_human_review,
            runtime_id=self._runtime_id,
            schema_version=schema_version,
        )


__all__ = [
    "RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION",
    "RELATIONSHIP_OUTCOME_UNKNOWN",
    "LlmStructuredRelationshipOutcomeTyper",
    "RelationshipOutcomeEvidenceBasis",
    "RelationshipOutcomeTypingResult",
    "RelationshipOutcomeTypingRuntime",
]
