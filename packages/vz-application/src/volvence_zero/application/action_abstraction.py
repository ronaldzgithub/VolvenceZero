"""Background-slow semantic abstraction for latent action families.

The temporal family id remains opaque.  This module is a CaseMemory-owner
collaborator that may propose a reusable semantic schema only after multiple
independent, schema-free experiences share one stable family identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Protocol

_MIN_EVIDENCE_COUNT = 2
_MIN_CANDIDATE_CONFIDENCE = 0.75
_RESOURCE_ROOT = Path(__file__).resolve().parent


class ActionAbstractionTextProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = ...,
        temperature: float = ...,
    ) -> str: ...


@dataclass(frozen=True)
class ActionApplicabilityDecision:
    """Structured readout for one promoted schema in the current situation."""

    applicable: bool
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "ActionApplicabilityDecision confidence must be in [0, 1]."
            )
        if not self.rationale.strip():
            raise ValueError(
                "ActionApplicabilityDecision rationale must be non-empty."
            )


class ActionApplicabilityEvaluator(Protocol):
    """CaseMemory collaborator; never an action or learning owner."""

    def evaluate(
        self,
        *,
        query_text: str,
        schema_id: str,
        applicability_conditions: tuple[str, ...],
        risk_markers: tuple[str, ...],
    ) -> ActionApplicabilityDecision | None: ...


class NoOpActionApplicabilityEvaluator:
    """Fail closed when no structured semantic evaluator is wired."""

    def evaluate(
        self,
        *,
        query_text: str,
        schema_id: str,
        applicability_conditions: tuple[str, ...],
        risk_markers: tuple[str, ...],
    ) -> ActionApplicabilityDecision | None:
        del query_text, schema_id, applicability_conditions, risk_markers
        return None


class LLMActionApplicabilityEvaluator:
    """Structured applicability evaluator backed by an injected provider."""

    def __init__(
        self,
        *,
        provider: ActionAbstractionTextProvider,
        max_new_tokens: int = 192,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        self._provider = provider
        self._max_new_tokens = max_new_tokens

    def evaluate(
        self,
        *,
        query_text: str,
        schema_id: str,
        applicability_conditions: tuple[str, ...],
        risk_markers: tuple[str, ...],
    ) -> ActionApplicabilityDecision | None:
        if (
            not query_text.strip()
            or not schema_id.strip()
            or not applicability_conditions
        ):
            return None
        prompt = _load_action_applicability_prompt_template().format(
            evidence_json=json.dumps(
                {
                    "current_situation_and_request": query_text,
                    "candidate_schema_id": schema_id,
                    "required_applicability_conditions": (
                        applicability_conditions
                    ),
                    "risk_markers": risk_markers,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            output_schema=_load_action_applicability_schema_text(),
        )
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        return _parse_action_applicability_decision(raw)


@dataclass(frozen=True)
class ActionAbstractionExperience:
    """Normalized schema-free evidence accepted by the abstraction owner."""

    outcome_id: str
    action_id: str
    action_family_id: str
    action_family_version: int
    situation_statement: str
    action_statement: str
    evidence: tuple[str, ...]
    confidence: float
    controller_code_digest: tuple[float, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("outcome_id", self.outcome_id),
            ("action_id", self.action_id),
            ("action_family_id", self.action_family_id),
            ("situation_statement", self.situation_statement),
            ("action_statement", self.action_statement),
        ):
            if not value.strip():
                raise ValueError(
                    f"ActionAbstractionExperience {name} must be non-empty."
                )
        if self.action_family_version <= 0:
            raise ValueError(
                "ActionAbstractionExperience action_family_version must be > 0."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "ActionAbstractionExperience confidence must be in [0, 1]."
            )


@dataclass(frozen=True)
class LearnedActionSchemaCandidate:
    """Typed, unpromoted semantic compression of one latent family."""

    schema_id: str
    action_family_id: str
    action_family_version: int
    applicability_conditions: tuple[str, ...]
    action_steps: tuple[str, ...]
    source_outcome_ids: tuple[str, ...]
    confidence: float
    description: str

    def __post_init__(self) -> None:
        for name, value in (
            ("schema_id", self.schema_id),
            ("action_family_id", self.action_family_id),
            ("description", self.description),
        ):
            if not value.strip():
                raise ValueError(
                    f"LearnedActionSchemaCandidate {name} must be non-empty."
                )
        if self.action_family_version <= 0:
            raise ValueError(
                "LearnedActionSchemaCandidate action_family_version must be > 0."
            )
        _require_non_empty_unique(
            "applicability_conditions",
            self.applicability_conditions,
        )
        _require_non_empty_unique("action_steps", self.action_steps)
        _require_non_empty_unique("source_outcome_ids", self.source_outcome_ids)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "LearnedActionSchemaCandidate confidence must be in [0, 1]."
            )


class ActionAbstractionDecoder(Protocol):
    """Background semantic decoder; never an Internal-RL signal owner."""

    def decode(
        self,
        *,
        family_id: str,
        family_version: int,
        experiences: tuple[ActionAbstractionExperience, ...],
    ) -> LearnedActionSchemaCandidate | None: ...


class NoOpActionAbstractionDecoder:
    """Fail-closed default when no structured background decoder is wired."""

    def decode(
        self,
        *,
        family_id: str,
        family_version: int,
        experiences: tuple[ActionAbstractionExperience, ...],
    ) -> LearnedActionSchemaCandidate | None:
        del family_id, family_version, experiences
        return None


class LLMActionAbstractionDecoder:
    """Structured background decoder backed by an injected text provider."""

    def __init__(
        self,
        *,
        provider: ActionAbstractionTextProvider,
        max_new_tokens: int = 384,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        self._provider = provider
        self._max_new_tokens = max_new_tokens

    def decode(
        self,
        *,
        family_id: str,
        family_version: int,
        experiences: tuple[ActionAbstractionExperience, ...],
    ) -> LearnedActionSchemaCandidate | None:
        evidence_payload = tuple(
            {
                "outcome_id": experience.outcome_id,
                "situation": experience.situation_statement,
                "executed_action": experience.action_statement,
            }
            for experience in experiences
        )
        prompt = _load_prompt_template().format(
            family_id=family_id,
            family_version=family_version,
            evidence_json=json.dumps(
                evidence_payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
            output_schema=_load_output_schema_text(),
        )
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        return _parse_candidate(raw)


class ActionAbstractionOwner:
    """Validate evidence closure and admit one unpromoted candidate."""

    def propose(
        self,
        *,
        experiences: tuple[ActionAbstractionExperience, ...],
        decoder: ActionAbstractionDecoder,
    ) -> LearnedActionSchemaCandidate | None:
        eligible = tuple(
            experience
            for experience in experiences
            if (
                experience.action_family_id
                and experience.action_family_version > 0
                and experience.situation_statement.strip()
            )
        )
        if len(eligible) < _MIN_EVIDENCE_COUNT:
            return None
        family_ids = {experience.action_family_id for experience in eligible}
        if len(family_ids) != 1:
            return None
        if len({item.outcome_id for item in eligible}) != len(eligible):
            return None
        if len({item.situation_statement for item in eligible}) < 2:
            return None
        family_id = next(iter(family_ids))
        # ``action_family_version`` is the temporal owner's global family-bank
        # revision, not a family incarnation id.  The opaque ``family_id`` is
        # the stable identity; use the newest observed bank revision for audit
        # and candidate publication without requiring byte-equal revisions
        # across experiences collected at different times.
        family_version = max(
            experience.action_family_version for experience in eligible
        )
        candidate = decoder.decode(
            family_id=family_id,
            family_version=family_version,
            experiences=eligible,
        )
        if candidate is None:
            return None
        if (
            candidate.action_family_id != family_id
            or candidate.action_family_version != family_version
            or set(candidate.source_outcome_ids)
            != {item.outcome_id for item in eligible}
            or candidate.confidence < _MIN_CANDIDATE_CONFIDENCE
        ):
            return None
        episode_sentences = {
            text
            for experience in eligible
            for text in (
                experience.situation_statement,
                experience.action_statement,
            )
        }
        if any(
            item in episode_sentences
            for item in (
                *candidate.applicability_conditions,
                *candidate.action_steps,
            )
        ):
            return None
        return candidate


def merge_action_abstraction_experiences(
    *groups: tuple[ActionAbstractionExperience, ...],
) -> tuple[ActionAbstractionExperience, ...]:
    """Deduplicate identical outcomes and reject contradictory owner state."""

    by_outcome: dict[str, ActionAbstractionExperience] = {}
    for experience in (item for group in groups for item in group):
        existing = by_outcome.get(experience.outcome_id)
        if existing is not None and existing != experience:
            raise ValueError(
                "Conflicting action-abstraction evidence for "
                f"outcome_id={experience.outcome_id!r}."
            )
        by_outcome[experience.outcome_id] = experience
    return tuple(by_outcome[outcome_id] for outcome_id in sorted(by_outcome))


def _load_prompt_template() -> str:
    return (
        _RESOURCE_ROOT / "prompts" / "action_abstraction.md"
    ).read_text(encoding="utf-8")


def _load_output_schema_text() -> str:
    return (
        _RESOURCE_ROOT / "schemas" / "action_abstraction.schema.json"
    ).read_text(encoding="utf-8")


def _load_action_applicability_prompt_template() -> str:
    return (
        _RESOURCE_ROOT / "prompts" / "action_applicability.md"
    ).read_text(encoding="utf-8")


def _load_action_applicability_schema_text() -> str:
    return (
        _RESOURCE_ROOT / "schemas" / "action_applicability.schema.json"
    ).read_text(encoding="utf-8")


def _parse_action_applicability_decision(
    text: str,
) -> ActionApplicabilityDecision | None:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"applicable", "confidence", "rationale"}:
        return None
    if not isinstance(payload["applicable"], bool):
        return None
    if (
        isinstance(payload["confidence"], bool)
        or not isinstance(payload["confidence"], (int, float))
        or not isinstance(payload["rationale"], str)
    ):
        return None
    try:
        return ActionApplicabilityDecision(
            applicable=payload["applicable"],
            confidence=float(payload["confidence"]),
            rationale=payload["rationale"],
        )
    except (TypeError, ValueError):
        return None


def _parse_candidate(text: str) -> LearnedActionSchemaCandidate | None:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    expected_keys = {
        "schema_id",
        "action_family_id",
        "action_family_version",
        "applicability_conditions",
        "action_steps",
        "source_outcome_ids",
        "confidence",
        "description",
    }
    if set(payload) != expected_keys:
        return None
    try:
        return LearnedActionSchemaCandidate(
            schema_id=str(payload["schema_id"]),
            action_family_id=str(payload["action_family_id"]),
            action_family_version=int(payload["action_family_version"]),
            applicability_conditions=tuple(
                str(item) for item in payload["applicability_conditions"]
            ),
            action_steps=tuple(str(item) for item in payload["action_steps"]),
            source_outcome_ids=tuple(
                str(item) for item in payload["source_outcome_ids"]
            ),
            confidence=float(payload["confidence"]),
            description=str(payload["description"]),
        )
    except (TypeError, ValueError):
        return None


def _require_non_empty_unique(
    name: str,
    values: tuple[str, ...],
) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values.")


__all__ = [
    "ActionApplicabilityDecision",
    "ActionApplicabilityEvaluator",
    "ActionAbstractionDecoder",
    "ActionAbstractionExperience",
    "ActionAbstractionOwner",
    "ActionAbstractionTextProvider",
    "LLMActionApplicabilityEvaluator",
    "LearnedActionSchemaCandidate",
    "LLMActionAbstractionDecoder",
    "merge_action_abstraction_experiences",
    "NoOpActionApplicabilityEvaluator",
    "NoOpActionAbstractionDecoder",
]
