"""Frozen semantic relationship-condition reader for preference forecasts.

The reader names an abstract condition from the current public observation,
then conditions the existing preference-action forecast on owner-persisted
experiences classified into the same semantic prototype space.  It never sees
evaluator labels, expected actions, future outcomes, PE, credit, or judge
scores.  The preference owner remains the sole publisher of the resulting
forecast and named readout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Protocol

from volvence_zero.social import (
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
)
from volvence_zero.social_cognition import (
    OtherMindRecord,
    PreferenceActionOutcomeEvidence,
    RelationshipConditionReadout,
)

from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)


RELATIONSHIP_CONDITION_READER_SCHEMA_VERSION = (
    "relationship-condition-reader-artifact.v1"
)


class RelationshipTextEmbedder(Protocol):
    """Frozen text encoder injected by the evidence/runtime composition root."""

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class RelationshipConditionPrototype:
    label: str
    summary: str

    def __post_init__(self) -> None:
        _require_text(self.label, "prototype label")
        _require_text(self.summary, "prototype summary")

    @property
    def summary_sha256(self) -> str:
        return hashlib.sha256(self.summary.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RelationshipConditionReaderArtifact:
    """Content-addressed relation concept vocabulary bound to frozen weights."""

    embedding_model_id: str
    embedding_weights_sha256: str
    prototypes: tuple[RelationshipConditionPrototype, ...]
    softmax_temperature: float = 0.05
    semantic_similarity: str = "cosine"
    schema_version: str = RELATIONSHIP_CONDITION_READER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_CONDITION_READER_SCHEMA_VERSION:
            raise ValueError("relationship condition reader schema mismatch")
        _require_text(self.embedding_model_id, "embedding_model_id")
        _require_sha256(
            self.embedding_weights_sha256,
            "embedding_weights_sha256",
        )
        if self.semantic_similarity != "cosine":
            raise ValueError("relationship condition reader requires cosine")
        if len(self.prototypes) < 2:
            raise ValueError("relationship condition reader requires at least two prototypes")
        labels = tuple(item.label for item in self.prototypes)
        if len(set(labels)) != len(labels):
            raise ValueError("relationship condition prototype labels must be unique")
        summaries = tuple(item.summary for item in self.prototypes)
        if len(set(summaries)) != len(summaries):
            raise ValueError("relationship condition prototype summaries must be unique")
        if (
            not math.isfinite(self.softmax_temperature)
            or not 0.001 <= self.softmax_temperature <= 1.0
        ):
            raise ValueError("softmax_temperature must be finite and in [0.001, 1]")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "embedding_model_id": self.embedding_model_id,
            "embedding_weights_sha256": self.embedding_weights_sha256,
            "semantic_similarity": self.semantic_similarity,
            "softmax_temperature": self.softmax_temperature,
            "prototypes": [
                {
                    "label": item.label,
                    "summary": item.summary,
                    "summary_sha256": item.summary_sha256,
                }
                for item in self.prototypes
            ],
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self.to_payload())


class PrototypeRelationshipPreferenceForecastRuntime:
    """Condition-labelled semantic collaborator of ``preference_about_other``."""

    def __init__(
        self,
        *,
        artifact: RelationshipConditionReaderArtifact,
        embedder: RelationshipTextEmbedder,
        prior_count: float = 1.0,
        evidence_weight: float = 4.0,
    ) -> None:
        self._artifact = artifact
        self._embedder = embedder
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._readouts: dict[str, RelationshipConditionReadout] = {}
        self._prototype_vectors = tuple(
            self._vector(item.summary) for item in artifact.prototypes
        )
        widths = {len(vector) for vector in self._prototype_vectors}
        if len(widths) != 1 or next(iter(widths)) < 2:
            raise ValueError(
                "relationship condition embedder must return one stable non-trivial width"
            )
        self._delegate = BoundedRelationshipPreferenceForecastRuntime(
            similarity=self._condition_similarity,
            prior_count=prior_count,
            evidence_weight=evidence_weight,
        )
        self.runtime_id = (
            "relationship-p2-prototype-condition-forecast.v1:"
            f"{artifact.artifact_id}"
        )

    @property
    def artifact(self) -> RelationshipConditionReaderArtifact:
        return self._artifact

    def read_condition(self, text: str) -> RelationshipConditionReadout:
        _require_text(text, "relationship condition source text")
        cached = self._readouts.get(text)
        if cached is not None:
            return cached
        vector = self._vector(text)
        scores = tuple(
            (
                prototype.label,
                _cosine(vector, prototype_vector),
            )
            for prototype, prototype_vector in zip(
                self._artifact.prototypes,
                self._prototype_vectors,
                strict=True,
            )
        )
        ordered = sorted(scores, key=lambda item: item[1], reverse=True)
        top_label, top_score = ordered[0]
        second_score = ordered[1][1]
        normalized_margin = min(
            1.0,
            max(0.0, (top_score - second_score) / 2.0),
        )
        maximum = max(score for _, score in scores)
        exponentials = tuple(
            math.exp(
                (score - maximum) / self._artifact.softmax_temperature
            )
            for _, score in scores
        )
        top_index = next(
            index for index, (label, _) in enumerate(scores) if label == top_label
        )
        confidence = exponentials[top_index] / math.fsum(exponentials)
        readout = RelationshipConditionReadout(
            condition_label=top_label,
            confidence=confidence,
            normalized_margin=normalized_margin,
            candidate_scores=scores,
            reader_artifact_id=self._artifact.artifact_id,
            source_observation_sha256=hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        )
        self._readouts[text] = readout
        return readout

    def _condition_similarity(self, left: str, right: str) -> float:
        left_readout = self.read_condition(left)
        right_readout = self.read_condition(right)
        if left_readout.condition_label != right_readout.condition_label:
            return 0.0
        return min(left_readout.confidence, right_readout.confidence)

    def _vector(self, text: str) -> tuple[float, ...]:
        cached = self._vectors.get(text)
        if cached is not None:
            return cached
        raw = self._embedder.embed(text)
        if (
            not raw
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw
            )
        ):
            raise ValueError(
                "relationship condition embedder returned an invalid vector"
            )
        vector = tuple(float(value) for value in raw)
        if self._vectors and len(vector) != len(next(iter(self._vectors.values()))):
            raise ValueError("relationship condition embedding width drift")
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm <= 1e-12:
            raise ValueError("relationship condition embedding norm must be positive")
        normalized = tuple(value / norm for value in vector)
        self._vectors[text] = normalized
        return normalized

    def propose(
        self,
        *,
        request: PreferenceActionForecastRequest,
        records: tuple[OtherMindRecord, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> PreferenceActionForecastProposal | None:
        proposal = self._delegate.propose(
            request=request,
            records=records,
            action_outcomes=action_outcomes,
        )
        if proposal is None:
            return None
        current = self.read_condition(request.current_observation)
        return replace(
            proposal,
            evidence=tuple(
                dict.fromkeys(
                    (
                        *proposal.evidence,
                        f"condition_reader:{self._artifact.artifact_id}",
                        f"condition_label:{current.condition_label}",
                    )
                )
            ),
            condition_readout=current,
        )


def _cosine(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError("relationship condition cosine width mismatch")
    value = math.fsum(
        left_item * right_item
        for left_item, right_item in zip(left, right, strict=True)
    )
    return max(-1.0, min(1.0, value))


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


__all__ = [
    "PrototypeRelationshipPreferenceForecastRuntime",
    "RELATIONSHIP_CONDITION_READER_SCHEMA_VERSION",
    "RelationshipConditionPrototype",
    "RelationshipConditionReaderArtifact",
    "RelationshipTextEmbedder",
]
