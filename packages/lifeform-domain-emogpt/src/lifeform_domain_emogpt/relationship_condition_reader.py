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
RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION = (
    "relationship-condition-reader-artifact.v2"
)
RELATIONSHIP_CONDITION_LINEAR_SOLVER = "unit_normalized_class_centroid_linear"
RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION = (
    "relationship-condition-centroid-solver.v1"
)


class RelationshipTextEmbedder(Protocol):
    """Frozen text encoder injected by the evidence/runtime composition root."""

    def embed(self, text: str) -> tuple[float, ...]: ...


class FrozenRelationshipTextEmbedder(Protocol):
    """Identity-bearing frozen encoder admitted by the v2 reader runtime."""

    model_source: str
    model_revision: str | None
    weights_sha256: str | None
    sentence_transformers_version: str | None

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


@dataclass(frozen=True)
class LabeledRelationshipConditionEmbeddingRow:
    """Condition-only offline input; it intentionally carries no outcome signal."""

    example_id: str
    condition_label: str
    embedding_hex: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.example_id, "example_id")
        _require_text(self.condition_label, "condition_label")
        if not isinstance(self.embedding_hex, tuple) or len(self.embedding_hex) < 2:
            raise ValueError("embedding_hex must be a tuple with width at least two")
        values = tuple(
            _decode_canonical_float_hex(value, "embedding_hex")
            for value in self.embedding_hex
        )
        if math.sqrt(math.fsum(value * value for value in values)) <= 1e-12:
            raise ValueError("offline relationship embedding norm must be positive")

    @property
    def embedding(self) -> tuple[float, ...]:
        return tuple(
            _decode_canonical_float_hex(value, "embedding_hex")
            for value in self.embedding_hex
        )


@dataclass(frozen=True)
class RelationshipConditionLinearClassParameters:
    """Complete immutable parameters for one centroid-linear class."""

    label: str
    example_count: int
    example_ids_sha256: str
    centroid_hex: tuple[str, ...]
    coefficient_hex: tuple[str, ...]
    bias_hex: str

    def __post_init__(self) -> None:
        _require_text(self.label, "linear class label")
        if (
            isinstance(self.example_count, bool)
            or not isinstance(self.example_count, int)
            or self.example_count < 1
        ):
            raise ValueError("linear class example_count must be a positive integer")
        _require_sha256(self.example_ids_sha256, "example_ids_sha256")
        if not isinstance(self.centroid_hex, tuple) or not self.centroid_hex:
            raise ValueError("centroid_hex must be a non-empty tuple")
        if not isinstance(self.coefficient_hex, tuple) or not self.coefficient_hex:
            raise ValueError("coefficient_hex must be a non-empty tuple")
        centroid = tuple(
            _decode_canonical_float_hex(value, "centroid_hex")
            for value in self.centroid_hex
        )
        coefficient = tuple(
            _decode_canonical_float_hex(value, "coefficient_hex")
            for value in self.coefficient_hex
        )
        if len(centroid) != len(coefficient):
            raise ValueError("centroid and coefficient widths must match")
        if self.centroid_hex != self.coefficient_hex:
            raise ValueError("centroid-linear coefficients must equal the centroid")
        if abs(math.sqrt(math.fsum(value * value for value in centroid)) - 1.0) > 1e-12:
            raise ValueError("centroid-linear class centroid must have unit norm")
        bias = _decode_canonical_float_hex(self.bias_hex, "bias_hex")
        if bias != 0.0 or self.bias_hex != _canonical_float_hex(0.0):
            raise ValueError("centroid-linear class bias must be canonical positive zero")

    def to_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "example_count": self.example_count,
            "example_ids_sha256": self.example_ids_sha256,
            "centroid_hex": list(self.centroid_hex),
            "coefficient_hex": list(self.coefficient_hex),
            "bias_hex": self.bias_hex,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> RelationshipConditionLinearClassParameters:
        if not isinstance(payload, dict):
            raise ValueError("linear class parameters payload must be an object")
        _require_exact_payload_keys(
            payload,
            {
                "label",
                "example_count",
                "example_ids_sha256",
                "centroid_hex",
                "coefficient_hex",
                "bias_hex",
            },
            "linear class parameters payload",
        )
        centroid_hex = payload["centroid_hex"]
        coefficient_hex = payload["coefficient_hex"]
        if not isinstance(centroid_hex, list):
            raise ValueError("linear class centroid_hex must be a JSON array")
        if not isinstance(coefficient_hex, list):
            raise ValueError("linear class coefficient_hex must be a JSON array")
        return cls(
            label=payload["label"],
            example_count=payload["example_count"],
            example_ids_sha256=payload["example_ids_sha256"],
            centroid_hex=tuple(centroid_hex),
            coefficient_hex=tuple(coefficient_hex),
            bias_hex=payload["bias_hex"],
        )


@dataclass(frozen=True)
class FrozenLinearRelationshipConditionReaderArtifact:
    """Content-addressed, fully frozen v2 condition-reader mechanism."""

    embedding_model_id: str
    embedding_model_revision: str
    embedding_weights_sha256: str
    embedding_runtime_version: str
    embedding_width: int
    labels: tuple[str, ...]
    condition_training_corpus_artifact_id: str
    condition_training_corpus_raw_sha256: str
    group_split_artifact_id: str
    class_parameters: tuple[RelationshipConditionLinearClassParameters, ...]
    solver: str = RELATIONSHIP_CONDITION_LINEAR_SOLVER
    solver_version: str = RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION
    schema_version: str = RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION:
            raise ValueError("linear relationship condition reader schema mismatch")
        _require_text(self.embedding_model_id, "embedding_model_id")
        _require_text(self.embedding_model_revision, "embedding_model_revision")
        _require_sha256(
            self.embedding_weights_sha256,
            "embedding_weights_sha256",
        )
        _require_text(self.embedding_runtime_version, "embedding_runtime_version")
        _require_embedding_width(self.embedding_width)
        _require_strict_ordered_labels(self.labels)
        _require_sha256(
            self.condition_training_corpus_artifact_id,
            "condition_training_corpus_artifact_id",
        )
        _require_sha256(
            self.condition_training_corpus_raw_sha256,
            "condition_training_corpus_raw_sha256",
        )
        _require_sha256(self.group_split_artifact_id, "group_split_artifact_id")
        if self.solver != RELATIONSHIP_CONDITION_LINEAR_SOLVER:
            raise ValueError("linear relationship condition reader solver mismatch")
        if self.solver_version != RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION:
            raise ValueError(
                "linear relationship condition reader solver version mismatch"
            )
        if not isinstance(self.class_parameters, tuple):
            raise ValueError("class_parameters must be a tuple")
        parameter_labels = tuple(item.label for item in self.class_parameters)
        if parameter_labels != self.labels:
            raise ValueError(
                "linear relationship condition parameter labels must match labels exactly"
            )
        if any(
            len(item.centroid_hex) != self.embedding_width
            or len(item.coefficient_hex) != self.embedding_width
            for item in self.class_parameters
        ):
            raise ValueError("linear relationship condition parameter width drift")

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "embedding_model_id": self.embedding_model_id,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_weights_sha256": self.embedding_weights_sha256,
            "embedding_runtime_version": self.embedding_runtime_version,
            "embedding_width": self.embedding_width,
            "labels": list(self.labels),
            "condition_training_corpus_artifact_id": (
                self.condition_training_corpus_artifact_id
            ),
            "condition_training_corpus_raw_sha256": (
                self.condition_training_corpus_raw_sha256
            ),
            "group_split_artifact_id": self.group_split_artifact_id,
            "solver": self.solver,
            "solver_version": self.solver_version,
            "class_parameters": [
                item.to_payload() for item in self.class_parameters
            ],
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self._content_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            **self._content_payload(),
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> FrozenLinearRelationshipConditionReaderArtifact:
        if not isinstance(payload, dict):
            raise ValueError("linear relationship reader payload must be an object")
        _require_exact_payload_keys(
            payload,
            {
                "schema_version",
                "embedding_model_id",
                "embedding_model_revision",
                "embedding_weights_sha256",
                "embedding_runtime_version",
                "embedding_width",
                "labels",
                "condition_training_corpus_artifact_id",
                "condition_training_corpus_raw_sha256",
                "group_split_artifact_id",
                "solver",
                "solver_version",
                "class_parameters",
                "artifact_id",
            },
            "linear relationship reader payload",
        )
        labels = payload["labels"]
        parameters = payload["class_parameters"]
        if not isinstance(labels, list):
            raise ValueError("linear relationship reader labels must be a JSON array")
        if not isinstance(parameters, list):
            raise ValueError(
                "linear relationship reader class_parameters must be a JSON array"
            )
        supplied_artifact_id = payload["artifact_id"]
        _require_sha256(supplied_artifact_id, "artifact_id")
        artifact = cls(
            embedding_model_id=payload["embedding_model_id"],
            embedding_model_revision=payload["embedding_model_revision"],
            embedding_weights_sha256=payload["embedding_weights_sha256"],
            embedding_runtime_version=payload["embedding_runtime_version"],
            embedding_width=payload["embedding_width"],
            labels=tuple(labels),
            condition_training_corpus_artifact_id=(
                payload["condition_training_corpus_artifact_id"]
            ),
            condition_training_corpus_raw_sha256=(
                payload["condition_training_corpus_raw_sha256"]
            ),
            group_split_artifact_id=payload["group_split_artifact_id"],
            class_parameters=tuple(
                RelationshipConditionLinearClassParameters.from_payload(item)
                for item in parameters
            ),
            solver=payload["solver"],
            solver_version=payload["solver_version"],
            schema_version=payload["schema_version"],
        )
        if supplied_artifact_id != artifact.artifact_id:
            raise ValueError("linear relationship reader artifact_id mismatch")
        return artifact

    def to_json(self) -> str:
        return _canonical_json(self.to_payload()) + "\n"

    def to_json_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(
        cls,
        raw: str | bytes,
    ) -> FrozenLinearRelationshipConditionReaderArtifact:
        raw_bytes = _require_utf8_json_bytes(raw)
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "linear relationship reader JSON must be exact UTF-8"
            ) from error
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError("invalid linear relationship reader JSON") from error
        artifact = cls.from_payload(payload)
        if raw_bytes != artifact.to_json_bytes():
            raise ValueError(
                "linear relationship reader JSON must use canonical UTF-8 bytes"
            )
        return artifact


def build_frozen_linear_relationship_condition_reader_artifact(
    *,
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_weights_sha256: str,
    embedding_runtime_version: str,
    embedding_width: int,
    labels: tuple[str, ...],
    condition_training_corpus_artifact_id: str,
    condition_training_corpus_raw_sha256: str,
    group_split_artifact_id: str,
    rows: tuple[LabeledRelationshipConditionEmbeddingRow, ...],
) -> FrozenLinearRelationshipConditionReaderArtifact:
    """Build v2 parameters offline from condition labels and frozen embeddings only."""

    _require_embedding_width(embedding_width)
    _require_strict_ordered_labels(labels)
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("offline relationship embedding rows must be a non-empty tuple")
    if any(
        not isinstance(row, LabeledRelationshipConditionEmbeddingRow)
        for row in rows
    ):
        raise ValueError("offline relationship embedding rows have invalid types")
    example_ids = tuple(row.example_id for row in rows)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("offline relationship embedding example ids must be unique")
    row_labels = {row.condition_label for row in rows}
    if row_labels != set(labels):
        raise ValueError("offline relationship embedding labels must match exactly")
    if any(len(row.embedding_hex) != embedding_width for row in rows):
        raise ValueError("offline relationship embedding width drift")

    parameters: list[RelationshipConditionLinearClassParameters] = []
    for label in labels:
        class_rows = tuple(
            sorted(
                (row for row in rows if row.condition_label == label),
                key=lambda row: row.example_id,
            )
        )
        normalized_rows = tuple(
            _normalize_numeric_vector(
                row.embedding,
                expected_width=embedding_width,
                field_name="offline relationship embedding",
            )
            for row in class_rows
        )
        mean = tuple(
            math.fsum(row[index] for row in normalized_rows)
            / len(normalized_rows)
            for index in range(embedding_width)
        )
        centroid = _normalize_numeric_vector(
            mean,
            expected_width=embedding_width,
            field_name="offline relationship class centroid",
        )
        centroid_hex = tuple(_canonical_float_hex(value) for value in centroid)
        parameters.append(
            RelationshipConditionLinearClassParameters(
                label=label,
                example_count=len(class_rows),
                example_ids_sha256=_sha256_json(
                    {
                        "example_ids": [row.example_id for row in class_rows],
                    }
                ),
                centroid_hex=centroid_hex,
                coefficient_hex=centroid_hex,
                bias_hex=_canonical_float_hex(0.0),
            )
        )
    return FrozenLinearRelationshipConditionReaderArtifact(
        embedding_model_id=embedding_model_id,
        embedding_model_revision=embedding_model_revision,
        embedding_weights_sha256=embedding_weights_sha256,
        embedding_runtime_version=embedding_runtime_version,
        embedding_width=embedding_width,
        labels=labels,
        condition_training_corpus_artifact_id=(
            condition_training_corpus_artifact_id
        ),
        condition_training_corpus_raw_sha256=(
            condition_training_corpus_raw_sha256
        ),
        group_split_artifact_id=group_split_artifact_id,
        class_parameters=tuple(parameters),
    )


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


class FrozenLinearRelationshipConditionReaderRuntime:
    """Inference-only v2 reader over public text and a pinned frozen embedder."""

    def __init__(
        self,
        *,
        artifact: FrozenLinearRelationshipConditionReaderArtifact,
        embedder: FrozenRelationshipTextEmbedder,
    ) -> None:
        _require_text(embedder.model_source, "embedder model_source")
        _require_text(embedder.model_revision, "embedder model_revision")
        _require_sha256(
            embedder.weights_sha256,
            "embedder weights_sha256",
        )
        _require_text(
            embedder.sentence_transformers_version,
            "embedder sentence_transformers_version",
        )
        identity = (
            embedder.model_source,
            embedder.model_revision,
            embedder.weights_sha256,
            embedder.sentence_transformers_version,
        )
        expected_identity = (
            artifact.embedding_model_id,
            artifact.embedding_model_revision,
            artifact.embedding_weights_sha256,
            artifact.embedding_runtime_version,
        )
        if identity != expected_identity:
            raise ValueError(
                "frozen relationship embedder identity does not match reader artifact"
            )
        self._artifact = artifact
        self._embedder = embedder
        self._parameters = tuple(
            (
                item.label,
                tuple(
                    _decode_canonical_float_hex(value, "coefficient_hex")
                    for value in item.coefficient_hex
                ),
                _decode_canonical_float_hex(item.bias_hex, "bias_hex"),
            )
            for item in artifact.class_parameters
        )
        self._readouts: dict[str, RelationshipConditionReadout] = {}
        self.runtime_id = (
            "relationship-condition-linear-reader.v2:"
            f"{artifact.artifact_id}"
        )

    @property
    def artifact(self) -> FrozenLinearRelationshipConditionReaderArtifact:
        return self._artifact

    def read_condition(self, text: str) -> RelationshipConditionReadout:
        _require_text(text, "relationship condition source text")
        cached = self._readouts.get(text)
        if cached is not None:
            return cached
        vector = _normalize_numeric_vector(
            self._embedder.embed(text),
            expected_width=self._artifact.embedding_width,
            field_name="frozen relationship embedder output",
        )
        scores = tuple(
            (
                label,
                max(-1.0, min(1.0, _cosine(vector, coefficient) + bias)),
            )
            for label, coefficient, bias in self._parameters
        )
        top_index = max(range(len(scores)), key=lambda index: scores[index][1])
        top_label, top_score = scores[top_index]
        second_score = max(
            score for index, (_, score) in enumerate(scores) if index != top_index
        )
        normalized_margin = min(
            1.0,
            max(0.0, (top_score - second_score) / 2.0),
        )
        maximum = max(score for _, score in scores)
        exponentials = tuple(math.exp(score - maximum) for _, score in scores)
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


def _canonical_float_hex(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("reader parameter floats must be finite")
    if numeric == 0.0:
        numeric = 0.0
    return numeric.hex()


def _decode_canonical_float_hex(value: str, field_name: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} entries must be canonical float hex strings")
    try:
        numeric = float.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} entries must be canonical float hex strings"
        ) from error
    if not math.isfinite(numeric) or _canonical_float_hex(numeric) != value:
        raise ValueError(f"{field_name} entries must be finite canonical float hex")
    return numeric


def _normalize_numeric_vector(
    values: object,
    *,
    expected_width: int,
    field_name: str,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != expected_width:
        raise ValueError(f"{field_name} width drift")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{field_name} must contain only finite numeric values")
    numeric = tuple(float(value) for value in values)
    norm = math.hypot(*numeric)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{field_name} norm must be positive and finite")
    return tuple(value / norm for value in numeric)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_payload_keys(
    payload: dict[object, object],
    expected: set[str],
    field_name: str,
) -> None:
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{field_name} keys must be strings")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{field_name} keys mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _require_utf8_json_bytes(raw: str | bytes) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if not isinstance(raw, str):
        raise ValueError("linear relationship reader JSON must be text or bytes")
    try:
        return raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "linear relationship reader JSON must be exact UTF-8"
        ) from error


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


def _require_embedding_width(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("embedding_width must be an integer of at least two")


def _require_strict_ordered_labels(labels: tuple[str, ...]) -> None:
    if not isinstance(labels, tuple) or len(labels) < 2:
        raise ValueError("linear relationship condition labels require a tuple of two")
    for label in labels:
        _require_text(label, "linear relationship condition label")
    if len(set(labels)) != len(labels):
        raise ValueError("linear relationship condition labels must be unique")
    expected = tuple(sorted(labels, key=lambda label: label.encode("utf-8")))
    if labels != expected:
        raise ValueError(
            "linear relationship condition labels must use strict UTF-8 byte order"
        )


__all__ = [
    "FrozenLinearRelationshipConditionReaderArtifact",
    "FrozenLinearRelationshipConditionReaderRuntime",
    "FrozenRelationshipTextEmbedder",
    "LabeledRelationshipConditionEmbeddingRow",
    "PrototypeRelationshipPreferenceForecastRuntime",
    "RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION",
    "RELATIONSHIP_CONDITION_LINEAR_SOLVER",
    "RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION",
    "RELATIONSHIP_CONDITION_READER_SCHEMA_VERSION",
    "RelationshipConditionPrototype",
    "RelationshipConditionLinearClassParameters",
    "RelationshipConditionReaderArtifact",
    "RelationshipTextEmbedder",
    "build_frozen_linear_relationship_condition_reader_artifact",
]
