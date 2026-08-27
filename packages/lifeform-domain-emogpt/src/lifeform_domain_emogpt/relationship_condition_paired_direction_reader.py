"""Fixed-scale relationship-condition reader fitted from matched semantic pairs.

This add-only v3 owner leaves the byte-frozen v1/v2 reader module untouched.
It consumes condition-only training embeddings and never accepts challenge,
action, outcome, prediction-error, credit, evaluation, or judge inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

from volvence_zero.social import (
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
)
from volvence_zero.social_cognition import (
    OtherMindRecord,
    PreferenceActionOutcomeEvidence,
    RelationshipConditionReadout,
)

from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenRelationshipTextEmbedder,
    LabeledRelationshipConditionEmbeddingRow,
    _canonical_float_hex,
    _canonical_json,
    _cosine,
    _decode_canonical_float_hex,
    _normalize_numeric_vector,
    _reject_json_constant,
    _require_embedding_width,
    _require_exact_payload_keys,
    _require_sha256,
    _require_strict_ordered_labels,
    _require_text,
    _require_utf8_json_bytes,
    _sha256_json,
    _unique_json_object,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)


RELATIONSHIP_CONDITION_PAIRED_DIRECTION_READER_SCHEMA_VERSION = (
    "relationship-condition-reader-artifact.v3"
)
RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER = (
    "matched_semantic_pair_direction_linear"
)
RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER_VERSION = (
    "relationship-condition-paired-direction-solver.v1"
)


@dataclass(frozen=True)
class MatchedRelationshipConditionEmbeddingPair:
    """Outcome-free semantic pair that differs only in its named condition."""

    pair_id: str
    semantic_group_id: str
    positive: LabeledRelationshipConditionEmbeddingRow
    negative: LabeledRelationshipConditionEmbeddingRow

    def __post_init__(self) -> None:
        _require_sha256(self.pair_id, "pair_id")
        _require_sha256(self.semantic_group_id, "semantic_group_id")
        if type(self.positive) is not LabeledRelationshipConditionEmbeddingRow:
            raise TypeError("positive must be a labeled relationship embedding row")
        if type(self.negative) is not LabeledRelationshipConditionEmbeddingRow:
            raise TypeError("negative must be a labeled relationship embedding row")
        if self.positive.example_id == self.negative.example_id:
            raise ValueError("matched relationship pair example ids must differ")
        if self.positive.condition_label == self.negative.condition_label:
            raise ValueError("matched relationship pair condition labels must differ")
        if len(self.positive.embedding_hex) != len(self.negative.embedding_hex):
            raise ValueError("matched relationship pair embedding widths must match")


@dataclass(frozen=True)
class FrozenPairedDirectionRelationshipConditionReaderArtifact:
    """Content-addressed v3 binary reader fitted from matched semantic pairs."""

    embedding_model_id: str
    embedding_model_revision: str
    embedding_weights_sha256: str
    embedding_runtime_version: str
    embedding_width: int
    labels: tuple[str, str]
    condition_training_corpus_artifact_id: str
    condition_training_corpus_raw_sha256: str
    training_group_split_artifact_id: str
    training_selection_receipt_artifact_id: str
    pair_count: int
    semantic_group_count: int
    pair_ids_sha256: str
    semantic_group_ids_sha256: str
    direction_hex: tuple[str, ...]
    threshold_hex: str
    solver: str = RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER
    solver_version: str = RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER_VERSION
    schema_version: str = RELATIONSHIP_CONDITION_PAIRED_DIRECTION_READER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_CONDITION_PAIRED_DIRECTION_READER_SCHEMA_VERSION:
            raise ValueError("paired-direction relationship condition reader schema mismatch")
        _require_text(self.embedding_model_id, "embedding_model_id")
        _require_text(self.embedding_model_revision, "embedding_model_revision")
        _require_sha256(self.embedding_weights_sha256, "embedding_weights_sha256")
        _require_text(self.embedding_runtime_version, "embedding_runtime_version")
        _require_embedding_width(self.embedding_width)
        _require_strict_ordered_labels(self.labels)
        if len(self.labels) != 2:
            raise ValueError("paired-direction relationship reader requires exactly two labels")
        for field_name, value in (
            (
                "condition_training_corpus_artifact_id",
                self.condition_training_corpus_artifact_id,
            ),
            (
                "condition_training_corpus_raw_sha256",
                self.condition_training_corpus_raw_sha256,
            ),
            ("training_group_split_artifact_id", self.training_group_split_artifact_id),
            (
                "training_selection_receipt_artifact_id",
                self.training_selection_receipt_artifact_id,
            ),
            ("pair_ids_sha256", self.pair_ids_sha256),
            ("semantic_group_ids_sha256", self.semantic_group_ids_sha256),
        ):
            _require_sha256(value, field_name)
        for field_name, value in (
            ("pair_count", self.pair_count),
            ("semantic_group_count", self.semantic_group_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.semantic_group_count > self.pair_count:
            raise ValueError("semantic_group_count cannot exceed pair_count")
        if (
            not isinstance(self.direction_hex, tuple)
            or len(self.direction_hex) != self.embedding_width
        ):
            raise ValueError("paired-direction reader direction width drift")
        direction = tuple(
            _decode_canonical_float_hex(value, "direction_hex")
            for value in self.direction_hex
        )
        if abs(math.hypot(*direction) - 1.0) > 1e-12:
            raise ValueError("paired-direction reader direction must have unit norm")
        threshold = _decode_canonical_float_hex(self.threshold_hex, "threshold_hex")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("paired-direction reader threshold must be in [-1, 1]")
        if self.solver != RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER:
            raise ValueError("paired-direction relationship reader solver mismatch")
        if self.solver_version != RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER_VERSION:
            raise ValueError("paired-direction relationship reader solver version mismatch")

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
            "training_group_split_artifact_id": self.training_group_split_artifact_id,
            "training_selection_receipt_artifact_id": (
                self.training_selection_receipt_artifact_id
            ),
            "pair_count": self.pair_count,
            "semantic_group_count": self.semantic_group_count,
            "pair_ids_sha256": self.pair_ids_sha256,
            "semantic_group_ids_sha256": self.semantic_group_ids_sha256,
            "direction_hex": list(self.direction_hex),
            "threshold_hex": self.threshold_hex,
            "solver": self.solver,
            "solver_version": self.solver_version,
        }

    @property
    def artifact_id(self) -> str:
        return _sha256_json(self._content_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self._content_payload(), "artifact_id": self.artifact_id}

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> FrozenPairedDirectionRelationshipConditionReaderArtifact:
        if not isinstance(payload, dict):
            raise ValueError("paired-direction relationship reader payload must be an object")
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
                "training_group_split_artifact_id",
                "training_selection_receipt_artifact_id",
                "pair_count",
                "semantic_group_count",
                "pair_ids_sha256",
                "semantic_group_ids_sha256",
                "direction_hex",
                "threshold_hex",
                "solver",
                "solver_version",
                "artifact_id",
            },
            "paired-direction relationship reader payload",
        )
        labels = payload["labels"]
        direction_hex = payload["direction_hex"]
        if not isinstance(labels, list):
            raise ValueError("paired-direction relationship reader labels must be a JSON array")
        if len(labels) != 2:
            raise ValueError("paired-direction relationship reader requires two labels")
        if not isinstance(direction_hex, list):
            raise ValueError(
                "paired-direction relationship reader direction_hex must be a JSON array"
            )
        supplied_artifact_id = payload["artifact_id"]
        _require_sha256(supplied_artifact_id, "artifact_id")
        artifact = cls(
            embedding_model_id=payload["embedding_model_id"],
            embedding_model_revision=payload["embedding_model_revision"],
            embedding_weights_sha256=payload["embedding_weights_sha256"],
            embedding_runtime_version=payload["embedding_runtime_version"],
            embedding_width=payload["embedding_width"],
            labels=(labels[0], labels[1]),
            condition_training_corpus_artifact_id=(
                payload["condition_training_corpus_artifact_id"]
            ),
            condition_training_corpus_raw_sha256=(
                payload["condition_training_corpus_raw_sha256"]
            ),
            training_group_split_artifact_id=payload["training_group_split_artifact_id"],
            training_selection_receipt_artifact_id=(
                payload["training_selection_receipt_artifact_id"]
            ),
            pair_count=payload["pair_count"],
            semantic_group_count=payload["semantic_group_count"],
            pair_ids_sha256=payload["pair_ids_sha256"],
            semantic_group_ids_sha256=payload["semantic_group_ids_sha256"],
            direction_hex=tuple(direction_hex),
            threshold_hex=payload["threshold_hex"],
            solver=payload["solver"],
            solver_version=payload["solver_version"],
            schema_version=payload["schema_version"],
        )
        if supplied_artifact_id != artifact.artifact_id:
            raise ValueError("paired-direction relationship reader artifact_id mismatch")
        return artifact

    def to_json(self) -> str:
        return _canonical_json(self.to_payload()) + "\n"

    def to_json_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(
        cls,
        raw: str | bytes,
    ) -> FrozenPairedDirectionRelationshipConditionReaderArtifact:
        raw_bytes = _require_utf8_json_bytes(raw)
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "paired-direction relationship reader JSON must be exact UTF-8"
            ) from error
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError("invalid paired-direction relationship reader JSON") from error
        artifact = cls.from_payload(payload)
        if raw_bytes != artifact.to_json_bytes():
            raise ValueError(
                "paired-direction relationship reader JSON must use canonical UTF-8 bytes"
            )
        return artifact


def build_frozen_paired_direction_relationship_condition_reader_artifact(
    *,
    embedding_model_id: str,
    embedding_model_revision: str,
    embedding_weights_sha256: str,
    embedding_runtime_version: str,
    embedding_width: int,
    labels: tuple[str, str],
    condition_training_corpus_artifact_id: str,
    condition_training_corpus_raw_sha256: str,
    training_group_split_artifact_id: str,
    training_selection_receipt_artifact_id: str,
    pairs: tuple[MatchedRelationshipConditionEmbeddingPair, ...],
) -> FrozenPairedDirectionRelationshipConditionReaderArtifact:
    """Fit the fixed-scale v3 direction and threshold from training pairs only."""

    _require_embedding_width(embedding_width)
    _require_strict_ordered_labels(labels)
    if len(labels) != 2:
        raise ValueError("paired-direction relationship reader requires exactly two labels")
    if not isinstance(pairs, tuple) or not pairs:
        raise ValueError("paired relationship embedding rows must be a non-empty tuple")
    if any(type(pair) is not MatchedRelationshipConditionEmbeddingPair for pair in pairs):
        raise TypeError("paired relationship embedding rows have invalid types")
    ordered_pairs = tuple(sorted(pairs, key=lambda pair: pair.pair_id))
    pair_ids = tuple(pair.pair_id for pair in ordered_pairs)
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("paired relationship embedding pair ids must be unique")
    example_ids = tuple(
        example_id
        for pair in ordered_pairs
        for example_id in (pair.positive.example_id, pair.negative.example_id)
    )
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("paired relationship embedding example ids must be globally unique")
    semantic_group_ids = tuple(pair.semantic_group_id for pair in ordered_pairs)
    if len(set(semantic_group_ids)) < 2:
        raise ValueError("paired relationship training requires at least two semantic groups")
    for pair in ordered_pairs:
        if (
            pair.positive.condition_label != labels[0]
            or pair.negative.condition_label != labels[1]
        ):
            raise ValueError("paired relationship labels must match the ordered artifact labels")
        if (
            len(pair.positive.embedding_hex) != embedding_width
            or len(pair.negative.embedding_hex) != embedding_width
        ):
            raise ValueError("paired relationship embedding width drift")

    normalized_pairs = tuple(
        (
            _normalize_numeric_vector(
                pair.positive.embedding,
                expected_width=embedding_width,
                field_name="paired positive relationship embedding",
            ),
            _normalize_numeric_vector(
                pair.negative.embedding,
                expected_width=embedding_width,
                field_name="paired negative relationship embedding",
            ),
        )
        for pair in ordered_pairs
    )
    mean_difference = tuple(
        math.fsum(
            positive[index] - negative[index]
            for positive, negative in normalized_pairs
        )
        / len(normalized_pairs)
        for index in range(embedding_width)
    )
    direction = _normalize_numeric_vector(
        mean_difference,
        expected_width=embedding_width,
        field_name="paired relationship mean direction",
    )
    midpoints = tuple(
        (_cosine(direction, positive) + _cosine(direction, negative)) / 2.0
        for positive, negative in normalized_pairs
    )
    ordered_midpoints = tuple(sorted(midpoints))
    midpoint_index = len(ordered_midpoints) // 2
    threshold = (
        ordered_midpoints[midpoint_index]
        if len(ordered_midpoints) % 2 == 1
        else math.fsum(
            (
                ordered_midpoints[midpoint_index - 1],
                ordered_midpoints[midpoint_index],
            )
        )
        / 2.0
    )
    return FrozenPairedDirectionRelationshipConditionReaderArtifact(
        embedding_model_id=embedding_model_id,
        embedding_model_revision=embedding_model_revision,
        embedding_weights_sha256=embedding_weights_sha256,
        embedding_runtime_version=embedding_runtime_version,
        embedding_width=embedding_width,
        labels=labels,
        condition_training_corpus_artifact_id=condition_training_corpus_artifact_id,
        condition_training_corpus_raw_sha256=condition_training_corpus_raw_sha256,
        training_group_split_artifact_id=training_group_split_artifact_id,
        training_selection_receipt_artifact_id=(
            training_selection_receipt_artifact_id
        ),
        pair_count=len(ordered_pairs),
        semantic_group_count=len(set(semantic_group_ids)),
        pair_ids_sha256=_sha256_json({"pair_ids": list(pair_ids)}),
        semantic_group_ids_sha256=_sha256_json(
            {"semantic_group_ids": sorted(set(semantic_group_ids))}
        ),
        direction_hex=tuple(_canonical_float_hex(value) for value in direction),
        threshold_hex=_canonical_float_hex(threshold),
    )


class FrozenPairedDirectionRelationshipConditionReaderRuntime:
    """Inference-only v3 reader with one fixed unit direction and midpoint."""

    def __init__(
        self,
        *,
        artifact: FrozenPairedDirectionRelationshipConditionReaderArtifact,
        embedder: FrozenRelationshipTextEmbedder,
    ) -> None:
        _require_text(embedder.model_source, "embedder model_source")
        _require_text(embedder.model_revision, "embedder model_revision")
        _require_sha256(embedder.weights_sha256, "embedder weights_sha256")
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
        self._direction = tuple(
            _decode_canonical_float_hex(value, "direction_hex")
            for value in artifact.direction_hex
        )
        self._threshold = _decode_canonical_float_hex(
            artifact.threshold_hex,
            "threshold_hex",
        )
        self._readouts: dict[str, RelationshipConditionReadout] = {}
        self.runtime_id = (
            "relationship-condition-paired-direction-reader.v3:"
            f"{artifact.artifact_id}"
        )

    @property
    def artifact(self) -> FrozenPairedDirectionRelationshipConditionReaderArtifact:
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
        signed_distance = _cosine(self._direction, vector) - self._threshold
        scores = (
            (self._artifact.labels[0], signed_distance / 2.0),
            (self._artifact.labels[1], -signed_distance / 2.0),
        )
        top_index = 0 if scores[0][1] >= scores[1][1] else 1
        top_label, top_score = scores[top_index]
        second_score = scores[1 - top_index][1]
        normalized_margin = (top_score - second_score) / 2.0
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


class FrozenPairedDirectionRelationshipPreferenceForecastRuntime:
    """Bind one v3 paired-direction readout to the preference forecast owner."""

    def __init__(
        self,
        *,
        reader: FrozenPairedDirectionRelationshipConditionReaderRuntime,
    ) -> None:
        if type(reader) is not FrozenPairedDirectionRelationshipConditionReaderRuntime:
            raise TypeError(
                "reader must be "
                "FrozenPairedDirectionRelationshipConditionReaderRuntime"
            )
        self._reader = reader
        self._delegate = BoundedRelationshipPreferenceForecastRuntime(
            similarity=self._condition_similarity,
        )
        self.runtime_id = (
            "relationship-condition-paired-direction-forecast.v3:"
            f"{reader.artifact.artifact_id}"
        )

    @property
    def artifact(self) -> FrozenPairedDirectionRelationshipConditionReaderArtifact:
        return self._reader.artifact

    def read_condition(self, text: str) -> RelationshipConditionReadout:
        return self._reader.read_condition(text)

    def _condition_similarity(self, left: str, right: str) -> float:
        left_readout = self.read_condition(left)
        right_readout = self.read_condition(right)
        if left_readout.condition_label != right_readout.condition_label:
            return 0.0
        return min(left_readout.confidence, right_readout.confidence)

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
                        f"runtime:{self.runtime_id}",
                        f"condition_reader:{self.artifact.artifact_id}",
                        f"condition_label:{current.condition_label}",
                    )
                )
            ),
            condition_readout=current,
        )


__all__ = [
    "FrozenPairedDirectionRelationshipConditionReaderArtifact",
    "FrozenPairedDirectionRelationshipConditionReaderRuntime",
    "FrozenPairedDirectionRelationshipPreferenceForecastRuntime",
    "MatchedRelationshipConditionEmbeddingPair",
    "RELATIONSHIP_CONDITION_PAIRED_DIRECTION_READER_SCHEMA_VERSION",
    "RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER",
    "RELATIONSHIP_CONDITION_PAIRED_DIRECTION_SOLVER_VERSION",
    "build_frozen_paired_direction_relationship_condition_reader_artifact",
]
