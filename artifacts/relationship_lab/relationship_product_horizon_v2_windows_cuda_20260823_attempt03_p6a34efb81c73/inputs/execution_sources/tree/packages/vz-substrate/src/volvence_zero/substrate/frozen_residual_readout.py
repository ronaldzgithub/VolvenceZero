"""Frozen linear readout over substrate-owned residual representations.

This module turns an immutable :class:`SubstrateForwardRepresentationSnapshot`
into a content-addressed, read-only classification artifact.  It deliberately
does not capture the model, parse text, choose actions, or mutate substrate
weights.  The only learned object is a closed-form ridge readout over the
already-published residual geometry.

Each class also publishes a normalized contrast axis (that class's effective
weight minus the mean effective weight of the other classes).  The axis is an
offline evidence surface for matched causal-steering experiments; installing
or using it in a live runtime requires a separate owner/wiring decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from volvence_zero.substrate.forward_representation import (
    SUBSTRATE_FORWARD_READOUT_KIND,
    SubstrateForwardRepresentationSnapshot,
)
from volvence_zero.substrate.substrate_fingerprint import SubstrateFingerprint


FROZEN_RESIDUAL_READOUT_SCHEMA_VERSION = "frozen-residual-readout.v1"
SUBSTRATE_RESIDUAL_READOUT_SCHEMA_VERSION = "substrate-residual-readout.v1"
FROZEN_RESIDUAL_READOUT_TRAINING_MODE = (
    "closed-form-standardized-ridge-one-hot.v1"
)

_MIN_SCALE = 1e-6
_MIN_AXIS_NORM = 1e-12


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalized(
    values: tuple[float, ...],
    *,
    context: str,
) -> tuple[float, ...]:
    norm = math.sqrt(_dot(values, values))
    if not math.isfinite(norm) or norm <= _MIN_AXIS_NORM:
        raise ValueError(
            f"frozen residual readout produced a degenerate {context} "
            f"(norm={norm!r})"
        )
    return tuple(value / norm for value in values)


@dataclass(frozen=True)
class FrozenResidualReadoutArtifact:
    """Content-addressed classifier and steering-axis geometry.

    ``class_weights`` are effective weights in the published representation
    coordinates (standardization has already been folded into them), so a
    score is exactly ``dot(values, class_weights[c]) + class_biases[c]``.
    ``class_axes`` contain no bias term and cannot reproduce the free-bias
    bypass diagnosed in ETA Stage-3.
    """

    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    source_readout_kind: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int
    class_ids: tuple[str, ...]
    ridge_alpha: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    class_weights: tuple[tuple[float, ...], ...]
    class_biases: tuple[float, ...]
    class_axes: tuple[tuple[float, ...], ...]
    training_snapshot_fingerprint: str
    training_labels_sha256: str
    training_support: int
    description: str
    training_mode: str = FROZEN_RESIDUAL_READOUT_TRAINING_MODE
    schema_version: str = FROZEN_RESIDUAL_READOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FROZEN_RESIDUAL_READOUT_SCHEMA_VERSION:
            raise ValueError("frozen residual readout schema_version mismatch")
        if self.training_mode != FROZEN_RESIDUAL_READOUT_TRAINING_MODE:
            raise ValueError("frozen residual readout training_mode mismatch")
        if not self.runtime_origin.strip():
            raise ValueError("frozen residual readout runtime_origin must be non-empty")
        if self.source_readout_kind != SUBSTRATE_FORWARD_READOUT_KIND:
            raise ValueError("frozen residual readout source_readout_kind mismatch")
        if not self.layer_indices or tuple(sorted(self.layer_indices)) != self.layer_indices:
            raise ValueError(
                "frozen residual readout layer_indices must be non-empty and sorted"
            )
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("frozen residual readout layer_indices must be unique")
        if len(self.activation_widths) != len(self.layer_indices) or any(
            width < 1 for width in self.activation_widths
        ):
            raise ValueError("frozen residual readout activation_widths are invalid")
        if self.representation_dim != sum(self.activation_widths):
            raise ValueError("frozen residual readout representation geometry mismatch")
        if len(self.class_ids) < 2 or any(not value.strip() for value in self.class_ids):
            raise ValueError("frozen residual readout requires at least two class_ids")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("frozen residual readout class_ids must be unique")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0.0:
            raise ValueError("frozen residual readout ridge_alpha must be positive")
        if self.training_support < len(self.class_ids):
            raise ValueError(
                "frozen residual readout training_support cannot cover all classes"
            )
        if not _is_sha256(self.training_snapshot_fingerprint):
            raise ValueError(
                "frozen residual readout training_snapshot_fingerprint is invalid"
            )
        if not _is_sha256(self.training_labels_sha256):
            raise ValueError("frozen residual readout training_labels_sha256 is invalid")
        if not self.description.strip():
            raise ValueError("frozen residual readout description must be non-empty")

        vectors = (self.feature_mean, self.feature_scale)
        if any(len(vector) != self.representation_dim for vector in vectors):
            raise ValueError("frozen residual readout feature-stat width mismatch")
        if not all(math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("frozen residual readout feature stats must be finite")
        if any(value < _MIN_SCALE for value in self.feature_scale):
            raise ValueError("frozen residual readout feature scales are too small")

        class_count = len(self.class_ids)
        if (
            len(self.class_weights) != class_count
            or len(self.class_biases) != class_count
            or len(self.class_axes) != class_count
        ):
            raise ValueError("frozen residual readout class geometry is misaligned")
        if any(
            len(vector) != self.representation_dim
            for vector in (*self.class_weights, *self.class_axes)
        ):
            raise ValueError("frozen residual readout class vector width mismatch")
        if not all(
            math.isfinite(value)
            for vector in (*self.class_weights, *self.class_axes)
            for value in vector
        ) or not all(math.isfinite(value) for value in self.class_biases):
            raise ValueError("frozen residual readout class geometry must be finite")
        for axis in self.class_axes:
            norm = math.sqrt(_dot(axis, axis))
            if not math.isclose(norm, 1.0, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("frozen residual readout class axes must be normalized")

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256(
            _canonical_json(self.as_json_dict(include_artifact_id=False)).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"{self.schema_version}:{digest}"

    def axis_for(self, class_id: str) -> tuple[float, ...]:
        try:
            index = self.class_ids.index(class_id)
        except ValueError:
            raise KeyError(
                f"unknown frozen residual readout class_id {class_id!r}"
            ) from None
        return self.class_axes[index]

    def as_json_dict(self, *, include_artifact_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "training_mode": self.training_mode,
            "model_fingerprint": {
                "model_id": self.model_fingerprint.model_id,
                "version": self.model_fingerprint.version,
                "weights_sha256": self.model_fingerprint.weights_sha256,
            },
            "runtime_origin": self.runtime_origin,
            "source_readout_kind": self.source_readout_kind,
            "layer_indices": list(self.layer_indices),
            "activation_widths": list(self.activation_widths),
            "representation_dim": self.representation_dim,
            "class_ids": list(self.class_ids),
            "ridge_alpha": self.ridge_alpha,
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "class_weights": [list(row) for row in self.class_weights],
            "class_biases": list(self.class_biases),
            "class_axes": [list(row) for row in self.class_axes],
            "training_snapshot_fingerprint": self.training_snapshot_fingerprint,
            "training_labels_sha256": self.training_labels_sha256,
            "training_support": self.training_support,
            "description": self.description,
        }
        if include_artifact_id:
            payload["artifact_id"] = self.artifact_id
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> FrozenResidualReadoutArtifact:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("frozen residual readout JSON must be an object")
        artifact_id = raw.get("artifact_id")
        model = raw.get("model_fingerprint")
        if not isinstance(model, dict):
            raise ValueError("frozen residual readout model_fingerprint is missing")
        artifact = cls(
            schema_version=str(raw["schema_version"]),
            training_mode=str(raw["training_mode"]),
            model_fingerprint=SubstrateFingerprint(
                model_id=str(model["model_id"]),
                version=str(model["version"]),
                weights_sha256=str(model["weights_sha256"]),
            ),
            runtime_origin=str(raw["runtime_origin"]),
            source_readout_kind=str(raw["source_readout_kind"]),
            layer_indices=tuple(int(value) for value in raw["layer_indices"]),
            activation_widths=tuple(
                int(value) for value in raw["activation_widths"]
            ),
            representation_dim=int(raw["representation_dim"]),
            class_ids=tuple(str(value) for value in raw["class_ids"]),
            ridge_alpha=float(raw["ridge_alpha"]),
            feature_mean=tuple(float(value) for value in raw["feature_mean"]),
            feature_scale=tuple(float(value) for value in raw["feature_scale"]),
            class_weights=tuple(
                tuple(float(value) for value in row)
                for row in raw["class_weights"]
            ),
            class_biases=tuple(float(value) for value in raw["class_biases"]),
            class_axes=tuple(
                tuple(float(value) for value in row) for row in raw["class_axes"]
            ),
            training_snapshot_fingerprint=str(
                raw["training_snapshot_fingerprint"]
            ),
            training_labels_sha256=str(raw["training_labels_sha256"]),
            training_support=int(raw["training_support"]),
            description=str(raw["description"]),
        )
        if artifact_id != artifact.artifact_id:
            raise ValueError("frozen residual readout artifact_id mismatch")
        return artifact


@dataclass(frozen=True)
class SubstrateResidualReadout:
    sample_id: str
    source_sha256: str
    class_scores: tuple[tuple[str, float], ...]
    predicted_class_id: str
    score_margin: float

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("substrate residual readout sample_id must be non-empty")
        if not _is_sha256(self.source_sha256):
            raise ValueError("substrate residual readout source_sha256 is invalid")
        if len(self.class_scores) < 2:
            raise ValueError("substrate residual readout requires at least two scores")
        class_ids = tuple(class_id for class_id, _ in self.class_scores)
        if any(not class_id.strip() for class_id in class_ids) or len(
            set(class_ids)
        ) != len(class_ids):
            raise ValueError("substrate residual readout class score ids are invalid")
        if not all(math.isfinite(score) for _, score in self.class_scores):
            raise ValueError("substrate residual readout scores must be finite")
        winner = max(
            range(len(self.class_scores)),
            key=lambda index: (self.class_scores[index][1], -index),
        )
        if self.predicted_class_id != self.class_scores[winner][0]:
            raise ValueError("substrate residual readout prediction/score mismatch")
        ordered = sorted((score for _, score in self.class_scores), reverse=True)
        expected_margin = ordered[0] - ordered[1]
        if not math.isfinite(self.score_margin) or not math.isclose(
            self.score_margin,
            expected_margin,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("substrate residual readout score_margin mismatch")


@dataclass(frozen=True)
class SubstrateResidualReadoutLineage:
    schema_version: str
    artifact_id: str
    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    source_snapshot_fingerprint: str
    source_readout_kind: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int

    def __post_init__(self) -> None:
        if self.schema_version != SUBSTRATE_RESIDUAL_READOUT_SCHEMA_VERSION:
            raise ValueError("substrate residual readout lineage schema mismatch")
        prefix = f"{FROZEN_RESIDUAL_READOUT_SCHEMA_VERSION}:"
        if not self.artifact_id.startswith(prefix) or not _is_sha256(
            self.artifact_id.removeprefix(prefix)
        ):
            raise ValueError("substrate residual readout artifact_id is invalid")
        if not self.runtime_origin.strip():
            raise ValueError("substrate residual readout runtime_origin is empty")
        if not _is_sha256(self.source_snapshot_fingerprint):
            raise ValueError(
                "substrate residual readout source snapshot fingerprint is invalid"
            )
        if self.source_readout_kind != SUBSTRATE_FORWARD_READOUT_KIND:
            raise ValueError("substrate residual readout source kind mismatch")
        if self.representation_dim != sum(self.activation_widths):
            raise ValueError("substrate residual readout lineage geometry mismatch")
        if len(self.layer_indices) != len(self.activation_widths):
            raise ValueError("substrate residual readout lineage layers misalign")


@dataclass(frozen=True)
class SubstrateResidualReadoutSnapshot:
    lineage: SubstrateResidualReadoutLineage
    readouts: tuple[SubstrateResidualReadout, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.readouts:
            raise ValueError("substrate residual readout snapshot must contain rows")
        sample_ids = tuple(row.sample_id for row in self.readouts)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("substrate residual readout sample_ids must be unique")
        if not self.description.strip():
            raise ValueError("substrate residual readout description must be non-empty")


def fit_frozen_residual_readout(
    *,
    torch_module: Any,
    snapshot: SubstrateForwardRepresentationSnapshot,
    labels: tuple[tuple[str, str], ...],
    class_ids: tuple[str, ...],
    ridge_alpha: float = 1.0,
) -> FrozenResidualReadoutArtifact:
    """Fit a deterministic, closed-form classifier on one training snapshot.

    Labels are an external typed supervision contract keyed by ``sample_id``;
    their semantic meaning remains with the caller.  This owner only learns
    geometry for the supplied opaque class ids.  Every representation row must
    receive exactly one label and every declared class must be represented.
    """

    if len(class_ids) < 2 or any(not value.strip() for value in class_ids):
        raise ValueError("frozen residual readout requires at least two class_ids")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("frozen residual readout class_ids must be unique")
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise ValueError("frozen residual readout ridge_alpha must be positive")
    label_map: dict[str, str] = {}
    for sample_id, class_id in labels:
        if sample_id in label_map:
            raise ValueError(
                f"frozen residual readout duplicate label for {sample_id!r}"
            )
        label_map[sample_id] = class_id
    sample_ids = tuple(row.sample_id for row in snapshot.representations)
    if set(label_map) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(label_map))
        extra = sorted(set(label_map) - set(sample_ids))
        raise ValueError(
            "frozen residual readout labels must match the training snapshot "
            f"exactly; missing={missing}, extra={extra}"
        )
    class_index = {class_id: index for index, class_id in enumerate(class_ids)}
    unknown = sorted(set(label_map.values()) - set(class_index))
    if unknown:
        raise ValueError(f"frozen residual readout labels contain unknown classes {unknown}")
    ordered_labels = tuple((sample_id, label_map[sample_id]) for sample_id in sample_ids)
    absent = tuple(
        class_id
        for class_id in class_ids
        if class_id not in {label for _, label in ordered_labels}
    )
    if absent:
        raise ValueError(
            f"frozen residual readout training snapshot misses classes {absent}"
        )

    torch = torch_module
    features = torch.tensor(
        [row.values for row in snapshot.representations],
        dtype=torch.float64,
        device="cpu",
    )
    target_indices = torch.tensor(
        [class_index[label_map[sample_id]] for sample_id in sample_ids],
        dtype=torch.long,
        device="cpu",
    )
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_scale = features.std(dim=0, correction=0, keepdim=True).clamp_min(
        _MIN_SCALE
    )
    standardized = (features - feature_mean) / feature_scale
    one_hot = torch.zeros(
        (len(sample_ids), len(class_ids)),
        dtype=torch.float64,
        device="cpu",
    )
    one_hot.scatter_(1, target_indices.unsqueeze(1), 1.0)
    target_mean = one_hot.mean(dim=0, keepdim=True)
    gram = standardized.T @ standardized + ridge_alpha * torch.eye(
        snapshot.lineage.representation_dim,
        dtype=torch.float64,
        device="cpu",
    )
    standardized_weights = torch.linalg.solve(
        gram,
        standardized.T @ (one_hot - target_mean),
    )
    effective_weights = standardized_weights / feature_scale.T
    effective_biases = target_mean - feature_mean @ effective_weights

    weights = tuple(
        tuple(float(value) for value in effective_weights[:, index].tolist())
        for index in range(len(class_ids))
    )
    biases = tuple(float(value) for value in effective_biases[0].tolist())
    axes: list[tuple[float, ...]] = []
    for class_position, class_id in enumerate(class_ids):
        others = tuple(
            row for index, row in enumerate(weights) if index != class_position
        )
        other_mean = tuple(
            sum(row[dimension] for row in others) / len(others)
            for dimension in range(snapshot.lineage.representation_dim)
        )
        contrast = tuple(
            value - other_mean[dimension]
            for dimension, value in enumerate(weights[class_position])
        )
        axes.append(_normalized(contrast, context=f"class-axis:{class_id}"))

    labels_sha256 = hashlib.sha256(
        _canonical_json(ordered_labels).encode("utf-8")
    ).hexdigest()
    return FrozenResidualReadoutArtifact(
        model_fingerprint=snapshot.lineage.model_fingerprint,
        runtime_origin=snapshot.lineage.runtime_origin,
        source_readout_kind=snapshot.lineage.readout_kind,
        layer_indices=snapshot.lineage.layer_indices,
        activation_widths=snapshot.lineage.activation_widths,
        representation_dim=snapshot.lineage.representation_dim,
        class_ids=class_ids,
        ridge_alpha=ridge_alpha,
        feature_mean=tuple(float(value) for value in feature_mean[0].tolist()),
        feature_scale=tuple(float(value) for value in feature_scale[0].tolist()),
        class_weights=weights,
        class_biases=biases,
        class_axes=tuple(axes),
        training_snapshot_fingerprint=snapshot.lineage.snapshot_fingerprint,
        training_labels_sha256=labels_sha256,
        training_support=len(sample_ids),
        description=(
            "Frozen closed-form ridge readout over substrate-owned residual "
            "representations; class ids are opaque and no result is fed back."
        ),
    )


class SubstrateResidualReadoutPublisher:
    """Apply one frozen artifact to compatible immutable representations."""

    def __init__(self, artifact: FrozenResidualReadoutArtifact) -> None:
        self._artifact = artifact

    @property
    def artifact(self) -> FrozenResidualReadoutArtifact:
        return self._artifact

    def publish(
        self,
        snapshot: SubstrateForwardRepresentationSnapshot,
    ) -> SubstrateResidualReadoutSnapshot:
        artifact = self._artifact
        lineage = snapshot.lineage
        expected_geometry = (
            artifact.model_fingerprint,
            artifact.runtime_origin,
            artifact.source_readout_kind,
            artifact.layer_indices,
            artifact.activation_widths,
            artifact.representation_dim,
        )
        actual_geometry = (
            lineage.model_fingerprint,
            lineage.runtime_origin,
            lineage.readout_kind,
            lineage.layer_indices,
            lineage.activation_widths,
            lineage.representation_dim,
        )
        if actual_geometry != expected_geometry:
            raise ValueError(
                "substrate residual readout representation lineage mismatch: "
                f"expected={expected_geometry!r}, actual={actual_geometry!r}"
            )

        readouts: list[SubstrateResidualReadout] = []
        for row in snapshot.representations:
            scores = tuple(
                _dot(row.values, weights) + bias
                for weights, bias in zip(
                    artifact.class_weights,
                    artifact.class_biases,
                    strict=True,
                )
            )
            winner = max(
                range(len(scores)),
                key=lambda index: (scores[index], -index),
            )
            ordered_scores = sorted(scores, reverse=True)
            readouts.append(
                SubstrateResidualReadout(
                    sample_id=row.sample_id,
                    source_sha256=row.source_sha256,
                    class_scores=tuple(zip(artifact.class_ids, scores, strict=True)),
                    predicted_class_id=artifact.class_ids[winner],
                    score_margin=ordered_scores[0] - ordered_scores[1],
                )
            )
        return SubstrateResidualReadoutSnapshot(
            lineage=SubstrateResidualReadoutLineage(
                schema_version=SUBSTRATE_RESIDUAL_READOUT_SCHEMA_VERSION,
                artifact_id=artifact.artifact_id,
                model_fingerprint=artifact.model_fingerprint,
                runtime_origin=artifact.runtime_origin,
                source_snapshot_fingerprint=lineage.snapshot_fingerprint,
                source_readout_kind=artifact.source_readout_kind,
                layer_indices=artifact.layer_indices,
                activation_widths=artifact.activation_widths,
                representation_dim=artifact.representation_dim,
            ),
            readouts=tuple(readouts),
            description=(
                "Read-only frozen residual class scores and margins; no raw "
                "text, action choice, learning signal, or substrate mutation."
            ),
        )


__all__ = [
    "FROZEN_RESIDUAL_READOUT_SCHEMA_VERSION",
    "FROZEN_RESIDUAL_READOUT_TRAINING_MODE",
    "SUBSTRATE_RESIDUAL_READOUT_SCHEMA_VERSION",
    "FrozenResidualReadoutArtifact",
    "SubstrateResidualReadout",
    "SubstrateResidualReadoutLineage",
    "SubstrateResidualReadoutPublisher",
    "SubstrateResidualReadoutSnapshot",
    "fit_frozen_residual_readout",
]
