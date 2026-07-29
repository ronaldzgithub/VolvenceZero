"""Learned control-basis extraction from frozen-substrate transition deltas.

The residual control path lifts the low-dimensional ``applied_control``
vector into the hidden space through a fixed row basis
(``control_vector @ control_basis``). The original basis rows are
arbitrary sinusoids: they have real behavioral effect (schema v33 showed
per-prefix realized-continuation NLL ranges >= 0.02) but the variance
decomposition on the v33 artifact showed the effects are entirely
prefix-specific (global action-main train->validation R^2 ~ -0.01,
within-route leave-one-prefix-out R^2 median ~ -0.31). Steering along
directions the model's own state actually traverses when a route
advances is the falsifiable next hypothesis for recovering transferable
action value.

This module is offline rare-heavy artifact preparation owned by the
substrate: it consumes hidden-state transition deltas captured from the
frozen model (``h_{i+1} - h_i`` at the hooked layers, training routes
only) and produces an orthonormal control basis:

- row 0: the normalized mean transition delta (the average "route
  advances" direction),
- rows 1..rank-1: the top principal components of the centered deltas,
  orthogonalized against row 0 and each other.

The implementation is dependency-free and bit-deterministic (fixed-seed
power iteration on the Gram matrix) so the resulting artifact can be
fingerprinted and preregistered in evidence manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

FIXED_SINUSOID_CONTROL_BASIS_PROVENANCE = "fixed-sinusoid-v1"
TRAIN_TRANSITION_PCA_CONTROL_BASIS_MODE = "train-transition-pca-v1"
FULL_CODE_SINUSOID_CONTROL_BASIS_MODE = "full-code-sinusoid-v1"
CONTROL_BASIS_ARTIFACT_SCHEMA_VERSION = "control-basis-artifact.v1"

_POWER_ITERATIONS = 300
_DEGENERATE_NORM = 1e-9


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class ControlBasisArtifact:
    """Versioned rare-heavy control-basis artifact.

    The artifact is immutable and self-fingerprinting. It carries the exact
    target model, hidden width, hooked layers, and per-layer gains needed to
    reject accidental cross-substrate installation.
    """

    model_id: str
    hidden_size: int
    basis: tuple[tuple[float, ...], ...]
    layer_indices: tuple[int, ...]
    layer_gains: tuple[float, ...]
    training_mode: str
    source_fingerprint: str
    sample_count: int
    description: str
    schema_version: str = CONTROL_BASIS_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_BASIS_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported control-basis schema {self.schema_version!r}"
            )
        if not self.model_id or self.hidden_size < 1:
            raise ValueError("control-basis artifact requires model_id and hidden_size")
        if not self.basis:
            raise ValueError("control-basis artifact requires at least one basis row")
        if any(len(row) != self.hidden_size for row in self.basis):
            raise ValueError("control-basis artifact row width does not match hidden_size")
        if not self.layer_indices or len(set(self.layer_indices)) != len(
            self.layer_indices
        ):
            raise ValueError(
                "control-basis artifact requires unique, non-empty layer_indices"
            )
        if len(self.layer_indices) != len(self.layer_gains):
            raise ValueError(
                "control-basis artifact layer_gains must align with layer_indices"
            )
        if any(not 0.0 < gain <= 1.0 for gain in self.layer_gains):
            raise ValueError("control-basis artifact layer gains must be in (0, 1]")
        if not self.training_mode or not self.source_fingerprint:
            raise ValueError(
                "control-basis artifact requires training_mode and source_fingerprint"
            )
        if self.sample_count < 1 or not self.description:
            raise ValueError(
                "control-basis artifact requires positive sample_count and description"
            )
        for row in self.basis:
            if not all(math.isfinite(value) for value in row):
                raise ValueError("control-basis artifact contains non-finite values")
            if _norm(row) < _DEGENERATE_NORM:
                raise ValueError("control-basis artifact contains a degenerate row")

    @property
    def rank(self) -> int:
        return len(self.basis)

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256(
            _canonical_json(self.as_json_dict(include_artifact_id=False)).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"{self.schema_version}:{digest}"

    def as_json_dict(self, *, include_artifact_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "basis": [list(row) for row in self.basis],
            "layer_indices": list(self.layer_indices),
            "layer_gains": list(self.layer_gains),
            "training_mode": self.training_mode,
            "source_fingerprint": self.source_fingerprint,
            "sample_count": self.sample_count,
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
    def from_json(cls, payload: str) -> "ControlBasisArtifact":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("control-basis artifact JSON must be an object")
        artifact_id = raw.pop("artifact_id", None)
        artifact = cls(
            schema_version=str(raw["schema_version"]),
            model_id=str(raw["model_id"]),
            hidden_size=int(raw["hidden_size"]),
            basis=tuple(
                tuple(float(value) for value in row)
                for row in raw["basis"]
            ),
            layer_indices=tuple(int(value) for value in raw["layer_indices"]),
            layer_gains=tuple(float(value) for value in raw["layer_gains"]),
            training_mode=str(raw["training_mode"]),
            source_fingerprint=str(raw["source_fingerprint"]),
            sample_count=int(raw["sample_count"]),
            description=str(raw["description"]),
        )
        if artifact_id != artifact.artifact_id:
            raise ValueError("control-basis artifact_id does not match payload")
        return artifact


def build_sinusoid_control_basis(
    *,
    hidden_size: int,
    rank: int,
) -> tuple[tuple[float, ...], ...]:
    """Build the deterministic diagnostic basis used by frozen runtimes."""

    if hidden_size < 1 or rank < 1 or rank > hidden_size:
        raise ValueError(
            "sinusoid control basis requires 1 <= rank <= hidden_size"
        )
    rows = []
    for row_index in range(rank):
        factor = float(row_index + 1)
        row = tuple(
            math.sin((position + 1.0) * 0.173 * factor)
            + math.cos((position + 1.0) * 0.117 * (factor + 1.0))
            for position in range(hidden_size)
        )
        rows.append(_normalized(row, context=f"sinusoid-row-{row_index}"))
    return tuple(rows)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalized(vector: Sequence[float], *, context: str) -> tuple[float, ...]:
    norm = _norm(vector)
    if not math.isfinite(norm) or norm < _DEGENERATE_NORM:
        raise ValueError(
            f"control basis extraction produced a degenerate {context} "
            f"vector (norm={norm!r}); the transition corpus does not span "
            "the requested rank"
        )
    return tuple(value / norm for value in vector)


def _project_out(
    vector: Sequence[float],
    directions: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    result = list(vector)
    for direction in directions:
        coefficient = _dot(result, direction)
        for index, component in enumerate(direction):
            result[index] -= coefficient * component
    return tuple(result)


def _deterministic_start_vector(size: int, *, component_index: int) -> tuple[float, ...]:
    # Linear congruential generator with fixed seed: identical across
    # platforms and Python versions, unlike random.Random float rounding
    # of numpy/torch RNG streams.
    state = 0x9E3779B9 ^ (component_index + 1)
    values = []
    for _ in range(size):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        values.append(((state >> 11) / float(1 << 53)) - 0.5)
    return tuple(values)


def fit_transition_control_basis(
    transition_deltas: Sequence[Sequence[float]],
    *,
    basis_rank: int = 3,
) -> tuple[tuple[float, ...], ...]:
    """Fit an orthonormal control basis from hidden-state transition deltas.

    ``transition_deltas`` must contain at least ``basis_rank + 1`` rows of
    identical width (the hidden size). Returns ``basis_rank`` orthonormal
    rows as described in the module docstring. Deterministic for a fixed
    input corpus.
    """

    if basis_rank < 1:
        raise ValueError(f"basis_rank must be >= 1, got {basis_rank}")
    rows = [tuple(float(value) for value in delta) for delta in transition_deltas]
    if len(rows) < basis_rank + 1:
        raise ValueError(
            "control basis extraction requires at least "
            f"{basis_rank + 1} transition deltas, got {len(rows)}"
        )
    width = len(rows[0])
    if width < basis_rank:
        raise ValueError(
            f"transition delta width {width} cannot support basis rank {basis_rank}"
        )
    for row_index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                "transition deltas must share one width: row "
                f"{row_index} has {len(row)}, expected {width}"
            )
        if not all(math.isfinite(value) for value in row):
            raise ValueError(
                f"transition delta row {row_index} contains non-finite values"
            )

    sample_count = len(rows)
    mean_delta = tuple(
        sum(row[index] for row in rows) / sample_count for index in range(width)
    )
    basis: list[tuple[float, ...]] = [
        _normalized(mean_delta, context="mean-transition")
    ]
    if basis_rank == 1:
        return tuple(basis)

    # Center the corpus and remove the mean-direction component so the
    # principal components are orthogonal to row 0 by construction.
    centered = [
        _project_out(
            tuple(row[index] - mean_delta[index] for index in range(width)),
            basis,
        )
        for row in rows
    ]

    # Gram-trick PCA: eigenvectors of the (sample x sample) Gram matrix
    # map back to feature space, avoiding a width x width covariance.
    gram = [
        [_dot(centered[i], centered[j]) for j in range(sample_count)]
        for i in range(sample_count)
    ]
    for component_index in range(basis_rank - 1):
        vector = list(
            _deterministic_start_vector(
                sample_count,
                component_index=component_index,
            )
        )
        for _ in range(_POWER_ITERATIONS):
            product = [
                sum(gram[i][j] * vector[j] for j in range(sample_count))
                for i in range(sample_count)
            ]
            norm = _norm(product)
            if norm < _DEGENERATE_NORM:
                raise ValueError(
                    "control basis extraction found no remaining variance "
                    f"for principal component {component_index + 1}; the "
                    "transition corpus does not span the requested rank"
                )
            vector = [value / norm for value in product]
        eigenvalue = _dot(
            vector,
            [
                sum(gram[i][j] * vector[j] for j in range(sample_count))
                for i in range(sample_count)
            ],
        )
        feature_vector = tuple(
            sum(vector[i] * centered[i][index] for i in range(sample_count))
            for index in range(width)
        )
        # Explicit re-orthogonalization guards against numerical drift in
        # the Gram-space iteration.
        feature_vector = _project_out(feature_vector, basis)
        feature_vector = _normalized(
            feature_vector,
            context=f"principal-component-{component_index + 1}",
        )
        max_abs_index = max(
            range(width), key=lambda index: abs(feature_vector[index])
        )
        if feature_vector[max_abs_index] < 0.0:
            feature_vector = tuple(-value for value in feature_vector)
        basis.append(feature_vector)
        # Deflate the Gram matrix for the next component.
        for i in range(sample_count):
            for j in range(sample_count):
                gram[i][j] -= eigenvalue * vector[i] * vector[j]
    return tuple(basis)


def control_basis_fingerprint(basis: Sequence[Sequence[float]]) -> str:
    """Stable sha256 fingerprint of a control basis for manifest provenance."""

    digest = hashlib.sha256()
    digest.update(f"rows={len(basis)}".encode("utf-8"))
    for row in basis:
        digest.update(f";width={len(row)};".encode("utf-8"))
        digest.update(
            ",".join(f"{float(value):.8e}" for value in row).encode("utf-8")
        )
    return digest.hexdigest()
