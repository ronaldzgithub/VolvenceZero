"""Versioned personal-conditioning projector artifacts.

The personal-conditioning owner publishes a frozen 16-dimensional semantic
readout. This module owns the substrate-side mapping from that readout into a
frozen model's hidden width. The default runtime still uses the deterministic
sine/cosine basis; a learned artifact is an explicit, reversible override.

Artifacts contain floats and metadata only. They never carry dialogue, user
facts, or torch tensors, and loading one does not mutate base-model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)

PERSONAL_CONDITIONING_PROJECTOR_SCHEMA_VERSION = (
    "personal-conditioning-projector.v1"
)
CONTRASTIVE_RESIDUAL_TRAINING_MODE = "contrastive-residual-v1"


@dataclass(frozen=True)
class PersonalConditioningProjectorArtifact:
    """Float-only projector compatible with one frozen substrate."""

    schema_version: str
    model_id: str
    hidden_size: int
    vector_labels: tuple[str, ...]
    basis_rows: tuple[tuple[float, ...], ...]
    layer_indices: tuple[int, ...]
    layer_gains: tuple[float, ...]
    training_mode: str
    source_fingerprint: str
    sample_count: int
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != PERSONAL_CONDITIONING_PROJECTOR_SCHEMA_VERSION:
            raise ValueError(
                "projector schema_version must be "
                f"{PERSONAL_CONDITIONING_PROJECTOR_SCHEMA_VERSION!r}."
            )
        if not self.model_id.strip():
            raise ValueError("projector model_id must be non-empty.")
        if self.hidden_size <= 0:
            raise ValueError("projector hidden_size must be positive.")
        if self.vector_labels != PERSONAL_CONDITIONING_VECTOR_LABELS:
            raise ValueError(
                "projector vector_labels must match personal-conditioning.v1."
            )
        if len(self.basis_rows) != len(self.vector_labels):
            raise ValueError(
                "projector must carry one basis row per conditioning coordinate."
            )
        if any(len(row) != self.hidden_size for row in self.basis_rows):
            raise ValueError(
                "every projector basis row must match hidden_size."
            )
        if any(
            not math.isfinite(value)
            for row in self.basis_rows
            for value in row
        ):
            raise ValueError("projector basis rows must contain finite floats.")
        for row in self.basis_rows:
            norm = math.sqrt(sum(value * value for value in row))
            if not 0.999 <= norm <= 1.001:
                raise ValueError(
                    "projector basis rows must be L2-normalized so the existing "
                    "personal-conditioning scale cap remains meaningful."
                )
        if not self.layer_indices:
            raise ValueError("projector layer_indices must be non-empty.")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("projector layer_indices must be unique.")
        if any(index < 0 for index in self.layer_indices):
            raise ValueError("projector layer_indices must be non-negative.")
        if len(self.layer_gains) != len(self.layer_indices):
            raise ValueError(
                "projector layer_gains must align with layer_indices."
            )
        if any(not 0.0 < gain <= 1.0 for gain in self.layer_gains):
            raise ValueError("projector layer gains must be in (0, 1].")
        if self.training_mode != CONTRASTIVE_RESIDUAL_TRAINING_MODE:
            raise ValueError(
                "unsupported projector training_mode "
                f"{self.training_mode!r}; expected "
                f"{CONTRASTIVE_RESIDUAL_TRAINING_MODE!r}."
            )
        if not self.source_fingerprint.strip():
            raise ValueError("projector source_fingerprint must be non-empty.")
        if self.sample_count <= 0:
            raise ValueError("projector sample_count must be positive.")
        if not self.description.strip():
            raise ValueError("projector description must be non-empty.")

    @property
    def artifact_id(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        payload = asdict(self)
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "PersonalConditioningProjectorArtifact":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("projector artifact must be a JSON object.")
        required_fields = {
            "artifact_id",
            "schema_version",
            "model_id",
            "hidden_size",
            "vector_labels",
            "basis_rows",
            "layer_indices",
            "layer_gains",
            "training_mode",
            "source_fingerprint",
            "sample_count",
            "description",
        }
        missing = sorted(required_fields - set(raw))
        extra = sorted(set(raw) - required_fields)
        if missing or extra:
            raise ValueError(
                "projector artifact fields do not match the frozen schema; "
                f"missing={missing}, extra={extra}"
            )
        declared_id = str(raw.pop("artifact_id"))
        artifact = cls(
            schema_version=str(raw["schema_version"]),
            model_id=str(raw["model_id"]),
            hidden_size=int(raw["hidden_size"]),
            vector_labels=tuple(str(value) for value in raw["vector_labels"]),
            basis_rows=tuple(
                tuple(float(value) for value in row)
                for row in raw["basis_rows"]
            ),
            layer_indices=tuple(int(value) for value in raw["layer_indices"]),
            layer_gains=tuple(float(value) for value in raw["layer_gains"]),
            training_mode=str(raw["training_mode"]),
            source_fingerprint=str(raw["source_fingerprint"]),
            sample_count=int(raw["sample_count"]),
            description=str(raw["description"]),
        )
        if declared_id != artifact.artifact_id:
            raise ValueError(
                "projector artifact_id does not match its canonical payload."
            )
        return artifact


def build_contrastive_projector_artifact(
    *,
    model_id: str,
    hidden_size: int,
    layer_indices: Sequence[int],
    contrastive_rows: Mapping[str, Sequence[float]],
    source_fingerprint: str,
    sample_count: int,
) -> PersonalConditioningProjectorArtifact:
    """Normalize model-derived contrastive rows into a bounded artifact."""

    missing = [
        label
        for label in PERSONAL_CONDITIONING_VECTOR_LABELS
        if label not in contrastive_rows
    ]
    extra = sorted(set(contrastive_rows) - set(PERSONAL_CONDITIONING_VECTOR_LABELS))
    if missing or extra:
        raise ValueError(
            "contrastive projector rows must match the frozen coordinate set; "
            f"missing={missing}, extra={extra}"
        )
    rows: list[tuple[float, ...]] = []
    for label in PERSONAL_CONDITIONING_VECTOR_LABELS:
        row = tuple(float(value) for value in contrastive_rows[label])
        if len(row) != hidden_size:
            raise ValueError(
                f"contrastive row {label!r} has width {len(row)}; "
                f"expected {hidden_size}"
            )
        norm = math.sqrt(sum(value * value for value in row))
        if norm <= 1e-8:
            raise ValueError(
                f"contrastive row {label!r} has zero norm; the substrate "
                "did not distinguish its positive and negative material."
            )
        rows.append(tuple(value / norm for value in row))
    layers = tuple(int(index) for index in layer_indices)
    return PersonalConditioningProjectorArtifact(
        schema_version=PERSONAL_CONDITIONING_PROJECTOR_SCHEMA_VERSION,
        model_id=model_id,
        hidden_size=hidden_size,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        basis_rows=tuple(rows),
        layer_indices=layers,
        layer_gains=tuple(1.0 for _ in layers),
        training_mode=CONTRASTIVE_RESIDUAL_TRAINING_MODE,
        source_fingerprint=source_fingerprint,
        sample_count=sample_count,
        description=(
            "Personal-conditioning projector baked from frozen-substrate "
            f"contrastive residuals; rows={len(rows)} layers={layers}."
        ),
    )


def load_projector_basis(
    *,
    torch_module: Any,
    artifact: PersonalConditioningProjectorArtifact,
    expected_model_id: str,
    expected_hidden_size: int,
    available_layer_indices: Sequence[int],
    device: Any,
) -> tuple[Any, dict[int, float]]:
    """Validate compatibility and materialize the artifact for one runtime."""

    if artifact.model_id != expected_model_id:
        raise ValueError(
            f"projector model_id {artifact.model_id!r} does not match runtime "
            f"{expected_model_id!r}."
        )
    if artifact.hidden_size != expected_hidden_size:
        raise ValueError(
            f"projector hidden_size {artifact.hidden_size} does not match "
            f"runtime {expected_hidden_size}."
        )
    available = set(int(index) for index in available_layer_indices)
    missing_layers = [
        index for index in artifact.layer_indices if index not in available
    ]
    if missing_layers:
        raise ValueError(
            "projector targets layers not hooked by this runtime: "
            f"{missing_layers}; available={sorted(available)}"
        )
    basis = torch_module.tensor(
        artifact.basis_rows,
        dtype=torch_module.float32,
        device=device,
    )
    gains = dict(zip(artifact.layer_indices, artifact.layer_gains, strict=True))
    return basis, gains


__all__ = [
    "CONTRASTIVE_RESIDUAL_TRAINING_MODE",
    "PERSONAL_CONDITIONING_PROJECTOR_SCHEMA_VERSION",
    "PersonalConditioningProjectorArtifact",
    "build_contrastive_projector_artifact",
    "load_projector_basis",
]
