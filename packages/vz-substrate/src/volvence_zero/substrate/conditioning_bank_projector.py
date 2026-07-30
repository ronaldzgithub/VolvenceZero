"""Bounded substrate projection for generic conditioning-bank carriers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from volvence_zero.conditioning_bank_contracts import (
    ConditioningBankLatentCarrier,
    ConditioningBankType,
)

RELATIONSHIP_CONDITIONING_PROJECTOR_SCHEMA_VERSION = (
    "relationship-conditioning-projector.v1"
)
RELATIONSHIP_CONTRASTIVE_RESIDUAL_TRAINING_MODE = (
    "relationship-contrastive-residual-v1"
)
RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION = (
    "relationship-conditioning-residual.v2"
)
RELATIONSHIP_RESIDUAL_DEFAULT_SCALE = 0.12


@dataclass(frozen=True)
class RelationshipConditioningProjectorArtifact:
    """Float-only Relationship projector for one frozen substrate."""

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
        if (
            self.schema_version
            != RELATIONSHIP_CONDITIONING_PROJECTOR_SCHEMA_VERSION
        ):
            raise ValueError(
                "Relationship projector schema_version must be "
                f"{RELATIONSHIP_CONDITIONING_PROJECTOR_SCHEMA_VERSION!r}."
            )
        if not self.model_id.strip():
            raise ValueError("Relationship projector model_id must be non-empty.")
        if self.hidden_size <= 0:
            raise ValueError(
                "Relationship projector hidden_size must be positive."
            )
        if not self.vector_labels:
            raise ValueError(
                "Relationship projector vector_labels must be non-empty."
            )
        if len(set(self.vector_labels)) != len(self.vector_labels):
            raise ValueError(
                "Relationship projector vector_labels must be unique."
            )
        if any(not label.strip() for label in self.vector_labels):
            raise ValueError(
                "Relationship projector vector_labels must be non-empty."
            )
        if len(self.basis_rows) != len(self.vector_labels):
            raise ValueError(
                "Relationship projector must carry one basis row per "
                "conditioning coordinate."
            )
        if any(len(row) != self.hidden_size for row in self.basis_rows):
            raise ValueError(
                "every Relationship projector basis row must match hidden_size."
            )
        if any(
            not math.isfinite(value)
            for row in self.basis_rows
            for value in row
        ):
            raise ValueError(
                "Relationship projector basis rows must contain finite floats."
            )
        for row in self.basis_rows:
            norm = math.sqrt(sum(value * value for value in row))
            if not 0.999 <= norm <= 1.001:
                raise ValueError(
                    "Relationship projector basis rows must be L2-normalized."
                )
        if not self.layer_indices:
            raise ValueError(
                "Relationship projector layer_indices must be non-empty."
            )
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError(
                "Relationship projector layer_indices must be unique."
            )
        if any(index < 0 for index in self.layer_indices):
            raise ValueError(
                "Relationship projector layer_indices must be non-negative."
            )
        if len(self.layer_gains) != len(self.layer_indices):
            raise ValueError(
                "Relationship projector layer_gains must align with "
                "layer_indices."
            )
        if any(not 0.0 < gain <= 1.0 for gain in self.layer_gains):
            raise ValueError(
                "Relationship projector layer gains must be in (0, 1]."
            )
        if (
            self.training_mode
            != RELATIONSHIP_CONTRASTIVE_RESIDUAL_TRAINING_MODE
        ):
            raise ValueError(
                "unsupported Relationship projector training_mode "
                f"{self.training_mode!r}; expected "
                f"{RELATIONSHIP_CONTRASTIVE_RESIDUAL_TRAINING_MODE!r}."
            )
        if not self.source_fingerprint.strip():
            raise ValueError(
                "Relationship projector source_fingerprint must be non-empty."
            )
        if self.sample_count <= 0:
            raise ValueError(
                "Relationship projector sample_count must be positive."
            )
        if not self.description.strip():
            raise ValueError(
                "Relationship projector description must be non-empty."
            )

    @property
    def artifact_id(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def projector_version(self) -> str:
        return (
            f"{RELATIONSHIP_CONTRASTIVE_RESIDUAL_TRAINING_MODE}:"
            f"{self.artifact_id}"
        )

    def to_json(self) -> str:
        payload = asdict(self)
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "RelationshipConditioningProjectorArtifact":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError(
                "Relationship projector artifact must be a JSON object."
            )
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
                "Relationship projector artifact fields do not match the "
                f"frozen schema; missing={missing}, extra={extra}"
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
                "Relationship projector artifact_id does not match its "
                "canonical payload."
            )
        return artifact


def build_relationship_contrastive_projector_artifact(
    *,
    model_id: str,
    hidden_size: int,
    vector_labels: Sequence[str],
    layer_indices: Sequence[int],
    contrastive_rows: Mapping[str, Sequence[float]],
    source_fingerprint: str,
    sample_count: int,
) -> RelationshipConditioningProjectorArtifact:
    """Normalize frozen-model contrastive rows into a bounded artifact."""

    labels = tuple(str(label) for label in vector_labels)
    missing = [label for label in labels if label not in contrastive_rows]
    extra = sorted(set(contrastive_rows) - set(labels))
    if missing or extra:
        raise ValueError(
            "Relationship contrastive rows must match vector_labels; "
            f"missing={missing}, extra={extra}"
        )
    rows: list[tuple[float, ...]] = []
    for label in labels:
        row = tuple(float(value) for value in contrastive_rows[label])
        if len(row) != hidden_size:
            raise ValueError(
                f"Relationship contrastive row {label!r} has width "
                f"{len(row)}; expected {hidden_size}"
            )
        norm = math.sqrt(sum(value * value for value in row))
        if norm <= 1e-8:
            raise ValueError(
                f"Relationship contrastive row {label!r} has zero norm."
            )
        rows.append(tuple(value / norm for value in row))
    layers = tuple(int(index) for index in layer_indices)
    return RelationshipConditioningProjectorArtifact(
        schema_version=RELATIONSHIP_CONDITIONING_PROJECTOR_SCHEMA_VERSION,
        model_id=model_id,
        hidden_size=hidden_size,
        vector_labels=labels,
        basis_rows=tuple(rows),
        layer_indices=layers,
        layer_gains=tuple(1.0 for _ in layers),
        training_mode=RELATIONSHIP_CONTRASTIVE_RESIDUAL_TRAINING_MODE,
        source_fingerprint=source_fingerprint,
        sample_count=sample_count,
        description=(
            "Relationship projector baked from frozen-substrate contrastive "
            f"residuals; rows={len(rows)} layers={layers}."
        ),
    )


def load_relationship_projector_basis(
    *,
    torch_module: Any,
    artifact: RelationshipConditioningProjectorArtifact,
    expected_model_id: str,
    expected_hidden_size: int,
    available_layer_indices: Sequence[int],
    device: Any,
) -> tuple[Any, dict[int, float]]:
    """Validate runtime compatibility and materialize one artifact."""

    if artifact.model_id != expected_model_id:
        raise ValueError(
            f"Relationship projector model_id {artifact.model_id!r} does not "
            f"match runtime {expected_model_id!r}."
        )
    if artifact.hidden_size != expected_hidden_size:
        raise ValueError(
            f"Relationship projector hidden_size {artifact.hidden_size} does "
            f"not match runtime {expected_hidden_size}."
        )
    available = set(int(index) for index in available_layer_indices)
    missing_layers = [
        index for index in artifact.layer_indices if index not in available
    ]
    if missing_layers:
        raise ValueError(
            "Relationship projector targets layers not hooked by this runtime: "
            f"{missing_layers}; available={sorted(available)}"
        )
    basis = torch_module.tensor(
        artifact.basis_rows,
        dtype=torch_module.float32,
        device=device,
    )
    gains = dict(zip(artifact.layer_indices, artifact.layer_gains, strict=True))
    return basis, gains


def build_conditioning_bank_residual_basis(
    *,
    torch_module: Any,
    hidden_size: int,
    vector_dim: int,
    device: Any | None = None,
):
    """Build the deterministic, row-normalized v1 bank projection basis."""

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")
    if vector_dim <= 0:
        raise ValueError("vector_dim must be positive.")
    positions = torch_module.arange(hidden_size, dtype=torch_module.float32)
    rows = []
    for factor in range(1, vector_dim + 1):
        row = torch_module.sin(
            (positions + 1.0) * 0.071 * (factor + 1.0)
        ) + torch_module.cos(
            (positions + 1.0) * 0.043 * (factor + 3.0)
        )
        rows.append(row / row.norm().clamp_min(1e-6))
    basis = torch_module.stack(rows, dim=0)
    return basis.to(device) if device is not None else basis


def build_conditioning_bank_residual_delta(
    *,
    torch_module: Any,
    carrier: ConditioningBankLatentCarrier,
    hidden_size: int,
    device: Any | None = None,
    basis: Any | None = None,
    vector_labels: tuple[str, ...] | None = None,
    expected_projector_version: str = RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION,
):
    """Project one admitted Relationship bank into a bounded hidden delta."""

    if carrier.bank.bank_type is not ConditioningBankType.RELATIONSHIP:
        raise ValueError(
            "conditioning-bank residual projector v1 supports only the "
            f"RELATIONSHIP bank, got {carrier.bank.bank_type.value!r}."
        )
    if carrier.projector_version != expected_projector_version:
        raise ValueError(
            "unsupported Relationship residual projector version "
            f"{carrier.projector_version!r}; expected "
            f"{expected_projector_version!r}."
        )
    if vector_labels is not None and carrier.bank.readout_labels != vector_labels:
        raise ValueError(
            "Relationship projector vector_labels do not match the admitted "
            "Relationship bank readout_labels."
        )
    if basis is None:
        basis = build_conditioning_bank_residual_basis(
            torch_module=torch_module,
            hidden_size=hidden_size,
            vector_dim=len(carrier.bank.readout),
            device=device,
        )
    if tuple(basis.shape) != (len(carrier.bank.readout), hidden_size):
        raise ValueError(
            "Relationship projector basis shape does not match the admitted "
            f"bank and runtime: got {tuple(basis.shape)}, expected "
            f"{(len(carrier.bank.readout), hidden_size)}."
        )
    # Owner coordinates are bounded probabilities/readouts where 0.5 is the
    # neutral point. Centering is essential: an uncentered projection is
    # dominated by the large common positive component shared by repair and
    # steady states, which makes distinct Relationship snapshots produce the
    # same steering direction.
    state = torch_module.tensor(
        tuple(2.0 * value - 1.0 for value in carrier.bank.readout),
        dtype=torch_module.float32,
        device=device,
    )
    if not bool((state.abs() > 1e-8).any().item()):
        return None
    delta = state @ basis
    norm = delta.norm()
    if float(norm.item()) <= 1e-8:
        return None
    delta = delta / norm
    return (
        delta
        * float(carrier.scale)
        * float(carrier.bank.confidence)
        * float(carrier.bank.freshness)
    )


__all__ = [
    "RELATIONSHIP_CONDITIONING_PROJECTOR_SCHEMA_VERSION",
    "RELATIONSHIP_CONTRASTIVE_RESIDUAL_TRAINING_MODE",
    "RELATIONSHIP_RESIDUAL_DEFAULT_SCALE",
    "RELATIONSHIP_RESIDUAL_PROJECTOR_VERSION",
    "RelationshipConditioningProjectorArtifact",
    "build_relationship_contrastive_projector_artifact",
    "build_conditioning_bank_residual_basis",
    "build_conditioning_bank_residual_delta",
    "load_relationship_projector_basis",
]
