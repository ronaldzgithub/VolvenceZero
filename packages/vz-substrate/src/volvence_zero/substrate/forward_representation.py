"""Frozen substrate-owned representation targets for N+1 prediction.

The substrate owns both the model capture and the interpretation of its
residual geometry.  Prediction-error consumers receive this immutable batch
snapshot (or its lineage), never raw text and never a second sentence encoder
that silently becomes another representation owner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import struct

from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime
from volvence_zero.substrate.substrate_fingerprint import SubstrateFingerprint


SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION = (
    "substrate-forward-representation.v1"
)
SUBSTRATE_FORWARD_READOUT_KIND = "latest-token-selected-layer-residual-l2.v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_vector(values: tuple[float, ...]) -> str:
    payload = struct.pack(f"!{len(values)}d", *values)
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class SubstrateForwardRepresentation:
    sample_id: str
    source_sha256: str
    values: tuple[float, ...]
    values_sha256: str

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("substrate representation sample_id must be non-empty")
        if not _is_sha256(self.source_sha256):
            raise ValueError("substrate representation source_sha256 is invalid")
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise ValueError(
                "substrate representation values must be non-empty and finite"
            )
        norm = math.sqrt(sum(value * value for value in self.values))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("substrate representation values must be L2-normalized")
        if self.values_sha256 != _sha256_vector(self.values):
            raise ValueError("substrate representation values_sha256 mismatch")


@dataclass(frozen=True)
class SubstrateForwardRepresentationLineage:
    schema_version: str
    snapshot_fingerprint: str
    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    readout_kind: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int

    def __post_init__(self) -> None:
        if self.schema_version != SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION:
            raise ValueError("substrate representation lineage schema_version mismatch")
        if not _is_sha256(self.snapshot_fingerprint):
            raise ValueError(
                "substrate representation lineage snapshot_fingerprint is invalid"
            )
        if not self.runtime_origin.strip():
            raise ValueError("substrate representation runtime_origin must be non-empty")
        if self.readout_kind != SUBSTRATE_FORWARD_READOUT_KIND:
            raise ValueError("substrate representation readout_kind mismatch")
        if not self.layer_indices or tuple(sorted(self.layer_indices)) != self.layer_indices:
            raise ValueError(
                "substrate representation layer_indices must be non-empty and sorted"
            )
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("substrate representation layer_indices must be unique")
        if len(self.activation_widths) != len(self.layer_indices) or any(
            width < 1 for width in self.activation_widths
        ):
            raise ValueError("substrate representation activation widths are invalid")
        if self.representation_dim != sum(self.activation_widths):
            raise ValueError("substrate representation dimension/geometry mismatch")


@dataclass(frozen=True)
class SubstrateForwardRepresentationSnapshot:
    lineage: SubstrateForwardRepresentationLineage
    representations: tuple[SubstrateForwardRepresentation, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.representations:
            raise ValueError("substrate representation snapshot must contain samples")
        sample_ids = tuple(row.sample_id for row in self.representations)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("substrate representation sample_ids must be unique")
        if any(
            len(row.values) != self.lineage.representation_dim
            for row in self.representations
        ):
            raise ValueError("substrate representation row dimension mismatch")
        if not self.description.strip():
            raise ValueError("substrate representation description must be non-empty")


def _snapshot_fingerprint(
    *,
    model_fingerprint: SubstrateFingerprint,
    runtime_origin: str,
    layer_indices: tuple[int, ...],
    activation_widths: tuple[int, ...],
    representations: tuple[SubstrateForwardRepresentation, ...],
) -> str:
    payload = {
        "schema_version": SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
        "model_fingerprint": {
            "model_id": model_fingerprint.model_id,
            "version": model_fingerprint.version,
            "weights_sha256": model_fingerprint.weights_sha256,
        },
        "runtime_origin": runtime_origin,
        "readout_kind": SUBSTRATE_FORWARD_READOUT_KIND,
        "layer_indices": layer_indices,
        "activation_widths": activation_widths,
        "representations": tuple(
            (row.sample_id, row.source_sha256, row.values_sha256)
            for row in representations
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class SubstrateForwardRepresentationPublisher:
    """Publish unconditioned frozen-LM residual targets with strict lineage."""

    def __init__(
        self,
        runtime: OpenWeightResidualRuntime,
        *,
        model_fingerprint: SubstrateFingerprint,
    ) -> None:
        if not runtime.is_frozen:
            raise ValueError("substrate representation runtime must be frozen")
        if runtime.model_id != model_fingerprint.model_id:
            raise ValueError(
                "substrate representation runtime/model fingerprint mismatch: "
                f"{runtime.model_id!r} != {model_fingerprint.model_id!r}"
            )
        if not _is_sha256(model_fingerprint.weights_sha256):
            raise ValueError(
                "substrate representation requires a full model weights SHA-256"
            )
        runtime_origin = runtime.runtime_origin
        if not isinstance(runtime_origin, str) or not runtime_origin.strip():
            raise ValueError("substrate representation runtime_origin must be explicit")
        self._runtime = runtime
        self._model_fingerprint = model_fingerprint
        self._runtime_origin = runtime_origin

    def publish(
        self,
        sample_sources: tuple[tuple[str, str], ...],
        *,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> SubstrateForwardRepresentationSnapshot:
        if not sample_sources:
            raise ValueError("substrate representation publish requires samples")
        sample_ids = tuple(sample_id for sample_id, _ in sample_sources)
        if any(not sample_id.strip() for sample_id in sample_ids):
            raise ValueError("substrate representation sample ids must be non-empty")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("substrate representation sample ids must be unique")

        geometry: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        rows: list[SubstrateForwardRepresentation] = []
        for sample_index, (sample_id, source_text) in enumerate(sample_sources):
            if not source_text.strip():
                raise ValueError(
                    f"substrate representation source for {sample_id!r} is empty"
                )
            capture = self._runtime.capture(source_text=source_text)
            if capture.personal_conditioning_applied:
                raise ValueError(
                    "substrate N+1 target capture must be unconditioned; personal "
                    f"conditioning was applied for {sample_id!r}"
                )
            activations = tuple(
                sorted(capture.residual_activations, key=lambda row: row.layer_index)
            )
            if not activations:
                raise ValueError(
                    f"substrate target capture for {sample_id!r} has no residual activations"
                )
            layer_indices = tuple(row.layer_index for row in activations)
            if len(set(layer_indices)) != len(layer_indices):
                raise ValueError(
                    f"substrate target capture for {sample_id!r} has duplicate layers"
                )
            activation_widths = tuple(len(row.activation) for row in activations)
            if any(width < 1 for width in activation_widths):
                raise ValueError(
                    f"substrate target capture for {sample_id!r} has empty activations"
                )
            current_geometry = (layer_indices, activation_widths)
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError(
                    "substrate target residual geometry drifted across samples: "
                    f"expected {geometry}, got {current_geometry} for {sample_id!r}"
                )
            flat = tuple(value for row in activations for value in row.activation)
            if not all(math.isfinite(value) for value in flat):
                raise ValueError(
                    f"substrate target capture for {sample_id!r} is non-finite"
                )
            norm = math.sqrt(sum(value * value for value in flat))
            if norm <= 1e-12:
                raise ValueError(
                    f"substrate target capture for {sample_id!r} has zero residual norm"
                )
            values = tuple(value / norm for value in flat)
            rows.append(
                SubstrateForwardRepresentation(
                    sample_id=sample_id,
                    source_sha256=_sha256_text(source_text),
                    values=values,
                    values_sha256=_sha256_vector(values),
                )
            )
            if progress is not None:
                progress(sample_id, sample_index + 1, len(sample_sources))

        if geometry is None:
            raise RuntimeError("substrate representation geometry was not initialized")
        representations = tuple(rows)
        layer_indices, activation_widths = geometry
        fingerprint = _snapshot_fingerprint(
            model_fingerprint=self._model_fingerprint,
            runtime_origin=self._runtime_origin,
            layer_indices=layer_indices,
            activation_widths=activation_widths,
            representations=representations,
        )
        lineage = SubstrateForwardRepresentationLineage(
            schema_version=SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
            snapshot_fingerprint=fingerprint,
            model_fingerprint=self._model_fingerprint,
            runtime_origin=self._runtime_origin,
            readout_kind=SUBSTRATE_FORWARD_READOUT_KIND,
            layer_indices=layer_indices,
            activation_widths=activation_widths,
            representation_dim=sum(activation_widths),
        )
        return SubstrateForwardRepresentationSnapshot(
            lineage=lineage,
            representations=representations,
            description=(
                "Frozen substrate N+1 targets from the latest token on selected "
                "residual layers; source text is retained only as SHA-256 lineage."
            ),
        )


__all__ = [
    "SUBSTRATE_FORWARD_READOUT_KIND",
    "SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION",
    "SubstrateForwardRepresentation",
    "SubstrateForwardRepresentationLineage",
    "SubstrateForwardRepresentationPublisher",
    "SubstrateForwardRepresentationSnapshot",
]
