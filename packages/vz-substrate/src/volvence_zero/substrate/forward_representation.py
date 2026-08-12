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

from volvence_zero.substrate.residual_contracts import OpenWeightRuntimeCapture
from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime
from volvence_zero.substrate.substrate_fingerprint import SubstrateFingerprint


SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION = (
    "substrate-forward-representation.v1"
)
SUBSTRATE_FORWARD_READOUT_KIND = "latest-token-selected-layer-residual-l2.v1"
# v2 readout: per-layer L2 normalization (so no single layer dominates the
# concatenated geometry) followed by subtraction of a frozen reference mean
# (and optional frozen principal directions) fitted on a train-split corpus.
# Motivation: the raw v1 readout carries >50% of its energy along one shared
# mean direction, which compresses genuine between-sample discriminability.
SUBSTRATE_FORWARD_CENTERED_READOUT_KIND = (
    "latest-token-selected-layer-centered-residual-l2.v2"
)
SUBSTRATE_READOUT_REFERENCE_STATISTICS_SCHEMA_VERSION = (
    "substrate-readout-reference-statistics.v1"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_vector(values: tuple[float, ...]) -> str:
    payload = struct.pack(f"!{len(values)}d", *values)
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validated_layer_geometry(
    layer_indices: tuple[int, ...],
    activation_widths: tuple[int, ...],
    *,
    context: str,
) -> None:
    if not layer_indices or tuple(sorted(layer_indices)) != layer_indices:
        raise ValueError(f"{context} layer_indices must be non-empty and sorted")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"{context} layer_indices must be unique")
    if len(activation_widths) != len(layer_indices) or any(
        width < 1 for width in activation_widths
    ):
        raise ValueError(f"{context} activation widths are invalid")


def layer_normalized_readout_vector(
    values: tuple[float, ...],
    *,
    activation_widths: tuple[int, ...],
) -> tuple[float, ...]:
    """L2-normalize each per-layer block of a concatenated residual vector.

    Per-block normalization is scale invariant, so this recovers the exact
    same geometry whether ``values`` is a raw residual concatenation or an
    already globally normalized v1 readout row.
    """

    if any(width < 1 for width in activation_widths):
        raise ValueError("layer normalization requires positive widths")
    if len(values) != sum(activation_widths):
        raise ValueError(
            "layer normalization dimension mismatch: "
            f"{len(values)} values for widths {activation_widths}"
        )
    normalized: list[float] = []
    offset = 0
    for width in activation_widths:
        block = values[offset : offset + width]
        offset += width
        norm = math.sqrt(sum(value * value for value in block))
        if norm <= 1e-12:
            raise ValueError("layer normalization found a zero-norm layer block")
        normalized.extend(value / norm for value in block)
    return tuple(normalized)


def _reference_statistics_sha256(
    *,
    corpus_id: str,
    layer_indices: tuple[int, ...],
    activation_widths: tuple[int, ...],
    sample_count: int,
    mean: tuple[float, ...],
    principal_components: tuple[tuple[float, ...], ...],
) -> str:
    payload = {
        "schema_version": SUBSTRATE_READOUT_REFERENCE_STATISTICS_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "layer_indices": layer_indices,
        "activation_widths": activation_widths,
        "sample_count": sample_count,
        "mean": mean,
        "principal_components": principal_components,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SubstrateReadoutReferenceStatistics:
    """Frozen train-split centering statistics for the v2 centered readout.

    The statistics are fitted once on a frozen reference corpus (never on
    evaluation or heldout data), published as a model-bound artifact, and
    bound into every v2 lineage via ``statistics_sha256``.
    """

    schema_version: str
    corpus_id: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    sample_count: int
    mean: tuple[float, ...]
    principal_components: tuple[tuple[float, ...], ...]
    statistics_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SUBSTRATE_READOUT_REFERENCE_STATISTICS_SCHEMA_VERSION
        ):
            raise ValueError("readout reference statistics schema_version mismatch")
        if not self.corpus_id.strip():
            raise ValueError("readout reference statistics corpus_id must be non-empty")
        _validated_layer_geometry(
            self.layer_indices,
            self.activation_widths,
            context="readout reference statistics",
        )
        if self.sample_count < 2:
            raise ValueError(
                "readout reference statistics require at least two fitting samples"
            )
        dimension = sum(self.activation_widths)
        if len(self.mean) != dimension or not all(
            math.isfinite(value) for value in self.mean
        ):
            raise ValueError("readout reference statistics mean is invalid")
        for index, component in enumerate(self.principal_components):
            if len(component) != dimension or not all(
                math.isfinite(value) for value in component
            ):
                raise ValueError(
                    f"readout reference principal component {index} is invalid"
                )
            norm = math.sqrt(sum(value * value for value in component))
            if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"readout reference principal component {index} must be unit norm"
                )
            for other_index in range(index):
                other = self.principal_components[other_index]
                dot = sum(a * b for a, b in zip(component, other, strict=True))
                if abs(dot) > 1e-6:
                    raise ValueError(
                        "readout reference principal components must be orthogonal"
                    )
        expected = _reference_statistics_sha256(
            corpus_id=self.corpus_id,
            layer_indices=self.layer_indices,
            activation_widths=self.activation_widths,
            sample_count=self.sample_count,
            mean=self.mean,
            principal_components=self.principal_components,
        )
        if self.statistics_sha256 != expected:
            raise ValueError("readout reference statistics_sha256 mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "layer_indices": list(self.layer_indices),
            "activation_widths": list(self.activation_widths),
            "sample_count": self.sample_count,
            "mean": list(self.mean),
            "principal_components": [
                list(component) for component in self.principal_components
            ],
            "statistics_sha256": self.statistics_sha256,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, object]
    ) -> "SubstrateReadoutReferenceStatistics":
        return cls(
            schema_version=str(payload["schema_version"]),
            corpus_id=str(payload["corpus_id"]),
            layer_indices=tuple(
                int(value) for value in payload["layer_indices"]  # type: ignore[union-attr]
            ),
            activation_widths=tuple(
                int(value) for value in payload["activation_widths"]  # type: ignore[union-attr]
            ),
            sample_count=int(payload["sample_count"]),  # type: ignore[arg-type]
            mean=tuple(float(value) for value in payload["mean"]),  # type: ignore[union-attr]
            principal_components=tuple(
                tuple(float(value) for value in component)
                for component in payload["principal_components"]  # type: ignore[union-attr]
            ),
            statistics_sha256=str(payload["statistics_sha256"]),
        )

    def apply(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Center one layer-normalized vector against the frozen statistics."""

        dimension = sum(self.activation_widths)
        if len(values) != dimension:
            raise ValueError(
                "readout reference statistics dimension mismatch: "
                f"{len(values)} values for dimension {dimension}"
            )
        centered = [value - offset for value, offset in zip(values, self.mean, strict=True)]
        for component in self.principal_components:
            projection = sum(
                a * b for a, b in zip(centered, component, strict=True)
            )
            centered = [
                value - projection * axis
                for value, axis in zip(centered, component, strict=True)
            ]
        norm = math.sqrt(sum(value * value for value in centered))
        if norm <= 1e-12:
            raise ValueError(
                "centered readout collapsed to zero norm; the sample coincides "
                "with the frozen reference statistics"
            )
        return tuple(value / norm for value in centered)


def _deterministic_unit_vector(dimension: int, *, seed_label: str) -> list[float]:
    """Deterministic pseudo-random unit vector for power-iteration init."""

    state = int.from_bytes(
        hashlib.sha256(seed_label.encode("utf-8")).digest()[:8], "big"
    )
    values: list[float] = []
    for _ in range(dimension):
        # 64-bit LCG (Knuth MMIX constants): deterministic across platforms.
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        values.append((state / float(1 << 64)) - 0.5)
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def fit_forward_readout_reference_statistics(
    *,
    corpus_id: str,
    layer_indices: tuple[int, ...],
    activation_widths: tuple[int, ...],
    vectors: tuple[tuple[float, ...], ...],
    principal_component_count: int = 0,
    power_iterations: int = 100,
) -> SubstrateReadoutReferenceStatistics:
    """Fit frozen centering statistics on a reference corpus.

    ``vectors`` are concatenated per-layer residual readouts (raw or v1
    globally normalized rows); per-layer normalization is applied here so the
    substrate remains the single owner of the readout geometry. The corpus
    must be a frozen train-split reference — fitting on evaluation or heldout
    data leaks the judgment surface and is forbidden by contract.
    """

    _validated_layer_geometry(
        layer_indices, activation_widths, context="readout reference fit"
    )
    if len(vectors) < 2:
        raise ValueError("readout reference fit requires at least two vectors")
    if principal_component_count < 0:
        raise ValueError("principal_component_count must be non-negative")
    if power_iterations < 1:
        raise ValueError("power_iterations must be positive")
    normalized = [
        list(
            layer_normalized_readout_vector(
                vector, activation_widths=activation_widths
            )
        )
        for vector in vectors
    ]
    dimension = sum(activation_widths)
    count = len(normalized)
    mean = tuple(
        sum(vector[index] for vector in normalized) / count
        for index in range(dimension)
    )
    centered = [
        [value - offset for value, offset in zip(vector, mean, strict=True)]
        for vector in normalized
    ]
    components: list[tuple[float, ...]] = []
    for component_index in range(principal_component_count):
        axis = _deterministic_unit_vector(
            dimension, seed_label=f"{corpus_id}:pc:{component_index}"
        )
        for _ in range(power_iterations):
            # One covariance matvec: X^T (X axis), without forming X^T X.
            projections = [
                sum(a * b for a, b in zip(row, axis, strict=True))
                for row in centered
            ]
            updated = [0.0] * dimension
            for row, projection in zip(centered, projections, strict=True):
                for index, value in enumerate(row):
                    updated[index] += projection * value
            # Gram-Schmidt against already extracted components: deflation
            # alone leaves numeric residue above the contract's 1e-6
            # orthogonality tolerance.
            for previous in components:
                projection = sum(
                    a * b for a, b in zip(updated, previous, strict=True)
                )
                for index, axis_value in enumerate(previous):
                    updated[index] -= projection * axis_value
            norm = math.sqrt(sum(value * value for value in updated))
            if norm <= 1e-12:
                raise ValueError(
                    "readout reference fit found no variance for principal "
                    f"component {component_index}"
                )
            axis = [value / norm for value in updated]
        unit = tuple(axis)
        components.append(unit)
        for row in centered:
            projection = sum(a * b for a, b in zip(row, unit, strict=True))
            for index, axis_value in enumerate(unit):
                row[index] -= projection * axis_value
    principal_components = tuple(components)
    return SubstrateReadoutReferenceStatistics(
        schema_version=SUBSTRATE_READOUT_REFERENCE_STATISTICS_SCHEMA_VERSION,
        corpus_id=corpus_id,
        layer_indices=layer_indices,
        activation_widths=activation_widths,
        sample_count=count,
        mean=mean,
        principal_components=principal_components,
        statistics_sha256=_reference_statistics_sha256(
            corpus_id=corpus_id,
            layer_indices=layer_indices,
            activation_widths=activation_widths,
            sample_count=count,
            mean=mean,
            principal_components=principal_components,
        ),
    )


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
    # v2 centered readout only: identity of the frozen reference statistics
    # the rows were centered against. Both stay None for the v1 readout.
    reference_corpus_id: str | None = None
    reference_statistics_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION:
            raise ValueError("substrate representation lineage schema_version mismatch")
        if not _is_sha256(self.snapshot_fingerprint):
            raise ValueError(
                "substrate representation lineage snapshot_fingerprint is invalid"
            )
        if not self.runtime_origin.strip():
            raise ValueError("substrate representation runtime_origin must be non-empty")
        if self.readout_kind == SUBSTRATE_FORWARD_READOUT_KIND:
            if (
                self.reference_corpus_id is not None
                or self.reference_statistics_sha256 is not None
            ):
                raise ValueError(
                    "v1 substrate representation lineage must not carry "
                    "reference statistics"
                )
        elif self.readout_kind == SUBSTRATE_FORWARD_CENTERED_READOUT_KIND:
            if self.reference_corpus_id is None or not self.reference_corpus_id.strip():
                raise ValueError(
                    "centered substrate representation lineage requires a "
                    "reference_corpus_id"
                )
            if self.reference_statistics_sha256 is None or not _is_sha256(
                self.reference_statistics_sha256
            ):
                raise ValueError(
                    "centered substrate representation lineage requires a "
                    "reference_statistics_sha256"
                )
        else:
            raise ValueError("substrate representation readout_kind mismatch")
        _validated_layer_geometry(
            self.layer_indices,
            self.activation_widths,
            context="substrate representation",
        )
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
    readout_kind: str,
    layer_indices: tuple[int, ...],
    activation_widths: tuple[int, ...],
    representations: tuple[SubstrateForwardRepresentation, ...],
    reference_statistics_sha256: str | None,
) -> str:
    payload = {
        "schema_version": SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
        "model_fingerprint": {
            "model_id": model_fingerprint.model_id,
            "version": model_fingerprint.version,
            "weights_sha256": model_fingerprint.weights_sha256,
        },
        "runtime_origin": runtime_origin,
        "readout_kind": readout_kind,
        "layer_indices": layer_indices,
        "activation_widths": activation_widths,
        "representations": tuple(
            (row.sample_id, row.source_sha256, row.values_sha256)
            for row in representations
        ),
    }
    # Key added only for the centered readout so historical v1 fingerprints
    # remain byte-for-byte reproducible.
    if reference_statistics_sha256 is not None:
        payload["reference_statistics_sha256"] = reference_statistics_sha256
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _representation_from_capture(
    *,
    sample_id: str,
    source_sha256: str,
    capture: OpenWeightRuntimeCapture,
    require_unconditioned: bool,
    reference_statistics: SubstrateReadoutReferenceStatistics | None = None,
) -> tuple[SubstrateForwardRepresentation, tuple[int, ...], tuple[int, ...]]:
    if require_unconditioned and capture.personal_conditioning_applied:
        raise ValueError(
            "substrate N+1 target capture must be unconditioned; personal "
            f"conditioning was applied for {sample_id!r}"
        )
    activations = tuple(
        sorted(capture.residual_activations, key=lambda row: row.layer_index)
    )
    if not activations:
        raise ValueError(
            f"substrate representation capture for {sample_id!r} has no "
            "residual activations"
        )
    layer_indices = tuple(row.layer_index for row in activations)
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(
            f"substrate representation capture for {sample_id!r} has duplicate layers"
        )
    activation_widths = tuple(len(row.activation) for row in activations)
    if any(width < 1 for width in activation_widths):
        raise ValueError(
            f"substrate representation capture for {sample_id!r} has empty activations"
        )
    flat = tuple(value for row in activations for value in row.activation)
    if not all(math.isfinite(value) for value in flat):
        raise ValueError(
            f"substrate representation capture for {sample_id!r} is non-finite"
        )
    if reference_statistics is None:
        # v1 readout: one global L2 normalization over the concatenation.
        norm = math.sqrt(sum(value * value for value in flat))
        if norm <= 1e-12:
            raise ValueError(
                f"substrate representation capture for {sample_id!r} has zero "
                "residual norm"
            )
        values = tuple(value / norm for value in flat)
    else:
        # v2 readout: per-layer normalization, then centering against the
        # frozen reference statistics, then one final normalization.
        if (
            reference_statistics.layer_indices != layer_indices
            or reference_statistics.activation_widths != activation_widths
        ):
            raise ValueError(
                "substrate representation reference statistics geometry "
                f"mismatch for {sample_id!r}: statistics were fitted on "
                f"layers {reference_statistics.layer_indices} widths "
                f"{reference_statistics.activation_widths}, capture has "
                f"layers {layer_indices} widths {activation_widths}"
            )
        values = reference_statistics.apply(
            layer_normalized_readout_vector(
                flat, activation_widths=activation_widths
            )
        )
    return (
        SubstrateForwardRepresentation(
            sample_id=sample_id,
            source_sha256=source_sha256,
            values=values,
            values_sha256=_sha256_vector(values),
        ),
        layer_indices,
        activation_widths,
    )


def publish_runtime_capture_representation(
    *,
    sample_id: str,
    source_sha256: str,
    capture: OpenWeightRuntimeCapture,
    model_fingerprint: SubstrateFingerprint,
    runtime_origin: str,
    reference_statistics: SubstrateReadoutReferenceStatistics | None = None,
) -> SubstrateForwardRepresentationSnapshot:
    """Publish one full-runtime conditioned context in substrate coordinates.

    Unlike N+1 targets, this evidence-only context may include bounded runtime
    conditioning. The substrate remains the sole interpreter of residual
    geometry and publishes the same L2 readout contract used by the target.
    With ``reference_statistics`` the centered v2 readout is published instead.
    """

    if not _is_sha256(source_sha256):
        raise ValueError("runtime context source_sha256 is invalid")
    if not _is_sha256(model_fingerprint.weights_sha256):
        raise ValueError("runtime context requires a full model weights SHA-256")
    if not runtime_origin.strip():
        raise ValueError("runtime context runtime_origin must be non-empty")
    row, layer_indices, activation_widths = _representation_from_capture(
        sample_id=sample_id,
        source_sha256=source_sha256,
        capture=capture,
        require_unconditioned=False,
        reference_statistics=reference_statistics,
    )
    readout_kind = (
        SUBSTRATE_FORWARD_READOUT_KIND
        if reference_statistics is None
        else SUBSTRATE_FORWARD_CENTERED_READOUT_KIND
    )
    representations = (row,)
    fingerprint = _snapshot_fingerprint(
        model_fingerprint=model_fingerprint,
        runtime_origin=runtime_origin,
        readout_kind=readout_kind,
        layer_indices=layer_indices,
        activation_widths=activation_widths,
        representations=representations,
        reference_statistics_sha256=(
            None
            if reference_statistics is None
            else reference_statistics.statistics_sha256
        ),
    )
    lineage = SubstrateForwardRepresentationLineage(
        schema_version=SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
        snapshot_fingerprint=fingerprint,
        model_fingerprint=model_fingerprint,
        runtime_origin=runtime_origin,
        readout_kind=readout_kind,
        layer_indices=layer_indices,
        activation_widths=activation_widths,
        representation_dim=sum(activation_widths),
        reference_corpus_id=(
            None if reference_statistics is None else reference_statistics.corpus_id
        ),
        reference_statistics_sha256=(
            None
            if reference_statistics is None
            else reference_statistics.statistics_sha256
        ),
    )
    return SubstrateForwardRepresentationSnapshot(
        lineage=lineage,
        representations=representations,
        description=(
            "Frozen substrate full-runtime context from the latest token on "
            "selected residual layers; source text is retained only as SHA-256."
        ),
    )


class SubstrateForwardRepresentationPublisher:
    """Publish unconditioned frozen-LM residual targets with strict lineage."""

    def __init__(
        self,
        runtime: OpenWeightResidualRuntime,
        *,
        model_fingerprint: SubstrateFingerprint,
        reference_statistics: SubstrateReadoutReferenceStatistics | None = None,
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
        self._reference_statistics = reference_statistics

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
            row, layer_indices, activation_widths = _representation_from_capture(
                sample_id=sample_id,
                source_sha256=_sha256_text(source_text),
                capture=capture,
                require_unconditioned=True,
                reference_statistics=self._reference_statistics,
            )
            current_geometry = (layer_indices, activation_widths)
            if geometry is None:
                geometry = current_geometry
            elif current_geometry != geometry:
                raise ValueError(
                    "substrate target residual geometry drifted across samples: "
                    f"expected {geometry}, got {current_geometry} for {sample_id!r}"
                )
            rows.append(row)
            if progress is not None:
                progress(sample_id, sample_index + 1, len(sample_sources))

        if geometry is None:
            raise RuntimeError("substrate representation geometry was not initialized")
        representations = tuple(rows)
        layer_indices, activation_widths = geometry
        readout_kind = (
            SUBSTRATE_FORWARD_READOUT_KIND
            if self._reference_statistics is None
            else SUBSTRATE_FORWARD_CENTERED_READOUT_KIND
        )
        reference_statistics_sha256 = (
            None
            if self._reference_statistics is None
            else self._reference_statistics.statistics_sha256
        )
        fingerprint = _snapshot_fingerprint(
            model_fingerprint=self._model_fingerprint,
            runtime_origin=self._runtime_origin,
            readout_kind=readout_kind,
            layer_indices=layer_indices,
            activation_widths=activation_widths,
            representations=representations,
            reference_statistics_sha256=reference_statistics_sha256,
        )
        lineage = SubstrateForwardRepresentationLineage(
            schema_version=SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION,
            snapshot_fingerprint=fingerprint,
            model_fingerprint=self._model_fingerprint,
            runtime_origin=self._runtime_origin,
            readout_kind=readout_kind,
            layer_indices=layer_indices,
            activation_widths=activation_widths,
            representation_dim=sum(activation_widths),
            reference_corpus_id=(
                None
                if self._reference_statistics is None
                else self._reference_statistics.corpus_id
            ),
            reference_statistics_sha256=reference_statistics_sha256,
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
    "SUBSTRATE_FORWARD_CENTERED_READOUT_KIND",
    "SUBSTRATE_FORWARD_READOUT_KIND",
    "SUBSTRATE_FORWARD_REPRESENTATION_SCHEMA_VERSION",
    "SUBSTRATE_READOUT_REFERENCE_STATISTICS_SCHEMA_VERSION",
    "SubstrateForwardRepresentation",
    "SubstrateForwardRepresentationLineage",
    "SubstrateForwardRepresentationPublisher",
    "SubstrateForwardRepresentationSnapshot",
    "SubstrateReadoutReferenceStatistics",
    "fit_forward_readout_reference_statistics",
    "layer_normalized_readout_vector",
    "publish_runtime_capture_representation",
]
