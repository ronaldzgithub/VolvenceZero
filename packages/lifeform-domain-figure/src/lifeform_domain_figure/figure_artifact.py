"""Frozen ``FigureArtifactBundle`` schema.

The single immutable artifact produced by the figure vertical and
consumed by the runtime + DLaaS layers. The bundle is the **only**
object that crosses the wheel boundary; consumers never reach into
the vertical's internal modules at runtime (R8 SSOT contract,
``ssot-module-boundaries.mdc``).

Field shape:

* :attr:`profile`            — the reviewed
  :class:`HistoricalFigureProfile` plus the active time-window
  selection.
* :attr:`domain_package`     — :class:`DomainExperiencePackage`
  compiled from the profile (knowledge / cases / playbook /
  boundaries) so the existing application owners pick it up
  unchanged.
* :attr:`vitals_bootstrap`   — :class:`VitalsBootstrap` for the
  figure's drive shape.
* :attr:`retrieval_index`    — L3 backbone (P2.1).
* :attr:`coverage_map`       — L4 backbone (P2.2).
* :attr:`style_prior`        — L1 backbone (P2.3).
* :attr:`steering`           — optional L2 artifact (filled by F5).
* :attr:`lora`               — optional L1+L2 artifact (filled by F6).
* :attr:`integrity_hash`     — SHA-256 over the load-bearing fields;
  byte-for-byte rollback contract (R15).
* :attr:`version_window`     — selected ``(year_start, year_end)``
  pair if a ``TimeWindowedView`` was applied, else ``(0, 0)``.

The ``steering`` and ``lora`` fields are typed as ``object | None``
here to avoid a forward import of types that land in F5 / F6. The
runtime consumer pattern (``isinstance(bundle.lora, FigureLoRAArtifact)``
when the F6 module has loaded) keeps the contract narrow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lifeform_core import VitalsBootstrap
from volvence_zero.application import DomainExperiencePackage

from lifeform_domain_figure.coverage_map import FigureCoverageMap
from lifeform_domain_figure.profile import HistoricalFigureProfile
from lifeform_domain_figure.retrieval_index import FigureRetrievalIndex
from lifeform_domain_figure.style_prior import FigureStylePrior


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FigureArtifactBundle:
    """Frozen, integrity-hashed bundle of every figure runtime artifact.

    Construct only via :func:`lifeform_domain_figure.build_figure_artifact_bundle`
    so the integrity hash is computed in one place.
    """

    schema_version: int
    bundle_id: str
    figure_id: str
    profile_version: str
    version_window: tuple[int, int]
    profile: HistoricalFigureProfile
    domain_package: DomainExperiencePackage
    vitals_bootstrap: VitalsBootstrap
    retrieval_index: FigureRetrievalIndex
    coverage_map: FigureCoverageMap
    style_prior: FigureStylePrior
    steering: object | None
    lora: object | None
    integrity_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"FigureArtifactBundle.schema_version mismatch: "
                f"got {self.schema_version!r}, expected {SCHEMA_VERSION!r}"
            )
        if not self.bundle_id.strip():
            raise ValueError("FigureArtifactBundle.bundle_id must be non-empty")
        if not self.figure_id.strip():
            raise ValueError("FigureArtifactBundle.figure_id must be non-empty")
        if not self.integrity_hash.strip():
            raise ValueError(
                "FigureArtifactBundle.integrity_hash must be non-empty; "
                "the rollback contract requires every bundle to be hash-addressed."
            )
        if self.figure_id != self.profile.profile_id:
            raise ValueError(
                f"FigureArtifactBundle.figure_id {self.figure_id!r} does "
                f"not match profile.profile_id {self.profile.profile_id!r}"
            )
        if self.figure_id != self.retrieval_index.figure_id:
            raise ValueError(
                "FigureArtifactBundle: retrieval_index.figure_id must "
                "match bundle.figure_id"
            )
        if self.figure_id != self.coverage_map.figure_id:
            raise ValueError(
                "FigureArtifactBundle: coverage_map.figure_id must "
                "match bundle.figure_id"
            )
        if self.figure_id != self.style_prior.figure_id:
            raise ValueError(
                "FigureArtifactBundle: style_prior.figure_id must "
                "match bundle.figure_id"
            )


def compute_bundle_integrity_hash(
    *,
    figure_id: str,
    profile_version: str,
    version_window: tuple[int, int],
    retrieval_integrity: str,
    coverage_integrity: str,
    style_integrity: str,
    steering_integrity: str,
    lora_integrity: str,
) -> str:
    """Deterministic SHA-256 over the bundle's load-bearing identity fields.

    Excludes the bundle id (which is derived from this hash) and any
    per-process audit fields. Two bundles with identical inputs
    produce the same integrity hash byte-for-byte (R15).
    """

    payload = (
        SCHEMA_VERSION,
        figure_id,
        profile_version,
        version_window,
        retrieval_integrity,
        coverage_integrity,
        style_integrity,
        steering_integrity,
        lora_integrity,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def bundle_id_from_hash(figure_id: str, integrity_hash: str) -> str:
    """Compose a stable, human-readable bundle id from figure + hash prefix."""

    return f"figure-bundle:{figure_id}:{integrity_hash[:16]}"


__all__ = [
    "SCHEMA_VERSION",
    "FigureArtifactBundle",
    "bundle_id_from_hash",
    "compute_bundle_integrity_hash",
]
