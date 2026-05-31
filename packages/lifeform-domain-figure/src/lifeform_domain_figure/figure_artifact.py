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
from dataclasses import dataclass, field

from lifeform_core import VitalsBootstrap
from volvence_zero.application import DomainExperiencePackage
from volvence_zero.substrate import SubstrateFingerprint, fingerprint_set_sha256

from lifeform_domain_figure.coverage_map import FigureCoverageMap
from lifeform_domain_figure.presence_artifact import (
    FigurePresenceArtifact,
    presence_fingerprint,
)
from lifeform_domain_figure.profile import HistoricalFigureProfile
from lifeform_domain_figure.retrieval_index import FigureRetrievalIndex
from lifeform_domain_figure.style_prior import FigureStylePrior


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FigureArtifactBundle:
    """Frozen, integrity-hashed bundle of every figure runtime artifact.

    Construct only via :func:`lifeform_domain_figure.build_figure_artifact_bundle`
    so the integrity hash is computed in one place.

    ``metadata_digest_fingerprint`` (debt #25 closure) is the
    SHA-256 fingerprint of the optional :class:`MetadataDigest` that
    enriched the profile prior to bundle compilation. Empty string
    when no metadata enrichment was applied; non-empty when present
    so an audit can reverse-look up which Wikidata / OpenAlex /
    Crossref / SEP snapshot produced this bundle.

    ``provenance_fingerprint`` (debt #24 closure) is the SHA-256
    fingerprint of the tuple of :class:`SourceProvenance` records
    the curator pinned at compile time (license / capture method /
    byte hash per source). Empty when no provenance records were
    supplied; non-empty makes a license-only edit on any source
    auditable through the bundle hash itself.
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
    metadata_digest_fingerprint: str = ""
    provenance_fingerprint: str = ""
    # debt #47 / F-C ACTIVE (2026-05-14): which substrate fingerprints
    # this bundle was baked against. Empty tuple = legacy bundle
    # (predates #47); migration shim in
    # ``lifeform_domain_figure.bundle_io`` injects ``LEGACY_FINGERPRINT``
    # so old pickles still load. Non-empty tuple is folded into
    # ``integrity_hash`` so upgrading substrate yields a different
    # bundle id (R15 byte-level rollback contract). Contract test
    # ``tests/contracts/test_substrate_fingerprint_propagation.py``
    # locks both the byte-stable empty-tuple and the
    # hash-changes-with-content invariants.
    compatible_substrates: tuple[SubstrateFingerprint, ...] = field(
        default_factory=tuple
    )
    # debt #58 / #59 / #60 (P1 figure-evidence-packet): readout reports
    # for L4 refusal accuracy / L3 grounding faithfulness / L1 voice
    # blind-test. None = not yet evaluated; non-None = evaluation
    # artifact attached. These fields **do not** affect
    # ``integrity_hash`` (eval is readout, not bundle identity); see
    # ``docs/specs/figure-refusal-gt-protocol.md`` for the audit
    # fingerprint scheme.
    refusal_eval_report: object | None = None
    grounding_eval_report: object | None = None
    voice_blind_test_report: object | None = None
    # L0 visual presence (web digital-human + lipsync metadata). Default
    # ``None`` keeps existing bundles byte-stable: only when a non-None
    # presence artifact is attached is its fingerprint folded into
    # ``integrity_hash``. The actual rendering plane lives in the
    # deploy-side ``apps/presence-service``; this field carries only the
    # hashed consent receipt + persona id + license label so audit can
    # reverse-look up which legal posture produced this bundle.
    presence: FigurePresenceArtifact | None = None

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
        if self.presence is not None and self.figure_id != self.presence.figure_id:
            raise ValueError(
                "FigureArtifactBundle: presence.figure_id must "
                "match bundle.figure_id"
            )


# D9 vocab alias: the volvence-press / novel-worlds products speak of an
# "author companion" rather than a "historical figure". The runtime
# artifact is identical — the author's reviewed profile + corpus compiles
# into the same immutable bundle. This alias lets Press-side code (and the
# ``figure-companion`` vertical) refer to the bundle by its product vocab
# without a parallel schema. It is a pure type alias, NOT a subclass: the
# integrity hash, byte layout, and ``FigureArtifactBundle`` identity are
# unchanged, so existing bundles stay byte-stable (additive-only).
AuthorCompanionBundle = FigureArtifactBundle


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
    metadata_digest_fingerprint: str = "",
    provenance_fingerprint: str = "",
    compatible_substrates: tuple[SubstrateFingerprint, ...] = (),
    presence_integrity: str = "",
) -> str:
    """Deterministic SHA-256 over the bundle's load-bearing identity fields.

    Excludes the bundle id (which is derived from this hash) and any
    per-process audit fields. Two bundles with identical inputs
    produce the same integrity hash byte-for-byte (R15).

    ``metadata_digest_fingerprint`` (debt #25 closure) defaults to
    ``""`` and is only folded into the hash when non-empty —
    preserving byte-for-byte stability of existing bundles that did
    not record metadata enrichment. Passing a non-empty fingerprint
    yields a different bundle id (any metadata-snapshot change is
    auditable through the bundle hash).

    ``provenance_fingerprint`` (debt #24 closure) follows the same
    rule: empty default keeps existing bundles' hashes byte-stable,
    a non-empty value (the SHA-256 over the supplied
    :class:`SourceProvenance` tuple) folds the curator's license /
    capture-method declarations into the bundle's identity so a
    license-only edit yields a fresh bundle id.

    ``presence_integrity`` follows the same byte-stable rule for the
    L0 visual presence layer: empty default keeps legacy bundles'
    hashes unchanged; non-empty (the
    :class:`FigurePresenceArtifact.integrity_hash`) folds reference
    image / consent token hash / license label / engine subset into
    the bundle id so a likeness re-bake or consent revoke yields a
    fresh bundle.
    """

    payload: tuple[object, ...] = (
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
    if metadata_digest_fingerprint:
        payload = payload + (("metadata_digest", metadata_digest_fingerprint),)
    if provenance_fingerprint:
        payload = payload + (("provenance", provenance_fingerprint),)
    substrate_fp = fingerprint_set_sha256(compatible_substrates)
    if substrate_fp:
        payload = payload + (("compatible_substrates", substrate_fp),)
    if presence_integrity:
        payload = payload + (("presence", presence_integrity),)
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def bundle_id_from_hash(figure_id: str, integrity_hash: str) -> str:
    """Compose a stable, human-readable bundle id from figure + hash prefix."""

    return f"figure-bundle:{figure_id}:{integrity_hash[:16]}"


__all__ = [
    "SCHEMA_VERSION",
    "FigureArtifactBundle",
    "FigurePresenceArtifact",
    "bundle_id_from_hash",
    "compute_bundle_integrity_hash",
    "presence_fingerprint",
]
