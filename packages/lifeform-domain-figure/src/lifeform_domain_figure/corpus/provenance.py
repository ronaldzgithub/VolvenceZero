"""Extended provenance records for figure-vertical corpus sources.

The kernel ``IngestionProvenance`` (in ``lifeform-ingestion``) carries
the minimum needed for runtime ingestion: ``uploader`` /
``upload_ts_ms`` / ``source_uri`` / ``integrity_hash``. The figure
vertical needs a richer provenance footprint so downstream
``LicenseGate`` checks (deferred packet), curation review, and IP
audit can answer:

* Which legal jurisdiction is this document public-domain in?
* Who digitised it (and when), so a re-fetch can be reproduced?
* Was it OCRed, transcribed, or born-digital?
* What URL or archive locator does the citation point at?
* What was the integrity hash of the source bytes the curator saw?

This module is **schema only**. There is deliberately no global
"provenance store" — each ``FigurePaperSource`` /
``FigureLetterSource`` / etc. carries its own provenance record so
the freeze step (P2.3 :func:`build_figure_artifact_bundle`) can fold
provenance fingerprints into the bundle's integrity hash.

The :class:`SourceProvenance` dataclass is intentionally not
embedded into the four source-kind records (``FigurePaperSource``
etc.) yet — those records currently carry only the minimum the
ingestion adapters need. The full provenance lifts at a higher level
(curation flow attaches one ``SourceProvenance`` per source id) so
the kernel-side ingestion path stays unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class CaptureMethod(str, Enum):
    """How the digital text was produced from the original document."""

    BORN_DIGITAL = "born_digital"  # author's word-processed manuscript / paper
    TRANSCRIBED = "transcribed"  # human typed from facsimile
    OCR = "ocr"  # automated OCR pipeline
    SCAN_REVIEWED_OCR = "scan_reviewed_ocr"  # OCR + reviewer hand-correction
    UNKNOWN = "unknown"


class LegalClearance(str, Enum):
    """Reviewer's documented legal clearance for shipping this source.

    Values are deliberately coarse: this enum exists to make the
    audit decision *visible*, not to encode jurisdictional nuance.
    A real production deployment maps these to its own legal review
    matrix.
    """

    PUBLIC_DOMAIN_GLOBAL = "public_domain_global"
    PUBLIC_DOMAIN_REGIONAL = "public_domain_regional"
    LICENSED_OPEN = "licensed_open"  # CC-BY, CC0, etc.
    LICENSED_RESTRICTED = "licensed_restricted"
    TENANT_DECLARED = "tenant_declared"  # tenant signs a "I have the rights" attestation
    UNCLEARED = "uncleared"  # forbidden in shipped bundles; aborts curation


_REQUIRED_NON_EMPTY_FIELDS = (
    "source_id",
    "figure_id",
    "source_url",
    "license_label",
    "captured_by",
    "captured_at_iso",
    "byte_sha256",
    "provenance_note",
)


@dataclass(frozen=True)
class SourceProvenance:
    """Per-source provenance record carried alongside the typed source.

    ``source_id`` is the canonical citation key used by the matching
    ``FigurePaperSource`` / ``FigureLetterSource`` / etc. record so
    a curator can look up "where did this paper come from?" by id.

    ``byte_sha256`` is the SHA-256 of the *original byte stream the
    curator captured*, not of the cleaned chunk text. The two hashes
    differ when normalisation / OCR-clean strips bytes; both are
    valuable — the byte hash proves "we got this exact upstream
    artifact", the chunk hash (computed by the ingestion adapter)
    proves "this is what the kernel actually saw".

    ``provenance_note`` is the short reviewer audit string; the
    curation reviewer (D6) refuses to admit a source whose note is
    empty so a human always pairs the record with a justification.
    """

    source_id: str
    figure_id: str
    source_url: str
    license_label: str
    legal_clearance: LegalClearance
    capture_method: CaptureMethod
    captured_by: str
    captured_at_iso: str
    byte_sha256: str
    provenance_note: str
    jurisdiction_hint: str = ""

    def __post_init__(self) -> None:
        for name in _REQUIRED_NON_EMPTY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"SourceProvenance.{name} must be a non-empty string for "
                    f"source_id={self.source_id!r}"
                )
        if len(self.byte_sha256) != 64:
            raise ValueError(
                f"SourceProvenance.byte_sha256 must be a 64-char hex digest, "
                f"got {self.byte_sha256!r} for source_id={self.source_id!r}"
            )
        if self.legal_clearance == LegalClearance.UNCLEARED:
            raise ValueError(
                f"SourceProvenance.legal_clearance UNCLEARED is forbidden in "
                f"shipped bundles (source_id={self.source_id!r}); the curation "
                f"flow refuses to admit uncleared sources. If the legal posture "
                f"is genuinely undecided, hold the source out of the bundle "
                f"until clearance is recorded."
            )


def fingerprint_provenance(records: tuple[SourceProvenance, ...]) -> str:
    """Compute a deterministic SHA-256 over a tuple of provenance records.

    Used by the curation freeze step to fold provenance into the
    bundle's integrity hash without exposing the full provenance
    payload through the bundle's public schema (the bundle stores a
    single fingerprint string; the full provenance log lives next to
    the bundle for audit).
    """

    payload = tuple(
        (
            rec.source_id,
            rec.figure_id,
            rec.source_url,
            rec.license_label,
            rec.legal_clearance.value,
            rec.capture_method.value,
            rec.captured_by,
            rec.captured_at_iso,
            rec.byte_sha256,
            rec.jurisdiction_hint,
            rec.provenance_note,
        )
        for rec in records
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


__all__ = [
    "CaptureMethod",
    "LegalClearance",
    "SourceProvenance",
    "fingerprint_provenance",
]
