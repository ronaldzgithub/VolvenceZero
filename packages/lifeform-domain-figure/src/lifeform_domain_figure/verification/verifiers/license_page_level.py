"""LICENSE_PAGE_LEVEL verifier (L2 first batch).

Compares two pieces of license evidence:

* ``provenance.license_label`` — the page-level license string the L1
  parser scraped (or the curator's override). For Wikisource this is
  typically a ``{{PD-old-100}}`` template; for Gutenberg it's the
  pre-START block; for CPAE it's the inside-cover ``"Copyright by
  Princeton University Press"`` notice.
* ``provenance.legal_clearance`` — the curator's enum-typed legal
  decision (``PUBLIC_DOMAIN_GLOBAL`` / ``LICENSED_OPEN`` / etc.).

Verdict rules:

* The license label contains a phrase from a clearance-specific
  allowlist -> ``PASS``
* The license label contains a hard-conflict phrase (``"all rights
  reserved"`` / ``"copyright "`` / ``"©"``) AND the clearance is one
  of the public-domain levels -> ``FAIL``
* No allowlist match and no hard-conflict (or the special L1 sentinel
  ``"(no license notice scraped)"``) -> ``NEEDS_REVIEW``

The ``TENANT_DECLARED`` clearance always passes (the tenant has
attested rights independently of any page-level evidence) and the
``UNCLEARED`` clearance never reaches this verifier — :class:`SourceProvenance`
already refuses construction with ``UNCLEARED``.

Pure function; no I/O, no metadata client, no kernel access.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.corpus.provenance import LegalClearance, SourceProvenance
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)

VERIFIER_VERSION = "1"
REVIEWER_ID = f"auto:license_page_level:{VERIFIER_VERSION}"

_ALLOWLIST: dict[LegalClearance, tuple[str, ...]] = {
    LegalClearance.PUBLIC_DOMAIN_GLOBAL: (
        "public domain",
        "pd-",
        "cc0",
        "expired",
        "no rights reserved",
        "no known copyright",
    ),
    LegalClearance.PUBLIC_DOMAIN_REGIONAL: (
        "public domain",
        "pd-",
        "expired",
        "no known copyright",
    ),
    LegalClearance.LICENSED_OPEN: (
        "cc-by",
        "cc by",
        "cc0",
        "creative commons",
        "open license",
        "mit license",
        "apache license",
    ),
    LegalClearance.LICENSED_RESTRICTED: (
        "all rights reserved",
        "copyright",
        "permission",
        "licensed",
    ),
}

_HARD_CONFLICT_PHRASES: tuple[str, ...] = (
    "all rights reserved",
    "copyright (c)",
    "(c) copyright",
    "\u00a9",
)

_PUBLIC_DOMAIN_CLEARANCES: frozenset[LegalClearance] = frozenset(
    {
        LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        LegalClearance.PUBLIC_DOMAIN_REGIONAL,
    }
)

L1_SENTINEL = "(no license notice scraped)"


def _normalise(text: str) -> str:
    return text.lower().strip()


def verify_license_page_level(
    provenance: SourceProvenance,
    *,
    now_iso: str | None = None,
) -> VerificationCheck:
    """Return a :class:`VerificationCheck` for the license-page-level axis."""

    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    label = provenance.license_label
    label_lower = _normalise(label)
    clearance = provenance.legal_clearance

    if clearance is LegalClearance.TENANT_DECLARED:
        return VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.PASS,
            evidence=(
                f"legal_clearance={clearance.value}",
                "tenant attestation supersedes page-level evidence",
                f"source_id={provenance.source_id}",
            ),
            reviewer_id=REVIEWER_ID,
            reviewed_at_iso=timestamp,
            source_byte_sha256=provenance.byte_sha256,
        )

    if label_lower == _normalise(L1_SENTINEL) or not label_lower:
        return VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                "license_label is empty or the L1 'no license notice scraped' sentinel",
                f"legal_clearance={clearance.value}",
                f"source_id={provenance.source_id}",
            ),
            reviewer_id=REVIEWER_ID,
            reviewed_at_iso=timestamp,
            source_byte_sha256=provenance.byte_sha256,
        )

    if clearance in _PUBLIC_DOMAIN_CLEARANCES:
        for phrase in _HARD_CONFLICT_PHRASES:
            if phrase in label_lower:
                return VerificationCheck(
                    check_kind=CheckKind.LICENSE_PAGE_LEVEL,
                    verdict=Verdict.FAIL,
                    evidence=(
                        f"license_label contains hard-conflict phrase {phrase!r}",
                        f"legal_clearance={clearance.value} (public-domain level)",
                        f"license_label_excerpt={label[:120]!r}",
                        f"source_id={provenance.source_id}",
                    ),
                    reviewer_id=REVIEWER_ID,
                    reviewed_at_iso=timestamp,
                    source_byte_sha256=provenance.byte_sha256,
                )

    allowlist = _ALLOWLIST.get(clearance, ())
    matched = next((phrase for phrase in allowlist if phrase in label_lower), None)
    if matched is not None:
        return VerificationCheck(
            check_kind=CheckKind.LICENSE_PAGE_LEVEL,
            verdict=Verdict.PASS,
            evidence=(
                f"license_label contains allowlist phrase {matched!r}",
                f"legal_clearance={clearance.value}",
                f"license_label_excerpt={label[:120]!r}",
                f"source_id={provenance.source_id}",
            ),
            reviewer_id=REVIEWER_ID,
            reviewed_at_iso=timestamp,
            source_byte_sha256=provenance.byte_sha256,
        )

    return VerificationCheck(
        check_kind=CheckKind.LICENSE_PAGE_LEVEL,
        verdict=Verdict.NEEDS_REVIEW,
        evidence=(
            "license_label matched neither allowlist nor hard-conflict phrases",
            f"legal_clearance={clearance.value}",
            f"license_label_excerpt={label[:120]!r}",
            f"source_id={provenance.source_id}",
        ),
        reviewer_id=REVIEWER_ID,
        reviewed_at_iso=timestamp,
        source_byte_sha256=provenance.byte_sha256,
    )


__all__ = [
    "L1_SENTINEL",
    "REVIEWER_ID",
    "VERIFIER_VERSION",
    "verify_license_page_level",
]
