"""DATE_PLAUSIBILITY verifier (L2 first batch).

Checks whether a source's claimed authoring year falls inside the
figure's lifespan. The figure vertical's lifespan field
(:attr:`HistoricalFigureProfile.figure_lifespan`) is a
``(birth_year, death_year)`` tuple; the boundaries are inclusive
(a paper dated the same year as the death year is plausible — many
posthumous publications carry the year of writing).

Pure function; no I/O, no metadata client, no kernel access.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.corpus.provenance import SourceProvenance
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)

VERIFIER_VERSION = "1"
REVIEWER_ID = f"auto:date_plausibility:{VERIFIER_VERSION}"


def verify_date_plausibility(
    provenance: SourceProvenance,
    *,
    document_year: int,
    figure_lifespan: tuple[int, int],
    now_iso: str | None = None,
) -> VerificationCheck:
    """Return a :class:`VerificationCheck` for the date-plausibility axis.

    * ``document_year`` is the curator-supplied authoring year for the
      source (taken from the matching ``CPAEPayload.year`` /
      ``GutenbergPayload.year`` / etc.).
    * ``figure_lifespan`` is the inclusive ``(birth, death)`` tuple
      from ``HistoricalFigureProfile.figure_lifespan``.

    Verdict rules:

    * ``birth <= document_year <= death`` -> ``PASS``
    * outside the closed interval -> ``FAIL``
    * lifespan tuple ill-formed (``birth > death``) -> ``NEEDS_REVIEW``
      (the verifier refuses to give a verdict against bad input)
    """

    birth, death = figure_lifespan
    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    if birth > death:
        return VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"figure_lifespan is ill-formed: birth={birth} > death={death}",
                f"source_id={provenance.source_id}",
                "verifier refuses to decide against malformed lifespan",
            ),
            reviewer_id=REVIEWER_ID,
            reviewed_at_iso=timestamp,
            source_byte_sha256=provenance.byte_sha256,
        )
    if birth <= document_year <= death:
        return VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=(
                f"document_year={document_year}",
                f"figure_lifespan=[{birth},{death}] (inclusive)",
                f"source_id={provenance.source_id}",
            ),
            reviewer_id=REVIEWER_ID,
            reviewed_at_iso=timestamp,
            source_byte_sha256=provenance.byte_sha256,
        )
    return VerificationCheck(
        check_kind=CheckKind.DATE_PLAUSIBILITY,
        verdict=Verdict.FAIL,
        evidence=(
            f"document_year={document_year}",
            f"figure_lifespan=[{birth},{death}] (inclusive)",
            "document_year out of range",
            f"source_id={provenance.source_id}",
        ),
        reviewer_id=REVIEWER_ID,
        reviewed_at_iso=timestamp,
        source_byte_sha256=provenance.byte_sha256,
    )


__all__ = ["REVIEWER_ID", "VERIFIER_VERSION", "verify_date_plausibility"]
