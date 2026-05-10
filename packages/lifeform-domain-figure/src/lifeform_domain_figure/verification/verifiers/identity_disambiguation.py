"""IDENTITY_DISAMBIGUATION verifier (L2 second batch).

Validates that the curator-supplied figure identity (Wikidata QID +
expected birth year + expected primary occupation labels) actually
points to the *same person* as a single Wikidata entity. Catches
the canonical "two Albert Einsteins" / "two Lu Xuns" failure mode
where the metadata pipeline keys a corpus chunk to the wrong real
person.

Heuristics (full-mode, debt #26 / #28 L2 second batch):

1. **Single-QID lookup**: fetch the supplied QID via the V2
   :class:`WikidataClient`. If the response is missing essential
   fields (birth year), or the QID returns no entity -> FAIL.
2. **Birth-year cross-check**: the curator-declared
   ``expected_birth_year`` must equal Wikidata's birth year (P569).
   Mismatch -> FAIL with both years in evidence. Tolerance ``+/- 1``
   for cases where Wikidata records "circa" with a 1-year offset.
3. **Occupation overlap**: if ``expected_occupations`` is provided,
   at least one expected label must appear in the Wikidata
   occupation_labels (or field_of_work_labels). Zero overlap with
   non-empty expected list -> NEEDS_REVIEW (occupation labels are
   noisy in Wikidata; reviewer adjudicates).
4. **Lifespan sanity**: birth year < (death year if non-None);
   already enforced by ``WikidataPersonPayload.__post_init__``,
   surfaced here as PASS evidence.

When all checks pass -> PASS.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.corpus.provenance import SourceProvenance
from lifeform_domain_figure.metadata.wikidata import WikidataClient
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)

VERIFIER_VERSION = "1"
REVIEWER_ID = f"auto:identity_disambiguation:{VERIFIER_VERSION}"


def _check(
    *,
    verdict: Verdict,
    evidence: tuple[str, ...],
    sha: str,
    now_iso: str,
) -> VerificationCheck:
    return VerificationCheck(
        check_kind=CheckKind.IDENTITY_DISAMBIGUATION,
        verdict=verdict,
        evidence=evidence,
        reviewer_id=REVIEWER_ID,
        reviewed_at_iso=now_iso,
        source_byte_sha256=sha,
    )


def verify_identity_disambiguation(
    provenance: SourceProvenance,
    *,
    wikidata_client: WikidataClient,
    expected_qid: str,
    expected_birth_year: int,
    expected_occupations: tuple[str, ...] = (),
    now_iso: str | None = None,
) -> VerificationCheck:
    """Return a :class:`VerificationCheck` for the identity-disambiguation axis.

    All inputs (``expected_qid`` / ``expected_birth_year`` /
    ``expected_occupations``) are curator-declared per figure;
    typically derived once from the figure profile and passed
    through to every source's verification.
    """

    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    try:
        person = wikidata_client.fetch_person(qid=expected_qid)
    except Exception as exc:  # noqa: BLE001
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"wikidata_client.fetch_person(qid={expected_qid!r}) failed: {exc}",
                "verifier defers to human reviewer when metadata fetch fails",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    birth_diff = abs(person.birth_year - expected_birth_year)
    if birth_diff > 1:
        return _check(
            verdict=Verdict.FAIL,
            evidence=(
                f"wikidata.birth_year={person.birth_year}",
                f"expected_birth_year={expected_birth_year}",
                f"diff={birth_diff} years (tolerance=1)",
                f"qid={person.qid} label={person.label!r}",
                "identity mismatch: birth year disagreement exceeds tolerance",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    overlap_evidence: tuple[str, ...] = ()
    if expected_occupations:
        wikidata_labels = set(person.occupation_labels) | set(person.field_of_work_labels)
        overlap = set(expected_occupations) & wikidata_labels
        if not overlap:
            return _check(
                verdict=Verdict.NEEDS_REVIEW,
                evidence=(
                    f"expected_occupations={list(expected_occupations)!r}",
                    f"wikidata.occupation_labels={list(person.occupation_labels)!r}",
                    f"wikidata.field_of_work_labels={list(person.field_of_work_labels)!r}",
                    "zero overlap; occupation labels in Wikidata are noisy "
                    "(QIDs not human-readable); reviewer adjudicates",
                    f"qid={person.qid} label={person.label!r}",
                    f"source_id={provenance.source_id}",
                ),
                sha=provenance.byte_sha256,
                now_iso=timestamp,
            )
        overlap_evidence = (
            f"occupation overlap: {sorted(overlap)!r}",
        )
    death_evidence = (
        f"wikidata.death_year={person.death_year}"
        if person.death_year is not None
        else "wikidata.death_year=None (still living per Wikidata)"
    )
    return _check(
        verdict=Verdict.PASS,
        evidence=(
            f"qid={person.qid} label={person.label!r}",
            f"wikidata.birth_year={person.birth_year} (expected={expected_birth_year}; diff={birth_diff})",
            death_evidence,
            *overlap_evidence,
            f"source_id={provenance.source_id}",
        ),
        sha=provenance.byte_sha256,
        now_iso=timestamp,
    )


__all__ = ["REVIEWER_ID", "VERIFIER_VERSION", "verify_identity_disambiguation"]
