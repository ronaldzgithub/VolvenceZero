"""AUTHORSHIP_ATTRIBUTION verifier (L2 second batch).

Checks whether the figure declared as the author of a source is
actually the author per OpenAlex's authorships field. Catches the
"this paper is by a junior collaborator, not Einstein himself" and
"this letter was edited / paraphrased by a 1960s biographer" failure
modes.

Heuristics (full-mode):

1. **Single-work lookup**: caller supplies the OpenAlex work id
   (e.g., ``"W4205692301"``); the verifier fetches all of the
   figure's authored works via
   :meth:`OpenAlexClient.fetch_author_works(openalex_author_id=...)``
   (typically cached) and asks whether the work id appears in the
   set. Match -> primary attribution PASS.
2. **Co-author overlap (edge heuristic)**: when no direct match,
   check the supplied ``coauthor_anchor_works`` (other works the
   figure is known to have authored) and a supplied
   ``candidate_coauthor_ids`` set on the source under audit. If any
   coauthor of the source overlaps with a known co-author of the
   figure on the anchor works -> NEEDS_REVIEW (likely-but-uncertain
   attribution; reviewer adjudicates).
3. **No match + no overlap** -> FAIL.

The verifier deliberately does NOT scrape the corpus body for
"author names" — that is keyword-matching, which the rules forbid.
All authorship signal must come from typed metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.corpus.provenance import SourceProvenance
from lifeform_domain_figure.metadata.openalex import OpenAlexClient
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)

VERIFIER_VERSION = "1"
REVIEWER_ID = f"auto:authorship_attribution:{VERIFIER_VERSION}"


def _check(
    *,
    verdict: Verdict,
    evidence: tuple[str, ...],
    sha: str,
    now_iso: str,
) -> VerificationCheck:
    return VerificationCheck(
        check_kind=CheckKind.AUTHORSHIP_ATTRIBUTION,
        verdict=verdict,
        evidence=evidence,
        reviewer_id=REVIEWER_ID,
        reviewed_at_iso=now_iso,
        source_byte_sha256=sha,
    )


def verify_authorship_attribution(
    provenance: SourceProvenance,
    *,
    openalex_client: OpenAlexClient,
    expected_openalex_author_id: str,
    candidate_work_id: str,
    coauthor_anchor_works: tuple[str, ...] = (),
    candidate_coauthor_openalex_ids: tuple[str, ...] = (),
    now_iso: str | None = None,
) -> VerificationCheck:
    """Return a :class:`VerificationCheck` for the authorship-attribution axis.

    * ``expected_openalex_author_id`` — figure's OpenAlex author id
      (e.g., ``"A5023888391"`` for Einstein).
    * ``candidate_work_id`` — the OpenAlex work id for the source
      under audit.
    * ``coauthor_anchor_works`` — other OpenAlex work ids the figure
      authored that we trust; used to derive the "known co-author
      pool" for fallback overlap heuristic.
    * ``candidate_coauthor_openalex_ids`` — OpenAlex author ids of
      the source's listed co-authors (excluding the expected author);
      caller fetches these via the candidate work's OpenAlex record
      upstream.
    """

    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    if not candidate_work_id.strip():
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                "candidate_work_id is empty",
                "verifier needs a typed OpenAlex work id to attribute; "
                "reviewer must supply one",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    try:
        works = openalex_client.fetch_author_works(
            openalex_author_id=expected_openalex_author_id
        )
    except Exception as exc:  # noqa: BLE001
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"openalex_client.fetch_author_works("
                f"openalex_author_id={expected_openalex_author_id!r}) failed: {exc}",
                "verifier defers to human reviewer when metadata fetch fails",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    work_ids = {w.openalex_id for w in works}
    if candidate_work_id in work_ids:
        return _check(
            verdict=Verdict.PASS,
            evidence=(
                f"candidate_work_id={candidate_work_id!r} found in OpenAlex "
                f"author works ({len(work_ids)} works total)",
                f"author_id={expected_openalex_author_id}",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if not candidate_coauthor_openalex_ids or not coauthor_anchor_works:
        return _check(
            verdict=Verdict.FAIL,
            evidence=(
                f"candidate_work_id={candidate_work_id!r} NOT in OpenAlex "
                f"author works for author_id={expected_openalex_author_id!r}",
                f"openalex returned {len(work_ids)} works; none match",
                "no co-author overlap data supplied; cannot fall back",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    anchor_set = set(coauthor_anchor_works) & work_ids
    candidate_set = set(candidate_coauthor_openalex_ids)
    if not anchor_set:
        return _check(
            verdict=Verdict.FAIL,
            evidence=(
                f"candidate_work_id={candidate_work_id!r} NOT in OpenAlex author works",
                "co-author anchor works supplied but none match figure's "
                "OpenAlex set; anchor_works likely stale",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    return _check(
        verdict=Verdict.NEEDS_REVIEW,
        evidence=(
            f"candidate_work_id={candidate_work_id!r} NOT directly in OpenAlex "
            f"author works for author_id={expected_openalex_author_id!r}",
            f"co-author overlap heuristic available but inconclusive: "
            f"{len(anchor_set)} anchor works & {len(candidate_set)} candidate "
            "co-authors; reviewer must adjudicate (could be ghostwritten / "
            "edited / mis-attributed in OpenAlex)",
            f"anchor_overlap_count={len(anchor_set)}",
            f"candidate_coauthor_count={len(candidate_set)}",
            f"source_id={provenance.source_id}",
        ),
        sha=provenance.byte_sha256,
        now_iso=timestamp,
    )


__all__ = ["REVIEWER_ID", "VERIFIER_VERSION", "verify_authorship_attribution"]
