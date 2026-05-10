"""TRANSLATION_LINEAGE verifier (L2 second batch).

Detects cases where a corpus chunk is **a translation of** the
figure's original work rather than the figure's own writing. The
distinction matters because:

* L2 fidelity (style / stance) on a translated text reflects the
  translator's voice, not the figure's.
* L3 citation locator should record the translation lineage so a
  reader knows "this is N. Wood's 1955 English rendering, not
  Einstein's German original".

Heuristics (full-mode):

1. **Single-DOI lookup** (Crossref): fetch the source's DOI raw
   message via the V2 :class:`CrossrefClient`. Read the
   ``translator`` field (a list of name dicts) — non-empty means
   this work IS a translation according to Crossref.
2. **Cross-check with curator language hint**: caller supplies
   ``figure_native_languages`` (the languages the figure originally
   wrote in, e.g., ``("de",)`` for Einstein) and the source's
   ``source_language``. Mismatch + non-empty translator field ->
   PASS with full lineage in evidence (translation correctly
   identified).
3. **Mismatch + empty translator field** -> NEEDS_REVIEW: language
   suggests translation but Crossref doesn't record one; reviewer
   verifies whether this is uncited translation or a genuine
   foreign-language original by the figure.
4. **Match (source_language in figure_native_languages) + empty
   translator field** -> PASS with "no translation" evidence.
5. **Match + non-empty translator field** -> NEEDS_REVIEW (rare:
   translator listed but language matches; could be a re-edition
   of the original).

Optional Wikidata cross-check via ``work_of_qid``: if supplied, the
verifier currently records the QID in evidence as a hint for the
reviewer; deeper cross-validation (Wikidata work-of P50 chain) is a
follow-up enhancement.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.corpus.provenance import SourceProvenance
from lifeform_domain_figure.metadata.crossref import (
    CrossrefClient,
    crossref_translator_names,
)
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)

VERIFIER_VERSION = "1"
REVIEWER_ID = f"auto:translation_lineage:{VERIFIER_VERSION}"


def _check(
    *,
    verdict: Verdict,
    evidence: tuple[str, ...],
    sha: str,
    now_iso: str,
) -> VerificationCheck:
    return VerificationCheck(
        check_kind=CheckKind.TRANSLATION_LINEAGE,
        verdict=verdict,
        evidence=evidence,
        reviewer_id=REVIEWER_ID,
        reviewed_at_iso=now_iso,
        source_byte_sha256=sha,
    )


def verify_translation_lineage(
    provenance: SourceProvenance,
    *,
    crossref_client: CrossrefClient,
    source_doi: str,
    source_language: str,
    figure_native_languages: tuple[str, ...],
    work_of_qid: str = "",
    now_iso: str | None = None,
) -> VerificationCheck:
    """Return a :class:`VerificationCheck` for the translation-lineage axis."""

    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    if not source_doi.strip():
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                "source_doi is empty",
                "verifier needs a typed Crossref DOI to read translator field",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if not source_language.strip() or not figure_native_languages:
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                "source_language and figure_native_languages must both be supplied",
                f"got source_language={source_language!r}, "
                f"figure_native_languages={list(figure_native_languages)!r}",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    fetch_raw = getattr(crossref_client, "fetch_raw_message", None)
    if not callable(fetch_raw):
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                "supplied crossref_client lacks fetch_raw_message(doi=...) method",
                "verifier needs Crossref's full message payload to read 'translator'",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    try:
        message = fetch_raw(doi=source_doi)
    except Exception as exc:  # noqa: BLE001
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"crossref_client.fetch_raw_message(doi={source_doi!r}) failed: {exc}",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if not isinstance(message, dict):
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"crossref message for doi={source_doi!r} is not a dict",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    translators = crossref_translator_names(message)
    is_translation_lang = source_language not in figure_native_languages
    qid_hint = (
        f"work_of_qid={work_of_qid!r} (reviewer cross-reference)"
        if work_of_qid
        else "work_of_qid not supplied"
    )
    if is_translation_lang and translators:
        return _check(
            verdict=Verdict.PASS,
            evidence=(
                f"source_language={source_language!r} not in "
                f"figure_native_languages={list(figure_native_languages)!r}",
                f"crossref translator field: {list(translators)!r}",
                "translation lineage correctly identified by Crossref",
                qid_hint,
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if is_translation_lang and not translators:
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"source_language={source_language!r} not in figure native langs "
                f"{list(figure_native_languages)!r}, suggesting translation",
                "but Crossref translator field is EMPTY",
                "reviewer verifies: (a) is this a genuine foreign-language "
                "original by the figure, or (b) an uncited translation?",
                qid_hint,
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if not is_translation_lang and translators:
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"source_language={source_language!r} matches figure native langs",
                f"but Crossref translator field NON-EMPTY: {list(translators)!r}",
                "rare combination; reviewer verifies whether this is a "
                "re-edition that re-translated the figure's original",
                qid_hint,
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    return _check(
        verdict=Verdict.PASS,
        evidence=(
            f"source_language={source_language!r} matches figure native langs "
            f"{list(figure_native_languages)!r}",
            "Crossref translator field is empty (no translation lineage)",
            qid_hint,
            f"source_id={provenance.source_id}",
        ),
        sha=provenance.byte_sha256,
        now_iso=timestamp,
    )


__all__ = ["REVIEWER_ID", "VERIFIER_VERSION", "verify_translation_lineage"]
