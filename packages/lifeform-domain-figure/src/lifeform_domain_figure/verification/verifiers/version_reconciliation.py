"""VERSION_RECONCILIATION verifier (L2 second batch).

Detects whether a source DOI has multiple Crossref-recognised
versions (preprint vs published, journal vs reprint, original vs
errata). The verifier surfaces them so the curator chooses one
canonical version rather than letting OpenAlex / cleaner silently
pick "whatever DOI was supplied first".

Heuristics (full-mode):

1. **Single-DOI lookup**: fetch the source's DOI via the V2
   :class:`CrossrefClient` (cached). On any fetch failure ->
   NEEDS_REVIEW with the failure reason in evidence.
2. **Relation map**: read Crossref ``relation`` for keys
   ``is-version-of`` / ``replaces`` / ``replaced-by`` /
   ``is-preprint-of`` / ``has-preprint`` /
   ``is-translation-of``. Combine the related DOIs into a
   ``related_dois`` set (excluding self).
3. **Verdict**:
   * Empty ``related_dois`` -> PASS (no other versions known).
   * Non-empty AND the supplied ``canonical_doi_hint`` matches the
     source's DOI -> PASS with ``related_dois`` listed in evidence
     (curator already chose canonical).
   * Non-empty AND ``canonical_doi_hint`` is empty or differs ->
     NEEDS_REVIEW with full DOI list and a "pick canonical via
     publication-date sort" recommendation.

The verifier never auto-rejects (FAIL) because version
disagreement is a legitimate workflow decision (curator may
intentionally choose preprint over journal); it only flags for
human review.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.corpus.provenance import SourceProvenance
from lifeform_domain_figure.metadata.crossref import (
    CrossrefClient,
    crossref_relations,
)
from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)

VERIFIER_VERSION = "1"
REVIEWER_ID = f"auto:version_reconciliation:{VERIFIER_VERSION}"


_VERSION_RELATION_KINDS = (
    "is-version-of",
    "replaces",
    "replaced-by",
    "is-preprint-of",
    "has-preprint",
    "is-translation-of",
    "has-translation",
)


def _check(
    *,
    verdict: Verdict,
    evidence: tuple[str, ...],
    sha: str,
    now_iso: str,
) -> VerificationCheck:
    return VerificationCheck(
        check_kind=CheckKind.VERSION_RECONCILIATION,
        verdict=verdict,
        evidence=evidence,
        reviewer_id=REVIEWER_ID,
        reviewed_at_iso=now_iso,
        source_byte_sha256=sha,
    )


def verify_version_reconciliation(
    provenance: SourceProvenance,
    *,
    crossref_client: CrossrefClient,
    source_doi: str,
    canonical_doi_hint: str = "",
    now_iso: str | None = None,
) -> VerificationCheck:
    """Return a :class:`VerificationCheck` for the version-reconciliation axis.

    ``source_doi`` is the DOI of the source under audit. The
    ``crossref_client`` MUST have a ``fetch_raw_message(doi=...)``
    method (live client does); the duck-typed access lets reviewer
    test fixtures swap in a stub without subclassing.
    """

    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    if not source_doi.strip():
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                "source_doi is empty",
                "verifier needs a typed Crossref DOI to reconcile versions",
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
                "verifier needs Crossref's full message payload to read 'relation'; "
                "use live_crossref_client(...) or supply a stub with fetch_raw_message",
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
                "verifier defers to human reviewer when metadata fetch fails",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if not isinstance(message, dict):
        return _check(
            verdict=Verdict.NEEDS_REVIEW,
            evidence=(
                f"crossref message for doi={source_doi!r} is not a dict; "
                f"got {type(message).__name__}",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    relations = crossref_relations(message)
    related_dois: set[str] = set()
    matched_kinds: list[str] = []
    for kind in _VERSION_RELATION_KINDS:
        items = relations.get(kind, ())
        if items:
            matched_kinds.append(kind)
            for d in items:
                if d != source_doi:
                    related_dois.add(d)
    if not related_dois:
        return _check(
            verdict=Verdict.PASS,
            evidence=(
                f"crossref relation map has no version-class entries for "
                f"doi={source_doi!r}",
                f"checked relation kinds: {list(_VERSION_RELATION_KINDS)!r}",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    if canonical_doi_hint and canonical_doi_hint == source_doi:
        return _check(
            verdict=Verdict.PASS,
            evidence=(
                f"doi={source_doi!r} declared canonical by curator "
                f"(canonical_doi_hint match)",
                f"related versions found: {sorted(related_dois)!r}",
                f"matched relation kinds: {matched_kinds!r}",
                f"source_id={provenance.source_id}",
            ),
            sha=provenance.byte_sha256,
            now_iso=timestamp,
        )
    return _check(
        verdict=Verdict.NEEDS_REVIEW,
        evidence=(
            f"doi={source_doi!r} has multiple related versions in Crossref",
            f"related_dois={sorted(related_dois)!r}",
            f"matched relation kinds: {matched_kinds!r}",
            f"canonical_doi_hint={canonical_doi_hint!r} (empty or non-matching)",
            "reviewer must pick canonical (recommend publication-date sort: "
            "earliest preprint vs latest journal version)",
            f"source_id={provenance.source_id}",
        ),
        sha=provenance.byte_sha256,
        now_iso=timestamp,
    )


__all__ = ["REVIEWER_ID", "VERIFIER_VERSION", "verify_version_reconciliation"]
