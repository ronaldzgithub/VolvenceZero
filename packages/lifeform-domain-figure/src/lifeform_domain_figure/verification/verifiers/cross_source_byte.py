"""CROSS_SOURCE_BYTE verifier (L2 first batch).

Detects multi-source byte disagreement: the same document (e.g.,
"Einstein's letter to Besso, 1909-04-29") may appear in CPAE,
Wikisource, and Internet Archive with three different cleaned text
bodies (different OCR, different transcription conventions, different
editorial corrections). The bundle should not silently choose one
without surfacing the disagreement.

Inputs:

* ``group`` — a tuple of :class:`SourceProvenance` records the
  curator declares are versions of the same source. Grouping is
  *reviewer-asserted*; this verifier does not infer document identity
  on its own (that is the deferred ``IDENTITY_DISAMBIGUATION`` axis,
  which depends on metadata).

Verdict rules per group:

* group of size 1 -> single PASS (trivially consistent)
* all members share the same ``byte_sha256`` -> all PASS
* members disagree on ``byte_sha256`` -> all NEEDS_REVIEW (curator
  must adjudicate which version is canonical; the gate then either
  blocks the bundle or admits the chosen one with an override)

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
REVIEWER_ID = f"auto:cross_source_byte:{VERIFIER_VERSION}"


def verify_cross_source_byte(
    group: tuple[SourceProvenance, ...],
    *,
    document_group_key: str,
    now_iso: str | None = None,
) -> tuple[VerificationCheck, ...]:
    """Return one :class:`VerificationCheck` per provenance in ``group``.

    ``document_group_key`` is the curator-supplied label that names
    the asserted shared document identity (e.g.,
    ``"einstein-besso-1909-04-29"``); it is included in every
    check's evidence so a downstream reviewer can see which group
    triggered which verdict.
    """

    if not isinstance(group, tuple) or not group:
        raise ValueError(
            "verify_cross_source_byte: group must be a non-empty tuple of "
            "SourceProvenance records"
        )
    if not isinstance(document_group_key, str) or not document_group_key.strip():
        raise ValueError(
            "verify_cross_source_byte: document_group_key must be a non-empty string"
        )
    timestamp = now_iso or datetime.now(timezone.utc).isoformat()
    if len(group) == 1:
        only = group[0]
        return (
            VerificationCheck(
                check_kind=CheckKind.CROSS_SOURCE_BYTE,
                verdict=Verdict.PASS,
                evidence=(
                    f"document_group_key={document_group_key!r}",
                    "group size = 1; trivially consistent",
                    f"source_id={only.source_id}",
                    f"byte_sha256={only.byte_sha256}",
                ),
                reviewer_id=REVIEWER_ID,
                reviewed_at_iso=timestamp,
                source_byte_sha256=only.byte_sha256,
            ),
        )
    distinct_hashes = {prov.byte_sha256 for prov in group}
    if len(distinct_hashes) == 1:
        evidence_common = (
            f"document_group_key={document_group_key!r}",
            f"group_size={len(group)}",
            "all members share the same byte_sha256",
        )
        return tuple(
            VerificationCheck(
                check_kind=CheckKind.CROSS_SOURCE_BYTE,
                verdict=Verdict.PASS,
                evidence=evidence_common
                + (
                    f"source_id={prov.source_id}",
                    f"byte_sha256={prov.byte_sha256}",
                ),
                reviewer_id=REVIEWER_ID,
                reviewed_at_iso=timestamp,
                source_byte_sha256=prov.byte_sha256,
            )
            for prov in group
        )
    conflict_lines = tuple(
        f"member: source_id={prov.source_id} source_url={prov.source_url} "
        f"byte_sha256={prov.byte_sha256}"
        for prov in group
    )
    evidence_common = (
        f"document_group_key={document_group_key!r}",
        f"group_size={len(group)}",
        f"distinct_byte_sha256_count={len(distinct_hashes)}",
        "members disagree on byte_sha256; reviewer must select canonical",
    ) + conflict_lines
    return tuple(
        VerificationCheck(
            check_kind=CheckKind.CROSS_SOURCE_BYTE,
            verdict=Verdict.NEEDS_REVIEW,
            evidence=evidence_common
            + (f"this_member_source_id={prov.source_id}",),
            reviewer_id=REVIEWER_ID,
            reviewed_at_iso=timestamp,
            source_byte_sha256=prov.byte_sha256,
        )
        for prov in group
    )


__all__ = ["REVIEWER_ID", "VERIFIER_VERSION", "verify_cross_source_byte"]
