"""Neutral verification record schema for the L2 verification pipeline.

The verification pipeline is figure-vertical-internal corpus auditing.
It sits **after** the L1 cleaning pipeline (a SourceProvenance is
attached to each cleaned source) and **before** the bundle build (a
gate decides whether a source is admissible).

This module owns only the neutral schema; verifiers, ledger, and
bundle gate live in sibling modules.

Three records:

* :class:`CheckKind` — closed enum of 7 verification axes (3 are
  implemented in this packet; 4 are deferred until the metadata
  client (debt #26) lands).
* :class:`Verdict` — closed enum of 3 outcomes (``PASS`` / ``FAIL``
  / ``NEEDS_REVIEW``). NEEDS_REVIEW means the auto verifier could
  not decide; a human must adjudicate.
* :class:`VerificationCheck` — one verifier verdict, anchored to a
  ``source_byte_sha256`` (the same hash that appears in
  :class:`SourceProvenance.byte_sha256` and L1
  :class:`RawDocument.raw_sha256`). Multiple checks per anchor are
  allowed (per-kind history); the ledger's ``latest_per_kind``
  reduces them to a single effective verdict per kind.

reviewer_id format
------------------

* ``auto:<verifier_id>:<int>`` for auto verifiers (e.g.,
  ``"auto:date_plausibility:1"``). The integer monotone-increments
  when a verifier's behaviour changes.
* ``human:<reviewer-id>`` for human adjudication entries. The string
  after ``human:`` is opaque (whatever the curator CLI passes); a
  later packet may add a reviewer registry.

The format is enforced by ``__post_init__`` so a "latest wins" reduce
is unambiguous about which kind of reviewer wrote which check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CheckKind(str, Enum):
    """Closed vocabulary of the 7 verification axes for figure corpora.

    The first three (``DATE_PLAUSIBILITY`` / ``LICENSE_PAGE_LEVEL`` /
    ``CROSS_SOURCE_BYTE``) are the L2 first batch — pure-function,
    no external metadata dependency.

    The last four are deferred until the metadata client (debt #26)
    lands; their entries in :data:`AUTO_VERIFIERS` raise
    :class:`NotImplementedError` with a pointer to the follow-up
    packet rather than silently passing.
    """

    DATE_PLAUSIBILITY = "date_plausibility"
    LICENSE_PAGE_LEVEL = "license_page_level"
    CROSS_SOURCE_BYTE = "cross_source_byte"
    IDENTITY_DISAMBIGUATION = "identity_disambiguation"
    AUTHORSHIP_ATTRIBUTION = "authorship_attribution"
    VERSION_RECONCILIATION = "version_reconciliation"
    TRANSLATION_LINEAGE = "translation_lineage"


IMPLEMENTED_CHECK_KINDS: frozenset[CheckKind] = frozenset(
    {
        CheckKind.DATE_PLAUSIBILITY,
        CheckKind.LICENSE_PAGE_LEVEL,
        CheckKind.CROSS_SOURCE_BYTE,
        CheckKind.IDENTITY_DISAMBIGUATION,
        CheckKind.AUTHORSHIP_ATTRIBUTION,
        CheckKind.VERSION_RECONCILIATION,
        CheckKind.TRANSLATION_LINEAGE,
    }
)
"""The full set of :class:`CheckKind` that L2 implements.

As of debt #28 L2 second batch (2026-05-10), all 7 kinds have real
verifiers (the second-batch four depend on the V2 metadata clients
landed by debt #26 closure). The bundle gate
(:func:`lifeform_domain_figure.verification.gate.assert_all_provenances_pass`)
requires all-PASS across this set; the contract test
``tests/contracts/test_bundle_admits_only_verified_sources.py``
fences the assumption.
"""


class Verdict(str, Enum):
    """Closed vocabulary of verification outcomes.

    ``PASS`` and ``FAIL`` are decisive. ``NEEDS_REVIEW`` means the
    auto verifier could not classify the input (typically: an empty
    sentinel field, an ambiguous phrase, or a multi-source disagreement
    that requires curator judgement).
    """

    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class VerificationCheck:
    """One verifier verdict on one source byte stream.

    ``source_byte_sha256`` is the content-addressable anchor used by
    the ledger. It MUST equal :class:`SourceProvenance.byte_sha256`
    for the same source, which itself equals
    :class:`RawDocument.raw_sha256` from the L1 pipeline. This is the
    single key that walks raw-bytes → cleaned text → provenance →
    verification verdict.

    ``evidence`` carries human-readable bullets (each a short string
    describing one piece of evidence the verifier considered). It is
    persisted verbatim into the ledger so a reviewer auditing the
    decision later sees exactly what the verifier saw.
    """

    check_kind: CheckKind
    verdict: Verdict
    evidence: tuple[str, ...]
    reviewer_id: str
    reviewed_at_iso: str
    source_byte_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError(
                f"VerificationCheck.evidence must be a non-empty tuple of "
                f"strings (check_kind={self.check_kind.value!r})"
            )
        for index, item in enumerate(self.evidence):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"VerificationCheck.evidence[{index}] must be a non-empty "
                    f"string (check_kind={self.check_kind.value!r})"
                )
        if not isinstance(self.reviewer_id, str) or ":" not in self.reviewer_id:
            raise ValueError(
                f"VerificationCheck.reviewer_id must be of the form "
                f"'auto:<verifier_id>:<int>' or 'human:<reviewer-id>'; "
                f"got {self.reviewer_id!r}"
            )
        kind_prefix = self.reviewer_id.split(":", 1)[0]
        if kind_prefix not in {"auto", "human"}:
            raise ValueError(
                f"VerificationCheck.reviewer_id prefix must be 'auto' or "
                f"'human'; got {self.reviewer_id!r}"
            )
        if not isinstance(self.reviewed_at_iso, str) or not self.reviewed_at_iso.strip():
            raise ValueError(
                "VerificationCheck.reviewed_at_iso must be a non-empty ISO-8601 string"
            )
        if (
            not isinstance(self.source_byte_sha256, str)
            or len(self.source_byte_sha256) != 64
        ):
            raise ValueError(
                f"VerificationCheck.source_byte_sha256 must be a 64-char hex "
                f"sha256; got {self.source_byte_sha256!r}"
            )


__all__ = [
    "CheckKind",
    "IMPLEMENTED_CHECK_KINDS",
    "Verdict",
    "VerificationCheck",
]
