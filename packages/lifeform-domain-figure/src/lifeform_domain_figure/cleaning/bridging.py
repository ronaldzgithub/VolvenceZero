"""Bridge cleaned text into the existing archive ``*Payload`` schemas.

The L1 cleaning pipeline produces a neutral
:class:`CleanedDocument`. Crossing into a figure-vertical typed
source record (``FigurePaperSource`` / ``FigureLetterSource`` / etc.)
is intentionally a TWO-step process to keep cleaner / parser
decoupled from typed source choice (R8):

1. ``cleaned_to_*_payload(...)`` — pure function in this module — wraps
   the cleaned text in the existing archive payload (``CPAEPayload``
   / ``WikisourcePayload`` / ``GutenbergPayload`` /
   ``InternetArchivePayload``). The curator / caller still supplies
   archive-specific metadata (``volume`` / ``document_number`` /
   ``page_title`` / etc.) that the parser cannot know.
2. ``<archive>_to_<source_kind>_source(...)`` — already existing in
   :mod:`lifeform_domain_figure.corpus.archives` — translates the
   payload into the typed ``Figure*Source``.

The bridging functions are pure and own no state. They never touch
the filesystem and never touch the cleaning store; the call site is
expected to read the cleaned text from the store first (via
``CleaningStore.get_cleaned``) when they need persistence.

Provenance bridge (debt #28 L2, recommendation 5)
-------------------------------------------------

In addition to the four ``cleaned_to_*_payload`` helpers, this
module exposes :func:`cleaned_to_source_provenance` which propagates
the L1 ``RawDocument.license_notice`` into the L2-facing
:class:`SourceProvenance.license_label`. This is the single seam
where parser-scraped license evidence flows into the verifier's
input. The cleaned text's ``raw_sha256`` lands in
:attr:`SourceProvenance.byte_sha256`, making the L1 / L2 chain
content-addressable end-to-end.
"""

from __future__ import annotations

from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    RawDocument,
)
from lifeform_domain_figure.corpus.archives.cpae import (
    CPAEDocumentKind,
    CPAEPayload,
)
from lifeform_domain_figure.corpus.archives.gutenberg import GutenbergPayload
from lifeform_domain_figure.corpus.archives.internet_archive import (
    InternetArchivePayload,
)
from lifeform_domain_figure.corpus.archives.wikisource import WikisourcePayload
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)


L1_LICENSE_SENTINEL = "(no license notice scraped)"
"""Sentinel string written to ``SourceProvenance.license_label`` when
neither the L1 parser scraped a license_notice nor the curator
supplied a ``license_label_override``. The L2
``LICENSE_PAGE_LEVEL`` verifier recognises this exact string and
emits ``NEEDS_REVIEW`` so the bundle gate refuses the source under
``require_verification_pass=True`` until a curator supplies a real
label."""


def cleaned_to_cpae_payload(
    cleaned: CleanedDocument,
    *,
    document_id: str,
    document_kind: CPAEDocumentKind,
    volume: int,
    document_number: int,
    title: str,
    year: int,
    language: str,
    source_url: str,
    sender_id: str = "",
    recipient_id: str = "",
    date_iso: str = "",
) -> CPAEPayload:
    """Wrap a :class:`CleanedDocument` as a :class:`CPAEPayload`.

    The cleaned ``text`` is the payload's ``body``. All
    archive-specific structural metadata (volume / document_number /
    title / year / language / sender / recipient / date) must be
    supplied by the curator: the cleaning pipeline does not infer
    them and the parser only sees raw bytes.
    """

    return CPAEPayload(
        document_id=document_id,
        document_kind=document_kind,
        volume=volume,
        document_number=document_number,
        title=title,
        year=year,
        language=language,
        body=cleaned.text,
        source_url=source_url,
        sender_id=sender_id,
        recipient_id=recipient_id,
        date_iso=date_iso,
    )


def cleaned_to_wikisource_payload(
    cleaned: CleanedDocument,
    *,
    page_title: str,
    language: str,
    source_url: str,
    year: int | None = None,
    author_id: str = "",
    venue_id: str = "",
    date_iso: str = "",
    audience: str = "",
) -> WikisourcePayload:
    """Wrap a :class:`CleanedDocument` as a :class:`WikisourcePayload`."""

    return WikisourcePayload(
        page_title=page_title,
        language=language,
        source_url=source_url,
        body=cleaned.text,
        year=year,
        author_id=author_id,
        venue_id=venue_id,
        date_iso=date_iso,
        audience=audience,
    )


def cleaned_to_gutenberg_payload(
    cleaned: CleanedDocument,
    *,
    ebook_id: int,
    title: str,
    language: str,
    source_url: str,
    section_label: str = "",
    year: int | None = None,
    author_id: str = "",
) -> GutenbergPayload:
    """Wrap a :class:`CleanedDocument` as a :class:`GutenbergPayload`."""

    return GutenbergPayload(
        ebook_id=ebook_id,
        title=title,
        language=language,
        body=cleaned.text,
        source_url=source_url,
        section_label=section_label,
        year=year,
        author_id=author_id,
    )


def cleaned_to_internet_archive_payload(
    cleaned: CleanedDocument,
    *,
    identifier: str,
    title: str,
    language: str,
    source_url: str,
    creator_id: str = "",
    year: int | None = None,
    venue_id: str = "",
    date_iso: str = "",
    audience: str = "",
) -> InternetArchivePayload:
    """Wrap a :class:`CleanedDocument` as an :class:`InternetArchivePayload`."""

    return InternetArchivePayload(
        identifier=identifier,
        title=title,
        language=language,
        body=cleaned.text,
        source_url=source_url,
        creator_id=creator_id,
        year=year,
        venue_id=venue_id,
        date_iso=date_iso,
        audience=audience,
    )


def cleaned_to_source_provenance(
    cleaned: CleanedDocument,
    raw: RawDocument,
    *,
    source_id: str,
    figure_id: str,
    source_url: str,
    legal_clearance: LegalClearance,
    capture_method: CaptureMethod,
    captured_by: str,
    captured_at_iso: str,
    provenance_note: str,
    license_label_override: str = "",
    jurisdiction_hint: str = "",
) -> SourceProvenance:
    """Wrap a cleaned/raw pair as a :class:`SourceProvenance` for L2.

    Forwards the L1 anchor ``raw.raw_sha256`` into
    :attr:`SourceProvenance.byte_sha256` so the L1 cleaning store
    and the L2 verification ledger share the same content-addressable
    key for one source.

    The ``license_label`` field resolves with priority:

    1. ``license_label_override`` (curator-supplied; wins),
    2. ``raw.license_notice`` (L1 parser-scraped; default),
    3. :data:`L1_LICENSE_SENTINEL` (when neither is present; the
       L2 ``LICENSE_PAGE_LEVEL`` verifier reads this sentinel and
       returns ``NEEDS_REVIEW`` so the gate fails loudly).

    The cleaned/raw mismatch in ``raw_sha256`` is also enforced so
    callers cannot accidentally pair a ``CleanedDocument`` with a
    ``RawDocument`` from a different source.
    """

    if cleaned.raw_sha256 != raw.raw_sha256:
        raise ValueError(
            f"cleaned_to_source_provenance: cleaned.raw_sha256 "
            f"({cleaned.raw_sha256!r}) does not match raw.raw_sha256 "
            f"({raw.raw_sha256!r}); refusing to fabricate a provenance "
            f"record for mismatched cleaning anchors."
        )
    if license_label_override.strip():
        license_label = license_label_override
    elif raw.license_notice.strip():
        license_label = raw.license_notice
    else:
        license_label = L1_LICENSE_SENTINEL
    return SourceProvenance(
        source_id=source_id,
        figure_id=figure_id,
        source_url=source_url,
        license_label=license_label,
        legal_clearance=legal_clearance,
        capture_method=capture_method,
        captured_by=captured_by,
        captured_at_iso=captured_at_iso,
        byte_sha256=raw.raw_sha256,
        provenance_note=provenance_note,
        jurisdiction_hint=jurisdiction_hint,
    )


__all__ = [
    "L1_LICENSE_SENTINEL",
    "cleaned_to_cpae_payload",
    "cleaned_to_gutenberg_payload",
    "cleaned_to_internet_archive_payload",
    "cleaned_to_source_provenance",
    "cleaned_to_wikisource_payload",
]
