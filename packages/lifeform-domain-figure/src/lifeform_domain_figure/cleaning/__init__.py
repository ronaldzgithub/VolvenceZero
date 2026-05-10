"""Figure-vertical L1 corpus cleaning pipeline.

This subpackage implements layer L1 of `docs/known-debts.md` debt #28
(see `docs/specs/figure-corpus-cleaning.md` for the spec):

    bytes + content_type + source_url
        -> parser   (cleaning/parsers/*)
        -> RawDocument
        -> cleaner pipeline (cleaning/cleaners/*)
        -> CleanedDocument
        -> bridging.cleaned_to_*_payload (curator metadata supplied)
        -> existing CPAEPayload / WikisourcePayload / ... archive payload
        -> existing *_to_*_source translator
        -> existing FigurePaperSource / FigureLetterSource / ...

Layer L0 (crawl orchestration) and L2 (multi-source verification) are
explicit follow-ups; this packet only implements L1.

Public surface:

* Schema: :class:`RawDocument`, :class:`CleanedDocument`,
  :class:`CleaningOpRecord`, :class:`CleaningOp`.
* Parsers: :func:`parse_cpae_pdf`, :func:`parse_wikisource_html`,
  :func:`parse_gutenberg`, :func:`parse_archive_org_ocr_json`,
  plus the dispatcher :func:`parse_by_content_type`.
* Cleaners: :func:`clean_raw_document` (orchestrator),
  :data:`CURRENT_CLEANER_PIPELINE_VERSION`, :func:`cleaner_for`.
* Storage: :class:`CleaningStore`, :class:`RawSidecar`.
* Bridging: :func:`cleaned_to_cpae_payload` /
  :func:`cleaned_to_wikisource_payload` /
  :func:`cleaned_to_gutenberg_payload` /
  :func:`cleaned_to_internet_archive_payload`.
"""

from __future__ import annotations

from lifeform_domain_figure.cleaning.bridging import (
    cleaned_to_cpae_payload,
    cleaned_to_gutenberg_payload,
    cleaned_to_internet_archive_payload,
    cleaned_to_wikisource_payload,
)
from lifeform_domain_figure.cleaning.cleaners import (
    CLEANER_PIPELINE_V1,
    CURRENT_CLEANER_PIPELINE_VERSION,
    clean_raw_document,
    cleaner_for,
)
from lifeform_domain_figure.cleaning.parsers import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
    CPAE_PDF_CONTENT_TYPE,
    GUTENBERG_HTML_CONTENT_TYPE,
    GUTENBERG_TEXT_CONTENT_TYPE,
    WIKISOURCE_HTML_CONTENT_TYPE,
    parse_archive_org_ocr_json,
    parse_by_content_type,
    parse_cpae_pdf,
    parse_gutenberg,
    parse_wikisource_html,
)
from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
    RawDocument,
)
from lifeform_domain_figure.cleaning.store import CleaningStore, RawSidecar


__all__ = [
    "ARCHIVE_ORG_OCR_CONTENT_TYPE",
    "CLEANER_PIPELINE_V1",
    "CPAE_PDF_CONTENT_TYPE",
    "CURRENT_CLEANER_PIPELINE_VERSION",
    "CleanedDocument",
    "CleaningOp",
    "CleaningOpRecord",
    "CleaningStore",
    "GUTENBERG_HTML_CONTENT_TYPE",
    "GUTENBERG_TEXT_CONTENT_TYPE",
    "RawDocument",
    "RawSidecar",
    "WIKISOURCE_HTML_CONTENT_TYPE",
    "clean_raw_document",
    "cleaned_to_cpae_payload",
    "cleaned_to_gutenberg_payload",
    "cleaned_to_internet_archive_payload",
    "cleaned_to_wikisource_payload",
    "cleaner_for",
    "parse_archive_org_ocr_json",
    "parse_by_content_type",
    "parse_cpae_pdf",
    "parse_gutenberg",
    "parse_wikisource_html",
]
