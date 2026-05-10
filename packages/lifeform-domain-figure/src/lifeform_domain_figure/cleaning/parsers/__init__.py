"""Parsers turn raw bytes into a typed :class:`RawDocument`.

The four parsers cover the formats the figure vertical needs to ingest
from the four V1 archives:

* :func:`parse_cpae_pdf` — Princeton CPAE PDF (``application/pdf``)
* :func:`parse_wikisource_html` — Wikisource MediaWiki page HTML
* :func:`parse_gutenberg` — Project Gutenberg HTML or plain text
* :func:`parse_archive_org_ocr_json` — Internet Archive DjVu OCR JSON

Each parser is a pure function::

    parser(data: bytes, *, source_url: str, content_type: str = ...) -> RawDocument

Parsers MUST NOT issue HTTP requests, MUST NOT touch the filesystem
(except as ``pypdf`` does internally on ``BytesIO``), and MUST NOT
import any figure-vertical typed source record (``FigurePaperSource``
etc.). Crossing into typed sources is the job of
:mod:`lifeform_domain_figure.cleaning.bridging`.

A small dispatcher :func:`parse_by_content_type` picks the right parser
from the MIME / pseudo-MIME label so the CLI can run without each
caller hardcoding the choice.
"""

from __future__ import annotations

from typing import Protocol

from lifeform_domain_figure.cleaning.parsers.archive_org_ocr import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
    parse_archive_org_ocr_json,
)
from lifeform_domain_figure.cleaning.parsers.cpae_pdf import (
    CPAE_PDF_CONTENT_TYPE,
    parse_cpae_pdf,
)
from lifeform_domain_figure.cleaning.parsers.gutenberg import (
    GUTENBERG_HTML_CONTENT_TYPE,
    GUTENBERG_TEXT_CONTENT_TYPE,
    parse_gutenberg,
)
from lifeform_domain_figure.cleaning.parsers.wikisource_html import (
    WIKISOURCE_HTML_CONTENT_TYPE,
    parse_wikisource_html,
)
from lifeform_domain_figure.cleaning.raw_document import RawDocument


class Parser(Protocol):
    """Pure-function shape every parser exposes."""

    def __call__(
        self,
        data: bytes,
        *,
        source_url: str,
        content_type: str,
    ) -> RawDocument: ...


def parse_by_content_type(
    data: bytes,
    *,
    source_url: str,
    content_type: str,
) -> RawDocument:
    """Dispatch ``data`` to the parser matching ``content_type``.

    The accepted ``content_type`` labels are the ones each parser
    module exports as ``*_CONTENT_TYPE`` constants. Unknown labels
    raise ``ValueError`` (no silent fallback — debt #28 requires
    explicit choice of parser per source).
    """

    if content_type == CPAE_PDF_CONTENT_TYPE:
        return parse_cpae_pdf(data, source_url=source_url, content_type=content_type)
    if content_type == WIKISOURCE_HTML_CONTENT_TYPE:
        return parse_wikisource_html(
            data, source_url=source_url, content_type=content_type
        )
    if content_type in {GUTENBERG_HTML_CONTENT_TYPE, GUTENBERG_TEXT_CONTENT_TYPE}:
        return parse_gutenberg(data, source_url=source_url, content_type=content_type)
    if content_type == ARCHIVE_ORG_OCR_CONTENT_TYPE:
        return parse_archive_org_ocr_json(
            data, source_url=source_url, content_type=content_type
        )
    raise ValueError(
        f"parse_by_content_type: no parser registered for content_type="
        f"{content_type!r}; expected one of "
        f"({CPAE_PDF_CONTENT_TYPE!r}, {WIKISOURCE_HTML_CONTENT_TYPE!r}, "
        f"{GUTENBERG_HTML_CONTENT_TYPE!r}, {GUTENBERG_TEXT_CONTENT_TYPE!r}, "
        f"{ARCHIVE_ORG_OCR_CONTENT_TYPE!r})"
    )


__all__ = [
    "ARCHIVE_ORG_OCR_CONTENT_TYPE",
    "CPAE_PDF_CONTENT_TYPE",
    "GUTENBERG_HTML_CONTENT_TYPE",
    "GUTENBERG_TEXT_CONTENT_TYPE",
    "Parser",
    "WIKISOURCE_HTML_CONTENT_TYPE",
    "parse_archive_org_ocr_json",
    "parse_by_content_type",
    "parse_cpae_pdf",
    "parse_gutenberg",
    "parse_wikisource_html",
]
