"""CPAE PDF parser (Princeton Collected Papers of Albert Einstein).

CPAE volumes are distributed as facsimile PDFs with embedded OCR text
(or hand-typeset text in modern volumes). This parser uses ``pypdf``
to extract page-by-page text and concatenates them with form-feed
separators so the cleaner can later strip page boundaries / page
numbers if it chooses to.

License notice scan: many CPAE PDFs carry an inside-cover statement
``"Copyright by Princeton University Press"`` plus a ``"Permission to
reproduce..."`` clause on the first 1-2 pages. We grep for those
substrings and stash whatever line(s) contain the keyword into
``license_notice``. If absent, ``license_notice`` is empty (downstream
verifier (#28 L2 follow-up) will flag).
"""

from __future__ import annotations

import hashlib
import io

from pypdf import PdfReader

from lifeform_domain_figure.cleaning.raw_document import RawDocument

CPAE_PDF_CONTENT_TYPE = "application/pdf"
PARSER_VERSION = "cpae-pdf:1"

_LICENSE_HINTS = (
    "copyright",
    "all rights reserved",
    "permission to reproduce",
    "princeton university press",
    "public domain",
    "creative commons",
)


def _scan_license_notice(text: str) -> str:
    """Return the first line(s) hinting at a license clause, or ``""``."""

    lines = text.splitlines()
    head_lines = lines[:80]
    notice_lines: list[str] = []
    lowered_seen: set[str] = set()
    for line in head_lines:
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(hint in lowered for hint in _LICENSE_HINTS):
            if lowered not in lowered_seen:
                notice_lines.append(stripped)
                lowered_seen.add(lowered)
    return " ".join(notice_lines)


def _detect_language_from_metadata(reader: PdfReader) -> str:
    metadata = reader.metadata
    if metadata is None:
        return ""
    lang_value = metadata.get("/Lang") if hasattr(metadata, "get") else None
    if not isinstance(lang_value, str):
        return ""
    candidate = lang_value.strip().lower()
    if not candidate:
        return ""
    return candidate.split("-", 1)[0][:2]


def parse_cpae_pdf(
    data: bytes,
    *,
    source_url: str,
    content_type: str = CPAE_PDF_CONTENT_TYPE,
) -> RawDocument:
    """Parse CPAE PDF bytes into a :class:`RawDocument`."""

    if content_type != CPAE_PDF_CONTENT_TYPE:
        raise ValueError(
            f"parse_cpae_pdf: refusing content_type={content_type!r}; "
            f"expected {CPAE_PDF_CONTENT_TYPE!r}"
        )
    if not data:
        raise ValueError(
            f"parse_cpae_pdf: empty bytes for source_url={source_url!r}"
        )
    reader = PdfReader(io.BytesIO(data))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    extracted = "\f".join(page_texts).strip()
    if not extracted:
        raise ValueError(
            f"parse_cpae_pdf: extracted text is empty for source_url={source_url!r} "
            f"(pages={len(reader.pages)}); upstream OCR likely missing"
        )
    bytes_count = max(len(data), 1)
    text_byte_ratio = min(1.0, len(extracted.encode("utf-8")) / bytes_count)
    page_count = max(len(reader.pages), 1)
    chars_per_page = len(extracted) / page_count
    layout_quality = min(1.0, max(0.0, 0.5 * text_byte_ratio + 0.5 * min(1.0, chars_per_page / 800.0)))
    language_detected = _detect_language_from_metadata(reader)
    license_notice = _scan_license_notice(extracted)
    raw_sha256 = hashlib.sha256(data).hexdigest()
    return RawDocument(
        text=extracted,
        parser_version=PARSER_VERSION,
        layout_quality=layout_quality,
        ocr_confidence=1.0,
        encoding_detected="utf-8",
        language_detected=language_detected,
        license_notice=license_notice,
        raw_sha256=raw_sha256,
    )


__all__ = [
    "CPAE_PDF_CONTENT_TYPE",
    "PARSER_VERSION",
    "parse_cpae_pdf",
]
