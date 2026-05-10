"""Project Gutenberg parser (HTML or plain-text variant).

Gutenberg ships each ebook in two delivery formats the figure
vertical may meet in the wild:

* ``text/plain`` — bare text, often Latin-1 encoded, wrapped between
  ``*** START OF THIS PROJECT GUTENBERG EBOOK ... ***`` and
  ``*** END OF THIS PROJECT GUTENBERG EBOOK ... ***`` markers. The
  block before START is a license / copyright notice; the block after
  END is project boilerplate. Both must be stripped from the body and
  the START block is captured into ``license_notice``.
* ``text/html`` — Gutenberg's rendered HTML. Same START/END markers
  appear as text inside the body once tags are stripped.

The parser handles both, returning a ``RawDocument`` whose ``text``
contains only the work itself with the START/END boilerplate
captured as license notice.
"""

from __future__ import annotations

import hashlib
import re

from bs4 import BeautifulSoup

from lifeform_domain_figure.cleaning.raw_document import RawDocument

GUTENBERG_HTML_CONTENT_TYPE = "text/html; profile=gutenberg"
GUTENBERG_TEXT_CONTENT_TYPE = "text/plain; profile=gutenberg"
PARSER_VERSION = "gutenberg:1"

_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)


def _decode_bytes(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8"


def _strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "lxml")
    body = soup.body or soup
    return body.get_text("\n", strip=False)


def _split_around_markers(text: str) -> tuple[str, str]:
    """Return ``(body, license_notice)`` carved from START/END markers."""

    start_match = _START_RE.search(text)
    end_match = _END_RE.search(text)
    license_notice = ""
    if start_match is not None:
        preface = text[: start_match.start()].strip()
        license_notice = preface
        body_start = start_match.end()
    else:
        body_start = 0
    body_end = end_match.start() if end_match is not None else len(text)
    body = text[body_start:body_end].strip()
    return body, license_notice


def parse_gutenberg(
    data: bytes,
    *,
    source_url: str,
    content_type: str = GUTENBERG_TEXT_CONTENT_TYPE,
) -> RawDocument:
    """Parse Gutenberg HTML or plain-text bytes into a :class:`RawDocument`."""

    if content_type not in {GUTENBERG_HTML_CONTENT_TYPE, GUTENBERG_TEXT_CONTENT_TYPE}:
        raise ValueError(
            f"parse_gutenberg: refusing content_type={content_type!r}; "
            f"expected one of "
            f"{(GUTENBERG_HTML_CONTENT_TYPE, GUTENBERG_TEXT_CONTENT_TYPE)!r}"
        )
    if not data:
        raise ValueError(
            f"parse_gutenberg: empty bytes for source_url={source_url!r}"
        )
    decoded, encoding_detected = _decode_bytes(data)
    if content_type == GUTENBERG_HTML_CONTENT_TYPE:
        decoded = _strip_html(decoded)
    body, license_notice = _split_around_markers(decoded)
    if not body.strip():
        raise ValueError(
            f"parse_gutenberg: body is empty after START/END strip for "
            f"source_url={source_url!r}"
        )
    raw_sha256 = hashlib.sha256(data).hexdigest()
    return RawDocument(
        text=body,
        parser_version=PARSER_VERSION,
        layout_quality=1.0,
        ocr_confidence=1.0,
        encoding_detected=encoding_detected,
        language_detected="",
        license_notice=license_notice,
        raw_sha256=raw_sha256,
    )


__all__ = [
    "GUTENBERG_HTML_CONTENT_TYPE",
    "GUTENBERG_TEXT_CONTENT_TYPE",
    "PARSER_VERSION",
    "parse_gutenberg",
]
