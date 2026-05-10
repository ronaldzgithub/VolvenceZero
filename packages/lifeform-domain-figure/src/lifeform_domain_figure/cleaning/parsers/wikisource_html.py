"""Wikisource MediaWiki page parser.

A Wikisource page download (``?action=raw`` or rendered HTML) needs:

1. Strip the MediaWiki chrome (sidebar, footer, edit links) and keep
   only the article body inside ``#mw-content-text``.
2. Parse remaining MediaWiki templates (``{{...}}`` / ``[[...]]``) so
   ``{{PD-old}}`` license tags are captured into ``license_notice`` and
   the visible reading order is preserved without template noise.

We accept either rendered HTML (``text/html`` from the standard URL)
or raw wikitext (``text/x-wiki``). Both arrive on the same parser
because Wikisource fetcher implementations may pick either.

Language is read from the ``<html lang="...">`` attribute when HTML
is supplied; for raw wikitext the parser falls back to scanning a
``{{header| ... |language=...}}`` template if present.
"""

from __future__ import annotations

import hashlib
import re

import mwparserfromhell
from bs4 import BeautifulSoup

from lifeform_domain_figure.cleaning.raw_document import RawDocument

WIKISOURCE_HTML_CONTENT_TYPE = "text/html; profile=wikisource"
WIKISOURCE_WIKITEXT_CONTENT_TYPE = "text/x-wiki"
PARSER_VERSION = "wikisource-html:1"

_LICENSE_TEMPLATE_PREFIXES = (
    "pd-",
    "public domain",
    "cc-",
    "license",
    "copyright",
)


def _decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode ``data`` as UTF-8 with Latin-1 fallback."""
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _strip_html_to_wikitext(text: str) -> tuple[str, str]:
    """Pull article body text + ``<html lang>`` attribute from rendered HTML.

    Returns a tuple ``(body_text, language_attr)``. Falls back to the
    full document when ``#mw-content-text`` is absent.
    """

    soup = BeautifulSoup(text, "lxml")
    html_tag = soup.find("html")
    language_attr = ""
    if html_tag is not None:
        lang_value = html_tag.get("lang", "")
        if isinstance(lang_value, str):
            language_attr = lang_value.strip().split("-", 1)[0][:2]
    body = soup.find(id="mw-content-text") or soup.body or soup
    return body.get_text("\n", strip=False), language_attr


def _wikitext_to_text(wikitext: str) -> tuple[str, list[str]]:
    """Strip wikitext templates, returning (visible_text, license_notes)."""

    parsed = mwparserfromhell.parse(wikitext)
    license_notes: list[str] = []
    for template in list(parsed.filter_templates()):
        try:
            template_name = template.name.strip_code().strip().lower()
        except (AttributeError, ValueError):
            template_name = ""
        if any(template_name.startswith(prefix) for prefix in _LICENSE_TEMPLATE_PREFIXES):
            license_notes.append(str(template).strip())
    visible = parsed.strip_code(normalize=True, collapse=True)
    return visible, license_notes


_HEADER_LANGUAGE_RE = re.compile(r"\|\s*language\s*=\s*([A-Za-z\-]{2,8})", re.IGNORECASE)


def parse_wikisource_html(
    data: bytes,
    *,
    source_url: str,
    content_type: str = WIKISOURCE_HTML_CONTENT_TYPE,
) -> RawDocument:
    """Parse Wikisource HTML or wikitext bytes into a :class:`RawDocument`."""

    if content_type not in {WIKISOURCE_HTML_CONTENT_TYPE, WIKISOURCE_WIKITEXT_CONTENT_TYPE}:
        raise ValueError(
            f"parse_wikisource_html: refusing content_type={content_type!r}; "
            f"expected one of "
            f"{(WIKISOURCE_HTML_CONTENT_TYPE, WIKISOURCE_WIKITEXT_CONTENT_TYPE)!r}"
        )
    if not data:
        raise ValueError(
            f"parse_wikisource_html: empty bytes for source_url={source_url!r}"
        )
    decoded, encoding_detected = _decode_bytes(data)
    if content_type == WIKISOURCE_HTML_CONTENT_TYPE:
        wikitext_or_text, language_html = _strip_html_to_wikitext(decoded)
    else:
        wikitext_or_text, language_html = decoded, ""
    visible, license_notes = _wikitext_to_text(wikitext_or_text)
    visible = visible.strip()
    if not visible:
        raise ValueError(
            f"parse_wikisource_html: extracted text is empty for source_url={source_url!r}"
        )
    language_detected = language_html
    if not language_detected:
        match = _HEADER_LANGUAGE_RE.search(decoded)
        if match is not None:
            language_detected = match.group(1).split("-", 1)[0][:2].lower()
    license_notice = "\n".join(license_notes).strip()
    raw_sha256 = hashlib.sha256(data).hexdigest()
    return RawDocument(
        text=visible,
        parser_version=PARSER_VERSION,
        layout_quality=1.0,
        ocr_confidence=1.0,
        encoding_detected=encoding_detected,
        language_detected=language_detected,
        license_notice=license_notice,
        raw_sha256=raw_sha256,
    )


__all__ = [
    "PARSER_VERSION",
    "WIKISOURCE_HTML_CONTENT_TYPE",
    "WIKISOURCE_WIKITEXT_CONTENT_TYPE",
    "parse_wikisource_html",
]
