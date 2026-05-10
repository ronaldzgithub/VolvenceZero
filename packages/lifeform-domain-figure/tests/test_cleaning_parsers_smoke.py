"""Smoke tests for the four L1 parsers (debt #28).

Each parser is exercised against a small in-memory fixture
(``cleaning_fixtures``) and asserts the parser's quality contracts:

* extracted text contains the expected substantive marker phrase
* parser_version follows the ``"<id>:<int>"`` convention
* license_notice is non-empty for fixtures that include a license
  block
* raw_sha256 equals the sha256 of the input bytes
"""

from __future__ import annotations

import hashlib

import pytest

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
from cleaning_fixtures import (
    EINSTEIN_QUOTE,
    build_archive_org_ocr_json_bytes,
    build_gutenberg_html_bytes,
    build_gutenberg_text_bytes,
    build_minimal_cpae_pdf_bytes,
    build_wikisource_html_bytes,
)


def test_parse_cpae_pdf_extracts_text_and_records_sha() -> None:
    data = build_minimal_cpae_pdf_bytes()
    raw = parse_cpae_pdf(
        data, source_url="test://cpae/vol1/doc1", content_type=CPAE_PDF_CONTENT_TYPE
    )
    assert EINSTEIN_QUOTE in raw.text
    assert raw.parser_version.startswith("cpae-pdf:")
    assert raw.raw_sha256 == hashlib.sha256(data).hexdigest()
    assert raw.ocr_confidence == 1.0
    assert 0.0 <= raw.layout_quality <= 1.0


def test_parse_cpae_pdf_rejects_wrong_content_type() -> None:
    data = build_minimal_cpae_pdf_bytes()
    with pytest.raises(ValueError):
        parse_cpae_pdf(data, source_url="test://x", content_type="text/plain")


def test_parse_wikisource_html_strips_chrome_and_keeps_body() -> None:
    data = build_wikisource_html_bytes()
    raw = parse_wikisource_html(
        data,
        source_url="test://en.wikisource.org/wiki/Sample",
        content_type=WIKISOURCE_HTML_CONTENT_TYPE,
    )
    assert "photoelectric effect" in raw.text
    assert raw.language_detected == "en"
    assert "PD-old" in raw.license_notice
    assert raw.raw_sha256 == hashlib.sha256(data).hexdigest()


def test_parse_gutenberg_text_carves_around_markers() -> None:
    data = build_gutenberg_text_bytes()
    raw = parse_gutenberg(
        data,
        source_url="test://gutenberg/1",
        content_type=GUTENBERG_TEXT_CONTENT_TYPE,
    )
    assert "Dear colleague" in raw.text
    assert "START OF" not in raw.text
    assert "END OF" not in raw.text
    assert "Project Gutenberg" in raw.license_notice
    assert raw.raw_sha256 == hashlib.sha256(data).hexdigest()


def test_parse_gutenberg_html_strips_tags_and_carves_markers() -> None:
    data = build_gutenberg_html_bytes()
    raw = parse_gutenberg(
        data,
        source_url="test://gutenberg/2",
        content_type=GUTENBERG_HTML_CONTENT_TYPE,
    )
    assert "special theory of relativity" in raw.text
    assert "<p>" not in raw.text
    assert "Trailer text" not in raw.text


def test_parse_archive_org_ocr_json_concatenates_pages_and_averages_confidence() -> None:
    data = build_archive_org_ocr_json_bytes()
    raw = parse_archive_org_ocr_json(
        data,
        source_url="test://archive.org/details/sample",
        content_type=ARCHIVE_ORG_OCR_CONTENT_TYPE,
    )
    assert "Page one" in raw.text
    assert "Page two" in raw.text
    assert "\f" in raw.text
    assert raw.ocr_confidence == pytest.approx((0.91 + 0.88) / 2, rel=1e-6)
    assert raw.language_detected == "eng"
    assert "creativecommons.org" in raw.license_notice


def test_parse_by_content_type_dispatches_each_format() -> None:
    cases = [
        (CPAE_PDF_CONTENT_TYPE, build_minimal_cpae_pdf_bytes(), EINSTEIN_QUOTE),
        (
            WIKISOURCE_HTML_CONTENT_TYPE,
            build_wikisource_html_bytes(),
            "photoelectric effect",
        ),
        (
            GUTENBERG_TEXT_CONTENT_TYPE,
            build_gutenberg_text_bytes(),
            "Dear colleague",
        ),
        (
            GUTENBERG_HTML_CONTENT_TYPE,
            build_gutenberg_html_bytes(),
            "special theory of relativity",
        ),
        (
            ARCHIVE_ORG_OCR_CONTENT_TYPE,
            build_archive_org_ocr_json_bytes(),
            "Page two",
        ),
    ]
    for content_type, data, expected_substring in cases:
        raw = parse_by_content_type(
            data, source_url=f"test://{content_type}", content_type=content_type
        )
        assert expected_substring in raw.text


def test_parse_by_content_type_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="no parser registered"):
        parse_by_content_type(
            b"some bytes",
            source_url="test://x",
            content_type="application/x-unknown",
        )
