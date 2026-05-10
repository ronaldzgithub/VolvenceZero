"""Smoke tests for the cleaner pipeline (debt #28)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.cleaning.cleaners import (
    CLEANER_PIPELINE_V1,
    CURRENT_CLEANER_PIPELINE_VERSION,
    clean_raw_document,
    cleaner_for,
)
from lifeform_domain_figure.cleaning.cleaners.boilerplate import strip_boilerplate
from lifeform_domain_figure.cleaning.cleaners.dedupe import dedupe_paragraphs
from lifeform_domain_figure.cleaning.cleaners.paragraph import normalise_paragraphs
from lifeform_domain_figure.cleaning.cleaners.pii import REDACTED_TOKEN, redact_pii
from lifeform_domain_figure.cleaning.cleaners.typography import normalise_typography
from lifeform_domain_figure.cleaning.cleaners.whitespace import normalise_whitespace
from lifeform_domain_figure.cleaning.raw_document import (
    CleaningOp,
    RawDocument,
)


_DUMMY_SHA = "0" * 64


def _make_raw(text: str) -> RawDocument:
    return RawDocument(
        text=text,
        parser_version="test:1",
        layout_quality=1.0,
        ocr_confidence=1.0,
        encoding_detected="utf-8",
        language_detected="en",
        license_notice="",
        raw_sha256=_DUMMY_SHA,
    )


def test_strip_boilerplate_removes_page_numbers_and_running_heads() -> None:
    text = "Body line one.\n   42  \nBody line two.\nVol. 2, p. 7\nMore body.\n[ p. 12 ]\nLast line."
    cleaned = strip_boilerplate(text)
    assert "42" not in cleaned
    assert "Vol. 2" not in cleaned
    assert "[ p. 12 ]" not in cleaned
    assert "Body line one." in cleaned
    assert "Last line." in cleaned


def test_normalise_whitespace_collapses_runs_and_blank_lines() -> None:
    text = "alpha    beta\n\n\n\ngamma   \n   "
    cleaned = normalise_whitespace(text)
    assert cleaned == "alpha beta\n\ngamma"


def test_normalise_typography_replaces_curly_quotes_and_dashes() -> None:
    text = "He said \u201chello\u201d. The en\u2013dash and em\u2014dash."
    cleaned = normalise_typography(text)
    assert '"hello"' in cleaned
    assert "\u2013" not in cleaned
    assert "\u2014" not in cleaned


def test_normalise_typography_rejoins_hyphenated_linebreaks() -> None:
    text = "experi-\nmental setup"
    cleaned = normalise_typography(text)
    assert "experimental setup" in cleaned


def test_dedupe_paragraphs_drops_repeated_long_paragraphs() -> None:
    paragraph = (
        "This is a long paragraph that the dedupe cleaner should treat as a "
        "deduplication candidate because its length exceeds the minimum threshold."
    )
    text = f"{paragraph}\n\nshort line\n\n{paragraph}\n\nshort line"
    cleaned = dedupe_paragraphs(text)
    assert cleaned.count(paragraph) == 1
    assert cleaned.count("short line") == 2


def test_redact_pii_replaces_emails_and_phones() -> None:
    text = (
        "Contact alice.smith@example.com or call +1-555-555-5555 today. "
        "My card is 4111 1111 1111 1111."
    )
    cleaned = redact_pii(text)
    assert "@example.com" not in cleaned
    assert "555-555-5555" not in cleaned
    assert "4111" not in cleaned
    assert REDACTED_TOKEN in cleaned


def test_normalise_paragraphs_reflows_continuation_lines() -> None:
    text = "First sentence ends here.\nA continuation\nthat keeps going\nuntil here."
    cleaned = normalise_paragraphs(text)
    assert "ends here.\nA continuation that keeps going until here." in cleaned


def test_pipeline_orchestrator_records_log_and_is_monotonic() -> None:
    text = (
        "Line one.   With   extra   spaces.\n\n\n\n"
        "   42   \n"
        "He \u201cquoted\u201d the email me@example.com.\n"
        "experi-\nmental setup follows."
    )
    raw = _make_raw(text)
    cleaned = clean_raw_document(raw)
    assert cleaned.cleaner_pipeline_version == CURRENT_CLEANER_PIPELINE_VERSION
    assert cleaned.parser_version == "test:1"
    assert cleaned.raw_sha256 == raw.raw_sha256
    assert len(cleaned.cleaning_log) == len(CLEANER_PIPELINE_V1)
    seen_ops = [record.op for record in cleaned.cleaning_log]
    assert seen_ops == [step[0] for step in CLEANER_PIPELINE_V1]
    for record in cleaned.cleaning_log:
        assert record.chars_after <= record.chars_before
    assert "@example.com" not in cleaned.text
    assert "experimental setup" in cleaned.text


def test_pipeline_reduces_noisy_text_by_more_than_five_percent() -> None:
    body_paragraph = (
        "This is a long substantive paragraph that the cleaner pipeline should "
        "preserve. It contains real content discussing the photoelectric effect "
        "and the special theory of relativity."
    )
    raw = _make_raw(
        "\n\n\n\n   42   \n\n\n"
        + body_paragraph
        + "\n\n   \n\n"
        + body_paragraph
        + "\n\n   12  \n\nVol. 1, p. 5\n\nfinal short tail.\n\n\n\n"
    )
    cleaned = clean_raw_document(raw)
    reduction = 1.0 - len(cleaned.text) / max(len(raw.text), 1)
    assert reduction > 0.05, (
        f"cleaner pipeline reduced text by only {reduction:.2%}; expected >5%"
    )


def test_cleaner_for_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="unknown pipeline_version"):
        cleaner_for(99)


def test_cleaning_op_record_invariant_rejects_expansion() -> None:
    from lifeform_domain_figure.cleaning.raw_document import CleaningOpRecord

    with pytest.raises(ValueError, match="must be <= chars_before"):
        CleaningOpRecord(
            op=CleaningOp.WHITESPACE_NORMALIZE,
            op_version="1",
            chars_before=10,
            chars_after=11,
        )
