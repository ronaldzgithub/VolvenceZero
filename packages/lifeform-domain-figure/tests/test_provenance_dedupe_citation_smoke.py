"""Smoke tests for the D2 provenance + dedupe + citation modules."""

from __future__ import annotations

import hashlib

import pytest

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    FigureLetterSource,
    FigurePaperSource,
    build_figure_ingestion_envelope,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.corpus import (
    CaptureMethod,
    DedupReport,
    LegalClearance,
    LocatorKind,
    ParsedLocator,
    SourceProvenance,
    compute_dedup_report,
    fingerprint_provenance,
    parse_locator,
)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _public_domain_provenance(source_id: str) -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        figure_id="einstein",
        source_url=f"https://example.invalid/{source_id}",
        license_label="public-domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.SCAN_REVIEWED_OCR,
        captured_by="curator-1",
        captured_at_iso="2026-05-10T00:00:00Z",
        byte_sha256="a" * 64,
        provenance_note="Captured from a public-domain facsimile.",
        jurisdiction_hint="US/EU",
    )


def test_source_provenance_validates_required_fields() -> None:
    record = _public_domain_provenance("synth-foundations-1")
    assert record.source_id == "synth-foundations-1"
    assert record.legal_clearance is LegalClearance.PUBLIC_DOMAIN_GLOBAL


def test_source_provenance_rejects_uncleared() -> None:
    with pytest.raises(ValueError, match="UNCLEARED"):
        SourceProvenance(
            source_id="x",
            figure_id="einstein",
            source_url="https://example.invalid/x",
            license_label="unknown",
            legal_clearance=LegalClearance.UNCLEARED,
            capture_method=CaptureMethod.UNKNOWN,
            captured_by="curator",
            captured_at_iso="2026-05-10T00:00:00Z",
            byte_sha256="0" * 64,
            provenance_note="cleared later",
        )


def test_source_provenance_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="byte_sha256"):
        SourceProvenance(
            source_id="x",
            figure_id="einstein",
            source_url="https://example.invalid/x",
            license_label="public-domain",
            legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
            capture_method=CaptureMethod.OCR,
            captured_by="curator",
            captured_at_iso="2026-05-10T00:00:00Z",
            byte_sha256="not-a-hash",
            provenance_note="cleared",
        )


def test_fingerprint_provenance_is_stable() -> None:
    a = _public_domain_provenance("synth-foundations-1")
    b = _public_domain_provenance("synth-letter-1935-04")
    fp_one = fingerprint_provenance((a, b))
    fp_two = fingerprint_provenance((a, b))
    assert fp_one == fp_two
    assert len(fp_one) == 64
    fp_swap = fingerprint_provenance((b, a))
    assert fp_swap != fp_one, "ordering of provenance records is significant"


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def _einstein_envelopes():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    return envelope_set.envelopes


def test_dedup_report_collapses_shared_header_paragraph() -> None:
    """The synthetic corpus prepends a shared ``_HEADER_NOTE`` paragraph to
    every body, so dedupe should identify exactly one duplicate group whose
    canonical chunk is on the highest-trust source kind (papers)."""
    report = compute_dedup_report(_einstein_envelopes())
    assert isinstance(report, DedupReport)
    assert report.total_chunks > report.unique_chunks
    header_groups = [
        group
        for group in report.duplicate_groups
        if any("Synthetic" in loc or loc.startswith("paper:") for loc in (group.canonical_locator,))
    ]
    assert header_groups, "expected at least one duplicate group from the shared header"
    for group in report.duplicate_groups:
        assert group.canonical_locator.startswith("paper:") or group.canonical_locator.startswith(
            "letter:"
        ), "canonical locator must be on a known source kind"


def test_dedup_report_collapses_byte_identical_chunk() -> None:
    duplicate_body = (
        "Reviewer-fabricated body for dedup test.\n\n"
        "Second paragraph for dedup test."
    )
    paper = FigurePaperSource(
        paper_id="p-1",
        title="Paper 1",
        year=1925,
        language="en",
        body=duplicate_body,
    )
    letter = FigureLetterSource(
        letter_id="l-1",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
        language="en",
        body=duplicate_body,
    )
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=(paper,),
        letters=(letter,),
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    report = compute_dedup_report(envelope_set.envelopes)
    assert report.total_chunks == 4
    assert report.unique_chunks == 2
    assert report.duplicate_chunk_count == 2
    assert len(report.duplicate_groups) == 2
    for group in report.duplicate_groups:
        assert group.canonical_locator.startswith("paper:"), (
            "papers must outrank letters when the text is byte-identical"
        )


def test_dedup_report_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_dedup_report(())


# ---------------------------------------------------------------------------
# citation parser
# ---------------------------------------------------------------------------


def test_parse_locator_paper() -> None:
    parsed = parse_locator(
        "paper:einstein-1905-001:lang=de:para=2:offset=100-250"
    )
    assert isinstance(parsed, ParsedLocator)
    assert parsed.kind is LocatorKind.PAPER
    assert parsed.document_id == "einstein-1905-001"
    assert parsed.language == "de"
    assert parsed.paragraph_index == 2
    assert parsed.offset.start == 100
    assert parsed.offset.end == 250
    assert parsed.sender_id == "" and parsed.recipient_id == ""


def test_parse_locator_letter() -> None:
    parsed = parse_locator(
        "letter:einstein-to-bohr:date=1935-04-12:lang=de:para=0:offset=0-50"
    )
    assert parsed.kind is LocatorKind.LETTER
    assert parsed.sender_id == "einstein"
    assert parsed.recipient_id == "bohr"
    assert parsed.date_iso == "1935-04-12"
    assert parsed.language == "de"


def test_parse_locator_lecture() -> None:
    parsed = parse_locator(
        "lecture:einstein-spencer-1933:venue=oxford-1933:date=1933-06-10:"
        "lang=en:para=4:offset=0-200"
    )
    assert parsed.kind is LocatorKind.LECTURE
    assert parsed.venue_id == "oxford-1933"
    assert parsed.date_iso == "1933-06-10"
    assert parsed.paragraph_index == 4


def test_parse_locator_notebook() -> None:
    parsed = parse_locator(
        "notebook:einstein-zurich-1912:vol=II:page=14:lang=de:para=1:offset=10-99"
    )
    assert parsed.kind is LocatorKind.NOTEBOOK
    assert parsed.volume == "II"
    assert parsed.page == 14


def test_parse_locator_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown locator kind"):
        parse_locator("podcast:some-show:para=0:offset=0-1")


def test_parse_locator_rejects_missing_para() -> None:
    with pytest.raises(ValueError, match="missing required key 'para'"):
        parse_locator("paper:einstein-1905-001:lang=de:offset=0-50")


def test_parse_locator_rejects_letter_without_thread() -> None:
    with pytest.raises(ValueError, match="-to-"):
        parse_locator(
            "letter:einstein-bohr:date=1935-04-12:lang=de:para=0:offset=0-50"
        )


def test_parse_locator_rejects_lecture_without_venue() -> None:
    with pytest.raises(ValueError, match="missing 'venue' key"):
        parse_locator(
            "lecture:einstein-spencer-1933:date=1933-06-10:lang=en:para=0:offset=0-50"
        )


def test_parse_locator_rejects_notebook_without_vol_page() -> None:
    with pytest.raises(ValueError, match="'vol'"):
        parse_locator(
            "notebook:einstein-zurich-1912:lang=de:para=0:offset=0-50"
        )


def test_parse_locator_preserves_extras() -> None:
    parsed = parse_locator(
        "paper:einstein-1905-001:lang=de:para=0:offset=0-50:tag=foundations"
    )
    assert ("tag", "foundations") in parsed.extras


def test_parsed_offset_validates_ordering() -> None:
    with pytest.raises(ValueError, match=">= start"):
        parse_locator(
            "paper:einstein-1905-001:lang=de:para=0:offset=200-100"
        )


def test_locator_round_trip_against_real_envelope() -> None:
    envelopes = _einstein_envelopes()
    for envelope in envelopes:
        for chunk in envelope.successful_chunks:
            parsed = parse_locator(chunk.locator)
            assert parsed.raw == chunk.locator
            assert parsed.paragraph_index >= 0
            assert parsed.offset.end >= parsed.offset.start


def test_real_chunk_byte_hash_matches_provenance_capture() -> None:
    """Sanity check: a curator computing byte_sha256 over a chunk's
    UTF-8 text should get the same hash as a runtime that hashes the
    chunk text the kernel saw, so cross-source dedupe and provenance
    checks share one truth."""
    envelopes = _einstein_envelopes()
    sample_chunk = envelopes[0].successful_chunks[0]
    expected = hashlib.sha256(sample_chunk.text.encode("utf-8")).hexdigest()
    record = SourceProvenance(
        source_id="check",
        figure_id="einstein",
        source_url="https://example.invalid/check",
        license_label="public-domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.OCR,
        captured_by="curator",
        captured_at_iso="2026-05-10T00:00:00Z",
        byte_sha256=expected,
        provenance_note="hash sanity",
    )
    assert record.byte_sha256 == expected
