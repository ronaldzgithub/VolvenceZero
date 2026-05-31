"""Source-adapter tests: plain_text + task_result.

Validates the chunking logic + envelope wiring without going through
a live kernel. Pure data in, pure data out.
"""

from __future__ import annotations

import pytest

from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionSourceKind,
    chunk_plain_text,
    envelope_from_task_result,
    envelope_from_text,
)


# ---------------------------------------------------------------------------
# chunk_plain_text
# ---------------------------------------------------------------------------


def test_chunk_plain_text_empty_input_returns_empty_tuple() -> None:
    assert chunk_plain_text("") == ()
    assert chunk_plain_text("   \n\n   \n") == ()


def test_chunk_plain_text_single_paragraph_becomes_one_chunk() -> None:
    text = "This is a short paragraph."
    chunks = chunk_plain_text(text)
    assert len(chunks) == 1
    segment, start, end = chunks[0]
    assert segment == text
    assert start == 0
    assert end == len(text)


def test_chunk_plain_text_respects_paragraph_boundaries() -> None:
    text = "Paragraph A.\n\nParagraph B.\n\nParagraph C."
    chunks = chunk_plain_text(text)
    assert len(chunks) == 3
    assert chunks[0][0] == "Paragraph A."
    assert chunks[1][0] == "Paragraph B."
    assert chunks[2][0] == "Paragraph C."


def test_chunk_plain_text_hard_cuts_oversized_paragraph() -> None:
    # Single paragraph of 5000 chars, max chunk 1000 -> 5 chunks.
    text = "x" * 5000
    chunks = chunk_plain_text(text, max_chunk_chars=1000)
    assert len(chunks) == 5
    # Each chunk is at most max_chunk_chars long.
    for segment, _, _ in chunks:
        assert len(segment) <= 1000
    # Offsets must cover the whole paragraph without gaps.
    assert chunks[0][1] == 0
    assert chunks[-1][2] == 5000


def test_chunk_plain_text_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError, match="max_chunk_chars"):
        chunk_plain_text("anything", max_chunk_chars=0)
    with pytest.raises(ValueError, match="max_chunk_chars"):
        chunk_plain_text("anything", max_chunk_chars=-1)


# ---------------------------------------------------------------------------
# envelope_from_text
# ---------------------------------------------------------------------------


def test_envelope_from_text_builds_well_formed_envelope() -> None:
    text = "Para one.\n\nPara two. It has more words."
    envelope = envelope_from_text(
        text,
        source_uri="file:///tmp/two-paragraphs.md",
    )
    assert envelope.source_kind is IngestionSourceKind.CORPUS
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED
    assert envelope.total_chunks == 2
    assert envelope.chunks[0].text == "Para one."
    assert envelope.chunks[1].text == "Para two. It has more words."
    # Provenance is populated with hash + uri.
    assert envelope.provenance.source_uri == "file:///tmp/two-paragraphs.md"
    assert len(envelope.provenance.integrity_hash) == 64  # SHA256 hex
    assert envelope.partial_failures == ()


def test_envelope_from_text_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="empty"):
        envelope_from_text("", source_uri="file:///tmp/empty.txt")
    with pytest.raises(ValueError, match="empty"):
        envelope_from_text("   \n\n   ", source_uri="file:///tmp/whitespace.txt")


def test_envelope_from_text_accepts_explicit_envelope_id() -> None:
    envelope = envelope_from_text(
        "content",
        source_uri="inline:fixture",
        envelope_id="ingestion:test-fixture",
    )
    assert envelope.envelope_id == "ingestion:test-fixture"


def test_envelope_from_text_consultative_profile_opts_out_of_forced() -> None:
    envelope = envelope_from_text(
        "content",
        source_uri="inline:test",
        compliance_profile=IngestionComplianceProfile.CONSULTATIVE,
    )
    assert envelope.compliance_profile is IngestionComplianceProfile.CONSULTATIVE


# ---------------------------------------------------------------------------
# envelope_from_task_result
# ---------------------------------------------------------------------------


def test_envelope_from_task_result_maps_fields_to_chunks() -> None:
    envelope = envelope_from_task_result(
        {
            "summary": "Found 3 related files.",
            "detail": "Files are: a.py, b.py, c.py",
            "status": "succeeded",
            "unused_field": "ignored",
        },
        task_id="search-123",
    )
    assert envelope.source_kind is IngestionSourceKind.TASK_RESULT
    # Known fields become chunks; unused_field is skipped.
    chunk_ids = [c.chunk_id for c in envelope.chunks]
    assert any("field:summary" in cid for cid in chunk_ids)
    assert any("field:detail" in cid for cid in chunk_ids)
    assert any("field:status" in cid for cid in chunk_ids)
    assert not any("unused_field" in cid for cid in chunk_ids)


def test_envelope_from_task_result_handles_list_value() -> None:
    envelope = envelope_from_task_result(
        {"findings": ["finding-a", "finding-b", "finding-c"]},
        task_id="t-list",
    )
    assert envelope.total_chunks == 1
    # List becomes bullet-list text.
    text = envelope.chunks[0].text
    assert "- finding-a" in text
    assert "- finding-c" in text


def test_envelope_from_task_result_skips_empty_fields() -> None:
    envelope = envelope_from_task_result(
        {"summary": "real", "detail": "", "status": "    "},
        task_id="t-empty-detail",
    )
    chunk_ids = [c.chunk_id for c in envelope.chunks]
    assert any("field:summary" in cid for cid in chunk_ids)
    assert not any("field:detail" in cid for cid in chunk_ids)
    assert not any("field:status" in cid for cid in chunk_ids)


def test_envelope_from_task_result_rejects_empty_task_id() -> None:
    with pytest.raises(ValueError, match="task_id"):
        envelope_from_task_result({"summary": "x"}, task_id="")


def test_envelope_from_task_result_rejects_all_empty_fields() -> None:
    with pytest.raises(ValueError, match="no non-empty chunks"):
        envelope_from_task_result(
            {"summary": "", "detail": "", "status": ""},
            task_id="t-totally-empty",
        )


def test_envelope_from_task_result_records_integrity_hash() -> None:
    a = envelope_from_task_result({"summary": "x"}, task_id="t-1")
    b = envelope_from_task_result({"summary": "x"}, task_id="t-1")
    # Same payload, same task_id => same integrity_hash + envelope_id.
    assert a.provenance.integrity_hash == b.provenance.integrity_hash
    assert a.envelope_id == b.envelope_id
    # Changed payload => different hash.
    c = envelope_from_task_result({"summary": "y"}, task_id="t-1")
    assert a.provenance.integrity_hash != c.provenance.integrity_hash


# ---------------------------------------------------------------------------
# D-de-1: chunked artifact ingestion for oversized task_result fields
# ---------------------------------------------------------------------------


def test_envelope_from_task_result_small_field_keeps_canonical_chunk_id() -> None:
    # A field that fits keeps the single ``:field:<name>`` id (no part suffix),
    # so existing audits / consumers are unaffected.
    envelope = envelope_from_task_result(
        {"detail": "short detail"},
        task_id="t-small",
    )
    assert envelope.total_chunks == 1
    assert envelope.chunks[0].chunk_id.endswith(":field:detail")


def test_envelope_from_task_result_oversized_field_is_subchunked() -> None:
    big_detail = "x" * 5000
    envelope = envelope_from_task_result(
        {"summary": "ok", "detail": big_detail},
        task_id="t-big",
        max_chunk_chars=1000,
    )
    detail_chunks = [c for c in envelope.chunks if ":field:detail" in c.chunk_id]
    # 5000 chars / 1000 -> 5 bounded sub-chunks, none truncated.
    assert len(detail_chunks) == 5
    for chunk in detail_chunks:
        assert len(chunk.text) <= 1000
        assert ":part:" in chunk.chunk_id
        assert "offset=" in chunk.locator
    # The full content survives: concatenating the parts reproduces it.
    assert "".join(c.text for c in detail_chunks) == big_detail
    # The small field stays a single canonical chunk.
    summary_chunks = [c for c in envelope.chunks if ":field:summary" in c.chunk_id]
    assert len(summary_chunks) == 1
    assert summary_chunks[0].chunk_id.endswith(":field:summary")


def test_envelope_from_task_result_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError, match="max_chunk_chars"):
        envelope_from_task_result({"summary": "x"}, task_id="t", max_chunk_chars=0)
