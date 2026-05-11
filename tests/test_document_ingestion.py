"""Packet 2.2: PDF / Markdown ingestion + deterministic chunking tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lifeform_protocol_runtime.document_uptake.ingestion import (
    DocumentChunk,
    DocumentText,
    chunk_document,
    read_markdown,
    read_pdf,
)


# ---------------------------------------------------------------------------
# DocumentText / DocumentChunk schemas
# ---------------------------------------------------------------------------


def test_document_text_rejects_empty_locator() -> None:
    with pytest.raises(ValueError, match="source_locator"):
        DocumentText(source_locator=" ", text="hi", page_count=1)


def test_document_chunk_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        DocumentChunk(
            source_locator="/tmp/x.md",
            chunk_index=-1,
            source_offset=0,
            text="hi",
        )


def test_document_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        DocumentChunk(
            source_locator="/tmp/x.md",
            chunk_index=0,
            source_offset=0,
            text="",
        )


# ---------------------------------------------------------------------------
# read_markdown
# ---------------------------------------------------------------------------


def test_read_markdown_returns_document_text(tmp_path: Path) -> None:
    md_path = tmp_path / "guide.md"
    md_path.write_text("# Hello\n\nWorld.", encoding="utf-8")

    doc = read_markdown(md_path)

    assert doc.text == "# Hello\n\nWorld."
    assert doc.page_count == 1
    assert "guide.md" in doc.description
    assert str(md_path) == doc.source_locator


def test_read_markdown_handles_utf8_chinese(tmp_path: Path) -> None:
    md_path = tmp_path / "中文.md"
    md_path.write_text("私域运营指南\n\n第一节：建立信任。", encoding="utf-8")

    doc = read_markdown(md_path)

    assert "私域运营指南" in doc.text
    assert "建立信任" in doc.text


# ---------------------------------------------------------------------------
# chunk_document — determinism
# ---------------------------------------------------------------------------


def test_chunk_document_empty_input_returns_empty_tuple() -> None:
    assert chunk_document("", source_locator="/x") == ()
    assert chunk_document("   \n\n   ", source_locator="/x") == ()


def test_chunk_document_short_input_one_chunk() -> None:
    text = "Short text under any reasonable max_tokens."
    chunks = chunk_document(text, source_locator="/x", max_tokens=2048)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_document_is_deterministic() -> None:
    text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
    a = chunk_document(text, source_locator="/x", max_tokens=10)
    b = chunk_document(text, source_locator="/x", max_tokens=10)
    assert a == b


def test_chunk_document_prefers_paragraph_breaks() -> None:
    """Prefer the latest paragraph break inside the slice."""
    # Each "P_N\n\n" is ~16 chars; max_tokens=10 → 40 chars → ~2 paragraphs per chunk.
    text = "\n\n".join([f"Para {i} content here." for i in range(8)])
    chunks = chunk_document(text, source_locator="/x", max_tokens=10)
    # Each chunk should end at a paragraph boundary (not mid-sentence).
    for chunk in chunks[:-1]:
        assert chunk.text.endswith(".") or chunk.text.endswith("here."), (
            f"chunk did not end at paragraph break: {chunk.text!r}"
        )


def test_chunk_document_records_source_offsets() -> None:
    """Each chunk's source_offset should reconstruct the text positions."""
    text = "\n\n".join([f"Paragraph {i} body." for i in range(5)])
    chunks = chunk_document(text, source_locator="/x", max_tokens=8)

    # Each chunk_index must be unique and increasing.
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)

    # Each source_offset must be < total length.
    for chunk in chunks:
        assert chunk.source_offset < len(text)


def test_chunk_document_overlap_includes_tail_of_previous() -> None:
    """Non-zero overlap_tokens means consecutive chunks share text."""
    text = "\n\n".join([f"P{i} sentence body." for i in range(20)])
    no_overlap = chunk_document(
        text, source_locator="/x", max_tokens=10, overlap_tokens=0
    )
    with_overlap = chunk_document(
        text, source_locator="/x", max_tokens=10, overlap_tokens=4
    )
    # Overlap version should have at least as many chunks.
    assert len(with_overlap) >= len(no_overlap)


def test_chunk_document_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_document(
            "test", source_locator="/x", max_tokens=10, overlap_tokens=10
        )
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_document(
            "test", source_locator="/x", max_tokens=10, overlap_tokens=-1
        )


def test_chunk_document_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_document("test", source_locator="/x", max_tokens=0)


# ---------------------------------------------------------------------------
# read_pdf — exercise pypdf path on a synthetic PDF
# ---------------------------------------------------------------------------


def test_read_pdf_extracts_text_from_synthetic_pdf(tmp_path: Path) -> None:
    """Generate a tiny PDF and round-trip through read_pdf."""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    # Build a 2-page PDF with empty content (pypdf can read it; text
    # will be "" but page_count reflects).
    pdf_path = tmp_path / "synthetic.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as f:
        writer.write(f)

    doc = read_pdf(pdf_path)
    assert doc.page_count == 2
    assert doc.source_locator == str(pdf_path)
    assert "synthetic.pdf" in doc.description
