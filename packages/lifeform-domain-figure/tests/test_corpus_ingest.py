"""Smoke tests for the F1.2 corpus ingestion adapters.

Validates citation-quality locator format + happy-path shapes for all
four ``corpus.ingest_*`` adapters and their typed source records.
"""

from __future__ import annotations

import pytest

from lifeform_ingestion.envelope import (
    IngestionComplianceProfile,
    IngestionSourceKind,
)

from lifeform_domain_figure import (
    FigureLectureSource,
    FigureLetterSource,
    FigureNotebookSource,
    FigurePaperSource,
    ingest_lectures,
    ingest_letters,
    ingest_notebooks,
    ingest_papers,
)


def _two_paragraph_body() -> str:
    return "First paragraph of the source.\n\nSecond paragraph of the source."


def test_ingest_papers_emits_paper_locator() -> None:
    source = FigurePaperSource(
        paper_id="einstein-1905-001",
        title="Zur Elektrodynamik bewegter Körper",
        year=1905,
        language="de",
        body=_two_paragraph_body(),
        figure_id="einstein",
    )
    envelope = ingest_papers((source,), uploader="test")
    assert envelope.source_kind == IngestionSourceKind.BOOK
    assert envelope.compliance_profile == IngestionComplianceProfile.FORCED
    assert len(envelope.chunks) == 2
    locators = [chunk.locator for chunk in envelope.chunks]
    for locator in locators:
        assert locator.startswith("paper:einstein-1905-001:lang=de:")
        assert "para=" in locator
        assert "offset=" in locator


def test_ingest_letters_emits_correspondence_locator() -> None:
    source = FigureLetterSource(
        letter_id="ein-to-bohr-1935-04-12",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
        language="de",
        body=_two_paragraph_body(),
        figure_id="einstein",
    )
    envelope = ingest_letters((source,), uploader="test")
    assert all(
        chunk.locator.startswith(
            "letter:einstein-to-bohr:date=1935-04-12:lang=de:"
        )
        for chunk in envelope.chunks
    )


def test_ingest_letters_handles_undated() -> None:
    source = FigureLetterSource(
        letter_id="ein-to-bohr-undated",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="",
        language="de",
        body=_two_paragraph_body(),
        figure_id="einstein",
    )
    envelope = ingest_letters((source,), uploader="test")
    assert all("date=undated" in chunk.locator for chunk in envelope.chunks)


def test_ingest_lectures_emits_venue_locator() -> None:
    source = FigureLectureSource(
        lecture_id="einstein-spencer-1933",
        venue_id="oxford-1933",
        date_iso="1933-06-10",
        audience="academic",
        language="en",
        body=_two_paragraph_body(),
        figure_id="einstein",
    )
    envelope = ingest_lectures((source,), uploader="test")
    assert all(
        chunk.locator.startswith(
            "lecture:einstein-spencer-1933:venue=oxford-1933:"
        )
        for chunk in envelope.chunks
    )


def test_ingest_notebooks_softer_default_confidence() -> None:
    source = FigureNotebookSource(
        notebook_id="einstein-zurich-1912",
        volume="vol-A",
        page=42,
        language="de",
        body=_two_paragraph_body(),
        figure_id="einstein",
    )
    envelope = ingest_notebooks((source,), uploader="test")
    assert all(chunk.confidence == 0.85 for chunk in envelope.chunks)
    assert all(
        chunk.locator.startswith(
            "notebook:einstein-zurich-1912:vol=vol-A:page=42:"
        )
        for chunk in envelope.chunks
    )


def test_ingest_papers_rejects_empty_sources() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ingest_papers((), uploader="test")


def test_paper_source_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="body"):
        FigurePaperSource(
            paper_id="einstein-1905-001",
            title="t",
            year=1905,
            language="de",
            body="",
        )


def test_letter_source_rejects_missing_participants() -> None:
    with pytest.raises(ValueError, match="sender_id"):
        FigureLetterSource(
            letter_id="x",
            sender_id="",
            recipient_id="bohr",
            date_iso="1935-04-12",
            language="de",
            body="x",
        )
