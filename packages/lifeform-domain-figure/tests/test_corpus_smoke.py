"""Smoke tests for the F1.2 corpus ingestion adapters.

Validates:

* Each source-specific adapter produces an :class:`IngestionEnvelope`
  with chunks whose locators carry the expected citation prefix.
* The top-level :func:`build_figure_ingestion_envelope` produces one
  envelope per non-empty source kind on the synthetic corpus.
* All adapters fail loudly on empty / whitespace-only bodies.
* Notebook chunks pick up the softer default confidence.
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    FigureLectureSource,
    FigureLetterSource,
    FigureNotebookSource,
    FigurePaperSource,
    build_figure_ingestion_envelope,
    ingest_lectures,
    ingest_letters,
    ingest_notebooks,
    ingest_papers,
    synthetic_einstein_corpus,
)


def test_paper_envelope_locators_carry_paper_id() -> None:
    paper = FigurePaperSource(
        paper_id="paper-1",
        title="On something",
        year=1925,
        language="en",
        body=("Paragraph one.\n\nParagraph two."),
    )
    envelope = ingest_papers((paper,), uploader="test")
    assert envelope.total_chunks == 2
    locators = [chunk.locator for chunk in envelope.chunks]
    assert all(loc.startswith("paper:paper-1:lang=en:para=") for loc in locators)
    assert envelope.partial_failures == ()


def test_letter_envelope_locators_thread_sender_recipient_date() -> None:
    letter = FigureLetterSource(
        letter_id="l-1",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
        language="en",
        body="Dear colleague.\n\nWith warm regards.",
    )
    envelope = ingest_letters((letter,), uploader="test")
    locators = [chunk.locator for chunk in envelope.chunks]
    assert all("letter:einstein-to-bohr" in loc for loc in locators)
    assert all("date=1935-04-12" in loc for loc in locators)


def test_lecture_envelope_locators_carry_venue_date() -> None:
    lecture = FigureLectureSource(
        lecture_id="lec-1",
        venue_id="princeton-1939",
        date_iso="1939-06-15",
        audience="university",
        language="en",
        body="Address paragraph one.\n\nAddress paragraph two.",
    )
    envelope = ingest_lectures((lecture,), uploader="test")
    locators = [chunk.locator for chunk in envelope.chunks]
    assert all("venue=princeton-1939" in loc for loc in locators)
    assert all("date=1939-06-15" in loc for loc in locators)


def test_notebook_chunks_inherit_default_softer_confidence() -> None:
    nb = FigureNotebookSource(
        notebook_id="nb-1",
        volume="II",
        page=14,
        language="en",
        body="Note paragraph one.\n\nNote paragraph two.",
    )
    envelope = ingest_notebooks((nb,), uploader="test")
    confidences = {chunk.confidence for chunk in envelope.chunks}
    assert confidences == {0.85}


def test_paper_source_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="body"):
        FigurePaperSource(
            paper_id="paper-x",
            title="x",
            year=1900,
            language="en",
            body="   ",
        )


def test_letter_source_rejects_empty_sender() -> None:
    with pytest.raises(ValueError, match="sender_id"):
        FigureLetterSource(
            letter_id="l-x",
            sender_id="  ",
            recipient_id="bohr",
            date_iso="1935-04-12",
            language="en",
            body="non-empty body",
        )


def test_build_figure_ingestion_envelope_produces_one_per_kind() -> None:
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    assert envelope_set.figure_id == "einstein"
    assert envelope_set.papers is not None
    assert envelope_set.letters is not None
    assert envelope_set.lectures is not None
    assert envelope_set.notebooks is not None
    assert len(envelope_set.envelopes) == 4
    for envelope in envelope_set.envelopes:
        assert envelope.envelope_id.startswith("figure:einstein:")
        assert envelope.total_chunks > 0
        assert envelope.partial_failures == ()


def test_build_figure_ingestion_envelope_skips_empty_kinds() -> None:
    papers, *_ = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    assert envelope_set.papers is not None
    assert envelope_set.letters is None
    assert envelope_set.lectures is None
    assert envelope_set.notebooks is None
    assert len(envelope_set.envelopes) == 1


def test_corpus_source_bundle_rejects_all_empty() -> None:
    with pytest.raises(ValueError, match="no source records"):
        FigureCorpusSourceBundle(figure_id="einstein")
