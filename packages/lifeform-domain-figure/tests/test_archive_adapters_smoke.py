"""Smoke tests for the D3 archive adapter facades.

Each adapter takes a pre-downloaded typed payload and emits a typed
:class:`FigurePaperSource` / :class:`FigureLetterSource` /
:class:`FigureLectureSource` record. No live HTTP — the V1 default
fetcher is :func:`offline_archive_fetcher` and always raises.
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureLetterSource,
    FigurePaperSource,
)
from lifeform_domain_figure.corpus.archives import (
    ArchiveFetchResult,
    CPAEDocumentKind,
    CPAEPayload,
    GutenbergPayload,
    InternetArchivePayload,
    WikisourcePayload,
    cpae_to_letter_source,
    cpae_to_paper_source,
    gutenberg_to_paper_source,
    internet_archive_to_lecture_source,
    internet_archive_to_paper_source,
    offline_archive_fetcher,
    wikisource_to_lecture_source,
    wikisource_to_paper_source,
)


_BODY = "Reviewer-paraphrased body paragraph one.\n\nReviewer-paraphrased body paragraph two."


# ---------------------------------------------------------------------------
# CPAE
# ---------------------------------------------------------------------------


def test_cpae_article_to_paper_source() -> None:
    payload = CPAEPayload(
        document_id="cpae-vol2-doc24",
        document_kind=CPAEDocumentKind.ARTICLE,
        volume=2,
        document_number=24,
        title="Zur Elektrodynamik bewegter Körper",
        year=1905,
        language="de",
        body=_BODY,
        source_url="https://einsteinpapers.press.princeton.edu/vol2-doc/153",
    )
    source = cpae_to_paper_source(payload, figure_id="einstein")
    assert isinstance(source, FigurePaperSource)
    assert source.paper_id == "cpae-vol2-doc24"
    assert source.figure_id == "einstein"
    assert source.year == 1905
    assert source.publication_locator == "cpae:vol=2:doc=24"


def test_cpae_letter_payload_to_letter_source() -> None:
    payload = CPAEPayload(
        document_id="cpae-vol5-doc207",
        document_kind=CPAEDocumentKind.LETTER,
        volume=5,
        document_number=207,
        title="Letter to Bohr",
        year=1935,
        language="de",
        body=_BODY,
        source_url="https://einsteinpapers.press.princeton.edu/vol5-doc/207",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
    )
    source = cpae_to_letter_source(payload, figure_id="einstein")
    assert isinstance(source, FigureLetterSource)
    assert source.sender_id == "einstein"
    assert source.recipient_id == "bohr"
    assert source.date_iso == "1935-04-12"
    assert source.figure_id == "einstein"


def test_cpae_letter_kind_must_have_sender_recipient() -> None:
    with pytest.raises(ValueError, match="sender_id"):
        CPAEPayload(
            document_id="x",
            document_kind=CPAEDocumentKind.LETTER,
            volume=1,
            document_number=1,
            title="t",
            year=1900,
            language="de",
            body=_BODY,
            source_url="https://example.invalid/x",
        )


def test_cpae_paper_helper_refuses_letter_payload() -> None:
    payload = CPAEPayload(
        document_id="cpae-letter",
        document_kind=CPAEDocumentKind.LETTER,
        volume=1,
        document_number=1,
        title="x",
        year=1900,
        language="de",
        body=_BODY,
        source_url="https://example.invalid/x",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
    )
    with pytest.raises(ValueError, match="LETTER payload"):
        cpae_to_paper_source(payload, figure_id="einstein")


# ---------------------------------------------------------------------------
# Wikisource
# ---------------------------------------------------------------------------


def test_wikisource_paper_source() -> None:
    payload = WikisourcePayload(
        page_title="Why War?",
        language="en",
        source_url="https://en.wikisource.org/wiki/Why_War%3F",
        body=_BODY,
        year=1933,
        author_id="einstein",
    )
    source = wikisource_to_paper_source(payload, figure_id="einstein")
    assert source.paper_id.startswith("paper:wikisource:en:")
    assert source.year == 1933
    assert source.publication_locator == "wikisource:en:Why War?"


def test_wikisource_lecture_requires_venue() -> None:
    payload = WikisourcePayload(
        page_title="Address at Some Conference",
        language="en",
        source_url="https://en.wikisource.org/wiki/x",
        body=_BODY,
    )
    with pytest.raises(ValueError, match="venue_id"):
        wikisource_to_lecture_source(payload, figure_id="einstein")


def test_wikisource_lecture_with_venue() -> None:
    payload = WikisourcePayload(
        page_title="Address at Princeton 1939",
        language="en",
        source_url="https://en.wikisource.org/wiki/x",
        body=_BODY,
        venue_id="princeton-1939",
        date_iso="1939-06-15",
        audience="university",
    )
    source = wikisource_to_lecture_source(payload, figure_id="einstein")
    assert source.venue_id == "princeton-1939"
    assert source.date_iso == "1939-06-15"


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------


def test_gutenberg_to_paper_source_with_section() -> None:
    payload = GutenbergPayload(
        ebook_id=12345,
        title="Relativity: The Special and the General Theory",
        language="en",
        body=_BODY,
        source_url="https://www.gutenberg.org/ebooks/12345",
        section_label="chapter-3",
        year=1916,
    )
    source = gutenberg_to_paper_source(payload, figure_id="einstein")
    assert source.paper_id == "gutenberg:12345:chapter-3"
    assert "section=chapter-3" in source.publication_locator
    assert "chapter-3" in source.title


def test_gutenberg_rejects_bad_ebook_id() -> None:
    with pytest.raises(ValueError, match="ebook_id"):
        GutenbergPayload(
            ebook_id=0,
            title="x",
            language="en",
            body=_BODY,
            source_url="https://www.gutenberg.org/x",
        )


# ---------------------------------------------------------------------------
# Internet Archive
# ---------------------------------------------------------------------------


def test_internet_archive_paper_source() -> None:
    payload = InternetArchivePayload(
        identifier="einstein-relativity-1920",
        title="Relativity: A Popular Exposition",
        language="en",
        body=_BODY,
        source_url="https://archive.org/details/einstein-relativity-1920",
        creator_id="einstein",
        year=1920,
    )
    source = internet_archive_to_paper_source(payload, figure_id="einstein")
    assert source.paper_id == "ia:einstein-relativity-1920"
    assert source.year == 1920
    assert source.publication_locator == (
        "internet-archive:identifier=einstein-relativity-1920"
    )


def test_internet_archive_lecture_requires_venue() -> None:
    payload = InternetArchivePayload(
        identifier="einstein-1933-lecture",
        title="A 1933 lecture",
        language="en",
        body=_BODY,
        source_url="https://archive.org/details/einstein-1933-lecture",
    )
    with pytest.raises(ValueError, match="venue_id"):
        internet_archive_to_lecture_source(payload, figure_id="einstein")


def test_internet_archive_lecture_with_venue() -> None:
    payload = InternetArchivePayload(
        identifier="einstein-1933-spencer",
        title="Herbert Spencer Lecture, Oxford",
        language="en",
        body=_BODY,
        source_url="https://archive.org/details/einstein-spencer-1933",
        venue_id="oxford-1933",
        date_iso="1933-06-10",
        audience="academic",
    )
    source = internet_archive_to_lecture_source(payload, figure_id="einstein")
    assert source.lecture_id == "ia:einstein-1933-spencer"
    assert source.venue_id == "oxford-1933"


# ---------------------------------------------------------------------------
# Offline fetcher stub
# ---------------------------------------------------------------------------


def test_offline_archive_fetcher_raises() -> None:
    fetcher = offline_archive_fetcher()
    with pytest.raises(NotImplementedError, match="V1 of the figure vertical"):
        fetcher.fetch("https://example.invalid/anything")


def test_archive_fetch_result_is_dataclass() -> None:
    result = ArchiveFetchResult(
        source_url="https://example.invalid/x",
        raw_payload=GutenbergPayload(
            ebook_id=1,
            title="t",
            language="en",
            body=_BODY,
            source_url="https://example.invalid/x",
        ),
    )
    assert result.source_url == "https://example.invalid/x"
    assert isinstance(result.raw_payload, GutenbergPayload)
