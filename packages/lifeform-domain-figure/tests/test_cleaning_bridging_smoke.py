"""Smoke tests for the cleaning -> archive payload -> typed source bridge."""

from __future__ import annotations

from lifeform_domain_figure.corpus.archives.cpae import (
    CPAEDocumentKind,
    cpae_to_letter_source,
    cpae_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.gutenberg import gutenberg_to_paper_source
from lifeform_domain_figure.corpus.archives.internet_archive import (
    internet_archive_to_lecture_source,
    internet_archive_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.wikisource import (
    wikisource_to_lecture_source,
    wikisource_to_paper_source,
)
from lifeform_domain_figure.cleaning.bridging import (
    cleaned_to_cpae_payload,
    cleaned_to_gutenberg_payload,
    cleaned_to_internet_archive_payload,
    cleaned_to_wikisource_payload,
)
from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
)


_DUMMY_SHA = "0" * 64


def _make_cleaned(text: str) -> CleanedDocument:
    return CleanedDocument(
        text=text,
        raw_sha256=_DUMMY_SHA,
        cleaner_pipeline_version=1,
        cleaning_log=(
            CleaningOpRecord(
                op=CleaningOp.WHITESPACE_NORMALIZE,
                op_version="1",
                chars_before=len(text) + 5,
                chars_after=len(text),
            ),
        ),
        parser_version="test:1",
    )


def test_cleaned_to_cpae_paper_full_chain() -> None:
    cleaned = _make_cleaned("On the photoelectric effect, by Einstein, 1905.")
    payload = cleaned_to_cpae_payload(
        cleaned,
        document_id="cpae-vol2-doc24",
        document_kind=CPAEDocumentKind.ARTICLE,
        volume=2,
        document_number=24,
        title="On a Heuristic Point of View Concerning Light",
        year=1905,
        language="de",
        source_url="https://einsteinpapers.press.princeton.edu/vol2-doc/24",
    )
    assert payload.body == cleaned.text
    assert payload.document_kind is CPAEDocumentKind.ARTICLE
    paper = cpae_to_paper_source(payload, figure_id="einstein")
    assert paper.body == cleaned.text
    assert paper.year == 1905
    assert paper.language == "de"
    assert paper.figure_id == "einstein"


def test_cleaned_to_cpae_letter_full_chain() -> None:
    cleaned = _make_cleaned("Lieber Kollege, vielen Dank fur den Brief.")
    payload = cleaned_to_cpae_payload(
        cleaned,
        document_id="cpae-vol5-doc100",
        document_kind=CPAEDocumentKind.LETTER,
        volume=5,
        document_number=100,
        title="To Michele Besso",
        year=1909,
        language="de",
        source_url="https://einsteinpapers.press.princeton.edu/vol5-doc/100",
        sender_id="einstein",
        recipient_id="besso",
        date_iso="1909-04-29",
    )
    letter = cpae_to_letter_source(payload, figure_id="einstein")
    assert letter.body == cleaned.text
    assert letter.sender_id == "einstein"
    assert letter.recipient_id == "besso"
    assert letter.date_iso == "1909-04-29"


def test_cleaned_to_wikisource_paper_and_lecture() -> None:
    cleaned = _make_cleaned("The body of the Wikisource transcription.")
    paper_payload = cleaned_to_wikisource_payload(
        cleaned,
        page_title="Annus_Mirabilis_Letter",
        language="en",
        source_url="https://en.wikisource.org/wiki/Annus_Mirabilis_Letter",
        year=1905,
    )
    paper = wikisource_to_paper_source(paper_payload, figure_id="einstein")
    assert paper.body == cleaned.text
    assert paper.year == 1905

    lecture_payload = cleaned_to_wikisource_payload(
        cleaned,
        page_title="Sample_Lecture",
        language="en",
        source_url="https://en.wikisource.org/wiki/Sample_Lecture",
        venue_id="solvay-1927",
        date_iso="1927-10-25",
    )
    lecture = wikisource_to_lecture_source(lecture_payload, figure_id="einstein")
    assert lecture.body == cleaned.text
    assert lecture.venue_id == "solvay-1927"


def test_cleaned_to_gutenberg_full_chain() -> None:
    cleaned = _make_cleaned("Chapter one of the Gutenberg ebook body.")
    payload = cleaned_to_gutenberg_payload(
        cleaned,
        ebook_id=12345,
        title="Sample Notes",
        language="en",
        source_url="https://www.gutenberg.org/files/12345/12345-0.txt",
        section_label="chapter-1",
        year=1916,
    )
    paper = gutenberg_to_paper_source(payload, figure_id="einstein")
    assert paper.body == cleaned.text
    assert "chapter-1" in paper.publication_locator


def test_cleaned_to_internet_archive_paper_and_lecture() -> None:
    cleaned = _make_cleaned("The transcript page concatenation goes here.")
    paper_payload = cleaned_to_internet_archive_payload(
        cleaned,
        identifier="sample-ia-paper",
        title="Scanned Article",
        language="eng",
        source_url="https://archive.org/details/sample-ia-paper",
        year=1921,
    )
    paper = internet_archive_to_paper_source(paper_payload, figure_id="einstein")
    assert paper.body == cleaned.text
    assert paper.year == 1921

    lecture_payload = cleaned_to_internet_archive_payload(
        cleaned,
        identifier="sample-ia-lecture",
        title="Recorded Lecture",
        language="eng",
        source_url="https://archive.org/details/sample-ia-lecture",
        venue_id="prague-1923",
        date_iso="1923-05-12",
    )
    lecture = internet_archive_to_lecture_source(lecture_payload, figure_id="einstein")
    assert lecture.body == cleaned.text
    assert lecture.venue_id == "prague-1923"
