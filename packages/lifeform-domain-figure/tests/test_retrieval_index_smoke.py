"""Smoke tests for the F2.1 retrieval index (L3 backbone).

Validates:

* Index builds from the synthetic corpus and contains a non-zero
  number of chunks across all four source kinds.
* ``retrieve(query)`` returns top-K evidence with locator preserved
  exactly (no string mangling).
* Citation locators carry the corpus-kind prefix that the L3
  contract reads.
* ``assertion_is_supported`` returns evidence above threshold for
  in-corpus assertions and empty for off-topic ones.
* Empty / whitespace queries return an empty tuple instead of
  random matches (fail-loud signal for L3 enforcement).
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    build_figure_ingestion_envelope,
    build_figure_retrieval_index,
    synthetic_einstein_corpus,
)


def _build_index_from_synthetic():
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    return build_figure_retrieval_index(
        figure_id="einstein",
        envelopes=envelope_set.envelopes,
    )


def test_retrieval_index_builds_from_synthetic_corpus() -> None:
    index = _build_index_from_synthetic()
    assert index.figure_id == "einstein"
    assert index.total_chunks >= 4
    assert index.integrity_hash
    locator_prefixes = {rec.locator.split(":", 1)[0] for rec in index.chunk_records}
    assert {"paper", "letter", "lecture", "notebook"}.issubset(locator_prefixes)


def test_retrieve_returns_top_k_with_citation_locators() -> None:
    index = _build_index_from_synthetic()
    results = index.retrieve("locally separable physical state", top_k=3)
    assert 1 <= len(results) <= 3
    assert all(result.score > 0.0 for result in results)
    for result in results:
        assert result.locator
        assert "|" in result.citation
        assert result.text


def test_retrieve_locator_is_unmodified() -> None:
    index = _build_index_from_synthetic()
    results = index.retrieve("incomplete quantum mechanics", top_k=5)
    locators_in_index = {rec.locator for rec in index.chunk_records}
    for result in results:
        assert result.locator in locators_in_index


def test_assertion_supported_for_in_corpus_topic() -> None:
    index = _build_index_from_synthetic()
    evidence = index.assertion_is_supported(
        "Physical reality should be locally separable; the theory is incomplete.",
        top_k=3,
    )
    assert evidence, "in-corpus assertion must surface at least one supporting chunk"


def test_assertion_unsupported_for_out_of_corpus_topic() -> None:
    index = _build_index_from_synthetic()
    evidence = index.assertion_is_supported(
        "TikTok algorithmic feed engagement metrics in 2024.",
        top_k=3,
        score_threshold=0.18,
    )
    assert not evidence, "off-topic assertion must not surface support above threshold"


def test_retrieve_empty_query_returns_empty() -> None:
    index = _build_index_from_synthetic()
    assert index.retrieve("   ", top_k=3) == ()


def test_retrieve_top_k_must_be_positive() -> None:
    index = _build_index_from_synthetic()
    with pytest.raises(ValueError, match="top_k"):
        index.retrieve("anything", top_k=0)


def test_build_index_rejects_empty_envelopes() -> None:
    with pytest.raises(ValueError, match="envelopes"):
        build_figure_retrieval_index(figure_id="einstein", envelopes=())


def test_index_rebuild_is_deterministic() -> None:
    a = _build_index_from_synthetic()
    b = _build_index_from_synthetic()
    assert a.integrity_hash == b.integrity_hash
    assert a.total_chunks == b.total_chunks
