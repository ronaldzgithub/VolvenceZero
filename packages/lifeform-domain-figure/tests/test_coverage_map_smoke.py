"""Smoke tests for the F2.2 coverage map (L4 backbone).

Validates:

* Map builds from Einstein profile + synthetic corpus retrieval index.
* In-corpus questions classify IN_DOMAIN with non-trivial similarity.
* Reviewer-declared out-of-scope topics classify BOUNDARY_BLOCKED.
* Genuinely off-topic queries classify OUT_OF_DOMAIN.
* Build rejects degenerate inputs (empty thresholds out of range).
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    CoverageDecision,
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_coverage_map,
    build_figure_ingestion_envelope,
    build_figure_retrieval_index,
    synthetic_einstein_corpus,
)


def _build_coverage_map():
    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    index = build_figure_retrieval_index(
        figure_id="einstein", envelopes=envelope_set.envelopes
    )
    return profile, build_figure_coverage_map(
        figure_id="einstein", profile=profile, retrieval_index=index
    )


def test_coverage_map_builds_with_centroids() -> None:
    profile, cmap = _build_coverage_map()
    assert cmap.figure_id == "einstein"
    assert cmap.domain_centroids
    assert cmap.out_of_scope_centroids
    assert cmap.integrity_hash
    profile_oos = {
        topic
        for boundary in profile.boundary_priors
        for topic in boundary.out_of_scope_topics
    }
    centroid_labels = {c.label for c in cmap.out_of_scope_centroids}
    assert centroid_labels.issubset(profile_oos)


def test_classify_in_domain_question_is_in_domain() -> None:
    _, cmap = _build_coverage_map()
    result = cmap.classify_query(
        "What did the author argue about the locality of physical states "
        "in his foundational paper on mechanics?"
    )
    assert result.decision == CoverageDecision.IN_DOMAIN
    assert result.closest_in_domain_score > result.closest_out_of_scope_score


def test_classify_out_of_scope_topic_is_boundary_blocked() -> None:
    _, cmap = _build_coverage_map()
    result = cmap.classify_query(
        "What is the contemporary AI policy stance on geopolitical events?"
    )
    assert result.decision == CoverageDecision.BOUNDARY_BLOCKED
    assert result.closest_out_of_scope_label
    assert result.closest_out_of_scope_score >= cmap.boundary_threshold


def test_classify_off_topic_question_is_out_of_domain() -> None:
    _, cmap = _build_coverage_map()
    result = cmap.classify_query(
        "What is the best apricot jam recipe for a sourdough breakfast?"
    )
    assert result.decision == CoverageDecision.OUT_OF_DOMAIN


def test_classify_empty_query_is_out_of_domain() -> None:
    _, cmap = _build_coverage_map()
    result = cmap.classify_query("a")
    assert result.decision == CoverageDecision.OUT_OF_DOMAIN


def test_build_rejects_invalid_thresholds() -> None:
    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    index = build_figure_retrieval_index(
        figure_id="einstein", envelopes=envelope_set.envelopes
    )
    with pytest.raises(ValueError, match="in_domain_threshold"):
        build_figure_coverage_map(
            figure_id="einstein",
            profile=profile,
            retrieval_index=index,
            in_domain_threshold=1.5,
        )
    with pytest.raises(ValueError, match="boundary_threshold"):
        build_figure_coverage_map(
            figure_id="einstein",
            profile=profile,
            retrieval_index=index,
            boundary_threshold=0.0,
        )


def test_classify_query_rationale_is_descriptive() -> None:
    _, cmap = _build_coverage_map()
    result = cmap.classify_query(
        "What did the author write about determinism, locality, and the "
        "incompleteness of the quantum description?"
    )
    assert result.rationale
    assert "cosine" in result.rationale or "out-of-domain" in result.rationale
