"""Smoke tests for the F1.3 / D1 bundle + freeze flow.

Validates:

* ``primary_source_from_envelope_chunk`` lifts an adapter chunk into a
  typed ``PrimarySource`` and computes a stable sha256.
* ``build_figure_corpus_bundle`` assembles a frozen
  ``FigureCorpusBundle`` with deterministic integrity_hash + bundle_id.
* The same inputs at two different wall-clock times produce identical
  ``integrity_hash`` and ``bundle_id`` (R15 reproducibility).
* License summary classifies public-domain / licensed / unknown.
* Bundle rejects empty corpus, mismatched figure_id, duplicate ids.
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    BuildBundleInputs,
    DocumentKind,
    EvidenceStrength,
    FigurePaperSource,
    PrimarySource,
    build_figure_corpus_bundle,
    compute_corpus_bundle_integrity_hash,
    ingest_papers,
    primary_source_from_envelope_chunk,
)


_TIMESTAMP_A = "2026-05-10T00:00:00Z"
_TIMESTAMP_B = "2026-05-11T12:34:56Z"


def _einstein_paper_envelope():
    source = FigurePaperSource(
        paper_id="einstein-1905-001",
        title="Zur Elektrodynamik bewegter Körper",
        year=1905,
        language="de",
        body=(
            "First paragraph: definition of synchronization across "
            "spatially separated clocks.\n\n"
            "Second paragraph: derivation of the Lorentz transformation "
            "from the principle of relativity."
        ),
        figure_id="einstein",
    )
    return ingest_papers((source,), uploader="test", upload_ts_ms=0)


def _einstein_corpus() -> tuple[PrimarySource, ...]:
    envelope = _einstein_paper_envelope()
    return tuple(
        primary_source_from_envelope_chunk(
            chunk=chunk,
            figure_id="einstein",
            document_kind=DocumentKind.PAPER,
            written_lang="de",
            license="public-domain",
            source_url="https://einsteinpapers.press.princeton.edu/vol2-doc/153",
            written_year=1905,
            evidence_strength=EvidenceStrength.FIRST_HAND,
        )
        for chunk in envelope.chunks
    )


def test_primary_source_from_envelope_chunk_computes_stable_sha() -> None:
    envelope = _einstein_paper_envelope()
    sources = tuple(
        primary_source_from_envelope_chunk(
            chunk=chunk,
            figure_id="einstein",
            document_kind=DocumentKind.PAPER,
            written_lang="de",
            license="public-domain",
            source_url="https://example.invalid/einstein-1905-001",
        )
        for chunk in envelope.chunks
    )
    assert len(sources) == len(envelope.chunks)
    for source, chunk in zip(sources, envelope.chunks, strict=True):
        assert source.text == chunk.text
        assert source.locator == chunk.locator
        assert len(source.sha256) == 64
        assert source.figure_id == "einstein"
        assert source.document_kind == DocumentKind.PAPER


def test_build_figure_corpus_bundle_happy_path() -> None:
    corpus = _einstein_corpus()
    inputs = BuildBundleInputs(
        figure_id="einstein",
        reviewed_by="test-reviewer",
        primary_corpus=corpus,
        audit_note="D1 smoke",
    )
    bundle = build_figure_corpus_bundle(inputs, created_at_utc=_TIMESTAMP_A)
    assert bundle.figure_id == "einstein"
    assert bundle.schema_version == 1
    assert bundle.reviewed_by == "test-reviewer"
    assert bundle.created_at_utc == _TIMESTAMP_A
    assert len(bundle.integrity_hash) == 64
    assert bundle.bundle_id == f"figure-corpus:einstein:{bundle.integrity_hash[:16]}"
    assert bundle.statistics.document_count == len({s.source_id for s in corpus})
    assert bundle.statistics.paragraph_count == len(corpus)
    assert "de" in bundle.statistics.languages
    assert ("paper", len(corpus)) in bundle.statistics.document_kind_counts
    assert bundle.statistics.earliest_year == 1905
    assert bundle.statistics.latest_year == 1905
    assert bundle.license_summary.public_domain_source_count == len(corpus)
    assert bundle.license_summary.licensed_source_count == 0
    assert bundle.license_summary.unknown_license_source_count == 0
    assert not bundle.has_contrast()
    assert not bundle.has_coverage_map()
    assert not bundle.has_time_windows()


def test_integrity_hash_is_wall_clock_independent() -> None:
    corpus = _einstein_corpus()
    inputs = BuildBundleInputs(
        figure_id="einstein",
        reviewed_by="test-reviewer",
        primary_corpus=corpus,
    )
    bundle_a = build_figure_corpus_bundle(inputs, created_at_utc=_TIMESTAMP_A)
    bundle_b = build_figure_corpus_bundle(inputs, created_at_utc=_TIMESTAMP_B)
    assert bundle_a.integrity_hash == bundle_b.integrity_hash
    assert bundle_a.bundle_id == bundle_b.bundle_id
    assert bundle_a.created_at_utc != bundle_b.created_at_utc


def test_integrity_hash_changes_when_corpus_changes() -> None:
    corpus = _einstein_corpus()
    inputs_a = BuildBundleInputs(
        figure_id="einstein",
        reviewed_by="test-reviewer",
        primary_corpus=corpus,
    )
    extra_source = PrimarySource(
        source_id="extra-doc",
        document_kind=DocumentKind.LETTER,
        figure_id="einstein",
        written_lang="de",
        license="public-domain",
        source_url="https://example.invalid/extra",
        sha256="0" * 64,
        locator="letter:einstein-to-besso:date=1916-08-11:para=0:offset=0-50",
        text="Reviewer-fabricated extra letter body for hash sensitivity test.",
        written_year=1916,
    )
    inputs_b = BuildBundleInputs(
        figure_id="einstein",
        reviewed_by="test-reviewer",
        primary_corpus=corpus + (extra_source,),
    )
    hash_a = compute_corpus_bundle_integrity_hash(inputs_a)
    hash_b = compute_corpus_bundle_integrity_hash(inputs_b)
    assert hash_a != hash_b


def test_bundle_rejects_mismatched_figure_id() -> None:
    corpus = _einstein_corpus()
    inputs = BuildBundleInputs(
        figure_id="bohr",
        reviewed_by="test-reviewer",
        primary_corpus=corpus,
    )
    with pytest.raises(ValueError, match="figure_id"):
        build_figure_corpus_bundle(inputs, created_at_utc=_TIMESTAMP_A)


def test_inputs_reject_empty_reviewed_by() -> None:
    corpus = _einstein_corpus()
    with pytest.raises(ValueError, match="reviewed_by"):
        BuildBundleInputs(
            figure_id="einstein",
            reviewed_by="",
            primary_corpus=corpus,
        )


def test_inputs_reject_empty_corpus() -> None:
    with pytest.raises(ValueError, match="primary_corpus"):
        BuildBundleInputs(
            figure_id="einstein",
            reviewed_by="test-reviewer",
            primary_corpus=(),
        )


def test_license_summary_classifies_unknown_and_licensed() -> None:
    corpus = _einstein_corpus()
    licensed = PrimarySource(
        source_id="licensed-source",
        document_kind=DocumentKind.PAPER,
        figure_id="einstein",
        written_lang="de",
        license="cc-by-4.0",
        source_url="https://example.invalid/cc-by",
        sha256="1" * 64,
        locator="paper:einstein-1922-001:lang=de:para=0:offset=0-50",
        text="Reviewer-fabricated licensed body for license summary test.",
        written_year=1922,
    )
    unknown = PrimarySource(
        source_id="unknown-source",
        document_kind=DocumentKind.NOTEBOOK,
        figure_id="einstein",
        written_lang="de",
        license="unknown",
        source_url="https://example.invalid/unknown",
        sha256="2" * 64,
        locator="notebook:einstein-1912:vol=A:page=1:lang=de:para=0:offset=0-50",
        text="Reviewer-fabricated unknown-license body for license summary test.",
        written_year=1912,
    )
    inputs = BuildBundleInputs(
        figure_id="einstein",
        reviewed_by="test-reviewer",
        primary_corpus=corpus + (licensed, unknown),
    )
    bundle = build_figure_corpus_bundle(inputs, created_at_utc=_TIMESTAMP_A)
    summary = bundle.license_summary
    assert summary.licensed_source_count == 1
    assert summary.unknown_license_source_count == 1
    assert summary.public_domain_source_count == len(corpus)
    assert "cc-by-4.0" in summary.distinct_licenses


def test_primary_source_rejects_bad_sha() -> None:
    with pytest.raises(ValueError, match="sha256"):
        PrimarySource(
            source_id="x",
            document_kind=DocumentKind.PAPER,
            figure_id="einstein",
            written_lang="de",
            license="public-domain",
            source_url="https://example.invalid/x",
            sha256="not-a-real-hash",
            locator="paper:x:para=0:offset=0-1",
            text="x",
        )
