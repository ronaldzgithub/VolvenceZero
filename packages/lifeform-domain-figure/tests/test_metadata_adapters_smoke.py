"""Smoke tests for the D4 metadata adapters."""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    HistoricalFigureProfile,
    build_einstein_profile,
    build_figure_coverage_map,
    build_figure_ingestion_envelope,
    build_figure_retrieval_index,
    FigureCorpusSourceBundle,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.metadata import (
    AuthoredWorkSummary,
    CrossrefWorkPayload,
    DomainCoverageHint,
    FigureLifespan,
    MetadataDigest,
    MetadataSource,
    OpenAlexWorkPayload,
    SEPEntryPayload,
    TimeWindowHint,
    WikidataPersonPayload,
    aggregate_metadata,
    build_time_window_hints_from_lifespan,
    crossref_to_authored_work,
    enrich_profile_with_metadata,
    offline_crossref_client,
    offline_openalex_client,
    offline_sep_client,
    offline_wikidata_client,
    openalex_to_authored_work,
    openalex_to_domain_hints,
    sep_to_domain_hints,
    wikidata_to_lifespan,
    wikidata_to_time_window_hints,
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_figure_lifespan_validates_year_order() -> None:
    with pytest.raises(ValueError, match="death_year"):
        FigureLifespan(
            figure_id="einstein",
            birth_year=1879,
            death_year=1800,
            source=MetadataSource.WIKIDATA,
            source_id="Q937",
            confidence=0.95,
        )


def test_aggregate_metadata_fingerprint_is_stable() -> None:
    lifespan = FigureLifespan(
        figure_id="einstein",
        birth_year=1879,
        death_year=1955,
        source=MetadataSource.WIKIDATA,
        source_id="Q937",
        confidence=0.95,
    )
    digest_a = aggregate_metadata(figure_id="einstein", lifespan=lifespan)
    digest_b = aggregate_metadata(figure_id="einstein", lifespan=lifespan)
    assert digest_a.fingerprint == digest_b.fingerprint
    assert len(digest_a.fingerprint) == 64


def test_aggregate_metadata_validates_lifespan_match() -> None:
    lifespan = FigureLifespan(
        figure_id="bohr",
        birth_year=1885,
        death_year=1962,
        source=MetadataSource.WIKIDATA,
        source_id="Q7185",
        confidence=0.95,
    )
    with pytest.raises(ValueError, match="figure_id"):
        aggregate_metadata(figure_id="einstein", lifespan=lifespan)


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def test_openalex_payload_to_authored_work_and_hints() -> None:
    payload = OpenAlexWorkPayload(
        openalex_id="W4205692301",
        title="Zur Elektrodynamik bewegter Körper",
        publication_year=1905,
        venue="Annalen der Physik",
        language="de",
        concept_labels=("Special relativity", "Electrodynamics", "Foundations of physics"),
        primary_topic="Special relativity",
        cited_by_count=12345,
    )
    work = openalex_to_authored_work(payload, figure_id="einstein")
    assert isinstance(work, AuthoredWorkSummary)
    assert work.work_id == "openalex:W4205692301"
    assert work.year == 1905
    assert work.source is MetadataSource.OPENALEX
    hints = openalex_to_domain_hints(payload)
    assert all(isinstance(h, DomainCoverageHint) for h in hints)
    assert {h.label for h in hints} == set(payload.concept_labels)
    assert all(not h.is_out_of_scope for h in hints)


def test_offline_openalex_client_raises() -> None:
    client = offline_openalex_client()
    with pytest.raises(NotImplementedError):
        client.fetch_author_works(openalex_author_id="A123")


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------


def test_wikidata_to_lifespan() -> None:
    payload = WikidataPersonPayload(
        qid="Q937",
        label="Albert Einstein",
        birth_year=1879,
        death_year=1955,
        occupation_labels=("theoretical physicist",),
    )
    lifespan = wikidata_to_lifespan(payload, figure_id="einstein")
    assert lifespan.birth_year == 1879
    assert lifespan.death_year == 1955
    assert lifespan.source is MetadataSource.WIKIDATA


def test_wikidata_time_window_hints_with_split() -> None:
    payload = WikidataPersonPayload(
        qid="Q937",
        label="Albert Einstein",
        birth_year=1879,
        death_year=1955,
    )
    hints = wikidata_to_time_window_hints(
        payload, figure_id="einstein", splits_at_years=(1925,)
    )
    assert len(hints) == 2
    assert hints[0].year_start == 1879
    assert hints[0].year_end == 1925
    assert hints[1].year_start == 1925
    assert hints[1].year_end == 1955
    assert hints[0].window_id.endswith(":early")
    assert hints[1].window_id.endswith(":late")


def test_wikidata_time_window_rejects_split_outside_lifespan() -> None:
    payload = WikidataPersonPayload(
        qid="Q937",
        label="Albert Einstein",
        birth_year=1879,
        death_year=1955,
    )
    with pytest.raises(ValueError, match="strictly inside"):
        wikidata_to_time_window_hints(
            payload, figure_id="einstein", splits_at_years=(1800,)
        )


def test_offline_wikidata_client_raises() -> None:
    with pytest.raises(NotImplementedError):
        offline_wikidata_client().fetch_person(qid="Q937")


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------


def test_crossref_payload_to_authored_work() -> None:
    payload = CrossrefWorkPayload(
        doi="10.1002/andp.19053221004",
        title="Zur Elektrodynamik bewegter Körper",
        publication_year=1905,
        container_title="Annalen der Physik",
        language="de",
        subject_tags=("foundations",),
        volume="322",
        issue="10",
    )
    work = crossref_to_authored_work(payload, figure_id="einstein")
    assert work.work_id == "crossref:10.1002/andp.19053221004"
    assert "vol=322" in work.venue
    assert "issue=10" in work.venue


def test_offline_crossref_client_raises() -> None:
    with pytest.raises(NotImplementedError):
        offline_crossref_client().fetch_work(doi="10.1234/x")


# ---------------------------------------------------------------------------
# SEP
# ---------------------------------------------------------------------------


def test_sep_payload_to_domain_hints() -> None:
    payload = SEPEntryPayload(
        entry_slug="einstein-philscience",
        title="Einstein's Philosophy of Science",
        section_titles=(
            "Realism and the EPR thought experiment",
            "Determinism and Hidden Variables",
            "Pacifism and Politics",
        ),
        summary="A reviewed encyclopedia entry on Einstein's philosophy.",
    )
    hints = sep_to_domain_hints(payload)
    assert len(hints) == 3
    assert all(h.source is MetadataSource.SEP for h in hints)
    assert all(not h.is_out_of_scope for h in hints)
    assert {h.label for h in hints} == set(payload.section_titles)


def test_sep_payload_rejects_empty_outline() -> None:
    with pytest.raises(ValueError, match="section_titles"):
        SEPEntryPayload(
            entry_slug="x",
            title="t",
            section_titles=(),
            summary="s",
        )


def test_offline_sep_client_raises() -> None:
    with pytest.raises(NotImplementedError):
        offline_sep_client().fetch_entry(slug="einstein-philscience")


# ---------------------------------------------------------------------------
# Time-window builder
# ---------------------------------------------------------------------------


def test_time_window_builder_single_window_when_no_splits() -> None:
    lifespan = FigureLifespan(
        figure_id="einstein",
        birth_year=1879,
        death_year=1955,
        source=MetadataSource.WIKIDATA,
        source_id="Q937",
        confidence=0.95,
    )
    hints = build_time_window_hints_from_lifespan(lifespan)
    assert len(hints) == 1
    assert hints[0].year_start == 1879
    assert hints[0].year_end == 1955


def test_time_window_builder_open_ended_for_living_figure() -> None:
    lifespan = FigureLifespan(
        figure_id="future-figure",
        birth_year=2000,
        death_year=None,
        source=MetadataSource.WIKIDATA,
        source_id="Q-future",
        confidence=0.5,
    )
    hints = build_time_window_hints_from_lifespan(lifespan)
    assert len(hints) == 1
    assert hints[0].year_end == 9999


# ---------------------------------------------------------------------------
# Coverage enrichment
# ---------------------------------------------------------------------------


def test_enrich_profile_widens_coverage_seed_and_adds_boundary() -> None:
    profile = build_einstein_profile()
    base_seed_count = len(profile.domain_coverage_seed)
    base_boundary_count = len(profile.boundary_priors)
    digest = aggregate_metadata(
        figure_id=profile.profile_id,
        lifespan=FigureLifespan(
            figure_id=profile.profile_id,
            birth_year=1879,
            death_year=1955,
            source=MetadataSource.WIKIDATA,
            source_id="Q937",
            confidence=0.95,
        ),
        coverage_hints=(
            DomainCoverageHint(
                label="Brownian motion",
                description="OpenAlex concept tag",
                is_out_of_scope=False,
                source=MetadataSource.OPENALEX,
                source_id="W-bm",
                confidence=0.6,
            ),
            DomainCoverageHint(
                label="medical_diagnosis",
                description="Out of scope",
                is_out_of_scope=True,
                source=MetadataSource.OPENALEX,
                source_id="W-skip",
                confidence=0.6,
            ),
        ),
    )
    enriched = enrich_profile_with_metadata(profile, digest)
    assert isinstance(enriched, HistoricalFigureProfile)
    assert "Brownian motion" in enriched.domain_coverage_seed
    assert "medical_diagnosis" not in enriched.domain_coverage_seed
    assert len(enriched.domain_coverage_seed) == base_seed_count + 1
    assert len(enriched.boundary_priors) == base_boundary_count


def test_enrich_profile_idempotent_for_existing_post_lifespan_boundary() -> None:
    profile = build_einstein_profile()
    digest = aggregate_metadata(
        figure_id=profile.profile_id,
        lifespan=FigureLifespan(
            figure_id=profile.profile_id,
            birth_year=1879,
            death_year=1955,
            source=MetadataSource.WIKIDATA,
            source_id="Q937",
            confidence=0.95,
        ),
    )
    enriched = enrich_profile_with_metadata(profile, digest)
    second_pass = enrich_profile_with_metadata(enriched, digest)
    assert len(enriched.boundary_priors) == len(second_pass.boundary_priors)


def test_enriched_profile_still_builds_coverage_map() -> None:
    """End-to-end sanity: enriched profile + retrieval index → coverage map."""
    profile = build_einstein_profile()
    digest = aggregate_metadata(
        figure_id=profile.profile_id,
        coverage_hints=(
            DomainCoverageHint(
                label="general_relativity",
                description="OpenAlex concept tag",
                is_out_of_scope=False,
                source=MetadataSource.OPENALEX,
                source_id="W-gr",
                confidence=0.6,
            ),
        ),
    )
    enriched = enrich_profile_with_metadata(profile, digest)
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    retrieval = build_figure_retrieval_index(
        figure_id="einstein",
        envelopes=envelope_set.envelopes,
    )
    coverage = build_figure_coverage_map(
        figure_id="einstein",
        profile=enriched,
        retrieval_index=retrieval,
    )
    assert coverage.figure_id == "einstein"
    assert coverage.domain_centroids


def test_enrich_profile_rejects_mismatched_figure_id() -> None:
    profile = build_einstein_profile()
    digest = aggregate_metadata(figure_id="bohr")
    with pytest.raises(ValueError, match="figure_id"):
        enrich_profile_with_metadata(profile, digest)
