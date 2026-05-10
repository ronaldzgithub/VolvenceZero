"""Smoke tests for the D7 Chinese figure PoC.

Validates:

* The reviewed 鲁迅 :class:`HistoricalFigureProfile` builds and
  passes the standard schema validators.
* The Chinese Text Project (CTP) archive adapter normalises a
  pre-downloaded chapter payload into a :class:`FigurePaperSource`
  with a citation locator under the ``ctext:`` namespace.
* The 鲁迅 profile compiles end-to-end with the existing F2.x
  pipeline (retrieval index + coverage map) using a synthetic
  Chinese-language paragraph as the primary corpus.
* The Wikidata + metadata enrichment loop works with the figure's
  Chinese lifespan (1881-1936).
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    CTPPayload,
    FigureCorpusSourceBundle,
    FigurePaperSource,
    HistoricalFigureProfile,
    aggregate_metadata,
    build_figure_coverage_map,
    build_figure_ingestion_envelope,
    build_figure_retrieval_index,
    build_lu_xun_profile,
    ctp_to_paper_source,
    enrich_profile_with_metadata,
)
from lifeform_domain_figure.metadata import (
    FigureLifespan,
    MetadataSource,
    WikidataPersonPayload,
    wikidata_to_lifespan,
    wikidata_to_time_window_hints,
)


_LU_XUN_SYNTHETIC_PARAGRAPH = (
    "[Synthetic reviewer paraphrase. Not derived from any published "
    "primary source.]\n\n"
    "白话文是文化普及的前提；没有共同的语言，便没有共同的对话。"
    "古代的辞章自有其价值，但若只让少数士大夫读得懂，则与广大群众"
    "无关。这并非否定传统，而是承认普及之需。\n\n"
    "讽刺的笔调并不是为了刻薄，而是为了让昏睡的人感到一种刺痛。"
    "若仅以颂扬为业，便忘了一个写作者的本份。\n\n"
    "对于政府的差使，作者一向有所节制；俸禄附带的条件，往往是"
    "言论的束缚。宁少一份薪水，不少一份独立。"
)


def test_lu_xun_profile_builds() -> None:
    profile = build_lu_xun_profile()
    assert isinstance(profile, HistoricalFigureProfile)
    assert profile.profile_id == "lu-xun"
    assert profile.figure_lifespan == (1881, 1936)
    assert profile.knowledge_seeds, "must have knowledge seeds"
    assert profile.signature_cases, "must have signature cases"
    assert profile.boundary_priors, "must have boundary priors"
    assert profile.drive_priors, "must have drive priors"
    assert "literature_and_critique" in profile.domain_coverage_seed
    assert {window.window_id for window in profile.time_windows} == {
        "early-1903-1918",
        "new-culture-1918-1927",
        "late-shanghai-1927-1936",
    }


def test_lu_xun_profile_post_lifespan_boundary_present() -> None:
    profile = build_lu_xun_profile()
    absolute_boundaries = [
        b
        for b in profile.boundary_priors
        if b.answer_depth_limit_hint == "absolute"
    ]
    assert absolute_boundaries, (
        "鲁迅 profile must declare an absolute post-lifespan boundary"
    )
    assert any(
        topic.startswith("post_1936")
        for boundary in absolute_boundaries
        for topic in boundary.out_of_scope_topics
    )


def test_ctp_adapter_normalises_payload() -> None:
    payload = CTPPayload(
        collection="analects",
        chapter_id="xueer",
        title="論語 · 學而",
        body=(
            "[Synthetic reviewer paraphrase.]\n\n"
            "學而時習之，不亦悅乎；有朋自遠方來，不亦樂乎。"
        ),
        source_url="https://ctext.org/analects/xueer",
        section_id="1",
        estimated_year=-500,
        language="zh-Hant",
    )
    source = ctp_to_paper_source(payload, figure_id="confucius")
    assert isinstance(source, FigurePaperSource)
    assert source.paper_id == "ctext:analects:xueer:1"
    assert source.publication_locator == "ctext:analects:xueer:section=1"
    assert source.figure_id == "confucius"
    assert source.year == -500
    assert source.language == "zh-Hant"


def test_ctp_payload_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="body"):
        CTPPayload(
            collection="analects",
            chapter_id="xueer",
            title="x",
            body="",
            source_url="https://ctext.org/x",
        )


def _build_lu_xun_corpus_bundle() -> FigureCorpusSourceBundle:
    paper = FigurePaperSource(
        paper_id="lu-xun-synth-1925-001",
        title="文化随感（synthetic placeholder）",
        year=1925,
        language="zh",
        body=_LU_XUN_SYNTHETIC_PARAGRAPH,
        figure_id="lu-xun",
    )
    return FigureCorpusSourceBundle(figure_id="lu-xun", papers=(paper,))


def test_lu_xun_full_chain_compiles() -> None:
    profile = build_lu_xun_profile()
    bundle = _build_lu_xun_corpus_bundle()
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    retrieval = build_figure_retrieval_index(
        figure_id="lu-xun",
        envelopes=envelope_set.envelopes,
    )
    coverage = build_figure_coverage_map(
        figure_id="lu-xun",
        profile=profile,
        retrieval_index=retrieval,
    )
    assert retrieval.figure_id == "lu-xun"
    assert retrieval.total_chunks >= 1
    assert coverage.figure_id == "lu-xun"
    assert coverage.domain_centroids


def test_lu_xun_metadata_enrichment_with_wikidata() -> None:
    profile = build_lu_xun_profile()
    payload = WikidataPersonPayload(
        qid="Q23114",  # actual Wikidata Q-id for 鲁迅
        label="鲁迅",
        birth_year=1881,
        death_year=1936,
        occupation_labels=("writer", "essayist", "translator"),
    )
    lifespan = wikidata_to_lifespan(payload, figure_id="lu-xun")
    assert isinstance(lifespan, FigureLifespan)
    assert lifespan.death_year == 1936
    hints = wikidata_to_time_window_hints(
        payload, figure_id="lu-xun", splits_at_years=(1918, 1927)
    )
    assert len(hints) == 3
    digest = aggregate_metadata(
        figure_id="lu-xun",
        lifespan=lifespan,
        time_window_hints=hints,
    )
    enriched = enrich_profile_with_metadata(profile, digest)
    assert enriched.profile_id == "lu-xun"
    # 鲁迅 profile already declares a post_1936 boundary so the
    # enrichment should be idempotent on that axis.
    assert len(enriched.boundary_priors) == len(profile.boundary_priors)


def test_lu_xun_query_classification_recognises_in_domain_chinese() -> None:
    profile = build_lu_xun_profile()
    bundle = _build_lu_xun_corpus_bundle()
    envelope_set = build_figure_ingestion_envelope(bundle, uploader="test")
    retrieval = build_figure_retrieval_index(
        figure_id="lu-xun",
        envelopes=envelope_set.envelopes,
    )
    coverage = build_figure_coverage_map(
        figure_id="lu-xun",
        profile=profile,
        retrieval_index=retrieval,
    )
    classification = coverage.classify_query("白话文 文化批判 现代性")
    assert classification.decision is not None
    assert classification.rationale
