"""Smoke tests for FigureArtifactBundle metadata fingerprint (debt #25)."""

from __future__ import annotations

from lifeform_domain_figure.compiler import (
    FigureBundleInputs,
    build_figure_artifact_bundle,
)
from lifeform_domain_figure.corpus.ingest_papers import ingest_papers
from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash
from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    DomainCoverageHint,
    FigureLifespan,
    MetadataDigest,
    MetadataSource,
    aggregate_metadata,
)
from lifeform_domain_figure.profiles.einstein import build_einstein_profile
from lifeform_domain_figure.sample_corpus import synthetic_einstein_corpus


def _lifespan() -> FigureLifespan:
    return FigureLifespan(
        figure_id="einstein",
        birth_year=1879,
        death_year=1955,
        source=MetadataSource.WIKIDATA,
        source_id="Q937",
        confidence=0.95,
    )


def _digest_v1() -> MetadataDigest:
    return aggregate_metadata(
        figure_id="einstein",
        lifespan=_lifespan(),
        authored_works=(
            AuthoredWorkSummary(
                work_id="openalex:W1",
                figure_id="einstein",
                title="t1",
                year=1905,
                venue="v",
                language="de",
                topic_tags=("a",),
                source=MetadataSource.OPENALEX,
                source_id="W1",
            ),
        ),
        coverage_hints=(
            DomainCoverageHint(
                label="topic-a",
                description="d",
                is_out_of_scope=False,
                source=MetadataSource.OPENALEX,
                source_id="W1",
                confidence=0.6,
            ),
        ),
        time_window_hints=(),
    )


def _digest_v2() -> MetadataDigest:
    return aggregate_metadata(
        figure_id="einstein",
        lifespan=_lifespan(),
        authored_works=(
            AuthoredWorkSummary(
                work_id="openalex:W2",
                figure_id="einstein",
                title="t2 (different snapshot)",
                year=1916,
                venue="v",
                language="de",
                topic_tags=("b",),
                source=MetadataSource.OPENALEX,
                source_id="W2",
            ),
        ),
        coverage_hints=(),
        time_window_hints=(),
    )


def _build_inputs(digest: MetadataDigest | None) -> FigureBundleInputs:
    papers, _l, _le, _n = synthetic_einstein_corpus()
    envelope = ingest_papers(
        (papers[0],), uploader="metadata-fp-test", upload_ts_ms=1_700_000_000_000
    )
    return FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        metadata_digest=digest,
    )


def test_bundle_without_digest_has_empty_fingerprint() -> None:
    bundle = build_figure_artifact_bundle(_build_inputs(None))
    assert bundle.metadata_digest_fingerprint == ""


def test_bundle_with_digest_records_fingerprint() -> None:
    digest = _digest_v1()
    bundle = build_figure_artifact_bundle(_build_inputs(digest))
    assert bundle.metadata_digest_fingerprint == digest.fingerprint


def test_two_distinct_digests_yield_distinct_bundle_hashes() -> None:
    bundle_v1 = build_figure_artifact_bundle(_build_inputs(_digest_v1()))
    bundle_v2 = build_figure_artifact_bundle(_build_inputs(_digest_v2()))
    assert bundle_v1.integrity_hash != bundle_v2.integrity_hash
    assert bundle_v1.bundle_id != bundle_v2.bundle_id


def test_no_digest_hash_equals_legacy_call_signature() -> None:
    """Backward compat: bundle without metadata_digest hashes identically to
    the pre-debt-#25 compute_bundle_integrity_hash path.
    """

    legacy = compute_bundle_integrity_hash(
        figure_id="x",
        profile_version="v1",
        version_window=(0, 0),
        retrieval_integrity="r",
        coverage_integrity="c",
        style_integrity="s",
        steering_integrity="absent",
        lora_integrity="absent",
    )
    new_no_digest = compute_bundle_integrity_hash(
        figure_id="x",
        profile_version="v1",
        version_window=(0, 0),
        retrieval_integrity="r",
        coverage_integrity="c",
        style_integrity="s",
        steering_integrity="absent",
        lora_integrity="absent",
        metadata_digest_fingerprint="",
    )
    assert legacy == new_no_digest


def test_with_digest_hash_differs_from_no_digest() -> None:
    no_digest = compute_bundle_integrity_hash(
        figure_id="x",
        profile_version="v1",
        version_window=(0, 0),
        retrieval_integrity="r",
        coverage_integrity="c",
        style_integrity="s",
        steering_integrity="absent",
        lora_integrity="absent",
    )
    with_digest = compute_bundle_integrity_hash(
        figure_id="x",
        profile_version="v1",
        version_window=(0, 0),
        retrieval_integrity="r",
        coverage_integrity="c",
        style_integrity="s",
        steering_integrity="absent",
        lora_integrity="absent",
        metadata_digest_fingerprint="deadbeef",
    )
    assert no_digest != with_digest
