"""Cross-cutting contract: metadata digest folds into bundle integrity hash.

Closes debt #25's R15 byte-level rollback contract. Three invariants:

1. Same profile + same digest -> same bundle hash.
2. Same profile + different digest -> different bundle hash.
3. Same profile + ``digest=None`` -> empty fingerprint field, bundle
   hash matches the legacy compute_bundle_integrity_hash signature
   without the new ``metadata_digest_fingerprint`` arg.

Contract anchor: anyone removing the metadata fingerprint folding
(or accidentally regressing it to "always include even when empty")
breaks one of these tests.
"""

from __future__ import annotations

from lifeform_domain_figure.compiler import (
    FigureBundleInputs,
    build_figure_artifact_bundle,
)
from lifeform_domain_figure.corpus.ingest_papers import ingest_papers
from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash
from lifeform_domain_figure.metadata.records import (
    AuthoredWorkSummary,
    FigureLifespan,
    MetadataDigest,
    MetadataSource,
    aggregate_metadata,
)
from lifeform_domain_figure.profiles.einstein import build_einstein_profile
from lifeform_domain_figure.sample_corpus import synthetic_einstein_corpus


def _build(digest: MetadataDigest | None) -> FigureBundleInputs:
    papers, _l, _le, _n = synthetic_einstein_corpus()
    envelope = ingest_papers(
        (papers[0],), uploader="contract-test", upload_ts_ms=1_700_000_000_000
    )
    return FigureBundleInputs(
        profile=build_einstein_profile(),
        envelopes=(envelope,),
        metadata_digest=digest,
    )


def _digest(work_id: str) -> MetadataDigest:
    return aggregate_metadata(
        figure_id="einstein",
        lifespan=FigureLifespan(
            figure_id="einstein",
            birth_year=1879,
            death_year=1955,
            source=MetadataSource.WIKIDATA,
            source_id="Q937",
            confidence=0.95,
        ),
        authored_works=(
            AuthoredWorkSummary(
                work_id=f"openalex:{work_id}",
                figure_id="einstein",
                title=f"title-{work_id}",
                year=1905,
                venue="v",
                language="de",
                topic_tags=("topic",),
                source=MetadataSource.OPENALEX,
                source_id=work_id,
            ),
        ),
        coverage_hints=(),
        time_window_hints=(),
    )


def test_same_digest_same_hash() -> None:
    digest = _digest("W1")
    a = build_figure_artifact_bundle(_build(digest))
    b = build_figure_artifact_bundle(_build(digest))
    assert a.integrity_hash == b.integrity_hash
    assert a.metadata_digest_fingerprint == digest.fingerprint


def test_different_digest_different_hash() -> None:
    a = build_figure_artifact_bundle(_build(_digest("W1")))
    b = build_figure_artifact_bundle(_build(_digest("W2")))
    assert a.integrity_hash != b.integrity_hash


def test_no_digest_yields_empty_fingerprint_and_legacy_hash() -> None:
    bundle = build_figure_artifact_bundle(_build(None))
    assert bundle.metadata_digest_fingerprint == ""
    legacy = compute_bundle_integrity_hash(
        figure_id=bundle.figure_id,
        profile_version=bundle.profile_version,
        version_window=bundle.version_window,
        retrieval_integrity=bundle.retrieval_index.integrity_hash,
        coverage_integrity=bundle.coverage_map.integrity_hash,
        style_integrity=bundle.style_prior.integrity_hash,
        steering_integrity="absent",
        lora_integrity="absent",
    )
    assert bundle.integrity_hash == legacy


def test_steering_attach_preserves_metadata_fingerprint() -> None:
    """attach_steering_to_bundle must preserve metadata_digest_fingerprint
    in its recomputed integrity hash so reattaching steering does not
    silently invalidate the digest audit chain.
    """

    from lifeform_domain_figure.compiler import attach_steering_to_bundle

    bundle = build_figure_artifact_bundle(_build(_digest("W1")))
    new_bundle = attach_steering_to_bundle(
        bundle, steering=object(), steering_integrity="abc"
    )
    assert new_bundle.metadata_digest_fingerprint == bundle.metadata_digest_fingerprint
