"""Smoke tests for the F2.3 style prior + bundle compiler.

Validates:

* :class:`FigureStylePrior` builds with non-empty top words /
  bigrams and reports a deterministic integrity hash.
* :class:`FigureArtifactBundle` assembles end-to-end from the
  Einstein profile + synthetic corpus.
* Steering / LoRA slots are ``None`` until later packets attach
  them, but their integrity always factors into the bundle hash.
* Bundle re-assembly is byte-deterministic (same inputs → same
  ``bundle_id`` and ``integrity_hash``).
* Time-window selection produces a different bundle id (R15
  rollback contract: any identity-bearing change yields a fresh
  bundle).
* Mismatched ``figure_id`` across artifacts is rejected by the
  bundle constructor.
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FIGURE_BUNDLE_SCHEMA_VERSION,
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    attach_lora_to_bundle,
    attach_steering_to_bundle,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    build_figure_style_prior,
    synthetic_einstein_corpus,
)


def _build_bundle(time_window_id: str | None = None):
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
    inputs = FigureBundleInputs(
        profile=profile,
        envelopes=envelope_set.envelopes,
        time_window_id=time_window_id,
    )
    return build_figure_artifact_bundle(inputs)


def test_style_prior_builds_top_words() -> None:
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
    style = build_figure_style_prior(
        figure_id="einstein", envelopes=envelope_set.envelopes
    )
    assert style.figure_id == "einstein"
    assert style.total_tokens > 0
    assert style.total_chunks > 0
    assert style.top_words
    assert style.term_list
    keys = {k for k, _ in style.sentence_length_percentiles}
    assert keys == {"p10", "p50", "p90"}


def test_style_prior_lookup_word_frequency() -> None:
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
    style = build_figure_style_prior(
        figure_id="einstein", envelopes=envelope_set.envelopes
    )
    assert style.lookup_word_frequency("the") >= 0.0
    assert style.lookup_word_frequency("zzznotpresent") == 0.0


def test_bundle_compiles_end_to_end() -> None:
    bundle = _build_bundle()
    assert bundle.schema_version == FIGURE_BUNDLE_SCHEMA_VERSION
    assert bundle.figure_id == "einstein"
    assert bundle.profile.profile_id == "einstein"
    assert bundle.domain_package.knowledge_records
    assert bundle.domain_package.case_records
    assert bundle.domain_package.playbook_rules
    assert bundle.domain_package.boundary_hints
    assert bundle.vitals_bootstrap.drives
    assert bundle.retrieval_index.total_chunks > 0
    assert bundle.coverage_map.domain_centroids
    assert bundle.style_prior.top_words
    assert bundle.steering is None
    assert bundle.lora is None
    assert bundle.integrity_hash
    assert bundle.bundle_id.startswith("figure-bundle:einstein:")
    assert bundle.version_window == (0, 0)


def test_bundle_rebuild_is_deterministic() -> None:
    a = _build_bundle()
    b = _build_bundle()
    assert a.integrity_hash == b.integrity_hash
    assert a.bundle_id == b.bundle_id


def test_bundle_with_time_window_yields_different_id() -> None:
    full = _build_bundle()
    early = _build_bundle(time_window_id="early-1905-1925")
    late = _build_bundle(time_window_id="late-1925-1955")
    assert full.integrity_hash != early.integrity_hash
    assert early.integrity_hash != late.integrity_hash
    assert full.bundle_id != early.bundle_id
    assert early.version_window == (1905, 1925)
    assert late.version_window == (1925, 1955)


def test_attach_steering_changes_bundle_id() -> None:
    bundle = _build_bundle()
    new = attach_steering_to_bundle(
        bundle, steering=("synthetic-steering",), steering_integrity="abc123"
    )
    assert new.bundle_id != bundle.bundle_id
    assert new.steering == ("synthetic-steering",)
    assert new.lora is None


def test_attach_lora_changes_bundle_id() -> None:
    bundle = _build_bundle()
    new = attach_lora_to_bundle(
        bundle, lora=("synthetic-lora",), lora_integrity="def456"
    )
    assert new.bundle_id != bundle.bundle_id
    assert new.lora == ("synthetic-lora",)
    assert new.steering is None


def test_bundle_rejects_mismatched_figure_id() -> None:
    bundle = _build_bundle()
    with pytest.raises(ValueError, match="figure_id"):
        type(bundle)(
            schema_version=bundle.schema_version,
            bundle_id=bundle.bundle_id,
            figure_id="not-einstein",
            profile_version=bundle.profile_version,
            version_window=bundle.version_window,
            profile=bundle.profile,
            domain_package=bundle.domain_package,
            vitals_bootstrap=bundle.vitals_bootstrap,
            retrieval_index=bundle.retrieval_index,
            coverage_map=bundle.coverage_map,
            style_prior=bundle.style_prior,
            steering=None,
            lora=None,
            integrity_hash=bundle.integrity_hash,
        )
