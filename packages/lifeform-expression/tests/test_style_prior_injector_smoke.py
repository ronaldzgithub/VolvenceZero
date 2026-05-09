"""Smoke tests for the F3.3 StylePriorInjector + synthesizer integration.

Validates:

* :class:`StylePriorInjector` produces a non-empty logit-bias dict
  via a duck-typed encode adapter.
* :meth:`render_style_hint_text` returns a multi-line hint that
  references the figure id and at least one top term.
* :class:`StyleHint` carries every relevant field.
* ``LifeformLLMResponseSynthesizer.with_figure_bundle`` clones the
  synthesizer with the bundle attached.
* ``clone_for_session`` propagates the figure bundle by reference
  (so per-session clones see the same frozen bundle).
"""

from __future__ import annotations

from typing import Any

import pytest

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    synthetic_einstein_corpus,
)
from lifeform_expression import (
    StyleHint,
    StylePriorInjector,
    StylePriorInjectorConfig,
)
from lifeform_expression.llm_synthesizer import LifeformLLMResponseSynthesizer


def _build_bundle():
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
    )
    return build_figure_artifact_bundle(inputs)


def test_style_prior_injector_logit_bias_uses_encoder() -> None:
    artifact_bundle = _build_bundle()
    injector = StylePriorInjector(artifact_bundle.style_prior)
    counter = {"calls": 0}

    def encoder(token: str) -> tuple[int, ...]:
        counter["calls"] += 1
        return (hash(token) & 0xFFFF,)

    biases = injector.compute_logit_bias(encode_token=encoder)
    assert biases
    assert counter["calls"] > 0
    cap = injector.config.logit_bias_cap
    assert all(0.0 <= bias <= cap for bias in biases.values())


def test_style_prior_injector_renders_style_hint_text() -> None:
    artifact_bundle = _build_bundle()
    injector = StylePriorInjector(artifact_bundle.style_prior)
    hint = injector.render_style_hint_text()
    assert "einstein" in hint
    assert "voice prior" in hint.lower()
    assert "median sentence length" in hint.lower()


def test_style_hint_returns_structured_payload() -> None:
    artifact_bundle = _build_bundle()
    injector = StylePriorInjector(artifact_bundle.style_prior)
    hint = injector.style_hint()
    assert isinstance(hint, StyleHint)
    assert hint.figure_id == "einstein"
    assert hint.voice_shape_text
    assert hint.top_term_examples
    assert hint.sentence_length_targets


def test_injector_skips_tokens_with_empty_encoder_output() -> None:
    artifact_bundle = _build_bundle()
    injector = StylePriorInjector(artifact_bundle.style_prior)

    def empty_encoder(_: str) -> tuple[int, ...]:
        return ()

    biases = injector.compute_logit_bias(encode_token=empty_encoder)
    assert biases == {}


def test_injector_config_validates() -> None:
    with pytest.raises(ValueError, match="top_words_for_bias"):
        StylePriorInjectorConfig(top_words_for_bias=0)
    with pytest.raises(ValueError, match="logit_bias_cap"):
        StylePriorInjectorConfig(logit_bias_cap=0.0)


def test_synthesizer_attaches_figure_bundle_by_reference() -> None:
    artifact_bundle = _build_bundle()

    class _NullRuntime:
        # The base LLMResponseSynthesizer accepts any object as runtime;
        # we never actually call it in this test (we only exercise
        # clone / attach paths).
        pass

    synthesizer = LifeformLLMResponseSynthesizer(runtime=_NullRuntime())
    assert synthesizer.figure_bundle is None

    bound = synthesizer.with_figure_bundle(artifact_bundle)
    assert bound.figure_bundle is artifact_bundle
    # The original synthesizer is not mutated.
    assert synthesizer.figure_bundle is None

    cloned = bound.clone_for_session()
    # Bundle is shared by reference (it's frozen, so this is safe).
    assert cloned.figure_bundle is artifact_bundle


def test_synthesizer_with_figure_bundle_none_clears_binding() -> None:
    artifact_bundle = _build_bundle()

    class _NullRuntime:
        pass

    bound = LifeformLLMResponseSynthesizer(
        runtime=_NullRuntime(), figure_bundle=artifact_bundle
    )
    cleared = bound.with_figure_bundle(None)
    assert cleared.figure_bundle is None
    # Original instance is preserved.
    assert bound.figure_bundle is artifact_bundle
