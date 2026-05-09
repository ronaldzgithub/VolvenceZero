"""Smoke tests for the F3.1 GroundedDecoder (L3 enforce).

Validates against the figure vertical's real
:class:`lifeform_domain_figure.FigureRetrievalIndex` so the duck-
typed Protocol is exercised end-to-end:

* Substantive in-corpus assertions yield ``passed=True`` with
  evidence pointers.
* Assertions about off-corpus topics yield ``passed=False`` with
  the unsupported assertion captured verbatim.
* :meth:`GroundedDecoder.verify_strict` raises
  :class:`UngroundedAssertionError` on L3 failure (no swallow).
* Empty / whitespace text is treated as trivially passing (no
  substantive assertion present).
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    build_figure_ingestion_envelope,
    build_figure_retrieval_index,
    synthetic_einstein_corpus,
)
from lifeform_expression import (
    GroundedDecoder,
    GroundedDecoderConfig,
    UngroundedAssertionError,
)
from volvence_zero.substrate import GroundedDecodeHook, GroundingVerdict


def _build_decoder() -> GroundedDecoder:
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
    return GroundedDecoder(index)


def test_grounded_decoder_passes_for_in_corpus_assertion() -> None:
    decoder = _build_decoder()
    text = (
        "Spatially separated subsystems should each carry their own "
        "definite physical state, and the theory remains incomplete "
        "until that locally separable description is recovered."
    )
    verdict = decoder.verify(text=text)
    assert verdict.passed
    assert verdict.evidence_pointers
    assert "verified" in verdict.rationale.lower()


def test_grounded_decoder_fails_for_off_corpus_assertion() -> None:
    decoder = _build_decoder()
    text = (
        "Apricot jam recipes pair beautifully with sourdough breakfast "
        "spreads and chilled almond milk lattes."
    )
    verdict = decoder.verify(text=text)
    assert not verdict.passed
    assert verdict.unsupported_assertions


def test_grounded_decoder_strict_raises_on_unsupported() -> None:
    decoder = _build_decoder()
    text = (
        "Apricot jam recipes pair beautifully with sourdough breakfast "
        "spreads and chilled almond milk lattes."
    )
    with pytest.raises(UngroundedAssertionError):
        decoder.verify_strict(text=text)


def test_grounded_decoder_treats_empty_as_trivially_passing() -> None:
    decoder = _build_decoder()
    verdict_empty = decoder.verify(text="")
    verdict_blank = decoder.verify(text="   ")
    assert verdict_empty.passed
    assert verdict_blank.passed
    assert verdict_empty.unsupported_assertions == ()


def test_grounded_decoder_short_fragment_is_skipped() -> None:
    decoder = _build_decoder()
    # below default min_assertion_tokens of 4
    verdict = decoder.verify(text="Yes.")
    assert verdict.passed


def test_grounded_decoder_satisfies_substrate_protocol() -> None:
    decoder = _build_decoder()
    # Structural compatibility check: GroundedDecoder.verify must
    # accept ``text=...`` and return GroundingVerdict.
    result: GroundingVerdict = decoder.verify(text="test text not in corpus today.")
    assert isinstance(result, GroundingVerdict)
    # Static-type witness: a GroundedDecoder instance must satisfy
    # the runtime-checkable Protocol.
    assert isinstance(decoder, GroundedDecodeHook)


def test_grounded_decoder_config_validates() -> None:
    with pytest.raises(ValueError, match="min_assertion_tokens"):
        GroundedDecoderConfig(min_assertion_tokens=0)
    with pytest.raises(ValueError, match="top_k"):
        GroundedDecoderConfig(top_k=0)
