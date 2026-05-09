"""Smoke tests for the F3.2 ScopeRefuser (L4 enforce).

Validates against the real
:class:`lifeform_domain_figure.FigureCoverageMap`:

* In-domain queries pass through with no disclaimers and no refusal.
* Boundary-blocked queries refuse outright under STRICT_REFUSE,
  carry a disclaimer under SOFT_DISCLAIM, and fall through under
  PASSTHROUGH.
* Out-of-domain queries follow the same three-arm policy with the
  matching reviewer-curated text.
* Unexpected classification shapes raise loudly (no silent
  fallback).
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_coverage_map,
    build_figure_ingestion_envelope,
    build_figure_retrieval_index,
    synthetic_einstein_corpus,
)
from lifeform_expression import (
    CoveragePolicy,
    ScopeRefuser,
    ScopeRefuserConfig,
)


def _build_refuser(policy: CoveragePolicy) -> ScopeRefuser:
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
    coverage_map = build_figure_coverage_map(
        figure_id="einstein", profile=profile, retrieval_index=index
    )
    return ScopeRefuser(coverage_map, config=ScopeRefuserConfig(policy=policy))


def test_in_domain_passes_through() -> None:
    refuser = _build_refuser(CoveragePolicy.STRICT_REFUSE)
    directive = refuser.evaluate(
        query=(
            "What did the author argue about the locality of physical states "
            "in his foundational paper on mechanics?"
        )
    )
    assert directive.should_refuse is False
    assert directive.refusal_text == ""
    assert directive.disclaimers == ()
    assert directive.coverage_decision == "in_domain"


def test_boundary_blocked_strict_refuses() -> None:
    refuser = _build_refuser(CoveragePolicy.STRICT_REFUSE)
    directive = refuser.evaluate(
        query="What is the contemporary AI policy stance on geopolitical events?"
    )
    assert directive.should_refuse is True
    assert directive.refusal_text
    assert directive.coverage_decision == "boundary_blocked"


def test_boundary_blocked_soft_disclaims() -> None:
    refuser = _build_refuser(CoveragePolicy.SOFT_DISCLAIM)
    directive = refuser.evaluate(
        query="What is the contemporary AI policy stance on geopolitical events?"
    )
    assert directive.should_refuse is False
    assert directive.disclaimers
    assert directive.coverage_decision == "boundary_blocked"


def test_boundary_blocked_passthrough() -> None:
    refuser = _build_refuser(CoveragePolicy.PASSTHROUGH)
    directive = refuser.evaluate(
        query="What is the contemporary AI policy stance on geopolitical events?"
    )
    assert directive.should_refuse is False
    assert directive.disclaimers == ()
    assert directive.coverage_decision == "boundary_blocked"


def test_out_of_domain_strict_refuses() -> None:
    refuser = _build_refuser(CoveragePolicy.STRICT_REFUSE)
    directive = refuser.evaluate(
        query="What is the best apricot jam recipe for a sourdough breakfast?"
    )
    assert directive.should_refuse is True
    assert directive.coverage_decision == "out_of_domain"


def test_out_of_domain_soft_disclaims() -> None:
    refuser = _build_refuser(CoveragePolicy.SOFT_DISCLAIM)
    directive = refuser.evaluate(
        query="What is the best apricot jam recipe for a sourdough breakfast?"
    )
    assert directive.should_refuse is False
    assert directive.disclaimers
    assert directive.coverage_decision == "out_of_domain"


def test_unexpected_classification_shape_raises() -> None:
    class _BadCoverageMap:
        def classify_query(self, query: str):
            class _R:
                decision = 12345  # neither enum nor str
                rationale = ""

            return _R()

    refuser = ScopeRefuser(_BadCoverageMap())
    with pytest.raises(TypeError, match="classification.decision"):
        refuser.evaluate(query="anything")
