"""Contract: ScopeRefuser + llm_synthesizer wire ``bundle.retrieval_index``
through to ``coverage_map.classify_query`` so the debt #39
retrieval-augmented floor actually fires at runtime.

Pre-fix state (before this contract was added):

* :func:`coverage_map.classify_query` accepted an optional
  ``retrieval_index`` kwarg and used it to lift in-corpus queries
  that missed every static centroid (the Wave K Einstein
  "equivalence principle / postulate / theory" false-refuse bug).
* :class:`ScopeRefuser.evaluate` called
  ``self._coverage_map.classify_query(query)`` **without** the
  kwarg, so the floor never fired at runtime.
* :func:`lifeform_expression.llm_synthesizer._evaluate_scope`
  constructed ``ScopeRefuser(coverage_map, config=...)`` **without**
  forwarding ``bundle.retrieval_index``, so even a bundle with a
  perfectly good retrieval index never got the floor benefit.

This module locks the wiring at three levels:

1. **Backward-compat** — a ScopeRefuser constructed without
   ``retrieval_index`` must keep calling ``classify_query(query)``
   byte-for-byte the same way, so duck-typed mocks / legacy callers
   with a narrow ``classify_query(self, query)`` signature stay
   working (debt #39 must not silently break the smaller surface).
2. **Forward wiring** — a ScopeRefuser constructed with a
   non-None ``retrieval_index`` must pass it through as a kwarg on
   every ``classify_query`` call.
3. **Synthesizer injection** — :func:`_evaluate_scope` must
   extract ``bundle.retrieval_index`` (via ``getattr``, since
   bundles without it are still valid) and inject it into the
   ScopeRefuser so the runtime ``synthesize()`` path benefits from
   the floor without any per-call ceremony.

Refs:

* docs/known-debts.md #39
* docs/specs/figure-persona-verification.md §refusal-precision
  ("ScopeRefuser 调用 classify_query 时显式传入 bundle.retrieval_index
  即可激活 floor")
* tests/contracts/test_figure_c1_coverage_and_refusal.py — locks
  the algorithm side of the same fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lifeform_domain_figure import (
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    synthetic_einstein_corpus,
)
from lifeform_domain_figure.compiler import FigureBundleInputs
from lifeform_domain_figure.coverage_map import (
    CoverageClassification,
    CoverageDecision,
)
from lifeform_expression.llm_synthesizer import _evaluate_scope
from lifeform_expression.scope_refuser import (
    CoveragePolicy,
    ScopeRefuser,
    ScopeRefuserConfig,
)


# ---------------------------------------------------------------------------
# Spy coverage_map / bundle stand-ins
# ---------------------------------------------------------------------------


@dataclass
class _SpyCoverageMap:
    """Records every classify_query call so the test can inspect kwargs.

    Always returns IN_DOMAIN so the test focuses purely on call
    shape, not on the directive translation logic (which is already
    covered by test_scope_refuser_smoke.py).

    Carries a ``figure_id`` attribute so :func:`dataclasses.replace`
    can swap this spy into a real :class:`FigureArtifactBundle` —
    the bundle's ``__post_init__`` validates
    ``coverage_map.figure_id == bundle.figure_id``.
    """

    figure_id: str = "einstein"
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def classify_query(self, query: str, **kwargs: object) -> CoverageClassification:
        self.calls.append((query, dict(kwargs)))
        return CoverageClassification(
            decision=CoverageDecision.IN_DOMAIN,
            closest_in_domain_label="spy",
            closest_in_domain_score=1.0,
            closest_out_of_scope_label="",
            closest_out_of_scope_score=0.0,
            rationale="spy: classify_query stub",
        )


@dataclass
class _SpyBundle:
    """Minimal bundle stand-in carrying just the two surfaces _evaluate_scope reads."""

    coverage_map: object
    retrieval_index: object | None


# ---------------------------------------------------------------------------
# Layer 1 — ScopeRefuser backward-compat: no retrieval_index → no kwarg
# ---------------------------------------------------------------------------


def test_scope_refuser_without_retrieval_index_passes_no_kwargs() -> None:
    """A ScopeRefuser built without retrieval_index must keep the legacy
    one-arg ``classify_query(query)`` call shape.

    This protects every duck-typed mock that defines
    ``classify_query(self, query)`` without a ``**kwargs`` sink —
    including ``_BadCoverageMap`` in test_scope_refuser_smoke.py —
    from breaking when debt #39 is wired in.
    """

    spy = _SpyCoverageMap()
    refuser = ScopeRefuser(spy)

    refuser.evaluate(query="anything")

    assert len(spy.calls) == 1, f"expected exactly one classify_query call, got {spy.calls}"
    query, kwargs = spy.calls[0]
    assert query == "anything"
    assert kwargs == {}, (
        f"backward-compat broken: classify_query was called with kwargs "
        f"{kwargs} when ScopeRefuser had no retrieval_index"
    )


# ---------------------------------------------------------------------------
# Layer 2 — ScopeRefuser forward wiring: retrieval_index reaches classify_query
# ---------------------------------------------------------------------------


def test_scope_refuser_with_retrieval_index_forwards_kwarg() -> None:
    """A ScopeRefuser built with a non-None retrieval_index must forward
    it as a kwarg on every classify_query call.

    The kwarg must be the exact object passed at construction time
    (identity, not just equality) — the coverage map needs to call
    ``retrieval_index.retrieve(...)`` on it, so anything weaker would
    be a hidden contract break.
    """

    spy = _SpyCoverageMap()
    sentinel = object()
    refuser = ScopeRefuser(spy, retrieval_index=sentinel)

    refuser.evaluate(query="equivalence principle")

    assert len(spy.calls) == 1
    query, kwargs = spy.calls[0]
    assert query == "equivalence principle"
    assert "retrieval_index" in kwargs, (
        f"forward wiring broken: retrieval_index kwarg missing from "
        f"classify_query kwargs {kwargs}"
    )
    assert kwargs["retrieval_index"] is sentinel, (
        "retrieval_index forwarded but not the same object instance; "
        "the coverage map's floor pass needs the actual index, not a copy"
    )


# ---------------------------------------------------------------------------
# Layer 3 — llm_synthesizer._evaluate_scope injection
# ---------------------------------------------------------------------------


def test_evaluate_scope_injects_bundle_retrieval_index() -> None:
    """When a bundle exposes ``retrieval_index``, ``_evaluate_scope``
    must thread it through to ``coverage_map.classify_query``.

    This is the runtime path used by ``synthesize()``; without this
    layer the algorithm-level fix in coverage_map.py and the
    constructor-level wiring in ScopeRefuser are both dead code as
    far as Patrick's demo is concerned.
    """

    spy = _SpyCoverageMap()
    sentinel_index = object()
    bundle = _SpyBundle(coverage_map=spy, retrieval_index=sentinel_index)

    directive = _evaluate_scope(bundle=bundle, query="equivalence principle")

    assert directive is not None, "spy bundle has coverage_map, directive must not be None"
    assert len(spy.calls) == 1
    _, kwargs = spy.calls[0]
    assert kwargs.get("retrieval_index") is sentinel_index, (
        f"_evaluate_scope failed to inject bundle.retrieval_index; "
        f"classify_query received kwargs={kwargs}"
    )


def test_evaluate_scope_omits_kwarg_when_bundle_has_no_retrieval_index() -> None:
    """When the bundle has no retrieval_index (or it is None),
    ``_evaluate_scope`` must NOT inject the kwarg, preserving the
    legacy call shape so older bundles / non-figure callers keep
    working unchanged.
    """

    spy = _SpyCoverageMap()
    bundle = _SpyBundle(coverage_map=spy, retrieval_index=None)

    directive = _evaluate_scope(bundle=bundle, query="anything")

    assert directive is not None
    assert len(spy.calls) == 1
    _, kwargs = spy.calls[0]
    assert "retrieval_index" not in kwargs, (
        f"_evaluate_scope leaked a retrieval_index kwarg when the "
        f"bundle had none; got kwargs={kwargs}"
    )


# ---------------------------------------------------------------------------
# Layer 4 — Real-bundle integration: verify the production data path
# (build_figure_artifact_bundle → bundle.coverage_map + bundle.retrieval_index
# → _evaluate_scope → ScopeRefuser → classify_query) actually threads the
# real bundle's retrieval_index through to the coverage map.
#
# We deliberately swap in a spy coverage_map via dataclasses.replace
# rather than asserting on the algorithm's IN_DOMAIN / OUT_OF_DOMAIN
# verdict, because:
#
# * The algorithm-level "floor lifts an OUT_OF_DOMAIN query" behaviour
#   is already locked by test_figure_c1_coverage_and_refusal.py
#   (test_floor_pass_lifts_an_otherwise_out_of_domain_query) — that
#   test calls coverage_map.classify_query directly with explicit
#   retrieval_floor=0.05 so it does not depend on corpus content
#   clearing the 0.30 production default.
# * Our concern in THIS file is wiring, not threshold tuning. Tying a
#   wiring contract to a specific corpus / threshold combination
#   would make this test flaky against unrelated corpus / threshold
#   changes.
# ---------------------------------------------------------------------------


def _build_real_einstein_bundle():
    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    corpus_bundle = FigureCorpusSourceBundle(
        figure_id="einstein",
        papers=papers,
        letters=letters,
        lectures=lectures,
        notebooks=notebooks,
    )
    envelope_set = build_figure_ingestion_envelope(corpus_bundle, uploader="test")
    return build_figure_artifact_bundle(
        FigureBundleInputs(
            profile=profile,
            envelopes=envelope_set.envelopes,
        )
    )


def test_real_bundle_ships_both_coverage_map_and_retrieval_index() -> None:
    """Structural invariant for the wiring to even be reachable.

    If a production Einstein bundle ever stopped shipping a
    retrieval_index, the runtime floor pass would be silently dead;
    locking this surface here prevents that regression in the bundle
    compiler.
    """

    bundle = _build_real_einstein_bundle()
    assert bundle.coverage_map is not None, (
        "production bundle missing coverage_map; ScopeRefuser has nothing to drive"
    )
    assert bundle.retrieval_index is not None, (
        "production bundle missing retrieval_index; debt #39 retrieval-augmented "
        "floor cannot fire and L4 will revert to centroid-only OUT_OF_DOMAIN "
        "false refusals on in-corpus paraphrases (Wave K Einstein bug)"
    )


def test_real_bundle_threads_retrieval_index_into_classify_query() -> None:
    """Production data path: a real bundle's retrieval_index reaches
    coverage_map.classify_query through the synthesize() entry point.

    This is the differential proof that debt #39 is fully wired:
    we swap the real bundle's coverage_map for a spy via
    dataclasses.replace, call _evaluate_scope as synthesize() does,
    and assert the spy received the bundle's own retrieval_index
    instance as a kwarg.
    """

    import dataclasses

    bundle = _build_real_einstein_bundle()
    spy = _SpyCoverageMap()
    bundle_with_spy = dataclasses.replace(bundle, coverage_map=spy)

    directive = _evaluate_scope(
        bundle=bundle_with_spy,
        query="any in-corpus relativity question",
    )

    assert directive is not None, (
        "spy returned IN_DOMAIN so _evaluate_scope must return a non-None directive"
    )
    assert len(spy.calls) == 1
    _, kwargs = spy.calls[0]
    assert kwargs.get("retrieval_index") is bundle.retrieval_index, (
        f"_evaluate_scope did not thread the real bundle's retrieval_index "
        f"through to coverage_map.classify_query; expected the bundle's own "
        f"retrieval_index instance, got {kwargs.get('retrieval_index')!r}"
    )
