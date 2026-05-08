"""VZ-MemProbe — Disambiguation / Context probe (CMA-2 Phase 2 W2.5).

Failure mode under test: same keyword carries different meanings under
different regime contexts. The agent must disambiguate by the active
regime when retrieving, NOT by the keyword alone — otherwise
problem-solving regime memories pollute casual_social regime
retrieval and vice versa.

Concrete narrative: the user uses the keyword "review" twice across
two sessions:

* Session 1 (regime=problem_solving): "I'm reviewing a colleague's
  pull request and the code feels off." — review = engineering
  artifact review.
* Session 2 (regime=casual_social): "We did a nice restaurant review
  with friends, the new ramen place is great." — review = casual
  social activity.

When the agent later asks about "review" in problem_solving regime,
top-1 must be the PR-review entry. When asked in casual_social regime,
top-1 must be the restaurant-review entry. Cross-context pollution
breaks the regime contract — the agent ends up replying with food
recommendations to a code question or vice versa.

Setup uses ``memory_store.write(...)`` with regime-aware tags, and the
probe queries using ``RetrievalQuery.facets`` (the same channel
``run_final_wiring_turn`` uses to inject runtime regime / dual-track
facets). Probe is deterministic and does NOT call any LLM runtime.
"""

from __future__ import annotations

from volvence_zero.memory import build_default_memory_store
from volvence_zero.memory.contracts import (
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)


def _seed_two_regime_meanings() -> object:
    store = build_default_memory_store()
    # Session 1: problem_solving regime — "review" means PR review.
    store.write(
        request=MemoryWriteRequest(
            content=(
                "User is reviewing a colleague pull request; the code "
                "structure feels off and they want help diagnosing why."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
            tags=(
                "regime:problem_solving",
                "review",
                "pull_request",
                "engineering",
            ),
        ),
        timestamp_ms=1_000,
    )
    # Session 2: casual_social regime — "review" means restaurant review.
    store.write(
        request=MemoryWriteRequest(
            content=(
                "User shared a casual restaurant review with friends; "
                "the new ramen place is great and they want to go again."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
            tags=(
                "regime:casual_social",
                "review",
                "restaurant",
                "ramen",
            ),
        ),
        timestamp_ms=2_000,
    )
    return store


# Phase 2 W4 (debt #10D close-out 2026-05-09):
# Previously this test was ``xfail(strict=True)`` because
# ``MemoryStore._score_entry`` had no explicit facet-match boost; the
# embedding-only path was too weak to disambiguate two entries that
# share a surface keyword in different ``regime:*`` contexts, so
# recency dominated and the symmetric test (forcing both directions
# to flip) correctly failed.
#
# debt #10D fix: ``_score_entry`` now adds ``facet_score = matched
# * 5.0`` for each ``query.facets`` element that matches an entry
# tag. This sits above the lexical band but below dominant semantic
# / learned channels, giving regime facets a real tie-breaker
# without overriding genuine content signal. The test is now expected
# to PASS in both directions.
def test_mp_context_regime_facet_disambiguates_both_directions() -> None:
    """``mp.context.regime_match_symmetric`` — facet MUST flip top-1.

    Two queries share the SAME text ("user review") but different
    regime facets. To prove the system genuinely uses the facet
    rather than a recency / strength tiebreak that happens to match
    one direction by accident, we assert BOTH:

    * ``regime:problem_solving`` -> top-1 is the PR-review entry
    * ``regime:casual_social`` -> top-1 is the restaurant-review entry

    Either direction failing is a fail.
    """
    store = _seed_two_regime_meanings()

    eng = store.retrieve(
        RetrievalQuery(
            text="user review",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=2,
            facets=("regime:problem_solving",),
        ),
        timestamp_ms=10_000,
    )
    soc = store.retrieve(
        RetrievalQuery(
            text="user review",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=2,
            facets=("regime:casual_social",),
        ),
        timestamp_ms=10_000,
    )
    assert eng.entries and soc.entries
    eng_top_is_pr = "pull_request" in eng.entries[0].tags
    soc_top_is_restaurant = "restaurant" in soc.entries[0].tags
    assert eng_top_is_pr and soc_top_is_restaurant, (
        "mp.context.regime_match_symmetric FAILED: facet did not "
        "flip top-1 between regimes. "
        f"problem_solving top-1 tags={eng.entries[0].tags!r} "
        f"casual_social top-1 tags={soc.entries[0].tags!r}. "
        "Cross-context pollution detected."
    )


def test_mp_context_no_facet_does_not_collapse_to_one_meaning() -> None:
    """``mp.context.no_facet_returns_both`` — when no regime facet is
    provided, BOTH entries should appear in top-K (the system has no
    way to disambiguate). This pins that the disambiguation seen in
    the two positive cases above is genuinely driven by the facet,
    not by the entries having different content scores by accident.
    """
    store = _seed_two_regime_meanings()
    result = store.retrieve(
        RetrievalQuery(
            text="user review",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=2,
            # facets intentionally empty
        ),
        timestamp_ms=10_000,
    )
    assert len(result.entries) == 2, (
        "mp.context.no_facet_returns_both FAILED: facet-less query "
        f"surfaced {len(result.entries)} entries instead of 2. "
        "Without a facet, the system must not silently pick one "
        "meaning."
    )
    tags_seen = set()
    for entry in result.entries:
        tags_seen.update(entry.tags)
    assert "pull_request" in tags_seen
    assert "restaurant" in tags_seen
