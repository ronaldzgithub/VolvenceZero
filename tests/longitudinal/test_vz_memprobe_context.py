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

import pytest

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


# NOTE — XFAIL with strict=True:
#
# The symmetric regime-disambiguation test below is expected-to-fail
# under the current MemoryStore retrieval implementation. The lexical
# / embedding score does not (yet) bias retrieval by
# ``RetrievalQuery.facets`` strongly enough to flip top-1 between two
# entries that share the same content keyword ("review") in different
# regime contexts. The stronger signal in current scoring is recency,
# so a single one-direction test would XPASS by accident; the
# two-direction symmetric test correctly XFAILS and captures the
# missing facet-driven disambiguation capability.
#
# This is a genuine Wave 3 evidence finding (CMA-2 "context probe"
# fails on the live retrieval path), not a setup bug. We keep the
# probe wired with clear assertions so the day the retrieval path
# starts honouring facets the test will go green and the new
# capability is detected automatically (``strict=True`` will then
# fail the suite, forcing us to remove the xfail).
#
# When it goes green, remove the xfail marker and update
# ``docs/known-debts.md`` to reflect the close-out (and add a
# closure note here citing the closing PR).
@pytest.mark.xfail(
    strict=True,
    reason=(
        "VZ-MemProbe Context: facet-driven regime disambiguation is "
        "not yet honoured by MemoryStore.retrieve scoring path. "
        "Tracked in known-debts as part of Wave 3 evidence run."
    ),
)
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
