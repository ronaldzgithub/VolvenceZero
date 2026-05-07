"""VZ-MemProbe — Temporal Association probe (CMA-2 Phase 2 W2.3).

Failure mode under test: given an "anchor" event the user discussed
in turn 3, the agent should be able to recall **its temporal
neighbourhood** (turn 2 / turn 4) when asked "what else was happening
around that time" — not just the anchor itself.

This is the canonical CMA "Temporal Association" probe: stale-fact
retrieval often fails to capture the surrounding context that gives
the anchor its meaning. In high-EQ chat, this manifests as
"remembers the breakup but not the lead-up" — the agent's responses
feel disembodied because the temporal continuity has collapsed.

Setup uses ``memory_store.write(...)`` with deterministic
``timestamp_ms`` increments to simulate a 5-turn session. Setup tags
the anchor event explicitly; the probe queries by an anchor-related
query string and asserts at least one of the temporal neighbours
shows up in the top-K retrieval results alongside the anchor.

The probe is deterministic and does NOT call any LLM runtime.
"""

from __future__ import annotations

from volvence_zero.memory import build_default_memory_store
from volvence_zero.memory.contracts import (
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)


# A small narrative the user might tell across 5 turns.
# Anchor = turn 3 (the breakup announcement).
_TURN_TIMELINE: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1_000, "User had coffee at the cafe and felt anxious before meeting Alex.", ("morning", "anxious", "cafe", "alex")),
    (2_000, "Alex arrived late looking tense; the conversation became cold.", ("alex", "tension", "context")),
    (3_000, "Alex told the user the relationship is ending; user is heartbroken.", ("alex", "breakup", "anchor", "heartbreak")),
    (4_000, "User cried in the cafe bathroom for ten minutes after Alex left.", ("alex", "aftermath", "tears", "cafe")),
    (5_000, "User went home, called mom, ate ice cream alone.", ("home", "mom", "ice_cream", "self_soothing")),
)


def _seed_timeline() -> object:
    store = build_default_memory_store()
    for ts_ms, content, tags in _TURN_TIMELINE:
        store.write(
            request=MemoryWriteRequest(
                content=content,
                track=Track.SELF,
                stratum=MemoryStratum.EPISODIC,
                strength=0.7,
                tags=tags,
            ),
            timestamp_ms=ts_ms,
        )
    return store


def test_mp_temporal_anchor_recalls_neighbourhood() -> None:
    """``mp.temporal.neighbour_recall`` — anchor query must surface at least
    one immediate neighbour (turn 2 or turn 4) in addition to the anchor.

    We query for the anchor topic ("Alex breakup") in a separate
    session (simulated by querying the shared store with a later
    timestamp). Top-K=3 retrieval MUST include the anchor itself
    (sanity check) and at least one of its turn-2 / turn-4
    neighbours.
    """
    store = _seed_timeline()

    result = store.retrieve(
        RetrievalQuery(
            text="alex breakup",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=3,
        ),
        timestamp_ms=10_000,
    )
    assert result.entries, "anchor query returned no entries; tag indexing broken"

    # Identify each retrieved entry by its anchor relationship.
    contents = tuple(entry.content for entry in result.entries)
    contains_anchor = any("relationship is ending" in c for c in contents)
    contains_lead_up = any("Alex arrived late" in c for c in contents)
    contains_aftermath = any("cried in the cafe bathroom" in c for c in contents)

    # Anchor itself should always retrieve (sanity).
    assert contains_anchor, (
        "mp.temporal.anchor_self_recall FAILED: anchor entry not in "
        f"top-3 retrieval. Got: {contents!r}"
    )
    # At least ONE temporal neighbour must come along.
    assert contains_lead_up or contains_aftermath, (
        "mp.temporal.neighbour_recall FAILED: top-3 retrieval is "
        "anchor-only, no temporal neighbourhood (turn 2 'tension' or "
        "turn 4 'aftermath') surfaced. This is the disembodied-memory "
        f"failure mode CMA-2 detects. Got: {contents!r}"
    )


def test_mp_temporal_unrelated_window_does_not_pollute_anchor_query() -> None:
    """Negative control: turn-1 (cafe morning, before Alex arrives) and
    turn-5 (home with mom) are temporally adjacent but **topically**
    unrelated to the breakup anchor. Top-1 retrieval for "alex breakup"
    must NOT be one of these temporally-adjacent-but-topic-unrelated
    entries.

    This pins that the temporal neighbour recall in the positive case
    above comes from genuine topical coupling (lead-up / aftermath are
    *about* Alex), not from a windowed bag-of-time-stamps where any
    entry near the anchor would surface.
    """
    store = _seed_timeline()

    result = store.retrieve(
        RetrievalQuery(
            text="alex breakup",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=1,
        ),
        timestamp_ms=10_000,
    )
    assert result.entries
    top_content = result.entries[0].content
    # Top-1 must not be the temporally-adjacent-but-unrelated turn-5
    # ("ice cream") nor turn-1 ("morning coffee anxiety alone").
    assert "ice cream" not in top_content, (
        "mp.temporal negative-control FAILED: top-1 is the "
        "topically-unrelated adjacent turn (turn 5)."
    )
    assert "had coffee at the cafe and felt anxious before meeting" not in top_content, (
        "mp.temporal negative-control FAILED: top-1 is the "
        "topically-unrelated adjacent turn (turn 1)."
    )
