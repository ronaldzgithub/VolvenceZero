"""VZ-MemProbe — Knowledge Updates probe (CMA-2 Phase 2 W2.2).

Failure mode under test: a later belief that **overrides** an earlier
belief on the same topic should win the retrieval rank when queried.
This is the "Knowledge Updates" probe from Logan 2026 (CMA paper):
the canonical long-horizon LLM agent failure mode where the agent
keeps surfacing stale facts after they have been corrected.

Setup uses the deterministic ``memory_store.write(...)`` path
documented in ``docs/specs/evaluation.md`` -> "Long-Horizon Memory
Probes". Setup writes simulate what session-post slow loop would have
consolidated; the probe exercises retrieval-side ranking via the
public ``retrieve(...)`` API and asserts the override wins.

The probe runs across two **simulated sessions** with a shared
``MemoryStore`` so the cross-session evidence is real (debt #10A
closure path). It does NOT call any LLM runtime — assertion is
fully deterministic on retrieval rank.
"""

from __future__ import annotations

from volvence_zero.memory import MemoryStore, build_default_memory_store
from volvence_zero.memory.contracts import (
    MemoryEntry,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)


_QUERY_TEXT = "user preferred beverage habit"


def _write_simulated_belief(
    store: MemoryStore,
    *,
    content: str,
    timestamp_ms: int,
    strength: float,
    tags: tuple[str, ...],
) -> MemoryEntry:
    return store.write(
        request=MemoryWriteRequest(
            content=content,
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=strength,
            tags=tags,
        ),
        timestamp_ms=timestamp_ms,
    )


def test_mp_update_override_wins_retrieval_rank() -> None:
    """``mp.update.top1_is_override`` — later belief beats earlier on same topic.

    Session 1: user said "I love coffee". Session 2 (months later in
    the simulated timeline): user said "actually I prefer tea now,
    coffee gives me anxiety". Top-1 retrieval for the
    preferred-beverage query MUST surface tea, not coffee.

    Strength A is set HIGHER than strength B so the test does not
    pass for the trivial reason of "newer-and-stronger". Override
    wins must come from the (recency * strength) ranking, where
    strength alone is insufficient.
    """
    store = build_default_memory_store()

    # Session 1 simulated consolidation
    entry_a = _write_simulated_belief(
        store,
        content="User loves coffee, drinks two cups a day.",
        timestamp_ms=1_000,
        strength=0.85,
        tags=("user_preference", "beverage", "coffee"),
    )
    # Session 2 simulated consolidation: override belief
    entry_b = _write_simulated_belief(
        store,
        content="User prefers tea now; coffee gives anxiety.",
        timestamp_ms=10_000,
        strength=0.70,
        tags=("user_preference", "beverage", "tea", "override"),
    )

    result = store.retrieve(
        RetrievalQuery(
            text=_QUERY_TEXT,
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=5,
        ),
        timestamp_ms=11_000,
    )
    assert result.entries, "retrieval returned no entries; both writes unreachable"
    top = result.entries[0]
    assert top.entry_id == entry_b.entry_id, (
        "mp.update.top1_is_override FAILED: expected override belief "
        f"({entry_b.content!r}) at top-1, got {top.content!r}. "
        "Stale-fact dominance is the canonical CMA failure mode "
        "this probe exists to detect."
    )
    # Both entries should still be reachable (retrieval is recency-
    # ranked, not destructive); the override just wins the rank.
    entry_ids = {entry.entry_id for entry in result.entries}
    assert entry_a.entry_id in entry_ids


def test_mp_update_no_override_keeps_original_rank() -> None:
    """Negative case: when no override happens, the original belief
    remains the top-1. This pins that the override-wins behaviour
    is *driven* by the override write, not a side effect of retrieval
    being timestamp-only.
    """
    store = build_default_memory_store()

    entry_a = _write_simulated_belief(
        store,
        content="User loves coffee, drinks two cups a day.",
        timestamp_ms=1_000,
        strength=0.85,
        tags=("user_preference", "beverage", "coffee"),
    )
    # Unrelated later belief — does not override
    _write_simulated_belief(
        store,
        content="User likes hiking on weekends.",
        timestamp_ms=10_000,
        strength=0.85,
        tags=("user_preference", "leisure", "hiking"),
    )

    result = store.retrieve(
        RetrievalQuery(
            text=_QUERY_TEXT,
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=5,
        ),
        timestamp_ms=11_000,
    )
    assert result.entries
    assert result.entries[0].entry_id == entry_a.entry_id, (
        "mp.update negative-control failed: unrelated later write "
        "won the rank, suggesting retrieval is dominated by recency "
        "and the positive case may pass for the wrong reason."
    )
