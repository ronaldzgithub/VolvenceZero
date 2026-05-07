"""VZ-MemProbe — Associative Recall probe (CMA-2 Phase 2 W2.4).

Failure mode under test: a 3-hop reasoning chain across the 9 typed
semantic owners must remain traversable from one end. Specifically:

* Hop 1: user_model — "User is recovering from burnout."
* Hop 2: relationship_state — "Relationship_state shows the user is
  currently asking for slower response cadence due to that burnout."
* Hop 3: boundary_consent — "User has explicitly granted consent for
  late-night disengagement when their burnout pressure is high."

A query that mentions only the proximal cue ("burnout") should be
able to surface entries on all three hops; in particular, the
``boundary_consent.granted`` end of the chain must be retrievable —
that's the operationally important fact at high-EQ time (the agent
needs to know it has explicit permission to slow down).

This is the canonical CMA "Associative Recall" probe. Failure mode in
chat: the agent remembers the user's burnout (hop 1) but not the
boundary contract (hop 3), so it answers questions about pacing as
if the consent didn't exist.

Setup uses ``memory_store.write(...)`` with explicit tag chains. The
9 semantic owners are simulated via tags: each entry's ``tags``
includes both its owner-name tag (``owner:user_model`` etc.) and the
shared chain key tag (``chain:burnout_pacing_consent``). Probe is
deterministic and does NOT call any LLM runtime.
"""

from __future__ import annotations

from volvence_zero.memory import build_default_memory_store
from volvence_zero.memory.contracts import (
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)


_CHAIN_TAG = "chain:burnout_pacing_consent"


def _seed_three_hop_chain() -> object:
    store = build_default_memory_store()

    # Hop 1: user_model
    store.write(
        request=MemoryWriteRequest(
            content="User is recovering from burnout from a high-stress job.",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=("owner:user_model", _CHAIN_TAG, "burnout", "user_model"),
        ),
        timestamp_ms=1_000,
    )
    # Hop 2: relationship_state
    store.write(
        request=MemoryWriteRequest(
            content=(
                "Relationship_state: user explicitly asked for slower response "
                "cadence given the burnout. Pacing is part of the contract."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=(
                "owner:relationship_state",
                _CHAIN_TAG,
                "burnout",
                "pacing",
                "relationship_state",
            ),
        ),
        timestamp_ms=2_000,
    )
    # Hop 3: boundary_consent (the operationally critical fact)
    store.write(
        request=MemoryWriteRequest(
            content=(
                "Boundary_consent.granted=True for late-night disengagement "
                "when burnout pressure is high. User issued this consent "
                "explicitly."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.9,
            tags=(
                "owner:boundary_consent",
                _CHAIN_TAG,
                "burnout",
                "boundary_consent",
                "granted",
            ),
        ),
        timestamp_ms=3_000,
    )
    # Distractor: unrelated boundary_consent for a different topic
    store.write(
        request=MemoryWriteRequest(
            content=(
                "Boundary_consent.granted=True for sharing technical jargon. "
                "Unrelated to burnout."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.7,
            tags=(
                "owner:boundary_consent",
                "chain:tech_jargon",
                "boundary_consent",
                "granted",
                "tech",
            ),
        ),
        timestamp_ms=4_000,
    )
    return store


def test_mp_assoc_three_hop_full_chain_recallable() -> None:
    """``mp.assoc.chain_complete`` — querying the proximal cue
    ("burnout") must surface all three chain hops in the top-K.
    """
    store = _seed_three_hop_chain()
    result = store.retrieve(
        RetrievalQuery(
            text="burnout pacing",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=5,
        ),
        timestamp_ms=10_000,
    )
    assert result.entries, "associative chain query returned nothing"

    owner_tags_seen = set()
    for entry in result.entries:
        for tag in entry.tags:
            if tag.startswith("owner:"):
                owner_tags_seen.add(tag)

    expected_owners = {
        "owner:user_model",
        "owner:relationship_state",
        "owner:boundary_consent",
    }
    assert expected_owners.issubset(owner_tags_seen), (
        "mp.assoc.chain_complete FAILED: top-5 retrieval did not "
        f"surface all three chain hops. Expected {expected_owners}, "
        f"got {owner_tags_seen}. Missing hops break high-EQ pacing "
        "decisions because the agent loses sight of the boundary "
        "contract that the user explicitly granted."
    )


def test_mp_assoc_three_hop_distal_consent_retrievable() -> None:
    """``mp.assoc.distal_consent_recall`` — the chain-3 end
    (``boundary_consent.granted``) must be retrievable when querying
    by the chain-1 cue, AND the matching entry must be the
    burnout-related boundary_consent rather than the unrelated
    tech-jargon one.
    """
    store = _seed_three_hop_chain()
    result = store.retrieve(
        RetrievalQuery(
            text="burnout boundary consent",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=5,
        ),
        timestamp_ms=10_000,
    )
    boundary_entries = tuple(
        entry for entry in result.entries
        if "owner:boundary_consent" in entry.tags
    )
    assert boundary_entries, (
        "mp.assoc.distal_consent_recall FAILED: no boundary_consent "
        "entries retrieved when querying by burnout cue. The "
        "operationally critical fact (consent exists) is unreachable "
        "from the proximal narrative cue."
    )
    top_boundary = boundary_entries[0]
    assert _CHAIN_TAG in top_boundary.tags, (
        "mp.assoc.distal_consent_recall FAILED: top boundary_consent "
        "result is from a DIFFERENT chain (likely the tech-jargon "
        f"distractor). Got tags={top_boundary.tags!r}. The chain "
        "binding has leaked across topics."
    )
