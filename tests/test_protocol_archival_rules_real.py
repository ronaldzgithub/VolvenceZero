"""Packet 9.2: real propose_knowledge_archival / propose_case_archival.

Asserts that with lineage maps supplied:

* No history → no proposal.
* Active protocol with seed never appearing in N turns → ARCHIVE proposal.
* Active protocol with seed appearing in window → no proposal.
* Inactive protocol (weight=0) gets no false positive even if seed unseen.
* Empty lineage map → no proposal.
"""

from __future__ import annotations

from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    ProtocolRevisionChangeKind,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.reflection import (
    KNOWLEDGE_ARCHIVAL_MIN_TURNS,
    propose_case_archival,
    propose_knowledge_archival,
)


def _mixture(*entries: tuple[str, float]) -> ActiveMixtureSnapshot:
    return ActiveMixtureSnapshot(
        active_protocols=tuple(
            ActiveProtocolEntry(protocol_id=pid, activation_weight=w)
            for pid, w in entries
        ),
        boundary_union_ids=(),
        revision_fingerprint="",
        description="",
    )


# ---------------------------------------------------------------------------
# Knowledge archival
# ---------------------------------------------------------------------------


def test_knowledge_archival_no_lineage_no_proposal() -> None:
    proposals = propose_knowledge_archival(
        pe_history=(),
        active_mixture_history=(),
        knowledge_hit_history=(),
        case_hit_history=(),
    )
    assert proposals == ()


def test_knowledge_archival_insufficient_history_no_proposal() -> None:
    n = KNOWLEDGE_ARCHIVAL_MIN_TURNS - 2
    am = tuple(_mixture(("p1", 0.8)) for _ in range(n))
    hits = tuple(() for _ in range(n))
    proposals = propose_knowledge_archival(
        pe_history=(),
        active_mixture_history=am,
        knowledge_hit_history=hits,
        knowledge_lineage_by_protocol={
            "p1": ("protocol:p1:knowledge:seed-x",),
        },
    )
    assert proposals == ()


def test_knowledge_archival_emits_for_unretrieved_seed() -> None:
    n = KNOWLEDGE_ARCHIVAL_MIN_TURNS + 2
    am = tuple(_mixture(("p1", 0.8)) for _ in range(n))
    # seed-y appears every turn; seed-x never appears.
    hits = tuple(
        ("protocol:p1:knowledge:seed-y",) for _ in range(n)
    )
    proposals = propose_knowledge_archival(
        pe_history=(),
        active_mixture_history=am,
        knowledge_hit_history=hits,
        knowledge_lineage_by_protocol={
            "p1": (
                "protocol:p1:knowledge:seed-x",
                "protocol:p1:knowledge:seed-y",
            ),
        },
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.target_protocol_id == "p1"
    assert p.target_entry_id == "seed-x"
    assert p.change_kind is ProtocolRevisionChangeKind.ARCHIVE
    assert p.target_field is ProtocolRevisionTargetField.KNOWLEDGE_SEED
    assert p.required_review_level is ReviewLevel.L2


def test_knowledge_archival_does_not_fire_when_seed_appears_at_least_once() -> None:
    n = KNOWLEDGE_ARCHIVAL_MIN_TURNS + 2
    am = tuple(_mixture(("p1", 0.8)) for _ in range(n))
    # seed-x appears once mid-window.
    hits = list(() for _ in range(n))
    hits[3] = ("protocol:p1:knowledge:seed-x",)
    proposals = propose_knowledge_archival(
        pe_history=(),
        active_mixture_history=am,
        knowledge_hit_history=tuple(hits),
        knowledge_lineage_by_protocol={
            "p1": ("protocol:p1:knowledge:seed-x",),
        },
    )
    assert proposals == ()


def test_knowledge_archival_skips_inactive_protocols() -> None:
    """Protocol with weight=0 every turn → no archival even if unseen."""
    n = KNOWLEDGE_ARCHIVAL_MIN_TURNS + 2
    am = tuple(_mixture(("p1", 0.0)) for _ in range(n))
    hits = tuple(() for _ in range(n))
    proposals = propose_knowledge_archival(
        pe_history=(),
        active_mixture_history=am,
        knowledge_hit_history=hits,
        knowledge_lineage_by_protocol={
            "p1": ("protocol:p1:knowledge:seed-x",),
        },
    )
    assert proposals == ()


# ---------------------------------------------------------------------------
# Case archival (mirror)
# ---------------------------------------------------------------------------


def test_case_archival_emits_for_unretrieved_case() -> None:
    n = KNOWLEDGE_ARCHIVAL_MIN_TURNS + 2
    am = tuple(_mixture(("p1", 0.8)) for _ in range(n))
    hits = tuple(() for _ in range(n))
    proposals = propose_case_archival(
        pe_history=(),
        active_mixture_history=am,
        case_hit_history=hits,
        case_lineage_by_protocol={
            "p1": ("protocol:p1:case:case-z",),
        },
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.target_entry_id == "case-z"
    assert p.target_field is ProtocolRevisionTargetField.SIGNATURE_CASE
    assert p.change_kind is ProtocolRevisionChangeKind.ARCHIVE
