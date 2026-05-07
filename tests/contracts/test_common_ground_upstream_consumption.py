"""Phase 1 W1.E contract test: CommonGroundModule consumes its
typed upstream dependencies (``conversational_role`` +
``belief_about_other``).

Before W1.E the owner declared four dependencies but immediately
``del upstream``-ed them, so the only path to atoms was an explicit
``proposal_runtime`` injection. This test pins the new typed
consumption: when ``conversational_role`` shows a confident dyad
context AND ``belief_about_other`` carries high-confidence records,
the owner derives one dyad ``CommonGroundAtom`` per record without
touching the ``proposal_runtime`` path.
"""

from __future__ import annotations

import asyncio

from volvence_zero.runtime import (
    Snapshot,
    WiringLevel,
    propagate,
)
from volvence_zero.runtime.kernel import (
    EventRecorder,
    DependencyGuard,
    ImmutabilityGuard,
)
from volvence_zero.social import CommonGroundModule
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    ConversationalRoleSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    SocialScopeKind,
)


def _belief_snapshot(records: tuple[OtherMindRecord, ...]) -> BeliefAboutOtherSnapshot:
    return BeliefAboutOtherSnapshot(
        records=records,
        active_predictions=(),
        control_signal=0.0,
        description="test belief snapshot",
    )


def _role_snapshot(*, role_confidence: float = 0.85) -> ConversationalRoleSnapshot:
    return ConversationalRoleSnapshot(
        active_speaker_id=PRIMARY_INTERLOCUTOR_ID,
        addressee_ids=(SELF_INTERLOCUTOR_ID,),
        subject_ids=(PRIMARY_INTERLOCUTOR_ID,),
        witness_ids=(),
        overhearer_ids=(),
        group_audience_ids=(),
        role_confidence=role_confidence,
        active_predictions=(),
        description="test role snapshot",
    )


def _belief_record(
    *,
    record_id: str,
    summary: str,
    confidence: float,
    evidence: str = "test evidence",
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=record_id,
        interlocutor_id=PRIMARY_INTERLOCUTOR_ID,
        kind=OtherMindRecordKind.BELIEF,
        summary=summary,
        detail="test detail",
        confidence=confidence,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=1,
        prediction_error_refs=(),
        evidence=evidence,
    )


def _build_upstream_view(
    *,
    role: ConversationalRoleSnapshot | None,
    belief: BeliefAboutOtherSnapshot | None,
) -> dict[str, Snapshot]:
    """Build a minimal upstream mapping the owner.process expects.

    Mirrors the kernel's ``UpstreamView`` for slots the owner declares
    as dependencies. Slots that resolve to ``None`` are omitted so the
    owner sees a missing-slot placeholder behaviour.
    """
    upstream: dict[str, Snapshot] = {}
    if role is not None:
        upstream["conversational_role"] = Snapshot(
            slot_name="conversational_role",
            owner="ConversationalRoleModule",
            version=1,
            timestamp_ms=1,
            value=role,
        )
    if belief is not None:
        upstream["belief_about_other"] = Snapshot(
            slot_name="belief_about_other",
            owner="BeliefAboutOtherModule",
            version=1,
            timestamp_ms=1,
            value=belief,
        )
    return upstream


def _run_owner(upstream: dict[str, Snapshot]) -> Snapshot:
    module = CommonGroundModule()
    return asyncio.run(module.process(upstream))


def test_no_role_means_no_upstream_atoms() -> None:
    snapshot = _run_owner(_build_upstream_view(role=None, belief=None)).value
    assert snapshot.dyad_atoms == ()
    assert snapshot.group_atoms == ()
    assert snapshot.control_signal == 0.0


def test_low_confidence_role_means_no_upstream_atoms() -> None:
    role = _role_snapshot(role_confidence=0.10)
    belief = _belief_snapshot(
        records=(
            _belief_record(
                record_id="b1",
                summary="meeting tomorrow",
                confidence=0.85,
            ),
        )
    )
    snapshot = _run_owner(_build_upstream_view(role=role, belief=belief)).value
    assert snapshot.dyad_atoms == ()


def test_low_confidence_belief_records_are_filtered() -> None:
    role = _role_snapshot()
    belief = _belief_snapshot(
        records=(
            _belief_record(
                record_id="b-low",
                summary="meeting tomorrow",
                confidence=0.30,
            ),
        )
    )
    snapshot = _run_owner(_build_upstream_view(role=role, belief=belief)).value
    assert snapshot.dyad_atoms == ()


def test_high_confidence_dyad_belief_yields_typed_atom() -> None:
    role = _role_snapshot()
    belief = _belief_snapshot(
        records=(
            _belief_record(
                record_id="b1",
                summary="meeting tomorrow",
                confidence=0.85,
                evidence="meeting tomorrow",
            ),
            _belief_record(
                record_id="b2",
                summary="prefers terse replies",
                confidence=0.70,
                evidence="terse",
            ),
        )
    )
    snapshot = _run_owner(_build_upstream_view(role=role, belief=belief)).value
    assert len(snapshot.dyad_atoms) == 2
    atom_a, atom_b = snapshot.dyad_atoms
    assert atom_a.scope_kind is SocialScopeKind.DYAD
    assert atom_b.scope_kind is SocialScopeKind.DYAD
    # scope_id must be deterministic and dyad-symmetric.
    assert atom_a.scope_id == atom_b.scope_id
    assert PRIMARY_INTERLOCUTOR_ID in atom_a.accepted_by_ids
    assert SELF_INTERLOCUTOR_ID in atom_a.accepted_by_ids
    # Recursion depth is 1 ("we know the user holds this belief").
    assert atom_a.recursion_depth == 1
    # Confidence is forwarded from the BELIEF record (typed, not text).
    assert 0.6 < atom_a.confidence <= 1.0
    # Mean confidence is published as control_signal.
    assert 0.0 < snapshot.control_signal <= 1.0


def test_multi_addressee_role_is_not_treated_as_dyad() -> None:
    """Group conversation (role with multiple addressees) suppresses
    the typed dyad atom path; group atoms remain spec-future-work.
    """
    role = ConversationalRoleSnapshot(
        active_speaker_id="alice",
        addressee_ids=("bob", "carol"),
        subject_ids=("alice",),
        witness_ids=(),
        overhearer_ids=(),
        group_audience_ids=("alice", "bob", "carol"),
        role_confidence=0.85,
        active_predictions=(),
        description="multi-addressee role",
    )
    belief = _belief_snapshot(
        records=(
            _belief_record(
                record_id="b1",
                summary="meeting tomorrow",
                confidence=0.85,
            ),
        )
    )
    snapshot = _run_owner(_build_upstream_view(role=role, belief=belief)).value
    assert snapshot.dyad_atoms == ()
    assert snapshot.group_atoms == ()
