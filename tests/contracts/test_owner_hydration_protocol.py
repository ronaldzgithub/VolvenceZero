"""Packet D (long-horizon-closure) — HydratableOwnerProtocol contract.

Three owners must satisfy the round-trip invariant:
``hydrate(export()) == export()`` after non-trivial state mutation.
This is the structural guarantee that owner hydration produces no
information loss across a process boundary.

If any of these fail, the corresponding owner's serialiser /
deserialiser disagree on what state matters — a fix is to either
add the missing field to ``export_persistence_snapshot`` or to make
``hydrate_from_persistence`` honour it.

Schema-version bumps and payload corruption / owner mismatch are
covered by ``test_owner_hydration_failures_loud.py``; this test
focuses on the happy round-trip.
"""

from __future__ import annotations

from volvence_zero.owner_hydration import OwnerPersistenceSnapshot


def test_semantic_state_store_round_trip() -> None:
    """SemanticStateStore: 9 slots, lifecycle / followup / outcome maps,
    after applying a meaningful set of proposals across multiple slots.
    """
    from volvence_zero.semantic_state.contracts import (
        SemanticProposal,
        SemanticProposalOperation,
    )
    from volvence_zero.semantic_state.store import SemanticStateStore

    source = SemanticStateStore()
    # Commitment: surface, then user pushes back -> BLOCK -> alignment=REJECT
    source.apply(
        slot="commitment",
        proposals=(
            SemanticProposal(
                proposal_id="c-1",
                target_slot="commitment",
                operation=SemanticProposalOperation.ACTIVATE,
                summary="commit to weekly check-in",
                detail="user proposed",
                evidence="user said yes once",
                confidence=0.8,
                control_signal=0.5,
            ),
            SemanticProposal(
                proposal_id="c-1",
                target_slot="commitment",
                operation=SemanticProposalOperation.BLOCK,
                summary="user retracted",
                detail="user later said no",
                evidence="user retracted",
                confidence=0.9,
                control_signal=0.0,
            ),
        ),
        turn_index=3,
    )
    # Open loop: just one OBSERVE
    source.apply(
        slot="open_loop",
        proposals=(
            SemanticProposal(
                proposal_id="ol-1",
                target_slot="open_loop",
                operation=SemanticProposalOperation.OBSERVE,
                summary="user mentioned travel plans",
                detail="follow up next week",
                evidence="user said upcoming trip",
                confidence=0.65,
                control_signal=0.0,
            ),
        ),
        turn_index=5,
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "semantic_state"
    assert exported.schema_version == 1

    target = SemanticStateStore()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, (
        "SemanticStateStore round-trip lost or mutated state."
    )
    # Spot-check that meaningful read accessors return identical results.
    assert target.records_for("commitment") == source.records_for("commitment")
    assert target.lifecycle_for("commitment") == source.lifecycle_for("commitment")
    assert target.records_for("open_loop") == source.records_for("open_loop")


def test_followup_manager_round_trip() -> None:
    from lifeform_core.followup_manager import FollowupManager

    source = FollowupManager(default_due_delay_ticks=60, max_pending=8)
    source.ingest_open_loops(
        unresolved_loops=("loop-a", "loop-b", "loop-c"),
        current_tick=10,
    )
    source.ingest_at_risk_commitments(
        at_risk_refs=("commit-1",),
        current_tick=15,
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "followup_manager"
    assert exported.schema_version == 1

    target = FollowupManager(default_due_delay_ticks=60, max_pending=8)
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, (
        "FollowupManager round-trip lost or mutated state."
    )
    assert target.pending == source.pending


def test_vitals_module_round_trip() -> None:
    from lifeform_core.vitals import (
        DriveSpec,
        VitalsBootstrap,
        VitalsModule,
    )
    from lifeform_core.types import TickEvent, TickKind

    bootstrap = VitalsBootstrap(
        schema_version=1,
        drives=(
            DriveSpec(
                name="bond_warmth",
                target=0.7,
                homeostatic_band=(0.5, 0.85),
                decay_per_tick=0.005,
                pe_weight=1.0,
                initial_level=0.6,
                recharge_per_turn=0.02,
            ),
            DriveSpec(
                name="user_engagement",
                target=0.6,
                homeostatic_band=(0.3, 0.85),
                decay_per_tick=0.02,
                pe_weight=0.5,
                initial_level=0.5,
                recharge_per_turn=0.3,
            ),
        ),
        proactive_pe_threshold=1.0,
        proactive_followup_priority=0.5,
        proactive_cooldown_ticks=60,
    )
    source = VitalsModule(bootstrap)
    # Drive the levels somewhere non-trivial via several SYSTEM ticks.
    for i in range(10):
        source.on_tick(
            TickEvent(
                tick_index=i + 1,
                kind=TickKind.SYSTEM,
                elapsed_seconds=1.0,
            )
        )
    source.on_turn(regime="emotional_support", user_input_present=True)
    # Force a proactive crossing to populate _last_proactive_at.
    source.consider_proactive_followup(current_tick=20)

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "vitals"
    assert exported.schema_version == 1

    target = VitalsModule(bootstrap)
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, (
        "VitalsModule round-trip lost or mutated state."
    )
    # Snapshot equality on observable fields.
    s = source.current_snapshot()
    t = target.current_snapshot()
    assert s.drive_levels == t.drive_levels
    assert s.tick_index == t.tick_index
    assert s.last_proactive_at_tick == t.last_proactive_at_tick


def test_round_trip_is_idempotent_on_double_hydrate() -> None:
    """Applying the same snapshot twice must produce the same store
    state (no accumulating effects).
    """
    from volvence_zero.semantic_state.contracts import (
        SemanticProposal,
        SemanticProposalOperation,
    )
    from volvence_zero.semantic_state.store import SemanticStateStore

    source = SemanticStateStore()
    source.apply(
        slot="open_loop",
        proposals=(
            SemanticProposal(
                proposal_id="ol-x",
                target_slot="open_loop",
                operation=SemanticProposalOperation.OBSERVE,
                summary="x",
                detail="",
                evidence="",
                confidence=0.5,
                control_signal=0.0,
            ),
        ),
        turn_index=1,
    )
    snap = source.export_persistence_snapshot()

    target = SemanticStateStore()
    target.hydrate_from_persistence(snap)
    after_first = target.export_persistence_snapshot()
    target.hydrate_from_persistence(snap)
    after_second = target.export_persistence_snapshot()

    assert after_first.payload == after_second.payload, (
        "Re-hydrating with the same snapshot must be idempotent."
    )


def test_owner_persistence_snapshot_validates_required_fields() -> None:
    """Frozen dataclass validation: empty owner_name, version 0, or
    non-Mapping payload must raise.
    """
    import pytest

    with pytest.raises(ValueError, match="owner_name"):
        OwnerPersistenceSnapshot(
            owner_name="", schema_version=1, payload={}
        )
    with pytest.raises(ValueError, match="schema_version"):
        OwnerPersistenceSnapshot(
            owner_name="x", schema_version=0, payload={}
        )
    with pytest.raises(TypeError, match="Mapping"):
        OwnerPersistenceSnapshot(
            owner_name="x", schema_version=1, payload="not a mapping"  # type: ignore[arg-type]
        )
