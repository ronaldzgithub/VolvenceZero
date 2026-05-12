"""Packet E (long-horizon-closure) — cross-session owner hydration e2e.

The packet's headline test: same user_id + same memory_scope_root_dir +
``BrainConfig.owner_hydration_wiring=ACTIVE`` -> destroy BrainSession ->
new BrainSession sees the prior session's relationship-axis state
(commitment lifecycle, followups, vitals).

Four scenarios:

1. ``commitment_lifecycle_continues`` — REJECT-aligned commitment from
   session 1 still visible on the new SemanticStateStore + still
   surfaced as a followup.
2. ``rupture_repair_durable_memory_continues`` — DURABLE rupture_repair
   memory entry from session 1 still surfaceable in session 2.
3. ``vitals_drive_levels_continue`` — drive that decayed below initial
   level in session 1 stays at the lowered level in session 2.
4. ``cross_user_isolation`` — none of the above leaks across user_id.

These tests use real ``Brain`` / ``Lifeform`` constructors with
identity providers + ``memory_scope_root_dir``. They do NOT use
SimpleNamespace fakes — that's the whole point: the new code must
be honest end-to-end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.memory import (
    StaticIdentityProvider,
    UserIdentity,
    list_durable_entries_for_scope,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    AdvocacyState,
    AlignmentState,
    SemanticProposal,
    SemanticProposalOperation,
)


def _make_brain(*, scope_root: Path, user_id: str = "alice") -> Brain:
    """Build a Brain wired for cross-session owner hydration."""
    identity = UserIdentity(user_id=user_id, scope_key=user_id)
    config = BrainConfig(
        memory_scope_root_dir=str(scope_root),
        owner_hydration_wiring=WiringLevel.ACTIVE,
    )
    return Brain(config, identity_provider=StaticIdentityProvider(identity=identity))


def _drive_commitment_to_reject(runner) -> None:
    """Force the SemanticStateStore to record a commitment that the
    user rejected. We bypass the proposal runtime so the test stays
    deterministic — the lifecycle path itself is what's being
    tested cross-session.
    """
    runner._semantic_state_store.apply(  # noqa: SLF001
        slot="commitment",
        proposals=(
            SemanticProposal(
                proposal_id="commit-cross-1",
                target_slot="commitment",
                operation=SemanticProposalOperation.ACTIVATE,
                summary="commit to nightly journal",
                detail="ai surfaced this commitment last turn",
                evidence="user said yes initially",
                confidence=0.85,
                control_signal=0.5,
            ),
            SemanticProposal(
                proposal_id="commit-cross-1",
                target_slot="commitment",
                operation=SemanticProposalOperation.BLOCK,
                summary="user retracted nightly journal",
                detail="user said this won't work right now",
                evidence="user rejected explicitly",
                confidence=0.95,
                control_signal=0.0,
            ),
        ),
        turn_index=2,
    )


def test_commitment_lifecycle_continues_across_session_boundary(
    tmp_path: Path,
) -> None:
    """REJECT-aligned commitment from session 1 must be visible on
    the new SemanticStateStore in session 2 (same user, same scope
    root).
    """
    brain1 = _make_brain(scope_root=tmp_path)
    session1 = brain1.create_session(session_id="alice-sess-1")
    runner1 = session1.runner
    _drive_commitment_to_reject(runner1)
    # Sanity: session 1 sees the lifecycle.
    lifecycle1 = runner1._semantic_state_store.lifecycle_for(  # noqa: SLF001
        "commitment"
    )
    assert "commit-cross-1" in lifecycle1
    assert lifecycle1["commit-cross-1"][1] is AlignmentState.REJECT
    # Persist + tear down.
    persisted = session1.persist_owners()
    assert "semantic_state" in persisted
    del session1
    del brain1

    # Session 2: brand-new Brain, same scope root + same user id.
    brain2 = _make_brain(scope_root=tmp_path)
    session2 = brain2.create_session(session_id="alice-sess-2")
    runner2 = session2.runner
    lifecycle2 = runner2._semantic_state_store.lifecycle_for(  # noqa: SLF001
        "commitment"
    )
    assert "commit-cross-1" in lifecycle2, (
        "REJECT-aligned commitment lost across BrainSession boundary "
        "with hydration ACTIVE; expected commit-cross-1 to survive."
    )
    advocacy, alignment = lifecycle2["commit-cross-1"]
    assert advocacy is AdvocacyState.PROPOSED
    assert alignment is AlignmentState.REJECT
    # Records and completed_refs survived too.
    records2 = runner2._semantic_state_store.records_for("commitment")  # noqa: SLF001
    assert any(r.record_id == "commit-cross-1" for r in records2)


def test_rupture_repair_durable_memory_continues_across_session_boundary(
    tmp_path: Path,
) -> None:
    """A DURABLE rupture_repair entry written in session 1 must still
    be queryable under the same user_scope in session 2.

    This already worked before Packet D (MemoryStore had its own
    persistence backend with eager load_from_backend on
    build_scoped_memory_store). The Packet E test pins it down so a
    future change to MemoryStore persistence can't silently regress
    cross-session memory continuity for user-scoped rupture-repair
    entries.
    """
    from volvence_zero.dialogue_trace import (
        DialogueExternalOutcomeEvidenceSource,
        DialogueExternalOutcomeKind,
    )
    from volvence_zero.memory import MemoryStratum

    brain1 = _make_brain(scope_root=tmp_path)
    session1 = brain1.create_session(session_id="alice-sess-1")

    asyncio.run(session1.run_turn_async("hi there"))
    session1.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
    )
    asyncio.run(session1.run_turn_async("that one missed"))
    runner1 = session1.runner
    runner1.begin_new_context(reason="cross-session-test-end")
    asyncio.run(runner1.drain_session_post_slow_loop())
    # Confirm a rupture_repair entry was written under alice's scope.
    alice_entries_session1 = list_durable_entries_for_scope(
        runner1._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    assert alice_entries_session1, (
        "Session 1 should produce at least one DURABLE rupture_repair "
        "entry under alice's scope; got none."
    )
    # MemoryStore writes via FileSystemPersistenceBackend save on
    # save_to_backend; the eager load_from_backend was called when
    # the scoped store was built. Force a save now to make sure the
    # latest state is on disk before tear-down.
    runner1._memory_store.save_to_backend()  # noqa: SLF001
    del session1
    del brain1

    brain2 = _make_brain(scope_root=tmp_path)
    session2 = brain2.create_session(session_id="alice-sess-2")
    runner2 = session2.runner
    alice_entries_session2 = list_durable_entries_for_scope(
        runner2._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    assert alice_entries_session2, (
        "Session 2 must surface the prior session's DURABLE rupture_repair "
        "entry under alice's scope; got none. This is the "
        "MemoryStore-side cross-session continuity invariant."
    )
    # Ensure the entries actually carry the rupture_repair tag (they
    # are the rupture-repair entries, not other DURABLE writes).
    repair_tagged = [
        e for e in alice_entries_session2
        if "rupture_repair" in e.tags
    ]
    assert repair_tagged, (
        "Surviving DURABLE entries should carry the rupture_repair "
        "tag; got tags: "
        + str(sorted({tag for e in alice_entries_session2 for tag in e.tags}))
    )


def test_vitals_drive_levels_continue_across_session_boundary(
    tmp_path: Path,
) -> None:
    """A drive that decayed below initial_level in session 1 must
    NOT bounce back to initial_level in session 2 (otherwise vitals
    is "starting fresh every restart").

    Uses the full ``Lifeform`` stack (not just ``Brain``) because
    ``VitalsModule`` lives at the lifeform layer and is hydrated
    by ``Lifeform.create_session`` reading
    ``brain_session.owner_hydration_store``.
    """
    from lifeform_core import (
        Lifeform,
        TickEngineConfig,
    )
    from lifeform_core.lifeform import LifeformConfig
    from lifeform_core.types import TickEvent, TickKind
    from lifeform_core.vitals import DriveSpec, VitalsBootstrap

    bootstrap = VitalsBootstrap(
        schema_version=1,
        drives=(
            DriveSpec(
                name="bond_warmth",
                target=0.7,
                homeostatic_band=(0.5, 0.85),
                decay_per_tick=0.05,
                pe_weight=1.0,
                initial_level=0.6,
                recharge_per_turn=0.0,
            ),
        ),
        proactive_pe_threshold=0.3,
        proactive_followup_priority=0.5,
        proactive_cooldown_ticks=10,
    )
    identity = UserIdentity(user_id="alice", scope_key="alice")
    brain_cfg = BrainConfig(
        memory_scope_root_dir=str(tmp_path),
        owner_hydration_wiring=WiringLevel.ACTIVE,
    )
    lifeform_config = LifeformConfig(
        brain_config=brain_cfg,
        tick=TickEngineConfig(system_tick_seconds=0.001),
        vitals_bootstrap=bootstrap,
    )
    identity_provider = StaticIdentityProvider(identity=identity)
    lifeform1 = Lifeform(
        lifeform_config, identity_provider=identity_provider
    )
    session1 = lifeform1.create_session(session_id="alice-vitals-1")
    # Decay the drive substantially below initial_level (0.6) by
    # advancing many SYSTEM ticks.
    vitals1 = session1.vitals_module
    assert vitals1 is not None
    for i in range(8):
        vitals1.on_tick(
            TickEvent(
                tick_index=i + 1,
                kind=TickKind.SYSTEM,
                elapsed_seconds=1.0,
            )
        )
    snap1 = session1.vitals_snapshot
    assert snap1 is not None
    bond_level_session1 = snap1.drive_levels[0].level
    assert bond_level_session1 < 0.5, (
        f"Test setup: expected bond_warmth to decay below 0.5 in "
        f"session 1; got {bond_level_session1!r}. Increase tick "
        f"count or decay_per_tick."
    )
    persisted = session1.persist_owners()
    assert "vitals" in persisted
    assert "followup_manager" in persisted
    del session1
    del lifeform1

    # Session 2: same scope root + same user.
    lifeform2 = Lifeform(
        lifeform_config, identity_provider=identity_provider
    )
    session2 = lifeform2.create_session(session_id="alice-vitals-2")
    snap2 = session2.vitals_snapshot
    assert snap2 is not None
    bond_level_session2 = snap2.drive_levels[0].level
    assert bond_level_session2 == pytest.approx(bond_level_session1, abs=1e-6), (
        f"Vitals must be hydrated across BrainSession boundary; "
        f"session 1 ended bond_warmth at {bond_level_session1!r} "
        f"but session 2 started at {bond_level_session2!r}. "
        f"Expected exact equality after hydration (no further ticks "
        f"in session 2 yet)."
    )


def test_cross_user_isolation_after_owner_hydration(
    tmp_path: Path,
) -> None:
    """Bob's session must NOT see Alice's persisted owner state when
    using the same scope_root. The scope_root is a directory under
    which each user gets their OWN subdirectory; isolation is by
    the per-user persistence backend created from
    build_scoped_memory_store.
    """
    # Alice's session: persist a commitment.
    brain_alice = _make_brain(scope_root=tmp_path, user_id="alice")
    session_alice = brain_alice.create_session(session_id="alice-only")
    _drive_commitment_to_reject(session_alice.runner)
    session_alice.persist_owners()
    del session_alice
    del brain_alice

    # Bob's session: must NOT see Alice's commit-cross-1.
    brain_bob = _make_brain(scope_root=tmp_path, user_id="bob")
    session_bob = brain_bob.create_session(session_id="bob-only")
    bob_lifecycle = session_bob.runner._semantic_state_store.lifecycle_for(  # noqa: SLF001
        "commitment"
    )
    assert "commit-cross-1" not in bob_lifecycle, (
        "Cross-user leakage: Bob's session 1 should not see Alice's "
        f"commit-cross-1 commitment lifecycle. Got: {bob_lifecycle!r}"
    )
    # Bob's records must be empty too.
    bob_records = session_bob.runner._semantic_state_store.records_for(  # noqa: SLF001
        "commitment"
    )
    assert bob_records == ()


def test_owner_hydration_disabled_by_default_does_not_persist(
    tmp_path: Path,
) -> None:
    """Default ``BrainConfig`` (owner_hydration_wiring=DISABLED) must
    NOT hydrate or persist anything. This guards against accidentally
    flipping the flag on without an explicit opt-in.
    """
    identity = UserIdentity(user_id="alice", scope_key="alice")
    # NOTE: explicitly DISABLED (the default; we set it for clarity).
    config = BrainConfig(
        memory_scope_root_dir=str(tmp_path),
        owner_hydration_wiring=WiringLevel.DISABLED,
    )
    brain = Brain(config, identity_provider=StaticIdentityProvider(identity=identity))
    session = brain.create_session(session_id="alice-disabled")
    # Drive a commitment, persist (no-op), tear down.
    _drive_commitment_to_reject(session.runner)
    persisted = session.persist_owners()
    assert persisted == (), (
        "Owner hydration disabled should make persist_owners a no-op; "
        f"got {persisted!r}."
    )
    del session
    del brain

    brain2 = Brain(config, identity_provider=StaticIdentityProvider(identity=identity))
    session2 = brain2.create_session(session_id="alice-disabled-2")
    lifecycle = session2.runner._semantic_state_store.lifecycle_for(  # noqa: SLF001
        "commitment"
    )
    assert "commit-cross-1" not in lifecycle, (
        "DISABLED hydration must not survive across BrainSession; "
        f"got: {lifecycle!r}"
    )
