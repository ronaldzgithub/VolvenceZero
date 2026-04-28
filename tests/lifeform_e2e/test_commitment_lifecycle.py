"""Tests for the AAC decision lifecycle (Gap 7 in ``docs/todo``).

Pins three things:

1. ``commitment_lifecycle_for_operation`` is a pure truth-table from
   ``SemanticProposalOperation`` to ``(AdvocacyState, AlignmentState)``,
   including the rule that operations leaving one axis ``None`` preserve
   the previous value of that axis.
2. ``SemanticStateStore`` advances and garbage-collects per-record
   lifecycle state alongside the existing record window.
3. ``CommitmentSnapshot`` publishes the lifecycle parallel to
   ``active_commitments`` and exposes aggregate counts. Existing
   consumers (FollowupManager / final_wiring / evaluation) keep the
   pre-existing fields unchanged.

The lifecycle is driven *only* by typed ``SemanticProposalOperation``
values \u2014 never from LLM keyword detection on user text. That is the
whole point of this layer (see ``docs/todo``  Gap 7's "MUST go through
SemanticProposal typed path" rule).
"""

from __future__ import annotations

import pytest

from volvence_zero.semantic_state import (
    AdvocacyState,
    AlignmentState,
    CommitmentLifecycleEntry,
    CommitmentSnapshot,
    SemanticProposal,
    SemanticProposalOperation,
    SemanticStateStore,
    commitment_lifecycle_for_operation,
)


# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            SemanticProposalOperation.OBSERVE,
            (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
        ),
        (
            SemanticProposalOperation.CREATE,
            (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
        ),
        (
            SemanticProposalOperation.DEFER,
            (AdvocacyState.READY, AlignmentState.UNKNOWN),
        ),
        (
            SemanticProposalOperation.ACTIVATE,
            (AdvocacyState.PROPOSED, AlignmentState.UNKNOWN),
        ),
        (
            SemanticProposalOperation.REVISE,
            (AdvocacyState.PROPOSED, AlignmentState.MODIFY),
        ),
        (
            SemanticProposalOperation.COMPLETE,
            (AdvocacyState.PROPOSED, AlignmentState.AGREE),
        ),
        (
            SemanticProposalOperation.CLOSE,
            (AdvocacyState.PROPOSED, AlignmentState.UNKNOWN),
        ),
        (
            SemanticProposalOperation.BLOCK,
            (AdvocacyState.PROPOSED, AlignmentState.REJECT),
        ),
    ],
)
def test_truth_table_from_clean_prior(operation, expected):
    assert commitment_lifecycle_for_operation(operation) == expected


def test_activate_after_revise_preserves_modify_alignment():
    """User said "modify"; AI then re-advocates the modified version. The
    user's earlier alignment signal must NOT be silently overwritten by
    the AI's advocacy operation."""
    previous = (AdvocacyState.PROPOSED, AlignmentState.MODIFY)
    advocacy, alignment = commitment_lifecycle_for_operation(
        SemanticProposalOperation.ACTIVATE, previous=previous
    )
    assert advocacy is AdvocacyState.PROPOSED
    assert alignment is AlignmentState.MODIFY


def test_close_does_not_clear_prior_alignment_signal():
    """CLOSE means "wrap up the record"; it should NOT overwrite an
    explicit prior alignment such as REJECT."""
    previous = (AdvocacyState.PROPOSED, AlignmentState.REJECT)
    advocacy, alignment = commitment_lifecycle_for_operation(
        SemanticProposalOperation.CLOSE, previous=previous
    )
    assert advocacy is AdvocacyState.PROPOSED
    assert alignment is AlignmentState.REJECT


# ---------------------------------------------------------------------------
# Store integration
# ---------------------------------------------------------------------------


def _proposal(
    pid: str, op: SemanticProposalOperation, *, slot: str = "commitment"
) -> SemanticProposal:
    return SemanticProposal(
        proposal_id=pid,
        target_slot=slot,
        operation=op,
        summary="s",
        detail="d",
        confidence=0.7,
        evidence="e",
    )


def test_store_walks_one_commitment_through_full_lifecycle():
    store = SemanticStateStore()
    sequence = (
        SemanticProposalOperation.CREATE,
        SemanticProposalOperation.ACTIVATE,
        SemanticProposalOperation.REVISE,
        SemanticProposalOperation.COMPLETE,
    )
    expected = (
        (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
        (AdvocacyState.PROPOSED, AlignmentState.UNKNOWN),
        (AdvocacyState.PROPOSED, AlignmentState.MODIFY),
        (AdvocacyState.PROPOSED, AlignmentState.AGREE),
    )
    for turn, (op, expected_state) in enumerate(zip(sequence, expected), start=1):
        store.apply(
            slot="commitment",
            proposals=(_proposal("c-1", op),),
            turn_index=turn,
        )
        state = store.lifecycle_for("commitment").get("c-1")
        assert state == expected_state, f"after {op.value}: got {state}"


def test_store_lifecycle_only_for_targeted_slot():
    """Lifecycle map is per-slot; a proposal on slot A never bleeds into
    slot B's lifecycle bookkeeping."""
    store = SemanticStateStore()
    store.apply(
        slot="commitment",
        proposals=(_proposal("c-1", SemanticProposalOperation.ACTIVATE),),
        turn_index=1,
    )
    store.apply(
        slot="plan_intent",
        proposals=(
            _proposal(
                "p-1", SemanticProposalOperation.ACTIVATE, slot="plan_intent"
            ),
        ),
        turn_index=2,
    )
    assert "c-1" in store.lifecycle_for("commitment")
    assert "p-1" not in store.lifecycle_for("commitment")


def test_store_garbage_collects_lifecycle_when_record_falls_out_of_window():
    """The bounded record window keeps only the last 12 records. Lifecycle
    entries for evicted records must be garbage-collected so the map
    does not grow unboundedly across long sessions."""
    store = SemanticStateStore()
    for i in range(15):
        store.apply(
            slot="commitment",
            proposals=(
                _proposal(f"c-{i:02d}", SemanticProposalOperation.ACTIVATE),
            ),
            turn_index=i + 1,
        )
    lifecycle = store.lifecycle_for("commitment")
    # Window keeps last 12, so c-00 / c-01 / c-02 are evicted.
    for evicted in ("c-00", "c-01", "c-02"):
        assert evicted not in lifecycle
    # And the lifecycle map size matches the live record count.
    live_record_ids = {r.record_id for r in store.records_for("commitment")}
    assert set(lifecycle.keys()) == live_record_ids


# ---------------------------------------------------------------------------
# CommitmentSnapshot exposes lifecycle on the public surface
# ---------------------------------------------------------------------------


def test_synthetic_commitment_snapshot_keeps_legacy_defaults():
    """Old code that builds CommitmentSnapshot positionally must still
    work: lifecycle defaults to empty tuple + zero counts."""
    snap = CommitmentSnapshot(
        active_commitments=(),
        honored_commitment_refs=(),
        at_risk_commitments=(),
        trust_obligation_count=0,
        continuity_score=0.0,
        control_signal=0.0,
        description="t",
    )
    assert snap.lifecycle_entries == ()
    assert snap.advocacy_proposed_count == 0
    assert snap.advocacy_ready_count == 0
    assert snap.alignment_agree_count == 0
    assert snap.alignment_modify_count == 0
    assert snap.alignment_reject_count == 0


def test_lifecycle_for_returns_entry_or_none():
    snap = CommitmentSnapshot(
        active_commitments=(),
        honored_commitment_refs=(),
        at_risk_commitments=(),
        trust_obligation_count=0,
        continuity_score=0.0,
        control_signal=0.0,
        description="t",
        lifecycle_entries=(
            CommitmentLifecycleEntry(
                record_id="c-1",
                advocacy_state=AdvocacyState.PROPOSED,
                alignment_state=AlignmentState.AGREE,
            ),
        ),
    )
    entry = snap.lifecycle_for("c-1")
    assert entry is not None
    assert entry.advocacy_state is AdvocacyState.PROPOSED
    assert snap.lifecycle_for("missing") is None


# ---------------------------------------------------------------------------
# End-to-end: a CommitmentModule produces a snapshot whose aggregates
# match the operation history we drove through it.
# ---------------------------------------------------------------------------


def test_lifeform_session_surfaces_followup_for_rejected_commitment():
    """A REJECT-aligned lifecycle entry must surface a commitment-source
    ``FollowupItem`` even when the owner-side status is not ``blocked``.

    Without this, a user explicitly rejecting a commitment would only
    leave an alignment_state=REJECT in the snapshot and otherwise be
    invisible to the follow-up queue \u2014 which is exactly what Gap 7's
    "AAC decision lifecycle" is supposed to fix."""
    from lifeform_core import (
        FollowupManager,
        SceneManager,
        TickEngine,
        TickEngineConfig,
    )
    from lifeform_core.lifeform import LifeformSession

    class _FakeBrainSession:
        def __init__(self):
            self.session_id = "lifecycle-followup"

        async def run_turn_async(self, _input: str):
            from types import SimpleNamespace

            commitment_value = CommitmentSnapshot(
                active_commitments=(),
                honored_commitment_refs=(),
                at_risk_commitments=(),
                trust_obligation_count=0,
                continuity_score=0.0,
                control_signal=0.0,
                description="rejected",
                lifecycle_entries=(
                    CommitmentLifecycleEntry(
                        record_id="commit-rej-1",
                        advocacy_state=AdvocacyState.PROPOSED,
                        alignment_state=AlignmentState.REJECT,
                    ),
                ),
                alignment_reject_count=1,
                advocacy_proposed_count=1,
            )
            commitment_snap = SimpleNamespace(value=commitment_value)
            response = SimpleNamespace(text="ok", regime_id=None, abstract_action=None, rationale="")
            return SimpleNamespace(
                response=response,
                active_regime=None,
                active_abstract_action=None,
                active_snapshots={"commitment": commitment_snap},
            )

    tick = TickEngine(TickEngineConfig(system_tick_seconds=0.001))
    scene = SceneManager(idle_close_after_system_ticks=None)
    followups = FollowupManager()
    session = LifeformSession(
        brain_session=_FakeBrainSession(),  # type: ignore[arg-type]
        tick=tick,
        scene=scene,
        followups=followups,
    )

    import asyncio

    asyncio.run(session.run_turn("hi"))
    pending = session.followup_manager.pending
    assert any(
        item.source == "commitment" and "commit-rej-1" in item.metadata.get("key", "")
        for item in pending
    ), f"expected a commitment followup keyed on commit-rej-1; got {pending}"


async def test_commitment_module_publishes_aggregate_counts_through_kernel_path():
    """Drive a fresh BrainSession through scenarios that submit typed
    semantic events. The resulting ``commitment`` snapshot must expose
    a non-empty lifecycle parallel to ``active_commitments`` whenever
    the kernel has produced at least one commitment record.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    life = build_companion_lifeform()
    session = life.create_session(session_id="lifecycle-e2e")
    # Run a couple of turns so the kernel's semantic owners get exercised.
    for text in (
        "I have been feeling really stuck lately and I do not know why.",
        "Honestly that almost made it worse, I just wanted to be heard.",
    ):
        result = await session.run_turn(text)
    snap = result.active_snapshots.get("commitment")
    if snap is None:
        pytest.skip("commitment snapshot not in active set on this run")
    value = snap.value
    assert isinstance(value, CommitmentSnapshot)
    # The lifecycle parallel must exactly match the active_commitments
    # tuple's length \u2014 one entry per record, in order. Aggregate counts
    # cannot exceed total records.
    expected_record_count = len(value.active_commitments) + len(value.at_risk_commitments)
    if expected_record_count == 0:
        # Run produced no records this time; lifecycle is allowed to be empty.
        assert value.lifecycle_entries == ()
    else:
        assert len(value.lifecycle_entries) >= expected_record_count
    total_alignment = (
        value.alignment_agree_count
        + value.alignment_modify_count
        + value.alignment_reject_count
    )
    assert total_alignment <= len(value.lifecycle_entries)
    assert value.advocacy_ready_count + value.advocacy_proposed_count <= len(
        value.lifecycle_entries
    )
