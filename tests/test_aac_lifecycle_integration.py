"""End-to-end tests for the AAC commitment lifecycle (Gap 7).

Scope (docs/specs/aac-lifecycle.md):

* **alignment_reject_triggers_pe_spike** \u2014 when a commitment transitions
  AGREE \u2192 REJECT turn-over-turn, ``PredictionErrorModule`` should emit a
  visibly higher magnitude + more-negative signed_reward than the prior
  turn's baseline. The spike is the discrete-event PE source required
  by R-PE / R14 to drive a ``repair_and_deescalation`` regime transition
  downstream.
* **followup_manager_policy_routing** \u2014 ``FollowupManager`` must route
  ``DEFER_ONLY`` commitments to a later due tick and lower priority than
  ``GENTLE_CHECKIN`` commitments. No keyword heuristics \u2014 strictly
  typed enum dispatch.
* **outcome_aggregate_counts** \u2014 ``CommitmentSnapshot`` must publish
  per-outcome counters that match the typed transitions for a
  multi-turn scripted scenario.

These exercise the PredictionErrorModule's alignment overlay directly
via ``process_standalone`` / synthetic CommitmentSnapshots; the full
``run_final_wiring_turn`` path is already covered by
``tests/test_semantic_state_owners.py``.
"""

from __future__ import annotations

import asyncio

from lifeform_core.followup_manager import FollowupManager
from lifeform_core.types import FollowupItem

from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import Track
from volvence_zero.prediction.error import PredictionErrorModule, PredictionErrorSnapshot
from volvence_zero.regime.identity import RegimeIdentity, RegimeSnapshot
from volvence_zero.semantic_state import (
    AdvocacyState,
    AlignmentState,
    CommitmentLifecycleEntry,
    CommitmentOutcomeKind,
    CommitmentSnapshot,
    FollowupPolicy,
    SemanticRecord,
)


# ---------------------------------------------------------------------------
# Synthetic snapshot builders
# ---------------------------------------------------------------------------


_NEUTRAL_CONTROLLER_CODE: tuple[float, ...] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)


def _track_state(track: Track) -> TrackState:
    return TrackState(
        track=track,
        active_goals=("goal-a",),
        recent_credits=(),
        controller_code=_NEUTRAL_CONTROLLER_CODE,
        tension_level=0.3,
    )


def _dual_track_snapshot() -> DualTrackSnapshot:
    return DualTrackSnapshot(
        world_track=_track_state(Track.WORLD),
        self_track=_track_state(Track.SELF),
        cross_track_tension=0.2,
        description="Dual-track snapshot for AAC integration test.",
    )


def _evaluation_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="relationship",
                metric_name="continuity",
                value=0.6,
                confidence=0.7,
                evidence="baseline relationship continuity",
            ),
            EvaluationScore(
                family="task",
                metric_name="usefulness",
                value=0.55,
                confidence=0.6,
                evidence="baseline task usefulness",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="baseline evaluation snapshot",
    )


def _regime_snapshot() -> RegimeSnapshot:
    identity = RegimeIdentity(
        regime_id="emotional_support",
        name="emotional support",
        embedding=_NEUTRAL_CONTROLLER_CODE,
        entry_conditions="user expressing distress",
        exit_conditions="distress acknowledged",
        historical_effectiveness=0.6,
    )
    return RegimeSnapshot(
        active_regime=identity,
        previous_regime=None,
        switch_reason="baseline",
        candidate_regimes=(("emotional_support", 0.6),),
        turns_in_current_regime=1,
        description="baseline regime snapshot",
        delayed_outcomes=(),
        effectiveness_trend=(("emotional_support", 0.6),),
    )


def _commitment_snapshot(*, entries: tuple[CommitmentLifecycleEntry, ...]) -> CommitmentSnapshot:
    records = tuple(
        SemanticRecord(
            record_id=entry.record_id,
            summary=f"commitment:{entry.record_id}",
            detail=f"detail for {entry.record_id}",
            confidence=0.8,
            status="blocked" if entry.alignment_state is AlignmentState.REJECT else "active",
            source_turn=1,
            evidence=f"evidence:{entry.record_id}",
        )
        for entry in entries
    )
    advocacy_counts = {AdvocacyState.READY: 0, AdvocacyState.PROPOSED: 0}
    alignment_counts = {
        AlignmentState.AGREE: 0,
        AlignmentState.MODIFY: 0,
        AlignmentState.REJECT: 0,
    }
    policy_counts = {FollowupPolicy.GENTLE_CHECKIN: 0, FollowupPolicy.DEFER_ONLY: 0}
    for entry in entries:
        if entry.advocacy_state in advocacy_counts:
            advocacy_counts[entry.advocacy_state] += 1
        if entry.alignment_state in alignment_counts:
            alignment_counts[entry.alignment_state] += 1
        if entry.followup_policy in policy_counts:
            policy_counts[entry.followup_policy] += 1
    return CommitmentSnapshot(
        active_commitments=tuple(
            record for record in records if record.status == "active"
        ),
        honored_commitment_refs=(),
        at_risk_commitments=tuple(
            record for record in records if record.status == "blocked"
        ),
        trust_obligation_count=len(records),
        continuity_score=0.6,
        control_signal=0.2,
        description="synthetic commitment snapshot",
        lifecycle_entries=entries,
        advocacy_proposed_count=advocacy_counts[AdvocacyState.PROPOSED],
        advocacy_ready_count=advocacy_counts[AdvocacyState.READY],
        alignment_agree_count=alignment_counts[AlignmentState.AGREE],
        alignment_modify_count=alignment_counts[AlignmentState.MODIFY],
        alignment_reject_count=alignment_counts[AlignmentState.REJECT],
        followup_gentle_count=policy_counts[FollowupPolicy.GENTLE_CHECKIN],
        followup_defer_only_count=policy_counts[FollowupPolicy.DEFER_ONLY],
    )


def _run_pe_turn(
    module: PredictionErrorModule,
    *,
    turn_index: int,
    commitment_snapshot: CommitmentSnapshot | None,
) -> PredictionErrorSnapshot:
    snapshot = asyncio.run(
        module.process_standalone(
            substrate_snapshot=None,
            evaluation_snapshot=_evaluation_snapshot(),
            dual_track_snapshot=_dual_track_snapshot(),
            regime_snapshot=_regime_snapshot(),
            commitment_snapshot=commitment_snapshot,
            turn_index=turn_index,
            previous_prediction=module._previous_prediction,  # noqa: SLF001 \u2014 test-only peek
            previous_substrate_snapshot=None,
        )
    )
    assert isinstance(snapshot.value, PredictionErrorSnapshot)
    module._previous_prediction = snapshot.value.next_prediction  # noqa: SLF001
    return snapshot.value


# ---------------------------------------------------------------------------
# alignment_reject_triggers_pe_spike
# ---------------------------------------------------------------------------


def test_alignment_reject_triggers_pe_spike() -> None:
    """Turn 1: commitment is AGREE. Turn 2: commitment flips to REJECT.

    The turn-2 PredictionError must carry a visibly larger magnitude and
    a more-negative signed_reward than turn 1's baseline. The description
    must mention the alignment transition so the audit trail is explicit.
    """
    module = PredictionErrorModule()
    turn_1_entry = CommitmentLifecycleEntry(
        record_id="c-1",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.AGREE,
    )
    turn_2_entry = CommitmentLifecycleEntry(
        record_id="c-1",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.REJECT,
        followup_policy=FollowupPolicy.DEFER_ONLY,
        last_outcome=CommitmentOutcomeKind.REJECTED,
        last_outcome_evidence="user said no, please drop it",
        last_outcome_at_turn=2,
    )

    turn_1 = _run_pe_turn(
        module,
        turn_index=1,
        commitment_snapshot=_commitment_snapshot(entries=(turn_1_entry,)),
    )
    turn_2 = _run_pe_turn(
        module,
        turn_index=2,
        commitment_snapshot=_commitment_snapshot(entries=(turn_2_entry,)),
    )

    # Turn 2 is the AGREE -> REJECT regression. The overlay must lift
    # magnitude and push signed_reward more negative.
    assert turn_2.error.magnitude > turn_1.error.magnitude + 0.5, (
        f"Expected alignment regression to lift PE magnitude; "
        f"got turn_1={turn_1.error.magnitude:.3f} turn_2={turn_2.error.magnitude:.3f}"
    )
    assert turn_2.error.signed_reward < turn_1.error.signed_reward, (
        f"Expected alignment regression to push signed_reward negative; "
        f"got turn_1={turn_1.error.signed_reward:.3f} "
        f"turn_2={turn_2.error.signed_reward:.3f}"
    )
    # Turn-2 relationship axis should also absorb the hit.
    assert turn_2.error.relationship_error < turn_1.error.relationship_error, (
        f"Expected relationship_error to drop after REJECT; "
        f"got turn_1={turn_1.error.relationship_error:.3f} "
        f"turn_2={turn_2.error.relationship_error:.3f}"
    )
    # Audit: the description must call out the alignment transition
    # so operators can trace *why* the PE spiked.
    assert "alignment_transition" in turn_2.error.description
    assert "agree->reject" in turn_2.error.description


def test_alignment_recovery_increases_magnitude_but_not_reward_negatively() -> None:
    """REJECT \u2192 AGREE is a recovery; magnitude still rises (it is a
    discrete-event surprise) but signed_reward should not drop; ideally
    it rises.
    """
    module = PredictionErrorModule()
    reject_entry = CommitmentLifecycleEntry(
        record_id="c-2",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.REJECT,
        followup_policy=FollowupPolicy.DEFER_ONLY,
        last_outcome=CommitmentOutcomeKind.REJECTED,
        last_outcome_evidence="initial rejection",
        last_outcome_at_turn=1,
    )
    agree_entry = CommitmentLifecycleEntry(
        record_id="c-2",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.AGREE,
        last_outcome=CommitmentOutcomeKind.COMPLETED,
        last_outcome_evidence="user changed mind, agreed",
        last_outcome_at_turn=2,
    )
    turn_1 = _run_pe_turn(
        module,
        turn_index=1,
        commitment_snapshot=_commitment_snapshot(entries=(reject_entry,)),
    )
    turn_2 = _run_pe_turn(
        module,
        turn_index=2,
        commitment_snapshot=_commitment_snapshot(entries=(agree_entry,)),
    )

    # Recovery still counts as a discrete-event PE \u2014 magnitude rises.
    assert turn_2.error.magnitude > turn_1.error.magnitude
    # Recovery pushes signed_reward in the positive direction (the
    # alignment contribution is the negative of the signed severity
    # for recoveries).
    assert turn_2.error.signed_reward >= turn_1.error.signed_reward
    assert "recovery" in turn_2.error.description


def test_no_alignment_change_produces_no_overlay() -> None:
    """If alignment stays the same across two turns, PE must be driven
    entirely by the base 4-axis computation \u2014 the alignment overlay
    must not fabricate a transition.
    """
    module = PredictionErrorModule()
    steady_entry = CommitmentLifecycleEntry(
        record_id="c-3",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.AGREE,
    )
    _ = _run_pe_turn(
        module,
        turn_index=1,
        commitment_snapshot=_commitment_snapshot(entries=(steady_entry,)),
    )
    turn_2 = _run_pe_turn(
        module,
        turn_index=2,
        commitment_snapshot=_commitment_snapshot(entries=(steady_entry,)),
    )
    # No transition \u2192 description must NOT mention alignment_transition.
    assert "alignment_transition" not in turn_2.error.description


# ---------------------------------------------------------------------------
# FollowupManager policy routing
# ---------------------------------------------------------------------------


def test_followup_manager_routes_defer_only_with_later_due_and_lower_priority() -> None:
    manager = FollowupManager(default_due_delay_ticks=90)
    current_tick = 100
    gentle_entry = CommitmentLifecycleEntry(
        record_id="gentle-commit",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.MODIFY,
        followup_policy=FollowupPolicy.GENTLE_CHECKIN,
    )
    defer_entry = CommitmentLifecycleEntry(
        record_id="defer-commit",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.REJECT,
        followup_policy=FollowupPolicy.DEFER_ONLY,
        last_outcome=CommitmentOutcomeKind.REJECTED,
        last_outcome_evidence="user rejected",
        last_outcome_at_turn=5,
    )
    produced = manager.ingest_commitment_lifecycle(
        lifecycle_entries=(gentle_entry, defer_entry),
        current_tick=current_tick,
    )
    assert len(produced) == 2
    by_record = {
        item.metadata.get("record_id"): item for item in produced
    }
    gentle_item = by_record["gentle-commit"]
    defer_item = by_record["defer-commit"]
    assert defer_item.due_at_tick > gentle_item.due_at_tick, (
        "DEFER_ONLY commitment must be scheduled later than GENTLE_CHECKIN"
    )
    assert defer_item.priority < gentle_item.priority, (
        "DEFER_ONLY commitment must have lower priority than GENTLE_CHECKIN"
    )
    assert defer_item.metadata["policy"] == "defer_only"
    assert gentle_item.metadata["policy"] == "gentle_checkin"


def test_followup_manager_skips_unknown_alignment_entries() -> None:
    """Freshly-observed commitments (UNKNOWN alignment, NOT_READY advocacy)
    should NOT produce a follow-up \u2014 they are not actionable yet.
    """
    manager = FollowupManager()
    fresh_entry = CommitmentLifecycleEntry(
        record_id="fresh-commit",
        advocacy_state=AdvocacyState.NOT_READY,
        alignment_state=AlignmentState.UNKNOWN,
    )
    produced = manager.ingest_commitment_lifecycle(
        lifecycle_entries=(fresh_entry,),
        current_tick=0,
    )
    assert produced == ()


def test_followup_manager_deduplicates_same_record_across_turns() -> None:
    manager = FollowupManager()
    entry = CommitmentLifecycleEntry(
        record_id="dup-commit",
        advocacy_state=AdvocacyState.PROPOSED,
        alignment_state=AlignmentState.REJECT,
        followup_policy=FollowupPolicy.DEFER_ONLY,
        last_outcome=CommitmentOutcomeKind.REJECTED,
        last_outcome_evidence="user rejected",
        last_outcome_at_turn=1,
    )
    first = manager.ingest_commitment_lifecycle(
        lifecycle_entries=(entry,), current_tick=10
    )
    second = manager.ingest_commitment_lifecycle(
        lifecycle_entries=(entry,), current_tick=20
    )
    assert len(first) == 1
    assert second == ()
    assert len(manager.pending) == 1


# ---------------------------------------------------------------------------
# CommitmentSnapshot aggregate counts
# ---------------------------------------------------------------------------


def test_commitment_snapshot_publishes_outcome_counts_via_entries() -> None:
    entries = (
        CommitmentLifecycleEntry(
            record_id="a",
            advocacy_state=AdvocacyState.PROPOSED,
            alignment_state=AlignmentState.AGREE,
            last_outcome=CommitmentOutcomeKind.COMPLETED,
            last_outcome_evidence="user agreed and moved on",
            last_outcome_at_turn=2,
        ),
        CommitmentLifecycleEntry(
            record_id="b",
            advocacy_state=AdvocacyState.PROPOSED,
            alignment_state=AlignmentState.REJECT,
            followup_policy=FollowupPolicy.DEFER_ONLY,
            last_outcome=CommitmentOutcomeKind.REJECTED,
            last_outcome_evidence="user said not now",
            last_outcome_at_turn=3,
        ),
        CommitmentLifecycleEntry(
            record_id="c",
            advocacy_state=AdvocacyState.READY,
            alignment_state=AlignmentState.UNKNOWN,
        ),
    )
    snapshot = _commitment_snapshot(entries=entries)
    assert snapshot.advocacy_proposed_count == 2
    assert snapshot.advocacy_ready_count == 1
    assert snapshot.alignment_agree_count == 1
    assert snapshot.alignment_reject_count == 1
    assert snapshot.followup_defer_only_count == 1
    assert snapshot.followup_gentle_count == 2
    # lifecycle_for must return the same per-entry data
    resolved = snapshot.lifecycle_for("b")
    assert resolved is not None
    assert resolved.alignment_state is AlignmentState.REJECT
    assert resolved.last_outcome is CommitmentOutcomeKind.REJECTED
    assert resolved.last_outcome_evidence == "user said not now"
