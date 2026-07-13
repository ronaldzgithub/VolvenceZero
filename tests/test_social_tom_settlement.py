"""W1.C acceptance: ToM / common-ground cross-turn settlement (CP-16/17 core).

Proves: (a) the promote/retire decision table, (b) prior-turn ToM
predictions settle against this turn's typed evidence and drive
ACTIVE -> CONTESTED -> RETIRED transitions, (c) the four ToM owners do
not cross-contaminate under a false-belief / preference-conflict double
probe, (d) settled outcomes lift into ``SocialPredictionError`` via the
error owner without reconstruction, and (e) common-ground predictions
settle the same way.
"""

from __future__ import annotations

import asyncio

from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social import (
    BeliefAboutOtherModule,
    CommonGroundModule,
    PreferenceAboutOtherModule,
    SocialPredictionErrorModule,
    SocialRecordStore,
    apply_outcome_to_record,
    settle_pending_predictions,
)
from volvence_zero.social.record_store import PendingSocialPrediction
from volvence_zero.social_cognition import (
    CommonGroundAtom,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    SocialPrediction,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialScopeKind,
)
from volvence_zero.runtime import Snapshot, WiringLevel


def _equality_similarity(left: str, right: str) -> float:
    return 1.0 if left == right else 0.0


class _ScriptedRuntime(SemanticProposalRuntime):
    """Emits one OBSERVE proposal for a single slot with a fixed summary."""

    runtime_id = "scripted-tom"

    def __init__(self, *, slot: str, summary: str, turn_index: int) -> None:
        self._slot = slot
        self._summary = summary
        self._turn_index = turn_index

    def propose(self, *, target_slot, user_input, substrate_snapshot, memory_snapshot, previous_snapshot, turn_index):
        del user_input, substrate_snapshot, memory_snapshot, previous_snapshot
        if target_slot != self._slot:
            return SemanticProposalBatch(
                proposals=(),
                runtime_id=self.runtime_id,
                schema_version=1,
                description="off-slot",
            )
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=f"{self._slot}:scripted:{turn_index}",
                    target_slot=self._slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=self._summary,
                    detail=f"scripted detail: {self._summary}",
                    confidence=0.8,
                    evidence=f"scripted evidence for {self._summary}",
                    control_signal=0.2,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description="scripted",
        )


def _record(record_id: str, *, status=OtherMindRecordStatus.ACTIVE, confidence=0.8) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=record_id,
        interlocutor_id="primary",
        kind=OtherMindRecordKind.BELIEF,
        summary="user believes the meeting is on friday",
        detail="scripted",
        confidence=confidence,
        status=status,
        source_turn=1,
        prediction_error_refs=(),
        evidence="scripted",
    )


def test_promote_retire_decision_table() -> None:
    active = _record("r1")
    confirmed = apply_outcome_to_record(
        active, SocialPredictionOutcome.CONFIRMED, error_id="e1"
    )
    assert confirmed.status is OtherMindRecordStatus.ACTIVE
    assert confirmed.confidence > active.confidence
    assert "e1" in confirmed.prediction_error_refs

    contested = apply_outcome_to_record(
        active, SocialPredictionOutcome.DISCONFIRMED, error_id="e2"
    )
    assert contested.status is OtherMindRecordStatus.CONTESTED
    assert contested.confidence < active.confidence

    retired = apply_outcome_to_record(
        contested, SocialPredictionOutcome.DISCONFIRMED, error_id="e3"
    )
    assert retired.status is OtherMindRecordStatus.RETIRED

    repromoted = apply_outcome_to_record(
        contested, SocialPredictionOutcome.CONFIRMED, error_id="e4"
    )
    assert repromoted.status is OtherMindRecordStatus.ACTIVE

    stale = apply_outcome_to_record(active, SocialPredictionOutcome.STALE)
    assert stale.status is active.status
    assert stale.confidence == active.confidence


def _pending(prediction_summary: str, *, issued_turn: int = 1) -> PendingSocialPrediction:
    return PendingSocialPrediction(
        prediction=SocialPrediction(
            prediction_id="belief_about_other:r1:prediction",
            kind=SocialPredictionKind.BELIEF_ABOUT_OTHER,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id="primary",
            subject_ids=("primary",),
            audience_ids=("self",),
            predicted_outcome=prediction_summary,
            confidence=0.8,
            evidence=("tom_record:r1",),
        ),
        source_record_id="r1",
        issued_turn=issued_turn,
    )


def test_settlement_confirms_disconfirms_and_stales() -> None:
    pending = (_pending("user believes the meeting is on friday"),)
    confirm = settle_pending_predictions(
        pending=pending,
        new_evidence_by_scope={
            "primary": (("r2", "user believes the meeting is on friday"),)
        },
        turn_index=2,
        owner="BeliefAboutOtherModule",
        similarity=_equality_similarity,
    )
    assert len(confirm.settled_errors) == 1
    assert confirm.settled_errors[0].outcome is SocialPredictionOutcome.CONFIRMED
    assert confirm.outcomes_by_record[0][0] == "r1"

    disconfirm = settle_pending_predictions(
        pending=pending,
        new_evidence_by_scope={
            "primary": (("r3", "user now says the meeting moved to monday"),)
        },
        turn_index=2,
        owner="BeliefAboutOtherModule",
        similarity=_equality_similarity,
    )
    assert disconfirm.settled_errors[0].outcome is SocialPredictionOutcome.DISCONFIRMED
    assert disconfirm.settled_errors[0].magnitude == 1.0

    # No evidence for the scope: stays pending until the age bound.
    quiet = settle_pending_predictions(
        pending=pending,
        new_evidence_by_scope={},
        turn_index=2,
        owner="BeliefAboutOtherModule",
        similarity=_equality_similarity,
    )
    assert quiet.settled_errors == ()
    assert len(quiet.still_pending) == 1

    stale = settle_pending_predictions(
        pending=pending,
        new_evidence_by_scope={},
        turn_index=20,
        owner="BeliefAboutOtherModule",
        similarity=_equality_similarity,
    )
    assert stale.settled_errors[0].outcome is SocialPredictionOutcome.STALE


async def _run_owner(module_type, *, store, slot, summary, turn_index):
    module = module_type(
        proposal_runtime=_ScriptedRuntime(slot=slot, summary=summary, turn_index=turn_index),
        user_input="probe input",
        turn_index=turn_index,
        wiring_level=WiringLevel.ACTIVE,
        record_store=store,
    )
    return (await module.process({})).value


def test_false_belief_probe_contests_then_retires_without_cross_talk() -> None:
    """False-belief + preference-conflict double probe (plan W1.C acceptance)."""

    store = SocialRecordStore(similarity=_equality_similarity)

    async def _probe() -> None:
        # Turn 1: belief and preference owners each record one claim.
        belief_1 = await _run_owner(
            BeliefAboutOtherModule,
            store=store,
            slot="belief_about_other",
            summary="user believes the meeting is on friday",
            turn_index=1,
        )
        pref_1 = await _run_owner(
            PreferenceAboutOtherModule,
            store=store,
            slot="preference_about_other",
            summary="user prefers written summaries",
            turn_index=1,
        )
        assert len(belief_1.records) == 1
        assert len(pref_1.records) == 1
        assert belief_1.settled_errors == ()

        # Turn 2: belief evidence contradicts turn 1; preference confirms.
        belief_2 = await _run_owner(
            BeliefAboutOtherModule,
            store=store,
            slot="belief_about_other",
            summary="user learned the meeting moved to monday",
            turn_index=2,
        )
        pref_2 = await _run_owner(
            PreferenceAboutOtherModule,
            store=store,
            slot="preference_about_other",
            summary="user prefers written summaries",
            turn_index=2,
        )
        assert len(belief_2.settled_errors) == 1
        assert belief_2.settled_errors[0].outcome is SocialPredictionOutcome.DISCONFIRMED
        assert belief_2.settled_errors[0].kind is SocialPredictionKind.BELIEF_ABOUT_OTHER
        old_belief = next(
            r for r in belief_2.records if r.source_turn == 1
        )
        assert old_belief.status is OtherMindRecordStatus.CONTESTED
        # Contested records stop asserting public predictions.
        assert all(
            p.predicted_outcome != old_belief.summary
            for p in belief_2.active_predictions
        )
        # Preference owner settled CONFIRMED — and only its own kind.
        assert len(pref_2.settled_errors) == 1
        assert pref_2.settled_errors[0].outcome is SocialPredictionOutcome.CONFIRMED
        assert pref_2.settled_errors[0].kind is SocialPredictionKind.PREFERENCE_ABOUT_OTHER
        old_pref = next(r for r in pref_2.records if r.source_turn == 1)
        assert old_pref.status is OtherMindRecordStatus.ACTIVE
        assert old_pref.confidence > 0.8

        # Turn 3: a second belief contradiction retires the contested record.
        belief_3 = await _run_owner(
            BeliefAboutOtherModule,
            store=store,
            slot="belief_about_other",
            summary="user reconfirmed the meeting is on monday now",
            turn_index=3,
        )
        retired = next(r for r in belief_3.records if r.source_turn == 1)
        assert retired.status is OtherMindRecordStatus.RETIRED
        assert retired.prediction_error_refs  # PE lineage attached

        # Cross-talk check: the preference slot never saw belief records.
        assert all(
            r.kind is OtherMindRecordKind.PREFERENCE
            for r in store.tom_records("preference_about_other")
        )
        assert all(
            r.kind is OtherMindRecordKind.BELIEF
            for r in store.tom_records("belief_about_other")
        )

    asyncio.run(_probe())


def test_error_owner_forwards_tom_settled_errors() -> None:
    store = SocialRecordStore(similarity=_equality_similarity)

    async def _run() -> None:
        await _run_owner(
            BeliefAboutOtherModule,
            store=store,
            slot="belief_about_other",
            summary="user believes the meeting is on friday",
            turn_index=1,
        )
        belief_2 = await _run_owner(
            BeliefAboutOtherModule,
            store=store,
            slot="belief_about_other",
            summary="user learned the meeting moved to monday",
            turn_index=2,
        )
        error_module = SocialPredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
        error_snapshot = (
            await error_module.process(
                {
                    "belief_about_other": Snapshot(
                        slot_name="belief_about_other",
                        owner="BeliefAboutOtherModule",
                        version=2,
                        timestamp_ms=0,
                        value=belief_2,
                    )
                }
            )
        ).value
        assert len(error_snapshot.errors) == 1
        forwarded = error_snapshot.errors[0]
        assert forwarded.owner == "BeliefAboutOtherModule"
        assert forwarded.outcome is SocialPredictionOutcome.DISCONFIRMED
        assert forwarded == belief_2.settled_errors[0]

    asyncio.run(_run())


def _atom(atom_id: str, summary: str) -> CommonGroundAtom:
    return CommonGroundAtom(
        atom_id=atom_id,
        scope_id="primary+self",
        scope_kind=SocialScopeKind.DYAD,
        summary=summary,
        recursion_depth=1,
        confidence=0.7,
        accepted_by_ids=("primary", "self"),
        evidence=("probe evidence",),
    )


def test_common_ground_settles_prior_predictions_against_new_atoms() -> None:
    store = SocialRecordStore(similarity=_equality_similarity)

    async def _run() -> None:
        module_1 = CommonGroundModule(
            dyad_atoms=(_atom("cg-1", "we agreed to review the tide tables"),),
            turn_index=1,
            wiring_level=WiringLevel.ACTIVE,
            record_store=store,
        )
        snap_1 = (await module_1.process({})).value
        assert snap_1.settled_errors == ()
        assert len(store.pending_common_ground_predictions) == 1

        # Turn 2: clarification/repair evidence contradicts the atom.
        module_2 = CommonGroundModule(
            dyad_atoms=(_atom("cg-2", "actually we postponed the tide review"),),
            turn_index=2,
            wiring_level=WiringLevel.ACTIVE,
            record_store=store,
        )
        snap_2 = (await module_2.process({})).value
        assert len(snap_2.settled_errors) == 1
        settled = snap_2.settled_errors[0]
        assert settled.outcome is SocialPredictionOutcome.DISCONFIRMED
        assert settled.kind is SocialPredictionKind.COMMON_GROUND_RESOLUTION
        assert settled.owner == "CommonGroundModule"
        # Cross-turn atom window retained both atoms.
        assert {a.atom_id for a in snap_2.dyad_atoms} == {"cg-1", "cg-2"}

    asyncio.run(_run())
