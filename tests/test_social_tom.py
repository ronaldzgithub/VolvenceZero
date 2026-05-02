from __future__ import annotations

import asyncio

from volvence_zero.memory import MemoryModule, MemoryStore
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    OtherMindRecordKind,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.social_identity import MultiPartyIdentityModule
from volvence_zero.social_tom import BeliefAboutOtherModule, PreferenceAboutOtherModule
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter, SubstrateModule


class ExplicitToMRuntime(SemanticProposalRuntime):
    runtime_id = "explicit-tom-test"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=f"{target_slot}:explicit:{turn_index}",
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=f"{target_slot}:explicit-record",
                    detail=user_input or "explicit ToM detail",
                    confidence=0.81,
                    evidence="explicit test proposal",
                    control_signal=0.37,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"explicit proposal for {target_slot}",
        )


def _substrate() -> SubstrateModule:
    return SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="tom-proposal-model",
            feature_surface=(FeatureSignal(name="tom_signal", values=(0.5,), source="test"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )


def test_explicit_tom_proposal_populates_belief_owner_only() -> None:
    runtime = ExplicitToMRuntime()
    belief = BeliefAboutOtherModule(
        proposal_runtime=runtime,
        user_input="Alice believes the meeting is tomorrow.",
        turn_index=4,
        wiring_level=WiringLevel.ACTIVE,
    )
    preference = PreferenceAboutOtherModule(wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(
        propagate(
            [
                _substrate(),
                MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
                MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
                belief,
                preference,
            ],
            session_id="tom-proposal-session",
            wave_id="tom-proposal-wave",
        )
    )

    belief_snapshot = result["belief_about_other"].value
    preference_snapshot = result["preference_about_other"].value
    assert isinstance(belief_snapshot, BeliefAboutOtherSnapshot)
    assert isinstance(preference_snapshot, PreferenceAboutOtherSnapshot)
    assert len(belief_snapshot.records) == 1
    assert belief_snapshot.records[0].kind is OtherMindRecordKind.BELIEF
    assert belief_snapshot.records[0].interlocutor_id == "primary"
    assert belief_snapshot.control_signal == 0.37
    assert preference_snapshot.records == ()


def test_explicit_tom_proposal_populates_preference_as_preference_kind() -> None:
    runtime = ExplicitToMRuntime()
    preference = PreferenceAboutOtherModule(
        proposal_runtime=runtime,
        user_input="Alice prefers slow planning.",
        turn_index=5,
        wiring_level=WiringLevel.ACTIVE,
    )

    result = asyncio.run(
        propagate(
            [
                _substrate(),
                MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
                MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
                preference,
            ],
            session_id="tom-preference-session",
            wave_id="tom-preference-wave",
        )
    )

    preference_snapshot = result["preference_about_other"].value
    assert isinstance(preference_snapshot, PreferenceAboutOtherSnapshot)
    assert len(preference_snapshot.records) == 1
    assert preference_snapshot.records[0].kind is OtherMindRecordKind.PREFERENCE


def test_false_belief_and_preference_conflict_probe_keeps_tom_owners_separate() -> None:
    runtime = ExplicitToMRuntime()
    belief = BeliefAboutOtherModule(
        proposal_runtime=runtime,
        user_input="Alice believes the meeting is tomorrow, but the calendar says today.",
        turn_index=6,
        wiring_level=WiringLevel.ACTIVE,
    )
    preference = PreferenceAboutOtherModule(
        proposal_runtime=runtime,
        user_input="Alice prefers slow planning, but today asks for one direct step.",
        turn_index=6,
        wiring_level=WiringLevel.ACTIVE,
    )

    result = asyncio.run(
        propagate(
            [
                _substrate(),
                MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
                MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
                belief,
                preference,
            ],
            session_id="tom-separation-probe",
            wave_id="tom-separation-wave",
        )
    )

    belief_snapshot = result["belief_about_other"].value
    preference_snapshot = result["preference_about_other"].value
    assert isinstance(belief_snapshot, BeliefAboutOtherSnapshot)
    assert isinstance(preference_snapshot, PreferenceAboutOtherSnapshot)
    assert len(belief_snapshot.records) == 1
    assert len(preference_snapshot.records) == 1
    assert belief_snapshot.records[0].kind is OtherMindRecordKind.BELIEF
    assert preference_snapshot.records[0].kind is OtherMindRecordKind.PREFERENCE
    assert belief_snapshot.records[0].record_id.startswith("belief_about_other:")
    assert preference_snapshot.records[0].record_id.startswith("preference_about_other:")


def test_response_assembly_surfaces_tom_owner_counts_diagnostically() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                belief_about_other=WiringLevel.ACTIVE,
                preference_about_other=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="tom-counts-model",
                feature_surface=(FeatureSignal(name="tom_counts", values=(0.5,), source="test"),),
            ),
            user_input="Alice believes one thing and prefers another.",
            tom_proposal_runtime=ExplicitToMRuntime(),
            session_id="tom-counts-session",
            wave_id="tom-counts-wave",
            turn_index=7,
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    assert counts["belief_about_other"] == 1
    assert counts["preference_about_other"] == 1
    assert "belief_about_other" not in response_assembly.semantic_residue_summary
    assert response_assembly.expression_intent
