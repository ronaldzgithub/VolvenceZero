"""Theory-of-Mind owner scaffolds (R17).

Slice 2 publishes empty SHADOW snapshots for the four ToM owners. These
modules establish ownership and wiring without changing response assembly,
planner, or renderer behavior.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.substrate import SubstrateSnapshot


class _OtherMindOwnerModule(RuntimeModule[Any]):
    record_kind: OtherMindRecordKind
    snapshot_type: type[Any]
    empty_description: str
    dependencies = ("substrate", "memory", "multi_party_identity")
    default_wiring_level = WiringLevel.SHADOW
    min_proposal_confidence = 0.50

    def __init__(
        self,
        *,
        proposal_runtime: SemanticProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._proposal_runtime = proposal_runtime
        self._user_input = user_input
        self._turn_index = turn_index

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[Any]:
        records: tuple[OtherMindRecord, ...] = ()
        control_signal = 0.0
        if self._proposal_runtime is not None:
            substrate_snapshot = upstream.get("substrate")
            memory_snapshot = upstream.get("memory")
            batch = self._proposal_runtime.propose(
                target_slot=self.slot_name,
                user_input=self._user_input,
                substrate_snapshot=(
                    substrate_snapshot.value
                    if substrate_snapshot is not None
                    and isinstance(substrate_snapshot.value, SubstrateSnapshot)
                    else None
                ),
                memory_snapshot=(
                    memory_snapshot.value
                    if memory_snapshot is not None
                    and isinstance(memory_snapshot.value, MemorySnapshot)
                    else None
                ),
                previous_snapshot=None,
                turn_index=self._turn_index,
            )
            proposals = tuple(
                proposal
                for proposal in batch.proposals
                if proposal.target_slot == self.slot_name
                and proposal.confidence >= self.min_proposal_confidence
            )
            records = tuple(
                _record_from_proposal(
                    proposal=proposal,
                    kind=self.record_kind,
                    turn_index=self._turn_index,
                )
                for proposal in proposals
            )
            control_signal = _mean_control_signal(proposals)
        return self.publish(self._snapshot(records=records, control_signal=control_signal))

    def _snapshot(
        self,
        *,
        records: tuple[OtherMindRecord, ...],
        control_signal: float,
    ) -> Any:
        return self.snapshot_type(
            records=records,
            active_predictions=(),
            control_signal=control_signal,
            description=(
                self.empty_description
                if not records
                else f"{self.owner} published explicit records={len(records)}."
            ),
        )


class BeliefAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "belief_about_other"
    owner = "BeliefAboutOtherModule"
    value_type = BeliefAboutOtherSnapshot
    record_kind = OtherMindRecordKind.BELIEF
    snapshot_type = BeliefAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no belief-about-other records yet."


class IntentAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "intent_about_other"
    owner = "IntentAboutOtherModule"
    value_type = IntentAboutOtherSnapshot
    record_kind = OtherMindRecordKind.INTENT
    snapshot_type = IntentAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no intent-about-other records yet."


class FeelingAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "feeling_about_other"
    owner = "FeelingAboutOtherModule"
    value_type = FeelingAboutOtherSnapshot
    record_kind = OtherMindRecordKind.FEELING
    snapshot_type = FeelingAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no feeling-about-other records yet."


class PreferenceAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "preference_about_other"
    owner = "PreferenceAboutOtherModule"
    value_type = PreferenceAboutOtherSnapshot
    record_kind = OtherMindRecordKind.PREFERENCE
    snapshot_type = PreferenceAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no preference-about-other records yet."


def _record_from_proposal(
    *,
    proposal: SemanticProposal,
    kind: OtherMindRecordKind,
    turn_index: int,
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=proposal.proposal_id,
        interlocutor_id="primary",
        kind=kind,
        summary=proposal.summary,
        detail=proposal.detail,
        confidence=proposal.confidence,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=turn_index,
        prediction_error_refs=(),
        evidence=proposal.evidence,
    )


def _mean_control_signal(proposals: tuple[SemanticProposal, ...]) -> float:
    if not proposals:
        return 0.0
    return sum(proposal.control_signal for proposal in proposals) / len(proposals)


__all__ = [
    "BeliefAboutOtherModule",
    "FeelingAboutOtherModule",
    "IntentAboutOtherModule",
    "PreferenceAboutOtherModule",
]
