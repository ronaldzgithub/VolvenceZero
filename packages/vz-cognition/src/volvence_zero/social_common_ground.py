"""Common-ground owner scaffold (R19).

Slice 2 publishes an empty SHADOW snapshot. Later slices will consume
role, identity, memory, and ToM state to build dyad/group common ground.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import CommonGroundAtom, CommonGroundSnapshot
from volvence_zero.social_common_ground_runtime import (
    CommonGroundProposal,
    LLMCommonGroundProposalRuntime,
)


class CommonGroundModule(RuntimeModule[CommonGroundSnapshot]):
    slot_name = "common_ground"
    owner = "CommonGroundModule"
    value_type = CommonGroundSnapshot
    dependencies = (
        "multi_party_identity",
        "conversational_role",
        "belief_about_other",
        "memory",
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        dyad_atoms: tuple[CommonGroundAtom, ...] = (),
        group_atoms: tuple[CommonGroundAtom, ...] = (),
        proposal_runtime: LLMCommonGroundProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._dyad_atoms = dyad_atoms
        self._group_atoms = group_atoms
        self._proposal_runtime = proposal_runtime
        self._user_input = user_input
        self._turn_index = turn_index

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[CommonGroundSnapshot]:
        del upstream
        runtime_atoms = self._runtime_atoms()
        dyad_atoms = (
            *self._dyad_atoms,
            *(atom for atom in runtime_atoms if atom.scope_kind.value == "dyad"),
        )
        group_atoms = (
            *self._group_atoms,
            *(atom for atom in runtime_atoms if atom.scope_kind.value == "group"),
        )
        control_signal = _mean_confidence(runtime_atoms)
        return self.publish(
            CommonGroundSnapshot(
                dyad_atoms=dyad_atoms,
                group_atoms=group_atoms,
                active_predictions=(),
                control_signal=control_signal,
                description=(
                    "R19 SHADOW scaffold: "
                    f"dyad_atoms={len(dyad_atoms)} group_atoms={len(group_atoms)}."
                ),
            )
        )

    def _runtime_atoms(self) -> tuple[CommonGroundAtom, ...]:
        if self._proposal_runtime is None:
            return ()
        batch = self._proposal_runtime.propose(
            user_input=self._user_input,
            turn_index=self._turn_index,
        )
        return tuple(
            _atom_from_proposal(proposal=proposal, turn_index=self._turn_index)
            for proposal in batch.proposals
        )


def _atom_from_proposal(
    *,
    proposal: CommonGroundProposal,
    turn_index: int,
) -> CommonGroundAtom:
    return CommonGroundAtom(
        atom_id=f"cg:{proposal.scope_kind.value}:{proposal.scope_id}:{turn_index}",
        scope_id=proposal.scope_id,
        scope_kind=proposal.scope_kind,
        summary=proposal.summary,
        recursion_depth=proposal.recursion_depth,
        confidence=proposal.confidence,
        accepted_by_ids=proposal.accepted_by_ids,
        evidence=proposal.evidence,
    )


def _mean_confidence(atoms: tuple[CommonGroundAtom, ...]) -> float:
    if not atoms:
        return 0.0
    return sum(atom.confidence for atom in atoms) / len(atoms)


__all__ = ["CommonGroundModule"]
