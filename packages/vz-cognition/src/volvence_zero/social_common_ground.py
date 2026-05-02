"""Common-ground owner scaffold (R19).

Slice 2 publishes an empty SHADOW snapshot. Later slices will consume
role, identity, memory, and ToM state to build dyad/group common ground.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import CommonGroundAtom, CommonGroundSnapshot


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
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._dyad_atoms = dyad_atoms
        self._group_atoms = group_atoms

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[CommonGroundSnapshot]:
        del upstream
        return self.publish(
            CommonGroundSnapshot(
                dyad_atoms=self._dyad_atoms,
                group_atoms=self._group_atoms,
                active_predictions=(),
                control_signal=0.0,
                description=(
                    "R19 SHADOW scaffold: "
                    f"dyad_atoms={len(self._dyad_atoms)} group_atoms={len(self._group_atoms)}."
                ),
            )
        )


__all__ = ["CommonGroundModule"]
