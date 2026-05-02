"""Group owner scaffold (R20).

Slice 2 publishes an empty SHADOW snapshot. Later slices will consume
role, common-ground, commitment, and open-loop state to learn group-level
continuity and joint commitments.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import GroupIdentity, GroupSnapshot


class GroupModule(RuntimeModule[GroupSnapshot]):
    slot_name = "groups"
    owner = "GroupModule"
    value_type = GroupSnapshot
    dependencies = (
        "multi_party_identity",
        "conversational_role",
        "common_ground",
        "commitment",
        "open_loop",
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        groups: tuple[GroupIdentity, ...] = (),
        active_group_id: str | None = None,
        joint_attention: tuple[str, ...] = (),
        joint_commitments: tuple[str, ...] = (),
        group_regime_id: str | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._groups = groups
        self._active_group_id = active_group_id
        self._joint_attention = joint_attention
        self._joint_commitments = joint_commitments
        self._group_regime_id = group_regime_id

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[GroupSnapshot]:
        del upstream
        return self.publish(
            GroupSnapshot(
                groups=self._groups,
                active_group_id=self._active_group_id,
                joint_attention=self._joint_attention,
                joint_commitments=self._joint_commitments,
                group_regime_id=self._group_regime_id,
                active_predictions=(),
                description=(
                    "R20 SHADOW scaffold: "
                    f"groups={len(self._groups)} "
                    f"joint_attention={len(self._joint_attention)} "
                    f"joint_commitments={len(self._joint_commitments)}."
                ),
            )
        )


__all__ = ["GroupModule"]
