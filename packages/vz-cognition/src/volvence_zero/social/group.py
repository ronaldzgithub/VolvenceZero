"""Group owner (R20 / CP-18).

Group identity comes from the canonical conversational frame: the
``multi_party_identity`` owner publishes speaker / addressee / audience
scope derived from ``EnvironmentEvent.frame``, and this owner derives a
deterministic ``GroupIdentity`` from that membership whenever more than
two distinct participants share the frame. The same membership always
yields the same ``group_id``, so a recurring group keeps a persistent
identity across scenes/sessions (R14) — never guessed from text.

Group regime is runtime state persisted per group in the session-held
``SocialRecordStore`` (single writer: this module). Orchestrator-injected
groups / joint state remain supported for product wiring; frame-derived
groups are merged in, never duplicated.

Wiring stays SHADOW by default: ACTIVE promotion requires the CP-18
evidence gate (group-level PE signal not reducible to individual owners).
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social.record_store import SocialRecordStore
from volvence_zero.social_cognition import (
    GroupIdentity,
    GroupSnapshot,
    MultiPartyIdentitySnapshot,
    SocialPrediction,
    SocialPredictionKind,
    SocialScopeKind,
)


def frame_group_id(member_ids: tuple[str, ...]) -> str:
    """Deterministic persistent id for a frame-derived group (R14).

    Same membership set -> same id, independent of member ordering and of
    when the group is observed. The prefix marks provenance so consumers
    can distinguish frame-derived groups from orchestrator-injected ones.
    """

    return "frame-group:" + "+".join(sorted(member_ids))


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
        record_store: SocialRecordStore | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._groups = groups
        self._active_group_id = active_group_id
        self._joint_attention = joint_attention
        self._joint_commitments = joint_commitments
        self._group_regime_id = group_regime_id
        self._record_store = record_store

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[GroupSnapshot]:
        frame_groups = self._frame_derived_groups(upstream.get("multi_party_identity"))
        injected_ids = {group.group_id for group in self._groups}
        groups = self._groups + tuple(
            group for group in frame_groups if group.group_id not in injected_ids
        )
        active_group_id = self._active_group_id
        if active_group_id is None and frame_groups:
            active_group_id = frame_groups[0].group_id

        # R14: group regime is persistent runtime state. An explicit
        # orchestrator-supplied regime is recorded for the active group;
        # otherwise the persisted regime for that group is rehydrated so a
        # recurring group resumes its regime instead of resetting.
        group_regime_id = self._group_regime_id
        if self._record_store is not None and active_group_id is not None:
            if group_regime_id is not None:
                self._record_store.record_group_regime(
                    active_group_id, group_regime_id
                )
            else:
                group_regime_id = self._record_store.group_regime_for(
                    active_group_id
                )

        frame_note = (
            f" frame_groups={len(frame_groups)}" if frame_groups else ""
        )
        return self.publish(
            GroupSnapshot(
                groups=groups,
                active_group_id=active_group_id,
                joint_attention=self._joint_attention,
                joint_commitments=self._joint_commitments,
                group_regime_id=group_regime_id,
                active_predictions=self._active_predictions(
                    groups=groups,
                    active_group_id=active_group_id,
                    group_regime_id=group_regime_id,
                ),
                description=(
                    "R20 group owner: "
                    f"groups={len(groups)} "
                    f"joint_attention={len(self._joint_attention)} "
                    f"joint_commitments={len(self._joint_commitments)}."
                    f"{frame_note}"
                ),
            )
        )

    @staticmethod
    def _frame_derived_groups(
        identity_snapshot: Snapshot[Any] | None,
    ) -> tuple[GroupIdentity, ...]:
        """Derive group identity from canonical frame membership (CP-18).

        Membership = speaker + addressees + audience from the
        ``multi_party_identity`` owner's published scope. Fewer than three
        distinct participants is a dyad (or solo), not a group — the
        single-party compatibility snapshot never produces one.
        """

        if identity_snapshot is None:
            return ()
        value = identity_snapshot.value
        if not isinstance(value, MultiPartyIdentitySnapshot):
            # Disabled / placeholder upstream: no frame membership basis.
            return ()
        members = tuple(
            sorted(
                {
                    value.active_speaker_id,
                    *value.addressee_ids,
                    *value.audience_ids,
                }
            )
        )
        if len(members) < 3:
            return ()
        return (
            GroupIdentity(
                group_id=frame_group_id(members),
                member_ids=members,
                display_name=None,
                confidence=1.0,
                evidence=(
                    "frame:multi_party_identity",
                    f"speaker:{value.active_speaker_id}",
                ),
            ),
        )

    def _active_predictions(
        self,
        *,
        groups: tuple[GroupIdentity, ...],
        active_group_id: str | None,
        group_regime_id: str | None,
    ) -> tuple[SocialPrediction, ...]:
        if active_group_id is None:
            return ()
        group = next(
            (item for item in groups if item.group_id == active_group_id),
            None,
        )
        if group is None:
            return ()
        if not self._joint_commitments and not self._joint_attention:
            return ()
        predicted = (
            f"joint_commitments={len(self._joint_commitments)} "
            f"joint_attention={len(self._joint_attention)} "
            f"group_regime={group_regime_id or 'none'}"
        )
        evidence = (
            *(f"group_evidence:{item}" for item in group.evidence),
            *(f"joint_commitment:{item}" for item in self._joint_commitments),
            *(f"joint_attention:{item}" for item in self._joint_attention),
        )
        return (
            SocialPrediction(
                prediction_id=f"groups:{group.group_id}:durability:prediction",
                kind=SocialPredictionKind.GROUP_COMMITMENT_DURABILITY,
                scope_kind=SocialScopeKind.GROUP,
                scope_id=group.group_id,
                subject_ids=group.member_ids,
                audience_ids=group.member_ids,
                predicted_outcome=predicted,
                confidence=group.confidence,
                evidence=evidence or (f"group:{group.group_id}",),
            ),
        )


__all__ = ["GroupModule", "frame_group_id"]
