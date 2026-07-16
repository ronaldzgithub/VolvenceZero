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

G1 (group-level PE settlement): each turn this owner settles the
prior turn's GROUP_COMMITMENT_DURABILITY predictions against this
turn's observed joint state (same ``settle_pending_predictions``
machinery the ToM / common-ground owners use — semantic similarity of
typed summaries, never keywords), publishes the settled errors on the
snapshot (forwarded by ``SocialPredictionErrorModule``), and folds the
outcomes into a bounded learned per-group ``durability score`` that
becomes the confidence of the next durability prediction. That closes
the loop: broken joint commitments lower future durability confidence,
kept ones raise it — the group-level PE the CP-18 evidence gate needs.

Wiring stays SHADOW by default: ACTIVE promotion requires the CP-18
evidence gate (group-level PE signal not reducible to individual owners).
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social.record_store import (
    PendingSocialPrediction,
    SocialRecordStore,
    settle_pending_predictions,
)
from volvence_zero.social_cognition import (
    GroupIdentity,
    GroupSnapshot,
    MultiPartyIdentitySnapshot,
    SocialPrediction,
    SocialPredictionError,
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
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._groups = groups
        self._active_group_id = active_group_id
        self._joint_attention = joint_attention
        self._joint_commitments = joint_commitments
        self._group_regime_id = group_regime_id
        self._record_store = record_store
        self._turn_index = turn_index

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

        settled_errors, durability_score = self._settle_group_predictions(
            active_group_id=active_group_id,
            group_regime_id=group_regime_id,
        )
        active_predictions = self._active_predictions(
            groups=groups,
            active_group_id=active_group_id,
            group_regime_id=group_regime_id,
            durability_score=durability_score,
        )
        self._park_pending_predictions(active_predictions)

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
                active_predictions=active_predictions,
                description=(
                    "R20 group owner: "
                    f"groups={len(groups)} "
                    f"joint_attention={len(self._joint_attention)} "
                    f"joint_commitments={len(self._joint_commitments)} "
                    f"settled={len(settled_errors)} "
                    f"durability={durability_score:.2f}."
                    f"{frame_note}"
                ),
                settled_errors=settled_errors,
                group_durability_score=durability_score,
            )
        )

    def _observed_state_summary(self, group_regime_id: str | None) -> str:
        """Typed summary of this turn's joint state for one group.

        Identical shape to ``predicted_outcome`` in the durability
        prediction so settlement compares like with like: an unchanged
        joint state confirms the prediction, a dropped commitment /
        regime flip disconfirms it (semantic similarity, no keywords
        driving behaviour — the summary is typed owner state).
        """

        return (
            f"joint_commitments={len(self._joint_commitments)} "
            f"joint_attention={len(self._joint_attention)} "
            f"group_regime={group_regime_id or 'none'}"
        )

    def _settle_group_predictions(
        self,
        *,
        active_group_id: str | None,
        group_regime_id: str | None,
    ) -> tuple[tuple[SocialPredictionError, ...], float]:
        """G1: settle prior-turn durability predictions, update the score.

        Without a store the owner stays stateless (original scaffold
        behavior: no settlement, uninformed 0.5 prior).
        """

        store = self._record_store
        if store is None:
            return ((), 0.5)
        evidence_by_scope: dict[str, tuple[tuple[str, str], ...]] = {}
        if active_group_id is not None:
            evidence_by_scope[active_group_id] = (
                (
                    f"groups:{active_group_id}:observed:{self._turn_index}",
                    self._observed_state_summary(group_regime_id),
                ),
            )
        result = settle_pending_predictions(
            pending=store.pending_group_predictions,
            new_evidence_by_scope=evidence_by_scope,
            turn_index=self._turn_index,
            owner=self.owner,
            similarity=store.similarity,
        )
        store.set_pending_group_predictions(result.still_pending)
        for error in result.settled_errors:
            store.apply_group_settlement(error.scope_id, error.outcome)
        durability = (
            store.group_durability_for(active_group_id)
            if active_group_id is not None
            else 0.5
        )
        return (result.settled_errors, durability)

    def _park_pending_predictions(
        self, active_predictions: tuple[SocialPrediction, ...]
    ) -> None:
        """Park this turn's predictions for next-turn settlement.

        Deduped by prediction id against what settlement left pending
        so a re-issued prediction refreshes rather than duplicates.
        """

        store = self._record_store
        if store is None or not active_predictions:
            return
        pending_by_id = {
            entry.prediction.prediction_id: entry
            for entry in store.pending_group_predictions
        }
        for prediction in active_predictions:
            pending_by_id[prediction.prediction_id] = PendingSocialPrediction(
                prediction=prediction,
                source_record_id=prediction.scope_id,
                issued_turn=self._turn_index,
            )
        store.set_pending_group_predictions(tuple(pending_by_id.values()))

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
        durability_score: float = 0.5,
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
        predicted = self._observed_state_summary(group_regime_id)
        evidence = (
            *(f"group_evidence:{item}" for item in group.evidence),
            *(f"joint_commitment:{item}" for item in self._joint_commitments),
            *(f"joint_attention:{item}" for item in self._joint_attention),
        )
        # G1: prediction confidence is the learned durability score
        # bounded by the group-identity confidence — the PE settlement
        # loop shapes future predictions instead of a static constant.
        confidence = max(0.0, min(group.confidence, durability_score))
        return (
            SocialPrediction(
                prediction_id=f"groups:{group.group_id}:durability:prediction",
                kind=SocialPredictionKind.GROUP_COMMITMENT_DURABILITY,
                scope_kind=SocialScopeKind.GROUP,
                scope_id=group.group_id,
                subject_ids=group.member_ids,
                audience_ids=group.member_ids,
                predicted_outcome=predicted,
                confidence=confidence,
                evidence=evidence or (f"group:{group.group_id}",),
            ),
        )


__all__ = ["GroupModule", "frame_group_id"]
