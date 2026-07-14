from __future__ import annotations

import asyncio

from volvence_zero.memory import MemoryModule, MemoryStore
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.social import CommonGroundModule
from volvence_zero.social_cognition import GroupIdentity, GroupSnapshot, SocialPredictionKind
from volvence_zero.social import GroupModule
from volvence_zero.social import MultiPartyIdentityModule
from volvence_zero.social import ConversationalRoleModule
from volvence_zero.social import BeliefAboutOtherModule
from volvence_zero.social import SocialPredictionAggregateModule
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter, SubstrateModule


def _substrate() -> SubstrateModule:
    return SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="group-test-model",
            feature_surface=(FeatureSignal(name="group_signal", values=(0.5,), source="test"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )


def _base_modules() -> list[object]:
    return [
        _substrate(),
        MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
        MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
        ConversationalRoleModule(wiring_level=WiringLevel.ACTIVE),
        BeliefAboutOtherModule(wiring_level=WiringLevel.ACTIVE),
        CommonGroundModule(wiring_level=WiringLevel.ACTIVE),
    ]


def test_group_module_publishes_empty_shadow_scaffold_by_default() -> None:
    result = asyncio.run(
        propagate(
            [*_base_modules(), GroupModule(wiring_level=WiringLevel.ACTIVE)],
            session_id="group-empty-session",
            wave_id="group-empty-wave",
        )
    )

    snapshot = result["groups"].value
    assert isinstance(snapshot, GroupSnapshot)
    assert snapshot.groups == ()
    assert snapshot.active_group_id is None
    assert snapshot.joint_attention == ()
    assert snapshot.joint_commitments == ()
    assert snapshot.group_regime_id is None
    assert snapshot.active_predictions == ()


def test_group_module_publishes_explicit_group_state() -> None:
    group = GroupIdentity(
        group_id="group:launch",
        member_ids=("alice", "bob", "carol"),
        display_name="Launch group",
        confidence=0.82,
        evidence=("host membership",),
    )
    result = asyncio.run(
        propagate(
            [
                *_base_modules(),
                GroupModule(
                    groups=(group,),
                    active_group_id="group:launch",
                    joint_attention=("launch-plan",),
                    joint_commitments=("commitment:ship",),
                    group_regime_id="problem_solving",
                    wiring_level=WiringLevel.ACTIVE,
                ),
            ],
            session_id="group-explicit-session",
            wave_id="group-explicit-wave",
        )
    )

    snapshot = result["groups"].value
    assert isinstance(snapshot, GroupSnapshot)
    assert snapshot.groups == (group,)
    assert snapshot.active_group_id == "group:launch"
    assert snapshot.joint_attention == ("launch-plan",)
    assert snapshot.joint_commitments == ("commitment:ship",)
    assert snapshot.group_regime_id == "problem_solving"
    assert len(snapshot.active_predictions) == 1
    prediction = snapshot.active_predictions[0]
    assert prediction.kind is SocialPredictionKind.GROUP_COMMITMENT_DURABILITY
    assert prediction.scope_id == "group:launch"
    assert prediction.subject_ids == group.member_ids
    assert "groups=1 joint_attention=1 joint_commitments=1" in snapshot.description


def test_social_prediction_aggregate_forwards_group_predictions() -> None:
    group = GroupIdentity(
        group_id="group:launch",
        member_ids=("alice", "bob", "carol"),
        display_name="Launch group",
        confidence=0.82,
        evidence=("host membership",),
    )
    result = asyncio.run(
        propagate(
            [
                *_base_modules(),
                GroupModule(
                    groups=(group,),
                    active_group_id="group:launch",
                    joint_attention=("launch-plan",),
                    joint_commitments=("commitment:ship",),
                    group_regime_id="problem_solving",
                    wiring_level=WiringLevel.ACTIVE,
                ),
                SocialPredictionAggregateModule(wiring_level=WiringLevel.ACTIVE),
            ],
            session_id="group-aggregate-session",
            wave_id="group-aggregate-wave",
        )
    )

    aggregate = result["social_prediction"].value
    predictions = tuple(
        prediction
        for prediction in aggregate.predictions
        if prediction.kind is SocialPredictionKind.GROUP_COMMITMENT_DURABILITY
    )
    assert len(predictions) == 1
    assert predictions[0].prediction_id == "groups:group:launch:durability:prediction"


def _identity_snapshot_envelope(
    *,
    speaker: str,
    addressees: tuple[str, ...],
    audience: tuple[str, ...],
) -> object:
    from volvence_zero.runtime import Snapshot
    from volvence_zero.social_cognition import (
        InterlocutorIdentity,
        MultiPartyIdentitySnapshot,
    )

    all_ids = tuple(dict.fromkeys((speaker, *addressees, *audience)))
    value = MultiPartyIdentitySnapshot(
        active_speaker_id=speaker,
        addressee_ids=addressees,
        subject_ids=addressees,
        audience_ids=audience,
        interlocutors=tuple(
            InterlocutorIdentity(
                interlocutor_id=identity_id,
                display_name=identity_id,
            )
            for identity_id in all_ids
        ),
        identity_predictions=(),
        description="synthetic multi-party frame for group tests",
    )
    return Snapshot(
        slot_name="multi_party_identity",
        owner="MultiPartyIdentityModule",
        version=1,
        timestamp_ms=1,
        value=value,
    )


async def test_group_identity_derives_from_canonical_frame_membership() -> None:
    """CP-18: >=3 distinct frame participants -> deterministic frame group."""
    from volvence_zero.social.group import frame_group_id

    upstream = {
        "multi_party_identity": _identity_snapshot_envelope(
            speaker="alice",
            addressees=("self",),
            audience=("self", "bob"),
        )
    }
    module = GroupModule(wiring_level=WiringLevel.ACTIVE)
    snapshot = (await module.process(upstream)).value
    assert isinstance(snapshot, GroupSnapshot)
    assert len(snapshot.groups) == 1
    group = snapshot.groups[0]
    assert group.member_ids == ("alice", "bob", "self")
    assert group.group_id == frame_group_id(("alice", "bob", "self"))
    assert snapshot.active_group_id == group.group_id
    assert "frame:multi_party_identity" in group.evidence

    # R14 determinism: the same membership (different ordering) observed by
    # a fresh module instance yields the SAME persistent group id.
    upstream_reordered = {
        "multi_party_identity": _identity_snapshot_envelope(
            speaker="bob",
            addressees=("alice",),
            audience=("self", "alice"),
        )
    }
    second = (await GroupModule(wiring_level=WiringLevel.ACTIVE).process(upstream_reordered)).value
    assert second.groups[0].group_id == group.group_id


async def test_dyad_frame_produces_no_group() -> None:
    upstream = {
        "multi_party_identity": _identity_snapshot_envelope(
            speaker="primary",
            addressees=("self",),
            audience=("self",),
        )
    }
    snapshot = (await GroupModule(wiring_level=WiringLevel.ACTIVE).process(upstream)).value
    assert snapshot.groups == ()
    assert snapshot.active_group_id is None


async def test_group_regime_persists_across_module_rebuilds() -> None:
    """R14: group regime is durable runtime state in SocialRecordStore."""
    from volvence_zero.social import SocialRecordStore

    store = SocialRecordStore()
    upstream = {
        "multi_party_identity": _identity_snapshot_envelope(
            speaker="alice",
            addressees=("self",),
            audience=("self", "bob"),
        )
    }
    # Turn 1: orchestrator supplies an explicit regime -> recorded.
    first_module = GroupModule(
        wiring_level=WiringLevel.ACTIVE,
        group_regime_id="problem_solving",
        record_store=store,
    )
    first = (await first_module.process(upstream)).value
    assert first.group_regime_id == "problem_solving"

    # Turn 2: fresh module (per-turn rebuild), no explicit regime ->
    # the persisted regime for the recurring group is rehydrated.
    second_module = GroupModule(wiring_level=WiringLevel.ACTIVE, record_store=store)
    second = (await second_module.process(upstream)).value
    assert second.active_group_id == first.active_group_id
    assert second.group_regime_id == "problem_solving"

    # A different group does not inherit the regime (no cross-group leak).
    other_upstream = {
        "multi_party_identity": _identity_snapshot_envelope(
            speaker="carol",
            addressees=("self",),
            audience=("self", "dave"),
        )
    }
    other = (await GroupModule(wiring_level=WiringLevel.ACTIVE, record_store=store).process(other_upstream)).value
    assert other.active_group_id != first.active_group_id
    assert other.group_regime_id is None


def test_response_assembly_surfaces_group_counts_diagnostically() -> None:
    group = GroupIdentity(
        group_id="group:launch",
        member_ids=("alice", "bob", "carol"),
        display_name="Launch group",
        confidence=0.82,
        evidence=("host membership",),
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(groups=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="group-counts-model",
                feature_surface=(FeatureSignal(name="group_counts", values=(0.5,), source="test"),),
            ),
            group_identities=(group,),
            active_group_id="group:launch",
            group_joint_attention=("launch-plan",),
            group_joint_commitments=("commitment:ship",),
            group_regime_id="problem_solving",
            session_id="group-counts-session",
            wave_id="group-counts-wave",
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    assert counts["groups"] == 1
    assert counts["group_joint_commitments"] == 1
    assert "groups" not in response_assembly.semantic_residue_summary
    assert response_assembly.expression_intent
