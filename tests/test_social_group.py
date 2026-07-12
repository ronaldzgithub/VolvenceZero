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
