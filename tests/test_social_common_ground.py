from __future__ import annotations

import asyncio

from volvence_zero.memory import MemoryModule, MemoryStore
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.social_cognition import CommonGroundAtom, CommonGroundSnapshot, SocialScopeKind
from volvence_zero.social_common_ground import CommonGroundModule
from volvence_zero.social_identity import MultiPartyIdentityModule
from volvence_zero.social_role import ConversationalRoleModule
from volvence_zero.social_tom import BeliefAboutOtherModule
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter, SubstrateModule


def _substrate() -> SubstrateModule:
    return SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="common-ground-test-model",
            feature_surface=(FeatureSignal(name="common_ground_signal", values=(0.5,), source="test"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )


def test_common_ground_module_publishes_empty_shadow_scaffold_by_default() -> None:
    result = asyncio.run(
        propagate(
            [
                _substrate(),
                MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
                MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
                ConversationalRoleModule(wiring_level=WiringLevel.ACTIVE),
                BeliefAboutOtherModule(wiring_level=WiringLevel.ACTIVE),
                CommonGroundModule(wiring_level=WiringLevel.ACTIVE),
            ],
            session_id="common-ground-empty-session",
            wave_id="common-ground-empty-wave",
        )
    )

    snapshot = result["common_ground"].value
    assert isinstance(snapshot, CommonGroundSnapshot)
    assert snapshot.dyad_atoms == ()
    assert snapshot.group_atoms == ()
    assert snapshot.active_predictions == ()
    assert snapshot.control_signal == 0.0


def test_common_ground_module_publishes_explicit_dyad_and_group_atoms() -> None:
    dyad = CommonGroundAtom(
        atom_id="cg:dyad:alice-bob:plan",
        scope_id="alice:bob",
        scope_kind=SocialScopeKind.DYAD,
        summary="Alice and Bob both know the plan changed.",
        recursion_depth=2,
        confidence=0.74,
        accepted_by_ids=("alice", "bob"),
        evidence=("both confirmed the change",),
    )
    group = CommonGroundAtom(
        atom_id="cg:group:team:deadline",
        scope_id="team:launch",
        scope_kind=SocialScopeKind.GROUP,
        summary="The launch team knows the deadline moved.",
        recursion_depth=1,
        confidence=0.69,
        accepted_by_ids=("alice", "bob", "carol"),
        evidence=("team acknowledgement",),
    )

    result = asyncio.run(
        propagate(
            [
                _substrate(),
                MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
                MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
                ConversationalRoleModule(wiring_level=WiringLevel.ACTIVE),
                BeliefAboutOtherModule(wiring_level=WiringLevel.ACTIVE),
                CommonGroundModule(
                    dyad_atoms=(dyad,),
                    group_atoms=(group,),
                    wiring_level=WiringLevel.ACTIVE,
                ),
            ],
            session_id="common-ground-explicit-session",
            wave_id="common-ground-explicit-wave",
        )
    )

    snapshot = result["common_ground"].value
    assert isinstance(snapshot, CommonGroundSnapshot)
    assert snapshot.dyad_atoms == (dyad,)
    assert snapshot.group_atoms == (group,)
    assert "dyad_atoms=1 group_atoms=1" in snapshot.description


def test_response_assembly_surfaces_common_ground_atom_count_diagnostically() -> None:
    dyad = CommonGroundAtom(
        atom_id="cg:dyad:alice-bob:plan",
        scope_id="alice:bob",
        scope_kind=SocialScopeKind.DYAD,
        summary="Alice and Bob both know the plan changed.",
        recursion_depth=2,
        confidence=0.74,
        accepted_by_ids=("alice", "bob"),
        evidence=("both confirmed the change",),
    )
    group = CommonGroundAtom(
        atom_id="cg:group:team:deadline",
        scope_id="team:launch",
        scope_kind=SocialScopeKind.GROUP,
        summary="The launch team knows the deadline moved.",
        recursion_depth=1,
        confidence=0.69,
        accepted_by_ids=("alice", "bob", "carol"),
        evidence=("team acknowledgement",),
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(common_ground=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="common-ground-counts-model",
                feature_surface=(FeatureSignal(name="common_ground_counts", values=(0.5,), source="test"),),
            ),
            common_ground_dyad_atoms=(dyad,),
            common_ground_group_atoms=(group,),
            session_id="common-ground-counts-session",
            wave_id="common-ground-counts-wave",
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    assert counts["common_ground"] == 2
    assert "common_ground" not in response_assembly.semantic_residue_summary
    assert response_assembly.expression_intent
