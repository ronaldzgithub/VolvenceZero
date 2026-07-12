from __future__ import annotations

import asyncio

from volvence_zero.memory import MemoryModule, MemoryStore
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.social_cognition import (
    CommonGroundAtom,
    CommonGroundSnapshot,
    SocialPredictionKind,
    SocialScopeKind,
)
from volvence_zero.social import CommonGroundModule
from volvence_zero.social import LLMCommonGroundProposalRuntime
from volvence_zero.social import MultiPartyIdentityModule
from volvence_zero.social import ConversationalRoleModule
from volvence_zero.social import BeliefAboutOtherModule
from volvence_zero.social import SocialPredictionAggregateModule
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter, SubstrateModule


class ScriptedProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        return self.response


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
    assert len(snapshot.active_predictions) == 2
    assert {
        prediction.kind for prediction in snapshot.active_predictions
    } == {SocialPredictionKind.COMMON_GROUND_RESOLUTION}
    assert "dyad_atoms=1 group_atoms=1" in snapshot.description


def test_social_prediction_aggregate_forwards_common_ground_predictions() -> None:
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

    result = asyncio.run(
        propagate(
            [
                _substrate(),
                MultiPartyIdentityModule(wiring_level=WiringLevel.ACTIVE),
                MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE),
                ConversationalRoleModule(wiring_level=WiringLevel.ACTIVE),
                BeliefAboutOtherModule(wiring_level=WiringLevel.ACTIVE),
                CommonGroundModule(dyad_atoms=(dyad,), wiring_level=WiringLevel.ACTIVE),
                SocialPredictionAggregateModule(wiring_level=WiringLevel.ACTIVE),
            ],
            session_id="common-ground-aggregate-session",
            wave_id="common-ground-aggregate-wave",
        )
    )

    aggregate = result["social_prediction"].value
    predictions = tuple(
        prediction
        for prediction in aggregate.predictions
        if prediction.kind is SocialPredictionKind.COMMON_GROUND_RESOLUTION
    )
    assert len(predictions) == 1
    assert predictions[0].prediction_id.startswith("common_ground:")


def test_common_ground_module_consumes_structured_runtime_atoms() -> None:
    runtime = LLMCommonGroundProposalRuntime(
        provider=ScriptedProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "We both know Alice wants a slower pace.",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "slow down like we agreed",
                "confidence": 0.82,
                "recursion_depth": 2,
                "control_signal": 0.40
              },
              {
                "scope_kind": "group",
                "scope_id": "team:launch",
                "summary": "The launch team accepted the new deadline.",
                "accepted_by_ids": ["alice", "bob", "carol"],
                "evidence": "all confirmed",
                "confidence": 0.78,
                "recursion_depth": 1,
                "control_signal": 0.35
              }
            ]
            """
        )
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
                    proposal_runtime=runtime,
                    user_input="Let's slow down like we agreed.",
                    turn_index=4,
                    wiring_level=WiringLevel.ACTIVE,
                ),
            ],
            session_id="common-ground-runtime-session",
            wave_id="common-ground-runtime-wave",
        )
    )

    snapshot = result["common_ground"].value
    assert isinstance(snapshot, CommonGroundSnapshot)
    assert len(snapshot.dyad_atoms) == 1
    assert len(snapshot.group_atoms) == 1
    assert len(snapshot.active_predictions) == 2
    assert snapshot.dyad_atoms[0].scope_id == "self:alice"
    assert snapshot.group_atoms[0].scope_id == "team:launch"
    assert snapshot.control_signal == 0.8


def test_common_ground_module_runtime_low_confidence_keeps_empty_snapshot() -> None:
    runtime = LLMCommonGroundProposalRuntime(
        provider=ScriptedProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "weak shared claim",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "weak evidence",
                "confidence": 0.25,
                "recursion_depth": 2
              }
            ]
            """
        )
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
                    proposal_runtime=runtime,
                    user_input="maybe shared",
                    turn_index=5,
                    wiring_level=WiringLevel.ACTIVE,
                ),
            ],
            session_id="common-ground-low-confidence-session",
            wave_id="common-ground-low-confidence-wave",
        )
    )

    snapshot = result["common_ground"].value
    assert snapshot.dyad_atoms == ()
    assert snapshot.group_atoms == ()


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


def test_final_wiring_structured_common_ground_runtime_populates_diagnostics() -> None:
    runtime = LLMCommonGroundProposalRuntime(
        provider=ScriptedProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "We both know Alice wants a slower pace.",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "slow down like we agreed",
                "confidence": 0.82,
                "recursion_depth": 2,
                "control_signal": 0.40
              }
            ]
            """
        )
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(common_ground=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="common-ground-runtime-counts-model",
                feature_surface=(FeatureSignal(name="common_ground_runtime", values=(0.5,), source="test"),),
            ),
            user_input="Let's slow down like we agreed.",
            common_ground_proposal_runtime=runtime,
            session_id="common-ground-runtime-counts-session",
            wave_id="common-ground-runtime-counts-wave",
            turn_index=6,
        )
    )

    common_ground = result.active_snapshots["common_ground"].value
    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    assert isinstance(common_ground, CommonGroundSnapshot)
    assert len(common_ground.dyad_atoms) == 1
    assert counts["common_ground"] == 1
    assert "common_ground" not in response_assembly.semantic_residue_summary
