from __future__ import annotations

import asyncio

from volvence_zero.credit import CreditModule, ModificationGate, ModificationProposal
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.regime import RegimeModule
from volvence_zero.reflection import ReflectionEngine, ReflectionModule, WritebackMode
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)


def test_reflection_module_builds_proposal_first_snapshot():
    reflection = ReflectionModule(wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(reflection.process_standalone(timestamp_ms=10))

    assert snapshot.value.writeback_mode == WritebackMode.PROPOSAL_ONLY.value
    assert snapshot.value.review_required is True
    assert snapshot.value.description.startswith("Reflection generated")


def test_reflection_module_consolidates_memory_and_policy_from_inputs():
    store = MemoryStore()
    memory_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="remember that the user values warmth and clarity",
            timestamp_ms=10,
        )
    ).value
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=11,
        )
    ).value
    credit_snapshot = asyncio.run(
        CreditModule(
            wiring_level=WiringLevel.ACTIVE,
            pending_proposals=(
                ModificationProposal(
                    target="retrieval_weight",
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="old",
                    new_value_hash="new",
                    justification="use audit trail in reflection",
                ),
            ),
        ).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=12,
        )
    ).value

    reflection_snapshot = asyncio.run(
        ReflectionModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            credit_snapshot=credit_snapshot,
            timestamp_ms=13,
        )
    ).value

    assert reflection_snapshot.memory_consolidation.promoted_entries or reflection_snapshot.lessons_extracted
    assert reflection_snapshot.policy_consolidation.controller_updates or reflection_snapshot.policy_consolidation.strategy_priors_updated


def test_reflection_module_runs_in_shadow_chain():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="deliver useful task help",
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
        ),
        timestamp_ms=10,
    )
    store.write(
        MemoryWriteRequest(
            content="stay emotionally attuned",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.75,
        ),
        timestamp_ms=11,
    )

    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="reflection-model",
            feature_surface=(FeatureSignal(name="reflection_context", values=(0.4,), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)
    credit = CreditModule(
        wiring_level=WiringLevel.ACTIVE,
        pending_proposals=(
            ModificationProposal(
                target="strategy_prior",
                desired_gate=ModificationGate.ONLINE,
                old_value_hash="old-prior",
                new_value_hash="new-prior",
                justification="reflection should see gate audit",
            ),
        )
    )
    reflection = ReflectionModule(wiring_level=WiringLevel.SHADOW)
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, memory, dual_track, evaluation, credit, reflection],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "credit" in result
    assert "reflection" not in result
    reflection_snapshot = shadow_snapshots["reflection"]
    assert reflection_snapshot.value.writeback_mode == WritebackMode.PROPOSAL_ONLY.value
    assert reflection_snapshot.value.review_required is True


def test_reflection_apply_path_supports_checkpoint_and_rollback():
    store = MemoryStore()
    regime = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    initial_snapshot = asyncio.run(
        MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE).process_standalone(
            user_text="remember this durable insight",
            timestamp_ms=40,
        )
    ).value
    regime_snapshot = asyncio.run(regime.process_standalone()).value
    reflection_snapshot = asyncio.run(
        ReflectionModule(
            engine=ReflectionEngine(writeback_mode=WritebackMode.APPLY),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone(
            memory_snapshot=initial_snapshot,
            regime_snapshot=regime_snapshot,
            timestamp_ms=41,
        )
    ).value
    engine = ReflectionEngine(writeback_mode=WritebackMode.APPLY)
    writeback = engine.apply(
        memory_store=store,
        reflection_snapshot=reflection_snapshot,
        credit_snapshot=None,
        regime_module=regime,
        checkpoint_id="rollback-checkpoint",
    )

    assert writeback.applied_operations
    assert writeback.checkpoint is not None

    engine.rollback(memory_store=store, checkpoint=writeback.checkpoint, regime_module=regime)
    restored_snapshot = store.snapshot(retrieved_entries=())
    assert restored_snapshot.description.startswith("Memory store")
