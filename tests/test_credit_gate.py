from __future__ import annotations

import asyncio

from volvence_zero.credit import (
    CreditModule,
    GateDecision,
    ModificationGate,
    ModificationProposal,
    derive_abstract_action_credit_records,
    extend_credit_snapshot,
    evaluate_gate,
    has_blocking_writeback,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.evaluation import EvaluationModule
from volvence_zero.memory import MemoryModule, MemoryStore, MemoryStratum, MemoryWriteRequest, Track
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import LearnedLiteTemporalPolicy, TemporalModule


def test_gate_blocks_online_modification_when_high_alert_present():
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
        )
    ).value
    evaluation_snapshot = type(evaluation_snapshot)(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=("HIGH: cross-track stability is degraded",),
        description=evaluation_snapshot.description,
    )

    decision = evaluate_gate(
        proposal=ModificationProposal(
            target="retrieval_weight",
            desired_gate=ModificationGate.ONLINE,
            old_value_hash="old",
            new_value_hash="new",
            justification="Tune retrieval after poor turn.",
        ),
        evaluation_snapshot=evaluation_snapshot,
    )

    assert decision is GateDecision.BLOCK


def test_credit_module_records_credits_and_modification_audit():
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=10,
        )
    ).value

    module = CreditModule(
        wiring_level=WiringLevel.ACTIVE,
        pending_proposals=(
            ModificationProposal(
                target="strategy_prior",
                desired_gate=ModificationGate.ONLINE,
                old_value_hash="old-prior",
                new_value_hash="new-prior",
                justification="Small turn-level policy adjustment.",
            ),
        ),
    )
    snapshot = asyncio.run(
        module.process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=20,
        )
    )

    assert snapshot.value.recent_credits
    assert snapshot.value.recent_modifications
    assert snapshot.value.cumulative_credit_by_level
    assert snapshot.value.recent_modifications[-1].decision is GateDecision.ALLOW


def test_credit_module_consumes_chain_in_shadow_mode():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="answer carefully and completely",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
        ),
        timestamp_ms=10,
    )
    store.write(
        MemoryWriteRequest(
            content="keep the user feeling understood",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.7,
        ),
        timestamp_ms=11,
    )
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="credit-model",
            feature_surface=(FeatureSignal(name="credit_context", values=(0.6,), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    dual_track = DualTrackModule(wiring_level=WiringLevel.ACTIVE)
    evaluation = EvaluationModule(session_id="s1", wave_id="w1", wiring_level=WiringLevel.ACTIVE)
    credit = CreditModule(
        pending_proposals=(
            ModificationProposal(
                target="retrieval_weight",
                desired_gate=ModificationGate.ONLINE,
                old_value_hash="old-weight",
                new_value_hash="new-weight",
                justification="Adjust retrieval after evaluation feedback.",
            ),
        ),
    )
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, memory, dual_track, evaluation, credit],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "evaluation" in result
    assert "credit" not in result
    credit_snapshot = shadow_snapshots["credit"]
    assert credit_snapshot.value.recent_credits
    assert credit_snapshot.value.recent_modifications


def test_has_blocking_writeback_detects_blocked_targets():
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s1",
            wave_id="w1",
            timestamp_ms=30,
        )
    ).value
    evaluation_snapshot = type(evaluation_snapshot)(
        turn_scores=evaluation_snapshot.turn_scores,
        session_scores=evaluation_snapshot.session_scores,
        alerts=("HIGH: writeback should pause",),
        description=evaluation_snapshot.description,
    )
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    credit_snapshot = asyncio.run(
        CreditModule(
            wiring_level=WiringLevel.ACTIVE,
            pending_proposals=(
                ModificationProposal(
                    target="memory.writeback.threshold",
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="old",
                    new_value_hash="new",
                    justification="guard writeback",
                ),
            ),
        ).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=31,
        )
    ).value

    assert has_blocking_writeback(credit_snapshot, target_prefix="memory")


def test_credit_snapshot_can_be_extended_with_abstract_action_credit():
    dual_track_snapshot = asyncio.run(
        DualTrackModule(wiring_level=WiringLevel.ACTIVE).process_standalone(world_entries=(), self_entries=())
    ).value
    evaluation_snapshot = asyncio.run(
        EvaluationModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            session_id="s2",
            wave_id="w2",
            timestamp_ms=50,
        )
    ).value
    temporal_snapshot = asyncio.run(
        TemporalModule(policy=LearnedLiteTemporalPolicy(), wiring_level=WiringLevel.ACTIVE).process_standalone(
            substrate_snapshot=asyncio.run(
                SubstrateModule(
                    adapter=FeatureSurfaceSubstrateAdapter(
                        model_id="temporal-credit-model",
                        feature_surface=(FeatureSignal(name="credit_signal", values=(0.5,), source="adapter"),),
                    ),
                    wiring_level=WiringLevel.ACTIVE,
                ).process_standalone()
            ).value
        )
    ).value
    base_snapshot = asyncio.run(
        CreditModule(wiring_level=WiringLevel.ACTIVE).process_standalone(
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=51,
        )
    ).value

    extended = extend_credit_snapshot(
        credit_snapshot=base_snapshot,
        extra_records=derive_abstract_action_credit_records(
            temporal_snapshot=temporal_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            timestamp_ms=52,
        ),
    )

    assert any(record.level == "abstract_action" for record in extended.recent_credits)
