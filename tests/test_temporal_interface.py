from __future__ import annotations

import asyncio

from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import SimulatedResidualSubstrateAdapter, build_training_trace
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import (
    HeuristicTemporalPolicy,
    LearnedLiteTemporalPolicy,
    PlaceholderTemporalPolicy,
    TemporalModule,
    fit_policy_from_trace_dataset,
)
from volvence_zero.substrate import TrainingTraceDataset


def test_temporal_module_builds_placeholder_snapshot():
    substrate_snapshot = asyncio.run(
        SubstrateModule(
            adapter=PlaceholderSubstrateAdapter(model_id="placeholder-model"),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    temporal = TemporalModule(policy=PlaceholderTemporalPolicy(), wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(temporal.process_standalone(substrate_snapshot=substrate_snapshot))

    assert snapshot.value.controller_state.code_dim == 3
    assert snapshot.value.controller_state.switch_gate == 0.0
    assert snapshot.value.active_abstract_action == "placeholder-controller"


def test_temporal_module_heuristic_switches_on_feature_signature_change():
    temporal = TemporalModule(policy=HeuristicTemporalPolicy(), wiring_level=WiringLevel.ACTIVE)
    first_substrate = asyncio.run(
        SubstrateModule(
            adapter=FeatureSurfaceSubstrateAdapter(
                model_id="heuristic-model",
                feature_surface=(FeatureSignal(name="context_a", values=(0.2, 0.3), source="adapter"),),
            ),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    second_substrate = asyncio.run(
        SubstrateModule(
            adapter=FeatureSurfaceSubstrateAdapter(
                model_id="heuristic-model",
                feature_surface=(FeatureSignal(name="context_b", values=(0.8, 0.9), source="adapter"),),
            ),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value

    first = asyncio.run(temporal.process_standalone(substrate_snapshot=first_substrate)).value
    second = asyncio.run(temporal.process_standalone(substrate_snapshot=second_substrate)).value

    assert first.controller_state.code_dim == 3
    assert second.controller_state.is_switching is True
    assert second.active_abstract_action.startswith("refresh-controller-context:")


def test_temporal_module_runs_in_shadow_chain():
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="shadow-temporal-model",
            feature_surface=(FeatureSignal(name="temporal_signal", values=(0.6, 0.2), source="adapter"),),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    temporal = TemporalModule()
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, temporal],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "substrate" in result
    assert "temporal_abstraction" not in result
    temporal_snapshot = shadow_snapshots["temporal_abstraction"]
    assert temporal_snapshot.value.controller_state.code_dim == 3
    assert temporal_snapshot.value.controller_params_hash


def test_learned_lite_policy_fits_from_trace_dataset_and_emits_controller_step():
    dataset = TrainingTraceDataset(
        (
            build_training_trace(trace_id="t1", source_text="steady progress"),
            build_training_trace(trace_id="t2", source_text="repair emotional tension"),
        )
    )
    policy = LearnedLiteTemporalPolicy()
    fit_policy_from_trace_dataset(policy=policy, dataset=dataset)
    substrate_snapshot = asyncio.run(
        SubstrateModule(
            adapter=SimulatedResidualSubstrateAdapter(trace=dataset.latest()),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    temporal = TemporalModule(policy=policy, wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(temporal.process_standalone(substrate_snapshot=substrate_snapshot))

    assert snapshot.value.controller_params_hash
    assert snapshot.value.active_abstract_action.endswith("learned-lite")


def test_learned_lite_policy_can_align_with_internal_rl_parameters():
    policy = LearnedLiteTemporalPolicy()
    initial = policy.export_parameters()

    policy.align_with_internal_rl(
        world_weights=(0.8, 0.1, 0.1),
        self_weights=(0.1, 0.8, 0.1),
        shared_weights=(0.4, 0.4, 0.2),
        persistence=0.7,
    )
    aligned = policy.export_parameters()

    assert aligned != initial
    assert aligned.switch_bias > 0.0


def test_learned_lite_policy_exports_runtime_visible_metacontroller_state():
    policy = LearnedLiteTemporalPolicy()

    runtime_state = policy.export_runtime_state()

    assert runtime_state.mode == "learned-lite"
    assert runtime_state.temporal_parameters.switch_bias >= 0.0
    assert len(runtime_state.track_parameters) == 3
    assert len(runtime_state.update_steps) == 3
