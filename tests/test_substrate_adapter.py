from __future__ import annotations

import asyncio

from volvence_zero.runtime import Snapshot, WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SimulatedResidualSubstrateAdapter,
    SubstrateModule,
    SurfaceKind,
    TrainingTraceDataset,
    build_training_trace,
)


def test_feature_surface_adapter_publishes_current_stable_contract():
    adapter = FeatureSurfaceSubstrateAdapter(
        model_id="test-model",
        feature_surface=(
            FeatureSignal(name="semantic_state", values=(0.1, 0.9), source="adapter"),
        ),
        token_logits=(0.6, 0.4),
    )
    module = SubstrateModule(adapter=adapter, source_text="hello", wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(propagate([module], session_id="s1", wave_id="w1"))
    snapshot = result["substrate"]

    assert snapshot.value.surface_kind is SurfaceKind.FEATURE_SURFACE
    assert snapshot.value.model_id == "test-model"
    assert snapshot.value.feature_surface[0].name == "semantic_state"
    assert snapshot.value.residual_activations == ()
    assert snapshot.value.unavailable_fields[0].field_name == "residual_activations"


def test_placeholder_adapter_keeps_shape_stable_for_downstream_consumers():
    adapter = PlaceholderSubstrateAdapter(model_id="placeholder-model")
    module = SubstrateModule(adapter=adapter, wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(propagate([module], session_id="s1", wave_id="w1"))
    snapshot = result["substrate"]

    assert snapshot.value.surface_kind is SurfaceKind.PLACEHOLDER
    assert snapshot.value.token_logits == ()
    assert snapshot.value.feature_surface == ()
    assert len(snapshot.value.unavailable_fields) >= 2


def test_shadow_substrate_module_publishes_shadow_only():
    adapter = FeatureSurfaceSubstrateAdapter(
        model_id="shadow-model",
        feature_surface=(FeatureSignal(name="shadow", values=(1.0,), source="adapter"),),
    )
    module = SubstrateModule(adapter=adapter)
    shadow_snapshots: dict[str, Snapshot[object]] = {}

    result = asyncio.run(
        propagate([module], session_id="s1", wave_id="w1", shadow_snapshots=shadow_snapshots)
    )

    assert "substrate" not in result
    assert shadow_snapshots["substrate"].value.model_id == "shadow-model"


def test_simulated_residual_adapter_exposes_executable_residual_surface():
    dataset = TrainingTraceDataset()
    dataset.add_trace(build_training_trace(trace_id="trace-1", source_text="calm reflective collaboration"))
    adapter = SimulatedResidualSubstrateAdapter(trace=dataset.latest())
    snapshot = asyncio.run(adapter.capture(source_text=dataset.latest().source_text))

    assert snapshot.surface_kind is SurfaceKind.RESIDUAL_STREAM
    assert snapshot.residual_activations
    assert snapshot.feature_surface
