from __future__ import annotations

import asyncio

from volvence_zero.runtime import Snapshot, WiringLevel, propagate
from volvence_zero.substrate import (
    apply_residual_control,
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    OpenWeightResidualInterventionBackend,
    OpenWeightResidualStreamSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SyntheticOpenWeightResidualRuntime,
    TraceResidualInterventionBackend,
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
    assert snapshot.value.residual_sequence == ()
    assert snapshot.value.unavailable_fields[0].field_name == "residual_activations"


def test_placeholder_adapter_keeps_shape_stable_for_downstream_consumers():
    adapter = PlaceholderSubstrateAdapter(model_id="placeholder-model")
    module = SubstrateModule(adapter=adapter, wiring_level=WiringLevel.ACTIVE)

    result = asyncio.run(propagate([module], session_id="s1", wave_id="w1"))
    snapshot = result["substrate"]

    assert snapshot.value.surface_kind is SurfaceKind.PLACEHOLDER
    assert snapshot.value.token_logits == ()
    assert snapshot.value.feature_surface == ()
    assert snapshot.value.residual_sequence == ()
    assert len(snapshot.value.unavailable_fields) >= 3


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
    assert snapshot.residual_sequence
    assert snapshot.residual_sequence[-1].token


def test_residual_control_application_updates_snapshot_and_effect():
    trace = build_training_trace(trace_id="control-trace", source_text="calm reflective collaboration")
    adapter = SimulatedResidualSubstrateAdapter(trace=trace)
    snapshot = asyncio.run(adapter.capture(source_text=trace.source_text))

    applied = apply_residual_control(
        substrate_snapshot=snapshot,
        applied_control=(0.4, 0.2, 0.3),
        track_scale=(1.0, 0.5, 0.5),
    )

    assert applied.applied_snapshot.residual_sequence
    assert applied.applied_snapshot.residual_activations != snapshot.residual_activations
    assert applied.downstream_effect != (0.0, 0.0, 0.0)
    assert applied.control_energy > 0.0
    assert applied.backend_name == "trace-residual-backend"


def test_trace_residual_backend_exposes_named_intervention_contract():
    trace = build_training_trace(trace_id="backend-trace", source_text="steady guided exploration")
    adapter = SimulatedResidualSubstrateAdapter(trace=trace)
    snapshot = asyncio.run(adapter.capture(source_text=trace.source_text))
    backend = TraceResidualInterventionBackend()

    applied = backend.apply_control(
        substrate_snapshot=snapshot,
        applied_control=(0.3, 0.3, 0.2),
    )

    assert applied.backend_name == "trace-residual-backend"
    assert "trace-residual-backend" in applied.description


def test_open_weight_residual_adapter_uses_runtime_capture_contract():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-runtime")
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)

    snapshot = asyncio.run(adapter.capture(source_text="real residual hooks later"))

    assert snapshot.model_id == "synthetic-runtime"
    assert snapshot.surface_kind is SurfaceKind.RESIDUAL_STREAM
    assert snapshot.residual_sequence


def test_open_weight_residual_backend_delegates_to_runtime():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-runtime")
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)
    snapshot = asyncio.run(adapter.capture(source_text="real residual hooks later"))
    backend = OpenWeightResidualInterventionBackend(
        runtime=runtime,
        source_text="real residual hooks later",
    )

    applied = backend.apply_control(
        substrate_snapshot=snapshot,
        applied_control=(0.2, 0.3, 0.1),
    )

    assert applied.backend_name == "open-weight:synthetic-runtime"
    assert applied.downstream_effect != (0.0, 0.0, 0.0)
