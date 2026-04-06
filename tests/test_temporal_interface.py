from __future__ import annotations

import asyncio

from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    PlaceholderSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import HeuristicTemporalPolicy, PlaceholderTemporalPolicy, TemporalModule


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
