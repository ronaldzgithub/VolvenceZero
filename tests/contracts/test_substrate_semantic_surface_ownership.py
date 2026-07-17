"""Semantic pull surface ownership merge (coverage 5.1 / #91 residual).

Contract under test: ``OpenWeightResidualStreamSubstrateAdapter.capture``
must not let the text-hash fallback (``semantic_feature_surface_from_text``)
shadow semantic readouts the runtime already publishes from its own
substrate-grounded machinery. ``feature_signal_map`` is last-wins by name,
so before this fix the appended hash surface silently overrode the
runtime's hybrid hidden-state pulls for every consumer. Now:

- names the runtime owns keep the runtime's values;
- the fallback only fills genuinely missing signals
  (``semantic_decision_delegation_pull`` / ``semantic_social_pull`` /
  ``semantic_surface_active`` / ``fallback_active`` when absent);
- runtimes without any semantic surface (synthetic) still receive the
  full fallback surface byte-identical to the direct helper output.
"""

from __future__ import annotations

import asyncio

from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    SyntheticOpenWeightResidualRuntime,
    semantic_feature_surface_from_text,
)
from volvence_zero.substrate.adapter import FeatureSignal, feature_signal_value
from volvence_zero.substrate.residual_contracts import OpenWeightRuntimeCapture
from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime


class _SemanticOwningRuntime(OpenWeightResidualRuntime):
    """Fake runtime that publishes its own semantic pulls (sentinel values)."""

    def __init__(self) -> None:
        self.model_id = "semantic-owning-runtime"
        self.is_frozen = True
        self.runtime_origin = "synthetic-open-weight"

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        del source_text
        return OpenWeightRuntimeCapture(
            token_logits=(0.5,),
            feature_surface=(
                FeatureSignal(
                    name="semantic_task_pull",
                    values=(0.777,),
                    source="runtime-owned-semantic",
                ),
                FeatureSignal(
                    name="semantic_support_pull",
                    values=(0.111,),
                    source="runtime-owned-semantic",
                ),
                FeatureSignal(
                    name="fallback_active",
                    values=(0.0,),
                    source="runtime-owned-semantic",
                ),
            ),
            residual_activations=(),
            residual_sequence=(),
            description="fake capture with runtime-owned semantic pulls",
        )

    def apply_control(self, **kwargs: object):  # pragma: no cover - unused
        raise NotImplementedError


def test_runtime_owned_pulls_are_not_shadowed_by_hash_fallback():
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=_SemanticOwningRuntime())
    snapshot = asyncio.run(adapter.capture(source_text="direct decision please"))

    assert feature_signal_value(snapshot.feature_surface, name="semantic_task_pull") == 0.777
    assert feature_signal_value(snapshot.feature_surface, name="semantic_support_pull") == 0.111
    # Runtime-owned fallback_active wins over the fallback's marker.
    assert feature_signal_value(snapshot.feature_surface, name="fallback_active") == 0.0

    # Owned names appear exactly once in the merged surface.
    names = [signal.name for signal in snapshot.feature_surface]
    assert names.count("semantic_task_pull") == 1
    assert names.count("semantic_support_pull") == 1
    assert names.count("fallback_active") == 1


def test_fallback_fills_missing_signals_only():
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=_SemanticOwningRuntime())
    snapshot = asyncio.run(adapter.capture(source_text="direct decision please"))

    # Signals the runtime did not publish still arrive from the fallback.
    assert feature_signal_value(snapshot.feature_surface, name="semantic_surface_active") == 1.0
    names = {signal.name for signal in snapshot.feature_surface}
    assert "semantic_decision_delegation_pull" in names
    assert "semantic_social_pull" in names
    # The fallback's copies of directive/exploration/repair also land
    # because this runtime does not own them.
    assert "semantic_directive_pull" in names
    assert "semantic_exploration_pull" in names
    assert "semantic_repair_pull" in names


def test_synthetic_runtime_receives_byte_identical_fallback_surface():
    source_text = "steady guided exploration"
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-runtime")
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)
    snapshot = asyncio.run(adapter.capture(source_text=source_text))

    expected = semantic_feature_surface_from_text(
        source_text,
        fallback_active=1.0,
        source="fallback:semantic-readout",
    )
    merged = {signal.name: signal for signal in snapshot.feature_surface}
    for signal in expected:
        assert merged[signal.name].values == signal.values
        assert merged[signal.name].source == signal.source


def test_runtime_with_surface_marker_skips_fallback_entirely():
    class _MarkerRuntime(_SemanticOwningRuntime):
        def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
            base = super().capture(source_text=source_text)
            return OpenWeightRuntimeCapture(
                token_logits=base.token_logits,
                feature_surface=base.feature_surface
                + (
                    FeatureSignal(
                        name="semantic_surface_active",
                        values=(1.0,),
                        source="runtime-owned-semantic",
                    ),
                ),
                residual_activations=(),
                residual_sequence=(),
                description=base.description,
            )

    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=_MarkerRuntime())
    snapshot = asyncio.run(adapter.capture(source_text="anything"))

    names = {signal.name for signal in snapshot.feature_surface}
    # No fallback signals were appended at all.
    assert "semantic_decision_delegation_pull" not in names
    assert "semantic_social_pull" not in names
