"""Tests for the per-turn persona-LoRA activation gate.

The synthesizer activates a persona LoRA only when (a) the bundle pins a
figure_id with a pool record, (b) the runtime is LoRA-aware, and (c) the
adopting contract's adapter_policy permits it (``enabled``). This pins
the policy gate (c), which is the R10 enforcement at the activation
site.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from lifeform_expression.llm_synthesizer import _maybe_activate_persona_lora
from volvence_zero.substrate import (
    SubstrateDeltaAdapterLayer,
    default_persona_lora_pool,
)


@dataclass
class _Bundle:
    figure_id: str


class _FakeRuntime:
    def __init__(self) -> None:
        self.activations: list[object] = []

    @contextlib.contextmanager
    def activate_lora(self, layers):
        self.activations.append(layers)
        yield


def _register(figure_id: str) -> None:
    default_persona_lora_pool().register(
        figure_id=figure_id,
        source_bundle_id=f"bundle:{figure_id}",
        backend_id="synthetic-v1",
        training_plan_hash="hash",
        adapter_layers=(
            SubstrateDeltaAdapterLayer(
                layer_index=0,
                delta_vector=(0.1, 0.2),
                mean_abs_delta=0.15,
                description="d",
            ),
        ),
        parameter_count=8,
    )


def test_gate_activates_when_enabled() -> None:
    figure_id = "gatetest-enabled"
    _register(figure_id)
    runtime = _FakeRuntime()
    with _maybe_activate_persona_lora(
        bundle=_Bundle(figure_id), runtime=runtime, enabled=True
    ):
        pass
    assert len(runtime.activations) == 1


def test_gate_skips_when_disabled() -> None:
    figure_id = "gatetest-disabled"
    _register(figure_id)
    runtime = _FakeRuntime()
    with _maybe_activate_persona_lora(
        bundle=_Bundle(figure_id), runtime=runtime, enabled=False
    ):
        pass
    assert runtime.activations == []


def test_synthesizer_propagates_persona_lora_flag() -> None:
    from lifeform_expression.llm_synthesizer import (
        LifeformLLMResponseSynthesizer,
    )

    synth = LifeformLLMResponseSynthesizer(runtime=object())
    assert synth.persona_lora_enabled is True
    disabled = synth.with_persona_lora_enabled(False)
    assert disabled.persona_lora_enabled is False
    # Cloning for a session preserves the flag.
    assert disabled.clone_for_session().persona_lora_enabled is False
    # Binding a bundle preserves the flag too.
    bound = disabled.with_figure_bundle(_Bundle("x"))
    assert bound.persona_lora_enabled is False
