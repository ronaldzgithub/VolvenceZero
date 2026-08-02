"""Switch-gated rate restores the ETA Eq.3 segment economics.

The 2026-08-02 smooth+v2 Gate-1 run showed a perfectly monotone rate axis but
zero switching and a near-horizontal rate-distortion curve. Root cause: the
objective charged posterior KL unconditionally every step, so holding a plan
for T steps cost T x KL and switching earned no discount -- the paper's
transmit-only-at-switch economics (pay KL once per segment) was absent, making
the near-vertical gap structurally impossible.

These tests pin the repair:

- the rollout exposes the effective code-mixing gate per step (1 at step 0),
- ``rate_gating="switch-gated"`` multiplies each step's KL by that gate, so a
  kept code contributes zero rate, and
- the legacy ``per-step`` default is byte-identical to the historical readout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from volvence_zero.substrate import (  # noqa: E402
    ExpertActionTarget,
    TraceStep,
    TrainingTrace,
    build_training_trace,
)
from volvence_zero.temporal.interface import (  # noqa: E402
    MetacontrollerParameterStore,
)
from volvence_zero.temporal.metacontroller_components import (  # noqa: E402
    RATE_GATING_PER_STEP,
    RATE_GATING_SWITCH,
)
from volvence_zero.temporal.torch_store_ssl import (  # noqa: E402
    StoreSSLTrainingSession,
    _TorchNdimMetacontroller,
)

_N_Z = 8


class _StubScorer:
    """Minimal SteeredActionNLLScorer: differentiable toy NLL over deltas."""

    hidden_size = 8

    def action_index(self, action_id: str) -> int:
        return {"move:left": 0, "move:right": 1}[action_id]

    def action_nll(
        self,
        *,
        source_texts: tuple[str, ...],
        control_deltas,
        action_indices: tuple[int, ...],
    ):
        del source_texts, action_indices
        return (
            control_deltas.to(dtype=torch.float64).pow(2).mean(dim=-1) + 1.0
        )

    def trainable_parameters(self) -> tuple:
        return ()


def _steered_trace(trace_id: str = "rate-gating") -> TrainingTrace:
    base = build_training_trace(
        trace_id=trace_id,
        source_text="steady waters carry the harbor plan through changing tides",
    )
    split = len(base.steps) // 2
    return TrainingTrace(
        trace_id=base.trace_id,
        source_text=base.source_text,
        steps=tuple(
            TraceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                expert_action_target=ExpertActionTarget(
                    action_id="move:left" if index < split else "move:right",
                    values=(1.0, 0.0) if index < split else (0.0, 1.0),
                    source="test-demonstrator",
                ),
                observation_text=f"at waypoint {index}; exits: left, right",
            )
            for index, step in enumerate(base.steps)
        ),
    )


def _module() -> _TorchNdimMetacontroller:
    store = MetacontrollerParameterStore(n_z=_N_Z, initialization_seed=11)
    return _TorchNdimMetacontroller(
        n_z=_N_Z,
        encoder=store.ndim_encoder_parameters,
        switch=store.ndim_switch_parameters,
        decoder=store.ndim_decoder_parameters,
    )


def _step_inputs(module: _TorchNdimMetacontroller) -> list[tuple[float, ...]]:
    generator = torch.Generator().manual_seed(3)
    return [
        tuple(
            float(v)
            for v in torch.randn(
                module.n_input, generator=generator, dtype=torch.float64
            )
        )
        for _ in range(5)
    ]


def test_rollout_exposes_the_code_mixing_gate() -> None:
    module = _module()
    inputs = _step_inputs(module)

    continuous = module.rollout(
        inputs, switch_threshold=0.55, generator=None, gate_mode="continuous"
    )
    gates = continuous["effective_gates"]
    assert len(gates) == len(inputs)
    assert float(gates[0].detach()) == pytest.approx(1.0)
    for gate, probability in zip(
        gates[1:], continuous["switch_probabilities"][1:], strict=True
    ):
        assert float(gate.detach()) == pytest.approx(float(probability.detach()))

    hard = module.rollout(
        inputs, switch_threshold=0.55, generator=None, gate_mode="hard-st"
    )
    for gate, hard_switch in zip(
        hard["effective_gates"], hard["hard_switches"], strict=True
    ):
        # Straight-through: forward value equals the hard 0/1 decision.
        assert float(gate.detach()) == pytest.approx(float(hard_switch.detach()))


def test_switch_gated_rate_is_cheaper_than_per_step_and_keeps_step0_charge() -> None:
    trace = _steered_trace()

    def kl_rate(rate_gating: str) -> float:
        store = MetacontrollerParameterStore(n_z=_N_Z, initialization_seed=7)
        session = StoreSSLTrainingSession(
            n_z=_N_Z,
            alpha=0.1,
            switch_rate_weight=0.0,
            switch_binary_weight=0.0,
            switch_group_weight=0.0,
            proposal_prediction_weight=0.0,
            gate_choice_weight=0.0,
            action_scorer=_StubScorer(),
            rate_gating=rate_gating,
        )
        report = session.evaluate_batch(
            store=store, traces=(trace,), batch_id=f"kl:{rate_gating}"
        )
        return report.kl_rate

    per_step = kl_rate(RATE_GATING_PER_STEP)
    gated = kl_rate(RATE_GATING_SWITCH)

    # Identical stores and deterministic rollouts, so the only difference is
    # the gating. Step 0 always pays (gate=1); later steps pay gate<1, so the
    # gated rate is strictly cheaper but not zero.
    assert per_step > 0.0
    assert 0.0 < gated < per_step


def test_unknown_rate_gating_is_rejected() -> None:
    with pytest.raises(ValueError, match="rate_gating"):
        StoreSSLTrainingSession(n_z=_N_Z, rate_gating="bogus")
