"""Owner-mainline torch autograd SSL over the live MetacontrollerParameterStore.

Phase A of the autograd-owner-integration plan. Unlike the standalone Phase-1
proof (`torch_metacontroller.py`, which owns a separate fused module), this
trainer operates **directly on the store's `Ndim*Parameters`**: it reads the
exact float weights the runtime metacontroller consumes, runs a faithful torch
forward of the ndim encoder/switch/decoder, optimizes the ETA Eq.3 objective
with real backprop (`D_KL(q || N(0, I))` + action MSE + STE switch), and writes
the updated floats back into the same store.

This is the bridge that makes `MetacontrollerSSLTrainer` capable of genuine
autograd learning of the runtime weights, gated by `WiringLevel`:

- DISABLED: not used (pure heuristic path is the live writer / rollback base).
- SHADOW: train a COPY seeded from the store; compare; do NOT write back.
- ACTIVE: train seeded from the store and write the refined weights back.

`torch` is imported lazily; this module is not re-exported from the temporal
facade. Weights cross the store boundary only as float tuples (R8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from volvence_zero.substrate import TrainingTrace
from volvence_zero.temporal.metacontroller_components import (
    NdimDecoderParameters,
    NdimEncoderParameters,
    NdimFFNParams,
    NdimGRUParams,
    NdimSwitchParameters,
    _project_to_ndim,
    summarize_residual_activations,
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded at call sites
        raise ImportError(
            "Torch store SSL requires torch. Install the vz-temporal '[torch]' extra."
        ) from exc
    return torch


@dataclass(frozen=True)
class StoreSSLReport:
    trace_id: str
    prediction_loss: float
    kl_loss: float
    total_loss: float
    trained_steps: int
    switch_sparsity: float
    binary_switch_ratio: float
    grad_norm: float
    parameters_changed: int
    parameter_change_rate: float
    wrote_back: bool
    # CP-05 (GAP-09): the refined weights as an exportable CANDIDATE
    # checkpoint. Under SHADOW (wrote_back=False) the live store is
    # untouched but the candidate is retained so promotion can restore
    # exactly what the SHADOW pass trained instead of retraining blind.
    # Under ACTIVE these equal what was written back.
    candidate_encoder_parameters: NdimEncoderParameters | None = None
    candidate_switch_parameters: NdimSwitchParameters | None = None
    candidate_decoder_parameters: NdimDecoderParameters | None = None
    description: str = ""


def _step_input_vectors(trace: TrainingTrace, n_input: int) -> list[tuple[float, ...]]:
    """Mirror `_summarize_substrate_ndim`: per-step n_input vector from a trace."""

    vectors: list[tuple[float, ...]] = []
    for step in trace.steps:
        raw: list[float] = []
        for act in step.residual_activations:
            raw.extend(act.activation)
        if not raw:
            for feat in step.feature_surface:
                raw.extend(feat.values)
        if not raw:
            raw = [0.0]
        vectors.append(_project_to_ndim(tuple(raw), n_input))
    return vectors


class _TorchNdimMetacontroller:
    """Torch mirror of the ndim encoder/switch/decoder, seeded from store floats."""

    def __init__(self, *, n_z: int, encoder: NdimEncoderParameters,
                 switch: NdimSwitchParameters, decoder: NdimDecoderParameters) -> None:
        torch = _require_torch()
        self._torch = torch
        self.n_z = n_z
        self.n_input = encoder.n_input
        self._dtype = torch.float64

        def mat(m) -> Any:
            return torch.tensor([list(row) for row in m], dtype=self._dtype, requires_grad=True)

        def vec(v) -> Any:
            return torch.tensor(list(v), dtype=self._dtype, requires_grad=True)

        g = encoder.gru
        self.W_z, self.U_z, self.b_z = mat(g.W_z), mat(g.U_z), vec(g.b_z)
        self.W_r, self.U_r, self.b_r = mat(g.W_r), mat(g.U_r), vec(g.b_r)
        self.W_h, self.U_h, self.b_h = mat(g.W_h), mat(g.U_h), vec(g.b_h)
        self.posterior_proj = mat(encoder.posterior_proj)
        self.posterior_std_proj = mat(encoder.posterior_std_proj)
        self.sw_W1, self.sw_b1 = mat(switch.gate_ffn.W1), vec(switch.gate_ffn.b1)
        self.sw_W2, self.sw_b2 = mat(switch.gate_ffn.W2), vec(switch.gate_ffn.b2)
        self.dec_W1, self.dec_b1 = mat(decoder.decoder_ffn.W1), vec(decoder.decoder_ffn.b1)
        self.dec_W2, self.dec_b2 = mat(decoder.decoder_ffn.W2), vec(decoder.decoder_ffn.b2)

    def parameters(self) -> list[Any]:
        return [
            self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h, self.posterior_proj, self.posterior_std_proj,
            self.sw_W1, self.sw_b1, self.sw_W2, self.sw_b2,
            self.dec_W1, self.dec_b1, self.dec_W2, self.dec_b2,
        ]

    def _gru_step(self, x: Any, h: Any) -> Any:
        torch = self._torch
        z = torch.sigmoid(torch.matmul(self.W_z, x) + torch.matmul(self.U_z, h) + self.b_z)
        r = torch.sigmoid(torch.matmul(self.W_r, x) + torch.matmul(self.U_r, h) + self.b_r)
        h_cand = torch.tanh(torch.matmul(self.W_h, x) + torch.matmul(self.U_h, r * h) + self.b_h)
        return (1.0 - z) * h + z * h_cand

    def rollout(self, step_inputs: Sequence[tuple[float, ...]], *, switch_threshold: float):
        """Faithful ndim forward; returns per-step controls, posteriors, betas."""

        torch = self._torch
        h = torch.zeros(self.n_z, dtype=self._dtype)
        prev_code = torch.zeros(self.n_z, dtype=self._dtype)
        hidden_sum = torch.zeros(self.n_z, dtype=self._dtype)
        controls: list[Any] = []
        means: list[Any] = []
        stds: list[Any] = []
        betas: list[Any] = []
        count = 0
        for raw in step_inputs:
            x = torch.tensor(list(raw), dtype=self._dtype)
            h = self._gru_step(x, h)
            count += 1
            hidden_sum = hidden_sum + h
            avg_hidden = hidden_sum / count
            posterior_mean = torch.clamp(
                0.7 * torch.matmul(self.posterior_proj, h) + 0.3 * avg_hidden, 0.0, 1.0
            )
            posterior_std = torch.clamp(
                torch.abs(torch.matmul(self.posterior_std_proj, h)), 0.05, 0.95
            )
            sample_noise = torch.clamp(avg_hidden - 0.5 * posterior_mean, -1.0, 1.0)
            z_tilde = torch.clamp(posterior_mean + 0.5 * posterior_std * sample_noise, 0.0, 1.0)
            # switch gate: the pure ndim code uses ``gate_input = delta + z_tilde``
            # as tuple CONCATENATION (2*n_z dims), matching the n_z*2-column W1.
            gate_input = torch.cat([torch.abs(z_tilde - prev_code), z_tilde])
            sw_hidden = torch.tanh(torch.matmul(self.sw_W1, gate_input) + self.sw_b1)
            raw_gate = torch.matmul(self.sw_W2, sw_hidden) + self.sw_b2
            beta_cont = torch.sigmoid(raw_gate)
            latent_code = torch.clamp(beta_cont * z_tilde + (1.0 - beta_cont) * prev_code, 0.0, 1.0)
            dec_hidden = torch.tanh(torch.matmul(self.dec_W1, latent_code) + self.dec_b1)
            decoder_output = torch.matmul(self.dec_W2, dec_hidden) + self.dec_b2
            applied_control = torch.clamp(0.65 * latent_code + 0.35 * decoder_output, 0.0, 1.0)
            controls.append(applied_control)
            means.append(posterior_mean)
            stds.append(posterior_std)
            betas.append(beta_cont)
            prev_code = latent_code
        return {"controls": controls, "means": means, "stds": stds, "betas": betas}

    # --- write-back to the ndim parameter dataclasses (float tuples) ---

    def _m(self, t: Any) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in row) for row in t.detach().tolist())

    def _v(self, t: Any) -> tuple[float, ...]:
        return tuple(float(v) for v in t.detach().tolist())

    def to_encoder_params(self, n_input: int) -> NdimEncoderParameters:
        return NdimEncoderParameters(
            n_input=n_input,
            gru=NdimGRUParams(
                W_z=self._m(self.W_z), U_z=self._m(self.U_z), b_z=self._v(self.b_z),
                W_r=self._m(self.W_r), U_r=self._m(self.U_r), b_r=self._v(self.b_r),
                W_h=self._m(self.W_h), U_h=self._m(self.U_h), b_h=self._v(self.b_h),
            ),
            posterior_proj=self._m(self.posterior_proj),
            posterior_std_proj=self._m(self.posterior_std_proj),
        )

    def to_switch_params(self) -> NdimSwitchParameters:
        return NdimSwitchParameters(
            gate_ffn=NdimFFNParams(
                W1=self._m(self.sw_W1), b1=self._v(self.sw_b1),
                W2=self._m(self.sw_W2), b2=self._v(self.sw_b2),
            )
        )

    def to_decoder_params(self) -> NdimDecoderParameters:
        return NdimDecoderParameters(
            decoder_ffn=NdimFFNParams(
                W1=self._m(self.dec_W1), b1=self._v(self.dec_b1),
                W2=self._m(self.dec_W2), b2=self._v(self.dec_b2),
            )
        )


def train_store_ssl(
    *,
    store: Any,
    trace: TrainingTrace,
    n_z: int,
    alpha: float = 0.1,
    learning_rate: float = 0.02,
    switch_threshold: float = 0.55,
    write_back: bool,
) -> StoreSSLReport:
    """Run one torch autograd SSL pass over the store's ndim params.

    When ``write_back`` is True the refined weights replace the store's ndim
    params (ACTIVE). When False, the store is left untouched (SHADOW evidence).
    """

    torch = _require_torch()
    if store.ndim_encoder_parameters is None or store.ndim_switch_parameters is None \
            or store.ndim_decoder_parameters is None:
        raise RuntimeError("torch store SSL requires ndim parameters (n_z > 3).")
    if len(trace.steps) < 2:
        return StoreSSLReport(
            trace_id=trace.trace_id, prediction_loss=0.0, kl_loss=0.0, total_loss=0.0,
            trained_steps=0, switch_sparsity=0.0, binary_switch_ratio=0.0, grad_norm=0.0,
            parameters_changed=0, parameter_change_rate=0.0, wrote_back=False,
            description="trace too short for store SSL",
        )

    module = _TorchNdimMetacontroller(
        n_z=n_z,
        encoder=store.ndim_encoder_parameters,
        switch=store.ndim_switch_parameters,
        decoder=store.ndim_decoder_parameters,
    )
    opt = torch.optim.Adam(module.parameters(), lr=learning_rate)
    inputs = _step_input_vectors(trace, module.n_input)
    targets = [
        _project_to_ndim(
            summarize_residual_activations(step.residual_activations, step.feature_surface), n_z
        )
        for step in trace.steps
    ]

    before = [p.detach().clone() for p in module.parameters()]
    opt.zero_grad()
    out = module.rollout(inputs[:-1], switch_threshold=switch_threshold)
    controls, means, stds, betas = out["controls"], out["means"], out["stds"], out["betas"]

    pred_terms = []
    kl_terms = []
    for t, control in enumerate(controls):
        target = torch.tensor(list(targets[t + 1]), dtype=torch.float64)
        pred_terms.append(torch.mean((control - target).pow(2)))
        m = means[t]
        s = stds[t]
        # D_KL(N(m, s^2) || N(0, I)) = 0.5 * sum(m^2 + s^2 - 1 - 2 ln s)
        kl_terms.append(0.5 * torch.sum(m.pow(2) + s.pow(2) - 1.0 - 2.0 * torch.log(s)))
    prediction_loss = torch.stack(pred_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    total = prediction_loss + alpha * kl_loss
    pred_v = float(prediction_loss.detach())
    kl_v = float(kl_loss.detach())
    total_v = float(total.detach())
    total.backward()
    grad_norm = math.sqrt(
        sum(float(p.grad.pow(2).sum()) for p in module.parameters() if p.grad is not None)
    )
    opt.step()

    after = module.parameters()
    changed = 0
    total_params = 0
    for b, a in zip(before, after):
        diff = (a.detach() - b).abs()
        changed += int((diff > 1e-12).sum())
        total_params += int(diff.numel())

    beta_means = [float(b.detach().mean()) for b in betas]
    mean_beta = sum(beta_means) / max(len(beta_means), 1)
    switches = sum(1 for b in beta_means if b >= switch_threshold)

    candidate_encoder = module.to_encoder_params(module.n_input)
    candidate_switch = module.to_switch_params()
    candidate_decoder = module.to_decoder_params()
    if write_back:
        store.ndim_encoder_parameters = candidate_encoder
        store.ndim_switch_parameters = candidate_switch
        store.ndim_decoder_parameters = candidate_decoder

    return StoreSSLReport(
        trace_id=trace.trace_id,
        prediction_loss=pred_v,
        kl_loss=kl_v,
        total_loss=total_v,
        trained_steps=len(controls),
        switch_sparsity=1.0 - mean_beta,
        binary_switch_ratio=switches / max(len(beta_means), 1),
        grad_norm=grad_norm,
        parameters_changed=changed,
        parameter_change_rate=changed / max(total_params, 1),
        wrote_back=write_back,
        candidate_encoder_parameters=candidate_encoder,
        candidate_switch_parameters=candidate_switch,
        candidate_decoder_parameters=candidate_decoder,
        description=(
            f"store SSL torch: pred={pred_v:.4f} kl={kl_v:.4f} "
            f"changed={changed} wrote_back={write_back}"
        ),
    )
