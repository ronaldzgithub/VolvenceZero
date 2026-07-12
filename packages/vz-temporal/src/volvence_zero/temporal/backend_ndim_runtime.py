"""Backend-routed ndim runtime metacontroller (autograd-owner-integration Phase B).

Mirrors the pure `NdimSequenceEncoder` / `NdimSwitchUnit` / `NdimResidualDecoder`
forward on a :class:`volvence_zero.tensor_backend.TensorBackend`, returning the
SAME `EncodedSequence` / `DecoderControl` dataclasses so it is a drop-in for the
three numeric calls inside `FullLearnedTemporalPolicy._step_impl_ndim`.

Because the forward is backend-agnostic and reads the live store ndim params,
the same weights produce the same controller step on either backend. That gives:

- DISABLED/SHADOW live path: pure components (rollback baseline, unchanged).
- ACTIVE: torch backend computes the runtime step (real autograd-capable path).
- SHADOW gate: :func:`runtime_ndim_shadow_compare` runs pure vs torch and blocks
  promotion unless the controller outputs match within tolerance (exact rollback)
  and latency is within budget.

Outputs cross back as float tuples inside the existing dataclasses (R8). ``torch``
is imported lazily via the backend; this module is not in the temporal facade.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from volvence_zero.tensor_backend import (
    BackendKind,
    FfnParams,
    GruParams,
    PurePythonBackend,
    TensorBackend,
    is_torch_available,
    resolve_backend,
)
from volvence_zero.temporal.metacontroller_components import (
    DecoderControl,
    EncodedSequence,
    NdimDecoderParameters,
    NdimEncoderParameters,
    NdimSwitchParameters,
    PosteriorState,
    _summarize_substrate_ndim,
)


def _gru_params(backend: TensorBackend, enc: NdimEncoderParameters) -> GruParams:
    g = enc.gru
    return GruParams(
        W_z=backend.matrix(g.W_z), U_z=backend.matrix(g.U_z), b_z=backend.vector(g.b_z),
        W_r=backend.matrix(g.W_r), U_r=backend.matrix(g.U_r), b_r=backend.vector(g.b_r),
        W_h=backend.matrix(g.W_h), U_h=backend.matrix(g.U_h), b_h=backend.vector(g.b_h),
    )


class BackendNdimMetacontroller:
    """Drop-in ndim encode/switch/decode on a chosen TensorBackend."""

    def __init__(self, backend: TensorBackend, *, n_z: int) -> None:
        self._b = backend
        self._n_z = n_z

    @property
    def backend(self) -> TensorBackend:
        return self._b

    # --- encoder (mirrors NdimSequenceEncoder.encode) ---

    def encode(
        self,
        *,
        substrate_snapshot: Any,
        previous_hidden_state: tuple[float, ...] | None,
        cms_context: tuple[float, ...] | None,
        params: NdimEncoderParameters,
    ) -> EncodedSequence:
        b = self._b
        n = self._n_z
        gru = _gru_params(b, params)
        step_vectors = _summarize_substrate_ndim(substrate_snapshot, params.n_input)
        prev = previous_hidden_state
        with b.no_grad():
            h = b.vector(prev) if prev is not None else b.zeros(n)
            hidden_sum = b.zeros(n)
            count = 0
            cms_vec = b.vector(cms_context) if cms_context is not None else None
            for step_vec in step_vectors:
                x = b.vector(step_vec)
                if cms_vec is not None:
                    x = b.add(b.scale(x, 0.8), b.scale(cms_vec, 0.2))
                h = b.gru_cell(x=x, h_prev=h, params=gru)
                hidden_sum = b.add(hidden_sum, h)
                count += 1
            if count == 0:
                h = b.zeros(n)
                hidden_sum = b.zeros(n)
                count = 1
            avg_hidden = b.scale(hidden_sum, 1.0 / count)
            posterior_proj = b.matrix(params.posterior_proj)
            posterior_std_proj = b.matrix(params.posterior_std_proj)
            posterior_mean = b.clamp(
                b.add(b.scale(b.matvec(posterior_proj, h), 0.7), b.scale(avg_hidden, 0.3)),
                0.0, 1.0,
            )
            posterior_std = b.clamp(b.abs(b.matvec(posterior_std_proj, h)), 0.05, 0.95)
            sample_noise = b.clamp(b.sub(avg_hidden, b.scale(posterior_mean, 0.5)), -1.0, 1.0)
            z_tilde = b.clamp(
                b.add(posterior_mean, b.scale(b.mul(posterior_std, sample_noise), 0.5)), 0.0, 1.0
            )
            base_for_prior = b.vector(prev) if prev is not None else h
            prior_mean = b.clamp(b.scale(base_for_prior, 0.35), 0.0, 1.0)
            h_f = b.to_floats(h)
            avg_f = b.to_floats(avg_hidden)
            prior_mean_f = b.to_floats(prior_mean)
            prior_std_f = tuple(
                max(0.05, 1.0 - abs(prior_mean_f[i] - avg_f[i]) * 0.5) for i in range(n)
            )
            prev_f = prev if prev is not None else tuple(0.0 for _ in range(n))
            drift = max(abs(h_f[i] - prev_f[i]) for i in range(n))
            posterior = PosteriorState(
                prior_mean=prior_mean_f,
                prior_std=prior_std_f,
                posterior_mean=b.to_floats(posterior_mean),
                posterior_std=b.to_floats(posterior_std),
                sample_noise=b.to_floats(sample_noise),
                z_tilde=b.to_floats(z_tilde),
                hidden_state=h_f,
                posterior_drift=drift,
            )
        return EncodedSequence(
            posterior=posterior,
            sequence_length=len(step_vectors),
            summary=(
                f"backend({self._b.kind.value}) ndim_encoder n_z={n} "
                f"len={len(step_vectors)} drift={drift:.3f}"
            ),
        )

    # --- switch (mirrors NdimSwitchUnit.compute) ---

    def compute(
        self,
        *,
        z_tilde: tuple[float, ...],
        previous_code: tuple[float, ...],
        memory_signal: float,
        reflection_signal: float,
        active_family_outcome: float,
        active_family_reuse: float,
        active_family_persistence: float,
        external_switch_pressure_delta: float,
        params: NdimSwitchParameters,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        b = self._b
        n = self._n_z
        # IMPORTANT: the pure NdimSwitchUnit computes ``gate_input = delta + z_tilde``
        # where both are tuples, so this is tuple CONCATENATION (2*n_z dims), not
        # an elementwise add. That is why the gate FFN W1 has n_z*2 columns. We
        # replicate the concatenation and use the full W1.
        ffn = FfnParams(
            W1=b.matrix(params.gate_ffn.W1), b1=b.vector(params.gate_ffn.b1),
            W2=b.matrix(params.gate_ffn.W2), b2=b.vector(params.gate_ffn.b2),
        )
        with b.no_grad():
            delta = tuple(abs(z_tilde[i] - previous_code[i]) for i in range(n))
            gate_input = b.vector(tuple(delta) + tuple(z_tilde))
            raw = b.ffn_2layer(x=gate_input, params=ffn)
            continuation_bias = max(0.0, min(1.0,
                active_family_outcome * 0.22
                + active_family_reuse * 0.33
                + active_family_persistence * 0.45
            ))
            bias = (
                memory_signal * 0.1
                + reflection_signal * 0.2
                - continuation_bias * 0.30
                + external_switch_pressure_delta
            )
            bias_vec = b.vector(tuple(bias for _ in range(n)))
            beta_cont = b.sigmoid(b.add(raw, bias_vec))
            beta_cont_f = b.to_floats(beta_cont)
        threshold = 0.55
        beta_binary = tuple(1.0 if v >= threshold else 0.0 for v in beta_cont_f)
        scalar_mean = sum(beta_cont_f) / max(len(beta_cont_f), 1)
        return beta_cont_f, beta_binary, scalar_mean

    # --- decoder (mirrors NdimResidualDecoder.decode) ---

    def decode(self, *, latent_code: tuple[float, ...], params: NdimDecoderParameters) -> DecoderControl:
        b = self._b
        ffn = FfnParams(
            W1=b.matrix(params.decoder_ffn.W1), b1=b.vector(params.decoder_ffn.b1),
            W2=b.matrix(params.decoder_ffn.W2), b2=b.vector(params.decoder_ffn.b2),
        )
        with b.no_grad():
            latent = b.vector(latent_code)
            decoder_output = b.ffn_2layer(x=latent, params=ffn)
            applied_control = b.clamp(
                b.add(b.scale(latent, 0.65), b.scale(decoder_output, 0.35)), 0.0, 1.0
            )
            decoder_output_f = b.to_floats(decoder_output)
            applied_f = b.to_floats(applied_control)
        return DecoderControl(
            decoder_output=decoder_output_f,
            applied_control=applied_f,
            summary=f"backend({self._b.kind.value}) ndim_decoder applied",
        )


def resolve_runtime_ndim_backend(wiring_level: Any) -> TensorBackend:
    """ACTIVE -> torch (fallback pure); otherwise pure rollback baseline."""

    from volvence_zero.runtime import WiringLevel

    if wiring_level is WiringLevel.ACTIVE:
        return resolve_backend(prefer=BackendKind.TORCH, allow_fallback=True).backend
    return PurePythonBackend()


@dataclass(frozen=True)
class RuntimeNdimShadowReport:
    steps_compared: int
    max_abs_diff_posterior_mean: float
    max_abs_diff_z_tilde: float
    max_abs_diff_beta: float
    max_abs_diff_applied: float
    within_tolerance: bool
    tolerance: float
    pure_latency_ms: float
    torch_latency_ms: float
    latency_budget_ms: float
    latency_ok: bool
    promotable: bool
    torch_available: bool
    description: str = ""


def runtime_ndim_shadow_compare(
    *,
    store: Any,
    substrate_snapshot: Any,
    previous_code: tuple[float, ...] | None = None,
    previous_hidden_state: tuple[float, ...] | None = None,
    cms_context: tuple[float, ...] | None = None,
    memory_signal: float = 0.0,
    reflection_signal: float = 0.0,
    active_family_outcome: float = 0.0,
    active_family_reuse: float = 0.0,
    active_family_persistence: float = 0.0,
    external_switch_pressure_delta: float = 0.0,
    tolerance: float = 1e-7,
    latency_budget_ms: float = 50.0,
) -> RuntimeNdimShadowReport:
    """Phase B SHADOW gate: pure vs torch ndim forward parity + latency.

    Runs encode -> switch -> decode on the SAME store params via both backends
    and reports the worst field divergence. Promotion to ACTIVE requires parity
    within tolerance and torch latency within budget.
    """

    n_z = store.n_z
    enc_p = store.ndim_encoder_parameters
    sw_p = store.ndim_switch_parameters
    dec_p = store.ndim_decoder_parameters
    if enc_p is None or sw_p is None or dec_p is None:
        raise RuntimeError("runtime_ndim_shadow_compare requires ndim params (n_z > 3).")
    prev = previous_code if previous_code is not None else tuple(0.0 for _ in range(n_z))

    def run(backend: TensorBackend) -> tuple[dict, float]:
        mc = BackendNdimMetacontroller(backend, n_z=n_z)
        t0 = time.perf_counter()
        enc = mc.encode(
            substrate_snapshot=substrate_snapshot,
            previous_hidden_state=previous_hidden_state,
            cms_context=cms_context,
            params=enc_p,
        )
        beta_cont, _, scalar = mc.compute(
            z_tilde=enc.posterior.z_tilde,
            previous_code=prev,
            memory_signal=memory_signal,
            reflection_signal=reflection_signal,
            active_family_outcome=active_family_outcome,
            active_family_reuse=active_family_reuse,
            active_family_persistence=active_family_persistence,
            external_switch_pressure_delta=external_switch_pressure_delta,
            params=sw_p,
        )
        latent = tuple(
            max(0.0, min(1.0, beta_cont[i] * enc.posterior.z_tilde[i] + (1.0 - beta_cont[i]) * prev[i]))
            for i in range(n_z)
        )
        dec = mc.decode(latent_code=latent, params=dec_p)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return (
            {
                "posterior_mean": enc.posterior.posterior_mean,
                "z_tilde": enc.posterior.z_tilde,
                "beta": beta_cont,
                "applied": dec.applied_control,
            },
            elapsed,
        )

    pure_out, pure_latency = run(PurePythonBackend())
    if not is_torch_available():
        return RuntimeNdimShadowReport(
            steps_compared=1, max_abs_diff_posterior_mean=0.0, max_abs_diff_z_tilde=0.0,
            max_abs_diff_beta=0.0, max_abs_diff_applied=0.0, within_tolerance=False,
            tolerance=tolerance, pure_latency_ms=pure_latency, torch_latency_ms=0.0,
            latency_budget_ms=latency_budget_ms, latency_ok=False, promotable=False,
            torch_available=False, description="torch unavailable; cannot promote",
        )
    res = resolve_backend(prefer=BackendKind.TORCH, allow_fallback=False)
    torch_out, torch_latency = run(res.backend)

    def md(key: str) -> float:
        return max((abs(a - b) for a, b in zip(pure_out[key], torch_out[key])), default=0.0)

    d_mean, d_z, d_beta, d_applied = md("posterior_mean"), md("z_tilde"), md("beta"), md("applied")
    within = max(d_mean, d_z, d_beta, d_applied) <= tolerance
    latency_ok = torch_latency <= latency_budget_ms
    return RuntimeNdimShadowReport(
        steps_compared=1,
        max_abs_diff_posterior_mean=d_mean,
        max_abs_diff_z_tilde=d_z,
        max_abs_diff_beta=d_beta,
        max_abs_diff_applied=d_applied,
        within_tolerance=within,
        tolerance=tolerance,
        pure_latency_ms=pure_latency,
        torch_latency_ms=torch_latency,
        latency_budget_ms=latency_budget_ms,
        latency_ok=latency_ok,
        promotable=within and latency_ok,
        torch_available=True,
        description=(
            f"runtime ndim SHADOW parity: mean={d_mean:.2e} z={d_z:.2e} "
            f"beta={d_beta:.2e} applied={d_applied:.2e} within={within} latency_ok={latency_ok}"
        ),
    )
