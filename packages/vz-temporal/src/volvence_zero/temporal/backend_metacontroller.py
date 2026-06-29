"""Backend-agnostic runtime metacontroller + SHADOW dual-run (NL/ETA Phase 3).

Phase 3 promotes the metacontroller's *runtime* forward from pure-Python tuples
to real torch autograd. The safety mechanism is a backend-agnostic forward:
the encoder (GRU) + switch gate + residual decoder are expressed entirely in
:class:`volvence_zero.tensor_backend.TensorBackend` primitives, so the **same
model with the same weights** can run on either backend.

That gives an exact rollback story (R15):

- ``WiringLevel.DISABLED`` -> pure backend only (the baseline).
- ``WiringLevel.SHADOW``   -> run BOTH backends, compare the controller output
  field by field (z_t / beta_t / control), and measure latency. Promotion is
  gated on parity-within-tolerance AND latency budget.
- ``WiringLevel.ACTIVE``   -> torch backend (falls back to pure if torch absent).

Because the forward is shared, the torch path is not a different model that
merely "looks like" the pure one — it computes the identical function, proven
numerically by the SHADOW comparison. Controller outputs are exported to float
tuples at the boundary so the public ``temporal_abstraction`` snapshot schema
is unchanged (R8).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

from volvence_zero.runtime import WiringLevel
from volvence_zero.tensor_backend import (
    BackendKind,
    FfnParams,
    GruParams,
    PurePythonBackend,
    TensorBackend,
    is_torch_available,
    resolve_backend,
)


@dataclass(frozen=True)
class RuntimeMetacontrollerConfig:
    n_z: int = 8
    input_dim: int = 3
    hidden_dim: int = 8
    action_dim: int = 3
    switch_threshold: float = 0.5
    seed: int = 1234


@dataclass(frozen=True)
class RuntimeMetacontrollerWeights:
    """Explicit float weights shared verbatim by both backends.

    Kept as plain Python floats so the same weights materialize identically on
    pure and torch backends (the basis for exact SHADOW parity), and so a
    trained model can be deployed via the rare-heavy artifact path.
    """

    # GRU
    W_z: tuple[tuple[float, ...], ...]
    U_z: tuple[tuple[float, ...], ...]
    b_z: tuple[float, ...]
    W_r: tuple[tuple[float, ...], ...]
    U_r: tuple[tuple[float, ...], ...]
    b_r: tuple[float, ...]
    W_h: tuple[tuple[float, ...], ...]
    U_h: tuple[tuple[float, ...], ...]
    b_h: tuple[float, ...]
    # latent head: hidden -> n_z (z_tilde)
    W_lat: tuple[tuple[float, ...], ...]
    b_lat: tuple[float, ...]
    # switch: [hidden, z_tilde, z_prev] -> 1
    W_sw: tuple[tuple[float, ...], ...]
    b_sw: tuple[float, ...]
    # decoder FFN: z_t -> hidden -> action_dim
    dec_W1: tuple[tuple[float, ...], ...]
    dec_b1: tuple[float, ...]
    dec_W2: tuple[tuple[float, ...], ...]
    dec_b2: tuple[float, ...]


def _rand_mat(rows: int, cols: int, rng: random.Random, scale: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(rng.gauss(0.0, scale) for _ in range(cols)) for _ in range(rows))


def build_seeded_weights(config: RuntimeMetacontrollerConfig) -> RuntimeMetacontrollerWeights:
    """Deterministic explicit weights (identical floats for both backends)."""

    rng = random.Random(config.seed)
    h = config.hidden_dim
    n_z = config.n_z
    in_dim = config.input_dim
    s_in = 1.0 / math.sqrt(max(in_dim, 1))
    s_h = 1.0 / math.sqrt(max(h, 1))
    s_sw = 1.0 / math.sqrt(max(h + 2 * n_z, 1))
    s_z = 1.0 / math.sqrt(max(h, 1))
    return RuntimeMetacontrollerWeights(
        W_z=_rand_mat(h, in_dim, rng, s_in), U_z=_rand_mat(h, h, rng, s_h), b_z=tuple(0.0 for _ in range(h)),
        W_r=_rand_mat(h, in_dim, rng, s_in), U_r=_rand_mat(h, h, rng, s_h), b_r=tuple(0.0 for _ in range(h)),
        W_h=_rand_mat(h, in_dim, rng, s_in), U_h=_rand_mat(h, h, rng, s_h), b_h=tuple(0.0 for _ in range(h)),
        W_lat=_rand_mat(n_z, h, rng, s_z), b_lat=tuple(0.0 for _ in range(n_z)),
        W_sw=_rand_mat(1, h + 2 * n_z, rng, s_sw), b_sw=(0.0,),
        dec_W1=_rand_mat(h, n_z, rng, s_z), dec_b1=tuple(0.0 for _ in range(h)),
        dec_W2=_rand_mat(config.action_dim, h, rng, s_h), dec_b2=tuple(0.0 for _ in range(config.action_dim)),
    )


@dataclass(frozen=True)
class RuntimeControllerStep:
    z_t: tuple[float, ...]
    beta: float
    switched: bool
    control: tuple[float, ...]
    steps_since_switch: int
    switch_sparsity: float


class BackendMetacontroller:
    """One metacontroller forward, runnable on any :class:`TensorBackend`."""

    def __init__(
        self,
        backend: TensorBackend,
        weights: RuntimeMetacontrollerWeights,
        config: RuntimeMetacontrollerConfig,
    ) -> None:
        self._b = backend
        self.config = config
        b = backend
        self._gru = GruParams(
            W_z=b.matrix(weights.W_z), U_z=b.matrix(weights.U_z), b_z=b.vector(weights.b_z),
            W_r=b.matrix(weights.W_r), U_r=b.matrix(weights.U_r), b_r=b.vector(weights.b_r),
            W_h=b.matrix(weights.W_h), U_h=b.matrix(weights.U_h), b_h=b.vector(weights.b_h),
        )
        self._W_lat = b.matrix(weights.W_lat)
        self._b_lat = b.vector(weights.b_lat)
        self._W_sw = b.matrix(weights.W_sw)
        self._b_sw = b.vector(weights.b_sw)
        self._dec = FfnParams(
            W1=b.matrix(weights.dec_W1), b1=b.vector(weights.dec_b1),
            W2=b.matrix(weights.dec_W2), b2=b.vector(weights.dec_b2),
        )

    def run_sequence(self, summaries: Sequence[tuple[float, ...]]) -> list[RuntimeControllerStep]:
        b = self._b
        h_state = b.zeros(self.config.hidden_dim)
        z_prev = b.zeros(self.config.n_z)
        steps_since_switch = 0
        out: list[RuntimeControllerStep] = []
        with b.no_grad():
            for raw in summaries:
                x = b.vector(raw)
                h_state = b.gru_cell(x=x, h_prev=h_state, params=self._gru)
                z_tilde = b.add(b.matvec(self._W_lat, h_state), self._b_lat)
                # switch gate over [h, z_tilde, z_prev]
                feat = b.vector(
                    tuple(b.to_floats(h_state)) + tuple(b.to_floats(z_tilde)) + tuple(b.to_floats(z_prev))
                )
                beta_vec = b.sigmoid(b.add(b.matvec(self._W_sw, feat), self._b_sw))
                beta = b.to_floats(beta_vec)[0]
                switched = beta >= self.config.switch_threshold
                hard = 1.0 if switched else 0.0
                # runtime: deterministic hard gate (no STE needed at inference)
                z_t = b.add(b.scale(z_tilde, hard), b.scale(z_prev, 1.0 - hard))
                control = b.ffn_2layer(x=z_t, params=self._dec)
                if switched:
                    steps_since_switch = 0
                else:
                    steps_since_switch += 1
                out.append(
                    RuntimeControllerStep(
                        z_t=b.to_floats(z_t),
                        beta=beta,
                        switched=switched,
                        control=b.to_floats(control),
                        steps_since_switch=steps_since_switch,
                        switch_sparsity=1.0 - beta,
                    )
                )
                z_prev = z_t
        return out


# ---------------------------------------------------------------------------
# SHADOW dual-run comparison + WiringLevel selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowComparisonReport:
    steps: int
    max_abs_diff_z: float
    max_abs_diff_beta: float
    max_abs_diff_control: float
    within_tolerance: bool
    tolerance: float
    pure_latency_ms: float
    torch_latency_ms: float
    latency_budget_ms: float
    latency_ok: bool
    promotable: bool
    torch_available: bool
    description: str = ""


def shadow_dual_run(
    *,
    config: RuntimeMetacontrollerConfig | None = None,
    weights: RuntimeMetacontrollerWeights | None = None,
    summaries: Sequence[tuple[float, ...]] | None = None,
    tolerance: float = 1e-7,
    latency_budget_ms: float = 50.0,
) -> ShadowComparisonReport:
    """Run the SAME metacontroller on pure + torch backends and compare.

    Phase 3 exit gate. Promotion to ACTIVE requires both: (1) the torch forward
    reproduces the pure forward within ``tolerance`` (exact-rollback guarantee),
    and (2) torch latency is within budget.
    """

    cfg = config or RuntimeMetacontrollerConfig()
    w = weights or build_seeded_weights(cfg)
    seqs = list(summaries) if summaries is not None else [
        (math.sin(i * 0.4), math.cos(i * 0.3), (i % 5) / 5.0) for i in range(12)
    ]

    pure_backend = PurePythonBackend()
    t0 = time.perf_counter()
    pure_steps = BackendMetacontroller(pure_backend, w, cfg).run_sequence(seqs)
    pure_latency = (time.perf_counter() - t0) * 1000.0

    if not is_torch_available():
        return ShadowComparisonReport(
            steps=len(pure_steps), max_abs_diff_z=0.0, max_abs_diff_beta=0.0,
            max_abs_diff_control=0.0, within_tolerance=False, tolerance=tolerance,
            pure_latency_ms=pure_latency, torch_latency_ms=0.0,
            latency_budget_ms=latency_budget_ms, latency_ok=False, promotable=False,
            torch_available=False,
            description="torch unavailable; SHADOW comparison cannot promote",
        )

    res = resolve_backend(prefer=BackendKind.TORCH, allow_fallback=False)
    t0 = time.perf_counter()
    torch_steps = BackendMetacontroller(res.backend, w, cfg).run_sequence(seqs)
    torch_latency = (time.perf_counter() - t0) * 1000.0

    max_z = max_beta = max_ctrl = 0.0
    for ps, ts in zip(pure_steps, torch_steps):
        max_z = max(max_z, max((abs(a - b) for a, b in zip(ps.z_t, ts.z_t)), default=0.0))
        max_beta = max(max_beta, abs(ps.beta - ts.beta))
        max_ctrl = max(max_ctrl, max((abs(a - b) for a, b in zip(ps.control, ts.control)), default=0.0))
    within = max(max_z, max_beta, max_ctrl) <= tolerance
    latency_ok = torch_latency <= latency_budget_ms
    return ShadowComparisonReport(
        steps=len(pure_steps),
        max_abs_diff_z=max_z,
        max_abs_diff_beta=max_beta,
        max_abs_diff_control=max_ctrl,
        within_tolerance=within,
        tolerance=tolerance,
        pure_latency_ms=pure_latency,
        torch_latency_ms=torch_latency,
        latency_budget_ms=latency_budget_ms,
        latency_ok=latency_ok,
        promotable=within and latency_ok,
        torch_available=True,
        description=(
            f"SHADOW parity: max_diff(z={max_z:.2e}, beta={max_beta:.2e}, "
            f"ctrl={max_ctrl:.2e}) within={within} latency_ok={latency_ok}"
        ),
    )


def resolve_runtime_backend(wiring_level: WiringLevel) -> TensorBackend:
    """Map WiringLevel to the runtime metacontroller backend.

    DISABLED -> pure baseline; SHADOW -> pure (torch runs in parallel via
    :func:`shadow_dual_run`, not on the live path); ACTIVE -> torch (explicit
    fallback to pure if torch is unavailable).
    """

    if wiring_level is WiringLevel.ACTIVE:
        return resolve_backend(prefer=BackendKind.TORCH, allow_fallback=True).backend
    return PurePythonBackend()
