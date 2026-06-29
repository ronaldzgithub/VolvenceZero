"""Real-autograd CMS band + SHADOW dual-run (NL/ETA migration, Phase 4).

The CMS band is NL's continuum-memory associative store. The production band
(:class:`volvence_zero.memory.cms_band_mlp.CMSBandMLP`) is a 2-layer residual
MLP ``y = clamp(x + W1 @ tanh(W2 @ x))`` learned by *manual* backprop + momentum.

Phase 4 migrates this to real torch autograd while preserving the legacy forward
exactly, so the pure-Python band stays the rollback baseline:

- :class:`BackendCMSBand` — backend-agnostic forward (pure OR torch) over the
  SAME weights, used for the SHADOW parity comparison.
- :class:`TorchCMSBand` — real autograd band: the same forward, learned by
  ``torch.autograd`` (MSE-to-target) instead of hand-derived gradients.
- :func:`cms_band_shadow_dual_run` — proves the torch forward reproduces BOTH
  the pure-backend forward AND the legacy ``CMSBandMLP.forward`` within a tight
  tolerance (exact rollback), plus a latency gate.

``torch`` is imported lazily and this module is NOT re-exported from the
``volvence_zero.memory`` facade, keeping it torch-free by default.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from volvence_zero.memory.cms_band_mlp import CMSBandMLP
from volvence_zero.memory.cms_math import _init_weight
from volvence_zero.tensor_backend import (
    BackendKind,
    PurePythonBackend,
    TensorBackend,
    is_torch_available,
    resolve_backend,
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded at call sites
        raise ImportError(
            "Torch CMS band requires torch. Install the vz-memory '[torch]' extra."
        ) from exc
    return torch


def _reshape(flat: Sequence[float], rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(flat[i * cols + j]) for j in range(cols)) for i in range(rows)
    )


@dataclass(frozen=True)
class CMSBandWeights:
    """Explicit float weights shared verbatim by all band implementations."""

    d_in: int
    d_hidden: int
    w1: tuple[float, ...]  # [d_in x d_hidden] row-major (legacy layout)
    w2: tuple[float, ...]  # [d_hidden x d_in] row-major (legacy layout)


def build_seeded_band_weights(*, d_in: int = 8, d_hidden: int = 16) -> CMSBandWeights:
    """Match the legacy band init: W1 == 0 (identity start), W2 == _init_weight."""

    return CMSBandWeights(
        d_in=d_in,
        d_hidden=d_hidden,
        w1=tuple(0.0 for _ in range(d_in * d_hidden)),
        w2=tuple(_init_weight(d_hidden * d_in, 0.01)),
    )


def _nontrivial_band_weights(*, d_in: int = 8, d_hidden: int = 16, seed: int = 7) -> CMSBandWeights:
    """Non-zero W1 so parity is exercised on a non-identity residual path."""

    import random

    rng = random.Random(seed)
    return CMSBandWeights(
        d_in=d_in,
        d_hidden=d_hidden,
        w1=tuple(rng.gauss(0.0, 0.05) for _ in range(d_in * d_hidden)),
        w2=tuple(_init_weight(d_hidden * d_in, 0.01)),
    )


class BackendCMSBand:
    """Backend-agnostic CMS band forward (matches the legacy forward exactly)."""

    def __init__(self, backend: TensorBackend, weights: CMSBandWeights) -> None:
        self._b = backend
        self.weights = weights
        self._W2 = backend.matrix(_reshape(weights.w2, weights.d_hidden, weights.d_in))
        self._W1 = backend.matrix(_reshape(weights.w1, weights.d_in, weights.d_hidden))

    def forward(self, x: Sequence[float]) -> tuple[float, ...]:
        b = self._b
        with b.no_grad():
            xv = b.vector(x)
            h = b.matvec(self._W2, xv)
            a = b.tanh(h)
            residual = b.matvec(self._W1, a)
            y = b.clamp(b.add(xv, residual), 0.0, 1.0)
            return b.to_floats(y)


def _legacy_band_from_weights(weights: CMSBandWeights) -> CMSBandMLP:
    band = CMSBandMLP(d_in=weights.d_in, d_hidden=weights.d_hidden)
    # restore_params expects: (state, state_momentum, w2, w1, w2_momentum, w1_momentum)
    band.restore_params(
        (
            tuple(0.0 for _ in range(weights.d_in)),
            tuple(0.0 for _ in range(weights.d_in)),
            tuple(weights.w2),
            tuple(weights.w1),
            tuple(0.0 for _ in range(weights.d_hidden * weights.d_in)),
            tuple(0.0 for _ in range(weights.d_in * weights.d_hidden)),
        )
    )
    return band


class TorchCMSBand:
    """Real-autograd CMS band: same forward, learned via torch.autograd."""

    def __init__(self, weights: CMSBandWeights, *, learning_rate: float = 0.1) -> None:
        self._torch = _require_torch()
        torch = self._torch
        self.weights = weights
        self._dtype = torch.float64
        self._W2 = torch.tensor(
            _reshape(weights.w2, weights.d_hidden, weights.d_in), dtype=self._dtype, requires_grad=True
        )
        self._W1 = torch.tensor(
            _reshape(weights.w1, weights.d_in, weights.d_hidden), dtype=self._dtype, requires_grad=True
        )
        self._opt = torch.optim.SGD([self._W1, self._W2], lr=learning_rate, momentum=0.9)

    def parameters(self) -> list[Any]:
        return [self._W1, self._W2]

    def forward_tensor(self, x: Any) -> Any:
        torch = self._torch
        h = torch.matmul(self._W2, x)
        a = torch.tanh(h)
        residual = torch.matmul(self._W1, a)
        return torch.clamp(x + residual, 0.0, 1.0)

    def forward(self, x: Sequence[float]) -> tuple[float, ...]:
        torch = self._torch
        with torch.no_grad():
            xv = torch.tensor(list(x), dtype=self._dtype)
            return tuple(float(v) for v in self.forward_tensor(xv).tolist())

    def update(self, *, x: Sequence[float], target: Sequence[float]) -> dict[str, float]:
        """One real-autograd SGD step minimizing ||forward(x) - target||^2."""

        torch = self._torch
        before = [p.detach().clone() for p in self.parameters()]
        xv = torch.tensor(list(x), dtype=self._dtype)
        tv = torch.tensor(list(target), dtype=self._dtype)
        self._opt.zero_grad()
        y = self.forward_tensor(xv)
        loss = torch.mean((y - tv).pow(2))
        loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in self.parameters() if p.grad is not None))
        self._opt.step()
        changed = 0
        total = 0
        for b, a in zip(before, self.parameters()):
            diff = (a.detach() - b).abs()
            changed += int((diff > 1e-12).sum())
            total += int(diff.numel())
        return {
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "parameters_changed": float(changed),
            "parameter_change_rate": changed / max(total, 1),
        }


@dataclass(frozen=True)
class CMSBandShadowReport:
    d_in: int
    d_hidden: int
    max_abs_diff_pure_torch: float
    max_abs_diff_legacy_torch: float
    within_tolerance: bool
    tolerance: float
    pure_latency_ms: float
    torch_latency_ms: float
    latency_budget_ms: float
    latency_ok: bool
    promotable: bool
    torch_available: bool
    description: str = ""


def cms_band_shadow_dual_run(
    *,
    weights: CMSBandWeights | None = None,
    inputs: Sequence[Sequence[float]] | None = None,
    tolerance: float = 1e-9,
    latency_budget_ms: float = 50.0,
) -> CMSBandShadowReport:
    """Compare torch band forward vs pure-backend AND legacy band forwards.

    Phase 4 exit gate. Promotion requires the torch forward to reproduce both
    the pure-backend forward and the *production* legacy ``CMSBandMLP.forward``
    within tolerance, plus a latency budget — the exact-rollback guarantee.
    """

    w = weights or _nontrivial_band_weights()
    xs = list(inputs) if inputs is not None else [
        tuple((math.sin(i * 0.3 + j) + 1.0) / 2.0 for j in range(w.d_in)) for i in range(8)
    ]

    pure_band = BackendCMSBand(PurePythonBackend(), w)
    legacy = _legacy_band_from_weights(w)
    t0 = time.perf_counter()
    pure_outs = [pure_band.forward(x) for x in xs]
    pure_latency = (time.perf_counter() - t0) * 1000.0
    legacy_outs = [legacy.forward(tuple(x)) for x in xs]

    if not is_torch_available():
        return CMSBandShadowReport(
            d_in=w.d_in, d_hidden=w.d_hidden, max_abs_diff_pure_torch=0.0,
            max_abs_diff_legacy_torch=0.0, within_tolerance=False, tolerance=tolerance,
            pure_latency_ms=pure_latency, torch_latency_ms=0.0,
            latency_budget_ms=latency_budget_ms, latency_ok=False, promotable=False,
            torch_available=False, description="torch unavailable; cannot promote",
        )

    torch_band = TorchCMSBand(w)
    t0 = time.perf_counter()
    torch_outs = [torch_band.forward(x) for x in xs]
    torch_latency = (time.perf_counter() - t0) * 1000.0

    def _maxdiff(a_seq, b_seq) -> float:
        m = 0.0
        for a, b in zip(a_seq, b_seq):
            m = max(m, max((abs(p - q) for p, q in zip(a, b)), default=0.0))
        return m

    diff_pt = _maxdiff(pure_outs, torch_outs)
    diff_lt = _maxdiff(legacy_outs, torch_outs)
    within = max(diff_pt, diff_lt) <= tolerance
    latency_ok = torch_latency <= latency_budget_ms
    return CMSBandShadowReport(
        d_in=w.d_in, d_hidden=w.d_hidden,
        max_abs_diff_pure_torch=diff_pt,
        max_abs_diff_legacy_torch=diff_lt,
        within_tolerance=within,
        tolerance=tolerance,
        pure_latency_ms=pure_latency,
        torch_latency_ms=torch_latency,
        latency_budget_ms=latency_budget_ms,
        latency_ok=latency_ok,
        promotable=within and latency_ok,
        torch_available=True,
        description=(
            f"CMS band SHADOW parity: pure~torch={diff_pt:.2e}, "
            f"legacy~torch={diff_lt:.2e}, within={within}, latency_ok={latency_ok}"
        ),
    )


def resolve_cms_backend(active: bool) -> TensorBackend:
    """ACTIVE -> torch (fallback pure); otherwise the pure rollback baseline."""

    if active:
        return resolve_backend(prefer=BackendKind.TORCH, allow_fallback=True).backend
    return PurePythonBackend()


@dataclass(frozen=True)
class TorchBandStepResult:
    w1_flat: tuple[float, ...]
    w2_flat: tuple[float, ...]
    loss: float
    grad_norm: float
    parameters_changed: int
    parameter_change_rate: float


def torch_band_update_from_params(
    *,
    d_in: int,
    d_hidden: int,
    w1_flat: Sequence[float],
    w2_flat: Sequence[float],
    state: Sequence[float],
    target: Sequence[float],
    learning_rate: float = 0.1,
) -> TorchBandStepResult:
    """Owner-mainline CMS band gradient step via real torch autograd.

    Seeds a :class:`TorchCMSBand` from a live ``CMSBandMLP``'s current weights
    (flat layouts), runs one autograd SGD step minimizing ``||y(state) - target||^2``,
    and returns the refined flat W1/W2 in the SAME layout so the caller can write
    them back into the band (keeping the band's pure state/momentum for coherent
    backflow / mix_from / checkpointing).
    """

    weights = CMSBandWeights(
        d_in=d_in, d_hidden=d_hidden, w1=tuple(w1_flat), w2=tuple(w2_flat)
    )
    band = TorchCMSBand(weights, learning_rate=learning_rate)
    metrics = band.update(x=tuple(state), target=tuple(target))
    torch = band._torch
    w1_out = tuple(float(v) for row in band._W1.detach().tolist() for v in row)
    w2_out = tuple(float(v) for row in band._W2.detach().tolist() for v in row)
    return TorchBandStepResult(
        w1_flat=w1_out,
        w2_flat=w2_out,
        loss=metrics["loss"],
        grad_norm=metrics["grad_norm"],
        parameters_changed=int(metrics["parameters_changed"]),
        parameter_change_rate=metrics["parameter_change_rate"],
    )
