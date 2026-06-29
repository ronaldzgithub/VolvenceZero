"""Parity harness for the pure-Python vs torch tensor backends (Phase 0 exit gate).

A SHADOW rollout must prove that the torch path reproduces the pure-Python
baseline forward pass field-by-field within a tight tolerance before any owner
is promoted to ``WiringLevel.ACTIVE``. This module provides a small, dependency
-light comparison utility plus canonical forward fixtures (GRU cell + 2-layer
FFN) wired from explicit shared weights so both backends start identical.

The harness is intentionally forward-only: parity is about numerical
equivalence of the deterministic forward computation. Learning-direction
agreement is checked separately in the per-phase training tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.tensor_backend import (
    BackendKind,
    FfnParams,
    GruParams,
    PurePythonBackend,
    TensorBackend,
    TorchBackend,
    is_torch_available,
)


@dataclass(frozen=True)
class ParityResult:
    """Field-by-field comparison of two backends on one forward fixture."""

    fixture: str
    max_abs_diff: float
    within_tolerance: bool
    tolerance: float
    pure_output: tuple[float, ...]
    torch_output: tuple[float, ...]


def _seeded_floats(n: int, *, seed: int, scale: float = 0.3) -> tuple[float, ...]:
    """Deterministic explicit weights shared by BOTH backends (no RNG skew)."""

    import random

    rng = random.Random(seed)
    return tuple(rng.gauss(0.0, scale) for _ in range(n))


def _gru_params(backend: TensorBackend, *, input_dim: int, hidden_dim: int) -> GruParams:
    def mat(seed: int) -> object:
        rows = [
            list(_seeded_floats(input_dim if seed % 2 == 0 else hidden_dim, seed=seed * 100 + r))
            for r in range(hidden_dim)
        ]
        return backend.matrix(rows)

    def w(seed: int) -> object:
        rows = [list(_seeded_floats(input_dim, seed=seed * 100 + r)) for r in range(hidden_dim)]
        return backend.matrix(rows)

    def u(seed: int) -> object:
        rows = [list(_seeded_floats(hidden_dim, seed=seed * 100 + r)) for r in range(hidden_dim)]
        return backend.matrix(rows)

    return GruParams(
        W_z=w(1), U_z=u(2), b_z=backend.vector(_seeded_floats(hidden_dim, seed=3)),
        W_r=w(4), U_r=u(5), b_r=backend.vector(_seeded_floats(hidden_dim, seed=6)),
        W_h=w(7), U_h=u(8), b_h=backend.vector(_seeded_floats(hidden_dim, seed=9)),
    )


def _ffn_params(backend: TensorBackend, *, input_dim: int, hidden_dim: int, output_dim: int) -> FfnParams:
    w1 = [list(_seeded_floats(input_dim, seed=10 * r + 1)) for r in range(hidden_dim)]
    w2 = [list(_seeded_floats(hidden_dim, seed=20 * r + 2)) for r in range(output_dim)]
    return FfnParams(
        W1=backend.matrix(w1),
        b1=backend.vector(_seeded_floats(hidden_dim, seed=99)),
        W2=backend.matrix(w2),
        b2=backend.vector(_seeded_floats(output_dim, seed=98)),
    )


def _gru_forward(backend: TensorBackend, *, input_dim: int = 6, hidden_dim: int = 8) -> tuple[float, ...]:
    params = _gru_params(backend, input_dim=input_dim, hidden_dim=hidden_dim)
    x = backend.vector(_seeded_floats(input_dim, seed=777))
    h = backend.zeros(hidden_dim)
    for step in range(4):
        x_step = backend.vector(_seeded_floats(input_dim, seed=777 + step))
        h = backend.gru_cell(x=x_step, h_prev=h, params=params)
    return backend.to_floats(h)


def _ffn_forward(
    backend: TensorBackend, *, input_dim: int = 8, hidden_dim: int = 10, output_dim: int = 4
) -> tuple[float, ...]:
    params = _ffn_params(backend, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    x = backend.vector(_seeded_floats(input_dim, seed=555))
    return backend.to_floats(backend.ffn_2layer(x=x, params=params))


_FIXTURES = {
    "gru_cell_4step": _gru_forward,
    "ffn_2layer": _ffn_forward,
}


def compare_fixture(fixture: str, *, tolerance: float = 1e-9) -> ParityResult:
    """Run one forward fixture on both backends and compare outputs.

    Raises if torch is unavailable — the caller (test) should skip in that case.
    """

    if not is_torch_available():
        raise RuntimeError(
            "Parity comparison needs both backends; torch is not installed."
        )
    if fixture not in _FIXTURES:
        raise KeyError(f"unknown parity fixture: {fixture!r}; known: {sorted(_FIXTURES)}")
    fn = _FIXTURES[fixture]
    pure_out = fn(PurePythonBackend())
    torch_out = fn(TorchBackend())
    diffs = [abs(p - t) for p, t in zip(pure_out, torch_out)]
    max_diff = max(diffs) if diffs else 0.0
    return ParityResult(
        fixture=fixture,
        max_abs_diff=max_diff,
        within_tolerance=max_diff <= tolerance and len(pure_out) == len(torch_out),
        tolerance=tolerance,
        pure_output=pure_out,
        torch_output=torch_out,
    )


def run_all_fixtures(*, tolerance: float = 1e-9) -> tuple[ParityResult, ...]:
    """Compare every canonical fixture; the Phase 0 exit gate asserts all pass."""

    return tuple(compare_fixture(name, tolerance=tolerance) for name in _FIXTURES)


def all_backend_kinds() -> tuple[BackendKind, ...]:
    """Backends actually available here (torch may be absent in minimal CI)."""

    if is_torch_available():
        return (BackendKind.PURE, BackendKind.TORCH)
    return (BackendKind.PURE,)
