"""Backend-agnostic tensor abstraction (Phase 0 of the NL/ETA full-autograd migration).

Two numeric backends share one interface so the same metacontroller / CMS
forward logic can run either as the zero-dependency pure-Python rollback
baseline or as a real ``torch`` autograd path:

- :class:`PurePythonBackend` — tuple-of-float math, no external dependency.
  This is the rollback baseline (``WiringLevel.DISABLED``). It has **no**
  autograd; owners that select it keep their existing manual update rules.
- :class:`TorchBackend` — ``torch.Tensor`` + autograd + ``nn.Parameter``. This
  is the migration target (``WiringLevel.ACTIVE``). It is forced onto CPU and
  configured deterministically so a SHADOW dual-run can be compared field by
  field against the pure baseline.

Design invariants (see ``docs/specs/temporal-abstraction.md`` and the
``first-principles-not-patches`` / ``ssot-module-boundaries`` rules):

- This module lives in ``vz-contracts`` — the zero-upstream foundation — so both
  ``vz-temporal`` (metacontroller / SSL / internal RL) and ``vz-memory`` (CMS)
  can share one backend without crossing a wheel boundary (R8).
- ``torch`` is imported lazily inside :class:`TorchBackend`; importing this
  module never forces a torch dependency. If torch is missing, resolution
  falls back to pure **explicitly** (a named reason, never a silent swallow).
- Snapshot boundary: callers MUST convert tensors back to float tuples via
  :meth:`TensorBackend.to_floats` before publishing into any public snapshot,
  so the public schema never carries torch tensors (R8 — torch stays owner-internal).
"""

from __future__ import annotations

import math
import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Sequence


class BackendKind(str, Enum):
    """Which numeric backend an owner is running on."""

    PURE = "pure"
    TORCH = "torch"


# Tensor handles are intentionally opaque (``Any``): pure backend uses
# ``float`` / ``tuple[float, ...]`` / ``tuple[tuple[float, ...], ...]``; torch
# backend uses ``torch.Tensor``. Callers route everything through the backend.
Tensor = Any
Vec = tuple[float, ...]
Mat = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class GruParams:
    """GRU cell parameter group (handles are backend-specific)."""

    W_z: Tensor
    U_z: Tensor
    b_z: Tensor
    W_r: Tensor
    U_r: Tensor
    b_r: Tensor
    W_h: Tensor
    U_h: Tensor
    b_h: Tensor


@dataclass(frozen=True)
class FfnParams:
    """2-layer FFN parameter group (handles are backend-specific)."""

    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor


# ---------------------------------------------------------------------------
# torch availability (lazy, no module-level import cost)
# ---------------------------------------------------------------------------


def is_torch_available() -> bool:
    """Return True if ``torch`` can be imported in this environment."""

    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def configure_determinism(seed: int = 1234) -> None:
    """Pin RNG + deterministic algorithms so SHADOW dual-runs are comparable.

    Always seeds the stdlib RNG (used by the pure backend init). If torch is
    available, additionally pins ``manual_seed`` and forces deterministic
    algorithms on CPU. This is a no-op-safe call: importing torch is guarded.
    """

    random.seed(seed)
    if not is_torch_available():
        return
    import torch

    torch.manual_seed(seed)
    # Deterministic algorithms make the torch path reproducible for parity.
    # ``warn_only`` keeps us from hard-failing if a downstream op lacks a
    # deterministic kernel; we never run on non-CPU in the live runtime.
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class TensorBackend(ABC):
    """Numeric ops shared by the pure-Python baseline and the torch path.

    The op surface is deliberately small but sufficient to express a GRU
    encoder, a switch gate, a 2-layer residual decoder, a CMS band MLP, and a
    causal z-policy — i.e. every learnable component in the NL/ETA stack.
    """

    kind: BackendKind

    @property
    @abstractmethod
    def supports_autograd(self) -> bool:
        """Whether :meth:`backward` / :meth:`grad` are real (torch) or absent."""

    # --- construction ---

    @abstractmethod
    def vector(self, values: Sequence[float], *, requires_grad: bool = False) -> Tensor: ...

    @abstractmethod
    def matrix(self, rows: Sequence[Sequence[float]], *, requires_grad: bool = False) -> Tensor: ...

    @abstractmethod
    def zeros(self, n: int, *, requires_grad: bool = False) -> Tensor: ...

    @abstractmethod
    def scalar(self, value: float) -> Tensor: ...

    @abstractmethod
    def parameter(self, values: Sequence[float] | Sequence[Sequence[float]]) -> Tensor:
        """Create a leaf tensor that accumulates gradients (torch) / a plain
        handle (pure)."""

    # --- linear / elementwise ---

    @abstractmethod
    def matvec(self, matrix: Tensor, vector: Tensor) -> Tensor: ...

    @abstractmethod
    def add(self, a: Tensor, b: Tensor) -> Tensor: ...

    @abstractmethod
    def sub(self, a: Tensor, b: Tensor) -> Tensor: ...

    @abstractmethod
    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise (Hadamard) product."""

    @abstractmethod
    def scale(self, a: Tensor, s: float) -> Tensor: ...

    @abstractmethod
    def sigmoid(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def tanh(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def clamp(self, x: Tensor, lo: float, hi: float) -> Tensor: ...

    # --- reductions (return a scalar tensor handle) ---

    @abstractmethod
    def mean(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def sum(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def dot(self, a: Tensor, b: Tensor) -> Tensor: ...

    @abstractmethod
    def abs(self, x: Tensor) -> Tensor: ...

    # --- composite layers ---

    def gru_cell(self, *, x: Tensor, h_prev: Tensor, params: GruParams) -> Tensor:
        """One GRU step. Default implementation is backend-agnostic: it is
        expressed purely in terms of the primitive ops above, so it works on
        either backend (and keeps the torch path differentiable end-to-end)."""

        z_gate = self.sigmoid(
            self.add(self.add(self.matvec(params.W_z, x), self.matvec(params.U_z, h_prev)), params.b_z)
        )
        r_gate = self.sigmoid(
            self.add(self.add(self.matvec(params.W_r, x), self.matvec(params.U_r, h_prev)), params.b_r)
        )
        h_candidate = self.tanh(
            self.add(
                self.add(self.matvec(params.W_h, x), self.matvec(params.U_h, self.mul(r_gate, h_prev))),
                params.b_h,
            )
        )
        one = self.scale(z_gate, 0.0)  # zeros like z_gate
        one = self.add(one, self.vector(tuple(1.0 for _ in range(self._len(z_gate)))))
        one_minus_z = self.sub(one, z_gate)
        return self.add(self.mul(one_minus_z, h_prev), self.mul(z_gate, h_candidate))

    def ffn_2layer(self, *, x: Tensor, params: FfnParams) -> Tensor:
        """2-layer FFN with tanh after layer 1 (backend-agnostic)."""

        hidden = self.tanh(self.add(self.matvec(params.W1, x), params.b1))
        return self.add(self.matvec(params.W2, hidden), params.b2)

    # --- autograd ---

    @abstractmethod
    def backward(self, loss: Tensor) -> None: ...

    @abstractmethod
    def grad(self, param: Tensor) -> Tensor | None: ...

    @abstractmethod
    def detach(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    @contextmanager
    def no_grad(self) -> Iterator[None]: ...

    # --- export (snapshot boundary) ---

    @abstractmethod
    def to_floats(self, x: Tensor) -> Vec: ...

    @abstractmethod
    def to_nested_floats(self, x: Tensor) -> Mat: ...

    @abstractmethod
    def item(self, x: Tensor) -> float: ...

    # --- internal helper ---

    @abstractmethod
    def _len(self, x: Tensor) -> int: ...


# ---------------------------------------------------------------------------
# Pure-Python backend (rollback baseline)
# ---------------------------------------------------------------------------


def _sigmoid(value: float) -> float:
    if value > 20.0:
        return 1.0
    if value < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


class PurePythonBackend(TensorBackend):
    """Zero-dependency tuple math. No autograd; rollback baseline."""

    kind = BackendKind.PURE

    @property
    def supports_autograd(self) -> bool:
        return False

    def vector(self, values: Sequence[float], *, requires_grad: bool = False) -> Vec:
        return tuple(float(v) for v in values)

    def matrix(self, rows: Sequence[Sequence[float]], *, requires_grad: bool = False) -> Mat:
        return tuple(tuple(float(v) for v in row) for row in rows)

    def zeros(self, n: int, *, requires_grad: bool = False) -> Vec:
        return tuple(0.0 for _ in range(n))

    def scalar(self, value: float) -> float:
        return float(value)

    def parameter(self, values: Sequence[float] | Sequence[Sequence[float]]) -> Tensor:
        if values and isinstance(values[0], (list, tuple)):
            return self.matrix(values)  # type: ignore[arg-type]
        return self.vector(values)  # type: ignore[arg-type]

    def matvec(self, matrix: Mat, vector: Vec) -> Vec:
        return tuple(sum(w * v for w, v in zip(row, vector)) for row in matrix)

    def add(self, a: Vec, b: Vec) -> Vec:
        return tuple(x + y for x, y in zip(a, b))

    def sub(self, a: Vec, b: Vec) -> Vec:
        return tuple(x - y for x, y in zip(a, b))

    def mul(self, a: Vec, b: Vec) -> Vec:
        return tuple(x * y for x, y in zip(a, b))

    def scale(self, a: Vec, s: float) -> Vec:
        return tuple(x * s for x in a)

    def sigmoid(self, x: Vec) -> Vec:
        return tuple(_sigmoid(v) for v in x)

    def tanh(self, x: Vec) -> Vec:
        return tuple(math.tanh(v) for v in x)

    def clamp(self, x: Vec, lo: float, hi: float) -> Vec:
        return tuple(max(lo, min(hi, v)) for v in x)

    def mean(self, x: Vec) -> float:
        return sum(x) / max(len(x), 1)

    def sum(self, x: Vec) -> float:
        return float(sum(x))

    def dot(self, a: Vec, b: Vec) -> float:
        return float(sum(p * q for p, q in zip(a, b)))

    def abs(self, x: Vec) -> Vec:
        return tuple(abs(v) for v in x)

    def backward(self, loss: float) -> None:
        raise NotImplementedError(
            "PurePythonBackend has no autograd. Select the torch backend "
            "(WiringLevel.ACTIVE) for gradient-based learning, or use the "
            "owner's manual update rule."
        )

    def grad(self, param: Tensor) -> None:
        return None

    def detach(self, x: Tensor) -> Tensor:
        return x

    @contextmanager
    def no_grad(self) -> Iterator[None]:
        yield

    def to_floats(self, x: Vec) -> Vec:
        return tuple(float(v) for v in x)

    def to_nested_floats(self, x: Mat) -> Mat:
        return tuple(tuple(float(v) for v in row) for row in x)

    def item(self, x: Any) -> float:
        return float(x)

    def _len(self, x: Vec) -> int:
        return len(x)


# ---------------------------------------------------------------------------
# Torch backend (migration target)
# ---------------------------------------------------------------------------


class TorchBackend(TensorBackend):
    """Real autograd backend. CPU + float64 + deterministic for parity."""

    kind = BackendKind.TORCH

    def __init__(self, *, dtype: str = "float64") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by resolve_backend
            raise ImportError(
                "TorchBackend requires torch. Install via the package "
                "'[torch]' extra, or select the pure backend "
                "(WiringLevel.DISABLED)."
            ) from exc
        self._torch = torch
        self._device = torch.device("cpu")
        # float64 keeps the torch path numerically close to the pure-Python
        # baseline so SHADOW parity comparisons use a tight tolerance.
        self._dtype = getattr(torch, dtype)

    @property
    def supports_autograd(self) -> bool:
        return True

    def _t(self, data: Any, *, requires_grad: bool = False) -> Any:
        return self._torch.tensor(
            data, dtype=self._dtype, device=self._device, requires_grad=requires_grad
        )

    def vector(self, values: Sequence[float], *, requires_grad: bool = False) -> Any:
        return self._t(list(float(v) for v in values), requires_grad=requires_grad)

    def matrix(self, rows: Sequence[Sequence[float]], *, requires_grad: bool = False) -> Any:
        return self._t([[float(v) for v in row] for row in rows], requires_grad=requires_grad)

    def zeros(self, n: int, *, requires_grad: bool = False) -> Any:
        return self._torch.zeros(
            n, dtype=self._dtype, device=self._device, requires_grad=requires_grad
        )

    def scalar(self, value: float) -> Any:
        return self._t(float(value))

    def parameter(self, values: Sequence[float] | Sequence[Sequence[float]]) -> Any:
        if len(values) > 0 and isinstance(values[0], (list, tuple)):
            return self.matrix(values, requires_grad=True)  # type: ignore[arg-type]
        return self.vector(values, requires_grad=True)  # type: ignore[arg-type]

    def matvec(self, matrix: Any, vector: Any) -> Any:
        return self._torch.matmul(matrix, vector)

    def add(self, a: Any, b: Any) -> Any:
        return a + b

    def sub(self, a: Any, b: Any) -> Any:
        return a - b

    def mul(self, a: Any, b: Any) -> Any:
        return a * b

    def scale(self, a: Any, s: float) -> Any:
        return a * float(s)

    def sigmoid(self, x: Any) -> Any:
        return self._torch.sigmoid(x)

    def tanh(self, x: Any) -> Any:
        return self._torch.tanh(x)

    def clamp(self, x: Any, lo: float, hi: float) -> Any:
        return self._torch.clamp(x, float(lo), float(hi))

    def mean(self, x: Any) -> Any:
        return self._torch.mean(x)

    def sum(self, x: Any) -> Any:
        return self._torch.sum(x)

    def dot(self, a: Any, b: Any) -> Any:
        return self._torch.dot(a, b)

    def abs(self, x: Any) -> Any:
        return self._torch.abs(x)

    def backward(self, loss: Any) -> None:
        loss.backward()

    def grad(self, param: Any) -> Any | None:
        return param.grad

    def detach(self, x: Any) -> Any:
        return x.detach()

    @contextmanager
    def no_grad(self) -> Iterator[None]:
        with self._torch.no_grad():
            yield

    def to_floats(self, x: Any) -> Vec:
        flat = x.detach().cpu().reshape(-1).tolist()
        return tuple(float(v) for v in flat)

    def to_nested_floats(self, x: Any) -> Mat:
        nested = x.detach().cpu().tolist()
        return tuple(tuple(float(v) for v in row) for row in nested)

    def item(self, x: Any) -> float:
        return float(x.detach().cpu().item())

    def _len(self, x: Any) -> int:
        return int(x.shape[0])


# ---------------------------------------------------------------------------
# Resolution (WiringLevel-aware, explicit fallback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendResolution:
    """Outcome of :func:`resolve_backend` — backend plus a named reason.

    The reason is published so a SHADOW/ACTIVE rollout can audit *why* a given
    owner ended up on pure vs torch (never a silent swallow — see
    ``no-swallow-errors-no-hasattr-abuse``).
    """

    backend: TensorBackend
    kind: BackendKind
    requested: BackendKind
    fell_back: bool
    reason: str


# Env override lets ops/CI force a backend without code changes.
_ENV_BACKEND_VAR = "VZ_TENSOR_BACKEND"


def _requested_kind(prefer: BackendKind | str | None) -> BackendKind:
    env = os.environ.get(_ENV_BACKEND_VAR, "").strip().lower()
    if env in (BackendKind.PURE.value, BackendKind.TORCH.value):
        return BackendKind(env)
    if prefer is None:
        return BackendKind.PURE
    if isinstance(prefer, BackendKind):
        return prefer
    return BackendKind(str(prefer).lower())


def resolve_backend(
    *,
    prefer: BackendKind | str | None = BackendKind.PURE,
    allow_fallback: bool = True,
) -> BackendResolution:
    """Resolve a concrete backend, falling back to pure with a named reason.

    - ``prefer`` (or the ``VZ_TENSOR_BACKEND`` env var) chooses the target.
    - If torch is requested but unavailable: fall back to pure when
      ``allow_fallback`` (DISABLED-style safety), else raise loudly.
    """

    requested = _requested_kind(prefer)
    if requested is BackendKind.PURE:
        return BackendResolution(
            backend=PurePythonBackend(),
            kind=BackendKind.PURE,
            requested=requested,
            fell_back=False,
            reason="pure backend requested",
        )
    # torch requested
    if is_torch_available():
        return BackendResolution(
            backend=TorchBackend(),
            kind=BackendKind.TORCH,
            requested=requested,
            fell_back=False,
            reason="torch backend active",
        )
    if not allow_fallback:
        raise ImportError(
            "Torch backend requested but torch is not installed and "
            "allow_fallback=False. Install the '[torch]' extra."
        )
    return BackendResolution(
        backend=PurePythonBackend(),
        kind=BackendKind.PURE,
        requested=requested,
        fell_back=True,
        reason="torch unavailable; fell back to pure rollback baseline",
    )
