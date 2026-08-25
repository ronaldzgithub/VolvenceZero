"""Pheromone snapshot bus — a multi-writer, append-only, decaying 2D field.

This is the digital analogue of stigmergy: the colony's externalised shared
memory. It is deliberately modelled as an SSOT snapshot bus, not a mutable
shared object that ants poke at:

- ``PheromoneField`` is an IMMUTABLE published snapshot (numpy grids are made
  read-only). Individuals READ the current published snapshot.
- Individuals never write into each other's state; they emit independent,
  additive ``deposit`` events. At the end of a round the bus AGGREGATES all
  deposits + applies decay and PUBLISHES the next immutable snapshot.

So two ants can never overwrite the same field cell — deposits only ADD. This
respects the module-boundary rule (individuals communicate only through the
published snapshot; zero direct calls between individuals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PheromoneField:
    """Immutable published pheromone snapshot (two channels: home, trail)."""

    home: np.ndarray
    trail: np.ndarray
    cell_size: float
    origin_x: float
    origin_y: float
    tick: int
    home_mass: float = field(init=False)
    trail_mass: float = field(init=False)
    home_entropy: float | None = field(init=False)
    trail_entropy: float | None = field(init=False)

    def __post_init__(self) -> None:
        # Back arrays with immutable ``bytes``. Merely calling
        # ``setflags(write=False)`` is reversible when an ndarray owns mutable
        # memory, which would let a consumer corrupt snapshot summaries.
        object.__setattr__(self, "home", self._immutable_grid(self.home))
        object.__setattr__(self, "trail", self._immutable_grid(self.trail))
        object.__setattr__(self, "home_mass", float(self.home.sum()))
        object.__setattr__(self, "trail_mass", float(self.trail.sum()))
        object.__setattr__(
            self,
            "home_entropy",
            self._normalized_entropy(self.home),
        )
        object.__setattr__(
            self,
            "trail_entropy",
            self._normalized_entropy(self.trail),
        )

    @staticmethod
    def _immutable_grid(grid: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(grid, dtype=float)
        return np.frombuffer(
            contiguous.tobytes(),
            dtype=float,
        ).reshape(contiguous.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.home.shape  # type: ignore[return-value]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int((x - self.origin_x) / self.cell_size)
        cy = int((y - self.origin_y) / self.cell_size)
        rows, cols = self.home.shape
        return max(0, min(rows - 1, cy)), max(0, min(cols - 1, cx))

    def contains(self, x: float, y: float) -> bool:
        rows, cols = self.home.shape
        return (
            self.origin_x <= x < self.origin_x + cols * self.cell_size
            and self.origin_y <= y < self.origin_y + rows * self.cell_size
        )

    def sample(self, x: float, y: float) -> tuple[float, float]:
        """Bilinearly interpolate both channels so the gradient is smooth.

        Nearest-cell sampling makes both antennae fall in one cell (zero
        gradient); bilinear interpolation gives a continuous field an ant can
        actually climb.
        """

        if not self.contains(x, y):
            return 0.0, 0.0
        return (self._bilinear(self.home, x, y), self._bilinear(self.trail, x, y))

    @staticmethod
    def _normalized_entropy(grid: np.ndarray) -> float | None:
        mass = float(grid.sum())
        if mass <= 0.0:
            return None
        probabilities = grid.ravel() / mass
        positive = probabilities[probabilities > 0.0]
        if grid.size <= 1:
            return 0.0
        return -float(np.sum(positive * np.log(positive))) / math.log(grid.size)

    def _bilinear(self, grid: np.ndarray, x: float, y: float) -> float:
        rows, cols = grid.shape
        fx = (x - self.origin_x) / self.cell_size - 0.5
        fy = (y - self.origin_y) / self.cell_size - 0.5
        x0 = int(math.floor(fx))
        y0 = int(math.floor(fy))
        tx = fx - x0
        ty = fy - y0

        def _at(cy: int, cx: int) -> float:
            cy = max(0, min(rows - 1, cy))
            cx = max(0, min(cols - 1, cx))
            return float(grid[cy, cx])

        top = _at(y0, x0) * (1.0 - tx) + _at(y0, x0 + 1) * tx
        bottom = _at(y0 + 1, x0) * (1.0 - tx) + _at(y0 + 1, x0 + 1) * tx
        return top * (1.0 - ty) + bottom * ty


@dataclass(frozen=True)
class _DepositEvent:
    x: float
    y: float
    home_amount: float
    trail_amount: float


class PheromoneBus:
    """Owns the pheromone snapshot; aggregates additive deposits + decay."""

    def __init__(
        self,
        *,
        width: float = 24.0,
        height: float = 24.0,
        cell_size: float = 1.0,
        decay: float = 0.05,
        deposit_amount: float = 1.0,
        origin_x: float | None = None,
        origin_y: float | None = None,
    ) -> None:
        self._cell_size = cell_size
        self._decay = decay
        self._deposit_amount = deposit_amount
        self._origin_x = origin_x if origin_x is not None else -width / 2.0
        self._origin_y = origin_y if origin_y is not None else -height / 2.0
        rows = int(math.ceil(height / cell_size))
        cols = int(math.ceil(width / cell_size))
        self._home = np.zeros((rows, cols), dtype=float)
        self._trail = np.zeros((rows, cols), dtype=float)
        self._tick = 0
        self._pending: list[_DepositEvent] = []
        self._snapshot = self._publish_snapshot()

    @property
    def snapshot(self) -> PheromoneField:
        return self._snapshot

    def _publish_snapshot(self) -> PheromoneField:
        return PheromoneField(
            home=self._home.copy(),
            trail=self._trail.copy(),
            cell_size=self._cell_size,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
            tick=self._tick,
        )

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int((x - self._origin_x) / self._cell_size)
        cy = int((y - self._origin_y) / self._cell_size)
        rows, cols = self._home.shape
        return max(0, min(rows - 1, cy)), max(0, min(cols - 1, cx))

    def deposit(self, *, x: float, y: float, home_amount: float = 0.0, trail_amount: float = 0.0) -> None:
        """Append an independent, additive deposit event (does not overwrite)."""

        if not self._snapshot.contains(x, y):
            return
        self._pending.append(
            _DepositEvent(
                x=x,
                y=y,
                home_amount=home_amount * self._deposit_amount,
                trail_amount=trail_amount * self._deposit_amount,
            )
        )

    def advance(self) -> PheromoneField:
        """Apply decay, fold in all pending deposits, publish the next snapshot."""

        self._home *= 1.0 - self._decay
        self._trail *= 1.0 - self._decay
        for event in self._pending:
            row, col = self._cell(event.x, event.y)
            self._home[row, col] += event.home_amount
            self._trail[row, col] += event.trail_amount
        self._pending.clear()
        self._tick += 1
        self._snapshot = self._publish_snapshot()
        return self._snapshot

    def total_mass(self) -> tuple[float, float]:
        return self._snapshot.home_mass, self._snapshot.trail_mass
