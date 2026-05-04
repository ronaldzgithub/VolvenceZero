"""Pure-Python low-level math helpers for CMS bands.

Kept separate so ``CMSBandMLP`` and ``CMSMemoryCore`` share the same
numerically-cheap primitives without dragging in ``numpy``. Private
to the memory wheel; see ``cms_band_mlp`` / ``cms`` for consumers.
"""

from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _matvec(mat: list[float], vec: list[float] | tuple[float, ...], rows: int, cols: int) -> list[float]:
    return [
        sum(mat[i * cols + j] * vec[j] for j in range(cols))
        for i in range(rows)
    ]


def _init_weight(size: int, scale: float = 0.01) -> list[float]:
    return [
        scale * (((i * 2654435761 + 17) % 65537) / 32768.5 - 1.0)
        for i in range(size)
    ]
