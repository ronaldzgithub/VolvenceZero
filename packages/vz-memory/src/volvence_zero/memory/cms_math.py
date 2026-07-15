"""Pure-Python low-level math helpers for CMS bands.

Kept separate so ``CMSBandMLP`` and ``CMSMemoryCore`` share the same
numerically-cheap primitives without dragging in ``numpy``. Private
to the memory wheel; see ``cms_band_mlp`` / ``cms`` for consumers.
"""

from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _matvec(mat: list[float], vec: list[float] | tuple[float, ...], rows: int, cols: int) -> list[float]:
    if not isinstance(rows, int) or rows < 0:
        raise ValueError(f"_matvec rows must be a non-negative int, got {rows!r}")
    if not isinstance(cols, int) or cols < 0:
        raise ValueError(f"_matvec cols must be a non-negative int, got {cols!r}")
    if len(vec) != cols:
        raise ValueError(f"_matvec vector length {len(vec)} != cols {cols}")
    expected = rows * cols
    if len(mat) != expected:
        raise ValueError(f"_matvec matrix length {len(mat)} != rows*cols {expected}")

    result: list[float] = []
    for row_index in range(rows):
        offset = row_index * cols
        total = 0.0
        for col_index in range(cols):
            total += float(mat[offset + col_index]) * float(vec[col_index])
        result.append(total)
    return result


def _init_weight(size: int, scale: float = 0.01) -> list[float]:
    return [
        scale * (((i * 2654435761 + 17) % 65537) / 32768.5 - 1.0)
        for i in range(size)
    ]
