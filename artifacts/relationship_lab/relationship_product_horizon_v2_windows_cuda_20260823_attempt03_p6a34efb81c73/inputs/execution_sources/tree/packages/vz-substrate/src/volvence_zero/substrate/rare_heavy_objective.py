"""Shared structural objective for synthetic rare-heavy transfer evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math


RARE_HEAVY_STRUCTURAL_OBJECTIVE_VERSION = "content-position-v1"


@dataclass(frozen=True)
class RareHeavyStructuralObjective:
    """Versioned content/position residual target shared by train and eval."""

    amplitude: float = 0.018
    position_weight: float = 0.65
    content_weight: float = 0.35
    version: str = RARE_HEAVY_STRUCTURAL_OBJECTIVE_VERSION

    def __post_init__(self) -> None:
        values = (
            self.amplitude,
            self.position_weight,
            self.content_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rare-heavy structural objective must be finite")
        if not 0.0 < self.amplitude <= 0.18:
            raise ValueError(
                "rare-heavy structural amplitude must be in (0, 0.18]"
            )
        if self.position_weight < 0.0 or self.content_weight < 0.0:
            raise ValueError(
                "rare-heavy structural weights must be non-negative"
            )
        if self.position_weight + self.content_weight <= 0.0:
            raise ValueError(
                "rare-heavy structural objective needs a non-zero weight"
            )
        if not self.version:
            raise ValueError(
                "rare-heavy structural objective version must be non-empty"
            )

    def residual_delta(
        self,
        *,
        source_text: str,
        layer_index: int,
        width: int,
    ) -> tuple[float, ...]:
        if not source_text.strip():
            raise ValueError(
                "rare-heavy structural objective needs non-empty source text"
            )
        if layer_index < 0:
            raise ValueError(
                "rare-heavy structural layer_index must be non-negative"
            )
        if width < 1:
            raise ValueError(
                "rare-heavy structural residual width must be positive"
            )
        tokens = tuple(source_text.casefold().split())
        raw = tuple(
            self.position_weight
            * math.sin((index + 1) * (layer_index + 1) * 0.73)
            + self.content_weight
            * sum(
                _token_projection(
                    token=token,
                    layer_index=layer_index,
                    coordinate=index,
                )
                for token in tokens
            )
            / len(tokens)
            for index in range(width)
        )
        mean_abs = sum(abs(value) for value in raw) / width
        if mean_abs <= 1e-12:
            raise RuntimeError(
                "rare-heavy structural objective collapsed to zero"
            )
        return tuple(
            self.amplitude * value / mean_abs
            for value in raw
        )


def _token_projection(
    *,
    token: str,
    layer_index: int,
    coordinate: int,
) -> float:
    digest = hashlib.blake2b(
        f"{token}:{layer_index}:{coordinate}".encode("utf-8"),
        digest_size=2,
    ).digest()
    unit = int.from_bytes(digest, "big") / 65535.0
    return 2.0 * unit - 1.0


__all__ = [
    "RARE_HEAVY_STRUCTURAL_OBJECTIVE_VERSION",
    "RareHeavyStructuralObjective",
]
