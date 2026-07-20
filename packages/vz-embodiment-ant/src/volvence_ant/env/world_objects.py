"""Immutable ecology objects owned by :mod:`volvence_ant.env`.

The browser and controllers never own these values.  ``AntWorld`` applies
typed placements at tick/round boundaries and publishes immutable snapshots
for rendering.  Controllers receive only local scalar samples through
``WorldObservation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class WorldObjectKind(str, Enum):
    BUTTER = "butter"
    WOOD_STICK = "wood_stick"
    BURNING_MATCH = "burning_match"


@dataclass(frozen=True)
class WorldObjectSnapshot:
    """Environment-authored public description of one placed object."""

    object_id: str
    kind: WorldObjectKind
    center: tuple[float, float]
    segment_start: tuple[float, float] | None
    segment_end: tuple[float, float] | None
    physical_radius: float
    effect_radius: float
    remaining: float | None
    active: bool
    description: str


def _require_finite(name: str, *values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite")


def _point_segment_distance(
    x: float,
    y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    dx = end_x - start_x
    dy = end_y - start_y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return math.hypot(x - start_x, y - start_y)
    fraction = ((x - start_x) * dx + (y - start_y) * dy) / length_sq
    fraction = max(0.0, min(1.0, fraction))
    projection_x = start_x + fraction * dx
    projection_y = start_y + fraction * dy
    return math.hypot(x - projection_x, y - projection_y)


def _orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(
    a_start: tuple[float, float],
    a_end: tuple[float, float],
    b_start: tuple[float, float],
    b_end: tuple[float, float],
) -> bool:
    ax, ay = a_start
    bx, by = a_end
    cx, cy = b_start
    dx, dy = b_end
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)
    epsilon = 1e-12
    if (o1 > epsilon and o2 < -epsilon or o1 < -epsilon and o2 > epsilon) and (
        o3 > epsilon and o4 < -epsilon or o3 < -epsilon and o4 > epsilon
    ):
        return True
    return (
        (
            abs(o1) <= epsilon
            and min(ax, bx) - epsilon <= cx <= max(ax, bx) + epsilon
            and min(ay, by) - epsilon <= cy <= max(ay, by) + epsilon
        )
        or (
            abs(o2) <= epsilon
            and min(ax, bx) - epsilon <= dx <= max(ax, bx) + epsilon
            and min(ay, by) - epsilon <= dy <= max(ay, by) + epsilon
        )
        or (
            abs(o3) <= epsilon
            and min(cx, dx) - epsilon <= ax <= max(cx, dx) + epsilon
            and min(cy, dy) - epsilon <= ay <= max(cy, dy) + epsilon
        )
        or (
            abs(o4) <= epsilon
            and min(cx, dx) - epsilon <= bx <= max(cx, dx) + epsilon
            and min(cy, dy) - epsilon <= by <= max(cy, dy) + epsilon
        )
    )


def _segment_distance(
    a_start: tuple[float, float],
    a_end: tuple[float, float],
    b_start: tuple[float, float],
    b_end: tuple[float, float],
) -> float:
    if _segments_intersect(a_start, a_end, b_start, b_end):
        return 0.0
    return min(
        _point_segment_distance(*a_start, *b_start, *b_end),
        _point_segment_distance(*a_end, *b_start, *b_end),
        _point_segment_distance(*b_start, *a_start, *a_end),
        _point_segment_distance(*b_end, *a_start, *a_end),
    )


@dataclass(frozen=True)
class ButterSource:
    """A finite or inexhaustible fat-rich food/odour source."""

    object_id: str
    x: float
    y: float
    strength: float = 1.6
    decay: float = 4.0
    radius: float = 1.2
    remaining: float = float("inf")

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("butter object_id must be non-empty")
        _require_finite("butter position", self.x, self.y)
        _require_finite("butter field", self.strength, self.decay, self.radius)
        if self.strength <= 0.0 or self.decay <= 0.0 or self.radius <= 0.0:
            raise ValueError("butter strength, decay and radius must be positive")
        if self.remaining != float("inf") and (not math.isfinite(self.remaining) or self.remaining < 0.0):
            raise ValueError("butter remaining must be finite and non-negative or inf")

    def intensity(self, x: float, y: float) -> float:
        if self.remaining <= 0.0:
            return 0.0
        return self.strength * math.exp(-math.hypot(x - self.x, y - self.y) / self.decay)

    def snapshot(self) -> WorldObjectSnapshot:
        visible_radius = self.decay * math.log(max(self.strength / 0.05, 1.0))
        return WorldObjectSnapshot(
            object_id=self.object_id,
            kind=WorldObjectKind.BUTTER,
            center=(self.x, self.y),
            segment_start=None,
            segment_end=None,
            physical_radius=self.radius,
            effect_radius=visible_radius,
            remaining=None if self.remaining == float("inf") else self.remaining,
            active=self.remaining > 0.0,
            description="fat-rich food source with a local odour field",
        )


@dataclass(frozen=True)
class AxisAlignedObstacle:
    """Immutable environment-owned rectangular obstacle."""

    obstacle_id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        bounds = (self.min_x, self.max_x, self.min_y, self.max_y)
        if not self.obstacle_id:
            raise ValueError("obstacle_id must be non-empty")
        _require_finite("obstacle bounds", *bounds)
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("obstacle bounds must satisfy min_x < max_x and min_y < max_y")

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def strictly_contains(self, x: float, y: float) -> bool:
        return self.min_x < x < self.max_x and self.min_y < y < self.max_y

    def penetration_depth(self, x: float, y: float) -> float:
        if not self.strictly_contains(x, y):
            return 0.0
        return min(
            x - self.min_x,
            self.max_x - x,
            y - self.min_y,
            self.max_y - y,
        )

    def entry_fraction(
        self,
        *,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
    ) -> float | None:
        entry = 0.0
        exit_ = 1.0
        for origin, delta, lower, upper in (
            (start_x, target_x - start_x, self.min_x, self.max_x),
            (start_y, target_y - start_y, self.min_y, self.max_y),
        ):
            if abs(delta) <= 1e-15:
                if origin < lower or origin > upper:
                    return None
                continue
            first = (lower - origin) / delta
            second = (upper - origin) / delta
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_ = min(exit_, second)
            if entry > exit_:
                return None
        if exit_ < 0.0 or entry > 1.0:
            return None
        return max(0.0, entry)


@dataclass(frozen=True)
class WoodStick:
    """A directional capsule obstacle defined by its centre line."""

    object_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float = 0.22

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("wood stick object_id must be non-empty")
        _require_finite(
            "wood stick geometry",
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.radius,
        )
        if self.radius <= 0.0:
            raise ValueError("wood stick radius must be positive")
        if math.hypot(self.end_x - self.start_x, self.end_y - self.start_y) <= 1e-6:
            raise ValueError("wood stick must have non-zero length")

    @property
    def obstacle_id(self) -> str:
        return self.object_id

    def distance(self, x: float, y: float) -> float:
        return _point_segment_distance(
            x,
            y,
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
        )

    def contains(self, x: float, y: float) -> bool:
        return self.distance(x, y) <= self.radius

    def strictly_contains(self, x: float, y: float) -> bool:
        return self.distance(x, y) < self.radius

    def penetration_depth(self, x: float, y: float) -> float:
        return max(0.0, self.radius - self.distance(x, y))

    def entry_fraction(
        self,
        *,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
    ) -> float | None:
        movement_start = (start_x, start_y)
        movement_end = (target_x, target_y)
        stick_start = (self.start_x, self.start_y)
        stick_end = (self.end_x, self.end_y)
        if _segment_distance(movement_start, movement_end, stick_start, stick_end) > self.radius:
            return None
        if self.contains(start_x, start_y):
            return 0.0
        low = 0.0
        high = 1.0
        for _ in range(60):
            midpoint = (low + high) / 2.0
            probe = (
                start_x + (target_x - start_x) * midpoint,
                start_y + (target_y - start_y) * midpoint,
            )
            if _segment_distance(movement_start, probe, stick_start, stick_end) <= self.radius:
                high = midpoint
            else:
                low = midpoint
        return high

    def snapshot(self) -> WorldObjectSnapshot:
        return WorldObjectSnapshot(
            object_id=self.object_id,
            kind=WorldObjectKind.WOOD_STICK,
            center=(
                (self.start_x + self.end_x) / 2.0,
                (self.start_y + self.end_y) / 2.0,
            ),
            segment_start=(self.start_x, self.start_y),
            segment_end=(self.end_x, self.end_y),
            physical_radius=self.radius,
            effect_radius=self.radius,
            remaining=None,
            active=True,
            description="directional, impassable 2D wood-stick obstacle",
        )


@dataclass(frozen=True)
class BurningMatch:
    """A persistent local heat source; normal avoidance remains learned."""

    object_id: str
    x: float
    y: float
    angle: float = 0.0
    length: float = 1.8
    heat_strength: float = 1.0
    heat_decay: float = 1.8
    harm_threshold: float = 0.55

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("burning match object_id must be non-empty")
        _require_finite(
            "burning match",
            self.x,
            self.y,
            self.angle,
            self.length,
            self.heat_strength,
            self.heat_decay,
            self.harm_threshold,
        )
        if self.length <= 0.0 or self.heat_strength <= 0.0 or self.heat_decay <= 0.0:
            raise ValueError("burning match length, heat_strength and heat_decay must be positive")
        if not 0.0 < self.harm_threshold <= 1.0:
            raise ValueError("burning match harm_threshold must be in (0, 1]")

    def intensity(self, x: float, y: float) -> float:
        return self.heat_strength * math.exp(-math.hypot(x - self.x, y - self.y) / self.heat_decay)

    def is_harmful(self, x: float, y: float) -> bool:
        return self.intensity(x, y) >= self.harm_threshold

    @property
    def harm_radius(self) -> float:
        return max(
            0.0,
            self.heat_decay * math.log(max(self.heat_strength / self.harm_threshold, 1.0)),
        )

    def snapshot(self) -> WorldObjectSnapshot:
        body_end = (
            self.x - math.cos(self.angle) * self.length,
            self.y - math.sin(self.angle) * self.length,
        )
        return WorldObjectSnapshot(
            object_id=self.object_id,
            kind=WorldObjectKind.BURNING_MATCH,
            center=(self.x, self.y),
            segment_start=body_end,
            segment_end=(self.x, self.y),
            physical_radius=0.12,
            effect_radius=self.harm_radius,
            remaining=None,
            active=True,
            description="burning match with a local directional-sensor heat field",
        )


WorldObject: TypeAlias = ButterSource | WoodStick | BurningMatch
WorldObstacle: TypeAlias = AxisAlignedObstacle | WoodStick


__all__ = [
    "AxisAlignedObstacle",
    "BurningMatch",
    "ButterSource",
    "WoodStick",
    "WorldObject",
    "WorldObjectKind",
    "WorldObjectSnapshot",
    "WorldObstacle",
]
