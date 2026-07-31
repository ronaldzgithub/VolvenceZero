"""Immutable relationship continuity evaluation exchange contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RelationshipContinuityConsoleOutcome:
    outcome_id: str
    observed_at_ms: int
    is_correction: bool
    is_wrong_user_attribution: bool
    is_boundary_violation: bool
    remembered_item_useful: bool | None

    def __post_init__(self) -> None:
        if not self.outcome_id.strip():
            raise ValueError("outcome_id must be non-empty")
        if self.observed_at_ms < 0:
            raise ValueError("observed_at_ms must be non-negative")
        if self.is_wrong_user_attribution and not self.is_correction:
            raise ValueError("wrong-user attribution must be a correction")


@dataclass(frozen=True)
class RelationshipContinuitySnapshot:
    schema_version: int
    user_scope_hash: str
    window_start_ms: int
    window_end_ms: int
    callback_hit_rate: float | None
    boundary_violation_rate: float | None
    wrong_user_attribution_rate: float | None
    open_loop_closure_rate: float | None
    user_correction_rate: float | None
    remembered_item_usefulness: float | None
    seven_day_trust_delta: float | None
    sample_sizes: tuple[tuple[str, int], ...]
    wiring_level: str
    description: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_scope_hash": self.user_scope_hash,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "callback_hit_rate": self.callback_hit_rate,
            "boundary_violation_rate": self.boundary_violation_rate,
            "wrong_user_attribution_rate": self.wrong_user_attribution_rate,
            "open_loop_closure_rate": self.open_loop_closure_rate,
            "user_correction_rate": self.user_correction_rate,
            "remembered_item_usefulness": self.remembered_item_usefulness,
            "seven_day_trust_delta": self.seven_day_trust_delta,
            "sample_sizes": dict(self.sample_sizes),
            "wiring_level": self.wiring_level,
            "description": self.description,
        }


__all__ = [
    "RelationshipContinuityConsoleOutcome",
    "RelationshipContinuitySnapshot",
]
