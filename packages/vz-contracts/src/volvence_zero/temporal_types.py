from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControllerState:
    code: tuple[float, ...]
    code_dim: int
    switch_gate: float
    is_switching: bool
    steps_since_switch: int
    track_codes: tuple[tuple[str, tuple[float, ...]], ...] = ()


@dataclass(frozen=True)
class TemporalSegmentClosure:
    segment_id: str
    open_turn_index: int
    close_turn_index: int
    abstract_action_id: str
    z_t_digest: tuple[float, ...]
    beta_open_digest: float
    beta_close_digest: float
    affordance_name: str | None = None
    description: str = ""


@dataclass(frozen=True)
class TemporalAbstractionSnapshot:
    controller_state: ControllerState
    active_abstract_action: str
    controller_params_hash: str
    description: str
    action_family_version: int = 0
    switch_gate_stats: Any | None = None
    memory_feedback_signal: tuple[float, ...] = ()
    closed_segments: tuple[TemporalSegmentClosure, ...] = ()
    memory_retrieval_facets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.memory_retrieval_facets:
            return
        object.__setattr__(
            self,
            "memory_retrieval_facets",
            (
                f"temporal:{self.active_abstract_action}",
                f"temporal:steps_since_switch:{self.controller_state.steps_since_switch}",
            ),
        )
