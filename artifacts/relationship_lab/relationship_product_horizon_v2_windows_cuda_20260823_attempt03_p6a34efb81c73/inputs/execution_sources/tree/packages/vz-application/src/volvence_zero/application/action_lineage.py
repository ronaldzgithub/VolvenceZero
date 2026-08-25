"""Typed Internal-RL lineage admitted by the CaseMemory owner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionLearningLineage:
    """Outcome-bound proof that a lived action entered Internal RL."""

    environment_outcome_id: str
    prediction_id: str
    world_capture_id: str
    self_capture_id: str
    credit_record_ids: tuple[str, ...]
    transition_count: int
    optimizer_consumed: bool
    policy_update_applied: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("environment_outcome_id", self.environment_outcome_id),
            ("prediction_id", self.prediction_id),
            ("world_capture_id", self.world_capture_id),
            ("self_capture_id", self.self_capture_id),
        ):
            if not value.strip():
                raise ValueError(
                    f"ActionLearningLineage {name} must be non-empty."
                )
        if not self.credit_record_ids or any(
            not record_id.strip() for record_id in self.credit_record_ids
        ):
            raise ValueError(
                "ActionLearningLineage requires non-empty credit_record_ids."
            )
        if self.transition_count < 2:
            raise ValueError(
                "ActionLearningLineage transition_count must cover world and self."
            )
        if not isinstance(self.optimizer_consumed, bool) or not isinstance(
            self.policy_update_applied, bool
        ):
            raise TypeError(
                "ActionLearningLineage consumption/update flags must be bool."
            )

    @property
    def admission_ready(self) -> bool:
        return self.optimizer_consumed and self.policy_update_applied
