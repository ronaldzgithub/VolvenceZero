from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from volvence_zero.conditioning_bank_contracts import ConditioningLineageRef


_ACTION_FAMILY_BY_EXACT_ID: dict[str, str] = {
    "task_controller": "task",
    "repair_controller": "repair",
    "stabilize_controller": "stabilize",
    "exploration_controller": "exploration",
}


def abstract_action_family_id(abstract_action_id: str | None) -> str | None:
    """Return the typed family encoded by an abstract action id.

    The parser accepts exact controller ids and structured ids with a
    separator suffix (``task_controller:v2``). It intentionally does not
    use substring matching.
    """

    if abstract_action_id is None:
        return None
    action_id = abstract_action_id.strip().lower()
    if not action_id:
        return None
    if action_id.startswith("latent-family-v"):
        return "latent"
    head = action_id.split(":", 1)[0].split("/", 1)[0]
    return _ACTION_FAMILY_BY_EXACT_ID.get(head)


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


class TemporalActionAdvisoryStatus(str, Enum):
    NONE = "none"
    SHADOW_RECORDED = "shadow_recorded"
    APPLIED = "applied"


@dataclass(frozen=True)
class TemporalActionAdvisoryProposal:
    """Typed collaborator proposal consumed only by ``self_temporal``.

    The proposal carries action identity and lineage, never expression text.
    ``active_authorized`` is false for development and closed-alpha artifacts;
    the temporal owner refuses to apply such a proposal under ACTIVE wiring.
    """

    advisory_id: str
    decision_id: str
    prediction_id: str
    action_id: str
    confidence: float
    policy_artifact_id: str
    policy_artifact_version: int
    evidence_refs: tuple[str, ...]
    rationale_codes: tuple[str, ...]
    evaluator_only: bool = False
    active_authorized: bool = False

    def __post_init__(self) -> None:
        for field_name, value in (
            ("advisory_id", self.advisory_id),
            ("decision_id", self.decision_id),
            ("prediction_id", self.prediction_id),
            ("action_id", self.action_id),
            ("policy_artifact_id", self.policy_artifact_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        if self.policy_artifact_version < 1:
            raise ValueError("policy_artifact_version must be >= 1")
        for field_name, values in (
            ("evidence_refs", self.evidence_refs),
            ("rationale_codes", self.rationale_codes),
        ):
            if not values or any(not item.strip() for item in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique strings")
        if self.evaluator_only and self.active_authorized:
            raise ValueError("evaluator-only advisory cannot be ACTIVE-authorized")


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
    conditioning_lineage_refs: tuple[ConditioningLineageRef, ...] = ()
    action_advisory: TemporalActionAdvisoryProposal | None = None
    action_advisory_status: TemporalActionAdvisoryStatus = (
        TemporalActionAdvisoryStatus.NONE
    )

    def __post_init__(self) -> None:
        if self.action_advisory is None:
            if self.action_advisory_status is not TemporalActionAdvisoryStatus.NONE:
                raise ValueError("action_advisory_status requires an advisory")
        else:
            if self.action_advisory_status is TemporalActionAdvisoryStatus.NONE:
                raise ValueError("action advisory requires a non-NONE status")
            if self.action_advisory_status is TemporalActionAdvisoryStatus.APPLIED:
                if self.active_abstract_action != self.action_advisory.action_id:
                    raise ValueError("applied action advisory must own active action")
                if not self.action_advisory.active_authorized:
                    raise ValueError("applied action advisory is not ACTIVE-authorized")
                if self.action_advisory.evaluator_only:
                    raise ValueError("evaluator-only advisory cannot be applied")
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
