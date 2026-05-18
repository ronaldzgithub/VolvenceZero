"""Request / response shapes for the lifeform HTTP service.

Every wire-format object is a frozen dataclass with explicit JSON
serialisation, so the service surface is auditable and the kernel never
sees raw HTTP payloads. The discipline mirrors the kernel's snapshot
contract: external code can only enter the lifeform via these typed
shapes, not by reaching into ``Lifeform`` internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CreateSessionRequest:
    session_id: str | None = None
    vertical: str | None = None  # reserved for future multi-vertical service


@dataclass(frozen=True)
class CreateSessionResponse:
    session_id: str
    vertical: str
    has_temporal_bootstrap: bool
    has_regime_bootstrap: bool
    user_id: str | None = None
    service_version: str = ""
    policy_version: str = ""
    alpha_disclaimer: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "vertical": self.vertical,
            "has_temporal_bootstrap": self.has_temporal_bootstrap,
            "has_regime_bootstrap": self.has_regime_bootstrap,
            "user_id": self.user_id,
            "service_version": self.service_version,
            "policy_version": self.policy_version,
            "alpha_disclaimer": self.alpha_disclaimer,
        }


@dataclass(frozen=True)
class TurnRequest:
    user_input: str


@dataclass(frozen=True)
class TurnResponse:
    """Compact projection of an ``AgentTurnResult`` over the wire.

    We deliberately do NOT serialise the full snapshot dict \u2014 the kernel's
    snapshot graph is large and tightly coupled to internal types. Product
    code that needs more should call ``GET /v1/sessions/{id}/state``.
    """

    session_id: str
    scene_id: str
    turn_index: int
    response_text: str
    active_regime: str | None
    active_abstract_action: str | None
    expression_intent: str | None
    pe_magnitude: float
    open_loop_count: int
    commitment_count: int
    response_rationale_tags: tuple[str, ...] = ()
    safety: dict[str, Any] = field(default_factory=dict)
    # Derived view of the prompt envelope that was assembled for this
    # turn's LLM call. SSOT note: ``system_prompt`` is rebuilt by the
    # handler via the pure ``build_system_prompt(assembly)`` function
    # using the same ``response_assembly`` snapshot the synthesizer
    # consumed, so this is a snapshot-derived projection — not a new
    # owner of prompt state. ``None`` when no LLM-backed synthesizer
    # ran (deterministic substrate, scope refusal, error path).
    llm_envelope: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scene_id": self.scene_id,
            "turn_index": self.turn_index,
            "response_text": self.response_text,
            "active_regime": self.active_regime,
            "active_abstract_action": self.active_abstract_action,
            "expression_intent": self.expression_intent,
            "pe_magnitude": self.pe_magnitude,
            "open_loop_count": self.open_loop_count,
            "commitment_count": self.commitment_count,
            "response_rationale_tags": list(self.response_rationale_tags),
            "safety": self.safety,
            "llm_envelope": self.llm_envelope,
        }


@dataclass(frozen=True)
class EndSceneResponse:
    session_id: str
    closed_scene_id: str | None
    slow_loop_drained: bool
    evidence_artifact_ref: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "closed_scene_id": self.closed_scene_id,
            "slow_loop_drained": self.slow_loop_drained,
            "evidence_artifact_ref": self.evidence_artifact_ref,
        }


@dataclass(frozen=True)
class SessionStateResponse:
    """A read-only summary of a session's current externally-visible state."""

    session_id: str
    open_scene_id: str | None
    open_scene_turn_count: int
    closed_scene_count: int
    turn_count: int
    pending_followup_count: int
    last_active_regime: str | None
    last_active_abstract_action: str | None
    last_response_text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "open_scene_id": self.open_scene_id,
            "open_scene_turn_count": self.open_scene_turn_count,
            "closed_scene_count": self.closed_scene_count,
            "turn_count": self.turn_count,
            "pending_followup_count": self.pending_followup_count,
            "last_active_regime": self.last_active_regime,
            "last_active_abstract_action": self.last_active_abstract_action,
            "last_response_text": self.last_response_text,
        }


@dataclass(frozen=True)
class HealthResponse:
    status: str
    session_count: int
    vertical: str

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_count": self.session_count,
            "vertical": self.vertical,
        }


@dataclass(frozen=True)
class ServiceInfoResponse:
    vertical: str
    has_temporal_bootstrap: bool
    has_regime_bootstrap: bool
    bootstraps_dir: str | None
    scenarios_dir: str | None
    substrate_shared: bool
    substrate_model_id: str | None
    substrate_runtime_origin: str | None
    alpha: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "vertical": self.vertical,
            "has_temporal_bootstrap": self.has_temporal_bootstrap,
            "has_regime_bootstrap": self.has_regime_bootstrap,
            "bootstraps_dir": self.bootstraps_dir,
            "scenarios_dir": self.scenarios_dir,
            "substrate_shared": self.substrate_shared,
            "substrate_model_id": self.substrate_model_id,
            "substrate_runtime_origin": self.substrate_runtime_origin,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class ErrorResponse:
    error: str
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"error": self.error, "detail": self.detail, **self.extra}
