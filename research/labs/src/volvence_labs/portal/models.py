"""Immutable public snapshot types for the local Research Lab portal."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any


class LifecycleStage(str, Enum):
    NEEDS_TASK_DESIGN = "NEEDS_TASK_DESIGN"
    AWAITING_A0 = "AWAITING_A0"
    PREFLIGHT = "PREFLIGHT"
    RESEARCH_RUNNING = "RESEARCH_RUNNING"
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"
    CANDIDATE_RETAINED = "CANDIDATE_RETAINED"
    FORMAL_VALIDATION = "FORMAL_VALIDATION"
    AWAITING_A1 = "AWAITING_A1"
    SHADOW = "SHADOW"
    AWAITING_A2 = "AWAITING_A2"
    ACTIVE = "ACTIVE"
    ROLLED_BACK = "ROLLED_BACK"
    BLOCKED = "BLOCKED"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class WarningSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PortalWarning:
    code: str
    message: str
    source: str
    severity: WarningSeverity = WarningSeverity.WARNING
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    kind: str
    locator: str
    sha256: str
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class NamedCount:
    name: str
    count: int


@dataclass(frozen=True, slots=True)
class SourceHealth:
    source: str
    status: HealthStatus
    artifacts_seen: int
    detail: str


@dataclass(frozen=True, slots=True)
class PraxistRunSnapshot:
    run_id: str
    state: str
    source: str
    pid: int | None
    task_path: str
    run_dir: str | None
    generation: int | None
    findings_total: int
    peers_total: int
    peer_health: tuple[NamedCount, ...]
    runtime: str | None
    model_provider: str | None
    model: str | None
    started_at: str | None
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    development: str
    formal: str
    shadow: str
    canary: str


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    a0_research_start_authorized: bool
    formal_validation_status: str
    modification_gate_decision: str
    authorized_wiring: str
    runtime_wiring: str
    target_adapter_apply_required: bool
    production_default_changed: bool
    evaluation_is_learning_source: bool


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    stage: LifecycleStage
    next_stage: LifecycleStage | None
    blocking_reason: str | None
    last_transition_at: str | None


@dataclass(frozen=True, slots=True)
class ResearchLabItem:
    item_id: str
    task_id: str
    research_mode: str
    claim_id: str
    title: str
    objective: str
    owner: str
    capability_axes: tuple[str, ...]
    release_target: str
    lifecycle: LifecycleSnapshot
    authority: AuthoritySnapshot
    evidence: EvidenceSnapshot
    bindings: tuple[ArtifactRef, ...]
    run: PraxistRunSnapshot | None
    available_actions: tuple[str, ...]
    warnings: tuple[PortalWarning, ...]
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class ResearchTopicSource:
    locator: str
    sha256: str
    claim: str


@dataclass(frozen=True, slots=True)
class ResearchTopicProposalSnapshot:
    proposal_id: str
    title: str
    hypothesis: str
    mechanism: str
    demand_relevance: str
    research_question: str
    suggested_method: str
    success_signals: tuple[str, ...]
    falsification_signals: tuple[str, ...]
    caveats: tuple[str, ...]
    source_refs: tuple[ResearchTopicSource, ...]
    effective_state: str
    mapping_id: str | None
    binding_decision: str | None
    reviewed_by: str | None
    request_id: str | None
    request_state: str | None
    artifact: ArtifactRef
    binding: ArtifactRef | None
    request: ArtifactRef | None
    available_actions: tuple[str, ...]
    created_at: str | None


@dataclass(frozen=True, slots=True)
class ResearchDemandSnapshot:
    demand_id: str
    claim_id: str
    title: str
    objective: str
    owner: str
    capability_axes: tuple[str, ...]
    status: str
    requested_mapping_id: str | None
    source_roots: tuple[str, ...]
    max_topics: int
    artifact: ArtifactRef
    latest_run: ArtifactRef | None
    run_backend: str | None
    run_model: str | None
    proposals: tuple[ResearchTopicProposalSnapshot, ...]
    created_at: str | None


@dataclass(frozen=True, slots=True)
class ResearchDiscoverySnapshot:
    registry: ArtifactRef | None
    demand_count: int
    open_demand_count: int
    proposal_count: int
    awaiting_binding_count: int
    awaiting_a0_count: int
    demands: tuple[ResearchDemandSnapshot, ...]

    def get_proposal(
        self,
        proposal_id: str,
    ) -> tuple[ResearchDemandSnapshot, ResearchTopicProposalSnapshot] | None:
        for demand in self.demands:
            for proposal in demand.proposals:
                if proposal.proposal_id == proposal_id:
                    return demand, proposal
        return None


@dataclass(frozen=True, slots=True)
class ResearchLabSummary:
    registered_tasks: int
    stage_counts: tuple[NamedCount, ...]
    active_runs: int
    blocked: int
    awaiting_human: int
    production_active: int


@dataclass(frozen=True, slots=True)
class ResearchLabSnapshot:
    schema_version: str
    generated_at: str
    revision: str
    repo_revision: str
    summary: ResearchLabSummary
    source_health: tuple[SourceHealth, ...]
    discovery: ResearchDiscoverySnapshot
    items: tuple[ResearchLabItem, ...]
    warnings: tuple[PortalWarning, ...]

    def to_jsonable(self) -> dict[str, Any]:
        value = _jsonable(self)
        if not isinstance(value, dict):  # pragma: no cover - defensive invariant
            raise TypeError("ResearchLabSnapshot must serialize to an object")
        return value

    def get_task(self, task_id: str) -> ResearchLabItem | None:
        return next((item for item in self.items if item.task_id == task_id), None)


def _jsonable(value: Any) -> Any:
    """Serialize portal-owned immutable values without exposing mutable internals."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
