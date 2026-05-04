"""Semantic state contract surface (R8 / R11).

Pure data-only module. Holds enums, dataclasses, event records, and
snapshot types for the nine semantic-state owners. Has no runtime
logic beyond tiny ``_clamp`` / ``semantic_control_signal`` /
``semantic_snapshot_description`` readout helpers.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``. External consumers continue to import
from ``volvence_zero.semantic_state`` and get these names via the
package ``__init__`` re-export.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from typing import Any


SEMANTIC_OWNER_SLOTS: tuple[str, ...] = (
    "plan_intent",
    "commitment",
    "open_loop",
    "user_model",
    "execution_result",
    "belief_assumption",
    "relationship_state",
    "goal_value",
    "boundary_consent",
)


def load_semantic_prompt_template(name: str = "extraction.md") -> str:
    return files("volvence_zero.semantic_state").joinpath("prompts", name).read_text(encoding="utf-8")


def load_semantic_json_schema(name: str = "proposal.schema.json") -> str:
    return files("volvence_zero.semantic_state").joinpath("schemas", name).read_text(encoding="utf-8")


class SemanticProposalOperation(str, Enum):
    OBSERVE = "observe"
    CREATE = "create"
    REVISE = "revise"
    DEFER = "defer"
    ACTIVATE = "activate"
    COMPLETE = "complete"
    CLOSE = "close"
    BLOCK = "block"


class AdvocacyState(str, Enum):
    """Where the AI sits on the path of advocating a commitment to the user.

    Models the AAC decision lifecycle (Advocacy \u2192 Alignment \u2192 Commitment \u2192
    Followup) at the AI side. A commitment record always has a value;
    ``not_ready`` means the AI is aware of it but has not chosen to
    surface it yet (e.g. still gathering evidence).
    """

    NOT_READY = "not_ready"  # observed / created but the AI is not surfacing it
    READY = "ready"          # AI has decided to surface but has not yet (e.g. DEFER)
    PROPOSED = "proposed"    # the AI advocated the commitment in-conversation


class AlignmentState(str, Enum):
    """The user's response to an advocated commitment.

    Stays ``unknown`` until the AI advocates AND we observe a typed
    user signal via the SemanticProposal pathway (REVISE / COMPLETE /
    BLOCK). NEVER set from LLM keyword detection on free text \u2014 the
    propose-then-observe contract is the whole point of having a
    typed lifecycle here.
    """

    UNKNOWN = "unknown"
    AGREE = "agree"
    MODIFY = "modify"
    REJECT = "reject"


class FollowupPolicy(str, Enum):
    """Who owns the re-engagement cadence for this commitment.

    Mirrors EmoGPT PRD \u00a75.6 (AAC lifecycle) but stripped of the
    product-side vocabulary: only two states that actually move the
    followup scheduler. Defaults to ``gentle_checkin`` \u2014 proactively
    due after the default delay. ``defer_only`` means the commitment
    was explicitly held back (user rejected, asked to come back later,
    or AI decided it wasn't the right moment) and the follow-up should
    NOT be surfaced unless the user brings it up first.
    """

    GENTLE_CHECKIN = "gentle_checkin"
    DEFER_ONLY = "defer_only"


class CommitmentOutcomeKind(str, Enum):
    """Typed outcome of a commitment lifecycle transition.

    Used by reflection writeback to record WHAT happened to a commitment
    at end-of-scene so ETA / regime / case_memory can learn from it.
    Not used for live routing decisions \u2014 that is what advocacy /
    alignment / followup_policy are for.
    """

    PROGRESSED = "commitment_progressed"
    COMPLETED = "commitment_completed"
    STALLED = "commitment_stalled"
    REJECTED = "commitment_rejected"
    FOLLOWUP_NO_RESPONSE = "followup_no_response"


class PlanIntentOutcome(str, Enum):
    """Typed outcome of a plan / intent lifecycle transition (Gap 10).

    Complements the generic ``status`` field on ``SemanticRecord`` by
    naming the *kind* of signal the transition represents. Used by
    reflection writeback to tag runtime events with a stable enum
    label instead of free-form strings.
    """

    DECISION_MADE = "decision_made"
    ASSUMPTION_RECORDED = "assumption_recorded"
    PROBLEM_PROGRESS_ASSESSED = "problem_progress_assessed"
    OUTCOME_OBSERVED = "outcome_observed"


class ExecutionResultOutcome(str, Enum):
    """Typed outcome of an execution-result record (Gap 10).

    Mirrors the EmoGPT PRD \u00a77.2 structured-runtime-event taxonomy,
    narrowed to what the execution_result owner can actually emit:
    tool outcomes, user feedback, teacher instructions, crystal
    evaluations, and the learning-artifact publication / bootstrap
    events.
    """

    USER_FEEDBACK_RECEIVED = "user_feedback_received"
    INSTRUCTION_RECEIVED = "instruction_received"
    TOOL_OUTCOME = "tool_outcome"
    CRYSTAL_EVALUATION = "crystal_evaluation"
    CRYSTAL_SUPPRESSION = "crystal_suppression"
    PACKAGE_PUBLICATION = "package_publication"
    BOOTSTRAP_CONSUMPTION = "bootstrap_consumption"



@dataclass(frozen=True)
class SemanticProposal:
    proposal_id: str
    target_slot: str
    operation: SemanticProposalOperation
    summary: str
    detail: str
    confidence: float
    evidence: str
    control_signal: float = 0.0
    requires_confirmation: bool = False


@dataclass(frozen=True)
class SemanticProposalBatch:
    proposals: tuple[SemanticProposal, ...]
    runtime_id: str
    schema_version: int
    description: str


@dataclass(frozen=True)
class ToolResultSemanticEvent:
    event_id: str
    tool_name: str
    action_id: str
    status: str
    summary: str
    detail: str
    confidence: float = 0.8
    artifact_refs: tuple[str, ...] = ()
    plan_ref: str | None = None


@dataclass(frozen=True)
class ProfileSemanticEvent:
    event_id: str
    source: str
    preferences: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    consent_grants: tuple[str, ...] = ()
    consent_denials: tuple[str, ...] = ()
    relationship_note: str = ""
    confidence: float = 0.75


@dataclass(frozen=True)
class TaskSemanticEvent:
    event_id: str
    task_id: str
    status: str
    summary: str
    detail: str
    due_hint: str | None = None
    commitment_ref: str | None = None
    confidence: float = 0.75


@dataclass(frozen=True)
class ReviewedKnowledgeSemanticEvent:
    event_id: str
    knowledge_id: str
    summary: str
    detail: str
    source_label: str
    confidence: float
    relevance_hint: str = ""
    needs_followup: bool = False


ExternalSemanticEvent = (
    ToolResultSemanticEvent
    | ProfileSemanticEvent
    | TaskSemanticEvent
    | ReviewedKnowledgeSemanticEvent
)


@dataclass(frozen=True)
class ExternalSemanticEventBatch:
    events: tuple[ExternalSemanticEvent, ...]
    source: str
    description: str


@dataclass(frozen=True)
class SemanticRecord:
    record_id: str
    summary: str
    detail: str
    confidence: float
    status: str
    source_turn: int
    evidence: str
    control_signal: float = 0.0


@dataclass(frozen=True)
class PlanIntentLifecycleEntry:
    """Per-record outcome state for a plan / intent record (Gap 10).

    Published alongside ``candidate_plans`` / ``deferred_intents`` so
    downstream consumers (reflection writeback, evaluation, credit
    attribution) can key off a typed outcome enum rather than parsing
    the free-form ``status`` string. Defaults represent a freshly
    observed record with no outcome yet.
    """

    record_id: str
    last_outcome: PlanIntentOutcome | None = None
    last_outcome_evidence: str = ""
    last_outcome_at_turn: int = -1

    def __post_init__(self) -> None:
        has_outcome = self.last_outcome is not None
        if has_outcome and not self.last_outcome_evidence.strip():
            raise ValueError(
                "PlanIntentLifecycleEntry.last_outcome set without "
                "non-empty last_outcome_evidence; every typed outcome "
                "MUST carry evidence for reflection writeback audit."
            )
        if has_outcome and self.last_outcome_at_turn < 0:
            raise ValueError(
                "PlanIntentLifecycleEntry.last_outcome set without "
                "last_outcome_at_turn (>= 0); outcome records must be "
                "anchored to a turn."
            )


@dataclass(frozen=True)
class PlanIntentSnapshot:
    active_plan_id: str | None
    active_goal: str
    active_step: str
    active_constraints: tuple[str, ...]
    deferred_intents: tuple[SemanticRecord, ...]
    standing_plans: tuple[SemanticRecord, ...]
    candidate_plans: tuple[SemanticRecord, ...]
    completed_plan_refs: tuple[str, ...]
    plan_revision_count: int
    continuity_score: float
    control_signal: float
    description: str
    # Gap 10 additions. Defaults preserve backwards compat for any
    # synthetic snapshot constructed outside the owner module.
    lifecycle_entries: tuple[PlanIntentLifecycleEntry, ...] = ()
    outcome_decision_made_count: int = 0
    outcome_assumption_recorded_count: int = 0
    outcome_problem_progress_assessed_count: int = 0
    outcome_observed_count: int = 0

    def lifecycle_for(self, record_id: str) -> PlanIntentLifecycleEntry | None:
        for entry in self.lifecycle_entries:
            if entry.record_id == record_id:
                return entry
        return None


@dataclass(frozen=True)
class CommitmentLifecycleEntry:
    """Per-record advocacy / alignment lifecycle state.

    Published parallel to ``active_commitments`` / ``at_risk_commitments``
    so consumers (reflection writeback, evaluation, prompt planner) can
    answer the AAC question \u2014 "where is each commitment in the
    Advocacy \u2192 Alignment \u2192 Commitment \u2192 Followup pipeline?" \u2014 without
    pattern-matching on free text. Both states are always present;
    a freshly-observed commitment with no AI advocacy yet shows
    ``(NOT_READY, UNKNOWN)``.

    ``followup_policy`` tells the lifeform-side ``FollowupManager`` how
    to treat this commitment when scheduling re-engagement. ``last_outcome``
    is only ever set when a typed outcome transition actually fires; an
    empty ``last_outcome_evidence`` with a non-None ``last_outcome`` is a
    contract violation enforced by ``tests/contracts/test_aac_lifecycle.py``
    (outcome-requires-evidence invariant).
    """

    record_id: str
    advocacy_state: AdvocacyState
    alignment_state: AlignmentState
    followup_policy: FollowupPolicy = FollowupPolicy.GENTLE_CHECKIN
    last_outcome: CommitmentOutcomeKind | None = None
    last_outcome_evidence: str = ""
    last_outcome_at_turn: int = -1

    def __post_init__(self) -> None:
        # Outcome-requires-evidence: if a typed outcome is recorded it MUST
        # carry non-empty evidence. Reflection writeback / audit downstream
        # rely on this being non-None-with-trace or None-without-trace;
        # silent None-with-evidence or outcome-without-evidence lets drift
        # sneak in.
        has_outcome = self.last_outcome is not None
        has_evidence = bool(self.last_outcome_evidence.strip())
        if has_outcome and not has_evidence:
            raise ValueError(
                "CommitmentLifecycleEntry.last_outcome set without "
                "non-empty last_outcome_evidence; every typed outcome "
                "MUST carry evidence for reflection writeback audit."
            )
        if has_outcome and self.last_outcome_at_turn < 0:
            raise ValueError(
                "CommitmentLifecycleEntry.last_outcome set without "
                "last_outcome_at_turn (>= 0); outcome records must be "
                "anchored to a turn for credit attribution."
            )


@dataclass(frozen=True)
class CommitmentSnapshot:
    active_commitments: tuple[SemanticRecord, ...]
    honored_commitment_refs: tuple[str, ...]
    at_risk_commitments: tuple[SemanticRecord, ...]
    trust_obligation_count: int
    continuity_score: float
    control_signal: float
    description: str
    # AAC lifecycle additions (Gap 7). Defaults preserve backwards
    # compat for any synthetic CommitmentSnapshot built outside the
    # owner module (e.g. older tests).
    lifecycle_entries: tuple[CommitmentLifecycleEntry, ...] = ()
    advocacy_proposed_count: int = 0
    advocacy_ready_count: int = 0
    alignment_agree_count: int = 0
    alignment_modify_count: int = 0
    alignment_reject_count: int = 0
    # Follow-up / outcome aggregates \u2014 published alongside per-entry
    # lifecycle for cheap O(1) consumption by FollowupManager /
    # evaluation / family report.
    followup_gentle_count: int = 0
    followup_defer_only_count: int = 0
    outcome_progressed_count: int = 0
    outcome_completed_count: int = 0
    outcome_stalled_count: int = 0
    outcome_rejected_count: int = 0
    outcome_followup_no_response_count: int = 0
    due_followup_count: int = 0
    stalled_commitment_count: int = 0
    recent_completion_count: int = 0

    def lifecycle_for(self, record_id: str) -> CommitmentLifecycleEntry | None:
        """Look up a single record's lifecycle, or ``None`` if absent."""
        for entry in self.lifecycle_entries:
            if entry.record_id == record_id:
                return entry
        return None


@dataclass(frozen=True)
class OpenLoopSnapshot:
    unresolved_loops: tuple[SemanticRecord, ...]
    pending_confirmations: tuple[SemanticRecord, ...]
    closure_refs: tuple[str, ...]
    highest_priority_loop_id: str | None
    closure_pressure: float
    control_signal: float
    description: str
    oldest_open_turn: int | None = None
    stale_loop_count: int = 0
    confirmation_debt_count: int = 0
    closure_readiness: float = 0.0


@dataclass(frozen=True)
class UserModelSnapshot:
    stable_preferences: tuple[SemanticRecord, ...]
    working_style_hints: tuple[SemanticRecord, ...]
    sensitive_boundaries: tuple[SemanticRecord, ...]
    durable_goals: tuple[SemanticRecord, ...]
    stability_score: float
    control_signal: float
    description: str
    preferred_support_pacing: str = "unknown"
    decision_style: str = "unknown"
    overwhelm_pattern_strength: float = 0.0


@dataclass(frozen=True)
class ExecutionResultLifecycleEntry:
    """Per-record typed outcome on an execution_result record (Gap 10).

    Mirrors ``PlanIntentLifecycleEntry`` and ``CommitmentLifecycleEntry``
    so reflection writeback can treat all three owners uniformly when
    building the outcome audit trail.
    """

    record_id: str
    last_outcome: ExecutionResultOutcome | None = None
    last_outcome_evidence: str = ""
    last_outcome_at_turn: int = -1

    def __post_init__(self) -> None:
        has_outcome = self.last_outcome is not None
        if has_outcome and not self.last_outcome_evidence.strip():
            raise ValueError(
                "ExecutionResultLifecycleEntry.last_outcome set without "
                "non-empty last_outcome_evidence."
            )
        if has_outcome and self.last_outcome_at_turn < 0:
            raise ValueError(
                "ExecutionResultLifecycleEntry.last_outcome set without "
                "last_outcome_at_turn (>= 0)."
            )


@dataclass(frozen=True)
class ExecutionResultSnapshot:
    attempted_actions: tuple[SemanticRecord, ...]
    completed_actions: tuple[SemanticRecord, ...]
    failed_actions: tuple[SemanticRecord, ...]
    artifact_refs: tuple[str, ...]
    execution_grounding_score: float
    control_signal: float
    description: str
    # Gap 10 additions.
    lifecycle_entries: tuple[ExecutionResultLifecycleEntry, ...] = ()
    outcome_user_feedback_count: int = 0
    outcome_instruction_received_count: int = 0
    outcome_tool_outcome_count: int = 0
    outcome_crystal_evaluation_count: int = 0
    outcome_crystal_suppression_count: int = 0
    outcome_package_publication_count: int = 0
    outcome_bootstrap_consumption_count: int = 0

    def lifecycle_for(self, record_id: str) -> ExecutionResultLifecycleEntry | None:
        for entry in self.lifecycle_entries:
            if entry.record_id == record_id:
                return entry
        return None


@dataclass(frozen=True)
class BeliefAssumptionSnapshot:
    beliefs: tuple[SemanticRecord, ...]
    assumptions: tuple[SemanticRecord, ...]
    verification_needs: tuple[SemanticRecord, ...]
    contradiction_refs: tuple[str, ...]
    mean_confidence: float
    control_signal: float
    description: str


@dataclass(frozen=True)
class RelationshipStateSnapshot:
    trust_level: float
    continuity_level: float
    repair_pressure: float
    rapport_signals: tuple[SemanticRecord, ...]
    relational_tensions: tuple[SemanticRecord, ...]
    control_signal: float
    description: str
    emotional_load: float = 0.0
    repair_need: float = 0.0
    trust_delta: float = 0.0
    attunement_gap: float = 0.0
    stabilization_need: float = 0.0
    recent_repair_count: int = 0
    unresolved_tension_count: int = 0
    attunement_trend: float = 0.0
    trust_recovery_signal: float = 0.0
    relationship_continuity_score: float = 0.0


@dataclass(frozen=True)
class GoalValueSnapshot:
    explicit_goals: tuple[SemanticRecord, ...]
    value_priorities: tuple[SemanticRecord, ...]
    tradeoff_notes: tuple[SemanticRecord, ...]
    active_goal_id: str | None
    alignment_score: float
    control_signal: float
    description: str
    value_conflict: float = 0.0
    decision_readiness: float = 0.0
    active_tradeoff_count: int = 0
    reversibility_need: float = 0.0
    goal_shift_pressure: float = 0.0
    active_goal_count: int = 0
    deferred_goal_count: int = 0
    conflicted_goal_count: int = 0
    resolved_goal_refs: tuple[str, ...] = ()
    goal_continuity_score: float = 0.0


@dataclass(frozen=True)
class BoundaryConsentSnapshot:
    granted_consents: tuple[SemanticRecord, ...]
    missing_consents: tuple[SemanticRecord, ...]
    denied_boundaries: tuple[SemanticRecord, ...]
    memory_consent: str
    external_action_consent: str
    compliance_score: float
    control_signal: float
    description: str
    autonomy_risk: float = 0.0
    consent_clarity: float = 0.0
    professional_scope_pressure: float = 0.0
    overreach_risk: float = 0.0
    active_scope_count: int = 0
    denial_count: int = 0
    revocation_count: int = 0
    external_action_blocked: bool = False
    memory_scope_status: str = "unknown"


SemanticSnapshotValue = (
    PlanIntentSnapshot
    | CommitmentSnapshot
    | OpenLoopSnapshot
    | UserModelSnapshot
    | ExecutionResultSnapshot
    | BeliefAssumptionSnapshot
    | RelationshipStateSnapshot
    | GoalValueSnapshot
    | BoundaryConsentSnapshot
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def semantic_control_signal(value: object) -> float:
    signal = getattr(value, "control_signal", 0.0)
    return _clamp(float(signal)) if isinstance(signal, int | float) else 0.0


def semantic_snapshot_description(value: object) -> str:
    description = getattr(value, "description", "")
    return description if isinstance(description, str) else type(value).__name__
