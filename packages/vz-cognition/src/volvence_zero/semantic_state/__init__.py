from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from typing import Any, ClassVar, Mapping

from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import SubstrateSnapshot


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


# Map each SemanticProposal operation to a lifecycle transition for
# commitment records. The mapping is intentionally narrow \u2014 only
# operations that meaningfully advance advocacy or alignment are
# represented; the rest leave the existing state in place. Missing
# entries keep whatever the previous operation set.
_COMMITMENT_LIFECYCLE_TRANSITIONS: dict[
    SemanticProposalOperation, tuple[AdvocacyState | None, AlignmentState | None]
] = {
    SemanticProposalOperation.OBSERVE: (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
    SemanticProposalOperation.CREATE: (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
    SemanticProposalOperation.DEFER: (AdvocacyState.READY, None),
    SemanticProposalOperation.ACTIVATE: (AdvocacyState.PROPOSED, None),
    SemanticProposalOperation.REVISE: (AdvocacyState.PROPOSED, AlignmentState.MODIFY),
    SemanticProposalOperation.COMPLETE: (AdvocacyState.PROPOSED, AlignmentState.AGREE),
    SemanticProposalOperation.CLOSE: (AdvocacyState.PROPOSED, None),
    SemanticProposalOperation.BLOCK: (AdvocacyState.PROPOSED, AlignmentState.REJECT),
}


# Map each operation to the follow-up policy that best fits the resulting
# lifecycle state. ``None`` means "leave whatever was set before". A
# freshly-observed commitment starts with ``GENTLE_CHECKIN`` so that the
# FollowupManager treats it as a normal engage-on-due item; ``BLOCK``
# (user rejected) and ``DEFER`` (explicit hold) flip it to ``DEFER_ONLY``
# so the lifeform does not badger the user about a commitment they just
# pushed back against.
_COMMITMENT_FOLLOWUP_POLICY_TRANSITIONS: dict[
    SemanticProposalOperation, FollowupPolicy | None
] = {
    SemanticProposalOperation.OBSERVE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.CREATE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.DEFER: FollowupPolicy.DEFER_ONLY,
    SemanticProposalOperation.ACTIVATE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.REVISE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.COMPLETE: None,
    SemanticProposalOperation.CLOSE: None,
    SemanticProposalOperation.BLOCK: FollowupPolicy.DEFER_ONLY,
}


# Map each operation to the typed outcome it produces, if any. Used by
# reflection writeback to record a single canonical outcome enum per
# commitment transition so downstream consumers (ETA credit, regime
# calibration, case_memory) can key off a stable label rather than
# reparse lifecycle state pairs. ``None`` means the operation does not
# represent a meaningful outcome (merely an observation / advocacy
# move), so no outcome is recorded.
_COMMITMENT_OUTCOME_TRANSITIONS: dict[
    SemanticProposalOperation, CommitmentOutcomeKind | None
] = {
    SemanticProposalOperation.OBSERVE: None,
    SemanticProposalOperation.CREATE: None,
    SemanticProposalOperation.DEFER: CommitmentOutcomeKind.STALLED,
    SemanticProposalOperation.ACTIVATE: CommitmentOutcomeKind.PROGRESSED,
    SemanticProposalOperation.REVISE: CommitmentOutcomeKind.PROGRESSED,
    SemanticProposalOperation.COMPLETE: CommitmentOutcomeKind.COMPLETED,
    SemanticProposalOperation.CLOSE: CommitmentOutcomeKind.STALLED,
    SemanticProposalOperation.BLOCK: CommitmentOutcomeKind.REJECTED,
}


def commitment_lifecycle_for_operation(
    operation: SemanticProposalOperation,
    *,
    previous: tuple[AdvocacyState, AlignmentState] | None = None,
) -> tuple[AdvocacyState, AlignmentState]:
    """Pure helper exposing the operation \u2192 lifecycle map.

    Public so reflection writeback / evaluation / tests can derive the
    same lifecycle the commitment owner uses, without duplicating the
    truth table. ``previous`` lets a transition leave one axis untouched
    (e.g. ``ACTIVATE`` advances advocacy but leaves alignment as
    whatever the user-side signals last said).
    """
    base_advocacy = previous[0] if previous else AdvocacyState.NOT_READY
    base_alignment = previous[1] if previous else AlignmentState.UNKNOWN
    advocacy, alignment = _COMMITMENT_LIFECYCLE_TRANSITIONS.get(
        operation, (None, None)
    )
    return (
        advocacy if advocacy is not None else base_advocacy,
        alignment if alignment is not None else base_alignment,
    )


def commitment_followup_policy_for_operation(
    operation: SemanticProposalOperation,
    *,
    previous: FollowupPolicy | None = None,
) -> FollowupPolicy:
    """Pure helper exposing the operation \u2192 follow-up policy map.

    Defaults to ``GENTLE_CHECKIN`` when both ``previous`` is None and the
    operation is unmapped, so that callers constructing a fresh lifecycle
    entry always get a usable policy.
    """
    policy = _COMMITMENT_FOLLOWUP_POLICY_TRANSITIONS.get(operation)
    if policy is not None:
        return policy
    return previous or FollowupPolicy.GENTLE_CHECKIN


def commitment_outcome_for_operation(
    operation: SemanticProposalOperation,
) -> CommitmentOutcomeKind | None:
    """Pure helper exposing the operation \u2192 outcome enum.

    ``None`` means the operation did not produce a durable outcome
    (observe / create is a status read, not an outcome). Callers must
    treat ``None`` as "leave the previous outcome in place" \u2014 never
    overwrite a real outcome with nothing.
    """
    return _COMMITMENT_OUTCOME_TRANSITIONS.get(operation)


# Gap 10: plan-intent outcome taxonomy. A plan / intent lifecycle
# maps to four named outcome kinds. OBSERVE / CREATE are "status
# reads" and intentionally do NOT produce a typed outcome; callers
# treat ``None`` as "leave the previous outcome in place".
_PLAN_INTENT_OUTCOME_TRANSITIONS: dict[
    SemanticProposalOperation, PlanIntentOutcome | None
] = {
    SemanticProposalOperation.OBSERVE: None,
    SemanticProposalOperation.CREATE: PlanIntentOutcome.ASSUMPTION_RECORDED,
    SemanticProposalOperation.DEFER: PlanIntentOutcome.PROBLEM_PROGRESS_ASSESSED,
    SemanticProposalOperation.ACTIVATE: PlanIntentOutcome.DECISION_MADE,
    SemanticProposalOperation.REVISE: PlanIntentOutcome.DECISION_MADE,
    SemanticProposalOperation.COMPLETE: PlanIntentOutcome.OUTCOME_OBSERVED,
    SemanticProposalOperation.CLOSE: PlanIntentOutcome.OUTCOME_OBSERVED,
    SemanticProposalOperation.BLOCK: PlanIntentOutcome.PROBLEM_PROGRESS_ASSESSED,
}


def plan_intent_outcome_for_operation(
    operation: SemanticProposalOperation,
) -> PlanIntentOutcome | None:
    """Pure helper exposing the operation \u2192 plan_intent outcome enum."""
    return _PLAN_INTENT_OUTCOME_TRANSITIONS.get(operation)


# Gap 10: execution-result outcome taxonomy. The execution_result
# owner receives external signals via adapters (tool results, profile
# events, etc.) so the mapping here focuses on the status bucket the
# owner actually writes. Product-specific subtypes (crystal
# evaluation / suppression, package publication, bootstrap
# consumption) do not arise from ``SemanticProposalOperation`` alone
# \u2014 they come in through reviewed-knowledge / task events / test
# writes. For those, ``None`` is returned here and the caller passes
# an explicit ``ExecutionResultOutcome`` to the store when writing.
_EXECUTION_RESULT_OUTCOME_TRANSITIONS: dict[
    SemanticProposalOperation, ExecutionResultOutcome | None
] = {
    SemanticProposalOperation.OBSERVE: None,
    SemanticProposalOperation.CREATE: None,
    SemanticProposalOperation.DEFER: None,
    SemanticProposalOperation.ACTIVATE: None,
    SemanticProposalOperation.REVISE: None,
    SemanticProposalOperation.COMPLETE: ExecutionResultOutcome.TOOL_OUTCOME,
    SemanticProposalOperation.CLOSE: ExecutionResultOutcome.TOOL_OUTCOME,
    SemanticProposalOperation.BLOCK: ExecutionResultOutcome.TOOL_OUTCOME,
}


def execution_result_outcome_for_operation(
    operation: SemanticProposalOperation,
) -> ExecutionResultOutcome | None:
    """Pure helper exposing the operation \u2192 execution_result outcome.

    Returns ``None`` for status-read / planning operations so the
    caller can leave the previous outcome in place. Callers that have
    direct typed information (e.g. a ``user_feedback_received`` event
    from a tool_result adapter) should pass an explicit outcome to
    the store rather than rely on this mapping.
    """
    return _EXECUTION_RESULT_OUTCOME_TRANSITIONS.get(operation)


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


class SemanticProposalRuntime(ABC):
    runtime_id: str

    @abstractmethod
    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: SemanticSnapshotValue | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        """Return typed semantic proposals for a single owner slot."""


class NoOpSemanticProposalRuntime(SemanticProposalRuntime):
    runtime_id = "semantic-noop"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: SemanticSnapshotValue | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        evidence = user_input or ""
        if not evidence:
            return SemanticProposalBatch(
                proposals=(),
                runtime_id=self.runtime_id,
                schema_version=1,
                description=f"No-op semantic runtime skipped {target_slot}; no user evidence.",
            )
        proposal = SemanticProposal(
            proposal_id=f"{target_slot}:observe:{turn_index}",
            target_slot=target_slot,
            operation=SemanticProposalOperation.OBSERVE,
            summary="latest-turn-observed",
            detail=evidence[:240],
            confidence=0.20 if evidence else 0.0,
            evidence=evidence[:240],
            control_signal=0.02 if evidence else 0.0,
        )
        return SemanticProposalBatch(
            proposals=(proposal,),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"No-op semantic runtime published observation for {target_slot}.",
        )


class SemanticEventAdapter(ABC):
    @abstractmethod
    def adapt(
        self,
        *,
        event: ExternalSemanticEvent,
        target_slot: str,
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        """Map a structured external event to owner-specific proposals."""


def _proposal(
    *,
    event_id: str,
    target_slot: str,
    operation: SemanticProposalOperation,
    summary: str,
    detail: str,
    confidence: float,
    evidence: str,
    turn_index: int,
    control_signal: float = 0.18,
    requires_confirmation: bool = False,
) -> SemanticProposal:
    return SemanticProposal(
        proposal_id=f"{event_id}:{target_slot}:{operation.value}:{turn_index}",
        target_slot=target_slot,
        operation=operation,
        summary=summary[:160],
        detail=detail[:320],
        confidence=_clamp(confidence),
        evidence=evidence[:320],
        control_signal=_clamp(control_signal),
        requires_confirmation=requires_confirmation,
    )


class ToolResultSemanticAdapter(SemanticEventAdapter):
    def adapt(
        self,
        *,
        event: ExternalSemanticEvent,
        target_slot: str,
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        if not isinstance(event, ToolResultSemanticEvent):
            return ()
        status = event.status
        succeeded = status in {"succeeded", "completed", "ok"}
        operation = SemanticProposalOperation.COMPLETE if succeeded else SemanticProposalOperation.BLOCK
        evidence = f"tool={event.tool_name} action={event.action_id} status={event.status} detail={event.detail}"
        if target_slot == "execution_result":
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=operation,
                    summary=event.summary,
                    detail=event.detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.34,
                ),
            )
        if target_slot == "belief_assumption":
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=f"tool-evidence:{event.tool_name}",
                    detail=event.summary,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.18,
                ),
            )
        if target_slot == "open_loop" and not succeeded:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.CREATE,
                    summary=f"follow-up:{event.tool_name}",
                    detail=event.detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.42,
                    requires_confirmation=True,
                ),
            )
        if target_slot == "plan_intent" and event.plan_ref is not None:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.REVISE,
                    summary=event.plan_ref,
                    detail=event.summary,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.24,
                ),
            )
        return ()


class ProfileSemanticAdapter(SemanticEventAdapter):
    def adapt(
        self,
        *,
        event: ExternalSemanticEvent,
        target_slot: str,
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        if not isinstance(event, ProfileSemanticEvent):
            return ()
        evidence = f"profile_source={event.source}"
        if target_slot == "user_model" and (event.preferences or event.goals or event.relationship_note):
            detail = "; ".join(event.preferences + event.goals + ((event.relationship_note,) if event.relationship_note else ()))
            profile_proposal = (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=f"profile:{event.source}",
                    detail=detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.26,
                ),
            )
            goal_proposals = tuple(
                _proposal(
                    event_id=f"{event.event_id}:durable-goal:{index}",
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=goal,
                    detail=goal,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.24,
                )
                for index, goal in enumerate(event.goals)
            )
            return profile_proposal + goal_proposals
        if target_slot == "goal_value" and event.goals:
            return tuple(
                _proposal(
                    event_id=f"{event.event_id}:goal:{index}",
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=goal,
                    detail=goal,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.22,
                )
                for index, goal in enumerate(event.goals)
            )
        if target_slot == "boundary_consent" and (event.consent_grants or event.consent_denials):
            grant_proposals = tuple(
                _proposal(
                    event_id=f"{event.event_id}:grant:{index}",
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=grant,
                    detail=grant,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.20,
                )
                for index, grant in enumerate(event.consent_grants)
            )
            denial_proposals = tuple(
                _proposal(
                    event_id=f"{event.event_id}:deny:{index}",
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.BLOCK,
                    summary=denial,
                    detail=denial,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.46,
                )
                for index, denial in enumerate(event.consent_denials)
            )
            return grant_proposals + denial_proposals
        if target_slot == "relationship_state" and event.relationship_note:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=f"relationship:{event.source}",
                    detail=event.relationship_note,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.22,
                ),
            )
        return ()


class TaskEventSemanticAdapter(SemanticEventAdapter):
    def adapt(
        self,
        *,
        event: ExternalSemanticEvent,
        target_slot: str,
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        if not isinstance(event, TaskSemanticEvent):
            return ()
        status = event.status
        evidence = f"task={event.task_id} status={status} due={event.due_hint or ''} detail={event.detail}"
        operation = {
            "deferred": SemanticProposalOperation.DEFER,
            "pending": SemanticProposalOperation.CREATE,
            "active": SemanticProposalOperation.ACTIVATE,
            "completed": SemanticProposalOperation.COMPLETE,
            "failed": SemanticProposalOperation.BLOCK,
            "blocked": SemanticProposalOperation.BLOCK,
        }.get(status, SemanticProposalOperation.OBSERVE)
        if target_slot == "plan_intent":
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=operation,
                    summary=event.summary,
                    detail=event.detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.36,
                ),
            )
        if target_slot == "open_loop" and status in {"pending", "deferred", "blocked", "failed"}:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.CREATE,
                    summary=event.summary,
                    detail=event.detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.38,
                    requires_confirmation=status in {"blocked", "failed"},
                ),
            )
        if target_slot == "commitment" and event.commitment_ref is not None:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=operation,
                    summary=event.commitment_ref,
                    detail=event.summary,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.28,
                ),
            )
        if target_slot == "execution_result" and status in {"completed", "failed", "blocked"}:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=operation,
                    summary=event.summary,
                    detail=event.detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.32,
                ),
            )
        return ()


class ReviewedKnowledgeSemanticAdapter(SemanticEventAdapter):
    def adapt(
        self,
        *,
        event: ExternalSemanticEvent,
        target_slot: str,
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        if not isinstance(event, ReviewedKnowledgeSemanticEvent):
            return ()
        evidence = f"knowledge={event.knowledge_id} source={event.source_label} detail={event.detail}"
        if target_slot == "belief_assumption":
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=event.summary,
                    detail=event.detail,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.24,
                ),
            )
        if target_slot == "goal_value" and event.relevance_hint:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=event.relevance_hint,
                    detail=event.summary,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.18,
                ),
            )
        if target_slot == "open_loop" and event.needs_followup:
            return (
                _proposal(
                    event_id=event.event_id,
                    target_slot=target_slot,
                    operation=SemanticProposalOperation.CREATE,
                    summary=f"review-followup:{event.knowledge_id}",
                    detail=event.summary,
                    confidence=event.confidence,
                    evidence=evidence,
                    turn_index=turn_index,
                    control_signal=0.34,
                    requires_confirmation=True,
                ),
            )
        return ()


DEFAULT_SEMANTIC_EVENT_ADAPTERS: tuple[SemanticEventAdapter, ...] = (
    ToolResultSemanticAdapter(),
    ProfileSemanticAdapter(),
    TaskEventSemanticAdapter(),
    ReviewedKnowledgeSemanticAdapter(),
)


class AdapterSemanticProposalRuntime(SemanticProposalRuntime):
    def __init__(
        self,
        *,
        base_runtime: SemanticProposalRuntime | None = None,
        external_events: tuple[ExternalSemanticEvent, ...] = (),
        adapters: tuple[SemanticEventAdapter, ...] = DEFAULT_SEMANTIC_EVENT_ADAPTERS,
    ) -> None:
        self._base_runtime = base_runtime or NoOpSemanticProposalRuntime()
        self._external_events = external_events
        self._adapters = adapters
        self.runtime_id = f"adapter-semantic+{self._base_runtime.runtime_id}"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: SemanticSnapshotValue | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        base_batch = self._base_runtime.propose(
            target_slot=target_slot,
            user_input=user_input,
            substrate_snapshot=substrate_snapshot,
            memory_snapshot=memory_snapshot,
            previous_snapshot=previous_snapshot,
            turn_index=turn_index,
        )
        adapter_proposals = tuple(
            proposal
            for event in self._external_events
            for adapter in self._adapters
            for proposal in adapter.adapt(
                event=event,
                target_slot=target_slot,
                turn_index=turn_index,
            )
        )
        proposals = base_batch.proposals + adapter_proposals
        return SemanticProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            schema_version=base_batch.schema_version,
            description=(
                f"{base_batch.description} Adapter runtime added {len(adapter_proposals)} "
                f"external proposal(s) for {target_slot}."
            ),
        )


def semantic_events_from_tool_result(
    *,
    event_id: str,
    tool_name: str,
    action_id: str,
    status: str,
    summary: str,
    detail: str,
    confidence: float = 0.8,
    artifact_refs: tuple[str, ...] = (),
    plan_ref: str | None = None,
) -> ExternalSemanticEventBatch:
    return ExternalSemanticEventBatch(
        events=(
            ToolResultSemanticEvent(
                event_id=event_id,
                tool_name=tool_name,
                action_id=action_id,
                status=status,
                summary=summary,
                detail=detail,
                confidence=confidence,
                artifact_refs=artifact_refs,
                plan_ref=plan_ref,
            ),
        ),
        source="tool-result",
        description=f"Tool result semantic event for {tool_name}:{action_id}.",
    )


def semantic_events_from_profile(
    *,
    event_id: str,
    source: str,
    preferences: tuple[str, ...] = (),
    goals: tuple[str, ...] = (),
    consent_grants: tuple[str, ...] = (),
    consent_denials: tuple[str, ...] = (),
    relationship_note: str = "",
    confidence: float = 0.75,
) -> ExternalSemanticEventBatch:
    return ExternalSemanticEventBatch(
        events=(
            ProfileSemanticEvent(
                event_id=event_id,
                source=source,
                preferences=preferences,
                goals=goals,
                consent_grants=consent_grants,
                consent_denials=consent_denials,
                relationship_note=relationship_note,
                confidence=confidence,
            ),
        ),
        source="profile",
        description=f"Profile semantic event from {source}.",
    )


def semantic_events_from_task_event(
    *,
    event_id: str,
    task_id: str,
    status: str,
    summary: str,
    detail: str,
    due_hint: str | None = None,
    commitment_ref: str | None = None,
    confidence: float = 0.75,
) -> ExternalSemanticEventBatch:
    return ExternalSemanticEventBatch(
        events=(
            TaskSemanticEvent(
                event_id=event_id,
                task_id=task_id,
                status=status,
                summary=summary,
                detail=detail,
                due_hint=due_hint,
                commitment_ref=commitment_ref,
                confidence=confidence,
            ),
        ),
        source="task-event",
        description=f"Task semantic event for {task_id}.",
    )


def semantic_events_from_reviewed_knowledge(
    *,
    event_id: str,
    knowledge_id: str,
    summary: str,
    detail: str,
    source_label: str,
    confidence: float,
    relevance_hint: str = "",
    needs_followup: bool = False,
) -> ExternalSemanticEventBatch:
    return ExternalSemanticEventBatch(
        events=(
            ReviewedKnowledgeSemanticEvent(
                event_id=event_id,
                knowledge_id=knowledge_id,
                summary=summary,
                detail=detail,
                source_label=source_label,
                confidence=confidence,
                relevance_hint=relevance_hint,
                needs_followup=needs_followup,
            ),
        ),
        source="reviewed-knowledge",
        description=f"Reviewed knowledge semantic event for {knowledge_id}.",
    )


@dataclass(frozen=True)
class _CommitmentOutcomeRecord:
    """Internal record: typed outcome + anchoring turn + evidence."""

    outcome: CommitmentOutcomeKind
    turn_index: int
    evidence: str


@dataclass(frozen=True)
class _PlanIntentOutcomeRecord:
    """Internal record for plan_intent outcome (Gap 10)."""

    outcome: PlanIntentOutcome
    turn_index: int
    evidence: str


@dataclass(frozen=True)
class _ExecutionResultOutcomeRecord:
    """Internal record for execution_result outcome (Gap 10)."""

    outcome: ExecutionResultOutcome
    turn_index: int
    evidence: str


# Per-slot dispatch for operation \u2192 outcome helpers. Lets ``apply``
# call the right helper per slot instead of branching on slot name.
# ``None`` means the slot does not participate in typed-outcome
# tracking (only commitment / plan_intent / execution_result do).
def _outcome_dispatch_for_slot(slot: str, operation: SemanticProposalOperation):
    if slot == "commitment":
        return commitment_outcome_for_operation(operation)
    if slot == "plan_intent":
        return plan_intent_outcome_for_operation(operation)
    if slot == "execution_result":
        return execution_result_outcome_for_operation(operation)
    return None


def _outcome_record_for_slot(
    slot: str,
    outcome: Any,
    *,
    turn_index: int,
    evidence: str,
):
    if slot == "commitment":
        return _CommitmentOutcomeRecord(
            outcome=outcome, turn_index=turn_index, evidence=evidence
        )
    if slot == "plan_intent":
        return _PlanIntentOutcomeRecord(
            outcome=outcome, turn_index=turn_index, evidence=evidence
        )
    if slot == "execution_result":
        return _ExecutionResultOutcomeRecord(
            outcome=outcome, turn_index=turn_index, evidence=evidence
        )
    raise ValueError(
        f"Unsupported outcome-tracking slot {slot!r}; expected one of "
        "commitment / plan_intent / execution_result."
    )


class SemanticStateStore:
    def __init__(self) -> None:
        self._records: dict[str, tuple[SemanticRecord, ...]] = {slot: () for slot in SEMANTIC_OWNER_SLOTS}
        self._completed_refs: dict[str, tuple[str, ...]] = {slot: () for slot in SEMANTIC_OWNER_SLOTS}
        self._revision_counts: dict[str, int] = {slot: 0 for slot in SEMANTIC_OWNER_SLOTS}
        # Per-record lifecycle state for the commitment owner (and any
        # other owner that later wants to consume it). Stored as
        # ``slot -> {record_id -> (advocacy, alignment)}`` so the latest
        # operation's transition wins and prior operations' state on the
        # untouched axis is preserved (see
        # ``commitment_lifecycle_for_operation``'s ``previous`` semantics).
        self._record_lifecycle: dict[
            str, dict[str, tuple[AdvocacyState, AlignmentState]]
        ] = {slot: {} for slot in SEMANTIC_OWNER_SLOTS}
        # Per-record follow-up policy. Same GC semantics as lifecycle.
        self._record_followup_policy: dict[str, dict[str, FollowupPolicy]] = {
            slot: {} for slot in SEMANTIC_OWNER_SLOTS
        }
        # Per-record typed outcome, anchored to the turn it was produced
        # and carrying non-empty evidence. Value type varies per slot:
        # - commitment   -> _CommitmentOutcomeRecord
        # - plan_intent  -> _PlanIntentOutcomeRecord  (Gap 10)
        # - execution_result -> _ExecutionResultOutcomeRecord  (Gap 10)
        # Other slots never populate this map.
        self._record_outcome: dict[str, dict[str, Any]] = {
            slot: {} for slot in SEMANTIC_OWNER_SLOTS
        }

    def apply(self, *, slot: str, proposals: tuple[SemanticProposal, ...], turn_index: int) -> tuple[SemanticRecord, ...]:
        existing = list(self._records[slot])
        completed_refs = list(self._completed_refs[slot])
        revision_count = self._revision_counts[slot]
        lifecycle_map = self._record_lifecycle[slot]
        policy_map = self._record_followup_policy[slot]
        outcome_map = self._record_outcome[slot]
        for proposal in proposals:
            if proposal.target_slot != slot:
                continue
            if proposal.operation in {SemanticProposalOperation.REVISE, SemanticProposalOperation.ACTIVATE}:
                revision_count += 1
            if proposal.operation in {SemanticProposalOperation.COMPLETE, SemanticProposalOperation.CLOSE}:
                completed_refs.append(proposal.proposal_id)
            status = {
                SemanticProposalOperation.DEFER: "deferred",
                SemanticProposalOperation.COMPLETE: "completed",
                SemanticProposalOperation.CLOSE: "closed",
                SemanticProposalOperation.BLOCK: "blocked",
            }.get(proposal.operation, "active")
            existing.append(
                SemanticRecord(
                    record_id=proposal.proposal_id,
                    summary=proposal.summary,
                    detail=proposal.detail,
                    confidence=_clamp(proposal.confidence),
                    status=status,
                    source_turn=turn_index,
                    evidence=proposal.evidence,
                    control_signal=_clamp(proposal.control_signal),
                )
            )
            previous = lifecycle_map.get(proposal.proposal_id)
            lifecycle_map[proposal.proposal_id] = (
                commitment_lifecycle_for_operation(
                    proposal.operation, previous=previous
                )
            )
            # Follow-up policy: keep previous if the operation does not
            # prescribe one; default is GENTLE_CHECKIN via the helper.
            policy_map[proposal.proposal_id] = commitment_followup_policy_for_operation(
                proposal.operation,
                previous=policy_map.get(proposal.proposal_id),
            )
            # Outcome: only record when the operation produces a typed
            # outcome. Evidence MUST be non-empty \u2014 fall back to the
            # proposal's evidence field or (as last resort) a short
            # operation+summary trace so the outcome never ships with an
            # empty audit string. Never silently overwrite an existing
            # outcome with None. Per-slot dispatch lets commitment /
            # plan_intent / execution_result each carry their own
            # outcome taxonomy without a mega-if.
            outcome_kind = _outcome_dispatch_for_slot(slot, proposal.operation)
            if outcome_kind is not None:
                evidence_text = proposal.evidence.strip() or (
                    f"op={proposal.operation.value} summary={proposal.summary}".strip()
                )
                if not evidence_text:
                    evidence_text = (
                        f"op={proposal.operation.value} "
                        f"record_id={proposal.proposal_id}"
                    )
                outcome_map[proposal.proposal_id] = _outcome_record_for_slot(
                    slot,
                    outcome_kind,
                    turn_index=turn_index,
                    evidence=evidence_text[:320],
                )
        self._records[slot] = tuple(existing[-12:])
        self._completed_refs[slot] = tuple(completed_refs[-12:])
        self._revision_counts[slot] = revision_count
        # Garbage-collect lifecycle / policy / outcome entries whose
        # record id has fallen out of the bounded window. Avoids
        # unbounded growth across long sessions while still letting
        # late-arriving proposals reuse earlier ids during the same
        # session.
        live_ids = {record.record_id for record in self._records[slot]}
        for record_id in tuple(lifecycle_map.keys()):
            if record_id not in live_ids:
                del lifecycle_map[record_id]
        for record_id in tuple(policy_map.keys()):
            if record_id not in live_ids:
                del policy_map[record_id]
        for record_id in tuple(outcome_map.keys()):
            if record_id not in live_ids:
                del outcome_map[record_id]
        return self._records[slot]

    def records_for(self, slot: str) -> tuple[SemanticRecord, ...]:
        return self._records[slot]

    def completed_refs_for(self, slot: str) -> tuple[str, ...]:
        return self._completed_refs[slot]

    def revision_count_for(self, slot: str) -> int:
        return self._revision_counts[slot]

    def lifecycle_for(
        self, slot: str
    ) -> dict[str, tuple[AdvocacyState, AlignmentState]]:
        """Return a copy of the per-record lifecycle map for ``slot``."""
        return dict(self._record_lifecycle[slot])

    def followup_policy_for(self, slot: str) -> dict[str, FollowupPolicy]:
        """Return a copy of the per-record follow-up policy map for ``slot``."""
        return dict(self._record_followup_policy[slot])

    def outcome_for(self, slot: str) -> dict[str, Any]:
        """Return a copy of the per-record typed-outcome map for ``slot``.

        Value type varies per slot (see ``_record_outcome`` attribute
        docstring). Callers that care about the typed enum should
        inspect ``record.outcome`` after lookup.
        """
        return dict(self._record_outcome[slot])


def _records_with_status(records: tuple[SemanticRecord, ...], *statuses: str) -> tuple[SemanticRecord, ...]:
    allowed = set(statuses)
    return tuple(record for record in records if record.status in allowed)


def _mean_record_control(records: tuple[SemanticRecord, ...]) -> float:
    if not records:
        return 0.0
    return _clamp(sum(record.control_signal for record in records) / len(records))


def _mean_record_confidence(records: tuple[SemanticRecord, ...]) -> float:
    if not records:
        return 0.0
    return _clamp(sum(record.confidence for record in records) / len(records))


class SemanticOwnerModule(RuntimeModule[SemanticSnapshotValue]):
    slot_name: ClassVar[str]
    owner: ClassVar[str]
    value_type: ClassVar[type[Any]]
    dependencies = ("substrate", "memory")
    default_wiring_level = WiringLevel.ACTIVE
    # Owners that want to drop low-confidence proposals before they
    # mutate the store override this. Default 0 keeps the historical
    # behaviour: every proposal that the runtime emits flows into
    # ``SemanticStateStore.apply``. Owners that absorb LLM-classified
    # proposals (e.g. ``CommitmentModule`` consuming
    # ``LLMSemanticProposalRuntime``) raise this so a routine OBSERVE
    # (confidence ~0.20-0.25) cannot accidentally enter the lifecycle
    # log and inflate AAC counters \u2014 the noise that phase B+A's
    # verify exposed (``outcome_rejected_count: 6`` from a single
    # explicit BLOCK probe). The threshold lives at the *owner*, not
    # the runtime, because the policy decision is owner-specific:
    # the user_model owner, for example, *wants* to absorb every
    # observation regardless of confidence.
    min_proposal_confidence: ClassVar[float] = 0.0

    def __init__(
        self,
        *,
        store: SemanticStateStore,
        proposal_runtime: SemanticProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._store = store
        self._proposal_runtime = proposal_runtime or NoOpSemanticProposalRuntime()
        self._user_input = user_input
        self._turn_index = turn_index
        self._last_snapshot: SemanticSnapshotValue | None = None

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[SemanticSnapshotValue]:
        substrate_value = upstream["substrate"].value
        memory_value = upstream["memory"].value
        substrate_snapshot = substrate_value if isinstance(substrate_value, SubstrateSnapshot) else None
        memory_snapshot = memory_value if isinstance(memory_value, MemorySnapshot) else None
        batch = self._proposal_runtime.propose(
            target_slot=self.slot_name,
            user_input=self._user_input,
            substrate_snapshot=substrate_snapshot,
            memory_snapshot=memory_snapshot,
            previous_snapshot=self._last_snapshot,
            turn_index=self._turn_index,
        )
        accepted = self._filter_proposals_by_confidence(batch.proposals)
        records = self._store.apply(
            slot=self.slot_name,
            proposals=accepted,
            turn_index=self._turn_index,
        )
        value = self._build_snapshot(records=records, batch=batch)
        self._last_snapshot = value
        return self.publish(value)

    def _filter_proposals_by_confidence(
        self, proposals: tuple[SemanticProposal, ...]
    ) -> tuple[SemanticProposal, ...]:
        """Keep only proposals at-or-above ``min_proposal_confidence``.

        Runs at the owner layer so the snapshot's ``batch`` field
        still reflects the ORIGINAL runtime emission (audit trail
        intact), while ``_store.apply`` only sees the accepted
        subset. Owners that want every proposal applied keep the
        default threshold of 0.0 \u2014 a strict ``>=`` check guarantees
        the historical behaviour for that case (a 0.0-confidence
        proposal still passes ``>= 0.0``).
        """
        threshold = self.min_proposal_confidence
        if threshold <= 0.0:
            return proposals
        return tuple(p for p in proposals if p.confidence >= threshold)

    async def process_standalone(self, **kwargs: Any) -> Snapshot[SemanticSnapshotValue]:
        user_input = kwargs.get("user_input")
        if user_input is not None and not isinstance(user_input, str):
            raise TypeError("user_input must be a string when provided.")
        self._user_input = user_input
        return await self.process(
            {
                "substrate": kwargs["substrate"],
                "memory": kwargs["memory"],
            }
        )

    def _latest_active(self, records: tuple[SemanticRecord, ...]) -> SemanticRecord | None:
        active = _records_with_status(records, "active")
        return active[-1] if active else None

    def _mean_confidence(self, records: tuple[SemanticRecord, ...]) -> float:
        if not records:
            return 0.0
        return _clamp(sum(record.confidence for record in records) / len(records))

    def _batch_signal(self, batch: SemanticProposalBatch) -> float:
        if not batch.proposals:
            return 0.0
        return _clamp(sum(_clamp(item.control_signal) for item in batch.proposals) / len(batch.proposals))

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> SemanticSnapshotValue:
        raise NotImplementedError


class PlanIntentModule(SemanticOwnerModule):
    slot_name = "plan_intent"
    owner = "PlanIntentModule"
    value_type = PlanIntentSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> PlanIntentSnapshot:
        latest = self._latest_active(records)
        confidence = self._mean_confidence(records)
        outcome_map = self._store.outcome_for(self.slot_name)
        lifecycle_entries: list[PlanIntentLifecycleEntry] = []
        decision_made = assumption_recorded = 0
        problem_progress_assessed = outcome_observed = 0
        for record in records:
            outcome_record = outcome_map.get(record.record_id)
            if outcome_record is None:
                lifecycle_entries.append(
                    PlanIntentLifecycleEntry(record_id=record.record_id)
                )
                continue
            last_outcome = outcome_record.outcome
            lifecycle_entries.append(
                PlanIntentLifecycleEntry(
                    record_id=record.record_id,
                    last_outcome=last_outcome,
                    last_outcome_evidence=outcome_record.evidence,
                    last_outcome_at_turn=outcome_record.turn_index,
                )
            )
            if last_outcome is PlanIntentOutcome.DECISION_MADE:
                decision_made += 1
            elif last_outcome is PlanIntentOutcome.ASSUMPTION_RECORDED:
                assumption_recorded += 1
            elif last_outcome is PlanIntentOutcome.PROBLEM_PROGRESS_ASSESSED:
                problem_progress_assessed += 1
            elif last_outcome is PlanIntentOutcome.OUTCOME_OBSERVED:
                outcome_observed += 1
        return PlanIntentSnapshot(
            active_plan_id=latest.record_id if latest else None,
            active_goal=latest.summary if latest else "",
            active_step=latest.detail if latest else "",
            active_constraints=tuple(record.detail for record in records if record.status == "blocked")[:4],
            deferred_intents=_records_with_status(records, "deferred"),
            standing_plans=(),
            candidate_plans=_records_with_status(records, "active"),
            completed_plan_refs=self._store.completed_refs_for(self.slot_name),
            plan_revision_count=self._store.revision_count_for(self.slot_name),
            continuity_score=confidence,
            control_signal=self._batch_signal(batch),
            description=(
                f"Plan/intent owner published {len(records)} records; "
                f"active={latest.record_id if latest else 'none'} "
                f"outcomes[decision={decision_made} "
                f"assumption={assumption_recorded} "
                f"progress={problem_progress_assessed} "
                f"observed={outcome_observed}]."
            ),
            lifecycle_entries=tuple(lifecycle_entries),
            outcome_decision_made_count=decision_made,
            outcome_assumption_recorded_count=assumption_recorded,
            outcome_problem_progress_assessed_count=problem_progress_assessed,
            outcome_observed_count=outcome_observed,
        )


class CommitmentModule(SemanticOwnerModule):
    slot_name = "commitment"
    owner = "CommitmentModule"
    value_type = CommitmentSnapshot
    # Confidence floor calibrated against the two runtimes that
    # currently feed this owner:
    #   * ``NoOpSemanticProposalRuntime``  emits OBSERVE @ 0.20
    #   * ``LLMSemanticProposalRuntime``   emits OBSERVE @ 0.25,
    #                                      DEFER @ 0.50,
    #                                      CREATE @ 0.55,
    #                                      COMPLETE / BLOCK @ 0.60
    # 0.40 is below DEFER (the lowest-confidence operation we WANT
    # in the lifecycle) and above OBSERVE (which is just "the user
    # said something, no commitment-relevant change") for both
    # runtimes. Result: AAC counters now reflect classified events,
    # not every turn the kernel observed.
    min_proposal_confidence: ClassVar[float] = 0.40

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> CommitmentSnapshot:
        active = _records_with_status(records, "active")
        at_risk = _records_with_status(records, "blocked")
        # Pull the per-record lifecycle / policy / outcome maps maintained
        # by the store and publish them as a parallel tuple. Records
        # present in the bounded window without a lifecycle entry
        # (legacy / synthetic records) default to
        # (NOT_READY, UNKNOWN, GENTLE_CHECKIN, no outcome).
        lifecycle_map = self._store.lifecycle_for(self.slot_name)
        policy_map = self._store.followup_policy_for(self.slot_name)
        outcome_map = self._store.outcome_for(self.slot_name)
        lifecycle_entries: list[CommitmentLifecycleEntry] = []
        ready = proposed = 0
        agree = modify = reject = 0
        gentle = defer_only = 0
        outcome_progressed = outcome_completed = outcome_stalled = 0
        outcome_rejected = outcome_followup_none = 0
        for record in records:
            advocacy, alignment = lifecycle_map.get(
                record.record_id, (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN)
            )
            policy = policy_map.get(record.record_id, FollowupPolicy.GENTLE_CHECKIN)
            outcome_record = outcome_map.get(record.record_id)
            last_outcome = outcome_record.outcome if outcome_record else None
            last_outcome_evidence = outcome_record.evidence if outcome_record else ""
            last_outcome_at_turn = outcome_record.turn_index if outcome_record else -1
            lifecycle_entries.append(
                CommitmentLifecycleEntry(
                    record_id=record.record_id,
                    advocacy_state=advocacy,
                    alignment_state=alignment,
                    followup_policy=policy,
                    last_outcome=last_outcome,
                    last_outcome_evidence=last_outcome_evidence,
                    last_outcome_at_turn=last_outcome_at_turn,
                )
            )
            if advocacy is AdvocacyState.READY:
                ready += 1
            elif advocacy is AdvocacyState.PROPOSED:
                proposed += 1
            if alignment is AlignmentState.AGREE:
                agree += 1
            elif alignment is AlignmentState.MODIFY:
                modify += 1
            elif alignment is AlignmentState.REJECT:
                reject += 1
            if policy is FollowupPolicy.GENTLE_CHECKIN:
                gentle += 1
            elif policy is FollowupPolicy.DEFER_ONLY:
                defer_only += 1
            if last_outcome is CommitmentOutcomeKind.PROGRESSED:
                outcome_progressed += 1
            elif last_outcome is CommitmentOutcomeKind.COMPLETED:
                outcome_completed += 1
            elif last_outcome is CommitmentOutcomeKind.STALLED:
                outcome_stalled += 1
            elif last_outcome is CommitmentOutcomeKind.REJECTED:
                outcome_rejected += 1
            elif last_outcome is CommitmentOutcomeKind.FOLLOWUP_NO_RESPONSE:
                outcome_followup_none += 1
        return CommitmentSnapshot(
            active_commitments=active,
            honored_commitment_refs=self._store.completed_refs_for(self.slot_name),
            at_risk_commitments=at_risk,
            trust_obligation_count=len(active),
            continuity_score=self._mean_confidence(active),
            control_signal=self._batch_signal(batch),
            description=(
                f"Commitment owner published active={len(active)} "
                f"at_risk={len(at_risk)} proposed={proposed} "
                f"agreed={agree} modify={modify} rejected={reject} "
                f"gentle={gentle} defer_only={defer_only} "
                f"outcome[progressed={outcome_progressed} "
                f"completed={outcome_completed} stalled={outcome_stalled} "
                f"rejected={outcome_rejected} "
                f"no_response={outcome_followup_none}]."
            ),
            lifecycle_entries=tuple(lifecycle_entries),
            advocacy_proposed_count=proposed,
            advocacy_ready_count=ready,
            alignment_agree_count=agree,
            alignment_modify_count=modify,
            alignment_reject_count=reject,
            followup_gentle_count=gentle,
            followup_defer_only_count=defer_only,
            outcome_progressed_count=outcome_progressed,
            outcome_completed_count=outcome_completed,
            outcome_stalled_count=outcome_stalled,
            outcome_rejected_count=outcome_rejected,
            outcome_followup_no_response_count=outcome_followup_none,
        )


class OpenLoopModule(SemanticOwnerModule):
    slot_name = "open_loop"
    owner = "OpenLoopModule"
    value_type = OpenLoopSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> OpenLoopSnapshot:
        unresolved = _records_with_status(records, "active", "deferred")
        confirmations = tuple(record for record in unresolved if record.confidence < 0.55)
        highest = unresolved[-1].record_id if unresolved else None
        return OpenLoopSnapshot(
            unresolved_loops=unresolved,
            pending_confirmations=confirmations,
            closure_refs=self._store.completed_refs_for(self.slot_name),
            highest_priority_loop_id=highest,
            closure_pressure=_clamp(len(unresolved) / 5.0),
            control_signal=max(self._batch_signal(batch), _clamp(len(confirmations) / 5.0)),
            description=f"Open-loop owner published unresolved={len(unresolved)} confirmations={len(confirmations)}.",
        )


class UserModelModule(SemanticOwnerModule):
    slot_name = "user_model"
    owner = "UserModelModule"
    value_type = UserModelSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> UserModelSnapshot:
        sensitive_boundaries = _records_with_status(records, "blocked")
        durable_goals = tuple(record for record in records if ":durable-goal:" in record.record_id)[-4:]
        active_records = _records_with_status(records, "active")
        stability_score = self._mean_confidence(records)
        overwhelm_pattern_strength = _clamp(
            _mean_record_control(records) * 0.52
            + (1.0 - stability_score) * 0.22
            + min(len(sensitive_boundaries) / 4.0, 1.0) * 0.26
        )
        preferred_support_pacing = (
            "support-first"
            if overwhelm_pattern_strength >= 0.35 or sensitive_boundaries
            else "standard"
        )
        decision_style = (
            "values-first"
            if durable_goals or len(active_records) >= 2
            else "unknown"
        )
        return UserModelSnapshot(
            stable_preferences=active_records[-4:],
            working_style_hints=records[-4:],
            sensitive_boundaries=sensitive_boundaries,
            durable_goals=durable_goals,
            stability_score=stability_score,
            control_signal=self._batch_signal(batch),
            description=(
                f"User-model owner published {len(records)} profile records; "
                f"pacing={preferred_support_pacing} decision_style={decision_style} "
                f"overwhelm={overwhelm_pattern_strength:.2f} durable_goals={len(durable_goals)}."
            ),
            preferred_support_pacing=preferred_support_pacing,
            decision_style=decision_style,
            overwhelm_pattern_strength=overwhelm_pattern_strength,
        )


class ExecutionResultModule(SemanticOwnerModule):
    slot_name = "execution_result"
    owner = "ExecutionResultModule"
    value_type = ExecutionResultSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> ExecutionResultSnapshot:
        completed = _records_with_status(records, "completed")
        failed = _records_with_status(records, "blocked")
        outcome_map = self._store.outcome_for(self.slot_name)
        lifecycle_entries: list[ExecutionResultLifecycleEntry] = []
        user_feedback = instruction = tool_outcome = 0
        crystal_eval = crystal_suppress = 0
        package_pub = bootstrap_cons = 0
        for record in records:
            outcome_record = outcome_map.get(record.record_id)
            if outcome_record is None:
                lifecycle_entries.append(
                    ExecutionResultLifecycleEntry(record_id=record.record_id)
                )
                continue
            last_outcome = outcome_record.outcome
            lifecycle_entries.append(
                ExecutionResultLifecycleEntry(
                    record_id=record.record_id,
                    last_outcome=last_outcome,
                    last_outcome_evidence=outcome_record.evidence,
                    last_outcome_at_turn=outcome_record.turn_index,
                )
            )
            if last_outcome is ExecutionResultOutcome.USER_FEEDBACK_RECEIVED:
                user_feedback += 1
            elif last_outcome is ExecutionResultOutcome.INSTRUCTION_RECEIVED:
                instruction += 1
            elif last_outcome is ExecutionResultOutcome.TOOL_OUTCOME:
                tool_outcome += 1
            elif last_outcome is ExecutionResultOutcome.CRYSTAL_EVALUATION:
                crystal_eval += 1
            elif last_outcome is ExecutionResultOutcome.CRYSTAL_SUPPRESSION:
                crystal_suppress += 1
            elif last_outcome is ExecutionResultOutcome.PACKAGE_PUBLICATION:
                package_pub += 1
            elif last_outcome is ExecutionResultOutcome.BOOTSTRAP_CONSUMPTION:
                bootstrap_cons += 1
        return ExecutionResultSnapshot(
            attempted_actions=records,
            completed_actions=completed,
            failed_actions=failed,
            artifact_refs=tuple(record.record_id for record in completed),
            execution_grounding_score=self._mean_confidence(completed or records),
            control_signal=self._batch_signal(batch),
            description=(
                f"Execution-result owner published attempted={len(records)} "
                f"completed={len(completed)} failed={len(failed)} "
                f"outcomes[tool={tool_outcome} feedback={user_feedback} "
                f"instruction={instruction} "
                f"crystal_eval={crystal_eval} crystal_suppress={crystal_suppress}]."
            ),
            lifecycle_entries=tuple(lifecycle_entries),
            outcome_user_feedback_count=user_feedback,
            outcome_instruction_received_count=instruction,
            outcome_tool_outcome_count=tool_outcome,
            outcome_crystal_evaluation_count=crystal_eval,
            outcome_crystal_suppression_count=crystal_suppress,
            outcome_package_publication_count=package_pub,
            outcome_bootstrap_consumption_count=bootstrap_cons,
        )


class BeliefAssumptionModule(SemanticOwnerModule):
    slot_name = "belief_assumption"
    owner = "BeliefAssumptionModule"
    value_type = BeliefAssumptionSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> BeliefAssumptionSnapshot:
        verification = tuple(record for record in records if record.confidence < 0.55)
        return BeliefAssumptionSnapshot(
            beliefs=tuple(record for record in records if record.confidence >= 0.55),
            assumptions=records,
            verification_needs=verification,
            contradiction_refs=tuple(record.record_id for record in _records_with_status(records, "blocked")),
            mean_confidence=self._mean_confidence(records),
            control_signal=max(self._batch_signal(batch), _clamp(len(verification) / 5.0)),
            description=f"Belief/assumption owner published assumptions={len(records)} verification={len(verification)}.",
        )


class RelationshipStateModule(SemanticOwnerModule):
    slot_name = "relationship_state"
    owner = "RelationshipStateModule"
    value_type = RelationshipStateSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> RelationshipStateSnapshot:
        tensions = _records_with_status(records, "blocked")
        confidence = self._mean_confidence(records)
        emotional_load = _clamp(
            _mean_record_control(records) * 0.44
            + (1.0 - confidence) * 0.24
            + min(len(records) / 6.0, 1.0) * 0.12
            + min(len(tensions) / 4.0, 1.0) * 0.20
        )
        repair_need = _clamp(
            min(len(tensions) / 4.0, 1.0) * 0.58
            + _mean_record_control(tensions) * 0.24
            + (1.0 - confidence) * 0.18
        )
        trust_level = _clamp(0.45 + confidence * 0.35 - len(tensions) * 0.05)
        continuity_level = _clamp(0.35 + len(records) / 10.0)
        trust_delta = _clamp(trust_level - 0.5)
        attunement_gap = _clamp((1.0 - trust_level) * 0.55 + repair_need * 0.45)
        stabilization_need = _clamp(emotional_load * 0.62 + repair_need * 0.22 + attunement_gap * 0.16)
        return RelationshipStateSnapshot(
            trust_level=trust_level,
            continuity_level=continuity_level,
            repair_pressure=_clamp(len(tensions) / 4.0),
            rapport_signals=records[-4:],
            relational_tensions=tensions,
            control_signal=self._batch_signal(batch),
            description=(
                f"Relationship-state owner published continuity={len(records)} tensions={len(tensions)} "
                f"emotional_load={emotional_load:.2f} repair_need={repair_need:.2f} "
                f"stabilization_need={stabilization_need:.2f}."
            ),
            emotional_load=emotional_load,
            repair_need=repair_need,
            trust_delta=trust_delta,
            attunement_gap=attunement_gap,
            stabilization_need=stabilization_need,
        )


class GoalValueModule(SemanticOwnerModule):
    slot_name = "goal_value"
    owner = "GoalValueModule"
    value_type = GoalValueSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> GoalValueSnapshot:
        latest = self._latest_active(records)
        tradeoffs = _records_with_status(records, "deferred", "blocked")
        alignment_score = self._mean_confidence(records)
        revision_count = self._store.revision_count_for(self.slot_name)
        active_tradeoff_count = len(tradeoffs)
        value_conflict = _clamp(
            min(active_tradeoff_count / 4.0, 1.0) * 0.50
            + (1.0 - alignment_score) * 0.24
            + _mean_record_control(tradeoffs) * 0.26
        )
        goal_shift_pressure = _clamp(min(revision_count / 4.0, 1.0) * 0.72 + value_conflict * 0.28)
        reversibility_need = _clamp(value_conflict * 0.50 + goal_shift_pressure * 0.30 + (1.0 - alignment_score) * 0.20)
        decision_readiness = _clamp(alignment_score * 0.62 + (1.0 - value_conflict) * 0.28 - reversibility_need * 0.10)
        return GoalValueSnapshot(
            explicit_goals=records,
            value_priorities=records[-4:],
            tradeoff_notes=tradeoffs,
            active_goal_id=latest.record_id if latest else None,
            alignment_score=alignment_score,
            control_signal=self._batch_signal(batch),
            description=(
                f"Goal/value owner published goals={len(records)} active={latest.record_id if latest else 'none'} "
                f"value_conflict={value_conflict:.2f} decision_readiness={decision_readiness:.2f} "
                f"reversibility_need={reversibility_need:.2f} shift={goal_shift_pressure:.2f}."
            ),
            value_conflict=value_conflict,
            decision_readiness=decision_readiness,
            active_tradeoff_count=active_tradeoff_count,
            reversibility_need=reversibility_need,
            goal_shift_pressure=goal_shift_pressure,
        )


class BoundaryConsentModule(SemanticOwnerModule):
    slot_name = "boundary_consent"
    owner = "BoundaryConsentModule"
    value_type = BoundaryConsentSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> BoundaryConsentSnapshot:
        granted = _records_with_status(records, "active", "completed")
        missing = tuple(record for record in records if record.confidence < 0.55 and record.status not in {"blocked", "closed"})
        denied = _records_with_status(records, "blocked")
        compliance = _clamp(1.0 - len(missing) * 0.12 - len(denied) * 0.20)
        autonomy_risk = _clamp(
            min((len(missing) + len(denied)) / 4.0, 1.0) * 0.44
            + _mean_record_control(denied or missing) * 0.34
            + (1.0 - compliance) * 0.22
        )
        consent_clarity = _clamp(
            len(granted) / max(len(granted) + len(missing) + len(denied), 1)
        )
        professional_scope_pressure = _clamp(
            min(len(denied) / 3.0, 1.0) * 0.52
            + _mean_record_confidence(denied) * 0.18
            + (1.0 - compliance) * 0.30
        )
        overreach_risk = _clamp(autonomy_risk * 0.62 + professional_scope_pressure * 0.38)
        return BoundaryConsentSnapshot(
            granted_consents=granted,
            missing_consents=missing,
            denied_boundaries=denied,
            memory_consent="unknown" if not granted else "granted",
            external_action_consent="unknown" if missing else "not-required",
            compliance_score=compliance,
            control_signal=max(self._batch_signal(batch), _clamp(len(missing) / 5.0)),
            description=(
                f"Boundary/consent owner published granted={len(granted)} missing={len(missing)} denied={len(denied)} "
                f"autonomy_risk={autonomy_risk:.2f} consent_clarity={consent_clarity:.2f} "
                f"overreach_risk={overreach_risk:.2f}."
            ),
            autonomy_risk=autonomy_risk,
            consent_clarity=consent_clarity,
            professional_scope_pressure=professional_scope_pressure,
            overreach_risk=overreach_risk,
        )


SEMANTIC_MODULE_TYPES = (
    PlanIntentModule,
    CommitmentModule,
    OpenLoopModule,
    UserModelModule,
    ExecutionResultModule,
    BeliefAssumptionModule,
    RelationshipStateModule,
    GoalValueModule,
    BoundaryConsentModule,
)


def build_semantic_modules(
    *,
    store: SemanticStateStore,
    proposal_runtime: SemanticProposalRuntime | None,
    user_input: str | None,
    turn_index: int,
    level_for: Any,
) -> tuple[SemanticOwnerModule, ...]:
    return tuple(
        module_type(
            store=store,
            proposal_runtime=proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
            wiring_level=level_for(module_type.slot_name, WiringLevel.ACTIVE),
        )
        for module_type in SEMANTIC_MODULE_TYPES
    )


def semantic_snapshot_counts(snapshots: Mapping[str, Snapshot[Any]]) -> tuple[tuple[str, int], ...]:
    counts: list[tuple[str, int]] = []
    for slot in SEMANTIC_OWNER_SLOTS:
        snapshot = snapshots.get(slot)
        value = snapshot.value if snapshot is not None else None
        if isinstance(value, PlanIntentSnapshot):
            counts.append((slot, len(value.candidate_plans) + len(value.deferred_intents)))
        elif isinstance(value, CommitmentSnapshot):
            counts.append((slot, len(value.active_commitments)))
        elif isinstance(value, OpenLoopSnapshot):
            counts.append((slot, len(value.unresolved_loops)))
        elif isinstance(value, UserModelSnapshot):
            counts.append((slot, len(value.stable_preferences)))
        elif isinstance(value, ExecutionResultSnapshot):
            counts.append((slot, len(value.attempted_actions)))
        elif isinstance(value, BeliefAssumptionSnapshot):
            counts.append((slot, len(value.assumptions)))
        elif isinstance(value, RelationshipStateSnapshot):
            counts.append((slot, len(value.rapport_signals) + len(value.relational_tensions)))
        elif isinstance(value, GoalValueSnapshot):
            counts.append((slot, len(value.explicit_goals)))
        elif isinstance(value, BoundaryConsentSnapshot):
            counts.append((slot, len(value.granted_consents) + len(value.missing_consents)))
    return tuple(counts)


def apply_semantic_writeback_result(
    *,
    store: SemanticStateStore,
    proposals: tuple[SemanticProposal, ...],
    turn_index: int,
) -> tuple[str, ...]:
    operations: list[str] = []
    for slot in SEMANTIC_OWNER_SLOTS:
        slot_proposals = tuple(proposal for proposal in proposals if proposal.target_slot == slot)
        if not slot_proposals:
            continue
        store.apply(slot=slot, proposals=slot_proposals, turn_index=turn_index)
        operations.append(f"semantic-state:{slot}:{len(slot_proposals)}")
    return tuple(operations)


def clone_semantic_store(source: SemanticStateStore) -> SemanticStateStore:
    target = SemanticStateStore()
    for slot in SEMANTIC_OWNER_SLOTS:
        target._records[slot] = source.records_for(slot)
        target._completed_refs[slot] = source.completed_refs_for(slot)
        target._revision_counts[slot] = source.revision_count_for(slot)
    return target
