"""Semantic state contract surface (R8 / R11).

SSOT split (oss-relationship-representation-standard.md, Phase A1):

* The *representation* — slot registry, outcome enums, ``SemanticRecord``,
  lifecycle entries, the nine snapshot value dataclasses, funnel-stage
  vocabulary, and the readout helpers — lives in
  ``companion_standard.semantic_state`` (the public Relationship
  Representation Standard) and is re-exported here so every existing
  ``volvence_zero.semantic_state`` import keeps working.
* The *write protocol* — ``SemanticProposal*`` / ``*SemanticEvent`` types —
  and the prompt / JSON-schema loaders stay private in this module: how
  state is mutated is runtime mechanism, not representation.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``. External consumers continue to import
from ``volvence_zero.semantic_state`` and get these names via the
package ``__init__`` re-export.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import files

# Re-exported representation types (public standard SSOT). Keep this list in
# sync with companion_standard.semantic_state.__all__-equivalent surface;
# tests/contracts/test_semantic_state_owners.py and the conformance kit
# exercise both import paths.
from companion_standard.semantic_state import (  # noqa: F401
    ALLOWED_FUNNEL_STAGES,
    FUNNEL_STAGE_CONVERTING,
    FUNNEL_STAGE_DISCOVERY,
    FUNNEL_STAGE_NURTURING,
    FUNNEL_STAGE_PROSPECTING,
    FUNNEL_STAGE_RECOMMENDING,
    FUNNEL_STAGE_REPURCHASING,
    FUNNEL_STAGE_UNKNOWN,
    SELF_SEMANTIC_OWNER_SLOTS,
    SEMANTIC_OWNER_SLOTS,
    WORLD_SEMANTIC_OWNER_SLOTS,
    AdvocacyState,
    AlignmentState,
    BeliefAssumptionSnapshot,
    BoundaryConsentSnapshot,
    CommitmentLifecycleEntry,
    CommitmentOutcomeKind,
    CommitmentSnapshot,
    ExecutionResultLifecycleEntry,
    ExecutionResultOutcome,
    ExecutionResultSnapshot,
    FollowupPolicy,
    GoalValueSnapshot,
    OpenLoopSnapshot,
    PlanIntentLifecycleEntry,
    PlanIntentOutcome,
    PlanIntentSnapshot,
    RelationshipStateSnapshot,
    SemanticRecord,
    SemanticSnapshotValue,
    UserModelSnapshot,
    _clamp,
    semantic_control_signal,
    semantic_snapshot_description,
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


@dataclass(frozen=True)
class GenericSemanticEvent:
    """Pre-formed typed semantic event from a reviewed external adapter.

    This is for sources that already know the target owner slot and
    operation (for example a human-reviewed character chapter ledger).
    It is still only a proposal source; ``SemanticStateStore`` remains
    the single writer.
    """

    event_id: str
    target_slot: str
    operation: SemanticProposalOperation
    summary: str
    detail: str
    confidence: float
    evidence: str
    control_signal: float = 0.0
    requires_confirmation: bool = False


ExternalSemanticEvent = (
    ToolResultSemanticEvent
    | ProfileSemanticEvent
    | TaskSemanticEvent
    | ReviewedKnowledgeSemanticEvent
    | GenericSemanticEvent
)


@dataclass(frozen=True)
class ExternalSemanticEventBatch:
    events: tuple[ExternalSemanticEvent, ...]
    source: str
    description: str
