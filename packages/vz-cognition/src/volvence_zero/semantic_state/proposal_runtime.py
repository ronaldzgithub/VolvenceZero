"""Semantic proposal runtime: ABC, adapters, and event builders.

This module hosts:

* :class:`SemanticProposalRuntime` — the ABC every structured-proposal
  source implements.
* :class:`NoOpSemanticProposalRuntime` — the default compatibility
  fallback used when no LLM / adapter runtime is wired.
* :class:`SemanticEventAdapter` + four concrete adapters that turn
  external environment events (tool results, profile batches, task
  events, reviewed knowledge) into :class:`SemanticProposal` records.
* :class:`AdapterSemanticProposalRuntime` — a composite runtime that
  fans a proposal request out to the adapter set.
* ``semantic_events_from_*`` helper builders that external lifeform
  hosts use to construct the concrete event dataclasses.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from volvence_zero.memory import MemorySnapshot
from volvence_zero.substrate import SubstrateSnapshot

from volvence_zero.semantic_state.contracts import (
    ExternalSemanticEvent,
    ExternalSemanticEventBatch,
    ProfileSemanticEvent,
    ReviewedKnowledgeSemanticEvent,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticSnapshotValue,
    TaskSemanticEvent,
    ToolResultSemanticEvent,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


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
