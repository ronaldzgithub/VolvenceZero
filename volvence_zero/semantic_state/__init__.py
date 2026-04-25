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
class SemanticRecord:
    record_id: str
    summary: str
    detail: str
    confidence: float
    status: str
    source_turn: int
    evidence: str


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


@dataclass(frozen=True)
class CommitmentSnapshot:
    active_commitments: tuple[SemanticRecord, ...]
    honored_commitment_refs: tuple[str, ...]
    at_risk_commitments: tuple[SemanticRecord, ...]
    trust_obligation_count: int
    continuity_score: float
    control_signal: float
    description: str


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


@dataclass(frozen=True)
class ExecutionResultSnapshot:
    attempted_actions: tuple[SemanticRecord, ...]
    completed_actions: tuple[SemanticRecord, ...]
    failed_actions: tuple[SemanticRecord, ...]
    artifact_refs: tuple[str, ...]
    execution_grounding_score: float
    control_signal: float
    description: str


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


@dataclass(frozen=True)
class GoalValueSnapshot:
    explicit_goals: tuple[SemanticRecord, ...]
    value_priorities: tuple[SemanticRecord, ...]
    tradeoff_notes: tuple[SemanticRecord, ...]
    active_goal_id: str | None
    alignment_score: float
    control_signal: float
    description: str


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


class SemanticStateStore:
    def __init__(self) -> None:
        self._records: dict[str, tuple[SemanticRecord, ...]] = {slot: () for slot in SEMANTIC_OWNER_SLOTS}
        self._completed_refs: dict[str, tuple[str, ...]] = {slot: () for slot in SEMANTIC_OWNER_SLOTS}
        self._revision_counts: dict[str, int] = {slot: 0 for slot in SEMANTIC_OWNER_SLOTS}

    def apply(self, *, slot: str, proposals: tuple[SemanticProposal, ...], turn_index: int) -> tuple[SemanticRecord, ...]:
        existing = list(self._records[slot])
        completed_refs = list(self._completed_refs[slot])
        revision_count = self._revision_counts[slot]
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
                )
            )
        self._records[slot] = tuple(existing[-12:])
        self._completed_refs[slot] = tuple(completed_refs[-12:])
        self._revision_counts[slot] = revision_count
        return self._records[slot]

    def records_for(self, slot: str) -> tuple[SemanticRecord, ...]:
        return self._records[slot]

    def completed_refs_for(self, slot: str) -> tuple[str, ...]:
        return self._completed_refs[slot]

    def revision_count_for(self, slot: str) -> int:
        return self._revision_counts[slot]


def _records_with_status(records: tuple[SemanticRecord, ...], *statuses: str) -> tuple[SemanticRecord, ...]:
    allowed = set(statuses)
    return tuple(record for record in records if record.status in allowed)


class SemanticOwnerModule(RuntimeModule[SemanticSnapshotValue]):
    slot_name: ClassVar[str]
    owner: ClassVar[str]
    value_type: ClassVar[type[Any]]
    dependencies = ("substrate", "memory")
    default_wiring_level = WiringLevel.ACTIVE

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
        records = self._store.apply(
            slot=self.slot_name,
            proposals=batch.proposals,
            turn_index=self._turn_index,
        )
        value = self._build_snapshot(records=records, batch=batch)
        self._last_snapshot = value
        return self.publish(value)

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
            description=f"Plan/intent owner published {len(records)} records; active={latest.record_id if latest else 'none'}.",
        )


class CommitmentModule(SemanticOwnerModule):
    slot_name = "commitment"
    owner = "CommitmentModule"
    value_type = CommitmentSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> CommitmentSnapshot:
        active = _records_with_status(records, "active")
        at_risk = _records_with_status(records, "blocked")
        return CommitmentSnapshot(
            active_commitments=active,
            honored_commitment_refs=self._store.completed_refs_for(self.slot_name),
            at_risk_commitments=at_risk,
            trust_obligation_count=len(active),
            continuity_score=self._mean_confidence(active),
            control_signal=self._batch_signal(batch),
            description=f"Commitment owner published active={len(active)} at_risk={len(at_risk)}.",
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
        return UserModelSnapshot(
            stable_preferences=records[-4:],
            working_style_hints=records[-4:],
            sensitive_boundaries=_records_with_status(records, "blocked"),
            durable_goals=(),
            stability_score=self._mean_confidence(records),
            control_signal=self._batch_signal(batch),
            description=f"User-model owner published {len(records)} profile records.",
        )


class ExecutionResultModule(SemanticOwnerModule):
    slot_name = "execution_result"
    owner = "ExecutionResultModule"
    value_type = ExecutionResultSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> ExecutionResultSnapshot:
        completed = _records_with_status(records, "completed")
        failed = _records_with_status(records, "blocked")
        return ExecutionResultSnapshot(
            attempted_actions=records,
            completed_actions=completed,
            failed_actions=failed,
            artifact_refs=tuple(record.record_id for record in completed),
            execution_grounding_score=self._mean_confidence(completed or records),
            control_signal=self._batch_signal(batch),
            description=f"Execution-result owner published attempted={len(records)} completed={len(completed)} failed={len(failed)}.",
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
        return RelationshipStateSnapshot(
            trust_level=_clamp(0.45 + confidence * 0.35 - len(tensions) * 0.05),
            continuity_level=_clamp(0.35 + len(records) / 10.0),
            repair_pressure=_clamp(len(tensions) / 4.0),
            rapport_signals=records[-4:],
            relational_tensions=tensions,
            control_signal=self._batch_signal(batch),
            description=f"Relationship-state owner published continuity={len(records)} tensions={len(tensions)}.",
        )


class GoalValueModule(SemanticOwnerModule):
    slot_name = "goal_value"
    owner = "GoalValueModule"
    value_type = GoalValueSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> GoalValueSnapshot:
        latest = self._latest_active(records)
        return GoalValueSnapshot(
            explicit_goals=records,
            value_priorities=records[-4:],
            tradeoff_notes=_records_with_status(records, "deferred", "blocked"),
            active_goal_id=latest.record_id if latest else None,
            alignment_score=self._mean_confidence(records),
            control_signal=self._batch_signal(batch),
            description=f"Goal/value owner published goals={len(records)} active={latest.record_id if latest else 'none'}.",
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
        return BoundaryConsentSnapshot(
            granted_consents=granted,
            missing_consents=missing,
            denied_boundaries=denied,
            memory_consent="unknown" if not granted else "granted",
            external_action_consent="unknown" if missing else "not-required",
            compliance_score=compliance,
            control_signal=max(self._batch_signal(batch), _clamp(len(missing) / 5.0)),
            description=f"Boundary/consent owner published granted={len(granted)} missing={len(missing)} denied={len(denied)}.",
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
