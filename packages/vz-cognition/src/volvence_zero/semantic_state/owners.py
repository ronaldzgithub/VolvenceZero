"""Semantic-state owner modules.

Nine concrete :class:`RuntimeModule` owners (``plan_intent``,
``commitment``, ``open_loop``, ``user_model``, ``execution_result``,
``belief_assumption``, ``relationship_state``, ``goal_value``,
``boundary_consent``) plus the shared :class:`SemanticOwnerModule`
base, the factory :func:`build_semantic_modules`, the readout
:func:`semantic_snapshot_counts`, and the writeback applier
:func:`apply_semantic_writeback_result`.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import SubstrateSnapshot

from volvence_zero.semantic_state.contracts import (
    SEMANTIC_OWNER_SLOTS,
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
    SemanticProposal,
    SemanticProposalOperation,
    SemanticRecord,
    SemanticSnapshotValue,
    UserModelSnapshot,
    semantic_control_signal,
    semantic_snapshot_description,
)
from volvence_zero.semantic_state.proposal_runtime import (
    SemanticProposalRuntime,
)
from volvence_zero.semantic_state.store import SemanticStateStore


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


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
        due_followup = sum(
            1
            for record in active
            if policy_map.get(record.record_id, FollowupPolicy.GENTLE_CHECKIN)
            is FollowupPolicy.GENTLE_CHECKIN
        )
        stalled_commitments = len(at_risk) + outcome_stalled + outcome_followup_none
        recent_completions = outcome_completed
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
                f"no_response={outcome_followup_none}] "
                f"continuity[due_followup={due_followup} "
                f"stalled={stalled_commitments} recent_completion={recent_completions}]."
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
            due_followup_count=due_followup,
            stalled_commitment_count=stalled_commitments,
            recent_completion_count=recent_completions,
        )


class OpenLoopModule(SemanticOwnerModule):
    slot_name = "open_loop"
    owner = "OpenLoopModule"
    value_type = OpenLoopSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> OpenLoopSnapshot:
        unresolved = _records_with_status(records, "active", "deferred")
        confirmations = tuple(record for record in unresolved if record.confidence < 0.55)
        highest = unresolved[-1].record_id if unresolved else None
        oldest_open_turn = min((record.source_turn for record in unresolved), default=None)
        stale_loops = sum(
            1
            for record in unresolved
            if self._turn_index - record.source_turn >= 3
        )
        confirmation_debt = len(confirmations)
        closure_readiness = _clamp(
            len(self._store.completed_refs_for(self.slot_name)) / max(len(records), 1)
        )
        return OpenLoopSnapshot(
            unresolved_loops=unresolved,
            pending_confirmations=confirmations,
            closure_refs=self._store.completed_refs_for(self.slot_name),
            highest_priority_loop_id=highest,
            closure_pressure=_clamp(len(unresolved) / 5.0),
            control_signal=max(self._batch_signal(batch), _clamp(len(confirmations) / 5.0)),
            description=(
                f"Open-loop owner published unresolved={len(unresolved)} confirmations={len(confirmations)} "
                f"oldest_turn={oldest_open_turn if oldest_open_turn is not None else 'none'} "
                f"stale={stale_loops} closure_readiness={closure_readiness:.2f}."
            ),
            oldest_open_turn=oldest_open_turn,
            stale_loop_count=stale_loops,
            confirmation_debt_count=confirmation_debt,
            closure_readiness=closure_readiness,
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
        repair_refs = self._store.completed_refs_for(self.slot_name)
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
        recent_repair_count = len(repair_refs)
        unresolved_tension_count = len(tensions)
        attunement_trend = _clamp(0.5 + trust_delta * 0.40 - attunement_gap * 0.25)
        trust_recovery_signal = _clamp(
            min(recent_repair_count / 3.0, 1.0) * 0.45
            + trust_level * 0.35
            + (1.0 - repair_need) * 0.20
        )
        relationship_continuity_score = _clamp(
            continuity_level * 0.45
            + trust_level * 0.35
            + trust_recovery_signal * 0.20
            - min(unresolved_tension_count / 4.0, 1.0) * 0.15
        )
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
                f"stabilization_need={stabilization_need:.2f} "
                f"repair_count={recent_repair_count} continuity_score={relationship_continuity_score:.2f}."
            ),
            emotional_load=emotional_load,
            repair_need=repair_need,
            trust_delta=trust_delta,
            attunement_gap=attunement_gap,
            stabilization_need=stabilization_need,
            recent_repair_count=recent_repair_count,
            unresolved_tension_count=unresolved_tension_count,
            attunement_trend=attunement_trend,
            trust_recovery_signal=trust_recovery_signal,
            relationship_continuity_score=relationship_continuity_score,
        )


class GoalValueModule(SemanticOwnerModule):
    slot_name = "goal_value"
    owner = "GoalValueModule"
    value_type = GoalValueSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> GoalValueSnapshot:
        latest = self._latest_active(records)
        tradeoffs = _records_with_status(records, "deferred", "blocked")
        active_goals = _records_with_status(records, "active")
        deferred_goals = _records_with_status(records, "deferred")
        conflicted_goals = _records_with_status(records, "blocked")
        resolved_goal_refs = self._store.completed_refs_for(self.slot_name)
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
        goal_continuity_score = _clamp(
            alignment_score * 0.55
            + min(len(active_goals) / 3.0, 1.0) * 0.20
            + min(len(resolved_goal_refs) / 3.0, 1.0) * 0.15
            - min(len(conflicted_goals) / 3.0, 1.0) * 0.10
        )
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
                f"reversibility_need={reversibility_need:.2f} shift={goal_shift_pressure:.2f} "
                f"lifecycle[active={len(active_goals)} deferred={len(deferred_goals)} "
                f"conflicted={len(conflicted_goals)} resolved={len(resolved_goal_refs)}]."
            ),
            value_conflict=value_conflict,
            decision_readiness=decision_readiness,
            active_tradeoff_count=active_tradeoff_count,
            reversibility_need=reversibility_need,
            goal_shift_pressure=goal_shift_pressure,
            active_goal_count=len(active_goals),
            deferred_goal_count=len(deferred_goals),
            conflicted_goal_count=len(conflicted_goals),
            resolved_goal_refs=resolved_goal_refs,
            goal_continuity_score=goal_continuity_score,
        )


class BoundaryConsentModule(SemanticOwnerModule):
    slot_name = "boundary_consent"
    owner = "BoundaryConsentModule"
    value_type = BoundaryConsentSnapshot

    def _build_snapshot(self, *, records: tuple[SemanticRecord, ...], batch: SemanticProposalBatch) -> BoundaryConsentSnapshot:
        granted = _records_with_status(records, "active", "completed")
        missing = tuple(record for record in records if record.confidence < 0.55 and record.status not in {"blocked", "closed"})
        denied = _records_with_status(records, "blocked")
        revoked = _records_with_status(records, "closed")
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
        active_scope_count = len(granted)
        denial_count = len(denied)
        revocation_count = len(revoked)
        external_action_blocked = denial_count > 0
        memory_scope_status = (
            "denied"
            if denial_count > 0
            else "granted" if active_scope_count > 0 else "unknown"
        )
        return BoundaryConsentSnapshot(
            granted_consents=granted,
            missing_consents=missing,
            denied_boundaries=denied,
            memory_consent=memory_scope_status,
            external_action_consent="denied" if external_action_blocked else ("unknown" if missing else "not-required"),
            compliance_score=compliance,
            control_signal=max(self._batch_signal(batch), _clamp(len(missing) / 5.0)),
            description=(
                f"Boundary/consent owner published granted={len(granted)} missing={len(missing)} denied={len(denied)} revoked={len(revoked)} "
                f"autonomy_risk={autonomy_risk:.2f} consent_clarity={consent_clarity:.2f} "
                f"overreach_risk={overreach_risk:.2f} memory_scope={memory_scope_status} "
                f"external_blocked={int(external_action_blocked)}."
            ),
            autonomy_risk=autonomy_risk,
            consent_clarity=consent_clarity,
            professional_scope_pressure=professional_scope_pressure,
            overreach_risk=overreach_risk,
            active_scope_count=active_scope_count,
            denial_count=denial_count,
            revocation_count=revocation_count,
            external_action_blocked=external_action_blocked,
            memory_scope_status=memory_scope_status,
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
