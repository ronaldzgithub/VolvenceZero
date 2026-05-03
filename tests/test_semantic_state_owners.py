from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    SEMANTIC_OWNER_SLOTS,
    AdapterSemanticProposalRuntime,
    CommitmentSnapshot,
    GoalValueSnapshot,
    NoOpSemanticProposalRuntime,
    OpenLoopSnapshot,
    PlanIntentSnapshot,
    RelationshipStateSnapshot,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
    SemanticStateStore,
    UserModelSnapshot,
    clone_semantic_store,
    semantic_events_from_profile,
    semantic_events_from_reviewed_knowledge,
    semantic_events_from_task_event,
    semantic_events_from_tool_result,
)
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


class DeterministicPlanRuntime(SemanticProposalRuntime):
    runtime_id = "deterministic-plan-test"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        operation = (
            SemanticProposalOperation.REVISE
            if target_slot == "plan_intent" and turn_index > 1
            else SemanticProposalOperation.OBSERVE
        )
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=f"{target_slot}:proposal:{turn_index}",
                    target_slot=target_slot,
                    operation=operation,
                    summary=f"{target_slot}:semantic-update",
                    detail=user_input or "",
                    confidence=0.82,
                    evidence=user_input or "",
                    control_signal=0.44 if target_slot == "plan_intent" else 0.12,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"test proposal for {target_slot}",
        )


class DeniedConsentRuntime(SemanticProposalRuntime):
    runtime_id = "denied-consent-test"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        operation = (
            SemanticProposalOperation.BLOCK
            if target_slot == "boundary_consent"
            else SemanticProposalOperation.OBSERVE
        )
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=f"{target_slot}:consent:{turn_index}",
                    target_slot=target_slot,
                    operation=operation,
                    summary=f"{target_slot}:consent-state",
                    detail=user_input or "",
                    confidence=0.80,
                    evidence=user_input or "",
                    control_signal=0.50 if target_slot == "boundary_consent" else 0.05,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"consent proposal for {target_slot}",
        )


class EmotionalDecisionRuntime(SemanticProposalRuntime):
    runtime_id = "emotional-decision-test"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        proposal: SemanticProposal | None = None
        if target_slot == "relationship_state":
            proposal = SemanticProposal(
                proposal_id=f"{target_slot}:emotional-load:{turn_index}",
                target_slot=target_slot,
                operation=SemanticProposalOperation.OBSERVE,
                summary="emotional-load",
                detail=user_input or "",
                confidence=0.72,
                evidence="typed emotional load evidence",
                control_signal=0.82,
            )
        elif target_slot == "goal_value":
            proposal = SemanticProposal(
                proposal_id=f"{target_slot}:tradeoff:{turn_index}",
                target_slot=target_slot,
                operation=SemanticProposalOperation.DEFER,
                summary="values-before-choice",
                detail="choice needs value clarification before commitment",
                confidence=0.68,
                evidence="typed value conflict evidence",
                control_signal=0.74,
            )
        elif target_slot == "boundary_consent":
            proposal = SemanticProposal(
                proposal_id=f"{target_slot}:autonomy:{turn_index}",
                target_slot=target_slot,
                operation=SemanticProposalOperation.OBSERVE,
                summary="do-not-decide-for-me",
                detail="user wants support without delegated decision",
                confidence=0.50,
                evidence="typed autonomy boundary evidence",
                control_signal=0.62,
            )
        elif target_slot == "user_model":
            proposal = SemanticProposal(
                proposal_id=f"{target_slot}:durable-goal:{turn_index}",
                target_slot=target_slot,
                operation=SemanticProposalOperation.OBSERVE,
                summary="steady-before-analysis",
                detail="prefers support-first pacing before decision analysis",
                confidence=0.76,
                evidence="typed user pacing evidence",
                control_signal=0.70,
            )
        proposals = () if proposal is None else (proposal,)
        return SemanticProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"emotional decision proposal for {target_slot}",
        )


def _adapter() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="semantic-state-test-model",
        feature_surface=(FeatureSignal(name="semantic_state_test", values=(0.4,), source="test"),),
    )


def test_semantic_owner_snapshots_are_frozen_and_active() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Help me plan this carefully.",
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=DeterministicPlanRuntime(),
            session_id="semantic-session",
            wave_id="semantic-wave",
            turn_index=1,
        )
    )

    for slot_name in SEMANTIC_OWNER_SLOTS:
        assert slot_name in result.active_snapshots

    plan_snapshot = result.active_snapshots["plan_intent"].value
    assert isinstance(plan_snapshot, PlanIntentSnapshot)
    assert plan_snapshot.active_plan_id == "plan_intent:proposal:1"
    with pytest.raises(FrozenInstanceError):
        plan_snapshot.active_goal = "mutated"  # type: ignore[misc]


def test_semantic_spine_publishes_cognitive_loop_readiness_evidence() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Track the goal, boundary, commitment, relationship, and result.",
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=DeterministicPlanRuntime(),
            session_id="semantic-spine",
            wave_id="semantic-spine-wave",
            turn_index=1,
        )
    )

    evaluation = result.active_snapshots["evaluation"].value
    metrics = {score.metric_name: score for score in evaluation.turn_scores}

    assert metrics["semantic_spine_coverage"].value == 1.0
    assert metrics["cognitive_loop_readiness"].value > 0.0
    assert "semantic owner snapshots" in metrics["cognitive_loop_readiness"].evidence


def test_semantic_proposals_persist_across_turns_and_revise_plan() -> None:
    store = SemanticStateStore()
    runtime = DeterministicPlanRuntime()

    first = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Build a launch plan.",
            semantic_state_store=store,
            semantic_proposal_runtime=runtime,
            session_id="semantic-session",
            wave_id="wave-1",
            turn_index=1,
        )
    )
    second = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Revise the second step with a rollout detail.",
            semantic_state_store=store,
            semantic_proposal_runtime=runtime,
            upstream_snapshots=first.active_snapshots,
            session_id="semantic-session",
            wave_id="wave-2",
            turn_index=2,
        )
    )

    plan_snapshot = second.active_snapshots["plan_intent"].value
    assert isinstance(plan_snapshot, PlanIntentSnapshot)
    assert plan_snapshot.plan_revision_count == 1
    assert plan_snapshot.active_plan_id == "plan_intent:proposal:2"
    response_assembly = second.active_snapshots["response_assembly"].value
    assert dict(response_assembly.semantic_record_counts)["plan_intent"] >= 1
    assert response_assembly.semantic_control_signal > 0.0


def test_clone_semantic_store_preserves_lifecycle_policy_and_outcomes() -> None:
    store = SemanticStateStore()
    proposals = (
        SemanticProposal(
            proposal_id="commitment:complete:1",
            target_slot="commitment",
            operation=SemanticProposalOperation.COMPLETE,
            summary="follow-up completed",
            detail="The commitment reached a typed completion outcome.",
            confidence=0.90,
            evidence="typed completion evidence",
            control_signal=0.40,
        ),
    )
    store.apply(slot="commitment", proposals=proposals, turn_index=3)

    cloned = clone_semantic_store(store)

    assert cloned.records_for("commitment") == store.records_for("commitment")
    assert cloned.completed_refs_for("commitment") == store.completed_refs_for("commitment")
    assert cloned.lifecycle_for("commitment") == store.lifecycle_for("commitment")
    assert cloned.followup_policy_for("commitment") == store.followup_policy_for("commitment")
    assert cloned.outcome_for("commitment") == store.outcome_for("commitment")


def test_commitment_owner_publishes_continuity_readouts() -> None:
    store = SemanticStateStore()
    store.apply(
        slot="commitment",
        proposals=(
            SemanticProposal(
                proposal_id="commitment:active:1",
                target_slot="commitment",
                operation=SemanticProposalOperation.CREATE,
                summary="check launch status",
                detail="Follow up on launch status.",
                confidence=0.90,
                evidence="typed active commitment evidence",
                control_signal=0.30,
            ),
            SemanticProposal(
                proposal_id="commitment:done:1",
                target_slot="commitment",
                operation=SemanticProposalOperation.COMPLETE,
                summary="publish checklist",
                detail="Checklist publication completed.",
                confidence=0.92,
                evidence="typed completion evidence",
                control_signal=0.30,
            ),
            SemanticProposal(
                proposal_id="commitment:stalled:1",
                target_slot="commitment",
                operation=SemanticProposalOperation.DEFER,
                summary="wait for review",
                detail="Review follow-up is deferred.",
                confidence=0.84,
                evidence="typed defer evidence",
                control_signal=0.40,
            ),
        ),
        turn_index=1,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=store,
            session_id="commitment-continuity",
            wave_id="commitment-continuity-wave",
            turn_index=2,
        )
    )

    commitment = result.active_snapshots["commitment"].value
    assert isinstance(commitment, CommitmentSnapshot)
    assert commitment.due_followup_count >= 1
    assert commitment.recent_completion_count == 1
    assert commitment.stalled_commitment_count >= 1


def test_open_loop_owner_publishes_continuity_readouts() -> None:
    store = SemanticStateStore()
    store.apply(
        slot="open_loop",
        proposals=(
            SemanticProposal(
                proposal_id="open-loop:old:1",
                target_slot="open_loop",
                operation=SemanticProposalOperation.CREATE,
                summary="old unanswered loop",
                detail="An unresolved follow-up from an earlier turn.",
                confidence=0.50,
                evidence="typed open loop evidence",
                control_signal=0.35,
                requires_confirmation=True,
            ),
            SemanticProposal(
                proposal_id="open-loop:closed:1",
                target_slot="open_loop",
                operation=SemanticProposalOperation.CLOSE,
                summary="closed loop",
                detail="A loop that has been closed.",
                confidence=0.88,
                evidence="typed closure evidence",
                control_signal=0.10,
            ),
        ),
        turn_index=1,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=store,
            session_id="open-loop-continuity",
            wave_id="open-loop-continuity-wave",
            turn_index=5,
        )
    )

    open_loop = result.active_snapshots["open_loop"].value
    assert isinstance(open_loop, OpenLoopSnapshot)
    assert open_loop.oldest_open_turn == 1
    assert open_loop.stale_loop_count >= 1
    assert open_loop.confirmation_debt_count >= 1
    assert open_loop.closure_readiness > 0.0


def test_noop_semantic_runtime_does_not_keyword_drive_operations() -> None:
    batch = NoOpSemanticProposalRuntime().propose(
        target_slot="plan_intent",
        user_input="预算 deadline 灰度 launch",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=3,
    )

    assert len(batch.proposals) == 1
    assert batch.proposals[0].operation is SemanticProposalOperation.OBSERVE
    assert batch.proposals[0].summary == "latest-turn-observed"


def test_semantic_owner_kill_switch_removes_active_slot() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(kill_switches=frozenset({"plan_intent"})),
            substrate_adapter=_adapter(),
            user_input="Plan something.",
            semantic_state_store=SemanticStateStore(),
            session_id="semantic-kill",
            wave_id="wave-kill",
        )
    )

    assert "plan_intent" not in result.active_snapshots
    assert "plan_intent" in result.acceptance_report.disabled_slots
    assert result.acceptance_report.passed is True


def test_session_post_request_carries_semantic_state_descriptions() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Keep this plan available for the next step.",
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=DeterministicPlanRuntime(),
            session_id="semantic-post",
            wave_id="wave-post",
            turn_index=1,
            apply_slow_writeback=False,
        )
    )

    assert result.session_post_writeback_request is not None
    assert result.session_post_writeback_request.semantic_state_descriptions
    assert any(
        "Plan/intent owner" in description
        for description in result.session_post_writeback_request.semantic_state_descriptions
    )


def test_boundary_consent_owner_tightens_boundary_policy() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Do not take external action without asking.",
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=DeniedConsentRuntime(),
            session_id="semantic-consent",
            wave_id="wave-consent",
            turn_index=1,
        )
    )

    boundary_consent = result.active_snapshots["boundary_consent"].value
    boundary_policy = result.active_snapshots["boundary_policy"].value

    assert boundary_consent.denied_boundaries
    assert boundary_consent.denial_count == 1
    assert boundary_consent.external_action_blocked is True
    assert boundary_consent.memory_scope_status == "denied"
    assert "consent-boundary-denied" in boundary_policy.trigger_reasons
    assert boundary_policy.active_decision.refer_out_required is True


def test_boundary_consent_owner_publishes_lifecycle_readouts() -> None:
    store = SemanticStateStore()
    store.apply(
        slot="boundary_consent",
        proposals=(
            SemanticProposal(
                proposal_id="boundary:grant:1",
                target_slot="boundary_consent",
                operation=SemanticProposalOperation.OBSERVE,
                summary="memory scope granted",
                detail="User grants memory scope for planning preferences.",
                confidence=0.86,
                evidence="typed consent grant evidence",
                control_signal=0.20,
            ),
            SemanticProposal(
                proposal_id="boundary:revoke:1",
                target_slot="boundary_consent",
                operation=SemanticProposalOperation.CLOSE,
                summary="old scope revoked",
                detail="User revokes an older consent scope.",
                confidence=0.88,
                evidence="typed revocation evidence",
                control_signal=0.30,
            ),
        ),
        turn_index=1,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=store,
            session_id="boundary-lifecycle",
            wave_id="boundary-lifecycle-wave",
            turn_index=2,
        )
    )

    boundary = result.active_snapshots["boundary_consent"].value
    assert isinstance(boundary, BoundaryConsentSnapshot)
    assert boundary.active_scope_count >= 1
    assert boundary.revocation_count == 1
    assert boundary.denial_count == 0
    assert boundary.external_action_blocked is False
    assert boundary.memory_scope_status == "granted"


def test_goal_value_owner_publishes_lifecycle_readouts() -> None:
    store = SemanticStateStore()
    store.apply(
        slot="goal_value",
        proposals=(
            SemanticProposal(
                proposal_id="goal:active:1",
                target_slot="goal_value",
                operation=SemanticProposalOperation.CREATE,
                summary="ship safely",
                detail="User wants a safe launch path.",
                confidence=0.90,
                evidence="typed goal evidence",
                control_signal=0.20,
            ),
            SemanticProposal(
                proposal_id="goal:defer:1",
                target_slot="goal_value",
                operation=SemanticProposalOperation.DEFER,
                summary="defer risky shortcut",
                detail="Shortcut is deferred pending value clarification.",
                confidence=0.82,
                evidence="typed tradeoff evidence",
                control_signal=0.55,
            ),
            SemanticProposal(
                proposal_id="goal:blocked:1",
                target_slot="goal_value",
                operation=SemanticProposalOperation.BLOCK,
                summary="conflicting priority",
                detail="A priority conflicts with the active goal.",
                confidence=0.78,
                evidence="typed conflict evidence",
                control_signal=0.60,
            ),
            SemanticProposal(
                proposal_id="goal:done:1",
                target_slot="goal_value",
                operation=SemanticProposalOperation.COMPLETE,
                summary="resolved prior goal",
                detail="A prior goal has been resolved.",
                confidence=0.84,
                evidence="typed resolution evidence",
                control_signal=0.10,
            ),
        ),
        turn_index=1,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=store,
            session_id="goal-lifecycle",
            wave_id="goal-lifecycle-wave",
            turn_index=2,
        )
    )

    goal = result.active_snapshots["goal_value"].value
    assert isinstance(goal, GoalValueSnapshot)
    assert goal.active_goal_count >= 1
    assert goal.deferred_goal_count == 1
    assert goal.conflicted_goal_count == 1
    assert goal.resolved_goal_refs
    assert goal.goal_continuity_score > 0.0


def test_relationship_state_owner_publishes_structured_continuity_readouts() -> None:
    store = SemanticStateStore()
    store.apply(
        slot="relationship_state",
        proposals=(
            SemanticProposal(
                proposal_id="relationship:rapport:1",
                target_slot="relationship_state",
                operation=SemanticProposalOperation.OBSERVE,
                summary="calm collaboration",
                detail="User and assistant are collaborating calmly.",
                confidence=0.84,
                evidence="typed rapport evidence",
                control_signal=0.20,
            ),
            SemanticProposal(
                proposal_id="relationship:tension:1",
                target_slot="relationship_state",
                operation=SemanticProposalOperation.BLOCK,
                summary="unresolved friction",
                detail="A relational tension remains unresolved.",
                confidence=0.70,
                evidence="typed tension evidence",
                control_signal=0.75,
            ),
            SemanticProposal(
                proposal_id="relationship:repair:1",
                target_slot="relationship_state",
                operation=SemanticProposalOperation.CLOSE,
                summary="repair completed",
                detail="A previous misunderstanding was repaired.",
                confidence=0.88,
                evidence="typed repair evidence",
                control_signal=0.20,
            ),
        ),
        turn_index=1,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=store,
            session_id="relationship-continuity",
            wave_id="relationship-continuity-wave",
            turn_index=2,
        )
    )

    relationship = result.active_snapshots["relationship_state"].value
    assert isinstance(relationship, RelationshipStateSnapshot)
    assert relationship.recent_repair_count == 1
    assert relationship.unresolved_tension_count == 1
    assert relationship.attunement_trend >= 0.0
    assert relationship.trust_recovery_signal > 0.0
    assert relationship.relationship_continuity_score > 0.0


def test_tool_result_adapter_updates_execution_and_open_loop() -> None:
    store = SemanticStateStore()
    events = semantic_events_from_tool_result(
        event_id="tool:event:1",
        tool_name="deploy",
        action_id="deploy:123",
        status="failed",
        summary="Deployment failed",
        detail="The deployment command returned a non-zero exit.",
        plan_ref="launch-plan",
    )
    runtime = AdapterSemanticProposalRuntime(
        base_runtime=NoOpSemanticProposalRuntime(),
        external_events=events.events,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="Continue the launch work.",
            semantic_state_store=store,
            semantic_proposal_runtime=runtime,
            session_id="semantic-tool",
            wave_id="wave-tool",
            turn_index=1,
        )
    )

    execution_result = result.active_snapshots["execution_result"].value
    open_loop = result.active_snapshots["open_loop"].value
    plan_intent = result.active_snapshots["plan_intent"].value

    assert execution_result.failed_actions
    assert open_loop.unresolved_loops
    assert plan_intent.plan_revision_count == 1


def test_profile_adapter_updates_user_model_goal_and_consent() -> None:
    events = semantic_events_from_profile(
        event_id="profile:event:1",
        source="product-profile",
        preferences=("prefers concise plans",),
        goals=("ship safely",),
        consent_grants=("remember planning preference",),
        consent_denials=("external action without confirmation",),
        relationship_note="prefers calm collaboration",
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=AdapterSemanticProposalRuntime(external_events=events.events),
            session_id="semantic-profile",
            wave_id="wave-profile",
            turn_index=1,
        )
    )

    assert result.active_snapshots["user_model"].value.stable_preferences
    assert result.active_snapshots["goal_value"].value.explicit_goals
    assert result.active_snapshots["relationship_state"].value.rapport_signals
    assert result.active_snapshots["boundary_consent"].value.denied_boundaries


def test_semantic_owners_publish_emotional_decision_readouts() -> None:
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            user_input="I am overwhelmed and need help deciding without being pushed.",
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=EmotionalDecisionRuntime(),
            session_id="semantic-emotional-decision",
            wave_id="wave-emotional-decision",
            turn_index=1,
        )
    )

    relationship = result.active_snapshots["relationship_state"].value
    goal = result.active_snapshots["goal_value"].value
    boundary = result.active_snapshots["boundary_consent"].value
    user_model = result.active_snapshots["user_model"].value
    response_assembly = result.active_snapshots["response_assembly"].value
    evaluation_metrics = {
        score.metric_name: score.value
        for score in result.active_snapshots["evaluation"].value.turn_scores
    }

    assert isinstance(relationship, RelationshipStateSnapshot)
    assert isinstance(goal, GoalValueSnapshot)
    assert isinstance(boundary, BoundaryConsentSnapshot)
    assert isinstance(user_model, UserModelSnapshot)
    assert relationship.emotional_load > 0.35
    assert relationship.stabilization_need > 0.30
    assert goal.value_conflict > 0.30
    assert goal.reversibility_need > 0.20
    assert boundary.autonomy_risk > 0.20
    assert boundary.consent_clarity < 1.0
    assert user_model.preferred_support_pacing == "support-first"
    assert user_model.decision_style == "values-first"
    assert user_model.durable_goals
    assert response_assembly.support_before_decision_pressure > 0.30
    assert response_assembly.eta_action_family in {
        "clarify_values_then_options",
        "hold_boundary_while_supporting",
        "small_reversible_next_step",
        "stabilize_before_deciding",
    }
    assert evaluation_metrics["owner_emotional_load"] == relationship.emotional_load
    assert evaluation_metrics["owner_value_conflict"] == goal.value_conflict
    assert evaluation_metrics["owner_autonomy_risk"] == boundary.autonomy_risk


def test_task_event_adapter_updates_plan_commitment_and_execution() -> None:
    events = semantic_events_from_task_event(
        event_id="task:event:1",
        task_id="task:launch",
        status="completed",
        summary="Launch checklist completed",
        detail="All launch checklist items were marked done.",
        due_hint="today",
        commitment_ref="finish launch checklist",
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=AdapterSemanticProposalRuntime(external_events=events.events),
            session_id="semantic-task",
            wave_id="wave-task",
            turn_index=1,
        )
    )

    assert result.active_snapshots["plan_intent"].value.completed_plan_refs
    assert result.active_snapshots["commitment"].value.honored_commitment_refs
    assert result.active_snapshots["execution_result"].value.completed_actions


def test_reviewed_knowledge_adapter_updates_belief_without_domain_write() -> None:
    events = semantic_events_from_reviewed_knowledge(
        event_id="knowledge:event:1",
        knowledge_id="knowledge:external:1",
        summary="Reviewed rollout guidance",
        detail="The reviewed source recommends staged rollout.",
        source_label="reviewed-doc",
        confidence=0.88,
        relevance_hint="rollout safety",
        needs_followup=True,
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=_adapter(),
            semantic_state_store=SemanticStateStore(),
            semantic_proposal_runtime=AdapterSemanticProposalRuntime(external_events=events.events),
            session_id="semantic-knowledge",
            wave_id="wave-knowledge",
            turn_index=1,
        )
    )

    belief = result.active_snapshots["belief_assumption"].value
    goal_value = result.active_snapshots["goal_value"].value
    open_loop = result.active_snapshots["open_loop"].value
    domain_knowledge = result.active_snapshots["domain_knowledge"].value

    assert any(record.summary == "Reviewed rollout guidance" for record in belief.beliefs)
    assert goal_value.explicit_goals
    assert open_loop.unresolved_loops
    assert not any(hit.hit_id == "knowledge:external:1" for hit in domain_knowledge.hits)
