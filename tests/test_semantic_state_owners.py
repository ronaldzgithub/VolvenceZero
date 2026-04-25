from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.semantic_state import (
    SEMANTIC_OWNER_SLOTS,
    NoOpSemanticProposalRuntime,
    PlanIntentSnapshot,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
    SemanticStateStore,
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
    assert "consent-boundary-denied" in boundary_policy.trigger_reasons
    assert boundary_policy.active_decision.refer_out_required is True
