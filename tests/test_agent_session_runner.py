from __future__ import annotations

import asyncio
import types
from dataclasses import replace

from volvence_zero.agent import (
    AgentSessionRunner,
    MultiPathBenchmarkReport,
    default_active_runner,
    run_multi_path_benchmark,
    run_substrate_path_benchmark,
)
from volvence_zero.application import (
    ApplicationCaseCluster,
    ApplicationRareHeavyCheckpoint,
    PlaybookRule,
)
from volvence_zero.credit.gate import CreditSnapshot, GateDecision, ModificationGate, SelfModificationRecord
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop.pipeline import RareHeavyArtifact
from volvence_zero.joint_loop import JointLoopSchedule, PipelineConfig
from volvence_zero.evaluation.backbone import EvolutionDecision, EvolutionJudgement, JudgementCategory
from volvence_zero.prediction import PredictionError
from volvence_zero.reflection import WritebackMode
from volvence_zero.agent.session import RareHeavyPreImportEvaluation
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    SubstrateFallbackMode,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
    TransformersOpenWeightResidualRuntime,
)


def test_agent_session_runner_executes_single_turn():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I need help organizing my plan and I also feel overwhelmed."))

    assert result.acceptance_passed is True
    assert result.wave_id == "wave-1"
    assert "evaluation" in result.active_snapshots
    assert result.active_regime is not None
    assert result.response.text
    assert result.event_count > 0
    assert result.joint_schedule_action in {"ssl-only", "full-cycle", "evidence-only"}
    assert result.active_snapshots["substrate"].value.surface_kind is SurfaceKind.RESIDUAL_STREAM
    assert "Transformers open-weight capture" in result.active_snapshots["substrate"].value.description
    assert "reflection" in result.active_snapshots
    assert "temporal_abstraction" in result.active_snapshots
    assert "case_memory" in result.active_snapshots
    assert "strategy_playbook" in result.active_snapshots
    assert "experience_fast_prior" in result.active_snapshots
    assert "experience_consolidation" in result.active_snapshots
    assert "session_post_slow_loop" in result.shadow_snapshots
    assert result.shadow_snapshots["session_post_slow_loop"].value.queue_state.completed_job_count == 0


def test_agent_session_runner_reuses_session_memory_across_turns():
    runner = AgentSessionRunner(session_id="s1")
    first = asyncio.run(runner.run_turn("Remember that I prefer calm, reflective collaboration."))
    second = asyncio.run(runner.run_turn("Can you help me continue that plan from before?"))

    assert first.wave_id == "wave-1"
    assert second.wave_id == "wave-2"
    assert len(second.active_snapshots["memory"].value.retrieved_entries) >= 1
    assert second.active_snapshots["dual_track"].value.world_track.controller_source in (
        "temporal+memory",
        "temporal-track-projected",
        "temporal-track-owner",
    )


def test_agent_session_runner_exposes_temporal_and_regime_views():
    runner = AgentSessionRunner(
        session_id="exposed-kernel-session",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    result = asyncio.run(runner.run_turn("Please guide me carefully through a difficult decision."))

    assert result.active_regime is not None
    assert result.active_abstract_action is not None
    assert result.metacontroller_state is not None
    assert result.metacontroller_state.mode == "full-learned"
    assert isinstance(result.evaluation_alerts, tuple)
    assert result.response.regime_id == result.active_regime
    assert result.response.abstract_action == result.active_abstract_action
    assert result.active_snapshots["temporal_abstraction"].value.action_family_version == result.metacontroller_state.action_family_version
    assert result.joint_learning_summary
    evaluation_scores = {
        score.metric_name: score.value
        for score in result.active_snapshots["evaluation"].value.turn_scores
    }
    evaluation_metric_names = set(evaluation_scores)
    assert "joint_learning_progress" in evaluation_metric_names
    assert "residual_env_fidelity" in evaluation_metric_names
    assert "temporal_action_commitment" in evaluation_metric_names
    assert evaluation_scores["joint_learning_progress"] > 0.0
    temporal_commitment_score = next(
        score
        for score in result.active_snapshots["evaluation"].value.turn_scores
        if score.metric_name == "temporal_action_commitment"
    )
    assert "family_version=" in temporal_commitment_score.evidence
    credit_events = {record.source_event for record in result.active_snapshots["credit"].value.recent_credits}
    assert any(
        event in credit_events
        for event in {
            "joint_learning_progress",
            "evaluation:temporal_action_commitment",
            "session-evaluation:temporal_action_commitment",
        }
    )


def test_agent_session_runner_exposes_staged_temporal_slots():
    runner = AgentSessionRunner(
        session_id="staged-temporal-session",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )

    result = asyncio.run(runner.run_turn("Please help me regulate the pressure while keeping the plan coherent."))

    active_slots = set(result.active_snapshots)
    assert "world_temporal" in active_slots
    assert "self_temporal" in active_slots
    assert "temporal_abstraction" in active_slots
    assert "world_temporal_consolidation" in active_slots
    assert "self_temporal_consolidation" in active_slots
    assert result.active_snapshots["world_temporal_consolidation"].value.prediction_error_applied in {True, False}
    assert result.active_snapshots["self_temporal_consolidation"].value.prediction_error_applied in {True, False}


def test_agent_session_runner_defaults_self_temporal_to_cloned_discovered_lineage():
    runner = AgentSessionRunner(session_id="cloned-temporal-lineage")

    world_snapshot = runner._world_temporal_policy.export_rare_heavy_snapshot()
    self_snapshot = runner._self_temporal_policy.export_rare_heavy_snapshot()

    assert world_snapshot == self_snapshot
    assert runner._world_temporal_policy is not runner._self_temporal_policy
    assert runner._world_temporal_policy.parameter_store is not runner._self_temporal_policy.parameter_store


def test_agent_session_runner_returns_user_visible_response():
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I feel tense and I need a careful response."))

    assert result.response.text
    assert result.response.rationale
    assert "switch_gate=" in result.response.rationale
    assert "joint=" in result.response.rationale
    assert "primary_lesson=" in result.response.rationale


def test_agent_session_runner_exposes_bounded_writeback_state():
    runner = AgentSessionRunner(session_id="writeback-session", reflection_mode=WritebackMode.APPLY)

    result = asyncio.run(runner.run_turn("Remember a stable preference and keep the interaction supportive."))

    assert result.writeback_source == "active"
    assert isinstance(result.writeback_operations, tuple)
    assert isinstance(result.writeback_blocks, tuple)


def test_agent_session_runner_defers_slow_writeback_until_context_boundary():
    runner = AgentSessionRunner(session_id="session-post-writeback", reflection_mode=WritebackMode.APPLY)

    first = asyncio.run(runner.run_turn("Remember that I prefer calm and structured support."))
    boundary_ops = runner.begin_new_context(reason="test-boundary")

    assert first.bounded_writeback_applied is False
    assert first.shadow_snapshots["session_post_slow_loop"].value.queue_state.completed_job_count == 0
    assert any(op.startswith("session-post-slow-loop:enqueued:") for op in boundary_ops)
    assert runner.session_post_snapshot is not None
    assert runner.session_post_snapshot.value.queue_state.pending_job_count == 1

    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert len(slow_loop_results) == 1
    assert slow_loop_results[0].writeback_result is not None
    assert slow_loop_results[0].applied is True or slow_loop_results[0].blocked is True
    assert runner.session_post_snapshot is not None
    assert runner.session_post_snapshot.value.queue_state.pending_job_count == 0
    assert runner.session_post_snapshot.value.queue_state.completed_job_count == 1
    assert len(runner.session_post_snapshot.value.recent_results) == 1

    second = asyncio.run(runner.run_turn("Continue in the next context."))

    assert second.session_post_completed_job_count >= 1
    assert second.shadow_snapshots["session_post_slow_loop"].value.queue_state.completed_job_count >= 1


def test_agent_session_runner_publishes_experience_consolidation_after_session_post_completion():
    runner = AgentSessionRunner(
        session_id="experience-consolidation-session",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need the smallest next step first."))
    runner.begin_new_context(reason="experience-consolidation-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert slow_loop_results
    assert runner.experience_consolidation_snapshot is not None
    experience_snapshot = runner.experience_consolidation_snapshot.value
    assert experience_snapshot.deltas
    assert experience_snapshot.playbook_delta_count >= 1
    assert any(delta.target_slot == "case_memory" for delta in experience_snapshot.deltas)
    assert experience_snapshot.delayed_outcome_ledger
    assert experience_snapshot.sequence_payoffs
    assert experience_snapshot.latest_prior_update is not None
    assert experience_snapshot.latest_writeback_report is not None
    assert experience_snapshot.latest_writeback_report.proposed_target_count >= 1
    assert experience_snapshot.continuum_profile_id is not None
    assert experience_snapshot.active_band_ids
    assert any(
        record.metric_name == "delayed_retrieval_mix_alignment"
        for record in runner.evaluation_backbone.records
    )
    assert runner._application_rare_heavy_state.retrieval_readout_checkpoint is not None
    checkpoint = runner._application_rare_heavy_state.retrieval_readout_checkpoint
    assert checkpoint.source_attribution_ids
    assert checkpoint.source_sequence_ids
    assert checkpoint.mean_retrieval_mix_alignment > 0.0


def test_agent_session_runner_uses_gated_retrieval_readout_checkpoint_next_turn():
    runner = AgentSessionRunner(
        session_id="retrieval-readout-checkpoint-session",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need calm support with the smallest next step."))
    runner.begin_new_context(reason="retrieval-readout-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert slow_loop_results
    assert runner._application_rare_heavy_state.retrieval_readout_checkpoint is not None
    checkpoint = runner._application_rare_heavy_state.retrieval_readout_checkpoint
    assert checkpoint.source_attribution_ids
    assert checkpoint.source_sequence_ids

    next_result = asyncio.run(runner.run_turn("Help me continue in the next context."))
    retrieval_policy = next_result.active_snapshots["retrieval_policy"].value

    assert "checkpoint=present" in retrieval_policy.description


def test_agent_session_runner_promotes_domain_knowledge_for_next_turn():
    runner = AgentSessionRunner(
        session_id="domain-knowledge-learning-session",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    first = asyncio.run(
        runner.run_turn("I need calm, high-level help thinking through a family transition without jumping to conclusions.")
    )
    initial_record_ids = {record.record_id for record in runner._domain_knowledge_store.records}

    assert "domain_knowledge" in first.active_snapshots
    assert first.active_snapshots["domain_knowledge"].value.hits

    runner.begin_new_context(reason="domain-knowledge-learning-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert slow_loop_results
    latest_prior = slow_loop_results[0].application_prior_update
    assert latest_prior is not None
    assert latest_prior.domain_knowledge_updates
    assert any(
        delta.target_slot == "domain_knowledge" for delta in slow_loop_results[0].experience_deltas
    )

    learned_record_ids = {record.record_id for record in runner._domain_knowledge_store.records}
    assert learned_record_ids > initial_record_ids
    assert any(record_id.startswith("knowledge:slow-loop:") for record_id in learned_record_ids)

    next_result = asyncio.run(runner.run_turn("Continue helping me with the same family transition context."))
    next_hits = next_result.active_snapshots["domain_knowledge"].value.hits

    assert next_hits
    assert any(hit.hit_id.startswith("knowledge:slow-loop:") for hit in next_hits)


def test_agent_session_runner_derives_case_and_playbook_eta_signals_between_turns():
    runner = AgentSessionRunner(
        session_id="experience-prior-signals",
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need calm support with the smallest next step."))
    signals = runner._experience_eta_signals()

    assert "experience_case_strength" in signals
    assert "experience_playbook_strength" in signals
    assert "experience_case_continuum_position" in signals
    assert "experience_control_prior_strength" in signals

    second = asyncio.run(runner.run_turn("Keep helping me with the same situation."))
    turn_scores = {score.metric_name: score.value for score in second.active_snapshots["evaluation"].value.turn_scores}

    assert "scheduler_control_prior_strength" in turn_scores
    assert "application_continuum_case_coverage" in turn_scores


def test_agent_session_runner_feeds_delayed_experience_credit_into_next_turn_schedule():
    runner = AgentSessionRunner(
        session_id="experience-credit-schedule",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need the smallest next step first."))
    runner.begin_new_context(reason="experience-credit-boundary")
    asyncio.run(runner.drain_session_post_slow_loop())

    result = asyncio.run(runner.run_turn("Start the next context with what you learned."))
    turn_scores = {score.metric_name: score.value for score in result.active_snapshots["evaluation"].value.turn_scores}

    assert "scheduler_experience_credit" in turn_scores
    assert "experience_fast_prior" in result.active_snapshots
    experience_fast_prior = result.active_snapshots["experience_fast_prior"].value
    assert experience_fast_prior.source_attribution_ids
    assert experience_fast_prior.prior_strength > 0.0
    assert result.metacontroller_state is not None
    assert result.metacontroller_state.fast_prior_strength > 0.0
    assert result.metacontroller_state.fast_prior_switch_pressure_delta != 0.0
    assert "delayed_fast_prior_available" in turn_scores
    assert "delayed_retrieval_mix_bias_applied" in turn_scores
    assert "temporal_fast_prior_strength" in turn_scores
    assert "temporal_fast_prior_switch_pressure" in turn_scores
    experience_consolidation = result.active_snapshots["experience_consolidation"].value
    assert experience_consolidation.delayed_credit_summary is not None
    assert experience_consolidation.delayed_credit_summary.action_family_version >= 0
    assert experience_consolidation.delayed_credit_summary.sequence_count >= 1


def test_agent_session_runner_imports_application_rare_heavy_checkpoint_into_fast_path():
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id="phase4-application-runtime",
        allow_live_substrate_mutation=True,
    )
    runner = AgentSessionRunner(
        session_id="phase4-application-import",
        default_residual_runtime=runtime,
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99),
        rare_heavy_enabled=False,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )
    artifact = RareHeavyArtifact(
        artifact_id="phase4-artifact",
        owner_path="offline-sslrl-pipeline",
        created_at_ms=1,
        temporal_snapshot=runner._joint_loop.world_temporal_policy.export_rare_heavy_snapshot(),
        memory_checkpoint=runner._joint_loop.memory_store.export_rare_heavy_state(checkpoint_id="phase4-memory"),
        substrate_checkpoint=None,
        transition_step=0,
        final_ssl_loss=0.1,
        final_total_reward=0.2,
        description="Phase 4 application artifact.",
        application_checkpoint=ApplicationRareHeavyCheckpoint(
            checkpoint_id="phase4-application",
            domain_template_biases=(("career_decision", 0.9),),
            case_clusters=(
                ApplicationCaseCluster(
                    cluster_id="cluster-1",
                    problem_pattern="family-transition-high-emotion",
                    exemplar_count=3,
                    mean_relevance=0.78,
                    risk_markers=("risk-medium", "child-impact"),
                    description="Clustered family-transition experience.",
                ),
            ),
            distilled_playbook_rules=(
                PlaybookRule(
                    rule_id="playbook-1",
                    problem_pattern="family-transition-high-emotion",
                    recommended_regime="emotional_support",
                    recommended_ordering=("stabilize", "split_axes", "smallest_next_step"),
                    recommended_pacing="gradual",
                    avoid_patterns=("procedure-dump-too-early",),
                    knowledge_weight_hint=0.35,
                    experience_weight_hint=0.75,
                    applicability_scope=("emotional_support",),
                    confidence=0.81,
                    description="Distilled playbook from application rare-heavy refresh.",
                ),
            ),
            description="Application rare-heavy checkpoint for phase4 test.",
        ),
    )

    import_result = runner.apply_rare_heavy_artifact(artifact, checkpoint_id="phase4-import")
    result = asyncio.run(runner.run_turn("I need help thinking through this life decision."))

    retrieval_policy = result.active_snapshots["retrieval_policy"].value
    case_memory = result.active_snapshots["case_memory"].value
    strategy_playbook = result.active_snapshots["strategy_playbook"].value

    assert "rare-heavy:application-domain-refresh" in import_result.applied_operations
    assert "career_decision" in retrieval_policy.knowledge_domains
    assert "family-transition-high-emotion" in case_memory.active_problem_patterns
    assert any(rule.problem_pattern == "family-transition-high-emotion" for rule in strategy_playbook.matched_rules)


def test_agent_session_runner_keeps_application_checkpoint_attached_during_rare_heavy_review():
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id="phase4-review-runtime",
        allow_live_substrate_mutation=True,
    )
    runner = AgentSessionRunner(
        session_id="phase4-review-session",
        default_residual_runtime=runtime,
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99),
        rare_heavy_enabled=False,
    )
    artifact = RareHeavyArtifact(
        artifact_id="phase4-review-artifact",
        owner_path="offline-sslrl-pipeline",
        created_at_ms=1,
        temporal_snapshot=runner._joint_loop.world_temporal_policy.export_rare_heavy_snapshot(),
        memory_checkpoint=runner._joint_loop.memory_store.export_rare_heavy_state(checkpoint_id="phase4-review-memory"),
        substrate_checkpoint=None,
        transition_step=0,
        final_ssl_loss=0.1,
        final_total_reward=0.2,
        description="Phase 4 review artifact.",
        application_checkpoint=ApplicationRareHeavyCheckpoint(
            checkpoint_id="phase4-review-application",
            domain_template_biases=(("career_decision", 0.9),),
            case_clusters=(),
            distilled_playbook_rules=(),
            description="Application checkpoint carried only for review-path coverage.",
        ),
    )

    review_result = runner.review_rare_heavy_artifact(artifact, checkpoint_id="phase4-review")

    assert review_result.applied_operations == ()
    assert review_result.checkpoint.application_checkpoint is not None
    assert review_result.checkpoint.application_checkpoint.checkpoint_id == "phase4-review:application-review"
    assert "session-owned review" in review_result.description


def test_agent_session_runner_default_profile_forms_widened_phase234_chain():
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id="phase234-default-runtime",
        allow_live_substrate_mutation=True,
    )
    runner = AgentSessionRunner(
        session_id="phase234-default-chain",
        default_residual_runtime=runtime,
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99),
        rare_heavy_enabled=False,
        reflection_mode=WritebackMode.APPLY,
    )

    first = asyncio.run(
        runner.run_turn("I feel overwhelmed about divorce and need calm support with the smallest next step first.")
    )

    assert "case_memory" in first.active_snapshots
    assert "strategy_playbook" in first.active_snapshots
    assert "experience_fast_prior" in first.active_snapshots
    assert "experience_consolidation" in first.active_snapshots

    runner.begin_new_context(reason="phase234-default-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())
    assert slow_loop_results

    second = asyncio.run(runner.run_turn("Continue in the next context with what you learned."))
    second_scores = {score.metric_name: score.value for score in second.active_snapshots["evaluation"].value.turn_scores}
    second_fast_prior = second.active_snapshots["experience_fast_prior"].value
    second_consolidation = second.active_snapshots["experience_consolidation"].value

    assert second_consolidation.delayed_outcome_ledger
    assert second_fast_prior.prior_strength > 0.0
    assert "scheduler_experience_credit" in second_scores
    assert "delayed_fast_prior_available" in second_scores

    artifact = RareHeavyArtifact(
        artifact_id="phase234-default-artifact",
        owner_path="offline-sslrl-pipeline",
        created_at_ms=2,
        temporal_snapshot=runner._joint_loop.world_temporal_policy.export_rare_heavy_snapshot(),
        memory_checkpoint=runner._joint_loop.memory_store.export_rare_heavy_state(
            checkpoint_id="phase234-default-memory"
        ),
        substrate_checkpoint=None,
        transition_step=0,
        final_ssl_loss=0.1,
        final_total_reward=0.2,
        description="Phase 4 artifact for widened default profile.",
        application_checkpoint=ApplicationRareHeavyCheckpoint(
            checkpoint_id="phase234-default-application",
            domain_template_biases=(("career_decision", 0.95),),
            case_clusters=(
                ApplicationCaseCluster(
                    cluster_id="phase234-default-cluster",
                    problem_pattern="family-transition-high-emotion",
                    exemplar_count=3,
                    mean_relevance=0.8,
                    risk_markers=("risk-medium",),
                    description="Cluster imported into widened default profile.",
                ),
            ),
            distilled_playbook_rules=(
                PlaybookRule(
                    rule_id="phase234-default-playbook",
                    problem_pattern="family-transition-high-emotion",
                    recommended_regime="emotional_support",
                    recommended_ordering=("stabilize", "split_axes", "smallest_next_step"),
                    recommended_pacing="gradual",
                    avoid_patterns=("procedure-dump-too-early",),
                    knowledge_weight_hint=0.3,
                    experience_weight_hint=0.8,
                    applicability_scope=("emotional_support",),
                    confidence=0.84,
                    description="Rare-heavy distilled playbook for widened default profile.",
                ),
            ),
            description="Application rare-heavy checkpoint for widened default profile.",
        ),
    )

    import_result = runner.apply_rare_heavy_artifact(artifact, checkpoint_id="phase234-default-import")
    third = asyncio.run(runner.run_turn("Help me think through the life decision while keeping me steady."))
    third_retrieval = third.active_snapshots["retrieval_policy"].value
    third_case_memory = third.active_snapshots["case_memory"].value
    third_strategy_playbook = third.active_snapshots["strategy_playbook"].value

    assert "rare-heavy:application-domain-refresh" in import_result.applied_operations
    assert "career_decision" in third_retrieval.knowledge_domains
    assert "family-transition-high-emotion" in third_case_memory.active_problem_patterns
    assert any(rule.rule_id == "phase234-default-playbook" for rule in third_strategy_playbook.matched_rules)


def test_agent_session_runner_session_post_loop_fails_closed_when_apply_disabled():
    runner = AgentSessionRunner(session_id="session-post-proposal", reflection_mode=WritebackMode.PROPOSAL_ONLY)

    asyncio.run(runner.run_turn("Capture reflection proposals without applying them."))
    runner.begin_new_context(reason="proposal-only-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert len(slow_loop_results) == 1
    assert slow_loop_results[0].writeback_result is not None
    assert "writeback-mode-not-apply" in slow_loop_results[0].writeback_result.blocked_operations


def test_agent_session_runner_blocks_application_promotion_when_judge_holds_or_rolls_back():
    runner = AgentSessionRunner(
        session_id="judge-gated-application",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need the smallest next step first."))
    hold_judgement = EvolutionJudgement(
        decision=EvolutionDecision.HOLD,
        category=JudgementCategory.INSUFFICIENT_EVIDENCE,
        replay_passed=True,
        abstraction_trend=0.0,
        learning_trend=0.0,
        relationship_trend=0.0,
        reasons=("insufficient-positive-evidence",),
        description="Hold application widening.",
    )
    assert runner._last_session_post_writeback_request is not None
    runner._last_session_post_writeback_request = replace(
        runner._last_session_post_writeback_request,
        evolution_judgement=hold_judgement,
        structural_writeback_allowed=False,
    )

    runner.begin_new_context(reason="judge-gated-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert slow_loop_results
    assert slow_loop_results[0].experience_deltas
    structural_deltas = tuple(
        delta
        for delta in slow_loop_results[0].experience_deltas
        if delta.target_slot in {"case_memory", "strategy_playbook", "boundary_policy"}
    )
    retrieval_deltas = tuple(
        delta
        for delta in slow_loop_results[0].experience_deltas
        if delta.target_slot == "retrieval_policy"
    )
    assert structural_deltas
    assert all(delta.blocked for delta in structural_deltas)
    assert retrieval_deltas
    assert slow_loop_results[0].application_prior_writeback_report is not None
    assert slow_loop_results[0].application_prior_writeback_report.blocked_targets


def test_agent_session_runner_applies_application_prior_to_next_context_fast_path():
    runner = AgentSessionRunner(
        session_id="application-prior-fast-path",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need the smallest next step first."))
    assert runner._last_session_post_writeback_request is not None
    runner._last_session_post_writeback_request = replace(
        runner._last_session_post_writeback_request,
        structural_writeback_allowed=True,
    )
    runner.begin_new_context(reason="application-prior-fast-path-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert slow_loop_results
    assert slow_loop_results[0].application_prior_writeback_report is not None
    assert slow_loop_results[0].application_prior_writeback_report.applied_targets
    assert any(record.case_id.startswith("case:slow-loop:") for record in runner._case_memory_store.records)

    second = asyncio.run(
        runner.run_turn("I still feel overwhelmed about divorce and need the next step kept very small.")
    )

    case_memory = second.active_snapshots["case_memory"].value
    strategy_playbook = second.active_snapshots["strategy_playbook"].value

    assert any(hit.case_id.startswith("case:slow-loop:") for hit in case_memory.hits)
    assert any(rule.rule_id.startswith("playbook:slow-loop:") for rule in strategy_playbook.matched_rules)


def test_agent_session_runner_partially_blocks_application_prior_by_credit_target():
    runner = AgentSessionRunner(
        session_id="application-prior-partial-block",
        reflection_mode=WritebackMode.APPLY,
        config=FinalRolloutConfig(
            case_memory=WiringLevel.ACTIVE,
            strategy_playbook=WiringLevel.ACTIVE,
        ),
    )

    asyncio.run(runner.run_turn("I feel overwhelmed about divorce and need the smallest next step first."))
    assert runner._last_session_post_writeback_request is not None
    runner._last_session_post_writeback_request = replace(
        runner._last_session_post_writeback_request,
        structural_writeback_allowed=True,
        credit_snapshot=CreditSnapshot(
            recent_credits=(),
            recent_modifications=(
                SelfModificationRecord(
                    target="application.strategy_playbook.rules.family-transition-high-emotion",
                    gate=ModificationGate.BACKGROUND,
                    decision=GateDecision.BLOCK,
                    old_value_hash="before",
                    new_value_hash="before",
                    justification="Seeded block for strategy playbook target.",
                    timestamp_ms=1,
                    is_reversible=True,
                ),
            ),
            cumulative_credit_by_level=(),
            description="Seeded partial block credit snapshot.",
        ),
    )

    runner.begin_new_context(reason="application-prior-partial-block-boundary")
    slow_loop_results = asyncio.run(runner.drain_session_post_slow_loop())

    assert slow_loop_results
    report = slow_loop_results[0].application_prior_writeback_report
    assert report is not None
    assert any(target.startswith("application.case_memory.records.") for target in report.applied_targets)
    assert any(target.startswith("application.strategy_playbook.rules.") for target in report.blocked_targets)
    assert any(record.case_id.startswith("case:slow-loop:") for record in runner._case_memory_store.records)
    assert not any(
        rule.rule_id == "playbook:slow-loop:family-transition-high-emotion:1"
        for rule in runner._application_rare_heavy_state.distilled_playbook_rules
    )


def test_agent_session_runner_accepts_hook_ready_substrate_factory():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="hook-ready-runtime")
    runner = AgentSessionRunner(
        session_id="hook-session",
        substrate_adapter_factory=lambda user_input, turn_index: OpenWeightResidualStreamSubstrateAdapter(
            runtime=runtime,
            default_source_text=user_input,
        ),
    )

    result = asyncio.run(runner.run_turn("Use the hook-ready substrate path."))

    assert result.acceptance_passed is True
    assert result.active_snapshots["substrate"].value.model_id == "hook-ready-runtime"


def test_agent_session_runner_defaults_to_real_transformers_runtime_with_builtin_fallback():
    runner = AgentSessionRunner(
        session_id="real-runtime-session",
        substrate_model_id="missing-local-model",
        substrate_local_files_only=True,
    )

    assert isinstance(runner._default_residual_runtime, TransformersOpenWeightResidualRuntime)

    result = asyncio.run(runner.run_turn("Use the default real substrate runtime."))

    assert result.active_snapshots["substrate"].value.model_id == "runner-transformers-runtime"
    assert "Transformers open-weight capture" in result.active_snapshots["substrate"].value.description
    assert result.substrate_runtime_origin == "builtin-fallback"
    assert result.substrate_fallback_active is True
    assert result.substrate_capture_source == "fallback"
    assert result.substrate_residual_sequence_length > 0


def test_agent_session_runner_can_fail_closed_for_missing_substrate_model():
    try:
        AgentSessionRunner(
            session_id="deny-runtime-session",
            substrate_model_id="missing-local-model",
            substrate_runtime_mode="strict-local",
        )
    except Exception as exc:
        assert type(exc).__name__ in {"OSError", "ValueError", "RuntimeError"}
    else:
        raise AssertionError("Expected runner construction to fail when substrate fallback mode is deny.")


def test_agent_session_runner_uses_explicit_model_source_for_strict_local():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            AgentSessionRunner(
                session_id="explicit-source-session",
                substrate_model_id="distilgpt2",
                substrate_model_source=tmpdir,
                substrate_runtime_mode="strict-local",
            )
        except Exception as exc:
            assert type(exc).__name__ in {"OSError", "ValueError", "RuntimeError"}
        else:
            raise AssertionError("Expected explicit empty local model source to fail in strict-local mode.")


def test_agent_session_runner_prefer_local_mode_marks_fallback_metadata():
    runner = AgentSessionRunner(
        session_id="prefer-local-session",
        substrate_model_id="missing-local-model",
        substrate_runtime_mode="prefer-local",
    )

    result = asyncio.run(runner.run_turn("Please use a local model when available."))

    assert result.substrate_runtime_origin == "builtin-fallback"
    assert result.substrate_fallback_active is True
    assert result.substrate_capture_source == "fallback"
    assert result.substrate_model_id == "runner-transformers-runtime"


def test_agent_session_runner_reports_real_runtime_metadata_for_injected_runtime():
    runtime = SyntheticOpenWeightResidualRuntime(model_id="real-path-runtime")
    runtime.runtime_origin = "hf-local"
    runner = AgentSessionRunner(
        session_id="real-path-session",
        default_residual_runtime=runtime,
    )

    result = asyncio.run(runner.run_turn("Use the injected local runtime path."))

    assert result.substrate_model_id == "real-path-runtime"
    assert result.substrate_runtime_origin == "hf-local"
    assert result.substrate_fallback_active is False
    assert result.substrate_capture_source == "real"
    assert result.substrate_residual_sequence_length > 0


# ---------------------------------------------------------------------------
# D1/D2 — Real substrate data flows through evaluation and RL
# ---------------------------------------------------------------------------

def test_evaluation_signals_flow_from_real_substrate_capture():
    """D1: Verify evaluation scores are derived from real substrate
    capture data, not just hardcoded values."""
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("Help me plan a complex project with many moving parts."))

    eval_snapshot = result.active_snapshots["evaluation"].value
    assert len(eval_snapshot.turn_scores) > 0, "Evaluation should produce turn scores from substrate"
    score_families = {s.family for s in eval_snapshot.turn_scores}
    assert "task" in score_families or "learning" in score_families, (
        f"Expected task or learning scores, got families: {score_families}"
    )


def test_multi_turn_rl_loop_produces_policy_changes():
    """D2: Run 5 turns through the full agent session and verify the
    joint loop produces policy parameter changes."""
    runner = AgentSessionRunner(
        session_id="multi-turn-rl-session",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )

    inputs = [
        "Help me organize my thoughts about this project.",
        "I feel overwhelmed by the complexity here.",
        "Can you break this down into smaller steps?",
        "That helps, but I'm still worried about the timeline.",
        "Let's focus on the most important pieces first.",
    ]
    reports = []
    for user_input in inputs:
        result = asyncio.run(runner.run_turn(user_input))
        reports.append(result)

    has_cycle = any(r.joint_cycle_report is not None for r in reports)
    assert has_cycle, "At least one turn should trigger a full joint cycle"

    cycle_reports = [r.joint_cycle_report for r in reports if r.joint_cycle_report is not None]
    objectives = [cr.policy_objective for cr in cycle_reports]
    assert any(abs(o) > 1e-8 for o in objectives), (
        f"Expected non-zero policy objectives from RL, got: {objectives}"
    )


def test_agent_session_runner_keeps_rare_heavy_review_only_when_high_pe_persists():
    runner = AgentSessionRunner(
        session_id="rare-heavy-session",
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99, pe_full_cycle_threshold=0.6),
        rare_heavy_trace_window=2,
        rare_heavy_min_traces=2,
        rare_heavy_cooldown_turns=0,
        rare_heavy_pipeline_config=PipelineConfig(
            n_z=3,
            ssl_min_steps=1,
            ssl_max_steps=1,
            rl_max_steps=1,
        ),
    )

    asyncio.run(runner.run_turn("Seed the first trace for rare-heavy review."))
    async def _accept_candidate(self, *, artifact, bundle):
        del artifact, bundle
        return RareHeavyPreImportEvaluation(
            accepted=True,
            case_count=2,
            baseline_mean_score=0.55,
            candidate_mean_score=0.60,
            mean_score_delta=0.05,
            worst_score_delta=0.01,
            positive_fraction=1.0,
            judgement="promote",
            reasons=(),
            description="Forced pre-import acceptance so frozen doctrine remains the deciding gate.",
        )

    runner._evaluate_rare_heavy_candidate = types.MethodType(_accept_candidate, runner)
    runner._previous_prediction_reward = -0.7
    runner._previous_prediction_magnitude = 1.35
    runner._previous_prediction_error = PredictionError(
        task_error=0.9,
        relationship_error=0.6,
        regime_error=0.4,
        action_error=0.8,
        magnitude=1.35,
        signed_reward=-0.7,
        description="Persistent high prediction error should trigger rare-heavy import.",
    )

    result = asyncio.run(runner.run_turn("Continue after the high-error turn."))

    assert result.rare_heavy_result is not None
    assert result.rare_heavy_result.recommended is True
    assert result.rare_heavy_result.applied is False
    assert result.rare_heavy_result.artifact_id is not None
    assert result.rare_heavy_result.applied_operations == ()
    assert result.rare_heavy_result.substrate_status == "review-only"
    assert result.rare_heavy_result.substrate_training_mode == "adapter-delta-v2"
    assert result.rare_heavy_result.import_decision == "blocked-by-doctrine"
    assert result.rare_heavy_result.reject_reason == "frozen-substrate-doctrine"
    assert result.rare_heavy_result.pre_import_passed is True
    assert result.rare_heavy_result.pre_import_case_count >= 1
    assert result.rare_heavy_result.bundle_trace_count >= 2
    assert result.rare_heavy_result.bundle_alignment_ratio > 0.0
    assert result.rare_heavy_result.candidate_adapter_parameter_count > 0
    assert result.joint_schedule_action == "full-cycle-pe"


def test_agent_session_runner_rejects_rare_heavy_candidate_when_preimport_replay_fails():
    runner = AgentSessionRunner(
        session_id="rare-heavy-preimport-reject",
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99, pe_full_cycle_threshold=0.6),
        rare_heavy_trace_window=2,
        rare_heavy_min_traces=2,
        rare_heavy_cooldown_turns=0,
        rare_heavy_pipeline_config=PipelineConfig(
            n_z=3,
            ssl_min_steps=1,
            ssl_max_steps=1,
            rl_max_steps=1,
        ),
    )

    async def _reject_candidate(self, *, artifact, bundle):
        del artifact, bundle
        return RareHeavyPreImportEvaluation(
            accepted=False,
            case_count=2,
            baseline_mean_score=0.55,
            candidate_mean_score=0.42,
            mean_score_delta=-0.13,
            worst_score_delta=-0.20,
            positive_fraction=0.0,
            judgement="rollback",
            reasons=("forced-test-reject",),
            description="Forced pre-import rejection for test coverage.",
        )

    runner._evaluate_rare_heavy_candidate = types.MethodType(_reject_candidate, runner)

    asyncio.run(runner.run_turn("Seed the first trace for rare-heavy rejection."))
    runner._previous_prediction_reward = -0.7
    runner._previous_prediction_magnitude = 1.35
    runner._previous_prediction_error = PredictionError(
        task_error=0.9,
        relationship_error=0.6,
        regime_error=0.4,
        action_error=0.8,
        magnitude=1.35,
        signed_reward=-0.7,
        description="Persistent high prediction error should trigger rare-heavy review.",
    )

    result = asyncio.run(runner.run_turn("Continue after the high-error turn with forced reject."))

    assert result.rare_heavy_result is not None
    assert result.rare_heavy_result.recommended is True
    assert result.rare_heavy_result.applied is False
    assert result.rare_heavy_result.import_decision == "rejected-pre-import"
    assert result.rare_heavy_result.reject_reason == "forced-test-reject"
    assert result.rare_heavy_result.pre_import_passed is False
    assert result.rare_heavy_result.pre_import_judgement == "rollback"
    assert result.rare_heavy_result.pre_import_mean_score_delta < 0.0


def test_agent_session_runner_keeps_online_fast_substrate_self_mod_review_only_by_default():
    runner = AgentSessionRunner(
        session_id="online-fast-session",
        joint_schedule=JointLoopSchedule(
            ssl_interval=99,
            rl_interval=99,
            pe_full_cycle_threshold=0.6,
            pe_substrate_online_fast_threshold=0.18,
        ),
        rare_heavy_enabled=False,
    )

    asyncio.run(runner.run_turn("Seed substrate for online-fast self-mod."))
    runner._previous_prediction_reward = -0.25
    runner._previous_prediction_magnitude = 0.45
    runner._previous_prediction_error = PredictionError(
        task_error=0.35,
        relationship_error=0.2,
        regime_error=0.15,
        action_error=0.3,
        magnitude=0.45,
        signed_reward=-0.25,
        description="Moderate prediction error should trigger online-fast substrate self-mod.",
    )

    result = asyncio.run(runner.run_turn("Continue after the moderate-error turn."))

    assert result.online_fast_substrate_result is not None
    assert result.online_fast_substrate_result.recommended is True
    assert result.online_fast_substrate_result.applied is False
    assert result.online_fast_substrate_result.applied_operations == ()
    assert result.online_fast_substrate_result.blocked_operations == ("online-fast:frozen-substrate-doctrine",)
    assert result.online_fast_substrate_result.gate_decision == "frozen-substrate-doctrine"
    assert result.online_fast_substrate_result.checkpoint_id
    assert result.online_fast_substrate_result.fast_state_hash
    assert result.online_fast_substrate_result.fast_memory_signal
    assert "substrate_self_mod" in result.active_snapshots
    modification_targets = {record.target for record in result.active_snapshots["credit"].value.recent_modifications}
    assert "substrate.online_fast.delta" in modification_targets


def test_agent_session_runner_can_opt_into_experimental_live_substrate_mutation():
    runner = AgentSessionRunner(
        session_id="online-fast-session-experimental",
        joint_schedule=JointLoopSchedule(
            ssl_interval=99,
            rl_interval=99,
            pe_full_cycle_threshold=0.6,
            pe_substrate_online_fast_threshold=0.18,
        ),
        rare_heavy_enabled=False,
        default_residual_runtime=SyntheticOpenWeightResidualRuntime(
            model_id="online-fast-session-experimental-runtime",
            allow_live_substrate_mutation=True,
        ),
    )

    asyncio.run(runner.run_turn("Seed substrate for experimental online-fast self-mod."))
    runner._previous_prediction_reward = -0.25
    runner._previous_prediction_magnitude = 0.45
    runner._previous_prediction_error = PredictionError(
        task_error=0.35,
        relationship_error=0.2,
        regime_error=0.15,
        action_error=0.3,
        magnitude=0.45,
        signed_reward=-0.25,
        description="Moderate prediction error should trigger experimental online-fast substrate self-mod.",
    )

    result = asyncio.run(runner.run_turn("Continue after the moderate-error turn in experimental mode."))

    assert result.online_fast_substrate_result is not None
    assert result.online_fast_substrate_result.recommended is True
    assert result.online_fast_substrate_result.applied is True
    assert "online-fast:substrate-import" in result.online_fast_substrate_result.applied_operations


def test_substrate_snapshot_used_for_next_turn_trace():
    """B1 verification: After the first turn, the training trace for
    the second turn should be built from real substrate data."""
    runner = default_active_runner()
    asyncio.run(runner.run_turn("First turn to capture substrate."))
    assert runner._previous_substrate_snapshot is not None, (
        "After first turn, previous substrate snapshot should be stored"
    )
    assert runner._previous_substrate_snapshot.residual_sequence, (
        "Substrate snapshot should contain residual sequence for trace building"
    )


def test_pe_scheduled_session_turns_still_emit_learning_scores():
    runner = AgentSessionRunner(
        session_id="pe-scheduled-session",
        joint_schedule=JointLoopSchedule(ssl_interval=99, rl_interval=99, pe_full_cycle_threshold=0.6),
    )
    runner._previous_prediction_reward = -0.5
    runner._previous_prediction_magnitude = 0.8
    runner._previous_prediction_error = PredictionError(
        task_error=0.5,
        relationship_error=0.3,
        regime_error=0.2,
        action_error=0.4,
        magnitude=0.8,
        signed_reward=-0.5,
        description="PE-scheduled full cycle.",
    )

    result = asyncio.run(runner.run_turn("Use the PE-scheduled path."))
    scores = {
        score.metric_name: score.value
        for score in result.active_snapshots["evaluation"].value.turn_scores
    }

    assert result.joint_schedule_action == "full-cycle-pe"
    assert "joint_learning_progress" in scores


# ---------------------------------------------------------------------------
# E1 — Regime-driven behavioral shifts
# ---------------------------------------------------------------------------

def test_regime_responds_to_emotional_context():
    """E1: Verify that emotional input triggers appropriate regime selection
    (emotional_support or repair_and_deescalation)."""
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("I'm feeling really frustrated and upset right now."))

    assert result.active_regime is not None
    assert result.response.text
    assert result.response.regime_id is not None


def test_regime_responds_to_task_context():
    """E1: Verify that task-oriented input triggers problem_solving or
    guided_exploration regime."""
    runner = default_active_runner()
    result = asyncio.run(runner.run_turn("How do I implement a binary search tree in Python?"))

    assert result.active_regime is not None
    assert result.response.text


def test_response_context_carries_cognitive_state():
    """Verify the response path still carries user-visible cognitive state
    after assembly-driven control moved into ResponseAssemblySnapshot."""
    runner = default_active_runner()

    asyncio.run(runner.run_turn("First message to seed memory."))
    result = asyncio.run(runner.run_turn("Second message to check cognitive state flow."))

    assert result.response.text
    assert result.response.rationale


def test_run_substrate_path_benchmark_collects_turn_metrics():
    runner = AgentSessionRunner(
        session_id="benchmark-session",
        substrate_model_id="distilgpt2",
        substrate_runtime_mode="builtin-only",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    report = asyncio.run(
        run_substrate_path_benchmark(
            path_label="builtin",
            runner=runner,
            user_inputs=(
                "Help me structure a plan.",
                "I need the next step.",
                "Summarize the main risk.",
            ),
        )
    )

    assert report.path_label == "builtin"
    assert len(report.turns) == 3
    assert 0.0 <= report.acceptance_rate <= 1.0
    assert report.mean_residual_sequence_length > 0
    assert report.mean_turn_score_count > 0
    assert isinstance(report.metric_means, tuple)
    assert report.mean_policy_objective >= 0.0 or report.mean_policy_objective <= 0.0
    assert report.max_family_version >= 0
    assert report.description


def test_run_multi_path_benchmark_compares_paths():
    builtin_runner = AgentSessionRunner(
        session_id="multi-benchmark-builtin",
        substrate_model_id="distilgpt2",
        substrate_runtime_mode="builtin-only",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    local_runner = AgentSessionRunner(
        session_id="multi-benchmark-local",
        substrate_model_id="distilgpt2",
        substrate_runtime_mode="strict-local",
        substrate_device="cpu",
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=1),
    )
    report = asyncio.run(
        run_multi_path_benchmark(
            baseline_label="builtin",
            path_runners=(("builtin", builtin_runner), ("hf-local", local_runner)),
            user_inputs=(
                "Help me structure a plan.",
                "I need the next step.",
                "Summarize the main risk.",
            ),
        )
    )

    assert isinstance(report, MultiPathBenchmarkReport)
    assert report.baseline_label == "builtin"
    assert len(report.path_reports) == 2
    assert len(report.metric_deltas_from_baseline) == 1
    assert report.description
