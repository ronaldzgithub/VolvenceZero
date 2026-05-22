from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from volvence_zero.agent import (
    AgentSessionRunner,
    DEFAULT_DIALOGUE_CASE_VARIANTS,
    DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
    DEFAULT_DIALOGUE_REPLAY_SEEDS,
    DEFAULT_DIALOGUE_PROOF_CASES,
    DEFAULT_OPEN_DIALOGUE_SCENARIOS,
    DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
    OpenDialogueEpisodeState,
    OpenDialogueScenario,
    OpenDialogueBenchmarkComparisonReport,
    OpenDialogueBenchmarkPathReport,
    OpenDialogueBenchmarkReport,
    TranscriptOnlyUserSimulator,
    DialogueArtifactAcceptanceGateConfig,
    DialogueComprehensiveStage,
    DialogueNLEssenceAcceptanceConfig,
    DialogueNLEssenceAcceptanceDecision,
    DialogueNLEssenceAssessmentReport,
    DialogueBenchmarkReport,
    DialogueBenchmarkCaseReport,
    DialogueBenchmarkPathReport,
    DialogueBenchmarkComparisonReport,
    DialogueLongitudinalBenchmarkReport,
    LongitudinalDialogueReport,
    LongitudinalDialogueSessionEvidence,
    MemoryStratumFlowReport,
    NLAblationMatrixReport,
    RegimeLockinReport,
    DialogueRealComprehensiveBenchmarkConfig,
    DialogueSharedRunnerFactories,
    DialoguePaperSuiteAggregateReport,
    DialogueOptionDiscoveryReport,
    PECounterfactualClosureReport,
    evaluate_dialogue_nl_essence_acceptance,
    build_dialogue_nl_essence_assessment,
    build_dialogue_emergence_dashboard,
    build_dialogue_emergence_dashboard_payload,
    build_longitudinal_dialogue_report,
    build_memory_stratum_flow_report,
    build_nl_ablation_matrix_report,
    build_regime_lockin_report,
    build_dialogue_expert_review_internal_key,
    build_dialogue_expert_review_packet,
    build_dialogue_option_discovery_report,
    build_dialogue_paper_suite_manifest,
    build_pe_counterfactual_closure_report,
    aggregate_dialogue_human_ratings,
    load_dialogue_human_rating_entries_csv,
    build_pe_dominance_case_diagnosis_report,
    build_pe_dominance_comparison_report,
    build_replay_selection_training_traces,
    build_dialogue_replay_selection_artifact,
    build_real_dialogue_comprehensive_runner_factories,
    default_longitudinal_dialogue_personas,
    build_standard_dialogue_runner,
    default_rare_heavy_candidate_configs,
    default_dialogue_ablation_profiles,
    default_dialogue_strong_proof_profiles,
    default_dialogue_comprehensive_profiles,
    default_phase2_shadow_evidence_profiles,
    dialogue_paper_suite_config,
    default_open_dialogue_ablation_profiles,
    default_dialogue_real_proof_config,
    dialogue_case_variants,
    dialogue_paraphrase_families,
    open_dialogue_scenarios,
    DialogueBenchmarkTurn,
    ScriptedDialogueCase,
    build_deterministic_user_simulator,
    build_dialogue_case_report,
    build_open_dialogue_case_report,
    build_dialogue_replay_ranking_report,
    get_open_dialogue_scenario,
    generate_stochastic_dialogue_case_variants,
    run_replay_selection_artifact_acceptance_benchmark,
    run_dialogue_pe_eta_ablation_benchmark,
    run_dialogue_pe_eta_benchmark,
    run_dialogue_pe_eta_case,
    run_dialogue_pe_eta_comprehensive_benchmark,
    run_dialogue_pe_eta_longitudinal_benchmark,
    run_dialogue_pe_eta_perturbation_benchmark,
    run_dialogue_pe_eta_systematic_replay_benchmark,
    run_phase2_shadow_evidence_smoke,
    run_open_dialogue_benchmark,
    run_open_dialogue_ablation_benchmark,
    run_open_dialogue_case,
    run_dialogue_paper_suite_repeated_benchmark,
    run_real_dialogue_pe_eta_comprehensive_benchmark,
    run_real_dialogue_pe_eta_comprehensive_benchmark_staged,
    run_multi_artifact_acceptance_benchmark,
    train_rare_heavy_artifact_from_replay_selection,
    export_dialogue_emergence_dashboard_artifact,
    export_dialogue_paper_suite_artifact_bundle,
)
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.evaluation import CrossSessionGrowthReport, MetricIntervalSummary, PairwiseMetricEffect
from volvence_zero.semantic_state import SemanticProposalOperation
from volvence_zero.semantic_state.quality import (
    SemanticProposalQualityCaseResult,
    SemanticProposalQualityReport,
)
from volvence_zero.substrate import LocalSubstrateRuntimeMode, SyntheticOpenWeightResidualRuntime
from volvence_zero.agent.dialogue import (
    DialogueExpertReviewDimension,
    DialogueExpertReviewInternalKey,
    DialogueExpertReviewInternalKeyEntry,
    DialogueExpertReviewItem,
    DialogueExpertReviewPacket,
    DialogueExpertReviewSample,
    _case_summary_metrics,
    _open_case_summary_metrics,
    evaluate_dialogue_artifact_acceptance,
)
from volvence_zero.agent.paper_suite import PaperSuiteProvenance


def _synthetic_runner(case: ScriptedDialogueCase) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id=f"dialogue-proof:{case.case_id}")
    runtime.runtime_origin = "synthetic-proof"
    return AgentSessionRunner(
        session_id=f"dialogue-proof:{case.case_id}",
        default_residual_runtime=runtime,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
    )


def _synthetic_ablation_runner(profile_label: str, case: ScriptedDialogueCase) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id=f"dialogue-ablation:{profile_label}:{case.case_id}")
    runtime.runtime_origin = f"synthetic-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=case,
        residual_runtime=runtime,
    )


def _synthetic_perturbation_runner(profile_label: str, variant) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"dialogue-perturb:{profile_label}:{variant.case.case_id}"
    )
    runtime.runtime_origin = f"synthetic-perturb-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=variant.case,
        residual_runtime=runtime,
    )


def _synthetic_systematic_runner(profile_label: str, variant) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"dialogue-systematic:{profile_label}:{variant.case.case_id}"
    )
    runtime.runtime_origin = f"synthetic-systematic-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=variant.case,
        residual_runtime=runtime,
    )


def _synthetic_acceptance_runner(variant) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="dialogue-acceptance:shared")
    runtime.runtime_origin = "synthetic-acceptance"
    runtime.supports_live_substrate_mutation = True
    return build_standard_dialogue_runner(
        profile_label="pe-eta",
        case=variant.case,
        residual_runtime=runtime,
    )


def _synthetic_open_ablation_runner(profile_label: str, scenario: OpenDialogueScenario) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(model_id=f"open-dialogue:{profile_label}:{scenario.scenario_id}")
    runtime.runtime_origin = f"synthetic-open-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=ScriptedDialogueCase(
            case_id=f"open-dialogue:{scenario.scenario_id}",
            description=scenario.description,
            user_inputs=scenario.opening_turns,
        ),
        residual_runtime=runtime,
    )


def _synthetic_shared_factories() -> DialogueSharedRunnerFactories:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="dialogue-staged-shared")
    runtime.runtime_origin = "synthetic-staged-shared"
    return DialogueSharedRunnerFactories(
        residual_runtime=runtime,
        canonical_runner_factory=lambda profile_label, case: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=case,
            residual_runtime=runtime,
        ),
        perturbation_runner_factory=lambda profile_label, variant: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
            residual_runtime=runtime,
        ),
        open_runner_factory=lambda profile_label, scenario: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=ScriptedDialogueCase(
                case_id=f"open-dialogue:{scenario.scenario_id}",
                description=scenario.description,
                user_inputs=scenario.opening_turns,
            ),
            residual_runtime=runtime,
        ),
        systematic_runner_factory=lambda profile_label, variant: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
            residual_runtime=runtime,
        ),
        acceptance_runner_factory=lambda variant: build_standard_dialogue_runner(
            profile_label="pe-eta",
            case=variant.case,
            residual_runtime=runtime,
        ),
        description="synthetic shared benchmark factories",
    )


def _benchmark_turn(
    *,
    turn_index: int,
    pe: float,
    reward: float,
    action: str,
    regime: str,
    abstract_action: str,
    switch_gate: float,
    delayed_metric: float,
    pe_triggered: bool = False,
    action_family_version: int = 0,
    bounded_writeback_applied: bool = False,
    reflection_promotion_eligible: bool = False,
    session_post_completed_job_count: int = 0,
    evolution_decision: str | None = None,
    evolution_category: str | None = None,
    cross_session_verdict: str = "",
    nested_profile_active: bool = True,
    nested_context_reset_applied: bool = False,
    nested_context_reset_total_count: int = 0,
    slow_to_fast_init_benefit: float = 0.0,
    slow_to_fast_target_distance_before: float = 0.0,
    slow_to_fast_target_distance_after: float = 0.0,
    slow_to_fast_target_alignment_gain: float = 0.0,
    tower_consolidation_count: int = 0,
    learned_recall_count: int = 0,
    learned_recall_confidence: float = 0.0,
    learned_recall_core_guided: bool = False,
    memory_tower_depth: int = 0,
    memory_tower_alignment: float = 0.0,
    memory_tower_profile_id: str = "",
    case_memory_surface_active: bool = False,
    strategy_playbook_surface_active: bool = False,
    experience_fast_prior_surface_active: bool = False,
    experience_consolidation_surface_active: bool = False,
    runtime_backbone_evidence_active: bool = True,
    runtime_backbone_signal_norm: float = 0.3,
    runtime_backbone_signal_quality: float = 0.28,
    runtime_backbone_signal_strength: float = 0.26,
    runtime_backbone_hook_coverage: float = 0.55,
    fast_memory_signal_norm: float = 0.24,
    fast_memory_runtime_alignment: float = 0.18,
    assistant_response_text: str | None = None,
    user_input: str | None = None,
    extra_outcome_metrics: tuple[tuple[str, float], ...] = (),
) -> DialogueBenchmarkTurn:
    return DialogueBenchmarkTurn(
        turn_index=turn_index,
        wave_id=f"wave-{turn_index}",
        user_input=user_input or f"turn-{turn_index}",
        assistant_response_text=assistant_response_text or f"assistant-response-{turn_index}",
        acceptance_passed=True,
        active_regime=regime,
        active_abstract_action=abstract_action,
        joint_schedule_action=action,
        switch_gate=switch_gate,
        action_family_version=action_family_version,
        prediction_error_magnitude=pe,
        prediction_error_reward=reward,
        task_error=pe * 0.5,
        relationship_error=pe * 0.3,
        regime_error=pe * 0.2,
        action_error=pe * 0.4,
        has_prediction_chain=True,
        bounded_writeback_applied=bounded_writeback_applied,
        reflection_promotion_eligible=reflection_promotion_eligible,
        session_post_completed_job_count=session_post_completed_job_count,
        rare_heavy_recommended=pe_triggered,
        rare_heavy_applied=False,
        evolution_decision=evolution_decision,
        evolution_category=evolution_category,
        cross_session_verdict=cross_session_verdict,
        nested_profile_active=nested_profile_active,
        nested_context_reset_applied=nested_context_reset_applied,
        nested_context_reset_total_count=nested_context_reset_total_count,
        slow_to_fast_init_benefit=slow_to_fast_init_benefit,
        outcome_metrics=(
            ("learning:joint_learning_progress", delayed_metric),
            ("relationship:cross_track_stability", delayed_metric),
            ("learning:default_continual_learning_active", 1.0),
            ("learning:default_owner_writeback_retained", 1.0 if bounded_writeback_applied or session_post_completed_job_count else 0.8),
            ("safety:default_substrate_live_mutation_suppressed", 1.0),
            ("safety:default_continual_rollback_clean", 1.0),
        )
        + extra_outcome_metrics,
        description="synthetic benchmark turn",
        tower_consolidation_count=tower_consolidation_count,
        learned_recall_count=learned_recall_count,
        learned_recall_confidence=learned_recall_confidence,
        learned_recall_core_guided=learned_recall_core_guided,
        memory_tower_depth=memory_tower_depth,
        memory_tower_alignment=memory_tower_alignment,
        memory_tower_profile_id=memory_tower_profile_id,
        slow_to_fast_target_distance_before=slow_to_fast_target_distance_before,
        slow_to_fast_target_distance_after=slow_to_fast_target_distance_after,
        slow_to_fast_target_alignment_gain=slow_to_fast_target_alignment_gain,
        case_memory_surface_active=case_memory_surface_active,
        strategy_playbook_surface_active=strategy_playbook_surface_active,
        experience_fast_prior_surface_active=experience_fast_prior_surface_active,
        experience_consolidation_surface_active=experience_consolidation_surface_active,
        runtime_backbone_evidence_active=runtime_backbone_evidence_active,
        runtime_backbone_signal_norm=runtime_backbone_signal_norm,
        runtime_backbone_signal_quality=runtime_backbone_signal_quality,
        runtime_backbone_signal_strength=runtime_backbone_signal_strength,
        runtime_backbone_hook_coverage=runtime_backbone_hook_coverage,
        fast_memory_signal_norm=fast_memory_signal_norm,
        fast_memory_runtime_alignment=fast_memory_runtime_alignment,
    )


def _synthetic_comprehensive_report_for_dashboard():
    from volvence_zero.agent.dialogue import (
        DialogueArtifactComparisonReport,
        DialogueBenchmarkPathReport,
        DialogueComprehensiveBenchmarkReport,
        DialogueLongitudinalBenchmarkReport,
        DialoguePerturbationBenchmarkReport,
        DialogueReplayRankingReport,
        DialogueReplaySelectionArtifact,
        DialogueSystematicReplayBenchmarkReport,
        OpenDialogueBenchmarkComparisonReport,
        OpenDialogueBenchmarkPathReport,
        OpenDialogueBenchmarkReport,
        _empty_emergence_dashboard,
    )
    from volvence_zero.evaluation import CrossSessionGrowthReport

    case = ScriptedDialogueCase(
        case_id="synthetic-dashboard",
        description="Synthetic dashboard case.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    baseline_case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.24,
                reward=0.22,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.75,
                delayed_metric=0.25,
                pe_triggered=True,
                nested_context_reset_applied=True,
                nested_context_reset_total_count=1,
                slow_to_fast_init_benefit=0.02,
                slow_to_fast_target_distance_before=0.02,
                slow_to_fast_target_distance_after=0.0,
                slow_to_fast_target_alignment_gain=0.02,
                tower_consolidation_count=1,
                memory_tower_depth=5,
                memory_tower_alignment=0.35,
                memory_tower_profile_id="mlp:nested:depth5",
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.08,
                reward=0.04,
                action="full-cycle-pe",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.25,
                delayed_metric=0.65,
                tower_consolidation_count=2,
                memory_tower_depth=5,
                memory_tower_alignment=0.42,
                memory_tower_profile_id="mlp:nested:depth5",
            ),
        ),
    )
    control_case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.18,
                reward=0.10,
                action="evidence-only",
                regime="repair",
                abstract_action="observe",
                switch_gate=0.2,
                delayed_metric=0.15,
                tower_consolidation_count=0,
                memory_tower_depth=3,
                memory_tower_alignment=0.12,
                memory_tower_profile_id="mlp:sequential:depth3",
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.14,
                reward=0.02,
                action="evidence-only",
                regime="repair",
                abstract_action="observe",
                switch_gate=0.1,
                delayed_metric=0.2,
                tower_consolidation_count=0,
                memory_tower_depth=3,
                memory_tower_alignment=0.08,
                memory_tower_profile_id="mlp:sequential:depth3",
            ),
        ),
    )
    baseline_report = DialogueBenchmarkReport(
        case_reports=(baseline_case_report,),
        passed_case_count=1,
        total_case_count=1,
        metric_means=(),
        description="baseline benchmark report",
    )
    baseline_report = replace(
        baseline_report,
        metric_means=tuple((key, value) for key, value in (
            ("passed", float(baseline_case_report.passed)),
            ("prediction_chain_turn_count", float(baseline_case_report.prediction_chain_turn_count)),
            ("pe_triggered_turn_count", float(baseline_case_report.pe_triggered_turn_count)),
            ("delayed_improvement_observed", float(baseline_case_report.delayed_improvement_observed)),
            ("online_learning_turn_count", float(baseline_case_report.online_learning_turn_count)),
            ("bounded_writeback_turn_count", float(baseline_case_report.bounded_writeback_turn_count)),
            ("session_post_completion_turn_count", float(baseline_case_report.session_post_completion_turn_count)),
            ("reflection_promotion_eligible_turn_count", float(baseline_case_report.reflection_promotion_eligible_turn_count)),
            ("rare_heavy_recommended_count", float(baseline_case_report.rare_heavy_recommended_count)),
            ("nested_profile_active_turn_count", float(baseline_case_report.nested_profile_active_turn_count)),
            ("learned_memory_primary_turn_count", float(baseline_case_report.learned_memory_primary_turn_count)),
            ("runtime_backbone_evidence_turn_count", float(baseline_case_report.runtime_backbone_evidence_turn_count)),
            ("mean_runtime_backbone_signal_quality", baseline_case_report.mean_runtime_backbone_signal_quality),
            ("mean_fast_memory_signal_norm", baseline_case_report.mean_fast_memory_signal_norm),
            ("mean_fast_memory_runtime_alignment", baseline_case_report.mean_fast_memory_runtime_alignment),
            ("mean_semantic_spine_coverage", 1.0),
            ("mean_cognitive_loop_readiness", 0.62),
            ("memory_tower_profile_turn_count", float(baseline_case_report.memory_tower_profile_turn_count)),
            ("mean_memory_tower_depth", baseline_case_report.mean_memory_tower_depth),
            ("mean_memory_tower_alignment", baseline_case_report.mean_memory_tower_alignment),
            ("max_tower_consolidation_count", float(baseline_case_report.max_tower_consolidation_count)),
            ("store_nested_context_reset_count", float(baseline_case_report.store_nested_context_reset_count)),
            ("mean_reset_turn_slow_to_fast_init_benefit", baseline_case_report.mean_reset_turn_slow_to_fast_init_benefit),
            ("mean_reset_turn_slow_to_fast_target_distance_before", baseline_case_report.mean_reset_turn_slow_to_fast_target_distance_before),
            ("mean_reset_turn_slow_to_fast_target_distance_after", baseline_case_report.mean_reset_turn_slow_to_fast_target_distance_after),
            ("mean_reset_turn_slow_to_fast_target_alignment_gain", baseline_case_report.mean_reset_turn_slow_to_fast_target_alignment_gain),
        )),
    )
    control_report = replace(
        baseline_report,
        case_reports=(control_case_report,),
        metric_means=(
            ("passed", float(control_case_report.passed)),
            ("prediction_chain_turn_count", float(control_case_report.prediction_chain_turn_count)),
            ("pe_triggered_turn_count", float(control_case_report.pe_triggered_turn_count)),
            ("delayed_improvement_observed", float(control_case_report.delayed_improvement_observed)),
            ("online_learning_turn_count", float(control_case_report.online_learning_turn_count)),
            ("bounded_writeback_turn_count", float(control_case_report.bounded_writeback_turn_count)),
            ("session_post_completion_turn_count", float(control_case_report.session_post_completion_turn_count)),
            ("reflection_promotion_eligible_turn_count", float(control_case_report.reflection_promotion_eligible_turn_count)),
            ("rare_heavy_recommended_count", float(control_case_report.rare_heavy_recommended_count)),
            ("nested_profile_active_turn_count", float(control_case_report.nested_profile_active_turn_count)),
            ("learned_memory_primary_turn_count", float(control_case_report.learned_memory_primary_turn_count)),
            ("runtime_backbone_evidence_turn_count", float(control_case_report.runtime_backbone_evidence_turn_count)),
            ("mean_runtime_backbone_signal_quality", control_case_report.mean_runtime_backbone_signal_quality),
            ("mean_fast_memory_signal_norm", control_case_report.mean_fast_memory_signal_norm),
            ("mean_fast_memory_runtime_alignment", control_case_report.mean_fast_memory_runtime_alignment),
            ("memory_tower_profile_turn_count", float(control_case_report.memory_tower_profile_turn_count)),
            ("mean_memory_tower_depth", control_case_report.mean_memory_tower_depth),
            ("mean_memory_tower_alignment", control_case_report.mean_memory_tower_alignment),
            ("max_tower_consolidation_count", float(control_case_report.max_tower_consolidation_count)),
            ("store_nested_context_reset_count", float(control_case_report.store_nested_context_reset_count)),
            ("mean_reset_turn_slow_to_fast_init_benefit", control_case_report.mean_reset_turn_slow_to_fast_init_benefit),
            ("mean_reset_turn_slow_to_fast_target_distance_before", control_case_report.mean_reset_turn_slow_to_fast_target_distance_before),
            ("mean_reset_turn_slow_to_fast_target_distance_after", control_case_report.mean_reset_turn_slow_to_fast_target_distance_after),
            ("mean_reset_turn_slow_to_fast_target_alignment_gain", control_case_report.mean_reset_turn_slow_to_fast_target_alignment_gain),
        ),
        description="control benchmark report",
    )
    canonical_comparison = DialogueBenchmarkComparisonReport(
        baseline_label="pe-eta",
        path_reports=(
            DialogueBenchmarkPathReport(path_label="pe-eta", benchmark_report=baseline_report, description="baseline"),
            DialogueBenchmarkPathReport(path_label="pe-drive-off", benchmark_report=control_report, description="control"),
        ),
        case_deltas_from_baseline=(),
        metric_deltas_from_baseline=(
            (
                "pe-drive-off",
                (
                    ("passed", -1.0),
                    ("pe_triggered_turn_count", -1.0),
                    ("delayed_improvement_observed", -1.0),
                    ("stability_after_recovery_score", -0.2),
                    ("mean_prediction_error", 0.1),
                    ("mean_memory_tower_depth", -2.0),
                    ("mean_memory_tower_alignment", -0.28),
                    ("max_tower_consolidation_count", -2.0),
                ),
            ),
        ),
        description="synthetic canonical comparison",
    )
    open_report = OpenDialogueBenchmarkReport(
        case_reports=(),
        passed_case_count=1,
        total_case_count=1,
        metric_means=(
            ("passed", 1.0),
            ("runtime_backbone_evidence_turn_count", 2.0),
            ("mean_runtime_backbone_signal_quality", 0.28),
            ("mean_fast_memory_runtime_alignment", 0.18),
            ("mean_memory_tower_depth", 4.0),
            ("mean_memory_tower_alignment", 0.25),
            ("max_tower_consolidation_count", 1.0),
        ),
        description="synthetic open report",
    )
    open_comparison = OpenDialogueBenchmarkComparisonReport(
        baseline_label="pe-eta",
        path_reports=(
            OpenDialogueBenchmarkPathReport(path_label="pe-eta", benchmark_report=open_report, description="baseline"),
            OpenDialogueBenchmarkPathReport(path_label="pe-drive-off", benchmark_report=open_report, description="control"),
        ),
        case_deltas_from_baseline=(),
        metric_deltas_from_baseline=(
            (
                "pe-drive-off",
                (
                    ("passed", -0.5),
                    ("pe_triggered_turn_count", -0.5),
                    ("delayed_improvement_observed", -0.5),
                    ("late_episode_stability_score", -0.1),
                    ("mean_prediction_error", 0.05),
                    ("mean_memory_tower_depth", -1.0),
                    ("mean_memory_tower_alignment", -0.1),
                    ("max_tower_consolidation_count", -1.0),
                ),
            ),
        ),
        description="synthetic open comparison",
    )
    essence_report = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=baseline_report,
        comparison_report=canonical_comparison,
    )
    essence_acceptance = evaluate_dialogue_nl_essence_acceptance(essence_report)
    longitudinal_report = DialogueLongitudinalBenchmarkReport(
        case_reports=(baseline_case_report,),
        session_reports=(),
        cross_session_report=CrossSessionGrowthReport(
            window_trends=(),
            family_persistence=0.6,
            regime_effectiveness_delta=0.1,
            verdict="stable",
            description="synthetic cross-session report",
        ),
        description="synthetic longitudinal report",
    )
    perturbation_report = DialoguePerturbationBenchmarkReport(
        variant_cases=(),
        ablation_report=canonical_comparison,
        description="synthetic perturbation report",
    )
    systematic_report = DialogueSystematicReplayBenchmarkReport(
        variant_cases=(),
        perturbation_report=perturbation_report,
        replay_ranking_report=DialogueReplayRankingReport(entries=(), description="synthetic replay ranking"),
        description="synthetic systematic report",
    )
    selection_artifact = DialogueReplaySelectionArtifact(
        artifact_id="synthetic-selection",
        selected_variants=(),
        ranking_entries=(),
        description="synthetic selection artifact",
    )
    artifact_comparison_report = DialogueArtifactComparisonReport(
        selection_artifact=selection_artifact,
        candidate_reports=(),
        chosen_candidate_label=None,
        chosen_accepted=False,
        description="synthetic artifact comparison",
    )
    provisional = DialogueComprehensiveBenchmarkReport(
        profile_labels=("pe-eta", "pe-drive-off"),
        canonical_ablation_report=canonical_comparison,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        open_ablation_report=open_comparison,
        systematic_replay_report=systematic_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        emergence_dashboard=_empty_emergence_dashboard(baseline_label="pe-eta"),
        description="synthetic comprehensive report",
    )
    return replace(
        provisional,
        emergence_dashboard=build_dialogue_emergence_dashboard(provisional),
    )


def test_dialogue_benchmark_exposes_default_scripted_cases():
    case_ids = tuple(case.case_id for case in DEFAULT_DIALOGUE_PROOF_CASES)

    assert case_ids == ("repair", "task_clarification", "repeated_failure", "goal_drift")
    assert all(len(case.user_inputs) >= 6 for case in DEFAULT_DIALOGUE_PROOF_CASES)


def test_dialogue_benchmark_exposes_default_ablation_profiles():
    assert default_dialogue_ablation_profiles() == ("pe-eta", "pe-drive-off", "eta-off", "timescale-off")


def test_phase2_shadow_profiles_can_build_standard_runners():
    for profile_label in default_phase2_shadow_evidence_profiles():
        runner = build_standard_dialogue_runner(
            profile_label=profile_label,
            case=DEFAULT_DIALOGUE_PROOF_CASES[0],
        )
        assert runner.session_id.startswith(f"dialogue-ablation:{profile_label}:")


def test_phase2_shadow_profiles_are_explicit_not_default():
    phase2_profiles = default_phase2_shadow_evidence_profiles()

    assert phase2_profiles == (
        "cpd-beta-switch",
        "counterfactual-credit",
        "tom-owner",
        "persona-geometry-readout",
    )
    assert not (set(phase2_profiles) & set(default_dialogue_ablation_profiles()))
    assert not (set(phase2_profiles) & set(default_dialogue_strong_proof_profiles()))


def test_phase2_shadow_evidence_smoke_runs_one_case():
    report = asyncio.run(
        run_phase2_shadow_evidence_smoke(
            cases=(DEFAULT_DIALOGUE_PROOF_CASES[0],),
            runner_factory=lambda _profile_label, case: _synthetic_runner(case),
        )
    )
    path_labels = tuple(path.path_label for path in report.path_reports)

    assert path_labels == ("pe-eta",) + default_phase2_shadow_evidence_profiles()
    assert report.baseline_label == "pe-eta"
    for path in report.path_reports:
        assert path.benchmark_report.total_case_count == 1
        metric_names = {name for name, _ in path.benchmark_report.metric_means}
        assert "mean_persona_geometry_drift" in metric_names
        assert "cpd_beta_switch_recommended_count" in metric_names
        assert "mean_least_control_score" in metric_names
        assert "tom_distinct_interlocutor_max" in metric_names


def test_phase2_candidate_metric_means_are_exported_from_case_summary():
    turn = DialogueBenchmarkTurn(
        turn_index=0,
        wave_id="w0",
        user_input="u",
        assistant_response_text="a",
        acceptance_passed=True,
        active_regime="r",
        active_abstract_action="act",
        joint_schedule_action="none",
        switch_gate=0.0,
        action_family_version=0,
        prediction_error_magnitude=0.0,
        prediction_error_reward=0.0,
        task_error=0.0,
        relationship_error=0.0,
        regime_error=0.0,
        action_error=0.0,
        has_prediction_chain=False,
        bounded_writeback_applied=False,
        reflection_promotion_eligible=False,
        session_post_completed_job_count=0,
        rare_heavy_recommended=False,
        rare_heavy_applied=False,
        evolution_decision=None,
        evolution_category=None,
        cross_session_verdict="",
        nested_profile_active=False,
        nested_context_reset_applied=False,
        nested_context_reset_total_count=0,
        slow_to_fast_init_benefit=0.0,
        outcome_metrics=(
            ("safety:persona_geometry_drift", 0.2),
            ("safety:persona_regime_geometry_alignment", 0.8),
            ("abstraction:cpd_beta_switch_recommended", 1.0),
            ("abstraction:cpd_pe_spike_score", 0.7),
            ("abstraction:cpd_reward_shift_score", 0.6),
            ("learning:least_control_score", 0.5),
            ("learning:least_control_effort", 0.25),
            ("learning:counterfactual_readout_count", 1.0),
            ("relationship:tom_distinct_interlocutor_count", 2.0),
            ("relationship:tom_record_total", 4.0),
        ),
        description="phase2 candidate metrics",
    )
    report = DialogueBenchmarkCaseReport(
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
        turns=(turn,),
        prediction_chain_turn_count=0,
        high_pe_turn_count=0,
        pe_schedule_due_turn_count=0,
        pe_triggered_turn_count=0,
        explicit_pe_schedule_turn_count=0,
        carryover_credit_turn_count=0,
        schedule_label_consistency=1.0,
        recovery_lag_turns=0,
        pressure_localization_score=0.0,
        over_response_cost=0.0,
        pressure_response_precision=0.0,
        pressure_response_recall=0.0,
        stability_after_recovery_score=0.0,
        online_learning_turn_count=0,
        bounded_writeback_turn_count=0,
        reflection_promotion_eligible_turn_count=0,
        session_post_completion_turn_count=0,
        rare_heavy_recommended_count=0,
        rare_heavy_applied_count=0,
        rare_heavy_pre_import_pass_count=0,
        rare_heavy_pre_import_reject_count=0,
        mean_rare_heavy_pre_import_score_delta=0.0,
        mean_rare_heavy_candidate_alignment=0.0,
        max_rare_heavy_candidate_adapter_parameter_count=0,
        evolution_judge_turn_count=0,
        evolution_judge_rollback_count=0,
        evolution_judge_structural_allow_count=0,
        nested_profile_active_turn_count=0,
        nested_context_reset_count=0,
        store_nested_context_reset_count=0,
        boundary_reset_observed_on_first_turn=False,
        first_turn_slow_to_fast_init_benefit=0.0,
        mean_reset_turn_slow_to_fast_init_benefit=0.0,
        mean_slow_to_fast_init_benefit=0.0,
        temporal_change_count=0,
        case_memory_surface_turn_count=0,
        strategy_playbook_surface_turn_count=0,
        experience_fast_prior_surface_turn_count=0,
        experience_consolidation_surface_turn_count=0,
        learned_memory_primary_turn_count=0,
        core_guided_recall_turn_count=0,
        mean_learned_recall_confidence=0.0,
        max_artifact_consolidation_count=0,
        max_tower_consolidation_count=0,
        mean_memory_tower_depth=0.0,
        mean_memory_tower_alignment=0.0,
        memory_tower_profile_turn_count=0,
        delayed_improvement_observed=False,
        acceptance_checks=(),
        passed=True,
        reasons=(),
        description="phase2 candidate report",
    )
    metrics = dict(_case_summary_metrics(report))

    assert metrics["mean_persona_geometry_drift"] == 0.2
    assert metrics["mean_persona_regime_geometry_alignment"] == 0.8
    assert metrics["cpd_beta_switch_recommended_count"] == 1.0
    assert metrics["mean_cpd_pe_spike_score"] == 0.7
    assert metrics["mean_least_control_score"] == 0.5
    assert metrics["counterfactual_readout_turn_count"] == 1.0
    assert metrics["tom_distinct_interlocutor_max"] == 2.0
    assert metrics["tom_record_total_max"] == 4.0


def test_dialogue_benchmark_exposes_default_strong_proof_profiles():
    assert default_dialogue_strong_proof_profiles() == (
        "pe-eta",
        "pe-eta-no-semantic-label",
        "pe-eta-no-reflection-cache",
        "pe-eta-pe-readout-only",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )


def test_dialogue_benchmark_exposes_default_comprehensive_profiles():
    assert default_dialogue_comprehensive_profiles() == (
        "pe-eta",
        "pe-eta-online-only",
        "pe-eta-no-writeback",
        "pe-eta-no-rare-heavy",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )


def test_dialogue_benchmark_exposes_default_case_variants():
    variants = dialogue_case_variants()

    assert len(variants) == len(DEFAULT_DIALOGUE_CASE_VARIANTS)
    labels = {(variant.base_case_id, variant.variant_label) for variant in variants}
    assert ("repair", "wording_shift") in labels
    assert ("goal_drift", "pressure_shift_late") in labels
    assert all(variant.case.case_id.startswith(f"{variant.base_case_id}__") for variant in variants)


def test_dialogue_benchmark_exposes_default_paraphrase_families():
    families = dialogue_paraphrase_families()

    assert len(families) == len(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES)
    family_ids = {(family.base_case_id, family.family_label) for family in families}
    assert ("repair", "repair_family") in family_ids
    assert ("goal_drift", "goal_drift_family") in family_ids


def test_dialogue_benchmark_exposes_default_open_dialogue_scenarios():
    scenarios = open_dialogue_scenarios()

    assert scenarios == DEFAULT_OPEN_DIALOGUE_SCENARIOS
    assert tuple(scenario.scenario_id for scenario in scenarios[:4]) == (
        "open_repair",
        "open_repair_family",
        "open_repair_heldout",
        "open_clarification",
    )
    assert any(scenario.split == "open_heldout" for scenario in scenarios)
    assert get_open_dialogue_scenario("intelligence_demo").family_id == "introspective_chat"
    emotional_decision = get_open_dialogue_scenario("open_emotional_decision_support")
    assert emotional_decision.expected_emotional_decision_support is True
    assert emotional_decision.pressure_shape == "support-before-decision"


def test_dialogue_benchmark_exposes_default_open_ablation_profiles():
    assert default_open_dialogue_ablation_profiles() == ("pe-eta", "pe-drive-off", "eta-off")


def test_build_deterministic_user_simulator_is_reproducible_for_same_seed():
    scenario = get_open_dialogue_scenario("open_repair")
    simulator_a = build_deterministic_user_simulator(scenario_id=scenario.scenario_id, seed=7, max_turns=4)
    simulator_b = build_deterministic_user_simulator(scenario_id=scenario.scenario_id, seed=7, max_turns=4)
    opening_a = simulator_a.next_turn()
    opening_b = simulator_b.next_turn()
    assert opening_a == opening_b

    escalation_turn = _benchmark_turn(
        turn_index=1,
        pe=0.45,
        reward=-0.12,
        action="evidence-only",
        regime="support",
        abstract_action="stay-cold",
        switch_gate=0.05,
        delayed_metric=0.1,
    )
    followup_a = simulator_a.next_turn(last_turn=escalation_turn)
    followup_b = simulator_b.next_turn(last_turn=escalation_turn)

    assert followup_a == followup_b
    assert simulator_a.episode_state == simulator_b.episode_state


def test_transcript_only_user_simulator_ignores_runtime_telemetry():
    scenario = get_open_dialogue_scenario("open_repair")
    simulator_a = TranscriptOnlyUserSimulator(scenario=scenario, seed=11)
    simulator_b = TranscriptOnlyUserSimulator(scenario=scenario, seed=11)
    opening_a = simulator_a.next_turn()
    opening_b = simulator_b.next_turn()
    assert opening_a == opening_b

    telemetry_heavy_turn = _benchmark_turn(
        turn_index=1,
        pe=0.95,
        reward=-0.9,
        action="rl+ssl-pe",
        regime="repair",
        abstract_action="high-telemetry",
        switch_gate=0.95,
        delayed_metric=0.1,
        bounded_writeback_applied=True,
        reflection_promotion_eligible=True,
        pe_triggered=True,
    )
    telemetry_light_turn = _benchmark_turn(
        turn_index=1,
        pe=0.0,
        reward=0.0,
        action="evidence-only",
        regime="repair",
        abstract_action="low-telemetry",
        switch_gate=0.0,
        delayed_metric=0.1,
        bounded_writeback_applied=False,
        reflection_promotion_eligible=False,
        pe_triggered=False,
    )

    assert simulator_a.next_turn(last_turn=telemetry_heavy_turn) == simulator_b.next_turn(last_turn=telemetry_light_turn)
    assert simulator_a.episode_state.user_policy_kind == "transcript-only"
    assert simulator_b.episode_state.user_policy_kind == "transcript-only"


def test_build_open_dialogue_case_report_uses_open_acceptance_surface():
    scenario = OpenDialogueScenario(
        scenario_id="open-proof",
        family_id="synthetic",
        split="open_core",
        description="Synthetic open scenario for report coverage.",
        opening_turns=("start",),
        escalation_turns=("push",),
        stabilization_turns=("stabilize",),
        consolidation_turns=("compress",),
        max_turns=4,
        hidden_perturbation_family="hidden_family_alpha",
        expected_repair_observable=True,
        expected_adaptation_signal=True,
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.32,
            reward=-0.08,
            action="evidence-only",
            regime="repair",
            abstract_action="probe",
            switch_gate=0.1,
            delayed_metric=0.1,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.28,
            reward=-0.07,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="repair",
            switch_gate=0.4,
            delayed_metric=0.2,
            bounded_writeback_applied=True,
            reflection_promotion_eligible=True,
            session_post_completed_job_count=1,
            pe_triggered=True,
            assistant_response_text="I misunderstood you; let's slow down and repair this.",
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.06,
            reward=-0.01,
            action="evidence-only",
            regime="support",
            abstract_action="stabilize",
            switch_gate=0.3,
            delayed_metric=0.5,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.03,
            reward=0.01,
            action="evidence-only",
            regime="support",
            abstract_action="consolidate",
            switch_gate=0.2,
            delayed_metric=0.7,
        ),
    )

    report = build_open_dialogue_case_report(
        scenario=scenario,
        final_episode_state=OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            turn_index=4,
            pressure_level=0,
            adaptive_response_count=1,
            calm_turn_count=2,
            last_stage="consolidation",
            completed=True,
            stop_reason="stable-consolidation",
            user_policy_kind="transcript-only",
        ),
        turns=turns,
    )

    assert report.final_episode_state.completed is True
    assert report.pe_triggered_turn_count > 0
    assert report.online_learning_turn_count > 0
    assert report.late_episode_stability_score >= 0.5
    check_map = dict(report.acceptance_checks)
    assert check_map["episode-runs-to-completion"] is True
    assert check_map["multi-timescale-evidence-observed"] is True
    assert report.passed is True
    metrics = dict(_open_case_summary_metrics(report))
    assert metrics["transcript_only_user_policy"] == 1.0
    assert metrics["hidden_label_leak_count"] == 0.0
    assert metrics["repair_observable"] == 1.0
    assert metrics["runtime_adaptation_evidence_observed"] == 1.0


def test_open_dialogue_emotional_decision_support_gate_is_fail_closed() -> None:
    scenario = OpenDialogueScenario(
        scenario_id="open-emotional-decision-proof",
        family_id="emotional_decision_support",
        split="open_core",
        description="Synthetic emotional decision support scenario.",
        opening_turns=("start",),
        escalation_turns=("push",),
        stabilization_turns=("stabilize",),
        consolidation_turns=("done",),
        max_turns=3,
        expected_adaptation_signal=True,
        expected_emotional_decision_support=True,
    )
    base_turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.32,
            reward=-0.08,
            action="ssl-only-pe",
            regime="problem_solving",
            abstract_action="generic_problem_solving",
            switch_gate=0.4,
            delayed_metric=0.2,
            bounded_writeback_applied=True,
            pe_triggered=True,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.12,
            reward=-0.02,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="generic_problem_solving",
            switch_gate=0.2,
            delayed_metric=0.5,
        ),
    )
    failing_report = build_open_dialogue_case_report(
        scenario=scenario,
        final_episode_state=OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            turn_index=2,
            completed=True,
            stop_reason="stable-consolidation",
        ),
        turns=base_turns,
    )

    assert failing_report.passed is False
    assert "emotional-decision-support-observed" in failing_report.reasons

    passing_turns = tuple(
        replace(
            turn,
            active_regime="emotional_support",
            active_abstract_action="clarify_values_then_options",
            outcome_metrics=turn.outcome_metrics
            + (
                ("interaction:support_presence", 0.62),
                ("interaction:warmth", 0.58),
                ("interaction:support_before_decision_pressure", 0.66),
                ("abstraction:emotional_decision_action_family", 1.0),
            ),
        )
        for turn in base_turns
    )
    passing_report = build_open_dialogue_case_report(
        scenario=scenario,
        final_episode_state=OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            turn_index=2,
            completed=True,
            stop_reason="stable-consolidation",
        ),
        turns=passing_turns,
    )

    assert "emotional-decision-support-observed" not in passing_report.reasons
    metrics = dict(_open_case_summary_metrics(passing_report))
    assert metrics["mean_support_before_decision_pressure"] >= 0.35
    assert metrics["mean_emotional_decision_action_family"] >= 0.25


def test_open_dialogue_case_report_tracks_hidden_label_leak_as_diagnostic() -> None:
    scenario = OpenDialogueScenario(
        scenario_id="open-leak-proof",
        family_id="synthetic",
        split="open_heldout",
        description="Synthetic hidden-label leak scenario.",
        opening_turns=("start",),
        escalation_turns=("push",),
        stabilization_turns=("stabilize",),
        consolidation_turns=("done",),
        hidden_perturbation_family="hidden_family_beta",
        expected_repair_observable=False,
        expected_adaptation_signal=False,
    )
    report = build_open_dialogue_case_report(
        scenario=scenario,
        final_episode_state=OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            turn_index=1,
            completed=True,
            stop_reason="complete",
            user_policy_kind="transcript-only",
        ),
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.1,
                reward=0.0,
                action="evidence-only",
                regime="support",
                abstract_action="support",
                switch_gate=0.0,
                delayed_metric=0.0,
                user_input="This hidden_family_beta label must never reach runtime.",
            ),
        ),
    )

    check_map = dict(report.acceptance_checks)
    assert report.hidden_label_leak_count == 1
    assert "hidden-perturbation-label-not-leaked" not in check_map
    assert "hidden-perturbation-label-not-leaked" not in report.reasons


def test_claim_beyond_scripted_requires_structured_open_repair(tmp_path) -> None:
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    provenance = PaperSuiteProvenance(
        git_sha="test",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        dependency_digest="test",
        manifest_hash="test",
        runtime_descriptor=(),
        description="Synthetic provenance for beyond-scripted claim test.",
    )
    scenario = OpenDialogueScenario(
        scenario_id="open-heldout-repair-proof",
        family_id="synthetic",
        split="open_heldout",
        description="Synthetic heldout repair case.",
        opening_turns=("start",),
        escalation_turns=("push",),
        stabilization_turns=("repair",),
        consolidation_turns=("done",),
        hidden_perturbation_family="hidden_family_gamma",
        expected_repair_observable=True,
        expected_adaptation_signal=True,
    )
    case_report = build_open_dialogue_case_report(
        scenario=scenario,
        final_episode_state=OpenDialogueEpisodeState(
            scenario_id=scenario.scenario_id,
            turn_index=2,
            completed=True,
            stop_reason="stable-consolidation",
            user_policy_kind="transcript-only",
        ),
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.35,
                reward=-0.2,
                action="evidence-only",
                regime="problem_solving",
                abstract_action="probe",
                switch_gate=0.1,
                delayed_metric=0.1,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.10,
                reward=0.1,
                action="ssl-only-pe",
                regime="repair_and_deescalation",
                abstract_action="repair_controller",
                switch_gate=0.4,
                delayed_metric=0.6,
                pe_triggered=True,
                bounded_writeback_applied=True,
                assistant_response_text="I misunderstood you; let's back up and repair this.",
            ),
        ),
    )
    open_report = OpenDialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=1,
        total_case_count=1,
        metric_means=_open_case_summary_metrics(case_report),
        description="synthetic open repair report",
    )
    open_comparison = OpenDialogueBenchmarkComparisonReport(
        baseline_label="pe-eta",
        path_reports=(
            OpenDialogueBenchmarkPathReport(path_label="pe-eta", benchmark_report=open_report, description="baseline"),
            OpenDialogueBenchmarkPathReport(path_label="pe-drive-off", benchmark_report=open_report, description="control"),
        ),
        case_deltas_from_baseline=(),
        metric_deltas_from_baseline=(),
        description="synthetic open comparison",
    )
    reference = replace(
        _synthetic_comprehensive_report_for_dashboard(),
        open_ablation_report=open_comparison,
    )
    aggregate = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        reference_run_report=reference,
        primary_metric_summaries=(
            MetricIntervalSummary(
                metric_name="perturbation_pass_rate_pe_eta",
                sample_count=1,
                mean=1.0,
                std=0.0,
                stderr=0.0,
                ci_low=1.0,
                ci_high=1.0,
                min_value=1.0,
                max_value=1.0,
                description="Synthetic perturbation pass.",
            ),
        ),
        secondary_metric_summaries=(),
        pairwise_effects=(
            PairwiseMetricEffect(
                metric_name="open_pass_rate",
                candidate_label="pe-eta",
                control_label="pe-drive-off",
                sample_count=1,
                mean_delta=0.5,
                std_delta=0.0,
                stderr_delta=0.0,
                ci_low=0.2,
                ci_high=0.5,
                effect_size=1.0,
                description="Synthetic positive open gap.",
            ),
        ),
        description="Synthetic aggregate for beyond-scripted repair claim.",
    )

    export_dialogue_paper_suite_artifact_bundle(
        aggregate,
        output_dir=tmp_path / "beyond-scripted",
    )
    payload = json.loads(
        (tmp_path / "beyond-scripted" / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    claim = next(
        verdict for verdict in payload["claim_verdicts"] if verdict["claim_id"] == "claim_beyond_scripted_canonical"
    )

    assert claim["status"] == "retain"
    evidence = {name: value for name, value in claim["evidence"]}
    assert evidence["has_open_heldout"] == 1.0
    assert evidence["transcript_only_open_case_count"] == 1.0
    assert evidence["open_hidden_label_leak_count"] == 0.0
    assert evidence["open_repair_observable_count"] == 1.0
    assert evidence["open_runtime_adaptation_evidence_count"] == 1.0


def test_run_open_dialogue_case_reuses_runner_turn_path():
    scenario = get_open_dialogue_scenario("open_repair")
    report = asyncio.run(
        run_open_dialogue_case(
            scenario=scenario,
            runner=_synthetic_runner(
                ScriptedDialogueCase(
                    case_id="open-repair-adapter",
                    description="Synthetic runner shell for open dialogue.",
                    user_inputs=("seed",),
                )
            ),
            seed=3,
            max_turns=3,
        )
    )

    assert len(report.turns) == 3
    assert report.final_episode_state.completed is True
    assert report.final_episode_state.stop_reason == "max-turns"
    assert all(turn.wave_id.startswith("wave-") for turn in report.turns)
    assert report.session_post_completion_turn_count >= 1
    assert report.turns[-1].session_post_completed_job_count >= 1


def test_run_open_dialogue_benchmark_aggregates_open_reports():
    benchmark = asyncio.run(
        run_open_dialogue_benchmark(
            scenarios=(get_open_dialogue_scenario("open_repair"),),
            runner_factory=lambda scenario: _synthetic_runner(
                ScriptedDialogueCase(
                    case_id=f"open-benchmark:{scenario.scenario_id}",
                    description="Synthetic open benchmark runner shell.",
                    user_inputs=("seed",),
                )
            ),
            seed=2,
        )
    )

    assert benchmark.total_case_count == 1
    assert len(benchmark.case_reports) == 1
    assert "episode_runs_to_completion" in dict(benchmark.metric_means)


def test_run_open_dialogue_ablation_benchmark_compares_small_profile_matrix():
    scenario = get_open_dialogue_scenario("open_repair")
    comparison = asyncio.run(
        run_open_dialogue_ablation_benchmark(
            scenarios=(scenario,),
            profile_labels=default_open_dialogue_ablation_profiles(),
            runner_factory=_synthetic_open_ablation_runner,
            seed=5,
        )
    )

    assert comparison.baseline_label == "pe-eta"
    assert tuple(path.path_label for path in comparison.path_reports) == (
        "pe-eta",
        "pe-drive-off",
        "eta-off",
    )
    assert len(comparison.case_deltas_from_baseline) == 1
    scenario_id, per_path = comparison.case_deltas_from_baseline[0]
    assert scenario_id == scenario.scenario_id
    assert {label for label, _ in per_path} == {"pe-eta", "pe-drive-off", "eta-off"}
    metric_deltas = dict(comparison.metric_deltas_from_baseline)
    assert "pe-drive-off" in metric_deltas
    assert "eta-off" in metric_deltas
    assert "pe_triggered_turn_count" in dict(metric_deltas["pe-drive-off"])


def test_open_ablation_runner_respects_profile_runtime_semantics():
    scenario = get_open_dialogue_scenario("open_repair")
    pe_drive_runner = _synthetic_open_ablation_runner("pe-drive-off", scenario)
    eta_off_runner = _synthetic_open_ablation_runner("eta-off", scenario)

    assert pe_drive_runner._external_prediction_error_drive is False
    assert eta_off_runner._external_prediction_error_drive is False
    assert eta_off_runner._joint_schedule.ssl_interval == 0
    assert eta_off_runner._joint_schedule.rl_interval == 0


def test_run_dialogue_pe_eta_case_collects_pe_and_eta_trajectories():
    case = ScriptedDialogueCase(
        case_id="mini-proof",
        description="Short scripted case for integration coverage.",
        user_inputs=(
            "I need help but the first answer was not enough.",
            "Try again and adjust to the feedback.",
            "Now tell me what changed in your frame.",
        ),
    )

    report = asyncio.run(
        run_dialogue_pe_eta_case(
            case=case,
            runner=_synthetic_runner(case),
        )
    )

    assert report.case.case_id == "mini-proof"
    assert len(report.turns) == 3
    assert report.prediction_chain_turn_count >= 1
    assert isinstance(report.turns[0].outcome_metrics, tuple)
    assert report.turns[0].joint_schedule_action
    assert report.turns[0].nested_profile_active is True
    assert report.turns[0].case_memory_surface_active is True
    assert report.turns[0].strategy_playbook_surface_active is True
    assert report.turns[0].experience_fast_prior_surface_active is True
    assert report.turns[0].experience_consolidation_surface_active is True
    assert report.session_post_completion_turn_count >= 1
    assert report.turns[-1].session_post_completed_job_count >= 1


def test_run_dialogue_pe_eta_benchmark_runs_complete_scripted_suite():
    report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            runner_factory=_synthetic_runner,
        )
    )

    assert report.total_case_count == len(DEFAULT_DIALOGUE_PROOF_CASES)
    assert len(report.case_reports) == len(DEFAULT_DIALOGUE_PROOF_CASES)
    assert report.description
    metric_means = dict(report.metric_means)
    assert "bounded_writeback_turn_count" in metric_means
    assert "rare_heavy_recommended_count" in metric_means
    assert "case_memory_surface_turn_count" in metric_means
    assert "strategy_playbook_surface_turn_count" in metric_means
    assert "experience_fast_prior_surface_turn_count" in metric_means
    assert "experience_consolidation_surface_turn_count" in metric_means
    assert metric_means["case_memory_surface_turn_count"] > 0.0
    assert metric_means["strategy_playbook_surface_turn_count"] > 0.0
    assert metric_means["experience_fast_prior_surface_turn_count"] > 0.0
    assert metric_means["experience_consolidation_surface_turn_count"] > 0.0
    for case_report in report.case_reports:
        assert len(case_report.turns) == len(case_report.case.user_inputs)
        assert isinstance(case_report.acceptance_checks, tuple)
        assert case_report.case_memory_surface_turn_count > 0
        assert case_report.strategy_playbook_surface_turn_count > 0
        assert case_report.experience_fast_prior_surface_turn_count > 0
        assert case_report.experience_consolidation_surface_turn_count > 0


def test_run_dialogue_pe_eta_ablation_benchmark_collects_path_deltas():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=_synthetic_ablation_runner,
        )
    )

    assert report.baseline_label == "pe-eta"
    assert len(report.path_reports) == 4
    assert len(report.case_deltas_from_baseline) == 2
    case_id, path_deltas = report.case_deltas_from_baseline[0]
    assert case_id in {"repair", "task_clarification"}
    delta_map = dict(path_deltas)
    assert "pe-eta" in delta_map
    assert "pe-drive-off" in delta_map
    assert "eta-off" in delta_map
    assert "timescale-off" in delta_map
    assert dict(delta_map["pe-eta"])["passed"] == 0.0
    assert dict(delta_map["pe-drive-off"])["pe_triggered_turn_count"] <= 0.0
    assert "recovery_lag_turns" in dict(delta_map["pe-eta"])
    assert "pressure_localization_score" in dict(delta_map["pe-eta"])
    assert "over_response_cost" in dict(delta_map["pe-eta"])
    assert "pressure_response_precision" in dict(delta_map["pe-eta"])
    assert "pressure_response_recall" in dict(delta_map["pe-eta"])
    assert "stability_after_recovery_score" in dict(delta_map["pe-eta"])
    profile_means = dict(report.metric_deltas_from_baseline)
    assert "pe-eta" in profile_means
    assert "pe-drive-off" in profile_means
    assert "bounded_writeback_turn_count" in dict(profile_means["pe-drive-off"])


def test_run_dialogue_pe_eta_perturbation_benchmark_collects_variant_reports():
    report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    assert len(report.variant_cases) == 2
    assert report.ablation_report.baseline_label == "pe-eta"
    assert len(report.ablation_report.path_reports) == 4
    variant_ids = {variant.case.case_id for variant in report.variant_cases}
    case_ids = {
        case_id
        for case_id, _ in report.ablation_report.case_deltas_from_baseline
    }
    assert case_ids == variant_ids


def test_generate_stochastic_dialogue_case_variants_is_deterministic():
    first = generate_stochastic_dialogue_case_variants(seeds=(0, 1))
    second = generate_stochastic_dialogue_case_variants(seeds=(0, 1))

    assert tuple(variant.case.case_id for variant in first) == tuple(variant.case.case_id for variant in second)
    assert all("__seed_" in variant.case.case_id for variant in first)
    assert len(first) == len(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES) * 2


def test_build_dialogue_replay_ranking_report_sorts_by_diagnostic_score():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )

    assert len(ranking.entries) == 2
    assert ranking.entries[0].diagnostic_score >= ranking.entries[1].diagnostic_score
    assert ranking.entries[0].gap_vs_eta_no_pe >= 0.0


def test_build_dialogue_replay_selection_artifact_keeps_top_k_entries():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:3],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:3],
        ablation_report=perturbation_report.ablation_report,
    )
    artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:3],
        replay_ranking_report=ranking,
        top_k=2,
    )

    assert artifact.artifact_id == "dialogue-replay-selection"
    assert len(artifact.selected_variants) == 2
    assert len(artifact.ranking_entries) == 2


def test_build_replay_selection_training_traces_matches_selected_variants():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    traces = build_replay_selection_training_traces(artifact)

    assert len(traces) == 2
    assert traces[0].trace_id.startswith("replay-selection:")


def test_train_rare_heavy_artifact_from_replay_selection_exports_artifact():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    rare_heavy_artifact = train_rare_heavy_artifact_from_replay_selection(artifact)

    assert rare_heavy_artifact.artifact_id.endswith(":rare-heavy")


def test_run_replay_selection_artifact_acceptance_benchmark_returns_case_reports():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
        )
    )

    assert len(acceptance_report.case_reports) == 2
    assert acceptance_report.artifact.artifact_id.endswith(":rare-heavy")
    assert acceptance_report.decision.accepted is False
    assert acceptance_report.decision.rollback_applied is True
    assert all(report.rollback_operations for report in acceptance_report.case_reports)


def test_replay_selection_artifact_acceptance_can_be_forced_accept():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
            gate_config=DialogueArtifactAcceptanceGateConfig(
                min_mean_score_delta=-1.0,
                min_passed_case_delta=-10,
                min_positive_case_fraction=0.0,
                    min_worst_case_delta=-2.0,
            ),
        )
    )

    assert acceptance_report.decision.accepted is True
    assert acceptance_report.decision.rollback_applied is False
    assert all(not report.rollback_operations for report in acceptance_report.case_reports)


def test_replay_selection_artifact_acceptance_stays_review_only_under_frozen_doctrine():
    def _frozen_acceptance_runner(variant) -> AgentSessionRunner:
        runtime = SyntheticOpenWeightResidualRuntime(model_id=f"dialogue-frozen:{variant.case.case_id}")
        runtime.runtime_origin = "synthetic-frozen-review"
        runtime.supports_live_substrate_mutation = False
        return build_standard_dialogue_runner(
            profile_label="pe-eta",
            case=variant.case,
            residual_runtime=runtime,
        )

    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
        replay_ranking_report=ranking,
        top_k=1,
    )

    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_frozen_acceptance_runner,
        )
    )

    assert len(acceptance_report.case_reports) == 1
    assert acceptance_report.case_reports[0].import_result.applied_operations == ()
    assert acceptance_report.case_reports[0].rollback_operations == ()


def test_run_dialogue_pe_eta_systematic_replay_benchmark_collects_generated_variants():
    report = asyncio.run(
        run_dialogue_pe_eta_systematic_replay_benchmark(
            seeds=(0,),
            include_fixed_variants=False,
            runner_factory=_synthetic_systematic_runner,
        )
    )

    assert len(report.variant_cases) == len(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES)
    assert len(report.replay_ranking_report.entries) == len(report.variant_cases)
    assert report.perturbation_report.ablation_report.baseline_label == "pe-eta"


def test_run_dialogue_pe_eta_comprehensive_benchmark_runs_end_to_end():
    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    assert report.profile_labels == default_dialogue_comprehensive_profiles()
    assert report.canonical_ablation_report.baseline_label == "pe-eta"
    assert report.longitudinal_report.cross_session_report.verdict in {"growing", "stable", "regressing", "insufficient-data"}
    assert isinstance(report.essence_report, DialogueNLEssenceAssessmentReport)
    assert isinstance(report.essence_acceptance, DialogueNLEssenceAcceptanceDecision)
    assert report.open_ablation_report is not None
    assert report.open_ablation_report.baseline_label == "pe-eta"
    assert report.emergence_dashboard.baseline_label == "pe-eta"
    assert report.emergence_dashboard.strong_proof_panels
    assert report.emergence_dashboard.open_environment_panels
    assert len(report.selection_artifact.selected_variants) == 1
    assert len(report.artifact_comparison_report.candidate_reports) == 1
    assert report.description
    canonical_path = next(
        path for path in report.canonical_ablation_report.path_reports if path.path_label == "pe-eta"
    )
    canonical_metric_means = dict(canonical_path.benchmark_report.metric_means)
    assert canonical_metric_means["case_memory_surface_turn_count"] > 0.0
    assert canonical_metric_means["strategy_playbook_surface_turn_count"] > 0.0
    assert canonical_metric_means["experience_fast_prior_surface_turn_count"] > 0.0
    assert canonical_metric_means["experience_consolidation_surface_turn_count"] > 0.0


def test_build_dialogue_emergence_dashboard_compresses_strong_proof_and_open_env_evidence():
    report = _synthetic_comprehensive_report_for_dashboard()

    dashboard = build_dialogue_emergence_dashboard(report)

    assert dashboard.baseline_label == "pe-eta"
    assert dashboard.canonical_case_count == 1
    assert dashboard.open_scenario_count == 1
    assert dashboard.strong_proof_panels
    assert dashboard.open_environment_panels
    assert dashboard.canonical_mean_memory_tower_depth >= 5.0
    assert dashboard.canonical_runtime_backbone_evidence_rate > 0.0
    assert dashboard.canonical_mean_semantic_spine_coverage == 1.0
    assert dashboard.canonical_mean_cognitive_loop_readiness > 0.0
    assert dashboard.tower_memory_gate_strength > 0.0
    assert dashboard.strongest_scaffold_path_label is not None
    assert dashboard.strongest_open_path_label is not None
    assert isinstance(dashboard.interpretation, str)
    assert "canonical_pass_rate=" in dashboard.description


def test_build_dialogue_emergence_dashboard_payload_exposes_summary_keys():
    report = _synthetic_comprehensive_report_for_dashboard()

    payload = build_dialogue_emergence_dashboard_payload(report)

    assert payload["baseline_label"] == "pe-eta"
    assert payload["canonical"]["case_count"] == 1
    assert "mean_memory_tower_depth" in payload["canonical"]
    assert "runtime_backbone_evidence_rate" in payload["canonical"]
    assert payload["canonical"]["mean_semantic_spine_coverage"] == 1.0
    assert payload["canonical"]["mean_cognitive_loop_readiness"] > 0.0
    assert "tower_memory_gate" in payload
    assert payload["open_environment"]["scenario_count"] == 1
    assert payload["strong_proof_panels"]
    assert payload["open_environment_panels"]
    assert "rare_heavy_gate" in payload
    assert "essence" in payload


def test_export_dialogue_emergence_dashboard_artifact_writes_json(tmp_path):
    report = _synthetic_comprehensive_report_for_dashboard()

    output_path = tmp_path / "emergence_dashboard.json"
    written_path = export_dialogue_emergence_dashboard_artifact(
        report,
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert written_path == output_path
    assert payload["baseline_label"] == "pe-eta"
    assert payload["canonical"]["case_count"] == 1
    assert payload["open_environment"]["scenario_count"] == 1
    assert payload["strong_proof_panels"]
    assert payload["tower_memory_gate"]["strength"] > 0.0
    assert payload["description"] == report.emergence_dashboard.description


def test_build_dialogue_paper_suite_manifest_and_config_freeze_expected_scope():
    manifest = build_dialogue_paper_suite_manifest(suite_tier="paper-suite-small")
    config = dialogue_paper_suite_config(
        manifest,
        runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
    )

    assert manifest.suite_kind == "dialogue-comprehensive"
    assert manifest.baseline_label == "pe-eta"
    assert manifest.repeat_count == 5
    assert manifest.seed_schedule == (0, 1, 2, 3, 4)
    assert any(metric.metric_name == "canonical_pass_rate_pe_eta" for metric in manifest.primary_metrics)
    assert any(metric.metric_name == "canonical_runtime_backbone_evidence_rate" for metric in manifest.primary_metrics)
    assert any(metric.metric_name == "canonical_mean_memory_tower_depth" for metric in manifest.secondary_metrics)
    assert any(metric.metric_name == "canonical_mean_semantic_spine_coverage" for metric in manifest.secondary_metrics)
    assert any(metric.metric_name == "canonical_mean_cognitive_loop_readiness" for metric in manifest.secondary_metrics)
    assert any(metric.metric_name == "tower_memory_gate_strength" for metric in manifest.secondary_metrics)
    assert all("heldout" not in scenario_id for scenario_id in manifest.case_groups[1][1])
    assert config.runtime_mode == LocalSubstrateRuntimeMode.BUILTIN_ONLY
    assert config.profile_labels == default_dialogue_comprehensive_profiles()
    assert config.open_profile_labels == default_open_dialogue_ablation_profiles()


def test_run_dialogue_paper_suite_repeated_benchmark_emits_interval_summaries(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")

    report = asyncio.run(
        run_dialogue_paper_suite_repeated_benchmark(
            manifest=manifest,
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
            output_dir=tmp_path,
        )
    )

    assert report.manifest.suite_id == "dialogue-ci-smoke"
    assert report.run_summaries
    assert report.primary_metric_summaries
    assert report.pairwise_effects
    assert report.claim_verdicts
    assert report.provenance.manifest_hash
    assert report.reference_run_report is not None
    companion_verdict = next(
        verdict
        for verdict in report.claim_verdicts
        if verdict.claim_id == "claim_companion_stateful_relationship"
    )
    assert companion_verdict.status in {"retain", "weak", "fail"}
    assert "semantic-spine-ready" in companion_verdict.required_gate_ids
    companion_evidence = dict(companion_verdict.evidence)
    assert "canonical_semantic_spine_summary_sample_count" in companion_evidence
    assert "canonical_cognitive_loop_summary_sample_count" in companion_evidence


def test_dialogue_paper_suite_artifact_bundle_exports_expert_review_packet(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    report = asyncio.run(
        run_dialogue_paper_suite_repeated_benchmark(
            manifest=manifest,
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        )
    )

    written_paths = export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=tmp_path,
    )
    review_packet_path = tmp_path / "expert_review_packet_blinded.json"
    review_key_path = tmp_path / "expert_review_key_internal.json"
    evidence_bundle_path = tmp_path / "evidence_bundle.json"

    assert written_paths
    assert review_packet_path.exists()
    assert review_key_path.exists()
    assert evidence_bundle_path.exists()
    payload = json.loads(review_packet_path.read_text(encoding="utf-8"))
    assert payload["items"]
    assert payload["review_dimensions"]
    assert "source_profile_label" not in review_packet_path.read_text(encoding="utf-8")

    key_payload = json.loads(review_key_path.read_text(encoding="utf-8"))
    assert key_payload["baseline_label"] == "pe-eta"
    assert key_payload["entries"]

    bundle_payload = json.loads(evidence_bundle_path.read_text(encoding="utf-8"))
    assert bundle_payload["pairwise_effects"]
    assert bundle_payload["claim_verdicts"]


def test_dialogue_paper_suite_exports_proposal_quality_shadow_artifact(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    report = asyncio.run(
        run_dialogue_paper_suite_repeated_benchmark(
            manifest=manifest,
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        )
    )
    shadow_report = SemanticProposalQualityReport(
        target_slot="boundary_consent",
        runtime_id="semantic-llm-commitment",
        case_results=(
            SemanticProposalQualityCaseResult(
                case_id="ambiguous-consent",
                target_slot="boundary_consent",
                expected_operations=(),
                observed_operations=(SemanticProposalOperation.BLOCK,),
                true_positive_count=0,
                false_positive_count=1,
                missing_count=0,
                confidence_floor_passed=True,
                fell_back=False,
                would_block=True,
                shadow_gate_reasons=("false-positive",),
                passed=False,
                description="shadow false-positive diagnostic",
            ),
        ),
        passed_case_count=0,
        total_case_count=1,
        precision=0.0,
        recall=1.0,
        false_positive_count=1,
        missing_count=0,
        fallback_count=0,
        would_block_count=1,
        would_allow_count=0,
        shadow_gate_reasons=(("ambiguous-consent", ("false-positive",)),),
        description="boundary consent shadow quality report",
    )

    written_paths = export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=tmp_path,
        proposal_quality_shadow_reports=(shadow_report,),
    )

    shadow_path = tmp_path / "semantic_proposal_quality_shadow.json"
    evidence_bundle_path = tmp_path / "evidence_bundle.json"
    assert shadow_path in written_paths
    shadow_payload = json.loads(shadow_path.read_text(encoding="utf-8"))
    assert shadow_payload["shadow_only"] is True
    assert shadow_payload["non_gating"] is True
    assert shadow_payload["reports"][0]["would_block_count"] == 1
    bundle_payload = json.loads(evidence_bundle_path.read_text(encoding="utf-8"))
    artifact_names = {item[0] for item in bundle_payload["reference_artifacts"]}
    assert "semantic_proposal_quality_shadow" in artifact_names


def test_dialogue_paper_suite_bundle_attaches_extra_reference_artifacts(tmp_path):
    """ETA open-weight aggregate (or any opaque payload) can be attached
    as a generic reference artifact in the dialogue evidence bundle without
    coupling dialogue to the upstream module."""

    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    report = asyncio.run(
        run_dialogue_paper_suite_repeated_benchmark(
            manifest=manifest,
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        )
    )

    eta_open_weight_payload = {
        "claim_id": "claim_eta_real_open_weight_residual_control",
        "status": "weak",
        "evidence": {
            "real_open_weight_fallback_rate": 0.0,
            "real_open_weight_hook_coverage": 1.0,
        },
    }

    written_paths = export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=tmp_path,
        extra_reference_artifacts=(
            ("eta_open_weight_paper_suite_aggregate", eta_open_weight_payload),
        ),
    )

    evidence_bundle_path = tmp_path / "evidence_bundle.json"
    assert evidence_bundle_path in written_paths
    bundle_payload = json.loads(evidence_bundle_path.read_text(encoding="utf-8"))
    artifact_names = {item[0] for item in bundle_payload["reference_artifacts"]}
    assert "eta_open_weight_paper_suite_aggregate" in artifact_names

    eta_artifact = next(
        item[1]
        for item in bundle_payload["reference_artifacts"]
        if item[0] == "eta_open_weight_paper_suite_aggregate"
    )
    assert eta_artifact["claim_id"] == "claim_eta_real_open_weight_residual_control"
    assert eta_artifact["evidence"]["real_open_weight_fallback_rate"] == 0.0


def _phase_a_option_benchmark_report() -> DialogueBenchmarkReport:
    case = ScriptedDialogueCase(
        case_id="phase-a-option",
        description="Synthetic case for Phase A option discovery report.",
        user_inputs=("one", "two", "three", "four"),
        expected_pressure_turns=(2,),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.1,
            reward=0.1,
            action="stabilize",
            regime="support",
            abstract_action="support",
            switch_gate=0.0,
            delayed_metric=0.1,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.8,
            reward=0.2,
            action="repair",
            regime="repair",
            abstract_action="repair",
            switch_gate=0.9,
            delayed_metric=0.4,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.3,
            reward=0.2,
            action="repair",
            regime="repair",
            abstract_action="repair",
            switch_gate=0.0,
            delayed_metric=0.5,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.2,
            reward=0.2,
            action="summarize",
            regime="support",
            abstract_action="summarize",
            switch_gate=0.7,
            delayed_metric=0.6,
        ),
    )
    case_report = build_dialogue_case_report(case=case, turns=turns)
    return DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=1,
        total_case_count=1,
        metric_means=tuple(_case_summary_metrics(case_report)),
        description="Synthetic benchmark report for Phase A option discovery.",
    )


def test_build_dialogue_option_discovery_report_from_turns_and_replay():
    report = _phase_a_option_benchmark_report()
    replay_artifact = {
        "action_replay": {
            "action_context": {
                "segment_id": "segment-repair",
                "z_t_digest": "z-repair",
            },
            "closed_segments": (
                {
                    "segment_id": "segment-support",
                    "z_t_digest": "z-support",
                    "abstract_action_id": "support",
                },
            ),
            "credit_records": (),
        }
    }

    option_report = build_dialogue_option_discovery_report(
        report,
        replay_artifacts=(replay_artifact,),
        source_id="test-option-discovery",
    )

    assert isinstance(option_report, DialogueOptionDiscoveryReport)
    assert option_report.non_gating is True
    assert option_report.case_count == 1
    assert option_report.turn_count == 4
    assert option_report.termination_event_count == 2
    assert option_report.abstract_action_diversity == 3
    assert option_report.evidence_quality == "snapshot-replay+turn-telemetry"
    assert option_report.pe_spike_near_termination_rate > 0.0
    assert option_report.case_reports[0].turn_evidence[0].z_t_digest == "z-repair"


def test_build_dialogue_option_discovery_report_empty_when_no_turns():
    empty_report = DialogueBenchmarkReport(
        case_reports=(),
        passed_case_count=0,
        total_case_count=0,
        metric_means=(),
        description="Empty source for option discovery.",
    )

    option_report = build_dialogue_option_discovery_report(empty_report)

    assert option_report.non_gating is True
    assert option_report.case_count == 0
    assert option_report.turn_count == 0
    assert option_report.termination_event_count == 0
    assert option_report.evidence_quality == "turn-telemetry-only"
    assert option_report.random_boundary_baseline_rate == 0.0


def _phase_a_counterfactual_comparison_report() -> DialogueBenchmarkComparisonReport:
    def path(
        label: str,
        *,
        passed: int,
        prediction_chain: float,
        pe_triggered: float,
        carryover_credit: float,
        bounded_writeback: float,
        delayed: float,
        precision: float,
        stability: float,
    ) -> DialogueBenchmarkPathReport:
        benchmark = DialogueBenchmarkReport(
            case_reports=(),
            passed_case_count=passed,
            total_case_count=2,
            metric_means=(
                ("prediction_chain_turn_count", prediction_chain),
                ("pe_triggered_turn_count", pe_triggered),
                ("carryover_credit_turn_count", carryover_credit),
                ("bounded_writeback_turn_count", bounded_writeback),
                ("delayed_improvement_observed", delayed),
                ("pressure_response_precision", precision),
                ("stability_after_recovery_score", stability),
                ("runtime_backbone_evidence_turn_count", 1.0),
            ),
            description=f"Synthetic benchmark report for {label}.",
        )
        return DialogueBenchmarkPathReport(
            path_label=label,
            benchmark_report=benchmark,
            description=f"Synthetic path {label}.",
        )

    path_reports = (
        path(
            "pe-eta",
            passed=2,
            prediction_chain=4.0,
            pe_triggered=3.0,
            carryover_credit=2.0,
            bounded_writeback=1.0,
            delayed=1.0,
            precision=0.8,
            stability=0.7,
        ),
        path(
            "pe-drive-off",
            passed=1,
            prediction_chain=4.0,
            pe_triggered=0.5,
            carryover_credit=0.4,
            bounded_writeback=0.2,
            delayed=0.3,
            precision=0.3,
            stability=0.2,
        ),
        path(
            "pe-eta-pe-readout-only",
            passed=1,
            prediction_chain=4.0,
            pe_triggered=1.0,
            carryover_credit=0.5,
            bounded_writeback=0.2,
            delayed=0.4,
            precision=0.35,
            stability=0.25,
        ),
        path(
            "eta-off",
            passed=1,
            prediction_chain=2.0,
            pe_triggered=0.5,
            carryover_credit=0.4,
            bounded_writeback=0.2,
            delayed=0.2,
            precision=0.25,
            stability=0.1,
        ),
    )
    return DialogueBenchmarkComparisonReport(
        baseline_label="pe-eta",
        path_reports=path_reports,
        case_deltas_from_baseline=(),
        metric_deltas_from_baseline=(),
        description="Synthetic comparison for PE counterfactual closure.",
    )


def test_build_pe_counterfactual_closure_report_from_existing_profiles():
    closure_report = build_pe_counterfactual_closure_report(
        _phase_a_counterfactual_comparison_report()
    )

    assert isinstance(closure_report, PECounterfactualClosureReport)
    assert closure_report.non_gating is True
    assert closure_report.baseline_label == "pe-eta"
    assert {variant.variant_label for variant in closure_report.variants} == {
        "pe-eta",
        "pe-drive-off",
        "pe-eta-pe-readout-only",
        "eta-off",
    }
    assert closure_report.pe_to_credit_drop > 0.0
    assert closure_report.pe_to_behavior_drop > 0.0
    assert closure_report.readout_only_gap > 0.0
    assert closure_report.eta_dependency_gap > 0.0
    assert closure_report.closure_strength > 0.0


def test_build_pe_counterfactual_closure_report_requires_all_profiles():
    comparison = _phase_a_counterfactual_comparison_report()
    incomplete = replace(
        comparison,
        path_reports=tuple(
            path for path in comparison.path_reports if path.path_label != "eta-off"
        ),
    )

    with pytest.raises(ValueError, match="eta-off"):
        build_pe_counterfactual_closure_report(incomplete)


def test_dialogue_paper_suite_export_writes_phase_a_trajectory_reports(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    provenance = PaperSuiteProvenance(
        git_sha="test",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        dependency_digest="test",
        manifest_hash="test",
        runtime_descriptor=(),
        description="Synthetic provenance for Phase A trajectory export test.",
    )
    aggregate_report = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        reference_run_report=None,
        primary_metric_summaries=(),
        secondary_metric_summaries=(),
        description="Synthetic aggregate for Phase A trajectory export test.",
    )
    option_report = build_dialogue_option_discovery_report(_phase_a_option_benchmark_report())
    closure_report = build_pe_counterfactual_closure_report(
        _phase_a_counterfactual_comparison_report()
    )

    written_paths = export_dialogue_paper_suite_artifact_bundle(
        aggregate_report,
        output_dir=tmp_path,
        dialogue_option_discovery_report=option_report,
        pe_counterfactual_closure_report=closure_report,
    )

    option_path = tmp_path / "dialogue_option_discovery_report.json"
    closure_path = tmp_path / "pe_counterfactual_closure_report.json"
    bundle_path = tmp_path / "evidence_bundle.json"
    assert option_path in written_paths
    assert closure_path in written_paths
    assert option_path.exists()
    assert closure_path.exists()
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    artifact_names = {item[0] for item in bundle_payload["reference_artifacts"]}
    assert "dialogue_option_discovery_report" in artifact_names
    assert "pe_counterfactual_closure_report" in artifact_names


def _phase_b_session_evidence() -> tuple[LongitudinalDialogueSessionEvidence, ...]:
    return (
        LongitudinalDialogueSessionEvidence(
            persona_id="delayed-return",
            session_id="delayed-return:s1",
            session_index=1,
            passed=True,
            mean_prediction_error=0.28,
            temporal_change_count=1,
            delayed_improvement_observed=True,
            mean_cognitive_loop_readiness=0.2,
            explicit_retention_observed=True,
            default_isolation_preserved=True,
            retrieved_preference_count=1,
            preference_conflict_observed=False,
            repair_observable=True,
            regime_stability_score=0.55,
            description="Session 1 writes the preference.",
        ),
        LongitudinalDialogueSessionEvidence(
            persona_id="delayed-return",
            session_id="delayed-return:s2",
            session_index=2,
            passed=True,
            mean_prediction_error=0.18,
            temporal_change_count=2,
            delayed_improvement_observed=True,
            mean_cognitive_loop_readiness=0.45,
            explicit_retention_observed=True,
            default_isolation_preserved=True,
            retrieved_preference_count=2,
            preference_conflict_observed=False,
            repair_observable=True,
            regime_stability_score=0.7,
            description="Session 2 retrieves the preference.",
        ),
        LongitudinalDialogueSessionEvidence(
            persona_id="preference-conflict",
            session_id="preference-conflict:s1",
            session_index=1,
            passed=True,
            mean_prediction_error=0.35,
            temporal_change_count=2,
            delayed_improvement_observed=True,
            mean_cognitive_loop_readiness=0.3,
            explicit_retention_observed=True,
            default_isolation_preserved=True,
            retrieved_preference_count=1,
            preference_conflict_observed=True,
            repair_observable=True,
            regime_stability_score=0.6,
            description="Preference conflict is repaired through clarification.",
        ),
    )


def test_build_longitudinal_dialogue_report_from_session_evidence():
    report = build_longitudinal_dialogue_report(
        session_evidence=_phase_b_session_evidence(),
        report_id="phase-b-test",
    )

    assert isinstance(report, LongitudinalDialogueReport)
    assert report.non_gating is True
    assert report.persona_count == 2
    assert report.session_count == 3
    assert report.retention_rate > 0.0
    assert report.isolation_pass_rate == 1.0
    assert report.trajectory_strength > 0.0
    delayed_return = next(
        persona for persona in report.persona_reports if persona.persona_id == "delayed-return"
    )
    assert delayed_return.retention_rate == 1.0
    assert delayed_return.adaptation_trend > 0.0
    preference_conflict = next(
        persona for persona in report.persona_reports if persona.persona_id == "preference-conflict"
    )
    assert preference_conflict.preference_conflict_repair_rate == 1.0


def test_build_longitudinal_dialogue_report_empty_is_non_gating_and_zero_strength():
    report = build_longitudinal_dialogue_report(report_id="phase-b-empty")

    assert report.non_gating is True
    assert report.persona_count == 0
    assert report.session_count == 0
    assert report.trajectory_strength == 0.0
    assert report.cross_session_verdict == "insufficient"
    assert report.evidence_quality == "empty"


def test_default_longitudinal_dialogue_personas_cover_v1_surface():
    personas = default_longitudinal_dialogue_personas()
    persona_ids = {persona_id for persona_id, _ in personas}

    assert {
        "direct-but-overwhelmed",
        "slow-trust-repair",
        "boundary-sensitive",
        "preference-conflict",
        "delayed-return",
    } <= persona_ids
    assert all(len(sessions) == 3 for _, sessions in personas)


def test_dialogue_paper_suite_export_writes_phase_b_longitudinal_report(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    provenance = PaperSuiteProvenance(
        git_sha="test",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        dependency_digest="test",
        manifest_hash="test",
        runtime_descriptor=(),
        description="Synthetic provenance for PhaseB longitudinal export test.",
    )
    aggregate_report = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        reference_run_report=None,
        primary_metric_summaries=(),
        secondary_metric_summaries=(),
        description="Synthetic aggregate for PhaseB longitudinal export test.",
    )
    longitudinal_report = build_longitudinal_dialogue_report(
        session_evidence=_phase_b_session_evidence(),
        report_id="phase-b-export-test",
    )

    written_paths = export_dialogue_paper_suite_artifact_bundle(
        aggregate_report,
        output_dir=tmp_path,
        longitudinal_dialogue_report=longitudinal_report,
    )

    report_path = tmp_path / "longitudinal_dialogue_report.json"
    bundle_path = tmp_path / "evidence_bundle.json"
    assert report_path in written_paths
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["non_gating"] is True
    assert payload["retention_rate"] > 0.0
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    artifact_names = {item[0] for item in bundle_payload["reference_artifacts"]}
    assert "longitudinal_dialogue_report" in artifact_names


def _phase_c_variant_metrics() -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    return (
        (
            "full-nl",
            (
                ("cross_session_growth_score", 0.8),
                ("heldout_payoff_score", 0.7),
                ("slow_to_fast_transfer_gain", 0.4),
                ("memory_churn_risk", 0.1),
                ("behavior_drift_risk", 0.1),
            ),
        ),
        (
            "no-ssl",
            (
                ("cross_session_growth_score", 0.35),
                ("heldout_payoff_score", 0.3),
                ("slow_to_fast_transfer_gain", 0.05),
                ("memory_churn_risk", 0.35),
                ("behavior_drift_risk", 0.25),
            ),
        ),
        (
            "no-rl",
            (
                ("cross_session_growth_score", 0.4),
                ("heldout_payoff_score", 0.35),
                ("slow_to_fast_transfer_gain", 0.1),
                ("memory_churn_risk", 0.25),
                ("behavior_drift_risk", 0.3),
            ),
        ),
        (
            "timescale-off",
            (
                ("cross_session_growth_score", 0.2),
                ("heldout_payoff_score", 0.25),
                ("slow_to_fast_transfer_gain", 0.0),
                ("memory_churn_risk", 0.5),
                ("behavior_drift_risk", 0.45),
            ),
        ),
    )


def test_build_nl_ablation_matrix_report_from_explicit_variant_metrics():
    report = build_nl_ablation_matrix_report(
        variant_metrics=_phase_c_variant_metrics(),
        report_id="phase-c-nl-test",
    )

    assert isinstance(report, NLAblationMatrixReport)
    assert report.non_gating is True
    assert report.evidence_quality == "explicit-variant-metrics"
    assert report.full_nl_advantage > 0.0
    assert report.slow_to_fast_transfer_gain > 0.0
    assert {variant.variant_label for variant in report.variants} >= {
        "full-nl",
        "no-ssl",
        "no-rl",
        "timescale-off",
    }


def test_build_memory_stratum_flow_report_from_dict_snapshots():
    report = build_memory_stratum_flow_report(
        (
            {
                "snapshot_id": "memory-1",
                "total_entries_by_stratum": (("transient", 2.0), ("episodic", 1.0), ("durable", 0.0)),
                "pending_promotions": 2,
                "pending_decays": 0,
                "cms_band_vectors": (),
                "lifecycle_metrics": (("slow_to_fast_init_benefit", 0.1),),
            },
            {
                "snapshot_id": "memory-2",
                "total_entries_by_stratum": (
                    ("transient", 0.0),
                    ("episodic", 2.0),
                    ("durable", 1.0),
                    ("derived", 2.0),
                ),
                "pending_promotions": 1,
                "pending_decays": 0,
                "cms_band_vectors": (("band-a", (0.1, 0.2)),),
                "lifecycle_metrics": (
                    ("slow_to_fast_init_benefit", 0.2),
                    ("tower_consolidation_count", 2.0),
                    ("continuum_band_count", 2.0),
                    ("fast_memory_signal_count", 1.0),
                ),
            },
        ),
        report_id="phase-c-memory-test",
    )

    assert isinstance(report, MemoryStratumFlowReport)
    assert report.non_gating is True
    assert report.snapshot_count == 2
    assert report.stratum_progression_score > 0.0
    assert report.derived_index_activity > 0.0
    assert report.lifecycle_signal_strength > 0.0
    assert report.memory_flow_strength > 0.0


def test_build_regime_lockin_report_from_dict_snapshots():
    report = build_regime_lockin_report(
        (
            {
                "snapshot_id": "regime-1",
                "active_regime": "emotional_support",
                "previous_regime": "acquaintance_building",
                "turns_in_current_regime": 1,
                "regime_changed": True,
                "candidate_regimes": ("emotional_support", "repair"),
                "delayed_attributions": (),
                "delayed_payoffs": (),
                "sequence_payoffs": (),
                "identity_hints": ("supportive",),
            },
            {
                "snapshot_id": "regime-2",
                "active_regime": "emotional_support",
                "previous_regime": "emotional_support",
                "turns_in_current_regime": 4,
                "regime_changed": False,
                "candidate_regimes": ("emotional_support", "repair"),
                "delayed_attributions": ("a1",),
                "delayed_payoffs": ("p1",),
                "sequence_payoffs": ("s1",),
                "identity_hints": ("supportive",),
            },
        ),
        report_id="phase-c-regime-test",
    )

    assert isinstance(report, RegimeLockinReport)
    assert report.non_gating is True
    assert report.snapshot_count == 2
    assert report.lockin_strength > 0.0
    assert report.hysteresis_proxy > 0.0
    assert report.delayed_attribution_strength > 0.0
    assert report.sequence_payoff_strength > 0.0
    assert report.regime_identity_stability == 1.0


def test_phase_c_empty_reports_are_non_gating_and_zero_strength():
    nl = build_nl_ablation_matrix_report(report_id="phase-c-empty-nl")
    memory = build_memory_stratum_flow_report(report_id="phase-c-empty-memory")
    regime = build_regime_lockin_report(report_id="phase-c-empty-regime")

    assert nl.non_gating is True
    assert nl.full_nl_advantage == 0.0
    assert memory.non_gating is True
    assert memory.memory_flow_strength == 0.0
    assert regime.non_gating is True
    assert regime.lockin_strength == 0.0


def test_dialogue_paper_suite_export_writes_phase_c_reports(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    provenance = PaperSuiteProvenance(
        git_sha="test",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        dependency_digest="test",
        manifest_hash="test",
        runtime_descriptor=(),
        description="Synthetic provenance for PhaseC export test.",
    )
    aggregate_report = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        reference_run_report=None,
        primary_metric_summaries=(),
        secondary_metric_summaries=(),
        description="Synthetic aggregate for PhaseC export test.",
    )
    nl_report = build_nl_ablation_matrix_report(variant_metrics=_phase_c_variant_metrics())
    memory_report = build_memory_stratum_flow_report(
        (
            {
                "snapshot_id": "memory-export",
                "total_entries_by_stratum": (("episodic", 1.0), ("durable", 1.0), ("derived", 1.0)),
                "pending_promotions": 1,
                "pending_decays": 0,
                "cms_band_vectors": (("band-a", (0.1,)),),
                "lifecycle_metrics": (("slow_to_fast_init_benefit", 0.2),),
            },
        )
    )
    regime_report = build_regime_lockin_report(
        (
            {
                "snapshot_id": "regime-export",
                "active_regime": "emotional_support",
                "previous_regime": "emotional_support",
                "turns_in_current_regime": 3,
                "regime_changed": False,
                "candidate_regimes": ("emotional_support", "repair"),
                "delayed_attributions": ("a1",),
                "sequence_payoffs": ("s1",),
                "identity_hints": ("supportive",),
            },
        )
    )

    written_paths = export_dialogue_paper_suite_artifact_bundle(
        aggregate_report,
        output_dir=tmp_path,
        nl_ablation_matrix_report=nl_report,
        memory_stratum_flow_report=memory_report,
        regime_lockin_report=regime_report,
    )

    assert tmp_path / "nl_ablation_matrix_report.json" in written_paths
    assert tmp_path / "memory_stratum_flow_report.json" in written_paths
    assert tmp_path / "regime_lockin_report.json" in written_paths
    bundle_payload = json.loads((tmp_path / "evidence_bundle.json").read_text(encoding="utf-8"))
    artifact_names = {item[0] for item in bundle_payload["reference_artifacts"]}
    assert {
        "nl_ablation_matrix_report",
        "memory_stratum_flow_report",
        "regime_lockin_report",
    } <= artifact_names


def test_build_dialogue_expert_review_packet_blinds_profile_labels():
    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    packet = build_dialogue_expert_review_packet(report)
    internal_key = build_dialogue_expert_review_internal_key(report)

    assert packet.items
    first_item = packet.items[0]
    assert first_item.samples
    assert all(sample.blinded_label.startswith("sample_") for sample in first_item.samples)
    assert all(sample.transcript for sample in first_item.samples)
    assert first_item.prompt_context
    assert internal_key.entries
    assert all(entry.source_profile_label in {"pe-eta", "pe-drive-off", "eta-off"} for entry in internal_key.entries)


def test_blind_packet_transcripts_have_no_profile_label_leak():
    """The blinded transcripts must not contain any profile-label string
    that would let the rater identify which condition produced a sample.

    We deliberately do NOT also assert the source case_id is absent: case
    ids can legitimately be common English words (``"repair"``,
    ``"goal_drift"``, ...) that appear in the conversation topic itself.
    Profile labels (``pe-eta``, ``pe-drive-off``, ``eta-off``) are
    hyphenated discriminators and never appear naturally in user text;
    if they do, the blind dispatch is broken.
    """

    report = asyncio.run(
        run_dialogue_pe_eta_comprehensive_benchmark(
            canonical_cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            open_scenarios=DEFAULT_OPEN_DIALOGUE_SCENARIOS[:1],
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:1],
            seeds=(0,),
            families=DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1],
            selection_top_k=1,
            candidate_configs=DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:1],
            canonical_runner_factory=_synthetic_ablation_runner,
            open_runner_factory=_synthetic_open_ablation_runner,
            perturbation_runner_factory=_synthetic_perturbation_runner,
            systematic_runner_factory=_synthetic_systematic_runner,
            acceptance_runner_factory=_synthetic_acceptance_runner,
        )
    )

    packet = build_dialogue_expert_review_packet(report)

    forbidden_profile_labels: set[str] = {"pe-eta", "pe-drive-off", "eta-off"}

    leak_findings: list[tuple[str, str, str]] = []
    for item in packet.items:
        for sample in item.samples:
            for turn_index, (user_text, assistant_text) in enumerate(sample.transcript):
                for forbidden in forbidden_profile_labels:
                    if forbidden in user_text:
                        leak_findings.append((sample.sample_id, f"user-turn-{turn_index}", forbidden))
                    if forbidden in assistant_text:
                        leak_findings.append((sample.sample_id, f"assistant-turn-{turn_index}", forbidden))

    assert not leak_findings, (
        "blind packet transcript leaked profile-label strings: "
        f"{leak_findings}. Profile labels must only live in "
        "DialogueExpertReviewInternalKey, never in transcripts."
    )


def test_dialogue_human_rating_csv_aggregate_exports_external_claim(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    dimension = DialogueExpertReviewDimension(
        dimension_id="relationship_continuity",
        prompt="Rate which transcript better preserves trust and continuity.",
        description="Synthetic external review dimension.",
    )
    samples = (
        DialogueExpertReviewSample(
            sample_id="item_01:sample_A",
            blinded_label="sample_A",
            transcript=(("user", "candidate"),),
            description="Candidate transcript.",
        ),
        DialogueExpertReviewSample(
            sample_id="item_01:sample_B",
            blinded_label="sample_B",
            transcript=(("user", "control"),),
            description="Control transcript.",
        ),
    )
    packet = DialogueExpertReviewPacket(
        packet_id="test-review",
        source_suite_id="dialogue-ci-smoke",
        items=(
            DialogueExpertReviewItem(
                item_id="item_01",
                prompt_context="Compare blinded transcripts.",
                samples=samples,
                review_dimensions=(dimension,),
                description="Synthetic review item.",
            ),
        ),
        review_dimensions=(dimension,),
        description="Synthetic review packet.",
    )
    internal_key = DialogueExpertReviewInternalKey(
        packet_id="test-review",
        baseline_label="pe-eta",
        entries=(
            DialogueExpertReviewInternalKeyEntry(
                item_id="item_01",
                sample_id="item_01:sample_A",
                blinded_label="sample_A",
                source_case_id="case-1",
                source_profile_label="pe-eta",
                description="Candidate key.",
            ),
            DialogueExpertReviewInternalKeyEntry(
                item_id="item_01",
                sample_id="item_01:sample_B",
                blinded_label="sample_B",
                source_case_id="case-1",
                source_profile_label="eta-off",
                description="Control key.",
            ),
        ),
        description="Synthetic internal key.",
    )
    report = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=PaperSuiteProvenance(
            git_sha="test",
            git_branch="test",
            working_tree_dirty=False,
            python_version="test",
            platform="test",
            dependency_versions=(),
            dependency_digest="test",
            manifest_hash="test",
            runtime_descriptor=(),
            description="Synthetic provenance for human rating aggregate export test.",
        ),
        run_summaries=(),
        reference_run_report=None,
        primary_metric_summaries=(),
        secondary_metric_summaries=(),
        description="Synthetic paper-suite aggregate for human rating aggregate export test.",
    )
    profile_by_sample = {
        entry.sample_id: entry.source_profile_label
        for entry in internal_key.entries
    }
    csv_path = tmp_path / "ratings.csv"
    lines = ["rater_id,item_id,sample_id,blinded_label,dimension_id,score"]
    for rater_id in ("rater-a", "rater-b", "rater-c"):
        for item in packet.items:
            for sample in item.samples:
                for dimension in packet.review_dimensions:
                    score = 5 if profile_by_sample[sample.sample_id] == "pe-eta" else 2
                    lines.append(
                        f"{rater_id},{item.item_id},{sample.sample_id},{sample.blinded_label},{dimension.dimension_id},{score}"
                    )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = load_dialogue_human_rating_entries_csv(csv_path)
    aggregate = aggregate_dialogue_human_ratings(
        packet=packet,
        entries=entries,
        internal_key=internal_key,
    )

    assert aggregate.rater_count == 3
    assert aggregate.profile_scores
    pe_eta_profile = next(
        profile for profile in aggregate.profile_scores if profile.source_profile_label == "pe-eta"
    )
    assert pe_eta_profile.mean_score == 5
    assert aggregate.pairwise_preferences
    assert all(pair.win_rate == 1.0 for pair in aggregate.pairwise_preferences)

    export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=tmp_path / "bundle",
        human_ratings_aggregate=aggregate,
    )
    aggregate_payload = json.loads(
        (tmp_path / "bundle" / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    external_claim = next(
        verdict
        for verdict in aggregate_payload["claim_verdicts"]
        if verdict["claim_id"] == "claim_external_human_legibility"
    )
    assert external_claim["status"] == "retain"


def test_load_dialogue_human_rating_entries_csv_dir_merges_per_rater_files(tmp_path):
    from volvence_zero.agent import load_dialogue_human_rating_entries_csv_dir

    csv_dir = tmp_path / "ratings"
    csv_dir.mkdir()
    header = "rater_id,item_id,sample_id,blinded_label,dimension_id,score"
    (csv_dir / "rater_a.csv").write_text(
        "\n".join(
            [
                header,
                "rater-a,item_01,item_01:sample_A,sample_A,relationship_continuity,5",
                "rater-a,item_01,item_01:sample_B,sample_B,relationship_continuity,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (csv_dir / "rater_b.csv").write_text(
        "\n".join(
            [
                header,
                "rater-b,item_01,item_01:sample_A,sample_A,relationship_continuity,4",
                "rater-b,item_01,item_01:sample_B,sample_B,relationship_continuity,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    merged = load_dialogue_human_rating_entries_csv_dir(csv_dir)
    rater_ids = {entry.rater_id for entry in merged}
    assert rater_ids == {"rater-a", "rater-b"}
    assert len(merged) == 4


def test_load_dialogue_human_rating_entries_csv_dir_rejects_rater_id_collision(tmp_path):
    from volvence_zero.agent import load_dialogue_human_rating_entries_csv_dir

    csv_dir = tmp_path / "ratings"
    csv_dir.mkdir()
    header = "rater_id,item_id,sample_id,blinded_label,dimension_id,score"
    (csv_dir / "rater_a_v1.csv").write_text(
        f"{header}\nrater-a,item_01,item_01:sample_A,sample_A,relationship_continuity,5\n",
        encoding="utf-8",
    )
    (csv_dir / "rater_a_v2.csv").write_text(
        f"{header}\nrater-a,item_01,item_01:sample_B,sample_B,relationship_continuity,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rater_id"):
        load_dialogue_human_rating_entries_csv_dir(csv_dir)

    merged = load_dialogue_human_rating_entries_csv_dir(
        csv_dir, forbid_rater_id_collision=False
    )
    assert len(merged) == 2


def test_companion_structural_gates_propagate_into_companion_stateful_verdict(tmp_path):
    """AAC1 / RGM1 / RFL1 gate results from upstream
    `companion_evidence` can be injected into the dialogue export and
    surface as additional retain checks + required_gate_ids on
    `claim_companion_stateful_relationship`. When the gate map is empty
    the verdict shape is unchanged (back-compat)."""

    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    report = asyncio.run(
        run_dialogue_paper_suite_repeated_benchmark(
            manifest=manifest,
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        )
    )

    bundle_with_gates_dir = tmp_path / "with_gates"
    written_paths = export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=bundle_with_gates_dir,
        companion_structural_gates=(("AAC1", True), ("RGM1", True), ("RFL1", False)),
    )
    aggregate_payload = json.loads(
        (bundle_with_gates_dir / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    companion_verdict = next(
        verdict
        for verdict in aggregate_payload["claim_verdicts"]
        if verdict["claim_id"] == "claim_companion_stateful_relationship"
    )
    required_gate_ids = set(companion_verdict["required_gate_ids"])
    assert {"AAC1", "RGM1", "RFL1"}.issubset(required_gate_ids)
    evidence_map = dict(companion_verdict["evidence"])
    assert evidence_map["AAC1_alignment_pe_repair_visibility"] == 1.0
    assert evidence_map["RGM1_regime_delayed_attribution_visibility"] == 1.0
    assert evidence_map["RFL1_reflection_writeback_stability"] == 0.0
    # When any structural gate fails, the verdict must NOT be retain
    # even if all spine checks would otherwise pass.
    assert companion_verdict["status"] in {"weak", "fail"}

    bundle_no_gates_dir = tmp_path / "no_gates"
    export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=bundle_no_gates_dir,
    )
    aggregate_no_gates = json.loads(
        (bundle_no_gates_dir / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    companion_verdict_no_gates = next(
        verdict
        for verdict in aggregate_no_gates["claim_verdicts"]
        if verdict["claim_id"] == "claim_companion_stateful_relationship"
    )
    no_gates_required = set(companion_verdict_no_gates["required_gate_ids"])
    assert "AAC1" not in no_gates_required
    assert "RGM1" not in no_gates_required
    assert "RFL1" not in no_gates_required
    no_gates_evidence_map = dict(companion_verdict_no_gates["evidence"])
    assert no_gates_evidence_map["AAC1_alignment_pe_repair_visibility"] == -1.0
    assert no_gates_evidence_map["RGM1_regime_delayed_attribution_visibility"] == -1.0
    assert no_gates_evidence_map["RFL1_reflection_writeback_stability"] == -1.0


def test_dialogue_temporal_advantage_claim_requires_runtime_backbone_consistency(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    provenance = PaperSuiteProvenance(
        git_sha="test",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        dependency_digest="test",
        manifest_hash="test",
        runtime_descriptor=(),
        description="Synthetic provenance for runtime consistency gate test.",
    )
    pairwise_effects = (
        PairwiseMetricEffect(
            metric_name="canonical_pass_rate",
            candidate_label="pe-eta",
            control_label="pe-drive-off",
            sample_count=3,
            mean_delta=0.4,
            std_delta=0.0,
            stderr_delta=0.0,
            ci_low=0.2,
            ci_high=0.5,
            effect_size=1.0,
            description="Synthetic positive PE drive gap.",
        ),
        PairwiseMetricEffect(
            metric_name="canonical_pass_rate",
            candidate_label="pe-eta",
            control_label="eta-off",
            sample_count=3,
            mean_delta=0.4,
            std_delta=0.0,
            stderr_delta=0.0,
            ci_low=0.2,
            ci_high=0.5,
            effect_size=1.0,
            description="Synthetic positive ETA off gap.",
        ),
    )
    report_without_runtime = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        reference_run_report=None,
        primary_metric_summaries=(),
        secondary_metric_summaries=(),
        pairwise_effects=pairwise_effects,
        description="Synthetic report without runtime backbone evidence.",
    )

    export_dialogue_paper_suite_artifact_bundle(
        report_without_runtime,
        output_dir=tmp_path / "without-runtime",
    )
    weak_payload = json.loads(
        (tmp_path / "without-runtime" / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    weak_claim = next(
        verdict for verdict in weak_payload["claim_verdicts"] if verdict["claim_id"] == "claim_temporal_advantage_over_controls"
    )
    assert weak_claim["status"] == "weak"
    assert ("runtime_backbone_consistency_observed", 0.0) in [tuple(item) for item in weak_claim["evidence"]]

    runtime_summaries = (
        MetricIntervalSummary(
            metric_name="canonical_runtime_backbone_evidence_rate",
            sample_count=3,
            mean=1.0,
            std=0.0,
            stderr=0.0,
            ci_low=1.0,
            ci_high=1.0,
            min_value=1.0,
            max_value=1.0,
            description="Synthetic runtime evidence rate.",
        ),
        MetricIntervalSummary(
            metric_name="canonical_mean_runtime_backbone_signal_quality",
            sample_count=3,
            mean=0.8,
            std=0.0,
            stderr=0.0,
            ci_low=0.8,
            ci_high=0.8,
            min_value=0.8,
            max_value=0.8,
            description="Synthetic runtime signal quality.",
        ),
    )
    report_with_runtime = replace(
        report_without_runtime,
        secondary_metric_summaries=runtime_summaries,
    )

    export_dialogue_paper_suite_artifact_bundle(
        report_with_runtime,
        output_dir=tmp_path / "with-runtime",
    )
    retain_payload = json.loads(
        (tmp_path / "with-runtime" / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    retain_claim = next(
        verdict for verdict in retain_payload["claim_verdicts"] if verdict["claim_id"] == "claim_temporal_advantage_over_controls"
    )
    assert retain_claim["status"] == "retain"
    assert ("runtime_backbone_consistency_observed", 1.0) in [tuple(item) for item in retain_claim["evidence"]]


def test_dialogue_rare_heavy_claim_requires_no_rare_heavy_control(tmp_path):
    manifest = build_dialogue_paper_suite_manifest(suite_tier="ci-smoke")
    provenance = PaperSuiteProvenance(
        git_sha="test",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        dependency_digest="test",
        manifest_hash="test",
        runtime_descriptor=(),
        description="Synthetic provenance for rare-heavy gate test.",
    )
    report_without_control = DialoguePaperSuiteAggregateReport(
        manifest=manifest,
        provenance=provenance,
        run_summaries=(),
        reference_run_report=None,
        primary_metric_summaries=(),
        secondary_metric_summaries=(),
        pairwise_effects=(),
        description="Synthetic report without rare-heavy matched control.",
    )

    export_dialogue_paper_suite_artifact_bundle(
        report_without_control,
        output_dir=tmp_path / "without-control",
    )
    missing_payload = json.loads(
        (tmp_path / "without-control" / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    missing_claim = next(
        verdict for verdict in missing_payload["claim_verdicts"] if verdict["claim_id"] == "claim_rare_heavy_net_benefit"
    )
    assert missing_claim["status"] == "fail"

    report_with_control = replace(
        report_without_control,
        pairwise_effects=(
            PairwiseMetricEffect(
                metric_name="canonical_pass_rate",
                candidate_label="pe-eta",
                control_label="pe-eta-no-rare-heavy",
                sample_count=3,
                mean_delta=0.25,
                std_delta=0.0,
                stderr_delta=0.0,
                ci_low=0.1,
                ci_high=0.3,
                effect_size=1.0,
                description="Synthetic rare-heavy net benefit over no-rare-heavy control.",
            ),
        ),
    )

    export_dialogue_paper_suite_artifact_bundle(
        report_with_control,
        output_dir=tmp_path / "with-control",
    )
    retained_payload = json.loads(
        (tmp_path / "with-control" / "paper_suite_aggregate.json").read_text(encoding="utf-8")
    )
    retained_claim = next(
        verdict for verdict in retained_payload["claim_verdicts"] if verdict["claim_id"] == "claim_rare_heavy_net_benefit"
    )
    assert retained_claim["status"] == "retain"
    assert ("canonical_gap_vs_no_rare_heavy_mean_delta", 0.25) in [tuple(item) for item in retained_claim["evidence"]]


def test_run_dialogue_pe_eta_longitudinal_benchmark_emits_cross_session_report():
    report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assert len(report.case_reports) == 2
    assert len(report.session_reports) == 2
    assert report.cross_session_report.verdict in {"growing", "stable", "regressing"}
    assert report.description


def test_build_dialogue_nl_essence_assessment_uses_nested_and_cross_session_evidence():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
    )

    assert assessment.path_label == "pe-eta"
    assert assessment.total_gate_count == 13
    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    assert "multi-timescale-default" in gate_map
    assert "semantic-spine-ready" in gate_map
    assert "default-continual-learner" in gate_map
    assert "online-fast-pe-coupling" in gate_map
    assert "cross-session-growth" in gate_map
    assert "rare-heavy-net-benefit" in gate_map


def test_dialogue_nl_essence_acceptance_fails_closed_without_cross_session_and_reset_evidence():
    benchmark_report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            runner_factory=_synthetic_runner,
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
    )
    decision = evaluate_dialogue_nl_essence_acceptance(assessment)

    assert isinstance(decision, DialogueNLEssenceAcceptanceDecision)
    assert decision.accepted is False
    assert "failed-gate:cross-session-growth" in decision.reasons


def test_slow_shapes_fast_gate_reports_config_artifact_when_case_count_is_below_proof_minimum():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
        proof_min_canonical_cases=2,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    slow_shapes_fast = gate_map["slow-shapes-fast"]

    assert slow_shapes_fast.passed is False
    evidence = dict(slow_shapes_fast.evidence)
    assert evidence["failure_mode"] == "config-artifact"
    assert evidence["proof_min_case_count_satisfied"] == 0.0


def test_dialogue_nl_essence_acceptance_default_now_passes_with_longitudinal_nested_evidence():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
    )

    decision = evaluate_dialogue_nl_essence_acceptance(assessment)

    assert decision.accepted is True
    assert not decision.blocked_gate_ids


def test_default_continual_learner_gate_retains_frozen_substrate_doctrine():
    ablation_report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )
    benchmark_report = next(
        path.benchmark_report
        for path in ablation_report.path_reports
        if path.path_label == "pe-eta"
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    gate = gate_map["default-continual-learner"]
    evidence = dict(gate.evidence)

    assert gate.passed is True
    assert evidence["mean_default_continual_learning_active"] > 0.0
    assert evidence["mean_default_owner_writeback_retained"] >= 0.5
    assert evidence["mean_default_substrate_live_mutation_suppressed"] >= 0.95
    assert evidence["online_fast_substrate_applied_count"] == 0.0


def test_rare_heavy_net_benefit_gate_fails_closed_without_no_rare_heavy_comparison():
    benchmark_report = asyncio.run(
        run_dialogue_pe_eta_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=_synthetic_runner,
        )
    )
    longitudinal_report = asyncio.run(
        run_dialogue_pe_eta_longitudinal_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
        )
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
    )

    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    rare_heavy_gate = gate_map["rare-heavy-net-benefit"]
    assert rare_heavy_gate.passed is False
    assert dict(rare_heavy_gate.evidence)["failure_mode"] == "missing-no-rare-heavy-comparison"


def test_explicit_pe_schedule_labels_do_not_receive_credit_below_runtime_reward_threshold():
    case = ScriptedDialogueCase(
        case_id="pe-threshold-alignment",
        description="Explicit -pe labels should still satisfy runtime PE schedule semantics.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.05,
            reward=0.10,
            action="evidence-only",
            regime="repair",
            abstract_action="observe",
            switch_gate=0.1,
            delayed_metric=0.2,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.04,
            reward=0.01,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="adapt",
            switch_gate=0.7,
            delayed_metric=0.3,
        ),
    )

    report = build_dialogue_case_report(
        case=case,
        turns=turns,
        allow_interval_carryover_credit=False,
    )

    assert report.high_pe_turn_count == 1
    assert report.pe_triggered_turn_count == 0


def test_case_report_tracks_store_nested_reset_deltas_without_turn_level_overcount():
    case = ScriptedDialogueCase(
        case_id="nested-reset-diagnostics",
        description="Case report should distinguish turn-level reset flags from owner-level reset count.",
        user_inputs=("turn-1", "turn-2", "turn-3"),
        expected_pressure_turns=(1,),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.25,
            reward=0.2,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="adapt",
            switch_gate=0.8,
            delayed_metric=0.2,
            pe_triggered=True,
            nested_context_reset_applied=True,
            nested_context_reset_total_count=1,
            slow_to_fast_init_benefit=0.02,
            slow_to_fast_target_distance_before=0.02,
            slow_to_fast_target_distance_after=0.0,
            slow_to_fast_target_alignment_gain=0.02,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.1,
            reward=0.02,
            action="evidence-only",
            regime="repair",
            abstract_action="stabilize",
            switch_gate=0.2,
            delayed_metric=0.4,
            nested_context_reset_total_count=1,
            slow_to_fast_init_benefit=0.02,
            slow_to_fast_target_distance_before=0.0,
            slow_to_fast_target_distance_after=0.0,
            slow_to_fast_target_alignment_gain=0.0,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.08,
            reward=0.01,
            action="evidence-only",
            regime="repair",
            abstract_action="stabilize",
            switch_gate=0.1,
            delayed_metric=0.6,
            nested_context_reset_total_count=1,
            slow_to_fast_init_benefit=0.02,
            slow_to_fast_target_distance_before=0.0,
            slow_to_fast_target_distance_after=0.0,
            slow_to_fast_target_alignment_gain=0.0,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.nested_context_reset_count == 1
    assert report.store_nested_context_reset_count == 1
    assert report.boundary_reset_observed_on_first_turn is True
    assert report.first_turn_slow_to_fast_init_benefit == 0.02
    assert report.mean_reset_turn_slow_to_fast_init_benefit == 0.02
    assert report.first_turn_slow_to_fast_target_distance_before == 0.02
    assert report.first_turn_slow_to_fast_target_distance_after == 0.0
    assert report.first_turn_slow_to_fast_target_alignment_gain == 0.02
    assert report.mean_reset_turn_slow_to_fast_target_distance_before == 0.02
    assert report.mean_reset_turn_slow_to_fast_target_distance_after == 0.0
    assert report.mean_reset_turn_slow_to_fast_target_alignment_gain == 0.02


def test_case_report_aggregates_memory_tower_metrics():
    case = ScriptedDialogueCase(
        case_id="tower-telemetry",
        description="Case report should aggregate memory tower telemetry beyond legacy reset metrics.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.22,
            reward=0.18,
            action="ssl-only-pe",
            regime="repair",
            abstract_action="adapt",
            switch_gate=0.75,
            delayed_metric=0.25,
            pe_triggered=True,
            tower_consolidation_count=1,
            memory_tower_depth=5,
            memory_tower_alignment=0.32,
            memory_tower_profile_id="mlp:nested:depth5",
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.08,
            reward=0.03,
            action="evidence-only",
            regime="repair",
            abstract_action="stabilize",
            switch_gate=0.20,
            delayed_metric=0.55,
            tower_consolidation_count=2,
            memory_tower_depth=5,
            memory_tower_alignment=0.46,
            memory_tower_profile_id="mlp:nested:depth5",
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.max_tower_consolidation_count == 2
    assert report.mean_memory_tower_depth == 5.0
    assert report.mean_memory_tower_alignment > 0.3
    assert report.memory_tower_profile_turn_count == 2


def test_case_report_aggregates_semantic_spine_readiness_metrics():
    case = ScriptedDialogueCase(
        case_id="semantic-spine-telemetry",
        description="Case report should expose semantic spine evidence from evaluation metrics.",
        user_inputs=("turn-1", "turn-2"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.08,
            reward=0.10,
            action="evidence-only",
            regime="stable",
            abstract_action="observe",
            switch_gate=0.10,
            delayed_metric=0.30,
            extra_outcome_metrics=(
                ("learning:semantic_spine_coverage", 1.0),
                ("learning:cognitive_loop_readiness", 0.42),
            ),
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.06,
            reward=0.12,
            action="ssl-only-pe",
            regime="stable",
            abstract_action="adapt",
            switch_gate=0.35,
            delayed_metric=0.45,
            extra_outcome_metrics=(
                ("learning:semantic_spine_coverage", 1.0),
                ("learning:cognitive_loop_readiness", 0.58),
            ),
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)
    summary = dict(_case_summary_metrics(report))

    assert report.mean_semantic_spine_coverage == 1.0
    assert report.mean_cognitive_loop_readiness == 0.5
    assert summary["mean_semantic_spine_coverage"] == 1.0
    assert summary["mean_cognitive_loop_readiness"] == 0.5


def test_nl_essence_assessment_surfaces_semantic_spine_ready_gate():
    case = ScriptedDialogueCase(
        case_id="semantic-spine-gate",
        description="NL essence should surface semantic spine readiness as an audit gate.",
        user_inputs=("turn-1", "turn-2"),
    )
    case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.10,
                reward=0.12,
                action="evidence-only",
                regime="stable",
                abstract_action="observe",
                switch_gate=0.10,
                delayed_metric=0.30,
                extra_outcome_metrics=(
                    ("learning:semantic_spine_coverage", 1.0),
                    ("learning:cognitive_loop_readiness", 0.40),
                ),
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.06,
                reward=0.15,
                action="ssl-only-pe",
                regime="stable",
                abstract_action="adapt",
                switch_gate=0.35,
                delayed_metric=0.55,
                extra_outcome_metrics=(
                    ("learning:semantic_spine_coverage", 1.0),
                    ("learning:cognitive_loop_readiness", 0.60),
                ),
            ),
        ),
    )
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=1,
        total_case_count=1,
        metric_means=_case_summary_metrics(case_report),
        description="semantic spine gate benchmark report",
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
    )
    gate = {item.gate_id: item for item in assessment.gates}["semantic-spine-ready"]
    evidence = dict(gate.evidence)

    assert gate.passed is True
    assert evidence["mean_semantic_spine_coverage"] == 1.0
    assert evidence["mean_cognitive_loop_readiness"] == 0.5


def test_case_report_surfaces_rare_heavy_preimport_selection_telemetry():
    case = ScriptedDialogueCase(
        case_id="rare-heavy-preimport-telemetry",
        description="Case report should expose pre-import rare-heavy selection and rejection evidence.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.28,
                reward=0.18,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.75,
                delayed_metric=0.2,
                pe_triggered=True,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.10,
                reward=0.01,
                action="full-cycle-pe",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.25,
                delayed_metric=0.6,
            ),
        ),
    )

    updated_turns = (
        replace(
            report.turns[0],
            rare_heavy_recommended=True,
            rare_heavy_import_decision="rejected-pre-import",
            rare_heavy_reject_reason="pre-import-positive-fraction-too-low",
            rare_heavy_pre_import_passed=False,
            rare_heavy_pre_import_mean_score_delta=-0.08,
            rare_heavy_candidate_alignment=1.0,
            rare_heavy_candidate_adapter_parameter_count=12,
        ),
        replace(
            report.turns[1],
            rare_heavy_recommended=True,
            rare_heavy_applied=True,
            rare_heavy_import_decision="imported",
            rare_heavy_pre_import_passed=True,
            rare_heavy_pre_import_mean_score_delta=0.12,
            rare_heavy_candidate_alignment=1.0,
            rare_heavy_candidate_adapter_parameter_count=18,
        ),
    )
    updated_report = build_dialogue_case_report(case=case, turns=updated_turns)
    assert updated_report.rare_heavy_pre_import_pass_count == 1
    assert updated_report.rare_heavy_pre_import_reject_count == 1
    assert updated_report.mean_rare_heavy_pre_import_score_delta > 0.0
    assert updated_report.mean_rare_heavy_candidate_alignment == 1.0
    assert updated_report.max_rare_heavy_candidate_adapter_parameter_count == 18


def test_nl_essence_assessment_reads_memory_tower_surface_metrics():
    case = ScriptedDialogueCase(
        case_id="tower-surface-gate",
        description="Assessment should read tower depth/alignment/consolidation evidence directly.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.25,
                reward=0.2,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.8,
                delayed_metric=0.25,
                pe_triggered=True,
                nested_context_reset_applied=True,
                nested_context_reset_total_count=1,
                slow_to_fast_init_benefit=0.02,
                slow_to_fast_target_distance_before=0.02,
                slow_to_fast_target_distance_after=0.0,
                slow_to_fast_target_alignment_gain=0.02,
                tower_consolidation_count=1,
                memory_tower_depth=5,
                memory_tower_alignment=0.35,
                memory_tower_profile_id="mlp:nested:depth5",
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.08,
                reward=0.03,
                action="full-cycle-pe",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.2,
                delayed_metric=0.6,
                tower_consolidation_count=2,
                memory_tower_depth=5,
                memory_tower_alignment=0.42,
                memory_tower_profile_id="mlp:nested:depth5",
            ),
        ),
    )
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=1,
        total_case_count=1,
        metric_means=(
            ("passed", 1.0),
            ("prediction_chain_turn_count", float(case_report.prediction_chain_turn_count)),
            ("pe_triggered_turn_count", float(case_report.pe_triggered_turn_count)),
            ("delayed_improvement_observed", 1.0),
            ("online_learning_turn_count", float(case_report.online_learning_turn_count)),
            ("bounded_writeback_turn_count", float(case_report.bounded_writeback_turn_count)),
            ("session_post_completion_turn_count", float(case_report.session_post_completion_turn_count)),
            ("reflection_promotion_eligible_turn_count", float(case_report.reflection_promotion_eligible_turn_count)),
            ("rare_heavy_recommended_count", float(case_report.rare_heavy_recommended_count)),
            ("nested_profile_active_turn_count", float(case_report.nested_profile_active_turn_count)),
            ("learned_memory_primary_turn_count", float(case_report.learned_memory_primary_turn_count)),
            ("memory_tower_profile_turn_count", float(case_report.memory_tower_profile_turn_count)),
            ("mean_memory_tower_depth", case_report.mean_memory_tower_depth),
            ("mean_memory_tower_alignment", case_report.mean_memory_tower_alignment),
            ("max_tower_consolidation_count", float(case_report.max_tower_consolidation_count)),
            ("store_nested_context_reset_count", float(case_report.store_nested_context_reset_count)),
            ("mean_reset_turn_slow_to_fast_init_benefit", case_report.mean_reset_turn_slow_to_fast_init_benefit),
            ("mean_reset_turn_slow_to_fast_target_distance_before", case_report.mean_reset_turn_slow_to_fast_target_distance_before),
            ("mean_reset_turn_slow_to_fast_target_distance_after", case_report.mean_reset_turn_slow_to_fast_target_distance_after),
            ("mean_reset_turn_slow_to_fast_target_alignment_gain", case_report.mean_reset_turn_slow_to_fast_target_alignment_gain),
        ),
        description="tower evidence benchmark report",
    )

    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=DialogueBenchmarkComparisonReport(
            baseline_label="pe-eta",
            path_reports=(),
            case_deltas_from_baseline=(),
            metric_deltas_from_baseline=(
                (
                    "pe-drive-off",
                    (
                        ("mean_memory_tower_depth", -1.2),
                        ("mean_memory_tower_alignment", -0.18),
                        ("max_tower_consolidation_count", -1.0),
                    ),
                ),
            ),
            description="synthetic comparison",
        ),
    )
    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    tower_gate = gate_map["tower-memory-surface"]

    assert tower_gate.passed is True
    evidence = dict(tower_gate.evidence)
    assert evidence["mean_memory_tower_depth"] >= 5.0
    assert evidence["max_tower_consolidation_count"] >= 2.0
    assert evidence["tower_effective_strength"] > 0.65
    assert evidence["tower_strength_gap_vs_best_control"] > 0.0


def test_tower_memory_surface_gate_fail_closes_when_depth_exists_but_alignment_and_consolidation_are_weak():
    case = ScriptedDialogueCase(
        case_id="tower-depth-only",
        description="Tower depth alone should not be enough when alignment and consolidation are weak.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.18,
                reward=0.08,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.7,
                delayed_metric=0.2,
                pe_triggered=True,
                memory_tower_depth=6,
                memory_tower_alignment=0.02,
                tower_consolidation_count=0,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.09,
                reward=0.04,
                action="evidence-only",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.2,
                delayed_metric=0.3,
                memory_tower_depth=6,
                memory_tower_alignment=0.03,
                tower_consolidation_count=0,
            ),
        ),
    )
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=int(case_report.passed),
        total_case_count=1,
        metric_means=(
            ("passed", 1.0),
            ("prediction_chain_turn_count", float(case_report.prediction_chain_turn_count)),
            ("pe_triggered_turn_count", float(case_report.pe_triggered_turn_count)),
            ("delayed_improvement_observed", 1.0),
            ("online_learning_turn_count", float(case_report.online_learning_turn_count)),
            ("bounded_writeback_turn_count", float(case_report.bounded_writeback_turn_count)),
            ("session_post_completion_turn_count", float(case_report.session_post_completion_turn_count)),
            ("reflection_promotion_eligible_turn_count", float(case_report.reflection_promotion_eligible_turn_count)),
            ("rare_heavy_recommended_count", float(case_report.rare_heavy_recommended_count)),
            ("nested_profile_active_turn_count", float(case_report.nested_profile_active_turn_count)),
            ("learned_memory_primary_turn_count", float(case_report.learned_memory_primary_turn_count)),
            ("memory_tower_profile_turn_count", float(case_report.memory_tower_profile_turn_count)),
            ("mean_memory_tower_depth", case_report.mean_memory_tower_depth),
            ("mean_memory_tower_alignment", case_report.mean_memory_tower_alignment),
            ("max_tower_consolidation_count", float(case_report.max_tower_consolidation_count)),
            ("max_artifact_consolidation_count", float(case_report.max_artifact_consolidation_count)),
        ),
        description="depth-only tower benchmark report",
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=DialogueBenchmarkComparisonReport(
            baseline_label="pe-eta",
            path_reports=(),
            case_deltas_from_baseline=(),
            metric_deltas_from_baseline=(
                (
                    "pe-drive-off",
                    (
                        ("mean_memory_tower_depth", -0.2),
                        ("mean_memory_tower_alignment", -0.01),
                        ("max_tower_consolidation_count", 0.0),
                    ),
                ),
            ),
            description="weak synthetic comparison",
        ),
        proof_min_canonical_cases=1,
    )
    tower_gate = {gate.gate_id: gate for gate in assessment.gates}["tower-memory-surface"]
    evidence = dict(tower_gate.evidence)

    assert tower_gate.passed is False
    assert evidence["tower_effective_strength"] < 0.65
    assert evidence["tower_consolidation_evidence"] == 0.0


def test_slow_shapes_fast_gate_identifies_already_near_target_when_benefit_is_small():
    case = ScriptedDialogueCase(
        case_id="nested-already-near-target",
        description="Small displacement can still mean reset is healthy if the state was already near its target.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.22,
                reward=0.12,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.8,
                delayed_metric=0.2,
                pe_triggered=True,
                nested_context_reset_applied=True,
                nested_context_reset_total_count=1,
                slow_to_fast_init_benefit=0.002,
                slow_to_fast_target_distance_before=0.002,
                slow_to_fast_target_distance_after=0.001,
                slow_to_fast_target_alignment_gain=0.001,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.09,
                reward=0.01,
                action="evidence-only",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.2,
                delayed_metric=0.5,
                nested_context_reset_total_count=1,
                slow_to_fast_target_distance_before=0.0,
                slow_to_fast_target_distance_after=0.0,
                slow_to_fast_target_alignment_gain=0.0,
            ),
        ),
    )
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=int(case_report.passed),
        total_case_count=1,
        metric_means=tuple(
            (key, float(value))
            for key, value in (
                ("prediction_chain_turn_count", float(case_report.prediction_chain_turn_count)),
                ("pe_triggered_turn_count", float(case_report.pe_triggered_turn_count)),
                ("delayed_improvement_observed", float(case_report.delayed_improvement_observed)),
                ("online_learning_turn_count", float(case_report.online_learning_turn_count)),
                ("bounded_writeback_turn_count", float(case_report.bounded_writeback_turn_count)),
                ("nested_profile_active_turn_count", float(case_report.nested_profile_active_turn_count)),
                ("learned_memory_primary_turn_count", float(case_report.learned_memory_primary_turn_count)),
                ("store_nested_context_reset_count", float(case_report.store_nested_context_reset_count)),
                ("mean_reset_turn_slow_to_fast_init_benefit", case_report.mean_reset_turn_slow_to_fast_init_benefit),
                (
                    "mean_reset_turn_slow_to_fast_target_distance_before",
                    case_report.mean_reset_turn_slow_to_fast_target_distance_before,
                ),
                (
                    "mean_reset_turn_slow_to_fast_target_distance_after",
                    case_report.mean_reset_turn_slow_to_fast_target_distance_after,
                ),
                (
                    "mean_reset_turn_slow_to_fast_target_alignment_gain",
                    case_report.mean_reset_turn_slow_to_fast_target_alignment_gain,
                ),
                ("boundary_reset_observed_on_first_turn", float(case_report.boundary_reset_observed_on_first_turn)),
                ("evolution_judge_turn_count", float(case_report.evolution_judge_turn_count)),
                ("evolution_judge_structural_allow_count", float(case_report.evolution_judge_structural_allow_count)),
                ("core_guided_recall_turn_count", float(case_report.core_guided_recall_turn_count)),
                ("mean_learned_recall_confidence", case_report.mean_learned_recall_confidence),
                ("max_artifact_consolidation_count", float(case_report.max_artifact_consolidation_count)),
            )
        ),
        description="single-case benchmark report",
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        proof_min_canonical_cases=1,
    )
    slow_shapes_fast_gate = {gate.gate_id: gate for gate in assessment.gates}["slow-shapes-fast"]

    assert slow_shapes_fast_gate.passed is False
    evidence = dict(slow_shapes_fast_gate.evidence)
    assert evidence["failure_mode"] == "already-near-target"
    assert evidence["alignment_interpretation"] == "already-near-target"
    assert evidence["weak_benefit_explained_by_target_proximity"] == 1.0


def test_slow_shapes_fast_gate_accepts_distributed_signal_when_reset_alignment_and_tower_evidence_are_strong():
    case = ScriptedDialogueCase(
        case_id="nested-distributed-signal",
        description="Slow-to-fast proof can pass with distributed nested evidence even when raw init benefit is modest.",
        user_inputs=("turn-1", "turn-2"),
        expected_pressure_turns=(1,),
    )
    case_report = build_dialogue_case_report(
        case=case,
        turns=(
            _benchmark_turn(
                turn_index=1,
                pe=0.24,
                reward=0.10,
                action="ssl-only-pe",
                regime="repair",
                abstract_action="adapt",
                switch_gate=0.8,
                delayed_metric=0.2,
                pe_triggered=True,
                nested_context_reset_applied=True,
                nested_context_reset_total_count=1,
                slow_to_fast_init_benefit=0.002,
                slow_to_fast_target_distance_before=0.02,
                slow_to_fast_target_distance_after=0.01,
                slow_to_fast_target_alignment_gain=0.004,
                memory_tower_depth=6,
                memory_tower_alignment=0.20,
                memory_tower_profile_id="mlp:nested:depth6",
                learned_recall_count=2,
                learned_recall_confidence=0.45,
                learned_recall_core_guided=True,
            ),
            _benchmark_turn(
                turn_index=2,
                pe=0.08,
                reward=0.04,
                action="full-cycle-pe",
                regime="repair",
                abstract_action="stabilize",
                switch_gate=0.2,
                delayed_metric=0.5,
                nested_context_reset_total_count=1,
                memory_tower_depth=6,
                memory_tower_alignment=0.22,
                memory_tower_profile_id="mlp:nested:depth6",
                learned_recall_count=2,
                learned_recall_confidence=0.45,
                learned_recall_core_guided=True,
                tower_consolidation_count=1,
            ),
        ),
    )
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(case_report,),
        passed_case_count=int(case_report.passed),
        total_case_count=1,
        metric_means=(
            ("prediction_chain_turn_count", float(case_report.prediction_chain_turn_count)),
            ("pe_triggered_turn_count", float(case_report.pe_triggered_turn_count)),
            ("delayed_improvement_observed", float(case_report.delayed_improvement_observed)),
            ("online_learning_turn_count", float(case_report.online_learning_turn_count)),
            ("bounded_writeback_turn_count", float(case_report.bounded_writeback_turn_count)),
            ("nested_profile_active_turn_count", float(case_report.nested_profile_active_turn_count)),
            ("learned_memory_primary_turn_count", float(case_report.learned_memory_primary_turn_count)),
            ("store_nested_context_reset_count", float(case_report.store_nested_context_reset_count)),
            ("mean_reset_turn_slow_to_fast_init_benefit", case_report.mean_reset_turn_slow_to_fast_init_benefit),
            (
                "mean_reset_turn_slow_to_fast_target_distance_before",
                case_report.mean_reset_turn_slow_to_fast_target_distance_before,
            ),
            (
                "mean_reset_turn_slow_to_fast_target_distance_after",
                case_report.mean_reset_turn_slow_to_fast_target_distance_after,
            ),
            (
                "mean_reset_turn_slow_to_fast_target_alignment_gain",
                case_report.mean_reset_turn_slow_to_fast_target_alignment_gain,
            ),
            ("boundary_reset_observed_on_first_turn", float(case_report.boundary_reset_observed_on_first_turn)),
            ("core_guided_recall_turn_count", float(case_report.core_guided_recall_turn_count)),
            ("mean_learned_recall_confidence", case_report.mean_learned_recall_confidence),
            ("max_artifact_consolidation_count", float(case_report.max_artifact_consolidation_count)),
            ("max_tower_consolidation_count", float(case_report.max_tower_consolidation_count)),
            ("mean_memory_tower_depth", case_report.mean_memory_tower_depth),
            ("mean_memory_tower_alignment", case_report.mean_memory_tower_alignment),
            ("memory_tower_profile_turn_count", float(case_report.memory_tower_profile_turn_count)),
        ),
        description="distributed slow-to-fast evidence benchmark report",
    )
    longitudinal_report = DialogueLongitudinalBenchmarkReport(
        case_reports=(case_report,),
        session_reports=(),
        cross_session_report=CrossSessionGrowthReport(
            window_trends=(),
            family_persistence=0.6,
            regime_effectiveness_delta=0.1,
            verdict="stable",
            description="synthetic cross-session report",
        ),
        description="distributed slow-to-fast longitudinal report",
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        longitudinal_report=longitudinal_report,
        proof_min_canonical_cases=1,
    )
    slow_shapes_fast_gate = {gate.gate_id: gate for gate in assessment.gates}["slow-shapes-fast"]
    evidence = dict(slow_shapes_fast_gate.evidence)

    assert slow_shapes_fast_gate.passed is True
    assert evidence["slow_to_fast_signal_strength"] > 0.42


def test_rare_heavy_gate_accepts_review_only_candidate_evidence_when_it_is_non_regressive():
    benchmark_report = DialogueBenchmarkReport(
        case_reports=(),
        passed_case_count=1,
        total_case_count=1,
        metric_means=(
            ("prediction_chain_turn_count", 1.0),
            ("pe_triggered_turn_count", 1.0),
            ("delayed_improvement_observed", 1.0),
            ("online_learning_turn_count", 1.0),
            ("bounded_writeback_turn_count", 1.0),
            ("nested_profile_active_turn_count", 1.0),
            ("learned_memory_primary_turn_count", 1.0),
            ("rare_heavy_recommended_count", 1.0),
            ("rare_heavy_applied_count", 0.0),
            ("rare_heavy_pre_import_pass_count", 1.0),
            ("mean_rare_heavy_pre_import_score_delta", 0.08),
            ("mean_rare_heavy_candidate_alignment", 0.62),
            ("max_rare_heavy_candidate_adapter_parameter_count", 128.0),
        ),
        description="review-only rare-heavy evidence benchmark report",
    )
    assessment = build_dialogue_nl_essence_assessment(
        path_label="pe-eta",
        benchmark_report=benchmark_report,
        comparison_report=DialogueBenchmarkComparisonReport(
            baseline_label="pe-eta",
            path_reports=(),
            case_deltas_from_baseline=(),
            metric_deltas_from_baseline=(),
            rare_heavy_metric_deltas=(
                ("rare_heavy_recommended_count", 0.0),
                ("rare_heavy_applied_count", 0.0),
                ("rare_heavy_pre_import_pass_count", 0.0),
                ("mean_rare_heavy_pre_import_score_delta", 0.0),
                ("mean_rare_heavy_candidate_alignment", 0.0),
                ("passed", 0.0),
                ("delayed_improvement_observed", 0.0),
                ("stability_after_recovery_score", 0.0),
                ("mean_prediction_error", 0.0),
            ),
            description="review-only rare-heavy comparison",
        ),
    )
    rare_heavy_gate = {gate.gate_id: gate for gate in assessment.gates}["rare-heavy-net-benefit"]
    evidence = dict(rare_heavy_gate.evidence)

    assert rare_heavy_gate.passed is True
    assert evidence["review_evidence_present"] == 1.0
    assert evidence["current_rare_heavy_pre_import_pass_count"] == 1.0


def test_build_real_dialogue_comprehensive_runner_factories_share_runtime():
    factories = build_real_dialogue_comprehensive_runner_factories(
        runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
    )
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    variant = DEFAULT_DIALOGUE_CASE_VARIANTS[0]
    scenario = DEFAULT_OPEN_DIALOGUE_SCENARIOS[0]
    runner = factories.canonical_runner_factory("pe-eta", case)
    acceptance_runner = factories.acceptance_runner_factory(variant)
    open_runner = factories.open_runner_factory("pe-eta", scenario)

    assert runner._default_residual_runtime is factories.residual_runtime
    assert acceptance_runner._default_residual_runtime is factories.residual_runtime
    assert open_runner._default_residual_runtime is factories.residual_runtime
    assert factories.description


def test_run_real_dialogue_pe_eta_comprehensive_benchmark_completes_with_builtin_runtime():
    progress_messages: list[str] = []
    report = asyncio.run(
        run_real_dialogue_pe_eta_comprehensive_benchmark(
            config=default_dialogue_real_proof_config(),
            progress_callback=progress_messages.append,
        )
    )

    assert report.canonical_ablation_report.baseline_label == "pe-eta"
    assert len(report.longitudinal_report.case_reports) == 2
    assert report.longitudinal_report.cross_session_report.verdict in {"growing", "stable", "regressing", "insufficient-data"}
    assert report.essence_report.total_gate_count == 13
    assert isinstance(report.essence_acceptance, DialogueNLEssenceAcceptanceDecision)
    assert report.open_ablation_report is not None
    assert report.open_ablation_report.baseline_label == "pe-eta"
    assert report.emergence_dashboard.strong_proof_panels
    assert report.emergence_dashboard.open_environment_panels
    assert len(report.selection_artifact.selected_variants) == 1
    assert len(report.artifact_comparison_report.candidate_reports) == 1
    assert "rare_heavy_gate_passed=" in report.description
    assert "rare_heavy_gate_failure_mode=" in report.description
    assert "emergence_interpretation=" in report.description
    assert "shared runtime" in report.description
    assert "open_scenarios=" in report.description
    assert progress_messages[0].startswith("Building shared real residual runtime")
    assert progress_messages[-1] == "Real comprehensive benchmark finished."


def test_run_real_dialogue_pe_eta_comprehensive_benchmark_staged_persists_and_resumes(tmp_path):
    config = default_dialogue_real_proof_config(runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY)
    first_progress: list[str] = []
    first_report = asyncio.run(
        run_real_dialogue_pe_eta_comprehensive_benchmark_staged(
            output_dir=tmp_path,
            config=config,
            shared_factories=_synthetic_shared_factories(),
            longitudinal_runner_factory=lambda: _synthetic_runner(DEFAULT_DIALOGUE_PROOF_CASES[0]),
            progress_callback=first_progress.append,
        )
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert DialogueComprehensiveStage.FINAL_REPORT.value in manifest["completed_stages"]
    assert manifest["summary"]["stage"] == DialogueComprehensiveStage.FINAL_REPORT.value
    assert manifest["summary"]["rare_heavy_gate"]["present"] is True
    assert "failure_mode" in manifest["summary"]["rare_heavy_gate"]
    assert "essence_accepted" in manifest["summary"]
    assert manifest["summary"]["artifact_acceptance"]["candidate_label"] is not None
    assert "override_mode" in manifest["summary"]["artifact_acceptance"]
    assert "reasons" in manifest["summary"]["artifact_acceptance"]
    assert "pre_import_evidence" in manifest["summary"]["artifact_acceptance"]
    assert "pre_import_pass_fraction" in manifest["summary"]["artifact_acceptance"]["pre_import_evidence"]
    assert "mean_pre_import_score_delta" in manifest["summary"]["artifact_acceptance"]["pre_import_evidence"]
    assert manifest["summary"]["emergence_dashboard"]["baseline_label"] == "pe-eta"
    assert "interpretation" in manifest["summary"]["emergence_dashboard"]
    assert manifest["summary"]["open_profile_count"] >= 1
    assert manifest["summary"]["open_scenario_count"] >= 1
    assert first_report.selection_artifact.selected_variants
    assert "rare_heavy_gate_passed=" in first_report.description
    assert "emergence_interpretation=" in first_report.description
    assert "open_scenarios=" in first_report.description
    assert any("Running canonical ablation" in message for message in first_progress)
    assert any("Running open-environment ablation" in message for message in first_progress)

    second_progress: list[str] = []
    resumed_report = asyncio.run(
        run_real_dialogue_pe_eta_comprehensive_benchmark_staged(
            output_dir=tmp_path,
            config=config,
            resume=True,
            progress_callback=second_progress.append,
        )
    )

    assert resumed_report.description == first_report.description
    assert second_progress == ["Resuming final comprehensive report from checkpoint."]


def test_default_dialogue_replay_seeds_are_exposed():
    assert DEFAULT_DIALOGUE_REPLAY_SEEDS == (0, 1, 2)


def test_default_dialogue_real_proof_config_requires_multiple_canonical_cases():
    config = default_dialogue_real_proof_config()

    assert isinstance(config, DialogueRealComprehensiveBenchmarkConfig)
    assert config.profile_labels == default_dialogue_strong_proof_profiles()
    assert config.open_profile_labels == default_open_dialogue_ablation_profiles()
    assert config.canonical_case_limit == 2
    assert config.open_scenario_limit == 1
    assert config.proof_min_canonical_cases == 2
    assert config.runtime_mode == LocalSubstrateRuntimeMode.BUILTIN_ONLY


def test_pe_drive_off_runner_disables_external_prediction_error_drive():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="pe-drive-off", case=case)

    assert runner._external_prediction_error_drive is False
    assert runner._joint_schedule.pe_ssl_threshold < 900.0
    assert runner.residual_runtime.supports_live_substrate_mutation is False


def test_timescale_off_runner_uses_non_nested_memory_profile():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="timescale-off", case=case)
    memory_snapshot = runner._memory_store.snapshot(retrieved_entries=())
    lifecycle_metrics = dict(memory_snapshot.lifecycle_metrics)

    assert lifecycle_metrics["nested_profile_active"] == 0.0
    assert runner.residual_runtime.supports_live_substrate_mutation is False


def test_eta_off_runner_uses_learned_lite_temporal_policy_and_no_joint_learning():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="eta-off", case=case)

    assert runner._temporal_policy.mode.value == "learned-lite"
    assert runner._joint_schedule.ssl_interval == 0
    assert runner._joint_schedule.rl_interval == 0
    assert runner._external_prediction_error_drive is False
    assert runner._external_prediction_error_drive is False
    assert runner.residual_runtime.supports_live_substrate_mutation is False


def test_pe_eta_runner_uses_live_substrate_mutation_by_default():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(profile_label="pe-eta", case=case)

    assert runner.residual_runtime.supports_live_substrate_mutation is True


def test_scaffold_ablation_runners_expose_expected_temporal_controls():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    no_label_runner = build_standard_dialogue_runner(
        profile_label="pe-eta-no-semantic-label",
        case=case,
    )
    no_reflection_cache_runner = build_standard_dialogue_runner(
        profile_label="pe-eta-no-reflection-cache",
        case=case,
    )

    assert no_label_runner._world_temporal_policy.mode.value == "full-learned"
    assert no_reflection_cache_runner._world_temporal_policy.mode.value == "full-learned"
    assert no_label_runner._external_prediction_error_drive is True
    assert no_reflection_cache_runner._external_prediction_error_drive is True


def test_pe_readout_only_runner_keeps_pe_visible_without_primary_dominance():
    case = DEFAULT_DIALOGUE_PROOF_CASES[0]
    runner = build_standard_dialogue_runner(
        profile_label="pe-eta-pe-readout-only",
        case=case,
    )

    assert runner._external_prediction_error_drive is False
    assert runner._prediction_error_readout_only is True
    assert runner._primary_prediction_error_dominance_enabled is False
    assert runner._joint_loop.primary_prediction_error_dominance_enabled is False
    assert runner._joint_loop._world_sandbox._env.primary_prediction_error_enabled is False


def test_scaffold_ablation_profiles_preserve_latent_family_and_pe_schedule():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-semantic-label", "pe-eta-no-reflection-cache"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    path_reports = {
        path.path_label: path.benchmark_report.case_reports[0]
        for path in report.path_reports
    }
    no_label = path_reports["pe-eta-no-semantic-label"]
    no_reflection_cache = path_reports["pe-eta-no-reflection-cache"]

    assert max(turn.action_family_version for turn in no_label.turns) > 0
    assert no_label.pe_triggered_turn_count > 0
    assert all(turn.active_abstract_action.startswith("latent-family-v") for turn in no_label.turns)

    assert max(turn.action_family_version for turn in no_reflection_cache.turns) > 0
    assert no_reflection_cache.pe_triggered_turn_count > 0
    assert no_reflection_cache.pressure_response_precision >= 0.0
    assert no_reflection_cache.stability_after_recovery_score >= 0.0

    strong_proof_metric_deltas = dict(report.strong_proof_metric_deltas)
    assert "pe-eta-no-semantic-label" in strong_proof_metric_deltas
    assert "pe-eta-no-reflection-cache" in strong_proof_metric_deltas


def test_strong_proof_deltas_surface_scaffold_and_pe_readout_profiles_explicitly():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=(
                "pe-eta",
                "pe-eta-no-semantic-label",
                "pe-eta-no-reflection-cache",
                "pe-eta-pe-readout-only",
            ),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    strong_case_map = {
        case_id: dict(path_deltas)
        for case_id, path_deltas in report.strong_proof_case_deltas
    }
    assert "repair" in strong_case_map
    assert {
        "pe-eta-no-semantic-label",
        "pe-eta-no-reflection-cache",
        "pe-eta-pe-readout-only",
    } <= set(strong_case_map["repair"])
    strong_metric_map = dict(report.strong_proof_metric_deltas)
    assert "pe-eta-no-semantic-label" in strong_metric_map
    assert "pe-eta-no-reflection-cache" in strong_metric_map
    assert "pe-eta-pe-readout-only" in strong_metric_map


def test_pe_readout_only_profile_preserves_prediction_chain_but_reduces_pe_trigger_dominance():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-pe-readout-only"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    path_reports = {
        path.path_label: path.benchmark_report.case_reports[0]
        for path in report.path_reports
    }
    pe_eta = path_reports["pe-eta"]
    pe_readout_only = path_reports["pe-eta-pe-readout-only"]
    delta_map = dict(report.case_deltas_from_baseline[0][1])

    assert pe_readout_only.prediction_chain_turn_count > 0
    assert max(turn.action_family_version for turn in pe_readout_only.turns) > 0
    assert pe_readout_only.pe_triggered_turn_count <= pe_eta.pe_triggered_turn_count
    assert dict(delta_map["pe-eta-pe-readout-only"])["pe_triggered_turn_count"] <= 0.0


def test_pe_dominance_comparison_report_quantifies_mechanism_retention():
    comparison = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-drive-off", "pe-eta-pe-readout-only"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    report = build_pe_dominance_comparison_report(comparison)

    assert report.baseline_label == "pe-eta"
    assert report.pe_drive_off_label == "pe-drive-off"
    assert report.pe_readout_only_label == "pe-eta-pe-readout-only"
    assert report.mechanism_retention_ratio >= 0.0
    assert report.pe_visibility_retention_ratio > 0.0
    assert isinstance(report.interpretation, str)
    assert "mechanism_retention=" in report.description


def test_pe_dominance_case_diagnosis_report_identifies_worst_case_and_failure_mode():
    comparison = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:2],
            profile_labels=("pe-eta", "pe-drive-off", "pe-eta-pe-readout-only"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    diagnosis = build_pe_dominance_case_diagnosis_report(comparison)

    assert diagnosis.baseline_label == "pe-eta"
    assert diagnosis.pe_drive_off_label == "pe-drive-off"
    assert diagnosis.pe_readout_only_label == "pe-eta-pe-readout-only"
    assert diagnosis.worst_case_id is not None
    assert diagnosis.dominant_failure_mode in {
        "schedule-driven",
        "reward-driven",
        "latent-fragility-driven",
    }
    assert len(diagnosis.case_diagnoses) == 2
    assert diagnosis.case_diagnoses[0].degradation_severity >= diagnosis.case_diagnoses[-1].degradation_severity
    assert "dominant_failure_mode=" in diagnosis.description


def test_default_rare_heavy_candidate_configs_are_exposed():
    configs = default_rare_heavy_candidate_configs()

    assert configs == DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS
    assert len(configs) >= 3


def test_eta_no_pe_baseline_does_not_receive_interval_carryover_credit():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            runner_factory=_synthetic_ablation_runner,
        )
    )

    case_id, path_deltas = report.case_deltas_from_baseline[0]
    assert case_id == "repair"
    delta_map = {label: dict(deltas) for label, deltas in path_deltas}

    assert delta_map["pe-eta"]["pe_triggered_turn_count"] == 0.0
    assert delta_map["pe-drive-off"]["pe_triggered_turn_count"] < 0.0


def test_run_multi_artifact_acceptance_benchmark_orders_candidates():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    comparison = asyncio.run(
        run_multi_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
        )
    )

    assert len(comparison.candidate_reports) == len(DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS)
    assert comparison.chosen_candidate_label is not None
    assert comparison.candidate_reports[0].candidate_score >= comparison.candidate_reports[-1].candidate_score


def test_run_multi_artifact_acceptance_benchmark_can_choose_accepted_candidate():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )
    comparison = asyncio.run(
        run_multi_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
            gate_config=DialogueArtifactAcceptanceGateConfig(
                min_mean_score_delta=-1.0,
                min_passed_case_delta=-10,
                min_positive_case_fraction=0.0,
                min_worst_case_delta=-1.0,
            ),
        )
    )

    assert comparison.chosen_candidate_label is not None
    assert comparison.chosen_accepted is True


def test_artifact_acceptance_can_use_minimal_graded_gain_override_when_only_passed_case_delta_is_missing():
    decision = evaluate_dialogue_artifact_acceptance(
        mean_score_delta=0.42,
        passed_case_delta=0,
        positive_case_fraction=0.8,
        worst_case_delta=0.0,
        substrate_checkpoint_present=True,
        substrate_update_count=2,
        substrate_source_batch_count=2,
        substrate_mean_sequence_length=1.2,
        substrate_mean_residual_magnitude=0.05,
        substrate_import_success_fraction=1.0,
        gate_config=DialogueArtifactAcceptanceGateConfig(),
    )

    assert decision.accepted is True
    assert decision.override_mode == "graded-gain"
    assert "passed-case-delta-below-threshold" not in decision.reasons


def test_artifact_acceptance_rejects_when_graded_gain_still_has_negative_tail():
    decision = evaluate_dialogue_artifact_acceptance(
        mean_score_delta=0.42,
        passed_case_delta=0,
        positive_case_fraction=0.8,
        worst_case_delta=-0.01,
        substrate_checkpoint_present=True,
        substrate_update_count=2,
        substrate_source_batch_count=2,
        substrate_mean_sequence_length=1.2,
        substrate_mean_residual_magnitude=0.05,
        substrate_import_success_fraction=1.0,
        gate_config=DialogueArtifactAcceptanceGateConfig(),
    )

    assert decision.accepted is False
    assert decision.override_mode == "none"
    assert "passed-case-delta-below-threshold" in decision.reasons


def test_replay_selection_acceptance_reports_substrate_evidence():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_perturbation_runner,
        )
    )
    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        replay_ranking_report=ranking,
        top_k=2,
    )

    acceptance_report = asyncio.run(
        run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            runner_factory=_synthetic_acceptance_runner,
        )
    )

    evidence = dict(acceptance_report.substrate_evidence)
    pre_import_evidence = dict(acceptance_report.pre_import_evidence)
    assert acceptance_report.artifact.substrate_checkpoint is not None
    assert evidence["substrate_checkpoint_present"] == 1.0
    assert evidence["substrate_update_count"] >= 1.0
    assert evidence["substrate_import_success_fraction"] == 1.0
    assert "pre_import_pass_fraction" in pre_import_evidence
    assert "mean_pre_import_score_delta" in pre_import_evidence


def test_ablation_report_quantifies_rare_heavy_gap_against_no_rare_heavy():
    report = asyncio.run(
        run_dialogue_pe_eta_ablation_benchmark(
            cases=DEFAULT_DIALOGUE_PROOF_CASES[:1],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_ablation_runner,
        )
    )

    assert report.rare_heavy_metric_deltas
    metric_delta_map = dict(report.rare_heavy_metric_deltas)
    assert metric_delta_map["rare_heavy_applied_count"] >= 0.0
    assert len(report.rare_heavy_case_deltas) == 1
    case_id, case_deltas = report.rare_heavy_case_deltas[0]
    assert case_id == "repair"
    assert "rare_heavy_applied_count" in dict(case_deltas)


def test_replay_ranking_quantifies_gap_vs_no_rare_heavy_when_profile_is_present():
    perturbation_report = asyncio.run(
        run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
            profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
            runner_factory=_synthetic_perturbation_runner,
        )
    )

    ranking = build_dialogue_replay_ranking_report(
        variant_cases=DEFAULT_DIALOGUE_CASE_VARIANTS[:2],
        ablation_report=perturbation_report.ablation_report,
    )

    assert ranking.entries
    assert ranking.mean_gap_vs_no_rare_heavy >= 0.0
    assert all(isinstance(entry.no_rare_heavy_score, float) for entry in ranking.entries)


def test_dialogue_case_report_flags_missing_pe_and_temporal_change():
    case = ScriptedDialogueCase(
        case_id="no-proof",
        description="A degenerate case with no PE or temporal variation.",
        user_inputs=("a", "b", "c"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.0,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.0,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.0,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.passed is False
    assert "high_pe_detected" in report.reasons
    assert "pe_triggered_temporal_response" in report.reasons
    assert "temporal_trajectory_nonconstant" in report.reasons
    assert "delayed_improvement_observed" in report.reasons


def test_dialogue_case_report_passes_when_pe_drives_temporal_and_delayed_change():
    case = ScriptedDialogueCase(
        case_id="proof-positive",
        description="Synthetic positive proof case.",
        user_inputs=("a", "b", "c", "d"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.85,
            reward=-0.50,
            action="full-cycle-pe",
            regime="repair_and_deescalation",
            abstract_action="repair",
            switch_gate=0.72,
            delayed_metric=0.20,
            pe_triggered=True,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.62,
            reward=-0.30,
            action="ssl-only-pe",
            regime="repair_and_deescalation",
            abstract_action="stabilize",
            switch_gate=0.61,
            delayed_metric=0.36,
            bounded_writeback_applied=True,
            reflection_promotion_eligible=True,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.35,
            reward=-0.10,
            action="full-cycle",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.42,
            delayed_metric=0.55,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.18,
            reward=0.12,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.28,
            delayed_metric=0.70,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.passed is True
    checks = dict(report.acceptance_checks)
    assert checks["prediction_chain_present"] is True
    assert checks["pe_triggered_temporal_response"] is True
    assert checks["temporal_trajectory_nonconstant"] is True
    assert checks["delayed_improvement_observed"] is True
    assert report.recovery_lag_turns == 0
    assert report.pressure_localization_score == 1.0
    assert report.over_response_cost == 0.0
    assert report.pressure_response_precision == 1.0
    assert report.pressure_response_recall == 1.0
    assert report.stability_after_recovery_score == 0.0
    assert report.bounded_writeback_turn_count == 1
    assert report.reflection_promotion_eligible_turn_count == 1


def test_dialogue_case_report_counts_cross_turn_pe_trigger():
    case = ScriptedDialogueCase(
        case_id="cross-turn-trigger",
        description="High PE on one turn should justify a PE schedule on the next turn.",
        user_inputs=("a", "b", "c", "d"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.24,
            reward=-0.06,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="diagnose",
            switch_gate=0.58,
            delayed_metric=0.25,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.12,
            reward=-0.02,
            action="ssl-only-pe",
            regime="guided_exploration",
            abstract_action="reframe",
            switch_gate=0.81,
            delayed_metric=0.38,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.10,
            reward=-0.01,
            action="full-cycle",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.43,
            delayed_metric=0.52,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.08,
            reward=0.01,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.29,
            delayed_metric=0.63,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.passed is True
    assert report.pe_triggered_turn_count >= 1
    checks = dict(report.acceptance_checks)
    assert checks["pe_triggered_temporal_response"] is True
    assert report.recovery_lag_turns == 1
    assert report.pressure_localization_score == 1.0
    assert report.over_response_cost == 0.0
    assert report.pressure_response_precision == 1.0
    assert report.pressure_response_recall == 1.0


def test_dialogue_case_report_counts_cross_turn_full_cycle_after_high_pe():
    case = ScriptedDialogueCase(
        case_id="cross-turn-full-cycle",
        description="High PE on one turn should count when the next turn runs a non-evidence cycle.",
        user_inputs=("a", "b", "c"),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.31,
            reward=-0.08,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="diagnose",
            switch_gate=0.66,
            delayed_metric=0.28,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.11,
            reward=-0.02,
            action="full-cycle",
            regime="problem_solving",
            abstract_action="reframe",
            switch_gate=0.74,
            delayed_metric=0.44,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.07,
            reward=0.01,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.32,
            delayed_metric=0.61,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.pe_triggered_turn_count >= 1
    assert dict(report.acceptance_checks)["pe_triggered_temporal_response"] is True
    assert report.recovery_lag_turns == 1
    assert report.pressure_localization_score == 1.0
    assert report.over_response_cost == 0.0
    assert report.pressure_response_precision == 1.0
    assert report.pressure_response_recall == 1.0


def test_dialogue_case_report_quantifies_recovery_lag_and_pressure_localization():
    case = ScriptedDialogueCase(
        case_id="quantitative-proof",
        description="Synthetic case for quantitative response metrics.",
        user_inputs=("a", "b", "c", "d", "e"),
        expected_pressure_turns=(2, 3),
    )
    turns = (
        _benchmark_turn(
            turn_index=1,
            pe=0.02,
            reward=0.0,
            action="evidence-only",
            regime="problem_solving",
            abstract_action="steady",
            switch_gate=0.10,
            delayed_metric=0.20,
        ),
        _benchmark_turn(
            turn_index=2,
            pe=0.26,
            reward=-0.06,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="diagnose",
            switch_gate=0.52,
            delayed_metric=0.24,
        ),
        _benchmark_turn(
            turn_index=3,
            pe=0.22,
            reward=-0.05,
            action="full-cycle",
            regime="guided_exploration",
            abstract_action="reframe",
            switch_gate=0.67,
            delayed_metric=0.40,
        ),
        _benchmark_turn(
            turn_index=4,
            pe=0.21,
            reward=-0.02,
            action="ssl-only",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.33,
            delayed_metric=0.57,
        ),
        _benchmark_turn(
            turn_index=5,
            pe=0.09,
            reward=0.02,
            action="ssl-only-pe",
            regime="problem_solving",
            abstract_action="plan",
            switch_gate=0.78,
            delayed_metric=0.64,
        ),
    )

    report = build_dialogue_case_report(case=case, turns=turns)

    assert report.recovery_lag_turns == 1
    assert report.pressure_localization_score == 2 / 3
    assert report.over_response_cost == 0.2
    assert report.pressure_response_precision == 2 / 3
    assert report.pressure_response_recall == 1.0
    assert report.stability_after_recovery_score == 0.0


def test_goal_drift_case_no_longer_fails_pe_trigger_check_with_synthetic_runner():
    goal_drift_case = next(case for case in DEFAULT_DIALOGUE_PROOF_CASES if case.case_id == "goal_drift")

    report = asyncio.run(
        run_dialogue_pe_eta_case(
            case=goal_drift_case,
            runner=_synthetic_runner(goal_drift_case),
        )
    )

    checks = dict(report.acceptance_checks)
    assert checks["high_pe_detected"] is True
    assert checks["pe_triggered_temporal_response"] is True
