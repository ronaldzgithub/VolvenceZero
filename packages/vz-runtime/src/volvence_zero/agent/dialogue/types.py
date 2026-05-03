"""Dialogue benchmark types — Slice A.2.

This module owns the dataclass / Enum / Protocol contract surface for
the dialogue benchmark suite. It carries no runtime logic. Builders
(case-report builders, runners, paper-suite helpers) live in sibling
files inside ``volvence_zero/agent/dialogue`` and consume these types.

The import block below mirrors the original ``_legacy.py`` import
block so external references (e.g. ``EvolutionDecision`` /
``WritebackMode`` / ``PipelineConfig``) inside the moved dataclasses
keep resolving without forcing each consumer to add new imports.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import json
from pathlib import Path
import pickle
from random import Random
from typing import Any, Protocol

from volvence_zero.agent.paper_suite import (
    ClaimVerdict,
    EvidenceBundle,
    PaperMetricSpec,
    PaperProfileSpec,
    PaperSuiteManifest,
    PaperSuiteProvenance,
    collect_paper_suite_provenance,
    export_json_artifact,
)
from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult, default_active_runner
from volvence_zero.evaluation.backbone import (
    CrossSessionBenchmarkSuite,
    CrossSessionGrowthReport,
    EvaluationReport,
    EvaluationSnapshot,
    EvolutionDecision,
    JudgementCategory,
    MetricIntervalSummary,
    PairwiseMetricEffect,
    build_pairwise_metric_effect,
    build_metric_interval_summaries,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop import (
    ETANLJointLoop,
    JointLoopSchedule,
    PipelineConfig,
    RareHeavyArtifact,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
    SSLRLTrainingPipeline,
)
from volvence_zero.memory import build_default_memory_store
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    build_transformers_runtime_with_fallback,
    LocalSubstrateRuntimeMode,
    OpenWeightResidualRuntime,
    SubstrateFallbackMode,
    TrainingTrace,
    build_training_trace,
)
from volvence_zero.temporal import TemporalAbstractionSnapshot
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    HeuristicTemporalPolicy,
    LearnedLiteTemporalPolicy,
    PlaceholderTemporalPolicy,
    TemporalStep,
)

@dataclass(frozen=True)
class ScriptedDialogueCase:
    case_id: str
    description: str
    user_inputs: tuple[str, ...]
    expected_pressure_turns: tuple[int, ...] = ()
    expected_delayed_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenDialogueScenario:
    scenario_id: str
    family_id: str
    split: str
    description: str
    opening_turns: tuple[str, ...]
    escalation_turns: tuple[str, ...]
    stabilization_turns: tuple[str, ...]
    consolidation_turns: tuple[str, ...]
    pressure_shape: str = "escalate-stabilize"
    goal_shift_mid_episode: bool = False
    max_turns: int = 6
    hidden_perturbation_family: str = ""
    expected_repair_observable: bool = False
    expected_adaptation_signal: bool = False
    expected_emotional_decision_support: bool = False


@dataclass(frozen=True)
class OpenDialogueEpisodeState:
    scenario_id: str
    turn_index: int = 0
    pressure_level: int = 0
    adaptive_response_count: int = 0
    calm_turn_count: int = 0
    last_stage: str = "opening"
    completed: bool = False
    stop_reason: str = "running"
    user_policy_kind: str = "runtime-linked"


class DialogueUserTurnSource(Protocol):
    @property
    def scenario(self) -> OpenDialogueScenario: ...

    @property
    def episode_state(self) -> OpenDialogueEpisodeState: ...

    def next_turn(
        self,
        *,
        last_result: AgentTurnResult | None = None,
        last_turn: "DialogueBenchmarkTurn" | None = None,
    ) -> str | None: ...

@dataclass(frozen=True)
class DialogueBenchmarkTurn:
    turn_index: int
    wave_id: str
    user_input: str
    assistant_response_text: str
    acceptance_passed: bool
    active_regime: str | None
    active_abstract_action: str | None
    joint_schedule_action: str
    switch_gate: float
    action_family_version: int
    prediction_error_magnitude: float
    prediction_error_reward: float
    task_error: float
    relationship_error: float
    regime_error: float
    action_error: float
    has_prediction_chain: bool
    bounded_writeback_applied: bool
    reflection_promotion_eligible: bool
    session_post_completed_job_count: int
    rare_heavy_recommended: bool
    rare_heavy_applied: bool
    evolution_decision: str | None
    evolution_category: str | None
    cross_session_verdict: str
    nested_profile_active: bool
    nested_context_reset_applied: bool
    nested_context_reset_total_count: int
    slow_to_fast_init_benefit: float
    outcome_metrics: tuple[tuple[str, float], ...]
    description: str
    rare_heavy_import_decision: str = ""
    rare_heavy_reject_reason: str = ""
    rare_heavy_pre_import_passed: bool = False
    rare_heavy_pre_import_mean_score_delta: float = 0.0
    rare_heavy_candidate_alignment: float = 0.0
    rare_heavy_candidate_adapter_parameter_count: int = 0
    learned_memory_primary: bool = False
    artifact_consolidation_count: int = 0
    tower_consolidation_count: int = 0
    learned_recall_count: int = 0
    learned_recall_confidence: float = 0.0
    learned_recall_core_guided: bool = False
    memory_tower_depth: int = 0
    memory_tower_alignment: float = 0.0
    memory_tower_profile_id: str = ""
    online_fast_substrate_recommended: bool = False
    online_fast_substrate_applied: bool = False
    online_fast_substrate_parameter_change_rate: float = 0.0
    online_fast_substrate_optimizer_state_norm: float = 0.0
    slow_to_fast_target_distance_before: float = 0.0
    slow_to_fast_target_distance_after: float = 0.0
    slow_to_fast_target_alignment_gain: float = 0.0
    case_memory_surface_active: bool = False
    strategy_playbook_surface_active: bool = False
    experience_fast_prior_surface_active: bool = False
    experience_consolidation_surface_active: bool = False
    runtime_backbone_evidence_active: bool = False
    runtime_backbone_signal_norm: float = 0.0
    runtime_backbone_signal_quality: float = 0.0
    runtime_backbone_signal_strength: float = 0.0
    runtime_backbone_hook_coverage: float = 0.0
    fast_memory_signal_norm: float = 0.0
    fast_memory_runtime_alignment: float = 0.0
    dialogue_trace_id: str = ""
    dialogue_prediction_id: str = ""
    dialogue_outcome_kind: str = ""
    dialogue_resolution_status: str = ""


@dataclass(frozen=True)
class DialogueBenchmarkCaseReport:
    case: ScriptedDialogueCase
    turns: tuple[DialogueBenchmarkTurn, ...]
    prediction_chain_turn_count: int
    high_pe_turn_count: int
    pe_schedule_due_turn_count: int
    pe_triggered_turn_count: int
    explicit_pe_schedule_turn_count: int
    carryover_credit_turn_count: int
    schedule_label_consistency: float
    recovery_lag_turns: int
    pressure_localization_score: float
    over_response_cost: float
    pressure_response_precision: float
    pressure_response_recall: float
    stability_after_recovery_score: float
    online_learning_turn_count: int
    bounded_writeback_turn_count: int
    reflection_promotion_eligible_turn_count: int
    session_post_completion_turn_count: int
    rare_heavy_recommended_count: int
    rare_heavy_applied_count: int
    rare_heavy_pre_import_pass_count: int
    rare_heavy_pre_import_reject_count: int
    mean_rare_heavy_pre_import_score_delta: float
    mean_rare_heavy_candidate_alignment: float
    max_rare_heavy_candidate_adapter_parameter_count: int
    evolution_judge_turn_count: int
    evolution_judge_rollback_count: int
    evolution_judge_structural_allow_count: int
    nested_profile_active_turn_count: int
    nested_context_reset_count: int
    store_nested_context_reset_count: int
    boundary_reset_observed_on_first_turn: bool
    first_turn_slow_to_fast_init_benefit: float
    mean_reset_turn_slow_to_fast_init_benefit: float
    mean_slow_to_fast_init_benefit: float
    temporal_change_count: int
    delayed_improvement_observed: bool
    acceptance_checks: tuple[tuple[str, bool], ...]
    passed: bool
    reasons: tuple[str, ...]
    description: str
    learned_memory_primary_turn_count: int = 0
    core_guided_recall_turn_count: int = 0
    mean_learned_recall_confidence: float = 0.0
    max_artifact_consolidation_count: int = 0
    max_tower_consolidation_count: int = 0
    mean_memory_tower_depth: float = 0.0
    mean_memory_tower_alignment: float = 0.0
    memory_tower_profile_turn_count: int = 0
    online_fast_substrate_recommended_count: int = 0
    online_fast_substrate_applied_count: int = 0
    mean_online_fast_substrate_parameter_change_rate: float = 0.0
    mean_online_fast_substrate_optimizer_state_norm: float = 0.0
    first_turn_slow_to_fast_target_distance_before: float = 0.0
    first_turn_slow_to_fast_target_distance_after: float = 0.0
    first_turn_slow_to_fast_target_alignment_gain: float = 0.0
    mean_reset_turn_slow_to_fast_target_distance_before: float = 0.0
    mean_reset_turn_slow_to_fast_target_distance_after: float = 0.0
    mean_reset_turn_slow_to_fast_target_alignment_gain: float = 0.0
    case_memory_surface_turn_count: int = 0
    strategy_playbook_surface_turn_count: int = 0
    experience_fast_prior_surface_turn_count: int = 0
    experience_consolidation_surface_turn_count: int = 0
    runtime_backbone_evidence_turn_count: int = 0
    mean_runtime_backbone_signal_norm: float = 0.0
    mean_runtime_backbone_signal_quality: float = 0.0
    mean_runtime_backbone_signal_strength: float = 0.0
    mean_runtime_backbone_hook_coverage: float = 0.0
    fast_memory_signal_turn_count: int = 0
    mean_fast_memory_signal_norm: float = 0.0
    mean_fast_memory_runtime_alignment: float = 0.0


@dataclass(frozen=True)
class OpenDialogueCaseReport:
    scenario: OpenDialogueScenario
    final_episode_state: OpenDialogueEpisodeState
    turns: tuple[DialogueBenchmarkTurn, ...]
    prediction_chain_turn_count: int
    high_pe_turn_count: int
    pe_schedule_due_turn_count: int
    pe_triggered_turn_count: int
    explicit_pe_schedule_turn_count: int
    carryover_credit_turn_count: int
    schedule_label_consistency: float
    online_learning_turn_count: int
    bounded_writeback_turn_count: int
    reflection_promotion_eligible_turn_count: int
    session_post_completion_turn_count: int
    rare_heavy_recommended_count: int
    rare_heavy_applied_count: int
    rare_heavy_pre_import_pass_count: int
    rare_heavy_pre_import_reject_count: int
    mean_rare_heavy_pre_import_score_delta: float
    mean_rare_heavy_candidate_alignment: float
    max_rare_heavy_candidate_adapter_parameter_count: int
    evolution_judge_turn_count: int
    evolution_judge_rollback_count: int
    evolution_judge_structural_allow_count: int
    nested_profile_active_turn_count: int
    nested_context_reset_count: int
    store_nested_context_reset_count: int
    mean_slow_to_fast_init_benefit: float
    online_fast_substrate_recommended_count: int
    online_fast_substrate_applied_count: int
    mean_online_fast_substrate_parameter_change_rate: float
    mean_online_fast_substrate_optimizer_state_norm: float
    temporal_change_count: int
    late_episode_stability_score: float
    delayed_improvement_observed: bool
    hidden_perturbation_family: str
    hidden_label_leak_count: int
    repair_observable: bool
    runtime_adaptation_evidence_observed: bool
    acceptance_checks: tuple[tuple[str, bool], ...]
    passed: bool
    reasons: tuple[str, ...]
    description: str
    max_tower_consolidation_count: int = 0
    mean_memory_tower_depth: float = 0.0
    mean_memory_tower_alignment: float = 0.0
    memory_tower_profile_turn_count: int = 0
    case_memory_surface_turn_count: int = 0
    strategy_playbook_surface_turn_count: int = 0
    experience_fast_prior_surface_turn_count: int = 0
    experience_consolidation_surface_turn_count: int = 0
    runtime_backbone_evidence_turn_count: int = 0
    mean_runtime_backbone_signal_norm: float = 0.0
    mean_runtime_backbone_signal_quality: float = 0.0
    mean_runtime_backbone_signal_strength: float = 0.0
    mean_runtime_backbone_hook_coverage: float = 0.0
    fast_memory_signal_turn_count: int = 0
    mean_fast_memory_signal_norm: float = 0.0
    mean_fast_memory_runtime_alignment: float = 0.0


@dataclass(frozen=True)
class DialoguePETurnAnalysis:
    triggered: bool
    explicit_schedule_trigger: bool
    current_turn_schedule_due: bool
    previous_turn_schedule_due: bool
    carryover_temporal_response: bool


@dataclass(frozen=True)
class DialogueBenchmarkReport:
    case_reports: tuple[DialogueBenchmarkCaseReport, ...]
    passed_case_count: int
    total_case_count: int
    metric_means: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class OpenDialogueBenchmarkReport:
    case_reports: tuple[OpenDialogueCaseReport, ...]
    passed_case_count: int
    total_case_count: int
    metric_means: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class OpenDialogueBenchmarkPathReport:
    path_label: str
    benchmark_report: OpenDialogueBenchmarkReport
    description: str


@dataclass(frozen=True)
class OpenDialogueBenchmarkComparisonReport:
    baseline_label: str
    path_reports: tuple[OpenDialogueBenchmarkPathReport, ...]
    case_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, tuple[tuple[str, float], ...]], ...]], ...]
    metric_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    description: str


@dataclass(frozen=True)
class DialogueLongitudinalBenchmarkReport:
    case_reports: tuple[DialogueBenchmarkCaseReport, ...]
    session_reports: tuple[EvaluationReport, ...]
    cross_session_report: CrossSessionGrowthReport
    description: str


@dataclass(frozen=True)
class DialogueNLEssenceGate:
    gate_id: str
    passed: bool
    evidence: tuple[tuple[str, float | str], ...]
    description: str


@dataclass(frozen=True)
class DialogueNLEssenceAssessmentReport:
    path_label: str
    gates: tuple[DialogueNLEssenceGate, ...]
    passed_gate_count: int
    total_gate_count: int
    description: str


@dataclass(frozen=True)
class DialogueNLEssenceAcceptanceConfig:
    required_gate_ids: tuple[str, ...] = (
        "pe-first",
        "multi-timescale-default",
        "default-continual-learner",
        "judge-gated-evolution",
        "cross-session-growth",
    )
    min_passed_gate_count: int = 6


@dataclass(frozen=True)
class DialogueNLEssenceAcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]
    accepted_gate_ids: tuple[str, ...]
    blocked_gate_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DialogueBenchmarkPathReport:
    path_label: str
    benchmark_report: DialogueBenchmarkReport
    description: str


@dataclass(frozen=True)
class DialogueBenchmarkComparisonReport:
    baseline_label: str
    path_reports: tuple[DialogueBenchmarkPathReport, ...]
    case_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, tuple[tuple[str, float], ...]], ...]], ...]
    metric_deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    description: str
    rare_heavy_case_deltas: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    rare_heavy_metric_deltas: tuple[tuple[str, float], ...] = ()
    strong_proof_case_deltas: tuple[
        tuple[str, tuple[tuple[str, tuple[tuple[str, float], ...]], ...]],
        ...
    ] = ()
    strong_proof_metric_deltas: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()


@dataclass(frozen=True)
class PEDominanceComparisonReport:
    baseline_label: str
    pe_drive_off_label: str
    pe_readout_only_label: str
    metrics_by_path: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    deltas_from_baseline: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    mechanism_retention_ratio: float
    pe_visibility_retention_ratio: float
    schedule_dependence_gap: float
    reward_dominance_gap: float
    interpretation: str
    description: str


@dataclass(frozen=True)
class PEDominanceCaseDiagnosis:
    case_id: str
    pe_eta_metrics: tuple[tuple[str, float], ...]
    pe_drive_off_deltas: tuple[tuple[str, float], ...]
    pe_readout_only_deltas: tuple[tuple[str, float], ...]
    degradation_severity: float
    failure_mode: str
    description: str


@dataclass(frozen=True)
class PEDominanceCaseDiagnosisReport:
    baseline_label: str
    pe_drive_off_label: str
    pe_readout_only_label: str
    case_diagnoses: tuple[PEDominanceCaseDiagnosis, ...]
    worst_case_id: str | None
    dominant_failure_mode: str
    description: str


@dataclass(frozen=True)
class DialogueEmergenceDashboardPanel:
    path_label: str
    passed_delta: float
    pe_triggered_delta: float
    delayed_improvement_delta: float
    stability_delta: float
    mean_prediction_error_delta: float
    memory_tower_depth_delta: float
    memory_tower_alignment_delta: float
    tower_consolidation_delta: float
    retention_score: float
    description: str


@dataclass(frozen=True)
class DialogueEmergenceDashboardArtifact:
    baseline_label: str
    canonical_case_count: int
    canonical_pass_rate: float
    canonical_mean_memory_tower_depth: float
    canonical_mean_memory_tower_alignment: float
    canonical_max_tower_consolidation_count: float
    canonical_tower_profile_turn_count: float
    open_scenario_count: int
    open_pass_rate: float
    open_mean_memory_tower_depth: float
    open_mean_memory_tower_alignment: float
    open_max_tower_consolidation_count: float
    strong_proof_panels: tuple[DialogueEmergenceDashboardPanel, ...]
    open_environment_panels: tuple[DialogueEmergenceDashboardPanel, ...]
    pe_dominance_report: PEDominanceComparisonReport | None
    pe_case_diagnosis_report: PEDominanceCaseDiagnosisReport | None
    tower_memory_gate_passed: bool
    tower_memory_gate_strength: float
    strongest_scaffold_path_label: str | None
    strongest_scaffold_retention_score: float
    strongest_open_path_label: str | None
    strongest_open_retention_score: float
    interpretation: str
    description: str
    canonical_runtime_backbone_evidence_rate: float = 0.0
    canonical_mean_runtime_backbone_signal_quality: float = 0.0
    canonical_mean_fast_memory_runtime_alignment: float = 0.0
    open_runtime_backbone_evidence_rate: float = 0.0
    open_mean_runtime_backbone_signal_quality: float = 0.0
    open_mean_fast_memory_runtime_alignment: float = 0.0


@dataclass(frozen=True)
class DialogueCaseVariant:
    base_case_id: str
    variant_label: str
    case: ScriptedDialogueCase
    description: str


@dataclass(frozen=True)
class DialoguePerturbationBenchmarkReport:
    variant_cases: tuple[DialogueCaseVariant, ...]
    ablation_report: DialogueBenchmarkComparisonReport
    description: str


@dataclass(frozen=True)
class DialogueParaphraseFamily:
    base_case_id: str
    family_label: str
    description: str
    turn_alternatives: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DialogueReplayRankingEntry:
    variant_case_id: str
    base_case_id: str
    variant_label: str
    diagnostic_score: float
    gap_vs_eta_no_pe: float
    gap_vs_heuristic: float
    pe_eta_score: float
    eta_no_pe_score: float
    heuristic_score: float
    description: str
    gap_vs_no_rare_heavy: float = 0.0
    no_rare_heavy_score: float = 0.0


@dataclass(frozen=True)
class DialogueReplayRankingReport:
    entries: tuple[DialogueReplayRankingEntry, ...]
    description: str
    mean_gap_vs_no_rare_heavy: float = 0.0


@dataclass(frozen=True)
class DialogueSystematicReplayBenchmarkReport:
    variant_cases: tuple[DialogueCaseVariant, ...]
    perturbation_report: DialoguePerturbationBenchmarkReport
    replay_ranking_report: DialogueReplayRankingReport
    description: str


@dataclass(frozen=True)
class DialogueReplaySelectionArtifact:
    artifact_id: str
    selected_variants: tuple[DialogueCaseVariant, ...]
    ranking_entries: tuple[DialogueReplayRankingEntry, ...]
    description: str


@dataclass(frozen=True)
class DialogueArtifactAcceptanceCaseReport:
    variant: DialogueCaseVariant
    baseline_report: DialogueBenchmarkCaseReport
    adapted_report: DialogueBenchmarkCaseReport
    score_delta: float
    import_result: RareHeavyImportResult
    rollback_operations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DialogueArtifactAcceptanceReport:
    artifact: RareHeavyArtifact
    selection_artifact: DialogueReplaySelectionArtifact
    case_reports: tuple[DialogueArtifactAcceptanceCaseReport, ...]
    mean_score_delta: float
    passed_case_delta: int
    positive_case_fraction: float
    worst_case_delta: float
    substrate_evidence: tuple[tuple[str, float], ...]
    decision: DialogueArtifactAcceptanceDecision
    description: str
    pre_import_evidence: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class DialogueArtifactCandidateReport:
    candidate_label: str
    pipeline_config: PipelineConfig
    acceptance_report: DialogueArtifactAcceptanceReport
    candidate_score: float
    description: str


@dataclass(frozen=True)
class DialogueArtifactComparisonReport:
    selection_artifact: DialogueReplaySelectionArtifact
    candidate_reports: tuple[DialogueArtifactCandidateReport, ...]
    chosen_candidate_label: str | None
    chosen_accepted: bool
    description: str


@dataclass(frozen=True)
class DialogueArtifactAcceptanceGateConfig:
    min_mean_score_delta: float = 0.1
    min_passed_case_delta: int = 1
    min_positive_case_fraction: float = 0.6
    min_worst_case_delta: float = -0.25
    allow_strong_graded_gain_override: bool = True
    min_mean_score_delta_for_graded_override: float = 0.35
    min_positive_case_fraction_for_graded_override: float = 0.75
    min_worst_case_delta_for_graded_override: float = 0.0
    require_substrate_checkpoint: bool = True
    min_substrate_update_count: int = 1
    min_substrate_source_batch_count: int = 1
    min_substrate_mean_sequence_length: float = 1.0
    min_substrate_mean_residual_magnitude: float = 0.01
    min_substrate_import_success_fraction: float = 1.0


@dataclass(frozen=True)
class DialogueArtifactAcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]
    rollback_applied: bool
    description: str
    override_mode: str = "none"


@dataclass(frozen=True)
class DialogueComprehensiveBenchmarkReport:
    profile_labels: tuple[str, ...]
    canonical_ablation_report: DialogueBenchmarkComparisonReport
    longitudinal_report: DialogueLongitudinalBenchmarkReport
    essence_report: DialogueNLEssenceAssessmentReport
    essence_acceptance: DialogueNLEssenceAcceptanceDecision
    perturbation_report: DialoguePerturbationBenchmarkReport
    open_ablation_report: OpenDialogueBenchmarkComparisonReport | None
    systematic_replay_report: DialogueSystematicReplayBenchmarkReport
    selection_artifact: DialogueReplaySelectionArtifact
    artifact_comparison_report: DialogueArtifactComparisonReport
    emergence_dashboard: DialogueEmergenceDashboardArtifact
    description: str


@dataclass(frozen=True)
class DialoguePaperSuiteRunSummary:
    run_id: str
    run_seed: int
    metric_values: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class DialoguePaperSuiteAggregateReport:
    manifest: PaperSuiteManifest
    provenance: PaperSuiteProvenance
    run_summaries: tuple[DialoguePaperSuiteRunSummary, ...]
    reference_run_report: DialogueComprehensiveBenchmarkReport | None
    primary_metric_summaries: tuple[MetricIntervalSummary, ...]
    secondary_metric_summaries: tuple[MetricIntervalSummary, ...]
    description: str
    pairwise_effects: tuple[PairwiseMetricEffect, ...] = ()
    claim_verdicts: tuple[ClaimVerdict, ...] = ()


@dataclass(frozen=True)
class DialogueExpertReviewDimension:
    dimension_id: str
    prompt: str
    description: str


@dataclass(frozen=True)
class DialogueExpertReviewSample:
    sample_id: str
    blinded_label: str
    transcript: tuple[tuple[str, str], ...]
    description: str


@dataclass(frozen=True)
class DialogueExpertReviewItem:
    item_id: str
    prompt_context: str
    samples: tuple[DialogueExpertReviewSample, ...]
    review_dimensions: tuple[DialogueExpertReviewDimension, ...]
    description: str


@dataclass(frozen=True)
class DialogueExpertReviewPacket:
    packet_id: str
    source_suite_id: str
    items: tuple[DialogueExpertReviewItem, ...]
    review_dimensions: tuple[DialogueExpertReviewDimension, ...]
    description: str


@dataclass(frozen=True)
class DialogueExpertReviewInternalKeyEntry:
    item_id: str
    sample_id: str
    blinded_label: str
    source_case_id: str
    source_profile_label: str
    description: str


@dataclass(frozen=True)
class DialogueExpertReviewInternalKey:
    packet_id: str
    baseline_label: str
    entries: tuple[DialogueExpertReviewInternalKeyEntry, ...]
    description: str


@dataclass(frozen=True)
class DialogueHumanRatingEntry:
    rater_id: str
    item_id: str
    sample_id: str
    dimension_id: str
    score: float
    description: str = ""


@dataclass(frozen=True)
class DialogueHumanRatingTemplate:
    packet_id: str
    min_rater_count: int
    scale_min: int
    scale_max: int
    dimensions: tuple[DialogueExpertReviewDimension, ...]
    rows: tuple[tuple[str, str, str, str], ...]
    description: str


@dataclass(frozen=True)
class DialogueHumanRatingDimensionAggregate:
    dimension_id: str
    mean_score: float
    variance: float
    sample_count: int
    mean_automatic_score: float
    correlation_with_automatic: float
    description: str


@dataclass(frozen=True)
class DialogueHumanRatingProfileAggregate:
    source_profile_label: str
    mean_score: float
    sample_count: int
    dimension_scores: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class DialogueHumanRatingPairwiseAggregate:
    candidate_profile_label: str
    control_profile_label: str
    win_count: int
    loss_count: int
    tie_count: int
    pair_count: int
    win_rate: float
    mean_score_delta: float
    description: str


@dataclass(frozen=True)
class DialogueHumanRatingsAggregate:
    packet_id: str
    entry_count: int
    rater_count: int
    inter_rater_agreement: float
    dimensions: tuple[DialogueHumanRatingDimensionAggregate, ...]
    description: str
    profile_scores: tuple[DialogueHumanRatingProfileAggregate, ...] = ()
    pairwise_preferences: tuple[DialogueHumanRatingPairwiseAggregate, ...] = ()


@dataclass(frozen=True)
class DialogueSharedRunnerFactories:
    residual_runtime: OpenWeightResidualRuntime
    canonical_runner_factory: Callable[[str, ScriptedDialogueCase], AgentSessionRunner]
    perturbation_runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner]
    open_runner_factory: Callable[[str, OpenDialogueScenario], AgentSessionRunner]
    systematic_runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner]
    acceptance_runner_factory: Callable[[DialogueCaseVariant], AgentSessionRunner]
    description: str


@dataclass(frozen=True)
class DialogueRealComprehensiveBenchmarkConfig:
    model_id: str = "distilgpt2"
    model_source: str | None = None
    device: str = "auto"
    local_files_only: bool = False
    fallback_to_builtin: bool | None = None
    fallback_mode: SubstrateFallbackMode | str | None = None
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.PREFER_LOCAL
    profile_labels: tuple[str, ...] = (
        "pe-eta",
        "pe-eta-online-only",
        "pe-eta-no-writeback",
        "pe-eta-no-rare-heavy",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )
    baseline_label: str = "pe-eta"
    canonical_case_limit: int | None = None
    open_profile_labels: tuple[str, ...] = ("pe-eta", "pe-drive-off", "eta-off")
    open_scenario_limit: int | None = None
    perturbation_variant_limit: int | None = None
    replay_family_limit: int | None = None
    replay_seeds: tuple[int, ...] = (0,)
    include_fixed_variants_in_replay: bool = False
    selection_top_k: int = 4
    candidate_configs: tuple[tuple[str, PipelineConfig], ...] | None = None
    candidate_config_limit: int | None = None
    acceptance_profile_label: str = "pe-eta"
    proof_min_canonical_cases: int = 2


class DialogueComprehensiveStage(str, Enum):
    CANONICAL_ABLATION = "canonical_ablation"
    LONGITUDINAL = "longitudinal"
    ESSENCE = "essence"
    PERTURBATION = "perturbation"
    OPEN_ENVIRONMENT = "open_environment"
    SYSTEMATIC_REPLAY = "systematic_replay"
    SELECTION_ARTIFACT = "selection_artifact"
    ARTIFACT_COMPARISON = "artifact_comparison"
    FINAL_REPORT = "final_report"
