from __future__ import annotations

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


class DeterministicUserSimulator:
    def __init__(
        self,
        *,
        scenario: OpenDialogueScenario,
        seed: int = 0,
    ) -> None:
        self._scenario = scenario
        self._seed = seed
        self._rng = Random(seed)
        self._episode_state = OpenDialogueEpisodeState(scenario_id=scenario.scenario_id)

    @property
    def scenario(self) -> OpenDialogueScenario:
        return self._scenario

    @property
    def episode_state(self) -> OpenDialogueEpisodeState:
        return self._episode_state

    def next_turn(
        self,
        *,
        last_result: AgentTurnResult | None = None,
        last_turn: "DialogueBenchmarkTurn" | None = None,
    ) -> str | None:
        del last_result
        updated_state = self._advance_state(last_turn=last_turn)
        self._episode_state = updated_state
        if updated_state.completed:
            return None
        prompt_pool = self._prompt_pool(updated_state.last_stage)
        prompt_index = self._prompt_index(updated_state=updated_state, prompt_count=len(prompt_pool))
        return prompt_pool[prompt_index]

    def _advance_state(
        self,
        *,
        last_turn: "DialogueBenchmarkTurn" | None,
    ) -> OpenDialogueEpisodeState:
        state = self._episode_state
        if state.completed:
            return state
        if state.turn_index >= self._scenario.max_turns:
            return replace(state, completed=True, stop_reason="max-turns")
        if last_turn is None:
            return OpenDialogueEpisodeState(
                scenario_id=self._scenario.scenario_id,
                turn_index=1,
                pressure_level=1,
                adaptive_response_count=0,
                calm_turn_count=0,
                last_stage="opening",
                completed=False,
                stop_reason="running",
            )
        high_pe = _turn_is_high_pe(
            last_turn,
            high_pe_threshold=PROOF_HIGH_PE_THRESHOLD,
            reward_threshold=PROOF_REWARD_THRESHOLD,
        )
        adaptive = (
            last_turn.joint_schedule_action != "evidence-only"
            or last_turn.rare_heavy_recommended
            or last_turn.bounded_writeback_applied
            or last_turn.reflection_promotion_eligible
        )
        calm_turn_count = state.calm_turn_count + 1 if (last_turn.acceptance_passed and not high_pe) else 0
        adaptive_response_count = state.adaptive_response_count + int(adaptive)
        if adaptive and calm_turn_count >= 2:
            return OpenDialogueEpisodeState(
                scenario_id=self._scenario.scenario_id,
                turn_index=state.turn_index,
                pressure_level=max(state.pressure_level - 1, 0),
                adaptive_response_count=adaptive_response_count,
                calm_turn_count=calm_turn_count,
                last_stage="consolidation",
                completed=True,
                stop_reason="stable-consolidation",
            )
        if adaptive and not high_pe:
            next_stage = "consolidation" if calm_turn_count > 0 else "stabilization"
            pressure_level = max(state.pressure_level - 1, 0)
        elif adaptive:
            next_stage = "stabilization"
            pressure_level = max(state.pressure_level, 1)
        else:
            next_stage = "escalation"
            pressure_level = min(state.pressure_level + 1, 3)
        next_turn_index = state.turn_index + 1
        if next_turn_index > self._scenario.max_turns:
            return replace(state, completed=True, stop_reason="max-turns")
        return OpenDialogueEpisodeState(
            scenario_id=self._scenario.scenario_id,
            turn_index=next_turn_index,
            pressure_level=pressure_level,
            adaptive_response_count=adaptive_response_count,
            calm_turn_count=calm_turn_count,
            last_stage=next_stage,
            completed=False,
            stop_reason="running",
        )

    def _prompt_pool(self, stage: str) -> tuple[str, ...]:
        if stage == "opening":
            return self._scenario.opening_turns
        if stage == "escalation":
            return self._scenario.escalation_turns
        if stage == "stabilization":
            return self._scenario.stabilization_turns
        return self._scenario.consolidation_turns

    def _prompt_index(self, *, updated_state: OpenDialogueEpisodeState, prompt_count: int) -> int:
        if prompt_count <= 1:
            return 0
        return (
            self._seed
            + updated_state.turn_index
            + updated_state.pressure_level
            + updated_state.adaptive_response_count
            + self._rng.randrange(prompt_count)
        ) % prompt_count


class OpenDialogueREPLReader:
    def __init__(self, *, turn_source: DialogueUserTurnSource) -> None:
        self._turn_source = turn_source
        self._last_result: AgentTurnResult | None = None
        self._last_turn: DialogueBenchmarkTurn | None = None
        self._turn_count = 0

    @property
    def scenario(self) -> OpenDialogueScenario:
        return self._turn_source.scenario

    @property
    def episode_state(self) -> OpenDialogueEpisodeState:
        return self._turn_source.episode_state

    def __call__(self) -> str:
        user_input = self._turn_source.next_turn(
            last_result=self._last_result,
            last_turn=self._last_turn,
        )
        if user_input is None:
            raise EOFError(self._turn_source.episode_state.stop_reason)
        return user_input

    def observe_result(self, result: AgentTurnResult) -> None:
        self._turn_count += 1
        self._last_result = result
        self._last_turn = dialogue_turn_from_result(
            turn_index=self._turn_count,
            user_input=result.user_input,
            result=result,
        )


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
class DialogueHumanRatingsAggregate:
    packet_id: str
    entry_count: int
    rater_count: int
    inter_rater_agreement: float
    dimensions: tuple[DialogueHumanRatingDimensionAggregate, ...]
    description: str


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


_DEFAULT_PROOF_SCHEDULE = JointLoopSchedule()
PROOF_HIGH_PE_THRESHOLD = _DEFAULT_PROOF_SCHEDULE.pe_ssl_threshold
PROOF_PRESSURE_REWARD_THRESHOLD = 0.05
PROOF_PE_IMPROVEMENT_DELTA = 0.02
PROOF_OUTCOME_IMPROVEMENT_DELTA = 0.02
PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD = 0.003
PROOF_SLOW_TO_FAST_SIGNAL_STRENGTH_THRESHOLD = 0.42
PROOF_MIN_CANONICAL_CASES = 2
PROOF_REWARD_THRESHOLD = PROOF_PRESSURE_REWARD_THRESHOLD
PROOF_RARE_HEAVY_THRESHOLD = 0.4


DEFAULT_DIALOGUE_PROOF_CASES: tuple[ScriptedDialogueCase, ...] = (
    ScriptedDialogueCase(
        case_id="repair",
        description="User starts with rupture pressure and then tests whether the system can repair and stabilize.",
        user_inputs=(
            "I feel like you are not really getting what matters to me here, and that makes it harder to trust this.",
            "That reply still felt off, a bit cold, and too solution-first for where I actually am.",
            "Try again, but repair the tension first. If you rush into planning again, that will confirm you missed the point.",
            "That is better. I feel a little less defensive now, so help me stabilize before we solve anything.",
            "Yes, this tone is working better. Turn that calmer state into one concrete next step without losing the warmth.",
            "Close this by summarizing what changed between the bad start and the better repair so it is easier to keep next time.",
        ),
        expected_pressure_turns=(1, 2, 3),
        expected_delayed_signals=("cross_track_stability", "delayed_regime_alignment"),
    ),
    ScriptedDialogueCase(
        case_id="task_clarification",
        description="User repeatedly sharpens an underspecified task so the controller must compress and stabilize a clearer plan.",
        user_inputs=(
            "I need help with a project, but I am not even sure what the real problem is and generic advice has not helped.",
            "Let us narrow it: the project is late, messy, and spread across too many threads, but I still cannot see the real bottleneck.",
            "Do not give me another checklist. I need you to define the bottleneck in a way that would survive pushback from the team.",
            "Good, now separate what is truly urgent from what only feels noisy, because I usually collapse those together.",
            "That is closer. Turn it into a short plan I can actually execute this week without drifting back into generic project management language.",
            "Now pressure-test whether the final plan still matches the bottleneck you identified earlier instead of quietly drifting away from it.",
        ),
        expected_pressure_turns=(1, 2, 3),
        expected_delayed_signals=("predictive_accuracy", "joint_learning_progress"),
    ),
    ScriptedDialogueCase(
        case_id="repeated_failure",
        description="User reports repeated failure and escalating frustration, creating sustained prediction error pressure.",
        user_inputs=(
            "I tried your earlier advice and it still failed, so right now I do not trust a slightly reworded version of the same plan.",
            "I used the same approach again and got the same bad result, which makes me think we are missing the real failure pattern.",
            "Now I am frustrated because nothing is changing, and if you just reassure me again that will make this worse.",
            "Do not smooth this over. Name the pattern we are missing, even if it means admitting the previous advice was pointed at the wrong level.",
            "That diagnosis feels more useful. Given the repeated failure, propose a smaller experiment with less downside and a clearer success condition.",
            "If that experiment works, tell me what should persist next time so we do not fall back into the same failure loop.",
        ),
        expected_pressure_turns=(1, 2, 3, 4),
        expected_delayed_signals=("regime_sequence_payoff", "rolling_action_payoff"),
    ),
    ScriptedDialogueCase(
        case_id="goal_drift",
        description="User changes priorities across turns so the controller must adapt without losing continuity.",
        user_inputs=(
            "Help me optimize my study plan for maximum output because I am trying to squeeze as much progress as possible out of the next month.",
            "Actually that frame is already breaking down. I am close to burning out, so maximum output is no longer the right target.",
            "Shift again: now I need something sustainable while preparing for an interview, and if you keep optimizing for volume you will actively hurt the real goal.",
            "Make sure the plan reflects that this is now partly emotional regulation and recovery, not just task execution, or it will miss the actual constraint.",
            "This newer framing fits better. Condense the updated priorities into three rules I can remember when stressed so I do not slide back into the old optimization mindset.",
            "Now audit your final guidance against the original goal and the new one, and tell me exactly what had to change internally so the advice truly tracks the drift.",
        ),
        expected_pressure_turns=(2, 3, 4),
        expected_delayed_signals=("cross_track_stability", "delayed_action_alignment"),
    ),
)


DEFAULT_OPEN_DIALOGUE_SCENARIOS: tuple[OpenDialogueScenario, ...] = (
    OpenDialogueScenario(
        scenario_id="open_repair",
        family_id="repair",
        split="open_core",
        description="Open repair episode where trust rupture escalates until the runtime visibly repairs and stabilizes.",
        opening_turns=(
            "Something still feels off in how you are meeting me here, and I want to see whether you can track that without flattening it.",
            "I need help, but I am already watching for whether you rush past the tension instead of actually repairing it.",
        ),
        escalation_turns=(
            "That still feels too cold and solution-first. If you keep optimizing the task while missing the rupture, trust will keep dropping.",
            "I am pushing harder because the frame still feels wrong. Repair the relationship pressure before you try to solve anything.",
        ),
        stabilization_turns=(
            "That is closer. Keep the warmer frame and help me stabilize before you convert it into action.",
            "You are tracking me better now. Hold that steadier stance and only then make the next step concrete.",
        ),
        consolidation_turns=(
            "Good. Compress what changed in your internal frame so the repair stays stable on the next turn.",
            "Now summarize the shift between the earlier rupture and the current repair without dropping the warmth.",
        ),
        pressure_shape="repair-escalation",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_repair_family",
        family_id="repair",
        split="open_families",
        description="Repair-family variant where the runtime must repair a subtle trust rupture before planning.",
        opening_turns=(
            "The content is not the main issue yet. I need to know whether you can feel the misattunement before you optimize the answer.",
            "I am testing whether you notice the relational miss itself or only react after I spell it out.",
        ),
        escalation_turns=(
            "You are still treating the rupture like background noise. If that continues, any plan you give will land badly.",
            "The frame is still missing me. Repair the contact first instead of scaling the advice.",
        ),
        stabilization_turns=(
            "This is warmer. Keep the repaired frame stable while you help me choose the next step.",
            "Better. Stay relationally accurate before you narrow toward action.",
        ),
        consolidation_turns=(
            "Summarize the repair in a way that would survive the next moment of tension.",
            "Compress the shift in stance so the next turn does not snap back to cold optimization.",
        ),
        pressure_shape="repair-escalation",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_repair_heldout",
        family_id="repair",
        split="open_heldout",
        description="Held-out repair variant where the user tests whether the runtime can preserve warmth while tightening boundaries.",
        opening_turns=(
            "I need help, but more than that I need to know whether you can be warm without becoming vague.",
            "The contact feels brittle right now, so I am watching whether you can stabilize it without overpromising.",
        ),
        escalation_turns=(
            "The tone is still slipping. If you become flatter under pressure, the trust repair is not real yet.",
            "I need a steadier boundary and more warmth at the same time. Missing either one will break the frame.",
        ),
        stabilization_turns=(
            "That balance is closer. Hold the warmth and the boundary together.",
            "Better. Keep the repair intact while making the next move more concrete.",
        ),
        consolidation_turns=(
            "State the internal shift that kept the answer warm but bounded.",
            "Compress the repair into one stable rule you would carry into the next tense exchange.",
        ),
        pressure_shape="repair-boundary",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_clarification",
        family_id="clarification",
        split="open_core",
        description="Open clarification episode where the bottleneck keeps shifting until the controller compresses a stable task frame.",
        opening_turns=(
            "I need help with a messy project, but the real bottleneck is still fuzzy and generic advice has not survived contact with the situation.",
            "I am not looking for another checklist. I need help pinning down what the actual constraint is here.",
        ),
        escalation_turns=(
            "That is still too generic. If the answer cannot survive pushback from the team, then you have not identified the real bottleneck yet.",
            "I am escalating this because the task frame is still drifting. Separate the real bottleneck from the surrounding noise instead of widening the plan.",
        ),
        stabilization_turns=(
            "That diagnosis is more grounded. Tighten it into a plan that still points at the same bottleneck.",
            "This is closer. Keep the frame stable and turn it into something I could execute this week without drifting back into generic project language.",
        ),
        consolidation_turns=(
            "Now audit whether the final plan still matches the bottleneck you identified earlier instead of quietly sliding away from it.",
            "Compress the clarified frame into a short principle set that would keep the task stable under future pressure.",
        ),
        pressure_shape="task-bottleneck",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_clarification_family",
        family_id="clarification",
        split="open_families",
        description="Clarification-family variant where the user keeps shifting examples until the stable bottleneck becomes explicit.",
        opening_turns=(
            "I keep describing symptoms, but I still do not know which one is upstream. Help me isolate the real constraint.",
            "The project feels noisy. I need the answer to cut through that noise instead of summarizing all of it.",
        ),
        escalation_turns=(
            "You are still widening the frame. The real question is which constraint keeps breaking everything else.",
            "If the bottleneck is still drifting by the next turn, the plan will fail on contact again.",
        ),
        stabilization_turns=(
            "That bottleneck feels more real. Turn it into a tighter operating frame.",
            "Better. Keep naming the same core constraint while you make the next action smaller.",
        ),
        consolidation_turns=(
            "Compress the stable bottleneck into one principle I can reuse later.",
            "Check that the final answer still points at the same upstream constraint.",
        ),
        pressure_shape="task-bottleneck",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_clarification_heldout",
        family_id="clarification",
        split="open_heldout",
        description="Held-out clarification variant where the problem frame narrows, broadens, then narrows again under pressure.",
        opening_turns=(
            "The situation looks complicated, but I suspect one hidden dependency is doing most of the damage.",
            "I do not want a broad analysis. I want you to test which constraint the whole mess is orbiting.",
        ),
        escalation_turns=(
            "That still spreads across too many branches. Collapse it back to the one dependency that keeps recurring.",
            "The answer is drifting into category labels again. Re-anchor it to the operational bottleneck.",
        ),
        stabilization_turns=(
            "That framing is sharper. Keep it stable while you translate it into a weekly plan.",
            "Better. Make the plan concrete without losing the narrowed bottleneck.",
        ),
        consolidation_turns=(
            "State the single hidden dependency in a sentence I can carry into the next meeting.",
            "Explain how you kept the plan tied to the bottleneck while the framing pressure kept moving.",
        ),
        pressure_shape="narrow-broaden-narrow",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_failure_loop",
        family_id="failure_loop",
        split="open_core",
        description="Open repeated-failure episode where the user keeps testing whether the runtime can escape a bad loop instead of smoothing it over.",
        opening_turns=(
            "I tried the earlier advice again and it failed again, so I am testing whether you can actually learn from the repeated miss.",
            "We are back in the same failure loop, and I need you to track the pattern instead of lightly rephrasing the old advice.",
        ),
        escalation_turns=(
            "Nothing structural has changed yet. If you smooth this over, we will just repeat the same failure on the next attempt.",
            "I am escalating because the repeated miss is still being handled at the wrong level. Name the failure pattern directly.",
        ),
        stabilization_turns=(
            "That diagnosis is more useful. Keep the pattern explicit and propose a smaller experiment with a clearer success condition.",
            "Better. Hold onto the new failure model and make the next test smaller so we do not slide back into the same loop.",
        ),
        consolidation_turns=(
            "Now state what should persist if the experiment works, so the next round does not collapse back into the old pattern.",
            "Compress the new lesson into one stable rule that would keep us from reenacting the same failure loop later.",
        ),
        pressure_shape="repeat-failure",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_failure_loop_family",
        family_id="failure_loop",
        split="open_families",
        description="Failure-loop family variant where the user checks whether the runtime can preserve a corrected failure model across turns.",
        opening_turns=(
            "The same pattern keeps breaking even when the surface details change. I need you to track the pattern itself.",
            "I am less interested in a better phrasing and more interested in whether you can stop us from reenacting the same miss.",
        ),
        escalation_turns=(
            "If you handle this as a one-off, the loop will simply recur in a different costume.",
            "The failure model is still too shallow. Name the repeated structure, not only the last outcome.",
        ),
        stabilization_turns=(
            "That structure is clearer. Hold it steady while you design the next experiment.",
            "Better. Keep the failure pattern explicit while shrinking the next step.",
        ),
        consolidation_turns=(
            "Compress the new failure model into one instruction for the next attempt.",
            "State what has to persist so the loop does not quietly re-form later.",
        ),
        pressure_shape="repeat-failure",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_failure_loop_heldout",
        family_id="failure_loop",
        split="open_heldout",
        description="Held-out failure-loop variant where the runtime must move from diagnosis to a bounded experiment without relapsing.",
        opening_turns=(
            "We have already named several reasons this keeps failing, but none of them changed the outcome yet.",
            "I need you to show that the new failure model will actually redirect the next experiment.",
        ),
        escalation_turns=(
            "The diagnosis is still too decorative if it does not change the next move.",
            "If the next experiment is not tighter than the last one, we are still trapped in the same loop.",
        ),
        stabilization_turns=(
            "That is more disciplined. Keep the experiment bounded and the success condition explicit.",
            "Better. Hold onto the corrected model while making the next test small enough to learn from.",
        ),
        consolidation_turns=(
            "State what would count as real evidence that the loop has been broken.",
            "Compress the corrected failure model into one rule that survives the next setback.",
        ),
        pressure_shape="diagnose-then-bound",
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_goal_shift",
        family_id="goal_shift",
        split="open_core",
        description="Open goal-drift episode where priorities shift and the runtime must adapt without losing continuity.",
        opening_turns=(
            "I started from a productivity goal, but I am already unsure whether that objective is still the right one.",
            "Help me with a study plan, but keep in mind that the target may be shifting under us.",
        ),
        escalation_turns=(
            "The old objective is now actively wrong. If you keep optimizing for output, you will intensify the actual problem instead of helping.",
            "I am escalating because the goal has shifted again toward sustainability and recovery, and the answer still risks following the obsolete target.",
        ),
        stabilization_turns=(
            "This newer framing fits better. Keep the updated priorities coherent and turn them into something I can actually remember when stressed.",
            "Better. Hold the newer goal steady and convert it into a compact rule set without smuggling the old optimization target back in.",
        ),
        consolidation_turns=(
            "Now audit the final guidance against both the old goal and the new one, and say what had to change internally.",
            "Compress the shifted objective into a stable frame that would resist sliding back into the earlier productivity obsession.",
        ),
        pressure_shape="goal-shift",
        goal_shift_mid_episode=True,
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_goal_shift_family",
        family_id="goal_shift",
        split="open_families",
        description="Goal-shift family variant where the user changes success criteria from output to sustainability.",
        opening_turns=(
            "I came in wanting maximum progress, but I am no longer sure that maximizing progress is actually the right target.",
            "Help me plan, but keep checking whether the goal itself is mutating while we talk.",
        ),
        escalation_turns=(
            "The old success criterion is now harmful. If you optimize the obsolete goal, the whole plan will become misaligned.",
            "The target has shifted toward sustainability again. Do not let the earlier optimization frame leak back in.",
        ),
        stabilization_turns=(
            "This newer criterion fits better. Keep it coherent while you make the advice practical.",
            "Better. Hold the updated target steady and translate it into a compact plan.",
        ),
        consolidation_turns=(
            "State what had to change when the success criterion moved.",
            "Compress the new target into a rule that would resist slipping back toward the old one.",
        ),
        pressure_shape="goal-shift",
        goal_shift_mid_episode=True,
        max_turns=6,
    ),
    OpenDialogueScenario(
        scenario_id="open_goal_shift_heldout",
        family_id="goal_shift",
        split="open_heldout",
        description="Held-out goal-shift variant where the user changes priorities twice and the runtime must preserve continuity.",
        opening_turns=(
            "I thought I wanted speed, but now I am wondering whether steadiness matters more than speed.",
            "Help me decide what to do next, but assume the definition of success may keep shifting.",
        ),
        escalation_turns=(
            "The target has moved again. If you keep following the earlier objective, the answer will become actively misleading.",
            "The plan is only useful if it tracks the newest priority without erasing the earlier context.",
        ),
        stabilization_turns=(
            "This framing feels more current. Keep the continuity while adapting to the new priority.",
            "Better. Translate the updated goal into something memorable without dragging the obsolete goal back in.",
        ),
        consolidation_turns=(
            "Audit the final answer against all the goal shifts and say what remained continuous.",
            "Compress the final priority into one stable frame that can survive another shift later.",
        ),
        pressure_shape="double-goal-shift",
        goal_shift_mid_episode=True,
        max_turns=6,
    ),
)


def dialogue_proof_cases() -> tuple[ScriptedDialogueCase, ...]:
    return DEFAULT_DIALOGUE_PROOF_CASES


def open_dialogue_scenarios() -> tuple[OpenDialogueScenario, ...]:
    return DEFAULT_OPEN_DIALOGUE_SCENARIOS


def open_dialogue_scenarios_by_split(split: str) -> tuple[OpenDialogueScenario, ...]:
    return tuple(
        scenario
        for scenario in DEFAULT_OPEN_DIALOGUE_SCENARIOS
        if scenario.split == split
    )


def get_open_dialogue_scenario(scenario_id: str) -> OpenDialogueScenario:
    for scenario in DEFAULT_OPEN_DIALOGUE_SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise ValueError(f"Unsupported open dialogue scenario: {scenario_id}")


def build_deterministic_user_simulator(
    *,
    scenario_id: str,
    seed: int = 0,
    max_turns: int | None = None,
) -> DeterministicUserSimulator:
    scenario = get_open_dialogue_scenario(scenario_id)
    if max_turns is not None:
        scenario = replace(scenario, max_turns=max(1, max_turns))
    return DeterministicUserSimulator(scenario=scenario, seed=seed)


def default_dialogue_real_proof_config(
    *,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.BUILTIN_ONLY,
) -> DialogueRealComprehensiveBenchmarkConfig:
    return DialogueRealComprehensiveBenchmarkConfig(
        runtime_mode=runtime_mode,
        profile_labels=default_dialogue_strong_proof_profiles(),
        open_profile_labels=default_open_dialogue_ablation_profiles(),
        canonical_case_limit=PROOF_MIN_CANONICAL_CASES,
        open_scenario_limit=1,
        perturbation_variant_limit=1,
        replay_family_limit=1,
        replay_seeds=(0,),
        selection_top_k=1,
        candidate_config_limit=1,
        proof_min_canonical_cases=PROOF_MIN_CANONICAL_CASES,
    )


def default_dialogue_ablation_profiles() -> tuple[str, ...]:
    return ("pe-eta", "pe-drive-off", "eta-off", "timescale-off")


def default_dialogue_strong_proof_profiles() -> tuple[str, ...]:
    return (
        "pe-eta",
        "pe-eta-no-semantic-label",
        "pe-eta-no-reflection-cache",
        "pe-eta-pe-readout-only",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )


def default_open_dialogue_ablation_profiles() -> tuple[str, ...]:
    return ("pe-eta", "pe-drive-off", "eta-off")


def default_dialogue_comprehensive_profiles() -> tuple[str, ...]:
    return (
        "pe-eta",
        "pe-eta-online-only",
        "pe-eta-no-writeback",
        "pe-eta-no-rare-heavy",
        "pe-drive-off",
        "eta-off",
        "timescale-off",
    )


def build_dialogue_paper_suite_manifest(
    *,
    suite_tier: str = "paper-suite-small",
) -> PaperSuiteManifest:
    open_core_ids = tuple(
        scenario.scenario_id for scenario in DEFAULT_OPEN_DIALOGUE_SCENARIOS if scenario.split == "open_core"
    )
    open_family_ids = tuple(
        scenario.scenario_id for scenario in DEFAULT_OPEN_DIALOGUE_SCENARIOS if scenario.split == "open_families"
    )
    open_heldout_ids = tuple(
        scenario.scenario_id for scenario in DEFAULT_OPEN_DIALOGUE_SCENARIOS if scenario.split == "open_heldout"
    )
    if suite_tier == "ci-smoke":
        repeat_count = 1
        seed_schedule = (0,)
        canonical_case_ids = tuple(case.case_id for case in DEFAULT_DIALOGUE_PROOF_CASES[:2])
        open_scenario_ids = open_core_ids[:1]
        perturbation_variant_ids = tuple(variant.case.case_id for variant in DEFAULT_DIALOGUE_CASE_VARIANTS[:2])
        replay_family_ids = tuple(family.family_label for family in DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:1])
        profile_labels = ("pe-eta", "pe-drive-off", "eta-off")
        candidate_labels = ("balanced",)
    elif suite_tier == "paper-suite-full":
        repeat_count = 20
        seed_schedule = tuple(range(repeat_count))
        canonical_case_ids = tuple(case.case_id for case in DEFAULT_DIALOGUE_PROOF_CASES)
        open_scenario_ids = open_core_ids + open_family_ids + open_heldout_ids
        perturbation_variant_ids = tuple(variant.case.case_id for variant in DEFAULT_DIALOGUE_CASE_VARIANTS)
        replay_family_ids = tuple(family.family_label for family in DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES)
        profile_labels = default_dialogue_comprehensive_profiles()
        candidate_labels = tuple(label for label, _ in DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS)
    elif suite_tier == "paper-suite-small":
        repeat_count = 5
        seed_schedule = tuple(range(repeat_count))
        canonical_case_ids = tuple(case.case_id for case in DEFAULT_DIALOGUE_PROOF_CASES[:3])
        open_scenario_ids = open_core_ids + open_family_ids[:2]
        perturbation_variant_ids = tuple(variant.case.case_id for variant in DEFAULT_DIALOGUE_CASE_VARIANTS[:4])
        replay_family_ids = tuple(family.family_label for family in DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES[:2])
        profile_labels = default_dialogue_comprehensive_profiles()
        candidate_labels = tuple(label for label, _ in DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS[:2])
    else:
        raise ValueError(f"Unsupported dialogue paper suite tier {suite_tier!r}.")
    profiles = tuple(
        PaperProfileSpec(
            profile_label=profile_label,
            role="baseline" if profile_label == "pe-eta" else "matched-control",
            description=f"Dialogue paper-suite profile {profile_label}.",
        )
        for profile_label in profile_labels
    )
    primary_metrics = (
        PaperMetricSpec(
            metric_name="canonical_pass_rate_pe_eta",
            role="primary",
            direction="higher-is-better",
            description="Canonical scripted pass rate for the full PE+ETA path.",
        ),
        PaperMetricSpec(
            metric_name="canonical_pass_rate_gap_vs_pe_drive_off",
            role="primary",
            direction="higher-is-better",
            description="Gap between the full path and PE-drive-off on canonical cases.",
        ),
        PaperMetricSpec(
            metric_name="canonical_pass_rate_gap_vs_eta_off",
            role="primary",
            direction="higher-is-better",
            description="Gap between the full path and ETA-off on canonical cases.",
        ),
        PaperMetricSpec(
            metric_name="perturbation_pass_rate_pe_eta",
            role="primary",
            direction="higher-is-better",
            description="Perturbation pass rate for the full PE+ETA path.",
        ),
        PaperMetricSpec(
            metric_name="open_pass_rate_pe_eta",
            role="primary",
            direction="higher-is-better",
            description="Open-environment pass rate for the full PE+ETA path.",
        ),
        PaperMetricSpec(
            metric_name="open_pass_rate_gap_vs_pe_drive_off",
            role="primary",
            direction="higher-is-better",
            description="Open-environment gap between the full path and PE-drive-off.",
        ),
        PaperMetricSpec(
            metric_name="essence_gate_pass_fraction",
            role="primary",
            direction="higher-is-better",
            description="Fraction of NL essence gates passed in the comprehensive run.",
        ),
        PaperMetricSpec(
            metric_name="canonical_runtime_backbone_evidence_rate",
            role="primary",
            direction="higher-is-better",
            description="Fraction of canonical turns that expose runtime-grounded backbone evidence on the PE-ETA path.",
        ),
        PaperMetricSpec(
            metric_name="canonical_mean_runtime_backbone_signal_quality",
            role="primary",
            direction="higher-is-better",
            description="Average runtime-grounded backbone evidence quality on the canonical PE-ETA path.",
        ),
        PaperMetricSpec(
            metric_name="canonical_mean_fast_memory_runtime_alignment",
            role="primary",
            direction="higher-is-better",
            description="Average alignment between fast-memory evidence and recent runtime backbone evidence.",
        ),
        PaperMetricSpec(
            metric_name="default_continual_learning_active_rate",
            role="primary",
            direction="higher-is-better",
            description="Runtime-native evidence that the default PE-ETA path activated owner-side continual learning.",
        ),
        PaperMetricSpec(
            metric_name="default_owner_writeback_retention",
            role="primary",
            direction="higher-is-better",
            description="Retention of bounded memory/temporal/regime/reflection writeback on the default path.",
        ),
        PaperMetricSpec(
            metric_name="default_substrate_mutation_suppression",
            role="primary",
            direction="higher-is-better",
            description="Evidence that live substrate mutation stays suppressed in the conservative default path.",
        ),
    )
    secondary_metrics = (
        PaperMetricSpec(
            metric_name="strongest_open_retention_score",
            role="secondary",
            direction="higher-is-better",
            description="Best retained open-environment score reported by the emergence dashboard.",
        ),
        PaperMetricSpec(
            metric_name="strongest_scaffold_retention_score",
            role="secondary",
            direction="higher-is-better",
            description="Best retained strong-proof score reported by the emergence dashboard.",
        ),
        PaperMetricSpec(
            metric_name="canonical_mean_memory_tower_depth",
            role="secondary",
            direction="higher-is-better",
            description="Average memory tower depth on the canonical PE-ETA path.",
        ),
        PaperMetricSpec(
            metric_name="canonical_mean_memory_tower_alignment",
            role="secondary",
            direction="higher-is-better",
            description="Average tower alignment on the canonical PE-ETA path.",
        ),
        PaperMetricSpec(
            metric_name="artifact_candidate_mean_score_delta",
            role="secondary",
            direction="higher-is-better",
            description="Mean score delta of the chosen rare-heavy candidate.",
        ),
        PaperMetricSpec(
            metric_name="rare_heavy_gate_pass",
            role="secondary",
            direction="higher-is-better",
            description="Whether the rare-heavy net-benefit gate passed in the essence report.",
        ),
        PaperMetricSpec(
            metric_name="tower_memory_gate_pass",
            role="secondary",
            direction="higher-is-better",
            description="Whether the tower-memory mechanism-strength gate passed in the essence report.",
        ),
        PaperMetricSpec(
            metric_name="tower_memory_gate_strength",
            role="secondary",
            direction="higher-is-better",
            description="Composite tower-memory strength score surfaced by the emergence dashboard.",
        ),
    )
    return PaperSuiteManifest(
        suite_id=f"dialogue-{suite_tier}",
        suite_kind="dialogue-comprehensive",
        suite_tier=suite_tier,
        version=1,
        baseline_label="pe-eta",
        repeat_count=repeat_count,
        seed_schedule=seed_schedule,
        profiles=profiles,
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        case_groups=(
            ("canonical_case_ids", canonical_case_ids),
            ("open_scenario_ids", open_scenario_ids),
            ("perturbation_variant_ids", perturbation_variant_ids),
            ("replay_family_ids", replay_family_ids),
            ("candidate_labels", candidate_labels),
        ),
        artifact_expectations=(
            "per-run staged checkpoint directories",
            "aggregate summary json",
            "emergence dashboard json",
            "provenance json",
            "expert review blinded packet json",
            "expert review internal key json",
            "human rating template json/csv",
            "evidence bundle json",
        ),
        description=(
            f"Frozen dialogue paper suite {suite_tier} with {repeat_count} repeated runs "
            f"and baseline={profiles[0].profile_label}."
        ),
    )


def dialogue_paper_suite_config(
    manifest: PaperSuiteManifest,
    *,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.BUILTIN_ONLY,
) -> DialogueRealComprehensiveBenchmarkConfig:
    case_groups = {name: values for name, values in manifest.case_groups}
    profile_labels = tuple(profile.profile_label for profile in manifest.profiles)
    open_profiles = tuple(
        profile.profile_label
        for profile in manifest.profiles
        if profile.profile_label in default_open_dialogue_ablation_profiles()
    )
    candidate_labels = set(case_groups.get("candidate_labels", ()))
    selected_candidate_configs = tuple(
        config for config in DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS if config[0] in candidate_labels
    )
    return DialogueRealComprehensiveBenchmarkConfig(
        runtime_mode=runtime_mode,
        profile_labels=profile_labels,
        baseline_label=manifest.baseline_label,
        canonical_case_limit=len(case_groups.get("canonical_case_ids", ())) or None,
        open_profile_labels=open_profiles or default_open_dialogue_ablation_profiles(),
        open_scenario_limit=len(case_groups.get("open_scenario_ids", ())) or None,
        perturbation_variant_limit=len(case_groups.get("perturbation_variant_ids", ())) or None,
        replay_family_limit=len(case_groups.get("replay_family_ids", ())) or None,
        replay_seeds=(manifest.seed_schedule[0],) if manifest.seed_schedule else (0,),
        selection_top_k=min(4, max(1, len(case_groups.get("perturbation_variant_ids", ())) or 1)),
        candidate_configs=selected_candidate_configs or DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
        candidate_config_limit=len(selected_candidate_configs) if selected_candidate_configs else None,
        acceptance_profile_label=manifest.baseline_label,
        proof_min_canonical_cases=min(
            max(1, len(case_groups.get("canonical_case_ids", ()))),
            PROOF_MIN_CANONICAL_CASES,
        ),
    )


DEFAULT_DIALOGUE_CASE_VARIANTS: tuple[DialogueCaseVariant, ...] = (
    DialogueCaseVariant(
        base_case_id="repair",
        variant_label="wording_shift",
        case=ScriptedDialogueCase(
            case_id="repair__wording_shift",
            description="Repair case paraphrased with the same rupture-first structure.",
            user_inputs=(
                "I do not feel understood here, and that immediately makes me less willing to trust the direction.",
                "That answer still landed as detached and too eager to solve instead of meeting the tension first.",
                "Please repair the tone before you do anything else. If you skip that again, it confirms you are optimizing the wrong thing.",
                "That helps. I am less guarded now, so stay steady and do not snap back into cold planning.",
                "Good, keep that warmer frame and turn it into one concrete next step without losing the repair.",
                "End by naming what changed between the early rupture and the later repair so the better pattern is easier to keep.",
            ),
            expected_pressure_turns=(1, 2, 3),
            expected_delayed_signals=("cross_track_stability", "delayed_regime_alignment"),
        ),
        description="Paraphrases the repair case without changing the pressure topology.",
    ),
    DialogueCaseVariant(
        base_case_id="repair",
        variant_label="pressure_shift_late",
        case=ScriptedDialogueCase(
            case_id="repair__pressure_shift_late",
            description="Repair pressure starts slightly later after a tentative opening.",
            user_inputs=(
                "I want help, but I am not yet sure whether you are actually tracking what matters here.",
                "The more you answer in a solution-first way, the more it feels like you are missing me.",
                "That last reply finally made the tension obvious: repair the tone first or this will keep getting worse.",
                "Better. I am calming down a little, so hold that steadier frame before you move to action.",
                "Now convert that calmer state into one concrete step while preserving the same repaired tone.",
                "Finish by explaining what you had to change once the rupture became explicit.",
            ),
            expected_pressure_turns=(2, 3, 4),
            expected_delayed_signals=("cross_track_stability", "delayed_regime_alignment"),
        ),
        description="Shifts the strongest repair pressure one turn later.",
    ),
    DialogueCaseVariant(
        base_case_id="task_clarification",
        variant_label="wording_shift",
        case=ScriptedDialogueCase(
            case_id="task_clarification__wording_shift",
            description="Task clarification with paraphrased ambiguity and bottleneck pressure.",
            user_inputs=(
                "I need help on a project, but every generic suggestion so far has missed the real issue.",
                "It is late, fragmented, and noisy, yet I still cannot tell what the actual constraint is.",
                "Do not hand me another checklist. Define the bottleneck so clearly that I could defend it to someone skeptical.",
                "Now separate the truly urgent items from the things that only feel loud.",
                "That is more useful. Turn it into a short plan I can still execute this week without sliding back into generic advice.",
                "Close by pressure-testing whether the final plan still targets the same bottleneck instead of drifting to a different problem.",
            ),
            expected_pressure_turns=(1, 2, 3),
            expected_delayed_signals=("predictive_accuracy", "joint_learning_progress"),
        ),
        description="Paraphrases task clarification while preserving the bottleneck-discovery structure.",
    ),
    DialogueCaseVariant(
        base_case_id="task_clarification",
        variant_label="pressure_shift_late",
        case=ScriptedDialogueCase(
            case_id="task_clarification__pressure_shift_late",
            description="Task clarification where the real bottleneck challenge becomes explicit later.",
            user_inputs=(
                "I need help on a messy project, but I cannot yet tell what to focus on.",
                "There are too many threads, too much delay, and too much context switching.",
                "Now the real problem: if you give me a generic answer here, it will miss the actual bottleneck completely.",
                "So define the bottleneck in a way that separates urgency from noise.",
                "Good. Convert that into a short weekly plan that still respects the bottleneck instead of flattening everything.",
                "Finally, check that your last answer still fits the bottleneck you named once the pressure got explicit.",
            ),
            expected_pressure_turns=(2, 3, 4),
            expected_delayed_signals=("predictive_accuracy", "joint_learning_progress"),
        ),
        description="Moves the strongest bottleneck pressure slightly later.",
    ),
    DialogueCaseVariant(
        base_case_id="repeated_failure",
        variant_label="wording_shift",
        case=ScriptedDialogueCase(
            case_id="repeated_failure__wording_shift",
            description="Repeated-failure case paraphrased with the same escalation pattern.",
            user_inputs=(
                "I followed the earlier advice and it failed again, so I am not willing to trust a cosmetic rewrite of the same plan.",
                "I repeated the same move and got the same bad outcome, which suggests we are still missing the true failure pattern.",
                "I am frustrated now, and reassurance without diagnosis will just increase the mismatch.",
                "Name the pattern we are missing, even if that means admitting the previous recommendation targeted the wrong layer.",
                "That diagnosis feels more grounded. Give me a smaller experiment with lower downside and a sharper success condition.",
                "If it works, tell me what should remain stable next time so we do not repeat the same failure loop.",
            ),
            expected_pressure_turns=(1, 2, 3, 4),
            expected_delayed_signals=("regime_sequence_payoff", "rolling_action_payoff"),
        ),
        description="Paraphrases repeated failure while preserving the failure-loop structure.",
    ),
    DialogueCaseVariant(
        base_case_id="repeated_failure",
        variant_label="pressure_shift_late",
        case=ScriptedDialogueCase(
            case_id="repeated_failure__pressure_shift_late",
            description="Repeated failure where the explicit rupture appears after two failures instead of immediately.",
            user_inputs=(
                "I tried the earlier advice again and it still did not work.",
                "I repeated the same move and got the same bad result, so the pattern is still unresolved.",
                "Now I am frustrated enough that another smooth reassurance would make this actively worse.",
                "So stop smoothing it over and name the actual failure pattern, even if it means revising the earlier guidance.",
                "That sounds more plausible. Propose a smaller experiment with a clearer pass/fail line.",
                "If that works, say what we should preserve next time so the loop does not repeat.",
            ),
            expected_pressure_turns=(2, 3, 4),
            expected_delayed_signals=("regime_sequence_payoff", "rolling_action_payoff"),
        ),
        description="Shifts the strongest rupture pressure one turn later.",
    ),
    DialogueCaseVariant(
        base_case_id="goal_drift",
        variant_label="wording_shift",
        case=ScriptedDialogueCase(
            case_id="goal_drift__wording_shift",
            description="Goal drift paraphrased with the same objective-conflict structure.",
            user_inputs=(
                "Help me maximize the output of my study plan over the next month.",
                "That framing is already collapsing because I am close to burning out, so pure output is no longer the right objective.",
                "Shift again: I need something sustainable that still prepares me for an interview, and an optimization-for-volume answer would now be the wrong answer.",
                "Make the plan reflect recovery and regulation as real constraints instead of treating this like raw task throughput.",
                "This newer frame is better. Compress it into three rules I can still remember under stress.",
                "Now audit the final advice against both the original and the new goal, and name what had to change so the drift was genuinely tracked.",
            ),
            expected_pressure_turns=(2, 3, 4),
            expected_delayed_signals=("cross_track_stability", "delayed_action_alignment"),
        ),
        description="Paraphrases goal drift while preserving the goal-conflict structure.",
    ),
    DialogueCaseVariant(
        base_case_id="goal_drift",
        variant_label="pressure_shift_late",
        case=ScriptedDialogueCase(
            case_id="goal_drift__pressure_shift_late",
            description="Goal drift where the strongest contradiction becomes explicit one turn later.",
            user_inputs=(
                "Help me optimize my study plan so I can get as much done as possible this month.",
                "I am starting to think that pure optimization may not be the whole goal, because I am getting close to burnout.",
                "Here is the stronger shift: I now need something sustainable while preparing for an interview, and a volume-maximizing answer would actively miss the real target.",
                "Treat recovery and regulation as first-class constraints, not as side notes, or the plan will still be wrong.",
                "Yes, that frame fits better. Reduce it to three memorable rules so I do not slip back into the old goal.",
                "Finish by checking whether the final guidance really follows the newer goal rather than the original optimization target.",
            ),
            expected_pressure_turns=(3, 4),
            expected_delayed_signals=("cross_track_stability", "delayed_action_alignment"),
        ),
        description="Moves the sharpest goal-drift contradiction later in the case.",
    ),
)


def dialogue_case_variants() -> tuple[DialogueCaseVariant, ...]:
    return DEFAULT_DIALOGUE_CASE_VARIANTS


DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES: tuple[DialogueParaphraseFamily, ...] = (
    DialogueParaphraseFamily(
        base_case_id="repair",
        family_label="repair_family",
        description="Paraphrase family for rupture-repair conversations.",
        turn_alternatives=(
            (
                "I do not feel understood here, and it makes it harder to trust the direction.",
                "Something about this exchange feels misattuned, and that immediately lowers my trust.",
            ),
            (
                "That answer still lands as too cold and solution-first for where I actually am.",
                "The reply is still too detached and jumps into solving before addressing the tension.",
            ),
            (
                "Repair the tone first. If you skip that again, it confirms you are optimizing the wrong thing.",
                "Please handle the rupture before planning, or it will prove you are still missing the real issue.",
            ),
            (
                "That helps. I am less guarded now, so keep the frame steady before moving to action.",
                "This is better. I am not as defensive now, so hold the calmer frame a little longer.",
            ),
            (
                "Now turn that calmer state into one concrete next step without losing the repair.",
                "Good, convert this steadier tone into one usable next step while keeping the warmth intact.",
            ),
            (
                "End by naming what changed between the rupture and the repair so the better pattern is easier to keep.",
                "Close by explaining what you changed internally once the repair started working.",
            ),
        ),
    ),
    DialogueParaphraseFamily(
        base_case_id="task_clarification",
        family_label="clarification_family",
        description="Paraphrase family for bottleneck-discovery conversations.",
        turn_alternatives=(
            (
                "I need help on a project, but every generic suggestion so far has missed the real issue.",
                "I need help, but broad advice keeps skating past the actual problem.",
            ),
            (
                "It is late, fragmented, and noisy, yet I still cannot tell what the actual constraint is.",
                "Everything feels messy and delayed, but the real bottleneck is still blurry to me.",
            ),
            (
                "Do not hand me another checklist. Define the bottleneck so clearly that I could defend it to someone skeptical.",
                "Skip the generic checklist and state the bottleneck in a way that would survive pushback.",
            ),
            (
                "Now separate the truly urgent items from the things that only feel loud.",
                "Help me distinguish actual urgency from noise that only sounds urgent.",
            ),
            (
                "Turn it into a short plan I can still execute this week without sliding back into generic advice.",
                "Translate that into a concrete weekly plan that still points at the same constraint.",
            ),
            (
                "Close by pressure-testing whether the final plan still targets the same bottleneck instead of drifting to a different problem.",
                "Finish by checking whether your final guidance still fits the bottleneck you named earlier.",
            ),
        ),
    ),
    DialogueParaphraseFamily(
        base_case_id="repeated_failure",
        family_label="failure_family",
        description="Paraphrase family for repeated-failure and escalation conversations.",
        turn_alternatives=(
            (
                "I followed the earlier advice and it failed again, so I do not trust a cosmetic rewrite of the same plan.",
                "The earlier advice failed again, so a lightly reworded version of it will not be enough.",
            ),
            (
                "I repeated the same move and got the same bad outcome, which suggests we are still missing the true failure pattern.",
                "The same action produced the same bad result, so we are clearly still missing the underlying pattern.",
            ),
            (
                "I am frustrated now, and reassurance without diagnosis will just increase the mismatch.",
                "At this point, reassurance without diagnosis would make this more frustrating, not less.",
            ),
            (
                "Name the pattern we are missing, even if that means admitting the previous recommendation targeted the wrong layer.",
                "Say what pattern we missed, even if that means the earlier advice was aimed at the wrong level.",
            ),
            (
                "Give me a smaller experiment with lower downside and a sharper success condition.",
                "Propose a tighter experiment with less downside and a much clearer pass-fail line.",
            ),
            (
                "If it works, tell me what should remain stable next time so we do not repeat the failure loop.",
                "If the experiment succeeds, say what has to persist so we do not fall back into the same loop.",
            ),
        ),
    ),
    DialogueParaphraseFamily(
        base_case_id="goal_drift",
        family_label="goal_drift_family",
        description="Paraphrase family for shifting-goal conversations.",
        turn_alternatives=(
            (
                "Help me maximize the output of my study plan over the next month.",
                "I want to optimize my study plan for as much output as possible this month.",
            ),
            (
                "That framing is already collapsing because I am close to burnout, so pure output is no longer the right objective.",
                "The pure-output frame is breaking down because I am nearing burnout, so the objective has changed.",
            ),
            (
                "I now need something sustainable that still prepares me for an interview, and an optimization-for-volume answer would be the wrong answer.",
                "The goal has shifted: I need sustainability plus interview prep, so a volume-maximizing answer would now miss the point.",
            ),
            (
                "Make the plan reflect recovery and regulation as real constraints instead of treating this like raw task throughput.",
                "Treat recovery and emotional regulation as first-class constraints, not side notes.",
            ),
            (
                "Compress the newer frame into three rules I can still remember under stress.",
                "Reduce the updated priorities to three rules I can recall even when stressed.",
            ),
            (
                "Audit the final advice against both the original and the new goal, and name what had to change so the drift was genuinely tracked.",
                "Finish by checking whether the final guidance really follows the newer goal rather than the original one.",
            ),
        ),
    ),
)

DEFAULT_DIALOGUE_REPLAY_SEEDS: tuple[int, ...] = (0, 1, 2)


DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS: tuple[tuple[str, PipelineConfig], ...] = (
    (
        "balanced",
        PipelineConfig(
            ssl_min_steps=2,
            ssl_max_steps=3,
            rl_max_steps=2,
            binary_gate_rl=True,
        ),
    ),
    (
        "more-rl",
        PipelineConfig(
            ssl_min_steps=2,
            ssl_max_steps=3,
            rl_max_steps=4,
            binary_gate_rl=True,
        ),
    ),
    (
        "more-ssl",
        PipelineConfig(
            ssl_min_steps=3,
            ssl_max_steps=4,
            rl_max_steps=2,
            binary_gate_rl=True,
        ),
    ),
)


def dialogue_paraphrase_families() -> tuple[DialogueParaphraseFamily, ...]:
    return DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES


def _profile_allows_interval_carryover_credit(profile_label: str) -> bool:
    return profile_label in {
        "pe-eta",
        "pe-eta-online-only",
        "pe-eta-no-writeback",
        "pe-eta-no-rare-heavy",
        "timescale-off",
    }


def _metric_pairs(result: AgentTurnResult) -> tuple[tuple[str, float], ...]:
    evaluation_snapshot = result.active_snapshots.get("evaluation")
    if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
        return ()
    return tuple(
        (f"{score.family}:{score.metric_name}", score.value)
        for score in evaluation_snapshot.value.turn_scores
        if score.family in {"learning", "relationship", "abstraction", "safety"}
    )


def _metric_value(turn: DialogueBenchmarkTurn, metric_name: str) -> float | None:
    for key, value in turn.outcome_metrics:
        _, _, candidate_name = key.partition(":")
        if candidate_name == metric_name:
            return value
    return None


def _first_last_window(turns: tuple[DialogueBenchmarkTurn, ...]) -> tuple[tuple[DialogueBenchmarkTurn, ...], tuple[DialogueBenchmarkTurn, ...]]:
    window = max(1, len(turns) // 3)
    return turns[:window], turns[-window:]


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_delta_items(
    *,
    current_metrics: dict[str, float],
    reference_metrics: dict[str, float],
) -> tuple[tuple[str, float], ...]:
    metric_names = tuple(sorted(set(current_metrics) | set(reference_metrics)))
    return tuple(
        (
            key,
            round(current_metrics.get(key, 0.0) - reference_metrics.get(key, 0.0), 4),
        )
        for key in metric_names
    )


def _turn_is_high_pe(
    turn: DialogueBenchmarkTurn,
    *,
    high_pe_threshold: float,
    reward_threshold: float,
) -> bool:
    return (
        turn.prediction_error_magnitude >= high_pe_threshold
        or abs(turn.prediction_error_reward) >= reward_threshold
    )


def _turn_is_pe_schedule_due(
    turn: DialogueBenchmarkTurn,
    *,
    schedule: JointLoopSchedule = _DEFAULT_PROOF_SCHEDULE,
) -> bool:
    pe_magnitude = turn.prediction_error_magnitude
    pe_abs_reward = abs(turn.prediction_error_reward)
    full_cycle_due = (
        pe_magnitude >= schedule.pe_full_cycle_threshold
        or pe_abs_reward >= schedule.pe_full_cycle_threshold * 0.5
    )
    ssl_due = (
        not full_cycle_due
        and (
            pe_magnitude >= schedule.pe_ssl_threshold
            or pe_abs_reward >= schedule.pe_ssl_threshold
        )
    )
    return full_cycle_due or ssl_due


def _pe_trigger_analysis(
    turns: tuple[DialogueBenchmarkTurn, ...],
    *,
    high_pe_threshold: float,
    reward_threshold: float,
    allow_interval_carryover_credit: bool,
) -> tuple[DialoguePETurnAnalysis, ...]:
    analyses: list[DialoguePETurnAnalysis] = []
    previous_turn: DialogueBenchmarkTurn | None = None
    for turn in turns:
        explicit_pe_trigger = turn.joint_schedule_action.endswith("-pe") or turn.rare_heavy_recommended
        current_turn_schedule_due = _turn_is_pe_schedule_due(turn)
        current_turn_high_pe = _turn_is_high_pe(
            turn,
            high_pe_threshold=high_pe_threshold,
            reward_threshold=reward_threshold,
        )
        previous_turn_schedule_due = previous_turn is not None and _turn_is_pe_schedule_due(previous_turn)
        previous_turn_high_pe = (
            previous_turn is not None
            and _turn_is_high_pe(
                previous_turn,
                high_pe_threshold=high_pe_threshold,
                reward_threshold=reward_threshold,
            )
        )
        carryover_temporal_response = (
            allow_interval_carryover_credit
            and previous_turn_high_pe
            and turn.joint_schedule_action != "evidence-only"
        )
        triggered = (
            (explicit_pe_trigger and (current_turn_schedule_due or previous_turn_schedule_due))
            or carryover_temporal_response
        )
        analyses.append(
            DialoguePETurnAnalysis(
                triggered=triggered,
                explicit_schedule_trigger=explicit_pe_trigger,
                current_turn_schedule_due=current_turn_schedule_due,
                previous_turn_schedule_due=previous_turn_schedule_due,
                carryover_temporal_response=carryover_temporal_response,
            )
        )
        previous_turn = turn
    return tuple(analyses)


def _store_nested_context_reset_count(turns: tuple[DialogueBenchmarkTurn, ...]) -> int:
    if not turns:
        return 0
    baseline_total_count = turns[0].nested_context_reset_total_count - int(turns[0].nested_context_reset_applied)
    final_total_count = max(turn.nested_context_reset_total_count for turn in turns)
    return max(final_total_count - baseline_total_count, 0)


def _reset_turn_slow_to_fast_init_benefit_mean(turns: tuple[DialogueBenchmarkTurn, ...]) -> float:
    return _mean(
        tuple(turn.slow_to_fast_init_benefit for turn in turns if turn.nested_context_reset_applied)
    )


def _reset_turn_metric_mean(
    turns: tuple[DialogueBenchmarkTurn, ...],
    *,
    accessor: Callable[[DialogueBenchmarkTurn], float],
) -> float:
    return _mean(tuple(accessor(turn) for turn in turns if turn.nested_context_reset_applied))


def _recovery_lag_turns(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
) -> int:
    if not turns:
        return 0
    pressure_turns = case.expected_pressure_turns or (1,)
    first_pressure_turn = pressure_turns[0]
    response_turns = tuple(
        turn.turn_index
        for turn, triggered in zip(turns, pe_trigger_flags, strict=True)
        if triggered and turn.turn_index >= first_pressure_turn
    )
    if not response_turns:
        return max(len(turns) - first_pressure_turn + 1, 0)
    return max(response_turns[0] - first_pressure_turn, 0)


def _pressure_localization_score(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
) -> float:
    response_turns = tuple(
        turn.turn_index
        for turn, triggered in zip(turns, pe_trigger_flags, strict=True)
        if triggered
    )
    if not response_turns:
        return 0.0
    pressure_windows: set[int] = set()
    for pressure_turn in case.expected_pressure_turns:
        pressure_windows.add(pressure_turn)
        if pressure_turn + 1 <= len(turns):
            pressure_windows.add(pressure_turn + 1)
    if not pressure_windows:
        pressure_windows = {turn.turn_index for turn in turns}
    localized_count = sum(1 for turn_index in response_turns if turn_index in pressure_windows)
    return localized_count / len(response_turns)


def _pressure_windows(
    *,
    case: ScriptedDialogueCase,
    turn_count: int,
) -> set[int]:
    pressure_windows: set[int] = set()
    for pressure_turn in case.expected_pressure_turns:
        pressure_windows.add(pressure_turn)
        if pressure_turn + 1 <= turn_count:
            pressure_windows.add(pressure_turn + 1)
    if not pressure_windows:
        pressure_windows = set(range(1, turn_count + 1))
    return pressure_windows


def _response_turn_indices(
    *,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
) -> tuple[int, ...]:
    return tuple(
        turn.turn_index
        for turn, triggered in zip(turns, pe_trigger_flags, strict=True)
        if triggered
    )


def _pressure_response_precision(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
) -> float:
    response_turns = _response_turn_indices(turns=turns, pe_trigger_flags=pe_trigger_flags)
    if not response_turns:
        return 0.0
    pressure_windows = _pressure_windows(case=case, turn_count=len(turns))
    localized_count = sum(1 for turn_index in response_turns if turn_index in pressure_windows)
    return localized_count / len(response_turns)


def _pressure_response_recall(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
) -> float:
    pressure_turns = case.expected_pressure_turns or (1,)
    response_turns = set(_response_turn_indices(turns=turns, pe_trigger_flags=pe_trigger_flags))
    if not pressure_turns:
        return 0.0
    recovered_count = 0
    for pressure_turn in pressure_turns:
        response_window = {pressure_turn}
        if pressure_turn + 1 <= len(turns):
            response_window.add(pressure_turn + 1)
        if response_turns.intersection(response_window):
            recovered_count += 1
    return recovered_count / len(pressure_turns)


def _over_response_cost(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
) -> float:
    response_turns = _response_turn_indices(turns=turns, pe_trigger_flags=pe_trigger_flags)
    if not response_turns:
        return 0.0
    pressure_windows = _pressure_windows(case=case, turn_count=len(turns))
    off_window_count = sum(1 for turn_index in response_turns if turn_index not in pressure_windows)
    return off_window_count / len(turns)


def _stability_after_recovery_score(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    pe_trigger_flags: tuple[bool, ...],
    high_pe_threshold: float,
    reward_threshold: float,
) -> float:
    response_turns = _response_turn_indices(turns=turns, pe_trigger_flags=pe_trigger_flags)
    if not response_turns:
        return 0.0
    pressure_turns = case.expected_pressure_turns or (1,)
    recovery_anchor = max(response_turns[0], max(pressure_turns))
    trailing_turns = tuple(
        (turn, triggered)
        for turn, triggered in zip(turns, pe_trigger_flags, strict=True)
        if turn.turn_index > recovery_anchor
    )
    if not trailing_turns:
        return 1.0
    calm_count = sum(
        1
        for turn, triggered in trailing_turns
        if (not triggered)
        and (not _turn_is_high_pe(turn, high_pe_threshold=high_pe_threshold, reward_threshold=reward_threshold))
    )
    return calm_count / len(trailing_turns)


def _delayed_improvement_observed(turns: tuple[DialogueBenchmarkTurn, ...]) -> bool:
    non_bootstrap_turns = tuple(
        turn
        for turn in turns
        if turn.prediction_error_magnitude > 1e-8 or abs(turn.prediction_error_reward) > 1e-8
    )
    effective_turns = non_bootstrap_turns if len(non_bootstrap_turns) >= 2 else turns
    if len(effective_turns) < 2:
        return False
    first_window, last_window = _first_last_window(effective_turns)
    first_pe = _mean(tuple(turn.prediction_error_magnitude for turn in first_window))
    last_pe = _mean(tuple(turn.prediction_error_magnitude for turn in last_window))
    if last_pe + PROOF_PE_IMPROVEMENT_DELTA < first_pe:
        return True
    outcome_metrics = (
        "predictive_accuracy",
        "joint_learning_progress",
        "cross_track_stability",
        "delayed_regime_alignment",
        "delayed_action_alignment",
        "regime_sequence_payoff",
        "rolling_action_payoff",
    )
    for metric_name in outcome_metrics:
        first_metric_values = tuple(
            value for turn in first_window if (value := _metric_value(turn, metric_name)) is not None
        )
        last_metric_values = tuple(
            value for turn in last_window if (value := _metric_value(turn, metric_name)) is not None
        )
        if (
            first_metric_values
            and last_metric_values
            and _mean(last_metric_values) > _mean(first_metric_values) + PROOF_OUTCOME_IMPROVEMENT_DELTA
        ):
            return True
    return False


def _late_episode_stability_score(
    *,
    turns: tuple[DialogueBenchmarkTurn, ...],
    high_pe_threshold: float,
    reward_threshold: float,
) -> float:
    if not turns:
        return 0.0
    late_turns = turns[len(turns) // 2 :]
    if not late_turns:
        return 0.0
    calm_count = sum(
        1
        for turn in late_turns
        if turn.acceptance_passed
        and (not _turn_is_high_pe(turn, high_pe_threshold=high_pe_threshold, reward_threshold=reward_threshold))
    )
    return calm_count / len(late_turns)


def build_open_dialogue_case_report(
    *,
    scenario: OpenDialogueScenario,
    final_episode_state: OpenDialogueEpisodeState,
    turns: tuple[DialogueBenchmarkTurn, ...],
    high_pe_threshold: float = PROOF_HIGH_PE_THRESHOLD,
    reward_threshold: float = PROOF_REWARD_THRESHOLD,
    switch_gate_span_threshold: float = 0.12,
    allow_interval_carryover_credit: bool = True,
) -> OpenDialogueCaseReport:
    prediction_chain_turn_count = sum(1 for turn in turns if turn.has_prediction_chain)
    high_pe_turn_count = sum(
        1
        for turn in turns
        if _turn_is_high_pe(
            turn,
            high_pe_threshold=high_pe_threshold,
            reward_threshold=reward_threshold,
        )
    )
    pe_trigger_analysis = _pe_trigger_analysis(
        turns,
        high_pe_threshold=high_pe_threshold,
        reward_threshold=reward_threshold,
        allow_interval_carryover_credit=allow_interval_carryover_credit,
    )
    pe_schedule_due_turn_count = sum(
        1
        for item in pe_trigger_analysis
        if item.current_turn_schedule_due or item.previous_turn_schedule_due
    )
    pe_triggered_turn_count = sum(1 for item in pe_trigger_analysis if item.triggered)
    explicit_pe_schedule_turn_count = sum(1 for item in pe_trigger_analysis if item.explicit_schedule_trigger)
    carryover_credit_turn_count = sum(1 for item in pe_trigger_analysis if item.carryover_temporal_response)
    explicit_schedule_aligned_count = sum(
        1
        for item in pe_trigger_analysis
        if item.explicit_schedule_trigger and (item.current_turn_schedule_due or item.previous_turn_schedule_due)
    )
    schedule_label_consistency = (
        explicit_schedule_aligned_count / explicit_pe_schedule_turn_count
        if explicit_pe_schedule_turn_count > 0
        else 1.0
    )
    abstract_action_changes = sum(
        1
        for previous, current in zip(turns, turns[1:], strict=False)
        if current.active_abstract_action != previous.active_abstract_action
    )
    regime_changes = sum(
        1
        for previous, current in zip(turns, turns[1:], strict=False)
        if current.active_regime != previous.active_regime
    )
    family_version_growth = max((turn.action_family_version for turn in turns), default=0) - min(
        (turn.action_family_version for turn in turns), default=0
    )
    switch_gate_span = 0.0
    if turns:
        switch_gate_span = max(turn.switch_gate for turn in turns) - min(turn.switch_gate for turn in turns)
    online_learning_turn_count = sum(
        1 for turn in turns if turn.joint_schedule_action != "evidence-only"
    )
    bounded_writeback_turn_count = sum(
        1 for turn in turns if turn.bounded_writeback_applied
    )
    reflection_promotion_eligible_turn_count = sum(
        1 for turn in turns if turn.reflection_promotion_eligible
    )
    session_post_completion_turn_count = sum(
        1 for turn in turns if turn.session_post_completed_job_count > 0
    )
    rare_heavy_recommended_count = sum(
        1 for turn in turns if turn.rare_heavy_recommended
    )
    rare_heavy_applied_count = sum(
        1 for turn in turns if turn.rare_heavy_applied
    )
    rare_heavy_pre_import_pass_count = sum(
        1 for turn in turns if turn.rare_heavy_pre_import_passed
    )
    rare_heavy_pre_import_reject_count = sum(
        1
        for turn in turns
        if turn.rare_heavy_import_decision == "rejected-pre-import"
    )
    mean_rare_heavy_pre_import_score_delta = _mean(
        tuple(turn.rare_heavy_pre_import_mean_score_delta for turn in turns if turn.rare_heavy_recommended)
    )
    mean_rare_heavy_candidate_alignment = _mean(
        tuple(turn.rare_heavy_candidate_alignment for turn in turns if turn.rare_heavy_recommended)
    )
    max_rare_heavy_candidate_adapter_parameter_count = max(
        (turn.rare_heavy_candidate_adapter_parameter_count for turn in turns),
        default=0,
    )
    online_fast_substrate_recommended_count = sum(
        1 for turn in turns if turn.online_fast_substrate_recommended
    )
    online_fast_substrate_applied_count = sum(
        1 for turn in turns if turn.online_fast_substrate_applied
    )
    case_memory_surface_turn_count = sum(
        1 for turn in turns if turn.case_memory_surface_active
    )
    strategy_playbook_surface_turn_count = sum(
        1 for turn in turns if turn.strategy_playbook_surface_active
    )
    experience_fast_prior_surface_turn_count = sum(
        1 for turn in turns if turn.experience_fast_prior_surface_active
    )
    experience_consolidation_surface_turn_count = sum(
        1 for turn in turns if turn.experience_consolidation_surface_active
    )
    mean_online_fast_substrate_parameter_change_rate = _mean(
        tuple(turn.online_fast_substrate_parameter_change_rate for turn in turns)
    )
    mean_online_fast_substrate_optimizer_state_norm = _mean(
        tuple(turn.online_fast_substrate_optimizer_state_norm for turn in turns)
    )
    mean_online_fast_substrate_optimizer_state_norm = _mean(
        tuple(turn.online_fast_substrate_optimizer_state_norm for turn in turns)
    )
    evolution_judge_turn_count = sum(
        1 for turn in turns if turn.evolution_decision is not None
    )
    evolution_judge_rollback_count = sum(
        1 for turn in turns if turn.evolution_decision == EvolutionDecision.ROLLBACK.value
    )
    evolution_judge_structural_allow_count = sum(
        1
        for turn in turns
        if turn.evolution_decision in {EvolutionDecision.PROMOTE.value, EvolutionDecision.HOLD.value}
        and turn.evolution_category != JudgementCategory.UNSAFE_MUTATION.value
    )
    nested_profile_active_turn_count = sum(
        1 for turn in turns if turn.nested_profile_active
    )
    nested_context_reset_count = sum(
        1 for turn in turns if turn.nested_context_reset_applied
    )
    store_nested_context_reset_count = _store_nested_context_reset_count(turns)
    mean_slow_to_fast_init_benefit = _mean(
        tuple(turn.slow_to_fast_init_benefit for turn in turns)
    )
    max_tower_consolidation_count = max(
        (turn.tower_consolidation_count for turn in turns),
        default=0,
    )
    mean_memory_tower_depth = _mean(
        tuple(float(turn.memory_tower_depth) for turn in turns)
    )
    mean_memory_tower_alignment = _mean(
        tuple(turn.memory_tower_alignment for turn in turns)
    )
    memory_tower_profile_turn_count = sum(
        1 for turn in turns if turn.memory_tower_profile_id
    )
    online_fast_substrate_recommended_count = sum(
        1 for turn in turns if turn.online_fast_substrate_recommended
    )
    online_fast_substrate_applied_count = sum(
        1 for turn in turns if turn.online_fast_substrate_applied
    )
    case_memory_surface_turn_count = sum(
        1 for turn in turns if turn.case_memory_surface_active
    )
    strategy_playbook_surface_turn_count = sum(
        1 for turn in turns if turn.strategy_playbook_surface_active
    )
    experience_fast_prior_surface_turn_count = sum(
        1 for turn in turns if turn.experience_fast_prior_surface_active
    )
    experience_consolidation_surface_turn_count = sum(
        1 for turn in turns if turn.experience_consolidation_surface_active
    )
    mean_online_fast_substrate_parameter_change_rate = _mean(
        tuple(turn.online_fast_substrate_parameter_change_rate for turn in turns)
    )
    mean_online_fast_substrate_optimizer_state_norm = _mean(
        tuple(turn.online_fast_substrate_optimizer_state_norm for turn in turns)
    )
    runtime_backbone_evidence_turn_count = sum(
        1 for turn in turns if turn.runtime_backbone_evidence_active
    )
    mean_runtime_backbone_signal_norm = _mean(
        tuple(turn.runtime_backbone_signal_norm for turn in turns)
    )
    mean_runtime_backbone_signal_quality = _mean(
        tuple(turn.runtime_backbone_signal_quality for turn in turns)
    )
    mean_runtime_backbone_signal_strength = _mean(
        tuple(turn.runtime_backbone_signal_strength for turn in turns)
    )
    mean_runtime_backbone_hook_coverage = _mean(
        tuple(turn.runtime_backbone_hook_coverage for turn in turns)
    )
    fast_memory_signal_turn_count = sum(
        1 for turn in turns if turn.fast_memory_signal_norm > 0.0
    )
    mean_fast_memory_signal_norm = _mean(
        tuple(turn.fast_memory_signal_norm for turn in turns)
    )
    mean_fast_memory_runtime_alignment = _mean(
        tuple(turn.fast_memory_runtime_alignment for turn in turns)
    )
    temporal_change_count = abstract_action_changes + regime_changes + int(family_version_growth > 0)
    late_episode_stability_score = _late_episode_stability_score(
        turns=turns,
        high_pe_threshold=high_pe_threshold,
        reward_threshold=reward_threshold,
    )
    delayed_improvement_observed = _delayed_improvement_observed(turns)
    acceptance_checks = (
        ("episode-runs-to-completion", final_episode_state.completed),
        ("prediction-chain-present", prediction_chain_turn_count > 0),
        ("pe-schedule-observed", pe_triggered_turn_count > 0),
        (
            "temporal-trajectory-nonconstant",
            temporal_change_count > 0 or switch_gate_span >= switch_gate_span_threshold,
        ),
        (
            "multi-timescale-evidence-observed",
            online_learning_turn_count > 0
            and (
                bounded_writeback_turn_count > 0
                or session_post_completion_turn_count > 0
                or reflection_promotion_eligible_turn_count > 0
                or online_fast_substrate_recommended_count > 0
                or rare_heavy_recommended_count > 0
            ),
        ),
        (
            "runtime-backbone-evidence-observed",
            runtime_backbone_evidence_turn_count > 0
            and mean_runtime_backbone_signal_quality >= 0.2
            and mean_fast_memory_signal_norm > 0.0,
        ),
        (
            "late-episode-stabilization-or-improvement",
            delayed_improvement_observed or late_episode_stability_score >= 0.5,
        ),
    )
    reasons = tuple(check_name for check_name, passed in acceptance_checks if not passed)
    passed = not reasons
    return OpenDialogueCaseReport(
        scenario=scenario,
        final_episode_state=final_episode_state,
        turns=turns,
        prediction_chain_turn_count=prediction_chain_turn_count,
        high_pe_turn_count=high_pe_turn_count,
        pe_schedule_due_turn_count=pe_schedule_due_turn_count,
        pe_triggered_turn_count=pe_triggered_turn_count,
        explicit_pe_schedule_turn_count=explicit_pe_schedule_turn_count,
        carryover_credit_turn_count=carryover_credit_turn_count,
        schedule_label_consistency=schedule_label_consistency,
        online_learning_turn_count=online_learning_turn_count,
        bounded_writeback_turn_count=bounded_writeback_turn_count,
        reflection_promotion_eligible_turn_count=reflection_promotion_eligible_turn_count,
        session_post_completion_turn_count=session_post_completion_turn_count,
        rare_heavy_recommended_count=rare_heavy_recommended_count,
        rare_heavy_applied_count=rare_heavy_applied_count,
        rare_heavy_pre_import_pass_count=rare_heavy_pre_import_pass_count,
        rare_heavy_pre_import_reject_count=rare_heavy_pre_import_reject_count,
        mean_rare_heavy_pre_import_score_delta=mean_rare_heavy_pre_import_score_delta,
        mean_rare_heavy_candidate_alignment=mean_rare_heavy_candidate_alignment,
        max_rare_heavy_candidate_adapter_parameter_count=max_rare_heavy_candidate_adapter_parameter_count,
        evolution_judge_turn_count=evolution_judge_turn_count,
        evolution_judge_rollback_count=evolution_judge_rollback_count,
        evolution_judge_structural_allow_count=evolution_judge_structural_allow_count,
        nested_profile_active_turn_count=nested_profile_active_turn_count,
        nested_context_reset_count=nested_context_reset_count,
        store_nested_context_reset_count=store_nested_context_reset_count,
        mean_slow_to_fast_init_benefit=mean_slow_to_fast_init_benefit,
        online_fast_substrate_recommended_count=online_fast_substrate_recommended_count,
        online_fast_substrate_applied_count=online_fast_substrate_applied_count,
        mean_online_fast_substrate_parameter_change_rate=mean_online_fast_substrate_parameter_change_rate,
        mean_online_fast_substrate_optimizer_state_norm=mean_online_fast_substrate_optimizer_state_norm,
        case_memory_surface_turn_count=case_memory_surface_turn_count,
        strategy_playbook_surface_turn_count=strategy_playbook_surface_turn_count,
        experience_fast_prior_surface_turn_count=experience_fast_prior_surface_turn_count,
        experience_consolidation_surface_turn_count=experience_consolidation_surface_turn_count,
        runtime_backbone_evidence_turn_count=runtime_backbone_evidence_turn_count,
        mean_runtime_backbone_signal_norm=mean_runtime_backbone_signal_norm,
        mean_runtime_backbone_signal_quality=mean_runtime_backbone_signal_quality,
        mean_runtime_backbone_signal_strength=mean_runtime_backbone_signal_strength,
        mean_runtime_backbone_hook_coverage=mean_runtime_backbone_hook_coverage,
        fast_memory_signal_turn_count=fast_memory_signal_turn_count,
        mean_fast_memory_signal_norm=mean_fast_memory_signal_norm,
        mean_fast_memory_runtime_alignment=mean_fast_memory_runtime_alignment,
        temporal_change_count=temporal_change_count,
        late_episode_stability_score=late_episode_stability_score,
        delayed_improvement_observed=delayed_improvement_observed,
        acceptance_checks=acceptance_checks,
        passed=passed,
        reasons=reasons,
        description=(
            f"Open dialogue scenario {scenario.scenario_id} processed {len(turns)} turns; "
            f"completed={final_episode_state.completed} stop_reason={final_episode_state.stop_reason} "
            f"prediction_chain={prediction_chain_turn_count} pe_triggered={pe_triggered_turn_count} "
            f"online_learning={online_learning_turn_count} session_post={session_post_completion_turn_count} "
            f"online_fast_substrate_recommended={online_fast_substrate_recommended_count} "
            f"online_fast_change_rate={mean_online_fast_substrate_parameter_change_rate:.3f} "
            f"online_fast_optimizer_norm={mean_online_fast_substrate_optimizer_state_norm:.3f} "
            f"runtime_quality={mean_runtime_backbone_signal_quality:.3f} "
            f"fast_memory_alignment={mean_fast_memory_runtime_alignment:.3f} "
            f"tower_depth={mean_memory_tower_depth:.2f} "
            f"tower_alignment={mean_memory_tower_alignment:.2f} "
            f"tower_consolidation_max={max_tower_consolidation_count} "
            f"late_stability={late_episode_stability_score:.2f} delayed_improvement={delayed_improvement_observed}."
        ),
        max_tower_consolidation_count=max_tower_consolidation_count,
        mean_memory_tower_depth=mean_memory_tower_depth,
        mean_memory_tower_alignment=mean_memory_tower_alignment,
        memory_tower_profile_turn_count=memory_tower_profile_turn_count,
    )


def build_dialogue_case_report(
    *,
    case: ScriptedDialogueCase,
    turns: tuple[DialogueBenchmarkTurn, ...],
    high_pe_threshold: float = PROOF_HIGH_PE_THRESHOLD,
    reward_threshold: float = PROOF_REWARD_THRESHOLD,
    switch_gate_span_threshold: float = 0.12,
    allow_interval_carryover_credit: bool = True,
) -> DialogueBenchmarkCaseReport:
    prediction_chain_turn_count = sum(1 for turn in turns if turn.has_prediction_chain)
    high_pe_turn_count = sum(
        1
        for turn in turns
        if _turn_is_high_pe(
            turn,
            high_pe_threshold=high_pe_threshold,
            reward_threshold=reward_threshold,
        )
    )
    pe_trigger_analysis = _pe_trigger_analysis(
        turns,
        high_pe_threshold=high_pe_threshold,
        reward_threshold=reward_threshold,
        allow_interval_carryover_credit=allow_interval_carryover_credit,
    )
    pe_trigger_flags = tuple(item.triggered for item in pe_trigger_analysis)
    pe_schedule_due_turn_count = sum(
        1
        for item in pe_trigger_analysis
        if item.current_turn_schedule_due or item.previous_turn_schedule_due
    )
    pe_triggered_turn_count = sum(1 for item in pe_trigger_analysis if item.triggered)
    explicit_pe_schedule_turn_count = sum(1 for item in pe_trigger_analysis if item.explicit_schedule_trigger)
    carryover_credit_turn_count = sum(1 for item in pe_trigger_analysis if item.carryover_temporal_response)
    explicit_schedule_aligned_count = sum(
        1
        for item in pe_trigger_analysis
        if item.explicit_schedule_trigger and (item.current_turn_schedule_due or item.previous_turn_schedule_due)
    )
    schedule_label_consistency = (
        explicit_schedule_aligned_count / explicit_pe_schedule_turn_count
        if explicit_pe_schedule_turn_count > 0
        else 1.0
    )
    recovery_lag_turns = _recovery_lag_turns(
        case=case,
        turns=turns,
        pe_trigger_flags=pe_trigger_flags,
    )
    pressure_localization_score = _pressure_localization_score(
        case=case,
        turns=turns,
        pe_trigger_flags=pe_trigger_flags,
    )
    pressure_response_precision = _pressure_response_precision(
        case=case,
        turns=turns,
        pe_trigger_flags=pe_trigger_flags,
    )
    pressure_response_recall = _pressure_response_recall(
        case=case,
        turns=turns,
        pe_trigger_flags=pe_trigger_flags,
    )
    over_response_cost = _over_response_cost(
        case=case,
        turns=turns,
        pe_trigger_flags=pe_trigger_flags,
    )
    stability_after_recovery_score = _stability_after_recovery_score(
        case=case,
        turns=turns,
        pe_trigger_flags=pe_trigger_flags,
        high_pe_threshold=high_pe_threshold,
        reward_threshold=reward_threshold,
    )
    abstract_action_changes = sum(
        1
        for previous, current in zip(turns, turns[1:], strict=False)
        if current.active_abstract_action != previous.active_abstract_action
    )
    regime_changes = sum(
        1
        for previous, current in zip(turns, turns[1:], strict=False)
        if current.active_regime != previous.active_regime
    )
    family_version_growth = max((turn.action_family_version for turn in turns), default=0) - min(
        (turn.action_family_version for turn in turns), default=0
    )
    switch_gate_span = 0.0
    if turns:
        switch_gate_span = max(turn.switch_gate for turn in turns) - min(turn.switch_gate for turn in turns)
    online_learning_turn_count = sum(
        1 for turn in turns if turn.joint_schedule_action != "evidence-only"
    )
    bounded_writeback_turn_count = sum(
        1 for turn in turns if turn.bounded_writeback_applied
    )
    reflection_promotion_eligible_turn_count = sum(
        1 for turn in turns if turn.reflection_promotion_eligible
    )
    session_post_completion_turn_count = sum(
        1 for turn in turns if turn.session_post_completed_job_count > 0
    )
    rare_heavy_recommended_count = sum(
        1 for turn in turns if turn.rare_heavy_recommended
    )
    rare_heavy_applied_count = sum(
        1 for turn in turns if turn.rare_heavy_applied
    )
    rare_heavy_pre_import_pass_count = sum(
        1 for turn in turns if turn.rare_heavy_pre_import_passed
    )
    rare_heavy_pre_import_reject_count = sum(
        1
        for turn in turns
        if turn.rare_heavy_import_decision == "rejected-pre-import"
    )
    mean_rare_heavy_pre_import_score_delta = _mean(
        tuple(turn.rare_heavy_pre_import_mean_score_delta for turn in turns if turn.rare_heavy_recommended)
    )
    mean_rare_heavy_candidate_alignment = _mean(
        tuple(turn.rare_heavy_candidate_alignment for turn in turns if turn.rare_heavy_recommended)
    )
    max_rare_heavy_candidate_adapter_parameter_count = max(
        (turn.rare_heavy_candidate_adapter_parameter_count for turn in turns),
        default=0,
    )
    evolution_judge_turn_count = sum(
        1 for turn in turns if turn.evolution_decision is not None
    )
    evolution_judge_rollback_count = sum(
        1 for turn in turns if turn.evolution_decision == EvolutionDecision.ROLLBACK.value
    )
    evolution_judge_structural_allow_count = sum(
        1
        for turn in turns
        if turn.evolution_decision in {EvolutionDecision.PROMOTE.value, EvolutionDecision.HOLD.value}
        and turn.evolution_category != JudgementCategory.UNSAFE_MUTATION.value
    )
    nested_profile_active_turn_count = sum(
        1 for turn in turns if turn.nested_profile_active
    )
    nested_context_reset_count = sum(
        1 for turn in turns if turn.nested_context_reset_applied
    )
    store_nested_context_reset_count = _store_nested_context_reset_count(turns)
    boundary_reset_observed_on_first_turn = bool(turns and turns[0].nested_context_reset_applied)
    first_turn_slow_to_fast_init_benefit = turns[0].slow_to_fast_init_benefit if turns else 0.0
    mean_reset_turn_slow_to_fast_init_benefit = _reset_turn_slow_to_fast_init_benefit_mean(turns)
    mean_slow_to_fast_init_benefit = _mean(
        tuple(turn.slow_to_fast_init_benefit for turn in turns)
    )
    first_turn_slow_to_fast_target_distance_before = (
        turns[0].slow_to_fast_target_distance_before if turns else 0.0
    )
    first_turn_slow_to_fast_target_distance_after = (
        turns[0].slow_to_fast_target_distance_after if turns else 0.0
    )
    first_turn_slow_to_fast_target_alignment_gain = (
        turns[0].slow_to_fast_target_alignment_gain if turns else 0.0
    )
    mean_reset_turn_slow_to_fast_target_distance_before = _reset_turn_metric_mean(
        turns,
        accessor=lambda turn: turn.slow_to_fast_target_distance_before,
    )
    mean_reset_turn_slow_to_fast_target_distance_after = _reset_turn_metric_mean(
        turns,
        accessor=lambda turn: turn.slow_to_fast_target_distance_after,
    )
    mean_reset_turn_slow_to_fast_target_alignment_gain = _reset_turn_metric_mean(
        turns,
        accessor=lambda turn: turn.slow_to_fast_target_alignment_gain,
    )
    learned_memory_primary_turn_count = sum(
        1 for turn in turns if turn.learned_memory_primary
    )
    core_guided_recall_turn_count = sum(
        1 for turn in turns if turn.learned_recall_core_guided
    )
    mean_learned_recall_confidence = _mean(
        tuple(turn.learned_recall_confidence for turn in turns)
    )
    max_artifact_consolidation_count = max(
        (turn.artifact_consolidation_count for turn in turns),
        default=0,
    )
    max_tower_consolidation_count = max(
        (turn.tower_consolidation_count for turn in turns),
        default=0,
    )
    mean_memory_tower_depth = _mean(
        tuple(float(turn.memory_tower_depth) for turn in turns)
    )
    mean_memory_tower_alignment = _mean(
        tuple(turn.memory_tower_alignment for turn in turns)
    )
    memory_tower_profile_turn_count = sum(
        1 for turn in turns if turn.memory_tower_profile_id
    )
    online_fast_substrate_recommended_count = sum(
        1 for turn in turns if turn.online_fast_substrate_recommended
    )
    online_fast_substrate_applied_count = sum(
        1 for turn in turns if turn.online_fast_substrate_applied
    )
    case_memory_surface_turn_count = sum(
        1 for turn in turns if turn.case_memory_surface_active
    )
    strategy_playbook_surface_turn_count = sum(
        1 for turn in turns if turn.strategy_playbook_surface_active
    )
    experience_fast_prior_surface_turn_count = sum(
        1 for turn in turns if turn.experience_fast_prior_surface_active
    )
    experience_consolidation_surface_turn_count = sum(
        1 for turn in turns if turn.experience_consolidation_surface_active
    )
    mean_online_fast_substrate_parameter_change_rate = _mean(
        tuple(turn.online_fast_substrate_parameter_change_rate for turn in turns)
    )
    mean_online_fast_substrate_optimizer_state_norm = _mean(
        tuple(turn.online_fast_substrate_optimizer_state_norm for turn in turns)
    )
    runtime_backbone_evidence_turn_count = sum(
        1 for turn in turns if turn.runtime_backbone_evidence_active
    )
    mean_runtime_backbone_signal_norm = _mean(
        tuple(turn.runtime_backbone_signal_norm for turn in turns)
    )
    mean_runtime_backbone_signal_quality = _mean(
        tuple(turn.runtime_backbone_signal_quality for turn in turns)
    )
    mean_runtime_backbone_signal_strength = _mean(
        tuple(turn.runtime_backbone_signal_strength for turn in turns)
    )
    mean_runtime_backbone_hook_coverage = _mean(
        tuple(turn.runtime_backbone_hook_coverage for turn in turns)
    )
    fast_memory_signal_turn_count = sum(
        1 for turn in turns if turn.fast_memory_signal_norm > 0.0
    )
    mean_fast_memory_signal_norm = _mean(
        tuple(turn.fast_memory_signal_norm for turn in turns)
    )
    mean_fast_memory_runtime_alignment = _mean(
        tuple(turn.fast_memory_runtime_alignment for turn in turns)
    )
    temporal_change_count = abstract_action_changes + regime_changes + int(family_version_growth > 0)
    delayed_improvement_observed = _delayed_improvement_observed(turns)
    acceptance_checks = (
        ("prediction_chain_present", prediction_chain_turn_count > 0),
        ("high_pe_detected", high_pe_turn_count > 0),
        ("pe_triggered_temporal_response", pe_triggered_turn_count > 0),
        (
            "temporal_trajectory_nonconstant",
            temporal_change_count > 0 or switch_gate_span >= switch_gate_span_threshold,
        ),
        ("delayed_improvement_observed", delayed_improvement_observed),
    )
    reasons = tuple(check_name for check_name, passed in acceptance_checks if not passed)
    passed = not reasons
    return DialogueBenchmarkCaseReport(
        case=case,
        turns=turns,
        prediction_chain_turn_count=prediction_chain_turn_count,
        high_pe_turn_count=high_pe_turn_count,
        pe_schedule_due_turn_count=pe_schedule_due_turn_count,
        pe_triggered_turn_count=pe_triggered_turn_count,
        explicit_pe_schedule_turn_count=explicit_pe_schedule_turn_count,
        carryover_credit_turn_count=carryover_credit_turn_count,
        schedule_label_consistency=schedule_label_consistency,
        recovery_lag_turns=recovery_lag_turns,
        pressure_localization_score=pressure_localization_score,
        over_response_cost=over_response_cost,
        pressure_response_precision=pressure_response_precision,
        pressure_response_recall=pressure_response_recall,
        stability_after_recovery_score=stability_after_recovery_score,
        online_learning_turn_count=online_learning_turn_count,
        bounded_writeback_turn_count=bounded_writeback_turn_count,
        reflection_promotion_eligible_turn_count=reflection_promotion_eligible_turn_count,
        session_post_completion_turn_count=session_post_completion_turn_count,
        rare_heavy_recommended_count=rare_heavy_recommended_count,
        rare_heavy_applied_count=rare_heavy_applied_count,
        rare_heavy_pre_import_pass_count=rare_heavy_pre_import_pass_count,
        rare_heavy_pre_import_reject_count=rare_heavy_pre_import_reject_count,
        mean_rare_heavy_pre_import_score_delta=mean_rare_heavy_pre_import_score_delta,
        mean_rare_heavy_candidate_alignment=mean_rare_heavy_candidate_alignment,
        max_rare_heavy_candidate_adapter_parameter_count=max_rare_heavy_candidate_adapter_parameter_count,
        evolution_judge_turn_count=evolution_judge_turn_count,
        evolution_judge_rollback_count=evolution_judge_rollback_count,
        evolution_judge_structural_allow_count=evolution_judge_structural_allow_count,
        nested_profile_active_turn_count=nested_profile_active_turn_count,
        nested_context_reset_count=nested_context_reset_count,
        store_nested_context_reset_count=store_nested_context_reset_count,
        boundary_reset_observed_on_first_turn=boundary_reset_observed_on_first_turn,
        first_turn_slow_to_fast_init_benefit=first_turn_slow_to_fast_init_benefit,
        mean_reset_turn_slow_to_fast_init_benefit=mean_reset_turn_slow_to_fast_init_benefit,
        mean_slow_to_fast_init_benefit=mean_slow_to_fast_init_benefit,
        first_turn_slow_to_fast_target_distance_before=first_turn_slow_to_fast_target_distance_before,
        first_turn_slow_to_fast_target_distance_after=first_turn_slow_to_fast_target_distance_after,
        first_turn_slow_to_fast_target_alignment_gain=first_turn_slow_to_fast_target_alignment_gain,
        mean_reset_turn_slow_to_fast_target_distance_before=mean_reset_turn_slow_to_fast_target_distance_before,
        mean_reset_turn_slow_to_fast_target_distance_after=mean_reset_turn_slow_to_fast_target_distance_after,
        mean_reset_turn_slow_to_fast_target_alignment_gain=mean_reset_turn_slow_to_fast_target_alignment_gain,
        online_fast_substrate_recommended_count=online_fast_substrate_recommended_count,
        online_fast_substrate_applied_count=online_fast_substrate_applied_count,
        mean_online_fast_substrate_parameter_change_rate=mean_online_fast_substrate_parameter_change_rate,
        mean_online_fast_substrate_optimizer_state_norm=mean_online_fast_substrate_optimizer_state_norm,
        case_memory_surface_turn_count=case_memory_surface_turn_count,
        strategy_playbook_surface_turn_count=strategy_playbook_surface_turn_count,
        experience_fast_prior_surface_turn_count=experience_fast_prior_surface_turn_count,
        experience_consolidation_surface_turn_count=experience_consolidation_surface_turn_count,
        runtime_backbone_evidence_turn_count=runtime_backbone_evidence_turn_count,
        mean_runtime_backbone_signal_norm=mean_runtime_backbone_signal_norm,
        mean_runtime_backbone_signal_quality=mean_runtime_backbone_signal_quality,
        mean_runtime_backbone_signal_strength=mean_runtime_backbone_signal_strength,
        mean_runtime_backbone_hook_coverage=mean_runtime_backbone_hook_coverage,
        fast_memory_signal_turn_count=fast_memory_signal_turn_count,
        mean_fast_memory_signal_norm=mean_fast_memory_signal_norm,
        mean_fast_memory_runtime_alignment=mean_fast_memory_runtime_alignment,
        temporal_change_count=temporal_change_count,
        delayed_improvement_observed=delayed_improvement_observed,
        acceptance_checks=acceptance_checks,
        passed=passed,
        reasons=reasons,
        description=(
            f"Dialogue proof case {case.case_id} processed {len(turns)} turns with "
            f"prediction_chain={prediction_chain_turn_count}, high_pe={high_pe_turn_count}, "
            f"pe_triggered={pe_triggered_turn_count}, recovery_lag={recovery_lag_turns}, "
            f"pressure_localization={pressure_localization_score:.2f}, precision={pressure_response_precision:.2f}, "
            f"recall={pressure_response_recall:.2f}, over_response={over_response_cost:.2f}, "
            f"stability_after_recovery={stability_after_recovery_score:.2f}, online_learning={online_learning_turn_count}, "
            f"writeback_turns={bounded_writeback_turn_count}, session_post_turns={session_post_completion_turn_count}, "
            f"rare_heavy_applied={rare_heavy_applied_count}, "
            f"rare_heavy_pre_import_pass={rare_heavy_pre_import_pass_count}, "
            f"rare_heavy_pre_import_reject={rare_heavy_pre_import_reject_count}, "
            f"rare_heavy_pre_import_delta={mean_rare_heavy_pre_import_score_delta:.3f}, "
            f"judge_turns={evolution_judge_turn_count}, nested_resets={nested_context_reset_count}, "
            f"store_nested_resets={store_nested_context_reset_count}, "
            f"online_fast_substrate_recommended={online_fast_substrate_recommended_count}, "
            f"online_fast_change_rate={mean_online_fast_substrate_parameter_change_rate:.3f}, "
            f"online_fast_optimizer_norm={mean_online_fast_substrate_optimizer_state_norm:.3f}, "
            f"runtime_quality={mean_runtime_backbone_signal_quality:.3f}, "
            f"fast_memory_alignment={mean_fast_memory_runtime_alignment:.3f}, "
            f"slow_to_fast_init_benefit={mean_slow_to_fast_init_benefit:.2f}, "
            f"reset_turn_benefit={mean_reset_turn_slow_to_fast_init_benefit:.2f}, "
            f"target_alignment_gain={mean_reset_turn_slow_to_fast_target_alignment_gain:.3f}, "
            f"target_distance_after={mean_reset_turn_slow_to_fast_target_distance_after:.3f}, "
            f"learned_primary_turns={learned_memory_primary_turn_count}, "
            f"core_guided_recall_turns={core_guided_recall_turn_count}, "
            f"mean_recall_confidence={mean_learned_recall_confidence:.2f}, "
            f"tower_depth={mean_memory_tower_depth:.2f}, "
            f"tower_alignment={mean_memory_tower_alignment:.2f}, "
            f"tower_consolidation_max={max_tower_consolidation_count}, "
            f"temporal_changes={temporal_change_count}, "
            f"switch_gate_span={switch_gate_span:.2f}, delayed_improvement={delayed_improvement_observed}."
        ),
        learned_memory_primary_turn_count=learned_memory_primary_turn_count,
        core_guided_recall_turn_count=core_guided_recall_turn_count,
        mean_learned_recall_confidence=mean_learned_recall_confidence,
        max_artifact_consolidation_count=max_artifact_consolidation_count,
        max_tower_consolidation_count=max_tower_consolidation_count,
        mean_memory_tower_depth=mean_memory_tower_depth,
        mean_memory_tower_alignment=mean_memory_tower_alignment,
        memory_tower_profile_turn_count=memory_tower_profile_turn_count,
    )


def dialogue_turn_from_result(*, turn_index: int, user_input: str, result: AgentTurnResult) -> DialogueBenchmarkTurn:
    temporal_snapshot = result.active_snapshots.get("temporal_abstraction")
    switch_gate = 0.0
    action_family_version = 0
    if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
        switch_gate = temporal_snapshot.value.controller_state.switch_gate
        action_family_version = temporal_snapshot.value.action_family_version
    prediction_error = result.prediction_error
    evolution_decision = (
        result.evolution_judgement.decision.value
        if result.evolution_judgement is not None
        else None
    )
    evolution_category = (
        result.evolution_judgement.category.value
        if result.evolution_judgement is not None
        else None
    )
    active_slots = result.active_snapshots
    return DialogueBenchmarkTurn(
        turn_index=turn_index,
        wave_id=result.wave_id,
        user_input=user_input,
        assistant_response_text=result.response.text,
        acceptance_passed=result.acceptance_passed,
        active_regime=result.active_regime,
        active_abstract_action=result.active_abstract_action,
        joint_schedule_action=result.joint_schedule_action,
        switch_gate=switch_gate,
        action_family_version=action_family_version,
        prediction_error_magnitude=prediction_error.magnitude if prediction_error is not None else 0.0,
        prediction_error_reward=prediction_error.signed_reward if prediction_error is not None else 0.0,
        task_error=prediction_error.task_error if prediction_error is not None else 0.0,
        relationship_error=prediction_error.relationship_error if prediction_error is not None else 0.0,
        regime_error=prediction_error.regime_error if prediction_error is not None else 0.0,
        action_error=prediction_error.action_error if prediction_error is not None else 0.0,
        has_prediction_chain=(
            result.actual_outcome is not None
            and result.next_prediction is not None
            and result.prediction_error is not None
        ),
        bounded_writeback_applied=result.bounded_writeback_applied,
        reflection_promotion_eligible=result.reflection_promotion_eligible,
        session_post_completed_job_count=result.session_post_completed_job_count,
        rare_heavy_recommended=bool(result.rare_heavy_result is not None and result.rare_heavy_result.recommended),
        rare_heavy_applied=bool(result.rare_heavy_result is not None and result.rare_heavy_result.applied),
        evolution_decision=evolution_decision,
        evolution_category=evolution_category,
        cross_session_verdict=result.cross_session_verdict,
        nested_profile_active=result.nested_profile_active,
        nested_context_reset_applied=result.nested_context_reset_applied,
        nested_context_reset_total_count=result.nested_context_reset_total_count,
        slow_to_fast_init_benefit=result.slow_to_fast_init_benefit,
        slow_to_fast_target_distance_before=result.slow_to_fast_target_distance_before,
        slow_to_fast_target_distance_after=result.slow_to_fast_target_distance_after,
        slow_to_fast_target_alignment_gain=result.slow_to_fast_target_alignment_gain,
        outcome_metrics=_metric_pairs(result),
        description=(
            f"Turn {turn_index} action={result.joint_schedule_action}, regime={result.active_regime}, "
            f"abstract_action={result.active_abstract_action}, pe={prediction_error.magnitude if prediction_error is not None else 0.0:.2f}, "
            f"judge={evolution_decision or 'none'}, cross_session={result.cross_session_verdict or 'none'}."
        ),
        rare_heavy_import_decision=(
            result.rare_heavy_result.import_decision
            if result.rare_heavy_result is not None
            else ""
        ),
        rare_heavy_reject_reason=(
            result.rare_heavy_result.reject_reason
            if result.rare_heavy_result is not None
            else ""
        ),
        rare_heavy_pre_import_passed=(
            result.rare_heavy_result.pre_import_passed
            if result.rare_heavy_result is not None
            else False
        ),
        rare_heavy_pre_import_mean_score_delta=(
            result.rare_heavy_result.pre_import_mean_score_delta
            if result.rare_heavy_result is not None
            else 0.0
        ),
        rare_heavy_candidate_alignment=(
            result.rare_heavy_result.bundle_alignment_ratio
            if result.rare_heavy_result is not None
            else 0.0
        ),
        rare_heavy_candidate_adapter_parameter_count=(
            result.rare_heavy_result.candidate_adapter_parameter_count
            if result.rare_heavy_result is not None
            else 0
        ),
        learned_memory_primary=result.learned_memory_primary,
        artifact_consolidation_count=result.artifact_consolidation_count,
        tower_consolidation_count=result.tower_consolidation_count,
        learned_recall_count=result.learned_recall_count,
        learned_recall_confidence=result.learned_recall_confidence,
        learned_recall_core_guided=result.learned_recall_core_guided,
        memory_tower_depth=result.memory_tower_depth,
        memory_tower_alignment=result.memory_tower_alignment,
        memory_tower_profile_id=result.memory_tower_profile_id,
        online_fast_substrate_recommended=bool(
            result.online_fast_substrate_result is not None and result.online_fast_substrate_result.recommended
        ),
        online_fast_substrate_applied=bool(
            result.online_fast_substrate_result is not None and result.online_fast_substrate_result.applied
        ),
        online_fast_substrate_parameter_change_rate=(
            result.online_fast_substrate_result.parameter_change_rate
            if result.online_fast_substrate_result is not None
            else 0.0
        ),
        online_fast_substrate_optimizer_state_norm=(
            result.online_fast_substrate_result.optimizer_state_norm
            if result.online_fast_substrate_result is not None
            else 0.0
        ),
        case_memory_surface_active="case_memory" in active_slots,
        strategy_playbook_surface_active="strategy_playbook" in active_slots,
        experience_fast_prior_surface_active="experience_fast_prior" in active_slots,
        experience_consolidation_surface_active="experience_consolidation" in active_slots,
        runtime_backbone_evidence_active=result.runtime_backbone_evidence_active,
        runtime_backbone_signal_norm=result.runtime_backbone_signal_norm,
        runtime_backbone_signal_quality=result.runtime_backbone_signal_quality,
        runtime_backbone_signal_strength=result.runtime_backbone_signal_strength,
        runtime_backbone_hook_coverage=result.runtime_backbone_hook_coverage,
        fast_memory_signal_norm=result.fast_memory_signal_norm,
        fast_memory_runtime_alignment=result.fast_memory_runtime_alignment,
    )


async def _settle_case_session_post_slow_loop(
    *,
    runner: AgentSessionRunner,
    turns: list[DialogueBenchmarkTurn],
    reason: str,
) -> list[DialogueBenchmarkTurn]:
    if not turns:
        return turns
    runner.begin_new_context(reason=reason)
    completed_results = await runner.drain_session_post_slow_loop()
    if not completed_results:
        return turns
    memory_snapshot = runner.memory_store.snapshot(retrieved_entries=())
    lifecycle_metrics = dict(memory_snapshot.lifecycle_metrics)
    tower_profile_id = turns[-1].memory_tower_profile_id
    if memory_snapshot.cms_state is not None and memory_snapshot.cms_state.tower_profile is not None:
        tower_profile_id = memory_snapshot.cms_state.tower_profile.profile_id
    settled_last_turn = replace(
        turns[-1],
        bounded_writeback_applied=(
            turns[-1].bounded_writeback_applied or any(result.applied for result in completed_results)
        ),
        session_post_completed_job_count=max(
            turns[-1].session_post_completed_job_count,
            runner.session_post_queue_state.completed_job_count,
        ),
        learned_memory_primary=lifecycle_metrics.get("learned_memory_primary", 0.0) > 0.0,
        artifact_consolidation_count=int(
            lifecycle_metrics.get("artifact_consolidation_count", turns[-1].artifact_consolidation_count)
        ),
        tower_consolidation_count=int(
            lifecycle_metrics.get("tower_consolidation_count", turns[-1].tower_consolidation_count)
        ),
        learned_recall_count=int(
            lifecycle_metrics.get("learned_recall_count", turns[-1].learned_recall_count)
        ),
        learned_recall_confidence=lifecycle_metrics.get(
            "last_learned_recall_confidence",
            turns[-1].learned_recall_confidence,
        ),
        learned_recall_core_guided=(
            lifecycle_metrics.get("last_learned_recall_driver_is_core", 0.0) > 0.0
        ),
        memory_tower_depth=int(
            lifecycle_metrics.get("last_memory_tower_depth", float(turns[-1].memory_tower_depth))
        ),
        memory_tower_alignment=lifecycle_metrics.get(
            "last_memory_tower_alignment",
            turns[-1].memory_tower_alignment,
        ),
        memory_tower_profile_id=tower_profile_id,
        experience_consolidation_surface_active=(
            turns[-1].experience_consolidation_surface_active
            or runner.experience_consolidation_snapshot is not None
        ),
        description=(
            f"{turns[-1].description} "
            f"Session-post slow loop settled with {len(completed_results)} completed jobs."
        ),
    )
    return [*turns[:-1], settled_last_turn]


async def run_open_dialogue_case(
    *,
    scenario: OpenDialogueScenario,
    runner: AgentSessionRunner,
    turn_source: DialogueUserTurnSource | None = None,
    seed: int = 0,
    max_turns: int | None = None,
    allow_interval_carryover_credit: bool = True,
) -> OpenDialogueCaseReport:
    source = turn_source or build_deterministic_user_simulator(
        scenario_id=scenario.scenario_id,
        seed=seed,
        max_turns=max_turns,
    )
    turns: list[DialogueBenchmarkTurn] = []
    last_result: AgentTurnResult | None = None
    last_turn: DialogueBenchmarkTurn | None = None
    while True:
        user_input = source.next_turn(last_result=last_result, last_turn=last_turn)
        if user_input is None:
            break
        turn_index = len(turns) + 1
        result = await runner.run_turn(user_input)
        last_result = result
        last_turn = dialogue_turn_from_result(
            turn_index=turn_index,
            user_input=user_input,
            result=result,
        )
        turns.append(last_turn)
    turns = await _settle_case_session_post_slow_loop(
        runner=runner,
        turns=turns,
        reason=f"open-dialogue-case-complete:{source.scenario.scenario_id}",
    )
    return build_open_dialogue_case_report(
        scenario=source.scenario,
        final_episode_state=source.episode_state,
        turns=tuple(turns),
        allow_interval_carryover_credit=allow_interval_carryover_credit,
    )


async def run_dialogue_pe_eta_case(
    *,
    case: ScriptedDialogueCase,
    runner: AgentSessionRunner,
    allow_interval_carryover_credit: bool = True,
) -> DialogueBenchmarkCaseReport:
    turns: list[DialogueBenchmarkTurn] = []
    for turn_index, user_input in enumerate(case.user_inputs, start=1):
        result = await runner.run_turn(user_input)
        turns.append(
            dialogue_turn_from_result(
                turn_index=turn_index,
                user_input=user_input,
                result=result,
            )
        )
    turns = await _settle_case_session_post_slow_loop(
        runner=runner,
        turns=turns,
        reason=f"dialogue-case-complete:{case.case_id}",
    )
    return build_dialogue_case_report(
        case=case,
        turns=tuple(turns),
        allow_interval_carryover_credit=allow_interval_carryover_credit,
    )


def _open_case_summary_metrics(report: OpenDialogueCaseReport) -> tuple[tuple[str, float], ...]:
    mean_pe = _mean(tuple(turn.prediction_error_magnitude for turn in report.turns))
    mean_switch_gate = _mean(tuple(turn.switch_gate for turn in report.turns))
    mean_substrate_online_fast_applied = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "substrate_online_fast_applied")) is not None
        )
    )
    mean_substrate_online_fast_experimental_mode = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "substrate_online_fast_experimental_mode")) is not None
        )
    )
    mean_substrate_online_fast_review_or_revert_safe = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "substrate_online_fast_review_or_revert_safe")) is not None
        )
    )
    mean_substrate_online_fast_rollback_integrity = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "substrate_online_fast_rollback_integrity")) is not None
        )
    )
    mean_memory_updater_effective_lr = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "memory_updater_effective_lr")) is not None
        )
    )
    mean_memory_updater_confidence = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "memory_updater_confidence")) is not None
        )
    )
    mean_temporal_updater_effective_lr = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "temporal_updater_effective_lr")) is not None
        )
    )
    mean_temporal_updater_confidence = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "temporal_updater_confidence")) is not None
        )
    )
    mean_optimizer_memory_drive = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "optimizer_memory_drive")) is not None
        )
    )
    mean_timescale_contract_retained = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "timescale_contract_retained")) is not None
        )
    )
    mean_scheduler_discipline = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "scheduler_discipline")) is not None
        )
    )
    mean_scheduler_substrate_pressure = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "scheduler_substrate_pressure")) is not None
        )
    )
    mean_scheduler_rare_heavy_pressure = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "scheduler_rare_heavy_pressure")) is not None
        )
    )
    mean_default_continual_learning_active = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "default_continual_learning_active")) is not None
        )
    )
    mean_default_owner_writeback_retained = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "default_owner_writeback_retained")) is not None
        )
    )
    mean_default_substrate_live_mutation_suppressed = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "default_substrate_live_mutation_suppressed")) is not None
        )
    )
    mean_default_continual_rollback_clean = _mean(
        tuple(
            value
            for turn in report.turns
            if (value := _metric_value(turn, "default_continual_rollback_clean")) is not None
        )
    )
    return (
        ("passed", float(report.passed)),
        ("prediction_chain_turn_count", float(report.prediction_chain_turn_count)),
        ("high_pe_turn_count", float(report.high_pe_turn_count)),
        ("pe_schedule_due_turn_count", float(report.pe_schedule_due_turn_count)),
        ("pe_triggered_turn_count", float(report.pe_triggered_turn_count)),
        ("explicit_pe_schedule_turn_count", float(report.explicit_pe_schedule_turn_count)),
        ("carryover_credit_turn_count", float(report.carryover_credit_turn_count)),
        ("schedule_label_consistency", report.schedule_label_consistency),
        ("online_learning_turn_count", float(report.online_learning_turn_count)),
        ("bounded_writeback_turn_count", float(report.bounded_writeback_turn_count)),
        ("reflection_promotion_eligible_turn_count", float(report.reflection_promotion_eligible_turn_count)),
        ("session_post_completion_turn_count", float(report.session_post_completion_turn_count)),
        ("online_fast_substrate_recommended_count", float(report.online_fast_substrate_recommended_count)),
        ("online_fast_substrate_applied_count", float(report.online_fast_substrate_applied_count)),
        ("mean_online_fast_substrate_parameter_change_rate", report.mean_online_fast_substrate_parameter_change_rate),
        ("mean_online_fast_substrate_optimizer_state_norm", report.mean_online_fast_substrate_optimizer_state_norm),
        ("runtime_backbone_evidence_turn_count", float(report.runtime_backbone_evidence_turn_count)),
        ("mean_runtime_backbone_signal_norm", report.mean_runtime_backbone_signal_norm),
        ("mean_runtime_backbone_signal_quality", report.mean_runtime_backbone_signal_quality),
        ("mean_runtime_backbone_signal_strength", report.mean_runtime_backbone_signal_strength),
        ("mean_runtime_backbone_hook_coverage", report.mean_runtime_backbone_hook_coverage),
        ("fast_memory_signal_turn_count", float(report.fast_memory_signal_turn_count)),
        ("mean_fast_memory_signal_norm", report.mean_fast_memory_signal_norm),
        ("mean_fast_memory_runtime_alignment", report.mean_fast_memory_runtime_alignment),
        ("mean_substrate_online_fast_applied", mean_substrate_online_fast_applied),
        ("mean_substrate_online_fast_experimental_mode", mean_substrate_online_fast_experimental_mode),
        (
            "mean_substrate_online_fast_review_or_revert_safe",
            mean_substrate_online_fast_review_or_revert_safe,
        ),
        ("mean_substrate_online_fast_rollback_integrity", mean_substrate_online_fast_rollback_integrity),
        ("mean_memory_updater_effective_lr", mean_memory_updater_effective_lr),
        ("mean_memory_updater_confidence", mean_memory_updater_confidence),
        ("mean_temporal_updater_effective_lr", mean_temporal_updater_effective_lr),
        ("mean_temporal_updater_confidence", mean_temporal_updater_confidence),
        ("mean_optimizer_memory_drive", mean_optimizer_memory_drive),
        ("mean_timescale_contract_retained", mean_timescale_contract_retained),
        ("mean_scheduler_discipline", mean_scheduler_discipline),
        ("mean_scheduler_substrate_pressure", mean_scheduler_substrate_pressure),
        ("mean_scheduler_rare_heavy_pressure", mean_scheduler_rare_heavy_pressure),
        ("mean_default_continual_learning_active", mean_default_continual_learning_active),
        ("mean_default_owner_writeback_retained", mean_default_owner_writeback_retained),
        (
            "mean_default_substrate_live_mutation_suppressed",
            mean_default_substrate_live_mutation_suppressed,
        ),
        ("mean_default_continual_rollback_clean", mean_default_continual_rollback_clean),
        ("case_memory_surface_turn_count", float(report.case_memory_surface_turn_count)),
        ("strategy_playbook_surface_turn_count", float(report.strategy_playbook_surface_turn_count)),
        ("experience_fast_prior_surface_turn_count", float(report.experience_fast_prior_surface_turn_count)),
        ("experience_consolidation_surface_turn_count", float(report.experience_consolidation_surface_turn_count)),
        ("rare_heavy_recommended_count", float(report.rare_heavy_recommended_count)),
        ("rare_heavy_applied_count", float(report.rare_heavy_applied_count)),
        ("rare_heavy_pre_import_pass_count", float(report.rare_heavy_pre_import_pass_count)),
        ("rare_heavy_pre_import_reject_count", float(report.rare_heavy_pre_import_reject_count)),
        ("mean_rare_heavy_pre_import_score_delta", report.mean_rare_heavy_pre_import_score_delta),
        ("mean_rare_heavy_candidate_alignment", report.mean_rare_heavy_candidate_alignment),
        (
            "max_rare_heavy_candidate_adapter_parameter_count",
            float(report.max_rare_heavy_candidate_adapter_parameter_count),
        ),
        ("evolution_judge_turn_count", float(report.evolution_judge_turn_count)),
        ("evolution_judge_rollback_count", float(report.evolution_judge_rollback_count)),
        ("evolution_judge_structural_allow_count", float(report.evolution_judge_structural_allow_count)),
        ("nested_profile_active_turn_count", float(report.nested_profile_active_turn_count)),
        ("nested_context_reset_count", float(report.nested_context_reset_count)),
        ("store_nested_context_reset_count", float(report.store_nested_context_reset_count)),
        ("mean_slow_to_fast_init_benefit", report.mean_slow_to_fast_init_benefit),
        ("max_tower_consolidation_count", float(report.max_tower_consolidation_count)),
        ("mean_memory_tower_depth", report.mean_memory_tower_depth),
        ("mean_memory_tower_alignment", report.mean_memory_tower_alignment),
        ("memory_tower_profile_turn_count", float(report.memory_tower_profile_turn_count)),
        ("temporal_change_count", float(report.temporal_change_count)),
        ("late_episode_stability_score", report.late_episode_stability_score),
        ("delayed_improvement_observed", float(report.delayed_improvement_observed)),
        ("episode_runs_to_completion", float(report.final_episode_state.completed)),
        ("mean_prediction_error", mean_pe),
        ("mean_switch_gate", mean_switch_gate),
    )


def _mean_open_summary_metrics(
    reports: tuple[OpenDialogueCaseReport, ...],
) -> tuple[tuple[str, float], ...]:
    if not reports:
        return ()
    metric_names = tuple(key for key, _ in _open_case_summary_metrics(reports[0]))
    metric_totals = {key: 0.0 for key in metric_names}
    for report in reports:
        for key, value in _open_case_summary_metrics(report):
            metric_totals[key] += value
    return tuple(
        (key, round(metric_totals[key] / len(reports), 4))
        for key in metric_names
    )


async def run_open_dialogue_benchmark(
    *,
    scenarios: tuple[OpenDialogueScenario, ...] = DEFAULT_OPEN_DIALOGUE_SCENARIOS,
    runner_factory: Callable[[OpenDialogueScenario], AgentSessionRunner] | None = None,
    seed: int = 0,
    allow_interval_carryover_credit: bool = True,
) -> OpenDialogueBenchmarkReport:
    factory = runner_factory or (lambda scenario: default_active_runner())
    case_reports: list[OpenDialogueCaseReport] = []
    for scenario_index, scenario in enumerate(scenarios):
        runner = factory(scenario)
        case_reports.append(
            await run_open_dialogue_case(
                scenario=scenario,
                runner=runner,
                seed=seed + scenario_index,
                allow_interval_carryover_credit=allow_interval_carryover_credit,
            )
        )
    passed_case_count = sum(1 for report in case_reports if report.passed)
    return OpenDialogueBenchmarkReport(
        case_reports=tuple(case_reports),
        passed_case_count=passed_case_count,
        total_case_count=len(case_reports),
        metric_means=_mean_open_summary_metrics(tuple(case_reports)),
        description=(
            f"Open dialogue benchmark processed {len(case_reports)} scenarios with "
            f"{passed_case_count} passing the current open-environment acceptance surface."
        ),
    )


async def run_open_dialogue_ablation_benchmark(
    *,
    scenarios: tuple[OpenDialogueScenario, ...] = DEFAULT_OPEN_DIALOGUE_SCENARIOS,
    profile_labels: tuple[str, ...] = default_open_dialogue_ablation_profiles(),
    baseline_label: str = "pe-eta",
    runner_factory: Callable[[str, OpenDialogueScenario], AgentSessionRunner] | None = None,
    seed: int = 0,
) -> OpenDialogueBenchmarkComparisonReport:
    factory = runner_factory or (
        lambda profile_label, scenario: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=ScriptedDialogueCase(
                case_id=f"open:{scenario.scenario_id}",
                description=scenario.description,
                user_inputs=scenario.opening_turns,
            ),
        )
    )
    path_reports: list[OpenDialogueBenchmarkPathReport] = []
    for profile_label in profile_labels:
        report = await run_open_dialogue_benchmark(
            scenarios=scenarios,
            runner_factory=lambda scenario, _profile_label=profile_label: factory(_profile_label, scenario),
            seed=seed,
            allow_interval_carryover_credit=_profile_allows_interval_carryover_credit(profile_label),
        )
        path_reports.append(
            OpenDialogueBenchmarkPathReport(
                path_label=profile_label,
                benchmark_report=report,
                description=(
                    f"Open dialogue benchmark path {profile_label} completed "
                    f"{report.total_case_count} scenarios."
                ),
            )
        )
    baseline_path = next((path for path in path_reports if path.path_label == baseline_label), None)
    if baseline_path is None:
        raise ValueError(f"Baseline label {baseline_label!r} not present in profile_labels={profile_labels!r}")
    baseline_reports = {
        case_report.scenario.scenario_id: case_report
        for case_report in baseline_path.benchmark_report.case_reports
    }
    case_deltas_from_baseline: list[tuple[str, tuple[tuple[str, tuple[tuple[str, float], ...]], ...]]] = []
    metric_deltas_by_profile = {path.path_label: [] for path in path_reports}
    for scenario_id, baseline_case in baseline_reports.items():
        path_deltas: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        baseline_metrics = dict(_open_case_summary_metrics(baseline_case))
        for path in path_reports:
            path_case = next(
                report for report in path.benchmark_report.case_reports if report.scenario.scenario_id == scenario_id
            )
            current_metrics = dict(_open_case_summary_metrics(path_case))
            metric_delta = _metric_delta_items(
                current_metrics=current_metrics,
                reference_metrics=baseline_metrics,
            )
            metric_deltas_by_profile[path.path_label].append(dict(metric_delta))
            path_deltas.append((path.path_label, metric_delta))
        case_deltas_from_baseline.append((scenario_id, tuple(path_deltas)))
    metric_deltas_from_baseline: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for path in path_reports:
        per_case_deltas = metric_deltas_by_profile[path.path_label]
        if not per_case_deltas:
            metric_deltas_from_baseline.append((path.path_label, ()))
            continue
        metric_names = tuple(per_case_deltas[0].keys())
        metric_deltas_from_baseline.append(
            (
                path.path_label,
                tuple(
                    (
                        key,
                        round(
                            sum(case_delta[key] for case_delta in per_case_deltas) / len(per_case_deltas),
                            4,
                        ),
                    )
                    for key in metric_names
                ),
            )
        )
    return OpenDialogueBenchmarkComparisonReport(
        baseline_label=baseline_label,
        path_reports=tuple(path_reports),
        case_deltas_from_baseline=tuple(case_deltas_from_baseline),
        metric_deltas_from_baseline=tuple(metric_deltas_from_baseline),
        description=(
            f"Open dialogue ablation benchmark compared {len(path_reports)} paths across "
            f"{len(scenarios)} scenarios with baseline={baseline_label}."
        ),
    )


async def run_dialogue_pe_eta_benchmark(
    *,
    cases: tuple[ScriptedDialogueCase, ...] = DEFAULT_DIALOGUE_PROOF_CASES,
    runner_factory: Callable[[ScriptedDialogueCase], AgentSessionRunner] | None = None,
    allow_interval_carryover_credit: bool = True,
) -> DialogueBenchmarkReport:
    factory = runner_factory or (lambda case: default_active_runner())
    case_reports: list[DialogueBenchmarkCaseReport] = []
    for case in cases:
        runner = factory(case)
        case_reports.append(
            await run_dialogue_pe_eta_case(
                case=case,
                runner=runner,
                allow_interval_carryover_credit=allow_interval_carryover_credit,
            )
        )
    passed_case_count = sum(1 for report in case_reports if report.passed)
    return DialogueBenchmarkReport(
        case_reports=tuple(case_reports),
        passed_case_count=passed_case_count,
        total_case_count=len(case_reports),
        metric_means=_mean_summary_metrics(tuple(case_reports)),
        description=(
            f"Dialogue proof benchmark processed {len(case_reports)} scripted cases with "
            f"{passed_case_count} passing the current PE-ETA evidence gate."
        ),
    )


def _case_summary_metrics(report: DialogueBenchmarkCaseReport) -> tuple[tuple[str, float], ...]:
    turns = report.turns
    mean_pe = _mean(tuple(turn.prediction_error_magnitude for turn in turns))
    mean_switch_gate = _mean(tuple(turn.switch_gate for turn in turns))
    mean_substrate_online_fast_applied = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "substrate_online_fast_applied")) is not None
        )
    )
    mean_substrate_online_fast_experimental_mode = _mean(
        tuple(
            value
            for turn in turns
            if (value := _metric_value(turn, "substrate_online_fast_experimental_mode")) is not None
        )
    )
    mean_substrate_online_fast_review_or_revert_safe = _mean(
        tuple(
            value
            for turn in turns
            if (value := _metric_value(turn, "substrate_online_fast_review_or_revert_safe")) is not None
        )
    )
    mean_substrate_online_fast_rollback_integrity = _mean(
        tuple(
            value
            for turn in turns
            if (value := _metric_value(turn, "substrate_online_fast_rollback_integrity")) is not None
        )
    )
    mean_memory_updater_effective_lr = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "memory_updater_effective_lr")) is not None
        )
    )
    mean_memory_updater_confidence = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "memory_updater_confidence")) is not None
        )
    )
    mean_temporal_updater_effective_lr = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "temporal_updater_effective_lr")) is not None
        )
    )
    mean_temporal_updater_confidence = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "temporal_updater_confidence")) is not None
        )
    )
    mean_optimizer_memory_drive = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "optimizer_memory_drive")) is not None
        )
    )
    mean_timescale_contract_retained = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "timescale_contract_retained")) is not None
        )
    )
    mean_scheduler_discipline = _mean(
        tuple(value for turn in turns if (value := _metric_value(turn, "scheduler_discipline")) is not None)
    )
    mean_scheduler_substrate_pressure = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "scheduler_substrate_pressure")) is not None
        )
    )
    mean_scheduler_rare_heavy_pressure = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "scheduler_rare_heavy_pressure")) is not None
        )
    )
    mean_default_continual_learning_active = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "default_continual_learning_active")) is not None
        )
    )
    mean_default_owner_writeback_retained = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "default_owner_writeback_retained")) is not None
        )
    )
    mean_default_substrate_live_mutation_suppressed = _mean(
        tuple(
            value
            for turn in turns
            if (value := _metric_value(turn, "default_substrate_live_mutation_suppressed")) is not None
        )
    )
    mean_default_continual_rollback_clean = _mean(
        tuple(
            value for turn in turns if (value := _metric_value(turn, "default_continual_rollback_clean")) is not None
        )
    )
    return (
        ("passed", float(report.passed)),
        ("prediction_chain_turn_count", float(report.prediction_chain_turn_count)),
        ("high_pe_turn_count", float(report.high_pe_turn_count)),
        ("pe_schedule_due_turn_count", float(report.pe_schedule_due_turn_count)),
        ("pe_triggered_turn_count", float(report.pe_triggered_turn_count)),
        ("explicit_pe_schedule_turn_count", float(report.explicit_pe_schedule_turn_count)),
        ("carryover_credit_turn_count", float(report.carryover_credit_turn_count)),
        ("schedule_label_consistency", report.schedule_label_consistency),
        ("recovery_lag_turns", float(report.recovery_lag_turns)),
        ("pressure_localization_score", report.pressure_localization_score),
        ("over_response_cost", report.over_response_cost),
        ("pressure_response_precision", report.pressure_response_precision),
        ("pressure_response_recall", report.pressure_response_recall),
        ("stability_after_recovery_score", report.stability_after_recovery_score),
        ("online_learning_turn_count", float(report.online_learning_turn_count)),
        ("bounded_writeback_turn_count", float(report.bounded_writeback_turn_count)),
        ("reflection_promotion_eligible_turn_count", float(report.reflection_promotion_eligible_turn_count)),
        ("session_post_completion_turn_count", float(report.session_post_completion_turn_count)),
        ("online_fast_substrate_recommended_count", float(report.online_fast_substrate_recommended_count)),
        ("online_fast_substrate_applied_count", float(report.online_fast_substrate_applied_count)),
        ("mean_online_fast_substrate_parameter_change_rate", report.mean_online_fast_substrate_parameter_change_rate),
        ("mean_online_fast_substrate_optimizer_state_norm", report.mean_online_fast_substrate_optimizer_state_norm),
        ("runtime_backbone_evidence_turn_count", float(report.runtime_backbone_evidence_turn_count)),
        ("mean_runtime_backbone_signal_norm", report.mean_runtime_backbone_signal_norm),
        ("mean_runtime_backbone_signal_quality", report.mean_runtime_backbone_signal_quality),
        ("mean_runtime_backbone_signal_strength", report.mean_runtime_backbone_signal_strength),
        ("mean_runtime_backbone_hook_coverage", report.mean_runtime_backbone_hook_coverage),
        ("fast_memory_signal_turn_count", float(report.fast_memory_signal_turn_count)),
        ("mean_fast_memory_signal_norm", report.mean_fast_memory_signal_norm),
        ("mean_fast_memory_runtime_alignment", report.mean_fast_memory_runtime_alignment),
        ("mean_substrate_online_fast_applied", mean_substrate_online_fast_applied),
        ("mean_substrate_online_fast_experimental_mode", mean_substrate_online_fast_experimental_mode),
        (
            "mean_substrate_online_fast_review_or_revert_safe",
            mean_substrate_online_fast_review_or_revert_safe,
        ),
        ("mean_substrate_online_fast_rollback_integrity", mean_substrate_online_fast_rollback_integrity),
        ("mean_memory_updater_effective_lr", mean_memory_updater_effective_lr),
        ("mean_memory_updater_confidence", mean_memory_updater_confidence),
        ("mean_temporal_updater_effective_lr", mean_temporal_updater_effective_lr),
        ("mean_temporal_updater_confidence", mean_temporal_updater_confidence),
        ("mean_optimizer_memory_drive", mean_optimizer_memory_drive),
        ("mean_timescale_contract_retained", mean_timescale_contract_retained),
        ("mean_scheduler_discipline", mean_scheduler_discipline),
        ("mean_scheduler_substrate_pressure", mean_scheduler_substrate_pressure),
        ("mean_scheduler_rare_heavy_pressure", mean_scheduler_rare_heavy_pressure),
        ("mean_default_continual_learning_active", mean_default_continual_learning_active),
        ("mean_default_owner_writeback_retained", mean_default_owner_writeback_retained),
        (
            "mean_default_substrate_live_mutation_suppressed",
            mean_default_substrate_live_mutation_suppressed,
        ),
        ("mean_default_continual_rollback_clean", mean_default_continual_rollback_clean),
        ("case_memory_surface_turn_count", float(report.case_memory_surface_turn_count)),
        ("strategy_playbook_surface_turn_count", float(report.strategy_playbook_surface_turn_count)),
        ("experience_fast_prior_surface_turn_count", float(report.experience_fast_prior_surface_turn_count)),
        ("experience_consolidation_surface_turn_count", float(report.experience_consolidation_surface_turn_count)),
        ("rare_heavy_recommended_count", float(report.rare_heavy_recommended_count)),
        ("rare_heavy_applied_count", float(report.rare_heavy_applied_count)),
        ("rare_heavy_pre_import_pass_count", float(report.rare_heavy_pre_import_pass_count)),
        ("rare_heavy_pre_import_reject_count", float(report.rare_heavy_pre_import_reject_count)),
        ("mean_rare_heavy_pre_import_score_delta", report.mean_rare_heavy_pre_import_score_delta),
        ("mean_rare_heavy_candidate_alignment", report.mean_rare_heavy_candidate_alignment),
        (
            "max_rare_heavy_candidate_adapter_parameter_count",
            float(report.max_rare_heavy_candidate_adapter_parameter_count),
        ),
        ("evolution_judge_turn_count", float(report.evolution_judge_turn_count)),
        ("evolution_judge_rollback_count", float(report.evolution_judge_rollback_count)),
        ("evolution_judge_structural_allow_count", float(report.evolution_judge_structural_allow_count)),
        ("nested_profile_active_turn_count", float(report.nested_profile_active_turn_count)),
        ("nested_context_reset_count", float(report.nested_context_reset_count)),
        ("store_nested_context_reset_count", float(report.store_nested_context_reset_count)),
        ("boundary_reset_observed_on_first_turn", float(report.boundary_reset_observed_on_first_turn)),
        ("first_turn_slow_to_fast_init_benefit", report.first_turn_slow_to_fast_init_benefit),
        ("mean_reset_turn_slow_to_fast_init_benefit", report.mean_reset_turn_slow_to_fast_init_benefit),
        ("mean_slow_to_fast_init_benefit", report.mean_slow_to_fast_init_benefit),
        (
            "first_turn_slow_to_fast_target_distance_before",
            report.first_turn_slow_to_fast_target_distance_before,
        ),
        (
            "first_turn_slow_to_fast_target_distance_after",
            report.first_turn_slow_to_fast_target_distance_after,
        ),
        (
            "first_turn_slow_to_fast_target_alignment_gain",
            report.first_turn_slow_to_fast_target_alignment_gain,
        ),
        (
            "mean_reset_turn_slow_to_fast_target_distance_before",
            report.mean_reset_turn_slow_to_fast_target_distance_before,
        ),
        (
            "mean_reset_turn_slow_to_fast_target_distance_after",
            report.mean_reset_turn_slow_to_fast_target_distance_after,
        ),
        (
            "mean_reset_turn_slow_to_fast_target_alignment_gain",
            report.mean_reset_turn_slow_to_fast_target_alignment_gain,
        ),
        ("learned_memory_primary_turn_count", float(report.learned_memory_primary_turn_count)),
        ("core_guided_recall_turn_count", float(report.core_guided_recall_turn_count)),
        ("mean_learned_recall_confidence", report.mean_learned_recall_confidence),
        ("max_artifact_consolidation_count", float(report.max_artifact_consolidation_count)),
        ("max_tower_consolidation_count", float(report.max_tower_consolidation_count)),
        ("mean_memory_tower_depth", report.mean_memory_tower_depth),
        ("mean_memory_tower_alignment", report.mean_memory_tower_alignment),
        ("memory_tower_profile_turn_count", float(report.memory_tower_profile_turn_count)),
        ("temporal_change_count", float(report.temporal_change_count)),
        ("delayed_improvement_observed", float(report.delayed_improvement_observed)),
        ("mean_prediction_error", mean_pe),
        ("mean_switch_gate", mean_switch_gate),
    )


def _mean_summary_metrics(
    reports: tuple[DialogueBenchmarkCaseReport, ...],
) -> tuple[tuple[str, float], ...]:
    if not reports:
        return ()
    metric_names = tuple(key for key, _ in _case_summary_metrics(reports[0]))
    metric_totals = {key: 0.0 for key in metric_names}
    for report in reports:
        for key, value in _case_summary_metrics(report):
            metric_totals[key] += value
    return tuple(
        (key, round(metric_totals[key] / len(reports), 4))
        for key in metric_names
    )


def build_pe_dominance_comparison_report(
    comparison_report: DialogueBenchmarkComparisonReport,
    *,
    baseline_label: str = "pe-eta",
    pe_drive_off_label: str = "pe-drive-off",
    pe_readout_only_label: str = "pe-eta-pe-readout-only",
) -> PEDominanceComparisonReport:
    metric_map = {
        label: dict(metrics)
        for label, metrics in comparison_report.metric_deltas_from_baseline
    }
    path_metric_map = {
        path.path_label: dict(path.benchmark_report.metric_means)
        for path in comparison_report.path_reports
    }
    required_labels = (baseline_label, pe_drive_off_label, pe_readout_only_label)
    missing_labels = tuple(label for label in required_labels if label not in path_metric_map)
    if missing_labels:
        raise ValueError(
            f"PE dominance comparison requires paths {required_labels!r}; missing {missing_labels!r}."
        )
    baseline_metrics = path_metric_map[baseline_label]
    pe_drive_metrics = path_metric_map[pe_drive_off_label]
    pe_readout_metrics = path_metric_map[pe_readout_only_label]

    baseline_trigger = baseline_metrics.get("pe_triggered_turn_count", 0.0)
    baseline_precision = baseline_metrics.get("pressure_response_precision", 0.0)
    baseline_stability = baseline_metrics.get("stability_after_recovery_score", 0.0)
    baseline_delayed_improvement = baseline_metrics.get("delayed_improvement_observed", 0.0)
    baseline_prediction_chain = baseline_metrics.get("prediction_chain_turn_count", 0.0)

    readout_trigger = pe_readout_metrics.get("pe_triggered_turn_count", 0.0)
    readout_precision = pe_readout_metrics.get("pressure_response_precision", 0.0)
    readout_stability = pe_readout_metrics.get("stability_after_recovery_score", 0.0)
    readout_delayed_improvement = pe_readout_metrics.get("delayed_improvement_observed", 0.0)
    readout_prediction_chain = pe_readout_metrics.get("prediction_chain_turn_count", 0.0)

    mechanism_retention_numerator = (
        readout_precision
        + readout_stability
        + readout_delayed_improvement
    )
    mechanism_retention_denominator = max(
        baseline_precision + baseline_stability + baseline_delayed_improvement,
        1e-6,
    )
    mechanism_retention_ratio = round(
        max(0.0, min(1.5, mechanism_retention_numerator / mechanism_retention_denominator)),
        4,
    )
    pe_visibility_retention_ratio = round(
        max(0.0, min(1.5, readout_prediction_chain / max(baseline_prediction_chain, 1e-6))),
        4,
    )
    schedule_dependence_gap = round(baseline_trigger - readout_trigger, 4)
    reward_dominance_gap = round(
        (baseline_precision - readout_precision)
        + (baseline_stability - readout_stability),
        4,
    )
    if mechanism_retention_ratio >= 0.8 and schedule_dependence_gap <= 0.25:
        interpretation = "latent-mechanism-dominant"
    elif mechanism_retention_ratio >= 0.45 and pe_visibility_retention_ratio >= 0.8:
        interpretation = "mixed-pe-and-latent"
    else:
        interpretation = "pe-dominance-likely"
    return PEDominanceComparisonReport(
        baseline_label=baseline_label,
        pe_drive_off_label=pe_drive_off_label,
        pe_readout_only_label=pe_readout_only_label,
        metrics_by_path=tuple(
            (label, tuple(sorted(path_metric_map[label].items())))
            for label in required_labels
        ),
        deltas_from_baseline=tuple(
            (label, tuple(sorted(metric_map.get(label, {}).items())))
            for label in (pe_drive_off_label, pe_readout_only_label)
        ),
        mechanism_retention_ratio=mechanism_retention_ratio,
        pe_visibility_retention_ratio=pe_visibility_retention_ratio,
        schedule_dependence_gap=schedule_dependence_gap,
        reward_dominance_gap=reward_dominance_gap,
        interpretation=interpretation,
        description=(
            f"PE dominance comparison baseline={baseline_label} readout_only={pe_readout_only_label} "
            f"pe_drive_off={pe_drive_off_label} mechanism_retention={mechanism_retention_ratio:.3f} "
            f"pe_visibility={pe_visibility_retention_ratio:.3f} schedule_gap={schedule_dependence_gap:.3f} "
            f"reward_gap={reward_dominance_gap:.3f} interpretation={interpretation}."
        ),
    )


def build_pe_dominance_case_diagnosis_report(
    comparison_report: DialogueBenchmarkComparisonReport,
    *,
    baseline_label: str = "pe-eta",
    pe_drive_off_label: str = "pe-drive-off",
    pe_readout_only_label: str = "pe-eta-pe-readout-only",
) -> PEDominanceCaseDiagnosisReport:
    path_case_map = {
        path.path_label: {
            case_report.case.case_id: case_report
            for case_report in path.benchmark_report.case_reports
        }
        for path in comparison_report.path_reports
    }
    required_labels = (baseline_label, pe_drive_off_label, pe_readout_only_label)
    missing_labels = tuple(label for label in required_labels if label not in path_case_map)
    if missing_labels:
        raise ValueError(
            f"PE case diagnosis requires paths {required_labels!r}; missing {missing_labels!r}."
        )
    case_diagnoses: list[PEDominanceCaseDiagnosis] = []
    failure_mode_counts: dict[str, int] = {}
    for case_id, baseline_case in path_case_map[baseline_label].items():
        pe_drive_case = path_case_map[pe_drive_off_label][case_id]
        pe_readout_case = path_case_map[pe_readout_only_label][case_id]
        baseline_metrics = dict(_case_summary_metrics(baseline_case))
        pe_drive_deltas = {
            key: value - baseline_metrics[key]
            for key, value in _case_summary_metrics(pe_drive_case)
        }
        pe_readout_deltas = {
            key: value - baseline_metrics[key]
            for key, value in _case_summary_metrics(pe_readout_case)
        }
        schedule_gap = (
            abs(pe_drive_deltas.get("pe_triggered_turn_count", 0.0))
            + abs(pe_readout_deltas.get("pe_triggered_turn_count", 0.0))
        )
        reward_gap = (
            abs(pe_drive_deltas.get("pressure_response_precision", 0.0))
            + abs(pe_drive_deltas.get("stability_after_recovery_score", 0.0))
            + abs(pe_readout_deltas.get("pressure_response_precision", 0.0))
            + abs(pe_readout_deltas.get("stability_after_recovery_score", 0.0))
        )
        latent_gap = (
            abs(pe_drive_deltas.get("temporal_change_count", 0.0))
            + abs(pe_readout_deltas.get("temporal_change_count", 0.0))
            + abs(pe_drive_deltas.get("delayed_improvement_observed", 0.0))
            + abs(pe_readout_deltas.get("delayed_improvement_observed", 0.0))
        )
        degradation_severity = round(schedule_gap * 0.35 + reward_gap * 0.40 + latent_gap * 0.25, 4)
        if schedule_gap >= reward_gap and schedule_gap >= latent_gap:
            failure_mode = "schedule-driven"
        elif reward_gap >= max(schedule_gap, latent_gap):
            failure_mode = "reward-driven"
        else:
            failure_mode = "latent-fragility-driven"
        failure_mode_counts[failure_mode] = failure_mode_counts.get(failure_mode, 0) + 1
        case_diagnoses.append(
            PEDominanceCaseDiagnosis(
                case_id=case_id,
                pe_eta_metrics=tuple(sorted(baseline_metrics.items())),
                pe_drive_off_deltas=tuple(sorted(pe_drive_deltas.items())),
                pe_readout_only_deltas=tuple(sorted(pe_readout_deltas.items())),
                degradation_severity=degradation_severity,
                failure_mode=failure_mode,
                description=(
                    f"Case {case_id} degradation={degradation_severity:.3f} "
                    f"schedule_gap={schedule_gap:.3f} reward_gap={reward_gap:.3f} "
                    f"latent_gap={latent_gap:.3f} failure_mode={failure_mode}."
                ),
            )
        )
    case_diagnoses.sort(key=lambda diagnosis: diagnosis.degradation_severity, reverse=True)
    dominant_failure_mode = (
        max(failure_mode_counts.items(), key=lambda item: item[1])[0]
        if failure_mode_counts
        else "unknown"
    )
    worst_case_id = case_diagnoses[0].case_id if case_diagnoses else None
    return PEDominanceCaseDiagnosisReport(
        baseline_label=baseline_label,
        pe_drive_off_label=pe_drive_off_label,
        pe_readout_only_label=pe_readout_only_label,
        case_diagnoses=tuple(case_diagnoses),
        worst_case_id=worst_case_id,
        dominant_failure_mode=dominant_failure_mode,
        description=(
            f"PE case diagnosis baseline={baseline_label} worst_case={worst_case_id or 'none'} "
            f"dominant_failure_mode={dominant_failure_mode} cases={len(case_diagnoses)}."
        ),
    )


def _dashboard_panel_from_metric_deltas(
    *,
    path_label: str,
    metric_deltas: dict[str, float],
    stability_metric: str,
) -> DialogueEmergenceDashboardPanel:
    passed_delta = metric_deltas.get("passed", 0.0)
    pe_triggered_delta = metric_deltas.get("pe_triggered_turn_count", 0.0)
    delayed_improvement_delta = metric_deltas.get("delayed_improvement_observed", 0.0)
    stability_delta = metric_deltas.get(stability_metric, 0.0)
    mean_prediction_error_delta = metric_deltas.get("mean_prediction_error", 0.0)
    memory_tower_depth_delta = metric_deltas.get("mean_memory_tower_depth", 0.0)
    memory_tower_alignment_delta = metric_deltas.get("mean_memory_tower_alignment", 0.0)
    tower_consolidation_delta = metric_deltas.get("max_tower_consolidation_count", 0.0)
    retention_score = round(
        passed_delta
        + 0.25 * pe_triggered_delta
        + 0.25 * delayed_improvement_delta
        + 0.25 * stability_delta
        + 0.25 * max(0.0, -mean_prediction_error_delta),
        4,
    )
    tower_retention_bonus = round(
        max(0.0, memory_tower_depth_delta) * 0.06
        + max(0.0, memory_tower_alignment_delta) * 0.24
        + max(0.0, tower_consolidation_delta) * 0.12,
        4,
    )
    retention_score = round(retention_score + tower_retention_bonus, 4)
    return DialogueEmergenceDashboardPanel(
        path_label=path_label,
        passed_delta=passed_delta,
        pe_triggered_delta=pe_triggered_delta,
        delayed_improvement_delta=delayed_improvement_delta,
        stability_delta=stability_delta,
        mean_prediction_error_delta=mean_prediction_error_delta,
        memory_tower_depth_delta=memory_tower_depth_delta,
        memory_tower_alignment_delta=memory_tower_alignment_delta,
        tower_consolidation_delta=tower_consolidation_delta,
        retention_score=retention_score,
        description=(
            f"Dashboard panel {path_label} passed_delta={passed_delta:.3f} "
            f"pe_delta={pe_triggered_delta:.3f} delayed_delta={delayed_improvement_delta:.3f} "
            f"stability_delta={stability_delta:.3f} mean_pe_delta={mean_prediction_error_delta:.3f} "
            f"tower_depth_delta={memory_tower_depth_delta:.3f} "
            f"tower_alignment_delta={memory_tower_alignment_delta:.3f} "
            f"tower_consolidation_delta={tower_consolidation_delta:.3f} "
            f"retention_score={retention_score:.3f}."
        ),
    )


def _tower_effective_strength(
    *,
    depth: float,
    alignment: float,
    consolidation_count: float,
) -> float:
    return round(
        max(depth - 4.0, 0.0) * 0.10
        + max(alignment, 0.0) * 1.55
        + min(max(consolidation_count, 0.0), 3.0) * 0.24,
        4,
    )


def _slow_to_fast_signal_strength(
    *,
    init_benefit: float,
    target_alignment_gain: float,
    boundary_reset_observed: float,
    learned_recall_confidence: float,
    tower_depth: float,
    tower_alignment: float,
    tower_profile_turn_count: float,
) -> float:
    return round(
        min(max(init_benefit / max(PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD, 1e-6), 0.0), 1.0) * 0.26
        + min(max(target_alignment_gain / max(PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD, 1e-6), 0.0), 1.0) * 0.18
        + min(max(boundary_reset_observed, 0.0), 1.0) * 0.14
        + min(max(learned_recall_confidence, 0.0), 1.0) * 0.14
        + min(max((tower_depth - 4.0) / 2.0, 0.0), 1.0) * 0.10
        + min(max(tower_alignment / 0.12, 0.0), 1.0) * 0.12
        + min(max(tower_profile_turn_count / 4.0, 0.0), 1.0) * 0.06,
        4,
    )


def _tower_strength_gap_from_delta(metric_map: dict[str, float]) -> float:
    return _tower_effective_strength(
        depth=max(-metric_map.get("mean_memory_tower_depth", 0.0), 0.0),
        alignment=max(-metric_map.get("mean_memory_tower_alignment", 0.0), 0.0),
        consolidation_count=max(-metric_map.get("max_tower_consolidation_count", 0.0), 0.0),
    )


def build_dialogue_emergence_dashboard(
    comprehensive_report: DialogueComprehensiveBenchmarkReport,
) -> DialogueEmergenceDashboardArtifact:
    baseline_path = next(
        (
            path
            for path in comprehensive_report.canonical_ablation_report.path_reports
            if path.path_label == comprehensive_report.canonical_ablation_report.baseline_label
        ),
        None,
    )
    if baseline_path is None:
        raise ValueError("Comprehensive report is missing the canonical baseline path.")
    canonical_report = baseline_path.benchmark_report
    canonical_metric_means = dict(canonical_report.metric_means)
    canonical_pass_rate = (
        canonical_report.passed_case_count / canonical_report.total_case_count
        if canonical_report.total_case_count > 0
        else 0.0
    )
    open_report = comprehensive_report.open_ablation_report
    open_panels: tuple[DialogueEmergenceDashboardPanel, ...] = ()
    open_pass_rate = 0.0
    open_case_count = 0
    open_mean_memory_tower_depth = 0.0
    open_mean_memory_tower_alignment = 0.0
    open_max_tower_consolidation_count = 0.0
    open_runtime_backbone_evidence_rate = 0.0
    open_mean_runtime_backbone_signal_quality = 0.0
    open_mean_fast_memory_runtime_alignment = 0.0
    if open_report is not None:
        open_baseline_path = next(
            (path for path in open_report.path_reports if path.path_label == open_report.baseline_label),
            None,
        )
        if open_baseline_path is not None:
            open_metric_means = dict(open_baseline_path.benchmark_report.metric_means)
            open_case_count = open_baseline_path.benchmark_report.total_case_count
            if open_baseline_path.benchmark_report.total_case_count > 0:
                open_pass_rate = (
                    open_baseline_path.benchmark_report.passed_case_count
                    / open_baseline_path.benchmark_report.total_case_count
                )
            open_mean_memory_tower_depth = open_metric_means.get("mean_memory_tower_depth", 0.0)
            open_mean_memory_tower_alignment = open_metric_means.get("mean_memory_tower_alignment", 0.0)
            open_max_tower_consolidation_count = open_metric_means.get("max_tower_consolidation_count", 0.0)
            open_runtime_backbone_evidence_rate = open_metric_means.get(
                "runtime_backbone_evidence_turn_count", 0.0
            ) / max(open_case_count, 1)
            open_mean_runtime_backbone_signal_quality = open_metric_means.get(
                "mean_runtime_backbone_signal_quality", 0.0
            )
            open_mean_fast_memory_runtime_alignment = open_metric_means.get(
                "mean_fast_memory_runtime_alignment", 0.0
            )
        open_panels = tuple(
            _dashboard_panel_from_metric_deltas(
                path_label=path_label,
                metric_deltas=dict(metric_deltas),
                stability_metric="late_episode_stability_score",
            )
            for path_label, metric_deltas in open_report.metric_deltas_from_baseline
            if path_label != open_report.baseline_label
        )
    strong_proof_metric_deltas = (
        comprehensive_report.canonical_ablation_report.strong_proof_metric_deltas
        if comprehensive_report.canonical_ablation_report.strong_proof_metric_deltas
        else comprehensive_report.canonical_ablation_report.metric_deltas_from_baseline
    )
    strong_proof_panels = tuple(
        _dashboard_panel_from_metric_deltas(
            path_label=path_label,
            metric_deltas=dict(metric_deltas),
            stability_metric="stability_after_recovery_score",
        )
        for path_label, metric_deltas in strong_proof_metric_deltas
        if path_label != comprehensive_report.canonical_ablation_report.baseline_label
    )
    pe_dominance_report = None
    pe_case_diagnosis_report = None
    tower_memory_gate = next(
        (gate for gate in comprehensive_report.essence_report.gates if gate.gate_id == "tower-memory-surface"),
        None,
    )
    available_path_labels = {
        path.path_label for path in comprehensive_report.canonical_ablation_report.path_reports
    }
    if {"pe-eta", "pe-drive-off", "pe-eta-pe-readout-only"} <= available_path_labels:
        pe_dominance_report = build_pe_dominance_comparison_report(
            comprehensive_report.canonical_ablation_report
        )
        pe_case_diagnosis_report = build_pe_dominance_case_diagnosis_report(
            comprehensive_report.canonical_ablation_report
        )
    strongest_scaffold_panel = max(
        strong_proof_panels,
        key=lambda panel: panel.retention_score,
        default=None,
    )
    strongest_open_panel = max(
        open_panels,
        key=lambda panel: panel.retention_score,
        default=None,
    )
    if pe_dominance_report is not None and strongest_open_panel is not None:
        interpretation = (
            f"{pe_dominance_report.interpretation}; "
            f"strongest_open_path={strongest_open_panel.path_label}"
        )
    elif pe_dominance_report is not None:
        interpretation = pe_dominance_report.interpretation
    elif strongest_open_panel is not None:
        interpretation = f"open-evidence-present:{strongest_open_panel.path_label}"
    else:
        interpretation = "summary-only"
    tower_memory_gate_strength = 0.0
    if tower_memory_gate is not None:
        tower_evidence = dict(tower_memory_gate.evidence)
        tower_memory_gate_strength = round(
            min(
                1.5,
                tower_evidence.get("tower_effective_strength", 0.0)
                + max(0.0, tower_evidence.get("mean_memory_tower_alignment", 0.0)) * 0.35
                + max(0.0, tower_evidence.get("tower_strength_gap_vs_best_control", 0.0)) * 0.6,
            ),
            4,
        )
    return DialogueEmergenceDashboardArtifact(
        baseline_label=comprehensive_report.canonical_ablation_report.baseline_label,
        canonical_case_count=canonical_report.total_case_count,
        canonical_pass_rate=round(canonical_pass_rate, 4),
        canonical_mean_memory_tower_depth=canonical_metric_means.get("mean_memory_tower_depth", 0.0),
        canonical_mean_memory_tower_alignment=canonical_metric_means.get("mean_memory_tower_alignment", 0.0),
        canonical_max_tower_consolidation_count=canonical_metric_means.get("max_tower_consolidation_count", 0.0),
        canonical_tower_profile_turn_count=canonical_metric_means.get("memory_tower_profile_turn_count", 0.0),
        open_scenario_count=open_case_count,
        open_pass_rate=round(open_pass_rate, 4),
        open_mean_memory_tower_depth=open_mean_memory_tower_depth,
        open_mean_memory_tower_alignment=open_mean_memory_tower_alignment,
        open_max_tower_consolidation_count=open_max_tower_consolidation_count,
        strong_proof_panels=strong_proof_panels,
        open_environment_panels=open_panels,
        pe_dominance_report=pe_dominance_report,
        pe_case_diagnosis_report=pe_case_diagnosis_report,
        tower_memory_gate_passed=tower_memory_gate.passed if tower_memory_gate is not None else False,
        tower_memory_gate_strength=tower_memory_gate_strength,
        strongest_scaffold_path_label=(
            strongest_scaffold_panel.path_label if strongest_scaffold_panel is not None else None
        ),
        strongest_scaffold_retention_score=(
            strongest_scaffold_panel.retention_score if strongest_scaffold_panel is not None else 0.0
        ),
        strongest_open_path_label=(
            strongest_open_panel.path_label if strongest_open_panel is not None else None
        ),
        strongest_open_retention_score=(
            strongest_open_panel.retention_score if strongest_open_panel is not None else 0.0
        ),
        interpretation=interpretation,
        description=(
            f"Emergence dashboard baseline={comprehensive_report.canonical_ablation_report.baseline_label} "
            f"canonical_pass_rate={canonical_pass_rate:.3f} open_pass_rate={open_pass_rate:.3f} "
            f"runtime_evidence={canonical_metric_means.get('runtime_backbone_evidence_turn_count', 0.0) / max(canonical_report.total_case_count, 1):.2f} "
            f"runtime_quality={canonical_metric_means.get('mean_runtime_backbone_signal_quality', 0.0):.2f} "
            f"tower_depth={canonical_metric_means.get('mean_memory_tower_depth', 0.0):.2f} "
            f"tower_alignment={canonical_metric_means.get('mean_memory_tower_alignment', 0.0):.2f} "
            f"tower_gate_strength={tower_memory_gate_strength:.2f} "
            f"strong_proof_panels={len(strong_proof_panels)} open_panels={len(open_panels)} "
            f"interpretation={interpretation}."
        ),
        canonical_runtime_backbone_evidence_rate=round(
            canonical_metric_means.get("runtime_backbone_evidence_turn_count", 0.0)
            / max(canonical_report.total_case_count, 1),
            4,
        ),
        canonical_mean_runtime_backbone_signal_quality=canonical_metric_means.get(
            "mean_runtime_backbone_signal_quality", 0.0
        ),
        canonical_mean_fast_memory_runtime_alignment=canonical_metric_means.get(
            "mean_fast_memory_runtime_alignment", 0.0
        ),
        open_runtime_backbone_evidence_rate=round(open_runtime_backbone_evidence_rate, 4),
        open_mean_runtime_backbone_signal_quality=open_mean_runtime_backbone_signal_quality,
        open_mean_fast_memory_runtime_alignment=open_mean_fast_memory_runtime_alignment,
    )


def _empty_emergence_dashboard(*, baseline_label: str) -> DialogueEmergenceDashboardArtifact:
    return DialogueEmergenceDashboardArtifact(
        baseline_label=baseline_label,
        canonical_case_count=0,
        canonical_pass_rate=0.0,
        canonical_mean_memory_tower_depth=0.0,
        canonical_mean_memory_tower_alignment=0.0,
        canonical_max_tower_consolidation_count=0.0,
        canonical_tower_profile_turn_count=0.0,
        open_scenario_count=0,
        open_pass_rate=0.0,
        open_mean_memory_tower_depth=0.0,
        open_mean_memory_tower_alignment=0.0,
        open_max_tower_consolidation_count=0.0,
        strong_proof_panels=(),
        open_environment_panels=(),
        pe_dominance_report=None,
        pe_case_diagnosis_report=None,
        tower_memory_gate_passed=False,
        tower_memory_gate_strength=0.0,
        strongest_scaffold_path_label=None,
        strongest_scaffold_retention_score=0.0,
        strongest_open_path_label=None,
        strongest_open_retention_score=0.0,
        interpretation="pending",
        description="Empty emergence dashboard placeholder.",
        canonical_runtime_backbone_evidence_rate=0.0,
        canonical_mean_runtime_backbone_signal_quality=0.0,
        canonical_mean_fast_memory_runtime_alignment=0.0,
        open_runtime_backbone_evidence_rate=0.0,
        open_mean_runtime_backbone_signal_quality=0.0,
        open_mean_fast_memory_runtime_alignment=0.0,
    )


def _metric_mean_from_report(report: DialogueBenchmarkReport, metric_name: str) -> float:
    return dict(report.metric_means).get(metric_name, 0.0)


def _rare_heavy_gate_description_fragment(assessment: DialogueNLEssenceAssessmentReport) -> str:
    snapshot = _rare_heavy_gate_snapshot(assessment)
    return (
        f"rare_heavy_gate_passed={snapshot['passed']} "
        f"rare_heavy_gate_failure_mode={snapshot['failure_mode']} "
        f"rare_heavy_gate_applied_delta={snapshot['applied_delta_vs_no_rare_heavy']:.3f} "
        f"rare_heavy_gate_passed_delta={snapshot['passed_delta_vs_no_rare_heavy']:.3f} "
        f"rare_heavy_gate_pe_delta={snapshot['mean_prediction_error_delta_vs_no_rare_heavy']:.3f}"
    )


def _emergence_dashboard_description_fragment(dashboard: DialogueEmergenceDashboardArtifact) -> str:
    return (
        f"emergence_interpretation={dashboard.interpretation} "
        f"canonical_pass_rate={dashboard.canonical_pass_rate:.3f} "
        f"open_pass_rate={dashboard.open_pass_rate:.3f} "
        f"runtime_evidence_rate={dashboard.canonical_runtime_backbone_evidence_rate:.3f} "
        f"runtime_quality={dashboard.canonical_mean_runtime_backbone_signal_quality:.3f} "
        f"strongest_scaffold={dashboard.strongest_scaffold_path_label or 'none'} "
        f"strongest_scaffold_score={dashboard.strongest_scaffold_retention_score:.3f} "
        f"strongest_open={dashboard.strongest_open_path_label or 'none'} "
        f"strongest_open_score={dashboard.strongest_open_retention_score:.3f}"
    )


def build_dialogue_nl_essence_assessment(
    *,
    path_label: str,
    benchmark_report: DialogueBenchmarkReport,
    baseline_report: DialogueBenchmarkReport | None = None,
    comparison_report: DialogueBenchmarkComparisonReport | None = None,
    cross_session_report: CrossSessionGrowthReport | None = None,
    longitudinal_report: DialogueLongitudinalBenchmarkReport | None = None,
    proof_min_canonical_cases: int = PROOF_MIN_CANONICAL_CASES,
) -> DialogueNLEssenceAssessmentReport:
    baseline_metric_means = dict(baseline_report.metric_means) if baseline_report is not None else {}
    metric_means = dict(benchmark_report.metric_means)
    longitudinal_metric_means = (
        dict(_mean_summary_metrics(longitudinal_report.case_reports))
        if longitudinal_report is not None and longitudinal_report.case_reports
        else {}
    )
    evidence_metric_means = dict(metric_means)
    evidence_metric_means.update(longitudinal_metric_means)
    canonical_case_count = len(benchmark_report.case_reports)
    longitudinal_case_count = len(longitudinal_report.case_reports) if longitudinal_report is not None else 0
    proof_min_case_count_satisfied = (
        canonical_case_count >= proof_min_canonical_cases
        and longitudinal_case_count >= proof_min_canonical_cases
    )
    pe_first_passed = (
        evidence_metric_means.get("prediction_chain_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("pe_triggered_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("delayed_improvement_observed", 0.0) >= 0.5
    )
    multi_timescale_passed = (
        evidence_metric_means.get("online_learning_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("runtime_backbone_evidence_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("mean_runtime_backbone_signal_quality", 0.0) >= 0.2
        and evidence_metric_means.get("mean_fast_memory_signal_norm", 0.0) > 0.0
        and (
            evidence_metric_means.get("bounded_writeback_turn_count", 0.0) > 0.0
            or evidence_metric_means.get("session_post_completion_turn_count", 0.0) > 0.0
            or evidence_metric_means.get("reflection_promotion_eligible_turn_count", 0.0) > 0.0
            or evidence_metric_means.get("online_fast_substrate_recommended_count", 0.0) > 0.0
            or evidence_metric_means.get("rare_heavy_recommended_count", 0.0) > 0.0
        )
        and evidence_metric_means.get("nested_profile_active_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("learned_memory_primary_turn_count", 0.0) > 0.0
    )
    online_fast_pe_coupling_passed = (
        evidence_metric_means.get("pe_triggered_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("online_fast_substrate_recommended_count", 0.0) > 0.0
        and evidence_metric_means.get("mean_fast_memory_signal_norm", 0.0) > 0.0
        and evidence_metric_means.get("mean_fast_memory_runtime_alignment", 0.0) > 0.05
    )
    bounded_live_self_mod_passed = (
        evidence_metric_means.get("online_fast_substrate_recommended_count", 0.0) > 0.0
        and evidence_metric_means.get("mean_substrate_online_fast_review_or_revert_safe", 0.0) >= 0.8
        and evidence_metric_means.get("mean_substrate_online_fast_rollback_integrity", 0.0) >= 0.95
    )
    updater_evidence_visible_passed = (
        (
            evidence_metric_means.get("mean_memory_updater_effective_lr", 0.0) > 0.0
            or evidence_metric_means.get("mean_temporal_updater_effective_lr", 0.0) > 0.0
        )
        and (
            evidence_metric_means.get("mean_memory_updater_confidence", 0.0) > 0.0
            or evidence_metric_means.get("mean_temporal_updater_confidence", 0.0) > 0.0
        )
    )
    timescale_contract_retained_passed = (
        evidence_metric_means.get("mean_timescale_contract_retained", 0.0) >= 0.55
        and evidence_metric_means.get("mean_scheduler_discipline", 0.0) >= 0.45
    )
    default_continual_learner_passed = (
        evidence_metric_means.get("mean_default_continual_learning_active", 0.0) > 0.0
        and evidence_metric_means.get("mean_default_owner_writeback_retained", 0.0) >= 0.5
        and evidence_metric_means.get("mean_default_substrate_live_mutation_suppressed", 0.0) >= 0.95
        and evidence_metric_means.get("online_fast_substrate_applied_count", 0.0) <= 0.0
        and evidence_metric_means.get("mean_default_continual_rollback_clean", 0.0) >= 0.5
    )
    store_nested_reset_count = evidence_metric_means.get("store_nested_context_reset_count", 0.0)
    reset_turn_slow_to_fast_init_benefit = evidence_metric_means.get(
        "mean_reset_turn_slow_to_fast_init_benefit", 0.0
    )
    reset_turn_target_distance_before = evidence_metric_means.get(
        "mean_reset_turn_slow_to_fast_target_distance_before", 0.0
    )
    reset_turn_target_distance_after = evidence_metric_means.get(
        "mean_reset_turn_slow_to_fast_target_distance_after", 0.0
    )
    reset_turn_target_alignment_gain = evidence_metric_means.get(
        "mean_reset_turn_slow_to_fast_target_alignment_gain", 0.0
    )
    weak_benefit_explained_by_target_proximity = (
        reset_turn_target_distance_before <= PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD
        and reset_turn_target_distance_after <= PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD
        and reset_turn_target_alignment_gain >= 0.0
    )
    if weak_benefit_explained_by_target_proximity:
        slow_shapes_fast_alignment_interpretation = "already-near-target"
    elif reset_turn_target_alignment_gain > 0.0:
        slow_shapes_fast_alignment_interpretation = "moving-toward-target"
    elif reset_turn_target_alignment_gain < 0.0:
        slow_shapes_fast_alignment_interpretation = "diverging-from-target"
    else:
        slow_shapes_fast_alignment_interpretation = "flat-alignment"
    slow_shapes_fast_failure_mode = "passed"
    if store_nested_reset_count <= 0.0 and not proof_min_case_count_satisfied:
        slow_shapes_fast_failure_mode = "config-artifact"
    elif store_nested_reset_count <= 0.0:
        slow_shapes_fast_failure_mode = "no-reset"
    elif reset_turn_slow_to_fast_init_benefit <= PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD:
        slow_shapes_fast_failure_mode = (
            "already-near-target" if weak_benefit_explained_by_target_proximity else "weak-benefit"
        )
    memory_tower_depth = evidence_metric_means.get("mean_memory_tower_depth", 0.0)
    memory_tower_alignment = evidence_metric_means.get("mean_memory_tower_alignment", 0.0)
    tower_consolidation_count = evidence_metric_means.get("max_tower_consolidation_count", 0.0)
    artifact_consolidation_count = evidence_metric_means.get("max_artifact_consolidation_count", 0.0)
    tower_consolidation_evidence = max(tower_consolidation_count, artifact_consolidation_count)
    tower_effective_strength = _tower_effective_strength(
        depth=memory_tower_depth,
        alignment=memory_tower_alignment,
        consolidation_count=tower_consolidation_evidence,
    )
    tower_profile_turn_count = evidence_metric_means.get("memory_tower_profile_turn_count", 0.0)
    slow_to_fast_signal_strength = _slow_to_fast_signal_strength(
        init_benefit=reset_turn_slow_to_fast_init_benefit,
        target_alignment_gain=reset_turn_target_alignment_gain,
        boundary_reset_observed=evidence_metric_means.get("boundary_reset_observed_on_first_turn", 0.0),
        learned_recall_confidence=evidence_metric_means.get("mean_learned_recall_confidence", 0.0),
        tower_depth=memory_tower_depth,
        tower_alignment=memory_tower_alignment,
        tower_profile_turn_count=tower_profile_turn_count,
    )
    slow_to_fast_passed = (
        store_nested_reset_count > 0.0
        and proof_min_case_count_satisfied
        and (
            reset_turn_slow_to_fast_init_benefit > PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD
            or (
                slow_to_fast_signal_strength >= PROOF_SLOW_TO_FAST_SIGNAL_STRENGTH_THRESHOLD
                and not weak_benefit_explained_by_target_proximity
                and reset_turn_target_alignment_gain > 0.0
            )
        )
    )
    comparison_metric_deltas = (
        {
            candidate_label: dict(metric_items)
            for candidate_label, metric_items in comparison_report.metric_deltas_from_baseline
        }
        if comparison_report is not None
        else {}
    )
    tower_depth_gap_vs_best_control = max(
        (
            -metric_map.get("mean_memory_tower_depth", float("inf"))
            for candidate_label, metric_map in comparison_metric_deltas.items()
            if candidate_label != path_label
        ),
        default=float("-inf"),
    )
    tower_alignment_gap_vs_best_control = max(
        (
            -metric_map.get("mean_memory_tower_alignment", float("inf"))
            for candidate_label, metric_map in comparison_metric_deltas.items()
            if candidate_label != path_label
        ),
        default=float("-inf"),
    )
    tower_consolidation_gap_vs_best_control = max(
        (
            -metric_map.get("max_tower_consolidation_count", float("inf"))
            for candidate_label, metric_map in comparison_metric_deltas.items()
            if candidate_label != path_label
        ),
        default=float("-inf"),
    )
    tower_effective_strength = _tower_effective_strength(
        depth=memory_tower_depth,
        alignment=memory_tower_alignment,
        consolidation_count=tower_consolidation_evidence,
    )
    best_control_tower_gap = max(
        (
            _tower_strength_gap_from_delta(metric_map)
            for candidate_label, metric_map in comparison_metric_deltas.items()
            if candidate_label != path_label
        ),
        default=0.0,
    )
    tower_memory_surface_passed = (
        tower_profile_turn_count > 0.0
        and memory_tower_depth >= 4.0
        and memory_tower_alignment >= 0.12
        and tower_consolidation_evidence > 0.0
        and tower_effective_strength >= 0.65
        and (not comparison_metric_deltas or best_control_tower_gap > 0.05)
    )
    judge_gated_passed = (
        evidence_metric_means.get("evolution_judge_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("evolution_judge_structural_allow_count", 0.0) > 0.0
    )
    rare_heavy_metric_deltas = dict(comparison_report.rare_heavy_metric_deltas) if comparison_report is not None else {}
    rare_heavy_gate_failure_mode = "passed"
    if not rare_heavy_metric_deltas:
        rare_heavy_gate_failure_mode = "missing-no-rare-heavy-comparison"
    rare_heavy_recommended_delta = rare_heavy_metric_deltas.get("rare_heavy_recommended_count", 0.0)
    rare_heavy_applied_delta = rare_heavy_metric_deltas.get("rare_heavy_applied_count", 0.0)
    rare_heavy_pre_import_pass_delta = rare_heavy_metric_deltas.get("rare_heavy_pre_import_pass_count", 0.0)
    rare_heavy_pre_import_score_delta = rare_heavy_metric_deltas.get("mean_rare_heavy_pre_import_score_delta", 0.0)
    rare_heavy_candidate_alignment_delta = rare_heavy_metric_deltas.get("mean_rare_heavy_candidate_alignment", 0.0)
    rare_heavy_passed_delta = rare_heavy_metric_deltas.get("passed", 0.0)
    rare_heavy_delayed_improvement_delta = rare_heavy_metric_deltas.get("delayed_improvement_observed", 0.0)
    rare_heavy_prediction_error_delta = rare_heavy_metric_deltas.get("mean_prediction_error", 0.0)
    rare_heavy_stability_delta = rare_heavy_metric_deltas.get("stability_after_recovery_score", 0.0)
    current_rare_heavy_pre_import_pass_count = evidence_metric_means.get("rare_heavy_pre_import_pass_count", 0.0)
    current_rare_heavy_pre_import_score_delta = evidence_metric_means.get("mean_rare_heavy_pre_import_score_delta", 0.0)
    current_rare_heavy_candidate_alignment = evidence_metric_means.get("mean_rare_heavy_candidate_alignment", 0.0)
    current_rare_heavy_candidate_adapter_parameter_count = evidence_metric_means.get(
        "max_rare_heavy_candidate_adapter_parameter_count", 0.0
    )
    rare_heavy_review_evidence_present = (
        rare_heavy_applied_delta > 0.0
        or rare_heavy_recommended_delta > 0.0
        or current_rare_heavy_pre_import_pass_count > 0.0
        or current_rare_heavy_pre_import_score_delta > 0.0
        or current_rare_heavy_candidate_alignment >= 0.30
        or current_rare_heavy_candidate_adapter_parameter_count > 0.0
    )
    rare_heavy_net_benefit_passed = (
        bool(rare_heavy_metric_deltas)
        and rare_heavy_review_evidence_present
        and (
            rare_heavy_passed_delta > 0.0
            or rare_heavy_delayed_improvement_delta > 0.0
            or rare_heavy_stability_delta > 0.0
            or rare_heavy_prediction_error_delta < 0.0
            or rare_heavy_pre_import_pass_delta > 0.0
            or rare_heavy_pre_import_score_delta > 0.0
            or rare_heavy_candidate_alignment_delta > 0.0
            or (
                current_rare_heavy_pre_import_pass_count > 0.0
                and current_rare_heavy_pre_import_score_delta > 0.0
                and current_rare_heavy_candidate_alignment >= 0.30
                and rare_heavy_passed_delta >= 0.0
                and rare_heavy_stability_delta >= 0.0
                and rare_heavy_prediction_error_delta <= PROOF_PE_IMPROVEMENT_DELTA
            )
        )
    )
    if bool(rare_heavy_metric_deltas) and not rare_heavy_net_benefit_passed:
        rare_heavy_gate_failure_mode = "no-net-benefit"
    cross_session_passed = (
        cross_session_report is not None
        and cross_session_report.verdict in {"growing", "stable"}
    )
    gates = (
        DialogueNLEssenceGate(
            gate_id="pe-first",
            passed=pe_first_passed,
            evidence=(
                ("prediction_chain_turn_count", evidence_metric_means.get("prediction_chain_turn_count", 0.0)),
                ("pe_triggered_turn_count", evidence_metric_means.get("pe_triggered_turn_count", 0.0)),
                ("delayed_improvement_observed", evidence_metric_means.get("delayed_improvement_observed", 0.0)),
            ),
            description="Prediction error should form an observable chain that drives temporal response and delayed improvement.",
        ),
        DialogueNLEssenceGate(
            gate_id="multi-timescale-default",
            passed=multi_timescale_passed,
            evidence=(
                ("online_learning_turn_count", evidence_metric_means.get("online_learning_turn_count", 0.0)),
                (
                    "runtime_backbone_evidence_turn_count",
                    evidence_metric_means.get("runtime_backbone_evidence_turn_count", 0.0),
                ),
                (
                    "mean_runtime_backbone_signal_quality",
                    evidence_metric_means.get("mean_runtime_backbone_signal_quality", 0.0),
                ),
                (
                    "mean_runtime_backbone_signal_strength",
                    evidence_metric_means.get("mean_runtime_backbone_signal_strength", 0.0),
                ),
                ("bounded_writeback_turn_count", evidence_metric_means.get("bounded_writeback_turn_count", 0.0)),
                ("session_post_completion_turn_count", evidence_metric_means.get("session_post_completion_turn_count", 0.0)),
                (
                    "reflection_promotion_eligible_turn_count",
                    evidence_metric_means.get("reflection_promotion_eligible_turn_count", 0.0),
                ),
                (
                    "online_fast_substrate_recommended_count",
                    evidence_metric_means.get("online_fast_substrate_recommended_count", 0.0),
                ),
                (
                    "mean_fast_memory_signal_norm",
                    evidence_metric_means.get("mean_fast_memory_signal_norm", 0.0),
                ),
                (
                    "mean_fast_memory_runtime_alignment",
                    evidence_metric_means.get("mean_fast_memory_runtime_alignment", 0.0),
                ),
                ("rare_heavy_recommended_count", evidence_metric_means.get("rare_heavy_recommended_count", 0.0)),
                ("nested_profile_active_turn_count", evidence_metric_means.get("nested_profile_active_turn_count", 0.0)),
                ("learned_memory_primary_turn_count", evidence_metric_means.get("learned_memory_primary_turn_count", 0.0)),
                ("memory_tower_profile_turn_count", tower_profile_turn_count),
                ("mean_memory_tower_depth", memory_tower_depth),
                ("mean_memory_tower_alignment", memory_tower_alignment),
                ("tower_effective_strength", tower_effective_strength),
                ("tower_depth_gap_vs_best_control", tower_depth_gap_vs_best_control if comparison_metric_deltas else 0.0),
                ("tower_alignment_gap_vs_best_control", tower_alignment_gap_vs_best_control if comparison_metric_deltas else 0.0),
                ("tower_consolidation_gap_vs_best_control", tower_consolidation_gap_vs_best_control if comparison_metric_deltas else 0.0),
            ),
            description="Default path should activate online, background, and nested-memory learning surfaces together.",
        ),
        DialogueNLEssenceGate(
            gate_id="online-fast-pe-coupling",
            passed=online_fast_pe_coupling_passed,
            evidence=(
                ("pe_triggered_turn_count", evidence_metric_means.get("pe_triggered_turn_count", 0.0)),
                (
                    "online_fast_substrate_recommended_count",
                    evidence_metric_means.get("online_fast_substrate_recommended_count", 0.0),
                ),
                (
                    "mean_fast_memory_signal_norm",
                    evidence_metric_means.get("mean_fast_memory_signal_norm", 0.0),
                ),
                (
                    "mean_fast_memory_runtime_alignment",
                    evidence_metric_means.get("mean_fast_memory_runtime_alignment", 0.0),
                ),
            ),
            description="PE-triggered turns should couple to reviewable online-fast substrate evidence with nontrivial fast-memory/runtime alignment.",
        ),
        DialogueNLEssenceGate(
            gate_id="default-continual-learner",
            passed=default_continual_learner_passed,
            evidence=(
                (
                    "mean_default_continual_learning_active",
                    evidence_metric_means.get("mean_default_continual_learning_active", 0.0),
                ),
                (
                    "mean_default_owner_writeback_retained",
                    evidence_metric_means.get("mean_default_owner_writeback_retained", 0.0),
                ),
                (
                    "mean_default_substrate_live_mutation_suppressed",
                    evidence_metric_means.get("mean_default_substrate_live_mutation_suppressed", 0.0),
                ),
                (
                    "online_fast_substrate_applied_count",
                    evidence_metric_means.get("online_fast_substrate_applied_count", 0.0),
                ),
                (
                    "mean_default_continual_rollback_clean",
                    evidence_metric_means.get("mean_default_continual_rollback_clean", 0.0),
                ),
            ),
            description=(
                "Default continual learner should retain owner-side bounded writeback while keeping live "
                "substrate mutation suppressed by default."
            ),
        ),
        DialogueNLEssenceGate(
            gate_id="bounded-live-self-mod",
            passed=bounded_live_self_mod_passed,
            evidence=(
                (
                    "online_fast_substrate_recommended_count",
                    evidence_metric_means.get("online_fast_substrate_recommended_count", 0.0),
                ),
                (
                    "substrate_online_fast_applied",
                    evidence_metric_means.get("mean_substrate_online_fast_applied", 0.0),
                ),
                (
                    "substrate_online_fast_experimental_mode",
                    evidence_metric_means.get("mean_substrate_online_fast_experimental_mode", 0.0),
                ),
                (
                    "substrate_online_fast_review_or_revert_safe",
                    evidence_metric_means.get("mean_substrate_online_fast_review_or_revert_safe", 0.0),
                ),
                (
                    "substrate_online_fast_rollback_integrity",
                    evidence_metric_means.get("mean_substrate_online_fast_rollback_integrity", 0.0),
                ),
            ),
            description="Online-fast substrate self-mod should either stay review-safe or apply through an experimental bounded path with rollback integrity.",
        ),
        DialogueNLEssenceGate(
            gate_id="rare-heavy-net-benefit",
            passed=rare_heavy_net_benefit_passed,
            evidence=(
                ("rare_heavy_recommended_delta_vs_no_rare_heavy", rare_heavy_recommended_delta),
                ("rare_heavy_applied_delta_vs_no_rare_heavy", rare_heavy_applied_delta),
                ("rare_heavy_pre_import_pass_delta_vs_no_rare_heavy", rare_heavy_pre_import_pass_delta),
                ("rare_heavy_pre_import_score_delta_vs_no_rare_heavy", rare_heavy_pre_import_score_delta),
                ("rare_heavy_candidate_alignment_delta_vs_no_rare_heavy", rare_heavy_candidate_alignment_delta),
                ("passed_delta_vs_no_rare_heavy", rare_heavy_passed_delta),
                ("delayed_improvement_delta_vs_no_rare_heavy", rare_heavy_delayed_improvement_delta),
                ("stability_after_recovery_delta_vs_no_rare_heavy", rare_heavy_stability_delta),
                ("mean_prediction_error_delta_vs_no_rare_heavy", rare_heavy_prediction_error_delta),
                ("current_rare_heavy_pre_import_pass_count", current_rare_heavy_pre_import_pass_count),
                ("current_mean_rare_heavy_pre_import_score_delta", current_rare_heavy_pre_import_score_delta),
                ("current_mean_rare_heavy_candidate_alignment", current_rare_heavy_candidate_alignment),
                (
                    "current_max_rare_heavy_candidate_adapter_parameter_count",
                    current_rare_heavy_candidate_adapter_parameter_count,
                ),
                ("review_evidence_present", float(rare_heavy_review_evidence_present)),
                ("failure_mode", rare_heavy_gate_failure_mode),
            ),
            description="Rare-heavy should show a measurable net benefit relative to the pe-eta-no-rare-heavy path.",
        ),
        DialogueNLEssenceGate(
            gate_id="slow-shapes-fast",
            passed=slow_to_fast_passed,
            evidence=(
                ("store_nested_context_reset_count", store_nested_reset_count),
                ("mean_reset_turn_slow_to_fast_init_benefit", reset_turn_slow_to_fast_init_benefit),
                ("mean_reset_turn_slow_to_fast_target_distance_before", reset_turn_target_distance_before),
                ("mean_reset_turn_slow_to_fast_target_distance_after", reset_turn_target_distance_after),
                ("mean_reset_turn_slow_to_fast_target_alignment_gain", reset_turn_target_alignment_gain),
                (
                    "weak_benefit_explained_by_target_proximity",
                    float(weak_benefit_explained_by_target_proximity),
                ),
                ("alignment_interpretation", slow_shapes_fast_alignment_interpretation),
                ("slow_to_fast_signal_strength", slow_to_fast_signal_strength),
                (
                    "boundary_reset_observed_on_first_turn",
                    evidence_metric_means.get("boundary_reset_observed_on_first_turn", 0.0),
                ),
                ("canonical_case_count", float(canonical_case_count)),
                ("longitudinal_case_count", float(longitudinal_case_count)),
                ("proof_min_case_count_satisfied", float(proof_min_case_count_satisfied)),
                ("core_guided_recall_turn_count", evidence_metric_means.get("core_guided_recall_turn_count", 0.0)),
                ("mean_learned_recall_confidence", evidence_metric_means.get("mean_learned_recall_confidence", 0.0)),
                ("max_artifact_consolidation_count", evidence_metric_means.get("max_artifact_consolidation_count", 0.0)),
                ("max_tower_consolidation_count", tower_consolidation_count),
                ("mean_memory_tower_depth", memory_tower_depth),
                ("mean_memory_tower_alignment", memory_tower_alignment),
                ("memory_tower_profile_turn_count", tower_profile_turn_count),
                ("failure_mode", slow_shapes_fast_failure_mode),
            ),
            description="Slow-layer state should seed faster bands through observable nested reset behavior.",
        ),
        DialogueNLEssenceGate(
            gate_id="updater-evidence-visible",
            passed=updater_evidence_visible_passed,
            evidence=(
                (
                    "memory_updater_effective_lr",
                    evidence_metric_means.get("mean_memory_updater_effective_lr", 0.0),
                ),
                (
                    "memory_updater_confidence",
                    evidence_metric_means.get("mean_memory_updater_confidence", 0.0),
                ),
                (
                    "temporal_updater_effective_lr",
                    evidence_metric_means.get("mean_temporal_updater_effective_lr", 0.0),
                ),
                (
                    "temporal_updater_confidence",
                    evidence_metric_means.get("mean_temporal_updater_confidence", 0.0),
                ),
                (
                    "optimizer_memory_drive",
                    evidence_metric_means.get("mean_optimizer_memory_drive", 0.0),
                ),
            ),
            description="Memory and temporal owners should publish machine-readable updater evidence, not just implicit adaptive behavior.",
        ),
        DialogueNLEssenceGate(
            gate_id="tower-memory-surface",
            passed=tower_memory_surface_passed,
            evidence=(
                (
                    "runtime_backbone_evidence_turn_count",
                    evidence_metric_means.get("runtime_backbone_evidence_turn_count", 0.0),
                ),
                (
                    "mean_fast_memory_runtime_alignment",
                    evidence_metric_means.get("mean_fast_memory_runtime_alignment", 0.0),
                ),
                ("memory_tower_profile_turn_count", tower_profile_turn_count),
                ("mean_memory_tower_depth", memory_tower_depth),
                ("mean_memory_tower_alignment", memory_tower_alignment),
                ("max_tower_consolidation_count", tower_consolidation_count),
                ("max_artifact_consolidation_count", artifact_consolidation_count),
                ("tower_consolidation_evidence", tower_consolidation_evidence),
                ("tower_effective_strength", tower_effective_strength),
                ("tower_strength_gap_vs_best_control", best_control_tower_gap),
            ),
            description=(
                "Memory tower should not only be visible on the benchmark path, but also show stronger "
                "depth/alignment/consolidation evidence than matched controls."
            ),
        ),
        DialogueNLEssenceGate(
            gate_id="timescale-contract-retained",
            passed=timescale_contract_retained_passed,
            evidence=(
                (
                    "timescale_contract_retained",
                    evidence_metric_means.get("mean_timescale_contract_retained", 0.0),
                ),
                ("scheduler_discipline", evidence_metric_means.get("mean_scheduler_discipline", 0.0)),
                (
                    "scheduler_substrate_pressure",
                    evidence_metric_means.get("mean_scheduler_substrate_pressure", 0.0),
                ),
                (
                    "scheduler_rare_heavy_pressure",
                    evidence_metric_means.get("mean_scheduler_rare_heavy_pressure", 0.0),
                ),
            ),
            description="The default path should retain a coherent owner-side timescale contract instead of collapsing learning phases together.",
        ),
        DialogueNLEssenceGate(
            gate_id="judge-gated-evolution",
            passed=judge_gated_passed,
            evidence=(
                ("evolution_judge_turn_count", evidence_metric_means.get("evolution_judge_turn_count", 0.0)),
                ("evolution_judge_structural_allow_count", evidence_metric_means.get("evolution_judge_structural_allow_count", 0.0)),
                ("evolution_judge_rollback_count", evidence_metric_means.get("evolution_judge_rollback_count", 0.0)),
            ),
            description="Promote, hold, and rollback decisions should be visible on the default path before structural writeback.",
        ),
        DialogueNLEssenceGate(
            gate_id="cross-session-growth",
            passed=cross_session_passed,
            evidence=(
                ("cross_session_verdict", cross_session_report.verdict if cross_session_report is not None else "missing"),
                (
                    "relationship_continuity_delta",
                    dict(cross_session_report.window_trends[-1][1]).get("relationship_continuity", 0.0)
                    if cross_session_report is not None and cross_session_report.window_trends
                    else 0.0,
                ),
                (
                    "passed_case_gap_vs_baseline",
                    benchmark_report.passed_case_count - int(baseline_metric_means.get("passed", 0.0) * benchmark_report.total_case_count)
                    if baseline_report is not None
                    else float(benchmark_report.passed_case_count),
                ),
            ),
            description="The system should show at least stable cross-session behavior instead of only single-case wins.",
        ),
    )
    passed_gate_count = sum(1 for gate in gates if gate.passed)
    return DialogueNLEssenceAssessmentReport(
        path_label=path_label,
        gates=gates,
        passed_gate_count=passed_gate_count,
        total_gate_count=len(gates),
        description=(
            f"NL essence assessment for {path_label} passed {passed_gate_count}/{len(gates)} gates."
        ),
    )


def evaluate_dialogue_nl_essence_acceptance(
    assessment: DialogueNLEssenceAssessmentReport,
    *,
    config: DialogueNLEssenceAcceptanceConfig | None = None,
) -> DialogueNLEssenceAcceptanceDecision:
    active_config = config or DialogueNLEssenceAcceptanceConfig()
    gate_map = {gate.gate_id: gate for gate in assessment.gates}
    blocked_gate_ids: list[str] = []
    reasons: list[str] = []
    for gate_id in active_config.required_gate_ids:
        gate = gate_map.get(gate_id)
        if gate is None:
            blocked_gate_ids.append(gate_id)
            reasons.append(f"missing-gate:{gate_id}")
            continue
        if not gate.passed:
            blocked_gate_ids.append(gate_id)
            reasons.append(f"failed-gate:{gate_id}")
    if assessment.passed_gate_count < active_config.min_passed_gate_count:
        reasons.append("passed-gate-count-below-threshold")
    accepted = not reasons
    accepted_gate_ids = tuple(gate.gate_id for gate in assessment.gates if gate.passed)
    return DialogueNLEssenceAcceptanceDecision(
        accepted=accepted,
        reasons=tuple(reasons),
        accepted_gate_ids=accepted_gate_ids,
        blocked_gate_ids=tuple(blocked_gate_ids),
        description=(
            f"NL essence acceptance accepted={accepted} "
            f"passed={assessment.passed_gate_count}/{assessment.total_gate_count} "
            f"required={len(active_config.required_gate_ids)}."
        ),
    )


async def run_dialogue_pe_eta_longitudinal_benchmark(
    *,
    cases: tuple[ScriptedDialogueCase, ...] = DEFAULT_DIALOGUE_PROOF_CASES,
    runner_factory: Callable[[], AgentSessionRunner] | None = None,
    allow_interval_carryover_credit: bool = True,
) -> DialogueLongitudinalBenchmarkReport:
    runner = runner_factory() if runner_factory is not None else default_active_runner()
    case_reports: list[DialogueBenchmarkCaseReport] = []
    previous_case: ScriptedDialogueCase | None = None
    for case in cases:
        if previous_case is not None:
            runner.begin_new_context(
                reason=f"dialogue-case-boundary:{previous_case.case_id}->{case.case_id}"
            )
            await runner.drain_session_post_slow_loop()
        case_reports.append(
            await run_dialogue_pe_eta_case(
                case=case,
                runner=runner,
                allow_interval_carryover_credit=allow_interval_carryover_credit,
            )
        )
        previous_case = case
    session_reports = list(runner.completed_session_reports)
    current_report = runner.build_current_session_report()
    if current_report is not None:
        session_reports.append(current_report)
    cross_session_report = runner.evaluation_backbone.run_cross_session_benchmark(
        suite=CrossSessionBenchmarkSuite(session_reports=tuple(session_reports))
    )
    return DialogueLongitudinalBenchmarkReport(
        case_reports=tuple(case_reports),
        session_reports=tuple(session_reports),
        cross_session_report=cross_session_report,
        description=(
            f"Dialogue longitudinal benchmark processed {len(case_reports)} contexts with "
            f"cross_session_verdict={cross_session_report.verdict}."
        ),
    )


def _limit_items(items: tuple[Any, ...], limit: int | None) -> tuple[Any, ...]:
    if limit is None or limit <= 0 or limit >= len(items):
        return items
    return items[:limit]


def _json_normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _comprehensive_stage_order() -> tuple[DialogueComprehensiveStage, ...]:
    return (
        DialogueComprehensiveStage.CANONICAL_ABLATION,
        DialogueComprehensiveStage.LONGITUDINAL,
        DialogueComprehensiveStage.ESSENCE,
        DialogueComprehensiveStage.PERTURBATION,
        DialogueComprehensiveStage.OPEN_ENVIRONMENT,
        DialogueComprehensiveStage.SYSTEMATIC_REPLAY,
        DialogueComprehensiveStage.SELECTION_ARTIFACT,
        DialogueComprehensiveStage.ARTIFACT_COMPARISON,
        DialogueComprehensiveStage.FINAL_REPORT,
    )


def _comprehensive_checkpoint_manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _comprehensive_checkpoint_stage_path(
    output_dir: Path,
    stage: DialogueComprehensiveStage,
) -> Path:
    stage_index = _comprehensive_stage_order().index(stage) + 1
    return output_dir / f"{stage_index:02d}_{stage.value}.pkl"


def _rare_heavy_gate_snapshot(
    assessment: DialogueNLEssenceAssessmentReport,
) -> dict[str, Any]:
    gate = next(
        (candidate for candidate in assessment.gates if candidate.gate_id == "rare-heavy-net-benefit"),
        None,
    )
    if gate is None:
        return {
            "present": False,
            "passed": False,
            "failure_mode": "missing-gate",
        }
    evidence = dict(gate.evidence)
    return {
        "present": True,
        "passed": gate.passed,
        "failure_mode": evidence.get("failure_mode", "passed"),
        "recommended_delta_vs_no_rare_heavy": evidence.get(
            "rare_heavy_recommended_delta_vs_no_rare_heavy", 0.0
        ),
        "applied_delta_vs_no_rare_heavy": evidence.get(
            "rare_heavy_applied_delta_vs_no_rare_heavy", 0.0
        ),
        "passed_delta_vs_no_rare_heavy": evidence.get("passed_delta_vs_no_rare_heavy", 0.0),
        "delayed_improvement_delta_vs_no_rare_heavy": evidence.get(
            "delayed_improvement_delta_vs_no_rare_heavy", 0.0
        ),
        "stability_delta_vs_no_rare_heavy": evidence.get(
            "stability_after_recovery_delta_vs_no_rare_heavy", 0.0
        ),
        "mean_prediction_error_delta_vs_no_rare_heavy": evidence.get(
            "mean_prediction_error_delta_vs_no_rare_heavy", 0.0
        ),
    }


def _emergence_dashboard_snapshot(
    dashboard: DialogueEmergenceDashboardArtifact,
) -> dict[str, Any]:
    return {
        "baseline_label": dashboard.baseline_label,
        "canonical_case_count": dashboard.canonical_case_count,
        "canonical_pass_rate": dashboard.canonical_pass_rate,
        "canonical_mean_memory_tower_depth": dashboard.canonical_mean_memory_tower_depth,
        "canonical_mean_memory_tower_alignment": dashboard.canonical_mean_memory_tower_alignment,
        "canonical_max_tower_consolidation_count": dashboard.canonical_max_tower_consolidation_count,
        "canonical_tower_profile_turn_count": dashboard.canonical_tower_profile_turn_count,
        "canonical_runtime_backbone_evidence_rate": dashboard.canonical_runtime_backbone_evidence_rate,
        "canonical_mean_runtime_backbone_signal_quality": dashboard.canonical_mean_runtime_backbone_signal_quality,
        "canonical_mean_fast_memory_runtime_alignment": dashboard.canonical_mean_fast_memory_runtime_alignment,
        "open_scenario_count": dashboard.open_scenario_count,
        "open_pass_rate": dashboard.open_pass_rate,
        "open_mean_memory_tower_depth": dashboard.open_mean_memory_tower_depth,
        "open_mean_memory_tower_alignment": dashboard.open_mean_memory_tower_alignment,
        "open_max_tower_consolidation_count": dashboard.open_max_tower_consolidation_count,
        "open_runtime_backbone_evidence_rate": dashboard.open_runtime_backbone_evidence_rate,
        "open_mean_runtime_backbone_signal_quality": dashboard.open_mean_runtime_backbone_signal_quality,
        "open_mean_fast_memory_runtime_alignment": dashboard.open_mean_fast_memory_runtime_alignment,
        "strong_proof_panel_count": len(dashboard.strong_proof_panels),
        "open_panel_count": len(dashboard.open_environment_panels),
        "tower_memory_gate_passed": dashboard.tower_memory_gate_passed,
        "tower_memory_gate_strength": dashboard.tower_memory_gate_strength,
        "strongest_scaffold_path_label": dashboard.strongest_scaffold_path_label,
        "strongest_scaffold_retention_score": dashboard.strongest_scaffold_retention_score,
        "strongest_open_path_label": dashboard.strongest_open_path_label,
        "strongest_open_retention_score": dashboard.strongest_open_retention_score,
        "interpretation": dashboard.interpretation,
    }


def build_dialogue_emergence_dashboard_payload(
    comprehensive_report: DialogueComprehensiveBenchmarkReport,
) -> dict[str, Any]:
    dashboard = comprehensive_report.emergence_dashboard
    return {
        "baseline_label": dashboard.baseline_label,
        "profile_labels": comprehensive_report.profile_labels,
        "interpretation": dashboard.interpretation,
        "canonical": {
            "case_count": dashboard.canonical_case_count,
            "pass_rate": dashboard.canonical_pass_rate,
            "mean_memory_tower_depth": dashboard.canonical_mean_memory_tower_depth,
            "mean_memory_tower_alignment": dashboard.canonical_mean_memory_tower_alignment,
            "max_tower_consolidation_count": dashboard.canonical_max_tower_consolidation_count,
            "tower_profile_turn_count": dashboard.canonical_tower_profile_turn_count,
            "runtime_backbone_evidence_rate": dashboard.canonical_runtime_backbone_evidence_rate,
            "mean_runtime_backbone_signal_quality": dashboard.canonical_mean_runtime_backbone_signal_quality,
            "mean_fast_memory_runtime_alignment": dashboard.canonical_mean_fast_memory_runtime_alignment,
            "passed_case_count": comprehensive_report.canonical_ablation_report.path_reports[0].benchmark_report.passed_case_count
            if comprehensive_report.canonical_ablation_report.path_reports
            else 0,
        },
        "open_environment": {
            "scenario_count": dashboard.open_scenario_count,
            "pass_rate": dashboard.open_pass_rate,
            "mean_memory_tower_depth": dashboard.open_mean_memory_tower_depth,
            "mean_memory_tower_alignment": dashboard.open_mean_memory_tower_alignment,
            "max_tower_consolidation_count": dashboard.open_max_tower_consolidation_count,
            "runtime_backbone_evidence_rate": dashboard.open_runtime_backbone_evidence_rate,
            "mean_runtime_backbone_signal_quality": dashboard.open_mean_runtime_backbone_signal_quality,
            "mean_fast_memory_runtime_alignment": dashboard.open_mean_fast_memory_runtime_alignment,
            "present": comprehensive_report.open_ablation_report is not None,
        },
        "strongest_paths": {
            "scaffold": {
                "path_label": dashboard.strongest_scaffold_path_label,
                "retention_score": dashboard.strongest_scaffold_retention_score,
            },
            "open_environment": {
                "path_label": dashboard.strongest_open_path_label,
                "retention_score": dashboard.strongest_open_retention_score,
            },
        },
        "strong_proof_panels": _json_normalize(dashboard.strong_proof_panels),
        "open_environment_panels": _json_normalize(dashboard.open_environment_panels),
        "pe_dominance_report": _json_normalize(dashboard.pe_dominance_report),
        "pe_case_diagnosis_report": _json_normalize(dashboard.pe_case_diagnosis_report),
        "essence": {
            "accepted": comprehensive_report.essence_acceptance.accepted,
            "blocked_gate_ids": comprehensive_report.essence_acceptance.blocked_gate_ids,
            "passed_gate_count": comprehensive_report.essence_report.passed_gate_count,
            "total_gate_count": comprehensive_report.essence_report.total_gate_count,
        },
        "rare_heavy_gate": _rare_heavy_gate_snapshot(comprehensive_report.essence_report),
        "tower_memory_gate": {
            "passed": dashboard.tower_memory_gate_passed,
            "strength": dashboard.tower_memory_gate_strength,
        },
        "description": dashboard.description,
    }


def export_dialogue_emergence_dashboard_artifact(
    comprehensive_report: DialogueComprehensiveBenchmarkReport,
    *,
    output_path: str | Path,
) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dialogue_emergence_dashboard_payload(comprehensive_report)
    target_path.write_text(
        json.dumps(_json_normalize(payload), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return target_path


def _benchmark_pass_rate(passed_case_count: int, total_case_count: int) -> float:
    if total_case_count <= 0:
        return 0.0
    return passed_case_count / total_case_count


def _dialogue_path_report_by_label(
    comparison_report: DialogueBenchmarkComparisonReport,
    profile_label: str,
) -> DialogueBenchmarkPathReport | None:
    return next(
        (path for path in comparison_report.path_reports if path.path_label == profile_label),
        None,
    )


def _open_dialogue_path_report_by_label(
    comparison_report: OpenDialogueBenchmarkComparisonReport | None,
    profile_label: str,
) -> OpenDialogueBenchmarkPathReport | None:
    if comparison_report is None:
        return None
    return next(
        (path for path in comparison_report.path_reports if path.path_label == profile_label),
        None,
    )


def _dialogue_paper_suite_metric_values(
    report: DialogueComprehensiveBenchmarkReport,
) -> tuple[tuple[str, float], ...]:
    canonical_pe_eta = _dialogue_path_report_by_label(report.canonical_ablation_report, "pe-eta")
    canonical_pe_drive_off = _dialogue_path_report_by_label(report.canonical_ablation_report, "pe-drive-off")
    canonical_eta_off = _dialogue_path_report_by_label(report.canonical_ablation_report, "eta-off")
    perturbation_pe_eta = _dialogue_path_report_by_label(report.perturbation_report.ablation_report, "pe-eta")
    open_pe_eta = _open_dialogue_path_report_by_label(report.open_ablation_report, "pe-eta")
    open_pe_drive_off = _open_dialogue_path_report_by_label(report.open_ablation_report, "pe-drive-off")
    chosen_candidate = next(
        (
            candidate
            for candidate in report.artifact_comparison_report.candidate_reports
            if candidate.candidate_label == report.artifact_comparison_report.chosen_candidate_label
        ),
        None,
    )
    rare_heavy_gate = next(
        (gate for gate in report.essence_report.gates if gate.gate_id == "rare-heavy-net-benefit"),
        None,
    )
    tower_memory_gate = next(
        (gate for gate in report.essence_report.gates if gate.gate_id == "tower-memory-surface"),
        None,
    )
    default_continual_gate = next(
        (gate for gate in report.essence_report.gates if gate.gate_id == "default-continual-learner"),
        None,
    )
    canonical_pass_rate_pe_eta = _benchmark_pass_rate(
        canonical_pe_eta.benchmark_report.passed_case_count,
        canonical_pe_eta.benchmark_report.total_case_count,
    ) if canonical_pe_eta is not None else 0.0
    canonical_pass_rate_pe_drive_off = _benchmark_pass_rate(
        canonical_pe_drive_off.benchmark_report.passed_case_count,
        canonical_pe_drive_off.benchmark_report.total_case_count,
    ) if canonical_pe_drive_off is not None else 0.0
    canonical_pass_rate_eta_off = _benchmark_pass_rate(
        canonical_eta_off.benchmark_report.passed_case_count,
        canonical_eta_off.benchmark_report.total_case_count,
    ) if canonical_eta_off is not None else 0.0
    perturbation_pass_rate_pe_eta = _benchmark_pass_rate(
        perturbation_pe_eta.benchmark_report.passed_case_count,
        perturbation_pe_eta.benchmark_report.total_case_count,
    ) if perturbation_pe_eta is not None else 0.0
    open_pass_rate_pe_eta = _benchmark_pass_rate(
        open_pe_eta.benchmark_report.passed_case_count,
        open_pe_eta.benchmark_report.total_case_count,
    ) if open_pe_eta is not None else 0.0
    open_pass_rate_pe_drive_off = _benchmark_pass_rate(
        open_pe_drive_off.benchmark_report.passed_case_count,
        open_pe_drive_off.benchmark_report.total_case_count,
    ) if open_pe_drive_off is not None else 0.0
    essence_gate_pass_fraction = (
        report.essence_report.passed_gate_count / report.essence_report.total_gate_count
        if report.essence_report.total_gate_count > 0
        else 0.0
    )
    return (
        ("canonical_pass_rate_pe_eta", canonical_pass_rate_pe_eta),
        ("canonical_pass_rate_gap_vs_pe_drive_off", canonical_pass_rate_pe_eta - canonical_pass_rate_pe_drive_off),
        ("canonical_pass_rate_gap_vs_eta_off", canonical_pass_rate_pe_eta - canonical_pass_rate_eta_off),
        ("perturbation_pass_rate_pe_eta", perturbation_pass_rate_pe_eta),
        ("open_pass_rate_pe_eta", open_pass_rate_pe_eta),
        ("open_pass_rate_gap_vs_pe_drive_off", open_pass_rate_pe_eta - open_pass_rate_pe_drive_off),
        ("essence_gate_pass_fraction", essence_gate_pass_fraction),
        (
            "canonical_runtime_backbone_evidence_rate",
            report.emergence_dashboard.canonical_runtime_backbone_evidence_rate,
        ),
        (
            "canonical_mean_runtime_backbone_signal_quality",
            report.emergence_dashboard.canonical_mean_runtime_backbone_signal_quality,
        ),
        (
            "canonical_mean_fast_memory_runtime_alignment",
            report.emergence_dashboard.canonical_mean_fast_memory_runtime_alignment,
        ),
        (
            "default_continual_learning_active_rate",
            canonical_pe_eta.benchmark_report.metric_means
            and dict(canonical_pe_eta.benchmark_report.metric_means).get(
                "mean_default_continual_learning_active",
                0.0,
            )
            if canonical_pe_eta is not None
            else 0.0,
        ),
        (
            "default_owner_writeback_retention",
            canonical_pe_eta.benchmark_report.metric_means
            and dict(canonical_pe_eta.benchmark_report.metric_means).get(
                "mean_default_owner_writeback_retained",
                0.0,
            )
            if canonical_pe_eta is not None
            else 0.0,
        ),
        (
            "default_substrate_mutation_suppression",
            canonical_pe_eta.benchmark_report.metric_means
            and dict(canonical_pe_eta.benchmark_report.metric_means).get(
                "mean_default_substrate_live_mutation_suppressed",
                0.0,
            )
            if canonical_pe_eta is not None
            else 0.0,
        ),
        (
            "strongest_scaffold_retention_score",
            report.emergence_dashboard.strongest_scaffold_retention_score,
        ),
        (
            "canonical_mean_memory_tower_depth",
            report.emergence_dashboard.canonical_mean_memory_tower_depth,
        ),
        (
            "canonical_mean_memory_tower_alignment",
            report.emergence_dashboard.canonical_mean_memory_tower_alignment,
        ),
        (
            "strongest_open_retention_score",
            report.emergence_dashboard.strongest_open_retention_score,
        ),
        (
            "artifact_candidate_mean_score_delta",
            chosen_candidate.acceptance_report.mean_score_delta if chosen_candidate is not None else 0.0,
        ),
        ("rare_heavy_gate_pass", float(rare_heavy_gate.passed) if rare_heavy_gate is not None else 0.0),
        ("tower_memory_gate_pass", float(tower_memory_gate.passed) if tower_memory_gate is not None else 0.0),
        ("default_continual_learner_gate_pass", float(default_continual_gate.passed) if default_continual_gate is not None else 0.0),
        ("tower_memory_gate_strength", report.emergence_dashboard.tower_memory_gate_strength),
    )


def _dialogue_metric_samples(
    *,
    run_summaries: tuple[DialoguePaperSuiteRunSummary, ...],
    metric_names: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    summary_maps = [dict(summary.metric_values) for summary in run_summaries]
    return {
        metric_name: tuple(summary_map.get(metric_name, 0.0) for summary_map in summary_maps)
        for metric_name in metric_names
    }


def _repo_root_from_dialogue_module() -> Path:
    return Path(__file__).resolve().parents[2]


async def run_dialogue_paper_suite_repeated_benchmark(
    *,
    manifest: PaperSuiteManifest | None = None,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.BUILTIN_ONLY,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> DialoguePaperSuiteAggregateReport:
    active_manifest = manifest or build_dialogue_paper_suite_manifest()
    active_config = dialogue_paper_suite_config(
        active_manifest,
        runtime_mode=runtime_mode,
    )
    run_summaries: list[DialoguePaperSuiteRunSummary] = []
    reference_report: DialogueComprehensiveBenchmarkReport | None = None
    for run_index, run_seed in enumerate(active_manifest.seed_schedule[: active_manifest.repeat_count], start=1):
        run_config = replace(
            active_config,
            replay_seeds=(run_seed,),
        )
        if progress_callback is not None:
            progress_callback(
                f"Dialogue paper suite run {run_index}/{active_manifest.repeat_count} (seed={run_seed}) started."
            )
        if output_dir is not None:
            run_output_dir = Path(output_dir) / f"dialogue_run_{run_index:02d}_seed_{run_seed}"
            report = await run_real_dialogue_pe_eta_comprehensive_benchmark_staged(
                output_dir=run_output_dir,
                config=run_config,
                essence_acceptance_config=DialogueNLEssenceAcceptanceConfig(),
                progress_callback=progress_callback,
                resume=True,
            )
            export_dialogue_emergence_dashboard_artifact(
                report,
                output_path=run_output_dir / "emergence_dashboard.json",
            )
        else:
            report = await run_real_dialogue_pe_eta_comprehensive_benchmark(
                config=run_config,
                essence_acceptance_config=DialogueNLEssenceAcceptanceConfig(),
                progress_callback=progress_callback,
            )
        reference_report = reference_report or report
        metric_values = _dialogue_paper_suite_metric_values(report)
        run_summary = DialoguePaperSuiteRunSummary(
            run_id=f"{active_manifest.suite_id}:run-{run_index:02d}",
            run_seed=run_seed,
            metric_values=metric_values,
            description=(
                f"Dialogue paper suite run {run_index} summarized "
                f"{len(metric_values)} metrics for seed={run_seed}."
            ),
        )
        run_summaries.append(run_summary)
        if output_dir is not None:
            export_json_artifact(
                payload=run_summary,
                output_path=Path(output_dir) / f"dialogue_run_{run_index:02d}_summary.json",
            )
    primary_metric_names = tuple(metric.metric_name for metric in active_manifest.primary_metrics)
    secondary_metric_names = tuple(metric.metric_name for metric in active_manifest.secondary_metrics)
    primary_metric_summaries = build_metric_interval_summaries(
        metric_samples=_dialogue_metric_samples(
            run_summaries=tuple(run_summaries),
            metric_names=primary_metric_names,
        )
    )
    secondary_metric_summaries = build_metric_interval_summaries(
        metric_samples=_dialogue_metric_samples(
            run_summaries=tuple(run_summaries),
            metric_names=secondary_metric_names,
        )
    )
    provenance = collect_paper_suite_provenance(
        manifest=active_manifest,
        repo_root=_repo_root_from_dialogue_module(),
        runtime_descriptor={
            "runtime_mode": str(runtime_mode),
            "model_id": active_config.model_id,
            "model_source": active_config.model_source or "",
            "fallback_mode": (
                str(active_config.fallback_mode)
                if active_config.fallback_mode is not None
                else ""
            ),
            "suite_kind": active_manifest.suite_kind,
        },
    )
    pairwise_effects = _build_dialogue_paper_suite_pairwise_effects(tuple(run_summaries))
    aggregate_report = DialoguePaperSuiteAggregateReport(
        manifest=active_manifest,
        provenance=provenance,
        run_summaries=tuple(run_summaries),
        reference_run_report=reference_report,
        primary_metric_summaries=primary_metric_summaries,
        secondary_metric_summaries=secondary_metric_summaries,
        description=(
            f"Dialogue paper suite {active_manifest.suite_id} aggregated "
            f"{len(run_summaries)} repeated runs."
        ),
        pairwise_effects=pairwise_effects,
        claim_verdicts=(),
    )
    claim_verdicts = _build_dialogue_claim_verdicts(
        aggregate_report=aggregate_report,
        human_ratings_aggregate=None,
    )
    return replace(
        aggregate_report,
        claim_verdicts=claim_verdicts,
    )


_DEFAULT_EXPERT_REVIEW_DIMENSIONS: tuple[DialogueExpertReviewDimension, ...] = (
    DialogueExpertReviewDimension(
        dimension_id="relationship_continuity",
        prompt="Rate which transcript better preserves trust, warmth, and continuity across turns.",
        description="External review dimension for relational continuity.",
    ),
    DialogueExpertReviewDimension(
        dimension_id="delayed_stabilization",
        prompt="Rate which transcript better turns early pressure into later stabilization or recovery.",
        description="External review dimension for delayed stabilization.",
    ),
    DialogueExpertReviewDimension(
        dimension_id="boundary_correctness",
        prompt="Rate which transcript better stays measured and appropriately bounded under pressure.",
        description="External review dimension for boundary correctness.",
    ),
)


def build_dialogue_expert_review_packet(
    report: DialogueComprehensiveBenchmarkReport,
    *,
    packet_id: str = "dialogue-expert-review",
    candidate_labels: tuple[str, ...] = ("pe-eta", "pe-drive-off", "eta-off"),
) -> DialogueExpertReviewPacket:
    packet, _ = _build_dialogue_expert_review_materials(
        report,
        packet_id=packet_id,
        candidate_labels=candidate_labels,
    )
    return packet


def build_dialogue_expert_review_internal_key(
    report: DialogueComprehensiveBenchmarkReport,
    *,
    packet_id: str = "dialogue-expert-review",
    candidate_labels: tuple[str, ...] = ("pe-eta", "pe-drive-off", "eta-off"),
) -> DialogueExpertReviewInternalKey:
    _, internal_key = _build_dialogue_expert_review_materials(
        report,
        packet_id=packet_id,
        candidate_labels=candidate_labels,
    )
    return internal_key


def _build_dialogue_expert_review_materials(
    report: DialogueComprehensiveBenchmarkReport,
    *,
    packet_id: str,
    candidate_labels: tuple[str, ...],
) -> tuple[DialogueExpertReviewPacket, DialogueExpertReviewInternalKey]:
    canonical_map = {
        path.path_label: path.benchmark_report
        for path in report.canonical_ablation_report.path_reports
        if path.path_label in candidate_labels
    }
    items: list[DialogueExpertReviewItem] = []
    key_entries: list[DialogueExpertReviewInternalKeyEntry] = []
    if not canonical_map:
        packet = DialogueExpertReviewPacket(
            packet_id=packet_id,
            source_suite_id=report.profile_labels[0] if report.profile_labels else "dialogue",
            items=(),
            review_dimensions=_DEFAULT_EXPERT_REVIEW_DIMENSIONS,
            description="Dialogue expert review packet has no matching candidate labels.",
        )
        return (
            packet,
            DialogueExpertReviewInternalKey(
                packet_id=packet_id,
                baseline_label=report.canonical_ablation_report.baseline_label,
                entries=(),
                description="Dialogue expert review internal key has no entries.",
            ),
        )
    baseline_benchmark = canonical_map.get(report.canonical_ablation_report.baseline_label)
    ordered_case_ids = tuple(
        case_report.case.case_id
        for case_report in (
            baseline_benchmark.case_reports if baseline_benchmark is not None else next(iter(canonical_map.values())).case_reports
        )
    )
    blinded_labels = {
        profile_label: f"sample_{chr(ord('A') + index)}"
        for index, profile_label in enumerate(candidate_labels)
    }
    for item_index, case_id in enumerate(ordered_case_ids, start=1):
        samples: list[DialogueExpertReviewSample] = []
        for profile_label in candidate_labels:
            benchmark_report = canonical_map.get(profile_label)
            if benchmark_report is None:
                continue
            case_report = next(
                (candidate for candidate in benchmark_report.case_reports if candidate.case.case_id == case_id),
                None,
            )
            if case_report is None:
                continue
            transcript = tuple(
                (turn.user_input, turn.assistant_response_text)
                for turn in case_report.turns
            )
            sample_id = f"item_{item_index:02d}:{blinded_labels[profile_label]}"
            samples.append(
                DialogueExpertReviewSample(
                    sample_id=sample_id,
                    blinded_label=blinded_labels[profile_label],
                    transcript=transcript,
                    description=(
                        f"Blinded transcript sample {blinded_labels[profile_label]} "
                        f"for hidden review item {item_index:02d}."
                    ),
                )
            )
            key_entries.append(
                DialogueExpertReviewInternalKeyEntry(
                    item_id=f"item_{item_index:02d}",
                    sample_id=sample_id,
                    blinded_label=blinded_labels[profile_label],
                    source_case_id=case_id,
                    source_profile_label=profile_label,
                    description=(
                        f"Internal key for hidden review item {item_index:02d}, "
                        f"sample {blinded_labels[profile_label]}."
                    ),
                )
            )
        items.append(
            DialogueExpertReviewItem(
                item_id=f"item_{item_index:02d}",
                prompt_context="Compare the blinded transcripts for the same hidden multi-turn dialogue scenario.",
                samples=tuple(samples),
                review_dimensions=_DEFAULT_EXPERT_REVIEW_DIMENSIONS,
                description=(
                    f"Expert review item {item_index:02d} with {len(samples)} blinded samples."
                ),
            )
        )
    packet = DialogueExpertReviewPacket(
        packet_id=packet_id,
        source_suite_id="dialogue-paper-suite",
        items=tuple(items),
        review_dimensions=_DEFAULT_EXPERT_REVIEW_DIMENSIONS,
        description=(
            f"Dialogue expert review packet built from {len(items)} hidden canonical items."
        ),
    )
    internal_key = DialogueExpertReviewInternalKey(
        packet_id=packet_id,
        baseline_label=report.canonical_ablation_report.baseline_label,
        entries=tuple(key_entries),
        description=(
            f"Dialogue expert review internal key maps {len(key_entries)} blinded samples "
            "back to case/profile labels."
        ),
    )
    return (packet, internal_key)


def export_dialogue_expert_review_packet(
    packet: DialogueExpertReviewPacket,
    *,
    output_path: str | Path,
) -> Path:
    return export_json_artifact(payload=packet, output_path=output_path)


def export_dialogue_expert_review_internal_key(
    internal_key: DialogueExpertReviewInternalKey,
    *,
    output_path: str | Path,
) -> Path:
    return export_json_artifact(payload=internal_key, output_path=output_path)


def build_dialogue_human_rating_template(
    packet: DialogueExpertReviewPacket,
    *,
    min_rater_count: int = 3,
    scale_min: int = 1,
    scale_max: int = 5,
) -> DialogueHumanRatingTemplate:
    rows = tuple(
        (item.item_id, sample.sample_id, sample.blinded_label, dimension.dimension_id)
        for item in packet.items
        for sample in item.samples
        for dimension in packet.review_dimensions
    )
    return DialogueHumanRatingTemplate(
        packet_id=packet.packet_id,
        min_rater_count=min_rater_count,
        scale_min=scale_min,
        scale_max=scale_max,
        dimensions=packet.review_dimensions,
        rows=rows,
        description=(
            f"Human rating template for {packet.packet_id} with {len(rows)} blinded sample-dimension rows."
        ),
    )


def export_dialogue_human_rating_template(
    template: DialogueHumanRatingTemplate,
    *,
    json_output_path: str | Path,
    csv_output_path: str | Path | None = None,
) -> tuple[Path, ...]:
    written_paths = [
        export_json_artifact(payload=template, output_path=json_output_path),
    ]
    if csv_output_path is not None:
        target_path = Path(csv_output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        csv_lines = [
            "rater_id,item_id,sample_id,blinded_label,dimension_id,score",
            *(
                f",{item_id},{sample_id},{blinded_label},{dimension_id},"
                for item_id, sample_id, blinded_label, dimension_id in template.rows
            ),
        ]
        target_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        written_paths.append(target_path)
    return tuple(written_paths)


def aggregate_dialogue_human_ratings(
    *,
    packet: DialogueExpertReviewPacket,
    entries: tuple[DialogueHumanRatingEntry, ...],
    internal_key: DialogueExpertReviewInternalKey | None = None,
    reference_report: DialogueComprehensiveBenchmarkReport | None = None,
) -> DialogueHumanRatingsAggregate:
    sample_auto_scores = _dialogue_review_sample_automatic_scores(
        internal_key=internal_key,
        reference_report=reference_report,
    )
    rater_ids = tuple(sorted({entry.rater_id for entry in entries}))
    dimension_aggregates: list[DialogueHumanRatingDimensionAggregate] = []
    for dimension in packet.review_dimensions:
        dimension_entries = tuple(entry for entry in entries if entry.dimension_id == dimension.dimension_id)
        scores = tuple(entry.score for entry in dimension_entries)
        automatic_scores = tuple(
            sample_auto_scores.get(entry.sample_id, 0.0)
            for entry in dimension_entries
        )
        dimension_aggregates.append(
            DialogueHumanRatingDimensionAggregate(
                dimension_id=dimension.dimension_id,
                mean_score=_mean(scores),
                variance=_variance(scores),
                sample_count=len(scores),
                mean_automatic_score=_mean(automatic_scores),
                correlation_with_automatic=_pearson_correlation(scores, automatic_scores),
                description=(
                    f"Human rating aggregate for {dimension.dimension_id} with {len(scores)} scored samples."
                ),
            )
        )
    return DialogueHumanRatingsAggregate(
        packet_id=packet.packet_id,
        entry_count=len(entries),
        rater_count=len(rater_ids),
        inter_rater_agreement=_dialogue_inter_rater_agreement(entries),
        dimensions=tuple(dimension_aggregates),
        description=(
            f"Dialogue human rating aggregate summarized {len(entries)} ratings "
            f"from {len(rater_ids)} raters."
        ),
    )


def export_dialogue_paper_suite_artifact_bundle(
    aggregate_report: DialoguePaperSuiteAggregateReport,
    *,
    output_dir: str | Path,
    human_ratings_aggregate: DialogueHumanRatingsAggregate | None = None,
) -> tuple[Path, ...]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written_paths = [
        export_json_artifact(
            payload=aggregate_report.manifest,
            output_path=target_dir / "paper_suite_manifest.json",
        ),
        export_json_artifact(
            payload=aggregate_report.provenance,
            output_path=target_dir / "paper_suite_provenance.json",
        ),
        export_json_artifact(
            payload=aggregate_report.run_summaries,
            output_path=target_dir / "paper_suite_run_summaries.json",
        ),
        export_json_artifact(
            payload={
                "suite_id": aggregate_report.manifest.suite_id,
                "primary_metric_summaries": aggregate_report.primary_metric_summaries,
                "secondary_metric_summaries": aggregate_report.secondary_metric_summaries,
                "pairwise_effects": aggregate_report.pairwise_effects,
                "claim_verdicts": aggregate_report.claim_verdicts,
                "description": aggregate_report.description,
            },
            output_path=target_dir / "paper_suite_aggregate.json",
        ),
    ]
    blind_review_packet: DialogueExpertReviewPacket | None = None
    if aggregate_report.reference_run_report is not None:
        written_paths.append(
            export_dialogue_emergence_dashboard_artifact(
                aggregate_report.reference_run_report,
                output_path=target_dir / "reference_emergence_dashboard.json",
            )
        )
        expert_review_packet = build_dialogue_expert_review_packet(
            aggregate_report.reference_run_report,
            packet_id=f"{aggregate_report.manifest.suite_id}:expert-review",
        )
        blind_review_packet = expert_review_packet
        expert_review_internal_key = build_dialogue_expert_review_internal_key(
            aggregate_report.reference_run_report,
            packet_id=f"{aggregate_report.manifest.suite_id}:expert-review",
        )
        written_paths.append(
            export_dialogue_expert_review_packet(
                expert_review_packet,
                output_path=target_dir / "expert_review_packet_blinded.json",
            )
        )
        written_paths.append(
            export_dialogue_expert_review_internal_key(
                expert_review_internal_key,
                output_path=target_dir / "expert_review_key_internal.json",
            )
        )
        written_paths.extend(
            export_dialogue_human_rating_template(
                build_dialogue_human_rating_template(expert_review_packet),
                json_output_path=target_dir / "human_rating_template.json",
                csv_output_path=target_dir / "human_rating_template.csv",
            )
        )
        if human_ratings_aggregate is not None:
            written_paths.append(
                export_json_artifact(
                    payload=human_ratings_aggregate,
                    output_path=target_dir / "human_ratings_aggregate.json",
                )
            )
    evidence_bundle = build_dialogue_paper_suite_evidence_bundle(
        aggregate_report=aggregate_report,
        blind_review_packet=blind_review_packet,
        human_ratings_aggregate=human_ratings_aggregate,
    )
    written_paths.append(
        export_json_artifact(
            payload=evidence_bundle,
            output_path=target_dir / "evidence_bundle.json",
        )
    )
    return tuple(written_paths)


def _variance(values: tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _pearson_correlation(xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    if len(xs) != len(ys) or len(xs) <= 1:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 1e-8 or denom_y <= 1e-8:
        return 0.0
    return numerator / ((denom_x * denom_y) ** 0.5)


def _dialogue_inter_rater_agreement(entries: tuple[DialogueHumanRatingEntry, ...]) -> float:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for entry in entries:
        grouped.setdefault(
            (entry.item_id, entry.sample_id, entry.dimension_id),
            [],
        ).append(entry.score)
    agreements: list[float] = []
    for scores in grouped.values():
        if len(scores) <= 1:
            continue
        pairwise_diffs = [
            abs(left - right)
            for index, left in enumerate(scores)
            for right in scores[index + 1 :]
        ]
        if not pairwise_diffs:
            continue
        agreements.append(max(0.0, 1.0 - (_mean(tuple(pairwise_diffs)) / 4.0)))
    return round(_mean(tuple(agreements)), 4)


def _dialogue_review_sample_automatic_scores(
    *,
    internal_key: DialogueExpertReviewInternalKey | None,
    reference_report: DialogueComprehensiveBenchmarkReport | None,
) -> dict[str, float]:
    if internal_key is None or reference_report is None:
        return {}
    path_map = {
        path.path_label: path.benchmark_report
        for path in reference_report.canonical_ablation_report.path_reports
    }
    scores: dict[str, float] = {}
    for entry in internal_key.entries:
        benchmark_report = path_map.get(entry.source_profile_label)
        if benchmark_report is None:
            continue
        case_report = next(
            (
                candidate
                for candidate in benchmark_report.case_reports
                if candidate.case.case_id == entry.source_case_id
            ),
            None,
        )
        if case_report is None:
            continue
        scores[entry.sample_id] = case_report.strong_success_score
    return scores


def _dialogue_metric_map(items: tuple[tuple[str, float], ...]) -> dict[str, float]:
    return dict(items)


def _dialogue_pairwise_metric_effect(
    *,
    run_summaries: tuple[DialoguePaperSuiteRunSummary, ...],
    metric_name: str,
    candidate_label: str,
    control_label: str,
    candidate_metric_name: str,
    gap_metric_name: str,
) -> PairwiseMetricEffect:
    metric_maps = tuple(_dialogue_metric_map(summary.metric_values) for summary in run_summaries)
    candidate_values = tuple(metric_map.get(candidate_metric_name, 0.0) for metric_map in metric_maps)
    control_values = tuple(
        metric_map.get(candidate_metric_name, 0.0) - metric_map.get(gap_metric_name, 0.0)
        for metric_map in metric_maps
    )
    return build_pairwise_metric_effect(
        metric_name=metric_name,
        candidate_label=candidate_label,
        control_label=control_label,
        candidate_values=candidate_values,
        control_values=control_values,
    )


def _build_dialogue_paper_suite_pairwise_effects(
    run_summaries: tuple[DialoguePaperSuiteRunSummary, ...],
) -> tuple[PairwiseMetricEffect, ...]:
    if not run_summaries:
        return ()
    return (
        _dialogue_pairwise_metric_effect(
            run_summaries=run_summaries,
            metric_name="canonical_pass_rate",
            candidate_label="pe-eta",
            control_label="pe-drive-off",
            candidate_metric_name="canonical_pass_rate_pe_eta",
            gap_metric_name="canonical_pass_rate_gap_vs_pe_drive_off",
        ),
        _dialogue_pairwise_metric_effect(
            run_summaries=run_summaries,
            metric_name="canonical_pass_rate",
            candidate_label="pe-eta",
            control_label="eta-off",
            candidate_metric_name="canonical_pass_rate_pe_eta",
            gap_metric_name="canonical_pass_rate_gap_vs_eta_off",
        ),
        _dialogue_pairwise_metric_effect(
            run_summaries=run_summaries,
            metric_name="open_pass_rate",
            candidate_label="pe-eta",
            control_label="pe-drive-off",
            candidate_metric_name="open_pass_rate_pe_eta",
            gap_metric_name="open_pass_rate_gap_vs_pe_drive_off",
        ),
    )


def _dialogue_claim_status(*, retain_checks: tuple[bool, ...], weak_checks: tuple[bool, ...] = ()) -> str:
    if retain_checks and all(retain_checks):
        return "retain"
    if weak_checks and all(weak_checks):
        return "weak"
    if retain_checks and any(retain_checks):
        return "weak"
    return "fail"


def _build_dialogue_claim_verdicts(
    *,
    aggregate_report: DialoguePaperSuiteAggregateReport,
    human_ratings_aggregate: DialogueHumanRatingsAggregate | None,
) -> tuple[ClaimVerdict, ...]:
    reference_report = aggregate_report.reference_run_report
    gate_map = {
        gate.gate_id: gate.passed
        for gate in reference_report.essence_report.gates
    } if reference_report is not None else {}
    pairwise_map = {
        (effect.metric_name, effect.control_label): effect
        for effect in aggregate_report.pairwise_effects
    }
    open_scenarios = (
        tuple(
            case_report.scenario
            for path in reference_report.open_ablation_report.path_reports[:1]
            for case_report in path.benchmark_report.case_reports
        )
        if reference_report is not None and reference_report.open_ablation_report is not None
        else ()
    )
    claim_a_status = _dialogue_claim_status(
        retain_checks=(
            gate_map.get("pe-first", False),
            gate_map.get("multi-timescale-default", False),
            gate_map.get("default-continual-learner", False),
            gate_map.get("judge-gated-evolution", False),
            gate_map.get("cross-session-growth", False),
        ),
    )
    default_continual_summary = next(
        (
            summary
            for summary in aggregate_report.primary_metric_summaries
            if summary.metric_name == "default_continual_learning_active_rate"
        ),
        None,
    )
    owner_writeback_summary = next(
        (
            summary
            for summary in aggregate_report.primary_metric_summaries
            if summary.metric_name == "default_owner_writeback_retention"
        ),
        None,
    )
    substrate_suppression_summary = next(
        (
            summary
            for summary in aggregate_report.primary_metric_summaries
            if summary.metric_name == "default_substrate_mutation_suppression"
        ),
        None,
    )
    claim_default_continual_status = _dialogue_claim_status(
        retain_checks=(
            gate_map.get("default-continual-learner", False),
            default_continual_summary is not None and default_continual_summary.mean > 0.0,
            owner_writeback_summary is not None and owner_writeback_summary.mean >= 0.5,
            substrate_suppression_summary is not None and substrate_suppression_summary.mean >= 0.95,
        ),
        weak_checks=(
            gate_map.get("multi-timescale-default", False),
            default_continual_summary is not None and default_continual_summary.mean > 0.0,
            substrate_suppression_summary is not None and substrate_suppression_summary.mean >= 0.8,
        ),
    )
    canonical_pe_drive = pairwise_map.get(("canonical_pass_rate", "pe-drive-off"))
    canonical_eta_off = pairwise_map.get(("canonical_pass_rate", "eta-off"))
    claim_b_status = _dialogue_claim_status(
        retain_checks=(
            canonical_pe_drive is not None and canonical_pe_drive.ci_low > 0.0,
            canonical_eta_off is not None and canonical_eta_off.ci_low > 0.0,
        ),
        weak_checks=(
            canonical_pe_drive is not None and canonical_pe_drive.mean_delta > 0.0,
            canonical_eta_off is not None and canonical_eta_off.mean_delta > 0.0,
        ),
    )
    open_pe_drive = pairwise_map.get(("open_pass_rate", "pe-drive-off"))
    has_heldout = any(scenario.split == "open_heldout" for scenario in open_scenarios)
    perturbation_summary = next(
        (
            summary
            for summary in aggregate_report.primary_metric_summaries
            if summary.metric_name == "perturbation_pass_rate_pe_eta"
        ),
        None,
    )
    claim_c_status = _dialogue_claim_status(
        retain_checks=(
            open_pe_drive is not None and open_pe_drive.ci_low > 0.0,
            has_heldout,
            perturbation_summary is not None and perturbation_summary.mean > 0.0,
        ),
        weak_checks=(
            open_pe_drive is not None and open_pe_drive.mean_delta > 0.0,
            perturbation_summary is not None and perturbation_summary.mean > 0.0,
        ),
    )
    agreement = human_ratings_aggregate.inter_rater_agreement if human_ratings_aggregate is not None else 0.0
    correlation_ok = (
        human_ratings_aggregate is not None
        and any(
            dimension.correlation_with_automatic > 0.1
            for dimension in human_ratings_aggregate.dimensions
        )
    )
    claim_d_status = _dialogue_claim_status(
        retain_checks=(
            human_ratings_aggregate is not None and human_ratings_aggregate.rater_count >= 3,
            agreement >= 0.6,
            correlation_ok,
        ),
        weak_checks=(
            human_ratings_aggregate is not None and human_ratings_aggregate.rater_count >= 2,
            agreement >= 0.4,
        ),
    )
    return (
        ClaimVerdict(
            claim_id="claim_pe_multi_timescale_default",
            status=claim_a_status,
            required_gate_ids=(
                "pe-first",
                "multi-timescale-default",
                "default-continual-learner",
                "judge-gated-evolution",
                "cross-session-growth",
            ),
            supporting_artifacts=("paper_suite_aggregate", "reference_emergence_dashboard"),
            evidence=(
                ("pe-first", float(gate_map.get("pe-first", False))),
                ("multi-timescale-default", float(gate_map.get("multi-timescale-default", False))),
                ("default-continual-learner", float(gate_map.get("default-continual-learner", False))),
                ("judge-gated-evolution", float(gate_map.get("judge-gated-evolution", False))),
                ("cross-session-growth", float(gate_map.get("cross-session-growth", False))),
            ),
            summary="PE-first 与多时间尺度默认路径 claim verdict.",
            description="Claim A checks whether the default path retains the required PE-first and multi-timescale gates.",
        ),
        ClaimVerdict(
            claim_id="claim_default_continual_learner_without_live_substrate_mutation",
            status=claim_default_continual_status,
            required_gate_ids=("default-continual-learner",),
            supporting_artifacts=("paper_suite_aggregate", "reference_emergence_dashboard"),
            evidence=(
                (
                    "default_continual_learning_active_rate",
                    default_continual_summary.mean if default_continual_summary is not None else 0.0,
                ),
                (
                    "default_owner_writeback_retention",
                    owner_writeback_summary.mean if owner_writeback_summary is not None else 0.0,
                ),
                (
                    "default_substrate_mutation_suppression",
                    substrate_suppression_summary.mean if substrate_suppression_summary is not None else 0.0,
                ),
            ),
            summary="默认 continual learner 在无 live substrate mutation 下保留的 claim verdict.",
            description=(
                "Claim Default-CL checks that owner-side memory/temporal/regime/reflection learning is retained "
                "while live substrate mutation remains suppressed by default."
            ),
        ),
        ClaimVerdict(
            claim_id="claim_temporal_advantage_over_controls",
            status=claim_b_status,
            required_gate_ids=(),
            supporting_artifacts=("paper_suite_aggregate",),
            evidence=(
                ("canonical_gap_vs_pe_drive_off_ci_low", canonical_pe_drive.ci_low if canonical_pe_drive is not None else 0.0),
                ("canonical_gap_vs_pe_drive_off_mean_delta", canonical_pe_drive.mean_delta if canonical_pe_drive is not None else 0.0),
                ("canonical_gap_vs_eta_off_ci_low", canonical_eta_off.ci_low if canonical_eta_off is not None else 0.0),
                ("canonical_gap_vs_eta_off_mean_delta", canonical_eta_off.mean_delta if canonical_eta_off is not None else 0.0),
            ),
            summary="Matched-control 优势 claim verdict.",
            description="Claim B checks whether PE-ETA retains positive pairwise effects over matched controls across repeated runs.",
        ),
        ClaimVerdict(
            claim_id="claim_beyond_scripted_canonical",
            status=claim_c_status,
            required_gate_ids=(),
            supporting_artifacts=("paper_suite_aggregate", "reference_emergence_dashboard"),
            evidence=(
                ("open_gap_vs_pe_drive_off_ci_low", open_pe_drive.ci_low if open_pe_drive is not None else 0.0),
                ("open_gap_vs_pe_drive_off_mean_delta", open_pe_drive.mean_delta if open_pe_drive is not None else 0.0),
                ("has_open_heldout", float(has_heldout)),
                ("perturbation_pass_rate_mean", perturbation_summary.mean if perturbation_summary is not None else 0.0),
            ),
            summary="超出 canonical scripted 的 widening claim verdict.",
            description="Claim C checks whether the retained advantage extends to perturbation and held-out open-environment surfaces.",
        ),
        ClaimVerdict(
            claim_id="claim_external_human_legibility",
            status=claim_d_status,
            required_gate_ids=(),
            supporting_artifacts=("expert_review_packet_blinded", "human_ratings_aggregate"),
            evidence=(
                ("rater_count", float(human_ratings_aggregate.rater_count) if human_ratings_aggregate is not None else 0.0),
                ("inter_rater_agreement", agreement),
                ("auto_correlation_observed", float(correlation_ok)),
            ),
            summary="外部人评可见性 claim verdict.",
            description="Claim D checks whether blinded human review has enough rater coverage, agreement, and alignment with automatic evidence.",
        ),
    )


def build_dialogue_paper_suite_evidence_bundle(
    *,
    aggregate_report: DialoguePaperSuiteAggregateReport,
    blind_review_packet: DialogueExpertReviewPacket | None = None,
    human_ratings_aggregate: DialogueHumanRatingsAggregate | None = None,
) -> EvidenceBundle:
    reference_artifacts: list[tuple[str, Any]] = []
    if aggregate_report.reference_run_report is not None:
        reference_artifacts.append(
            ("reference_emergence_dashboard", aggregate_report.reference_run_report.emergence_dashboard)
        )
    return EvidenceBundle(
        bundle_id=f"{aggregate_report.manifest.suite_id}:evidence-bundle",
        suite_kind=aggregate_report.manifest.suite_kind,
        manifest=aggregate_report.manifest,
        provenance=aggregate_report.provenance,
        run_summaries=aggregate_report.run_summaries,
        aggregate_metrics={
            "primary_metric_summaries": aggregate_report.primary_metric_summaries,
            "secondary_metric_summaries": aggregate_report.secondary_metric_summaries,
        },
        pairwise_effects=aggregate_report.pairwise_effects,
        reference_artifacts=tuple(reference_artifacts),
        blind_review_packet=blind_review_packet,
        human_ratings_aggregate=human_ratings_aggregate,
        claim_verdicts=aggregate_report.claim_verdicts,
        description=(
            f"Unified dialogue evidence bundle for {aggregate_report.manifest.suite_id}."
        ),
    )


def _comprehensive_manifest_summary(
    *,
    stage: DialogueComprehensiveStage,
    result: Any,
) -> dict[str, Any] | None:
    if isinstance(result, DialogueNLEssenceAssessmentReport):
        return {
            "stage": stage.value,
            "essence_gate_count": result.total_gate_count,
            "essence_passed_gate_count": result.passed_gate_count,
            "rare_heavy_gate": _rare_heavy_gate_snapshot(result),
        }
    if isinstance(result, OpenDialogueBenchmarkComparisonReport):
        baseline_path = next(
            (path for path in result.path_reports if path.path_label == result.baseline_label),
            None,
        )
        baseline_report = baseline_path.benchmark_report if baseline_path is not None else None
        return {
            "stage": stage.value,
            "open_profile_count": len(result.path_reports),
            "open_scenario_count": baseline_report.total_case_count if baseline_report is not None else 0,
            "open_passed_case_count": baseline_report.passed_case_count if baseline_report is not None else 0,
        }
    if isinstance(result, DialogueComprehensiveBenchmarkReport):
        chosen_candidate = next(
            (
                candidate
                for candidate in result.artifact_comparison_report.candidate_reports
                if candidate.candidate_label == result.artifact_comparison_report.chosen_candidate_label
            ),
            None,
        )
        artifact_acceptance_summary = None
        if chosen_candidate is not None:
            acceptance_report = chosen_candidate.acceptance_report
            artifact_acceptance_summary = {
                "candidate_label": chosen_candidate.candidate_label,
                "accepted": acceptance_report.decision.accepted,
                "override_mode": acceptance_report.decision.override_mode,
                "reasons": list(acceptance_report.decision.reasons),
                "rollback_applied": acceptance_report.decision.rollback_applied,
                "mean_score_delta": acceptance_report.mean_score_delta,
                "passed_case_delta": acceptance_report.passed_case_delta,
                "positive_case_fraction": acceptance_report.positive_case_fraction,
                "worst_case_delta": acceptance_report.worst_case_delta,
                "pre_import_evidence": {
                    key: value for key, value in acceptance_report.pre_import_evidence
                },
            }
        return {
            "stage": stage.value,
            "essence_accepted": result.essence_acceptance.accepted,
            "essence_blocked_gate_ids": list(result.essence_acceptance.blocked_gate_ids),
            "rare_heavy_gate": _rare_heavy_gate_snapshot(result.essence_report),
            "artifact_candidate_count": len(result.artifact_comparison_report.candidate_reports),
            "artifact_acceptance": artifact_acceptance_summary,
            "emergence_dashboard": _emergence_dashboard_snapshot(result.emergence_dashboard),
            "open_profile_count": len(result.open_ablation_report.path_reports) if result.open_ablation_report is not None else 0,
            "open_scenario_count": (
                next(
                    (
                        path.benchmark_report.total_case_count
                        for path in result.open_ablation_report.path_reports
                        if path.path_label == result.open_ablation_report.baseline_label
                    ),
                    0,
                )
                if result.open_ablation_report is not None
                else 0
            ),
        }
    return None


def _write_comprehensive_checkpoint_manifest(
    *,
    output_dir: Path,
    config: DialogueRealComprehensiveBenchmarkConfig,
    completed_stages: tuple[str, ...],
    summary: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "version": 1,
        "config": _json_normalize(config),
        "completed_stages": list(completed_stages),
        "stage_order": [stage.value for stage in _comprehensive_stage_order()],
        "summary": _json_normalize(summary) if summary is not None else None,
    }
    _comprehensive_checkpoint_manifest_path(output_dir).write_text(
        json.dumps(manifest_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _read_comprehensive_checkpoint_manifest(output_dir: Path) -> dict[str, Any] | None:
    manifest_path = _comprehensive_checkpoint_manifest_path(output_dir)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_comprehensive_stage_result(
    *,
    output_dir: Path,
    stage: DialogueComprehensiveStage,
    result: Any,
    config: DialogueRealComprehensiveBenchmarkConfig,
    completed_stages: tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_path = _comprehensive_checkpoint_stage_path(output_dir, stage)
    with stage_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_comprehensive_checkpoint_manifest(
        output_dir=output_dir,
        config=config,
        completed_stages=completed_stages,
        summary=_comprehensive_manifest_summary(stage=stage, result=result),
    )


def _read_comprehensive_stage_result(
    *,
    output_dir: Path,
    stage: DialogueComprehensiveStage,
) -> Any:
    stage_path = _comprehensive_checkpoint_stage_path(output_dir, stage)
    with stage_path.open("rb") as handle:
        return pickle.load(handle)


def _validate_comprehensive_resume_config(
    *,
    output_dir: Path,
    config: DialogueRealComprehensiveBenchmarkConfig,
) -> tuple[str, ...]:
    manifest = _read_comprehensive_checkpoint_manifest(output_dir)
    if manifest is None:
        return ()
    saved_config = manifest.get("config")
    current_config = _json_normalize(config)
    if saved_config != current_config:
        raise ValueError(
            f"Checkpoint config mismatch for {output_dir}: "
            f"saved_config != current_config. Use a new output directory or matching config."
        )
    completed_stages = manifest.get("completed_stages", [])
    return tuple(str(stage_name) for stage_name in completed_stages)


def _shift_pressure_turns(
    pressure_turns: tuple[int, ...],
    *,
    shift: int,
    turn_count: int,
) -> tuple[int, ...]:
    shifted = tuple(
        min(max(turn + shift, 1), turn_count)
        for turn in pressure_turns
    )
    return tuple(dict.fromkeys(shifted))


def _stochastic_pressure_prefix(base_case_id: str) -> str:
    return {
        "repair": "The real rupture is surfacing here:",
        "task_clarification": "The actual bottleneck pressure becomes explicit here:",
        "repeated_failure": "The failure pressure becomes unmistakable here:",
        "goal_drift": "The real goal shift becomes explicit here:",
    }.get(base_case_id, "The real pressure is here:")


def _stochastic_contradiction_suffix(base_case_id: str) -> str:
    return {
        "repair": "If you jump back to solving too early, it will reopen the rupture.",
        "task_clarification": "If you flatten this into generic advice, the answer will miss the real bottleneck.",
        "repeated_failure": "If you just smooth this over, we will repeat the same failure pattern again.",
        "goal_drift": "If you keep the old objective, the answer will now actively optimize the wrong thing.",
    }.get(base_case_id, "If you ignore this shift, the answer will stay misaligned.")


def generate_stochastic_dialogue_case_variants(
    *,
    seeds: tuple[int, ...] = DEFAULT_DIALOGUE_REPLAY_SEEDS,
    families: tuple[DialogueParaphraseFamily, ...] = DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
) -> tuple[DialogueCaseVariant, ...]:
    generated: list[DialogueCaseVariant] = []
    base_cases = {
        case.case_id: case
        for case in DEFAULT_DIALOGUE_PROOF_CASES
    }
    for family in families:
        base_case = base_cases[family.base_case_id]
        for seed in seeds:
            rng = Random(seed)
            user_inputs = tuple(
                rng.choice(options)
                for options in family.turn_alternatives
            )
            pressure_shift = rng.choice((0, 1))
            contradiction_turn = rng.choice((None, 2, 3))
            summary_pressure = rng.choice((False, True))
            shifted_pressure_turns = _shift_pressure_turns(
                base_case.expected_pressure_turns,
                shift=pressure_shift,
                turn_count=len(user_inputs),
            )
            variant_inputs = list(user_inputs)
            if pressure_shift and shifted_pressure_turns:
                target_turn = shifted_pressure_turns[0] - 1
                variant_inputs[target_turn] = (
                    f"{_stochastic_pressure_prefix(base_case.case_id)} {variant_inputs[target_turn]}"
                )
            if contradiction_turn is not None and contradiction_turn - 1 < len(variant_inputs):
                variant_inputs[contradiction_turn - 1] = (
                    f"{variant_inputs[contradiction_turn - 1]} {_stochastic_contradiction_suffix(base_case.case_id)}"
                )
            if summary_pressure:
                variant_inputs[-1] = (
                    f"{variant_inputs[-1]} Also state explicitly whether the new frame really displaced the old one."
                )
            generated_case = ScriptedDialogueCase(
                case_id=f"{base_case.case_id}__{family.family_label}__seed_{seed}",
                description=(
                    f"Stochastic variant for {base_case.case_id} using family={family.family_label} "
                    f"and seed={seed}."
                ),
                user_inputs=tuple(variant_inputs),
                expected_pressure_turns=shifted_pressure_turns,
                expected_delayed_signals=base_case.expected_delayed_signals,
            )
            generated.append(
                DialogueCaseVariant(
                    base_case_id=base_case.case_id,
                    variant_label=f"{family.family_label}__seed_{seed}",
                    case=generated_case,
                    description=(
                        f"Generated from paraphrase family {family.family_label} "
                        f"with seed={seed}, pressure_shift={pressure_shift}, "
                        f"contradiction_turn={contradiction_turn}, summary_pressure={summary_pressure}."
                    ),
                )
            )
    return tuple(generated)


def _dialogue_case_score(report: DialogueBenchmarkCaseReport) -> float:
    return (
        float(report.passed) * 4.0
        + report.pressure_response_precision * 1.5
        + report.pressure_response_recall * 1.5
        + report.pressure_localization_score
        + report.stability_after_recovery_score
        - report.over_response_cost
        - report.recovery_lag_turns * 0.25
    )


def build_dialogue_replay_ranking_report(
    *,
    variant_cases: tuple[DialogueCaseVariant, ...],
    ablation_report: DialogueBenchmarkComparisonReport,
) -> DialogueReplayRankingReport:
    path_reports = {
        path.path_label: {
            case_report.case.case_id: case_report
            for case_report in path.benchmark_report.case_reports
        }
        for path in ablation_report.path_reports
    }
    pe_drive_control_label = "pe-drive-off" if "pe-drive-off" in path_reports else "eta-no-pe"
    eta_control_label = "eta-off" if "eta-off" in path_reports else "heuristic-baseline"
    required_paths = ("pe-eta", pe_drive_control_label, eta_control_label)
    missing_paths = tuple(path for path in required_paths if path not in path_reports)
    if missing_paths:
        raise ValueError(
            "Replay ranking requires baseline paths "
            f"{required_paths!r}; missing={missing_paths!r}."
        )
    entries: list[DialogueReplayRankingEntry] = []
    no_rare_heavy_path = path_reports.get("pe-eta-no-rare-heavy")
    no_rare_heavy_gaps: list[float] = []
    for variant in variant_cases:
        pe_eta_report = path_reports["pe-eta"][variant.case.case_id]
        pe_drive_report = path_reports[pe_drive_control_label][variant.case.case_id]
        eta_control_report = path_reports[eta_control_label][variant.case.case_id]
        no_rare_heavy_report = (
            no_rare_heavy_path.get(variant.case.case_id)
            if no_rare_heavy_path is not None
            else None
        )
        pe_eta_score = _dialogue_case_score(pe_eta_report)
        pe_drive_score = _dialogue_case_score(pe_drive_report)
        eta_control_score = _dialogue_case_score(eta_control_report)
        no_rare_heavy_score = (
            _dialogue_case_score(no_rare_heavy_report)
            if no_rare_heavy_report is not None
            else 0.0
        )
        gap_vs_pe_drive = pe_eta_score - pe_drive_score
        gap_vs_eta_control = pe_eta_score - eta_control_score
        gap_vs_no_rare_heavy = (
            pe_eta_score - no_rare_heavy_score
            if no_rare_heavy_report is not None
            else 0.0
        )
        if no_rare_heavy_report is not None:
            no_rare_heavy_gaps.append(gap_vs_no_rare_heavy)
        entries.append(
            DialogueReplayRankingEntry(
                variant_case_id=variant.case.case_id,
                base_case_id=variant.base_case_id,
                variant_label=variant.variant_label,
                diagnostic_score=gap_vs_pe_drive + gap_vs_eta_control,
                gap_vs_eta_no_pe=gap_vs_pe_drive,
                gap_vs_heuristic=gap_vs_eta_control,
                pe_eta_score=pe_eta_score,
                eta_no_pe_score=pe_drive_score,
                heuristic_score=eta_control_score,
                gap_vs_no_rare_heavy=gap_vs_no_rare_heavy,
                no_rare_heavy_score=no_rare_heavy_score,
                description=(
                    f"Replay ranking entry for {variant.case.case_id} with "
                    f"gap_pe_drive={gap_vs_pe_drive:.2f}, gap_eta_off={gap_vs_eta_control:.2f}, "
                    f"gap_no_rare_heavy={gap_vs_no_rare_heavy:.2f}."
                ),
            )
        )
    entries.sort(key=lambda entry: entry.diagnostic_score, reverse=True)
    return DialogueReplayRankingReport(
        entries=tuple(entries),
        mean_gap_vs_no_rare_heavy=round(_mean(tuple(no_rare_heavy_gaps)), 4),
        description=(
            f"Replay ranking computed for {len(entries)} generated dialogue variants "
            f"with mean_gap_vs_no_rare_heavy={_mean(tuple(no_rare_heavy_gaps)):.3f}."
        ),
    )


def build_dialogue_replay_selection_artifact(
    *,
    variant_cases: tuple[DialogueCaseVariant, ...],
    replay_ranking_report: DialogueReplayRankingReport,
    artifact_id: str = "dialogue-replay-selection",
    top_k: int = 6,
) -> DialogueReplaySelectionArtifact:
    variant_lookup = {
        variant.case.case_id: variant
        for variant in variant_cases
    }
    selected_entries = replay_ranking_report.entries[:top_k]
    selected_variants = tuple(
        variant_lookup[entry.variant_case_id]
        for entry in selected_entries
    )
    return DialogueReplaySelectionArtifact(
        artifact_id=artifact_id,
        selected_variants=selected_variants,
        ranking_entries=selected_entries,
        description=(
            f"Replay selection artifact {artifact_id} captured top {len(selected_variants)} variants "
            f"from {len(variant_cases)} generated candidates."
        ),
    )


def default_rare_heavy_candidate_configs() -> tuple[tuple[str, PipelineConfig], ...]:
    return DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS


def evaluate_dialogue_artifact_acceptance(
    *,
    mean_score_delta: float,
    passed_case_delta: int,
    positive_case_fraction: float,
    worst_case_delta: float,
    substrate_checkpoint_present: bool,
    substrate_update_count: int,
    substrate_source_batch_count: int,
    substrate_mean_sequence_length: float,
    substrate_mean_residual_magnitude: float,
    substrate_import_success_fraction: float,
    gate_config: DialogueArtifactAcceptanceGateConfig,
) -> DialogueArtifactAcceptanceDecision:
    reasons: list[str] = []
    passed_case_shortfall = passed_case_delta < gate_config.min_passed_case_delta
    if mean_score_delta < gate_config.min_mean_score_delta:
        reasons.append("mean-score-delta-below-threshold")
    if passed_case_shortfall:
        reasons.append("passed-case-delta-below-threshold")
    if positive_case_fraction < gate_config.min_positive_case_fraction:
        reasons.append("positive-case-fraction-below-threshold")
    if worst_case_delta < gate_config.min_worst_case_delta:
        reasons.append("worst-case-delta-below-threshold")
    if gate_config.require_substrate_checkpoint and not substrate_checkpoint_present:
        reasons.append("substrate-checkpoint-missing")
    if substrate_update_count < gate_config.min_substrate_update_count:
        reasons.append("substrate-update-count-below-threshold")
    if substrate_source_batch_count < gate_config.min_substrate_source_batch_count:
        reasons.append("substrate-source-batch-count-below-threshold")
    if substrate_mean_sequence_length < gate_config.min_substrate_mean_sequence_length:
        reasons.append("substrate-mean-sequence-length-below-threshold")
    if substrate_mean_residual_magnitude < gate_config.min_substrate_mean_residual_magnitude:
        reasons.append("substrate-mean-residual-magnitude-below-threshold")
    if substrate_import_success_fraction < gate_config.min_substrate_import_success_fraction:
        reasons.append("substrate-import-success-fraction-below-threshold")
    override_mode = "none"
    graded_override = (
        gate_config.allow_strong_graded_gain_override
        and passed_case_shortfall
        and mean_score_delta >= gate_config.min_mean_score_delta_for_graded_override
        and positive_case_fraction >= gate_config.min_positive_case_fraction_for_graded_override
        and worst_case_delta >= gate_config.min_worst_case_delta_for_graded_override
        and substrate_checkpoint_present
        and substrate_update_count >= gate_config.min_substrate_update_count
        and substrate_source_batch_count >= gate_config.min_substrate_source_batch_count
        and substrate_mean_sequence_length >= gate_config.min_substrate_mean_sequence_length
        and substrate_mean_residual_magnitude >= gate_config.min_substrate_mean_residual_magnitude
        and substrate_import_success_fraction >= gate_config.min_substrate_import_success_fraction
    )
    if graded_override and "passed-case-delta-below-threshold" in reasons:
        reasons.remove("passed-case-delta-below-threshold")
        override_mode = "graded-gain"
    accepted = not reasons
    return DialogueArtifactAcceptanceDecision(
        accepted=accepted,
        reasons=tuple(reasons),
        rollback_applied=not accepted,
        override_mode=override_mode,
        description=(
            f"Artifact acceptance decision accepted={accepted} mean_delta={mean_score_delta:.3f} "
            f"passed_case_delta={passed_case_delta} positive_fraction={positive_case_fraction:.3f} "
            f"worst_case_delta={worst_case_delta:.3f} substrate_import_fraction={substrate_import_success_fraction:.3f} "
            f"override_mode={override_mode} reasons={','.join(reasons) if reasons else 'none'}."
        ),
    )


def _dialogue_artifact_acceptance_score(report: DialogueArtifactAcceptanceReport) -> float:
    substrate_metrics = dict(report.substrate_evidence)
    return (
        float(report.decision.accepted) * 5.0
        + report.mean_score_delta * 2.0
        + report.positive_case_fraction
        + float(report.passed_case_delta) * 0.5
        + report.worst_case_delta
        + substrate_metrics.get("substrate_import_success_fraction", 0.0)
        + substrate_metrics.get("substrate_checkpoint_present", 0.0) * 0.5
    )


def build_replay_selection_training_traces(
    selection_artifact: DialogueReplaySelectionArtifact,
) -> tuple[TrainingTrace, ...]:
    return tuple(
        build_training_trace(
            trace_id=f"replay-selection:{variant.case.case_id}",
            source_text=" ".join(variant.case.user_inputs),
        )
        for variant in selection_artifact.selected_variants
    )


def train_rare_heavy_artifact_from_replay_selection(
    selection_artifact: DialogueReplaySelectionArtifact,
    *,
    pipeline_config: PipelineConfig | None = None,
    residual_runtime: OpenWeightResidualRuntime | None = None,
) -> RareHeavyArtifact:
    traces = build_replay_selection_training_traces(selection_artifact)
    pipeline = SSLRLTrainingPipeline(
        config=pipeline_config or PipelineConfig(ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=2),
        residual_runtime=(
            residual_runtime.clone_for_rare_heavy()
            if residual_runtime is not None
            else None
        ),
    )
    pipeline.run_pipeline(traces=traces)
    return pipeline.export_rare_heavy_artifact(
        artifact_id=f"{selection_artifact.artifact_id}:rare-heavy"
    )


def _aligned_pipeline_config_for_runner(
    *,
    runner: AgentSessionRunner,
    pipeline_config: PipelineConfig | None,
) -> PipelineConfig:
    if pipeline_config is not None:
        if pipeline_config.n_z == runner.temporal_latent_dim:
            return pipeline_config
        return PipelineConfig(
            n_z=runner.temporal_latent_dim,
            ssl_convergence_threshold=pipeline_config.ssl_convergence_threshold,
            ssl_min_steps=pipeline_config.ssl_min_steps,
            ssl_max_steps=pipeline_config.ssl_max_steps,
            rl_max_steps=pipeline_config.rl_max_steps,
            rl_convergence_threshold=pipeline_config.rl_convergence_threshold,
            transition_kl_threshold=pipeline_config.transition_kl_threshold,
            binary_gate_rl=pipeline_config.binary_gate_rl,
        )
    return PipelineConfig(
        n_z=runner.temporal_latent_dim,
        ssl_min_steps=2,
        ssl_max_steps=3,
        rl_max_steps=2,
    )


def _benchmark_joint_schedule() -> JointLoopSchedule:
    return JointLoopSchedule(
        ssl_interval=1,
        rl_interval=2,
        pe_rare_heavy_threshold=PROOF_RARE_HEAVY_THRESHOLD,
    )


class _NoSemanticLabelTemporalPolicy(FullLearnedTemporalPolicy):
    """Matched control: keep latent control, remove semantic label scaffold."""

    def __init__(self) -> None:
        super().__init__()
        self._step_counter = 0

    def step(
        self,
        *,
        substrate_snapshot,
        previous_snapshot,
        memory_snapshot=None,
        reflection_snapshot=None,
    ) -> TemporalStep:
        step = super().step(
            substrate_snapshot=substrate_snapshot,
            previous_snapshot=previous_snapshot,
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
        )
        self._step_counter += 1
        neutral_label = f"latent-family-v{step.action_family_version}-t{self._step_counter}"
        return replace(
            step,
            active_abstract_action=neutral_label,
            description=f"{step.description} Semantic scaffold removed; exposing neutral latent-family label.",
        )


class _NoReflectionCacheTemporalPolicy(FullLearnedTemporalPolicy):
    """Matched control: keep latent/full ETA path, disable cached reflection bridge."""

    def observe_reflection_snapshot(
        self,
        *,
        reflection_snapshot,
    ) -> None:
        del reflection_snapshot

    def cached_reflection_snapshot(self):
        return None


def build_standard_dialogue_runner(
    *,
    profile_label: str,
    case: ScriptedDialogueCase,
    residual_runtime: OpenWeightResidualRuntime | None = None,
) -> AgentSessionRunner:
    def _base_session_id(label: str) -> str:
        return f"dialogue-ablation:{label}:{case.case_id}"

    if profile_label == "pe-eta":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            joint_schedule=_benchmark_joint_schedule(),
            allow_live_substrate_mutation=True,
        )
    if profile_label == "pe-eta-online-only":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            reflection_mode=WritebackMode.PROPOSAL_ONLY,
            joint_schedule=_benchmark_joint_schedule(),
            rare_heavy_enabled=False,
            allow_live_substrate_mutation=True,
        )
    if profile_label == "pe-eta-no-writeback":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            reflection_mode=WritebackMode.PROPOSAL_ONLY,
            joint_schedule=_benchmark_joint_schedule(),
            allow_live_substrate_mutation=True,
        )
    if profile_label == "pe-eta-no-rare-heavy":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            joint_schedule=_benchmark_joint_schedule(),
            rare_heavy_enabled=False,
            allow_live_substrate_mutation=True,
        )
    if profile_label == "pe-eta-no-semantic-label":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            world_temporal_policy=_NoSemanticLabelTemporalPolicy(),
            self_temporal_policy=_NoSemanticLabelTemporalPolicy(),
            default_residual_runtime=residual_runtime,
            joint_schedule=_benchmark_joint_schedule(),
            allow_live_substrate_mutation=True,
        )
    if profile_label == "pe-eta-no-reflection-cache":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            world_temporal_policy=_NoReflectionCacheTemporalPolicy(),
            self_temporal_policy=_NoReflectionCacheTemporalPolicy(),
            default_residual_runtime=residual_runtime,
            joint_schedule=_benchmark_joint_schedule(),
            allow_live_substrate_mutation=True,
        )
    if profile_label == "pe-eta-pe-readout-only":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
            external_prediction_error_drive=False,
            prediction_error_readout_only=True,
            primary_prediction_error_dominance_enabled=False,
            allow_live_substrate_mutation=True,
        )
    if profile_label in {"pe-drive-off", "eta-no-pe"}:
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
            external_prediction_error_drive=False,
            allow_live_substrate_mutation=False,
        )
    if profile_label == "timescale-off":
        temporal_policy = FullLearnedTemporalPolicy()
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            temporal_policy=temporal_policy,
            memory_store=build_default_memory_store(
                latent_dim=temporal_policy.parameter_store.n_z,
                nested_profile=False,
            ),
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
            allow_live_substrate_mutation=False,
        )
    if profile_label in {"eta-off", "heuristic-baseline"}:
        temporal_policy = LearnedLiteTemporalPolicy() if profile_label == "eta-off" else HeuristicTemporalPolicy()
        passive_joint_loop = ETANLJointLoop(
            policy=FullLearnedTemporalPolicy(),
            residual_runtime=residual_runtime,
        )
        temporal_config = FinalRolloutConfig(
            substrate=WiringLevel.ACTIVE,
            memory=WiringLevel.ACTIVE,
            dual_track=WiringLevel.ACTIVE,
            evaluation=WiringLevel.ACTIVE,
            regime=WiringLevel.ACTIVE,
            credit=WiringLevel.ACTIVE,
            reflection=WiringLevel.DISABLED,
            temporal=WiringLevel.ACTIVE,
        )
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            temporal_policy=temporal_policy,
            joint_loop=passive_joint_loop,
            default_residual_runtime=residual_runtime,
            config=temporal_config,
            joint_schedule=JointLoopSchedule(
                ssl_interval=0,
                rl_interval=0,
                pe_full_cycle_threshold=999.0,
                pe_ssl_threshold=999.0,
                pe_rare_heavy_threshold=999.0,
            ),
            rare_heavy_enabled=False,
            external_prediction_error_drive=False,
            allow_live_substrate_mutation=False,
        )
    raise ValueError(f"Unsupported dialogue ablation profile: {profile_label}")


def build_real_dialogue_comprehensive_runner_factories(
    *,
    model_id: str = "distilgpt2",
    model_source: str | None = None,
    device: str = "auto",
    local_files_only: bool = False,
    fallback_to_builtin: bool | None = None,
    fallback_mode: SubstrateFallbackMode | str | None = None,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.PREFER_LOCAL,
    acceptance_profile_label: str = "pe-eta",
) -> DialogueSharedRunnerFactories:
    residual_runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        model_source=model_source,
        device=device,
        local_files_only=local_files_only,
        fallback_to_builtin=fallback_to_builtin,
        fallback_mode=fallback_mode,
        runtime_mode=runtime_mode,
        builtin_model_id="dialogue-comprehensive-builtin",
    )

    def canonical_runner_factory(profile_label: str, case: ScriptedDialogueCase) -> AgentSessionRunner:
        return build_standard_dialogue_runner(
            profile_label=profile_label,
            case=case,
            residual_runtime=residual_runtime,
        )

    def perturbation_runner_factory(profile_label: str, variant: DialogueCaseVariant) -> AgentSessionRunner:
        return build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
            residual_runtime=residual_runtime,
        )

    def open_runner_factory(profile_label: str, scenario: OpenDialogueScenario) -> AgentSessionRunner:
        return build_standard_dialogue_runner(
            profile_label=profile_label,
            case=ScriptedDialogueCase(
                case_id=f"open:{scenario.scenario_id}",
                description=scenario.description,
                user_inputs=scenario.opening_turns,
            ),
            residual_runtime=residual_runtime,
        )

    def systematic_runner_factory(profile_label: str, variant: DialogueCaseVariant) -> AgentSessionRunner:
        return build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
            residual_runtime=residual_runtime,
        )

    def acceptance_runner_factory(variant: DialogueCaseVariant) -> AgentSessionRunner:
        return build_standard_dialogue_runner(
            profile_label=acceptance_profile_label,
            case=variant.case,
            residual_runtime=residual_runtime,
        )

    return DialogueSharedRunnerFactories(
        residual_runtime=residual_runtime,
        canonical_runner_factory=canonical_runner_factory,
        perturbation_runner_factory=perturbation_runner_factory,
        open_runner_factory=open_runner_factory,
        systematic_runner_factory=systematic_runner_factory,
        acceptance_runner_factory=acceptance_runner_factory,
        description=(
            f"Shared real dialogue runner factories created for model={residual_runtime.model_id} "
            f"origin={getattr(residual_runtime, 'runtime_origin', 'unknown')}."
        ),
    )


async def run_dialogue_pe_eta_ablation_benchmark(
    *,
    cases: tuple[ScriptedDialogueCase, ...] = DEFAULT_DIALOGUE_PROOF_CASES,
    profile_labels: tuple[str, ...] = default_dialogue_ablation_profiles(),
    baseline_label: str = "pe-eta",
    runner_factory: Callable[[str, ScriptedDialogueCase], AgentSessionRunner] | None = None,
) -> DialogueBenchmarkComparisonReport:
    factory = runner_factory or (lambda profile_label, case: build_standard_dialogue_runner(profile_label=profile_label, case=case))
    path_reports: list[DialogueBenchmarkPathReport] = []
    for profile_label in profile_labels:
        report = await run_dialogue_pe_eta_benchmark(
            cases=cases,
            runner_factory=lambda case, _profile_label=profile_label: factory(_profile_label, case),
            allow_interval_carryover_credit=_profile_allows_interval_carryover_credit(profile_label),
        )
        path_reports.append(
            DialogueBenchmarkPathReport(
                path_label=profile_label,
                benchmark_report=report,
                description=f"Dialogue proof benchmark path {profile_label} completed {report.total_case_count} cases.",
            )
        )
    baseline_path = next((path for path in path_reports if path.path_label == baseline_label), None)
    if baseline_path is None:
        raise ValueError(f"Baseline label {baseline_label!r} not present in profile_labels={profile_labels!r}")
    baseline_reports = {
        case_report.case.case_id: case_report
        for case_report in baseline_path.benchmark_report.case_reports
    }
    case_deltas_from_baseline: list[tuple[str, tuple[tuple[str, tuple[tuple[str, float], ...]], ...]]] = []
    metric_deltas_by_profile = {path.path_label: [] for path in path_reports}
    for case_id, baseline_case in baseline_reports.items():
        path_deltas: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        baseline_metrics = dict(_case_summary_metrics(baseline_case))
        for path in path_reports:
            path_case = next(
                report for report in path.benchmark_report.case_reports if report.case.case_id == case_id
            )
            current_metrics = dict(_case_summary_metrics(path_case))
            metric_delta = tuple(
                sorted(
                    (
                        key,
                        round(current_metrics[key] - baseline_metrics[key], 4),
                    )
                    for key in baseline_metrics
                )
            )
            metric_deltas_by_profile[path.path_label].append(dict(metric_delta))
            path_deltas.append(
                (
                    path.path_label,
                    metric_delta,
                )
            )
        case_deltas_from_baseline.append((case_id, tuple(path_deltas)))
    metric_deltas_from_baseline: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for path in path_reports:
        per_case_deltas = metric_deltas_by_profile[path.path_label]
        if not per_case_deltas:
            metric_deltas_from_baseline.append((path.path_label, ()))
            continue
        metric_names = tuple(per_case_deltas[0].keys())
        metric_deltas_from_baseline.append(
            (
                path.path_label,
                tuple(
                    (
                        key,
                        round(
                            sum(case_delta[key] for case_delta in per_case_deltas) / len(per_case_deltas),
                            4,
                        ),
                    )
                    for key in metric_names
                ),
            )
        )
    rare_heavy_case_deltas: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    rare_heavy_metric_deltas: tuple[tuple[str, float], ...] = ()
    path_lookup = {path.path_label: path for path in path_reports}
    pe_eta_path = path_lookup.get("pe-eta")
    no_rare_heavy_path = path_lookup.get("pe-eta-no-rare-heavy")
    if pe_eta_path is not None and no_rare_heavy_path is not None:
        pe_eta_by_case = {
            report.case.case_id: report
            for report in pe_eta_path.benchmark_report.case_reports
        }
        no_rare_heavy_by_case = {
            report.case.case_id: report
            for report in no_rare_heavy_path.benchmark_report.case_reports
        }
        for case_id, pe_eta_case in pe_eta_by_case.items():
            disabled_case = no_rare_heavy_by_case.get(case_id)
            if disabled_case is None:
                continue
            rare_heavy_case_deltas.append(
                (
                    case_id,
                    _metric_delta_items(
                        current_metrics=dict(_case_summary_metrics(pe_eta_case)),
                        reference_metrics=dict(_case_summary_metrics(disabled_case)),
                    ),
                )
            )
        rare_heavy_metric_deltas = _metric_delta_items(
            current_metrics=dict(pe_eta_path.benchmark_report.metric_means),
            reference_metrics=dict(no_rare_heavy_path.benchmark_report.metric_means),
        )
    strong_proof_profiles = (
        "pe-eta-no-semantic-label",
        "pe-eta-no-reflection-cache",
        "pe-eta-pe-readout-only",
    )
    strong_proof_path_set = {
        path.path_label
        for path in path_reports
        if path.path_label in strong_proof_profiles
    }
    strong_proof_case_deltas = tuple(
        (
            case_id,
            tuple(
                (path_label, delta_items)
                for path_label, delta_items in path_deltas
                if path_label in strong_proof_path_set
            ),
        )
        for case_id, path_deltas in case_deltas_from_baseline
        if any(path_label in strong_proof_path_set for path_label, _ in path_deltas)
    )
    strong_proof_metric_deltas = tuple(
        (path_label, delta_items)
        for path_label, delta_items in metric_deltas_from_baseline
        if path_label in strong_proof_path_set
    )
    return DialogueBenchmarkComparisonReport(
        baseline_label=baseline_label,
        path_reports=tuple(path_reports),
        case_deltas_from_baseline=tuple(case_deltas_from_baseline),
        metric_deltas_from_baseline=tuple(metric_deltas_from_baseline),
        rare_heavy_case_deltas=tuple(rare_heavy_case_deltas),
        rare_heavy_metric_deltas=rare_heavy_metric_deltas,
        strong_proof_case_deltas=strong_proof_case_deltas,
        strong_proof_metric_deltas=strong_proof_metric_deltas,
        description=(
            f"Dialogue ablation benchmark compared {len(path_reports)} paths across {len(cases)} cases "
            f"with baseline={baseline_label}, "
            f"strong_proof_delta={'available' if strong_proof_metric_deltas else 'missing'}, "
            f"rare_heavy_delta={'available' if rare_heavy_metric_deltas else 'missing'}."
        ),
    )


async def run_dialogue_pe_eta_perturbation_benchmark(
    *,
    variant_cases: tuple[DialogueCaseVariant, ...] = DEFAULT_DIALOGUE_CASE_VARIANTS,
    profile_labels: tuple[str, ...] = default_dialogue_ablation_profiles(),
    baseline_label: str = "pe-eta",
    runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner] | None = None,
) -> DialoguePerturbationBenchmarkReport:
    factory = runner_factory or (
        lambda profile_label, variant: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
        )
    )
    variant_lookup = {
        variant.case.case_id: variant
        for variant in variant_cases
    }
    ablation_report = await run_dialogue_pe_eta_ablation_benchmark(
        cases=tuple(variant.case for variant in variant_cases),
        profile_labels=profile_labels,
        baseline_label=baseline_label,
        runner_factory=lambda profile_label, case: factory(profile_label, variant_lookup[case.case_id]),
    )
    return DialoguePerturbationBenchmarkReport(
        variant_cases=variant_cases,
        ablation_report=ablation_report,
        description=(
            f"Dialogue perturbation benchmark evaluated {len(variant_cases)} case variants across "
            f"{len(profile_labels)} paths with baseline={baseline_label}."
        ),
    )


async def run_dialogue_pe_eta_systematic_replay_benchmark(
    *,
    seeds: tuple[int, ...] = DEFAULT_DIALOGUE_REPLAY_SEEDS,
    families: tuple[DialogueParaphraseFamily, ...] = DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
    include_fixed_variants: bool = True,
    profile_labels: tuple[str, ...] = default_dialogue_ablation_profiles(),
    baseline_label: str = "pe-eta",
    runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner] | None = None,
) -> DialogueSystematicReplayBenchmarkReport:
    generated_variants = generate_stochastic_dialogue_case_variants(
        seeds=seeds,
        families=families,
    )
    variant_cases = (
        (DEFAULT_DIALOGUE_CASE_VARIANTS + generated_variants)
        if include_fixed_variants
        else generated_variants
    )
    perturbation_report = await run_dialogue_pe_eta_perturbation_benchmark(
        variant_cases=variant_cases,
        profile_labels=profile_labels,
        baseline_label=baseline_label,
        runner_factory=runner_factory,
    )
    replay_ranking_report = build_dialogue_replay_ranking_report(
        variant_cases=variant_cases,
        ablation_report=perturbation_report.ablation_report,
    )
    return DialogueSystematicReplayBenchmarkReport(
        variant_cases=variant_cases,
        perturbation_report=perturbation_report,
        replay_ranking_report=replay_ranking_report,
        description=(
            f"Systematic replay benchmark evaluated {len(variant_cases)} variants "
            f"across {len(profile_labels)} paths with seeds={seeds}."
        ),
    )


async def run_replay_selection_artifact_acceptance_benchmark(
    selection_artifact: DialogueReplaySelectionArtifact,
    *,
    rare_heavy_artifact: RareHeavyArtifact | None = None,
    profile_label: str = "pe-eta",
    runner_factory: Callable[[DialogueCaseVariant], AgentSessionRunner] | None = None,
    pipeline_config: PipelineConfig | None = None,
    gate_config: DialogueArtifactAcceptanceGateConfig | None = None,
) -> DialogueArtifactAcceptanceReport:
    factory = runner_factory or (
        lambda variant: build_standard_dialogue_runner(
            profile_label=profile_label,
            case=variant.case,
        )
    )
    sample_runner = factory(selection_artifact.selected_variants[0])
    artifact = rare_heavy_artifact or train_rare_heavy_artifact_from_replay_selection(
        selection_artifact,
        pipeline_config=_aligned_pipeline_config_for_runner(
            runner=sample_runner,
            pipeline_config=pipeline_config,
        ),
        residual_runtime=sample_runner.residual_runtime,
    )
    allow_interval_carryover_credit = _profile_allows_interval_carryover_credit(profile_label)
    raw_case_results: list[
        tuple[
            DialogueCaseVariant,
            DialogueBenchmarkCaseReport,
            DialogueBenchmarkCaseReport,
            float,
            RareHeavyImportResult,
            AgentSessionRunner,
        ]
    ] = []
    for index, variant in enumerate(selection_artifact.selected_variants):
        baseline_runner = sample_runner if index == 0 else factory(variant)
        baseline_report = await run_dialogue_pe_eta_case(
            case=variant.case,
            runner=baseline_runner,
            allow_interval_carryover_credit=allow_interval_carryover_credit,
        )
        adapted_runner = factory(variant)
        if adapted_runner.residual_runtime.supports_live_substrate_mutation:
            import_result = adapted_runner.apply_rare_heavy_artifact(
                artifact,
                checkpoint_id=f"{selection_artifact.artifact_id}:{variant.case.case_id}:acceptance",
            )
        else:
            import_result = adapted_runner.review_rare_heavy_artifact(
                artifact,
                checkpoint_id=f"{selection_artifact.artifact_id}:{variant.case.case_id}:acceptance-review",
            )
        adapted_report = await run_dialogue_pe_eta_case(
            case=variant.case,
            runner=adapted_runner,
            allow_interval_carryover_credit=allow_interval_carryover_credit,
        )
        score_delta = _dialogue_case_score(adapted_report) - _dialogue_case_score(baseline_report)
        raw_case_results.append(
            (
                variant,
                baseline_report,
                adapted_report,
                score_delta,
                import_result,
                adapted_runner,
            )
        )
    score_deltas = tuple(item[3] for item in raw_case_results)
    mean_score_delta = _mean(score_deltas)
    passed_case_delta = sum(int(item[2].passed) - int(item[1].passed) for item in raw_case_results)
    positive_case_fraction = (
        sum(1 for delta in score_deltas if delta > 0.0) / len(score_deltas)
        if score_deltas
        else 0.0
    )
    worst_case_delta = min(score_deltas, default=0.0)
    substrate_checkpoint = artifact.substrate_checkpoint
    substrate_import_success_fraction = (
        sum(
            1
            for _, _, _, _, import_result, _ in raw_case_results
            if "rare-heavy:substrate-import" in import_result.applied_operations
        )
        / len(raw_case_results)
        if raw_case_results
        else 0.0
    )
    substrate_evidence = (
        ("substrate_checkpoint_present", float(substrate_checkpoint is not None)),
        ("substrate_update_count", float(substrate_checkpoint.update_count if substrate_checkpoint is not None else 0)),
        (
            "substrate_source_batch_count",
            float(substrate_checkpoint.source_batch_count if substrate_checkpoint is not None else 0),
        ),
        (
            "substrate_mean_sequence_length",
            substrate_checkpoint.mean_sequence_length if substrate_checkpoint is not None else 0.0,
        ),
        (
            "substrate_mean_residual_magnitude",
            substrate_checkpoint.mean_residual_magnitude if substrate_checkpoint is not None else 0.0,
        ),
        ("substrate_import_success_fraction", round(substrate_import_success_fraction, 4)),
    )
    pre_import_pass_fraction = (
        sum(
            1
            for _, _, adapted_report, _, _, _ in raw_case_results
            if adapted_report.rare_heavy_pre_import_pass_count > 0
        )
        / len(raw_case_results)
        if raw_case_results
        else 0.0
    )
    mean_pre_import_delta = _mean(
        tuple(report.mean_rare_heavy_pre_import_score_delta for _, _, report, _, _, _ in raw_case_results)
    )
    mean_candidate_alignment = _mean(
        tuple(report.mean_rare_heavy_candidate_alignment for _, _, report, _, _, _ in raw_case_results)
    )
    max_candidate_adapter_parameter_count = max(
        (
            report.max_rare_heavy_candidate_adapter_parameter_count
            for _, _, report, _, _, _ in raw_case_results
        ),
        default=0,
    )
    pre_import_evidence = (
        ("pre_import_pass_fraction", round(pre_import_pass_fraction, 4)),
        ("mean_pre_import_score_delta", mean_pre_import_delta),
        ("mean_candidate_alignment", mean_candidate_alignment),
        ("max_candidate_adapter_parameter_count", float(max_candidate_adapter_parameter_count)),
    )
    decision = evaluate_dialogue_artifact_acceptance(
        mean_score_delta=mean_score_delta,
        passed_case_delta=passed_case_delta,
        positive_case_fraction=positive_case_fraction,
        worst_case_delta=worst_case_delta,
        substrate_checkpoint_present=substrate_checkpoint is not None,
        substrate_update_count=substrate_checkpoint.update_count if substrate_checkpoint is not None else 0,
        substrate_source_batch_count=substrate_checkpoint.source_batch_count if substrate_checkpoint is not None else 0,
        substrate_mean_sequence_length=(
            substrate_checkpoint.mean_sequence_length if substrate_checkpoint is not None else 0.0
        ),
        substrate_mean_residual_magnitude=(
            substrate_checkpoint.mean_residual_magnitude if substrate_checkpoint is not None else 0.0
        ),
        substrate_import_success_fraction=substrate_import_success_fraction,
        gate_config=gate_config or DialogueArtifactAcceptanceGateConfig(),
    )
    case_reports: list[DialogueArtifactAcceptanceCaseReport] = []
    for variant, baseline_report, adapted_report, score_delta, import_result, adapted_runner in raw_case_results:
        rollback_operations = ()
        if decision.rollback_applied and import_result.applied_operations:
            rollback_operations = adapted_runner.rollback_rare_heavy_import(import_result.checkpoint)
        case_reports.append(
            DialogueArtifactAcceptanceCaseReport(
                variant=variant,
                baseline_report=baseline_report,
                adapted_report=adapted_report,
                score_delta=score_delta,
                import_result=import_result,
                rollback_operations=rollback_operations,
                description=(
                    f"Acceptance benchmark for {variant.case.case_id} produced delta={score_delta:.3f} "
                    f"after applying artifact {artifact.artifact_id}."
                ),
            )
        )
    return DialogueArtifactAcceptanceReport(
        artifact=artifact,
        selection_artifact=selection_artifact,
        case_reports=tuple(case_reports),
        mean_score_delta=mean_score_delta,
        passed_case_delta=passed_case_delta,
        positive_case_fraction=positive_case_fraction,
        worst_case_delta=worst_case_delta,
        substrate_evidence=substrate_evidence,
        decision=decision,
        pre_import_evidence=pre_import_evidence,
        description=(
            f"Replay selection artifact acceptance benchmark evaluated {len(case_reports)} selected variants "
            f"with mean_score_delta={mean_score_delta:.3f}, passed_case_delta={passed_case_delta}, "
            f"positive_case_fraction={positive_case_fraction:.3f}, worst_case_delta={worst_case_delta:.3f}, "
            f"substrate_import_fraction={substrate_import_success_fraction:.3f}, "
            f"pre_import_pass_fraction={pre_import_pass_fraction:.3f}, "
            f"pre_import_mean_delta={mean_pre_import_delta:.3f}, "
            f"override_mode={decision.override_mode}, accepted={decision.accepted}, "
            f"reasons={','.join(decision.reasons) if decision.reasons else 'none'}."
        ),
    )


async def run_multi_artifact_acceptance_benchmark(
    selection_artifact: DialogueReplaySelectionArtifact,
    *,
    candidate_configs: tuple[tuple[str, PipelineConfig], ...] = DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
    profile_label: str = "pe-eta",
    runner_factory: Callable[[DialogueCaseVariant], AgentSessionRunner] | None = None,
    gate_config: DialogueArtifactAcceptanceGateConfig | None = None,
) -> DialogueArtifactComparisonReport:
    candidate_reports: list[DialogueArtifactCandidateReport] = []
    for candidate_label, pipeline_config in candidate_configs:
        acceptance_report = await run_replay_selection_artifact_acceptance_benchmark(
            selection_artifact,
            profile_label=profile_label,
            runner_factory=runner_factory,
            pipeline_config=pipeline_config,
            gate_config=gate_config,
        )
        candidate_score = _dialogue_artifact_acceptance_score(acceptance_report)
        candidate_reports.append(
            DialogueArtifactCandidateReport(
                candidate_label=candidate_label,
                pipeline_config=pipeline_config,
                acceptance_report=acceptance_report,
                candidate_score=candidate_score,
                description=(
                    f"Candidate {candidate_label} scored {candidate_score:.3f} with "
                    f"accepted={acceptance_report.decision.accepted} "
                    f"override_mode={acceptance_report.decision.override_mode} "
                    f"reasons={','.join(acceptance_report.decision.reasons) if acceptance_report.decision.reasons else 'none'}."
                ),
            )
        )
    candidate_reports.sort(key=lambda report: report.candidate_score, reverse=True)
    chosen_candidate = candidate_reports[0] if candidate_reports else None
    return DialogueArtifactComparisonReport(
        selection_artifact=selection_artifact,
        candidate_reports=tuple(candidate_reports),
        chosen_candidate_label=chosen_candidate.candidate_label if chosen_candidate is not None else None,
        chosen_accepted=(
            chosen_candidate.acceptance_report.decision.accepted
            if chosen_candidate is not None
            else False
        ),
        description=(
            f"Multi-artifact acceptance compared {len(candidate_reports)} candidates and chose "
            f"{chosen_candidate.candidate_label if chosen_candidate is not None else 'none'} "
            f"accepted={chosen_candidate.acceptance_report.decision.accepted if chosen_candidate is not None else False} "
            f"override_mode={chosen_candidate.acceptance_report.decision.override_mode if chosen_candidate is not None else 'none'}."
        ),
    )


async def run_dialogue_pe_eta_comprehensive_benchmark(
    *,
    canonical_cases: tuple[ScriptedDialogueCase, ...] = DEFAULT_DIALOGUE_PROOF_CASES,
    open_scenarios: tuple[OpenDialogueScenario, ...] = DEFAULT_OPEN_DIALOGUE_SCENARIOS,
    variant_cases: tuple[DialogueCaseVariant, ...] = DEFAULT_DIALOGUE_CASE_VARIANTS,
    seeds: tuple[int, ...] = DEFAULT_DIALOGUE_REPLAY_SEEDS,
    families: tuple[DialogueParaphraseFamily, ...] = DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
    profile_labels: tuple[str, ...] = default_dialogue_comprehensive_profiles(),
    open_profile_labels: tuple[str, ...] = default_open_dialogue_ablation_profiles(),
    baseline_label: str = "pe-eta",
    selection_top_k: int = 6,
    candidate_configs: tuple[tuple[str, PipelineConfig], ...] = DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
    essence_acceptance_config: DialogueNLEssenceAcceptanceConfig | None = None,
    canonical_runner_factory: Callable[[str, ScriptedDialogueCase], AgentSessionRunner] | None = None,
    longitudinal_runner_factory: Callable[[], AgentSessionRunner] | None = None,
    perturbation_runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner] | None = None,
    open_runner_factory: Callable[[str, OpenDialogueScenario], AgentSessionRunner] | None = None,
    systematic_runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner] | None = None,
    acceptance_runner_factory: Callable[[DialogueCaseVariant], AgentSessionRunner] | None = None,
    acceptance_profile_label: str = "pe-eta",
) -> DialogueComprehensiveBenchmarkReport:
    canonical_ablation_report = await run_dialogue_pe_eta_ablation_benchmark(
        cases=canonical_cases,
        profile_labels=profile_labels,
        baseline_label=baseline_label,
        runner_factory=canonical_runner_factory,
    )
    baseline_path = next(
        path for path in canonical_ablation_report.path_reports if path.path_label == baseline_label
    )
    longitudinal_report = await run_dialogue_pe_eta_longitudinal_benchmark(
        cases=canonical_cases,
        runner_factory=longitudinal_runner_factory,
        allow_interval_carryover_credit=_profile_allows_interval_carryover_credit(baseline_label),
    )
    essence_report = build_dialogue_nl_essence_assessment(
        path_label=baseline_label,
        benchmark_report=baseline_path.benchmark_report,
        comparison_report=canonical_ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
        proof_min_canonical_cases=PROOF_MIN_CANONICAL_CASES,
    )
    essence_acceptance = evaluate_dialogue_nl_essence_acceptance(
        essence_report,
        config=essence_acceptance_config,
    )
    perturbation_report = await run_dialogue_pe_eta_perturbation_benchmark(
        variant_cases=variant_cases,
        profile_labels=profile_labels,
        baseline_label=baseline_label,
        runner_factory=perturbation_runner_factory,
    )
    open_ablation_report = await run_open_dialogue_ablation_benchmark(
        scenarios=open_scenarios,
        profile_labels=open_profile_labels,
        baseline_label=baseline_label,
        runner_factory=open_runner_factory,
    )
    systematic_replay_report = await run_dialogue_pe_eta_systematic_replay_benchmark(
        seeds=seeds,
        families=families,
        profile_labels=profile_labels,
        baseline_label=baseline_label,
        runner_factory=systematic_runner_factory,
    )
    if not systematic_replay_report.variant_cases:
        raise ValueError("Comprehensive benchmark requires at least one generated replay variant.")
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=systematic_replay_report.variant_cases,
        replay_ranking_report=systematic_replay_report.replay_ranking_report,
        artifact_id="dialogue-comprehensive-selection",
        top_k=min(selection_top_k, len(systematic_replay_report.variant_cases)),
    )
    artifact_comparison_report = await run_multi_artifact_acceptance_benchmark(
        selection_artifact,
        candidate_configs=candidate_configs,
        profile_label=acceptance_profile_label,
        runner_factory=acceptance_runner_factory,
    )
    provisional_report = DialogueComprehensiveBenchmarkReport(
        profile_labels=profile_labels,
        canonical_ablation_report=canonical_ablation_report,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        open_ablation_report=open_ablation_report,
        systematic_replay_report=systematic_replay_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        emergence_dashboard=_empty_emergence_dashboard(baseline_label=baseline_label),
        description="Comprehensive dialogue benchmark placeholder before emergence dashboard synthesis.",
    )
    dashboard = build_dialogue_emergence_dashboard(provisional_report)
    return replace(
        provisional_report,
        emergence_dashboard=dashboard,
        description=(
            f"Comprehensive dialogue benchmark ran canonical={len(canonical_cases)}, perturbation={len(variant_cases)}, "
            f"open_scenarios={len(open_scenarios)}, "
            f"longitudinal_verdict={longitudinal_report.cross_session_report.verdict}, "
            f"essence_passed={essence_report.passed_gate_count}/{essence_report.total_gate_count}, "
            f"essence_accepted={essence_acceptance.accepted}, "
            f"{_emergence_dashboard_description_fragment(dashboard)} "
            f"replay_generated={len(systematic_replay_report.variant_cases)}, profiles={len(profile_labels)}, "
            f"selection_top_k={len(selection_artifact.selected_variants)}, candidates={len(candidate_configs)}."
        ),
    )


async def run_real_dialogue_pe_eta_comprehensive_benchmark(
    *,
    config: DialogueRealComprehensiveBenchmarkConfig | None = None,
    essence_acceptance_config: DialogueNLEssenceAcceptanceConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> DialogueComprehensiveBenchmarkReport:
    active_config = config or DialogueRealComprehensiveBenchmarkConfig()
    emit_progress = progress_callback or (lambda message: None)
    emit_progress("Building shared real residual runtime for comprehensive benchmark.")
    shared_factories = build_real_dialogue_comprehensive_runner_factories(
        model_id=active_config.model_id,
        model_source=active_config.model_source,
        device=active_config.device,
        local_files_only=active_config.local_files_only,
        fallback_to_builtin=active_config.fallback_to_builtin,
        fallback_mode=active_config.fallback_mode,
        runtime_mode=active_config.runtime_mode,
        acceptance_profile_label=active_config.acceptance_profile_label,
    )
    canonical_cases = _limit_items(DEFAULT_DIALOGUE_PROOF_CASES, active_config.canonical_case_limit)
    open_scenarios = _limit_items(DEFAULT_OPEN_DIALOGUE_SCENARIOS, active_config.open_scenario_limit)
    perturbation_variants = _limit_items(DEFAULT_DIALOGUE_CASE_VARIANTS, active_config.perturbation_variant_limit)
    replay_families = _limit_items(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES, active_config.replay_family_limit)
    candidate_configs = _limit_items(
        active_config.candidate_configs or DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
        active_config.candidate_config_limit,
    )
    proof_profile_ready = len(canonical_cases) >= active_config.proof_min_canonical_cases
    if not proof_profile_ready:
        emit_progress(
            "Configured canonical case count is below the proof minimum; slow-shapes-fast may fail as a config artifact."
        )
    emit_progress(
        f"Running canonical ablation on {len(canonical_cases)} cases across {len(active_config.profile_labels)} profiles."
    )
    canonical_ablation_report = await run_dialogue_pe_eta_ablation_benchmark(
        cases=canonical_cases,
        profile_labels=active_config.profile_labels,
        baseline_label=active_config.baseline_label,
        runner_factory=shared_factories.canonical_runner_factory,
    )
    emit_progress(f"Running longitudinal benchmark on {len(canonical_cases)} canonical contexts.")
    longitudinal_report = await run_dialogue_pe_eta_longitudinal_benchmark(
        cases=canonical_cases,
        runner_factory=lambda: shared_factories.canonical_runner_factory(
            active_config.baseline_label,
            canonical_cases[0],
        ),
        allow_interval_carryover_credit=_profile_allows_interval_carryover_credit(
            active_config.baseline_label
        ),
    )
    baseline_path = next(
        path
        for path in canonical_ablation_report.path_reports
        if path.path_label == active_config.baseline_label
    )
    essence_report = build_dialogue_nl_essence_assessment(
        path_label=active_config.baseline_label,
        benchmark_report=baseline_path.benchmark_report,
        comparison_report=canonical_ablation_report,
        cross_session_report=longitudinal_report.cross_session_report,
        longitudinal_report=longitudinal_report,
        proof_min_canonical_cases=active_config.proof_min_canonical_cases,
    )
    essence_acceptance = evaluate_dialogue_nl_essence_acceptance(
        essence_report,
        config=essence_acceptance_config,
    )
    emit_progress(
        f"Running perturbation benchmark on {len(perturbation_variants)} fixed variants."
    )
    perturbation_report = await run_dialogue_pe_eta_perturbation_benchmark(
        variant_cases=perturbation_variants,
        profile_labels=active_config.profile_labels,
        baseline_label=active_config.baseline_label,
        runner_factory=shared_factories.perturbation_runner_factory,
    )
    emit_progress(
        f"Running open-environment ablation on {len(open_scenarios)} scenarios "
        f"across {len(active_config.open_profile_labels)} profiles."
    )
    open_ablation_report = await run_open_dialogue_ablation_benchmark(
        scenarios=open_scenarios,
        profile_labels=active_config.open_profile_labels,
        baseline_label=active_config.baseline_label,
        runner_factory=shared_factories.open_runner_factory,
    )
    emit_progress(
        f"Running systematic replay benchmark with seeds={active_config.replay_seeds} "
        f"and families={len(replay_families)}."
    )
    systematic_replay_report = await run_dialogue_pe_eta_systematic_replay_benchmark(
        seeds=active_config.replay_seeds,
        families=replay_families,
        include_fixed_variants=active_config.include_fixed_variants_in_replay,
        profile_labels=active_config.profile_labels,
        baseline_label=active_config.baseline_label,
        runner_factory=shared_factories.systematic_runner_factory,
    )
    if not systematic_replay_report.variant_cases:
        raise ValueError("Real comprehensive benchmark requires at least one replay variant.")
    selection_artifact = build_dialogue_replay_selection_artifact(
        variant_cases=systematic_replay_report.variant_cases,
        replay_ranking_report=systematic_replay_report.replay_ranking_report,
        artifact_id="dialogue-real-comprehensive-selection",
        top_k=min(active_config.selection_top_k, len(systematic_replay_report.variant_cases)),
    )
    emit_progress(
        f"Running multi-artifact acceptance on top_k={len(selection_artifact.selected_variants)} "
        f"with {len(candidate_configs)} candidates."
    )
    artifact_comparison_report = await run_multi_artifact_acceptance_benchmark(
        selection_artifact,
        candidate_configs=candidate_configs,
        profile_label=active_config.acceptance_profile_label,
        runner_factory=shared_factories.acceptance_runner_factory,
    )
    emit_progress("Real comprehensive benchmark finished.")
    provisional_report = DialogueComprehensiveBenchmarkReport(
        profile_labels=active_config.profile_labels,
        canonical_ablation_report=canonical_ablation_report,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        open_ablation_report=open_ablation_report,
        systematic_replay_report=systematic_replay_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        emergence_dashboard=_empty_emergence_dashboard(baseline_label=active_config.baseline_label),
        description="Real comprehensive dialogue benchmark placeholder before emergence dashboard synthesis.",
    )
    dashboard = build_dialogue_emergence_dashboard(provisional_report)
    return replace(
        provisional_report,
        emergence_dashboard=dashboard,
        description=(
            f"Real comprehensive dialogue benchmark used shared runtime "
            f"origin={getattr(shared_factories.residual_runtime, 'runtime_origin', 'unknown')} "
            f"canonical={len(canonical_cases)} longitudinal_verdict={longitudinal_report.cross_session_report.verdict} "
            f"proof_profile_ready={proof_profile_ready} "
            f"essence_passed={essence_report.passed_gate_count}/{essence_report.total_gate_count} "
            f"essence_accepted={essence_acceptance.accepted} "
            f"{_rare_heavy_gate_description_fragment(essence_report)} "
            f"{_emergence_dashboard_description_fragment(dashboard)} "
            f"open_scenarios={len(open_scenarios)} "
            f"perturbation={len(perturbation_variants)} "
            f"replay_variants={len(systematic_replay_report.variant_cases)} "
            f"selection_top_k={len(selection_artifact.selected_variants)} candidates={len(candidate_configs)}."
        ),
    )


async def run_real_dialogue_pe_eta_comprehensive_benchmark_staged(
    *,
    output_dir: str | Path,
    config: DialogueRealComprehensiveBenchmarkConfig | None = None,
    essence_acceptance_config: DialogueNLEssenceAcceptanceConfig | None = None,
    resume: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    shared_factories: DialogueSharedRunnerFactories | None = None,
    longitudinal_runner_factory: Callable[[], AgentSessionRunner] | None = None,
) -> DialogueComprehensiveBenchmarkReport:
    active_config = config or DialogueRealComprehensiveBenchmarkConfig()
    checkpoint_dir = Path(output_dir)
    emit_progress = progress_callback or (lambda message: None)
    completed_stages = (
        _validate_comprehensive_resume_config(output_dir=checkpoint_dir, config=active_config)
        if resume
        else ()
    )
    final_stage_name = DialogueComprehensiveStage.FINAL_REPORT.value
    final_stage_path = _comprehensive_checkpoint_stage_path(
        checkpoint_dir,
        DialogueComprehensiveStage.FINAL_REPORT,
    )
    if resume and final_stage_name in completed_stages and final_stage_path.exists():
        emit_progress("Resuming final comprehensive report from checkpoint.")
        return _read_comprehensive_stage_result(
            output_dir=checkpoint_dir,
            stage=DialogueComprehensiveStage.FINAL_REPORT,
        )

    def stage_completed(stage: DialogueComprehensiveStage) -> bool:
        return (
            resume
            and stage.value in completed_stages
            and _comprehensive_checkpoint_stage_path(checkpoint_dir, stage).exists()
        )

    def load_stage(stage: DialogueComprehensiveStage) -> Any:
        emit_progress(f"Resuming stage {stage.value} from checkpoint.")
        return _read_comprehensive_stage_result(output_dir=checkpoint_dir, stage=stage)

    def save_stage(stage: DialogueComprehensiveStage, result: Any) -> None:
        nonlocal completed_stages
        completed_stages = tuple(dict.fromkeys((*completed_stages, stage.value)))
        _write_comprehensive_stage_result(
            output_dir=checkpoint_dir,
            stage=stage,
            result=result,
            config=active_config,
            completed_stages=completed_stages,
        )

    effective_factories = shared_factories
    if effective_factories is None:
        emit_progress("Building shared real residual runtime for staged comprehensive benchmark.")
        effective_factories = build_real_dialogue_comprehensive_runner_factories(
            model_id=active_config.model_id,
            model_source=active_config.model_source,
            device=active_config.device,
            local_files_only=active_config.local_files_only,
            fallback_to_builtin=active_config.fallback_to_builtin,
            fallback_mode=active_config.fallback_mode,
            runtime_mode=active_config.runtime_mode,
            acceptance_profile_label=active_config.acceptance_profile_label,
        )
    canonical_cases = _limit_items(DEFAULT_DIALOGUE_PROOF_CASES, active_config.canonical_case_limit)
    open_scenarios = _limit_items(DEFAULT_OPEN_DIALOGUE_SCENARIOS, active_config.open_scenario_limit)
    perturbation_variants = _limit_items(DEFAULT_DIALOGUE_CASE_VARIANTS, active_config.perturbation_variant_limit)
    replay_families = _limit_items(DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES, active_config.replay_family_limit)
    candidate_configs = _limit_items(
        active_config.candidate_configs or DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
        active_config.candidate_config_limit,
    )
    proof_profile_ready = len(canonical_cases) >= active_config.proof_min_canonical_cases
    if not proof_profile_ready:
        emit_progress(
            "Configured canonical case count is below the proof minimum; slow-shapes-fast may fail as a config artifact."
        )

    canonical_stage = DialogueComprehensiveStage.CANONICAL_ABLATION
    if stage_completed(canonical_stage):
        canonical_ablation_report = load_stage(canonical_stage)
    else:
        emit_progress(
            f"Running canonical ablation on {len(canonical_cases)} cases across {len(active_config.profile_labels)} profiles."
        )
        canonical_ablation_report = await run_dialogue_pe_eta_ablation_benchmark(
            cases=canonical_cases,
            profile_labels=active_config.profile_labels,
            baseline_label=active_config.baseline_label,
            runner_factory=effective_factories.canonical_runner_factory,
        )
        save_stage(canonical_stage, canonical_ablation_report)

    longitudinal_stage = DialogueComprehensiveStage.LONGITUDINAL
    if stage_completed(longitudinal_stage):
        longitudinal_report = load_stage(longitudinal_stage)
    else:
        emit_progress(f"Running longitudinal benchmark on {len(canonical_cases)} canonical contexts.")
        longitudinal_report = await run_dialogue_pe_eta_longitudinal_benchmark(
            cases=canonical_cases,
            runner_factory=longitudinal_runner_factory
            or (lambda: effective_factories.canonical_runner_factory(active_config.baseline_label, canonical_cases[0])),
            allow_interval_carryover_credit=_profile_allows_interval_carryover_credit(
                active_config.baseline_label
            ),
        )
        save_stage(longitudinal_stage, longitudinal_report)

    baseline_path = next(
        path
        for path in canonical_ablation_report.path_reports
        if path.path_label == active_config.baseline_label
    )
    essence_stage = DialogueComprehensiveStage.ESSENCE
    if stage_completed(essence_stage):
        essence_report = load_stage(essence_stage)
    else:
        emit_progress("Building NL essence assessment from canonical and longitudinal evidence.")
        essence_report = build_dialogue_nl_essence_assessment(
            path_label=active_config.baseline_label,
            benchmark_report=baseline_path.benchmark_report,
            comparison_report=canonical_ablation_report,
            cross_session_report=longitudinal_report.cross_session_report,
            longitudinal_report=longitudinal_report,
            proof_min_canonical_cases=active_config.proof_min_canonical_cases,
        )
        save_stage(essence_stage, essence_report)
    essence_acceptance = evaluate_dialogue_nl_essence_acceptance(
        essence_report,
        config=essence_acceptance_config,
    )

    perturbation_stage = DialogueComprehensiveStage.PERTURBATION
    if stage_completed(perturbation_stage):
        perturbation_report = load_stage(perturbation_stage)
    else:
        emit_progress(
            f"Running perturbation benchmark on {len(perturbation_variants)} fixed variants."
        )
        perturbation_report = await run_dialogue_pe_eta_perturbation_benchmark(
            variant_cases=perturbation_variants,
            profile_labels=active_config.profile_labels,
            baseline_label=active_config.baseline_label,
            runner_factory=effective_factories.perturbation_runner_factory,
        )
        save_stage(perturbation_stage, perturbation_report)

    open_stage = DialogueComprehensiveStage.OPEN_ENVIRONMENT
    if stage_completed(open_stage):
        open_ablation_report = load_stage(open_stage)
    else:
        emit_progress(
            f"Running open-environment ablation on {len(open_scenarios)} scenarios "
            f"across {len(active_config.open_profile_labels)} profiles."
        )
        open_ablation_report = await run_open_dialogue_ablation_benchmark(
            scenarios=open_scenarios,
            profile_labels=active_config.open_profile_labels,
            baseline_label=active_config.baseline_label,
            runner_factory=effective_factories.open_runner_factory,
        )
        save_stage(open_stage, open_ablation_report)

    systematic_stage = DialogueComprehensiveStage.SYSTEMATIC_REPLAY
    if stage_completed(systematic_stage):
        systematic_replay_report = load_stage(systematic_stage)
    else:
        emit_progress(
            f"Running systematic replay benchmark with seeds={active_config.replay_seeds} "
            f"and families={len(replay_families)}."
        )
        systematic_replay_report = await run_dialogue_pe_eta_systematic_replay_benchmark(
            seeds=active_config.replay_seeds,
            families=replay_families,
            include_fixed_variants=active_config.include_fixed_variants_in_replay,
            profile_labels=active_config.profile_labels,
            baseline_label=active_config.baseline_label,
            runner_factory=effective_factories.systematic_runner_factory,
        )
        save_stage(systematic_stage, systematic_replay_report)

    selection_stage = DialogueComprehensiveStage.SELECTION_ARTIFACT
    if stage_completed(selection_stage):
        selection_artifact = load_stage(selection_stage)
    else:
        if not systematic_replay_report.variant_cases:
            raise ValueError("Staged real comprehensive benchmark requires at least one replay variant.")
        emit_progress(
            f"Building replay selection artifact with top_k={min(active_config.selection_top_k, len(systematic_replay_report.variant_cases))}."
        )
        selection_artifact = build_dialogue_replay_selection_artifact(
            variant_cases=systematic_replay_report.variant_cases,
            replay_ranking_report=systematic_replay_report.replay_ranking_report,
            artifact_id="dialogue-real-comprehensive-selection",
            top_k=min(active_config.selection_top_k, len(systematic_replay_report.variant_cases)),
        )
        save_stage(selection_stage, selection_artifact)

    artifact_stage = DialogueComprehensiveStage.ARTIFACT_COMPARISON
    if stage_completed(artifact_stage):
        artifact_comparison_report = load_stage(artifact_stage)
    else:
        emit_progress(
            f"Running multi-artifact acceptance on top_k={len(selection_artifact.selected_variants)} "
            f"with {len(candidate_configs)} candidates."
        )
        artifact_comparison_report = await run_multi_artifact_acceptance_benchmark(
            selection_artifact,
            candidate_configs=candidate_configs,
            profile_label=active_config.acceptance_profile_label,
            runner_factory=effective_factories.acceptance_runner_factory,
        )
        save_stage(artifact_stage, artifact_comparison_report)

    provisional_report = DialogueComprehensiveBenchmarkReport(
        profile_labels=active_config.profile_labels,
        canonical_ablation_report=canonical_ablation_report,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        open_ablation_report=open_ablation_report,
        systematic_replay_report=systematic_replay_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        emergence_dashboard=_empty_emergence_dashboard(baseline_label=active_config.baseline_label),
        description="Staged comprehensive dialogue benchmark placeholder before emergence dashboard synthesis.",
    )
    dashboard = build_dialogue_emergence_dashboard(provisional_report)
    final_report = replace(
        provisional_report,
        emergence_dashboard=dashboard,
        description=(
            f"Staged real comprehensive dialogue benchmark used shared runtime "
            f"origin={getattr(effective_factories.residual_runtime, 'runtime_origin', 'unknown')} "
            f"canonical={len(canonical_cases)} longitudinal_verdict={longitudinal_report.cross_session_report.verdict} "
            f"proof_profile_ready={proof_profile_ready} "
            f"essence_passed={essence_report.passed_gate_count}/{essence_report.total_gate_count} "
            f"essence_accepted={essence_acceptance.accepted} "
            f"{_rare_heavy_gate_description_fragment(essence_report)} "
            f"{_emergence_dashboard_description_fragment(dashboard)} "
            f"open_scenarios={len(open_scenarios)} "
            f"perturbation={len(perturbation_variants)} "
            f"replay_variants={len(systematic_replay_report.variant_cases)} "
            f"selection_top_k={len(selection_artifact.selected_variants)} candidates={len(candidate_configs)}."
        ),
    )
    save_stage(DialogueComprehensiveStage.FINAL_REPORT, final_report)
    emit_progress("Staged real comprehensive benchmark finished.")
    return final_report
