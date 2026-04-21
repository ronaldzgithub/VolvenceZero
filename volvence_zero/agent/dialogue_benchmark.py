from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
from pathlib import Path
import pickle
from random import Random
from typing import Any

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult, default_active_runner
from volvence_zero.evaluation import (
    CrossSessionBenchmarkSuite,
    CrossSessionGrowthReport,
    EvaluationReport,
    EvaluationSnapshot,
    EvolutionDecision,
    JudgementCategory,
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
    PlaceholderTemporalPolicy,
)


@dataclass(frozen=True)
class ScriptedDialogueCase:
    case_id: str
    description: str
    user_inputs: tuple[str, ...]
    expected_pressure_turns: tuple[int, ...] = ()
    expected_delayed_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DialogueBenchmarkTurn:
    turn_index: int
    wave_id: str
    user_input: str
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
    learned_memory_primary: bool = False
    artifact_consolidation_count: int = 0
    learned_recall_count: int = 0
    learned_recall_confidence: float = 0.0
    learned_recall_core_guided: bool = False
    slow_to_fast_target_distance_before: float = 0.0
    slow_to_fast_target_distance_after: float = 0.0
    slow_to_fast_target_alignment_gain: float = 0.0


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
    rare_heavy_recommended_count: int
    rare_heavy_applied_count: int
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
    first_turn_slow_to_fast_target_distance_before: float = 0.0
    first_turn_slow_to_fast_target_distance_after: float = 0.0
    first_turn_slow_to_fast_target_alignment_gain: float = 0.0
    mean_reset_turn_slow_to_fast_target_distance_before: float = 0.0
    mean_reset_turn_slow_to_fast_target_distance_after: float = 0.0
    mean_reset_turn_slow_to_fast_target_alignment_gain: float = 0.0


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
        "rare-heavy-net-benefit",
        "slow-shapes-fast",
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


@dataclass(frozen=True)
class DialogueComprehensiveBenchmarkReport:
    profile_labels: tuple[str, ...]
    canonical_ablation_report: DialogueBenchmarkComparisonReport
    longitudinal_report: DialogueLongitudinalBenchmarkReport
    essence_report: DialogueNLEssenceAssessmentReport
    essence_acceptance: DialogueNLEssenceAcceptanceDecision
    perturbation_report: DialoguePerturbationBenchmarkReport
    systematic_replay_report: DialogueSystematicReplayBenchmarkReport
    selection_artifact: DialogueReplaySelectionArtifact
    artifact_comparison_report: DialogueArtifactComparisonReport
    description: str


@dataclass(frozen=True)
class DialogueSharedRunnerFactories:
    residual_runtime: OpenWeightResidualRuntime
    canonical_runner_factory: Callable[[str, ScriptedDialogueCase], AgentSessionRunner]
    perturbation_runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner]
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
PROOF_MIN_CANONICAL_CASES = 2
PROOF_REWARD_THRESHOLD = PROOF_PRESSURE_REWARD_THRESHOLD


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


def dialogue_proof_cases() -> tuple[ScriptedDialogueCase, ...]:
    return DEFAULT_DIALOGUE_PROOF_CASES


def default_dialogue_real_proof_config(
    *,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = LocalSubstrateRuntimeMode.BUILTIN_ONLY,
) -> DialogueRealComprehensiveBenchmarkConfig:
    return DialogueRealComprehensiveBenchmarkConfig(
        runtime_mode=runtime_mode,
        profile_labels=("pe-eta", "pe-eta-no-rare-heavy", "pe-drive-off", "eta-off", "timescale-off"),
        canonical_case_limit=PROOF_MIN_CANONICAL_CASES,
        perturbation_variant_limit=1,
        replay_family_limit=1,
        replay_seeds=(0,),
        selection_top_k=1,
        candidate_config_limit=1,
        proof_min_canonical_cases=PROOF_MIN_CANONICAL_CASES,
    )


def default_dialogue_ablation_profiles() -> tuple[str, ...]:
    return ("pe-eta", "pe-drive-off", "eta-off", "timescale-off")


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
        if score.family in {"learning", "relationship", "abstraction"}
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
    rare_heavy_recommended_count = sum(
        1 for turn in turns if turn.rare_heavy_recommended
    )
    rare_heavy_applied_count = sum(
        1 for turn in turns if turn.rare_heavy_applied
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
        rare_heavy_recommended_count=rare_heavy_recommended_count,
        rare_heavy_applied_count=rare_heavy_applied_count,
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
            f"writeback_turns={bounded_writeback_turn_count}, rare_heavy_applied={rare_heavy_applied_count}, "
            f"judge_turns={evolution_judge_turn_count}, nested_resets={nested_context_reset_count}, "
            f"store_nested_resets={store_nested_context_reset_count}, "
            f"slow_to_fast_init_benefit={mean_slow_to_fast_init_benefit:.2f}, "
            f"reset_turn_benefit={mean_reset_turn_slow_to_fast_init_benefit:.2f}, "
            f"target_alignment_gain={mean_reset_turn_slow_to_fast_target_alignment_gain:.3f}, "
            f"target_distance_after={mean_reset_turn_slow_to_fast_target_distance_after:.3f}, "
            f"learned_primary_turns={learned_memory_primary_turn_count}, "
            f"core_guided_recall_turns={core_guided_recall_turn_count}, "
            f"mean_recall_confidence={mean_learned_recall_confidence:.2f}, "
            f"temporal_changes={temporal_change_count}, "
            f"switch_gate_span={switch_gate_span:.2f}, delayed_improvement={delayed_improvement_observed}."
        ),
        learned_memory_primary_turn_count=learned_memory_primary_turn_count,
        core_guided_recall_turn_count=core_guided_recall_turn_count,
        mean_learned_recall_confidence=mean_learned_recall_confidence,
        max_artifact_consolidation_count=max_artifact_consolidation_count,
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
    return DialogueBenchmarkTurn(
        turn_index=turn_index,
        wave_id=result.wave_id,
        user_input=user_input,
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
        learned_memory_primary=result.learned_memory_primary,
        artifact_consolidation_count=result.artifact_consolidation_count,
        learned_recall_count=result.learned_recall_count,
        learned_recall_confidence=result.learned_recall_confidence,
        learned_recall_core_guided=result.learned_recall_core_guided,
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
    return build_dialogue_case_report(
        case=case,
        turns=tuple(turns),
        allow_interval_carryover_credit=allow_interval_carryover_credit,
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
        ("rare_heavy_recommended_count", float(report.rare_heavy_recommended_count)),
        ("rare_heavy_applied_count", float(report.rare_heavy_applied_count)),
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
        and evidence_metric_means.get("bounded_writeback_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("nested_profile_active_turn_count", 0.0) > 0.0
        and evidence_metric_means.get("learned_memory_primary_turn_count", 0.0) > 0.0
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
    slow_to_fast_passed = (
        store_nested_reset_count > 0.0
        and reset_turn_slow_to_fast_init_benefit > PROOF_SLOW_TO_FAST_INIT_BENEFIT_THRESHOLD
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
    rare_heavy_passed_delta = rare_heavy_metric_deltas.get("passed", 0.0)
    rare_heavy_delayed_improvement_delta = rare_heavy_metric_deltas.get("delayed_improvement_observed", 0.0)
    rare_heavy_prediction_error_delta = rare_heavy_metric_deltas.get("mean_prediction_error", 0.0)
    rare_heavy_stability_delta = rare_heavy_metric_deltas.get("stability_after_recovery_score", 0.0)
    rare_heavy_net_benefit_passed = (
        bool(rare_heavy_metric_deltas)
        and (rare_heavy_applied_delta > 0.0 or rare_heavy_recommended_delta > 0.0)
        and (
            rare_heavy_passed_delta > 0.0
            or rare_heavy_delayed_improvement_delta > 0.0
            or rare_heavy_stability_delta > 0.0
            or rare_heavy_prediction_error_delta < 0.0
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
                ("bounded_writeback_turn_count", evidence_metric_means.get("bounded_writeback_turn_count", 0.0)),
                ("nested_profile_active_turn_count", evidence_metric_means.get("nested_profile_active_turn_count", 0.0)),
                ("learned_memory_primary_turn_count", evidence_metric_means.get("learned_memory_primary_turn_count", 0.0)),
            ),
            description="Default path should activate online, background, and nested-memory learning surfaces together.",
        ),
        DialogueNLEssenceGate(
            gate_id="rare-heavy-net-benefit",
            passed=rare_heavy_net_benefit_passed,
            evidence=(
                ("rare_heavy_recommended_delta_vs_no_rare_heavy", rare_heavy_recommended_delta),
                ("rare_heavy_applied_delta_vs_no_rare_heavy", rare_heavy_applied_delta),
                ("passed_delta_vs_no_rare_heavy", rare_heavy_passed_delta),
                ("delayed_improvement_delta_vs_no_rare_heavy", rare_heavy_delayed_improvement_delta),
                ("stability_after_recovery_delta_vs_no_rare_heavy", rare_heavy_stability_delta),
                ("mean_prediction_error_delta_vs_no_rare_heavy", rare_heavy_prediction_error_delta),
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
                ("failure_mode", slow_shapes_fast_failure_mode),
            ),
            description="Slow-layer state should seed faster bands through observable nested reset behavior.",
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
    if isinstance(result, DialogueComprehensiveBenchmarkReport):
        return {
            "stage": stage.value,
            "essence_accepted": result.essence_acceptance.accepted,
            "essence_blocked_gate_ids": list(result.essence_acceptance.blocked_gate_ids),
            "rare_heavy_gate": _rare_heavy_gate_snapshot(result.essence_report),
            "artifact_candidate_count": len(result.artifact_comparison_report.candidate_reports),
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
    if mean_score_delta < gate_config.min_mean_score_delta:
        reasons.append("mean-score-delta-below-threshold")
    if passed_case_delta < gate_config.min_passed_case_delta:
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
    accepted = not reasons
    return DialogueArtifactAcceptanceDecision(
        accepted=accepted,
        reasons=tuple(reasons),
        rollback_applied=not accepted,
        description=(
            f"Artifact acceptance decision accepted={accepted} mean_delta={mean_score_delta:.3f} "
            f"passed_case_delta={passed_case_delta} positive_fraction={positive_case_fraction:.3f} "
            f"worst_case_delta={worst_case_delta:.3f} substrate_import_fraction={substrate_import_success_fraction:.3f}."
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
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
        )
    if profile_label == "pe-eta-online-only":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            reflection_mode=WritebackMode.PROPOSAL_ONLY,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
            rare_heavy_enabled=False,
        )
    if profile_label == "pe-eta-no-writeback":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            reflection_mode=WritebackMode.PROPOSAL_ONLY,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
        )
    if profile_label == "pe-eta-no-rare-heavy":
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
            rare_heavy_enabled=False,
        )
    if profile_label in {"pe-drive-off", "eta-no-pe"}:
        return AgentSessionRunner(
            session_id=_base_session_id(profile_label),
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
            external_prediction_error_drive=False,
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
        )
    if profile_label in {"eta-off", "heuristic-baseline"}:
        temporal_policy = PlaceholderTemporalPolicy() if profile_label == "eta-off" else HeuristicTemporalPolicy()
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
    return DialogueBenchmarkComparisonReport(
        baseline_label=baseline_label,
        path_reports=tuple(path_reports),
        case_deltas_from_baseline=tuple(case_deltas_from_baseline),
        metric_deltas_from_baseline=tuple(metric_deltas_from_baseline),
        rare_heavy_case_deltas=tuple(rare_heavy_case_deltas),
        rare_heavy_metric_deltas=rare_heavy_metric_deltas,
        description=(
            f"Dialogue ablation benchmark compared {len(path_reports)} paths across {len(cases)} cases "
            f"with baseline={baseline_label} and "
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
        import_result = adapted_runner.apply_rare_heavy_artifact(
            artifact,
            checkpoint_id=f"{selection_artifact.artifact_id}:{variant.case.case_id}:acceptance",
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
        if decision.rollback_applied:
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
        description=(
            f"Replay selection artifact acceptance benchmark evaluated {len(case_reports)} selected variants "
            f"with mean_score_delta={mean_score_delta:.3f}, passed_case_delta={passed_case_delta}, "
            f"positive_case_fraction={positive_case_fraction:.3f}, worst_case_delta={worst_case_delta:.3f}, "
            f"substrate_import_fraction={substrate_import_success_fraction:.3f}, accepted={decision.accepted}."
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
                    f"accepted={acceptance_report.decision.accepted}."
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
            f"{chosen_candidate.candidate_label if chosen_candidate is not None else 'none'}."
        ),
    )


async def run_dialogue_pe_eta_comprehensive_benchmark(
    *,
    canonical_cases: tuple[ScriptedDialogueCase, ...] = DEFAULT_DIALOGUE_PROOF_CASES,
    variant_cases: tuple[DialogueCaseVariant, ...] = DEFAULT_DIALOGUE_CASE_VARIANTS,
    seeds: tuple[int, ...] = DEFAULT_DIALOGUE_REPLAY_SEEDS,
    families: tuple[DialogueParaphraseFamily, ...] = DEFAULT_DIALOGUE_PARAPHRASE_FAMILIES,
    profile_labels: tuple[str, ...] = default_dialogue_comprehensive_profiles(),
    baseline_label: str = "pe-eta",
    selection_top_k: int = 6,
    candidate_configs: tuple[tuple[str, PipelineConfig], ...] = DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS,
    essence_acceptance_config: DialogueNLEssenceAcceptanceConfig | None = None,
    canonical_runner_factory: Callable[[str, ScriptedDialogueCase], AgentSessionRunner] | None = None,
    longitudinal_runner_factory: Callable[[], AgentSessionRunner] | None = None,
    perturbation_runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner] | None = None,
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
    return DialogueComprehensiveBenchmarkReport(
        profile_labels=profile_labels,
        canonical_ablation_report=canonical_ablation_report,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        systematic_replay_report=systematic_replay_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        description=(
            f"Comprehensive dialogue benchmark ran canonical={len(canonical_cases)}, perturbation={len(variant_cases)}, "
            f"longitudinal_verdict={longitudinal_report.cross_session_report.verdict}, "
            f"essence_passed={essence_report.passed_gate_count}/{essence_report.total_gate_count}, "
            f"essence_accepted={essence_acceptance.accepted}, "
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
    return DialogueComprehensiveBenchmarkReport(
        profile_labels=active_config.profile_labels,
        canonical_ablation_report=canonical_ablation_report,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        systematic_replay_report=systematic_replay_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        description=(
            f"Real comprehensive dialogue benchmark used shared runtime "
            f"origin={getattr(shared_factories.residual_runtime, 'runtime_origin', 'unknown')} "
            f"canonical={len(canonical_cases)} longitudinal_verdict={longitudinal_report.cross_session_report.verdict} "
            f"proof_profile_ready={proof_profile_ready} "
            f"essence_passed={essence_report.passed_gate_count}/{essence_report.total_gate_count} "
            f"essence_accepted={essence_acceptance.accepted} "
            f"{_rare_heavy_gate_description_fragment(essence_report)} "
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

    final_report = DialogueComprehensiveBenchmarkReport(
        profile_labels=active_config.profile_labels,
        canonical_ablation_report=canonical_ablation_report,
        longitudinal_report=longitudinal_report,
        essence_report=essence_report,
        essence_acceptance=essence_acceptance,
        perturbation_report=perturbation_report,
        systematic_replay_report=systematic_replay_report,
        selection_artifact=selection_artifact,
        artifact_comparison_report=artifact_comparison_report,
        description=(
            f"Staged real comprehensive dialogue benchmark used shared runtime "
            f"origin={getattr(effective_factories.residual_runtime, 'runtime_origin', 'unknown')} "
            f"canonical={len(canonical_cases)} longitudinal_verdict={longitudinal_report.cross_session_report.verdict} "
            f"proof_profile_ready={proof_profile_ready} "
            f"essence_passed={essence_report.passed_gate_count}/{essence_report.total_gate_count} "
            f"essence_accepted={essence_acceptance.accepted} "
            f"{_rare_heavy_gate_description_fragment(essence_report)} "
            f"perturbation={len(perturbation_variants)} "
            f"replay_variants={len(systematic_replay_report.variant_cases)} "
            f"selection_top_k={len(selection_artifact.selected_variants)} candidates={len(candidate_configs)}."
        ),
    )
    save_stage(DialogueComprehensiveStage.FINAL_REPORT, final_report)
    emit_progress("Staged real comprehensive benchmark finished.")
    return final_report
