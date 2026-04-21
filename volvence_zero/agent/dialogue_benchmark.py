from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult, default_active_runner
from volvence_zero.evaluation import EvaluationSnapshot
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
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import OpenWeightResidualRuntime, TrainingTrace, build_training_trace
from volvence_zero.temporal import TemporalAbstractionSnapshot
from volvence_zero.temporal import FullLearnedTemporalPolicy, HeuristicTemporalPolicy


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
    rare_heavy_recommended: bool
    rare_heavy_applied: bool
    outcome_metrics: tuple[tuple[str, float], ...]
    description: str


@dataclass(frozen=True)
class DialogueBenchmarkCaseReport:
    case: ScriptedDialogueCase
    turns: tuple[DialogueBenchmarkTurn, ...]
    prediction_chain_turn_count: int
    high_pe_turn_count: int
    pe_triggered_turn_count: int
    recovery_lag_turns: int
    pressure_localization_score: float
    over_response_cost: float
    pressure_response_precision: float
    pressure_response_recall: float
    stability_after_recovery_score: float
    temporal_change_count: int
    delayed_improvement_observed: bool
    acceptance_checks: tuple[tuple[str, bool], ...]
    passed: bool
    reasons: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DialogueBenchmarkReport:
    case_reports: tuple[DialogueBenchmarkCaseReport, ...]
    passed_case_count: int
    total_case_count: int
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
    description: str


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


@dataclass(frozen=True)
class DialogueReplayRankingReport:
    entries: tuple[DialogueReplayRankingEntry, ...]
    description: str


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


@dataclass(frozen=True)
class DialogueArtifactAcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]
    rollback_applied: bool
    description: str


PROOF_HIGH_PE_THRESHOLD = 0.18
PROOF_REWARD_THRESHOLD = 0.05
PROOF_PE_IMPROVEMENT_DELTA = 0.02
PROOF_OUTCOME_IMPROVEMENT_DELTA = 0.02


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


def default_dialogue_ablation_profiles() -> tuple[str, ...]:
    return ("pe-eta", "eta-no-pe", "heuristic-baseline")


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
    return profile_label == "pe-eta"


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


def _pe_trigger_flags(
    turns: tuple[DialogueBenchmarkTurn, ...],
    *,
    high_pe_threshold: float,
    reward_threshold: float,
    allow_interval_carryover_credit: bool,
) -> tuple[bool, ...]:
    flags: list[bool] = []
    previous_turn: DialogueBenchmarkTurn | None = None
    for turn in turns:
        explicit_pe_trigger = turn.joint_schedule_action.endswith("-pe") or turn.rare_heavy_recommended
        current_turn_high_pe = _turn_is_high_pe(
            turn,
            high_pe_threshold=high_pe_threshold,
            reward_threshold=reward_threshold,
        )
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
        flags.append(
            (explicit_pe_trigger and (current_turn_high_pe or previous_turn_high_pe))
            or carryover_temporal_response
        )
        previous_turn = turn
    return tuple(flags)


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
    pe_trigger_flags = _pe_trigger_flags(
        turns,
        high_pe_threshold=high_pe_threshold,
        reward_threshold=reward_threshold,
        allow_interval_carryover_credit=allow_interval_carryover_credit,
    )
    pe_triggered_turn_count = sum(1 for triggered in pe_trigger_flags if triggered)
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
        pe_triggered_turn_count=pe_triggered_turn_count,
        recovery_lag_turns=recovery_lag_turns,
        pressure_localization_score=pressure_localization_score,
        over_response_cost=over_response_cost,
        pressure_response_precision=pressure_response_precision,
        pressure_response_recall=pressure_response_recall,
        stability_after_recovery_score=stability_after_recovery_score,
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
            f"stability_after_recovery={stability_after_recovery_score:.2f}, temporal_changes={temporal_change_count}, "
            f"switch_gate_span={switch_gate_span:.2f}, delayed_improvement={delayed_improvement_observed}."
        ),
    )


def dialogue_turn_from_result(*, turn_index: int, user_input: str, result: AgentTurnResult) -> DialogueBenchmarkTurn:
    temporal_snapshot = result.active_snapshots.get("temporal_abstraction")
    switch_gate = 0.0
    action_family_version = 0
    if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
        switch_gate = temporal_snapshot.value.controller_state.switch_gate
        action_family_version = temporal_snapshot.value.action_family_version
    prediction_error = result.prediction_error
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
        rare_heavy_recommended=bool(result.rare_heavy_result is not None and result.rare_heavy_result.recommended),
        rare_heavy_applied=bool(result.rare_heavy_result is not None and result.rare_heavy_result.applied),
        outcome_metrics=_metric_pairs(result),
        description=(
            f"Turn {turn_index} action={result.joint_schedule_action}, regime={result.active_regime}, "
            f"abstract_action={result.active_abstract_action}, pe={prediction_error.magnitude if prediction_error is not None else 0.0:.2f}."
        ),
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
        description=(
            f"Dialogue proof benchmark processed {len(case_reports)} scripted cases with "
            f"{passed_case_count} passing the current PE-ETA evidence gate."
        ),
    )


def _case_summary_metrics(report: DialogueBenchmarkCaseReport) -> tuple[tuple[str, float], ...]:
    turns = report.turns
    mean_pe = _mean(tuple(turn.prediction_error_magnitude for turn in turns))
    mean_switch_gate = _mean(tuple(turn.switch_gate for turn in turns))
    rare_heavy_count = sum(1 for turn in turns if turn.rare_heavy_applied)
    return (
        ("passed", float(report.passed)),
        ("prediction_chain_turn_count", float(report.prediction_chain_turn_count)),
        ("high_pe_turn_count", float(report.high_pe_turn_count)),
        ("pe_triggered_turn_count", float(report.pe_triggered_turn_count)),
        ("recovery_lag_turns", float(report.recovery_lag_turns)),
        ("pressure_localization_score", report.pressure_localization_score),
        ("over_response_cost", report.over_response_cost),
        ("pressure_response_precision", report.pressure_response_precision),
        ("pressure_response_recall", report.pressure_response_recall),
        ("stability_after_recovery_score", report.stability_after_recovery_score),
        ("temporal_change_count", float(report.temporal_change_count)),
        ("delayed_improvement_observed", float(report.delayed_improvement_observed)),
        ("mean_prediction_error", mean_pe),
        ("mean_switch_gate", mean_switch_gate),
        ("rare_heavy_applied_count", float(rare_heavy_count)),
    )


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
    entries: list[DialogueReplayRankingEntry] = []
    for variant in variant_cases:
        pe_eta_report = path_reports["pe-eta"][variant.case.case_id]
        eta_no_pe_report = path_reports["eta-no-pe"][variant.case.case_id]
        heuristic_report = path_reports["heuristic-baseline"][variant.case.case_id]
        pe_eta_score = _dialogue_case_score(pe_eta_report)
        eta_no_pe_score = _dialogue_case_score(eta_no_pe_report)
        heuristic_score = _dialogue_case_score(heuristic_report)
        gap_vs_eta_no_pe = pe_eta_score - eta_no_pe_score
        gap_vs_heuristic = pe_eta_score - heuristic_score
        entries.append(
            DialogueReplayRankingEntry(
                variant_case_id=variant.case.case_id,
                base_case_id=variant.base_case_id,
                variant_label=variant.variant_label,
                diagnostic_score=gap_vs_eta_no_pe + gap_vs_heuristic,
                gap_vs_eta_no_pe=gap_vs_eta_no_pe,
                gap_vs_heuristic=gap_vs_heuristic,
                pe_eta_score=pe_eta_score,
                eta_no_pe_score=eta_no_pe_score,
                heuristic_score=heuristic_score,
                description=(
                    f"Replay ranking entry for {variant.case.case_id} with "
                    f"gap_eta={gap_vs_eta_no_pe:.2f} and gap_heuristic={gap_vs_heuristic:.2f}."
                ),
            )
        )
    entries.sort(key=lambda entry: entry.diagnostic_score, reverse=True)
    return DialogueReplayRankingReport(
        entries=tuple(entries),
        description=f"Replay ranking computed for {len(entries)} generated dialogue variants.",
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
    accepted = not reasons
    return DialogueArtifactAcceptanceDecision(
        accepted=accepted,
        reasons=tuple(reasons),
        rollback_applied=not accepted,
        description=(
            f"Artifact acceptance decision accepted={accepted} mean_delta={mean_score_delta:.3f} "
            f"passed_case_delta={passed_case_delta} positive_fraction={positive_case_fraction:.3f} "
            f"worst_case_delta={worst_case_delta:.3f}."
        ),
    )


def _dialogue_artifact_acceptance_score(report: DialogueArtifactAcceptanceReport) -> float:
    return (
        float(report.decision.accepted) * 5.0
        + report.mean_score_delta * 2.0
        + report.positive_case_fraction
        + float(report.passed_case_delta) * 0.5
        + report.worst_case_delta
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
) -> RareHeavyArtifact:
    traces = build_replay_selection_training_traces(selection_artifact)
    pipeline = SSLRLTrainingPipeline(
        config=pipeline_config or PipelineConfig(ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=2),
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
    if profile_label == "pe-eta":
        return AgentSessionRunner(
            session_id=f"dialogue-ablation:{profile_label}:{case.case_id}",
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=2),
        )
    if profile_label == "eta-no-pe":
        return AgentSessionRunner(
            session_id=f"dialogue-ablation:{profile_label}:{case.case_id}",
            default_residual_runtime=residual_runtime,
            joint_schedule=JointLoopSchedule(
                ssl_interval=1,
                rl_interval=2,
                pe_full_cycle_threshold=999.0,
                pe_ssl_threshold=999.0,
                pe_rare_heavy_threshold=999.0,
            ),
        )
    if profile_label == "heuristic-baseline":
        temporal_policy = HeuristicTemporalPolicy()
        passive_joint_loop = ETANLJointLoop(
            policy=FullLearnedTemporalPolicy(),
            residual_runtime=residual_runtime,
        )
        return AgentSessionRunner(
            session_id=f"dialogue-ablation:{profile_label}:{case.case_id}",
            temporal_policy=temporal_policy,
            joint_loop=passive_joint_loop,
            default_residual_runtime=residual_runtime,
            config=FinalRolloutConfig(
                substrate=WiringLevel.ACTIVE,
                memory=WiringLevel.ACTIVE,
                dual_track=WiringLevel.ACTIVE,
                evaluation=WiringLevel.ACTIVE,
                regime=WiringLevel.ACTIVE,
                credit=WiringLevel.ACTIVE,
                reflection=WiringLevel.DISABLED,
                temporal=WiringLevel.ACTIVE,
            ),
            joint_schedule=JointLoopSchedule(
                ssl_interval=0,
                rl_interval=0,
                pe_full_cycle_threshold=999.0,
                pe_ssl_threshold=999.0,
                pe_rare_heavy_threshold=999.0,
            ),
            rare_heavy_enabled=False,
        )
    raise ValueError(f"Unsupported dialogue ablation profile: {profile_label}")


async def run_dialogue_pe_eta_ablation_benchmark(
    *,
    cases: tuple[ScriptedDialogueCase, ...] = DEFAULT_DIALOGUE_PROOF_CASES,
    profile_labels: tuple[str, ...] = ("pe-eta", "eta-no-pe", "heuristic-baseline"),
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
    for case_id, baseline_case in baseline_reports.items():
        path_deltas: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        baseline_metrics = dict(_case_summary_metrics(baseline_case))
        for path in path_reports:
            path_case = next(
                report for report in path.benchmark_report.case_reports if report.case.case_id == case_id
            )
            current_metrics = dict(_case_summary_metrics(path_case))
            path_deltas.append(
                (
                    path.path_label,
                    tuple(
                        sorted(
                            (
                                key,
                                round(current_metrics[key] - baseline_metrics[key], 4),
                            )
                            for key in baseline_metrics
                        )
                    ),
                )
            )
        case_deltas_from_baseline.append((case_id, tuple(path_deltas)))
    return DialogueBenchmarkComparisonReport(
        baseline_label=baseline_label,
        path_reports=tuple(path_reports),
        case_deltas_from_baseline=tuple(case_deltas_from_baseline),
        description=(
            f"Dialogue ablation benchmark compared {len(path_reports)} paths across {len(cases)} cases "
            f"with baseline={baseline_label}."
        ),
    )


async def run_dialogue_pe_eta_perturbation_benchmark(
    *,
    variant_cases: tuple[DialogueCaseVariant, ...] = DEFAULT_DIALOGUE_CASE_VARIANTS,
    profile_labels: tuple[str, ...] = ("pe-eta", "eta-no-pe", "heuristic-baseline"),
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
    include_fixed_variants: bool = True,
    profile_labels: tuple[str, ...] = ("pe-eta", "eta-no-pe", "heuristic-baseline"),
    baseline_label: str = "pe-eta",
    runner_factory: Callable[[str, DialogueCaseVariant], AgentSessionRunner] | None = None,
) -> DialogueSystematicReplayBenchmarkReport:
    generated_variants = generate_stochastic_dialogue_case_variants(seeds=seeds)
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
    decision = evaluate_dialogue_artifact_acceptance(
        mean_score_delta=mean_score_delta,
        passed_case_delta=passed_case_delta,
        positive_case_fraction=positive_case_fraction,
        worst_case_delta=worst_case_delta,
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
        decision=decision,
        description=(
            f"Replay selection artifact acceptance benchmark evaluated {len(case_reports)} selected variants "
            f"with mean_score_delta={mean_score_delta:.3f}, passed_case_delta={passed_case_delta}, "
            f"positive_case_fraction={positive_case_fraction:.3f}, worst_case_delta={worst_case_delta:.3f}, "
            f"accepted={decision.accepted}."
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
