from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult, default_active_runner
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop import ETANLJointLoop, JointLoopSchedule
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import OpenWeightResidualRuntime
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
