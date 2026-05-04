"""Scheduling mixin for ETA/NL joint-loop cadence decisions."""

from __future__ import annotations

from volvence_zero.joint_loop.contracts import JointLoopSchedule
from volvence_zero.temporal import MetacontrollerRuntimeState


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class _JointLoopSchedulingMixin:
    def _experience_credit_signal(self) -> float:
        keys = (
            "delayed_retrieval_mix_alignment",
            "delayed_regime_alignment",
            "delayed_abstract_action_alignment",
            "regime_sequence_payoff",
        )
        values = [self._external_learning_signals[key] for key in keys if key in self._external_learning_signals]
        if not values:
            return 0.0
        return _clamp(sum(values) / len(values))

    def _experience_control_prior_signal(self) -> float:
        keys = (
            "experience_case_strength",
            "experience_playbook_strength",
            "experience_control_prior_strength",
            "experience_playbook_knowledge_hint",
            "experience_playbook_experience_hint",
        )
        values = [self._external_learning_signals[key] for key in keys if key in self._external_learning_signals]
        if not values:
            return 0.0
        return _clamp(sum(values) / len(values))

    def _record_schedule_outcome(
        self,
        *,
        turn_index: int,
        schedule_action: str,
        metacontroller_state: MetacontrollerRuntimeState | None,
    ) -> None:
        self._last_schedule_action = schedule_action
        if schedule_action != "evidence-only":
            self._last_learning_turn_index = turn_index
        if metacontroller_state is not None:
            self._previous_metacontroller_state = metacontroller_state

    def _latent_continuation_due(
        self,
        *,
        turn_index: int,
        schedule: JointLoopSchedule,
    ) -> bool:
        state = self._previous_metacontroller_state
        if state is None:
            return False
        if self._last_schedule_action == "evidence-only":
            return False
        if self._last_learning_turn_index <= 0 or turn_index - self._last_learning_turn_index != 1:
            return False
        active_family = state.active_family_summary
        if active_family is None:
            return False
        family_support = _clamp(min(active_family.support / 4.0, 1.0))
        family_strength = _clamp(
            active_family.stability * 0.30
            + state.active_family_competition_score * 0.25
            + family_support * 0.20
            + (1.0 - state.action_family_monopoly_pressure) * 0.10
            + state.switch_sparsity * 0.15
        )
        continuation_bias = _clamp(
            (1.0 if state.latest_switch_gate >= 0.55 else 0.0) * 0.45
            + min(state.mean_persistence_window / 2.0, 1.0) * 0.35
            + family_strength * 0.20
        )
        return _clamp(family_strength * 0.65 + continuation_bias * 0.35) >= schedule.latent_continuation_threshold


    def _schedule_telemetry(
        self,
        *,
        turn_index: int,
        schedule: JointLoopSchedule,
    ) -> tuple[tuple[str, int], ...]:
        ssl_due = int(schedule.ssl_interval > 0 and turn_index % schedule.ssl_interval == 0)
        rl_due = int(schedule.rl_interval > 0 and turn_index % schedule.rl_interval == 0)
        pe_magnitude = self._external_learning_signals.get("prediction_error_magnitude", 0.0)
        pe_abs_reward = abs(self._external_learning_signals.get("prediction_error_reward", 0.0))
        pe_full_cycle_due = int(self._pe_full_cycle_due(schedule=schedule))
        pe_ssl_due = int(self._pe_ssl_due(schedule=schedule))
        pe_substrate_online_fast_due = int(self._pe_substrate_online_fast_due(schedule=schedule))
        pe_rare_heavy_due = int(self._pe_rare_heavy_due(schedule=schedule))
        latent_continuation_due = int(self._latent_continuation_due(turn_index=turn_index, schedule=schedule))
        batch_target = self._effective_rl_batch_target()
        pending_batch_count = self._pending_rl_batch_count()
        rl_batch_ready_due = int(self._rl_batch_ready_due())
        rl_batch_wait_due = int(self._rl_batch_wait_due(turn_index=turn_index, schedule=schedule))
        (
            pe_pressure,
            family_stability,
            rollback_risk,
            transition_pressure,
            substrate_pressure,
            rare_heavy_pressure,
        ) = self._joint_schedule_inputs(
            turn_index=turn_index,
            schedule=schedule,
        )
        experience_credit = self._experience_credit_signal()
        control_prior_strength = self._experience_control_prior_signal()
        return (
            ("turn_index", turn_index),
            ("ssl_interval", schedule.ssl_interval),
            ("rl_interval", schedule.rl_interval),
            ("rl_batch_target", batch_target),
            ("pending_batch_count", pending_batch_count),
            ("ssl_due", ssl_due),
            ("rl_due", rl_due),
            ("rl_batch_ready_due", rl_batch_ready_due),
            ("rl_batch_wait_due", rl_batch_wait_due),
            ("pe_full_cycle_due", pe_full_cycle_due),
            ("pe_ssl_due", pe_ssl_due),
            ("pe_substrate_online_fast_due", pe_substrate_online_fast_due),
            ("pe_rare_heavy_due", pe_rare_heavy_due),
            ("latent_continuation_due", latent_continuation_due),
            ("pe_magnitude_x1000", int(pe_magnitude * 1000)),
            ("pe_abs_reward_x1000", int(pe_abs_reward * 1000)),
            ("pe_pressure_x1000", int(pe_pressure * 1000)),
            ("family_stability_x1000", int(family_stability * 1000)),
            ("rollback_risk_x1000", int(rollback_risk * 1000)),
            ("transition_pressure_x1000", int(transition_pressure * 1000)),
            ("substrate_pressure_x1000", int(substrate_pressure * 1000)),
            ("rare_heavy_pressure_x1000", int(rare_heavy_pressure * 1000)),
            ("experience_credit_x1000", int(experience_credit * 1000)),
            ("control_prior_strength_x1000", int(control_prior_strength * 1000)),
        )

    def _pe_full_cycle_due(self, *, schedule: JointLoopSchedule) -> bool:
        pe_magnitude = self._external_learning_signals.get("prediction_error_magnitude", 0.0)
        pe_abs_reward = abs(self._external_learning_signals.get("prediction_error_reward", 0.0))
        return pe_magnitude >= schedule.pe_full_cycle_threshold or pe_abs_reward >= schedule.pe_full_cycle_threshold * 0.5

    def _pe_ssl_due(self, *, schedule: JointLoopSchedule) -> bool:
        pe_magnitude = self._external_learning_signals.get("prediction_error_magnitude", 0.0)
        pe_abs_reward = abs(self._external_learning_signals.get("prediction_error_reward", 0.0))
        return (
            not self._pe_full_cycle_due(schedule=schedule)
            and (pe_magnitude >= schedule.pe_ssl_threshold or pe_abs_reward >= schedule.pe_ssl_threshold)
        )

    def _pe_rare_heavy_due(self, *, schedule: JointLoopSchedule) -> bool:
        pe_magnitude = self._external_learning_signals.get("prediction_error_magnitude", 0.0)
        pe_abs_reward = abs(self._external_learning_signals.get("prediction_error_reward", 0.0))
        return (
            pe_magnitude >= schedule.pe_rare_heavy_threshold
            or pe_abs_reward >= schedule.pe_rare_heavy_threshold * 0.35
        )

    def _pe_substrate_online_fast_due(self, *, schedule: JointLoopSchedule) -> bool:
        pe_magnitude = self._external_learning_signals.get("prediction_error_magnitude", 0.0)
        pe_abs_reward = abs(self._external_learning_signals.get("prediction_error_reward", 0.0))
        return (
            pe_magnitude >= schedule.pe_substrate_online_fast_threshold
            or pe_abs_reward >= schedule.pe_substrate_online_fast_threshold * 0.5
        )

    def _effective_rl_batch_target(self) -> int:
        return max(1, self._rl_batch_accumulation_size)

    def _pending_rl_batch_count(self) -> int:
        return len(self._pending_task_rollouts)

    def _rl_batch_ready_due(self) -> bool:
        target = self._effective_rl_batch_target()
        if target <= 1:
            return False
        return self._pending_rl_batch_count() + 1 >= target

    def _rl_batch_wait_due(
        self,
        *,
        turn_index: int,
        schedule: JointLoopSchedule,
    ) -> bool:
        if self._effective_rl_batch_target() <= 1:
            return False
        if self._pending_rl_batch_count() <= 0:
            return False
        if schedule.rl_batch_max_wait_turns <= 0:
            return False
        if self._last_learning_turn_index <= 0:
            return False
        return turn_index - self._last_learning_turn_index >= schedule.rl_batch_max_wait_turns

    def _joint_schedule_inputs(
        self,
        *,
        turn_index: int,
        schedule: JointLoopSchedule,
    ) -> tuple[float, float, float, float, float, float]:
        pe_magnitude = self._external_learning_signals.get("prediction_error_magnitude", 0.0)
        pe_abs_reward = abs(self._external_learning_signals.get("prediction_error_reward", 0.0))
        experience_credit = self._experience_credit_signal()
        control_prior_strength = self._experience_control_prior_signal()
        pe_pressure = _clamp(
            max(
                pe_magnitude / max(schedule.pe_full_cycle_threshold, 1e-6),
                pe_abs_reward / max(schedule.pe_ssl_threshold, 1e-6),
            )
            * 0.42
            + (1.0 - experience_credit) * 0.08
        )
        state = self._previous_metacontroller_state
        if state is None:
            family_stability = _clamp(0.45 + experience_credit * 0.20 + control_prior_strength * 0.10)
        else:
            active_family = state.active_family_summary
            support = min((active_family.support if active_family is not None else 0) / 4.0, 1.0)
            stability = active_family.stability if active_family is not None else 0.5
            competition = active_family.competition_score if active_family is not None else 0.5
            family_stability = _clamp(
                stability * 0.30
                + competition * 0.18
                + support * 0.12
                + state.switch_sparsity * 0.14
                + (1.0 - state.action_family_monopoly_pressure) * 0.14
                + experience_credit * 0.08
                + control_prior_strength * 0.04
            )
        family_signals = self._previous_family_signals
        rollback_risk = _clamp(
            max(
                1.0 - family_signals.get("safety", 1.0),
                max(0.0, 0.55 - family_signals.get("relationship", 0.55)),
                max(0.0, 0.55 - family_signals.get("learning", 0.55)),
                max(0.0, 0.50 - family_signals.get("abstraction", 0.50)),
                (state.action_family_monopoly_pressure if state is not None else 0.0) * 0.8,
            )
        )
        transition_pressure = _clamp(
            (
                (1.0 - family_stability) * 0.30
                + (state.posterior_drift if state is not None else 0.0) * 0.18
                + (1.0 - min(state.policy_replacement_score, 1.0) if state is not None else 0.5) * 0.18
                + (0.0 if state is None else abs(state.latest_switch_gate - 0.5) * 2.0) * 0.08
                + min((state.active_family_summary.support if state is not None and state.active_family_summary is not None else 0) / 4.0, 1.0) * 0.12
                + (1.0 - control_prior_strength) * 0.14
            )
        )
        substrate_pressure = _clamp(
            max(
                pe_magnitude / max(schedule.pe_substrate_online_fast_threshold, 1e-6),
                pe_abs_reward / max(schedule.pe_substrate_online_fast_threshold, 1e-6),
            )
            * 0.44
            + (1.0 - experience_credit) * 0.04
        )
        rare_heavy_pressure = _clamp(
            pe_magnitude / max(schedule.pe_rare_heavy_threshold, 1e-6)
            + (1.0 - experience_credit) * 0.10
        )
        return (
            pe_pressure,
            family_stability,
            rollback_risk,
            transition_pressure,
            substrate_pressure,
            rare_heavy_pressure,
        )

    def _batch_schedule_action(
        self,
        *,
        turn_index: int,
        schedule: JointLoopSchedule,
        pe_full_cycle_due: bool,
        pe_ssl_due: bool,
        rl_due: bool,
        rl_batch_ready_due: bool,
        rl_batch_wait_due: bool,
        substrate_online_fast_due: bool,
        rare_heavy_review_recommended: bool,
    ) -> str | None:
        if self._effective_rl_batch_target() <= 1:
            return None
        (
            pe_pressure,
            family_stability,
            rollback_risk,
            transition_pressure,
            substrate_pressure,
            rare_heavy_pressure,
        ) = self._joint_schedule_inputs(
            turn_index=turn_index,
            schedule=schedule,
        )
        if rare_heavy_review_recommended and rollback_risk >= 0.70 and rare_heavy_pressure >= 0.75:
            return "ssl-only-rare-heavy-hold"
        if pe_full_cycle_due:
            return "full-cycle-pe"
        if rl_batch_wait_due:
            if transition_pressure >= 0.55 or substrate_pressure >= 0.55:
                return "full-cycle-batch-forced"
            if pe_pressure >= 0.55 or family_stability >= 0.55 or rollback_risk <= 0.35:
                return "full-cycle-batch-forced"
            return "ssl-only-risk-hold" if pe_ssl_due or family_stability >= 0.35 else "evidence-only-risk-hold"
        if rl_batch_ready_due:
            if rare_heavy_review_recommended and rollback_risk >= 0.75 and transition_pressure < 0.55:
                return "ssl-only-rare-heavy-hold"
            if rollback_risk >= 0.75 and pe_pressure < 0.55:
                return "ssl-only-risk-hold"
            if transition_pressure >= 0.55:
                return "full-cycle-batch-transition"
            return "full-cycle-batch"
        if rl_due:
            if substrate_online_fast_due and substrate_pressure >= 0.50 and rollback_risk < 0.75:
                return "full-cycle-collect-substrate"
            if transition_pressure >= 0.60 and family_stability < 0.55 and rollback_risk < 0.75:
                return "full-cycle-collect-transition"
            if rollback_risk >= 0.70 and pe_pressure < 0.45:
                return "ssl-only-risk-hold" if family_stability >= 0.30 else "evidence-only-risk-hold"
            if pe_pressure >= 0.40 or family_stability >= 0.45:
                return "full-cycle-collect"
        return None
