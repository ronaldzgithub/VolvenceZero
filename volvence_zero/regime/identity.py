from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

if TYPE_CHECKING:
    from volvence_zero.temporal.interface import MetacontrollerRuntimeState


@dataclass(frozen=True)
class RegimeIdentity:
    regime_id: str
    name: str
    embedding: tuple[float, ...]
    entry_conditions: str
    exit_conditions: str
    historical_effectiveness: float


@dataclass(frozen=True)
class RegimeSnapshot:
    active_regime: RegimeIdentity
    previous_regime: RegimeIdentity | None
    switch_reason: str
    candidate_regimes: tuple[tuple[str, float], ...]
    turns_in_current_regime: int
    description: str
    delayed_outcomes: tuple[tuple[str, float], ...] = ()
    identity_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegimeCheckpoint:
    checkpoint_id: str
    historical_effectiveness: tuple[tuple[str, float], ...]
    strategy_priors: tuple[tuple[str, float], ...]
    active_regime_id: str | None
    previous_regime_id: str | None
    turns_in_current_regime: int
    pending_outcome_regimes: tuple[str, ...] = ()
    last_delayed_outcomes: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PendingRegimeOutcome:
    regime_id: str


@dataclass(frozen=True)
class _RegimeTemplate:
    regime_id: str
    name: str
    embedding: tuple[float, ...]
    entry_conditions: str
    exit_conditions: str


REGIME_TEMPLATES: tuple[_RegimeTemplate, ...] = (
    _RegimeTemplate(
        regime_id="casual_social",
        name="casual social contact",
        embedding=(0.10, 0.15, 0.20),
        entry_conditions="low overall tension and stable interaction quality",
        exit_conditions="task urgency or relationship tension rises",
    ),
    _RegimeTemplate(
        regime_id="acquaintance_building",
        name="acquaintance building",
        embedding=(0.20, 0.35, 0.25),
        entry_conditions="self-track engagement without acute rupture",
        exit_conditions="clear task urgency or support need dominates",
    ),
    _RegimeTemplate(
        regime_id="emotional_support",
        name="emotional support",
        embedding=(0.25, 0.70, 0.30),
        entry_conditions="self-track tension or support need dominates",
        exit_conditions="support need cools down or task urgency dominates",
    ),
    _RegimeTemplate(
        regime_id="guided_exploration",
        name="guided exploration",
        embedding=(0.45, 0.45, 0.55),
        entry_conditions="balanced task and self signals invite exploration",
        exit_conditions="system needs to narrow into task solving or repair",
    ),
    _RegimeTemplate(
        regime_id="problem_solving",
        name="problem solving",
        embedding=(0.80, 0.20, 0.35),
        entry_conditions="world-track urgency and task signal dominate",
        exit_conditions="relationship repair or support becomes primary",
    ),
    _RegimeTemplate(
        regime_id="repair_and_deescalation",
        name="repair and de-escalation",
        embedding=(0.35, 0.80, 0.80),
        entry_conditions="cross-track tension or degraded relationship stability is high",
        exit_conditions="tension stabilizes and another regime out-scores repair",
    ),
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _metric(evaluation_snapshot: EvaluationSnapshot | None, metric_name: str, default: float = 0.0) -> float:
    if evaluation_snapshot is None:
        return default
    for score in evaluation_snapshot.turn_scores:
        if score.metric_name == metric_name:
            return score.value
    return default


def _track_counts(memory_snapshot: MemorySnapshot | None) -> tuple[int, int]:
    if memory_snapshot is None:
        return (0, 0)
    world_count = sum(1 for entry in memory_snapshot.retrieved_entries if entry.track is Track.WORLD)
    self_count = sum(1 for entry in memory_snapshot.retrieved_entries if entry.track is Track.SELF)
    return (world_count, self_count)


def _controller_profile(dual_track_snapshot: DualTrackSnapshot | None) -> tuple[float, float, float, float]:
    if dual_track_snapshot is None:
        return (0.0, 0.0, 0.0, 0.0)
    world_code = dual_track_snapshot.world_track.controller_code
    self_code = dual_track_snapshot.self_track.controller_code
    world_drive = world_code[0] if len(world_code) > 0 else 0.0
    self_drive = self_code[0] if len(self_code) > 0 else 0.0
    shared_drive = (
        ((world_code[1] if len(world_code) > 1 else 0.0) + (self_code[1] if len(self_code) > 1 else 0.0)) / 2.0
    )
    switch_pressure = (
        max(
            world_code[2] if len(world_code) > 2 else 0.0,
            self_code[2] if len(self_code) > 2 else 0.0,
        )
        if dual_track_snapshot is not None
        else 0.0
    )
    return (
        _clamp(world_drive),
        _clamp(self_drive),
        _clamp(shared_drive),
        _clamp(switch_pressure),
    )


def _abstract_action_profile(dual_track_snapshot: DualTrackSnapshot | None) -> tuple[float, float, float, float]:
    if dual_track_snapshot is None:
        return (0.0, 0.0, 0.0, 0.0)
    hints = tuple(
        hint
        for hint in (
            dual_track_snapshot.world_track.abstract_action_hint,
            dual_track_snapshot.self_track.abstract_action_hint,
        )
        if hint is not None
    )
    if not hints:
        return (0.0, 0.0, 0.0, 0.0)
    repair_bias = sum(1.0 for hint in hints if hint == "repair_controller") / len(hints)
    task_bias = sum(1.0 for hint in hints if hint == "task_controller") / len(hints)
    exploration_bias = sum(1.0 for hint in hints if hint == "exploration_controller") / len(hints)
    stabilize_bias = sum(1.0 for hint in hints if hint == "stabilize_controller") / len(hints)
    return (
        _clamp(repair_bias),
        _clamp(task_bias),
        _clamp(exploration_bias),
        _clamp(stabilize_bias),
    )


def _alert_pressure(evaluation_snapshot: EvaluationSnapshot | None) -> float:
    if evaluation_snapshot is None:
        return 0.0
    relevant_alerts = tuple(
        alert
        for alert in evaluation_snapshot.alerts
        if "cross-track stability" in alert.lower() or "rollback pressure" in alert.lower()
    )
    if any(alert.startswith("CRITICAL") for alert in relevant_alerts):
        return 1.0
    if any(alert.startswith("HIGH") for alert in relevant_alerts):
        return 0.8
    if relevant_alerts:
        return 0.5
    return 0.0


def score_regimes(
    *,
    memory_snapshot: MemorySnapshot | None,
    dual_track_snapshot: DualTrackSnapshot | None,
    evaluation_snapshot: EvaluationSnapshot | None,
    historical_effectiveness: Mapping[str, float],
    strategy_priors: Mapping[str, float] | None = None,
) -> tuple[tuple[str, float], ...]:
    regime_priors = strategy_priors or {}
    if dual_track_snapshot is None:
        base = 0.1
        return tuple(
            (
                template.regime_id,
                _clamp(
                    base
                    + historical_effectiveness.get(template.regime_id, 0.0) * 0.1
                    + regime_priors.get(template.regime_id, 0.0)
                ),
            )
            for template in REGIME_TEMPLATES
        )

    world_tension = dual_track_snapshot.world_track.tension_level
    self_tension = dual_track_snapshot.self_track.tension_level
    cross_tension = dual_track_snapshot.cross_track_tension
    task_score = _metric(evaluation_snapshot, "info_integration", default=0.4)
    task_pressure = _metric(evaluation_snapshot, "task_pressure", default=task_score)
    warmth = _metric(evaluation_snapshot, "warmth", default=0.4)
    support_presence = _metric(evaluation_snapshot, "support_presence", default=warmth)
    relationship_stability = _metric(evaluation_snapshot, "cross_track_stability", default=0.4)
    alert_pressure = _alert_pressure(evaluation_snapshot)
    world_count, self_count = _track_counts(memory_snapshot)
    balance = _clamp(1.0 - abs(world_tension - self_tension))
    world_presence = _clamp(world_count / 3.0)
    self_presence = _clamp(self_count / 3.0)
    task_dominance = _clamp(max(task_pressure - support_presence, 0.0) / 0.35)
    support_dominance = _clamp(max(support_presence - task_pressure, 0.0) / 0.35)
    world_drive, self_drive, shared_drive, switch_pressure = _controller_profile(dual_track_snapshot)
    repair_bias, task_bias, exploration_bias, stabilize_bias = _abstract_action_profile(dual_track_snapshot)

    scores = {
        "casual_social": _clamp(
            0.34 * (1.0 - max(world_tension, self_tension))
            + 0.24 * warmth
            + 0.20 * relationship_stability
            + 0.12 * balance
            - 0.22 * switch_pressure
            + 0.10 * stabilize_bias
            - 0.14 * repair_bias
            - 0.08 * task_bias
            - 0.12 * exploration_bias
        ),
        "acquaintance_building": _clamp(
            0.26 * self_presence
            + 0.16 * warmth
            + 0.10 * support_presence
            + 0.18 * relationship_stability
            + 0.16 * self_drive
            + 0.12 * shared_drive
            + 0.08 * balance
            - 0.08 * world_tension
            + 0.06 * stabilize_bias
        ),
        "emotional_support": _clamp(
            0.28 * self_tension
            + 0.18 * self_presence
            + 0.20 * self_drive
            + 0.08 * warmth
            + 0.18 * support_presence
            + 0.14 * support_dominance
            + 0.08 * relationship_stability
            + 0.12 * shared_drive
            + 0.08 * switch_pressure
            + 0.12 * repair_bias
        ),
        "guided_exploration": _clamp(
            0.24 * balance
            + 0.12 * task_score
            + 0.10 * task_pressure
            + 0.18 * shared_drive
            + 0.18 * switch_pressure
            + 0.12 * self_presence
            + 0.10 * world_presence
            + 0.26 * exploration_bias
        ),
        "problem_solving": _clamp(
            0.25 * world_tension
            + 0.16 * task_score
            + 0.16 * task_pressure
            + 0.16 * task_dominance
            + 0.20 * world_presence
            + 0.20 * world_drive
            + 0.11 * switch_pressure
            + 0.22 * task_bias
        ),
        "repair_and_deescalation": _clamp(
            0.28 * cross_tension
            + 0.20 * (1.0 - relationship_stability)
            + 0.12 * self_tension
            + 0.16 * alert_pressure
            + 0.12 * shared_drive
            + 0.12 * switch_pressure
            + 0.30 * repair_bias
            - 0.08 * task_dominance
            - 0.10 * support_dominance
        ),
    }

    ranked: list[tuple[str, float]] = []
    for template in REGIME_TEMPLATES:
        historical = historical_effectiveness.get(template.regime_id, 0.5)
        blended = _clamp(
            scores[template.regime_id] * 0.80
            + historical * 0.15
            + regime_priors.get(template.regime_id, 0.0)
        )
        ranked.append((template.regime_id, round(blended, 4)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return tuple(ranked)


def build_regime_identity(
    *,
    regime_id: str,
    historical_effectiveness: Mapping[str, float],
) -> RegimeIdentity:
    for template in REGIME_TEMPLATES:
        if template.regime_id == regime_id:
            return RegimeIdentity(
                regime_id=template.regime_id,
                name=template.name,
                embedding=template.embedding,
                entry_conditions=template.entry_conditions,
                exit_conditions=template.exit_conditions,
                historical_effectiveness=historical_effectiveness.get(regime_id, 0.5),
            )
    raise KeyError(f"Unknown regime_id: {regime_id}")


class RegimeModule(RuntimeModule[RegimeSnapshot]):
    slot_name = "regime"
    owner = "RegimeModule"
    value_type = RegimeSnapshot
    dependencies = ("memory", "dual_track", "evaluation")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(self, *, wiring_level: WiringLevel | None = None) -> None:
        super().__init__(wiring_level=wiring_level)
        self._historical_effectiveness: dict[str, float] = {
            template.regime_id: 0.5 for template in REGIME_TEMPLATES
        }
        self._strategy_priors: dict[str, float] = {
            template.regime_id: 0.0 for template in REGIME_TEMPLATES
        }
        self._active_regime_id: str | None = None
        self._previous_regime_id: str | None = None
        self._turns_in_current_regime = 0
        self._pending_outcomes: list[PendingRegimeOutcome] = []
        self._last_delayed_outcomes: tuple[tuple[str, float], ...] = ()

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[RegimeSnapshot]:
        memory_snapshot = upstream["memory"]
        dual_track_snapshot = upstream["dual_track"]
        evaluation_snapshot = upstream["evaluation"]

        memory_value = memory_snapshot.value if isinstance(memory_snapshot.value, MemorySnapshot) else None
        dual_track_value = dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        evaluation_value = (
            evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        )

        delayed_outcomes = self._apply_delayed_outcomes(evaluation_value)
        self._update_historical_effectiveness(evaluation_value)
        candidates = score_regimes(
            memory_snapshot=memory_value,
            dual_track_snapshot=dual_track_value,
            evaluation_snapshot=evaluation_value,
            historical_effectiveness=self._historical_effectiveness,
            strategy_priors=self._strategy_priors,
        )
        chosen_regime_id = candidates[0][0]
        switch_reason = self._update_active_regime(chosen_regime_id=chosen_regime_id, candidates=candidates)
        self._pending_outcomes.append(PendingRegimeOutcome(regime_id=self._active_regime_id or chosen_regime_id))
        identity_hints = self._identity_hints(memory_value)
        active_regime = build_regime_identity(
            regime_id=self._active_regime_id or chosen_regime_id,
            historical_effectiveness=self._historical_effectiveness,
        )
        previous_regime = (
            build_regime_identity(
                regime_id=self._previous_regime_id,
                historical_effectiveness=self._historical_effectiveness,
            )
            if self._previous_regime_id is not None
            else None
        )
        description = (
            f"Regime module active={active_regime.regime_id}, previous={self._previous_regime_id}, "
            f"turns_in_current_regime={self._turns_in_current_regime}, "
            f"delayed_outcomes={len(delayed_outcomes)}, identity_hints={len(identity_hints)}."
        )
        return self.publish(
            RegimeSnapshot(
                active_regime=active_regime,
                previous_regime=previous_regime,
                switch_reason=switch_reason,
                candidate_regimes=candidates,
                turns_in_current_regime=self._turns_in_current_regime,
                delayed_outcomes=delayed_outcomes,
                identity_hints=identity_hints,
                description=description,
            )
        )

    async def process_standalone(self, **kwargs: object) -> Snapshot[RegimeSnapshot]:
        memory_snapshot = kwargs.get("memory_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        evaluation_snapshot = kwargs.get("evaluation_snapshot")
        if not isinstance(memory_snapshot, MemorySnapshot):
            memory_snapshot = None
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            dual_track_snapshot = None
        if not isinstance(evaluation_snapshot, EvaluationSnapshot):
            evaluation_snapshot = None

        delayed_outcomes = self._apply_delayed_outcomes(evaluation_snapshot)
        self._update_historical_effectiveness(evaluation_snapshot)
        candidates = score_regimes(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            historical_effectiveness=self._historical_effectiveness,
            strategy_priors=self._strategy_priors,
        )
        chosen_regime_id = candidates[0][0]
        switch_reason = self._update_active_regime(chosen_regime_id=chosen_regime_id, candidates=candidates)
        self._pending_outcomes.append(PendingRegimeOutcome(regime_id=self._active_regime_id or chosen_regime_id))
        identity_hints = self._identity_hints(memory_snapshot)
        active_regime = build_regime_identity(
            regime_id=self._active_regime_id or chosen_regime_id,
            historical_effectiveness=self._historical_effectiveness,
        )
        previous_regime = (
            build_regime_identity(
                regime_id=self._previous_regime_id,
                historical_effectiveness=self._historical_effectiveness,
            )
            if self._previous_regime_id is not None
            else None
        )
        return self.publish(
            RegimeSnapshot(
                active_regime=active_regime,
                previous_regime=previous_regime,
                switch_reason=switch_reason,
                candidate_regimes=candidates,
                turns_in_current_regime=self._turns_in_current_regime,
                delayed_outcomes=delayed_outcomes,
                identity_hints=identity_hints,
                description="Standalone regime snapshot.",
            )
        )

    def _update_historical_effectiveness(self, evaluation_snapshot: EvaluationSnapshot | None) -> None:
        if evaluation_snapshot is None or self._active_regime_id is None:
            return
        relationship_score = _metric(evaluation_snapshot, "cross_track_stability", default=0.5)
        warmth_score = _metric(evaluation_snapshot, "warmth", default=0.5)
        task_score = _metric(evaluation_snapshot, "info_integration", default=0.5)
        blended = _clamp((relationship_score + warmth_score + task_score) / 3.0)
        current = self._historical_effectiveness[self._active_regime_id]
        self._historical_effectiveness[self._active_regime_id] = round(current * 0.7 + blended * 0.3, 4)

    def _apply_delayed_outcomes(
        self,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> tuple[tuple[str, float], ...]:
        if evaluation_snapshot is None or not self._pending_outcomes:
            self._last_delayed_outcomes = ()
            return ()
        relationship_score = _metric(evaluation_snapshot, "cross_track_stability", default=0.5)
        warmth_score = _metric(evaluation_snapshot, "warmth", default=0.5)
        task_score = _metric(evaluation_snapshot, "info_integration", default=0.5)
        delayed_score = _clamp((relationship_score + warmth_score + task_score) / 3.0)
        pending = self._pending_outcomes.pop(0)
        current = self._historical_effectiveness.get(pending.regime_id, 0.5)
        self._historical_effectiveness[pending.regime_id] = round(current * 0.8 + delayed_score * 0.2, 4)
        self._strategy_priors[pending.regime_id] = _clamp(
            self._strategy_priors.get(pending.regime_id, 0.0) + (delayed_score - 0.5) * 0.08
        )
        self._last_delayed_outcomes = ((pending.regime_id, round(delayed_score, 4)),)
        return self._last_delayed_outcomes

    def _identity_hints(self, memory_snapshot: MemorySnapshot | None) -> tuple[str, ...]:
        if memory_snapshot is None:
            return ()
        hints: list[str] = []
        for entry in memory_snapshot.retrieved_entries[:4]:
            if entry.track is Track.SELF:
                hints.append(f"identity:relationship:{entry.content}")
            elif entry.track is Track.SHARED and "user_input" in entry.tags:
                hints.append(f"identity:user:{entry.content}")
        return tuple(dict.fromkeys(hints))[:3]

    def _update_active_regime(
        self,
        *,
        chosen_regime_id: str,
        candidates: tuple[tuple[str, float], ...],
    ) -> str:
        top_score = candidates[0][1]
        if self._active_regime_id is None:
            self._active_regime_id = chosen_regime_id
            self._turns_in_current_regime = 1
            return f"initial selection from candidate score {top_score:.2f}"
        if chosen_regime_id == self._active_regime_id:
            self._turns_in_current_regime += 1
            return f"hold current regime with candidate score {top_score:.2f}"

        self._previous_regime_id = self._active_regime_id
        self._active_regime_id = chosen_regime_id
        self._turns_in_current_regime = 1
        return f"switch to higher-scoring regime with candidate score {top_score:.2f}"

    def apply_policy_consolidation(
        self,
        *,
        strategy_updates: tuple[str, ...],
        regime_effectiveness_updates: tuple[tuple[str, float], ...],
        strategy_gain: float = 0.05,
        effectiveness_gain: float = 0.4,
    ) -> tuple[str, ...]:
        applied: list[str] = []
        for update in strategy_updates:
            if update == "increase_self_track_priority":
                self._strategy_priors["emotional_support"] = _clamp(
                    self._strategy_priors["emotional_support"] + strategy_gain * 1.35
                )
                self._strategy_priors["repair_and_deescalation"] = _clamp(
                    self._strategy_priors["repair_and_deescalation"] + strategy_gain * 0.95
                )
                self._strategy_priors["acquaintance_building"] = _clamp(
                    self._strategy_priors["acquaintance_building"] + strategy_gain * 0.7
                )
                applied.append("strategy-prior:self-track")
            elif update == "increase_world_track_priority":
                self._strategy_priors["problem_solving"] = _clamp(
                    self._strategy_priors["problem_solving"] + strategy_gain * 1.35
                )
                self._strategy_priors["guided_exploration"] = _clamp(
                    self._strategy_priors["guided_exploration"] + strategy_gain * 0.8
                )
                applied.append("strategy-prior:world-track")
        for regime_id, value in regime_effectiveness_updates:
            if regime_id not in self._historical_effectiveness:
                continue
            current = self._historical_effectiveness[regime_id]
            self._historical_effectiveness[regime_id] = round(
                current * (1.0 - effectiveness_gain) + value * effectiveness_gain,
                4,
            )
            applied.append(f"regime-effectiveness:{regime_id}")
        return tuple(applied)

    def apply_metacontroller_evidence(
        self,
        *,
        metacontroller_state: "MetacontrollerRuntimeState | None",
        rollback_reasons: tuple[str, ...],
    ) -> tuple[str, ...]:
        if metacontroller_state is None:
            return ()
        applied: list[str] = []
        if metacontroller_state.active_label == "repair_controller":
            for regime_id in ("repair_and_deescalation", "emotional_support"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:repair")
        elif metacontroller_state.active_label == "task_controller":
            for regime_id in ("problem_solving", "guided_exploration"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:task")
        elif metacontroller_state.active_label == "exploration_controller":
            for regime_id in ("guided_exploration", "acquaintance_building"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:exploration")
        else:
            self._strategy_priors["casual_social"] = _clamp(self._strategy_priors["casual_social"] + 0.02)
            applied.append("metacontroller:stabilize")
        if metacontroller_state.binary_switch_rate > 0.55:
            self._strategy_priors["guided_exploration"] = _clamp(
                self._strategy_priors["guided_exploration"] + 0.03
            )
            applied.append("metacontroller:sparse-switch")
        if metacontroller_state.posterior_drift > 0.45:
            self._strategy_priors["repair_and_deescalation"] = _clamp(
                self._strategy_priors["repair_and_deescalation"] + 0.03
            )
            applied.append("metacontroller:posterior-guard")
        if metacontroller_state.policy_replacement_score > 0.45:
            self._strategy_priors["problem_solving"] = _clamp(
                self._strategy_priors["problem_solving"] + 0.03
            )
            applied.append("metacontroller:replacement")
        if rollback_reasons:
            self._strategy_priors["repair_and_deescalation"] = _clamp(
                self._strategy_priors["repair_and_deescalation"] + 0.05
            )
            applied.append("metacontroller:guard")
        return tuple(applied)

    def create_checkpoint(self, *, checkpoint_id: str) -> RegimeCheckpoint:
        return RegimeCheckpoint(
            checkpoint_id=checkpoint_id,
            historical_effectiveness=tuple(sorted(self._historical_effectiveness.items())),
            strategy_priors=tuple(sorted(self._strategy_priors.items())),
            active_regime_id=self._active_regime_id,
            previous_regime_id=self._previous_regime_id,
            turns_in_current_regime=self._turns_in_current_regime,
            pending_outcome_regimes=tuple(item.regime_id for item in self._pending_outcomes),
            last_delayed_outcomes=self._last_delayed_outcomes,
        )

    def restore_checkpoint(self, checkpoint: RegimeCheckpoint) -> None:
        self._historical_effectiveness = dict(checkpoint.historical_effectiveness)
        self._strategy_priors = dict(checkpoint.strategy_priors)
        self._active_regime_id = checkpoint.active_regime_id
        self._previous_regime_id = checkpoint.previous_regime_id
        self._turns_in_current_regime = checkpoint.turns_in_current_regime
        self._pending_outcomes = [PendingRegimeOutcome(regime_id=value) for value in checkpoint.pending_outcome_regimes]
        self._last_delayed_outcomes = checkpoint.last_delayed_outcomes
