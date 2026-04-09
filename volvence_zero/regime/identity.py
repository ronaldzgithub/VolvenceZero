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
class RegimeSelectionWeights:
    weights: tuple[tuple[str, float], ...]
    learning_rate: float = 0.02


@dataclass(frozen=True)
class RegimeSnapshot:
    active_regime: RegimeIdentity
    previous_regime: RegimeIdentity | None
    switch_reason: str
    candidate_regimes: tuple[tuple[str, float], ...]
    turns_in_current_regime: int
    description: str
    delayed_outcomes: tuple[tuple[str, float], ...] = ()
    delayed_attributions: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_attribution_ledger: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_payoffs: tuple["DelayedOutcomePayoff", ...] = ()
    sequence_payoffs: tuple["RegimeSequencePayoff", ...] = ()
    identity_hints: tuple[str, ...] = ()
    effectiveness_trend: tuple[tuple[str, float], ...] = ()
    regime_changed: bool = False
    selection_weights: RegimeSelectionWeights | None = None


@dataclass(frozen=True)
class RegimeCheckpoint:
    checkpoint_id: str
    historical_effectiveness: tuple[tuple[str, float], ...]
    strategy_priors: tuple[tuple[str, float], ...]
    active_regime_id: str | None
    previous_regime_id: str | None
    turns_in_current_regime: int
    turn_index: int = 0
    pending_outcomes: tuple["PendingRegimeOutcome", ...] = ()
    last_delayed_outcomes: tuple[tuple[str, float], ...] = ()
    last_delayed_attributions: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_attribution_ledger: tuple["DelayedOutcomeAttribution", ...] = ()
    delayed_payoffs: tuple["DelayedOutcomePayoff", ...] = ()
    turn_evaluation_scores: tuple[float, ...] = ()
    regime_sequence: tuple[str, ...] = ()
    sequence_payoffs: tuple["RegimeSequencePayoff", ...] = ()
    attribution_horizons: tuple[int, ...] = (2,)


@dataclass(frozen=True)
class PendingRegimeOutcome:
    regime_id: str
    source_turn_index: int
    source_wave_id: str
    abstract_action: str | None = None
    action_family_version: int = 0
    resolution_horizon_turns: int = 2


@dataclass(frozen=True)
class DelayedOutcomeAttribution:
    regime_id: str
    outcome_score: float
    source_turn_index: int
    source_wave_id: str
    abstract_action: str | None = None
    action_family_version: int = 0
    resolved_turn_index: int = 0


@dataclass(frozen=True)
class DelayedOutcomePayoff:
    regime_id: str
    abstract_action: str | None
    action_family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    last_source_wave_id: str


@dataclass(frozen=True)
class RegimeSequencePayoff:
    regime_sequence: tuple[str, ...]
    family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    last_source_wave_id: str


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


def _metacontroller_action_profile(metacontroller_state: "MetacontrollerRuntimeState") -> tuple[float, float, float]:
    track_parameters = {track_name: values for track_name, values in metacontroller_state.track_parameters}
    world_track = track_parameters.get("world", ())
    self_track = track_parameters.get("self", ())
    shared_track = track_parameters.get("shared", ())
    decoder = (
        metacontroller_state.decoder_applied_control
        if metacontroller_state.decoder_applied_control
        else metacontroller_state.decoder_control
    )
    latent = metacontroller_state.latent_mean
    world_bias = (
        (sum(world_track) / len(world_track) if world_track else 0.0) * 0.35
        + (decoder[0] if len(decoder) > 0 else 0.0) * 0.40
        + (latent[0] if len(latent) > 0 else 0.0) * 0.25
    )
    self_bias = (
        (sum(self_track) / len(self_track) if self_track else 0.0) * 0.35
        + (decoder[1] if len(decoder) > 1 else 0.0) * 0.40
        + (latent[1] if len(latent) > 1 else 0.0) * 0.25
    )
    shared_bias = (
        (sum(shared_track) / len(shared_track) if shared_track else 0.0) * 0.35
        + (decoder[2] if len(decoder) > 2 else 0.0) * 0.40
        + (latent[2] if len(latent) > 2 else 0.0) * 0.25
    )
    return (world_bias, self_bias, shared_bias)


def _dominant_abstract_action_context(
    dual_track_snapshot: DualTrackSnapshot | None,
) -> tuple[str | None, int]:
    if dual_track_snapshot is None:
        return (None, 0)
    hints = tuple(
        hint
        for hint in (
            dual_track_snapshot.world_track.abstract_action_hint,
            dual_track_snapshot.self_track.abstract_action_hint,
        )
        if hint is not None
    )
    versions = tuple(
        version
        for version in (
            dual_track_snapshot.world_track.action_family_version_hint,
            dual_track_snapshot.self_track.action_family_version_hint,
        )
        if version > 0
    )
    dominant_hint = hints[0] if hints else None
    dominant_version = max(versions) if versions else 0
    return (dominant_hint, dominant_version)


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
    selection_weights: Mapping[str, float] | None = None,
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
            - 0.10 * task_pressure
            - 0.16 * task_dominance
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
            + 0.08 * relationship_stability
            + 0.20 * world_presence
            + 0.20 * world_drive
            + 0.11 * switch_pressure
            + 0.22 * task_bias
        ),
        "repair_and_deescalation": _clamp(
            0.34 * cross_tension
            + 0.24 * (1.0 - relationship_stability)
            + 0.12 * self_tension
            + 0.16 * alert_pressure
            + 0.12 * shared_drive
            + 0.12 * switch_pressure
            + 0.30 * repair_bias
            - 0.08 * task_dominance
            - 0.10 * support_dominance
        ),
    }

    learned_weights = selection_weights or {}
    ranked: list[tuple[str, float]] = []
    for template in REGIME_TEMPLATES:
        historical = historical_effectiveness.get(template.regime_id, 0.5)
        base_score = _clamp(
            scores[template.regime_id] * 0.80
            + historical * 0.15
            + regime_priors.get(template.regime_id, 0.0)
        )
        weight = learned_weights.get(template.regime_id, 1.0)
        blended = _clamp(base_score * weight)
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

    def __init__(
        self,
        *,
        attribution_horizons: tuple[int, ...] = (2,),
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._attribution_horizons = tuple(min(max(h, 1), 8) for h in attribution_horizons) or (2,)
        self._historical_effectiveness: dict[str, float] = {
            template.regime_id: 0.5 for template in REGIME_TEMPLATES
        }
        self._strategy_priors: dict[str, float] = {
            template.regime_id: 0.0 for template in REGIME_TEMPLATES
        }
        self._selection_weights: dict[str, float] = {
            template.regime_id: 1.0 for template in REGIME_TEMPLATES
        }
        self._selection_weight_lr = 0.02
        self._active_regime_id: str | None = None
        self._previous_regime_id: str | None = None
        self._turns_in_current_regime = 0
        self._turn_index = 0
        self._pending_outcomes: list[PendingRegimeOutcome] = []
        self._last_delayed_outcomes: tuple[tuple[str, float], ...] = ()
        self._last_delayed_attributions: tuple[DelayedOutcomeAttribution, ...] = ()
        self._delayed_attribution_ledger: list[DelayedOutcomeAttribution] = []
        self._delayed_payoffs: dict[tuple[str, str | None, int], DelayedOutcomePayoff] = {}
        self._turn_evaluation_scores: list[float] = []
        self._regime_sequence: list[str] = []
        self._sequence_payoffs: dict[tuple[tuple[str, ...], int], RegimeSequencePayoff] = {}
        self._effectiveness_history: dict[str, list[float]] = {
            template.regime_id: [] for template in REGIME_TEMPLATES
        }

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[RegimeSnapshot]:
        memory_snapshot = upstream["memory"]
        dual_track_snapshot = upstream["dual_track"]
        evaluation_snapshot = upstream["evaluation"]

        memory_value = memory_snapshot.value if isinstance(memory_snapshot.value, MemorySnapshot) else None
        dual_track_value = dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        evaluation_value = (
            evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        )

        self._turn_index += 1
        self._record_turn_score(evaluation_value)
        delayed_attributions = self._apply_delayed_outcomes(evaluation_value)
        delayed_outcomes = tuple(
            (item.regime_id, item.outcome_score) for item in delayed_attributions
        )
        self._update_historical_effectiveness(evaluation_value)
        previous_active = self._active_regime_id
        candidates = score_regimes(
            memory_snapshot=memory_value,
            dual_track_snapshot=dual_track_value,
            evaluation_snapshot=evaluation_value,
            historical_effectiveness=self._historical_effectiveness,
            strategy_priors=self._strategy_priors,
            selection_weights=self._selection_weights,
        )
        chosen_regime_id = candidates[0][0]
        switch_reason = self._update_active_regime(chosen_regime_id=chosen_regime_id, candidates=candidates)
        regime_changed = self._active_regime_id != previous_active and previous_active is not None
        abstract_action, action_family_version = _dominant_abstract_action_context(dual_track_value)
        self._enqueue_pending_outcomes(
            regime_id=self._active_regime_id or chosen_regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
        )
        self._regime_sequence.append(self._active_regime_id or chosen_regime_id)
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
                delayed_attributions=delayed_attributions,
                delayed_attribution_ledger=tuple(self._delayed_attribution_ledger),
                delayed_payoffs=self._sorted_delayed_payoffs(),
                sequence_payoffs=self._sorted_sequence_payoffs(),
                identity_hints=identity_hints,
                effectiveness_trend=self._compute_effectiveness_trend(),
                regime_changed=regime_changed,
                selection_weights=RegimeSelectionWeights(
                    weights=tuple(sorted(self._selection_weights.items())),
                    learning_rate=self._selection_weight_lr,
                ),
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

        self._turn_index += 1
        self._record_turn_score(evaluation_snapshot)
        delayed_attributions = self._apply_delayed_outcomes(evaluation_snapshot)
        delayed_outcomes = tuple(
            (item.regime_id, item.outcome_score) for item in delayed_attributions
        )
        self._update_historical_effectiveness(evaluation_snapshot)
        previous_active = self._active_regime_id
        candidates = score_regimes(
            memory_snapshot=memory_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            historical_effectiveness=self._historical_effectiveness,
            strategy_priors=self._strategy_priors,
            selection_weights=self._selection_weights,
        )
        chosen_regime_id = candidates[0][0]
        switch_reason = self._update_active_regime(chosen_regime_id=chosen_regime_id, candidates=candidates)
        regime_changed = self._active_regime_id != previous_active and previous_active is not None
        abstract_action, action_family_version = _dominant_abstract_action_context(dual_track_snapshot)
        self._enqueue_pending_outcomes(
            regime_id=self._active_regime_id or chosen_regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
        )
        self._regime_sequence.append(self._active_regime_id or chosen_regime_id)
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
                delayed_attributions=delayed_attributions,
                delayed_attribution_ledger=tuple(self._delayed_attribution_ledger),
                delayed_payoffs=self._sorted_delayed_payoffs(),
                sequence_payoffs=self._sorted_sequence_payoffs(),
                identity_hints=identity_hints,
                effectiveness_trend=self._compute_effectiveness_trend(),
                regime_changed=regime_changed,
                selection_weights=RegimeSelectionWeights(
                    weights=tuple(sorted(self._selection_weights.items())),
                    learning_rate=self._selection_weight_lr,
                ),
                description="Standalone regime snapshot.",
            )
        )

    def _compute_effectiveness_trend(self) -> tuple[tuple[str, float], ...]:
        trends: list[tuple[str, float]] = []
        for regime_id, history in self._effectiveness_history.items():
            if len(history) < 2:
                trends.append((regime_id, 0.0))
                continue
            recent = history[-5:]
            if len(recent) < 2:
                trends.append((regime_id, 0.0))
                continue
            slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
            trends.append((regime_id, round(slope, 4)))
        return tuple(sorted(trends))

    def _update_historical_effectiveness(self, evaluation_snapshot: EvaluationSnapshot | None) -> None:
        if evaluation_snapshot is None or self._active_regime_id is None:
            return
        relationship_score = _metric(evaluation_snapshot, "cross_track_stability", default=0.5)
        warmth_score = _metric(evaluation_snapshot, "warmth", default=0.5)
        task_score = _metric(evaluation_snapshot, "info_integration", default=0.5)
        blended = _clamp((relationship_score + warmth_score + task_score) / 3.0)
        current = self._historical_effectiveness[self._active_regime_id]
        self._historical_effectiveness[self._active_regime_id] = round(current * 0.7 + blended * 0.3, 4)
        self._effectiveness_history.setdefault(self._active_regime_id, []).append(
            self._historical_effectiveness[self._active_regime_id]
        )
        if len(self._effectiveness_history[self._active_regime_id]) > 20:
            self._effectiveness_history[self._active_regime_id] = (
                self._effectiveness_history[self._active_regime_id][-20:]
            )

    def _record_turn_score(self, evaluation_snapshot: EvaluationSnapshot | None) -> None:
        if evaluation_snapshot is None:
            self._turn_evaluation_scores.append(0.5)
            return
        relationship = _metric(evaluation_snapshot, "cross_track_stability", default=0.5)
        warmth = _metric(evaluation_snapshot, "warmth", default=0.5)
        task = _metric(evaluation_snapshot, "info_integration", default=0.5)
        self._turn_evaluation_scores.append(_clamp((relationship + warmth + task) / 3.0))

    def _nstep_blended_score(self, source_turn_index: int) -> float:
        gamma = 0.85
        horizon = self._turn_index - source_turn_index
        if horizon <= 0:
            return self._turn_evaluation_scores[-1] if self._turn_evaluation_scores else 0.5
        scores: list[float] = []
        weights: list[float] = []
        for step in range(horizon):
            idx = source_turn_index + step  # 0-based: turn (source+1+step) stored at index (source+step)
            if 0 <= idx < len(self._turn_evaluation_scores):
                scores.append(self._turn_evaluation_scores[idx])
                weights.append(gamma ** (horizon - 1 - step))
        if not scores:
            return self._turn_evaluation_scores[-1] if self._turn_evaluation_scores else 0.5
        return sum(s * w for s, w in zip(scores, weights)) / max(sum(weights), 1e-6)

    def _enqueue_pending_outcomes(
        self,
        *,
        regime_id: str,
        abstract_action: str | None,
        action_family_version: int,
    ) -> None:
        for horizon in self._attribution_horizons:
            self._pending_outcomes.append(
                PendingRegimeOutcome(
                    regime_id=regime_id,
                    source_turn_index=self._turn_index,
                    source_wave_id=f"wave-{self._turn_index}",
                    abstract_action=abstract_action,
                    action_family_version=action_family_version,
                    resolution_horizon_turns=horizon,
                )
            )

    def _apply_delayed_outcomes(
        self,
        evaluation_snapshot: EvaluationSnapshot | None,
    ) -> tuple[DelayedOutcomeAttribution, ...]:
        if evaluation_snapshot is None or not self._pending_outcomes:
            self._last_delayed_outcomes = ()
            self._last_delayed_attributions = ()
            return ()
        matured: list[DelayedOutcomeAttribution] = []
        remaining: list[PendingRegimeOutcome] = []
        for pending in self._pending_outcomes:
            age = self._turn_index - pending.source_turn_index
            if age < pending.resolution_horizon_turns:
                remaining.append(pending)
                continue
            delayed_score = self._nstep_blended_score(pending.source_turn_index)
            current = self._historical_effectiveness.get(pending.regime_id, 0.5)
            self._historical_effectiveness[pending.regime_id] = round(current * 0.8 + delayed_score * 0.2, 4)
            self._strategy_priors[pending.regime_id] = _clamp(
                self._strategy_priors.get(pending.regime_id, 0.0) + (delayed_score - 0.5) * 0.08
            )
            current_weight = self._selection_weights.get(pending.regime_id, 1.0)
            advantage = delayed_score - 0.5
            self._selection_weights[pending.regime_id] = max(
                0.3, min(2.0, current_weight + self._selection_weight_lr * advantage * current_weight)
            )
            matured.append(
                DelayedOutcomeAttribution(
                    regime_id=pending.regime_id,
                    outcome_score=round(delayed_score, 4),
                    source_turn_index=pending.source_turn_index,
                    source_wave_id=pending.source_wave_id,
                    abstract_action=pending.abstract_action,
                    action_family_version=pending.action_family_version,
                    resolved_turn_index=self._turn_index,
                )
            )
        self._pending_outcomes = remaining
        if not matured:
            self._last_delayed_outcomes = ()
            self._last_delayed_attributions = ()
            return ()
        self._delayed_attribution_ledger.extend(matured)
        self._delayed_attribution_ledger = self._delayed_attribution_ledger[-24:]
        for attribution in matured:
            self._update_delayed_payoff(attribution)
            self._update_sequence_payoff(attribution)
        self._last_delayed_outcomes = tuple(
            (item.regime_id, item.outcome_score) for item in matured
        )
        self._last_delayed_attributions = tuple(matured)
        return self._last_delayed_attributions

    def _update_delayed_payoff(self, attribution: DelayedOutcomeAttribution) -> None:
        key = (
            attribution.regime_id,
            attribution.abstract_action,
            attribution.action_family_version,
        )
        current = self._delayed_payoffs.get(key)
        if current is None:
            self._delayed_payoffs[key] = DelayedOutcomePayoff(
                regime_id=attribution.regime_id,
                abstract_action=attribution.abstract_action,
                action_family_version=attribution.action_family_version,
                sample_count=1,
                rolling_payoff=attribution.outcome_score,
                latest_outcome=attribution.outcome_score,
                last_source_wave_id=attribution.source_wave_id,
            )
            return
        sample_count = current.sample_count + 1
        rolling_payoff = round(
            current.rolling_payoff * 0.7 + attribution.outcome_score * 0.3,
            4,
        )
        self._delayed_payoffs[key] = DelayedOutcomePayoff(
            regime_id=current.regime_id,
            abstract_action=current.abstract_action,
            action_family_version=current.action_family_version,
            sample_count=sample_count,
            rolling_payoff=rolling_payoff,
            latest_outcome=attribution.outcome_score,
            last_source_wave_id=attribution.source_wave_id,
        )

    def _sorted_delayed_payoffs(self) -> tuple[DelayedOutcomePayoff, ...]:
        ranked = sorted(
            self._delayed_payoffs.values(),
            key=lambda payoff: (payoff.rolling_payoff, payoff.sample_count),
            reverse=True,
        )
        return tuple(ranked[:12])

    def _update_sequence_payoff(self, attribution: DelayedOutcomeAttribution) -> None:
        src_idx = attribution.source_turn_index - 1
        seq_start = max(0, src_idx - 1)
        regime_seq = tuple(self._regime_sequence[seq_start : src_idx + 1])
        if not regime_seq:
            regime_seq = (attribution.regime_id,)
        key = (regime_seq, attribution.action_family_version)
        current = self._sequence_payoffs.get(key)
        if current is None:
            self._sequence_payoffs[key] = RegimeSequencePayoff(
                regime_sequence=regime_seq,
                family_version=attribution.action_family_version,
                sample_count=1,
                rolling_payoff=attribution.outcome_score,
                latest_outcome=attribution.outcome_score,
                last_source_wave_id=attribution.source_wave_id,
            )
            return
        self._sequence_payoffs[key] = RegimeSequencePayoff(
            regime_sequence=regime_seq,
            family_version=current.family_version,
            sample_count=current.sample_count + 1,
            rolling_payoff=round(current.rolling_payoff * 0.7 + attribution.outcome_score * 0.3, 4),
            latest_outcome=attribution.outcome_score,
            last_source_wave_id=attribution.source_wave_id,
        )

    def _sorted_sequence_payoffs(self) -> tuple[RegimeSequencePayoff, ...]:
        ranked = sorted(
            self._sequence_payoffs.values(),
            key=lambda p: (p.rolling_payoff, p.sample_count),
            reverse=True,
        )
        return tuple(ranked[:12])

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
        world_bias, self_bias, shared_bias = _metacontroller_action_profile(metacontroller_state)
        if self_bias >= world_bias and self_bias >= shared_bias:
            for regime_id in ("repair_and_deescalation", "emotional_support"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:repair")
        elif world_bias >= self_bias and world_bias >= shared_bias:
            for regime_id in ("problem_solving", "guided_exploration"):
                self._strategy_priors[regime_id] = _clamp(self._strategy_priors[regime_id] + 0.04)
            applied.append("metacontroller:task")
        elif shared_bias >= max(world_bias, self_bias):
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
            turn_index=self._turn_index,
            pending_outcomes=tuple(self._pending_outcomes),
            last_delayed_outcomes=self._last_delayed_outcomes,
            last_delayed_attributions=self._last_delayed_attributions,
            delayed_attribution_ledger=tuple(self._delayed_attribution_ledger),
            delayed_payoffs=self._sorted_delayed_payoffs(),
            turn_evaluation_scores=tuple(self._turn_evaluation_scores),
            regime_sequence=tuple(self._regime_sequence),
            sequence_payoffs=self._sorted_sequence_payoffs(),
            attribution_horizons=self._attribution_horizons,
        )

    def restore_checkpoint(self, checkpoint: RegimeCheckpoint) -> None:
        self._historical_effectiveness = dict(checkpoint.historical_effectiveness)
        self._strategy_priors = dict(checkpoint.strategy_priors)
        self._active_regime_id = checkpoint.active_regime_id
        self._previous_regime_id = checkpoint.previous_regime_id
        self._turns_in_current_regime = checkpoint.turns_in_current_regime
        self._turn_index = checkpoint.turn_index
        self._pending_outcomes = list(checkpoint.pending_outcomes)
        self._last_delayed_outcomes = checkpoint.last_delayed_outcomes
        self._last_delayed_attributions = checkpoint.last_delayed_attributions
        self._delayed_attribution_ledger = list(checkpoint.delayed_attribution_ledger)
        self._delayed_payoffs = {
            (item.regime_id, item.abstract_action, item.action_family_version): item
            for item in checkpoint.delayed_payoffs
        }
        self._turn_evaluation_scores = list(checkpoint.turn_evaluation_scores)
        self._regime_sequence = list(checkpoint.regime_sequence)
        self._sequence_payoffs = {
            (item.regime_sequence, item.family_version): item
            for item in checkpoint.sequence_payoffs
        }
        self._attribution_horizons = checkpoint.attribution_horizons
