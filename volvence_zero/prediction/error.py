from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import SubstrateSnapshot, feature_signal_value

if TYPE_CHECKING:
    from volvence_zero.regime import RegimeSnapshot
    from volvence_zero.temporal import TemporalAbstractionSnapshot


@dataclass(frozen=True)
class PredictedOutcome:
    source_turn_index: int
    target_turn_index: int
    predicted_task_progress: float
    predicted_relationship_delta: float
    predicted_regime_stability: float
    predicted_action_payoff: float
    confidence: float
    description: str


@dataclass(frozen=True)
class ActualOutcome:
    observed_turn_index: int
    task_progress: float
    relationship_delta: float
    regime_stability: float
    action_payoff: float
    description: str


@dataclass(frozen=True)
class PredictionError:
    task_error: float
    relationship_error: float
    regime_error: float
    action_error: float
    magnitude: float
    signed_reward: float
    description: str


@dataclass(frozen=True)
class PredictionErrorSnapshot:
    evaluated_prediction: PredictedOutcome | None
    actual_outcome: ActualOutcome
    next_prediction: PredictedOutcome
    error: PredictionError
    turn_index: int
    bootstrap: bool
    description: str


class PredictionErrorModule(RuntimeModule[PredictionErrorSnapshot]):
    slot_name = "prediction_error"
    owner = "PredictionErrorModule"
    value_type = PredictionErrorSnapshot
    dependencies = ("substrate", "evaluation", "dual_track", "regime")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(self, *, wiring_level: WiringLevel | None = None) -> None:
        super().__init__(wiring_level=wiring_level)
        self._previous_prediction: PredictedOutcome | None = None
        self._previous_substrate_snapshot: SubstrateSnapshot | None = None
        self._turn_index = 0

    def compute_prediction(
        self,
        *,
        source_turn_index: int,
        substrate_snapshot: SubstrateSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot,
        dual_track_snapshot: DualTrackSnapshot,
        regime_snapshot: RegimeSnapshot | None,
    ) -> PredictedOutcome:
        family_signals = _family_signals(evaluation_snapshot)
        substrate_signals = _substrate_semantic_signals(substrate_snapshot)
        task_progress = _clamp_unit(
            family_signals.get("task", 0.5) * 0.55
            + substrate_signals["task_pull"] * 0.45
        )
        relationship_signal = _clamp_unit(
            family_signals.get("relationship", 0.5) * 0.45
            + (1.0 - dual_track_snapshot.cross_track_tension) * 0.25
            + substrate_signals["support_pull"] * 0.30
        )
        regime_stability = 0.5
        if regime_snapshot is not None:
            trend_map = dict(regime_snapshot.effectiveness_trend)
            regime_stability = _clamp_unit(
                trend_map.get(regime_snapshot.active_regime.regime_id, 0.5)
            )
        action_payoff = _clamp_unit(
            family_signals.get("abstraction", 0.5) * 0.5
            + substrate_signals["directive_pull"] * 0.25
            + substrate_signals["exploration_pull"] * 0.25
        )
        confidence = _clamp_unit(
            family_signals.get("learning", 0.5) * 0.5 + family_signals.get("safety", 0.5) * 0.5
        )
        return PredictedOutcome(
            source_turn_index=source_turn_index,
            target_turn_index=source_turn_index + 1,
            predicted_task_progress=task_progress,
            predicted_relationship_delta=relationship_signal,
            predicted_regime_stability=regime_stability,
            predicted_action_payoff=action_payoff,
            confidence=confidence,
            description=(
                f"Predicted next-turn outcome task={task_progress:.2f} relationship={relationship_signal:.2f} "
                f"regime={regime_stability:.2f} action={action_payoff:.2f} confidence={confidence:.2f} "
                f"substrate_task={substrate_signals['task_pull']:.2f} substrate_support={substrate_signals['support_pull']:.2f}."
            ),
        )

    def compute_prediction_error(
        self,
        *,
        predicted: PredictedOutcome,
        actual_outcome: ActualOutcome,
    ) -> PredictionError:
        task_error = _clamp_signed(actual_outcome.task_progress - predicted.predicted_task_progress)
        relationship_error = _clamp_signed(
            actual_outcome.relationship_delta - predicted.predicted_relationship_delta
        )
        regime_error = _clamp_signed(
            actual_outcome.regime_stability - predicted.predicted_regime_stability
        )
        action_error = _clamp_signed(actual_outcome.action_payoff - predicted.predicted_action_payoff)
        magnitude = abs(task_error) + abs(relationship_error) + abs(regime_error) + abs(action_error)
        signed_reward = (task_error + relationship_error + regime_error + action_error) / 4.0
        return PredictionError(
            task_error=task_error,
            relationship_error=relationship_error,
            regime_error=regime_error,
            action_error=action_error,
            magnitude=round(magnitude, 4),
            signed_reward=round(signed_reward, 4),
            description=(
                f"Prediction error task={task_error:.2f} relationship={relationship_error:.2f} "
                f"regime={regime_error:.2f} action={action_error:.2f} magnitude={magnitude:.2f}."
            ),
        )

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[PredictionErrorSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        substrate_snapshot = upstream["substrate"]
        evaluation_snapshot = upstream["evaluation"]
        dual_track_snapshot = upstream["dual_track"]
        regime_snapshot = upstream["regime"]
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        evaluation_value = evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        dual_track_value = dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        regime_value = regime_snapshot.value if isinstance(regime_snapshot.value, RegimeSnapshot) else None
        if evaluation_value is None or dual_track_value is None:
            return self.publish(_bootstrap_snapshot(turn_index=self._turn_index))
        self._turn_index += 1
        snapshot = self._advance(
            turn_index=self._turn_index,
            previous_prediction=self._previous_prediction,
            previous_substrate_snapshot=self._previous_substrate_snapshot,
            substrate_snapshot=substrate_value,
            evaluation_snapshot=evaluation_value,
            dual_track_snapshot=dual_track_value,
            regime_snapshot=regime_value,
        )
        self._previous_prediction = snapshot.next_prediction
        self._previous_substrate_snapshot = substrate_value
        return self.publish(snapshot)

    async def process_standalone(self, **kwargs: Any) -> Snapshot[PredictionErrorSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        substrate_snapshot = kwargs.get("substrate_snapshot")
        evaluation_snapshot = kwargs.get("evaluation_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        regime_snapshot = kwargs.get("regime_snapshot")
        previous_prediction = kwargs.get("previous_prediction")
        previous_substrate_snapshot = kwargs.get("previous_substrate_snapshot")
        turn_index = int(kwargs.get("turn_index", 0))
        if not isinstance(evaluation_snapshot, EvaluationSnapshot) or not isinstance(dual_track_snapshot, DualTrackSnapshot):
            return self.publish(_bootstrap_snapshot(turn_index=turn_index))
        regime_value = regime_snapshot if isinstance(regime_snapshot, RegimeSnapshot) else None
        return self.publish(
            self._advance(
                turn_index=turn_index,
                previous_prediction=previous_prediction if isinstance(previous_prediction, PredictedOutcome) else None,
                previous_substrate_snapshot=(
                    previous_substrate_snapshot if isinstance(previous_substrate_snapshot, SubstrateSnapshot) else None
                ),
                substrate_snapshot=substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None,
                evaluation_snapshot=evaluation_snapshot,
                dual_track_snapshot=dual_track_snapshot,
                regime_snapshot=regime_value,
            )
        )

    def _advance(
        self,
        *,
        turn_index: int,
        previous_prediction: PredictedOutcome | None,
        previous_substrate_snapshot: SubstrateSnapshot | None,
        substrate_snapshot: SubstrateSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot,
        dual_track_snapshot: DualTrackSnapshot,
        regime_snapshot: RegimeSnapshot | None,
    ) -> PredictionErrorSnapshot:
        next_prediction = self.compute_prediction(
            source_turn_index=turn_index,
            substrate_snapshot=substrate_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
        )
        actual_outcome = derive_actual_outcome(
            observed_turn_index=turn_index,
            substrate_snapshot=substrate_snapshot,
            previous_substrate_snapshot=previous_substrate_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
        )
        bootstrap = previous_prediction is None
        evaluated_prediction = previous_prediction
        error = (
            self.compute_prediction_error(
                predicted=evaluated_prediction,
                actual_outcome=actual_outcome,
            )
            if evaluated_prediction is not None
            else PredictionError(
                task_error=0.0,
                relationship_error=0.0,
                regime_error=0.0,
                action_error=0.0,
                magnitude=0.0,
                signed_reward=0.0,
                description="Bootstrap turn: no previous prediction to evaluate yet.",
            )
        )
        return PredictionErrorSnapshot(
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            next_prediction=next_prediction,
            error=error,
            turn_index=turn_index,
            bootstrap=bootstrap,
            description=(
                f"{'Bootstrap' if bootstrap else 'Evaluated'} prediction chain at turn {turn_index}. "
                f"{error.description} Next prediction targets turn {next_prediction.target_turn_index}."
            ),
        )


def derive_actual_outcome(
    *,
    observed_turn_index: int,
    substrate_snapshot: SubstrateSnapshot | None,
    previous_substrate_snapshot: SubstrateSnapshot | None,
    evaluation_snapshot: EvaluationSnapshot,
    dual_track_snapshot: DualTrackSnapshot,
    regime_snapshot: RegimeSnapshot | None,
) -> ActualOutcome:
    family_signals = _family_signals(evaluation_snapshot)
    substrate_signals = _substrate_semantic_signals(substrate_snapshot)
    previous_substrate_signals = _substrate_semantic_signals(previous_substrate_snapshot)
    substrate_delta = _substrate_delta(substrate_snapshot, previous_substrate_snapshot)
    regime_stability = 0.5
    if regime_snapshot is not None:
        trend_map = dict(regime_snapshot.effectiveness_trend)
        regime_stability = _clamp_unit(trend_map.get(regime_snapshot.active_regime.regime_id, 0.5))
    action_payoff = _clamp_unit(
        family_signals.get("abstraction", 0.5) * 0.45
        + substrate_signals["directive_pull"] * 0.20
        + substrate_signals["exploration_pull"] * 0.20
        + substrate_delta["residual_shift"] * 0.15
    )
    task_progress = _clamp_unit(
        family_signals.get("task", 0.5) * 0.35
        + substrate_signals["task_pull"] * 0.35
        + substrate_delta["task_shift"] * 0.30
    )
    relationship_delta = _clamp_unit(
        family_signals.get("relationship", 0.5) * 0.30
        + (1.0 - dual_track_snapshot.cross_track_tension) * 0.20
        + substrate_signals["support_pull"] * 0.25
        + substrate_signals["repair_pull"] * 0.10
        + max(0.0, substrate_signals["support_pull"] - previous_substrate_signals["support_pull"]) * 0.15
    )
    return ActualOutcome(
        observed_turn_index=observed_turn_index,
        task_progress=task_progress,
        relationship_delta=relationship_delta,
        regime_stability=regime_stability,
        action_payoff=action_payoff,
        description=(
            f"Observed outcome turn={observed_turn_index} task={task_progress:.2f} "
            f"relationship={relationship_delta:.2f} regime={regime_stability:.2f} action={action_payoff:.2f} "
            f"substrate_task={substrate_signals['task_pull']:.2f} support={substrate_signals['support_pull']:.2f} "
            f"task_shift={substrate_delta['task_shift']:.2f} residual_shift={substrate_delta['residual_shift']:.2f}."
        ),
    )


def derive_actual_outcome_from_substrate(
    *,
    observed_turn_index: int,
    substrate_snapshot: SubstrateSnapshot | None,
    previous_substrate_snapshot: SubstrateSnapshot | None,
) -> ActualOutcome:
    substrate_signals = _substrate_semantic_signals(substrate_snapshot)
    previous_substrate_signals = _substrate_semantic_signals(previous_substrate_snapshot)
    substrate_delta = _substrate_delta(substrate_snapshot, previous_substrate_snapshot)
    task_progress = _clamp_unit(substrate_signals["task_pull"] * 0.7 + substrate_delta["task_shift"] * 0.3)
    relationship_delta = _clamp_unit(
        substrate_signals["support_pull"] * 0.45
        + substrate_signals["repair_pull"] * 0.20
        + max(0.0, substrate_signals["support_pull"] - previous_substrate_signals["support_pull"]) * 0.35
    )
    regime_stability = _clamp_unit(1.0 - substrate_delta["residual_shift"])
    action_payoff = _clamp_unit(
        substrate_signals["directive_pull"] * 0.35
        + substrate_signals["exploration_pull"] * 0.35
        + substrate_delta["residual_shift"] * 0.30
    )
    return ActualOutcome(
        observed_turn_index=observed_turn_index,
        task_progress=task_progress,
        relationship_delta=relationship_delta,
        regime_stability=regime_stability,
        action_payoff=action_payoff,
        description=(
            f"Substrate-derived outcome turn={observed_turn_index} task={task_progress:.2f} "
            f"relationship={relationship_delta:.2f} regime={regime_stability:.2f} action={action_payoff:.2f}."
        ),
    )


def _family_signals(evaluation_snapshot: EvaluationSnapshot) -> dict[str, float]:
    families: dict[str, list[float]] = {}
    for score in evaluation_snapshot.turn_scores + evaluation_snapshot.session_scores:
        value = 1.0 - score.value if score.metric_name == "fallback_reliance" else score.value
        families.setdefault(score.family, []).append(value)
    return {
        family: _clamp_unit(sum(values) / len(values)) if values else 0.5
        for family, values in families.items()
    }


def _substrate_semantic_signals(substrate_snapshot: SubstrateSnapshot | None) -> dict[str, float]:
    if substrate_snapshot is None:
        return {
            "task_pull": 0.5,
            "support_pull": 0.5,
            "repair_pull": 0.5,
            "exploration_pull": 0.5,
            "directive_pull": 0.5,
        }
    return {
        "task_pull": _clamp_unit(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_task_pull", default=0.5)),
        "support_pull": _clamp_unit(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_support_pull", default=0.5)),
        "repair_pull": _clamp_unit(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_repair_pull", default=0.5)),
        "exploration_pull": _clamp_unit(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_exploration_pull", default=0.5)),
        "directive_pull": _clamp_unit(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_directive_pull", default=0.5)),
    }


def _substrate_delta(
    substrate_snapshot: SubstrateSnapshot | None,
    previous_substrate_snapshot: SubstrateSnapshot | None,
) -> dict[str, float]:
    current = _substrate_semantic_signals(substrate_snapshot)
    previous = _substrate_semantic_signals(previous_substrate_snapshot)
    current_length = len(substrate_snapshot.residual_sequence) if substrate_snapshot is not None else 0
    previous_length = len(previous_substrate_snapshot.residual_sequence) if previous_substrate_snapshot is not None else 0
    length_delta = _clamp_unit(0.5 + (current_length - previous_length) * 0.1)
    task_shift = _clamp_unit(0.5 + current["task_pull"] - previous["task_pull"])
    support_shift = _clamp_unit(0.5 + current["support_pull"] - previous["support_pull"])
    directive_shift = _clamp_unit(0.5 + current["directive_pull"] - previous["directive_pull"])
    residual_shift = abs(task_shift - 0.5) + abs(support_shift - 0.5) + abs(directive_shift - 0.5)
    residual_shift = _clamp_unit(residual_shift / 1.5 * 0.5 + length_delta * 0.5)
    return {
        "length_delta": length_delta,
        "task_shift": task_shift,
        "support_shift": support_shift,
        "directive_shift": directive_shift,
        "residual_shift": residual_shift,
    }


def _bootstrap_snapshot(*, turn_index: int) -> PredictionErrorSnapshot:
    predicted = PredictedOutcome(
        source_turn_index=turn_index,
        target_turn_index=turn_index + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.0,
        description="Bootstrap prediction placeholder.",
    )
    actual = ActualOutcome(
        observed_turn_index=turn_index,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="Bootstrap actual outcome placeholder.",
    )
    error = PredictionError(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Bootstrap prediction error placeholder.")
    return PredictionErrorSnapshot(
        evaluated_prediction=None,
        actual_outcome=actual,
        next_prediction=predicted,
        error=error,
        turn_index=turn_index,
        bootstrap=True,
        description="Bootstrap prediction error snapshot.",
    )


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))
