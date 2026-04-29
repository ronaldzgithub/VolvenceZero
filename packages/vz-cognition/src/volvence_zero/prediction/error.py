from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation.backbone import EvaluationSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import CommitmentSnapshot
from volvence_zero.substrate import SubstrateSnapshot, feature_signal_value

if TYPE_CHECKING:
    from volvence_zero.regime import RegimeSnapshot
    from volvence_zero.temporal import TemporalAbstractionSnapshot


# AAC alignment transition severities (Gap 7 / docs/specs/aac-lifecycle.md).
# Moving from unknown to any known state is a low-PE disclosure (user is
# finally telling us where they stand); the high-PE events are transitions
# WITHIN the known-states lattice, especially away from AGREE. Magnitudes
# are chosen to keep the alignment contribution bounded to the same
# [0, 1] envelope as the existing 4-axis errors (see
# ``PredictionError.magnitude``).
_ALIGNMENT_TRANSITION_SEVERITY: dict[tuple[str, str], float] = {
    # unknown -> any known: low (initial disclosure signal)
    ("unknown", "agree"): 0.10,
    ("unknown", "modify"): 0.25,
    ("unknown", "reject"): 0.45,
    # AGREE -> weaker alignment: highest PE (expectations violated)
    ("agree", "modify"): 0.45,
    ("agree", "reject"): 0.90,
    # MODIFY -> stronger alignment: positive PE (recovered alignment)
    ("modify", "agree"): 0.20,
    # MODIFY -> reject: high PE (alignment failed)
    ("modify", "reject"): 0.70,
    # REJECT recovery paths: moderate-positive PE
    ("reject", "modify"): 0.35,
    ("reject", "agree"): 0.60,
}


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


@dataclass(frozen=True)
class _OutcomeEvidence:
    family_signals: dict[str, float]
    substrate_signals: dict[str, float]
    previous_substrate_signals: dict[str, float]
    substrate_delta: dict[str, float]
    cross_track_tension: float
    regime_stability: float


@dataclass(frozen=True)
class _OutcomeAxisCalibration:
    family_weight: float
    substrate_weight: float
    delta_weight: float
    continuity_weight: float = 0.0
    stability_weight: float = 0.0
    tension_weight: float = 0.0
    novelty_weight: float = 0.0


class _PredictionErrorHead:
    """Owner-side mapper from runtime evidence to prediction/outcome axes."""

    def __init__(self) -> None:
        self._prediction_axes = {
            "task": _OutcomeAxisCalibration(
                family_weight=0.40,
                substrate_weight=0.35,
                delta_weight=0.10,
                continuity_weight=0.05,
                novelty_weight=0.10,
            ),
            "relationship": _OutcomeAxisCalibration(
                family_weight=0.30,
                substrate_weight=0.28,
                delta_weight=0.12,
                continuity_weight=0.12,
                stability_weight=0.08,
                tension_weight=0.10,
            ),
            "regime": _OutcomeAxisCalibration(
                family_weight=0.10,
                substrate_weight=0.08,
                delta_weight=0.05,
                stability_weight=0.67,
                tension_weight=0.10,
            ),
            "action": _OutcomeAxisCalibration(
                family_weight=0.32,
                substrate_weight=0.28,
                delta_weight=0.18,
                continuity_weight=0.10,
                novelty_weight=0.12,
            ),
        }
        self._actual_axes = {
            "task": _OutcomeAxisCalibration(
                family_weight=0.28,
                substrate_weight=0.30,
                delta_weight=0.24,
                continuity_weight=0.04,
                novelty_weight=0.14,
            ),
            "relationship": _OutcomeAxisCalibration(
                family_weight=0.24,
                substrate_weight=0.26,
                delta_weight=0.16,
                continuity_weight=0.14,
                stability_weight=0.08,
                tension_weight=0.12,
            ),
            "regime": _OutcomeAxisCalibration(
                family_weight=0.05,
                substrate_weight=0.05,
                delta_weight=0.05,
                stability_weight=0.75,
                tension_weight=0.10,
            ),
            "action": _OutcomeAxisCalibration(
                family_weight=0.22,
                substrate_weight=0.28,
                delta_weight=0.28,
                continuity_weight=0.08,
                novelty_weight=0.14,
            ),
        }

    def build_prediction(
        self,
        *,
        source_turn_index: int,
        evidence: _OutcomeEvidence,
    ) -> PredictedOutcome:
        task_progress = self._axis_value("task", evidence=evidence, calibrations=self._prediction_axes)
        relationship_signal = self._axis_value("relationship", evidence=evidence, calibrations=self._prediction_axes)
        regime_stability = self._axis_value("regime", evidence=evidence, calibrations=self._prediction_axes)
        action_payoff = self._axis_value("action", evidence=evidence, calibrations=self._prediction_axes)
        confidence = self._confidence(evidence=evidence)
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
                f"task_signal={self._task_signal(evidence):.2f} relationship_signal={self._relationship_signal(evidence):.2f}."
            ),
        )

    def build_actual_outcome(
        self,
        *,
        observed_turn_index: int,
        evidence: _OutcomeEvidence,
    ) -> ActualOutcome:
        task_progress = self._axis_value("task", evidence=evidence, calibrations=self._actual_axes)
        relationship_delta = self._axis_value("relationship", evidence=evidence, calibrations=self._actual_axes)
        regime_stability = self._axis_value("regime", evidence=evidence, calibrations=self._actual_axes)
        action_payoff = self._axis_value("action", evidence=evidence, calibrations=self._actual_axes)
        return ActualOutcome(
            observed_turn_index=observed_turn_index,
            task_progress=task_progress,
            relationship_delta=relationship_delta,
            regime_stability=regime_stability,
            action_payoff=action_payoff,
            description=(
                f"Observed outcome turn={observed_turn_index} task={task_progress:.2f} "
                f"relationship={relationship_delta:.2f} regime={regime_stability:.2f} action={action_payoff:.2f} "
                f"task_shift={evidence.substrate_delta['task_shift']:.2f} support_shift={evidence.substrate_delta['support_shift']:.2f} "
                f"residual_shift={evidence.substrate_delta['residual_shift']:.2f}."
            ),
        )

    def compute_error(
        self,
        *,
        predicted: PredictedOutcome,
        actual_outcome: ActualOutcome,
    ) -> PredictionError:
        task_error = _clamp_signed(actual_outcome.task_progress - predicted.predicted_task_progress)
        relationship_error = _clamp_signed(
            actual_outcome.relationship_delta - predicted.predicted_relationship_delta
        )
        regime_error = _clamp_signed(actual_outcome.regime_stability - predicted.predicted_regime_stability)
        action_error = _clamp_signed(actual_outcome.action_payoff - predicted.predicted_action_payoff)
        weighted_axis_errors = self._weighted_axis_errors(
            predicted=predicted,
            task_error=task_error,
            relationship_error=relationship_error,
            regime_error=regime_error,
            action_error=action_error,
        )
        weighted_abs_error = sum(abs(error) * weight for _, error, weight in weighted_axis_errors)
        weighted_signed_error = sum(error * weight for _, error, weight in weighted_axis_errors)
        magnitude = _clamp_unit(weighted_abs_error) * 4.0
        signed_reward = _clamp_signed(weighted_signed_error)
        weight_text = ", ".join(
            f"{axis}={weight:.2f}" for axis, _, weight in weighted_axis_errors
        )
        return PredictionError(
            task_error=task_error,
            relationship_error=relationship_error,
            regime_error=regime_error,
            action_error=action_error,
            magnitude=round(magnitude, 4),
            signed_reward=round(signed_reward, 4),
            description=(
                f"Prediction error task={task_error:.2f} relationship={relationship_error:.2f} "
                f"regime={regime_error:.2f} action={action_error:.2f} magnitude={magnitude:.2f} "
                f"signed_reward={signed_reward:.2f} weighted_axes[{weight_text}]."
            ),
        )

    def _axis_value(
        self,
        axis_name: str,
        *,
        evidence: _OutcomeEvidence,
        calibrations: dict[str, _OutcomeAxisCalibration],
    ) -> float:
        calibration = calibrations[axis_name]
        base_signal = {
            "task": self._task_signal(evidence),
            "relationship": self._relationship_signal(evidence),
            "regime": evidence.regime_stability,
            "action": self._action_signal(evidence),
        }[axis_name]
        delta_signal = {
            "task": evidence.substrate_delta["task_shift"],
            "relationship": evidence.substrate_delta["support_shift"],
            "regime": 1.0 - evidence.substrate_delta["residual_shift"],
            "action": evidence.substrate_delta["directive_shift"],
        }[axis_name]
        continuity_signal = {
            "task": evidence.previous_substrate_signals["task_pull"],
            "relationship": evidence.previous_substrate_signals["support_pull"],
            "regime": evidence.regime_stability,
            "action": evidence.previous_substrate_signals["directive_pull"],
        }[axis_name]
        novelty_signal = {
            "task": evidence.substrate_delta["length_delta"],
            "relationship": evidence.substrate_delta["support_shift"],
            "regime": evidence.substrate_delta["residual_shift"],
            "action": evidence.substrate_delta["residual_shift"],
        }[axis_name]
        tension_signal = 1.0 - evidence.cross_track_tension
        total_weight = (
            calibration.family_weight
            + calibration.substrate_weight
            + calibration.delta_weight
            + calibration.continuity_weight
            + calibration.stability_weight
            + calibration.tension_weight
            + calibration.novelty_weight
        )
        if total_weight <= 0.0:
            return 0.5
        weighted_sum = (
            base_signal * (calibration.family_weight + calibration.substrate_weight)
            + delta_signal * calibration.delta_weight
            + continuity_signal * calibration.continuity_weight
            + evidence.regime_stability * calibration.stability_weight
            + tension_signal * calibration.tension_weight
            + novelty_signal * calibration.novelty_weight
        )
        return _clamp_unit(weighted_sum / total_weight)

    def _task_signal(self, evidence: _OutcomeEvidence) -> float:
        return _clamp_unit(
            evidence.family_signals.get("task", 0.5) * 0.52
            + evidence.substrate_signals["task_pull"] * 0.33
            + evidence.substrate_signals["directive_pull"] * 0.15
        )

    def _relationship_signal(self, evidence: _OutcomeEvidence) -> float:
        return _clamp_unit(
            evidence.family_signals.get("relationship", 0.5) * 0.35
            + evidence.substrate_signals["support_pull"] * 0.28
            + evidence.substrate_signals["repair_pull"] * 0.17
            + (1.0 - evidence.cross_track_tension) * 0.20
        )

    def _action_signal(self, evidence: _OutcomeEvidence) -> float:
        return _clamp_unit(
            evidence.family_signals.get("abstraction", 0.5) * 0.34
            + evidence.substrate_signals["directive_pull"] * 0.26
            + evidence.substrate_signals["exploration_pull"] * 0.18
            + evidence.substrate_delta["residual_shift"] * 0.22
        )

    def _confidence(self, *, evidence: _OutcomeEvidence) -> float:
        learning_signal = evidence.family_signals.get("learning", 0.5)
        safety_signal = evidence.family_signals.get("safety", 0.5)
        stability_signal = evidence.regime_stability
        coherence_signal = 1.0 - min(
            abs(evidence.substrate_delta["task_shift"] - 0.5)
            + abs(evidence.substrate_delta["support_shift"] - 0.5)
            + abs(evidence.substrate_delta["directive_shift"] - 0.5),
            1.0,
        )
        return _clamp_unit(
            learning_signal * 0.35
            + safety_signal * 0.35
            + stability_signal * 0.20
            + coherence_signal * 0.10
        )

    def _weighted_axis_errors(
        self,
        *,
        predicted: PredictedOutcome,
        task_error: float,
        relationship_error: float,
        regime_error: float,
        action_error: float,
    ) -> tuple[tuple[str, float, float], ...]:
        axis_predictions = (
            ("task", predicted.predicted_task_progress, task_error),
            ("relationship", predicted.predicted_relationship_delta, relationship_error),
            ("regime", predicted.predicted_regime_stability, regime_error),
            ("action", predicted.predicted_action_payoff, action_error),
        )
        raw_weights = []
        for axis_name, predicted_value, _ in axis_predictions:
            expectation_strength = abs(predicted_value - 0.5) * 2.0
            raw_weight = 0.55 + predicted.confidence * 0.30 + expectation_strength * 0.15
            raw_weights.append((axis_name, raw_weight))
        total_weight = sum(weight for _, weight in raw_weights)
        normalized = {
            axis_name: weight / total_weight
            for axis_name, weight in raw_weights
        }
        return tuple(
            (axis_name, axis_error, normalized[axis_name])
            for axis_name, _, axis_error in axis_predictions
        )


class PredictionErrorModule(RuntimeModule[PredictionErrorSnapshot]):
    slot_name = "prediction_error"
    owner = "PredictionErrorModule"
    value_type = PredictionErrorSnapshot
    # Commitment is added as a dependency so that AAC alignment transitions
    # (Gap 7 / docs/specs/aac-lifecycle.md) can enter the PE signal chain
    # as a discrete-event PE source distinct from continuous substrate PE.
    # Consumers still read a single unified ``PredictionErrorSnapshot``; the
    # commitment contribution is overlaid inside _advance and described in
    # the snapshot description so the origin remains auditable.
    dependencies = ("substrate", "evaluation", "dual_track", "regime", "commitment")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(self, *, wiring_level: WiringLevel | None = None) -> None:
        super().__init__(wiring_level=wiring_level)
        self._previous_prediction: PredictedOutcome | None = None
        self._previous_substrate_snapshot: SubstrateSnapshot | None = None
        self._previous_alignment_by_record: dict[str, str] = {}
        self._turn_index = 0
        self._outcome_head = _PredictionErrorHead()

    def compute_prediction(
        self,
        *,
        source_turn_index: int,
        substrate_snapshot: SubstrateSnapshot | None,
        previous_substrate_snapshot: SubstrateSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot,
        dual_track_snapshot: DualTrackSnapshot,
        regime_snapshot: RegimeSnapshot | None,
    ) -> PredictedOutcome:
        evidence = _build_outcome_evidence(
            substrate_snapshot=substrate_snapshot,
            previous_substrate_snapshot=previous_substrate_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
        )
        return self._outcome_head.build_prediction(
            source_turn_index=source_turn_index,
            evidence=evidence,
        )

    def compute_prediction_error(
        self,
        *,
        predicted: PredictedOutcome,
        actual_outcome: ActualOutcome,
    ) -> PredictionError:
        return self._outcome_head.compute_error(
            predicted=predicted,
            actual_outcome=actual_outcome,
        )

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[PredictionErrorSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        substrate_snapshot = upstream["substrate"]
        evaluation_snapshot = upstream["evaluation"]
        dual_track_snapshot = upstream["dual_track"]
        regime_snapshot = upstream["regime"]
        commitment_snapshot = upstream.get("commitment")
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        evaluation_value = evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        dual_track_value = dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        regime_value = regime_snapshot.value if isinstance(regime_snapshot.value, RegimeSnapshot) else None
        commitment_value: CommitmentSnapshot | None = None
        if commitment_snapshot is not None and isinstance(
            commitment_snapshot.value, CommitmentSnapshot
        ):
            commitment_value = commitment_snapshot.value
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
            commitment_snapshot=commitment_value,
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
        commitment_snapshot = kwargs.get("commitment_snapshot")
        previous_prediction = kwargs.get("previous_prediction")
        previous_substrate_snapshot = kwargs.get("previous_substrate_snapshot")
        turn_index = int(kwargs.get("turn_index", 0))
        if not isinstance(evaluation_snapshot, EvaluationSnapshot) or not isinstance(dual_track_snapshot, DualTrackSnapshot):
            return self.publish(_bootstrap_snapshot(turn_index=turn_index))
        regime_value = regime_snapshot if isinstance(regime_snapshot, RegimeSnapshot) else None
        commitment_value = (
            commitment_snapshot if isinstance(commitment_snapshot, CommitmentSnapshot) else None
        )
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
                commitment_snapshot=commitment_value,
            )
        )

    def _compute_alignment_contribution(
        self, commitment_snapshot: CommitmentSnapshot | None
    ) -> tuple[float, str]:
        """Return (signed_severity, audit_description) for this turn's deltas.

        Diffs per-record alignment state against the previous turn's map
        and sums matched transition severities. Updates the internal
        previous-state map as a side effect.

        Sign convention:

        * **Positive** signed severity = *regression* (alignment got
          weaker, e.g. AGREE -> REJECT). Indicates a relationship debt
          taken on this turn; the caller should push relationship_error
          and signed_reward in the NEGATIVE direction.
        * **Negative** signed severity = *recovery* (alignment got
          stronger, e.g. REJECT -> AGREE). The caller should push
          relationship_error and signed_reward in the POSITIVE
          direction.

        The magnitude part is always ``|signed_severity|`` and is what
        the caller adds to ``PredictionError.magnitude`` regardless of
        direction: discrete-event PE registers as surprise either way.

        Severity is clamped to ``[-1.0, 1.0]`` so the contribution never
        overwhelms the base 4-axis error budget.
        """
        if commitment_snapshot is None:
            return 0.0, ""
        # Recovery transitions: current alignment strictly stronger than
        # the previous. ``unknown -> agree`` is a disclosure + positive
        # signal (user just told us they agree); we treat it as a
        # recovery so it pushes signed_reward positive rather than
        # pretending it's a regression.
        recovery_keys: frozenset[tuple[str, str]] = frozenset(
            {
                ("unknown", "agree"),
                ("modify", "agree"),
                ("reject", "modify"),
                ("reject", "agree"),
            }
        )
        regression_weight = 0.0
        recovery_weight = 0.0
        transitions: list[str] = []
        current: dict[str, str] = {}
        for entry in commitment_snapshot.lifecycle_entries:
            current[entry.record_id] = entry.alignment_state.value
            previous_alignment = self._previous_alignment_by_record.get(
                entry.record_id, "unknown"
            )
            if previous_alignment == entry.alignment_state.value:
                continue
            key = (previous_alignment, entry.alignment_state.value)
            severity = _ALIGNMENT_TRANSITION_SEVERITY.get(key, 0.0)
            if severity <= 0.0:
                continue
            if key in recovery_keys:
                recovery_weight += severity
            else:
                regression_weight += severity
            transitions.append(
                f"{entry.record_id}:{previous_alignment}->{entry.alignment_state.value}"
            )
        self._previous_alignment_by_record = current
        signed_severity = regression_weight - recovery_weight
        if signed_severity == 0.0 and regression_weight == 0.0 and recovery_weight == 0.0:
            return 0.0, ""
        signed_severity = max(-1.0, min(1.0, signed_severity))
        sign = "regression" if signed_severity >= 0.0 else "recovery"
        summary = (
            f"alignment_transition[{sign}] severity={abs(signed_severity):.2f} "
            f"changes={';'.join(transitions[:4])}"
        )
        return signed_severity, summary

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
        commitment_snapshot: CommitmentSnapshot | None = None,
    ) -> PredictionErrorSnapshot:
        next_prediction = self.compute_prediction(
            source_turn_index=turn_index,
            substrate_snapshot=substrate_snapshot,
            previous_substrate_snapshot=previous_substrate_snapshot,
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
        base_error = (
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
        # Overlay the AAC alignment-transition contribution. We bump the
        # relationship axis (alignment is fundamentally a relationship
        # signal) and the overall magnitude, and push signed_reward in the
        # regression / recovery direction. All three are re-clamped so
        # they stay inside the existing [0,1] / [-1,1] envelopes.
        #
        # ``signed_severity`` is POSITIVE for regressions (AGREE -> REJECT)
        # and NEGATIVE for recoveries (REJECT -> AGREE). Both
        # relationship_error and signed_reward subtract it: regressions
        # pull both negative (user pushed back, we owe relationship work),
        # recoveries pull both positive (relationship debt reduced).
        signed_severity, alignment_audit = self._compute_alignment_contribution(
            commitment_snapshot
        )
        if signed_severity != 0.0:
            severity = abs(signed_severity)
            relationship_error = _clamp_signed(
                base_error.relationship_error - signed_severity
            )
            signed_reward = _clamp_signed(
                base_error.signed_reward - signed_severity
            )
            # magnitude: alignment transitions are discrete events, so we
            # lift the magnitude regardless of direction; recoveries are
            # informative PE too (they reduce relationship_error but
            # still register as a surprise relative to the last turn).
            magnitude = round(min(base_error.magnitude + severity, 4.0), 4)
            error = PredictionError(
                task_error=base_error.task_error,
                relationship_error=round(relationship_error, 4),
                regime_error=base_error.regime_error,
                action_error=base_error.action_error,
                magnitude=magnitude,
                signed_reward=round(signed_reward, 4),
                description=(
                    f"{base_error.description} {alignment_audit}."
                ),
            )
        else:
            error = base_error
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
    evidence = _build_outcome_evidence(
        substrate_snapshot=substrate_snapshot,
        previous_substrate_snapshot=previous_substrate_snapshot,
        evaluation_snapshot=evaluation_snapshot,
        dual_track_snapshot=dual_track_snapshot,
        regime_snapshot=regime_snapshot,
    )
    return _PredictionErrorHead().build_actual_outcome(
        observed_turn_index=observed_turn_index,
        evidence=evidence,
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


def _build_outcome_evidence(
    *,
    substrate_snapshot: SubstrateSnapshot | None,
    previous_substrate_snapshot: SubstrateSnapshot | None,
    evaluation_snapshot: EvaluationSnapshot,
    dual_track_snapshot: DualTrackSnapshot,
    regime_snapshot: RegimeSnapshot | None,
) -> _OutcomeEvidence:
    regime_stability = 0.5
    if regime_snapshot is not None:
        trend_map = dict(regime_snapshot.effectiveness_trend)
        regime_stability = _clamp_unit(
            trend_map.get(regime_snapshot.active_regime.regime_id, 0.5)
        )
    return _OutcomeEvidence(
        family_signals=_family_signals(evaluation_snapshot),
        substrate_signals=_substrate_semantic_signals(substrate_snapshot),
        previous_substrate_signals=_substrate_semantic_signals(previous_substrate_snapshot),
        substrate_delta=_substrate_delta(substrate_snapshot, previous_substrate_snapshot),
        cross_track_tension=_clamp_unit(dual_track_snapshot.cross_track_tension),
        regime_stability=regime_stability,
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
