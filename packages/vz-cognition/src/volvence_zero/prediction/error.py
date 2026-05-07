from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import CommitmentSnapshot
from volvence_zero.substrate import SubstrateSnapshot, feature_signal_value

if TYPE_CHECKING:
    from volvence_zero.regime import RegimeSnapshot


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
class PredictionActionContext:
    segment_id: str = ""
    abstract_action_id: str = ""
    z_t_digest: tuple[float, ...] = ()
    regime_id: str = ""
    affordance_name: str = ""
    environment_event_id: str = ""
    environment_outcome_id: str = ""


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
    action_context: PredictionActionContext = field(default_factory=PredictionActionContext)


@dataclass(frozen=True)
class ActualOutcome:
    observed_turn_index: int
    task_progress: float
    relationship_delta: float
    regime_stability: float
    action_payoff: float
    description: str
    action_context: PredictionActionContext = field(default_factory=PredictionActionContext)
    # Provenance: evidence_ids of DialogueExternalOutcomeEvidence entries
    # that biased this outcome. Empty tuple when PE derived the outcome
    # purely from internal upstream signals. Populated via
    # ``_apply_external_outcome_bias``; never set by external writers.
    external_outcome_refs: tuple[str, ...] = ()


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
class PEDecomposition:
    """Owner-internal, running-stats based PE decomposition (Phase 1.B).

    Inspired by Curiosity-Critic (PE = aleatoric noise floor + epistemic
    reducible part). The PE owner maintains an EMA mean / variance per
    ``(axis, regime_id_or_segment_id)`` bucket. The variance floor is
    treated as aleatoric (noise the policy cannot remove). The
    deviation of the absolute error above the noise floor is treated
    as epistemic (the part learning can drive down).

    Both magnitudes are normalized to ``[0, 1]``. ``per_axis`` lists each
    axis' (axis_name, aleatoric, epistemic) for fine-grained inspection.
    Field ordering and typing are append-only stable for downstream
    consumers.

    This is a Phase 1 readout: it is **not** an acceptance gate input
    and does not feed any RL reward. Phase 2.B uplift may replace the
    running stats with a learned critic head; see
    ``research/papers/core-author-paper-assessment-2026-05.md``.
    """

    aleatoric_magnitude: float
    epistemic_magnitude: float
    per_axis: tuple[tuple[str, float, float], ...]
    description: str = ""
    critic_predicted_magnitude: float = 0.0
    improvement_magnitude: float = 0.0
    critic_update_count: int = 0
    critic_checkpoint_id: str = ""
    critic_gate_decision: str = "shadow"


@dataclass(frozen=True)
class PECriticHeadState:
    rule_id: str
    feature_dim: int
    update_count: int
    axis_weights: tuple[tuple[str, tuple[float, ...]], ...]
    axis_biases: tuple[tuple[str, float], ...]
    last_prediction: float
    last_target: float
    last_validation_delta: float
    last_capacity_cost: float
    last_rollback_evidence: str
    description: str = ""


@dataclass(frozen=True)
class PredictionErrorSnapshot:
    evaluated_prediction: PredictedOutcome | None
    actual_outcome: ActualOutcome
    next_prediction: PredictedOutcome
    error: PredictionError
    turn_index: int
    bootstrap: bool
    description: str
    action_context: PredictionActionContext = field(default_factory=PredictionActionContext)
    memory_retrieval_facets: tuple[str, ...] = ()
    pe_decomposition: PEDecomposition | None = None

    def __post_init__(self) -> None:
        if self.memory_retrieval_facets:
            return
        if self.bootstrap:
            object.__setattr__(self, "memory_retrieval_facets", ())
            return
        dominant_dimension = max(
            (
                ("task", abs(self.error.task_error)),
                ("relationship", abs(self.error.relationship_error)),
                ("regime", abs(self.error.regime_error)),
                ("action", abs(self.error.action_error)),
            ),
            key=lambda item: item[1],
        )[0]
        object.__setattr__(
            self,
            "memory_retrieval_facets",
            (
                f"prediction_error:{dominant_dimension}",
                f"prediction_reward:{self.error.signed_reward:.2f}",
            ),
        )


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
        action_context: PredictionActionContext,
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
            action_context=action_context,
        )

    def build_actual_outcome(
        self,
        *,
        observed_turn_index: int,
        evidence: _OutcomeEvidence,
        action_context: PredictionActionContext,
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
            action_context=action_context,
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
    # dialogue_external_outcome (Rupture-and-Repair M2) is added so the
    # PE owner can fuse externally-confirmed outcome signals into its
    # internally-derived actual outcome; the single legal channel for
    # external outcomes into the kernel. No external writer mutates PE
    # state directly; PE consumes the snapshot inside its own ``process``.
    # Consumers still read a single unified ``PredictionErrorSnapshot``; the
    # commitment contribution is overlaid inside _advance and described in
    # the snapshot description so the origin remains auditable.
    dependencies = (
        "substrate",
        "evaluation",
        "dual_track",
        "regime",
        "commitment",
        "dialogue_external_outcome",
    )
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        action_context: PredictionActionContext | None = None,
        pe_critic_decay: float = 0.9,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._previous_prediction: PredictedOutcome | None = None
        self._previous_substrate_snapshot: SubstrateSnapshot | None = None
        self._previous_alignment_by_record: dict[str, str] = {}
        self._turn_index = 0
        self._outcome_head = _PredictionErrorHead()
        self._action_context = action_context or PredictionActionContext()
        # Phase 1.B: owner-internal Curiosity-Critic running-stats head.
        # Decay is the EMA retention factor; pe_critic_decay = 0.9 means
        # each new sample contributes 10% to the EMA mean / variance.
        self._critic = _PECriticHead(decay=pe_critic_decay)

    def set_action_context(self, action_context: PredictionActionContext | None) -> None:
        """Set the current-turn action context without resetting prediction state."""
        self._action_context = action_context or PredictionActionContext()

    def compute_prediction(
        self,
        *,
        source_turn_index: int,
        substrate_snapshot: SubstrateSnapshot | None,
        previous_substrate_snapshot: SubstrateSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot,
        dual_track_snapshot: DualTrackSnapshot,
        regime_snapshot: RegimeSnapshot | None,
        action_context: PredictionActionContext | None = None,
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
            action_context=action_context or PredictionActionContext(),
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
        external_outcome_snapshot = upstream.get("dialogue_external_outcome")
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        evaluation_value = evaluation_snapshot.value if isinstance(evaluation_snapshot.value, EvaluationSnapshot) else None
        dual_track_value = dual_track_snapshot.value if isinstance(dual_track_snapshot.value, DualTrackSnapshot) else None
        regime_value = regime_snapshot.value if isinstance(regime_snapshot.value, RegimeSnapshot) else None
        commitment_value: CommitmentSnapshot | None = None
        if commitment_snapshot is not None and isinstance(
            commitment_snapshot.value, CommitmentSnapshot
        ):
            commitment_value = commitment_snapshot.value
        external_outcome_value: DialogueExternalOutcomeSnapshot | None = None
        if (
            external_outcome_snapshot is not None
            and isinstance(
                external_outcome_snapshot.value, DialogueExternalOutcomeSnapshot
            )
        ):
            external_outcome_value = external_outcome_snapshot.value
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
            external_outcome_snapshot=external_outcome_value,
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
        external_outcome_snapshot = kwargs.get("external_outcome_snapshot")
        previous_prediction = kwargs.get("previous_prediction")
        previous_substrate_snapshot = kwargs.get("previous_substrate_snapshot")
        turn_index = int(kwargs.get("turn_index", 0))
        if not isinstance(evaluation_snapshot, EvaluationSnapshot) or not isinstance(dual_track_snapshot, DualTrackSnapshot):
            return self.publish(_bootstrap_snapshot(turn_index=turn_index))
        regime_value = regime_snapshot if isinstance(regime_snapshot, RegimeSnapshot) else None
        commitment_value = (
            commitment_snapshot if isinstance(commitment_snapshot, CommitmentSnapshot) else None
        )
        external_outcome_value = (
            external_outcome_snapshot
            if isinstance(external_outcome_snapshot, DialogueExternalOutcomeSnapshot)
            else None
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
                external_outcome_snapshot=external_outcome_value,
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
        external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None = None,
    ) -> PredictionErrorSnapshot:
        next_prediction = self.compute_prediction(
            source_turn_index=turn_index,
            substrate_snapshot=substrate_snapshot,
            previous_substrate_snapshot=previous_substrate_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
            action_context=self._action_context,
        )
        actual_outcome = derive_actual_outcome(
            observed_turn_index=turn_index,
            substrate_snapshot=substrate_snapshot,
            previous_substrate_snapshot=previous_substrate_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
            action_context=self._action_context,
            external_outcome_snapshot=external_outcome_snapshot,
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
        # Phase 1.B: owner-internal Curiosity-Critic readout. Bootstrap
        # turns publish ``pe_decomposition=None`` so legacy consumers
        # that ignore the field stay byte-for-byte compatible.
        decomposition: PEDecomposition | None = None
        if not bootstrap:
            decomposition = self._critic.update(
                error=error,
                action_context=self._action_context,
                substrate_snapshot=substrate_snapshot,
                timestamp_ms=turn_index,
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
            action_context=self._action_context,
            pe_decomposition=decomposition,
        )


def derive_actual_outcome(
    *,
    observed_turn_index: int,
    substrate_snapshot: SubstrateSnapshot | None,
    previous_substrate_snapshot: SubstrateSnapshot | None,
    evaluation_snapshot: EvaluationSnapshot,
    dual_track_snapshot: DualTrackSnapshot,
    regime_snapshot: RegimeSnapshot | None,
    action_context: PredictionActionContext | None = None,
    external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None = None,
) -> ActualOutcome:
    evidence = _build_outcome_evidence(
        substrate_snapshot=substrate_snapshot,
        previous_substrate_snapshot=previous_substrate_snapshot,
        evaluation_snapshot=evaluation_snapshot,
        dual_track_snapshot=dual_track_snapshot,
        regime_snapshot=regime_snapshot,
    )
    base = _PredictionErrorHead().build_actual_outcome(
        observed_turn_index=observed_turn_index,
        evidence=evidence,
        action_context=action_context or PredictionActionContext(),
    )
    return _apply_external_outcome_bias(
        base=base,
        external_outcome_snapshot=external_outcome_snapshot,
    )


# Per-kind bias table for external outcomes. Each bias is a signed delta
# applied to the corresponding axis of ``ActualOutcome``; the result is
# clamped to the axis' valid range (task / regime / action to ``[0, 1]``,
# relationship_delta to ``[-1, 1]``). Values are scaled by the evidence's
# confidence before being applied. The table is a documented static
# mapping, NOT learned — learned outcome weighting is explicitly
# post-v0 (see docs/specs/rupture-and-repair.md).
_EXTERNAL_OUTCOME_AXIS_BIAS: dict[
    DialogueExternalOutcomeKind,
    tuple[float, float, float, float],
] = {
    # (task_progress_delta, relationship_delta, regime_stability_delta, action_payoff_delta)
    DialogueExternalOutcomeKind.HELPED: (0.0, +0.50, 0.0, +0.30),
    DialogueExternalOutcomeKind.FELT_HEARD: (0.0, +0.60, 0.0, 0.0),
    DialogueExternalOutcomeKind.DECISION_CLEARER: (+0.50, 0.0, 0.0, +0.50),
    DialogueExternalOutcomeKind.COME_BACK: (0.0, 0.0, -0.20, -0.30),
    DialogueExternalOutcomeKind.MISSED: (0.0, -0.60, 0.0, -0.20),
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: (0.0, -0.40, -0.10, -0.30),
    DialogueExternalOutcomeKind.UNSAFE: (0.0, -0.70, -0.30, -0.50),
    DialogueExternalOutcomeKind.ABANDONED: (-0.40, -0.60, 0.0, -0.40),
}


def _apply_external_outcome_bias(
    *,
    base: ActualOutcome,
    external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None,
) -> ActualOutcome:
    """Bias ``base`` by per-entry kind deltas from the external snapshot.

    This helper preserves provenance via ``external_outcome_refs``. The
    bias magnitudes are the only place where the external-outcome
    vocabulary meets the PE axes; the mapping is documented and closed.
    """

    if external_outcome_snapshot is None:
        return base
    entries: tuple[DialogueExternalOutcomeEvidence, ...] = external_outcome_snapshot.entries
    if not entries:
        return base

    task_bias = 0.0
    relationship_bias = 0.0
    regime_bias = 0.0
    action_bias = 0.0
    refs: list[str] = []
    for entry in entries:
        weights = _EXTERNAL_OUTCOME_AXIS_BIAS.get(entry.kind)
        if weights is None:
            continue
        scale = max(0.0, min(1.0, float(entry.confidence)))
        task_delta, rel_delta, regime_delta, action_delta = weights
        task_bias += task_delta * scale
        relationship_bias += rel_delta * scale
        regime_bias += regime_delta * scale
        action_bias += action_delta * scale
        refs.append(entry.evidence_id)

    if not refs:
        return base

    task_progress = _clamp_unit(base.task_progress + task_bias)
    relationship_delta = _clamp_signed(base.relationship_delta + relationship_bias)
    regime_stability = _clamp_unit(base.regime_stability + regime_bias)
    action_payoff = _clamp_unit(base.action_payoff + action_bias)
    ref_list = ", ".join(refs[:3]) + (f" (+{len(refs) - 3} more)" if len(refs) > 3 else "")
    return ActualOutcome(
        observed_turn_index=base.observed_turn_index,
        task_progress=task_progress,
        relationship_delta=relationship_delta,
        regime_stability=regime_stability,
        action_payoff=action_payoff,
        description=(
            f"{base.description} | external-outcome bias applied "
            f"task+{task_bias:.2f} rel{relationship_bias:+.2f} "
            f"regime{regime_bias:+.2f} action{action_bias:+.2f} refs=[{ref_list}]"
        ),
        action_context=base.action_context,
        external_outcome_refs=tuple(refs),
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


_PE_AXES: tuple[str, ...] = ("task", "relationship", "regime", "action")


@dataclass
class _AxisRunningStats:
    """EMA mean / variance for one (axis, bucket_key) cell.

    Welford-style is overkill here; a simple EMA (``alpha`` = 1 - decay)
    is enough to filter out short-window noise without retaining a full
    history. Decay is supplied by the owner so it stays auditable and
    is not hardcoded into the dataclass.
    """

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    def update(self, *, value: float, decay: float) -> None:
        # alpha is the new-sample weight: alpha = 1 - decay. With
        # decay = 0.9 each new sample contributes 10% to the EMA mean
        # and the variance estimator follows the standard EMA form:
        #   delta = value - prev_mean
        #   new_mean = prev_mean + alpha * delta
        #   new_variance = (1 - alpha) * (prev_variance + alpha * delta**2)
        if self.count == 0:
            self.mean = value
            self.variance = 0.0
            self.count = 1
            return
        alpha = max(0.0, min(1.0, 1.0 - decay))
        delta = value - self.mean
        self.mean = self.mean + alpha * delta
        self.variance = (1.0 - alpha) * (self.variance + alpha * delta * delta)
        self.count += 1


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _unit_digest(value: str, *, width: int = 2) -> tuple[float, ...]:
    if not value:
        return tuple(0.0 for _ in range(width))
    total = sum((index + 1) * ord(char) for index, char in enumerate(value))
    return tuple(
        ((total + index * 131) % 257) / 128.0 - 1.0
        for index in range(width)
    )


def _substrate_feature_digest(
    substrate_snapshot: SubstrateSnapshot | None,
    *,
    dim: int,
) -> tuple[float, ...]:
    if substrate_snapshot is None or not substrate_snapshot.feature_surface:
        return tuple(0.0 for _ in range(dim))
    values: list[float] = []
    for signal in sorted(substrate_snapshot.feature_surface, key=lambda item: item.name):
        values.extend(float(value) for value in signal.values)
        if len(values) >= dim:
            break
    if not values:
        return tuple(0.0 for _ in range(dim))
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    normalized = tuple(_clamp_signed(value / norm) for value in values[:dim])
    if len(normalized) < dim:
        normalized = normalized + tuple(0.0 for _ in range(dim - len(normalized)))
    return normalized


def _critic_features(
    *,
    substrate_snapshot: SubstrateSnapshot | None,
    action_context: PredictionActionContext,
    dim: int,
) -> tuple[float, ...]:
    substrate_digest = _substrate_feature_digest(substrate_snapshot, dim=8)
    z_digest = tuple(action_context.z_t_digest[:4])
    if len(z_digest) < 4:
        z_digest = z_digest + tuple(0.0 for _ in range(4 - len(z_digest)))
    context_digest = (
        _unit_digest(action_context.regime_id, width=2)
        + _unit_digest(action_context.abstract_action_id, width=2)
        + _unit_digest(action_context.segment_id, width=2)
    )
    features = substrate_digest + z_digest + context_digest
    if len(features) >= dim:
        return tuple(_clamp_signed(features[index]) for index in range(dim))
    return features + tuple(0.0 for _ in range(dim - len(features)))


class _PELearnedCritic:
    """Bounded contextual regressor for expected absolute PE per axis."""

    _FEATURE_DIM = 18

    def __init__(self, *, learning_rate: float = 0.06) -> None:
        self._learning_rate = max(0.01, min(learning_rate, 0.18))
        self._axis_weights = {
            axis: tuple(0.0 for _ in range(self._FEATURE_DIM))
            for axis in _PE_AXES
        }
        self._axis_biases = {axis: 0.0 for axis in _PE_AXES}
        self._update_count = 0
        self._last_prediction = 0.0
        self._last_target = 0.0
        self._last_validation_delta = 0.0
        self._last_capacity_cost = 0.0
        self._last_rollback_evidence = ""

    @property
    def update_count(self) -> int:
        return self._update_count

    def export_state(self) -> PECriticHeadState:
        return PECriticHeadState(
            rule_id="prediction.pe_critic_head.v1",
            feature_dim=self._FEATURE_DIM,
            update_count=self._update_count,
            axis_weights=tuple(
                (axis, self._axis_weights[axis])
                for axis in _PE_AXES
            ),
            axis_biases=tuple((axis, self._axis_biases[axis]) for axis in _PE_AXES),
            last_prediction=self._last_prediction,
            last_target=self._last_target,
            last_validation_delta=self._last_validation_delta,
            last_capacity_cost=self._last_capacity_cost,
            last_rollback_evidence=self._last_rollback_evidence,
            description=(
                f"PE learned critic updates={self._update_count} "
                f"last_prediction={self._last_prediction:.3f} target={self._last_target:.3f} "
                f"validation_delta={self._last_validation_delta:.3f} "
                f"capacity_cost={self._last_capacity_cost:.3f}."
            ),
        )

    def restore_state(self, state: PECriticHeadState) -> None:
        if state.feature_dim <= 0:
            raise ValueError("PE critic feature_dim must be positive")
        weights = dict(state.axis_weights)
        biases = dict(state.axis_biases)
        self._axis_weights = {
            axis: self._align(weights.get(axis, ()))
            for axis in _PE_AXES
        }
        self._axis_biases = {
            axis: _clamp_unit(float(biases.get(axis, 0.0)))
            for axis in _PE_AXES
        }
        self._update_count = max(0, state.update_count)
        self._last_prediction = _clamp_unit(state.last_prediction)
        self._last_target = _clamp_unit(state.last_target)
        self._last_validation_delta = float(state.last_validation_delta)
        self._last_capacity_cost = _clamp_unit(state.last_capacity_cost)
        self._last_rollback_evidence = state.last_rollback_evidence

    def predict_axis(self, *, axis: str, features: tuple[float, ...], fallback: float) -> float:
        if self._update_count == 0:
            return _clamp_unit(fallback)
        aligned = self._align(features)
        weights = self._axis_weights[axis]
        prediction = self._axis_biases[axis] + sum(
            weight * value for weight, value in zip(weights, aligned, strict=True)
        )
        return _clamp_unit(prediction)

    def update(
        self,
        *,
        axis_targets: tuple[tuple[str, float], ...],
        features: tuple[float, ...],
        timestamp_ms: int,
    ) -> tuple[str, str]:
        aligned = self._align(features)
        old_predictions = tuple(
            (
                axis,
                _clamp_unit(
                    self._axis_biases[axis]
                    + sum(
                        weight * value
                        for weight, value in zip(self._axis_weights[axis], aligned, strict=True)
                    )
                ),
            )
            for axis, target in axis_targets
        )
        old_error = sum(
            abs(_clamp_unit(target) - prediction)
            for (axis, target), (_, prediction) in zip(axis_targets, old_predictions, strict=True)
        ) / max(1, len(axis_targets))
        step = self._learning_rate * max(0.25, min(1.0, _mean_abs(aligned) + 0.25))
        proposed_weights = dict(self._axis_weights)
        proposed_biases = dict(self._axis_biases)
        deltas: list[float] = []
        for axis, target in axis_targets:
            prediction = _clamp_unit(
                self._axis_biases[axis]
                + sum(
                    weight * value
                    for weight, value in zip(self._axis_weights[axis], aligned, strict=True)
                )
            )
            error = _clamp_unit(target) - prediction
            current_weights = self._axis_weights[axis]
            next_weights = tuple(
                _clamp_signed(weight + step * error * value)
                for weight, value in zip(current_weights, aligned, strict=True)
            )
            next_bias = _clamp_unit(self._axis_biases[axis] + step * error)
            deltas.extend(
                abs(next_weight - weight)
                for next_weight, weight in zip(next_weights, current_weights, strict=True)
            )
            deltas.append(abs(next_bias - self._axis_biases[axis]))
            proposed_weights[axis] = next_weights
            proposed_biases[axis] = next_bias
        new_error = 0.0
        for axis, target in axis_targets:
            prediction = _clamp_unit(
                proposed_biases[axis]
                + sum(
                    weight * value
                    for weight, value in zip(proposed_weights[axis], aligned, strict=True)
                )
            )
            new_error += abs(_clamp_unit(target) - prediction)
        new_error /= max(1, len(axis_targets))
        validation_delta = old_error - new_error
        capacity_cost = sum(deltas) / max(1, len(deltas))
        checkpoint_id = f"prediction-pe-critic:{timestamp_ms}:{self._update_count + 1}"
        gate_decision = "allow"
        if validation_delta < 0.0 or capacity_cost > 0.20 or not checkpoint_id:
            gate_decision = "block"
        if gate_decision == "allow":
            self._axis_weights = proposed_weights
            self._axis_biases = proposed_biases
            self._update_count += 1
        self._last_prediction = sum(prediction for _, prediction in old_predictions) / max(1, len(old_predictions))
        self._last_target = sum(_clamp_unit(target) for _, target in axis_targets) / max(1, len(axis_targets))
        self._last_validation_delta = validation_delta
        self._last_capacity_cost = capacity_cost
        self._last_rollback_evidence = checkpoint_id
        return gate_decision, checkpoint_id

    def _align(self, features: tuple[float, ...]) -> tuple[float, ...]:
        if not features:
            return tuple(0.0 for _ in range(self._FEATURE_DIM))
        if len(features) >= self._FEATURE_DIM:
            return tuple(_clamp_signed(features[index]) for index in range(self._FEATURE_DIM))
        return tuple(_clamp_signed(features[index % len(features)]) for index in range(self._FEATURE_DIM))


class _PECriticHead:
    """Owner-internal running-stats critic for PE decomposition.

    Maintains per-axis EMA stats per ``bucket_key`` (regime_id, falling
    back to segment_id, falling back to "default") and emits a
    ``PEDecomposition`` summarizing aleatoric vs epistemic magnitudes.

    The critic does *not* mutate ``PredictionError`` or any other
    upstream snapshot. It is fully owner-internal and is reset only by
    the owner. The decay rate is configurable via the module init.
    """

    def __init__(self, *, decay: float = 0.9) -> None:
        self._decay = max(0.0, min(0.999, decay))
        self._stats: dict[tuple[str, str], _AxisRunningStats] = {}
        self._learned = _PELearnedCritic()

    @property
    def decay(self) -> float:
        return self._decay

    def reset(self) -> None:
        self._stats.clear()

    def export_state(self) -> PECriticHeadState:
        return self._learned.export_state()

    def restore_state(self, state: PECriticHeadState) -> None:
        self._learned.restore_state(state)

    def _bucket_key(self, action_context: PredictionActionContext) -> str:
        if action_context.regime_id:
            return f"regime:{action_context.regime_id}"
        if action_context.segment_id:
            return f"segment:{action_context.segment_id}"
        if action_context.abstract_action_id:
            return f"action:{action_context.abstract_action_id}"
        return "default"

    def _axis_value(self, *, axis: str, error: PredictionError) -> float:
        if axis == "task":
            return abs(error.task_error)
        if axis == "relationship":
            return abs(error.relationship_error)
        if axis == "regime":
            return abs(error.regime_error)
        if axis == "action":
            return abs(error.action_error)
        return 0.0

    def update(
        self,
        *,
        error: PredictionError,
        action_context: PredictionActionContext,
        substrate_snapshot: SubstrateSnapshot | None = None,
        timestamp_ms: int = 0,
    ) -> PEDecomposition:
        bucket = self._bucket_key(action_context)
        features = _critic_features(
            substrate_snapshot=substrate_snapshot,
            action_context=action_context,
            dim=_PELearnedCritic._FEATURE_DIM,
        )
        per_axis: list[tuple[str, float, float]] = []
        axis_targets: list[tuple[str, float]] = []
        total_aleatoric = 0.0
        total_epistemic = 0.0
        total_predicted = 0.0
        for axis in _PE_AXES:
            value = self._axis_value(axis=axis, error=error)
            axis_targets.append((axis, value))
            stats = self._stats.setdefault((axis, bucket), _AxisRunningStats())
            stats.update(value=value, decay=self._decay)
            # Aleatoric = sqrt(variance), bounded to [0, 1]. Variance is
            # the EMA of squared deviation from EMA mean, so this is the
            # noise floor that learning cannot remove on its own.
            aleatoric = max(0.0, min(1.0, math.sqrt(max(stats.variance, 0.0))))
            learned_prediction = self._learned.predict_axis(
                axis=axis,
                features=features,
                fallback=stats.mean,
            )
            # Phase 2.B: epistemic is the improvement-PE component,
            # ``actual |axis_error| - critic_prediction``. The EMA
            # variance remains the aleatoric readout and the learned
            # critic explains away stable contextual PE.
            epistemic = max(0.0, value - learned_prediction)
            epistemic = min(1.0, epistemic)
            per_axis.append((axis, round(aleatoric, 4), round(epistemic, 4)))
            total_aleatoric += aleatoric
            total_epistemic += epistemic
            total_predicted += learned_prediction
        gate_decision, checkpoint_id = self._learned.update(
            axis_targets=tuple(axis_targets),
            features=features,
            timestamp_ms=timestamp_ms,
        )
        per_axis_count = float(len(_PE_AXES))
        aggregate_aleatoric = round(min(1.0, total_aleatoric / per_axis_count), 4)
        aggregate_epistemic = round(min(1.0, total_epistemic / per_axis_count), 4)
        aggregate_predicted = round(min(1.0, total_predicted / per_axis_count), 4)
        aggregate_improvement = aggregate_epistemic
        description = (
            f"pe_decomposition[bucket={bucket}; "
            f"aleatoric={aggregate_aleatoric:.3f}; "
            f"epistemic={aggregate_epistemic:.3f}; "
            f"critic_predicted={aggregate_predicted:.3f}; "
            f"improvement={aggregate_improvement:.3f}; "
            f"updates={self._learned.update_count}; "
            f"gate={gate_decision}; "
            f"decay={self._decay:.2f}]"
        )
        return PEDecomposition(
            aleatoric_magnitude=aggregate_aleatoric,
            epistemic_magnitude=aggregate_epistemic,
            per_axis=tuple(per_axis),
            description=description,
            critic_predicted_magnitude=aggregate_predicted,
            improvement_magnitude=aggregate_improvement,
            critic_update_count=self._learned.update_count,
            critic_checkpoint_id=checkpoint_id,
            critic_gate_decision=gate_decision,
        )


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
