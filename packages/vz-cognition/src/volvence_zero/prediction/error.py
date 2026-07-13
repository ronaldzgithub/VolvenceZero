from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.apprenticeship import ApprenticeshipAlignmentSnapshot
from volvence_zero.owner_prediction import (
    OwnerPredictionSettlement,
    OwnerPredictionSignal,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.prediction.distribution import DistributionSummary
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    CommitmentSnapshot,
    ExecutionResultSnapshot,
    GoalValueSnapshot,
    RelationshipStateSnapshot,
)
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
    # Packet A (long-horizon-closure): plan_ref / prediction_id lineage
    # threaded through from AffordanceInvoker.invoke(plan_ref=...) -> 
    # BrainSession.submit_tool_result -> EnvironmentOutcome.prediction_id
    # -> next-turn PredictionActionContext.prediction_id. Empty when the
    # caller did not supply a plan_ref (back-compat path); non-empty when
    # the affordance call was bound to a specific prior prediction id.
    prediction_id: str = ""


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
    # CP-10: owner-issued pre-action prediction id. Stamped by
    # ``PredictionErrorModule`` when it PUBLISHES ``next_prediction``; callers
    # (affordance tool loop, renderers) forward this reference as ``plan_ref``
    # instead of fabricating their own. Empty string = the prediction was
    # built outside the owner's publish path (bootstrap / standalone helpers)
    # and downstream lineage must treat the action as explicitly unknown.
    prediction_id: str = ""


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
    # Phase 2 W1.1 (DM-1) — optional per-axis distribution shape descriptor
    # over a recent PE window. ``None`` until the PE owner has observed
    # the minimum window of samples (cold-start safety). Read-only for
    # downstream consumers; only ``vitals`` and audit / evaluation
    # surfaces are intended readers in Wave 1. Existing scalar fields
    # above remain the canonical inputs for credit gate / regime scoring
    # / ModificationGate.
    distribution_summary: DistributionSummary | None = None


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
class PredictiveHeadReadout:
    """CP-11 SHADOW readout: learned world/self heads vs the hand-crafted head.

    Report-only. The live prediction chain still comes from
    ``_PredictionErrorHead``; the learned heads dual-run inside the PE owner
    and this readout carries their running mean-absolute-error next to the
    baseline's, so the ACTIVE gate (>= 0.02 improvement over >= 200 turns,
    plan CP-11) can be evaluated from artifacts without flipping anything.
    ``improvement`` is ``baseline_mae - learned_mae`` (positive = learned
    heads are better).
    """

    world_learned_mae: float
    world_baseline_mae: float
    self_learned_mae: float
    self_baseline_mae: float
    sample_count: int
    world_improvement: float
    self_improvement: float
    description: str = ""


PREDICTIVE_HEAD_CHECKPOINT_SCHEMA_VERSION = "predictive-head-checkpoint.v1"

# Rolling window used by the CP-11 self-reward autocorrelation kill check.
# 64 samples matches the PE distribution window's max size so both owner
# readouts describe the same recent horizon.
_HEAD_KILL_WINDOW = 64


class PredictiveHeadCheckpointError(ValueError):
    """Raised when a CP-11 head checkpoint cannot be restored (schema or
    dimension mismatch). Fail-loudly per the contract-first rule; callers must
    not silently continue with cold-start heads."""


@dataclass(frozen=True)
class PredictiveHeadCheckpoint:
    """CP-11 session-medium checkpoint for the SHADOW world/self heads.

    Float-only, immutable, owner-exported. Lets the learned heads survive a
    session restart (session-medium timescale) without touching the live
    prediction chain: restoring only reloads SHADOW weights and readout
    counters. Weight payloads are ``(axis, weights)`` tuples so the artifact
    is JSON-serialisable without dict ordering ambiguity.
    """

    checkpoint_id: str
    feature_dim: int
    world_weights: tuple[tuple[str, tuple[float, ...]], ...]
    self_weights: tuple[tuple[str, tuple[float, ...]], ...]
    sample_count: int
    abs_error_sums: tuple[tuple[str, float], ...]
    schema_version: str = PREDICTIVE_HEAD_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")


@dataclass(frozen=True)
class PredictiveHeadKillCriteria:
    """CP-11 kill-criteria readout: self-reward autocorrelation check.

    A learned head that stops tracking realized outcomes and instead echoes
    its own previous predictions is in a self-reward loop and must be killed
    before any ACTIVE consideration. ``kill_triggered`` is True when the
    window is full, the prediction series is highly self-correlated
    (lag-1 autocorrelation >= 0.98, with a degenerate constant series
    counted as perfectly self-correlated) AND essentially uncorrelated with
    the realized targets (<= 0.05). Report-only: the PE owner never acts on
    this automatically; operators / gate evaluators consume it.
    """

    window_size: int
    samples_in_window: int
    window_filled: bool
    prediction_self_autocorrelation: float
    prediction_target_correlation: float
    kill_triggered: bool
    description: str


def _series_correlation(
    a: list[float], b: list[float], *, degenerate: float
) -> float:
    """Pearson correlation of two equal-length series.

    ``degenerate`` is returned when either series has ~zero variance, so the
    caller chooses the conservative interpretation (a frozen prediction
    series counts as self-correlated 1.0 but target-correlated 0.0).
    """

    n = len(a)
    if n < 2:
        return degenerate
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    if var_a < 1e-12 or var_b < 1e-12:
        return degenerate
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    return max(-1.0, min(1.0, cov / math.sqrt(var_a * var_b)))


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
    # CP-12 owner prediction signal contract: per-signal mismatch records
    # for owner-published predictions settled this turn. Report-only in v1
    # (they do not enter the magnitude formula); only this owner constructs
    # OwnerPredictionSettlement values.
    owner_prediction_settlements: tuple[OwnerPredictionSettlement, ...] = ()
    # CP-11 world/self predictive heads SHADOW readout (report-only; None
    # until both heads have at least one scored sample).
    predictive_head_readout: PredictiveHeadReadout | None = None
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


def _featurize_outcome_evidence(evidence: "_OutcomeEvidence") -> tuple[float, ...]:
    """Fixed-order compact feature vector shared by the CP-11 learned heads.

    Uses stable aggregates rather than raw dict keys so the feature
    dimensionality never drifts with upstream signal vocabularies.
    """

    def _mean(values: dict[str, float], default: float = 0.5) -> float:
        if not values:
            return default
        return sum(values.values()) / len(values)

    return (
        _clamp_unit(_mean(evidence.family_signals)),
        _clamp_unit(_mean(evidence.substrate_signals)),
        _clamp_unit(_mean(evidence.previous_substrate_signals)),
        _clamp_unit(
            sum(abs(v) for v in evidence.substrate_delta.values())
            / max(len(evidence.substrate_delta), 1)
        ),
        _clamp_unit(evidence.cross_track_tension),
        _clamp_unit(evidence.regime_stability),
        1.0,  # bias
    )


class _LinearAxisHead:
    """Bounded online-SGD linear predictor for one outcome axis (CP-11).

    Owner-internal learned component: weights stay clamped, updates are
    single-step SGD on the realized outcome, and nothing here reaches the
    live prediction chain — the readout is SHADOW evidence only.
    """

    def __init__(self, *, feature_dim: int, learning_rate: float = 0.05) -> None:
        self._weights = [0.0] * feature_dim
        # Start from the neutral 0.5 prior via the bias term so the head's
        # cold-start behavior matches the bootstrap placeholder axes.
        self._weights[-1] = 0.5
        self._learning_rate = learning_rate

    def predict(self, features: tuple[float, ...]) -> float:
        raw = sum(w * f for w, f in zip(self._weights, features, strict=True))
        return _clamp_unit(raw)

    def update(self, *, features: tuple[float, ...], target: float) -> None:
        prediction = self.predict(features)
        gradient_scale = self._learning_rate * (target - prediction)
        self._weights = [
            max(-2.0, min(2.0, w + gradient_scale * f))
            for w, f in zip(self._weights, features, strict=True)
        ]

    def export_weights(self) -> tuple[float, ...]:
        return tuple(self._weights)

    def restore_weights(self, weights: tuple[float, ...]) -> None:
        if len(weights) != len(self._weights):
            raise PredictiveHeadCheckpointError(
                f"axis head expects {len(self._weights)} weights, got {len(weights)}"
            )
        self._weights = [max(-2.0, min(2.0, w)) for w in weights]


class _AxisBankPredictiveHead:
    """Shared base for the CP-11 axis-bank heads (world / self)."""

    axes: tuple[str, ...] = ()

    def __init__(self, *, feature_dim: int) -> None:
        self._heads = {axis: _LinearAxisHead(feature_dim=feature_dim) for axis in self.axes}

    def predict(self, features: tuple[float, ...]) -> dict[str, float]:
        return {axis: head.predict(features) for axis, head in self._heads.items()}

    def update(self, *, features: tuple[float, ...], targets: dict[str, float]) -> None:
        for axis, head in self._heads.items():
            head.update(features=features, target=targets[axis])

    def export_weights(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        return tuple((axis, self._heads[axis].export_weights()) for axis in self.axes)

    def restore_weights(
        self, weights: tuple[tuple[str, tuple[float, ...]], ...]
    ) -> None:
        payload = dict(weights)
        if set(payload) != set(self.axes):
            raise PredictiveHeadCheckpointError(
                f"axis set mismatch: expected {sorted(self.axes)}, got {sorted(payload)}"
            )
        for axis in self.axes:
            self._heads[axis].restore_weights(payload[axis])


class _WorldPredictiveHead(_AxisBankPredictiveHead):
    """CP-11 learned world head: task / regime / action outcome axes."""

    axes = ("task_progress", "regime_stability", "action_payoff")


class _SelfPredictiveHead(_AxisBankPredictiveHead):
    """CP-11 learned self head: relationship outcome axis."""

    axes = ("relationship_delta",)


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
    # apprenticeship_alignment (docs/specs/apprenticeship-alignment.md) is
    # added so that operator-guidance-vs-cognition mismatch / version-space
    # collapse can enter the PE chain as a discrete-event PE source,
    # analogous to the AAC alignment overlay. It is SHADOW by default, so PE
    # receives a placeholder and the overlay is a no-op until the owner is
    # promoted to ACTIVE (R15 reversibility).
    # CP-12: the four additional semantic owners publish typed
    # owner-prediction signals; PE (the single mismatch computer) consumes
    # their settled signals in-process, exactly like the commitment overlay
    # precedent. Their snapshots are read via ``upstream.get`` so disabled
    # owners degrade to "no settlements" rather than failing the DAG.
    dependencies = (
        "substrate",
        "evaluation",
        "dual_track",
        "regime",
        "commitment",
        "dialogue_external_outcome",
        "apprenticeship_alignment",
        "relationship_state",
        "goal_value",
        "boundary_consent",
        "execution_result",
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
        # Phase 2 W1.2 (DM-1): owner-internal per-axis distribution window.
        # Owner-internal constants only; downstream consumers MUST NOT
        # depend on min/max sizes (the public contract is the published
        # ``DistributionSummary`` payload). 8/64 chosen post Wave A
        # mechanism validation (debt #11 close-out 2026-05-08):
        # ``artifacts/eq_uplift/pe_window_long_form.json`` shows the
        # 8-sample IQR estimate converges to the 16-sample IQR within
        # ratio 1.0 across all four axes once both windows are warm,
        # so 8 produces statistically usable distribution shape with
        # 50% less cold-start cost. min_window=8 lets typical 8-15
        # turn benchmark scenarios surface non-None summaries (W3.1
        # probe previously got 0/5 with min_window=16). max_window=64
        # caps memory at 4 * 64 = 256 floats per session.
        self._distribution_window = _PEDistributionWindow(
            min_window=8,
            max_window=64,
        )
        # autograd-owner-integration Phase F: offline gradient-LSS calibration.
        # This is owner-internal state updated only via rare-heavy LSS import; it
        # never enters the published PredictionErrorSnapshot (schema unchanged).
        self._offline_lss_magnitude_ema = 0.0
        self._offline_lss_update_count = 0
        # CP-11 world/self predictive heads (SHADOW dual-run inside the single
        # PE owner; no new slot). The learned heads predict next-turn outcome
        # axes from the same compact evidence features; on the following turn
        # they are scored against the realized ActualOutcome next to the
        # hand-crafted head, and updated with bounded online SGD. Their
        # readout is report-only until the plan CP-11 gate is met.
        feature_dim = 7
        self._head_feature_dim = feature_dim
        self._world_head = _WorldPredictiveHead(feature_dim=feature_dim)
        self._self_head = _SelfPredictiveHead(feature_dim=feature_dim)
        self._pending_head_forecast: dict[str, Any] | None = None
        self._head_sample_count = 0
        self._head_abs_error_sums = {
            "world_learned": 0.0,
            "world_baseline": 0.0,
            "self_learned": 0.0,
            "self_baseline": 0.0,
        }
        # CP-11 kill-criteria window: (mean learned prediction, mean realized
        # target) per scored sample, bounded to the recent horizon.
        self._head_recent_pairs: deque[tuple[float, float]] = deque(
            maxlen=_HEAD_KILL_WINDOW
        )

    # ------------------------------------------------------------------
    # Offline gradient-LSS rare-heavy surface (Phase F). Runtime PE is
    # unchanged; these methods only build/consume the float-only LSS artifact.
    # ------------------------------------------------------------------

    def export_rare_heavy_lss(
        self,
        *,
        samples: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
        checkpoint_id: str,
    ):
        """Build a float-only LSS rare-heavy checkpoint from (predicted, actual)
        PE-axis samples using real torch autograd (grounding gate enforced)."""

        from volvence_zero.prediction.lss_rare_heavy import build_lss_rare_heavy_checkpoint

        return build_lss_rare_heavy_checkpoint(samples=samples, checkpoint_id=checkpoint_id)

    def import_rare_heavy_lss(self, checkpoint) -> None:
        """Apply an LSS checkpoint into owner-internal calibration only.

        Updates an EMA of the gradient-LSS magnitude. Does NOT modify the
        published snapshot in any way (R8). Fail-closed if ungrounded.
        """

        if not checkpoint.all_grounded:
            raise ValueError(
                "Refusing to import an ungrounded LSS rare-heavy checkpoint."
            )
        beta = 0.5
        self._offline_lss_magnitude_ema = (
            beta * self._offline_lss_magnitude_ema + (1.0 - beta) * checkpoint.mean_magnitude
        )
        self._offline_lss_update_count += 1

    @property
    def rare_heavy_lss_calibration(self) -> dict:
        return {
            "magnitude_ema": self._offline_lss_magnitude_ema,
            "update_count": self._offline_lss_update_count,
        }

    def export_rare_heavy_lss_state(self) -> tuple[float, int]:
        """Owner-internal LSS calibration state (for rollback)."""

        return (self._offline_lss_magnitude_ema, self._offline_lss_update_count)

    def restore_rare_heavy_lss_state(self, state: tuple[float, int]) -> None:
        """Restore owner-internal LSS calibration (rollback)."""

        self._offline_lss_magnitude_ema = float(state[0])
        self._offline_lss_update_count = int(state[1])

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
        apprenticeship_snapshot = upstream.get("apprenticeship_alignment")
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
        apprenticeship_value: ApprenticeshipAlignmentSnapshot | None = None
        if apprenticeship_snapshot is not None and isinstance(
            apprenticeship_snapshot.value, ApprenticeshipAlignmentSnapshot
        ):
            apprenticeship_value = apprenticeship_snapshot.value
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
            apprenticeship_snapshot=apprenticeship_value,
        )
        settlements = self._settle_owner_predictions(
            upstream=upstream,
            settled_turn_index=self._turn_index,
        )
        if settlements:
            snapshot = replace(snapshot, owner_prediction_settlements=settlements)
        self._previous_prediction = snapshot.next_prediction
        self._previous_substrate_snapshot = substrate_value
        return self.publish(snapshot)

    _OWNER_PREDICTION_SLOTS = (
        "commitment",
        "relationship_state",
        "goal_value",
        "boundary_consent",
        "execution_result",
    )

    def _settle_owner_predictions(
        self,
        *,
        upstream: Mapping[str, Snapshot[Any]],
        settled_turn_index: int,
    ) -> tuple[OwnerPredictionSettlement, ...]:
        """Compute mismatch for owner-published settled prediction signals.

        Only this owner computes mismatch (CP-12). Source owners publish
        ``owner_prediction_signals`` on their own snapshots; a signal with a
        ``settled_vector`` is scored here as the mean absolute component
        difference. Unsettled signals are ignored (they settle on a later
        turn). Slots without the field fail loudly — schema drift must not
        be papered over.
        """

        publisher_types = (
            CommitmentSnapshot,
            RelationshipStateSnapshot,
            GoalValueSnapshot,
            BoundaryConsentSnapshot,
            ExecutionResultSnapshot,
        )
        settlements: list[OwnerPredictionSettlement] = []
        for slot in self._OWNER_PREDICTION_SLOTS:
            upstream_snapshot = upstream.get(slot)
            if upstream_snapshot is None:
                continue
            value = upstream_snapshot.value
            if not isinstance(value, publisher_types):
                # Disabled / bootstrap slots can publish placeholder values;
                # a publisher-typed snapshot ALWAYS carries the field.
                continue
            signals: tuple[OwnerPredictionSignal, ...] = value.owner_prediction_signals
            for signal in signals:
                if not signal.settled:
                    continue
                assert signal.settled_vector is not None
                mismatch = sum(
                    abs(predicted - settled)
                    for predicted, settled in zip(
                        signal.predicted_vector, signal.settled_vector, strict=True
                    )
                ) / len(signal.predicted_vector)
                settlements.append(
                    OwnerPredictionSettlement(
                        prediction_id=signal.prediction_id,
                        source_owner=signal.source_owner,
                        source_slot=signal.source_slot,
                        track=signal.track,
                        kind=signal.kind,
                        mismatch_magnitude=max(0.0, min(1.0, mismatch)),
                        confidence=signal.confidence,
                        settled_turn_index=settled_turn_index,
                        description=(
                            f"{signal.source_slot} {signal.kind.value} settlement: "
                            f"mismatch={mismatch:.3f} over "
                            f"{len(signal.predicted_vector)} components."
                        ),
                    )
                )
        return tuple(settlements)

    async def process_standalone(self, **kwargs: Any) -> Snapshot[PredictionErrorSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        substrate_snapshot = kwargs.get("substrate_snapshot")
        evaluation_snapshot = kwargs.get("evaluation_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        regime_snapshot = kwargs.get("regime_snapshot")
        commitment_snapshot = kwargs.get("commitment_snapshot")
        external_outcome_snapshot = kwargs.get("external_outcome_snapshot")
        apprenticeship_snapshot = kwargs.get("apprenticeship_snapshot")
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
        apprenticeship_value = (
            apprenticeship_snapshot
            if isinstance(apprenticeship_snapshot, ApprenticeshipAlignmentSnapshot)
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
                apprenticeship_snapshot=apprenticeship_value,
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

    def _compute_apprenticeship_contribution(
        self, apprenticeship_snapshot: ApprenticeshipAlignmentSnapshot | None
    ) -> tuple[float, float, str]:
        """Return (signed_severity, surprise, audit) for this turn's guidance.

        * ``surprise`` (eluder informativeness) always lifts PE magnitude:
          informative guidance is a surprise relative to prior cognition.
        * ``signed_severity`` is POSITIVE when guidance clashes with the
          AI's cognition (mismatches) or the version space collapses
          (contradictions); the caller pushes regime_error / signed_reward
          NEGATIVE by it. Pure novelty without a confirmed clash carries
          surprise but near-zero signed severity (epistemic, not punitive).
        * Returns ``(0.0, 0.0, "")`` when the owner is absent / idle so the
          overlay is a strict no-op until the owner is promoted to ACTIVE.
        """
        if apprenticeship_snapshot is None:
            return 0.0, 0.0, ""
        snapshot = apprenticeship_snapshot
        if snapshot.version_space_status == "idle":
            return 0.0, 0.0, ""
        surprise = _clamp_unit(snapshot.guidance_surprise)
        contradiction_severity = max(
            (finding.severity for finding in snapshot.contradiction_findings),
            default=0.0,
        )
        mismatch_severity = (
            sum(ref.severity for ref in snapshot.mismatch_refs)
            / len(snapshot.mismatch_refs)
            if snapshot.mismatch_refs
            else 0.0
        )
        signed_severity = _clamp_signed(
            contradiction_severity * 0.8 + mismatch_severity * 0.2
        )
        audit = (
            f"apprenticeship[{snapshot.version_space_status}] "
            f"surprise={surprise:.2f} severity={signed_severity:.2f} "
            f"reliability={snapshot.reliability} "
            f"mismatches={len(snapshot.mismatch_refs)} "
            f"contradictions={len(snapshot.contradiction_findings)}"
        )
        return signed_severity, surprise, audit

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
        apprenticeship_snapshot: ApprenticeshipAlignmentSnapshot | None = None,
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
        # CP-10: the PE owner (and only the PE owner) issues the pre-action
        # prediction id on the prediction it publishes. Downstream actors
        # (affordance tool loop, expression) forward this id as lineage; they
        # must not fabricate their own.
        next_prediction = replace(
            next_prediction,
            prediction_id=f"pe:{self.slot_name}:turn-{turn_index}:next",
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
        # Overlay the apprenticeship-alignment contribution (Phase 2,
        # docs/specs/apprenticeship-alignment.md). Operator guidance that
        # diverges from / contradicts current cognition is informative,
        # discrete-event PE: it lifts magnitude (the AI was surprised) and,
        # for confirmed contradictions / mismatches, pushes the regime axis
        # and signed_reward negative (the AI's identity/cognition is out of
        # line with what it is being taught). No-op when the owner is SHADOW
        # (snapshot absent => placeholder => None) or idle.
        appr_severity, appr_surprise, appr_audit = (
            self._compute_apprenticeship_contribution(apprenticeship_snapshot)
        )
        if appr_surprise > 0.0 or appr_severity != 0.0:
            regime_error = _clamp_signed(error.regime_error - appr_severity)
            signed_reward = _clamp_signed(error.signed_reward - appr_severity)
            magnitude = round(
                min(error.magnitude + max(appr_surprise, abs(appr_severity)), 4.0),
                4,
            )
            error = PredictionError(
                task_error=error.task_error,
                relationship_error=error.relationship_error,
                regime_error=round(regime_error, 4),
                action_error=error.action_error,
                magnitude=magnitude,
                signed_reward=round(signed_reward, 4),
                description=f"{error.description} {appr_audit}.",
                distribution_summary=error.distribution_summary,
            )
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
        # Phase 2 W1.2 (DM-1): push the finalised PE into the rolling
        # distribution window AFTER alignment overlay so the window
        # tracks the same per-axis errors downstream consumers see on
        # ``error``. ``DistributionSummary`` returns ``None`` until the
        # window holds at least ``min_window`` samples (cold-start
        # safety). Bootstrap turns SKIP the update so the very first
        # turn's all-zero error does not pollute the window.
        # CP-11 SHADOW dual-run: score last turn's learned/baseline forecasts
        # against this turn's realized outcome, update the learned heads, and
        # queue this turn's forecast for the next turn.
        predictive_head_readout = self._advance_predictive_heads(
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            substrate_snapshot=substrate_snapshot,
            previous_substrate_snapshot=previous_substrate_snapshot,
            evaluation_snapshot=evaluation_snapshot,
            dual_track_snapshot=dual_track_snapshot,
            regime_snapshot=regime_snapshot,
        )
        if not bootstrap:
            self._distribution_window.update(error)
        distribution_summary = self._distribution_window.summarise()
        if distribution_summary is not None:
            error = PredictionError(
                task_error=error.task_error,
                relationship_error=error.relationship_error,
                regime_error=error.regime_error,
                action_error=error.action_error,
                magnitude=error.magnitude,
                signed_reward=error.signed_reward,
                description=error.description,
                distribution_summary=distribution_summary,
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
            predictive_head_readout=predictive_head_readout,
            pe_decomposition=decomposition,
        )

    def _advance_predictive_heads(
        self,
        *,
        evaluated_prediction: PredictedOutcome | None,
        actual_outcome: ActualOutcome,
        substrate_snapshot: SubstrateSnapshot | None,
        previous_substrate_snapshot: SubstrateSnapshot | None,
        evaluation_snapshot: EvaluationSnapshot,
        dual_track_snapshot: DualTrackSnapshot,
        regime_snapshot: "RegimeSnapshot | None",
    ) -> PredictiveHeadReadout | None:
        """CP-11 SHADOW step. Report-only: never touches the live chain."""

        pending = self._pending_head_forecast
        if pending is not None and evaluated_prediction is not None:
            world_targets = {
                "task_progress": _clamp_unit(actual_outcome.task_progress),
                "regime_stability": _clamp_unit(actual_outcome.regime_stability),
                "action_payoff": _clamp_unit(actual_outcome.action_payoff),
            }
            self_targets = {
                "relationship_delta": _clamp_unit(actual_outcome.relationship_delta),
            }
            world_learned: dict[str, float] = pending["world"]
            self_learned: dict[str, float] = pending["self"]
            self._head_abs_error_sums["world_learned"] += sum(
                abs(world_learned[axis] - world_targets[axis]) for axis in world_targets
            ) / len(world_targets)
            self._head_abs_error_sums["world_baseline"] += (
                abs(evaluated_prediction.predicted_task_progress - world_targets["task_progress"])
                + abs(
                    evaluated_prediction.predicted_regime_stability
                    - world_targets["regime_stability"]
                )
                + abs(
                    evaluated_prediction.predicted_action_payoff
                    - world_targets["action_payoff"]
                )
            ) / len(world_targets)
            self._head_abs_error_sums["self_learned"] += abs(
                self_learned["relationship_delta"] - self_targets["relationship_delta"]
            )
            self._head_abs_error_sums["self_baseline"] += abs(
                _clamp_unit(evaluated_prediction.predicted_relationship_delta)
                - self_targets["relationship_delta"]
            )
            self._head_sample_count += 1
            # Kill-criteria evidence: mean learned prediction vs mean realized
            # target across all four axes for the self-reward autocorr check.
            learned_mean = (
                sum(world_learned[axis] for axis in world_targets)
                + self_learned["relationship_delta"]
            ) / 4.0
            target_mean = (
                sum(world_targets.values()) + self_targets["relationship_delta"]
            ) / 4.0
            self._head_recent_pairs.append((learned_mean, target_mean))
            features: tuple[float, ...] = pending["features"]
            self._world_head.update(features=features, targets=world_targets)
            self._self_head.update(features=features, targets=self_targets)

        next_features = _featurize_outcome_evidence(
            _build_outcome_evidence(
                substrate_snapshot=substrate_snapshot,
                previous_substrate_snapshot=previous_substrate_snapshot,
                evaluation_snapshot=evaluation_snapshot,
                dual_track_snapshot=dual_track_snapshot,
                regime_snapshot=regime_snapshot,
            )
        )
        self._pending_head_forecast = {
            "features": next_features,
            "world": self._world_head.predict(next_features),
            "self": self._self_head.predict(next_features),
        }

        if self._head_sample_count == 0:
            return None
        count = self._head_sample_count
        world_learned_mae = self._head_abs_error_sums["world_learned"] / count
        world_baseline_mae = self._head_abs_error_sums["world_baseline"] / count
        self_learned_mae = self._head_abs_error_sums["self_learned"] / count
        self_baseline_mae = self._head_abs_error_sums["self_baseline"] / count
        return PredictiveHeadReadout(
            world_learned_mae=round(world_learned_mae, 4),
            world_baseline_mae=round(world_baseline_mae, 4),
            self_learned_mae=round(self_learned_mae, 4),
            self_baseline_mae=round(self_baseline_mae, 4),
            sample_count=count,
            world_improvement=round(world_baseline_mae - world_learned_mae, 4),
            self_improvement=round(self_baseline_mae - self_learned_mae, 4),
            description=(
                f"CP-11 SHADOW heads over {count} samples: world learned/baseline "
                f"MAE {world_learned_mae:.3f}/{world_baseline_mae:.3f}, self "
                f"{self_learned_mae:.3f}/{self_baseline_mae:.3f}. Report-only."
            ),
        )

    # ------------------------------------------------------------------
    # CP-11 gate completeness surface (W1.D): session-medium checkpoint
    # export/restore for the SHADOW heads plus the self-reward
    # autocorrelation kill-criteria readout. All report-only; nothing here
    # touches the live prediction chain.
    # ------------------------------------------------------------------

    def export_predictive_head_checkpoint(
        self, *, checkpoint_id: str
    ) -> PredictiveHeadCheckpoint:
        """Export the SHADOW heads' weights + readout counters (float-only)."""

        return PredictiveHeadCheckpoint(
            checkpoint_id=checkpoint_id,
            feature_dim=self._head_feature_dim,
            world_weights=self._world_head.export_weights(),
            self_weights=self._self_head.export_weights(),
            sample_count=self._head_sample_count,
            abs_error_sums=tuple(sorted(self._head_abs_error_sums.items())),
        )

    def restore_predictive_head_checkpoint(
        self, checkpoint: PredictiveHeadCheckpoint
    ) -> None:
        """Restore SHADOW head weights + readout counters from a checkpoint.

        Fails loudly (PredictiveHeadCheckpointError) on schema / feature-dim /
        axis-set mismatch. The kill-criteria window intentionally restarts
        empty: its evidence must come from the live session, not history.
        """

        if checkpoint.schema_version != PREDICTIVE_HEAD_CHECKPOINT_SCHEMA_VERSION:
            raise PredictiveHeadCheckpointError(
                f"schema_version mismatch: expected "
                f"{PREDICTIVE_HEAD_CHECKPOINT_SCHEMA_VERSION!r}, got "
                f"{checkpoint.schema_version!r}"
            )
        if checkpoint.feature_dim != self._head_feature_dim:
            raise PredictiveHeadCheckpointError(
                f"feature_dim mismatch: expected {self._head_feature_dim}, "
                f"got {checkpoint.feature_dim}"
            )
        restored_sums = dict(checkpoint.abs_error_sums)
        if set(restored_sums) != set(self._head_abs_error_sums):
            raise PredictiveHeadCheckpointError(
                f"abs_error_sums key mismatch: expected "
                f"{sorted(self._head_abs_error_sums)}, got {sorted(restored_sums)}"
            )
        self._world_head.restore_weights(checkpoint.world_weights)
        self._self_head.restore_weights(checkpoint.self_weights)
        self._head_sample_count = checkpoint.sample_count
        self._head_abs_error_sums = restored_sums
        self._head_recent_pairs.clear()
        self._pending_head_forecast = None

    def predictive_head_kill_criteria(self) -> PredictiveHeadKillCriteria:
        """Self-reward autocorrelation check over the recent scored window."""

        pairs = list(self._head_recent_pairs)
        predictions = [p for p, _ in pairs]
        targets = [t for _, t in pairs]
        window_filled = len(pairs) >= _HEAD_KILL_WINDOW
        if len(pairs) >= 3:
            # A frozen (constant) prediction series counts as perfectly
            # self-correlated and target-uncorrelated: that IS the failure
            # mode this check exists to catch.
            self_autocorr = _series_correlation(
                predictions[1:], predictions[:-1], degenerate=1.0
            )
            target_corr = _series_correlation(predictions, targets, degenerate=0.0)
        else:
            self_autocorr = 0.0
            target_corr = 0.0
        kill = window_filled and self_autocorr >= 0.98 and target_corr <= 0.05
        return PredictiveHeadKillCriteria(
            window_size=_HEAD_KILL_WINDOW,
            samples_in_window=len(pairs),
            window_filled=window_filled,
            prediction_self_autocorrelation=round(self_autocorr, 4),
            prediction_target_correlation=round(target_corr, 4),
            kill_triggered=kill,
            description=(
                f"CP-11 kill check over {len(pairs)}/{_HEAD_KILL_WINDOW} samples: "
                f"self-autocorr={self_autocorr:.3f}, target-corr={target_corr:.3f}, "
                f"kill_triggered={kill}. Report-only."
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
    # ------------------------------------------------------------------
    # W3-A LTV / conversion-funnel outcomes. Magnitudes reflect that
    # purchase / repurchase signals are stronger evidence of action
    # payoff than any single dialogue-level signal, and that churn is
    # a relationship-axis catastrophe that must visibly drive PE.
    # ------------------------------------------------------------------
    DialogueExternalOutcomeKind.LEAD_QUALIFIED: (+0.20, +0.30, 0.0, +0.20),
    DialogueExternalOutcomeKind.RECOMMENDATION_MADE: (+0.30, 0.0, 0.0, +0.30),
    DialogueExternalOutcomeKind.PURCHASE_CONFIRMED: (+0.50, +0.40, 0.0, +0.50),
    DialogueExternalOutcomeKind.REPURCHASE: (+0.60, +0.50, 0.0, +0.60),
    DialogueExternalOutcomeKind.CHURNED: (-0.30, -0.60, -0.10, -0.40),
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


_PE_DECOUPLE_TRUTHY = frozenset({"1", "true", "yes", "on", "active"})


def pe_evaluation_decoupled_active() -> bool:
    """R-PE invariant #1 gate: evaluation is a readout, not a learning source.

    When ACTIVE (``VZ_PE_EVALUATION_DECOUPLED`` truthy) the PE actual
    outcome no longer derives its ``family_signals`` from the
    :class:`EvaluationSnapshot`; the outcome is driven only by substrate /
    dual-track / regime / external-outcome evidence. Default is SHADOW
    (env unset/false), which preserves the legacy behaviour exactly so
    this packet is a no-op until an operator opts in (R15 reversibility).

    Decoupling mechanism: an empty ``family_signals`` mapping makes every
    ``evidence.family_signals.get(family, 0.5)`` lookup in
    :meth:`_PredictionErrorHead.build_actual_outcome` fall back to the
    neutral 0.5, so evaluation can no longer bias the outcome while the
    substrate-derived axes keep contributing unchanged.
    """
    raw = (os.environ.get("VZ_PE_EVALUATION_DECOUPLED", "") or "").strip().lower()
    return raw in _PE_DECOUPLE_TRUTHY


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
    # R-PE #1: when the decouple gate is ACTIVE, evaluation scores do NOT
    # feed PE actual outcome (neutral family_signals). SHADOW (default)
    # keeps the legacy evaluation-derived signals.
    family_signals = (
        {} if pe_evaluation_decoupled_active() else _family_signals(evaluation_snapshot)
    )
    return _OutcomeEvidence(
        family_signals=family_signals,
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


class _PEDistributionWindow:
    """Owner-internal per-axis rolling window for PE distribution shape.

    Phase 2 W1.2 (DM-1). Maintains 4 axis bounded deques over the most
    recent ``max_window`` PE samples and computes IQR / entropy /
    asymmetry on demand. Used by ``PredictionErrorModule._advance`` to
    fill the optional ``DistributionSummary`` slot on
    :class:`PredictionError`.

    Cold-start safety: returns ``None`` until at least ``min_window``
    samples have been observed on EVERY axis, so consumers never see
    a partial / single-sample summary.

    Owner-internal constants only. Other modules MUST NOT depend on
    ``min_window`` / ``max_window`` / bin count; the public contract
    is the published :class:`DistributionSummary`.
    """

    _AXES: tuple[str, ...] = ("task", "relationship", "regime", "action")
    # Number of equal-width bins over [0, 1] (we bin |axis_error|) used
    # to compute Shannon entropy. 5 bins gives a stable estimate from
    # the smallest legal window (16 samples) without saturating.
    _ENTROPY_BIN_COUNT: int = 5

    def __init__(self, *, min_window: int = 16, max_window: int = 64) -> None:
        if min_window < 4:
            raise ValueError("min_window must be >= 4 for IQR to be defined")
        if max_window < min_window:
            raise ValueError("max_window must be >= min_window")
        self._min_window = int(min_window)
        self._max_window = int(max_window)
        self._samples: dict[str, list[float]] = {axis: [] for axis in self._AXES}

    def update(self, error: PredictionError) -> None:
        """Append one PE sample (4 axes) to the rolling window."""
        per_axis = (
            ("task", float(error.task_error)),
            ("relationship", float(error.relationship_error)),
            ("regime", float(error.regime_error)),
            ("action", float(error.action_error)),
        )
        for axis, value in per_axis:
            buffer = self._samples[axis]
            buffer.append(value)
            if len(buffer) > self._max_window:
                # Drop the oldest sample so the window stays bounded.
                del buffer[0]

    def summarise(self) -> DistributionSummary | None:
        """Compute the current per-axis summary or ``None`` if cold."""
        sizes = [len(self._samples[axis]) for axis in self._AXES]
        window_size = min(sizes) if sizes else 0
        if window_size < self._min_window:
            return None
        iqr_pairs: list[tuple[str, float]] = []
        entropy_pairs: list[tuple[str, float]] = []
        asymmetry_pairs: list[tuple[str, float]] = []
        for axis in self._AXES:
            values = tuple(self._samples[axis])
            iqr_pairs.append((axis, _iqr_of_abs(values)))
            entropy_pairs.append(
                (axis, _binned_entropy(values, bins=self._ENTROPY_BIN_COUNT))
            )
            asymmetry_pairs.append((axis, _asymmetry(values)))
        return DistributionSummary(
            window_size=window_size,
            iqr=tuple(iqr_pairs),
            entropy=tuple(entropy_pairs),
            asymmetry=tuple(asymmetry_pairs),
            description=(
                f"PE distribution window={window_size} bins={self._ENTROPY_BIN_COUNT}"
            ),
        )


def _iqr_of_abs(values: tuple[float, ...]) -> float:
    """Return Q3 - Q1 of |values|, clamped to ``[0, 1]``.

    Computed on absolute values so the IQR captures distribution width
    independent of sign (signed drift is captured separately by
    ``_asymmetry``).
    """
    if not values:
        return 0.0
    sorted_abs = sorted(abs(v) for v in values)
    q1 = _percentile(sorted_abs, 0.25)
    q3 = _percentile(sorted_abs, 0.75)
    iqr = max(0.0, q3 - q1)
    return round(min(iqr, 1.0), 4)


def _binned_entropy(values: tuple[float, ...], *, bins: int) -> float:
    """Shannon entropy of ``|values|`` discretised into ``bins`` bins.

    Bins are equal-width over ``[0, 1]``. Returned in nats and clamped
    to ``[0, log(bins)]`` (the maximum possible entropy for a uniform
    distribution over ``bins`` bins).
    """
    if not values or bins <= 1:
        return 0.0
    counts = [0] * bins
    for value in values:
        magnitude = min(1.0, max(0.0, abs(value)))
        index = min(bins - 1, int(magnitude * bins))
        counts[index] += 1
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return round(min(entropy, math.log(bins)), 4)


def _asymmetry(values: tuple[float, ...]) -> float:
    """Signed skew proxy ``(mean - median) / (iqr + eps)``, clamped to ``[-1, 1]``.

    Computed on signed values (NOT absolute), so positive output means
    the distribution leans toward larger positive errors and negative
    output means it leans toward larger negative errors.
    """
    if not values:
        return 0.0
    sorted_signed = sorted(values)
    mean = sum(sorted_signed) / len(sorted_signed)
    median = _percentile(sorted_signed, 0.5)
    spread = _iqr_of_signed(sorted_signed)
    asymmetry = (mean - median) / (spread + 1e-6)
    return round(max(-1.0, min(1.0, asymmetry)), 4)


def _iqr_of_signed(sorted_signed: list[float]) -> float:
    """Q3 - Q1 of already-sorted SIGNED values.

    Used internally by ``_asymmetry`` to keep the asymmetry denominator
    in the same units as the numerator.
    """
    q1 = _percentile(sorted_signed, 0.25)
    q3 = _percentile(sorted_signed, 0.75)
    return max(0.0, q3 - q1)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile on a presorted list.

    Cheap stand-in for numpy.percentile. ``q`` in ``[0, 1]``.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = q * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


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
