"""Report-only learned scorer candidate for affordance selection (G3).

``AffordanceModule``'s live selection stays on the z_t projection +
threshold path (scorer.v1 semantics). This learner dual-runs a bounded
linear residual head over the same candidates and publishes a
report-only ``shadow_learned_score`` per candidate, settled by realized
tool invocation outcomes (SUCCEEDED vs BACKEND_FAILED) flowing back
through the invoker's outcome listener seam.

Same convergence pattern as ``RegimeScoreLearner`` /
``DualTrackGateLearner``: the hand-crafted score is the initialisation
and the rollback point; weights are envelope-bounded; ``reset()``
restores byte-identical baseline behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.affordance import AffordanceKind, AffordanceLatencyClass

_MIN_SETTLES = 50
_MAE_MARGIN = 0.02
_KILL_DEGRADATION = 0.10
_LR = 0.08
_WEIGHT_LIMIT = 1.5
_FEATURE_DIM = 7

_LATENCY_SCALAR: dict[AffordanceLatencyClass, float] = {
    AffordanceLatencyClass.INSTANT: 0.0,
    AffordanceLatencyClass.FAST: 0.33,
    AffordanceLatencyClass.SLOW: 0.66,
    AffordanceLatencyClass.VERY_SLOW: 1.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class AffordanceScoreLearnerState:
    """Float-only checkpointable learner state."""

    weights: tuple[float, ...]
    settled_count: int
    abs_error_sum: float
    baseline_abs_error_sum: float
    last_settled_descriptor: str


@dataclass(frozen=True)
class AffordanceScorePromotionReadout:
    """Report-only promotion / kill evidence for the learned candidate."""

    ready: bool
    kill_recommended: bool
    settled_count: int
    learned_mae: float
    baseline_mae: float
    mae_improvement: float
    blocking_reasons: tuple[str, ...]
    description: str


class AffordanceScoreLearner:
    """Bounded linear residual head over affordance candidates.

    Feature vector per candidate:
    ``(base_score, kind_tool, kind_action, kind_organ, kind_shell,
    latency_scalar, bias)``. The prediction is
    ``clamp01(base_score + w·features)`` so an all-zero weight vector is
    byte-identical to the live scorer's output (rollback point).
    """

    def __init__(self) -> None:
        self._weights = [0.0] * _FEATURE_DIM
        self._latest_features: dict[str, tuple[float, ...]] = {}
        self._latest_base: dict[str, float] = {}
        self._settled_count = 0
        self._abs_error_sum = 0.0
        self._baseline_abs_error_sum = 0.0
        self._last_settled_descriptor = ""

    def shadow_score(
        self,
        *,
        descriptor_name: str,
        base_score: float,
        kind: AffordanceKind,
        latency_class: AffordanceLatencyClass,
    ) -> float:
        features = self._featurize(
            base_score=base_score, kind=kind, latency_class=latency_class
        )
        self._latest_features[descriptor_name] = features
        self._latest_base[descriptor_name] = _clamp01(base_score)
        return round(self._predict(features, base_score=base_score), 4)

    def begin_turn(self) -> None:
        """Reset the per-turn feature window before rescoring.

        Only descriptors scored THIS turn are settleable by this turn's
        invocation outcomes; stale windows from earlier turns must not
        absorb credit for outcomes they did not propose.
        """

        self._latest_features = {}
        self._latest_base = {}

    def observe_invocation_outcome(
        self, *, descriptor_name: str, succeeded: bool
    ) -> bool:
        """Settle one realized tool outcome against the latest shadow score.

        Returns True when an SGD step was applied (False when the
        descriptor was not scored in the current window).
        """

        features = self._latest_features.get(descriptor_name)
        if features is None:
            return False
        base = self._latest_base.get(descriptor_name, 0.5)
        target = 1.0 if succeeded else 0.0
        prediction = self._predict(features, base_score=base)
        error = target - prediction
        self._weights = [
            max(-_WEIGHT_LIMIT, min(_WEIGHT_LIMIT, weight + _LR * error * feature))
            for weight, feature in zip(self._weights, features, strict=True)
        ]
        self._settled_count += 1
        self._abs_error_sum += abs(error)
        self._baseline_abs_error_sum += abs(target - base)
        self._last_settled_descriptor = descriptor_name
        return True

    def promotion_readout(self) -> AffordanceScorePromotionReadout:
        learned_mae = (
            self._abs_error_sum / self._settled_count if self._settled_count else 0.0
        )
        baseline_mae = (
            self._baseline_abs_error_sum / self._settled_count
            if self._settled_count
            else 0.0
        )
        improvement = baseline_mae - learned_mae
        blocking: list[str] = []
        if self._settled_count < _MIN_SETTLES:
            blocking.append(f"settled_count<{_MIN_SETTLES}")
        if improvement < _MAE_MARGIN:
            blocking.append(f"mae_improvement<{_MAE_MARGIN:.2f}")
        kill = self._settled_count >= _MIN_SETTLES and improvement <= -_KILL_DEGRADATION
        return AffordanceScorePromotionReadout(
            ready=not blocking,
            kill_recommended=kill,
            settled_count=self._settled_count,
            learned_mae=round(learned_mae, 6),
            baseline_mae=round(baseline_mae, 6),
            mae_improvement=round(improvement, 6),
            blocking_reasons=tuple(blocking),
            description=(
                "AffordanceScoreLearner SHADOW candidate; live selection "
                f"unchanged; settled={self._settled_count} "
                f"learned_mae={learned_mae:.4f} baseline_mae={baseline_mae:.4f}."
            ),
        )

    def export_state(self) -> AffordanceScoreLearnerState:
        return AffordanceScoreLearnerState(
            weights=tuple(self._weights),
            settled_count=self._settled_count,
            abs_error_sum=self._abs_error_sum,
            baseline_abs_error_sum=self._baseline_abs_error_sum,
            last_settled_descriptor=self._last_settled_descriptor,
        )

    def restore_state(self, state: AffordanceScoreLearnerState) -> None:
        if len(state.weights) != _FEATURE_DIM:
            raise ValueError(
                f"affordance score learner state has {len(state.weights)} "
                f"weights, expected {_FEATURE_DIM}"
            )
        self._weights = [
            max(-_WEIGHT_LIMIT, min(_WEIGHT_LIMIT, float(value)))
            for value in state.weights
        ]
        self._settled_count = max(0, state.settled_count)
        self._abs_error_sum = max(0.0, state.abs_error_sum)
        self._baseline_abs_error_sum = max(0.0, state.baseline_abs_error_sum)
        self._last_settled_descriptor = state.last_settled_descriptor
        self._latest_features = {}
        self._latest_base = {}

    def reset(self) -> None:
        """Kill-path rollback: return to the zero-residual baseline."""

        self.restore_state(
            AffordanceScoreLearnerState(
                weights=tuple([0.0] * _FEATURE_DIM),
                settled_count=0,
                abs_error_sum=0.0,
                baseline_abs_error_sum=0.0,
                last_settled_descriptor="",
            )
        )

    def _featurize(
        self,
        *,
        base_score: float,
        kind: AffordanceKind,
        latency_class: AffordanceLatencyClass,
    ) -> tuple[float, ...]:
        return (
            _clamp01(base_score),
            1.0 if kind is AffordanceKind.TOOL else 0.0,
            1.0 if kind is AffordanceKind.ACTION else 0.0,
            1.0 if kind is AffordanceKind.ORGAN else 0.0,
            1.0 if kind is AffordanceKind.SHELL else 0.0,
            _LATENCY_SCALAR[latency_class],
            1.0,
        )

    def _predict(self, features: tuple[float, ...], *, base_score: float) -> float:
        residual = sum(
            weight * feature
            for weight, feature in zip(self._weights, features, strict=True)
        )
        return _clamp01(_clamp01(base_score) + residual)


__all__ = [
    "AffordanceScoreLearner",
    "AffordanceScoreLearnerState",
    "AffordanceScorePromotionReadout",
]
