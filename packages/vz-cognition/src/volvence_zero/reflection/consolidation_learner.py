"""Report-only learned candidate for the reflection consolidation score (G4).

The hand-crafted ``_consolidation_score`` formula stays the live writer:
promote / decay decisions and writeback gating never read this learner.
The learner dual-runs a bounded linear residual head over the same
features and publishes a report-only ``learned_promotion_score`` on the
``ConsolidationScore`` readout, settled one turn later against the
realized prediction-error magnitude (a consolidation choice followed by
low PE scores closer to its own promotion estimate; high realized PE
pulls it down).

Session-held (same lifetime pattern as ``DualTrackGateLearner``): the
per-turn ``ReflectionEngine`` rebuild receives the ONE learner owned by
the session so weights accumulate across turns.
"""

from __future__ import annotations

from dataclasses import dataclass

_FEATURE_DIM = 7
_LR = 0.06
_WEIGHT_LIMIT = 1.5
_MIN_SETTLES = 50
_MAE_MARGIN = 0.02
_KILL_DEGRADATION = 0.10


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ConsolidationScoreLearnerState:
    weights: tuple[float, ...]
    settled_count: int
    abs_error_sum: float
    baseline_abs_error_sum: float


@dataclass(frozen=True)
class ConsolidationPromotionReadout:
    ready: bool
    kill_recommended: bool
    settled_count: int
    learned_mae: float
    baseline_mae: float
    mae_improvement: float
    blocking_reasons: tuple[str, ...]
    description: str


class ConsolidationScoreLearner:
    """Bounded linear residual head over consolidation features.

    Feature vector:
    ``(memory_pressure, cross_tension, alert_pressure, positive_credit,
    negative_credit, pe_penalty, bias)``. Prediction is
    ``clamp01(baseline_promotion + w·features)`` so zero weights are
    byte-identical to the live formula (rollback point).
    """

    def __init__(self) -> None:
        self._weights = [0.0] * _FEATURE_DIM
        # Features / predictions issued this turn, settled by the NEXT
        # realized PE observation (one-turn window, gate-learner style).
        self._latest_features: tuple[float, ...] | None = None
        self._latest_baseline = 0.0
        self._settleable_features: tuple[float, ...] | None = None
        self._settleable_baseline = 0.0
        self._settled_count = 0
        self._abs_error_sum = 0.0
        self._baseline_abs_error_sum = 0.0

    def shadow_score(
        self,
        *,
        features: tuple[float, ...],
        baseline_promotion: float,
    ) -> float:
        if len(features) != _FEATURE_DIM:
            raise ValueError(
                f"consolidation features have dim {len(features)}, "
                f"expected {_FEATURE_DIM}"
            )
        self._settleable_features = self._latest_features
        self._settleable_baseline = self._latest_baseline
        self._latest_features = features
        self._latest_baseline = _clamp01(baseline_promotion)
        return round(self._predict(features, baseline=baseline_promotion), 4)

    def observe_realized_outcome(self, *, pe_magnitude: float) -> bool:
        """Settle the previous turn's shadow score against realized PE.

        Target semantics: a consolidation issued right before a low-PE
        turn should have promoted confidently; a high-PE follow-up turn
        means the consolidation underestimated open tension. Returns
        True when an SGD step was applied.
        """

        features = self._settleable_features
        baseline = self._settleable_baseline
        self._settleable_features = None
        if features is None:
            return False
        target = _clamp01(1.0 - min(max(pe_magnitude, 0.0) / 4.0, 1.0))
        prediction = self._predict(features, baseline=baseline)
        error = target - prediction
        self._weights = [
            max(-_WEIGHT_LIMIT, min(_WEIGHT_LIMIT, weight + _LR * error * feature))
            for weight, feature in zip(self._weights, features, strict=True)
        ]
        self._settled_count += 1
        self._abs_error_sum += abs(error)
        self._baseline_abs_error_sum += abs(target - baseline)
        return True

    def promotion_readout(self) -> ConsolidationPromotionReadout:
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
        return ConsolidationPromotionReadout(
            ready=not blocking,
            kill_recommended=kill,
            settled_count=self._settled_count,
            learned_mae=round(learned_mae, 6),
            baseline_mae=round(baseline_mae, 6),
            mae_improvement=round(improvement, 6),
            blocking_reasons=tuple(blocking),
            description=(
                "ConsolidationScoreLearner SHADOW candidate; live writeback "
                f"unchanged; settled={self._settled_count} "
                f"learned_mae={learned_mae:.4f} baseline_mae={baseline_mae:.4f}."
            ),
        )

    def export_state(self) -> ConsolidationScoreLearnerState:
        return ConsolidationScoreLearnerState(
            weights=tuple(self._weights),
            settled_count=self._settled_count,
            abs_error_sum=self._abs_error_sum,
            baseline_abs_error_sum=self._baseline_abs_error_sum,
        )

    def restore_state(self, state: ConsolidationScoreLearnerState) -> None:
        if len(state.weights) != _FEATURE_DIM:
            raise ValueError(
                f"consolidation learner state has {len(state.weights)} "
                f"weights, expected {_FEATURE_DIM}"
            )
        self._weights = [
            max(-_WEIGHT_LIMIT, min(_WEIGHT_LIMIT, float(value)))
            for value in state.weights
        ]
        self._settled_count = max(0, state.settled_count)
        self._abs_error_sum = max(0.0, state.abs_error_sum)
        self._baseline_abs_error_sum = max(0.0, state.baseline_abs_error_sum)
        self._latest_features = None
        self._settleable_features = None

    def reset(self) -> None:
        self.restore_state(
            ConsolidationScoreLearnerState(
                weights=tuple([0.0] * _FEATURE_DIM),
                settled_count=0,
                abs_error_sum=0.0,
                baseline_abs_error_sum=0.0,
            )
        )

    def _predict(self, features: tuple[float, ...], *, baseline: float) -> float:
        residual = sum(
            weight * feature
            for weight, feature in zip(self._weights, features, strict=True)
        )
        return _clamp01(_clamp01(baseline) + residual)


__all__ = [
    "ConsolidationPromotionReadout",
    "ConsolidationScoreLearner",
    "ConsolidationScoreLearnerState",
]
