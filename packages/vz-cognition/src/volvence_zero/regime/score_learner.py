"""Report-only learned scorer for regime selection.

The learner is owned by ``RegimeModule`` and never selects the live
regime. It dual-runs against the existing hand-crafted scorer so traces
can settle whether a learned candidate should later replace the fixed
formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.regime.contracts import RegimeLearnedScoreShadow

_MIN_UPDATES = 50
_MAE_MARGIN = 0.02
_KILL_DEGRADATION = 0.10
_LR = 0.06
_WEIGHT_LIMIT = 1.5


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class RegimeScoreLearnerState:
    weights: tuple[tuple[str, tuple[float, ...]], ...]
    update_count: int
    abs_error_sum: float
    baseline_abs_error_sum: float
    settled_count: int
    last_target_regime_id: str


class RegimeScoreLearner:
    """Bounded linear head over baseline regime candidates.

    Feature vector per candidate:
    ``(baseline_score, historical_effectiveness, strategy_prior, bias)``.
    This keeps the first SHADOW slice grounded in the existing scorer while
    letting delayed outcome payoff move the candidate toward observed wins.
    """

    _FEATURE_DIM = 4

    def __init__(self) -> None:
        self._weights: dict[str, list[float]] = {}
        self._latest_features: dict[str, tuple[float, ...]] = {}
        self._latest_baseline: dict[str, float] = {}
        self._update_count = 0
        self._abs_error_sum = 0.0
        self._baseline_abs_error_sum = 0.0
        self._settled_count = 0
        self._last_target_regime_id = ""

    def score(
        self,
        *,
        baseline_scores: tuple[tuple[str, float], ...],
        historical_effectiveness: dict[str, float],
        strategy_priors: dict[str, float],
    ) -> RegimeLearnedScoreShadow:
        baseline_by_id = {regime_id: score for regime_id, score in baseline_scores}
        learned: list[tuple[str, float]] = []
        features_by_id: dict[str, tuple[float, ...]] = {}
        for regime_id, baseline in baseline_scores:
            features = (
                _clamp(baseline),
                _clamp(historical_effectiveness.get(regime_id, 0.5)),
                max(-0.5, min(0.5, strategy_priors.get(regime_id, 0.0))),
                1.0,
            )
            features_by_id[regime_id] = features
            learned.append((regime_id, round(self._predict(regime_id, features, baseline), 4)))
        learned.sort(key=lambda item: item[1], reverse=True)
        self._latest_features = features_by_id
        self._latest_baseline = baseline_by_id
        return self._readout(learned_scores=tuple(learned), baseline_scores=baseline_scores)

    def observe_delayed_payoff(self, *, regime_id: str, outcome_score: float) -> None:
        features = self._latest_features.get(regime_id)
        if features is None:
            return
        target = _clamp(outcome_score)
        baseline = _clamp(self._latest_baseline.get(regime_id, 0.5))
        prediction = self._predict(regime_id, features, baseline)
        error = target - prediction
        weights = self._weights_for(regime_id)
        self._weights[regime_id] = [
            max(-_WEIGHT_LIMIT, min(_WEIGHT_LIMIT, weight + _LR * error * feature))
            for weight, feature in zip(weights, features, strict=True)
        ]
        self._update_count += 1
        self._settled_count += 1
        self._abs_error_sum += abs(error)
        self._baseline_abs_error_sum += abs(target - baseline)
        self._last_target_regime_id = regime_id

    def export_state(self) -> RegimeScoreLearnerState:
        return RegimeScoreLearnerState(
            weights=tuple(
                (regime_id, tuple(weights))
                for regime_id, weights in sorted(self._weights.items())
            ),
            update_count=self._update_count,
            abs_error_sum=self._abs_error_sum,
            baseline_abs_error_sum=self._baseline_abs_error_sum,
            settled_count=self._settled_count,
            last_target_regime_id=self._last_target_regime_id,
        )

    def restore_state(self, state: RegimeScoreLearnerState) -> None:
        self._weights = {
            regime_id: self._align(weights)
            for regime_id, weights in state.weights
        }
        self._update_count = max(0, state.update_count)
        self._abs_error_sum = max(0.0, state.abs_error_sum)
        self._baseline_abs_error_sum = max(0.0, state.baseline_abs_error_sum)
        self._settled_count = max(0, state.settled_count)
        self._last_target_regime_id = state.last_target_regime_id
        self._latest_features = {}
        self._latest_baseline = {}

    def reset(self) -> None:
        self.restore_state(
            RegimeScoreLearnerState(
                weights=(),
                update_count=0,
                abs_error_sum=0.0,
                baseline_abs_error_sum=0.0,
                settled_count=0,
                last_target_regime_id="",
            )
        )

    def _predict(
        self, regime_id: str, features: tuple[float, ...], baseline: float
    ) -> float:
        weights = self._weights_for(regime_id)
        residual = sum(weight * feature for weight, feature in zip(weights, features, strict=True))
        return _clamp(baseline + residual)

    def _weights_for(self, regime_id: str) -> list[float]:
        if regime_id not in self._weights:
            self._weights[regime_id] = [0.0] * self._FEATURE_DIM
        return self._weights[regime_id]

    def _align(self, weights: tuple[float, ...]) -> list[float]:
        values = [max(-_WEIGHT_LIMIT, min(_WEIGHT_LIMIT, float(value))) for value in weights]
        if len(values) >= self._FEATURE_DIM:
            return values[: self._FEATURE_DIM]
        return values + [0.0] * (self._FEATURE_DIM - len(values))

    def _readout(
        self,
        *,
        learned_scores: tuple[tuple[str, float], ...],
        baseline_scores: tuple[tuple[str, float], ...],
    ) -> RegimeLearnedScoreShadow:
        learned_mae = self._abs_error_sum / self._settled_count if self._settled_count else 0.0
        baseline_mae = (
            self._baseline_abs_error_sum / self._settled_count
            if self._settled_count
            else 0.0
        )
        improvement = baseline_mae - learned_mae
        blocking: list[str] = []
        if self._settled_count < _MIN_UPDATES:
            blocking.append(f"settled_count<{_MIN_UPDATES}")
        if improvement < _MAE_MARGIN:
            blocking.append(f"mae_improvement<{_MAE_MARGIN:.2f}")
        kill = self._settled_count >= _MIN_UPDATES and improvement <= -_KILL_DEGRADATION
        return RegimeLearnedScoreShadow(
            learned_scores=learned_scores,
            baseline_scores=baseline_scores,
            update_count=self._update_count,
            running_abs_error=round(learned_mae, 6),
            last_target_regime_id=self._last_target_regime_id,
            ready=not blocking,
            kill_recommended=kill,
            blocking_reasons=tuple(blocking),
            description=(
                "RegimeScoreLearner SHADOW candidate; live scorer unchanged; "
                f"settled={self._settled_count} learned_mae={learned_mae:.4f} "
                f"baseline_mae={baseline_mae:.4f} improvement={improvement:.4f}."
            ),
        )


__all__ = ["RegimeScoreLearner", "RegimeScoreLearnerState"]
