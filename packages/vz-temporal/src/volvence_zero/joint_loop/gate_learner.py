"""Learned SSL->RL schedule gate for the ETA/NL joint loop (T3, #88).

The rule cascade in ``_batch_schedule_action`` stays the live writer
(rollback baseline). This module adds the bounded learned head that the
cascade should eventually hand over to: an online logistic regressor over
the same schedule inputs, settled against the realized SSL improvement of
the learning turns it observed. Report-only SHADOW semantics — the shadow
recommendation is published in schedule telemetry for dual-run evidence
and never reaches the live decision until the promotion gate (>=500-turn
validation evidence, #86/#88) clears.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


_FEATURE_DIM = 11
_WEIGHT_BOUND = 3.0
_LEARNING_RATE = 0.05


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exp_value = math.exp(max(value, -60.0))
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class ScheduleGateShadow:
    """Frozen report of one shadow gate derivation."""

    predicted_learning_gain: float
    recommends_learning: bool
    observation_count: int
    settled_count: int
    summary: str


def build_schedule_gate_features(
    *,
    pe_pressure: float,
    family_stability: float,
    rollback_risk: float,
    transition_pressure: float,
    substrate_pressure: float,
    rare_heavy_pressure: float,
    experience_credit: float,
    control_prior_strength: float,
    pe_full_cycle_due: bool,
    rl_due: bool,
) -> tuple[float, ...]:
    """Feature vector shared by shadow derivation and settlement."""

    return (
        1.0,
        pe_pressure,
        family_stability,
        rollback_risk,
        transition_pressure,
        substrate_pressure,
        rare_heavy_pressure,
        experience_credit,
        control_prior_strength,
        1.0 if pe_full_cycle_due else 0.0,
        1.0 if rl_due else 0.0,
    )


class ScheduleGateLearner:
    """Bounded online logistic head over joint-loop schedule inputs.

    Owner: the joint loop runtime (controller layer, online-fast
    timescale). Weights start at zero (prediction 0.5 — no opinion),
    stay inside ``[-_WEIGHT_BOUND, _WEIGHT_BOUND]``, and update only
    from settled outcomes of learning turns. Rollback = fresh instance.
    """

    def __init__(self) -> None:
        self._weights: list[float] = [0.0] * _FEATURE_DIM
        self._pending_features: tuple[float, ...] | None = None
        self._observation_count = 0
        self._settled_count = 0

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def settled_count(self) -> int:
        return self._settled_count

    def weights(self) -> tuple[float, ...]:
        return tuple(self._weights)

    def derive_shadow(self, features: tuple[float, ...]) -> ScheduleGateShadow:
        if len(features) != _FEATURE_DIM:
            raise ValueError(
                f"schedule gate expects {_FEATURE_DIM} features, got {len(features)}"
            )
        self._observation_count += 1
        self._pending_features = features
        logit = sum(w * x for w, x in zip(self._weights, features, strict=True))
        predicted = _sigmoid(logit)
        return ScheduleGateShadow(
            predicted_learning_gain=predicted,
            recommends_learning=predicted >= 0.5,
            observation_count=self._observation_count,
            settled_count=self._settled_count,
            summary=(
                f"schedule-gate-shadow predicted={predicted:.3f} "
                f"observations={self._observation_count} settled={self._settled_count}"
            ),
        )

    def observe_realized_outcome(self, *, realized_gain: float) -> bool:
        """Settle the most recent shadow against a realized learning gain.

        ``realized_gain`` > 0 means the learning turn improved the SSL
        objective (previous prediction loss minus current). One bounded
        logistic SGD step; weights never leave the bound.
        """

        features = self._pending_features
        if features is None:
            return False
        self._pending_features = None
        target = 1.0 if realized_gain > 0.0 else 0.0
        magnitude = min(abs(realized_gain), 1.0)
        if magnitude == 0.0:
            return False
        logit = sum(w * x for w, x in zip(self._weights, features, strict=True))
        predicted = _sigmoid(logit)
        gradient_scale = (target - predicted) * _LEARNING_RATE * (0.25 + magnitude * 0.75)
        for index, feature in enumerate(features):
            candidate = self._weights[index] + gradient_scale * feature
            self._weights[index] = max(-_WEIGHT_BOUND, min(_WEIGHT_BOUND, candidate))
        self._settled_count += 1
        return True
