"""Session-held learned gate for the dual-track fusion SHADOW readout.

W1.A of the intent-alignment remediation plan: replaces the fixed
``derive_learned_gate_shadow`` formula with a bounded online-SGD linear
learner (same pattern as the CP-11 ``_LinearAxisHead`` inside the PE
owner). The learner is owned by the dual-track capability but held by the
session so its weights survive the per-turn ``DualTrackModule`` rebuild —
mirroring the ``_tom_proposal_runtime`` precedent in
``volvence_zero.agent.session``.

Learning loop (report-only, CP-19 kill conditions unchanged):

1. Turn ``t``: ``derive_shadow`` featurizes the typed track readouts,
   predicts ``world_weight`` and remembers the features.
2. Turn ``t+1``: the orchestrator reads the PE owner's realized
   ``ActualOutcome`` (task / relationship axes) from the published
   snapshot and calls ``observe_realized_outcome``; the turn-``t``
   features are scored against a target derived from which track's axis
   actually moved, and the weights take one clamped SGD step.

Nothing here reaches the live world/self track fusion: the output is the
same ``DualTrackLearnedGateShadow`` report-only readout, with the weight
movement capped around the neutral 0.5/0.5 prior so the SHADOW candidate
cannot silently become policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.dual_track.core import (
    DualTrackLearnedGateShadow,
    TrackState,
)

_FEATURE_DIM = 6
# Same movement cap as the heuristic candidate: the learned weight may not
# leave [0.5 - CAP, 0.5 + CAP] while the gate is SHADOW.
_WEIGHT_MOVEMENT_CAP = 0.35


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _track_pressure(track: TrackState) -> float:
    return _clamp(
        track.tension_level * 0.45
        + track.controller_code[0] * 0.25
        + track.controller_code[2] * 0.15
        + min(len(track.active_goals) / 3.0, 1.0) * 0.15
    )


@dataclass(frozen=True)
class DualTrackGateLearnerReadout:
    """Typed learning-state readout for evidence artifacts."""

    update_count: int
    running_abs_error: float
    last_prediction: float
    last_target: float
    weights: tuple[float, ...]
    description: str


class DualTrackGateLearner:
    """Bounded online-SGD learner for the dual-track fusion gate weight."""

    def __init__(self, *, learning_rate: float = 0.08) -> None:
        self._weights = [0.0] * _FEATURE_DIM
        # Neutral prior through the bias term: cold-start predicts 0.5/0.5.
        self._weights[-1] = 0.5
        self._learning_rate = learning_rate
        # Features issued this turn (settled by NEXT turn's realized outcome).
        self._latest_features: tuple[float, ...] | None = None
        # Features from the previous turn, awaiting settlement.
        self._settleable_features: tuple[float, ...] | None = None
        self._update_count = 0
        self._abs_error_sum = 0.0
        self._last_prediction = 0.5
        self._last_target = 0.5

    # ----- prediction path (called by DualTrackModule each turn) -----

    def _featurize(
        self,
        *,
        world_track: TrackState,
        self_track: TrackState,
        cross_track_tension: float,
    ) -> tuple[float, ...]:
        return (
            _track_pressure(world_track),
            _track_pressure(self_track),
            _clamp(cross_track_tension),
            _clamp(world_track.tension_level),
            _clamp(self_track.tension_level),
            1.0,  # bias
        )

    def _predict_world_weight(self, features: tuple[float, ...]) -> float:
        raw = sum(w * f for w, f in zip(self._weights, features, strict=True))
        return _clamp(raw, 0.5 - _WEIGHT_MOVEMENT_CAP, 0.5 + _WEIGHT_MOVEMENT_CAP)

    def derive_shadow(
        self,
        *,
        world_track: TrackState,
        self_track: TrackState,
        cross_track_tension: float,
    ) -> DualTrackLearnedGateShadow:
        features = self._featurize(
            world_track=world_track,
            self_track=self_track,
            cross_track_tension=cross_track_tension,
        )
        # Rotate the settlement window: the previous turn's features become
        # settleable by this turn's realized outcome consumer.
        self._settleable_features = self._latest_features
        self._latest_features = features
        world_weight = self._predict_world_weight(features)
        self_weight = _clamp(1.0 - world_weight)
        running_mae = self.running_abs_error
        confidence = _clamp(
            min(self._update_count / 20.0, 1.0) * (1.0 - min(running_mae, 1.0))
        )
        return DualTrackLearnedGateShadow(
            world_weight=round(world_weight, 4),
            self_weight=round(self_weight, 4),
            cross_track_pressure=round(_clamp(cross_track_tension), 4),
            confidence=round(confidence, 4),
            description=(
                "CP-19 SHADOW dual-track gate candidate (v2 online-SGD "
                "learned, session-held); report-only, "
                f"updates={self._update_count} running_mae={running_mae:.3f}."
            ),
        )

    # ----- learning path (called by the orchestrator after each turn) -----

    def observe_realized_outcome(
        self,
        *,
        task_progress: float,
        relationship_delta: float,
    ) -> bool:
        """Score the previous turn's gate against the realized outcome.

        ``task_progress`` is the PE owner's realized world-axis value in
        [0, 1]; ``relationship_delta`` is the realized self-axis value in
        [-1, 1]. The target favors the track whose axis realized stronger.
        Returns True when an SGD update was applied (False during the
        bootstrap turn, when no prior features exist yet).
        """

        features = self._settleable_features
        self._settleable_features = None
        if features is None:
            return False
        relationship_unit = _clamp((relationship_delta + 1.0) / 2.0)
        target = _clamp(
            0.5 + (_clamp(task_progress) - relationship_unit) * 0.5,
            0.5 - _WEIGHT_MOVEMENT_CAP,
            0.5 + _WEIGHT_MOVEMENT_CAP,
        )
        prediction = self._predict_world_weight(features)
        gradient_scale = self._learning_rate * (target - prediction)
        self._weights = [
            max(-2.0, min(2.0, w + gradient_scale * f))
            for w, f in zip(self._weights, features, strict=True)
        ]
        self._update_count += 1
        self._abs_error_sum += abs(target - prediction)
        self._last_prediction = prediction
        self._last_target = target
        return True

    # ----- readouts -----

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def running_abs_error(self) -> float:
        if self._update_count == 0:
            return 0.0
        return self._abs_error_sum / self._update_count

    def readout(self) -> DualTrackGateLearnerReadout:
        return DualTrackGateLearnerReadout(
            update_count=self._update_count,
            running_abs_error=round(self.running_abs_error, 6),
            last_prediction=round(self._last_prediction, 6),
            last_target=round(self._last_target, 6),
            weights=tuple(round(w, 6) for w in self._weights),
            description=(
                "Session-held dual-track gate learner (bounded online-SGD); "
                "report-only SHADOW candidate."
            ),
        )
