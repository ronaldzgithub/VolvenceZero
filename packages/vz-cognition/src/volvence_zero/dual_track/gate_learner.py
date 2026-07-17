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

from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
)
from volvence_zero.dual_track.core import (
    DualTrackLearnedGateShadow,
    TrackState,
    derive_learned_gate_shadow,
)

_DUAL_TRACK_GATE_OWNER_NAME = "dual_track_gate_learner"
_DUAL_TRACK_GATE_SCHEMA_VERSION = 1
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


@dataclass(frozen=True)
class DualTrackGateLearnerState:
    """Checkpointable learner state (C3 rollback contract).

    ``export_state()`` / ``restore_state()`` give the promotion path a
    concrete rollback artifact: before any ACTIVE flip the caller
    checkpoints this state, and rolling back means restoring it (or
    constructing a fresh learner, which is the neutral-prior reset).
    """

    weights: tuple[float, ...]
    update_count: int
    abs_error_sum: float
    heuristic_abs_error_sum: float
    settled_comparison_count: int


# C3 promotion exit conditions (code-complete; the ACTIVE flip itself is
# gated on external-anchor evidence per CP-19 and is NOT taken here).
_PROMOTION_MIN_UPDATES = 50
# The learned gate must beat the fixed-prior heuristic candidate's MAE by
# at least this margin over the same settled turns.
_PROMOTION_MAE_MARGIN = 0.02
# Kill condition: after enough settlements, a learner clearly WORSE than
# the heuristic candidate should be reset to the neutral prior.
_KILL_MAE_DEGRADATION = 0.10


@dataclass(frozen=True)
class DualTrackGatePromotionReadout:
    """CP-19 promotion / kill evidence for the learned fusion gate.

    Report-only: ``ready`` means every code-level exit condition holds
    and only external-anchor evidence remains; ``kill_recommended``
    means the SHADOW dual-run shows sustained degradation versus the
    fixed-prior candidate and the learner should be ``reset()``.
    """

    ready: bool
    kill_recommended: bool
    update_count: int
    learned_mae: float
    heuristic_mae: float
    mae_improvement: float
    blocking_reasons: tuple[str, ...]
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
        # C3 SHADOW dual-run: the fixed-prior heuristic candidate's
        # world_weight for the same turn, settled against the same target
        # so promotion evidence compares like-for-like.
        self._latest_heuristic_weight: float | None = None
        self._settleable_heuristic_weight: float | None = None
        self._heuristic_abs_error_sum = 0.0
        self._settled_comparison_count = 0

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
        # settleable by this turn's realized outcome consumer. The
        # fixed-prior heuristic candidate's weight rotates in lockstep so
        # both candidates are settled against the same realized target
        # (C3 SHADOW dual-run).
        self._settleable_features = self._latest_features
        self._latest_features = features
        self._settleable_heuristic_weight = self._latest_heuristic_weight
        self._latest_heuristic_weight = derive_learned_gate_shadow(
            world_track=world_track,
            self_track=self_track,
            cross_track_tension=cross_track_tension,
        ).world_weight
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
        heuristic_weight = self._settleable_heuristic_weight
        self._settleable_heuristic_weight = None
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
        if heuristic_weight is not None:
            self._heuristic_abs_error_sum += abs(target - heuristic_weight)
            self._settled_comparison_count += 1
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

    @property
    def heuristic_running_abs_error(self) -> float:
        if self._settled_comparison_count == 0:
            return 0.0
        return self._heuristic_abs_error_sum / self._settled_comparison_count

    # ----- C3 promotion / rollback contract -----

    def promotion_readout(self) -> DualTrackGatePromotionReadout:
        """Evaluate the code-level exit conditions for promotion or kill.

        Report-only: nothing here flips any wiring. ``ready=True`` means
        "only external-anchor evidence remains before an ACTIVE packet";
        ``kill_recommended=True`` means the dual-run shows sustained
        degradation and the caller should ``reset()``.
        """

        learned_mae = self.running_abs_error
        heuristic_mae = self.heuristic_running_abs_error
        improvement = heuristic_mae - learned_mae
        blocking: list[str] = []
        if self._update_count < _PROMOTION_MIN_UPDATES:
            blocking.append(
                f"updates {self._update_count} below required {_PROMOTION_MIN_UPDATES}"
            )
        if self._settled_comparison_count < _PROMOTION_MIN_UPDATES:
            blocking.append(
                f"settled dual-run comparisons {self._settled_comparison_count} "
                f"below required {_PROMOTION_MIN_UPDATES}"
            )
        if improvement < _PROMOTION_MAE_MARGIN:
            blocking.append(
                f"MAE improvement {improvement:.4f} below margin "
                f"{_PROMOTION_MAE_MARGIN:.4f} (learned={learned_mae:.4f} "
                f"heuristic={heuristic_mae:.4f})"
            )
        kill = (
            self._settled_comparison_count >= _PROMOTION_MIN_UPDATES
            and improvement <= -_KILL_MAE_DEGRADATION
        )
        ready = not blocking
        return DualTrackGatePromotionReadout(
            ready=ready,
            kill_recommended=kill,
            update_count=self._update_count,
            learned_mae=round(learned_mae, 6),
            heuristic_mae=round(heuristic_mae, 6),
            mae_improvement=round(improvement, 6),
            blocking_reasons=tuple(blocking),
            description=(
                "CP-19 gate promotion evidence: "
                + ("READY (evidence-gated ACTIVE flip remains)" if ready else "NOT READY")
                + ("; KILL recommended (reset to neutral prior)" if kill else "")
                + f"; learned_mae={learned_mae:.4f} heuristic_mae={heuristic_mae:.4f}."
            ),
        )

    def export_state(self) -> DualTrackGateLearnerState:
        return DualTrackGateLearnerState(
            weights=tuple(self._weights),
            update_count=self._update_count,
            abs_error_sum=self._abs_error_sum,
            heuristic_abs_error_sum=self._heuristic_abs_error_sum,
            settled_comparison_count=self._settled_comparison_count,
        )

    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot:
        state = self.export_state()
        return OwnerPersistenceSnapshot(
            owner_name=_DUAL_TRACK_GATE_OWNER_NAME,
            schema_version=_DUAL_TRACK_GATE_SCHEMA_VERSION,
            payload={
                "weights": list(state.weights),
                "update_count": state.update_count,
                "abs_error_sum": state.abs_error_sum,
                "heuristic_abs_error_sum": state.heuristic_abs_error_sum,
                "settled_comparison_count": state.settled_comparison_count,
            },
            description=(
                "DualTrackGateLearner snapshot "
                f"updates={state.update_count} settled={state.settled_comparison_count}"
            ),
        )

    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None:
        if snapshot.owner_name != _DUAL_TRACK_GATE_OWNER_NAME:
            raise HydrationOwnerMismatchError(
                "DualTrackGateLearner expected owner_name="
                f"{_DUAL_TRACK_GATE_OWNER_NAME!r}, got {snapshot.owner_name!r}"
            )
        if snapshot.schema_version != _DUAL_TRACK_GATE_SCHEMA_VERSION:
            raise HydrationVersionMismatchError(
                "DualTrackGateLearner unsupported schema_version="
                f"{snapshot.schema_version!r}; expected "
                f"{_DUAL_TRACK_GATE_SCHEMA_VERSION}"
            )
        try:
            weights = snapshot.payload["weights"]
            if not isinstance(weights, list | tuple):
                raise HydrationPayloadInvalidError(
                    "DualTrackGateLearner weights must be a list; "
                    f"got {type(weights).__name__}"
                )
            self.restore_state(
                DualTrackGateLearnerState(
                    weights=tuple(float(value) for value in weights),
                    update_count=int(snapshot.payload["update_count"]),
                    abs_error_sum=float(snapshot.payload["abs_error_sum"]),
                    heuristic_abs_error_sum=float(
                        snapshot.payload["heuristic_abs_error_sum"]
                    ),
                    settled_comparison_count=int(
                        snapshot.payload["settled_comparison_count"]
                    ),
                )
            )
        except KeyError as exc:
            raise HydrationPayloadInvalidError(
                "DualTrackGateLearner payload missing key "
                f"{exc.args[0]!r}"
            ) from exc
        except ValueError as exc:
            raise HydrationPayloadInvalidError(
                f"DualTrackGateLearner payload is structurally invalid: {exc}"
            ) from exc

    def restore_state(self, state: DualTrackGateLearnerState) -> None:
        if len(state.weights) != _FEATURE_DIM:
            raise ValueError(
                f"gate learner state has {len(state.weights)} weights, "
                f"expected {_FEATURE_DIM}"
            )
        self._weights = list(state.weights)
        self._update_count = max(0, state.update_count)
        self._abs_error_sum = max(0.0, state.abs_error_sum)
        self._heuristic_abs_error_sum = max(0.0, state.heuristic_abs_error_sum)
        self._settled_comparison_count = max(0, state.settled_comparison_count)
        # Pending settlement windows do not survive a restore: they refer
        # to turns the restored learner never saw.
        self._latest_features = None
        self._settleable_features = None
        self._latest_heuristic_weight = None
        self._settleable_heuristic_weight = None

    def reset(self) -> None:
        """Kill-path rollback: return to the neutral 0.5/0.5 prior."""

        self.restore_state(
            DualTrackGateLearnerState(
                weights=tuple([0.0] * (_FEATURE_DIM - 1) + [0.5]),
                update_count=0,
                abs_error_sum=0.0,
                heuristic_abs_error_sum=0.0,
                settled_comparison_count=0,
            )
        )
