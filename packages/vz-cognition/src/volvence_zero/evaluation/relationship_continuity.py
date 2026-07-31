"""Read-only seven-day relationship continuity evaluation owner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from volvence_zero.memory.persistence import PersistenceBackend
from volvence_zero.owner_prediction import OwnerPredictionKind
from volvence_zero.relationship_continuity import (
    RelationshipContinuityConsoleOutcome,
    RelationshipContinuitySnapshot,
)
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    OpenLoopSnapshot,
    RelationshipStateSnapshot,
)

_SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class RelationshipContinuityObservation:
    observation_id: str
    observed_at_ms: int
    callback_hits: int
    callback_total: int
    boundary_violation: bool
    open_loop_closed: int
    open_loop_total: int
    trust_level: float | None
    boundary_observed: bool = False


class RelationshipContinuityEvaluationModule:
    """Owns a persisted read-only window; never feeds a learning signal."""

    slot_name = "relationship_continuity"
    _SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        persistence_root: str | Path | None = None,
        persistence_backend: PersistenceBackend | None = None,
        wiring_level: WiringLevel = WiringLevel.SHADOW,
    ) -> None:
        if persistence_root is not None and persistence_backend is not None:
            raise ValueError(
                "pass persistence_root or persistence_backend, not both"
            )
        self._wiring_level = wiring_level
        self._observations: dict[
            str, dict[str, RelationshipContinuityObservation]
        ] = {}
        self._persistence_path = (
            Path(persistence_root) / "relationship-continuity-evaluation.json"
            if persistence_root is not None
            else None
        )
        self._persistence_backend = persistence_backend
        if self._persistence_path is not None and self._persistence_path.exists():
            self._hydrate_file()
        elif self._persistence_backend is not None:
            self._hydrate_backend()

    def observe(
        self,
        *,
        user_id: str,
        observation_id: str,
        observed_at_ms: int,
        snapshots: Mapping[str, Snapshot[Any]],
        console_outcomes: tuple[RelationshipContinuityConsoleOutcome, ...],
    ) -> RelationshipContinuitySnapshot:
        if self._wiring_level is WiringLevel.DISABLED:
            raise RuntimeError("relationship_continuity wiring is DISABLED")
        if self._persistence_backend is not None:
            self._hydrate_backend()
        scope_hash = _scope_hash(user_id)
        observation = _observation_from_snapshots(
            observation_id=observation_id,
            observed_at_ms=observed_at_ms,
            snapshots=snapshots,
        )
        self._observations.setdefault(scope_hash, {})[observation_id] = observation
        self._prune(window_end_ms=observed_at_ms)
        self._persist()
        return self.readout(
            user_id=user_id,
            window_end_ms=observed_at_ms,
            console_outcomes=console_outcomes,
        )

    def readout(
        self,
        *,
        user_id: str,
        window_end_ms: int,
        console_outcomes: tuple[RelationshipContinuityConsoleOutcome, ...],
    ) -> RelationshipContinuitySnapshot:
        scope_hash = _scope_hash(user_id)
        window_start_ms = max(0, window_end_ms - _SEVEN_DAYS_MS)
        observations = tuple(
            sorted(
                (
                    item
                    for item in self._observations.get(scope_hash, {}).values()
                    if window_start_ms <= item.observed_at_ms <= window_end_ms
                ),
                key=lambda item: (item.observed_at_ms, item.observation_id),
            )
        )
        outcomes = tuple(
            item
            for item in console_outcomes
            if window_start_ms <= item.observed_at_ms <= window_end_ms
        )
        callback_hits = sum(item.callback_hits for item in observations)
        callback_total = sum(item.callback_total for item in observations)
        boundary_total = sum(item.boundary_observed for item in observations)
        boundary_violations = sum(
            item.boundary_violation
            for item in observations
            if item.boundary_observed
        )
        boundary_violations += sum(item.is_boundary_violation for item in outcomes)
        boundary_total += len(outcomes)
        correction_total = sum(item.is_correction for item in outcomes)
        wrong_user_total = sum(item.is_wrong_user_attribution for item in outcomes)
        usefulness = tuple(
            item.remembered_item_useful
            for item in outcomes
            if item.remembered_item_useful is not None
        )
        latest = observations[-1] if observations else None
        trust_points = tuple(
            item.trust_level for item in observations if item.trust_level is not None
        )
        trust_delta = (
            _clamp_signed(trust_points[-1] - trust_points[0])
            if len(trust_points) >= 2
            else None
        )
        return RelationshipContinuitySnapshot(
            schema_version=self._SCHEMA_VERSION,
            user_scope_hash=scope_hash,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            callback_hit_rate=_rate(callback_hits, callback_total),
            boundary_violation_rate=_rate(boundary_violations, boundary_total),
            wrong_user_attribution_rate=_rate(wrong_user_total, correction_total),
            open_loop_closure_rate=(
                _rate(latest.open_loop_closed, latest.open_loop_total)
                if latest is not None
                else None
            ),
            user_correction_rate=_rate(correction_total, len(outcomes)),
            remembered_item_usefulness=(
                sum(bool(item) for item in usefulness) / len(usefulness)
                if usefulness
                else None
            ),
            seven_day_trust_delta=trust_delta,
            sample_sizes=(
                ("callback", callback_total),
                ("boundary", boundary_total),
                ("console", len(outcomes)),
                ("correction", correction_total),
                ("open_loop", latest.open_loop_total if latest is not None else 0),
                ("trust", len(trust_points)),
                ("usefulness", len(usefulness)),
            ),
            wiring_level=self._wiring_level.value,
            description=(
                "Read-only seven-day continuity evaluation from typed owner "
                "snapshots and explicit console outcomes; not a learning source."
            ),
        )

    def _prune(self, *, window_end_ms: int) -> None:
        cutoff = max(0, window_end_ms - _SEVEN_DAYS_MS)
        for scope_hash, observations in tuple(self._observations.items()):
            retained = {
                key: item
                for key, item in observations.items()
                if item.observed_at_ms >= cutoff
            }
            if retained:
                self._observations[scope_hash] = retained
            else:
                del self._observations[scope_hash]

    def _persist(self) -> None:
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "observations": {
                scope_hash: [asdict(item) for item in observations.values()]
                for scope_hash, observations in self._observations.items()
            },
        }
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        if self._persistence_backend is not None:
            self._persistence_backend.save_checkpoint(
                key="evaluation/relationship_continuity",
                data=serialized.encode("utf-8"),
                version=self._SCHEMA_VERSION,
            )
        if self._persistence_path is None:
            return
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._persistence_path.with_suffix(".tmp")
        temporary.write_text(
            serialized,
            encoding="utf-8",
        )
        temporary.replace(self._persistence_path)

    def _hydrate_file(self) -> None:
        if self._persistence_path is None:
            raise RuntimeError("relationship continuity persistence path is missing")
        payload = json.loads(self._persistence_path.read_text(encoding="utf-8"))
        self._hydrate_payload(payload)

    def _hydrate_backend(self) -> None:
        if self._persistence_backend is None:
            raise RuntimeError("relationship continuity persistence backend is missing")
        loaded = self._persistence_backend.load_checkpoint(
            key="evaluation/relationship_continuity"
        )
        if loaded is None:
            return
        data, version = loaded
        if version != self._SCHEMA_VERSION:
            raise ValueError("relationship continuity backend version mismatch")
        self._hydrate_payload(json.loads(data.decode("utf-8")))

    def _hydrate_payload(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("relationship continuity payload must be an object")
        if payload.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError("relationship continuity schema_version mismatch")
        raw_observations = payload.get("observations")
        if not isinstance(raw_observations, dict):
            raise ValueError("relationship continuity observations must be an object")
        self._observations = {}
        for scope_hash, items in raw_observations.items():
            if not isinstance(items, list):
                raise ValueError("relationship continuity observation list is invalid")
            self._observations[str(scope_hash)] = {
                str(item["observation_id"]): RelationshipContinuityObservation(**item)
                for item in items
            }


def _observation_from_snapshots(
    *,
    observation_id: str,
    observed_at_ms: int,
    snapshots: Mapping[str, Snapshot[Any]],
) -> RelationshipContinuityObservation:
    from volvence_zero.prediction.error import PredictionErrorSnapshot

    pe = _typed_snapshot_value(snapshots, "prediction_error", PredictionErrorSnapshot)
    boundary = _typed_snapshot_value(
        snapshots, "boundary_consent", BoundaryConsentSnapshot
    )
    open_loop = _typed_snapshot_value(snapshots, "open_loop", OpenLoopSnapshot)
    relationship = _typed_snapshot_value(
        snapshots, "relationship_state", RelationshipStateSnapshot
    )
    callback_settlements = (
        tuple(
            item
            for item in pe.owner_prediction_settlements
            if item.kind
            in {
                OwnerPredictionKind.COMMITMENT_FOLLOW_THROUGH,
                OwnerPredictionKind.OPEN_LOOP_CLOSURE,
            }
        )
        if pe is not None
        else ()
    )
    callback_hits = sum(
        item.mismatch_magnitude <= 0.25 for item in callback_settlements
    )
    open_loop_closed = len(open_loop.closure_refs) if open_loop is not None else 0
    open_loop_total = (
        open_loop_closed + len(open_loop.unresolved_loops)
        if open_loop is not None
        else 0
    )
    return RelationshipContinuityObservation(
        observation_id=observation_id,
        observed_at_ms=observed_at_ms,
        callback_hits=callback_hits,
        callback_total=len(callback_settlements),
        boundary_violation=(
            boundary.overreach_risk > 0.5 if boundary is not None else False
        ),
        open_loop_closed=open_loop_closed,
        open_loop_total=open_loop_total,
        trust_level=(
            relationship.cumulative_trust_level
            if relationship is not None
            else None
        ),
        boundary_observed=boundary is not None,
    )


def _typed_snapshot_value(
    snapshots: Mapping[str, Snapshot[Any]],
    slot: str,
    expected_type: type[Any],
) -> Any | None:
    snapshot = snapshots.get(slot)
    if snapshot is None:
        return None
    if not isinstance(snapshot, Snapshot):
        raise TypeError(f"{slot} must be a Snapshot")
    if not isinstance(snapshot.value, expected_type):
        raise TypeError(
            f"{slot} snapshot value must be {expected_type.__name__}, "
            f"got {type(snapshot.value).__name__}"
        )
    return snapshot.value


def _scope_hash(user_id: str) -> str:
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    return sha256(user_id.encode("utf-8")).hexdigest()[:24]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


__all__ = [
    "RelationshipContinuityConsoleOutcome",
    "RelationshipContinuityEvaluationModule",
    "RelationshipContinuityObservation",
    "RelationshipContinuitySnapshot",
]
