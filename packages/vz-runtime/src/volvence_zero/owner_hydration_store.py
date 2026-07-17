"""Packet D (long-horizon-closure) — runtime-side hydration store.

Thin orchestrator that reads / writes ``OwnerPersistenceSnapshot``
payloads through the same ``PersistenceBackend`` the ``MemoryStore``
already uses.

Key conventions:

- Storage key: ``owner_hydration/<owner_name>``. The
  ``owner_hydration/`` prefix lets ``backend.list_checkpoints`` /
  operator tooling tell hydration entries apart from
  ``memory/store`` and other future namespaces.
- Payload format: JSON, same as MemoryStore (we reuse
  ``serialize_checkpoint`` / ``deserialize_checkpoint`` shape but
  without the MemoryStore-specific dataclass).
- Versioning: per-owner ``schema_version`` lives on
  ``OwnerPersistenceSnapshot``; the store's persistence_backend
  also stamps a monotonic ``version`` number for IO ordering.

WiringLevel semantics:

- ``DISABLED`` — store is never constructed (BrainConfig never
  reaches this code path).
- ``SHADOW`` — store writes on save; ``load_snapshot`` returns ``None``.
- ``ACTIVE`` — store writes on save AND returns existing payload on
  load. This is the path that actually hydrates new sessions.

The store is owned by ``Brain`` and passed through to
``BrainSession`` / ``Lifeform`` / ``LifeformSession`` so that all
three hydratable owners (SemanticStateStore, FollowupManager,
VitalsModule) share the same backend without each layer needing to
know about backend internals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from volvence_zero.memory.persistence import PersistenceBackend
from volvence_zero.owner_hydration import (
    HydratableOwnerProtocol,
    HydrationError,
    OwnerPersistenceSnapshot,
)
from volvence_zero.runtime import WiringLevel


_LOG = logging.getLogger("volvence_zero.owner_hydration_store")
_KEY_PREFIX = "owner_hydration/"


@dataclass(frozen=True)
class OwnerHydrationMatrixEntry:
    owner_name: str
    decision: str
    storage_key: str
    reason: str

    def __post_init__(self) -> None:
        if not self.owner_name.strip():
            raise ValueError("owner_name must be non-empty")
        if self.decision not in {"hydrate", "external-owner", "explicit-no-hydrate"}:
            raise ValueError(f"unknown hydration decision {self.decision!r}")
        if not self.storage_key.strip():
            raise ValueError("storage_key must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")


def _key_for(owner_name: str) -> str:
    return f"{_KEY_PREFIX}{owner_name}"


OWNER_HYDRATION_MATRIX: tuple[OwnerHydrationMatrixEntry, ...] = (
    OwnerHydrationMatrixEntry(
        owner_name="semantic_state",
        decision="hydrate",
        storage_key=_key_for("semantic_state"),
        reason="Nine semantic owner slots are single-writer state and export via SemanticStateStore.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="followup_manager",
        decision="hydrate",
        storage_key=_key_for("followup_manager"),
        reason="Pending followups are lifeform-side owner state and must survive session boundaries.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="vitals",
        decision="hydrate",
        storage_key=_key_for("vitals"),
        reason="Drive levels and proactive cooldown are lifeform-side owner state.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="protocol_registry",
        decision="hydrate",
        storage_key=_key_for("protocol_registry"),
        reason="Protocol registry is hydratable when the application owner exposes the protocol.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="social_record_store",
        decision="hydrate",
        storage_key=_key_for("social_record_store"),
        reason="ToM, common-ground, and group social state are owner-held records that must survive session boundaries.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="prediction_error_heads",
        decision="hydrate",
        storage_key=_key_for("prediction_error_heads"),
        reason="PredictionErrorModule owns learned PE critic and predictive-head parameters.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="dual_track_gate_learner",
        decision="hydrate",
        storage_key=_key_for("dual_track_gate_learner"),
        reason="DualTrackGateLearner is session-held learned state for the dual-track SHADOW gate candidate.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="credit_heads",
        decision="hydrate",
        storage_key=_key_for("credit_heads"),
        reason="CreditModule owns the COCOA rewarding-state head and the SHADOW gate-risk learner.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="memory",
        decision="external-owner",
        storage_key="memory/store",
        reason="MemoryStore already owns save_to_backend/load_from_backend and is not duplicated here.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="regime",
        decision="hydrate",
        storage_key=_key_for("regime"),
        reason="RegimeModule owns persistent regime identity, delayed payoff, and bounded calibration state.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="world_temporal",
        decision="explicit-no-hydrate",
        storage_key="none",
        reason="Temporal continuity is checkpoint/rare-heavy owned; live owner hydration would duplicate controller ownership.",
    ),
    OwnerHydrationMatrixEntry(
        owner_name="self_temporal",
        decision="explicit-no-hydrate",
        storage_key="none",
        reason="Temporal continuity is checkpoint/rare-heavy owned; live owner hydration would duplicate controller ownership.",
    ),
)


class OwnerHydrationStore:
    """Read / write OwnerPersistenceSnapshot payloads via a backend.

    Construct with the same ``PersistenceBackend`` the MemoryStore
    is using (``memory_store.persistence_backend`` accessor) so the
    files land under the per-user scope directory.
    """

    def __init__(
        self,
        *,
        backend: PersistenceBackend,
        wiring_level: WiringLevel,
    ) -> None:
        if wiring_level is WiringLevel.DISABLED:
            raise ValueError(
                "OwnerHydrationStore should not be constructed when "
                "wiring_level is DISABLED; the BrainConfig path should "
                "skip construction entirely."
            )
        self._backend = backend
        self._wiring_level = wiring_level
        self._versions: dict[str, int] = {}

    @property
    def wiring_level(self) -> WiringLevel:
        return self._wiring_level

    def load_snapshot(self, owner_name: str) -> OwnerPersistenceSnapshot | None:
        """Return the most-recent persisted snapshot for ``owner_name``,
        or ``None`` when none exists or the store is in SHADOW (write-only).

        SHADOW behavior is intentional: it lets operators turn writes on
        first, observe round-trip behavior in logs, then flip to ACTIVE
        without surprising production sessions with stale state.
        """
        if self._wiring_level is not WiringLevel.ACTIVE:
            return None
        key = _key_for(owner_name)
        result = self._backend.load_checkpoint(key=key)
        if result is None:
            return None
        data, version = result
        try:
            payload_dict = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Fail-loudly per the no-swallow rule; an operator must
            # decide whether to delete the corrupt key.
            raise HydrationError(
                f"OwnerHydrationStore: backend returned non-JSON payload "
                f"for owner_name={owner_name!r}: {exc}"
            ) from exc
        # Track the latest version we've observed so subsequent saves
        # preserve monotonic ordering across the same process.
        if version > self._versions.get(owner_name, 0):
            self._versions[owner_name] = version
        try:
            return OwnerPersistenceSnapshot(
                owner_name=str(payload_dict["owner_name"]),
                schema_version=int(payload_dict["schema_version"]),
                payload=dict(payload_dict["payload"]),
                description=str(payload_dict.get("description", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HydrationError(
                f"OwnerHydrationStore: backend payload for "
                f"owner_name={owner_name!r} is structurally invalid: "
                f"{exc}; raw_keys={sorted(payload_dict)!r}"
            ) from exc

    def save_snapshot(self, snapshot: OwnerPersistenceSnapshot) -> None:
        """Serialise + persist ``snapshot`` under its owner_name key.

        Always-on regardless of SHADOW vs ACTIVE; the difference is
        only on the read side. Backend errors propagate (fail-loudly)
        so the operator sees them.
        """
        key = _key_for(snapshot.owner_name)
        payload = {
            "owner_name": snapshot.owner_name,
            "schema_version": snapshot.schema_version,
            "payload": dict(snapshot.payload),
            "description": snapshot.description,
        }
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        version = self._versions.get(snapshot.owner_name, 0) + 1
        self._versions[snapshot.owner_name] = version
        self._backend.save_checkpoint(key=key, data=data, version=version)

    def hydrate_owner_if_present(
        self,
        owner: HydratableOwnerProtocol,
        owner_name: str,
    ) -> bool:
        """Convenience: load the latest snapshot for ``owner_name``
        and apply it to ``owner``. Returns True iff hydration happened.

        Hydration errors propagate as ``HydrationError`` (typed) so
        the caller can decide whether to abort or proceed with
        bootstrap defaults. SHADOW returns False without touching
        ``owner``.
        """
        snapshot = self.load_snapshot(owner_name)
        if snapshot is None:
            return False
        try:
            owner.hydrate_from_persistence(snapshot)
        except HydrationError:
            # Re-raise so the caller (Brain.create_session) can surface
            # the failure rather than silently dropping back to bootstrap.
            raise
        _LOG.info(
            "OwnerHydrationStore: hydrated %s (schema_version=%s)",
            owner_name,
            snapshot.schema_version,
        )
        return True

    def export_and_save_owner(
        self,
        owner: HydratableOwnerProtocol,
        owner_name: str,
    ) -> OwnerPersistenceSnapshot:
        """Convenience: ask the owner for its current snapshot and
        persist it. Returns the produced snapshot for inspection.
        """
        snapshot = owner.export_persistence_snapshot()
        if snapshot.owner_name != owner_name:
            raise ValueError(
                f"OwnerHydrationStore.export_and_save_owner: owner "
                f"published owner_name={snapshot.owner_name!r}, but "
                f"caller said owner_name={owner_name!r}; refusing to "
                f"save under the wrong key."
            )
        self.save_snapshot(snapshot)
        return snapshot


__all__ = [
    "OWNER_HYDRATION_MATRIX",
    "OwnerHydrationMatrixEntry",
    "OwnerHydrationStore",
]
