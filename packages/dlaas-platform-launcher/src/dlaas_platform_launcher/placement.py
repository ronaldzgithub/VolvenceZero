"""Multi-pod placement + routing for DLaaS runtime instances (debt #17).

The single-process :class:`InstanceManager` caps at "one substrate
runtime + N ai_id" — it cannot horizontally scale the same substrate
across GPUs, and a single process crash takes every ai_id down. This
module adds the platform-owned **placement record** and a sticky,
capacity-aware **router** so an ai_id can be assigned to one of many
runtime pods (each pod = one substrate copy on one GPU / one process).

Scope:

* Pure placement *logic* (no process spawning) so it is unit-testable
  and so the api / gateway can resolve ``ai_id -> runtime_pod_id`` before
  forwarding traffic (sticky routing while awake, per
  ``docs/deployment/routing.md``).
* :class:`MultiPodLauncher` composes a placement router with a set of
  per-pod :class:`InstanceManager` objects so ``acquire`` / ``get`` /
  ``release`` route to the owning pod. Each pod owns exactly one
  substrate runtime; spawning the pods (subprocess / container / GPU
  pinning) is an operational concern injected from outside.

The placement record is platform lifecycle state only — it never holds
memory, regime, PE, or semantic-owner snapshots (R8 boundary).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace


def _now_ms() -> int:
    return int(time.time() * 1000)


class PlacementCapacityError(RuntimeError):
    """Raised when no pod has capacity for a new placement."""


class PlacementNotFound(LookupError):
    """Raised when an ai_id has no active placement."""


class PodNotRegistered(LookupError):
    """Raised when a referenced runtime_pod_id is not registered."""


@dataclass(frozen=True)
class RuntimePod:
    """A runtime pod: one substrate copy on one GPU / process.

    ``capacity`` caps how many ai_ids the pod hosts (memory / VRAM
    bound). ``substrate_profile`` records which profile this pod's
    substrate was started with so the router only places matching
    ai_ids on it.
    """

    runtime_pod_id: str
    runtime_pool_id: str = "default"
    substrate_profile: str = ""
    gpu_id: str = ""
    capacity: int = 64

    def __post_init__(self) -> None:
        if not self.runtime_pod_id.strip():
            raise ValueError("RuntimePod.runtime_pod_id must be non-empty")
        if self.capacity < 1:
            raise ValueError(
                f"RuntimePod.capacity must be >= 1, got {self.capacity!r}"
            )


@dataclass(frozen=True)
class PlacementRecord:
    """Platform-owned ``ai_id -> pod`` placement (see routing.md)."""

    ai_id: str
    tenant_id: str
    runtime_pool_id: str
    runtime_pod_id: str
    vertical: str = ""
    substrate_profile: str = ""
    lifecycle_state: str = "awake"
    last_wake_reason: str = "acquire"
    last_interaction_at_ms: int = 0
    placement_generation: int = 1
    released: bool = False


class AiIdPlacementRouter:
    """Sticky, capacity-aware ``ai_id -> runtime_pod`` router.

    Thread-safe (single lock; placement ops are infrequent vs chat
    traffic). Placement is sticky: an already-placed, non-released
    ai_id always resolves to the same pod (routing.md: "sticky routing
    is mandatory while the instance is awake"). New placements pick the
    least-loaded pod whose ``substrate_profile`` matches the request.
    """

    def __init__(self) -> None:
        self._pods: dict[str, RuntimePod] = {}
        self._placements: dict[str, PlacementRecord] = {}
        self._lock = threading.Lock()

    # -- pod registry --------------------------------------------------

    def register_pod(self, pod: RuntimePod) -> None:
        with self._lock:
            self._pods[pod.runtime_pod_id] = pod

    def pods(self) -> tuple[RuntimePod, ...]:
        with self._lock:
            return tuple(self._pods[name] for name in sorted(self._pods))

    def load_of(self, runtime_pod_id: str) -> int:
        """Active (non-released) placement count on a pod."""

        with self._lock:
            return self._load_locked(runtime_pod_id)

    def _load_locked(self, runtime_pod_id: str) -> int:
        return sum(
            1
            for record in self._placements.values()
            if record.runtime_pod_id == runtime_pod_id and not record.released
        )

    # -- placement -----------------------------------------------------

    def place(
        self,
        *,
        ai_id: str,
        tenant_id: str = "",
        vertical: str = "",
        substrate_profile: str = "",
        last_wake_reason: str = "acquire",
    ) -> PlacementRecord:
        """Return the sticky placement for ``ai_id``, creating one if new.

        Re-placing an already-placed, non-released ai_id returns the
        existing record (sticky). A released ai_id is re-placed fresh
        with a bumped ``placement_generation``. Raises
        :class:`PlacementCapacityError` when no matching pod has room.
        """

        if not ai_id.strip():
            raise ValueError("place: ai_id must be non-empty")
        with self._lock:
            existing = self._placements.get(ai_id)
            if existing is not None and not existing.released:
                return existing
            pod = self._select_pod_locked(substrate_profile=substrate_profile)
            generation = (
                existing.placement_generation + 1 if existing is not None else 1
            )
            record = PlacementRecord(
                ai_id=ai_id,
                tenant_id=tenant_id,
                runtime_pool_id=pod.runtime_pool_id,
                runtime_pod_id=pod.runtime_pod_id,
                vertical=vertical,
                substrate_profile=substrate_profile or pod.substrate_profile,
                lifecycle_state="awake",
                last_wake_reason=last_wake_reason,
                last_interaction_at_ms=_now_ms(),
                placement_generation=generation,
                released=False,
            )
            self._placements[ai_id] = record
            return record

    def _select_pod_locked(self, *, substrate_profile: str) -> RuntimePod:
        candidates = [
            pod
            for pod in self._pods.values()
            if not substrate_profile
            or not pod.substrate_profile
            or pod.substrate_profile == substrate_profile
        ]
        if not candidates:
            raise PlacementCapacityError(
                "no runtime pod registered"
                + (
                    f" for substrate_profile={substrate_profile!r}"
                    if substrate_profile
                    else ""
                )
            )
        # Least-loaded with remaining capacity; deterministic tie-break
        # by pod id so placement is reproducible.
        best: tuple[int, str, RuntimePod] | None = None
        for pod in candidates:
            load = self._load_locked(pod.runtime_pod_id)
            if load >= pod.capacity:
                continue
            key = (load, pod.runtime_pod_id, pod)
            if best is None or key[:2] < best[:2]:
                best = key
        if best is None:
            raise PlacementCapacityError(
                "all matching runtime pods are at capacity"
                + (
                    f" for substrate_profile={substrate_profile!r}"
                    if substrate_profile
                    else ""
                )
            )
        return best[2]

    def resolve(self, ai_id: str) -> PlacementRecord:
        """Return the active placement for ``ai_id`` (fail loud)."""

        with self._lock:
            record = self._placements.get(ai_id)
            if record is None or record.released:
                raise PlacementNotFound(ai_id)
            return record

    def record_interaction(self, ai_id: str) -> None:
        with self._lock:
            record = self._placements.get(ai_id)
            if record is None or record.released:
                return
            self._placements[ai_id] = replace(
                record, last_interaction_at_ms=_now_ms()
            )

    def release(self, ai_id: str) -> None:
        """Mark ``ai_id``'s placement released (frees pod capacity)."""

        with self._lock:
            record = self._placements.get(ai_id)
            if record is None or record.released:
                return
            self._placements[ai_id] = replace(
                record, released=True, lifecycle_state="released"
            )

    def migrate(self, *, ai_id: str, target_pod_id: str) -> PlacementRecord:
        """Move ``ai_id`` to ``target_pod_id`` with a bumped generation.

        Migration is the platform's responsibility to pause/drain and
        rehydrate (routing.md); this only updates the placement record
        so traffic re-routes to the new pod.
        """

        with self._lock:
            if target_pod_id not in self._pods:
                raise PodNotRegistered(target_pod_id)
            record = self._placements.get(ai_id)
            if record is None:
                raise PlacementNotFound(ai_id)
            target = self._pods[target_pod_id]
            migrated = replace(
                record,
                runtime_pod_id=target.runtime_pod_id,
                runtime_pool_id=target.runtime_pool_id,
                placement_generation=record.placement_generation + 1,
                released=False,
                lifecycle_state="awake",
                last_interaction_at_ms=_now_ms(),
            )
            self._placements[ai_id] = migrated
            return migrated

    def snapshot(self) -> tuple[PlacementRecord, ...]:
        with self._lock:
            return tuple(
                self._placements[name] for name in sorted(self._placements)
            )


__all__ = [
    "AiIdPlacementRouter",
    "PlacementCapacityError",
    "PlacementNotFound",
    "PlacementRecord",
    "PodNotRegistered",
    "RuntimePod",
]
