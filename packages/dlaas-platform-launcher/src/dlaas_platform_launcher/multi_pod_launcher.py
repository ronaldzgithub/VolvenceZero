"""Multi-pod launcher: route ai_id traffic across per-GPU substrate pods.

Composes an :class:`AiIdPlacementRouter` with a set of per-pod
:class:`InstanceManager` objects (one substrate copy each). It presents
the same ``acquire`` / ``get`` / ``release`` surface the api wheel uses,
but routes each ai_id to its owning pod's InstanceManager so one logical
DLaaS deployment can span multiple GPUs / processes (debt #17).

This does not spawn the pods; pod processes / GPU pinning are injected
from outside (an operator wires N InstanceManagers, each constructed
with its own substrate runtime, then registers them here). Keeping the
spawn out of this class makes the routing logic unit-testable and lets
the same class work for in-process pods (tests) and remote pods (a thin
remote InstanceManager proxy) alike.
"""

from __future__ import annotations

from typing import Any

from dlaas_platform_contracts import InstanceLifecycleState, InstanceStatus

from dlaas_platform_launcher.instance_manager import InstanceNotFound
from dlaas_platform_launcher.placement import (
    AiIdPlacementRouter,
    PlacementNotFound,
    PlacementRecord,
    RuntimePod,
)


def _status_from_placement(record: PlacementRecord) -> InstanceStatus:
    """Build a platform-status readout from a placement record.

    Multi-pod status is derived from the placement record (parent-owned
    lifecycle state) without a synchronous remote call. Per-pod session
    counts are a pod-local detail; the parent reports lifecycle +
    routing only here. A deeper readout uses the async
    :meth:`RemoteInstanceManager.status` directly.
    """

    try:
        state = InstanceLifecycleState(record.lifecycle_state)
    except ValueError:
        state = InstanceLifecycleState.AWAKE
    return InstanceStatus(
        ai_id=record.ai_id,
        lifecycle_state=state,
        vertical=record.vertical or "unknown",
        last_interaction_at_ms=record.last_interaction_at_ms,
        last_wake_reason=record.last_wake_reason,
    )


class MultiPodLauncher:
    """Route ai_id lifecycle + interactions across multiple substrate pods.

    Each pod is represented by a manager object (a
    :class:`RemoteInstanceManager` HTTP proxy in production; a fake in
    tests) registered against a :class:`RuntimePod`. The
    :class:`AiIdPlacementRouter` decides which pod owns each ai_id
    (sticky, capacity- and substrate_profile-aware).

    Because pod sessions live in the child process, the api forwards the
    interaction envelope here via :meth:`forward_interaction` instead of
    resolving a local ``SessionManager`` — that method's presence is the
    api's multi-process discriminator.
    """

    def __init__(self, router: AiIdPlacementRouter | None = None) -> None:
        self._router = router or AiIdPlacementRouter()
        self._managers: dict[str, Any] = {}

    @property
    def router(self) -> AiIdPlacementRouter:
        return self._router

    def register_pod(self, pod: RuntimePod, manager: Any) -> None:
        """Register a runtime pod and its (remote or in-process) manager."""

        self._router.register_pod(pod)
        self._managers[pod.runtime_pod_id] = manager

    def manager_for(self, ai_id: str) -> Any:
        """Return the manager of the pod that owns ``ai_id``.

        Raises :class:`InstanceNotFound` when ``ai_id`` is not placed
        (the api dispatch path maps that to a 404).
        """

        try:
            record = self._router.resolve(ai_id)
        except PlacementNotFound as exc:
            raise InstanceNotFound(ai_id) from exc
        manager = self._managers.get(record.runtime_pod_id)
        if manager is None:
            raise InstanceNotFound(ai_id)
        return manager

    async def acquire(
        self,
        *,
        ai_id: str,
        runtime_template_id: str,
        tenant_id: str = "",
        substrate_profile: str = "",
        **acquire_kwargs: Any,
    ):
        """Place ``ai_id`` on a pod (sticky) then acquire it there."""

        record = self._router.place(
            ai_id=ai_id,
            tenant_id=tenant_id,
            vertical=runtime_template_id,
            substrate_profile=substrate_profile,
        )
        manager = self._managers[record.runtime_pod_id]
        return await manager.acquire(
            ai_id=ai_id,
            runtime_template_id=runtime_template_id,
            tenant_id=tenant_id,
            **acquire_kwargs,
        )

    async def forward_interaction(self, *, ai_id: str, envelope: Any) -> dict:
        """Forward an interaction envelope to the pod that owns ``ai_id``."""

        manager = self.manager_for(ai_id)
        forward = getattr(manager, "forward_interaction", None)
        if not callable(forward):
            raise RuntimeError(
                f"pod manager for ai_id={ai_id!r} does not support remote "
                "interaction forwarding."
            )
        self._router.record_interaction(ai_id)
        return await forward(ai_id=ai_id, envelope=envelope)

    async def wake(
        self,
        *,
        ai_id: str,
        runtime_template_id: str = "",
        tenant_id: str = "",
        substrate_profile: str = "",
        **wake_kwargs: Any,
    ):
        """Place ``ai_id`` (if needed) then wake it on the owning pod."""

        try:
            self._router.resolve(ai_id)
        except PlacementNotFound:
            if not runtime_template_id.strip():
                raise InstanceNotFound(ai_id) from None
            self._router.place(
                ai_id=ai_id,
                tenant_id=tenant_id,
                vertical=runtime_template_id,
                substrate_profile=substrate_profile,
            )
        manager = self.manager_for(ai_id)
        return await manager.wake(
            ai_id=ai_id,
            runtime_template_id=runtime_template_id,
            **wake_kwargs,
        )

    async def sleep(self, *, ai_id: str, **sleep_kwargs: Any):
        """Sleep ``ai_id`` on its owning pod (releases placement if asked)."""

        manager = self.manager_for(ai_id)
        result = await manager.sleep(ai_id=ai_id, **sleep_kwargs)
        if sleep_kwargs.get("release_instance"):
            self._router.release(ai_id)
        return result

    def status(self, ai_id: str) -> InstanceStatus:
        """Placement-derived status (sync). Raises if not placed."""

        try:
            record = self._router.resolve(ai_id)
        except PlacementNotFound as exc:
            raise InstanceNotFound(ai_id) from exc
        return _status_from_placement(record)

    def overview(self) -> tuple[dict[str, Any], ...]:
        """Placement-derived snapshot of every placed ai_id (sync)."""

        return tuple(
            _status_from_placement(record).to_json()
            for record in self._router.snapshot()
            if not record.released
        )

    def get(self, ai_id: str):
        """Not supported in multi-pod mode (sessions live in the pod).

        The api forwards interactions via :meth:`forward_interaction`
        rather than resolving a local SessionManager, so this is never
        called on the multi-pod path. Present so the launcher conforms
        to :class:`LauncherProtocol`.
        """

        raise NotImplementedError(
            "MultiPodLauncher.get is not supported; sessions live in the "
            "owning pod process. Use forward_interaction for dispatch."
        )

    def has(self, ai_id: str) -> bool:
        try:
            self._router.resolve(ai_id)
        except PlacementNotFound:
            return False
        return True

    async def release(self, ai_id: str) -> None:
        """Release ``ai_id`` from its pod and free placement capacity."""

        try:
            manager = self.manager_for(ai_id)
        except (InstanceNotFound, PlacementNotFound):
            self._router.release(ai_id)
            return
        release = getattr(manager, "release", None)
        if callable(release):
            await release(ai_id)
        self._router.release(ai_id)


__all__ = ["MultiPodLauncher"]
