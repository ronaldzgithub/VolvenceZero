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

from dlaas_platform_launcher.instance_manager import (
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_launcher.placement import (
    AiIdPlacementRouter,
    PlacementNotFound,
    RuntimePod,
)


class MultiPodLauncher:
    """Route ai_id lifecycle calls across multiple substrate pods."""

    def __init__(self, router: AiIdPlacementRouter | None = None) -> None:
        self._router = router or AiIdPlacementRouter()
        self._managers: dict[str, InstanceManager] = {}

    @property
    def router(self) -> AiIdPlacementRouter:
        return self._router

    def register_pod(
        self, pod: RuntimePod, manager: InstanceManager
    ) -> None:
        """Register a runtime pod and its in-process InstanceManager."""

        self._router.register_pod(pod)
        self._managers[pod.runtime_pod_id] = manager

    def manager_for(self, ai_id: str) -> InstanceManager:
        """Return the InstanceManager of the pod that owns ``ai_id``."""

        record = self._router.resolve(ai_id)
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
            **acquire_kwargs,
        )

    def get(self, ai_id: str):
        """Return the SessionManager for ``ai_id`` from its owning pod."""

        manager = self.manager_for(ai_id)
        self._router.record_interaction(ai_id)
        return manager.get(ai_id)

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
