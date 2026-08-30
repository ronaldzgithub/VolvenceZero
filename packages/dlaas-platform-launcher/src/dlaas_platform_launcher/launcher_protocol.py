"""Launcher protocol shared by the in-process and multi-pod launchers.

The api wheel routes ai_id traffic + lifecycle calls through "the
launcher" stored at ``INSTANCE_MANAGER_APP_KEY``. Historically that was
always an :class:`InstanceManager` and the handlers gated on
``isinstance(launcher, InstanceManager)``. To let a
:class:`MultiPodLauncher` (multi-process / multi-GPU) drop in, both
satisfy this :class:`LauncherProtocol`; the handlers check the Protocol
instead of a concrete class.

``forward_interaction`` is the primary multi-process discriminator. Optional
``forward_session_create`` and ``forward_brain_request`` methods extend the
same pod affinity to explicit session and vertical Brain traffic; remote pods
retain ownership of their sessions and product-lineage controllers.  The
Operations-specific protocol remains as a compatibility surface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LauncherProtocol(Protocol):
    """Lifecycle + routing surface the api wheel depends on."""

    def get(self, ai_id: str) -> Any: ...

    def overview(self) -> Any: ...

    def status(self, ai_id: str) -> Any: ...

    async def wake(self, *, ai_id: str, **kwargs: Any) -> Any: ...

    async def sleep(self, *, ai_id: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class InteractionForwardingLauncherProtocol(Protocol):
    """Capability contract for launchers that own remote interaction routing."""

    async def forward_interaction(self, *, ai_id: str, envelope: Any) -> Any: ...


@runtime_checkable
class ExplicitSessionForwardingLauncherProtocol(Protocol):
    """Capability contract for explicit session creation on an owning pod."""

    async def forward_session_create(
        self,
        *,
        ai_id: str,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]: ...


@runtime_checkable
class VerticalBrainForwardingLauncherProtocol(Protocol):
    """Capability contract for sticky vertical Brain request forwarding."""

    async def forward_brain_request(
        self,
        *,
        ai_id: str,
        session_id: str,
        operation: str,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]: ...


@runtime_checkable
class OperationsForwardingLauncherProtocol(Protocol):
    """Compatibility contract for legacy Operations-only forwarding."""

    async def forward_operations_request(
        self,
        *,
        ai_id: str,
        session_id: str,
        operation: str,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]: ...


__all__ = [
    "ExplicitSessionForwardingLauncherProtocol",
    "InteractionForwardingLauncherProtocol",
    "LauncherProtocol",
    "OperationsForwardingLauncherProtocol",
    "VerticalBrainForwardingLauncherProtocol",
]
