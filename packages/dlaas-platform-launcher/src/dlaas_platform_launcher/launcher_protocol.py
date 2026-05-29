"""Launcher protocol shared by the in-process and multi-pod launchers.

The api wheel routes ai_id traffic + lifecycle calls through "the
launcher" stored at ``INSTANCE_MANAGER_APP_KEY``. Historically that was
always an :class:`InstanceManager` and the handlers gated on
``isinstance(launcher, InstanceManager)``. To let a
:class:`MultiPodLauncher` (multi-process / multi-GPU) drop in, both
satisfy this :class:`LauncherProtocol`; the handlers check the Protocol
instead of a concrete class.

``forward_interaction`` is the multi-process discriminator: when present
and not ``None`` the api forwards the whole interaction envelope to the
owning pod over RPC instead of resolving a local ``SessionManager``
(remote pods hold their own sessions).
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


__all__ = ["LauncherProtocol"]
