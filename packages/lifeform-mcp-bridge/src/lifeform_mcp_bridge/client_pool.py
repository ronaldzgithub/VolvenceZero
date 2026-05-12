"""``MCPClientPool`` — process supervision for the bridge.

One pool per ``Lifeform`` (sessions of the same lifeform share
clients). Per ``MCPServerSpec``:

* ``ensure_started(spec)`` — spawn (or reuse) a client; runs
  ``initialize`` once.
* ``client_for(name)`` — read a started client by spec name; raises
  if not started.
* ``shutdown_all()`` — close every spawned subprocess; idempotent.
* ``mark_unavailable(name, reason)`` — used by the resource /
  affordance adapters when a per-call failure indicates the
  subprocess died; downstream snapshots can then surface the
  ``mcp_unavailable`` ``blocked_reason`` without the pool having to
  poll process state on every snapshot build.

Restart policy:

The pool does NOT auto-restart on every crash; doing so silently
would mask a misbehaving server. Instead:

* ``"never"`` — leave the client dead after first crash.
* ``"on_crash"`` — at the next ``client_for(name)`` call, if the
  subprocess has died with non-zero exit code, re-spawn once. Repeat
  crash within a short window leaves it dead.
* ``"always"`` — at the next ``client_for(name)`` call, re-spawn no
  matter why it died.

This deferred-restart model lets the affordance snapshot reflect
"server X is down" for at least one turn so the lifeform notices the
gap, instead of papering over it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from lifeform_mcp_bridge.client import (
    MCPClientProtocol,
    StdioMCPClient,
)
from lifeform_mcp_bridge.errors import (
    MCPConnectionLostError,
    MCPServerSpawnError,
)
from lifeform_mcp_bridge.server_spec import MCPServerSpec


_LOG = logging.getLogger("lifeform_mcp_bridge.client_pool")
_RESTART_BACKOFF_SECONDS = 5.0


class MCPClientPool:
    """Per-lifeform pool of MCP clients keyed by ``MCPServerSpec.name``.

    Tests can inject a ``client_factory`` so the pool builds stub
    clients instead of spawning real subprocesses; production uses
    the default ``StdioMCPClient``-based factory.
    """

    def __init__(
        self,
        *,
        client_factory: Any = None,
    ) -> None:
        self._clients: dict[str, MCPClientProtocol] = {}
        self._specs: dict[str, MCPServerSpec] = {}
        self._unavailable: dict[str, str] = {}
        self._last_restart_at: dict[str, float] = {}
        # Tests pass in a sync callable that returns an
        # MCPClientProtocol-conforming object given a spec; production
        # wraps StdioMCPClient + start().
        self._client_factory = client_factory or _default_client_factory

    async def ensure_started(self, spec: MCPServerSpec) -> MCPClientProtocol:
        """Spawn (or reuse) the client for ``spec``.

        Idempotent: subsequent calls with the same ``spec.name``
        return the existing client. If the existing client is dead
        and ``restart_policy`` allows, the pool will re-spawn.
        """
        existing = self._clients.get(spec.name)
        if existing is not None and existing.is_alive:
            return existing
        if existing is not None:
            await self._maybe_restart(spec, prior=existing)
            existing = self._clients.get(spec.name)
            if existing is not None and existing.is_alive:
                return existing
            raise MCPConnectionLostError(
                f"MCPClientPool: server {spec.name!r} is dead and the "
                f"restart_policy={spec.restart_policy!r} did not yield "
                f"a live client."
            )
        # Fresh start.
        client = await self._client_factory(spec)
        self._clients[spec.name] = client
        self._specs[spec.name] = spec
        self._unavailable.pop(spec.name, None)
        return client

    def client_for(self, name: str) -> MCPClientProtocol:
        """Look up a started client by name. Raises if missing."""
        client = self._clients.get(name)
        if client is None:
            raise MCPConnectionLostError(
                f"MCPClientPool: no client started for server name "
                f"{name!r}; call ensure_started(spec) first."
            )
        return client

    def is_unavailable(self, name: str) -> bool:
        """Return True iff the server has been marked unavailable.

        ``MCPAffordanceAdapter`` checks this when building each snapshot
        so MCP-supplied candidates can surface ``blocked_reason``
        cleanly without the snapshot path itself catching protocol
        exceptions.
        """
        client = self._clients.get(name)
        if client is not None and not client.is_alive:
            return True
        return name in self._unavailable

    def unavailable_reason(self, name: str) -> str:
        return self._unavailable.get(name, "")

    def mark_unavailable(self, name: str, *, reason: str) -> None:
        self._unavailable[name] = reason

    def specs(self) -> Mapping[str, MCPServerSpec]:
        return dict(self._specs)

    async def shutdown_all(self) -> None:
        """Close every spawned client. Idempotent.

        Errors during shutdown are logged but never re-raised — the
        pool is a process-level resource and shutdown must succeed
        even if one client has misbehaved.
        """
        for name, client in list(self._clients.items()):
            try:
                await client.shutdown()
            except Exception as exc:  # noqa: BLE001 — last-ditch shutdown
                _LOG.warning(
                    "MCPClientPool: error shutting down client %r: %s",
                    name,
                    exc,
                )
            finally:
                self._clients.pop(name, None)
        self._unavailable.clear()
        self._last_restart_at.clear()

    async def _maybe_restart(
        self,
        spec: MCPServerSpec,
        *,
        prior: MCPClientProtocol,
    ) -> None:
        if spec.restart_policy == "never":
            self.mark_unavailable(
                spec.name,
                reason="server crashed; restart_policy=never",
            )
            return
        # Back-off: don't thrash if the server keeps dying.
        last = self._last_restart_at.get(spec.name, 0.0)
        if (time.monotonic() - last) < _RESTART_BACKOFF_SECONDS:
            self.mark_unavailable(
                spec.name,
                reason=(
                    f"server crashed; in restart back-off window "
                    f"(<{_RESTART_BACKOFF_SECONDS:.0f}s)"
                ),
            )
            return
        try:
            await prior.shutdown()
        except Exception:  # noqa: BLE001 — already dead, swallowing OK on restart path
            pass
        self._last_restart_at[spec.name] = time.monotonic()
        try:
            client = await self._client_factory(spec)
        except MCPServerSpawnError as exc:
            self.mark_unavailable(
                spec.name,
                reason=f"restart failed: {exc}",
            )
            self._clients.pop(spec.name, None)
            return
        self._clients[spec.name] = client
        self._unavailable.pop(spec.name, None)


async def _default_client_factory(spec: MCPServerSpec) -> MCPClientProtocol:
    """Default factory: spawn a real ``StdioMCPClient`` and run start()."""
    client = StdioMCPClient(spec=spec)
    await client.start()
    return client


__all__ = [
    "MCPClientPool",
]
