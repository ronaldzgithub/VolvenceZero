"""Stdio JSON-RPC 2.0 client for MCP servers.

The bridge implements its own thin MCP client over subprocess
stdio so the wheel installs without ``pip install mcp``. The wire
protocol (JSON-RPC 2.0 + the small set of MCP method names used
here) is documented at https://modelcontextprotocol.io/.

What we implement:

* Initialize handshake (``initialize``)
* ``tools/list`` and ``tools/call``
* ``resources/list`` and ``resources/read``
* ``prompts/list`` and ``prompts/get``

What we do not implement (yet):

* HTTP+SSE transport (``MCPServerSpec.transport == "http"`` raises)
* Notifications / cancellation streams
* Sampling / completion request flow
* Roots / logging negotiated capabilities

The client is async; one client per server. Subprocess lifecycle
(spawn, restart, kill) lives in ``client_pool.MCPClientPool``;
this module is the wire protocol.

Tests can substitute a ``MCPClientProtocol``-conforming stub
without spawning an actual subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from typing import Any, Protocol

from lifeform_mcp_bridge.errors import (
    MCPCallTimeoutError,
    MCPConnectionLostError,
    MCPProtocolError,
    MCPServerSpawnError,
)
from lifeform_mcp_bridge.server_spec import MCPServerSpec


_LOG = logging.getLogger("lifeform_mcp_bridge.client")
_JSONRPC_VERSION = "2.0"
# MCP protocol revision the bridge was written against. Servers may
# negotiate to a different (compatible) revision during initialize;
# we record what the server sent for diagnostic logs but do not
# branch behaviour on it.
_BRIDGE_PROTOCOL_VERSION = "2024-11-05"


class MCPClientProtocol(Protocol):
    """Minimal client surface the bridge consumes.

    Tests substitute a fake conforming to this protocol; production
    uses ``StdioMCPClient``. Adding an HTTP backend later means
    writing another class with this same surface.
    """

    @property
    def server_name(self) -> str: ...

    async def initialize(self) -> Mapping[str, Any]: ...

    async def list_tools(self) -> tuple[Mapping[str, Any], ...]: ...

    async def call_tool(
        self, *, name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    async def list_resources(self) -> tuple[Mapping[str, Any], ...]: ...

    async def read_resource(self, *, uri: str) -> Mapping[str, Any]: ...

    async def list_prompts(self) -> tuple[Mapping[str, Any], ...]: ...

    async def get_prompt(
        self, *, name: str, arguments: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]: ...

    async def shutdown(self) -> None: ...

    @property
    def is_alive(self) -> bool: ...


class StdioMCPClient:
    """JSON-RPC 2.0 over stdio against a spawned MCP server subprocess.

    Lifecycle:

    1. ``await client.start()`` spawns the subprocess and runs the
       MCP ``initialize`` handshake.
    2. ``await client.list_tools()`` / ``call_tool()`` / etc.
    3. ``await client.shutdown()`` closes stdin and waits for exit.

    All RPCs share a single mutex so we serialise requests on the
    stdio pipe (MCP allows concurrent requests with distinct ids,
    but mixing stdout writes from concurrent tasks invites parsing
    issues; serialisation is simpler and matches the typical
    one-tool-at-a-time pattern of an LLM session).
    """

    def __init__(self, *, spec: MCPServerSpec) -> None:
        if spec.transport != "stdio":
            raise NotImplementedError(
                f"StdioMCPClient only supports stdio transport; got "
                f"transport={spec.transport!r}. HTTP+SSE transport is "
                f"a planning stub."
            )
        self._spec = spec
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._initialize_response: Mapping[str, Any] = {}

    @property
    def server_name(self) -> str:
        return self._spec.name

    @property
    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.returncode is None

    async def start(self) -> None:
        """Spawn the subprocess and run the MCP initialize handshake.

        Raises ``MCPServerSpawnError`` when the subprocess cannot be
        created (PATH issue, missing module, etc).
        """
        if self.is_alive:
            return
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in self._spec.env.items()})
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._spec.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise MCPServerSpawnError(
                f"failed to spawn MCP server {self._spec.name!r} via "
                f"command={self._spec.command!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._initialize_response = await self.initialize()
        except MCPProtocolError:
            await self.shutdown()
            raise

    async def initialize(self) -> Mapping[str, Any]:
        """Run the MCP ``initialize`` handshake.

        We declare ``protocolVersion = _BRIDGE_PROTOCOL_VERSION`` and
        empty client capabilities (tools / resources / prompts are
        discovered via dedicated list calls, not via capability
        negotiation in this minimal client).
        """
        return await self._rpc(
            method="initialize",
            params={
                "protocolVersion": _BRIDGE_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "lifeform-mcp-bridge",
                    "version": "0.1.0",
                },
            },
        )

    async def list_tools(self) -> tuple[Mapping[str, Any], ...]:
        result = await self._rpc(method="tools/list", params={})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPProtocolError(
                f"server {self._spec.name!r} returned non-list 'tools' "
                f"in tools/list result: {result!r}"
            )
        return tuple(tools)

    async def call_tool(
        self, *, name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return await self._rpc(
            method="tools/call",
            params={"name": name, "arguments": dict(arguments)},
        )

    async def list_resources(self) -> tuple[Mapping[str, Any], ...]:
        result = await self._rpc(method="resources/list", params={})
        resources = result.get("resources")
        if not isinstance(resources, list):
            raise MCPProtocolError(
                f"server {self._spec.name!r} returned non-list "
                f"'resources' in resources/list result: {result!r}"
            )
        return tuple(resources)

    async def read_resource(self, *, uri: str) -> Mapping[str, Any]:
        return await self._rpc(
            method="resources/read",
            params={"uri": uri},
        )

    async def list_prompts(self) -> tuple[Mapping[str, Any], ...]:
        result = await self._rpc(method="prompts/list", params={})
        prompts = result.get("prompts")
        if not isinstance(prompts, list):
            raise MCPProtocolError(
                f"server {self._spec.name!r} returned non-list "
                f"'prompts' in prompts/list result: {result!r}"
            )
        return tuple(prompts)

    async def get_prompt(
        self, *, name: str, arguments: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = dict(arguments)
        return await self._rpc(method="prompts/get", params=params)

    async def shutdown(self) -> None:
        """Close stdin and wait for the subprocess to exit.

        Force-kill after a short grace period if the server does not
        cooperate. Always sets ``self._proc = None`` so subsequent
        ``is_alive`` returns False.
        """
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.returncode is None:
                if proc.stdin is not None and not proc.stdin.is_closing():
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, RuntimeError):
                        pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        _LOG.warning(
                            "MCP server %r did not exit after kill",
                            self._spec.name,
                        )
        finally:
            self._proc = None

    async def _rpc(
        self, *, method: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        async with self._lock:
            return await self._rpc_locked(method=method, params=params)

    async def _rpc_locked(
        self, *, method: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise MCPConnectionLostError(
                f"MCP server {self._spec.name!r}: subprocess not started "
                f"or stdio pipes already closed."
            )
        request_id = self._next_id
        self._next_id += 1
        envelope = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        line = (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            proc.stdin.write(line)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise MCPConnectionLostError(
                f"MCP server {self._spec.name!r}: stdio write failed "
                f"({type(exc).__name__}: {exc}). The subprocess likely "
                f"exited."
            ) from exc
        try:
            raw_line = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=self._spec.call_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise MCPCallTimeoutError(
                f"MCP server {self._spec.name!r} did not respond to "
                f"{method!r} within {self._spec.call_timeout_seconds:.1f}s."
            ) from exc
        if not raw_line:
            raise MCPConnectionLostError(
                f"MCP server {self._spec.name!r}: stdout closed before "
                f"responding to {method!r}; subprocess likely exited "
                f"(returncode={proc.returncode!r})."
            )
        try:
            response = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MCPProtocolError(
                f"MCP server {self._spec.name!r}: response to "
                f"{method!r} is not valid JSON: {exc}; raw="
                f"{raw_line[:256]!r}"
            ) from exc
        if not isinstance(response, Mapping):
            raise MCPProtocolError(
                f"MCP server {self._spec.name!r}: response to "
                f"{method!r} must be a JSON object; got "
                f"{type(response).__name__}."
            )
        if response.get("jsonrpc") != _JSONRPC_VERSION:
            raise MCPProtocolError(
                f"MCP server {self._spec.name!r}: response to "
                f"{method!r} has wrong jsonrpc version: "
                f"{response.get('jsonrpc')!r}; expected "
                f"{_JSONRPC_VERSION!r}."
            )
        if response.get("id") != request_id:
            raise MCPProtocolError(
                f"MCP server {self._spec.name!r}: response id mismatch "
                f"for {method!r}; expected {request_id!r}, got "
                f"{response.get('id')!r}."
            )
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(
                f"MCP server {self._spec.name!r} returned JSON-RPC "
                f"error for {method!r}: {err!r}"
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise MCPProtocolError(
                f"MCP server {self._spec.name!r}: response to "
                f"{method!r} missing 'result' object; got "
                f"{response!r}."
            )
        return result


__all__ = [
    "MCPClientProtocol",
    "StdioMCPClient",
]
