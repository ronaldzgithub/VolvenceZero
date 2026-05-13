"""mcp-tools-bundle-bridge: safety manifest is mandatory.

Per ``docs/specs/mcp-bridge.md`` invariant 2 (R10): the bridge must
fail-loud if a tool exposed by an MCP server has no entry in the
per-server ``.vzbridge.yaml`` safety manifest. Hash-defaulting to a
silent "unsafe" path would create a hole through the safety gate
that nothing else catches.

This test exercises every documented failure mode of the manifest
loader + the affordance adapter so a regression accidentally
making any of these silent gets caught at CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lifeform_affordance import AffordanceInvoker, AffordanceRegistry
from lifeform_mcp_bridge import (
    MCPClientPool,
    MCPMissingSafetyManifestError,
    MCPSafetyManifestSchemaError,
    MCPServerSpec,
    populate_registry,
)


# ----------------------------------------------------------------------
# Stub MCP client (no subprocess spawn)
# ----------------------------------------------------------------------


class _StubMCPClient:
    """In-process MCPClientProtocol-conforming stub.

    ``tools_listing`` is the only knob the manifest tests poke;
    everything else is shaped to satisfy the protocol but always
    returns empty lists so the resource / prompt paths stay quiet.
    """

    def __init__(self, *, server_name: str, tools_listing) -> None:
        self._server_name = server_name
        self._tools_listing = tuple(tools_listing)
        self._alive = True

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def is_alive(self) -> bool:
        return self._alive

    async def initialize(self):
        return {"protocolVersion": "2024-11-05"}

    async def list_tools(self):
        return self._tools_listing

    async def call_tool(self, *, name, arguments):
        return {"content": [{"type": "text", "text": f"stub:{name}"}]}

    async def list_resources(self):
        return ()

    async def read_resource(self, *, uri):
        return {"contents": []}

    async def list_prompts(self):
        return ()

    async def get_prompt(self, *, name, arguments=None):
        return {"messages": []}

    async def shutdown(self):
        self._alive = False


def _make_stub_factory(server_name: str, tools_listing):
    async def factory(spec):
        return _StubMCPClient(server_name=server_name, tools_listing=tools_listing)

    return factory


# ----------------------------------------------------------------------
# Manifest fixtures
# ----------------------------------------------------------------------


_VALID_MANIFEST = """
schema_version: 1
server:
  name: test-srv
  description: test
tools:
  - name: echo
    when_to_use: |
      Use the echo tool whenever you want to verify the manifest test
      pipeline is functioning end-to-end without surprises.
    when_not_to_use: |
      Don't use echo for anything that requires real side effects;
      it is purely a manifest-fixture probe.
    cost_model:
      latency_class: instant
      monetary_class: free
    safety_model:
      requires_user_confirmation: false
      irreversible: false
      requires_consent_grant: []
      blocked_in_regimes: []
      audit_required: false
    affordance_tags: ["probe"]
"""


def _write_manifest(tmp_path: Path, body: str) -> str:
    path = tmp_path / ".vzbridge.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _make_spec(*, name: str, manifest_path: str) -> MCPServerSpec:
    return MCPServerSpec(
        name=name,
        command=("python", "-c", "pass"),
        safety_manifest_path=manifest_path,
        autostart=False,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_without_manifest_entry_fails_loud(tmp_path: Path) -> None:
    """Server lists ``unmanifested_tool`` but manifest only has ``echo``."""
    manifest_path = _write_manifest(tmp_path, _VALID_MANIFEST)
    spec = _make_spec(name="test-srv", manifest_path=manifest_path)
    pool = MCPClientPool(
        client_factory=_make_stub_factory(
            "test-srv",
            [
                {"name": "echo", "description": "ok", "inputSchema": {"type": "object"}},
                {"name": "unmanifested_tool", "description": "no entry", "inputSchema": {"type": "object"}},
            ],
        )
    )
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(MCPMissingSafetyManifestError, match="unmanifested_tool"):
        await populate_registry(
            pool=pool, specs=[spec], registry=registry, invoker=invoker
        )
    await pool.shutdown_all()


@pytest.mark.asyncio
async def test_manifest_when_to_use_below_minimum_fails_loud(tmp_path: Path) -> None:
    """when_to_use < 50 chars must be rejected at manifest load time."""
    manifest_path = _write_manifest(
        tmp_path,
        """
schema_version: 1
server: {name: test-srv, description: t}
tools:
  - name: echo
    when_to_use: "too short"
    when_not_to_use: |
      This is a valid long when_not_to_use string with more than fifty characters.
    cost_model: {latency_class: instant, monetary_class: free}
    safety_model: {}
""",
    )
    spec = _make_spec(name="test-srv", manifest_path=manifest_path)
    pool = MCPClientPool(client_factory=_make_stub_factory("test-srv", []))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(MCPSafetyManifestSchemaError, match="when_to_use"):
        await populate_registry(
            pool=pool, specs=[spec], registry=registry, invoker=invoker
        )


@pytest.mark.asyncio
async def test_manifest_server_name_mismatch_fails_loud(tmp_path: Path) -> None:
    """manifest.server.name != MCPServerSpec.name => fail-loud."""
    manifest_path = _write_manifest(tmp_path, _VALID_MANIFEST)
    spec = _make_spec(name="actually-different-name", manifest_path=manifest_path)
    pool = MCPClientPool(
        client_factory=_make_stub_factory("actually-different-name", [])
    )
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(
        MCPSafetyManifestSchemaError, match="does not match the BrainConfig"
    ):
        await populate_registry(
            pool=pool, specs=[spec], registry=registry, invoker=invoker
        )


@pytest.mark.asyncio
async def test_missing_manifest_file_fails_loud(tmp_path: Path) -> None:
    spec = _make_spec(
        name="test-srv",
        manifest_path=str(tmp_path / "does-not-exist.yaml"),
    )
    pool = MCPClientPool(client_factory=_make_stub_factory("test-srv", []))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(MCPSafetyManifestSchemaError, match="not found"):
        await populate_registry(
            pool=pool, specs=[spec], registry=registry, invoker=invoker
        )


@pytest.mark.asyncio
async def test_invalid_latency_class_fails_loud(tmp_path: Path) -> None:
    bad_latency = """
schema_version: 1
server: {name: test-srv, description: t}
tools:
  - name: echo
    when_to_use: |
      Long enough to satisfy the minimum-fifty-character bridge rule for
      manifest entries; descriptive prose about the echo tool here.
    when_not_to_use: |
      Long enough to satisfy the minimum-fifty-character bridge rule for
      manifest entries on the don't-use side as well, equally descriptive.
    cost_model:
      latency_class: ludicrous   # not a valid enum
      monetary_class: free
    safety_model: {}
"""
    manifest_path = _write_manifest(tmp_path, bad_latency)
    spec = _make_spec(name="test-srv", manifest_path=manifest_path)
    pool = MCPClientPool(client_factory=_make_stub_factory("test-srv", []))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(MCPSafetyManifestSchemaError, match="latency_class"):
        await populate_registry(
            pool=pool, specs=[spec], registry=registry, invoker=invoker
        )


def test_mcp_server_spec_requires_safety_manifest_path() -> None:
    """``safety_manifest_path`` is required at the spec level too —
    not even constructing a spec without it is allowed.
    """
    with pytest.raises(ValueError, match="safety_manifest_path is required"):
        MCPServerSpec(name="x", command=("p",), safety_manifest_path="")
