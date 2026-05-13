"""mcp-tools-bundle-bridge: MCP tool def -> AffordanceDescriptor.

Per ``docs/specs/mcp-bridge.md`` § "Tool translation table", the
bridge must produce an ``AffordanceDescriptor`` whose:

* ``name`` is prefixed with the server name (``<server>.<tool>``)
* ``parameters_schema`` is the MCP ``inputSchema``
* ``safety_model`` / ``cost_model`` / ``when_to_use`` / ``when_not_to_use``
  come from the manifest (NOT the MCP server)
* ``affordance_tags`` always include ``"mcp"`` even when the manifest
  did not list it
* ``source_path`` is ``mcp://<server>/<tool>``

This is the structural contract test; it uses the same
``_StubMCPClient`` pattern as the safety-manifest test so no
subprocesses are spawned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lifeform_affordance import (
    AffordanceInvoker,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceRegistry,
)
from lifeform_mcp_bridge import (
    MCPClientPool,
    MCPServerSpec,
    descriptor_name_for,
    populate_registry,
)


_MANIFEST = """
schema_version: 1
server:
  name: test-srv
  description: Translation-test server.
tools:
  - name: read_file
    when_to_use: |
      Reading a file from the workspace gives the lifeform exact
      knowledge of the file's contents before reasoning about it.
    when_not_to_use: |
      Don't use this tool to enumerate large directories; that should
      be a different list_dir affordance bound to a different backend.
    cost_model:
      latency_class: fast
      monetary_class: free
      rate_limit_per_minute: 60
    safety_model:
      requires_user_confirmation: false
      irreversible: false
      requires_consent_grant: ["filesystem_read"]
      blocked_in_regimes: ["emotional_support"]
      audit_required: false
    affordance_tags: ["read", "filesystem"]
"""


class _StubMCPClient:
    def __init__(self, *, server_name, tools_listing):
        self._server_name = server_name
        self._tools_listing = tuple(tools_listing)
        self._alive = True

    @property
    def server_name(self): return self._server_name

    @property
    def is_alive(self): return self._alive

    async def initialize(self): return {"protocolVersion": "2024-11-05"}
    async def list_tools(self): return self._tools_listing
    async def call_tool(self, *, name, arguments):
        return {"content": [{"type": "text", "text": f"stub:{name}"}]}
    async def list_resources(self): return ()
    async def read_resource(self, *, uri): return {"contents": []}
    async def list_prompts(self): return ()
    async def get_prompt(self, *, name, arguments=None):
        return {"messages": []}
    async def shutdown(self): self._alive = False


def _make_factory(server_name, tools_listing):
    async def factory(spec):
        return _StubMCPClient(server_name=server_name, tools_listing=tools_listing)
    return factory


def _make_spec(*, name, manifest_path):
    return MCPServerSpec(
        name=name,
        command=("python", "-c", "pass"),
        safety_manifest_path=manifest_path,
        autostart=False,
    )


@pytest.fixture
def manifest_path(tmp_path: Path) -> str:
    p = tmp_path / ".vzbridge.yaml"
    p.write_text(_MANIFEST, encoding="utf-8")
    return str(p)


@pytest.mark.asyncio
async def test_tool_translation_produces_correct_descriptor(manifest_path) -> None:
    spec = _make_spec(name="test-srv", manifest_path=manifest_path)
    tools_listing = [
        {
            "name": "read_file",
            "description": "Read a UTF-8 file from the bundle sandbox.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    pool = MCPClientPool(client_factory=_make_factory("test-srv", tools_listing))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    audit = await populate_registry(
        pool=pool, specs=[spec], registry=registry, invoker=invoker
    )
    assert len(audit) == 1
    record = audit[0]
    assert record.server_name == "test-srv"
    assert record.mcp_tool_name == "read_file"
    expected_name = descriptor_name_for(
        server_name="test-srv", mcp_tool_name="read_file"
    )
    assert record.descriptor_name == expected_name == "test-srv.read_file"
    descriptor = registry.get(expected_name)
    # Field-by-field translation invariants:
    assert descriptor.name == "test-srv.read_file"
    assert descriptor.description == "Read a UTF-8 file from the bundle sandbox."
    assert descriptor.parameters_schema["type"] == "object"
    assert descriptor.parameters_schema["properties"] == {"path": {"type": "string"}}
    assert descriptor.parameters_schema["required"] == ["path"]
    assert descriptor.cost_model.latency_class is AffordanceLatencyClass.FAST
    assert descriptor.cost_model.monetary_class is AffordanceMonetaryClass.FREE
    assert descriptor.cost_model.rate_limit_per_minute == 60
    assert descriptor.safety_model.requires_consent_grant == ("filesystem_read",)
    assert descriptor.safety_model.blocked_in_regimes == ("emotional_support",)
    assert descriptor.safety_model.requires_user_confirmation is False
    assert descriptor.safety_model.irreversible is False
    assert descriptor.source_path == "mcp://test-srv/read_file"
    # ``mcp`` tag is auto-added even when the manifest did not list it
    assert "mcp" in descriptor.affordance_tags
    assert "read" in descriptor.affordance_tags
    assert "filesystem" in descriptor.affordance_tags
    # version is a stable string suffix so consumers can detect MCP origin
    assert descriptor.version.endswith("+mcp")
    await pool.shutdown_all()


@pytest.mark.asyncio
async def test_descriptor_name_collision_across_servers_does_not_clash(
    tmp_path: Path,
) -> None:
    """Two servers with the same short tool name produce two
    distinct descriptor names because of the server-prefix
    convention.
    """
    manifest_a = tmp_path / "a.yaml"
    manifest_b = tmp_path / "b.yaml"
    manifest_a.write_text(_MANIFEST.replace("test-srv", "alpha"), encoding="utf-8")
    manifest_b.write_text(_MANIFEST.replace("test-srv", "beta"), encoding="utf-8")
    tools_listing = [
        {
            "name": "read_file",
            "description": "x",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }
    ]
    spec_a = _make_spec(name="alpha", manifest_path=str(manifest_a))
    spec_b = _make_spec(name="beta", manifest_path=str(manifest_b))

    # Use independent pools so each spec gets its own factory instance.
    pool_a = MCPClientPool(client_factory=_make_factory("alpha", tools_listing))
    pool_b = MCPClientPool(client_factory=_make_factory("beta", tools_listing))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    await populate_registry(
        pool=pool_a, specs=[spec_a], registry=registry, invoker=invoker
    )
    await populate_registry(
        pool=pool_b, specs=[spec_b], registry=registry, invoker=invoker
    )
    descriptors = sorted(registry.names())
    assert descriptors == ["alpha.read_file", "beta.read_file"]
    await pool_a.shutdown_all()
    await pool_b.shutdown_all()


@pytest.mark.asyncio
async def test_input_schema_with_wrong_top_level_type_is_wrapped(
    manifest_path,
) -> None:
    """If a server returns a non-object inputSchema (rare bug), the
    bridge wraps it rather than failing the whole bringup so the
    other tools still register.
    """
    spec = _make_spec(name="test-srv", manifest_path=manifest_path)
    tools_listing = [
        {
            "name": "read_file",
            "description": "Schema is wrong type below.",
            "inputSchema": {"type": "array", "items": {"type": "string"}},
        }
    ]
    pool = MCPClientPool(client_factory=_make_factory("test-srv", tools_listing))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    await populate_registry(
        pool=pool, specs=[spec], registry=registry, invoker=invoker
    )
    descriptor = registry.get("test-srv.read_file")
    assert descriptor.parameters_schema["type"] == "object"
    # Original bad schema preserved for audit so an operator can
    # diagnose the malformed server.
    assert "raw_input_schema" in descriptor.parameters_schema
    await pool.shutdown_all()


@pytest.mark.asyncio
async def test_tool_marked_excluded_in_manifest_is_registered_but_not_invocable(
    tmp_path: Path,
) -> None:
    """``excluded: true`` in manifest -> descriptor exists but
    ``excluded_from_runtime_selection`` is True; no backend bound.
    """
    excluded_manifest = _MANIFEST.replace(
        'affordance_tags: ["read", "filesystem"]',
        'affordance_tags: ["read", "filesystem"]\n    excluded: true',
    )
    p = tmp_path / "excl.yaml"
    p.write_text(excluded_manifest, encoding="utf-8")
    spec = _make_spec(name="test-srv", manifest_path=str(p))
    tools_listing = [
        {
            "name": "read_file",
            "description": "x",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        }
    ]
    pool = MCPClientPool(client_factory=_make_factory("test-srv", tools_listing))
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    audit = await populate_registry(
        pool=pool, specs=[spec], registry=registry, invoker=invoker
    )
    assert audit[0].excluded is True
    descriptor = registry.get("test-srv.read_file")
    assert descriptor.excluded_from_runtime_selection is True
    # No backend bound for excluded tools, so the invoker treats it
    # as BACKEND_MISSING. (Confirmed indirectly via backend names.)
    assert "test-srv.read_file" not in invoker.backend_names()
    await pool.shutdown_all()
