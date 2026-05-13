"""Translate MCP ``tools/list`` -> ``AffordanceDescriptor`` + register backend.

Per ``docs/specs/mcp-bridge.md`` § "Tool translation":

* The MCP server is the source of ``name`` / ``description`` /
  ``inputSchema`` / optional ``outputSchema``.
* The ``.vzbridge.yaml`` safety manifest is the source of
  ``safety_model`` / ``cost_model`` / ``when_to_use`` /
  ``when_not_to_use`` / ``affordance_tags``. Without a manifest entry
  the bridge refuses to register the tool (R10 invariant).
* ``AffordanceDescriptor.name`` is prefixed with the server name so
  cross-server tools with the same short name can co-exist
  (``coding-bundle.read_file`` and ``research-bundle.read_file``).
* The registered ``AffordanceBackend`` closes over the pool +
  spec.name, so a per-call ``call_tool`` is routed to the right
  client. Pool-level failures (server died) are surfaced as backend
  raises that the ``AffordanceInvoker`` catches and converts to
  ``BACKEND_FAILED``.

Snapshot rendering (``AffordanceModule``) does NOT live here; it is
already implemented in ``lifeform-affordance.module``. The adapter
just feeds the registry + invoker so the same z_t-driven scoring
applies to MCP-supplied tools.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from volvence_zero.affordance import (
    AffordanceDescriptor,
    AffordanceKind,
)

from lifeform_affordance.invoker import AffordanceBackend, AffordanceInvoker
from lifeform_affordance.registry import AffordanceRegistry

from lifeform_mcp_bridge.client_pool import MCPClientPool
from lifeform_mcp_bridge.errors import (
    MCPBridgeError,
    MCPMissingSafetyManifestError,
    MCPProtocolError,
)
from lifeform_mcp_bridge.safety_manifest import (
    SafetyManifest,
    SafetyManifestEntry,
    load_manifest,
)
from lifeform_mcp_bridge.server_spec import MCPServerSpec


_LOG = logging.getLogger("lifeform_mcp_bridge.affordance_adapter")
_DEFAULT_OUTPUT_SCHEMA: Mapping[str, Any] = {"type": "object"}


@dataclass(frozen=True)
class MCPAffordanceRegistration:
    """Audit record for one MCP tool that became an affordance.

    Returned from ``populate_registry`` so callers (tests, dashboards,
    the lifeform startup logger) can inspect the tool->descriptor
    mapping without re-querying the registry.
    """

    server_name: str
    mcp_tool_name: str
    descriptor_name: str
    excluded: bool


def descriptor_name_for(*, server_name: str, mcp_tool_name: str) -> str:
    """Stable convention: ``<server>.<tool>``.

    The dot acts as the boundary marker. ``MCPServerSpec`` rejects
    server names containing dots, so the prefix can be split back
    out reliably.
    """
    return f"{server_name}.{mcp_tool_name}"


async def populate_registry(
    *,
    pool: MCPClientPool,
    specs: Iterable[MCPServerSpec],
    registry: AffordanceRegistry,
    invoker: AffordanceInvoker,
) -> tuple[MCPAffordanceRegistration, ...]:
    """Discover MCP tools across all ``specs`` and register them.

    For each spec:

    1. Ensure the client is started (``pool.ensure_started(spec)``).
    2. Load the per-spec safety manifest.
    3. Call ``tools/list`` on the client.
    4. For every tool in the listing, look up the manifest entry —
       missing => ``MCPMissingSafetyManifestError`` (fail loud).
    5. Build the descriptor + register it. Excluded entries are
       still recorded in the audit so an operator can grep "what
       tools did this server expose vs which did we register".
    6. Bind a backend that calls back through the pool when invoked.

    Returns the audit tuple. Raises typed ``MCPBridgeError`` on any
    failure; the caller decides whether to abort session creation.
    """
    audit: list[MCPAffordanceRegistration] = []
    for spec in specs:
        manifest = load_manifest(
            path=spec.safety_manifest_path,
            expected_server_name=spec.name,
        )
        client = await pool.ensure_started(spec)
        tools = await client.list_tools()
        for raw_tool in tools:
            audit.append(
                _register_one_tool(
                    spec=spec,
                    raw_tool=raw_tool,
                    manifest=manifest,
                    pool=pool,
                    registry=registry,
                    invoker=invoker,
                )
            )
    return tuple(audit)


def _register_one_tool(
    *,
    spec: MCPServerSpec,
    raw_tool: Mapping[str, Any],
    manifest: SafetyManifest,
    pool: MCPClientPool,
    registry: AffordanceRegistry,
    invoker: AffordanceInvoker,
) -> MCPAffordanceRegistration:
    mcp_tool_name = raw_tool.get("name", "")
    if not isinstance(mcp_tool_name, str) or not mcp_tool_name.strip():
        raise MCPProtocolError(
            f"server {spec.name!r}: tool entry missing 'name' or has "
            f"non-string name: {raw_tool!r}"
        )
    entry = manifest.lookup(mcp_tool_name)
    if entry is None:
        raise MCPMissingSafetyManifestError(
            f"server {spec.name!r}: tool {mcp_tool_name!r} is exposed by "
            f"the MCP server but has no entry in the safety manifest "
            f"({manifest.manifest_path}). R10 forbids hash-defaulting "
            f"to unsafe; add a reviewed tools[] entry to the manifest."
        )
    descriptor_name = descriptor_name_for(
        server_name=spec.name, mcp_tool_name=mcp_tool_name
    )
    descriptor = _build_descriptor(
        descriptor_name=descriptor_name,
        spec=spec,
        raw_tool=raw_tool,
        entry=entry,
    )
    if descriptor.excluded_from_runtime_selection:
        # Still register so the snapshot-side audit knows the tool
        # exists; AffordanceModule honours the flag and skips it
        # from candidates.
        registry.register(descriptor)
        return MCPAffordanceRegistration(
            server_name=spec.name,
            mcp_tool_name=mcp_tool_name,
            descriptor_name=descriptor_name,
            excluded=True,
        )
    registry.register(descriptor)
    invoker.register_backend(
        descriptor_name,
        _build_backend(
            pool=pool,
            spec_name=spec.name,
            mcp_tool_name=mcp_tool_name,
        ),
    )
    return MCPAffordanceRegistration(
        server_name=spec.name,
        mcp_tool_name=mcp_tool_name,
        descriptor_name=descriptor_name,
        excluded=False,
    )


def _build_descriptor(
    *,
    descriptor_name: str,
    spec: MCPServerSpec,
    raw_tool: Mapping[str, Any],
    entry: SafetyManifestEntry,
) -> AffordanceDescriptor:
    description = raw_tool.get("description", "")
    if not isinstance(description, str) or not description.strip():
        # MCP spec allows empty / missing description; we synthesise
        # a stable fallback rather than failing — the manifest's
        # when_to_use is the real selection signal.
        description = (
            f"MCP tool {raw_tool.get('name', '<unknown>')} from server "
            f"{spec.name}."
        )
    parameters_schema = _coerce_input_schema(raw_tool, descriptor_name=descriptor_name)
    output_schema = raw_tool.get("outputSchema", _DEFAULT_OUTPUT_SCHEMA)
    if not isinstance(output_schema, Mapping):
        output_schema = _DEFAULT_OUTPUT_SCHEMA
    if "type" not in output_schema:
        output_schema = {**dict(output_schema), "type": "object"}
    display_name = raw_tool.get("displayName")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = entry.tool_name.replace("_", " ").title()
    affordance_tags = entry.affordance_tags
    if "mcp" not in affordance_tags:
        affordance_tags = (*affordance_tags, "mcp")
    return AffordanceDescriptor(
        name=descriptor_name,
        kind=AffordanceKind.TOOL,
        version="0.1.0+mcp",
        display_name=display_name,
        description=description,
        when_to_use=entry.when_to_use,
        when_not_to_use=entry.when_not_to_use,
        parameters_schema=parameters_schema,
        output_schema=output_schema,
        cost_model=entry.cost_model,
        safety_model=entry.safety_model,
        preconditions=(),
        affordance_tags=affordance_tags,
        examples=(),
        source_path=f"mcp://{spec.name}/{entry.tool_name}",
        excluded_from_runtime_selection=entry.excluded,
    )


def _coerce_input_schema(
    raw_tool: Mapping[str, Any], *, descriptor_name: str
) -> Mapping[str, Any]:
    """Map MCP ``inputSchema`` to the AffordanceDescriptor JSON Schema.

    MCP requires ``inputSchema`` for tool definitions; we still
    defend against malformed servers by synthesising a permissive
    object schema rather than failing the whole bringup. The bridge
    invoker validates parameters at call time per JSON Schema.
    """
    raw = raw_tool.get("inputSchema")
    if isinstance(raw, Mapping) and raw.get("type") == "object":
        return dict(raw)
    if isinstance(raw, Mapping):
        # Has a schema but with the wrong top-level type. Wrap rather
        # than fail outright — the manifest already declared this
        # tool safe to expose.
        _LOG.warning(
            "MCP tool %r has inputSchema with type=%r; wrapping as "
            "object so the descriptor invariant holds.",
            descriptor_name,
            raw.get("type"),
        )
        return {"type": "object", "properties": {}, "raw_input_schema": dict(raw)}
    # No schema at all: build a permissive default.
    return {"type": "object", "properties": {}}


def _build_backend(
    *,
    pool: MCPClientPool,
    spec_name: str,
    mcp_tool_name: str,
) -> AffordanceBackend:
    """Closure over the pool: each invocation looks up the live client.

    Looking up per-call (rather than capturing the client object at
    register time) means a restarted server transparently rebinds.
    Pool-level "server unavailable" surfaces as
    ``MCPConnectionLostError`` from ``client_for`` -> bridges to
    ``AffordanceInvocationStatus.BACKEND_FAILED`` via the invoker.
    """

    async def _backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            client = pool.client_for(spec_name)
            response = await client.call_tool(
                name=mcp_tool_name, arguments=parameters
            )
        except MCPBridgeError:
            # The invoker catches all backend exceptions and turns
            # them into typed error_class on the result; we re-raise
            # so the typed MCPBridgeError class makes it into the
            # error_class field via the AffordanceInvocationResult.
            raise
        # MCP `tools/call` response shape: {"content": [...], "isError": bool}.
        # We pass the raw mapping back through; downstream semantic
        # adapters extract what they need from the dict. Errors flagged
        # by the server (``isError=True``) get re-raised so the
        # invoker treats them as backend failures.
        if response.get("isError") is True:
            raise MCPProtocolError(
                f"MCP server {spec_name!r} tool {mcp_tool_name!r} "
                f"returned isError=True; content={response.get('content')!r}"
            )
        return dict(response)

    return _backend


__all__ = [
    "MCPAffordanceRegistration",
    "descriptor_name_for",
    "populate_registry",
]
