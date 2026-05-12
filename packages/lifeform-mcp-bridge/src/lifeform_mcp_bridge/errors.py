"""Typed error hierarchy for the MCP bundle bridge.

Per the workspace ``no-swallow-errors-no-hasattr-abuse`` rule and the
``mcp-bridge.md`` spec invariant 3, every failure mode the bridge can
encounter must be a typed exception. Callers can either:

* Catch ``MCPBridgeError`` to handle "any bridge failure" uniformly.
* Catch a specific subclass when behaviour should differ by failure
  mode (e.g. ``MCPMissingSafetyManifestError`` is a configuration bug
  the operator must fix; ``MCPCallTimeoutError`` is a transient runtime
  condition that the AffordanceInvoker can surface as
  ``BACKEND_FAILED`` and let the metacontroller learn from).

These exceptions never wrap silently: any catch site that swallows
them must turn them into typed evidence (e.g. an
``AffordanceCandidate.blocked_reason``) so the failure stays
inspectable on the snapshot.
"""

from __future__ import annotations


class MCPBridgeError(Exception):
    """Base class for every failure inside the MCP bundle bridge.

    Catch this when the caller wants to handle any bridge failure
    uniformly (e.g. at session-create time decide whether to abort
    bringup or proceed without that server). For more granular
    handling, catch one of the specific subclasses below.
    """


class MCPServerSpawnError(MCPBridgeError):
    """The bridge failed to start the MCP server subprocess.

    Typical causes: ``command`` not on PATH, missing Python module,
    permission denied. Operators must remove or fix the failing
    ``MCPServerSpec`` from ``BrainConfig.mcp_server_specs`` to
    recover; bridge does not silently down-grade by skipping the
    spec.
    """


class MCPConnectionLostError(MCPBridgeError):
    """The MCP server process exited (or stdio pipe closed) mid-session.

    The pool may attempt to restart per the ``restart_policy``; while
    the server is unavailable the corresponding affordances surface
    as ``AffordanceCandidate.blocked_reason="mcp_unavailable:<name>"``
    rather than disappearing from the snapshot — the consumer needs
    to see the gap.
    """


class MCPCallTimeoutError(MCPBridgeError):
    """A single ``tools/call`` / ``resources/read`` / ``prompts/get``
    exceeded ``MCPServerSpec.call_timeout_seconds``.

    Surfaces from ``AffordanceInvoker`` as ``BACKEND_FAILED`` with
    ``error_class="mcp_timeout"``. The tool stays registered (it may
    succeed on the next call); only this specific call failed.
    """


class MCPProtocolError(MCPBridgeError):
    """The MCP server returned a JSON-RPC error or a payload that
    violates the MCP protocol shape (missing required fields, wrong
    types, etc).

    Wrapped on ``AffordanceInvocationResult`` with
    ``error_class="mcp_protocol_error"``. Distinct from
    ``MCPCallTimeoutError`` because the operator response differs:
    a protocol error indicates a bug in the server (or a version
    mismatch), not a transient capacity issue.
    """


class MCPMissingSafetyManifestError(MCPBridgeError):
    """The MCP server lists a tool that is NOT in the per-server
    ``.vzbridge.yaml`` safety manifest.

    Hard configuration error: the operator must add a manifest entry
    (with reviewed ``safety_model`` / ``cost_model`` /
    ``when_to_use`` / ``when_not_to_use``) before the bridge will
    register the tool. Hash-defaulting to "unsafe" would create a
    silent path through R10, which is forbidden.
    """


class MCPSafetyManifestSchemaError(MCPBridgeError):
    """The ``.vzbridge.yaml`` file is structurally invalid.

    Typical causes: ``schema_version`` mismatch, missing required
    field on a tool entry, ``when_to_use`` < 50 chars, illegal enum
    value for ``cost_model.latency_class``, server name does not
    match the ``MCPServerSpec.name`` configured on the BrainConfig.

    The error message includes the offending key path so the
    external repo author can fix the YAML directly.
    """


__all__ = [
    "MCPBridgeError",
    "MCPCallTimeoutError",
    "MCPConnectionLostError",
    "MCPMissingSafetyManifestError",
    "MCPProtocolError",
    "MCPSafetyManifestSchemaError",
    "MCPServerSpawnError",
]
