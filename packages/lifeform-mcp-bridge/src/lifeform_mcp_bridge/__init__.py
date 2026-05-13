"""Bridge external MCP servers (tools / resources / prompts / evals)
into the lifeform stack.

See ``docs/specs/mcp-bridge.md`` for the spec; ``packages/lifeform-mcp-bridge/pyproject.toml``
for the wheel boundary.

Public API:

* ``MCPServerSpec`` — frozen configuration dataclass; one per
  external MCP server you want to attach to a Lifeform.
* ``MCPClientPool`` — process supervision, lifecycle, restart policy.
* ``populate_registry(...)`` — register every MCP-tool affordance into
  the existing ``AffordanceRegistry`` + ``AffordanceInvoker``.
* ``fetch_envelopes(...)`` — pull MCP resources into ``IngestionEnvelope``
  payloads ready for ``BrainSession.run_turn(... trigger_kind=INGESTION)``.
* ``fetch_prompt_events(...)`` — pull MCP prompts into payloads ready
  for ``BrainSession.submit_reviewed_knowledge_event``.
* ``load_scenarios(...)`` — load ``eval-scenarios/*.json`` from the
  external bundle (no MCP RPC).
* ``SafetyManifest`` / ``load_manifest(...)`` — reviewed safety
  manifest loader.
* ``MCPClientProtocol`` / ``StdioMCPClient`` — wire-protocol layer.
* Typed errors: ``MCPBridgeError`` + 6 subclasses.

Stability: 0.1.0 — public surface is ``populate_registry`` /
``fetch_envelopes`` / ``fetch_prompt_events`` / ``load_scenarios``
plus the spec dataclasses. The lower-level adapters / client classes
are exported for advanced consumers and tests but their signatures
may change in 0.2.x.
"""

from lifeform_mcp_bridge.affordance_adapter import (
    MCPAffordanceRegistration,
    descriptor_name_for,
    populate_registry,
)
from lifeform_mcp_bridge.client import (
    MCPClientProtocol,
    StdioMCPClient,
)
from lifeform_mcp_bridge.client_pool import MCPClientPool
from lifeform_mcp_bridge.errors import (
    MCPBridgeError,
    MCPCallTimeoutError,
    MCPConnectionLostError,
    MCPMissingSafetyManifestError,
    MCPProtocolError,
    MCPSafetyManifestSchemaError,
    MCPServerSpawnError,
)
from lifeform_mcp_bridge.eval_loader import (
    EvalScenarioLoadError,
    MCPEvalScenario,
    load_scenarios,
)
from lifeform_mcp_bridge.prompt_adapter import (
    MCPPromptEvent,
    fetch_prompt_events,
)
from lifeform_mcp_bridge.resource_adapter import fetch_envelopes
from lifeform_mcp_bridge.safety_manifest import (
    SafetyManifest,
    SafetyManifestEntry,
    load_manifest,
)
from lifeform_mcp_bridge.server_spec import MCPServerSpec


__all__ = [
    "EvalScenarioLoadError",
    "MCPAffordanceRegistration",
    "MCPBridgeError",
    "MCPCallTimeoutError",
    "MCPClientPool",
    "MCPClientProtocol",
    "MCPConnectionLostError",
    "MCPEvalScenario",
    "MCPMissingSafetyManifestError",
    "MCPProtocolError",
    "MCPPromptEvent",
    "MCPSafetyManifestSchemaError",
    "MCPServerSpawnError",
    "MCPServerSpec",
    "SafetyManifest",
    "SafetyManifestEntry",
    "StdioMCPClient",
    "descriptor_name_for",
    "fetch_envelopes",
    "fetch_prompt_events",
    "load_manifest",
    "load_scenarios",
    "populate_registry",
]
