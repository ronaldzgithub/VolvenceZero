"""Translate MCP ``prompts/list`` -> reviewed knowledge events.

Per ``docs/specs/mcp-bridge.md`` § "Prompt translation":

* Default off (``MCPServerSpec.enable_prompts=False`` AND
  ``manifest.prompts.enabled=False``); both must be True for the
  adapter to do anything.
* Each MCP prompt becomes a dict ready for
  ``BrainSession.submit_reviewed_knowledge_event`` (the canonical
  reviewed-knowledge entry path). Confidence defaults to 0.7 — low
  enough that the lifeform treats it as guidance, not as authoritative
  fact.
* Like the resource adapter, the prompt adapter is pure: it produces
  the event payloads, the caller routes them through the brain
  session.

This adapter is deliberately small in v0 because most MCP servers
do not expose useful prompts yet. It exists so the spec coverage
is complete and the wiring is in place for the day the ecosystem
matures.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lifeform_mcp_bridge.client_pool import MCPClientPool
from lifeform_mcp_bridge.errors import MCPBridgeError
from lifeform_mcp_bridge.safety_manifest import load_manifest
from lifeform_mcp_bridge.server_spec import MCPServerSpec


_LOG = logging.getLogger("lifeform_mcp_bridge.prompt_adapter")
_DEFAULT_CONFIDENCE = 0.7


@dataclass(frozen=True)
class MCPPromptEvent:
    """Plain payload for ``BrainSession.submit_reviewed_knowledge_event``.

    Caller does the actual submission so this adapter stays free of
    a hard dependency on a brain session object — easier to test +
    keeps the wheel boundary clean.
    """

    knowledge_id: str
    summary: str
    detail: str
    source_label: str
    confidence: float


async def fetch_prompt_events(
    *,
    pool: MCPClientPool,
    specs: Iterable[MCPServerSpec],
) -> tuple[MCPPromptEvent, ...]:
    """Discover MCP prompts and turn them into knowledge event payloads.

    Skips specs where either the server-side toggle
    (``enable_prompts``) or the manifest-side toggle
    (``prompts.enabled``) is False — both gates must agree.
    """
    events: list[MCPPromptEvent] = []
    for spec in specs:
        if not spec.enable_prompts:
            continue
        manifest = load_manifest(
            path=spec.safety_manifest_path,
            expected_server_name=spec.name,
        )
        if not manifest.prompts_enabled:
            _LOG.info(
                "MCP server %r: enable_prompts=True on spec but "
                "manifest.prompts.enabled=False; skipping prompts.",
                spec.name,
            )
            continue
        client = await pool.ensure_started(spec)
        try:
            prompts = await client.list_prompts()
        except MCPBridgeError as exc:
            _LOG.warning(
                "MCP server %r: prompts/list failed (%s); skipping "
                "prompt ingestion this session.",
                spec.name,
                exc,
            )
            continue
        for raw in prompts:
            event = await _build_event(pool=pool, spec=spec, raw=raw)
            if event is not None:
                events.append(event)
    return tuple(events)


async def _build_event(
    *,
    pool: MCPClientPool,
    spec: MCPServerSpec,
    raw: Mapping[str, Any],
) -> MCPPromptEvent | None:
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        _LOG.warning(
            "MCP server %r: prompts/list entry missing name: %r",
            spec.name,
            raw,
        )
        return None
    description = str(raw.get("description", ""))
    detail = description
    try:
        client = pool.client_for(spec.name)
        full = await client.get_prompt(name=name)
    except MCPBridgeError as exc:
        _LOG.warning(
            "MCP server %r: prompts/get for %r failed (%s); falling "
            "back to description only.",
            spec.name,
            name,
            exc,
        )
    else:
        rendered = _render_prompt(full)
        if rendered:
            detail = rendered
    summary = description or f"MCP prompt {name} from server {spec.name}"
    return MCPPromptEvent(
        knowledge_id=f"mcp_prompt:{spec.name}:{name}",
        summary=summary[:240],
        detail=detail[:4000],
        source_label=f"mcp:{spec.name}:prompts",
        confidence=_DEFAULT_CONFIDENCE,
    )


def _render_prompt(full: Mapping[str, Any]) -> str:
    """MCP ``prompts/get`` returns ``{"messages": [{"role": .., "content": {"type": "text", "text": ".."}}, ...]}``.

    We concatenate the text bodies, prefixed by role, separated by
    blank lines. Non-text content is annotated as ``[non-text content]``
    so the consumer sees the prompt is not pure text.
    """
    messages = full.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user"))
        content = message.get("content")
        if isinstance(content, Mapping) and content.get("type") == "text":
            text = content.get("text")
            if isinstance(text, str) and text:
                parts.append(f"[{role}] {text}")
        elif isinstance(content, str) and content:
            parts.append(f"[{role}] {content}")
        else:
            parts.append(f"[{role}] [non-text content]")
    return "\n\n".join(parts)


__all__ = [
    "MCPPromptEvent",
    "fetch_prompt_events",
]
