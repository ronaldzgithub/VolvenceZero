"""Helpers that build :class:`OutputAct` instances from kernel results.

The platform-api dispatch handlers each produce one or more
``OutputAct`` entries and a small JSON envelope that carries response
metadata. Centralising the construction here keeps the dispatchers
short and keeps the output-act schema as the SSOT for the platform
wire format (see ``docs/specs/dlaas-platform.md`` §"OutputAct 包装").

Slice 2 only emits ``act_type='text'`` and ``act_type='system'`` with
``capability='text_streaming'`` / ``capability='system_notice'``;
later slices (Ops + Streaming) extend this with ``ack`` / ``chunk`` /
``done`` shapes.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from dlaas_platform_contracts import DEFAULT_PROTOCOL_VERSION, OutputAct


def make_response_id() -> str:
    """Return a fresh ``resp_*`` identifier for one platform response."""
    return f"resp_{uuid4().hex[:12]}"


def text_act(content: str, *, capability: str = "text_streaming") -> OutputAct:
    """Wrap a plain-text response in the canonical text OutputAct."""
    return OutputAct(
        act_type="text",
        capability=capability,
        payload={"content": content},
        degraded=False,
        original_capability="",
    )


def system_act(content: str) -> OutputAct:
    """Wrap a platform-internal status message as a ``system`` OutputAct.

    Used for non-cognitive responses (operator placeholders, command
    acknowledgements, "report drained" notifications). The kernel
    never produces these — they originate from the platform layer and
    must be visible to the integrator without being conflated with
    AI-generated content.
    """
    return OutputAct(
        act_type="system",
        capability="system_notice",
        payload={"content": content},
        degraded=False,
        original_capability="",
    )


def ok_envelope(
    *,
    ai_id: str,
    contract_id: str,
    session_id: str,
    interaction_type: str,
    output_acts: tuple[OutputAct, ...],
    protocol_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body of a successful interaction response.

    Keeps the wire-level keys deterministic so integrators can rely
    on them without inspecting the response shape on a per-type
    basis. ``extra`` lets a dispatcher attach interaction-specific
    metadata (``active_regime``, ``ingestion_report``, etc.) without
    each handler re-deriving the base envelope shape.
    """
    body: dict[str, Any] = {
        "status": "ok",
        "ai_id": ai_id,
        "contract_id": contract_id,
        "session_id": session_id,
        "response_id": make_response_id(),
        "protocol_version": protocol_version or DEFAULT_PROTOCOL_VERSION,
        "interaction_type": interaction_type,
        "output_acts": [act.to_json() for act in output_acts],
    }
    if extra:
        body.update(extra)
    return body


__all__ = [
    "make_response_id",
    "ok_envelope",
    "system_act",
    "text_act",
]
