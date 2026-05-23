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

from typing import Any, Iterable
from uuid import uuid4

from dlaas_platform_contracts import DEFAULT_PROTOCOL_VERSION, OutputAct


# Capability identifiers exposed to integrators. The platform-api
# attaches one of these on every OutputAct so the shell can route the
# payload to the right rendering surface (text widget, voice player,
# talking-head <video>, ...). Shells advertise the subset they support
# in the dispatch envelope under ``shell.embodiment``; the wrapper
# below degrades unsupported capabilities to ``CAPABILITY_TEXT``.
CAPABILITY_TEXT = "text_streaming"
CAPABILITY_TALKING_HEAD = "talking_head_video"


def make_response_id() -> str:
    """Return a fresh ``resp_*`` identifier for one platform response."""
    return f"resp_{uuid4().hex[:12]}"


def text_act(content: str, *, capability: str = CAPABILITY_TEXT) -> OutputAct:
    """Wrap a plain-text response in the canonical text OutputAct."""
    return OutputAct(
        act_type="text",
        capability=capability,
        payload={"content": content},
        degraded=False,
        original_capability="",
    )


def talking_head_video_act(
    *,
    content: str,
    persona_id: str,
    session_token: str,
    scope_key: str,
    shell_capabilities: Iterable[str] = (),
) -> OutputAct:
    """Wrap a response for the L0 visual presence layer.

    The kernel always produces text; this helper attaches the
    talking-head payload that an apps/presence-service-aware shell
    knows how to render.

    When the active shell does not advertise ``talking_head_video`` in
    ``shell_capabilities``, the act is downgraded to plain text with
    ``degraded=True`` and ``original_capability="talking_head_video"``
    so the integrator surfaces a no-regression text bubble. This is
    the platform-side enforcement of the spec contract in
    ``docs/specs/dlaas-platform.md`` §"OutputAct 包装":

        shell 不接受的 capability 由 platform-api 在出站时 degrade 到
        ``text`` + ``degraded=True``；不让 kernel 感知 shell embodiment.
    """

    advertised = frozenset(shell_capabilities)
    if CAPABILITY_TALKING_HEAD not in advertised:
        return OutputAct(
            act_type="text",
            capability=CAPABILITY_TEXT,
            payload={"content": content},
            degraded=True,
            original_capability=CAPABILITY_TALKING_HEAD,
        )
    return OutputAct(
        act_type="text",
        capability=CAPABILITY_TALKING_HEAD,
        payload={
            "content": content,
            "persona_id": persona_id,
            "session_token": session_token,
            "scope_key": scope_key,
        },
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


def tool_call_act(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> OutputAct:
    """Ask a native client to execute a tool and return observe/tool_result."""

    return OutputAct(
        act_type="tool_call",
        capability="tool_calling",
        payload={
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": dict(arguments),
        },
        degraded=False,
        original_capability="",
    )


def tool_task_act(
    *,
    task_id: str,
    status: str,
    poll_after_ms: int,
) -> OutputAct:
    """Surface an async affordance task handle to native clients."""

    return OutputAct(
        act_type="tool_task",
        capability="tool_calling_async",
        payload={
            "task_id": task_id,
            "status": status,
            "poll_after_ms": poll_after_ms,
        },
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
    "CAPABILITY_TALKING_HEAD",
    "CAPABILITY_TEXT",
    "make_response_id",
    "ok_envelope",
    "system_act",
    "talking_head_video_act",
    "text_act",
    "tool_call_act",
    "tool_task_act",
]
