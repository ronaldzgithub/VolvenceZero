"""SSE streaming surface for ``/dlaas/instances/{ai_id}/interactions``.

Partially closes known-debt **#12** (DLaaS Slice 5.4 真流式 SSE): the
interaction route now honours ``output_contract.stream=true`` for the
streamable interaction types and answers with the documented native
event scheme (``DLAAS_README.md`` §"Send A Chat Interaction"):

```text
event: ack    -> {"ai_id", "session_id", "contract_id", "interaction_type"}
event: chunk  -> {"content": "<text segment>"}          (0..n per text act)
event: act    -> one OutputAct JSON object              (1..n)
event: done   -> the FULL non-streaming JSON body       (exactly once)
event: error  -> {"error", "detail", "status"}          (terminal, replaces done)
```

Contract notes (consumers MUST rely on these, nothing else):

* ``ack`` is written **before** the kernel turn runs, so callers get an
  immediate "accepted" signal even when cognition takes seconds.
* ``chunk`` frames carry plain-text segments of every ``act_type='text'``
  OutputAct, in order; their concatenation equals the final text. Today
  the kernel produces the turn text atomically
  (``OpenWeightResidualRuntime.generate`` is a sync block — debt #12),
  so chunks are emitted post-generation at the platform layer. When the
  substrate streaming additive interface lands, chunks become genuine
  incremental tokens — the wire contract does not change.
* ``act`` frames are the structured authority; ``done`` carries the full
  response body byte-equal in shape to the non-streaming JSON response,
  so a consumer can persist from ``done`` exactly as it would from the
  JSON path.
* A typed :class:`~dlaas_platform_api.dispatch.DispatchError` raised
  after ``ack`` is surfaced as an ``error`` frame (never a silent EOF);
  unexpected exceptions emit an ``error`` frame and then re-raise so the
  server logs the failure loudly.
* Non-streamable interaction types and the paused / operator-takeover
  path silently degrade to plain JSON, matching the
  :class:`dlaas_platform_contracts.OutputContract` "best-effort" clause.
  Clients must therefore branch on the response ``Content-Type``.

Spec: ``docs/specs/dlaas-api-v1.md`` §"Interactions (native runtime
envelope)" documents the same frames.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web
from dlaas_platform_contracts import InteractionEnvelope, InteractionType

from dlaas_platform_api.dispatch import DispatchError

_LOG = logging.getLogger("dlaas_platform_api.streaming")

# Only the conversational turn types stream. ``observe`` / ``feedback`` /
# ``report`` / ``command`` responses are platform notices with no
# incremental text, so they keep the JSON shape even when the caller
# sets ``output_contract.stream=true``.
STREAMABLE_INTERACTION_TYPES: frozenset[InteractionType] = frozenset(
    {
        InteractionType.CHAT,
        InteractionType.TEACH,
        InteractionType.TASK,
    }
)

# Presentational segmentation width while the kernel emits text
# atomically (see module docstring). Not a wire-contract value — when
# real token hooks land the segment size simply becomes "whatever the
# substrate produced".
STREAM_CHUNK_CHARS = 120

_SSE_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def interaction_stream_requested(envelope: InteractionEnvelope) -> bool:
    """True when the caller opted into SSE AND the type is streamable."""

    return (
        envelope.output_contract.stream
        and envelope.interaction_type in STREAMABLE_INTERACTION_TYPES
    )


def chunk_text(content: str, size: int = STREAM_CHUNK_CHARS) -> tuple[str, ...]:
    """Split ``content`` into ordered segments whose concatenation is exact."""

    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size!r}")
    if not content:
        return ()
    return tuple(content[i : i + size] for i in range(0, len(content), size))


async def _write_event(
    response: web.StreamResponse, event: str, data: Any
) -> None:
    payload = json.dumps(data, ensure_ascii=False)
    await response.write(f"event: {event}\ndata: {payload}\n\n".encode())


async def respond_with_interaction_stream(
    request: web.Request,
    *,
    ai_id: str,
    envelope: InteractionEnvelope,
    run_dispatch: Callable[[], Awaitable[dict[str, Any]]],
    on_success: Callable[[dict[str, Any]], None],
) -> web.StreamResponse:
    """Frame one interaction dispatch as the native SSE event scheme.

    ``run_dispatch`` performs the actual kernel call and returns the
    same JSON body the non-streaming path would serialise; it may raise
    :class:`DispatchError`. ``on_success`` runs the caller's
    audit / usage / snapshot bookkeeping exactly once after a successful
    dispatch, before the terminal ``done`` frame is written.
    """

    response = web.StreamResponse(status=200, headers=dict(_SSE_HEADERS))
    await response.prepare(request)
    await _write_event(
        response,
        "ack",
        {
            "ai_id": ai_id,
            "session_id": envelope.session_id,
            "contract_id": envelope.contract_id,
            "interaction_type": envelope.interaction_type.value,
        },
    )
    try:
        body = await run_dispatch()
    except DispatchError as exc:
        await _write_event(
            response,
            "error",
            {"error": exc.code, "detail": exc.detail, "status": exc.status},
        )
        await response.write_eof()
        return response
    except Exception:
        # Surface the failure on the open stream (the client must never
        # see a silent EOF), then re-raise so the server error path logs
        # it exactly like the non-streaming route would.
        _LOG.exception(
            "interaction stream dispatch failed for ai_id=%s session_id=%s",
            ai_id,
            envelope.session_id,
        )
        await _write_event(
            response,
            "error",
            {
                "error": "internal_error",
                "detail": "interaction dispatch raised an unexpected error",
                "status": 500,
            },
        )
        await response.write_eof()
        raise

    # ``ok_envelope`` guarantees ``output_acts`` on every dispatch body;
    # a missing key is a contract bug and must fail loudly (KeyError).
    output_acts = body["output_acts"]
    for act in output_acts:
        if act["act_type"] == "text":
            for piece in chunk_text(str(act["payload"]["content"] or "")):
                await _write_event(response, "chunk", {"content": piece})
    for act in output_acts:
        await _write_event(response, "act", act)
    on_success(body)
    await _write_event(response, "done", body)
    await response.write_eof()
    return response


__all__ = [
    "STREAMABLE_INTERACTION_TYPES",
    "STREAM_CHUNK_CHARS",
    "chunk_text",
    "interaction_stream_requested",
    "respond_with_interaction_stream",
]
