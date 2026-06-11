"""aiohttp routes for the deferred affordance task lifecycle.

Endpoint summary::

    GET  /dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}           — poll status
    POST /dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}/complete  — submit deferred result

Spec: ``docs/specs/dlaas-api-v1.md`` §"Tool Task Lifecycle".

Ownership (R8): the task store lives on the session-scoped
``AffordanceInvoker`` (``lifeform_affordance.invoker``). This surface is
a thin HTTP adapter over ``invoker.get_task_handle`` /
``invoker.submit_deferred_result`` — it never re-implements task state,
and completion flows through the canonical tool bus
(``session.submit_tool_result``) so PE lineage / credit attribution via
``plan_ref`` is preserved exactly like a fast tool result.

Auth: same level as the ``/dlaas/v1/instances/{ai_id}/interactions``
runtime route — the platform's interaction surface does not gate
per-request tenant credentials at the handler today; tool-task polling
and completion are part of the same runtime conversation surface, so
they inherit that posture (operator-plane raw access stays on the admin
snapshot route).

Session resolution mirrors ``app._resolve_session_manager``: the
launcher (when bound) owns the ai_id → SessionManager mapping; the
Slice 1 single-instance fallback is ``app["session_manager"]``. The
``session_id`` is required (query param on GET, body field on POST)
because the invoker that holds the task handle is reached through the
session — an unknown session is a typed 404, never a silent create.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceNotFound,
    LauncherProtocol,
)
from lifeform_service.session_manager import (
    SessionManager,
    SessionNotFoundError,
)

_LOG = logging.getLogger("dlaas_platform_api.tool_tasks")

#: Task statuses that accept a deferred completion. Terminal handles
#: (succeeded / failed / cancelled) reject re-completion with a 409 so a
#: retried HTTP callback cannot double-feed the kernel tool bus.
_COMPLETABLE_TASK_STATUSES = frozenset({"queued", "running"})

_COMPLETION_STATUSES = frozenset({"succeeded", "failed"})

_DEFAULT_ERROR_CLASS = "deferred_backend_failed"


def attach_tool_task_routes(app: web.Application) -> web.Application:
    """Register the tool-task lifecycle routes on ``app``.

    Requires the same session wiring as the interactions route
    (``app["session_manager"]`` and/or a bound launcher); no extra app
    state is introduced.
    """
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}",
        _handle_get_task,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/tool-tasks/{task_id}/complete",
        _handle_complete_task,
    )
    return app


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_get_task(request: web.Request) -> web.Response:
    ai_id = request.match_info["ai_id"]
    task_id = request.match_info["task_id"]
    session_id = request.query.get("session_id", "").strip()
    if not session_id:
        return _error(
            400,
            "missing_session_id",
            "query parameter session_id is required",
        )
    resolved = await _resolve_invoker(request, ai_id=ai_id, session_id=session_id)
    if isinstance(resolved, web.Response):
        return resolved
    invoker, _session = resolved
    try:
        handle = invoker.get_task_handle(task_id)
    except KeyError:
        return _task_not_found(task_id)
    return web.json_response(_handle_to_json(handle))


async def _handle_complete_task(request: web.Request) -> web.Response:
    ai_id = request.match_info["ai_id"]
    task_id = request.match_info["task_id"]
    try:
        body = await _read_json(request)
    except web.HTTPBadRequest as exc:
        return exc

    session_id = str(body.get("session_id", "") or "").strip()
    if not session_id:
        return _error(
            400,
            "missing_session_id",
            "body field session_id is required",
        )

    status = body.get("status")
    if status not in _COMPLETION_STATUSES:
        return _error(
            400,
            "invalid_completion_status",
            "status must be 'succeeded' or 'failed'",
        )

    payload = body.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        return _error(400, "invalid_payload", "payload must be a JSON object")

    error_detail = str(body.get("error", "") or "")
    error_class = str(body.get("error_class", "") or "")
    if status == "failed":
        if not error_detail.strip():
            return _error(
                400,
                "missing_error",
                "status='failed' requires a non-empty 'error' string",
            )
        if not error_class:
            error_class = _DEFAULT_ERROR_CLASS
    else:
        if error_detail or error_class:
            return _error(
                400,
                "conflicting_completion",
                "status='succeeded' must not carry error/error_class",
            )

    latency_ms = body.get("latency_ms")
    if latency_ms is not None and not isinstance(latency_ms, int):
        return _error(400, "invalid_latency_ms", "latency_ms must be an integer")

    resolved = await _resolve_invoker(request, ai_id=ai_id, session_id=session_id)
    if isinstance(resolved, web.Response):
        return resolved
    invoker, session = resolved

    try:
        handle = invoker.get_task_handle(task_id)
    except KeyError:
        return _task_not_found(task_id)
    if handle.status.value not in _COMPLETABLE_TASK_STATUSES:
        return _error(
            409,
            "tool_task_already_terminal",
            (
                f"task_id={task_id!r} is already {handle.status.value}; "
                "deferred completion can only be submitted once."
            ),
        )

    result = invoker.submit_deferred_result(
        task_id=task_id,
        session=session,
        payload=dict(payload) if payload is not None else None,
        error_class=error_class,
        error_detail=error_detail,
        latency_ms=latency_ms,
    )
    completed = invoker.get_task_handle(task_id)
    _LOG.info(
        "tool task completed: ai_id=%s task_id=%s status=%s tool_events=%d",
        ai_id,
        task_id,
        completed.status.value,
        len(result.tool_event_ids),
    )
    return web.json_response(
        {
            **_handle_to_json(completed),
            "result": {
                "status": result.status.value,
                "tool_event_ids": list(result.tool_event_ids),
                "kernel_summary_truncated": result.kernel_summary_truncated,
            },
        }
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _resolve_invoker(
    request: web.Request,
    *,
    ai_id: str,
    session_id: str,
) -> tuple[Any, Any] | web.Response:
    """Resolve ``(invoker, session)`` for the addressed instance session.

    Mirrors ``app._resolve_session_manager`` resolution order, then reads
    the session's ``mcp_invoker``. The invoker is documented optional
    session behaviour (``None`` when no affordance bridge is wired) —
    absence is a typed 501, matching dispatch's
    ``tool_invoker_unavailable`` error.
    """
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if isinstance(launcher, LauncherProtocol):
        try:
            manager: SessionManager = launcher.get(ai_id)
        except InstanceNotFound:
            return _error(
                404,
                "ai_id_not_found",
                f"ai_id={ai_id!r} is not adopted on this server.",
            )
    else:
        manager = request.app["session_manager"]
    try:
        session = await manager.get_session(session_id)
    except SessionNotFoundError as exc:
        return _error(404, "session_not_found", str(exc))
    invoker = getattr(session, "mcp_invoker", None)
    if invoker is None:
        return _error(
            501,
            "tool_invoker_unavailable",
            "session does not expose an affordance invoker",
        )
    return invoker, session


def _handle_to_json(handle: Any) -> dict[str, Any]:
    """Project an ``AffordanceTaskHandle`` onto the wire shape.

    ``status`` is the task lifecycle status (``queued`` / ``running`` /
    ``succeeded`` / ``failed`` / ``cancelled``), per the spec's Tool Task
    Lifecycle section.
    """
    return {
        "task_id": handle.task_id,
        "descriptor_name": handle.descriptor_name,
        "status": handle.status.value,
        "poll_after_ms": handle.poll_after_ms,
        "plan_ref": handle.plan_ref,
    }


def _task_not_found(task_id: str) -> web.Response:
    return _error(
        404,
        "tool_task_not_found",
        f"no affordance task registered with task_id={task_id!r}",
    )


async def _read_json(request: web.Request) -> Mapping[str, Any]:
    if not request.body_exists:
        raise _bad_request("missing_body", "Body required")
    text = await request.text()
    if not text.strip():
        raise _bad_request("missing_body", "Empty body")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_request("invalid_json", f"Body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _bad_request("invalid_envelope", "Top-level body must be a JSON object")
    return data


def _bad_request(code: str, detail: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = ["attach_tool_task_routes"]
