"""aiohttp route handlers for protocol uptake (PDF / MD / desc / API).

Mounted by ``create_app`` when a :class:`ProtocolUptakeService` is
injected. Routes:

* ``GET    /v1/protocols``                    list active protocols (in-memory registry)
* ``GET    /v1/protocols/candidates``         list pending candidates
* ``POST   /v1/protocols/upload-pdf``         multipart PDF → candidate
* ``POST   /v1/protocols/upload-markdown``    multipart MD → candidate
* ``POST   /v1/protocols/from-description``   {description, protocol_id, advisor_name} → candidate
* ``POST   /v1/protocols/from-payload``       full BehaviorProtocol payload → candidate
* ``POST   /v1/protocols/candidates/{id}/approve``  approve → load + persist to library
* ``POST   /v1/protocols/candidates/{id}/reject``   reject
* ``DELETE /v1/protocols/{id}``               unload approved from active set (keeps library file)

Library routes (only when ``--protocol-approved-dir`` is set):

* ``GET    /v1/protocols/library``            list disk-persisted protocols + active flag
* ``POST   /v1/protocols/library/{id}/load``  read JSON → load into active set
* ``POST   /v1/protocols/library/{id}/unload`` remove from active set (file stays)
* ``DELETE /v1/protocols/library/{id}``       remove JSON + unload

When the service is started without ``--protocol-approved-dir``,
library routes respond ``503 protocol_library_not_configured``.

All extraction routes that need an LLM respond ``503 protocol_llm_not_configured``
when the service has no client. ``from-payload`` works without an LLM.

Auth: routes mirror the existing service auth model. When alpha
mode is on, the alpha user header is required and forwarded into
audit notes. No alpha mode → no auth (intended for local dev /
single-tenant).
"""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

from lifeform_service.alpha import AlphaServiceConfig
from lifeform_service.protocol_uptake import (
    ProtocolUptakeService,
    candidate_to_json,
    protocol_to_json,
)


_APP_KEY = "protocol_uptake_service"


def register_protocol_routes(
    app: web.Application,
    *,
    uptake_service: ProtocolUptakeService,
) -> None:
    """Attach the uptake service to the app and register routes."""

    app[_APP_KEY] = uptake_service
    app.router.add_get("/v1/protocols", _handle_list_protocols)
    app.router.add_get("/v1/protocols/candidates", _handle_list_candidates)
    app.router.add_post(
        "/v1/protocols/upload-pdf", _handle_upload_pdf
    )
    app.router.add_post(
        "/v1/protocols/upload-markdown", _handle_upload_markdown
    )
    app.router.add_post(
        "/v1/protocols/from-description", _handle_from_description
    )
    app.router.add_post(
        "/v1/protocols/from-payload", _handle_from_payload
    )
    app.router.add_post(
        "/v1/protocols/candidates/{protocol_id}/approve",
        _handle_approve,
    )
    app.router.add_post(
        "/v1/protocols/candidates/{protocol_id}/reject",
        _handle_reject,
    )
    # Library (disk-backed) routes — handlers themselves return
    # ``503 protocol_library_not_configured`` when no persistence
    # store is wired, so registration is unconditional.
    app.router.add_get(
        "/v1/protocols/library", _handle_list_library
    )
    app.router.add_post(
        "/v1/protocols/library/{protocol_id}/load",
        _handle_library_load,
    )
    app.router.add_post(
        "/v1/protocols/library/{protocol_id}/unload",
        _handle_library_unload,
    )
    app.router.add_delete(
        "/v1/protocols/library/{protocol_id}",
        _handle_library_delete,
    )
    # NOTE: the legacy "/v1/protocols/{id}" DELETE route is
    # registered AFTER the library routes so the more specific
    # path ``/v1/protocols/library/{id}`` matches first.
    app.router.add_delete(
        "/v1/protocols/{protocol_id}", _handle_unload
    )


def _service(request: web.Request) -> ProtocolUptakeService:
    return request.app[_APP_KEY]


def _reviewer_id(request: web.Request) -> str:
    """Best-effort reviewer attribution."""
    alpha: AlphaServiceConfig | None = request.app.get("alpha_config")
    if alpha is not None and alpha.enabled:
        header = request.headers.get("X-Alpha-User", "").strip()
        if header:
            return header
    return "anonymous"


async def _handle_list_protocols(request: web.Request) -> web.Response:
    service = _service(request)
    protocols = await service.list_approved()
    return web.json_response(
        {
            "protocols": [protocol_to_json(p) for p in protocols],
            "count": len(protocols),
        }
    )


async def _handle_list_candidates(request: web.Request) -> web.Response:
    service = _service(request)
    pending = await service.list_pending()
    return web.json_response(
        {
            "candidates": [candidate_to_json(e) for e in pending],
            "count": len(pending),
        }
    )


async def _handle_upload_pdf(request: web.Request) -> web.Response:
    service = _service(request)
    if service.llm_client is None:
        return _llm_not_configured()

    reader = await request.multipart()
    pdf_bytes: bytes | None = None
    filename: str = "uploaded.pdf"
    protocol_id_seed: str | None = None

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            if part.filename:
                filename = part.filename
            pdf_bytes = await part.read(decode=False)
        elif part.name == "protocol_id_seed":
            value_bytes = await part.read(decode=False)
            protocol_id_seed = value_bytes.decode("utf-8", errors="ignore").strip() or None

    if not pdf_bytes:
        return web.json_response(
            {"error": "missing_file", "detail": "multipart 'file' field is required"},
            status=400,
        )

    try:
        candidate = await service.extract_from_pdf_bytes(
            pdf_bytes,
            filename=filename,
            protocol_id_seed=protocol_id_seed,
        )
    except (ValueError, RuntimeError) as exc:
        return web.json_response(
            {"error": "extraction_failed", "detail": str(exc)}, status=422
        )
    pid = await service.submit_candidate(
        candidate, note=f"upload-pdf:{filename}"
    )
    return web.json_response(
        {
            "submitted": True,
            "protocol_id": pid,
            "candidate": candidate_to_json(
                (await service.list_pending())[-1]
            ),
        },
        status=201,
    )


async def _handle_upload_markdown(request: web.Request) -> web.Response:
    service = _service(request)
    if service.llm_client is None:
        return _llm_not_configured()

    reader = await request.multipart()
    text: str | None = None
    filename: str = "uploaded.md"
    protocol_id_seed: str | None = None

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            if part.filename:
                filename = part.filename
            raw = await part.read(decode=False)
            text = raw.decode("utf-8", errors="replace")
        elif part.name == "protocol_id_seed":
            value_bytes = await part.read(decode=False)
            protocol_id_seed = value_bytes.decode("utf-8", errors="ignore").strip() or None

    if not text or not text.strip():
        return web.json_response(
            {"error": "missing_file", "detail": "multipart 'file' field is required"},
            status=400,
        )
    try:
        candidate = await service.extract_from_markdown_text(
            text,
            source_label=filename,
            protocol_id_seed=protocol_id_seed,
        )
    except (ValueError, RuntimeError) as exc:
        return web.json_response(
            {"error": "extraction_failed", "detail": str(exc)}, status=422
        )
    pid = await service.submit_candidate(
        candidate, note=f"upload-markdown:{filename}"
    )
    return web.json_response(
        {"submitted": True, "protocol_id": pid}, status=201
    )


async def _handle_from_description(request: web.Request) -> web.Response:
    service = _service(request)
    if service.llm_client is None:
        return _llm_not_configured()
    payload = await _require_json(request)
    description = (payload.get("description") or "").strip()
    protocol_id = (payload.get("protocol_id") or "").strip()
    advisor_name = (payload.get("advisor_name") or "").strip()
    if not description or not protocol_id or not advisor_name:
        return web.json_response(
            {
                "error": "missing_fields",
                "detail": "description, protocol_id, advisor_name are required",
            },
            status=400,
        )
    try:
        candidate = await service.extract_from_description(
            description, protocol_id=protocol_id, advisor_name=advisor_name
        )
    except (ValueError, RuntimeError) as exc:
        return web.json_response(
            {"error": "extraction_failed", "detail": str(exc)}, status=422
        )
    pid = await service.submit_candidate(
        candidate, note=f"from-description:{protocol_id}"
    )
    return web.json_response(
        {"submitted": True, "protocol_id": pid}, status=201
    )


async def _handle_from_payload(request: web.Request) -> web.Response:
    service = _service(request)
    payload = await _require_json(request)
    request_id = (payload.get("request_id") or "").strip()
    spec = payload.get("protocol")
    if not request_id or not isinstance(spec, dict):
        return web.json_response(
            {
                "error": "missing_fields",
                "detail": "'request_id' and 'protocol' (object) are required",
            },
            status=400,
        )
    try:
        candidate = await service.inject_from_payload(
            spec, request_id=request_id
        )
    except (ValueError, RuntimeError) as exc:
        return web.json_response(
            {"error": "injection_failed", "detail": str(exc)}, status=422
        )
    pid = await service.submit_candidate(
        candidate, note=f"from-payload:{request_id}"
    )
    return web.json_response(
        {"submitted": True, "protocol_id": pid}, status=201
    )


async def _handle_approve(request: web.Request) -> web.Response:
    service = _service(request)
    protocol_id = request.match_info["protocol_id"]
    try:
        approved = await service.approve_pending(
            protocol_id, reviewer_id=_reviewer_id(request)
        )
    except KeyError as exc:
        return web.json_response(
            {"error": "candidate_not_found", "detail": str(exc)}, status=404
        )
    except ValueError as exc:
        return web.json_response(
            {"error": "invalid_candidate", "detail": str(exc)}, status=400
        )
    return web.json_response(
        {
            "approved": True,
            "protocol": protocol_to_json(approved),
        }
    )


async def _handle_reject(request: web.Request) -> web.Response:
    service = _service(request)
    protocol_id = request.match_info["protocol_id"]
    payload = await _maybe_json(request) or {}
    reason = ""
    if isinstance(payload, dict):
        raw = payload.get("reason", "")
        if isinstance(raw, str):
            reason = raw.strip()
    try:
        await service.reject_pending(
            protocol_id,
            reviewer_id=_reviewer_id(request),
            reason=reason or "rejected via API",
        )
    except KeyError as exc:
        return web.json_response(
            {"error": "candidate_not_found", "detail": str(exc)}, status=404
        )
    return web.json_response({"rejected": True, "protocol_id": protocol_id})


async def _handle_unload(request: web.Request) -> web.Response:
    service = _service(request)
    protocol_id = request.match_info["protocol_id"]
    removed = await service.unload_protocol(protocol_id)
    if not removed:
        return web.json_response(
            {"error": "protocol_not_loaded", "detail": protocol_id}, status=404
        )
    return web.json_response({"unloaded": True, "protocol_id": protocol_id})


# ---------------------------------------------------------------------------
# Library (disk-backed) routes
# ---------------------------------------------------------------------------


async def _handle_list_library(request: web.Request) -> web.Response:
    service = _service(request)
    if service.persistence is None:
        return _library_not_configured()
    entries = service.library_state_snapshot()
    return web.json_response(
        {
            "approved_dir": str(service.persistence.approved_dir),
            "count": len(entries),
            "entries": [
                {
                    **protocol_to_json(p),
                    "is_active": active,
                }
                for p, active in entries
            ],
        }
    )


async def _handle_library_load(request: web.Request) -> web.Response:
    service = _service(request)
    if service.persistence is None:
        return _library_not_configured()
    protocol_id = request.match_info["protocol_id"]
    try:
        loaded = await service.load_from_library(protocol_id)
    except KeyError as exc:
        return web.json_response(
            {"error": "library_entry_not_found", "detail": str(exc)},
            status=404,
        )
    except ValueError as exc:
        return web.json_response(
            {"error": "library_entry_invalid", "detail": str(exc)},
            status=422,
        )
    return web.json_response(
        {
            "loaded": True,
            "protocol_id": protocol_id,
            "protocol": protocol_to_json(loaded),
        }
    )


async def _handle_library_unload(request: web.Request) -> web.Response:
    service = _service(request)
    if service.persistence is None:
        return _library_not_configured()
    protocol_id = request.match_info["protocol_id"]
    removed = await service.unload_from_registry(protocol_id)
    return web.json_response(
        {
            "unloaded": removed,
            "protocol_id": protocol_id,
            "detail": (
                "removed from active set; file remains on disk"
                if removed
                else "no active entry for this protocol_id; file (if any) remains on disk"
            ),
        }
    )


async def _handle_library_delete(request: web.Request) -> web.Response:
    service = _service(request)
    if service.persistence is None:
        return _library_not_configured()
    protocol_id = request.match_info["protocol_id"]
    removed = await service.delete_from_library(protocol_id)
    if not removed:
        return web.json_response(
            {
                "error": "library_entry_not_found",
                "detail": f"no persisted protocol with id {protocol_id!r}",
            },
            status=404,
        )
    return web.json_response(
        {"deleted": True, "protocol_id": protocol_id}
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_not_configured() -> web.Response:
    return web.json_response(
        {
            "error": "protocol_llm_not_configured",
            "detail": (
                "this route requires an OpenAI-compatible LLM. Set "
                "PROTOCOL_LLM_BASE_URL and PROTOCOL_LLM_API_KEY (and "
                "optionally PROTOCOL_LLM_MODEL / "
                "PROTOCOL_LLM_TIMEOUT_SECONDS) and restart the service."
            ),
        },
        status=503,
    )


def _library_not_configured() -> web.Response:
    return web.json_response(
        {
            "error": "protocol_library_not_configured",
            "detail": (
                "this route requires a disk-backed protocol library. "
                "Start the service with --protocol-approved-dir <path> "
                "and restart."
            ),
        },
        status=503,
    )


async def _maybe_json(request: web.Request) -> Any:
    if not request.body_exists:
        return None
    try:
        text = await request.text()
    except Exception as exc:
        raise web.HTTPBadRequest(text=f"could not read body: {exc}") from exc
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc


async def _require_json(request: web.Request) -> dict[str, Any]:
    body = await _maybe_json(request)
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="expected JSON object body")
    return body


__all__ = ["register_protocol_routes"]
