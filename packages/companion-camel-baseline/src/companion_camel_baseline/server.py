# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""aiohttp server exposing the CAMEL baseline as an OpenAI-compatible endpoint.

Routes:

* ``POST /v1/chat/completions`` — the only route CompanionBench strictly
  requires. Each request flows through session-boundary handling (compact any
  stale sessions of this user), then the backend produces a reply given the
  prior cross-session memory + in-session transcript, and the upstream-shaped
  body is re-emitted verbatim.
* ``POST /v1/sessions/{session_id}/close`` — explicit session boundary; the
  server compacts that session's transcript into a durable memory record.
* ``GET /healthz`` — boot-time liveness probe.

Response contract (identical guarantees to companion-ref-harness):

* The body is exactly an OpenAI ``chat.completion`` envelope. No keys added.
* No ``x-camel-*`` / ``x-volvence-*`` / ``x-companionbench-*`` headers. The
  endpoint is shape-indistinguishable from a raw OpenAI-compat endpoint.
* HTTP error mapping (fail-loud, no swallowing):
  - request body invalid JSON / shape -> 400 ``invalid_body`` / ``invalid_request``
  - backend / upstream failure -> 502 ``camel_baseline_internal``

Session boundary detection:

1. Explicit close: ``POST /v1/sessions/{id}/close``.
2. Lazy on new session: when a request arrives for ``user_id == U`` with a new
   ``session_id``, every other seen session of ``U`` that has no stored memory
   record yet gets compacted before the new session's reply is produced.

In-flight session transcripts live in process memory only; a restart drops
in-progress sessions but not already-compacted records (those live in SQLite).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import re
import time
import uuid
from typing import Any

from aiohttp import web

from companion_camel_baseline.backend import AgentReply, CamelBackend, CamelBackendError
from companion_camel_baseline.memory_store import MemoryStore, SessionMemoryRecord


_LOG = logging.getLogger("companion_camel_baseline.server")

_OPENAI_CHAT_ROUTE: str = "/v1/chat/completions"
_SESSION_CLOSE_ROUTE: str = "/v1/sessions/{session_id}/close"
_HEALTH_ROUTE: str = "/healthz"

_APP_KEY: str = "companion_camel_baseline_app"

# CompanionBench session_id convention: "{arc_id}-s{idx}". The arc id is the
# cross-session relationship key when no explicit user_id is sent.
_ARC_SESSION_RE: re.Pattern[str] = re.compile(r"^(.*)-s\d+$")


# ---------------------------------------------------------------------------
# In-flight transcript buffer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _InflightSession:
    scope_key: str
    session_id: str
    messages: list[dict[str, str]] = dataclasses.field(default_factory=list)
    last_touched_monotonic: float = dataclasses.field(default_factory=time.monotonic)

    def append_pair(self, *, user: str, assistant: str) -> None:
        self.messages.append({"role": "user", "content": user})
        self.messages.append({"role": "assistant", "content": assistant})
        self.last_touched_monotonic = time.monotonic()


# ---------------------------------------------------------------------------
# BaselineApp
# ---------------------------------------------------------------------------


class BaselineApp:
    """Bundles backend / store / default system prompt for one server."""

    def __init__(
        self,
        *,
        backend: CamelBackend,
        store: MemoryStore,
        default_system_prompt: str = "You are a long-running companion AI.",
    ) -> None:
        self._backend = backend
        self._store = store
        self._default_system_prompt = default_system_prompt
        self._inflight: dict[tuple[str, str], _InflightSession] = {}
        self._seen_sessions_by_scope: dict[str, set[str]] = {}
        self._compact_lock = asyncio.Lock()

    @property
    def backend(self) -> CamelBackend:
        return self._backend

    @property
    def store(self) -> MemoryStore:
        return self._store

    # ---- pure helpers ------------------------------------------------------

    @staticmethod
    def derive_scope_key(
        *,
        metadata_user_id: str | None,
        header_user_id: str | None,
        request_headers: dict[str, str],
        session_id: str | None = None,
    ) -> str:
        """SSOT for scope-key derivation.

        Order: metadata.user_id > X-Companion-User-Id header > arc scope parsed
        from the ``{arc_id}-s{idx}`` session_id convention (CompanionBench sends
        no user_id; one arc = one relationship, so sessions of an arc share
        memory while arcs stay isolated) > per-session fallback > header
        surrogate (only when there is no session_id at all).
        """

        if metadata_user_id and metadata_user_id.strip():
            return metadata_user_id.strip()
        if header_user_id and header_user_id.strip():
            return header_user_id.strip()
        if session_id and session_id.strip():
            sid = session_id.strip()
            match = _ARC_SESSION_RE.match(sid)
            if match:
                return f"arc:{match.group(1)}"
            return f"session:{sid}"
        anchor = "|".join(
            request_headers.get(h, "") for h in ("User-Agent", "Authorization")
        )
        digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()[:16]
        return f"anon:{digest}"

    # ---- memory lifecycle --------------------------------------------------

    async def close_session(
        self, *, scope_key: str, session_id: str,
    ) -> SessionMemoryRecord | None:
        """Compact ``(scope_key, session_id)`` into a durable record (idempotent)."""

        async with self._compact_lock:
            existing = self._store.session_memory_get(
                scope_key=scope_key, session_id=session_id,
            )
            if existing is not None:
                self._inflight.pop((scope_key, session_id), None)
                return existing
            inflight = self._inflight.pop((scope_key, session_id), None)
            if inflight is None or not inflight.messages:
                return None
            record = await self._backend.compact(
                scope_key=scope_key,
                session_id=session_id,
                transcript=list(inflight.messages),
            )
            self._store.session_memory_put(record)
            return record

    async def _maybe_close_stale_sessions(
        self, *, scope_key: str, new_session_id: str,
    ) -> None:
        seen = self._seen_sessions_by_scope.setdefault(scope_key, set())
        seen.add(new_session_id)
        stale: list[str] = []
        for sid in seen:
            if sid == new_session_id:
                continue
            if self._store.session_memory_get(
                scope_key=scope_key, session_id=sid,
            ) is not None:
                continue
            stale.append(sid)
        for sid in stale:
            try:
                await self.close_session(scope_key=scope_key, session_id=sid)
            except CamelBackendError as exc:
                # A compaction failure on a stale session must not block the new
                # session's turn. Log loud + continue; the record stays missing
                # and will retry on the next request from the same user.
                _LOG.warning(
                    "stale session compaction failed: scope=%s session=%s err=%s",
                    scope_key, sid, exc,
                )

    # ---- per-turn pipeline -------------------------------------------------

    async def handle_chat_turn(
        self,
        *,
        scope_key: str,
        session_id: str,
        request_messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        await self._maybe_close_stale_sessions(
            scope_key=scope_key, new_session_id=session_id,
        )
        prior_memory = self._store.session_memory_list_for_scope(
            scope_key=scope_key, exclude_session_id=session_id, limit=None,
        )
        inflight = self._get_inflight(scope_key=scope_key, session_id=session_id)
        # The session messages handed to the backend are the in-flight buffer
        # (prior turns of this session) plus the latest incoming user turn.
        system_prompt = _extract_system_prompt(request_messages) or self._default_system_prompt
        latest_user = next(
            (m["content"] for m in reversed(request_messages) if m.get("role") == "user"),
            "",
        )
        session_messages = list(inflight.messages)
        if latest_user:
            session_messages.append({"role": "user", "content": latest_user})
        reply: AgentReply = await self._backend.respond(
            scope_key=scope_key,
            session_id=session_id,
            system_prompt=system_prompt,
            prior_memory=prior_memory,
            session_messages=session_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if latest_user:
            inflight.append_pair(user=latest_user, assistant=reply.text)
        return reply.raw

    def _get_inflight(self, *, scope_key: str, session_id: str) -> _InflightSession:
        key = (scope_key, session_id)
        existing = self._inflight.get(key)
        if existing is not None:
            return existing
        fresh = _InflightSession(scope_key=scope_key, session_id=session_id)
        self._inflight[key] = fresh
        self._seen_sessions_by_scope.setdefault(scope_key, set()).add(session_id)
        return fresh


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_app(
    *,
    backend: CamelBackend,
    store: MemoryStore,
    default_system_prompt: str = "You are a long-running companion AI.",
) -> web.Application:
    """Construct the aiohttp web.Application bound to a BaselineApp."""

    baseline = BaselineApp(
        backend=backend,
        store=store,
        default_system_prompt=default_system_prompt,
    )
    app = web.Application()
    app[_APP_KEY] = baseline
    app.router.add_post(_OPENAI_CHAT_ROUTE, _handle_chat_completions)
    app.router.add_post(_SESSION_CLOSE_ROUTE, _handle_session_close)
    app.router.add_get(_HEALTH_ROUTE, _handle_healthz)
    app.on_cleanup.append(_on_cleanup)
    return app


def get_baseline_app(app: web.Application) -> BaselineApp:
    baseline = app.get(_APP_KEY)
    if not isinstance(baseline, BaselineApp):
        raise RuntimeError(
            "BaselineApp is not attached to this aiohttp Application. "
            "Use build_app() to construct the server."
        )
    return baseline


async def _on_cleanup(app: web.Application) -> None:
    baseline = get_baseline_app(app)
    try:
        await baseline.backend.close()
    finally:
        baseline.store.close()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_chat_completions(request: web.Request) -> web.Response:
    payload = await _read_json_body(request)
    if payload is None:
        return _error(
            status=400,
            code="invalid_body",
            message="POST body must be a non-empty JSON object",
        )
    try:
        parsed = _ChatRequest.from_payload(payload)
    except ValueError as exc:
        return _error(status=400, code="invalid_request", message=str(exc))

    baseline = get_baseline_app(request.app)
    request_headers = {k: v for k, v in request.headers.items()}
    scope_key = baseline.derive_scope_key(
        metadata_user_id=parsed.user_id,
        header_user_id=request_headers.get("X-Companion-User-Id"),
        request_headers=request_headers,
        session_id=parsed.session_id,
    )
    try:
        body = await baseline.handle_chat_turn(
            scope_key=scope_key,
            session_id=parsed.session_id,
            request_messages=parsed.messages,
            max_tokens=parsed.max_tokens,
            temperature=parsed.temperature,
        )
    except CamelBackendError as exc:
        return _error(
            status=502,
            code="camel_baseline_internal",
            message=str(exc),
        )
    return web.json_response(body, status=200)


async def _handle_session_close(request: web.Request) -> web.Response:
    payload = await _read_json_body(request)
    if payload is None:
        payload = {}
    session_id = request.match_info["session_id"]
    request_headers = {k: v for k, v in request.headers.items()}
    metadata_user_id = None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict):
        metadata_user_id = metadata.get("user_id")
    baseline = get_baseline_app(request.app)
    scope_key = baseline.derive_scope_key(
        metadata_user_id=metadata_user_id,
        header_user_id=request_headers.get("X-Companion-User-Id"),
        request_headers=request_headers,
        session_id=session_id,
    )
    try:
        record = await baseline.close_session(
            scope_key=scope_key, session_id=session_id,
        )
    except CamelBackendError as exc:
        return _error(
            status=502,
            code="camel_baseline_internal",
            message=str(exc),
        )
    if record is None:
        return web.json_response({"closed": True, "record": None}, status=200)
    return web.json_response(
        {"closed": True, "record": record.to_payload()}, status=200,
    )


async def _handle_healthz(request: web.Request) -> web.Response:
    baseline = get_baseline_app(request.app)
    return web.json_response(
        {"ok": True, "backend_model": baseline.backend.model},
        status=200,
    )


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ChatRequest:
    model: str
    messages: list[dict[str, str]]
    session_id: str
    user_id: str | None
    max_tokens: int | None
    temperature: float | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "_ChatRequest":
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object, got {type(payload).__name__}")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("'model' must be a non-empty string")
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("'messages' must be a non-empty list")
        messages: list[dict[str, str]] = []
        for i, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                raise ValueError(
                    f"messages[{i}] must be an object, got {type(msg).__name__}"
                )
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("system", "user", "assistant"):
                raise ValueError(
                    f"messages[{i}].role must be system/user/assistant; got {role!r}"
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"messages[{i}].content must be a string; got {type(content).__name__}"
                )
            messages.append({"role": role, "content": content})
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"'metadata' must be an object if present; got {type(metadata).__name__}"
            )
        session_id = str(metadata.get("session_id") or "").strip()
        if not session_id:
            session_id = f"auto-{uuid.uuid4().hex[:16]}"
        user_id_raw = metadata.get("user_id")
        user_id = str(user_id_raw).strip() if user_id_raw else None
        max_tokens = _parse_optional_int(payload.get("max_tokens"), "max_tokens")
        temperature = _parse_optional_float(payload.get("temperature"), "temperature")
        return cls(
            model=model.strip(),
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _extract_system_prompt(messages: list[dict[str, str]]) -> str:
    parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    return "\n\n".join(parts).strip()


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name!r} must be int, got bool")
    if isinstance(value, int):
        return value
    raise ValueError(f"{field_name!r} must be int or null; got {type(value).__name__}")


def _parse_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name!r} must be float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{field_name!r} must be number or null; got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _read_json_body(request: web.Request) -> dict[str, Any] | None:
    if request.body_exists is False:
        return None
    try:
        payload = await request.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _error(*, status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message}}, status=status,
    )
