# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""aiohttp server exposing the harness as an OpenAI-compatible endpoint.

The server attaches three routes:

* ``POST /v1/chat/completions`` — the only route CompanionBench
  strictly requires. Each request flows through
  :class:`HarnessPolicy` (which may splice in component prefixes),
  then is forwarded to the configured :class:`UpstreamClient`, and
  the response is re-emitted verbatim in the standard
  ``chat.completion`` envelope.
* ``POST /v1/sessions/{session_id}/close`` — clients may explicitly
  announce a session boundary; the server triggers a summary
  extraction for that session.
* ``GET /healthz`` — boot-time liveness probe.

Response contract:

* The response body is exactly the upstream's ``chat.completion``
  body (translated for non-OpenAI families). No keys are added or
  rewritten.
* No ``x-ref-harness-*`` / ``x-lifeform-*`` / ``x-volvence-*`` /
  ``x-companionbench-*`` headers are emitted. The harness is
  shape-indistinguishable from a raw OpenAI-compat endpoint from
  the CompanionBench runner's perspective.
* HTTP error mapping (fail-loud, no swallowing):
  - request body invalid JSON / shape -> 400 ``invalid_body``
  - upstream HTTP failure -> 502 ``ref_harness_upstream_error``
  - harness-internal failure (store / extractor) -> 502
    ``ref_harness_internal``

Session boundary detection (H-A scope):

1. Explicit close: ``POST /v1/sessions/{id}/close``.
2. Lazy on new session: when a request arrives for ``user_id == U``
   with ``session_id == S2``, every other pending session of ``U``
   (any session for which a summary does not yet exist in the
   store) gets its summary extracted before the new session's
   prompt is built.

In-flight session transcripts are held in process memory only.
This is intentional: a server restart drops in-progress sessions
but does NOT drop already-extracted summaries (those live in
SQLite). Mid-arc crashes therefore degrade gracefully — the
benchmark runner restarts the arc and the previously-summarised
sessions are still available.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import time
import uuid
from typing import Any

from aiohttp import web

from companion_ref_harness.policy import (
    ComponentSet,
    HarnessComponent,
    HarnessPolicy,
)
from companion_ref_harness.session_summary import (
    SessionSummary,
    SummaryExtractor,
)
from companion_ref_harness.store.sqlite_store import HarnessStore
from companion_ref_harness.upstream_client import (
    UpstreamClient,
    UpstreamError,
)


_LOG = logging.getLogger("companion_ref_harness.server")

_OPENAI_CHAT_ROUTE: str = "/v1/chat/completions"
_SESSION_CLOSE_ROUTE: str = "/v1/sessions/{session_id}/close"
_HEALTH_ROUTE: str = "/healthz"

# Key under which we stash the HarnessApp on the aiohttp.web.Application.
_APP_KEY: str = "companion_ref_harness_app"


# ---------------------------------------------------------------------------
# In-flight transcript buffer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _InflightSession:
    """Per-session in-process transcript buffer.

    Mutated as turns flow in; never persisted. When the session
    closes (explicit / lazy), the buffer is handed to the summary
    extractor and the result is persisted via the store.
    """

    scope_key: str
    session_id: str
    messages: list[dict[str, str]] = dataclasses.field(default_factory=list)
    last_touched_monotonic: float = dataclasses.field(default_factory=time.monotonic)

    def append_pair(self, *, user: str, assistant: str) -> None:
        self.messages.append({"role": "user", "content": user})
        self.messages.append({"role": "assistant", "content": assistant})
        self.last_touched_monotonic = time.monotonic()


# ---------------------------------------------------------------------------
# HarnessApp — the runtime container for one server process
# ---------------------------------------------------------------------------


class HarnessApp:
    """Bundles upstream / store / policy / extractor for one server.

    The aiohttp ``web.Application`` keeps a reference to this object
    so handlers can pull the runtime state without globals.
    """

    def __init__(
        self,
        *,
        upstream: UpstreamClient,
        store: HarnessStore,
        policy: HarnessPolicy,
        summary_extractor: SummaryExtractor,
    ) -> None:
        self._upstream = upstream
        self._store = store
        self._policy = policy
        self._summary_extractor = summary_extractor
        # (scope_key, session_id) -> _InflightSession
        self._inflight: dict[tuple[str, str], _InflightSession] = {}
        # Per-scope: which sessions we have seen at least once during
        # this process lifetime. Used for "lazy on new session"
        # summary extraction.
        self._seen_sessions_by_scope: dict[str, set[str]] = {}
        self._extract_lock = asyncio.Lock()

    @property
    def upstream(self) -> UpstreamClient:
        return self._upstream

    @property
    def store(self) -> HarnessStore:
        return self._store

    @property
    def policy(self) -> HarnessPolicy:
        return self._policy

    @property
    def summary_extractor(self) -> SummaryExtractor:
        return self._summary_extractor

    # ---- pure helpers ------------------------------------------------------

    @staticmethod
    def derive_scope_key(
        *,
        metadata_user_id: str | None,
        header_user_id: str | None,
        request_headers: dict[str, str],
    ) -> str:
        """SSOT for scope-key derivation. Used by the request handler.

        Order: metadata.user_id > X-Companion-User-Id header > stable
        surrogate derived from a small set of request headers. The
        surrogate is **only** for tests / sanity smoke — production
        CompanionBench runs always supply ``metadata.user_id``.
        """

        if metadata_user_id and metadata_user_id.strip():
            return metadata_user_id.strip()
        if header_user_id and header_user_id.strip():
            return header_user_id.strip()
        # Last-resort surrogate. Hash a stable subset of headers so
        # the same anonymous caller maps to the same scope across
        # requests within one server lifetime.
        anchor = "|".join(
            request_headers.get(h, "")
            for h in ("User-Agent", "Authorization")
        )
        digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()[:16]
        return f"anon:{digest}"

    # ---- summary lifecycle -------------------------------------------------

    async def close_session(
        self,
        *,
        scope_key: str,
        session_id: str,
    ) -> SessionSummary | None:
        """Force-extract a summary for ``(scope_key, session_id)``.

        Returns the extracted summary, or ``None`` if the session
        has no in-flight transcript (already closed / never
        active). Idempotent: a second call returns the previously
        stored summary.
        """

        if not self._policy.components.has(HarnessComponent.SUMMARY):
            return None

        async with self._extract_lock:
            existing = self._store.session_summary_get(
                scope_key=scope_key, session_id=session_id,
            )
            if existing is not None:
                self._inflight.pop((scope_key, session_id), None)
                return existing
            inflight = self._inflight.pop((scope_key, session_id), None)
            if inflight is None or not inflight.messages:
                return None
            summary = await self._summary_extractor.extract(
                scope_key=scope_key,
                session_id=session_id,
                transcript=list(inflight.messages),
            )
            self._store.session_summary_put(summary)
            return summary

    async def _maybe_close_stale_sessions(
        self,
        *,
        scope_key: str,
        new_session_id: str,
    ) -> None:
        """Trigger summary extraction for any older sessions of this scope.

        Called whenever a new session_id arrives for ``scope_key``.
        Iterates over the per-scope seen-sessions set and closes
        any session != ``new_session_id`` that does not yet have a
        stored summary.
        """

        if not self._policy.components.has(HarnessComponent.SUMMARY):
            return
        seen = self._seen_sessions_by_scope.setdefault(scope_key, set())
        seen.add(new_session_id)
        stale: list[str] = []
        for sid in seen:
            if sid == new_session_id:
                continue
            if self._store.session_summary_get(
                scope_key=scope_key, session_id=sid,
            ) is not None:
                continue
            stale.append(sid)
        for sid in stale:
            try:
                await self.close_session(scope_key=scope_key, session_id=sid)
            except Exception as exc:  # noqa: BLE001 -- intentional re-raise as warning
                # An extraction failure on a stale session must not block
                # the new session's turn. Log loud + continue. The
                # missing summary stays missing (it will retry on next
                # request from the same user).
                _LOG.warning(
                    "stale session summary extraction failed: "
                    "scope=%s session=%s err=%s",
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
        metadata_user_id: str | None,
    ) -> dict[str, Any]:
        """The per-turn pipeline: stale-close -> blend -> upstream -> buffer.

        Returns the upstream's OpenAI-compat response body verbatim.
        """

        await self._maybe_close_stale_sessions(
            scope_key=scope_key, new_session_id=session_id,
        )
        prior_summaries = self._store.session_summary_list_for_scope(
            scope_key=scope_key, exclude_session_id=session_id, limit=None,
        )
        blended = self._policy.blend(
            scope_key=scope_key,
            session_id=session_id,
            messages=request_messages,
            prior_summaries=prior_summaries,
        )
        upstream_resp = await self._upstream.chat(
            messages=blended.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
            user_id=metadata_user_id,
        )
        # Record the (user, assistant) pair into the in-flight buffer
        # for later summary extraction. We use the **un-blended**
        # original messages so the transcript stored locally reflects
        # what the user actually said, not what the harness spliced.
        latest_user = next(
            (m["content"] for m in reversed(request_messages) if m.get("role") == "user"),
            "",
        )
        if latest_user:
            self._get_inflight(scope_key=scope_key, session_id=session_id).append_pair(
                user=latest_user,
                assistant=upstream_resp.text,
            )
        return upstream_resp.raw

    def _get_inflight(
        self,
        *,
        scope_key: str,
        session_id: str,
    ) -> _InflightSession:
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
    upstream: UpstreamClient,
    store: HarnessStore,
    components: ComponentSet,
    summary_extractor: SummaryExtractor,
) -> web.Application:
    """Construct the aiohttp web.Application bound to a HarnessApp."""

    policy = HarnessPolicy(components)
    harness = HarnessApp(
        upstream=upstream,
        store=store,
        policy=policy,
        summary_extractor=summary_extractor,
    )
    app = web.Application()
    app[_APP_KEY] = harness
    app.router.add_post(_OPENAI_CHAT_ROUTE, _handle_chat_completions)
    app.router.add_post(_SESSION_CLOSE_ROUTE, _handle_session_close)
    app.router.add_get(_HEALTH_ROUTE, _handle_healthz)
    app.on_cleanup.append(_on_cleanup)
    return app


def get_harness_app(app: web.Application) -> HarnessApp:
    """Pull the :class:`HarnessApp` off the aiohttp Application."""

    harness = app.get(_APP_KEY)
    if not isinstance(harness, HarnessApp):
        raise RuntimeError(
            "HarnessApp is not attached to this aiohttp Application. "
            "Use build_app() to construct the server."
        )
    return harness


async def _on_cleanup(app: web.Application) -> None:
    harness = get_harness_app(app)
    try:
        await harness.upstream.close()
    finally:
        harness.store.close()


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

    harness = get_harness_app(request.app)

    request_headers = {k: v for k, v in request.headers.items()}
    scope_key = harness.derive_scope_key(
        metadata_user_id=parsed.user_id,
        header_user_id=request_headers.get("X-Companion-User-Id"),
        request_headers=request_headers,
    )

    try:
        body = await harness.handle_chat_turn(
            scope_key=scope_key,
            session_id=parsed.session_id,
            request_messages=parsed.messages,
            max_tokens=parsed.max_tokens,
            temperature=parsed.temperature,
            metadata_user_id=parsed.user_id,
        )
    except UpstreamError as exc:
        return _error(
            status=502,
            code="ref_harness_upstream_error",
            message=f"upstream HTTP {exc.status}: {exc.detail[:200]}",
        )
    except ValueError as exc:
        return _error(
            status=502,
            code="ref_harness_internal",
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
    harness = get_harness_app(request.app)
    scope_key = harness.derive_scope_key(
        metadata_user_id=metadata_user_id,
        header_user_id=request_headers.get("X-Companion-User-Id"),
        request_headers=request_headers,
    )
    try:
        summary = await harness.close_session(
            scope_key=scope_key, session_id=session_id,
        )
    except ValueError as exc:
        return _error(
            status=502,
            code="ref_harness_internal",
            message=str(exc),
        )
    if summary is None:
        return web.json_response(
            {"closed": True, "summary": None}, status=200,
        )
    return web.json_response(
        {"closed": True, "summary": summary.to_payload()}, status=200,
    )


async def _handle_healthz(request: web.Request) -> web.Response:
    harness = get_harness_app(request.app)
    return web.json_response(
        {
            "ok": True,
            "components": harness.policy.components.to_csv(),
            "upstream_family": harness.upstream.family.value,
            "upstream_model": harness.upstream.model,
        },
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
            raise ValueError(
                f"expected JSON object, got {type(payload).__name__}"
            )
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
            # Auto-mint a session_id if the caller didn't supply one.
            # Stable per request (UUID4); the absence of session_id is
            # legitimate for stateless smoke probes.
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
    """Emit a JSON error response in OpenAI's standard shape."""

    return web.json_response(
        {"error": {"code": code, "message": message}},
        status=status,
    )
