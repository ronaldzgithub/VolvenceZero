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
import datetime as _dt
import hashlib
import logging
import re
import time
import uuid
from typing import Any

from aiohttp import web

from companion_ref_harness.embed import EmbedEntry, Embedder, top_k
from companion_ref_harness.episodic import EpisodicExtractor
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
from companion_ref_harness.user_model import UserFactExtractor


_LOG = logging.getLogger("companion_ref_harness.server")

_OPENAI_CHAT_ROUTE: str = "/v1/chat/completions"
_SESSION_CLOSE_ROUTE: str = "/v1/sessions/{session_id}/close"
_HEALTH_ROUTE: str = "/healthz"

# Key under which we stash the HarnessApp on the aiohttp.web.Application.
_APP_KEY: str = "companion_ref_harness_app"

# CompanionBench session_id convention: "{arc_id}-s{idx}" (arc_runner). The arc
# id is the cross-session relationship key when no explicit user_id is sent.
_ARC_SESSION_RE: re.Pattern[str] = re.compile(r"^(.*)-s\d+$")


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
        embedder: Embedder | None = None,
        user_fact_extractor: UserFactExtractor | None = None,
        episodic_extractor: EpisodicExtractor | None = None,
    ) -> None:
        self._upstream = upstream
        self._store = store
        self._policy = policy
        self._summary_extractor = summary_extractor
        self._embedder = embedder
        self._user_fact_extractor = user_fact_extractor
        self._episodic_extractor = episodic_extractor
        # (scope_key, session_id) -> _InflightSession
        self._inflight: dict[tuple[str, str], _InflightSession] = {}
        # Per-scope: which sessions we have seen at least once during
        # this process lifetime. Used for "lazy on new session"
        # extraction.
        self._seen_sessions_by_scope: dict[str, set[str]] = {}
        # (scope_key, session_id) already compacted this process lifetime.
        self._compacted: set[tuple[str, str]] = set()
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

    def _has_session_close_component(self) -> bool:
        c = self._policy.components
        return (
            c.has(HarnessComponent.SUMMARY)
            or c.has(HarnessComponent.USER_MODEL)
            or c.has(HarnessComponent.EPISODIC)
        )

    # ---- pure helpers ------------------------------------------------------

    @staticmethod
    def derive_scope_key(
        *,
        metadata_user_id: str | None,
        header_user_id: str | None,
        request_headers: dict[str, str],
        session_id: str | None = None,
    ) -> str:
        """SSOT for scope-key derivation. Used by the request handler.

        Order:
        1. ``metadata.user_id`` — production / explicit identity.
        2. ``X-Companion-User-Id`` header.
        3. **Arc scope from the session_id convention.** CompanionBench sends
           no ``user_id`` and names sessions ``{arc_id}-s{idx}`` (one arc = one
           continuous relationship across sessions; different arcs = different
           users). Without this, every arc would collapse onto the same
           header surrogate and memory would bleed across unrelated arcs. So
           when no explicit identity is given we key on the arc id (the
           session_id with its ``-s<N>`` suffix stripped). This is a protocol
           convention parse, not a behavioral heuristic.
        4. Per-session fallback (isolation) when the session_id has no arc
           suffix, then a header surrogate only if there is no session_id at
           all (tests / sanity smoke).
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
        # Last-resort surrogate (no identity, no session_id): hash a stable
        # subset of headers so one anonymous caller maps to one scope.
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
        """Compact ``(scope_key, session_id)`` for every enabled close component.

        Runs the summary / user-model / episodic extractors (whichever are
        enabled) over the session transcript and persists their output.
        Returns the session summary (when the summary component is enabled),
        otherwise ``None``. Idempotent: a second call returns the already-stored
        summary and does not re-extract.
        """

        if not self._has_session_close_component():
            return None

        async with self._extract_lock:
            key = (scope_key, session_id)
            if key in self._compacted:
                self._inflight.pop(key, None)
                return self._store.session_summary_get(
                    scope_key=scope_key, session_id=session_id,
                )
            existing_summary = self._store.session_summary_get(
                scope_key=scope_key, session_id=session_id,
            )
            inflight = self._inflight.pop(key, None)
            if inflight is None or not inflight.messages:
                return existing_summary
            transcript = list(inflight.messages)
            components = self._policy.components

            summary: SessionSummary | None = existing_summary
            if components.has(HarnessComponent.SUMMARY) and existing_summary is None:
                summary = await self._summary_extractor.extract(
                    scope_key=scope_key,
                    session_id=session_id,
                    transcript=transcript,
                )
                self._store.session_summary_put(summary)
            if components.has(HarnessComponent.USER_MODEL) and self._user_fact_extractor is not None:
                facts = await self._user_fact_extractor.extract(
                    scope_key=scope_key,
                    session_id=session_id,
                    transcript=transcript,
                )
                for fact in facts:
                    self._store.user_fact_put(fact)
            if components.has(HarnessComponent.EPISODIC) and self._episodic_extractor is not None:
                events = await self._episodic_extractor.extract(
                    scope_key=scope_key,
                    session_id=session_id,
                    transcript=transcript,
                )
                for event in events:
                    self._store.episodic_put(event)
            self._compacted.add(key)
            return summary

    async def _maybe_close_stale_sessions(
        self,
        *,
        scope_key: str,
        new_session_id: str,
    ) -> None:
        """Compact any older sessions of this scope before the new turn.

        Called whenever a new session_id arrives for ``scope_key``. Iterates
        over the per-scope seen-sessions set and compacts any session !=
        ``new_session_id`` that has not yet been compacted this process
        lifetime.
        """

        if not self._has_session_close_component():
            return
        seen = self._seen_sessions_by_scope.setdefault(scope_key, set())
        seen.add(new_session_id)
        stale = [
            sid
            for sid in seen
            if sid != new_session_id and (scope_key, sid) not in self._compacted
        ]
        for sid in stale:
            try:
                await self.close_session(scope_key=scope_key, session_id=sid)
            except Exception as exc:  # noqa: BLE001 -- intentional re-raise as warning
                # An extraction failure on a stale session must not block the
                # new session's turn. Log loud + continue; the missing record
                # retries on the next request from the same user.
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
        metadata_user_id: str | None,
    ) -> dict[str, Any]:
        """The per-turn pipeline: stale-close -> blend -> upstream -> buffer.

        Returns the upstream's OpenAI-compat response body verbatim.
        """

        await self._maybe_close_stale_sessions(
            scope_key=scope_key, new_session_id=session_id,
        )
        components = self._policy.components
        latest_user = next(
            (m["content"] for m in reversed(request_messages) if m.get("role") == "user"),
            "",
        )

        prior_summaries: tuple[SessionSummary, ...] = ()
        if components.has(HarnessComponent.SUMMARY):
            prior_summaries = self._store.session_summary_list_for_scope(
                scope_key=scope_key, exclude_session_id=session_id, limit=None,
            )
        retrieved_turns: tuple[EmbedEntry, ...] = ()
        if components.has(HarnessComponent.EMBED) and self._embedder is not None and latest_user:
            query_vec = self._embedder.embed(latest_user)
            retrieved_turns = top_k(
                query=query_vec,
                entries=self._store.embed_index_list_for_scope(scope_key=scope_key),
            )
        user_facts = ()
        if components.has(HarnessComponent.USER_MODEL):
            user_facts = self._store.user_fact_list_for_scope(scope_key=scope_key)
        episodic_events = ()
        if components.has(HarnessComponent.EPISODIC):
            episodic_events = self._store.episodic_list_for_scope(
                scope_key=scope_key, limit=None,
            )

        blended = self._policy.blend(
            scope_key=scope_key,
            session_id=session_id,
            messages=request_messages,
            prior_summaries=prior_summaries,
            retrieved_turns=retrieved_turns,
            user_facts=user_facts,
            episodic_events=episodic_events,
        )
        upstream_resp = await self._upstream.chat(
            messages=blended.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
            user_id=metadata_user_id,
        )
        # Record the (user, assistant) pair into the in-flight buffer for later
        # extraction. We use the **un-blended** original user text so the local
        # transcript reflects what the user actually said, not the spliced
        # prompt.
        if latest_user:
            inflight = self._get_inflight(scope_key=scope_key, session_id=session_id)
            pair_idx = len(inflight.messages) // 2
            inflight.append_pair(user=latest_user, assistant=upstream_resp.text)
            if components.has(HarnessComponent.EMBED) and self._embedder is not None:
                self._index_turn(
                    scope_key=scope_key,
                    turn_id=f"{session_id}:u{pair_idx}",
                    role="user",
                    content=latest_user,
                )
                if upstream_resp.text:
                    self._index_turn(
                        scope_key=scope_key,
                        turn_id=f"{session_id}:a{pair_idx}",
                        role="assistant",
                        content=upstream_resp.text,
                    )
        return upstream_resp.raw

    def _index_turn(
        self, *, scope_key: str, turn_id: str, role: str, content: str,
    ) -> None:
        assert self._embedder is not None  # guarded by caller
        self._store.embed_index_put(
            EmbedEntry(
                scope_key=scope_key,
                turn_id=turn_id,
                role=role,
                content=content,
                embedding=self._embedder.embed(content),
                ts=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            )
        )

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
    embedder: Embedder | None = None,
    user_fact_extractor: UserFactExtractor | None = None,
    episodic_extractor: EpisodicExtractor | None = None,
) -> web.Application:
    """Construct the aiohttp web.Application bound to a HarnessApp."""

    policy = HarnessPolicy(components)
    harness = HarnessApp(
        upstream=upstream,
        store=store,
        policy=policy,
        summary_extractor=summary_extractor,
        embedder=embedder,
        user_fact_extractor=user_fact_extractor,
        episodic_extractor=episodic_extractor,
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
        session_id=parsed.session_id,
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
        session_id=session_id,
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
