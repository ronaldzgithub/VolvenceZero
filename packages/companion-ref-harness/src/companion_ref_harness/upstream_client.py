# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Upstream chat-completions client.

The harness forwards a (possibly-blended) ``messages`` list to an
upstream model and returns the assistant message. Three families are
supported:

* ``openai-compat`` (default) — any endpoint serving the OpenAI
  ``POST /v1/chat/completions`` shape (OpenAI, Together, DeepSeek,
  Groq, Mistral, vLLM, ...).
* ``anthropic`` — the Anthropic ``POST /v1/messages`` shape; the
  harness translates request and response to / from the OpenAI shape
  so the calling code stays uniform.
* ``passthrough`` — used by tests; loops a deterministic echo so
  smoke runs do not require any network access.

We deliberately use ``aiohttp.ClientSession`` rather than the
official SDKs to (1) keep the dependency footprint small, (2)
remove version-churn risk against fast-moving SDK releases, and
(3) match the existing
``packages/companion-bench/src/companion_bench/sut_client.py``
pattern.

Family selection precedence inside the server:

1. ``X-Compat-Upstream-Family`` request header
2. CLI ``--upstream-family`` flag (set at boot)
3. Default = ``openai-compat``
"""

from __future__ import annotations

import dataclasses
import enum
import json
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import aiohttp


def _compose_chat_completions_url(base_url: str) -> str:
    """Append ``/chat/completions`` to ``base_url`` preserving any query string.

    A naive ``base_url + "/chat/completions"`` breaks when ``base_url`` carries
    a query (e.g. the same-substrate ablation raw track uses
    ``http://127.0.0.1:8000/v1?mode=raw`` to select the OpenAI-compat router's
    raw dispatch): the query would swallow the appended path.
    """

    parts = urlsplit(base_url)
    path = parts.path.rstrip("/") + "/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


# ---------------------------------------------------------------------------
# Family enum + dispatcher
# ---------------------------------------------------------------------------


class UpstreamFamily(str, enum.Enum):
    OPENAI_COMPAT = "openai-compat"
    ANTHROPIC = "anthropic"
    PASSTHROUGH = "passthrough"


_VALID_FAMILIES: frozenset[str] = frozenset(f.value for f in UpstreamFamily)


def parse_upstream_family(raw: str | None) -> UpstreamFamily:
    """Parse a family selector string. Defaults to ``openai-compat``."""

    if raw is None or not raw.strip():
        return UpstreamFamily.OPENAI_COMPAT
    try:
        return UpstreamFamily(raw.strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"unknown upstream family {raw!r}; allowed: {sorted(_VALID_FAMILIES)}"
        ) from exc


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UpstreamResponse:
    """One assistant turn from the upstream backend."""

    text: str
    model_id: str
    usage_prompt_tokens: int | None
    usage_completion_tokens: int | None
    # ``raw`` is the unparsed OpenAI-compat response body so the server
    # can re-emit it verbatim to the harness client without re-shaping.
    raw: dict[str, Any]


@runtime_checkable
class UpstreamClient(Protocol):
    """The minimum surface the server needs.

    Implementations are async; the server awaits on every call.
    """

    @property
    def family(self) -> UpstreamFamily: ...

    @property
    def model(self) -> str: ...

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        session_id: str | None,
        user_id: str | None,
    ) -> UpstreamResponse: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# OpenAI-compat client
# ---------------------------------------------------------------------------


class OpenAICompatUpstreamClient:
    """Async client for OpenAI ``/v1/chat/completions``-shape endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._extra_headers = dict(extra_headers or {})
        self._session: aiohttp.ClientSession | None = None

    @property
    def family(self) -> UpstreamFamily:
        return UpstreamFamily.OPENAI_COMPAT

    @property
    def model(self) -> str:
        return self._model

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        session_id: str | None,
        user_id: str | None,
    ) -> UpstreamResponse:
        metadata: dict[str, str] = {}
        if session_id:
            metadata["session_id"] = session_id
        if user_id:
            metadata["user_id"] = user_id
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if metadata:
            body["metadata"] = metadata
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        session = await self._ensure_session()
        async with session.post(
            _compose_chat_completions_url(self._base_url),
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise UpstreamError(
                    status=resp.status,
                    detail=detail,
                    upstream=self._base_url,
                )
            raw_body = await resp.json()
        return _openai_compat_response_to_upstream(raw_body, default_model=self._model)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Anthropic client (translates Messages API <-> OpenAI shape)
# ---------------------------------------------------------------------------


class AnthropicUpstreamClient:
    """Async client for Anthropic ``/v1/messages``.

    Translates the incoming OpenAI-shape ``messages`` list to
    Anthropic's split ``system`` + ``messages`` shape, and translates
    the response back to OpenAI shape so the server can re-emit it
    without per-family branching.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        anthropic_version: str = "2023-06-01",
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._anthropic_version = anthropic_version
        self._extra_headers = dict(extra_headers or {})
        self._session: aiohttp.ClientSession | None = None

    @property
    def family(self) -> UpstreamFamily:
        return UpstreamFamily.ANTHROPIC

    @property
    def model(self) -> str:
        return self._model

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        session_id: str | None,
        user_id: str | None,
    ) -> UpstreamResponse:
        system_text, conversation = _split_openai_for_anthropic(messages)
        body: dict[str, Any] = {
            "model": self._model,
            "messages": conversation,
            "max_tokens": max_tokens if max_tokens is not None else 1024,
        }
        if system_text:
            body["system"] = system_text
        if temperature is not None:
            body["temperature"] = temperature
        if user_id:
            body["metadata"] = {"user_id": user_id}
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        session = await self._ensure_session()
        async with session.post(
            f"{self._base_url}/messages",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        ) as resp:
            if resp.status >= 400:
                detail = await resp.text()
                raise UpstreamError(
                    status=resp.status,
                    detail=detail,
                    upstream=self._base_url,
                )
            raw_body = await resp.json()
        return _anthropic_response_to_upstream(raw_body, default_model=self._model)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Passthrough echo client (tests)
# ---------------------------------------------------------------------------


class EchoUpstreamClient:
    """In-process fake that echoes the latest user message.

    Used by unit tests and by the CLI ``--upstream-family passthrough``
    boot mode so that smoke runs without any network access still
    exercise the full server pipeline end-to-end.

    Records every call into :attr:`calls` for assertion in tests.
    """

    def __init__(self, *, model: str = "ref-harness/echo-upstream-v1") -> None:
        self._model = model
        self.calls: list[dict[str, Any]] = []

    @property
    def family(self) -> UpstreamFamily:
        return UpstreamFamily.PASSTHROUGH

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        session_id: str | None,
        user_id: str | None,
    ) -> UpstreamResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        text = f"[echo:{session_id or 'no_session'}] {last_user}"
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "session_id": session_id,
                "user_id": user_id,
            }
        )
        return UpstreamResponse(
            text=text,
            model_id=self._model,
            usage_prompt_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
            usage_completion_tokens=max(1, len(text) // 4),
            raw=_build_openai_compat_envelope(text=text, model=self._model),
        )

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UpstreamError(RuntimeError):
    """Raised when the upstream returns an HTTP error.

    The server maps this to a 502 with body ``{"error": {"code":
    "ref_harness_upstream_error", "message": "..."}}``. Fail-loud per
    the no-swallow-errors rule.
    """

    def __init__(self, *, status: int, detail: str, upstream: str) -> None:
        super().__init__(
            f"upstream HTTP {status} from {upstream}: {detail[:200]}"
        )
        self.status = status
        self.detail = detail
        self.upstream = upstream


# ---------------------------------------------------------------------------
# Translation helpers (public for testability)
# ---------------------------------------------------------------------------


def _openai_compat_response_to_upstream(
    raw_body: dict[str, Any],
    *,
    default_model: str,
) -> UpstreamResponse:
    choices = raw_body.get("choices") or []
    if not choices:
        raise ValueError(
            f"upstream returned no choices; raw[:200]={str(raw_body)[:200]!r}"
        )
    msg = choices[0].get("message") or {}
    text = str(msg.get("content", "")).strip()
    usage = raw_body.get("usage") or {}
    return UpstreamResponse(
        text=text,
        model_id=str(raw_body.get("model", default_model)),
        usage_prompt_tokens=_safe_int(usage.get("prompt_tokens")),
        usage_completion_tokens=_safe_int(usage.get("completion_tokens")),
        raw=raw_body,
    )


def _anthropic_response_to_upstream(
    raw_body: dict[str, Any],
    *,
    default_model: str,
) -> UpstreamResponse:
    content_blocks = raw_body.get("content") or []
    text_chunks = [
        str(block.get("text", ""))
        for block in content_blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "".join(text_chunks).strip()
    usage = raw_body.get("usage") or {}
    model_id = str(raw_body.get("model", default_model))
    return UpstreamResponse(
        text=text,
        model_id=model_id,
        usage_prompt_tokens=_safe_int(usage.get("input_tokens")),
        usage_completion_tokens=_safe_int(usage.get("output_tokens")),
        raw=_build_openai_compat_envelope(text=text, model=model_id, usage=usage),
    )


def _split_openai_for_anthropic(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Pull all ``role == 'system'`` messages into a single system string.

    Anthropic's ``/v1/messages`` API takes a top-level ``system``
    string plus a ``messages`` array of strictly user/assistant
    turns. We concatenate any system messages (the harness policy
    may have prepended a memory prefix) in order.
    """

    system_lines: list[str] = []
    conversation: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role == "system":
            system_lines.append(content)
        elif role in ("user", "assistant"):
            conversation.append({"role": role, "content": content})
        else:
            raise ValueError(
                f"unsupported role in anthropic translation: {role!r}"
            )
    return ("\n\n".join(s for s in system_lines if s), conversation)


def _build_openai_compat_envelope(
    *,
    text: str,
    model: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap an assistant text in the OpenAI chat-completions response shape."""

    envelope: dict[str, Any] = {
        "id": "ref-harness-stub",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not None:
        # Anthropic shape is input_tokens / output_tokens; expose both.
        envelope["usage"] = {
            "prompt_tokens": _safe_int(usage.get("input_tokens")) or 0,
            "completion_tokens": _safe_int(usage.get("output_tokens")) or 0,
            "total_tokens": (
                (_safe_int(usage.get("input_tokens")) or 0)
                + (_safe_int(usage.get("output_tokens")) or 0)
            ),
        }
    return envelope


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
