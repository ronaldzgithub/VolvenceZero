# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.

"""Pluggable client surface for talking to the system under test (SUT).

The arc runner needs to send a per-turn request and receive an
assistant message back. We define a thin Protocol here so the runner
is decoupled from any specific HTTP library — tests use a
:class:`FakeSUTClient` in-process; production runs use the
:class:`OpenAIChatClient` against a real OpenAI-compatible endpoint.

The SUT client is a strict consumer of the OpenAI ``/v1/chat/completions``
HTTP contract. We never reach into the SUT's internals — that is the
RFC §3 P4 outcome-level evaluation contract.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclasses.dataclass(frozen=True)
class SUTResponse:
    """One assistant turn from the SUT.

    ``raw`` is the unparsed JSON body so callers (cost telemetry,
    telemetry header inspection) can pull whatever they need without
    forcing a structured shape on every backend.
    """

    text: str
    model_id: str
    response_headers: Mapping[str, str]
    usage_prompt_tokens: int | None
    usage_completion_tokens: int | None
    raw: dict[str, Any]


@runtime_checkable
class SUTClient(Protocol):
    """The minimum surface arc_runner needs."""

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> SUTResponse: ...


# ---------------------------------------------------------------------------
# Real OpenAI-compatible client
# ---------------------------------------------------------------------------


class OpenAIChatClient:
    """POST to /v1/chat/completions on any OpenAI-compatible endpoint.

    Stays minimal and dependency-free (urllib only). For high-throughput
    benchmark runs the caller can wrap this in an asyncio executor.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        request_timeout_s: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = request_timeout_s
        self._extra_headers = dict(extra_headers or {})

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> SUTResponse:
        import urllib.request

        metadata: dict[str, str] = {"session_id": session_id}
        if user_id:
            metadata["user_id"] = user_id
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "metadata": metadata,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw_body = json.loads(resp.read().decode("utf-8"))
            response_headers = {k: v for k, v in resp.headers.items()}
        choices = raw_body.get("choices") or []
        if not choices:
            raise RuntimeError(f"SUT returned no choices: {raw_body}")
        text = str(choices[0].get("message", {}).get("content", "")).strip()
        usage = raw_body.get("usage") or {}
        return SUTResponse(
            text=text,
            model_id=str(raw_body.get("model", self._model)),
            response_headers=response_headers,
            usage_prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            usage_completion_tokens=_int_or_none(usage.get("completion_tokens")),
            raw=raw_body,
        )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fake SUT for tests
# ---------------------------------------------------------------------------


class EchoFakeSUTClient:
    """In-process fake that echoes the latest user message with a prefix.

    Used by arc_runner unit tests so we never need a network call. It
    also records every call for inspection (per-test history checks).
    """

    def __init__(self, *, model: str = "fake-sut/echo") -> None:
        self._model = model
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> SUTResponse:
        self.calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        text = f"[echo:{session_id[-6:]}] {last_user}"
        return SUTResponse(
            text=text,
            model_id=self._model,
            response_headers={"x-fake-sut": "1"},
            usage_prompt_tokens=sum(len(m.get("content", "")) // 4 for m in messages),
            usage_completion_tokens=max(1, len(text) // 4),
            raw={"model": self._model, "choices": [{"message": {"role": "assistant", "content": text}}]},
        )
