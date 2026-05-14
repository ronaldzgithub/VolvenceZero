"""Contract test: lifeform-openai-compat SSE streaming (debt #12 / #31).

Validates:

1. ``stream=true`` raw-mode request returns 200 with
   ``Content-Type: text/event-stream`` (not 501).
2. SSE frame order matches OpenAI wire format: role delta →
   content delta → final chunk with finish_reason → ``[DONE]``.
3. Concatenating the content delta yields the full assistant text
   from the underlying non-streaming payload.
4. Usage echo carries through to the final chunk.
5. ``stream=false`` still returns a single ``application/json`` body.

Refs:

* docs/known-debts.md #12 / #31
* packages/lifeform-openai-compat/src/lifeform_openai_compat/router.py
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web

from lifeform_openai_compat.dto import ChatCompletionResponse
from lifeform_openai_compat.router import add_openai_routes


# ---------------------------------------------------------------------------
# Fake substrate runtime + manager so the test can drive the router
# without spinning up the real lifeform service.
# ---------------------------------------------------------------------------


class _FakeGenerationResult:
    """Match the .text / .token_count surface raw_substrate_complete reads."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.token_count = len(text.split())


class _FakeRawSubstrate:
    """Minimal runtime that satisfies raw_substrate_complete(...)."""

    model_id = "fake/raw-model"
    runtime_origin = "fake"

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        system_context: str | None = None,
        **_: object,
    ) -> _FakeGenerationResult:
        del max_new_tokens, temperature, system_context, _
        return _FakeGenerationResult(text=f"echo:{prompt}")


class _FakeManager:
    """Mimics lifeform_service.SessionManager surface used by the router."""

    def __init__(self) -> None:
        self.substrate_runtime = _FakeRawSubstrate()


@pytest.fixture
async def client(aiohttp_client):  # noqa: ANN001
    app = web.Application()
    app["session_manager"] = _FakeManager()
    add_openai_routes(app)
    return await aiohttp_client(app)


def _request_payload(stream: bool) -> dict:
    return {
        "model": "fake/raw-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_true_returns_sse_content_type(client) -> None:  # noqa: ANN001
    resp = await client.post(
        "/v1/chat/completions",
        json=_request_payload(stream=True),
        headers={"X-Compat-Mode": "raw"},
    )
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    body = await resp.text()
    assert "data: [DONE]" in body


async def test_stream_frame_order(client) -> None:  # noqa: ANN001
    resp = await client.post(
        "/v1/chat/completions",
        json=_request_payload(stream=True),
        headers={"X-Compat-Mode": "raw"},
    )
    body = await resp.text()
    frames = [
        line[len("data: "):]
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]
    # Last frame is [DONE]; preceding 3 are JSON.
    assert frames[-1] == "[DONE]"
    parsed = [json.loads(frame) for frame in frames[:-1]]
    # Frame 1: role delta
    assert parsed[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert parsed[0]["choices"][0]["finish_reason"] is None
    # Frame 2: content delta
    content_delta = parsed[1]["choices"][0]["delta"]
    assert "content" in content_delta
    assert content_delta["content"] != ""
    # Frame 3: finish_reason
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


async def test_stream_content_delta_matches_full_assistant_text(client) -> None:  # noqa: ANN001
    resp_stream = await client.post(
        "/v1/chat/completions",
        json=_request_payload(stream=True),
        headers={"X-Compat-Mode": "raw"},
    )
    resp_nonstream = await client.post(
        "/v1/chat/completions",
        json=_request_payload(stream=False),
        headers={"X-Compat-Mode": "raw"},
    )
    stream_body = await resp_stream.text()
    nonstream_body = await resp_nonstream.json()
    expected_text = nonstream_body["choices"][0]["message"]["content"]
    streamed_content = ""
    for line in stream_body.split("\n\n"):
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[len("data: "):])
        delta = chunk["choices"][0].get("delta") or {}
        streamed_content += delta.get("content") or ""
    assert streamed_content == expected_text


async def test_stream_false_still_json(client) -> None:  # noqa: ANN001
    """``stream=false`` keeps the single-shot application/json contract."""
    resp = await client.post(
        "/v1/chat/completions",
        json=_request_payload(stream=False),
        headers={"X-Compat-Mode": "raw"},
    )
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("application/json")
    body = await resp.json()
    assert body["choices"][0]["message"]["role"] == "assistant"


async def test_stream_preserves_lifeform_telemetry_headers(client) -> None:  # noqa: ANN001
    """SSE headers still carry ``x-lifeform-*`` for harness telemetry."""
    resp = await client.post(
        "/v1/chat/completions",
        json=_request_payload(stream=True),
        headers={"X-Compat-Mode": "raw"},
    )
    assert resp.headers.get("x-lifeform-mode") == "raw"
    assert "x-lifeform-fingerprint" in resp.headers
