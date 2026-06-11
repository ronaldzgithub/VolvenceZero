"""Tests for the interaction SSE streaming surface (debt #12, API layer).

Covers four contract points:

1. ``output_contract.stream=false`` keeps the legacy JSON response.
2. ``stream=true`` + ``chat`` answers ``text/event-stream`` with the
   documented ``ack -> chunk* -> act* -> done`` frame order, where the
   concatenated chunks equal the final text and ``done`` carries the
   full non-streaming body (so consumers persist from one shape).
3. A typed ``DispatchError`` raised after ``ack`` surfaces as a
   terminal ``error`` frame — never a silent EOF.
4. Non-streamable interaction types degrade to JSON even when the
   caller requested a stream (OutputContract best-effort clause).

Plus unit coverage for the ``chunk_text`` segmentation helper.
"""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.app import attach_dlaas_routes
from dlaas_platform_api.streaming import chunk_text


# ---------------------------------------------------------------------------
# Fakes: minimal session manager + session surface used by _handle_chat.
# ---------------------------------------------------------------------------


class _FakeTurnResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.rationale_tags: tuple[str, ...] = ()


class _FakeTurnResult:
    def __init__(self, text: str) -> None:
        self.response = _FakeTurnResponse(text)
        self.active_regime = "task_focus"
        self.active_abstract_action = None


class _FakeSession:
    """Echo session: run_turn returns a deterministic long-ish text."""

    latest_active_snapshots: dict[str, Any] = {}

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def run_turn(self, brief: str, **_kwargs: Any) -> _FakeTurnResult:
        return _FakeTurnResult(self._reply)

    def submit_reviewed_knowledge_event(self, **_kwargs: Any) -> tuple[str, ...]:
        return ("evt_1",)


class _FakeSessionManager:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self._sessions: dict[str, _FakeSession] = {}

    async def get_session(self, session_id: str) -> _FakeSession:
        from lifeform_service import SessionNotFoundError

        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        return self._sessions[session_id]

    async def create_session(
        self, *, session_id: str, user_id: str | None = None
    ) -> _FakeSession:
        session = _FakeSession(self._reply)
        self._sessions[session_id] = session
        return session

    def session_end_user(self, session_id: str) -> str | None:
        return None


def _build_app(reply: str) -> web.Application:
    app = web.Application()
    app["session_manager"] = _FakeSessionManager(reply)
    return attach_dlaas_routes(app)


def _envelope_body(
    *,
    stream: bool,
    interaction_type: str = "chat",
    human_brief: str = "hello there",
    structured_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract_id": "ctr_1",
        "session_id": "s1",
        "end_user_ref": "alice",
        "interaction_type": interaction_type,
        "human_brief": human_brief,
        "structured_context": structured_context or {},
        "output_contract": {
            "delivery_channel": "dlaas",
            "format": "text",
            "stream": stream,
        },
        "lang": "en",
    }


def _parse_sse(raw: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into ordered (event, parsed-data) tuples."""

    frames: list[tuple[str, Any]] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        frames.append((event, json.loads(data)))
    return frames


async def _post(app: web.Application, body: dict[str, Any]):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post(
            "/dlaas/v1/instances/ai_1/interactions", json=body
        )
        text = await resp.text()
        return resp.status, resp.headers.get("Content-Type", ""), text
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Route-level contract tests
# ---------------------------------------------------------------------------


async def test_stream_false_keeps_json_response() -> None:
    status, content_type, text = await _post(
        _build_app("plain reply"), _envelope_body(stream=False)
    )
    assert status == 200
    assert "application/json" in content_type
    body = json.loads(text)
    assert body["status"] == "ok"
    assert body["output_acts"][0]["payload"]["content"] == "plain reply"


async def test_stream_true_chat_emits_ack_chunk_act_done() -> None:
    reply = "x" * 250  # forces > 1 chunk at STREAM_CHUNK_CHARS=120
    status, content_type, text = await _post(
        _build_app(reply), _envelope_body(stream=True)
    )
    assert status == 200
    assert "text/event-stream" in content_type

    frames = _parse_sse(text)
    events = [event for event, _data in frames]
    assert events[0] == "ack"
    assert events[-1] == "done"
    assert "chunk" in events and "act" in events
    # Frame order: every chunk precedes every act, which precede done.
    assert events.index("act") > events.index("chunk")

    ack = frames[0][1]
    assert ack["ai_id"] == "ai_1"
    assert ack["session_id"] == "s1"
    assert ack["interaction_type"] == "chat"

    chunks = [data["content"] for event, data in frames if event == "chunk"]
    assert len(chunks) == 3  # 250 chars / 120 per chunk
    assert "".join(chunks) == reply

    done = frames[-1][1]
    # `done` carries the full non-streaming body shape.
    assert done["status"] == "ok"
    assert done["output_acts"][0]["payload"]["content"] == reply
    assert done["response_id"].startswith("resp_")

    acts = [data for event, data in frames if event == "act"]
    assert acts == done["output_acts"]


async def test_stream_dispatch_error_surfaces_error_frame() -> None:
    # Empty human_brief passes envelope parsing but raises the typed
    # DispatchError(invalid_human_brief) inside _handle_chat — i.e.
    # AFTER the ack frame committed the response to SSE.
    status, content_type, text = await _post(
        _build_app("unused"), _envelope_body(stream=True, human_brief="")
    )
    assert status == 200
    assert "text/event-stream" in content_type
    frames = _parse_sse(text)
    events = [event for event, _data in frames]
    assert events == ["ack", "error"]
    error = frames[-1][1]
    assert error["error"] == "invalid_human_brief"
    assert error["status"] == 400


async def test_stream_request_on_observe_degrades_to_json() -> None:
    status, content_type, text = await _post(
        _build_app("unused"),
        _envelope_body(
            stream=True,
            interaction_type="observe",
            human_brief="note text",
            structured_context={
                "observation_type": "class_note",
                "event_id": "evt_1",
                "knowledge_id": "k_1",
                "summary": "note text",
            },
        ),
    )
    assert status == 200
    assert "application/json" in content_type
    body = json.loads(text)
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# chunk_text unit coverage
# ---------------------------------------------------------------------------


def test_chunk_text_concatenation_is_exact() -> None:
    content = "abcdef" * 50  # 300 chars
    pieces = chunk_text(content, 120)
    assert len(pieces) == 3
    assert "".join(pieces) == content


def test_chunk_text_empty_returns_no_pieces() -> None:
    assert chunk_text("") == ()


def test_chunk_text_rejects_non_positive_size() -> None:
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", 0)
