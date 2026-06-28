# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""End-to-end server tests with the deterministic EchoCamelBackend.

Covers the cross-session memory contract: a new session for the same user must
carry the prior session's compacted memory into the agent's context, and a
different user must NOT see another user's memory (cross-user isolation).
"""

from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from companion_camel_baseline.backend import EchoCamelBackend
from companion_camel_baseline.memory_store import StoreMode, open_store
from companion_camel_baseline.server import build_app


async def _build_test_client() -> tuple[TestClient, EchoCamelBackend]:
    backend = EchoCamelBackend()
    store = open_store(StoreMode.MEMORY)
    app = build_app(backend=backend, store=store)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, backend


async def _post_turn(
    client: TestClient, *, session_id: str, user_id: str, text: str,
) -> dict:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-raw",
            "messages": [{"role": "user", "content": text}],
            "metadata": {"session_id": session_id, "user_id": user_id},
        },
    )
    assert resp.status == 200, await resp.text()
    return await resp.json()


async def test_healthz_reports_backend_model() -> None:
    client, _backend = await _build_test_client()
    try:
        resp = await client.get("/healthz")
        body = await resp.json()
        assert body["ok"] is True
        assert body["backend_model"] == "camel-baseline/echo-agent-v1"
    finally:
        await client.close()


async def test_response_shape_is_vendor_neutral() -> None:
    client, _backend = await _build_test_client()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "lifeform-raw",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"session_id": "s1", "user_id": "u1"},
            },
        )
        assert resp.status == 200
        # No baseline-identifying headers leak.
        for header in resp.headers:
            low = header.lower()
            assert not low.startswith("x-camel")
            assert not low.startswith("x-volvence")
            assert not low.startswith("x-companionbench")
        body = await resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
    finally:
        await client.close()


async def test_lazy_compaction_carries_memory_into_next_session() -> None:
    client, backend = await _build_test_client()
    try:
        await _post_turn(client, session_id="s1", user_id="alice", text="I moved to a new flat")
        await _post_turn(client, session_id="s1", user_id="alice", text="the cat is hiding")
        # New session triggers lazy compaction of s1, then responds to s2.
        body = await _post_turn(client, session_id="s2", user_id="alice", text="hi again")
        # The echo backend names recalled topics; s1's topic must appear.
        content = body["choices"][0]["message"]["content"]
        assert "I moved to a new flat" in content
        # Backend saw the prior memory.
        last_call = backend.calls[-1]
        assert last_call["session_id"] == "s2"
        assert "I moved to a new flat" in last_call["prior_memory_topics"]
    finally:
        await client.close()


async def test_explicit_close_persists_record() -> None:
    client, _backend = await _build_test_client()
    try:
        await _post_turn(client, session_id="s1", user_id="alice", text="topic of session one")
        resp = await client.post(
            "/v1/sessions/s1/close", json={"metadata": {"user_id": "alice"}},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["closed"] is True
        assert body["record"]["topic"] == "topic of session one"
    finally:
        await client.close()


async def test_cross_user_isolation() -> None:
    client, backend = await _build_test_client()
    try:
        await _post_turn(client, session_id="alice-s1", user_id="alice", text="alice secret")
        await client.post(
            "/v1/sessions/alice-s1/close", json={"metadata": {"user_id": "alice"}},
        )
        await _post_turn(client, session_id="bob-s1", user_id="bob", text="bob first turn")
        bob_call = next(c for c in backend.calls if c["session_id"] == "bob-s1")
        assert bob_call["prior_memory_topics"] == []
    finally:
        await client.close()


async def test_within_session_transcript_accumulates() -> None:
    client, backend = await _build_test_client()
    try:
        await _post_turn(client, session_id="s1", user_id="alice", text="turn one")
        await _post_turn(client, session_id="s1", user_id="alice", text="turn two")
        last_call = backend.calls[-1]
        # Second turn's session_messages should include the first (user+assistant) pair
        # plus the new user turn.
        roles = [m["role"] for m in last_call["session_messages"]]
        assert roles == ["user", "assistant", "user"]
        assert last_call["session_messages"][-1]["content"] == "turn two"
    finally:
        await client.close()


def test_scope_key_uses_arc_prefix_when_no_user_id() -> None:
    """Sessions of one arc share scope; different arcs isolated (no user_id)."""
    from companion_camel_baseline.server import BaselineApp

    headers = {"User-Agent": "x", "Authorization": "Bearer k"}
    s1 = BaselineApp.derive_scope_key(
        metadata_user_id=None, header_user_id=None,
        request_headers=headers, session_id="arc-ABC-s1",
    )
    s2 = BaselineApp.derive_scope_key(
        metadata_user_id=None, header_user_id=None,
        request_headers=headers, session_id="arc-ABC-s2",
    )
    other = BaselineApp.derive_scope_key(
        metadata_user_id=None, header_user_id=None,
        request_headers=headers, session_id="arc-XYZ-s1",
    )
    assert s1 == s2 == "arc:arc-ABC"
    assert other == "arc:arc-XYZ"
    assert s1 != other


async def test_invalid_body_returns_400() -> None:
    client, _backend = await _build_test_client()
    try:
        resp = await client.post("/v1/chat/completions", data=b"not json")
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] in ("invalid_body", "invalid_request")
    finally:
        await client.close()
