# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""End-to-end test of the harness server with the ``summary`` component.

Covers the H-A ACTIVE exit-criterion (c): "session N+1 头部自动注入
前 N 个 session 的 summary 作为 system message 前缀".

The test uses the deterministic StubSummaryExtractor so no LLM
calls are made. The flow:

1. Boot the harness with ``--components summary``, memory store.
2. Drive 3 turns of session ``s1`` for user ``user-a``.
3. Drive 3 turns of session ``s2`` for the same user.
4. Assert the upstream received a system-message prefix containing
   the s1 summary on the first turn of s2 (and on every subsequent
   turn of s2 — the prefix is per-call, not per-session-once).
5. Assert that GET /healthz reports ``components == "summary"``.
"""

from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from companion_ref_harness.policy import parse_component_set
from companion_ref_harness.server import build_app
from companion_ref_harness.session_summary import StubSummaryExtractor
from companion_ref_harness.store.sqlite_store import open_store, StoreMode
from companion_ref_harness.upstream_client import EchoUpstreamClient


async def _build_test_client() -> tuple[TestClient, EchoUpstreamClient]:
    upstream = EchoUpstreamClient(model="ref-harness/echo-summary")
    store = open_store(StoreMode.MEMORY)
    app = build_app(
        upstream=upstream,
        store=store,
        components=parse_component_set("summary"),
        summary_extractor=StubSummaryExtractor(),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, upstream


async def _post_turn(
    client: TestClient,
    *,
    session_id: str,
    user_id: str,
    text: str,
) -> dict:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-5",
            "messages": [{"role": "user", "content": text}],
            "metadata": {"session_id": session_id, "user_id": user_id},
        },
    )
    assert resp.status == 200, await resp.text()
    return await resp.json()


# ---------------------------------------------------------------------------
# Healthz
# ---------------------------------------------------------------------------


async def test_healthz_reports_summary_component_enabled() -> None:
    client, _upstream = await _build_test_client()
    try:
        resp = await client.get("/healthz")
        body = await resp.json()
        assert body["components"] == "summary"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Session boundary + injection
# ---------------------------------------------------------------------------


async def test_explicit_close_extracts_summary_and_persists() -> None:
    client, _upstream = await _build_test_client()
    try:
        await _post_turn(
            client, session_id="s1", user_id="user-a", text="I moved to a new flat today",
        )
        await _post_turn(
            client, session_id="s1", user_id="user-a", text="the cat is hiding under the bed",
        )
        resp = await client.post(
            "/v1/sessions/s1/close",
            json={"metadata": {"user_id": "user-a"}},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["closed"] is True
        assert body["summary"] is not None
        # Stub uses the first user turn as topic.
        assert body["summary"]["topic"] == "I moved to a new flat today"
    finally:
        await client.close()


async def test_lazy_close_triggers_on_new_session_for_same_user() -> None:
    client, upstream = await _build_test_client()
    try:
        # Session 1: 2 turns.
        await _post_turn(
            client, session_id="s1", user_id="user-a", text="I moved to a new flat",
        )
        await _post_turn(
            client, session_id="s1", user_id="user-a", text="the cat is hiding",
        )

        # Session 2 starts: lazy-close should fire for s1 before s2's first
        # turn is forwarded. The system prefix injected into s2's first turn
        # must contain s1's topic.
        await _post_turn(
            client, session_id="s2", user_id="user-a", text="hi again",
        )

        # The very last upstream call should be s2's turn, blended with
        # s1's summary in a system-message prefix.
        last_call = upstream.calls[-1]
        assert last_call["session_id"] == "s2"
        sent_messages = last_call["messages"]
        # The blend must put a system message at index 0 with the harness header.
        assert sent_messages[0]["role"] == "system"
        assert "[ref-harness · cross-session memory" in sent_messages[0]["content"]
        # The stub summarised s1 using its first user turn as the topic.
        assert "I moved to a new flat" in sent_messages[0]["content"]
        # The actual user turn is preserved.
        assert sent_messages[-1] == {"role": "user", "content": "hi again"}
    finally:
        await client.close()


async def test_summary_prefix_appears_on_every_subsequent_turn_of_session_2() -> None:
    client, upstream = await _build_test_client()
    try:
        await _post_turn(
            client, session_id="s1", user_id="user-a", text="topic of session one",
        )
        await client.post(
            "/v1/sessions/s1/close",
            json={"metadata": {"user_id": "user-a"}},
        )
        # Two turns in s2; both should see the same prefix (HarnessPolicy is
        # stateless across turns).
        await _post_turn(
            client, session_id="s2", user_id="user-a", text="first s2 turn",
        )
        await _post_turn(
            client, session_id="s2", user_id="user-a", text="second s2 turn",
        )
        s2_calls = [c for c in upstream.calls if c["session_id"] == "s2"]
        assert len(s2_calls) == 2
        for call in s2_calls:
            assert call["messages"][0]["role"] == "system"
            assert "topic of session one" in call["messages"][0]["content"]
    finally:
        await client.close()


async def test_different_user_does_not_see_other_user_summaries() -> None:
    """Cross-user memory isolation (RFC §7.2 attestation)."""
    client, upstream = await _build_test_client()
    try:
        await _post_turn(
            client, session_id="alice-s1", user_id="alice",
            text="alice's secret topic",
        )
        await client.post(
            "/v1/sessions/alice-s1/close",
            json={"metadata": {"user_id": "alice"}},
        )
        await _post_turn(
            client, session_id="bob-s1", user_id="bob", text="bob's first turn",
        )
        bob_call = next(c for c in upstream.calls if c["session_id"] == "bob-s1")
        # Bob's prompt must NOT carry Alice's summary. In the empty-prior-
        # summaries case the policy emits no system prefix at all, so the
        # messages list is just Bob's user turn.
        assert bob_call["messages"] == [
            {"role": "user", "content": "bob's first turn"},
        ]
    finally:
        await client.close()


async def test_session_1_first_turn_has_no_prefix() -> None:
    """The first session of a user has no prior summaries to inject."""
    client, upstream = await _build_test_client()
    try:
        await _post_turn(
            client, session_id="s1", user_id="user-a", text="first ever turn",
        )
        first_call = upstream.calls[0]
        # No system message inserted by the harness.
        assert first_call["messages"] == [
            {"role": "user", "content": "first ever turn"},
        ]
    finally:
        await client.close()
