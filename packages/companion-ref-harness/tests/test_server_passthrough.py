# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""End-to-end test of the harness server in passthrough mode.

Passthrough mode is the SHADOW boot config: no components enabled,
the harness simply forwards every request to the upstream. This
test covers the H-A SHADOW exit-criterion (b): "server起 aiohttp
端点能转发 raw chat-completions (passthrough mode)".

Uses aiohttp's TestServer + TestClient so no real port is bound.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from companion_ref_harness.policy import parse_component_set
from companion_ref_harness.server import build_app
from companion_ref_harness.session_summary import StubSummaryExtractor
from companion_ref_harness.store.sqlite_store import open_store, StoreMode
from companion_ref_harness.upstream_client import EchoUpstreamClient


async def _build_test_client(*, components: str = "") -> tuple[TestClient, EchoUpstreamClient]:
    upstream = EchoUpstreamClient(model="ref-harness/echo-passthrough")
    store = open_store(StoreMode.MEMORY)
    app = build_app(
        upstream=upstream,
        store=store,
        components=parse_component_set(components),
        summary_extractor=StubSummaryExtractor(),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, upstream


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_healthz_reports_components_and_upstream() -> None:
    client, _upstream = await _build_test_client(components="")
    try:
        resp = await client.get("/healthz")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["components"] == ""
        assert body["upstream_family"] == "passthrough"
        assert body["upstream_model"] == "ref-harness/echo-passthrough"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Passthrough mode
# ---------------------------------------------------------------------------


async def test_passthrough_forwards_request_unchanged() -> None:
    client, upstream = await _build_test_client(components="")
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5",
                "messages": [
                    {"role": "system", "content": "you are a companion"},
                    {"role": "user", "content": "hi"},
                ],
                "metadata": {"session_id": "arc-s1", "user_id": "user-a"},
            },
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["choices"][0]["message"]["content"].endswith("hi")
        # Response must not carry any harness telemetry headers (vendor-neutral
        # contract per server.py docstring + packet §3.1 H-A 子任务 2).
        forbidden_header_prefixes = (
            "x-ref-harness-",
            "x-lifeform-",
            "x-volvence-",
            "x-companionbench-",
        )
        for header_name in resp.headers:
            lk = header_name.lower()
            for prefix in forbidden_header_prefixes:
                assert not lk.startswith(prefix), (
                    f"forbidden header {header_name!r} in response. "
                    f"The harness must be shape-indistinguishable from a raw "
                    f"OpenAI-compat endpoint."
                )
    finally:
        await client.close()
    # Upstream should have received exactly the messages we sent (no blending).
    assert len(upstream.calls) == 1
    sent_messages = upstream.calls[0]["messages"]
    assert sent_messages == [
        {"role": "system", "content": "you are a companion"},
        {"role": "user", "content": "hi"},
    ]
    assert upstream.calls[0]["session_id"] == "arc-s1"
    assert upstream.calls[0]["user_id"] == "user-a"


async def test_passthrough_auto_mints_session_id_if_missing() -> None:
    client, upstream = await _build_test_client(components="")
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status == 200
    finally:
        await client.close()
    # Upstream should have received SOME session_id, auto-minted.
    sent_sid = upstream.calls[0]["session_id"]
    assert sent_sid is not None
    assert sent_sid.startswith("auto-")


async def test_invalid_body_returns_400_with_error_envelope() -> None:
    client, _upstream = await _build_test_client(components="")
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-5"},  # missing messages
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "invalid_request"
        assert "messages" in body["error"]["message"]
    finally:
        await client.close()


async def test_invalid_role_returns_400() -> None:
    client, _upstream = await _build_test_client(components="")
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5",
                "messages": [{"role": "function", "content": "x"}],
            },
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "invalid_request"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Session close
# ---------------------------------------------------------------------------


async def test_session_close_returns_null_summary_in_passthrough() -> None:
    """In passthrough mode (no summary component), close is a no-op."""
    client, _upstream = await _build_test_client(components="")
    try:
        # Drive one turn so the in-flight buffer has something to summarise.
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5",
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"session_id": "s1", "user_id": "user-a"},
            },
        )
        # Close.
        resp = await client.post(
            "/v1/sessions/s1/close",
            json={"metadata": {"user_id": "user-a"}},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["closed"] is True
        # Passthrough has no summary component, so summary stays null.
        assert body["summary"] is None
    finally:
        await client.close()
