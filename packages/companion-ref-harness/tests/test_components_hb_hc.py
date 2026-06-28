# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Tests for H-B (embed retrieval) + H-C (user-model + episodic) components.

Covers the component units plus an end-to-end server run with all four
components enabled, asserting cross-session retrieval / facts / events land in
the spliced system prefix.
"""

from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from companion_ref_harness.embed import EmbedEntry, HashingEmbedder, cosine, top_k
from companion_ref_harness.episodic import StubEpisodicExtractor
from companion_ref_harness.policy import (
    EPISODIC_SECTION_TAG,
    RETRIEVAL_SECTION_TAG,
    USER_MODEL_SECTION_TAG,
    ComponentSet,
    HarnessComponent,
    HarnessPolicy,
)
from companion_ref_harness.server import build_app
from companion_ref_harness.session_summary import StubSummaryExtractor
from companion_ref_harness.store.sqlite_store import StoreMode, open_store
from companion_ref_harness.upstream_client import EchoUpstreamClient
from companion_ref_harness.user_model import StubUserFactExtractor


# ---------------------------------------------------------------------------
# Embedder unit
# ---------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder()
    v1 = emb.embed("the cat sat on the mat")
    v2 = emb.embed("the cat sat on the mat")
    assert v1 == v2
    assert len(v1) == emb.dim
    # Self-cosine of a non-empty vector is 1.0.
    assert abs(cosine(v1, v1) - 1.0) < 1e-9


def test_retrieval_prefers_lexically_similar_turn() -> None:
    emb = HashingEmbedder()

    def entry(turn_id: str, content: str, ts: str) -> EmbedEntry:
        return EmbedEntry(
            scope_key="alice",
            turn_id=turn_id,
            role="user",
            content=content,
            embedding=emb.embed(content),
            ts=ts,
        )

    entries = [
        entry("t1", "I adopted a kitten named Mia", "2026-01-01T00:00:00+00:00"),
        entry("t2", "the weather is cold today", "2026-01-02T00:00:00+00:00"),
        entry("t3", "my favourite food is ramen", "2026-01-03T00:00:00+00:00"),
    ]
    hits = top_k(query=emb.embed("how is your kitten Mia doing"), entries=entries, k=1)
    assert len(hits) == 1
    assert hits[0].turn_id == "t1"


# ---------------------------------------------------------------------------
# Multi-component blend
# ---------------------------------------------------------------------------


def test_blend_composes_all_enabled_sections() -> None:
    emb = HashingEmbedder()
    policy = HarnessPolicy(ComponentSet(frozenset(HarnessComponent)))
    from companion_ref_harness.episodic import EpisodicEvent
    from companion_ref_harness.user_model import UserFact

    out = policy.blend(
        scope_key="alice",
        session_id="s2",
        messages=[{"role": "user", "content": "hi"}],
        prior_summaries=(),
        retrieved_turns=(
            EmbedEntry(
                scope_key="alice", turn_id="t1", role="user",
                content="I adopted a kitten", embedding=emb.embed("I adopted a kitten"),
                ts="2026-01-01T00:00:00+00:00",
            ),
        ),
        user_facts=(
            UserFact(
                scope_key="alice", key="user_name", value="Alice",
                source_turn="s1", confidence=0.9, ts="2026-01-01T00:00:00+00:00",
            ),
        ),
        episodic_events=(
            EpisodicEvent(
                scope_key="alice", event_id="ev-1", summary="user moved to Berlin",
                source_turn="s1", ts="2026-01-01T00:00:00+00:00",
            ),
        ),
    )
    assert out.blended is True
    sys_content = out.messages[0]["content"]
    assert RETRIEVAL_SECTION_TAG in sys_content
    assert USER_MODEL_SECTION_TAG in sys_content
    assert EPISODIC_SECTION_TAG in sys_content
    assert "I adopted a kitten" in sys_content
    assert "user_name: Alice" in sys_content
    assert "user moved to Berlin" in sys_content


# ---------------------------------------------------------------------------
# End-to-end server with all components
# ---------------------------------------------------------------------------


async def _build_full_client() -> tuple[TestClient, EchoUpstreamClient]:
    upstream = EchoUpstreamClient(model="ref-harness/echo-full")
    store = open_store(StoreMode.MEMORY)
    app = build_app(
        upstream=upstream,
        store=store,
        components=ComponentSet(frozenset(HarnessComponent)),
        summary_extractor=StubSummaryExtractor(),
        embedder=HashingEmbedder(),
        user_fact_extractor=StubUserFactExtractor(),
        episodic_extractor=StubEpisodicExtractor(),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, upstream


async def _post(client: TestClient, *, session_id: str, user_id: str, text: str) -> dict:
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


async def test_full_stack_carries_facts_and_retrieval_across_sessions() -> None:
    client, upstream = await _build_full_client()
    try:
        await _post(client, session_id="s1", user_id="alice", text="my name is Alice")
        await _post(client, session_id="s1", user_id="alice", text="I adopted a kitten named Mia")
        # New session triggers compaction of s1 (summary + facts + episodic).
        await _post(client, session_id="s2", user_id="alice", text="how is Mia the kitten")
        last_call = upstream.calls[-1]
        assert last_call["session_id"] == "s2"
        sys_msg = last_call["messages"][0]
        assert sys_msg["role"] == "system"
        content = sys_msg["content"]
        # User-model fact recalled.
        assert "user_name: Alice" in content
        # Embed retrieval surfaced the kitten turn from s1.
        assert "kitten" in content
        # The actual user turn is preserved at the end.
        assert last_call["messages"][-1] == {"role": "user", "content": "how is Mia the kitten"}
    finally:
        await client.close()


async def test_full_stack_cross_user_isolation() -> None:
    client, upstream = await _build_full_client()
    try:
        await _post(client, session_id="alice-s1", user_id="alice", text="my name is Alice")
        await client.post("/v1/sessions/alice-s1/close", json={"metadata": {"user_id": "alice"}})
        await _post(client, session_id="bob-s1", user_id="bob", text="hello there")
        bob_call = next(c for c in upstream.calls if c["session_id"] == "bob-s1")
        # No system prefix at all for Bob's first turn — nothing known about Bob.
        assert bob_call["messages"] == [{"role": "user", "content": "hello there"}]
    finally:
        await client.close()
