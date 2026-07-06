# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the deterministic EchoCamelBackend + shared helpers."""

from __future__ import annotations

import pytest

from companion_camel_baseline.backend import (
    CamelBackendError,
    EchoCamelBackend,
    _normalize_upstream_openai_base_url,
    build_memory_preamble,
    parse_compaction_json,
)
from companion_camel_baseline.memory_store import SessionMemoryRecord


def _record(*, session_id: str, topic: str) -> SessionMemoryRecord:
    return SessionMemoryRecord(
        scope_key="alice",
        session_id=session_id,
        topic=topic,
        salient=("fact one", "fact two"),
        extracted_at="2026-01-01T00:00:00+00:00",
        extractor_model="test",
    )


async def test_respond_echoes_last_user_and_names_recalled_topics() -> None:
    backend = EchoCamelBackend()
    reply = await backend.respond(
        scope_key="alice",
        session_id="s2",
        system_prompt="You are a companion.",
        prior_memory=(_record(session_id="s1", topic="moved to a new flat"),),
        session_messages=[{"role": "user", "content": "hi again"}],
        max_tokens=64,
        temperature=0.0,
    )
    assert "hi again" in reply.text
    assert "moved to a new flat" in reply.text
    assert reply.raw["choices"][0]["message"]["content"] == reply.text


async def test_respond_without_prior_memory_has_no_recall_note() -> None:
    backend = EchoCamelBackend()
    reply = await backend.respond(
        scope_key="alice",
        session_id="s1",
        system_prompt="You are a companion.",
        prior_memory=(),
        session_messages=[{"role": "user", "content": "first turn"}],
        max_tokens=None,
        temperature=None,
    )
    assert "recalling" not in reply.text
    assert "first turn" in reply.text


async def test_compact_uses_first_user_turn_as_topic() -> None:
    backend = EchoCamelBackend()
    record = await backend.compact(
        scope_key="alice",
        session_id="s1",
        transcript=[
            {"role": "user", "content": "I moved to a new flat today"},
            {"role": "assistant", "content": "congrats"},
            {"role": "user", "content": "the cat is hiding"},
        ],
    )
    assert record.topic == "I moved to a new flat today"
    assert "the cat is hiding" in record.salient


def test_normalize_upstream_openai_base_url_moves_mode_to_header() -> None:
    base_url, headers = _normalize_upstream_openai_base_url(
        "http://127.0.0.1:8000/v1?mode=raw"
    )
    assert base_url == "http://127.0.0.1:8000/v1"
    assert headers == {"X-Compat-Mode": "raw"}


def test_build_memory_preamble_empty_is_none() -> None:
    assert build_memory_preamble(()) is None


def test_build_memory_preamble_includes_header_and_topics() -> None:
    preamble = build_memory_preamble((_record(session_id="s1", topic="topic A"),))
    assert preamble is not None
    assert "camel-baseline · cross-session memory" in preamble
    assert "topic A" in preamble


def test_parse_compaction_json_recovers_from_fenced_block() -> None:
    raw = "```json\n{\"topic\": \"t\", \"salient\": [\"a\"]}\n```"
    parsed = parse_compaction_json(raw)
    assert parsed["topic"] == "t"
    assert parsed["salient"] == ["a"]


def test_parse_compaction_json_fail_loud_on_garbage() -> None:
    with pytest.raises(CamelBackendError):
        parse_compaction_json("no json here at all")
