# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Unit tests for upstream_client (no network access)."""

from __future__ import annotations

import pytest

from companion_ref_harness.upstream_client import (
    EchoUpstreamClient,
    UpstreamFamily,
    _anthropic_response_to_upstream,
    _build_openai_compat_envelope,
    _compose_chat_completions_url,
    _openai_compat_response_to_upstream,
    _split_openai_for_anthropic,
    parse_upstream_family,
)


# ---------------------------------------------------------------------------
# Family parsing
# ---------------------------------------------------------------------------


def test_parse_upstream_family_defaults_to_openai_compat() -> None:
    assert parse_upstream_family(None) is UpstreamFamily.OPENAI_COMPAT
    assert parse_upstream_family("") is UpstreamFamily.OPENAI_COMPAT
    assert parse_upstream_family("  ") is UpstreamFamily.OPENAI_COMPAT


def test_parse_upstream_family_accepts_known_values() -> None:
    assert parse_upstream_family("openai-compat") is UpstreamFamily.OPENAI_COMPAT
    assert parse_upstream_family("anthropic") is UpstreamFamily.ANTHROPIC
    assert parse_upstream_family("passthrough") is UpstreamFamily.PASSTHROUGH


def test_parse_upstream_family_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown upstream family"):
        parse_upstream_family("gemini-flash-direct")


def test_compose_chat_completions_url_preserves_mode_query() -> None:
    assert _compose_chat_completions_url(
        "http://127.0.0.1:8000/v1?mode=raw",
    ) == "http://127.0.0.1:8000/v1/chat/completions?mode=raw"


# ---------------------------------------------------------------------------
# Echo client
# ---------------------------------------------------------------------------


async def test_echo_upstream_client_echoes_last_user_message() -> None:
    client = EchoUpstreamClient()
    resp = await client.chat(
        messages=[
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hello there"},
        ],
        max_tokens=None,
        temperature=None,
        session_id="arc-s1",
        user_id="user-a",
    )
    assert resp.text.endswith("hello there")
    assert "arc-s1" in resp.text
    assert resp.raw["choices"][0]["message"]["content"] == resp.text
    assert len(client.calls) == 1
    assert client.calls[0]["session_id"] == "arc-s1"
    await client.close()


async def test_echo_upstream_client_handles_no_user_message() -> None:
    client = EchoUpstreamClient()
    resp = await client.chat(
        messages=[{"role": "system", "content": "persona"}],
        max_tokens=None,
        temperature=None,
        session_id=None,
        user_id=None,
    )
    # Should not crash; produces a degenerate echo.
    assert resp.text.startswith("[echo:no_session]")
    await client.close()


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def test_openai_compat_response_parsing() -> None:
    body = {
        "model": "openai/gpt-5",
        "choices": [
            {"message": {"role": "assistant", "content": "  hi there  "}}
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }
    resp = _openai_compat_response_to_upstream(body, default_model="fallback")
    assert resp.text == "hi there"
    assert resp.model_id == "openai/gpt-5"
    assert resp.usage_prompt_tokens == 42
    assert resp.usage_completion_tokens == 7


def test_openai_compat_response_fails_loud_on_no_choices() -> None:
    with pytest.raises(ValueError, match="no choices"):
        _openai_compat_response_to_upstream({"model": "m"}, default_model="m")


def test_anthropic_response_parsing() -> None:
    body = {
        "model": "anthropic/claude-opus-4.6",
        "content": [
            {"type": "text", "text": "hi "},
            {"type": "text", "text": "there"},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 5},
    }
    resp = _anthropic_response_to_upstream(body, default_model="fallback")
    assert resp.text == "hi there"
    assert resp.model_id == "anthropic/claude-opus-4.6"
    assert resp.usage_prompt_tokens == 100
    assert resp.usage_completion_tokens == 5
    # raw is an OpenAI-shape envelope so the server can re-emit verbatim.
    assert resp.raw["choices"][0]["message"]["content"] == "hi there"
    assert resp.raw["usage"]["prompt_tokens"] == 100


def test_split_openai_for_anthropic_concatenates_systems() -> None:
    system, conv = _split_openai_for_anthropic([
        {"role": "system", "content": "memory block"},
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi back"},
    ])
    assert system == "memory block\n\npersona"
    assert conv == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi back"},
    ]


def test_split_openai_for_anthropic_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unsupported role"):
        _split_openai_for_anthropic([
            {"role": "function", "content": "should fail"},
        ])


def test_envelope_helper_round_trips() -> None:
    env = _build_openai_compat_envelope(text="hello", model="m/1")
    assert env["choices"][0]["message"]["content"] == "hello"
    assert env["model"] == "m/1"
    assert "usage" not in env
    env_with_usage = _build_openai_compat_envelope(
        text="hello", model="m/1",
        usage={"input_tokens": 10, "output_tokens": 2},
    )
    assert env_with_usage["usage"]["prompt_tokens"] == 10
    assert env_with_usage["usage"]["completion_tokens"] == 2
    assert env_with_usage["usage"]["total_tokens"] == 12
