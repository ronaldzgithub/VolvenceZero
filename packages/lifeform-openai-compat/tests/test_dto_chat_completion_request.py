"""Unit tests for the OpenAI ChatCompletion request DTO parsing.

Lock down the parser against (a) round-trip with realistic
EQ-Bench / OpenAI Python client payloads and (b) failure-mode
clarity (we want every malformed payload to surface as a typed
``ValueError`` with a stable error code suffix the router can map
to a 400 response).
"""

from __future__ import annotations

import pytest

from lifeform_openai_compat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    GenerationConfig,
)


# ---------------------------------------------------------------------------
# Round-trip happy paths
# ---------------------------------------------------------------------------


def test_parses_minimal_eqbench_style_payload() -> None:
    """Single-turn user message + a model name is enough to parse."""

    payload = {
        "model": "lifeform-companion@qwen2.5-1.5b",
        "messages": [
            {"role": "user", "content": "Tell me how you would handle a friend who feels overwhelmed."},
        ],
    }
    parsed = ChatCompletionRequest.from_payload(payload)
    assert parsed.model == "lifeform-companion@qwen2.5-1.5b"
    assert parsed.messages == (
        ChatMessage(role="user", content="Tell me how you would handle a friend who feels overwhelmed."),
    )
    assert parsed.generation == GenerationConfig()
    assert parsed.metadata == {}
    assert parsed.user == ""


def test_parses_three_turn_roleplay_with_full_generation_config() -> None:
    payload = {
        "model": "lifeform-companion-cold@qwen2.5-1.5b",
        "messages": [
            {"role": "system", "content": "You are a warm, attentive companion."},
            {"role": "user", "content": "I had a fight with my dad."},
            {"role": "assistant", "content": "I hear you. Tell me more about what happened."},
            {"role": "user", "content": "He never listens."},
        ],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 512,
        "stop": ["</end>", "###"],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 42,
        "metadata": {
            "session_id": "eqbench-arc-007",
            "user_id": "harness-tester",
            "scenario_id": "repair-arc-001",
        },
        "user": "harness-tester",
    }
    parsed = ChatCompletionRequest.from_payload(payload)
    assert parsed.model == "lifeform-companion-cold@qwen2.5-1.5b"
    assert len(parsed.messages) == 4
    assert parsed.messages[0].role == "system"
    assert parsed.messages[-1].content == "He never listens."
    assert parsed.generation.temperature == 0.6
    assert parsed.generation.top_p == 0.9
    assert parsed.generation.max_tokens == 512
    assert parsed.generation.stop == ("</end>", "###")
    assert parsed.generation.seed == 42
    assert parsed.generation.stream is False
    assert parsed.metadata["session_id"] == "eqbench-arc-007"
    assert parsed.metadata["scenario_id"] == "repair-arc-001"
    assert parsed.user == "harness-tester"


def test_string_stop_normalizes_to_single_element_tuple() -> None:
    """OpenAI accepts ``stop`` as either string or array; we always store a tuple."""

    parsed = ChatCompletionRequest.from_payload(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "###",
        }
    )
    assert parsed.generation.stop == ("###",)


def test_unknown_payload_keys_are_ignored() -> None:
    """OpenAI clients send a superset; the parser must drop fields silently."""

    parsed = ChatCompletionRequest.from_payload(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            # Supplying every non-supported field that real harnesses use.
            "logprobs": True,
            "top_logprobs": 5,
            "tools": [],
            "tool_choice": "auto",
            "response_format": {"type": "text"},
            "n": 1,
        }
    )
    assert parsed.generation == GenerationConfig()


# ---------------------------------------------------------------------------
# Failure modes — every error has a stable code suffix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected_prefix"),
    [
        (None, "invalid_request"),
        ({}, "invalid_model"),
        ({"model": "", "messages": [{"role": "user", "content": "hi"}]}, "invalid_model"),
        ({"model": "x"}, "invalid_messages"),
        ({"model": "x", "messages": []}, "invalid_messages"),
        ({"model": "x", "messages": [{"role": "wizard", "content": "hi"}]}, "invalid_role_at_0"),
        ({"model": "x", "messages": [{"role": "user", "content": ["hi"]}]}, "invalid_content_at_0"),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "temperature": "warm"},
            "invalid_temperature",
        ),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stop": [1, 2]},
            "invalid_stop",
        ),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1.5},
            "invalid_max_tokens",
        ),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": "yes"},
            "invalid_stream",
        ),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": [1]},
            "invalid_metadata",
        ),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {"k": 1}},
            "invalid_metadata",
        ),
    ],
)
def test_invalid_payloads_fail_with_stable_code(payload, expected_prefix) -> None:
    with pytest.raises(ValueError) as excinfo:
        ChatCompletionRequest.from_payload(payload)
    assert str(excinfo.value).startswith(expected_prefix), (
        f"expected error message to start with {expected_prefix!r}, got: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# Response envelope shape
# ---------------------------------------------------------------------------


def test_response_envelope_matches_openai_chat_completion_keys() -> None:
    """OpenAI Python client expects exactly these top-level keys."""

    response = ChatCompletionResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=1_700_000_000,
        model="lifeform-companion@qwen2.5-1.5b",
        choices=(
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            ),
        ),
        usage=ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
        ),
    )
    body = response.to_json()
    assert set(body.keys()) == {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
    }
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 12
