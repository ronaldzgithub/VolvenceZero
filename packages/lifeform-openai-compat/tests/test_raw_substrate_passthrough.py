"""Unit tests for raw substrate passthrough mode.

These tests exercise the OpenAI request → ``runtime.generate(...)``
mapping with a fake runtime, so they do not require torch /
transformers / a real Qwen download. The fake runtime captures the
arguments it was called with so we can assert the mapping is
correct (system messages joined into ``system_context``, history
preserved, last user message becomes ``prompt``, generation knobs
forwarded).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lifeform_openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    GenerationConfig,
    RawSubstrateUnavailable,
    estimate_prompt_tokens,
    raw_substrate_complete,
    split_messages,
)


# ---------------------------------------------------------------------------
# Fake runtime
# ---------------------------------------------------------------------------


@dataclass
class _FakeGenerationResult:
    text: str
    token_count: int


class _FakeRuntime:
    """In-memory stand-in for OpenWeightResidualRuntime.

    Records the most recent ``generate`` call for inspection.
    """

    def __init__(
        self,
        *,
        canned_text: str = "I hear you. That sounds really hard.",
        canned_token_count: int = 12,
        model_id: str = "fake/qwen2.5-1.5b-instruct",
        runtime_origin: str = "test-fake",
    ) -> None:
        self.canned_text = canned_text
        self.canned_token_count = canned_token_count
        self.model_id = model_id
        self.runtime_origin = runtime_origin
        self.last_call: dict[str, object] | None = None

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> _FakeGenerationResult:
        self.last_call = {
            "prompt": prompt,
            "system_context": system_context,
            "chat_messages": chat_messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        return _FakeGenerationResult(
            text=self.canned_text, token_count=self.canned_token_count
        )


# ---------------------------------------------------------------------------
# split_messages
# ---------------------------------------------------------------------------


def test_split_messages_extracts_system_history_and_prompt() -> None:
    messages = (
        ChatMessage(role="system", content="You are a warm companion."),
        ChatMessage(role="user", content="I'm feeling overwhelmed."),
        ChatMessage(role="assistant", content="Tell me what's weighing on you."),
        ChatMessage(role="user", content="My job is too much right now."),
    )
    system_context, prompt, history = split_messages(messages)
    assert system_context == "You are a warm companion."
    assert prompt == "My job is too much right now."
    assert history == (
        ("user", "I'm feeling overwhelmed."),
        ("assistant", "Tell me what's weighing on you."),
    )


def test_split_messages_concatenates_multiple_system_blocks() -> None:
    messages = (
        ChatMessage(role="system", content="You are warm."),
        ChatMessage(role="developer", content="Avoid clinical jargon."),
        ChatMessage(role="user", content="Tell me a story."),
    )
    system_context, prompt, _ = split_messages(messages)
    assert system_context == "You are warm.\n\nAvoid clinical jargon."
    assert prompt == "Tell me a story."


def test_split_messages_handles_assistant_at_end_as_continuation() -> None:
    messages = (
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi!"),
    )
    system_context, prompt, history = split_messages(messages)
    assert system_context == ""
    assert prompt == "Hi!"
    assert history == (("user", "Hello"),)


def test_split_messages_empty_raises_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        split_messages(())
    assert str(excinfo.value).startswith("invalid_messages")


# ---------------------------------------------------------------------------
# estimate_prompt_tokens
# ---------------------------------------------------------------------------


def test_estimate_prompt_tokens_chars_over_4_heuristic() -> None:
    assert estimate_prompt_tokens("", "", ()) == 1  # at least 1
    assert estimate_prompt_tokens("hello", "", ()) == 1  # 5 chars / 4 = 1
    assert estimate_prompt_tokens("hello world", "", ()) == 2  # 11 / 4 = 2
    assert estimate_prompt_tokens(
        "system" * 4,  # 24 chars
        "prompt" * 4,  # 24 chars
        (("user", "history" * 2),),  # 14 chars
    ) == (24 + 24 + 14) // 4


# ---------------------------------------------------------------------------
# raw_substrate_complete
# ---------------------------------------------------------------------------


def _request_from_messages(
    *messages: tuple[str, str],
    model: str = "raw-substrate-test",
    generation: GenerationConfig | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=tuple(ChatMessage(role=role, content=content) for role, content in messages),
        generation=generation or GenerationConfig(),
    )


def test_raw_passthrough_forwards_messages_to_runtime_generate() -> None:
    runtime = _FakeRuntime()
    request = _request_from_messages(
        ("system", "Be supportive."),
        ("user", "I feel low."),
    )
    response = raw_substrate_complete(request=request, runtime=runtime)

    # Runtime saw the right arguments.
    assert runtime.last_call is not None
    assert runtime.last_call["system_context"] == "Be supportive."
    assert runtime.last_call["prompt"] == "I feel low."
    assert runtime.last_call["chat_messages"] == ()

    # Response shape is OpenAI-compatible.
    assert response.model == "raw-substrate-test"
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert choice.message.content == "I hear you. That sounds really hard."
    assert choice.finish_reason == "stop"
    assert response.usage.completion_tokens == 12
    assert response.usage.total_tokens == response.usage.prompt_tokens + 12
    assert response.system_fingerprint.startswith("raw-substrate:fake/qwen2.5-1.5b-instruct")


def test_raw_passthrough_uses_user_supplied_generation_config() -> None:
    runtime = _FakeRuntime()
    request = _request_from_messages(
        ("user", "hi"),
        generation=GenerationConfig(temperature=0.2, max_tokens=64),
    )
    raw_substrate_complete(request=request, runtime=runtime)
    assert runtime.last_call["temperature"] == 0.2
    assert runtime.last_call["max_new_tokens"] == 64


def test_raw_passthrough_falls_back_to_defaults_when_unspecified() -> None:
    runtime = _FakeRuntime()
    request = _request_from_messages(("user", "hi"))
    raw_substrate_complete(request=request, runtime=runtime)
    assert runtime.last_call["temperature"] == 0.7
    assert runtime.last_call["max_new_tokens"] == 512


def test_raw_passthrough_marks_finish_reason_length_when_budget_exhausted() -> None:
    runtime = _FakeRuntime(canned_token_count=64)
    request = _request_from_messages(
        ("user", "hi"),
        generation=GenerationConfig(max_tokens=64),
    )
    response = raw_substrate_complete(request=request, runtime=runtime)
    assert response.choices[0].finish_reason == "length"


def test_raw_passthrough_with_no_runtime_raises_typed_unavailable() -> None:
    request = _request_from_messages(("user", "hi"))
    with pytest.raises(RawSubstrateUnavailable) as excinfo:
        raw_substrate_complete(request=request, runtime=None)
    msg = str(excinfo.value)
    assert "raw substrate mode requires" in msg
    assert "lifeform-serve" in msg
    assert "synthetic" in msg


def test_raw_passthrough_id_falls_back_to_minted_chatcmpl_when_unspecified() -> None:
    runtime = _FakeRuntime()
    response = raw_substrate_complete(
        request=_request_from_messages(("user", "hi")), runtime=runtime
    )
    assert response.id.startswith("chatcmpl-")
    assert len(response.id) > len("chatcmpl-")


def test_raw_passthrough_id_uses_supplied_request_id_when_provided() -> None:
    runtime = _FakeRuntime()
    response = raw_substrate_complete(
        request=_request_from_messages(("user", "hi")),
        runtime=runtime,
        request_id="explicit-id-007",
    )
    assert response.id == "explicit-id-007"


def test_raw_passthrough_history_preserves_role_order() -> None:
    runtime = _FakeRuntime()
    request = _request_from_messages(
        ("system", "warm"),
        ("user", "u1"),
        ("assistant", "a1"),
        ("user", "u2"),
        ("assistant", "a2"),
        ("user", "u3"),
    )
    raw_substrate_complete(request=request, runtime=runtime)
    assert runtime.last_call["system_context"] == "warm"
    assert runtime.last_call["prompt"] == "u3"
    assert runtime.last_call["chat_messages"] == (
        ("user", "u1"),
        ("assistant", "a1"),
        ("user", "u2"),
        ("assistant", "a2"),
    )


def test_raw_passthrough_response_envelope_round_trips_through_to_json() -> None:
    runtime = _FakeRuntime(canned_text="OK.", canned_token_count=2)
    response = raw_substrate_complete(
        request=_request_from_messages(("user", "test")), runtime=runtime
    )
    body = response.to_json()
    # OpenAI Python client requires these top-level keys.
    assert set(body.keys()) >= {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
    }
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "OK."
