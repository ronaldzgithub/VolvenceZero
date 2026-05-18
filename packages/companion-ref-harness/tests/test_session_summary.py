# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the session summary extractor surface."""

from __future__ import annotations

import json

import pytest

from companion_ref_harness.session_summary import (
    LLMSummaryExtractor,
    SUMMARY_EXTRACTOR_PROMPT_TEMPLATE,
    SessionSummary,
    StubSummaryExtractor,
    _parse_extractor_json,
)


# ---------------------------------------------------------------------------
# SessionSummary value object
# ---------------------------------------------------------------------------


def test_session_summary_is_frozen() -> None:
    s = SessionSummary(
        scope_key="u",
        session_id="s1",
        topic="t",
        commitments=(),
        open_loops=(),
        extracted_at="2026-05-17T00:00:00+00:00",
        extractor_model="m",
    )
    with pytest.raises(Exception):
        s.topic = "mutated"  # type: ignore[misc]


def test_session_summary_prompt_block_format() -> None:
    s = SessionSummary(
        scope_key="u",
        session_id="s1",
        topic="moved to a new flat",
        commitments=("user: will hang shelves",),
        open_loops=("assistant has not replied about the cat",),
        extracted_at="2026-05-17T00:00:00+00:00",
        extractor_model="m",
    )
    block = s.to_prompt_block()
    assert "- topic: moved to a new flat" in block
    assert "- commitments:" in block
    assert "    - user: will hang shelves" in block
    assert "- open loops:" in block
    assert "    - assistant has not replied about the cat" in block


def test_session_summary_to_payload_is_stable() -> None:
    s = SessionSummary(
        scope_key="u",
        session_id="s1",
        topic="t",
        commitments=("c1", "c2"),
        open_loops=("o1",),
        extracted_at="2026-05-17T00:00:00+00:00",
        extractor_model="m",
    )
    assert s.to_payload() == {
        "topic": "t",
        "commitments": ["c1", "c2"],
        "open_loops": ["o1"],
    }


# ---------------------------------------------------------------------------
# StubSummaryExtractor
# ---------------------------------------------------------------------------


async def test_stub_summary_uses_first_user_turn_as_topic() -> None:
    extractor = StubSummaryExtractor()
    transcript = [
        {"role": "system", "content": "you are a companion"},
        {"role": "user", "content": "I moved to a new flat today"},
        {"role": "assistant", "content": "tell me about it"},
        {"role": "user", "content": "the cat is hiding under the bed"},
    ]
    summary = await extractor.extract(
        scope_key="user-a",
        session_id="s1",
        transcript=transcript,
    )
    assert summary.scope_key == "user-a"
    assert summary.session_id == "s1"
    assert summary.topic == "I moved to a new flat today"
    assert summary.commitments == ()
    assert summary.open_loops == ()
    assert summary.extractor_model == "ref-harness/stub-summary-v1"


async def test_stub_summary_flags_unanswered_user_question() -> None:
    extractor = StubSummaryExtractor()
    summary = await extractor.extract(
        scope_key="user-a",
        session_id="s1",
        transcript=[
            {"role": "user", "content": "are you there?"},
        ],
    )
    assert summary.open_loops == ("assistant has not replied yet",)


async def test_stub_summary_handles_empty_transcript() -> None:
    extractor = StubSummaryExtractor()
    summary = await extractor.extract(
        scope_key="user-a",
        session_id="s1",
        transcript=[],
    )
    assert summary.topic == "(empty)"


# ---------------------------------------------------------------------------
# LLMSummaryExtractor + JSON parser
# ---------------------------------------------------------------------------


def test_parse_extractor_json_clean_payload() -> None:
    raw = json.dumps({"topic": "t", "commitments": ["c1"], "open_loops": []})
    parsed = _parse_extractor_json(raw)
    assert parsed["topic"] == "t"


def test_parse_extractor_json_recovers_from_prose_wrapping() -> None:
    raw = "Here is the JSON you asked for:\n```json\n{\"topic\": \"t\", \"commitments\": [], \"open_loops\": []}\n```\nThanks!"
    parsed = _parse_extractor_json(raw)
    assert parsed["topic"] == "t"


def test_parse_extractor_json_fails_loud_on_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_extractor_json("this response has no JSON at all")


def test_parse_extractor_json_fails_loud_on_malformed_json() -> None:
    # A balanced {...} block that the regex extracts but json.loads rejects.
    with pytest.raises(ValueError, match="malformed JSON"):
        _parse_extractor_json("preamble {topic: missing_quotes, broken}")


async def test_llm_extractor_calls_upstream_with_template_prompt() -> None:
    captured: dict[str, list[dict[str, str]]] = {}

    async def fake_upstream_call(messages: list[dict[str, str]]) -> str:
        captured["messages"] = messages
        return json.dumps(
            {
                "topic": "moved to a new flat",
                "commitments": ["user: hang shelves saturday"],
                "open_loops": ["cat hiding under the bed"],
            }
        )

    extractor = LLMSummaryExtractor(
        model="test/extractor",
        upstream_call=fake_upstream_call,
    )
    transcript = [
        {"role": "user", "content": "I moved to a new flat today"},
        {"role": "assistant", "content": "tell me about it"},
    ]
    summary = await extractor.extract(
        scope_key="user-a",
        session_id="s1",
        transcript=transcript,
    )
    assert summary.topic == "moved to a new flat"
    assert summary.commitments == ("user: hang shelves saturday",)
    assert summary.open_loops == ("cat hiding under the bed",)
    assert summary.extractor_model == "test/extractor"
    # The upstream got a single user message whose content embeds the prompt
    # template; check the template is actually being used.
    assert len(captured["messages"]) == 1
    assert captured["messages"][0]["role"] == "user"
    assert "strict transcript summariser" in captured["messages"][0]["content"]
    assert "I moved to a new flat today" in captured["messages"][0]["content"]


async def test_llm_extractor_rejects_empty_topic() -> None:
    async def fake_upstream_call(messages: list[dict[str, str]]) -> str:
        return json.dumps({"topic": "", "commitments": [], "open_loops": []})

    extractor = LLMSummaryExtractor(
        model="test/extractor",
        upstream_call=fake_upstream_call,
    )
    with pytest.raises(ValueError, match="empty topic"):
        await extractor.extract(
            scope_key="user-a",
            session_id="s1",
            transcript=[{"role": "user", "content": "hi"}],
        )


async def test_llm_extractor_rejects_non_list_commitments() -> None:
    async def fake_upstream_call(messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {"topic": "t", "commitments": "this should be a list", "open_loops": []}
        )

    extractor = LLMSummaryExtractor(
        model="test/extractor",
        upstream_call=fake_upstream_call,
    )
    with pytest.raises(ValueError, match="expected list"):
        await extractor.extract(
            scope_key="user-a",
            session_id="s1",
            transcript=[{"role": "user", "content": "hi"}],
        )


def test_prompt_template_is_inline_in_source() -> None:
    """The reproducibility contract requires the prompt to be inline.

    If someone moves it to a remote file (or worse, a remote URL) this
    test catches the violation. Inline means the constant exists in
    this Python module and has the expected schema markers.
    """
    assert "strict transcript summariser" in SUMMARY_EXTRACTOR_PROMPT_TEMPLATE
    assert "\"topic\"" in SUMMARY_EXTRACTOR_PROMPT_TEMPLATE
    assert "\"commitments\"" in SUMMARY_EXTRACTOR_PROMPT_TEMPLATE
    assert "\"open_loops\"" in SUMMARY_EXTRACTOR_PROMPT_TEMPLATE
