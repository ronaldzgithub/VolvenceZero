# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Unit tests for HarnessPolicy + ComponentSet parsing."""

from __future__ import annotations

import pytest

from companion_ref_harness.policy import (
    ComponentSet,
    HarnessComponent,
    HarnessPolicy,
    MAX_PRIOR_SUMMARIES,
    SYSTEM_PREFIX_FOOTER,
    SYSTEM_PREFIX_HEADER,
    parse_component_set,
)
from companion_ref_harness.session_summary import SessionSummary


def _summary(*, session_id: str, topic: str, extracted_at: str) -> SessionSummary:
    return SessionSummary(
        scope_key="user-a",
        session_id=session_id,
        topic=topic,
        commitments=(),
        open_loops=(),
        extracted_at=extracted_at,
        extractor_model="test/m",
    )


# ---------------------------------------------------------------------------
# parse_component_set
# ---------------------------------------------------------------------------


def test_parse_component_set_empty_is_passthrough() -> None:
    assert parse_component_set("").is_empty()
    assert parse_component_set(None).is_empty()
    assert parse_component_set("   ").is_empty()


def test_parse_component_set_summary_only() -> None:
    cs = parse_component_set("summary")
    assert cs.has(HarnessComponent.SUMMARY)
    assert not cs.has(HarnessComponent.EMBED)


def test_parse_component_set_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        parse_component_set("summary,bogus")


def test_parse_component_set_csv_canonicalises() -> None:
    cs = parse_component_set("summary, summary , summary")
    assert cs.to_csv() == "summary"


# ---------------------------------------------------------------------------
# HarnessPolicy gating (H-A only wires summary)
# ---------------------------------------------------------------------------


def test_policy_rejects_h_b_h_c_components() -> None:
    cs = ComponentSet(frozenset({HarnessComponent.SUMMARY, HarnessComponent.EMBED}))
    with pytest.raises(NotImplementedError, match="reserved for H-B"):
        HarnessPolicy(cs)


def test_policy_accepts_empty_set() -> None:
    HarnessPolicy(ComponentSet(frozenset()))


def test_policy_accepts_summary_only() -> None:
    HarnessPolicy(ComponentSet(frozenset({HarnessComponent.SUMMARY})))


# ---------------------------------------------------------------------------
# HarnessPolicy.blend
# ---------------------------------------------------------------------------


def test_blend_returns_unchanged_when_passthrough() -> None:
    policy = HarnessPolicy(ComponentSet(frozenset()))
    messages = [{"role": "user", "content": "hi"}]
    out = policy.blend(
        scope_key="user-a",
        session_id="s1",
        messages=messages,
        prior_summaries=(),
    )
    assert out.blended is False
    assert out.messages == [{"role": "user", "content": "hi"}]
    # Original list must not be mutated.
    assert messages == [{"role": "user", "content": "hi"}]


def test_blend_with_summary_but_no_prior_returns_unchanged() -> None:
    policy = HarnessPolicy(ComponentSet(frozenset({HarnessComponent.SUMMARY})))
    messages = [{"role": "user", "content": "hi"}]
    out = policy.blend(
        scope_key="user-a",
        session_id="s1",
        messages=messages,
        prior_summaries=(),
    )
    assert out.blended is False
    assert out.messages == [{"role": "user", "content": "hi"}]


def test_blend_inserts_system_prefix_when_no_existing_system() -> None:
    policy = HarnessPolicy(ComponentSet(frozenset({HarnessComponent.SUMMARY})))
    out = policy.blend(
        scope_key="user-a",
        session_id="s2",
        messages=[{"role": "user", "content": "how are you?"}],
        prior_summaries=[
            _summary(session_id="s1", topic="first session topic",
                     extracted_at="2026-05-15T00:00:00+00:00"),
        ],
    )
    assert out.blended is True
    assert out.messages[0]["role"] == "system"
    sys_content = out.messages[0]["content"]
    assert SYSTEM_PREFIX_HEADER in sys_content
    assert SYSTEM_PREFIX_FOOTER in sys_content
    assert "first session topic" in sys_content
    assert "session s1" in sys_content
    # The user turn must still be present and unchanged.
    assert out.messages[1] == {"role": "user", "content": "how are you?"}


def test_blend_prepends_to_existing_system_message() -> None:
    policy = HarnessPolicy(ComponentSet(frozenset({HarnessComponent.SUMMARY})))
    out = policy.blend(
        scope_key="user-a",
        session_id="s2",
        messages=[
            {"role": "system", "content": "you are a long-running companion AI"},
            {"role": "user", "content": "how are you?"},
        ],
        prior_summaries=[
            _summary(session_id="s1", topic="moved to new flat",
                     extracted_at="2026-05-15T00:00:00+00:00"),
        ],
    )
    assert out.blended is True
    assert out.messages[0]["role"] == "system"
    sys_content = out.messages[0]["content"]
    # Both the harness prefix and the original system prompt are present.
    assert SYSTEM_PREFIX_HEADER in sys_content
    assert "you are a long-running companion AI" in sys_content
    # Order: harness prefix comes first so the model sees memory before persona.
    header_idx = sys_content.index(SYSTEM_PREFIX_HEADER)
    persona_idx = sys_content.index("you are a long-running companion AI")
    assert header_idx < persona_idx


def test_blend_caps_at_max_prior_summaries() -> None:
    policy = HarnessPolicy(ComponentSet(frozenset({HarnessComponent.SUMMARY})))
    summaries = [
        _summary(
            session_id=f"s{i}",
            topic=f"topic-{i}",
            extracted_at=f"2026-05-{i+1:02d}T00:00:00+00:00",
        )
        for i in range(MAX_PRIOR_SUMMARIES + 3)
    ]
    out = policy.blend(
        scope_key="user-a",
        session_id="latest",
        messages=[{"role": "user", "content": "hi"}],
        prior_summaries=summaries,
    )
    sys_content = out.messages[0]["content"]
    # Only the most-recent MAX_PRIOR_SUMMARIES should appear; the first
    # three (oldest) must be dropped.
    assert "topic-0" not in sys_content
    assert "topic-1" not in sys_content
    assert "topic-2" not in sys_content
    assert "topic-3" in sys_content
    # The most recent one must appear.
    assert f"topic-{MAX_PRIOR_SUMMARIES + 2}" in sys_content


def test_blend_does_not_mutate_input_messages() -> None:
    policy = HarnessPolicy(ComponentSet(frozenset({HarnessComponent.SUMMARY})))
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
    ]
    snapshot = [dict(m) for m in messages]
    policy.blend(
        scope_key="user-a",
        session_id="s2",
        messages=messages,
        prior_summaries=[
            _summary(session_id="s1", topic="t",
                     extracted_at="2026-05-15T00:00:00+00:00"),
        ],
    )
    assert messages == snapshot
