"""Unit tests for session-bridge id resolution and lifeform-mode dispatch.

These tests exercise the OpenAI → SessionManager mapping with a
fake SessionManager + fake LifeformSession, so they do not need a
real lifeform / Qwen substrate. The contract is:

* ``derive_session_id`` is deterministic on (model + system + first
  user message) and produces stable ids across calls within an arc.
* ``extract_user_input`` returns only the LATEST user message — the
  kernel's own memory carries earlier turns forward.
* ``lifeform_complete`` calls ``run_turn(user_input)`` exactly once
  per request, on a session keyed by the resolved id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lifeform_openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    GenerationConfig,
    LifeformCompletionResult,
    SessionResolution,
    derive_session_id,
    extract_user_input,
    lifeform_complete,
    reserved_metadata_keys,
)


# ---------------------------------------------------------------------------
# Fake LifeformSession + SessionManager
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    text: str = ""
    rationale_tags: tuple[str, ...] = ()


@dataclass
class _FakeRunResult:
    response: _FakeResponse
    active_regime: str | None = "acquaintance_building"
    active_abstract_action: str | None = "ground"
    active_snapshots: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeTurnSummary:
    pe_magnitude: float = 0.0


class _FakeLifeformSession:
    def __init__(self, *, session_id: str) -> None:
        self.session_id = session_id
        self.turn_summaries: list[_FakeTurnSummary] = []
        self.run_turn_calls: list[str] = []
        # The real LifeformSession always exposes an affordance invoker
        # (the service attaches a default MCP bundle). ``None`` here means
        # "no server-side tool loop", so non-tool turns go straight to
        # ``run_turn`` — which is what these bridge tests exercise.
        self.mcp_invoker = None

    async def run_turn(self, user_input: str) -> _FakeRunResult:
        self.run_turn_calls.append(user_input)
        self.turn_summaries.append(_FakeTurnSummary(pe_magnitude=0.42))
        return _FakeRunResult(
            response=_FakeResponse(
                text=f"<reply turn={len(self.run_turn_calls)} input={user_input!r}>",
                rationale_tags=("intent=ground", "stage=opener"),
            ),
        )


class _FakeSessionManager:
    """Minimal SessionManager protocol surface for the session-bridge tests.

    Implements the methods the bridge actually calls:
    ``has_session`` / ``get_session`` / ``create_session`` plus the
    ``vertical_name`` property. Same semantics as the real
    ``lifeform_service.SessionManager`` for the relevant behaviors.
    """

    def __init__(self, *, vertical_name: str = "companion") -> None:
        self.vertical_name = vertical_name
        self._sessions: dict[str, _FakeLifeformSession] = {}
        self._session_verticals: dict[str, str] = {}
        self.create_calls: list[tuple[str | None, str | None]] = []
        self.vertical_calls: list[str | None] = []
        # D22: record the per-call tenant_id the bridge plumbs through.
        self.tenant_calls: list[str | None] = []

    async def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def get_session(self, session_id: str) -> _FakeLifeformSession:
        return self._sessions[session_id]

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        template_id: str | None = None,  # noqa: ARG002 - parity with real signature
        tenant_id: str | None = None,
        vertical_name: str | None = None,
    ) -> _FakeLifeformSession:
        sid = session_id or "auto-fake"
        self.create_calls.append((sid, user_id))
        self.tenant_calls.append(tenant_id)
        self.vertical_calls.append(vertical_name)
        if sid in self._sessions:
            from lifeform_service import SessionAlreadyExistsError

            raise SessionAlreadyExistsError(sid)
        session = _FakeLifeformSession(session_id=sid)
        self._sessions[sid] = session
        self._session_verticals[sid] = vertical_name or self.vertical_name
        return session

    def vertical_name_for(self, session_id: str) -> str:
        return self._session_verticals[session_id]


def _request(
    *messages: tuple[str, str],
    model: str = "lifeform-companion@qwen2.5-1.5b",
    metadata: dict[str, str] | None = None,
    generation: GenerationConfig | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=tuple(ChatMessage(role=role, content=content) for role, content in messages),
        generation=generation or GenerationConfig(),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# derive_session_id
# ---------------------------------------------------------------------------


def test_derive_session_id_explicit_metadata_wins() -> None:
    request = _request(
        ("system", "warm"),
        ("user", "anything"),
        metadata={"session_id": "eqbench-arc-007"},
    )
    resolution = derive_session_id(request)
    assert resolution == SessionResolution(
        session_id="eqbench-arc-007", kind="explicit", derivation_input=""
    )


def test_derive_session_id_single_message_is_fresh_unique() -> None:
    request_a = _request(("user", "hello"))
    request_b = _request(("user", "hello"))
    res_a = derive_session_id(request_a)
    res_b = derive_session_id(request_b)
    assert res_a.kind == "fresh"
    assert res_b.kind == "fresh"
    assert res_a.session_id != res_b.session_id  # uniqueness for single-message
    assert res_a.session_id.startswith("auto-")


def test_derive_session_id_multi_turn_is_stable_across_calls() -> None:
    """Same arc → same session id even across calls (sticky reuse)."""

    request_turn_2 = _request(
        ("system", "warm"),
        ("user", "I feel low."),
        ("assistant", "Tell me more."),
        ("user", "My job."),
    )
    request_turn_3 = _request(
        ("system", "warm"),
        ("user", "I feel low."),
        ("assistant", "Tell me more."),
        ("user", "My job."),
        ("assistant", "What about it?"),
        ("user", "It's draining."),
    )
    res_2 = derive_session_id(request_turn_2)
    res_3 = derive_session_id(request_turn_3)
    assert res_2.kind == "derived"
    assert res_3.kind == "derived"
    # Same model + same system + same first user message → same session id.
    assert res_2.session_id == res_3.session_id


def test_derive_session_id_different_first_user_diverges() -> None:
    res_a = derive_session_id(
        _request(
            ("system", "warm"),
            ("user", "scenario A opening"),
            ("assistant", "..."),
            ("user", "follow up"),
        )
    )
    res_b = derive_session_id(
        _request(
            ("system", "warm"),
            ("user", "scenario B opening"),
            ("assistant", "..."),
            ("user", "follow up"),
        )
    )
    assert res_a.session_id != res_b.session_id


def test_derive_session_id_different_model_diverges() -> None:
    common_messages = (
        ("system", "warm"),
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "and?"),
    )
    res_a = derive_session_id(_request(*common_messages, model="lifeform-companion@qwen2.5-1.5b"))
    res_b = derive_session_id(_request(*common_messages, model="lifeform-companion@qwen2.5-7b"))
    assert res_a.session_id != res_b.session_id


def test_derive_session_id_different_vertical_diverges() -> None:
    common_messages = (
        ("system", "warm"),
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "and?"),
    )
    res_a = derive_session_id(
        _request(*common_messages),
        vertical_name="companion",
    )
    res_b = derive_session_id(
        _request(*common_messages),
        vertical_name="companion-cold",
    )
    assert res_a.session_id != res_b.session_id


def test_derive_session_id_different_system_diverges() -> None:
    res_a = derive_session_id(
        _request(
            ("system", "warm"),
            ("user", "same"),
            ("assistant", "..."),
            ("user", "..."),
        )
    )
    res_b = derive_session_id(
        _request(
            ("system", "stoic"),
            ("user", "same"),
            ("assistant", "..."),
            ("user", "..."),
        )
    )
    assert res_a.session_id != res_b.session_id


# ---------------------------------------------------------------------------
# extract_user_input
# ---------------------------------------------------------------------------


def test_extract_user_input_returns_latest_user_message() -> None:
    messages = (
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="latest"),
    )
    assert extract_user_input(messages) == "latest"


def test_extract_user_input_handles_assistant_at_end_as_continuation() -> None:
    """Last-message-is-assistant: harness wants a continuation hand-off."""
    messages = (
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="prefilled assistant text"),
    )
    assert extract_user_input(messages) == "prefilled assistant text"


def test_extract_user_input_rejects_system_at_end() -> None:
    messages = (
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="system", content="system at end"),
    )
    with pytest.raises(ValueError) as excinfo:
        extract_user_input(messages)
    assert str(excinfo.value).startswith("invalid_messages")


def test_extract_user_input_rejects_empty() -> None:
    with pytest.raises(ValueError):
        extract_user_input(())


# ---------------------------------------------------------------------------
# reserved_metadata_keys
# ---------------------------------------------------------------------------


def test_reserved_metadata_keys_includes_session_id_and_user_id() -> None:
    keys = reserved_metadata_keys()
    assert "session_id" in keys
    assert "user_id" in keys


def test_reserved_metadata_keys_includes_tenant_id() -> None:
    # D22: tenant_id is interpreted by the bridge (two-layer scope).
    assert "tenant_id" in reserved_metadata_keys()


# ---------------------------------------------------------------------------
# lifeform_complete (async)
# ---------------------------------------------------------------------------


async def test_lifeform_complete_creates_session_on_first_call() -> None:
    manager = _FakeSessionManager()
    request = _request(("user", "First contact."))
    result = await lifeform_complete(request=request, manager=manager)

    # New session was created.
    assert len(manager.create_calls) == 1
    sid_passed, user_id = manager.create_calls[0]
    assert sid_passed.startswith("auto-")
    assert user_id is None

    assert isinstance(result, LifeformCompletionResult)
    assert result.resolution.kind == "fresh"
    assert result.response.id.startswith("auto-")
    assert result.response.choices[0].message.role == "assistant"
    assert "input='First contact.'" in result.response.choices[0].message.content


async def test_lifeform_complete_reuses_session_across_arc_via_derived_id() -> None:
    """All three turns of one arc share the same derived session id.

    The harness sends progressively longer ``messages`` arrays for the
    same arc. Because (model, system, first_user_message) is constant
    across the three calls, ``derive_session_id`` returns the same
    derived id, and the SessionManager reuses the kernel session.
    """

    manager = _FakeSessionManager()

    common_opening = (
        ("system", "warm"),
        ("user", "I feel low."),
    )
    request_turn_1 = _request(*common_opening)
    request_turn_2 = _request(
        *common_opening,
        ("assistant", "<some assistant reply>"),
        ("user", "It's been a long week."),
    )
    request_turn_3 = _request(
        *common_opening,
        ("assistant", "<some assistant reply>"),
        ("user", "It's been a long week."),
        ("assistant", "<another reply>"),
        ("user", "What do I do?"),
    )

    res_1 = await lifeform_complete(request=request_turn_1, manager=manager)
    res_2 = await lifeform_complete(request=request_turn_2, manager=manager)
    res_3 = await lifeform_complete(request=request_turn_3, manager=manager)

    # All three resolutions are "derived" (they each have ≥ 2 messages
    # which never matches the len==1 fresh-mint branch).
    assert res_1.resolution.kind == "derived"
    assert res_2.resolution.kind == "derived"
    assert res_3.resolution.kind == "derived"

    # And — critically — they all share the same kernel session id.
    assert res_1.response.id == res_2.response.id == res_3.response.id

    # Only ONE kernel session was ever created.
    assert len(manager.create_calls) == 1

    # That single session saw three sequential run_turn calls — one
    # per harness call — and each run_turn received only the LATEST
    # user message, not the whole history.
    derived_session = manager._sessions[res_1.response.id]  # noqa: SLF001 - test internals
    assert derived_session.run_turn_calls == [
        "I feel low.",
        "It's been a long week.",
        "What do I do?",
    ]


async def test_lifeform_complete_single_message_opener_is_fresh_per_call() -> None:
    """A bare single-user-message request mints a fresh session each time."""

    manager = _FakeSessionManager()
    res_a = await lifeform_complete(
        request=_request(("user", "Hello there.")),
        manager=manager,
    )
    res_b = await lifeform_complete(
        request=_request(("user", "Hello there.")),
        manager=manager,
    )
    # Both are "fresh" — the documented behavior is that single-message
    # requests are independent (no implicit hash-based reuse). If the
    # harness wants reuse it must pass metadata.session_id explicitly.
    assert res_a.resolution.kind == "fresh"
    assert res_b.resolution.kind == "fresh"
    assert res_a.response.id != res_b.response.id
    assert len(manager.create_calls) == 2


async def test_lifeform_complete_explicit_session_id_overrides_derivation() -> None:
    manager = _FakeSessionManager()
    request = _request(
        ("user", "hi"),
        metadata={"session_id": "harness-arc-001"},
    )
    result = await lifeform_complete(request=request, manager=manager)
    assert result.resolution.kind == "explicit"
    assert result.response.id == "harness-arc-001"
    assert manager.create_calls[0][0] == "harness-arc-001"


async def test_lifeform_complete_passes_user_id_to_create_session() -> None:
    manager = _FakeSessionManager()
    request = _request(
        ("user", "hi"),
        metadata={"session_id": "harness-arc-002", "user_id": "alice"},
    )
    await lifeform_complete(request=request, manager=manager)
    assert manager.create_calls[0] == ("harness-arc-002", "alice")


async def test_lifeform_complete_passes_tenant_id_to_create_session() -> None:
    # D22: metadata.tenant_id is plumbed through to create_session so the
    # two-layer {tenant}:{end_user} scope partitions memory per tenant.
    manager = _FakeSessionManager()
    request = _request(
        ("user", "hi"),
        metadata={
            "session_id": "harness-arc-003",
            "user_id": "alice",
            "tenant_id": "brand_a",
        },
    )
    await lifeform_complete(request=request, manager=manager)
    assert manager.create_calls[0] == ("harness-arc-003", "alice")
    assert manager.tenant_calls[0] == "brand_a"


async def test_lifeform_complete_tenant_id_defaults_none_when_absent() -> None:
    manager = _FakeSessionManager()
    request = _request(("user", "hi"), metadata={"session_id": "no-tenant"})
    await lifeform_complete(request=request, manager=manager)
    assert manager.tenant_calls[0] is None


async def test_lifeform_complete_passes_selected_vertical_to_create_session() -> None:
    manager = _FakeSessionManager()
    request = _request(
        ("user", "hi"),
        metadata={"session_id": "vertical-arc"},
    )
    result = await lifeform_complete(
        request=request,
        manager=manager,
        vertical_name="companion-eta-off",
    )
    assert manager.vertical_calls[0] == "companion-eta-off"
    assert result.response.system_fingerprint == "lifeform:companion-eta-off"


async def test_lifeform_complete_rejects_explicit_cross_vertical_reuse() -> None:
    manager = _FakeSessionManager()
    request = _request(
        ("user", "hi"),
        metadata={"session_id": "shared-explicit"},
    )
    await lifeform_complete(
        request=request,
        manager=manager,
        vertical_name="companion",
    )
    with pytest.raises(ValueError, match="invalid_session_vertical_mismatch"):
        await lifeform_complete(
            request=request,
            manager=manager,
            vertical_name="companion-cold",
        )


async def test_lifeform_complete_sends_only_latest_user_message_to_kernel() -> None:
    """Critical: full history is NOT replayed into run_turn — kernel has its own memory."""

    manager = _FakeSessionManager()
    request = _request(
        ("system", "warm"),
        ("user", "old turn 1"),
        ("assistant", "..."),
        ("user", "old turn 2"),
        ("assistant", "..."),
        ("user", "latest turn"),
        metadata={"session_id": "explicit-test-arc"},
    )
    await lifeform_complete(request=request, manager=manager)
    session = manager._sessions["explicit-test-arc"]  # noqa: SLF001
    assert session.run_turn_calls == ["latest turn"]


async def test_lifeform_complete_fingerprint_includes_vertical_name() -> None:
    manager = _FakeSessionManager(vertical_name="companion-cold")
    request = _request(("user", "hi"))
    result = await lifeform_complete(request=request, manager=manager)
    assert result.response.system_fingerprint == "lifeform:companion-cold"


async def test_lifeform_complete_surfaces_telemetry_on_wrapper() -> None:
    manager = _FakeSessionManager()
    request = _request(("user", "hi"))
    result = await lifeform_complete(request=request, manager=manager)
    assert result.active_regime == "acquaintance_building"
    assert result.active_abstract_action == "ground"
    assert result.pe_magnitude == pytest.approx(0.42)
    assert "intent=ground" in result.rationale_tags


async def test_lifeform_complete_response_envelope_round_trips_to_json() -> None:
    manager = _FakeSessionManager()
    result = await lifeform_complete(
        request=_request(("user", "test")),
        manager=manager,
    )
    body = result.response.to_json()
    assert set(body.keys()) >= {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
    }
    assert body["object"] == "chat.completion"
