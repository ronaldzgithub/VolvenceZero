"""Integration tests for ``POST /v1/chat/completions`` over aiohttp.

These tests bind to an ephemeral port via ``aiohttp.test_utils``'s
test client and exercise the real router code (no mocking of the
adapter), but inject a fake SessionManager / fake substrate runtime
into the app so the tests do not require a real lifeform / Qwen.

Coverage:

* POST /v1/chat/completions default mode → lifeform path
* mode=raw via query param + via X-Compat-Mode header → raw path
* invalid bodies → 400 ``invalid_*``
* stream=true → OpenAI-shaped SSE stream
* raw mode with no runtime → 503
* unknown mode → 400 ``invalid_mode``
* response shape is byte-compatible with OpenAI Python client
* lifeform telemetry surfaces on response headers

We deliberately do NOT cover ``/v1/models`` — that route is owned
by lifeform-service (with its own substrate-provider schema).
EQ-Bench 3 and similar harnesses POST directly to
``/v1/chat/completions`` without consulting ``/v1/models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceInvoker,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceRegistry,
    AffordanceSafety,
)
from lifeform_openai_compat import add_openai_routes


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    text: str = "fake assistant reply"
    rationale_tags: tuple[str, ...] = ("intent=ground", "stage=opener")


@dataclass
class _ResponseAssemblyValue:
    expression_intent: str = "support-first"


@dataclass
class _FakeSnapshot:
    value: Any


def _default_active_snapshots() -> dict[str, Any]:
    return {"response_assembly": _FakeSnapshot(_ResponseAssemblyValue())}


@dataclass
class _FakeRunResult:
    response: _FakeResponse
    active_regime: str | None = "acquaintance_building"
    active_abstract_action: str | None = "ground"
    active_snapshots: dict[str, Any] = field(
        default_factory=_default_active_snapshots
    )


@dataclass
class _FakeTurnSummary:
    pe_magnitude: float = 0.13


class _FakeLifeformSession:
    def __init__(self, *, session_id: str) -> None:
        self.session_id = session_id
        self.turn_summaries: list[_FakeTurnSummary] = []
        self.run_turn_calls: list[str] = []
        self.submitted_tool_results: list[dict[str, Any]] = []
        self.mcp_invoker = _fake_invoker()

    @property
    def brain_session(self) -> "_FakeLifeformSession":
        return self

    @property
    def latest_active_snapshots(self) -> dict[str, Any]:
        return {}

    async def run_turn(self, user_input: str, **_: Any) -> _FakeRunResult:
        self.run_turn_calls.append(user_input)
        self.turn_summaries.append(_FakeTurnSummary())
        return _FakeRunResult(
            response=_FakeResponse(
                text=f"reply turn={len(self.run_turn_calls)} input={user_input!r}",
            )
        )

    def submit_tool_result(self, **kwargs: Any) -> tuple[str, ...]:
        self.submitted_tool_results.append(dict(kwargs))
        return (str(kwargs["event_id"]),)


def _fake_invoker() -> AffordanceInvoker:
    hint = (
        "Use this test tool when a structured OpenAI tool call requests it "
        "and the adapter needs a deterministic callable descriptor."
    )
    descriptor = AffordanceDescriptor(
        name="lookup_profile",
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Lookup Profile",
        description="Lookup fake profile data.",
        when_to_use=hint,
        when_not_to_use=hint + " Do not use outside OpenAI tool-call tests.",
        parameters_schema={
            "type": "object",
            "properties": {"user": {"type": "string"}},
            "required": ["user"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.FAST),
        safety_model=AffordanceSafety(),
    )
    registry = AffordanceRegistry()
    registry.register(descriptor)
    invoker = AffordanceInvoker(registry=registry)

    async def backend(parameters: dict[str, Any]) -> dict[str, Any]:
        return {"profile": parameters["user"]}

    invoker.register_backend("lookup_profile", backend)
    return invoker


@dataclass
class _FakeGenerationResult:
    text: str
    token_count: int


class _FakeRuntime:
    def __init__(self) -> None:
        self.model_id = "fake/qwen2.5-1.5b-instruct"
        self.runtime_origin = "test-fake"

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> _FakeGenerationResult:
        if "available_tools_json" in prompt:
            return _FakeGenerationResult(
                text=(
                    '{"decision":"invoke_tool","tool_name":"lookup_profile",'
                    '"parameters":{"user":"alice"},"rationale":"descriptor match"}'
                ),
                token_count=24,
            )
        return _FakeGenerationResult(
            text=f"raw reply: {prompt}",
            token_count=8,
        )


class _FakeSessionManager:
    def __init__(
        self,
        *,
        vertical_name: str = "companion",
        with_runtime: bool = True,
    ) -> None:
        self.vertical_name = vertical_name
        self.substrate_runtime: _FakeRuntime | None = (
            _FakeRuntime() if with_runtime else None
        )
        self._sessions: dict[str, _FakeLifeformSession] = {}

    async def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def get_session(self, session_id: str) -> _FakeLifeformSession:
        return self._sessions[session_id]

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,  # noqa: ARG002 - signature parity
        template_id: str | None = None,  # noqa: ARG002
        tenant_id: str | None = None,  # noqa: ARG002 - D22 signature parity
    ) -> _FakeLifeformSession:
        sid = session_id or "auto-fake"
        if sid in self._sessions:
            from lifeform_service import SessionAlreadyExistsError

            raise SessionAlreadyExistsError(sid)
        session = _FakeLifeformSession(session_id=sid)
        self._sessions[sid] = session
        return session


def _build_app(*, with_runtime: bool = True, vertical_name: str = "companion") -> web.Application:
    app = web.Application()
    app["session_manager"] = _FakeSessionManager(
        vertical_name=vertical_name, with_runtime=with_runtime
    )
    add_openai_routes(app)
    return app


@pytest.fixture
async def lifeform_client(aiohttp_client):
    return await aiohttp_client(_build_app(with_runtime=True))


@pytest.fixture
async def synthetic_client(aiohttp_client):
    # No shared runtime — raw mode should 503.
    return await aiohttp_client(_build_app(with_runtime=False))


# ---------------------------------------------------------------------------
# Adapter does NOT mount /v1/models — that route is owned by
# lifeform-service. The fake SessionManager-only fixture has no
# /v1/models handler, so the route should 404.
# ---------------------------------------------------------------------------


async def test_adapter_does_not_mount_v1_models(lifeform_client) -> None:
    """Confirms the adapter is /v1/chat/completions only.

    Background: lifeform-service has its own ``/v1/models`` route
    with a vertical-specific schema. To avoid a route collision,
    the adapter intentionally does not mount its own. EQ-Bench 3
    does not hit ``/v1/models`` so the lifeform-service route
    suffices for harness compatibility.
    """
    resp = await lifeform_client.get("/v1/models")
    assert resp.status == 404


# ---------------------------------------------------------------------------
# Chat completions — happy paths
# ---------------------------------------------------------------------------


async def test_default_mode_is_lifeform(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "Hello there."}],
        },
    )
    assert resp.status == 200
    assert resp.headers["x-lifeform-mode"] == "lifeform"
    body = await resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "Hello there." in body["choices"][0]["message"]["content"]


async def test_lifeform_mode_surfaces_telemetry_headers(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "How are you?"}],
        },
    )
    assert resp.status == 200
    assert resp.headers["x-lifeform-regime"] == "acquaintance_building"
    assert resp.headers["x-lifeform-abstract-action"] == "ground"
    assert resp.headers["x-lifeform-pe-magnitude"] == "0.1300"
    assert resp.headers["x-lifeform-session-resolution"] == "fresh"
    assert "intent=ground" in resp.headers["x-lifeform-rationale-tags"]
    # Published expression intent (subjectivity-transport fix): lets
    # OpenAI-compat consumers drive the presence avatar from the core's
    # real intent.
    assert resp.headers["x-lifeform-expression-intent"] == "support-first"


async def test_openai_forced_tool_choice_returns_tool_calls(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "lookup this profile"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_profile",
                        "description": "Lookup fake profile data.",
                        "parameters": {
                            "type": "object",
                            "properties": {"user": {"type": "string"}},
                            "required": ["user"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "lookup_profile",
                    "arguments": {"user": "alice"},
                },
            },
        },
    )

    assert resp.status == 200
    body = await resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "lookup_profile"
    assert '"alice"' in tool_call["function"]["arguments"]


async def test_openai_tool_role_message_submits_tool_result(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "metadata": {"session_id": "tool-session"},
            "messages": [
                {"role": "user", "content": "lookup this profile"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "lookup_profile",
                                "arguments": "{\"user\":\"alice\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "{\"profile\":\"alice\"}",
                },
            ],
        },
    )

    assert resp.status == 200
    body = await resp.json()
    assert "submitted tool result" in body["choices"][0]["message"]["content"]


async def test_server_side_conversation_can_drive_tool_loop(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "metadata": {
                "session_id": "chat-tool-session",
                "dlaas.tool_loop": "server",
            },
            "messages": [{"role": "user", "content": "Please look up Alice's profile."}],
        },
    )

    assert resp.status == 200
    body = await resp.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "Continue the turn using" in body["choices"][0]["message"]["content"]


async def test_raw_mode_via_query_param(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions?mode=raw",
        json={
            "model": "lifeform-companion-raw",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert resp.status == 200
    assert resp.headers["x-lifeform-mode"] == "raw"
    body = await resp.json()
    assert body["choices"][0]["message"]["content"] == "raw reply: ping"
    assert body["system_fingerprint"].startswith("raw-substrate:")


async def test_raw_mode_via_header(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion-raw",
            "messages": [{"role": "user", "content": "ping"}],
        },
        headers={"X-Compat-Mode": "raw"},
    )
    assert resp.status == 200
    assert resp.headers["x-lifeform-mode"] == "raw"


async def test_raw_mode_with_no_substrate_returns_503(synthetic_client) -> None:
    resp = await synthetic_client.post(
        "/v1/chat/completions?mode=raw",
        json={
            "model": "lifeform-companion-raw",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"]["code"] == "raw_substrate_unavailable"
    assert "synthetic" in body["error"]["message"]


async def test_lifeform_mode_works_without_substrate(synthetic_client) -> None:
    """lifeform mode is independent of the shared runtime."""
    resp = await synthetic_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200


async def test_sticky_session_reuse_across_three_arc_turns(lifeform_client) -> None:
    arc_messages = [
        {"role": "system", "content": "Be warm."},
        {"role": "user", "content": "I'm overwhelmed."},
    ]
    resp1 = await lifeform_client.post(
        "/v1/chat/completions",
        json={"model": "lifeform-companion", "messages": arc_messages},
    )
    assert resp1.status == 200
    body1 = await resp1.json()
    arc_messages.append(
        {"role": "assistant", "content": body1["choices"][0]["message"]["content"]}
    )
    arc_messages.append({"role": "user", "content": "Where do I start?"})
    resp2 = await lifeform_client.post(
        "/v1/chat/completions",
        json={"model": "lifeform-companion", "messages": arc_messages},
    )
    assert resp2.status == 200
    body2 = await resp2.json()
    # Same session id across the arc → sticky reuse.
    assert body1["id"] == body2["id"]
    assert resp2.headers["x-lifeform-session-resolution"] == "derived"


async def test_explicit_session_id_via_metadata(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"session_id": "harness-arc-009"},
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["id"] == "harness-arc-009"
    assert resp.headers["x-lifeform-session-resolution"] == "explicit"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_missing_model_returns_400_invalid_model(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["code"] == "invalid_model"


async def test_empty_messages_returns_400_invalid_messages(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={"model": "lifeform-companion", "messages": []},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["code"] == "invalid_messages"


async def test_invalid_role_returns_400(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "wizard", "content": "hi"}],
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["code"] == "invalid_role_at_0"


async def test_streaming_returns_openai_sse(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    text = await resp.text()
    assert "data: " in text
    assert '"delta": {"role": "assistant"}' in text
    assert "reply turn=1 input='hi'" in text
    assert "data: [DONE]" in text


async def test_invalid_mode_returns_400(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions?mode=hybrid",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["code"] == "invalid_mode"


async def test_empty_body_returns_400(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        data="",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["code"] == "invalid_body"


async def test_malformed_json_returns_400(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        data="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"]["code"] == "invalid_body"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_double_mount_raises() -> None:
    app = _build_app(with_runtime=True)
    with pytest.raises(RuntimeError) as excinfo:
        add_openai_routes(app)
    assert "already mounted" in str(excinfo.value)


# ---------------------------------------------------------------------------
# OpenAI client compatibility — the response body must contain exactly the
# top-level keys the OpenAI Python client expects, so harnesses that use
# ``openai.OpenAI(base_url=...)`` work without bespoke parsing.
# ---------------------------------------------------------------------------


async def test_response_body_matches_openai_chat_completion_shape(lifeform_client) -> None:
    resp = await lifeform_client.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    body = await resp.json()
    assert set(body.keys()) >= {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
    }
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert set(choice.keys()) == {"index", "message", "finish_reason"}
    assert set(choice["message"].keys()) == {"role", "content"}
    assert set(body["usage"].keys()) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
