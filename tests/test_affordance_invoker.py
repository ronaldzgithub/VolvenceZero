"""AffordanceInvoker unit + integration tests (Gap 1 slice 2a).

Covers the 5-stage pipeline (lookup \u2192 boundary \u2192 rate limit \u2192
parameter validation \u2192 backend) plus the optional
``submit_tool_result`` kernel wiring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from lifeform_affordance import (
    AffordanceBackend,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceInvocationError,
    AffordanceInvocationResult,
    AffordanceInvocationStatus,
    AffordanceInvoker,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceRegistry,
    AffordanceSafety,
    BoundaryCheckContext,
    BoundaryDenial,
    DescriptorDerivedBoundaryPolicy,
    validate_parameters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_HINT = (
    "This is a hint long enough to clear the 50-character minimum; "
    "production descriptors carry much richer copy."
)


def _tool(
    name: str,
    *,
    kind: AffordanceKind = AffordanceKind.TOOL,
    excluded: bool = False,
    safety: AffordanceSafety | None = None,
    rate_limit: int | None = None,
    parameters_schema: Mapping[str, Any] | None = None,
) -> AffordanceDescriptor:
    default_schema: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "n": {"type": "integer"},
        },
        "required": ["q"],
    }
    return AffordanceDescriptor(
        name=name,
        kind=kind,
        version="0.1.0",
        display_name=name.title(),
        description=f"Test tool {name}.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " (negative).",
        parameters_schema=parameters_schema or default_schema,
        output_schema={"type": "object"},
        cost_model=AffordanceCost(
            latency_class=AffordanceLatencyClass.FAST,
            monetary_class=AffordanceMonetaryClass.FREE,
            rate_limit_per_minute=rate_limit,
        ),
        safety_model=safety or AffordanceSafety(),
        excluded_from_runtime_selection=excluded,
    )


def _registry_with(*descriptors: AffordanceDescriptor) -> AffordanceRegistry:
    r = AffordanceRegistry()
    r.register_all(list(descriptors))
    return r


def _echo_backend(payload_key: str = "echoed") -> AffordanceBackend:
    async def backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        return {payload_key: dict(parameters)}

    return backend


async def _boom_backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    raise RuntimeError(f"simulated failure for params={dict(parameters)!r}")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_invoke_success_returns_payload() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())

    result = await invoker.invoke("ping", {"q": "hi"})
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert result.payload == {"echoed": {"q": "hi"}}
    assert result.error_class == ""
    assert result.error_detail == ""
    assert result.tool_event_ids == ()  # no session supplied


async def test_invoke_or_raise_raises_on_failure() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    # No backend -> BACKEND_MISSING.
    with pytest.raises(AffordanceInvocationError) as excinfo:
        await invoker.invoke_or_raise("ping", {"q": "hi"})
    assert excinfo.value.result.status is AffordanceInvocationStatus.BACKEND_MISSING


async def test_invoke_or_raise_returns_result_on_success() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    result = await invoker.invoke_or_raise("ping", {"q": "ok"})
    assert result.status is AffordanceInvocationStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Lookup / registration
# ---------------------------------------------------------------------------


def test_register_backend_rejects_unknown_descriptor_name() -> None:
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(KeyError, match="unknown"):
        invoker.register_backend("not-registered", _echo_backend())


async def test_invoke_unknown_name_raises_key_error() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(KeyError):
        await invoker.invoke("no-such-tool", {})


def test_register_backend_replaces_previous() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)

    async def backend_a(_p):
        return {"who": "A"}

    async def backend_b(_p):
        return {"who": "B"}

    invoker.register_backend("ping", backend_a)
    invoker.register_backend("ping", backend_b)
    assert "ping" in invoker.backend_names()


def test_unregister_backend_reports_presence() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    assert invoker.unregister_backend("ping") is True
    assert invoker.unregister_backend("ping") is False
    assert invoker.backend_names() == ()


async def test_invoke_without_backend_returns_backend_missing() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    result = await invoker.invoke("ping", {"q": "hi"})
    assert result.status is AffordanceInvocationStatus.BACKEND_MISSING
    assert result.error_class == "backend_missing"


# ---------------------------------------------------------------------------
# Boundary policy
# ---------------------------------------------------------------------------


async def test_boundary_denies_excluded_descriptor() -> None:
    registry = _registry_with(_tool("admin_tool", excluded=True))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("admin_tool", _echo_backend())
    result = await invoker.invoke("admin_tool", {"q": "hi"})
    assert result.status is AffordanceInvocationStatus.EXCLUDED
    assert result.error_class == "descriptor_excluded"


async def test_boundary_denies_regime_blocked() -> None:
    safety = AffordanceSafety(blocked_in_regimes=("casual_social",))
    registry = _registry_with(_tool("solver", safety=safety))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("solver", _echo_backend())
    result = await invoker.invoke(
        "solver",
        {"q": "hi"},
        active_regime_id="casual_social",
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "regime_blocked"


async def test_boundary_allows_when_regime_not_blocked() -> None:
    safety = AffordanceSafety(blocked_in_regimes=("casual_social",))
    registry = _registry_with(_tool("solver", safety=safety))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("solver", _echo_backend())
    result = await invoker.invoke(
        "solver",
        {"q": "hi"},
        active_regime_id="problem_solving",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED


async def test_boundary_denies_missing_consent() -> None:
    safety = AffordanceSafety(requires_consent_grant=("tool_use", "filesystem_read"))
    registry = _registry_with(_tool("reader", safety=safety))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("reader", _echo_backend())
    result = await invoker.invoke(
        "reader",
        {"q": "hi"},
        granted_consents=frozenset({"tool_use"}),  # missing filesystem_read
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "consent_not_granted"
    assert "filesystem_read" in result.error_detail


async def test_boundary_allows_when_all_consents_granted() -> None:
    safety = AffordanceSafety(requires_consent_grant=("tool_use",))
    registry = _registry_with(_tool("reader", safety=safety))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("reader", _echo_backend())
    result = await invoker.invoke(
        "reader",
        {"q": "hi"},
        granted_consents=frozenset({"tool_use"}),
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED


async def test_boundary_requires_confirmation_when_safety_says_so() -> None:
    safety = AffordanceSafety(requires_user_confirmation=True)
    registry = _registry_with(_tool("write_file", safety=safety))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("write_file", _echo_backend())

    denied = await invoker.invoke("write_file", {"q": "hi"})
    assert denied.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert denied.error_class == "confirmation_required"

    allowed = await invoker.invoke("write_file", {"q": "hi"}, user_confirmed=True)
    assert allowed.status is AffordanceInvocationStatus.SUCCEEDED


async def test_custom_boundary_policy_is_honoured() -> None:
    """A host-supplied policy can override / augment the default."""

    class _AlwaysDeny:
        def check(self, ctx: BoundaryCheckContext) -> BoundaryDenial | None:
            return BoundaryDenial(reason_code="tenant_quota_hit", detail="over budget")

    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(
        registry=registry,
        boundary_policy=_AlwaysDeny(),
    )
    invoker.register_backend("ping", _echo_backend())
    result = await invoker.invoke("ping", {"q": "hi"})
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "tenant_quota_hit"


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


async def test_rate_limit_fires_after_quota_exhausted() -> None:
    # Fake clock we can drive forward manually.
    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

    clock = _Clock()
    registry = _registry_with(_tool("limited", rate_limit=2))
    invoker = AffordanceInvoker(registry=registry, clock=clock)
    invoker.register_backend("limited", _echo_backend())

    # Two successes within the window.
    clock.t = 0.0
    ok1 = await invoker.invoke("limited", {"q": "a"})
    assert ok1.status is AffordanceInvocationStatus.SUCCEEDED
    clock.t = 1.0
    ok2 = await invoker.invoke("limited", {"q": "b"})
    assert ok2.status is AffordanceInvocationStatus.SUCCEEDED

    # Third within 60s -> rate limited.
    clock.t = 2.0
    limited = await invoker.invoke("limited", {"q": "c"})
    assert limited.status is AffordanceInvocationStatus.RATE_LIMITED
    assert limited.error_class == "rate_limited"

    # Advance beyond the window; the quota replenishes.
    clock.t = 65.0
    unblocked = await invoker.invoke("limited", {"q": "d"})
    assert unblocked.status is AffordanceInvocationStatus.SUCCEEDED


async def test_rate_limit_none_means_no_limit() -> None:
    registry = _registry_with(_tool("freebie"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("freebie", _echo_backend())
    # 5 calls in a row, no rate limit set \u2014 all succeed.
    for i in range(5):
        result = await invoker.invoke("freebie", {"q": f"q{i}"})
        assert result.status is AffordanceInvocationStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


async def test_invoke_rejects_missing_required_parameter() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    result = await invoker.invoke("ping", {"n": 3})  # missing required q
    assert result.status is AffordanceInvocationStatus.PARAMETER_INVALID
    assert "q" in result.error_detail


async def test_invoke_rejects_wrong_parameter_type() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    # q is typed as string; integer should be rejected.
    result = await invoker.invoke("ping", {"q": 42})
    assert result.status is AffordanceInvocationStatus.PARAMETER_INVALID
    assert "type" in result.error_detail.lower() or "expected" in result.error_detail.lower()


async def test_invoke_rejects_boolean_where_integer_is_required() -> None:
    # True is an instance of int in Python; the validator must
    # special-case bool to avoid silent coercion.
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    result = await invoker.invoke("ping", {"q": "hi", "n": True})
    assert result.status is AffordanceInvocationStatus.PARAMETER_INVALID


async def test_invoke_allows_unknown_parameter_by_default() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    # No additionalProperties constraint => unknown key "extra" is OK.
    result = await invoker.invoke("ping", {"q": "hi", "extra": "whatever"})
    assert result.status is AffordanceInvocationStatus.SUCCEEDED


async def test_invoke_rejects_unknown_parameter_when_schema_forbids() -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    registry = _registry_with(_tool("strict", parameters_schema=schema))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("strict", _echo_backend())
    result = await invoker.invoke("strict", {"q": "hi", "extra": "nope"})
    assert result.status is AffordanceInvocationStatus.PARAMETER_INVALID
    assert "extra" in result.error_detail


def test_validate_parameters_raises_on_unsupported_schema_type() -> None:
    """A descriptor author who writes ``type: "bananas"`` for a
    property triggers a loud raise at invoke time \u2014 no silent
    passthrough.
    """
    bad_schema = {
        "type": "object",
        "properties": {"q": {"type": "bananas"}},
    }
    descriptor = _tool("bad", parameters_schema=bad_schema)
    with pytest.raises(ValueError, match="Unsupported JSON Schema type"):
        validate_parameters(descriptor, {"q": "hi"})


def test_validate_parameters_rejects_non_object_top_level() -> None:
    bad_schema = {"type": "string"}
    descriptor = _tool("bad", parameters_schema=bad_schema)
    with pytest.raises(ValueError, match="'object'"):
        validate_parameters(descriptor, {"q": "hi"})


# ---------------------------------------------------------------------------
# Backend failure
# ---------------------------------------------------------------------------


async def test_backend_exception_becomes_backend_failed() -> None:
    registry = _registry_with(_tool("boom"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("boom", _boom_backend)
    result = await invoker.invoke("boom", {"q": "trigger"})
    assert result.status is AffordanceInvocationStatus.BACKEND_FAILED
    assert result.error_class == "RuntimeError"
    assert "simulated failure" in result.error_detail


async def test_backend_returning_non_mapping_yields_empty_payload() -> None:
    async def backend(_p):
        return "not a mapping"

    registry = _registry_with(_tool("weird"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("weird", backend)
    result = await invoker.invoke("weird", {"q": "hi"})
    # Status succeeds (backend did not raise) but payload is
    # normalised to an empty mapping because the backend broke
    # the return contract.
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert result.payload == {}


# ---------------------------------------------------------------------------
# Kernel wiring (submit_tool_result)
# ---------------------------------------------------------------------------


@dataclass
class _ToolCall:
    event_id: str
    tool_name: str
    action_id: str
    status: str
    summary: str
    detail: str
    confidence: float


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[_ToolCall] = []

    def submit_tool_result(
        self,
        *,
        event_id: str,
        tool_name: str,
        action_id: str,
        status: str,
        summary: str,
        detail: str,
        confidence: float = 0.8,
        artifact_refs: tuple[str, ...] = (),
        plan_ref: str | None = None,
    ) -> tuple[str, ...]:
        self.calls.append(
            _ToolCall(
                event_id=event_id,
                tool_name=tool_name,
                action_id=action_id,
                status=status,
                summary=summary,
                detail=detail,
                confidence=confidence,
            )
        )
        return (f"semantic:{event_id}",)


async def test_success_feeds_submit_tool_result_when_session_supplied() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    session = _FakeSession()
    result = await invoker.invoke(
        "ping",
        {"q": "hi"},
        session=session,
        event_id="evt-1",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert result.tool_event_ids == ("semantic:evt-1",)
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call.tool_name == "ping"
    assert call.status == "succeeded"
    assert call.confidence == pytest.approx(1.0)


async def test_failure_also_feeds_submit_tool_result_with_failed_status() -> None:
    registry = _registry_with(_tool("boom"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("boom", _boom_backend)
    session = _FakeSession()
    result = await invoker.invoke(
        "boom",
        {"q": "trigger"},
        session=session,
        event_id="evt-fail",
    )
    assert result.status is AffordanceInvocationStatus.BACKEND_FAILED
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call.status == "failed"
    assert "RuntimeError" in call.detail


async def test_no_session_means_no_tool_event_ids() -> None:
    registry = _registry_with(_tool("ping"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("ping", _echo_backend())
    result = await invoker.invoke("ping", {"q": "hi"})
    assert result.tool_event_ids == ()


async def test_action_kind_does_not_feed_tool_result_bus() -> None:
    """Only AffordanceKind.TOOL results belong on the tool_result bus.
    ACTION / ORGAN / SHELL successes skip the session write.
    """
    registry = _registry_with(_tool("clarify", kind=AffordanceKind.ACTION))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("clarify", _echo_backend())
    session = _FakeSession()
    result = await invoker.invoke(
        "clarify",
        {"q": "are you sure?"},
        session=session,
        event_id="evt-action",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert session.calls == []
    assert result.tool_event_ids == ()


async def test_boundary_denial_does_not_call_backend() -> None:
    calls: list[Mapping[str, Any]] = []

    async def backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(parameters)
        return {}

    safety = AffordanceSafety(blocked_in_regimes=("casual_social",))
    registry = _registry_with(_tool("blocked", safety=safety))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("blocked", backend)
    result = await invoker.invoke(
        "blocked",
        {"q": "hi"},
        active_regime_id="casual_social",
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert calls == []


# ---------------------------------------------------------------------------
# Result invariants
# ---------------------------------------------------------------------------


def test_invocation_result_rejects_empty_error_class_on_failure() -> None:
    with pytest.raises(ValueError, match="error_class"):
        AffordanceInvocationResult(
            descriptor_name="x",
            status=AffordanceInvocationStatus.BACKEND_FAILED,
            error_class="",
            error_detail="",
        )


def test_invocation_result_rejects_error_fields_on_success() -> None:
    with pytest.raises(ValueError, match="SUCCEEDED"):
        AffordanceInvocationResult(
            descriptor_name="x",
            status=AffordanceInvocationStatus.SUCCEEDED,
            error_class="should-be-empty",
        )
