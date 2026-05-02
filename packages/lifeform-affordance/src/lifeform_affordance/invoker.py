"""AffordanceInvoker \u2014 slice 2a execution bridge.

Ties together:

* ``AffordanceRegistry`` (slice 1) \u2014 the registered descriptor set
* a **backend registry** \u2014 host-supplied callables that actually
  run each affordance (stdlib dispatch, HTTP, shell, whatever)
* a **boundary policy** \u2014 pluggable safety gate that decides
  whether a given ``(descriptor, parameters, context)`` triple is
  allowed right now
* a **rate limiter** \u2014 per-descriptor sliding window to match
  ``cost_model.rate_limit_per_minute``
* a **parameter validator** \u2014 narrow JSON-Schema subset enforcer
  (type + required keys) with stdlib only

Design constraints:

1. **Read-only on kernel state.** The invoker calls a supplied
   ``session.submit_tool_result`` to feed results back, but it
   never touches any owner store. Host code that doesn't want
   kernel integration can omit the session entirely; the invoker
   still returns an ``AffordanceInvocationResult`` for the caller.
2. **Fail-loud safety gates.** Blocking conditions raise typed
   exceptions; a "silent maybe-blocked" result would paper over
   misconfiguration. Callers that want to fall through a block
   catch the specific exception.
3. **Deterministic ordering in ``invoke``.** Every invocation
   goes through the same five stages in the same order: lookup
   \u2192 safety gate \u2192 rate limit \u2192 parameter validation \u2192 backend
   call \u2192 kernel result wiring. A future audit trail can record
   exactly which stage a failure occurred at.
4. **Backends are async callables.** Sync backends wrap via
   ``functools.partial`` + ``asyncio.to_thread`` at the host;
   the invoker contract is single-signature
   ``async (parameters: Mapping[str, Any]) -> AffordancePayload``.

The invoker itself is stateless w.r.t. affordance history; it
holds the registry / backend map / rate-limiter state / boundary
policy as immutable configuration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from volvence_zero.affordance import AffordanceDescriptor, AffordanceKind

from lifeform_affordance.registry import AffordanceRegistry


_LOG = logging.getLogger("lifeform_affordance.invoker")


AffordanceBackend = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any] | None]]
"""Host-supplied async callable for one affordance.

Receives the validated parameter mapping. Returns the tool's
result payload (JSON-serialisable mapping) or ``None`` when the
backend has no structured output. Exceptions are caught by the
invoker and turned into ``AffordanceInvocationResult(status=FAILED,
error_class=...)`` records.
"""


class AffordanceInvocationStatus(str, Enum):
    """Terminal status of one invoke call.

    * ``SUCCEEDED`` \u2014 the backend returned cleanly.
    * ``DENIED_BY_BOUNDARY`` \u2014 boundary policy vetoed
      (regime-blocked, missing consent, confirmation required).
    * ``RATE_LIMITED`` \u2014 the per-affordance rate limit fired.
    * ``PARAMETER_INVALID`` \u2014 the ``parameters_schema`` gate
      rejected the supplied parameters.
    * ``BACKEND_MISSING`` \u2014 no backend registered for this
      descriptor name.
    * ``BACKEND_FAILED`` \u2014 backend raised; details captured.
    * ``EXCLUDED`` \u2014 descriptor has
      ``excluded_from_runtime_selection=True`` and the invoker was
      called anyway (usually a caller bug).
    """

    SUCCEEDED = "succeeded"
    DENIED_BY_BOUNDARY = "denied_by_boundary"
    RATE_LIMITED = "rate_limited"
    PARAMETER_INVALID = "parameter_invalid"
    BACKEND_MISSING = "backend_missing"
    BACKEND_FAILED = "backend_failed"
    EXCLUDED = "excluded"


_TERMINAL_FAIL_STATUSES: frozenset[AffordanceInvocationStatus] = frozenset(
    {
        AffordanceInvocationStatus.DENIED_BY_BOUNDARY,
        AffordanceInvocationStatus.RATE_LIMITED,
        AffordanceInvocationStatus.PARAMETER_INVALID,
        AffordanceInvocationStatus.BACKEND_MISSING,
        AffordanceInvocationStatus.BACKEND_FAILED,
        AffordanceInvocationStatus.EXCLUDED,
    }
)


@dataclass(frozen=True)
class AffordanceInvocationResult:
    """Immutable audit record for one invoke call.

    ``payload`` is populated only for SUCCEEDED; every failure
    status supplies ``error_class`` + ``error_detail`` with an
    operator-readable reason. ``tool_event_ids`` is non-empty when
    the result was routed through ``session.submit_tool_result``;
    callers that bypass session wiring see it empty.
    """

    descriptor_name: str
    status: AffordanceInvocationStatus
    payload: Mapping[str, Any] | None = None
    error_class: str = ""
    error_detail: str = ""
    tool_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        is_success = self.status is AffordanceInvocationStatus.SUCCEEDED
        if is_success and (self.error_class or self.error_detail):
            raise ValueError(
                "AffordanceInvocationResult: error_class/error_detail "
                "must be empty when status is SUCCEEDED"
            )
        if (not is_success) and not self.error_class.strip():
            raise ValueError(
                f"AffordanceInvocationResult: error_class must be non-empty "
                f"for non-SUCCEEDED status {self.status.value!r}"
            )


class AffordanceInvocationError(Exception):
    """Base class for invoker errors when the caller wants to
    handle failure via try/except rather than inspect
    ``AffordanceInvocationResult.status``.

    Invoker entry points return ``AffordanceInvocationResult`` by
    default; explicit ``invoke_or_raise`` variant converts a
    non-SUCCEEDED result into this exception so callers can push
    control flow up instead of threading status checks.
    """

    def __init__(self, result: AffordanceInvocationResult) -> None:
        super().__init__(
            f"{result.descriptor_name}: {result.status.value} "
            f"({result.error_class}: {result.error_detail})"
        )
        self.result = result


# ---------------------------------------------------------------------------
# Boundary policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryCheckContext:
    """Minimal context the boundary policy sees at invoke time.

    Deliberately narrow: we pass the descriptor + the claimed
    consent grants + the active regime id. Richer context (full
    snapshots) should go through a custom ``BoundaryPolicy``
    subclass.
    """

    descriptor: AffordanceDescriptor
    granted_consents: frozenset[str]
    active_regime_id: str | None
    user_confirmed: bool


@dataclass(frozen=True)
class BoundaryDenial:
    """Why a boundary denied an invocation."""

    reason_code: str
    detail: str


class BoundaryPolicy(Protocol):
    """Pluggable safety gate protocol.

    Return ``None`` to allow the invocation; return a
    ``BoundaryDenial`` to block it. The invoker calls this once
    per invoke, between lookup and rate-limit stages.
    """

    def check(self, context: BoundaryCheckContext) -> BoundaryDenial | None: ...


class DescriptorDerivedBoundaryPolicy:
    """Default policy that applies exactly the descriptor's safety model.

    Checks, in order:

    1. descriptor is not excluded_from_runtime_selection
    2. active_regime_id is NOT in safety_model.blocked_in_regimes
    3. every name in safety_model.requires_consent_grant is in
       ``granted_consents``
    4. if safety_model.requires_user_confirmation, then
       ``user_confirmed`` is True

    This is the sensible default for hosts that don't have their
    own richer policy. Advanced hosts build their own
    ``BoundaryPolicy`` subclass (e.g. quota-aware, tenant-aware).
    """

    def check(self, context: BoundaryCheckContext) -> BoundaryDenial | None:
        d = context.descriptor
        if d.excluded_from_runtime_selection:
            return BoundaryDenial(
                reason_code="descriptor_excluded",
                detail=(
                    f"{d.name!r} has excluded_from_runtime_selection=True; "
                    f"invoker should not be called for this descriptor."
                ),
            )
        if (
            context.active_regime_id is not None
            and context.active_regime_id in d.safety_model.blocked_in_regimes
        ):
            return BoundaryDenial(
                reason_code="regime_blocked",
                detail=(
                    f"{d.name!r} is blocked in regime "
                    f"{context.active_regime_id!r}; safety_model."
                    f"blocked_in_regimes={d.safety_model.blocked_in_regimes!r}"
                ),
            )
        missing = [
            grant
            for grant in d.safety_model.requires_consent_grant
            if grant not in context.granted_consents
        ]
        if missing:
            return BoundaryDenial(
                reason_code="consent_not_granted",
                detail=(
                    f"{d.name!r} requires consent grant(s) {missing!r} "
                    f"but granted={sorted(context.granted_consents)!r}"
                ),
            )
        if d.safety_model.requires_user_confirmation and not context.user_confirmed:
            return BoundaryDenial(
                reason_code="confirmation_required",
                detail=(
                    f"{d.name!r} requires explicit user confirmation; "
                    f"host must surface a prompt + call invoke(..., "
                    f"user_confirmed=True) before proceeding."
                ),
            )
        return None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _PerAffordanceRateLimiter:
    """Sliding 60-second window, per descriptor name.

    Sliding rather than fixed-60s-window avoids the "burst at
    minute boundary" problem. Lightweight: uses a ``deque`` per
    descriptor so space is O(active limit). No global state.

    ``monotonic`` clock is used so test harnesses can inject a
    fake clock via ``_now``. Tests override ``_now`` to drive
    deterministic timing.
    """

    _WINDOW_SECONDS: float = 60.0

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._windows: dict[str, deque[float]] = {}

    def check_and_record(self, descriptor: AffordanceDescriptor) -> BoundaryDenial | None:
        """Return None to allow, a denial to block. Records the
        invocation timestamp on allow so subsequent calls see it.

        If the descriptor has no rate limit, this is always a
        no-op that returns None.
        """
        limit = descriptor.cost_model.rate_limit_per_minute
        if limit is None:
            return None
        now = self._clock()
        window = self._windows.setdefault(descriptor.name, deque())
        threshold = now - self._WINDOW_SECONDS
        while window and window[0] < threshold:
            window.popleft()
        if len(window) >= limit:
            oldest_in_window = window[0]
            return BoundaryDenial(
                reason_code="rate_limited",
                detail=(
                    f"{descriptor.name!r} hit rate_limit_per_minute="
                    f"{limit}; {len(window)} call(s) in the last 60s, "
                    f"oldest at t={oldest_in_window:.2f}"
                ),
            )
        window.append(now)
        return None


# ---------------------------------------------------------------------------
# Parameter validation (narrow JSON Schema subset)
# ---------------------------------------------------------------------------


_JSON_TYPE_PYTHON: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    # ``bool`` must come BEFORE ``int`` in the integer check because
    # True/False are also instances of int in Python. We special-case
    # that in ``_check_type``.
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _check_type(value: Any, json_type: str) -> bool:
    if json_type not in _JSON_TYPE_PYTHON:
        # Unsupported type in descriptor; fail-loud at validation time.
        raise ValueError(
            f"Unsupported JSON Schema type {json_type!r}. Supported: "
            f"{sorted(_JSON_TYPE_PYTHON)!r}"
        )
    expected = _JSON_TYPE_PYTHON[json_type]
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_parameters(
    descriptor: AffordanceDescriptor,
    parameters: Mapping[str, Any],
) -> BoundaryDenial | None:
    """Validate ``parameters`` against ``descriptor.parameters_schema``.

    Supports the narrow JSON Schema subset the affordance authoring
    path uses:

    * top-level ``type: "object"``
    * ``required: [keys]`` for required keys
    * ``properties: {name: {"type": T, ...}}`` for per-field types

    Returns ``None`` when valid, a ``BoundaryDenial`` with
    ``reason_code="parameter_invalid"`` when not. A ``BoundaryDenial``
    is used (not a Python exception) because this is a request
    validation result, not a bug.

    Unknown fields in ``parameters`` are accepted (JSON Schema
    default). To reject unknowns, the descriptor author can set
    ``additionalProperties: false``, and we honour it.
    """
    schema = descriptor.parameters_schema
    schema_type = schema.get("type")
    if schema_type != "object":
        raise ValueError(
            f"{descriptor.name!r}: parameters_schema.type must be "
            f"'object', got {schema_type!r}. The invoker only "
            f"validates object-shaped parameters."
        )
    required = tuple(schema.get("required", ()))
    properties: Mapping[str, Any] = schema.get("properties", {})
    for key in required:
        if key not in parameters:
            return BoundaryDenial(
                reason_code="parameter_invalid",
                detail=(
                    f"{descriptor.name!r}: required parameter "
                    f"{key!r} missing."
                ),
            )
    for key, value in parameters.items():
        prop_schema = properties.get(key)
        if prop_schema is None:
            if schema.get("additionalProperties") is False:
                return BoundaryDenial(
                    reason_code="parameter_invalid",
                    detail=(
                        f"{descriptor.name!r}: unknown parameter "
                        f"{key!r}; additionalProperties=False."
                    ),
                )
            continue
        prop_type = prop_schema.get("type")
        if prop_type is None:
            continue  # no type constraint
        if not _check_type(value, prop_type):
            return BoundaryDenial(
                reason_code="parameter_invalid",
                detail=(
                    f"{descriptor.name!r}: parameter {key!r} expected "
                    f"type {prop_type!r}, got "
                    f"{type(value).__name__} ({value!r})"
                ),
            )
    return None


# ---------------------------------------------------------------------------
# Session protocol
# ---------------------------------------------------------------------------


class _SessionLike(Protocol):
    """Structural protocol for the optional session the invoker
    feeds results into. Matches ``BrainSession`` exactly.
    """

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
        latency_ms: int | None = None,
        monetary_cost: float = 0.0,
        reversibility: str = "reversible",
        environment_state_delta_kind: str = "none",
    ) -> tuple[str, ...]: ...


# ---------------------------------------------------------------------------
# The invoker
# ---------------------------------------------------------------------------


class AffordanceInvoker:
    """Orchestrates one affordance invocation end-to-end.

    Construct once per lifeform session (or process, if the
    registry + boundary policy are session-agnostic). Then call
    ``invoke(...)`` per affordance call.
    """

    def __init__(
        self,
        *,
        registry: AffordanceRegistry,
        boundary_policy: BoundaryPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._boundary_policy = boundary_policy or DescriptorDerivedBoundaryPolicy()
        self._backends: dict[str, AffordanceBackend] = {}
        self._rate_limiter = _PerAffordanceRateLimiter(clock=clock)

    # ------------------------------------------------------------------
    # Backend registration
    # ------------------------------------------------------------------

    def register_backend(self, descriptor_name: str, backend: AffordanceBackend) -> None:
        """Register the async callable that implements ``descriptor_name``.

        Re-registration replaces the previous backend; this matches
        hot-reloading a plugin. Hosts that want "register once,
        fail on duplicate" can subclass and override.
        """
        # Catch the typo "register a backend for a name the registry
        # does not know about" early \u2014 otherwise it'd be caught
        # at invoke time as BACKEND_MISSING, which is less helpful.
        if descriptor_name not in self._registry:
            raise KeyError(
                f"Cannot register backend for unknown affordance "
                f"{descriptor_name!r}; descriptor must be registered "
                f"with AffordanceRegistry first. Known: "
                f"{sorted(self._registry.names())!r}"
            )
        self._backends[descriptor_name] = backend

    def unregister_backend(self, descriptor_name: str) -> bool:
        """Remove a backend. Returns True if one was present."""
        return self._backends.pop(descriptor_name, None) is not None

    def backend_names(self) -> tuple[str, ...]:
        """Names of descriptors that currently have a backend."""
        return tuple(sorted(self._backends))

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------

    async def invoke(
        self,
        descriptor_name: str,
        parameters: Mapping[str, Any],
        *,
        active_regime_id: str | None = None,
        granted_consents: frozenset[str] = frozenset(),
        user_confirmed: bool = False,
        session: _SessionLike | None = None,
        event_id: str | None = None,
        action_id: str | None = None,
    ) -> AffordanceInvocationResult:
        """Run the 5-stage invocation pipeline.

        Stages in order:

        1. Registry lookup. Unknown name => KeyError (caller bug).
        2. Boundary policy check. Denial => DENIED_BY_BOUNDARY.
        3. Rate limit. Denial => RATE_LIMITED.
        4. Parameter validation. Denial => PARAMETER_INVALID.
        5. Backend call. Missing => BACKEND_MISSING; exception =>
           BACKEND_FAILED; success => SUCCEEDED.

        When ``session`` is supplied, the result is also fed back
        through ``session.submit_tool_result`` so the kernel's
        execution_result / open_loop / belief_assumption owners
        see the outcome. ``event_id`` + ``action_id`` default to
        ``descriptor_name``-based values if not supplied.
        """
        descriptor = self._registry.get(descriptor_name)

        # Stage 2: boundary.
        boundary_context = BoundaryCheckContext(
            descriptor=descriptor,
            granted_consents=granted_consents,
            active_regime_id=active_regime_id,
            user_confirmed=user_confirmed,
        )
        boundary_denial = self._boundary_policy.check(boundary_context)
        if boundary_denial is not None:
            status = (
                AffordanceInvocationStatus.EXCLUDED
                if boundary_denial.reason_code == "descriptor_excluded"
                else AffordanceInvocationStatus.DENIED_BY_BOUNDARY
            )
            return self._finalize(
                descriptor=descriptor,
                status=status,
                session=session,
                event_id=event_id,
                action_id=action_id,
                error_class=boundary_denial.reason_code,
                error_detail=boundary_denial.detail,
            )

        # Stage 3: rate limit.
        rate_denial = self._rate_limiter.check_and_record(descriptor)
        if rate_denial is not None:
            return self._finalize(
                descriptor=descriptor,
                status=AffordanceInvocationStatus.RATE_LIMITED,
                session=session,
                event_id=event_id,
                action_id=action_id,
                error_class=rate_denial.reason_code,
                error_detail=rate_denial.detail,
            )

        # Stage 4: parameter validation.
        param_denial = validate_parameters(descriptor, parameters)
        if param_denial is not None:
            return self._finalize(
                descriptor=descriptor,
                status=AffordanceInvocationStatus.PARAMETER_INVALID,
                session=session,
                event_id=event_id,
                action_id=action_id,
                error_class=param_denial.reason_code,
                error_detail=param_denial.detail,
            )

        # Stage 5: backend call.
        backend = self._backends.get(descriptor_name)
        if backend is None:
            return self._finalize(
                descriptor=descriptor,
                status=AffordanceInvocationStatus.BACKEND_MISSING,
                session=session,
                event_id=event_id,
                action_id=action_id,
                error_class="backend_missing",
                error_detail=(
                    f"No backend registered for {descriptor_name!r}; "
                    f"call AffordanceInvoker.register_backend(name, fn)."
                ),
            )
        backend_started_at = time.monotonic()
        try:
            raw_payload = await backend(parameters)
        except Exception as exc:  # noqa: BLE001 \u2014 invoker isolation boundary
            _LOG.exception(
                "AffordanceInvoker: backend for %s raised",
                descriptor_name,
            )
            return self._finalize(
                descriptor=descriptor,
                status=AffordanceInvocationStatus.BACKEND_FAILED,
                session=session,
                event_id=event_id,
                action_id=action_id,
                latency_ms=int((time.monotonic() - backend_started_at) * 1000),
                error_class=type(exc).__name__,
                error_detail=str(exc)[:512],
            )
        payload: Mapping[str, Any] = (
            dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        )
        return self._finalize(
            descriptor=descriptor,
            status=AffordanceInvocationStatus.SUCCEEDED,
            session=session,
            event_id=event_id,
            action_id=action_id,
            payload=payload,
            latency_ms=int((time.monotonic() - backend_started_at) * 1000),
        )

    async def invoke_or_raise(
        self,
        descriptor_name: str,
        parameters: Mapping[str, Any],
        **invoke_kwargs: Any,
    ) -> AffordanceInvocationResult:
        """Variant of ``invoke`` that raises on non-SUCCEEDED status.

        Convenience for host code that prefers exception-based
        control flow. The raised ``AffordanceInvocationError``
        carries the full ``AffordanceInvocationResult`` for
        inspection.
        """
        result = await self.invoke(descriptor_name, parameters, **invoke_kwargs)
        if result.status is not AffordanceInvocationStatus.SUCCEEDED:
            raise AffordanceInvocationError(result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _finalize(
        self,
        *,
        descriptor: AffordanceDescriptor,
        status: AffordanceInvocationStatus,
        session: _SessionLike | None,
        event_id: str | None,
        action_id: str | None,
        payload: Mapping[str, Any] | None = None,
        latency_ms: int | None = None,
        error_class: str = "",
        error_detail: str = "",
    ) -> AffordanceInvocationResult:
        """Build the result + optionally feed it to ``session.submit_tool_result``.

        Tool-bus wiring policy:

        * ``SUCCEEDED`` \u2014 the backend ran. The kernel MUST see the
          outcome so ``execution_result.completed_actions`` /
          ``belief_assumption`` reflect reality.
        * ``BACKEND_FAILED`` \u2014 the backend ran and raised. This is
          real tool signal; the kernel sees it as
          ``execution_result.failed_actions``.
        * Everything else (BOUNDARY / RATE_LIMITED / PARAMETER_INVALID
          / BACKEND_MISSING / EXCLUDED) \u2014 nothing reached the
          backend. These pre-flight gates did NOT invoke the tool,
          so leaking them onto the tool-result bus would lie to
          the kernel about what was attempted. Silent to the kernel;
          fully auditable on the returned ``AffordanceInvocationResult``.

        Only ``AffordanceKind.TOOL`` results ever flow through
        ``submit_tool_result`` regardless of status \u2014 ACTION /
        ORGAN / SHELL results do not match that event shape.
        """
        ran_backend = status in {
            AffordanceInvocationStatus.SUCCEEDED,
            AffordanceInvocationStatus.BACKEND_FAILED,
        }
        tool_event_ids: tuple[str, ...] = ()
        if (
            session is not None
            and descriptor.kind is AffordanceKind.TOOL
            and event_id is not None
            and ran_backend
        ):
            tool_action_id = action_id or f"{descriptor.name}:{event_id}"
            summary, detail = _summarise_for_kernel(
                status=status,
                payload=payload,
                error_class=error_class,
                error_detail=error_detail,
            )
            tool_event_ids = session.submit_tool_result(
                event_id=event_id,
                tool_name=descriptor.name,
                action_id=tool_action_id,
                status=(
                    "succeeded"
                    if status is AffordanceInvocationStatus.SUCCEEDED
                    else "failed"
                ),
                summary=summary,
                detail=detail,
                confidence=1.0 if status is AffordanceInvocationStatus.SUCCEEDED else 0.6,
                latency_ms=latency_ms,
                monetary_cost=_monetary_cost_from_descriptor(descriptor),
                reversibility=(
                    "irreversible"
                    if descriptor.safety_model.irreversible
                    else "reversible"
                ),
                environment_state_delta_kind=descriptor.kind.value,
            )
        return AffordanceInvocationResult(
            descriptor_name=descriptor.name,
            status=status,
            payload=payload if status is AffordanceInvocationStatus.SUCCEEDED else None,
            error_class=error_class,
            error_detail=error_detail,
            tool_event_ids=tool_event_ids,
        )


def _monetary_cost_from_descriptor(descriptor: AffordanceDescriptor) -> float:
    monetary_class = descriptor.cost_model.monetary_class
    value = monetary_class.value if hasattr(monetary_class, "value") else str(monetary_class)
    return {
        "free": 0.0,
        "low": 0.25,
        "medium": 0.5,
        "high": 1.0,
    }.get(value, 0.0)


def _summarise_for_kernel(
    *,
    status: AffordanceInvocationStatus,
    payload: Mapping[str, Any] | None,
    error_class: str,
    error_detail: str,
    max_chars: int = 320,
) -> tuple[str, str]:
    """Build ``(summary, detail)`` suitable for ``submit_tool_result``.

    Kernel semantic adapters rely on the tool result fields having
    real content; an empty summary looks like a no-op and can
    confuse the execution_result owner's classification.
    """
    if status is AffordanceInvocationStatus.SUCCEEDED:
        summary = "affordance invocation succeeded"
        if payload:
            # Keep the detail short + JSON-ish so the adapter can
            # treat it as evidence without parsing every backend.
            text_preview = ", ".join(f"{k}={payload[k]!r}" for k in list(payload)[:4])
            detail = f"payload: {text_preview}"[:max_chars]
        else:
            detail = "payload: (empty)"
        return summary, detail
    summary = f"affordance invocation {status.value}"
    detail = f"{error_class}: {error_detail}"[:max_chars]
    return summary, detail


__all__ = [
    "AffordanceBackend",
    "AffordanceInvocationError",
    "AffordanceInvocationResult",
    "AffordanceInvocationStatus",
    "AffordanceInvoker",
    "BoundaryCheckContext",
    "BoundaryDenial",
    "BoundaryPolicy",
    "DescriptorDerivedBoundaryPolicy",
    "validate_parameters",
]
