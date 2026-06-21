"""Session-bridge: OpenAI request → lifeform-service SessionManager.

This is the **stateful track** ("mode=lifeform") of the OpenAI-compat
adapter. Where :mod:`lifeform_openai_compat.raw_substrate` calls
``runtime.generate(...)`` directly and ignores the lifeform pipeline,
this module routes each request through a :class:`LifeformSession`
so the full lifeform stack (PromptPlanner / ResponseSynthesizer /
memory / regime / adaptive controllers) runs on every turn.

Key design problem: OpenAI's ``/v1/chat/completions`` is conceptually
stateless — the harness re-sends the entire conversation history on
every call. Our SessionManager is stateful — each ``session_id`` has
in-process kernel state for memory / regime / controllers. We need
to bridge these two contracts so:

1. A fresh single-user-message request (``len(messages) == 1`` with
   no ``metadata.session_id``) creates a new session and runs the
   first turn against it.
2. A continuation request (``len(messages) > 1``) for the same arc
   reuses the same kernel session, so memory and regime carry
   forward turn-to-turn.
3. The harness can force an explicit session via
   ``metadata.session_id`` — useful for testing cross-session
   memory by reusing a session id across "scenarios".
4. Two scenarios with similar opening prompts must NOT be treated
   as the same session (cross-scenario contamination).

Strategy: derive a deterministic auto session id from
(model + system_concat + first_user_message). Two requests with
identical opening map to the same kernel session; different openings
map to different sessions. Within an arc, the opening is stable
(harness keeps appending to the messages array), so the auto id is
stable too. Across arcs, different system prompts / different first
user messages produce different ids.

This module is read-only with respect to the kernel: it never
mutates SessionManager internals; never touches LifeformSession
private attrs; always reaches the kernel through SessionManager's
public ``create_session`` / ``get_session`` / ``has_session`` API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from lifeform_service import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
)

from lifeform_openai_compat.dto import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ChatToolCall,
    ChatToolCallFunction,
)
from lifeform_openai_compat.raw_substrate import (
    estimate_prompt_tokens,
    split_messages,
)


class SessionEndUserMismatchError(ValueError):
    """A reused session_id is bound to a different end-user than requested.

    Subclasses ``ValueError`` so existing router error handling still
    catches it; the router maps it to a 409 specifically.
    """


def _session_end_user_remap_allowed() -> bool:
    return os.environ.get("VZ_ALLOW_SESSION_END_USER_REMAP", "").strip() in (
        "1",
        "true",
        "True",
    )


_DEFAULT_MAX_NEW_TOKENS: int = 512
_AUTO_SESSION_ID_PREFIX: str = "auto-"
_AUTO_SESSION_ID_HEX_LEN: int = 24


# ---------------------------------------------------------------------------
# Session id derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionResolution:
    """Outcome of mapping an OpenAI request to a session id.

    ``kind`` annotates how the id was chosen:

    * ``"explicit"``: ``metadata.session_id`` was set; we use it
      verbatim. The harness gets exact control.
    * ``"fresh"``: single-user-message request, no metadata id.
      A fresh ``chatcmpl-style`` id is minted.
    * ``"derived"``: multi-turn or multi-message request without
      metadata id. The id is a stable SHA-256 hash of
      ``(model, system_concat, first_user_message)``.
    """

    session_id: str
    kind: str
    derivation_input: str = ""


def derive_session_id(request: ChatCompletionRequest) -> SessionResolution:
    """Translate an OpenAI request to the session id we'll use.

    See :class:`SessionResolution` for the three kinds of outcome.
    """

    explicit = request.metadata.get("session_id", "").strip()
    if explicit:
        return SessionResolution(session_id=explicit, kind="explicit")

    messages = request.messages
    if len(messages) == 1 and messages[0].role in {"user", "system", "developer"}:
        return SessionResolution(
            session_id=_fresh_auto_session_id(),
            kind="fresh",
        )

    system_context, _, _ = split_messages(messages)
    first_user_msg = next(
        (msg.content for msg in messages if msg.role == "user"),
        # Fallback: if no user message yet (e.g. assistant-prefilled
        # transcript), use the first message of any role.
        messages[0].content if messages else "",
    )
    derivation_input = f"{request.model}|{system_context}|{first_user_msg}"
    digest = hashlib.sha256(derivation_input.encode("utf-8")).hexdigest()
    return SessionResolution(
        session_id=f"{_AUTO_SESSION_ID_PREFIX}{digest[:_AUTO_SESSION_ID_HEX_LEN]}",
        kind="derived",
        derivation_input=derivation_input,
    )


def _fresh_auto_session_id() -> str:
    """Mint a non-colliding session id for a single-message request.

    We do NOT hash these because each ``len(messages) == 1`` request
    is conceptually a fresh conversation and should NOT collide with
    a previous fresh request that happened to send the same opener.
    Auto-derived hashing is only useful when the harness will send
    the SAME messages array again in a follow-up.
    """
    return f"{_AUTO_SESSION_ID_PREFIX}{uuid.uuid4().hex[:_AUTO_SESSION_ID_HEX_LEN]}"


# ---------------------------------------------------------------------------
# Last-user-message extraction
# ---------------------------------------------------------------------------


def extract_user_input(messages: tuple[ChatMessage, ...]) -> str:
    """Return the kernel-bound ``user_input`` for ``session.run_turn``.

    We send only the LATEST user message to the kernel — the kernel's
    own memory carries prior turns forward. Sending the full history
    on every turn would replay turns and corrupt the regime / memory
    state.

    Edge cases:

    * Last message is user: that is the input.
    * Last message is assistant or tool: a "model-please-continue"
      request from the harness side. The kernel does not have a
      first-class continue path; we treat the last assistant content
      as the input so the kernel sees something to ground on, and
      the regime owner will treat it as a regular turn.
    * Last message is system: we surface a typed error rather than
      silently treating system text as user input — that pattern
      almost always indicates a malformed harness request.
    """

    if not messages:
        raise ValueError("invalid_messages: at least one message required")
    last = messages[-1]
    if last.role in {"system", "developer"}:
        raise ValueError(
            "invalid_messages: last message must be user / assistant / tool, "
            "not system. The lifeform path needs an actionable user input "
            "(or a continuation hand-off from a prior assistant turn)."
        )
    return last.content


# ---------------------------------------------------------------------------
# Lifeform completion
# ---------------------------------------------------------------------------


def _expression_intent_from_result(result: object) -> str | None:
    """Read ``response_assembly.expression_intent`` off the kernel run
    result's published snapshots.

    Mirrors the DLaaS dispatch + readout-builder pattern: we read only
    the public snapshot value field, never owner internals, and treat
    a missing snapshot as ``None`` (the turn simply produced no
    response_assembly — e.g. a tool-only turn).
    """
    snapshots = getattr(result, "active_snapshots", None)
    if not snapshots:
        return None
    snapshot = snapshots.get("response_assembly")
    if snapshot is None:
        return None
    value = getattr(snapshot, "value", None)
    if value is None:
        return None
    intent = getattr(value, "expression_intent", None)
    return intent if isinstance(intent, str) and intent else None


def _kernel_confidence_from_result(result: object) -> float | None:
    """Read the kernel's calibrated forward-looking confidence.

    Source: ``prediction_error`` snapshot value
    (``PredictionErrorSnapshot.next_prediction.confidence``), the PE
    owner's own [0, 1] confidence for the upcoming turn. Mirrors the
    DLaaS dispatch ``_kernel_confidence`` projection (upstream half of
    deploy debt ``D-collab-pe``) so OpenAI-compat consumers get the same
    signal on a response header. ``None`` (header absent) when the turn
    published no PE snapshot — consumers must NOT fabricate a value.
    """
    snapshots = getattr(result, "active_snapshots", None)
    if not snapshots:
        return None
    snapshot = snapshots.get("prediction_error")
    if snapshot is None:
        return None
    value = getattr(snapshot, "value", None)
    if value is None:
        return None
    next_prediction = getattr(value, "next_prediction", None)
    if next_prediction is None:
        return None
    raw = getattr(next_prediction, "confidence", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return max(0.0, min(1.0, float(raw)))


@dataclass(frozen=True)
class LifeformCompletionResult:
    """Internal struct produced by :func:`lifeform_complete`.

    Wraps :class:`ChatCompletionResponse` with extra telemetry that
    the router exposes via response headers (``x-lifeform-*``) for
    downstream ablation analysis. The OpenAI body itself stays
    schema-compatible.

    ``evidence_pointers`` (U6 / family-memorial enabler) carries the
    structured L3 ``EvidencePointer`` records (as JSON-safe dicts)
    that the figure-vertical's ``GroundedDecoder`` produced when the
    response was grounded against a ``FigureRetrievalIndex``. The
    router writes these into an ``event: evidence`` SSE frame at
    stream close so downstream clients (apps/family-memorial's
    ``CitationCard``) can render clickable citations linking back to
    the original family corpus. Empty tuple = "no figure bundle was
    bound to this ai_id" OR "no assertions cleared verify".
    """

    response: ChatCompletionResponse
    resolution: SessionResolution
    active_regime: str | None
    active_abstract_action: str | None
    pe_magnitude: float
    rationale_tags: tuple[str, ...]
    evidence_pointers: tuple[dict, ...] = ()
    # Published expression intent (from the ``response_assembly``
    # snapshot). Surfaced on the ``x-lifeform-expression-intent``
    # response header so OpenAI-compat consumers (einstein /
    # family-memorial) can drive the presence avatar from the core's
    # real intent instead of a hardcoded preset — the OpenAI-compat
    # half of the "subjectivity transport gap" fix. ``None`` when the
    # turn produced no response_assembly.
    expression_intent: str | None = None
    # Kernel-calibrated forward-looking confidence [0, 1] from the PE
    # owner (``next_prediction.confidence``). Surfaced on the
    # ``x-lifeform-confidence`` response header so a BFF collaboration
    # gate can consume a REAL kernel-PE signal (deploy debt
    # ``D-collab-pe``). ``None`` when the turn published no PE snapshot.
    confidence: float | None = None
    # The resolved ``LifeformSession`` this turn ran on. Carried so the
    # router can hand it to an optional post-turn observability hook
    # (``app['openai_compat_on_turn']``) that records a DLaaS cognition
    # snapshot — closing the gap where OpenAI-path turns advanced kernel
    # state but were invisible to ``/cognition/health`` / ``/readouts`` /
    # ``/explain``. ``None`` only on the early tool-call return path.
    session: Any | None = None


async def lifeform_complete(
    *,
    request: ChatCompletionRequest,
    manager: SessionManager,
) -> LifeformCompletionResult:
    """Run an OpenAI chat completion through the lifeform pipeline.

    Steps:
      1. Resolve the session id (explicit / fresh / derived).
      2. Get-or-create the SessionManager session at that id.
      3. Extract the latest user input from the messages array.
      4. Call ``session.run_turn(user_input)`` — this is the only
         state-advancing call we make on the kernel.
      5. Build the OpenAI-shaped response from the lifeform result
         and surface the lifeform telemetry on the wrapper struct.
    """

    resolution = derive_session_id(request)
    user_id = request.metadata.get("user_id", "").strip() or None
    # D22: honour an explicit ``tenant_id`` so a multi-tenant front door
    # can route end-user memory into the two-layer ``{tenant}:{end_user}``
    # scope. Empty / missing falls back to the manager's adopting tenant
    # (then the closed-alpha default) inside ``create_session``.
    tenant_id = request.metadata.get("tenant_id", "").strip() or None
    # NW9: honour the baked LifeformTemplate id when the caller binds
    # one. Without this the session falls back to the vertical default
    # profile and the character's trained memory_checkpoint never loads.
    # Only applies on first create_session for this session_id.
    template_id = request.metadata.get("dlaas.template_id", "").strip() or None

    session = await _get_or_create_session(
        manager=manager,
        session_id=resolution.session_id,
        user_id=user_id,
        template_id=template_id,
        tenant_id=tenant_id,
    )

    if request.messages[-1].role == "tool":
        _submit_openai_tool_message(session=session, messages=request.messages)
        user_input = "Continue the turn using the submitted tool result."
        result = await session.run_turn(user_input)
    else:
        user_input = extract_user_input(request.messages)
        tool_intent = _forced_tool_intent(request)
        invoker = _session_invoker(session)
        tool_loop_mode = request.metadata.get("dlaas.tool_loop", "").strip().lower()
        if invoker is not None and (tool_intent is not None or tool_loop_mode == "server"):
            if tool_loop_mode == "server":
                from lifeform_affordance import (
                    ToolLoopOrchestrator,
                    ToolLoopPolicy,
                )

                orchestrator = ToolLoopOrchestrator(
                    registry=invoker.registry,
                    invoker=invoker,
                    policy=ToolLoopPolicy(server_side_execution=True),
                    contract_id=request.metadata.get("dlaas.contract_id", "").strip() or None,
                    intent_proposer=_llm_tool_intent_proposer(manager),
                )
                loop_result = await orchestrator.run(
                    session=session,
                    user_input=user_input,
                    initial_intents=(tool_intent,) if tool_intent is not None else (),
                )
                result = loop_result.final_turn_result
            elif tool_intent is not None:
                response = _tool_call_response(
                    request=request,
                    resolution=resolution,
                    manager=manager,
                    tool_call=_intent_to_tool_call(tool_intent),
                )
                return LifeformCompletionResult(
                    response=response,
                    resolution=resolution,
                    active_regime=None,
                    active_abstract_action=None,
                    pe_magnitude=0.0,
                    rationale_tags=("tool-call",),
                )
        else:
            result = await session.run_turn(user_input)

    response_text = result.response.text
    rationale_tags = tuple(getattr(result.response, "rationale_tags", ()))
    # U6: copy structured L3 evidence pointers (added by lifeform-expression
    # LLMResponseSynthesizer when a figure bundle is bound). Empty tuple
    # when no bundle / no grounding ran. Kept as opaque dicts to avoid
    # pulling lifeform-domain-figure types across the wheel boundary.
    evidence_pointers = tuple(
        getattr(result.response, "evidence_pointers", ()) or ()
    )
    active_regime = getattr(result, "active_regime", None)
    active_abstract_action = getattr(result, "active_abstract_action", None)
    expression_intent = _expression_intent_from_result(result)
    confidence = _kernel_confidence_from_result(result)

    summaries = session.turn_summaries
    last_summary = summaries[-1] if summaries else None
    pe_magnitude = float(getattr(last_summary, "pe_magnitude", 0.0)) if last_summary else 0.0

    system_context, prompt_text, history = split_messages(request.messages)
    prompt_tokens = estimate_prompt_tokens(system_context, prompt_text, history)
    completion_tokens = max(1, len(response_text) // 4)
    gen_max = request.generation.max_tokens
    finish_reason = "length" if (gen_max and completion_tokens >= gen_max) else "stop"

    response = ChatCompletionResponse(
        id=resolution.session_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=(
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason=finish_reason,
            ),
        ),
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        system_fingerprint=f"lifeform:{manager.vertical_name}",
    )

    return LifeformCompletionResult(
        response=response,
        resolution=resolution,
        active_regime=active_regime,
        active_abstract_action=active_abstract_action,
        pe_magnitude=pe_magnitude,
        rationale_tags=rationale_tags,
        evidence_pointers=evidence_pointers,
        expression_intent=expression_intent,
        confidence=confidence,
        session=session,
    )


def _session_invoker(session: Any) -> Any:
    try:
        return session.mcp_invoker
    except AttributeError as exc:
        raise ValueError(
            "invalid_tools: selected session does not expose an affordance invoker"
        ) from exc


def _forced_tool_intent(request: ChatCompletionRequest) -> Any:
    choice = request.tool_choice
    if not isinstance(choice, dict):
        return None
    if choice.get("type") != "function":
        raise ValueError("invalid_tool_choice: only function tool_choice is supported")
    function = choice.get("function")
    if not isinstance(function, dict):
        raise ValueError("invalid_tool_choice: function object is required")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("invalid_tool_choice: function.name must be non-empty")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_tool_choice: function.arguments must be JSON") from exc
    elif isinstance(arguments, dict):
        parsed_arguments = dict(arguments)
    else:
        raise ValueError("invalid_tool_choice: function.arguments must be object or JSON string")
    from lifeform_affordance import ToolCallIntent

    return ToolCallIntent(
        descriptor_name=name,
        parameters=parsed_arguments,
        call_id=f"call_{uuid.uuid4().hex[:16]}",
        source="openai_tool_choice",
    )


def _llm_tool_intent_proposer(manager: SessionManager) -> Any:
    runtime = manager.substrate_runtime
    if runtime is None:
        raise ValueError(
            "invalid_tools: server-side conversational tool loop requires a substrate runtime"
        )

    def provider(prompt: str) -> str:
        generated = runtime.generate(
            prompt=prompt,
            system_context=(
                "Return only JSON for the tool-intent decision. "
                "Do not include markdown."
            ),
            chat_messages=(),
            max_new_tokens=512,
            temperature=0.0,
        )
        return str(generated.text)

    from lifeform_affordance import LLMToolIntentProposer

    return LLMToolIntentProposer(provider)


def _intent_to_tool_call(intent: Any) -> ChatToolCall:
    return ChatToolCall(
        id=intent.stable_call_id,
        type="function",
        function=ChatToolCallFunction(
            name=intent.descriptor_name,
            arguments=json.dumps(dict(intent.parameters), ensure_ascii=False),
        ),
    )


def _tool_call_response(
    *,
    request: ChatCompletionRequest,
    resolution: SessionResolution,
    manager: SessionManager,
    tool_call: ChatToolCall,
) -> ChatCompletionResponse:
    system_context, prompt_text, history = split_messages(request.messages)
    prompt_tokens = estimate_prompt_tokens(system_context, prompt_text, history)
    return ChatCompletionResponse(
        id=resolution.session_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=(
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=(tool_call,),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            total_tokens=prompt_tokens + 1,
        ),
        system_fingerprint=f"lifeform:{manager.vertical_name}",
    )


def _submit_openai_tool_message(
    *,
    session: Any,
    messages: tuple[ChatMessage, ...],
) -> None:
    last = messages[-1]
    if not last.tool_call_id.strip():
        raise ValueError("invalid_tool_message: tool_call_id is required")
    tool_name = last.name.strip() or _tool_name_for_call_id(
        messages=messages, tool_call_id=last.tool_call_id
    )
    session.submit_tool_result(
        event_id=last.tool_call_id,
        tool_name=tool_name,
        action_id=f"{tool_name}:{last.tool_call_id}",
        status="succeeded",
        summary="OpenAI tool message received",
        detail=last.content,
        confidence=1.0,
        plan_ref=last.tool_call_id,
    )


def _tool_name_for_call_id(
    *,
    messages: tuple[ChatMessage, ...],
    tool_call_id: str,
) -> str:
    for message in reversed(messages[:-1]):
        for call in message.tool_calls:
            if call.id == tool_call_id:
                return call.function.name
    raise ValueError(
        "invalid_tool_message: tool name missing and no matching assistant tool_call found"
    )


async def _get_or_create_session(
    *,
    manager: SessionManager,
    session_id: str,
    user_id: str | None,
    template_id: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    """SessionManager get-or-create with a tiny race-tolerant fallback.

    Concurrent requests with the same auto-derived session id can
    race on ``create_session``. We swallow the typed
    ``SessionAlreadyExistsError`` (from lifeform-service) and fall
    back to ``get_session``, which the loser of the race will see
    as the existing session. This is the standard lifeform-service
    pattern; we are not introducing new error semantics.

    NW9: ``template_id`` (when set) selects a baked LifeformTemplate so
    the session is reincarnated from that template's memory_checkpoint
    rather than the vertical default profile. It only takes effect on
    the first ``create_session`` for a given ``session_id`` (sessions
    are sticky); reused sessions keep their original template binding.
    """

    if await manager.has_session(session_id):
        session = await manager.get_session(session_id)
        if user_id and not _session_end_user_remap_allowed():
            reader = getattr(manager, "session_end_user", None)
            bound = reader(session_id) if callable(reader) else None
            if bound is not None and bound != user_id:
                raise SessionEndUserMismatchError(
                    f"session_id={session_id!r} is bound to user_id={bound!r} "
                    f"but the request carries user_id={user_id!r}"
                )
        return session
    try:
        return await manager.create_session(
            session_id=session_id,
            user_id=user_id,
            template_id=template_id,
            tenant_id=tenant_id,
        )
    except SessionAlreadyExistsError:
        # Lost the race; fall back to the existing session.
        return await manager.get_session(session_id)
    except SessionNotFoundError:
        # Pathological case: has_session=False, create raised
        # NotFound (rare; would mean LifeformFactory returned None).
        # Surface as a clean ValueError for the router to map to 500.
        raise


def reserved_metadata_keys() -> tuple[str, ...]:
    """Reserved keys in the OpenAI ``metadata`` extension.

    External harnesses may pass arbitrary metadata; these keys
    are interpreted by the bridge:

    * ``session_id`` — explicit override (sticky reuse on a chosen id)
    * ``user_id`` — passed through to ``create_session(user_id=...)``
    * ``tenant_id`` — passed through to ``create_session(tenant_id=...)``
      so the two-layer ``{tenant}:{end_user}`` scope partitions memory
      per tenant (D22)
    * ``dlaas.template_id`` — baked LifeformTemplate id; selects the
      reincarnation template on first ``create_session`` (NW9)

    All other keys are stored on the request DTO unchanged.
    """
    return ("session_id", "user_id", "tenant_id", "dlaas.template_id")
