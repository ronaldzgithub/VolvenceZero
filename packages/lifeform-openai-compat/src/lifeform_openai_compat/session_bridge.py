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
)
from lifeform_openai_compat.raw_substrate import (
    estimate_prompt_tokens,
    split_messages,
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


@dataclass(frozen=True)
class LifeformCompletionResult:
    """Internal struct produced by :func:`lifeform_complete`.

    Wraps :class:`ChatCompletionResponse` with extra telemetry that
    the router exposes via response headers (``x-lifeform-*``) for
    downstream ablation analysis. The OpenAI body itself stays
    schema-compatible.
    """

    response: ChatCompletionResponse
    resolution: SessionResolution
    active_regime: str | None
    active_abstract_action: str | None
    pe_magnitude: float
    rationale_tags: tuple[str, ...]


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
    user_input = extract_user_input(request.messages)

    session = await _get_or_create_session(
        manager=manager,
        session_id=resolution.session_id,
        user_id=user_id,
    )

    result = await session.run_turn(user_input)

    response_text = result.response.text
    rationale_tags = tuple(getattr(result.response, "rationale_tags", ()))
    active_regime = getattr(result, "active_regime", None)
    active_abstract_action = getattr(result, "active_abstract_action", None)

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
    )


async def _get_or_create_session(
    *,
    manager: SessionManager,
    session_id: str,
    user_id: str | None,
) -> Any:
    """SessionManager get-or-create with a tiny race-tolerant fallback.

    Concurrent requests with the same auto-derived session id can
    race on ``create_session``. We swallow the typed
    ``SessionAlreadyExistsError`` (from lifeform-service) and fall
    back to ``get_session``, which the loser of the race will see
    as the existing session. This is the standard lifeform-service
    pattern; we are not introducing new error semantics.
    """

    if await manager.has_session(session_id):
        return await manager.get_session(session_id)
    try:
        return await manager.create_session(
            session_id=session_id, user_id=user_id
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

    External harnesses may pass arbitrary metadata; these two keys
    are interpreted by the bridge:

    * ``session_id`` — explicit override (sticky reuse on a chosen id)
    * ``user_id`` — passed through to ``create_session(user_id=...)``

    All other keys are stored on the request DTO unchanged.
    """
    return ("session_id", "user_id")
