"""Raw substrate passthrough mode for the OpenAI-compat adapter.

This module implements the **ablation track 3** for external benchmark
scoring (EQ-Bench 3 / EmpathyBench / etc.). It bypasses the lifeform
pipeline entirely (no PromptPlanner, no ResponseSynthesizer, no
expression / refusal layer, no memory, no regime classifier) and maps
the OpenAI ``/v1/chat/completions`` request directly onto a single
``runtime.generate(...)`` call against whatever shared
``OpenWeightResidualRuntime`` the SessionManager is holding.

Why three tracks (companion / companion-cold / raw):

The same Qwen substrate is reachable two ways through this adapter:

1. ``mode=lifeform`` (Packet 3 + 4) — the full lifeform pipeline
   wraps the substrate. Differences from raw are the lifeform's
   adaptive controllers, regime priors, expression-layer rewriting,
   and memory.
2. ``mode=raw`` (this module) — bypass everything; the OpenAI
   message history goes straight into ``runtime.generate``.

By scoring both modes on the same external benchmark we get a
**measurable delta** for "what does the lifeform pipeline add (or
subtract) on top of the bare LLM?". That delta is the only honest
answer to the fundraising-due-diligence question "is your wrapper
making the model better, or just slower?".

Architecture invariants:

* This module imports only stdlib + typing + this wheel's own DTOs.
  It does NOT import ``volvence_zero.*`` or ``lifeform_service`` —
  the runtime is duck-typed via :class:`SubstrateRuntimeProtocol`,
  so the router (Packet 4) injects whatever
  ``SessionManager.substrate_runtime`` returns and the contract
  surface is captured here in code rather than via cross-wheel
  imports.
* Read-only with respect to the runtime. We call
  ``runtime.generate(...)`` and read attributes like
  ``runtime.model_id`` / ``runtime.runtime_origin`` for the response
  fingerprint. We never mutate the runtime, never set attributes on
  it, never call any private method.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Protocol, runtime_checkable

from lifeform_openai_compat.dto import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)

_DEFAULT_MAX_NEW_TOKENS: int = 512
_DEFAULT_TEMPERATURE: float = 0.7


# ---------------------------------------------------------------------------
# Runtime protocol
#
# Duck-typed contract that the SessionManager's substrate_runtime must
# satisfy. The actual concrete class is
# ``volvence_zero.substrate.OpenWeightResidualRuntime`` (and its
# transformers subclass), but this wheel does not import those —
# instead we declare the surface here and rely on Python's structural
# typing. Any object that satisfies this Protocol is a valid argument
# to :func:`raw_substrate_complete`.
# ---------------------------------------------------------------------------


@runtime_checkable
class SubstrateRuntimeProtocol(Protocol):
    """The minimal slice of OpenWeightResidualRuntime this module uses.

    See ``volvence_zero.substrate.residual_interfaces`` (kernel) for
    the canonical definition; the contract below is intentionally
    narrower so that future kernel-side additions cannot accidentally
    leak into the adapter's API surface.
    """

    model_id: str
    runtime_origin: str

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = ...,
        chat_messages: tuple[tuple[str, str], ...] = ...,
        max_new_tokens: int = ...,
        temperature: float = ...,
    ) -> Any: ...


class RawSubstrateUnavailable(RuntimeError):
    """Raised when no shared substrate runtime is available.

    The adapter's raw mode requires the host service to be started
    with a real runtime (typically ``--substrate-mode=hf-shared`` on
    ``lifeform-serve``). Synthetic / per-session-runtime mode has no
    runtime to pass through, and silently falling back to the
    lifeform pipeline would defeat the purpose of the ablation. We
    surface a typed error so the router maps it to a clean 503
    response.
    """


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------


def split_messages(
    messages: tuple[ChatMessage, ...],
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Translate OpenAI messages → ``(system_context, prompt, history)``.

    Mapping rules (mirrors the runtime.generate contract):

    * Every leading ``system`` or ``developer`` message is concatenated
      with ``\\n\\n`` separators and becomes ``system_context``.
    * Every non-last assistant / user message becomes one entry in
      ``history`` (a tuple of ``(role, content)``). Order preserved.
    * The last message becomes ``prompt``:
        - If user / tool: prompt is its content.
        - If assistant: same — the runtime treats prompt as "what the
          model has just been handed" and continues from it.
        - If system / developer: prompt is empty (all-system payloads
          are typically system-prompt warmups; the runtime returns a
          dry response with empty prompt).
    """

    if not messages:
        raise ValueError("invalid_messages: at least one message required")

    system_parts: list[str] = []
    history: list[tuple[str, str]] = []

    for msg in messages[:-1]:
        if msg.role in {"system", "developer"}:
            system_parts.append(msg.content)
        else:
            history.append((msg.role, msg.content))

    last = messages[-1]
    if last.role in {"system", "developer"}:
        system_parts.append(last.content)
        prompt = ""
    else:
        prompt = last.content

    return ("\n\n".join(system_parts), prompt, tuple(history))


def estimate_prompt_tokens(
    system_context: str,
    prompt: str,
    chat_messages: tuple[tuple[str, str], ...],
) -> int:
    """Rough chars/4 heuristic for the OpenAI usage envelope.

    Real tokenization would require importing the runtime's
    tokenizer (forbidden by the import boundary). Since most external
    harnesses do not validate ``usage.prompt_tokens`` byte-exactly,
    a stable estimate is better than a brittle exact count. For
    payment / billing systems this would be inadequate; this is for
    benchmarks where ``usage`` is purely informational.
    """

    total_chars = len(system_context) + len(prompt)
    for _, content in chat_messages:
        total_chars += len(content)
    return max(1, total_chars // 4)


# ---------------------------------------------------------------------------
# The completion call
# ---------------------------------------------------------------------------


def raw_substrate_complete(
    *,
    request: ChatCompletionRequest,
    runtime: SubstrateRuntimeProtocol | None,
    request_id: str | None = None,
) -> ChatCompletionResponse:
    """Translate an OpenAI request to ``runtime.generate(...)`` and back.

    Args:
        request: parsed OpenAI ChatCompletion payload.
        runtime: shared substrate runtime, or ``None`` to raise
            :class:`RawSubstrateUnavailable` (the SessionManager
            ``substrate_runtime`` property returns ``None`` when the
            host is in synthetic mode).
        request_id: optional pre-minted id for the response. When
            ``None``, mint a fresh ``chatcmpl-<hex>`` to mirror
            OpenAI's id format.

    Raises:
        RawSubstrateUnavailable: if ``runtime is None``.

    Returns:
        ChatCompletionResponse byte-compatible with OpenAI Python
        client's ``ChatCompletion`` model.
    """

    if runtime is None:
        raise RawSubstrateUnavailable(
            "raw substrate mode requires the host service to be started "
            "with a shared runtime (lifeform-serve --substrate-mode=hf-shared, "
            "or equivalent). The current service is in synthetic / per-session "
            "mode and has no runtime to pass through. Use mode=lifeform to "
            "exercise the lifeform pipeline path instead, or restart the "
            "service with a real substrate."
        )

    system_context, prompt, history = split_messages(request.messages)

    gen = request.generation
    max_new = gen.max_tokens if gen.max_tokens is not None else _DEFAULT_MAX_NEW_TOKENS
    temperature = gen.temperature if gen.temperature is not None else _DEFAULT_TEMPERATURE

    result = runtime.generate(
        prompt=prompt,
        system_context=system_context,
        chat_messages=history,
        max_new_tokens=max_new,
        temperature=temperature,
    )

    text = getattr(result, "text", "")
    completion_tokens_attr = getattr(result, "token_count", 0)
    completion_tokens = int(completion_tokens_attr or 0)
    prompt_tokens = estimate_prompt_tokens(system_context, prompt, history)

    # Heuristic finish_reason: if we hit the max_new_tokens budget,
    # the model was likely cut short. Otherwise we trust the runtime
    # to have stopped on EOS / stop sequence.
    finish_reason = "length" if completion_tokens >= max_new else "stop"

    response_id = request_id or _fresh_completion_id()

    runtime_model_id = getattr(runtime, "model_id", "unknown")
    runtime_origin = getattr(runtime, "runtime_origin", "unknown")

    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=(
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            ),
        ),
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        system_fingerprint=f"raw-substrate:{runtime_model_id}@{runtime_origin}",
    )


def _fresh_completion_id() -> str:
    """OpenAI uses ``chatcmpl-<base16>`` as the id format; mirror it."""
    return f"chatcmpl-{uuid.uuid4().hex}"
