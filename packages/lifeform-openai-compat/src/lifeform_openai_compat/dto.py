"""Typed DTOs for OpenAI Chat Completions request / response.

These mirror the fields actually used by external chat harnesses
(EQ-Bench 3, EmpathyBench, OpenRouter passthrough, OpenAI Python
client) and are intentionally a STRICT SUBSET of the upstream OpenAI
schema. Fields the lifeform service has no mapping for (function
calls, tool choice, log probs, image inputs) are silently dropped on
parse rather than error: external harnesses commonly send a superset
of fields, and we want broad compatibility without echoing modes we
do not support.

Schema reference: https://platform.openai.com/docs/api-reference/chat/create

Frozen-dataclass invariants:

* All DTOs are frozen so once parsed a request cannot be mutated
  before reaching the adapter. This is consistent with the rest of
  the codebase's snapshot-style data contracts (R8).
* ``messages`` is a tuple, not a list. Same reason.
* Numeric fields are ``float | int | None``. ``None`` means "use the
  underlying lifeform / substrate default" rather than "use 0".
* ``metadata`` is a free-form ``dict[str, str]`` mirror of OpenAI's
  ``metadata`` extension surface (added in 2024). The adapter uses two
  reserved keys: ``session_id`` (forces sticky reuse of a specific
  lifeform session) and ``user_id`` (alpha-mode identity binding).
  Other keys are accepted and surfaced to the response unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    """A single message in the OpenAI chat history.

    ``role`` is one of ``"system"`` / ``"user"`` / ``"assistant"`` /
    ``"tool"`` (the last one is accepted on parse but routed to a
    typed error response: this adapter does not support tool calls).
    """

    role: str
    content: str

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationConfig:
    """Generation knobs the harness can override per-request.

    Empty / None means "use the underlying runtime's default". The
    adapter forwards what it can map onto the lifeform's generation
    path; fields with no mapping (e.g. logprobs) are dropped.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    stream: bool = False  # only ``False`` is supported by the adapter for now


@dataclass(frozen=True)
class ChatCompletionRequest:
    """Parsed OpenAI Chat Completions POST body.

    The ``model`` field is preserved for echo into the response but is
    NOT used to route between verticals — that decision is made by the
    underlying lifeform-service vertical the adapter is mounted onto.
    Use the ``X-Compat-Mode`` header or ``mode`` query param (added in
    Packet 4) to switch between the lifeform path and the raw substrate
    passthrough path.
    """

    model: str
    messages: tuple[ChatMessage, ...]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: dict[str, str] = field(default_factory=dict)
    user: str = ""  # OpenAI's top-level ``user`` field (auditing hint)

    @staticmethod
    def from_payload(payload: object) -> "ChatCompletionRequest":
        """Parse an OpenAI-shaped JSON payload.

        Raises :class:`ValueError` with a stable error code suffix
        (``invalid_*`` matching lifeform-service convention) when the
        payload is malformed; the router translates these into 400
        responses.
        """

        if not isinstance(payload, dict):
            raise ValueError("invalid_request: body must be a JSON object")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("invalid_model: 'model' must be a non-empty string")
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(
                "invalid_messages: 'messages' must be a non-empty array"
            )
        parsed_messages: list[ChatMessage] = []
        for index, item in enumerate(raw_messages):
            if not isinstance(item, dict):
                raise ValueError(
                    f"invalid_message_at_{index}: each message must be an object"
                )
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or role not in {
                "system", "user", "assistant", "tool", "developer",
            }:
                raise ValueError(
                    f"invalid_role_at_{index}: role must be one of "
                    "system/user/assistant/tool/developer"
                )
            if not isinstance(content, str):
                # OpenAI also allows multi-modal content arrays; we treat
                # those as unsupported and surface a 400 rather than
                # silently dropping them, because EQ-Bench harnesses use
                # plain string content and any array form would mean a
                # new harness contract we have not validated.
                raise ValueError(
                    f"invalid_content_at_{index}: content must be a string "
                    "(multi-modal content arrays are not supported by this adapter)"
                )
            parsed_messages.append(ChatMessage(role=role, content=content))

        gen = _parse_generation_config(payload)

        raw_metadata = payload.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, dict):
            raise ValueError("invalid_metadata: 'metadata' must be an object")
        normalized_metadata: dict[str, str] = {}
        for key, value in raw_metadata.items():
            if not isinstance(key, str):
                raise ValueError("invalid_metadata: metadata keys must be strings")
            if not isinstance(value, str):
                raise ValueError(
                    "invalid_metadata: metadata values must be strings "
                    "(OpenAI extension surface)"
                )
            normalized_metadata[key] = value

        user = payload.get("user", "")
        if user is None:
            user = ""
        if not isinstance(user, str):
            raise ValueError("invalid_user: 'user' must be a string when provided")

        return ChatCompletionRequest(
            model=model.strip(),
            messages=tuple(parsed_messages),
            generation=gen,
            metadata=normalized_metadata,
            user=user,
        )


def _parse_generation_config(payload: dict[str, Any]) -> GenerationConfig:
    def _opt_float(name: str) -> float | None:
        if name not in payload or payload[name] is None:
            return None
        value = payload[name]
        if not isinstance(value, (int, float)):
            raise ValueError(f"invalid_{name}: must be a number")
        return float(value)

    def _opt_int(name: str) -> int | None:
        if name not in payload or payload[name] is None:
            return None
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"invalid_{name}: must be an integer")
        return value

    raw_stop = payload.get("stop")
    stop_tuple: tuple[str, ...] = ()
    if isinstance(raw_stop, str):
        stop_tuple = (raw_stop,)
    elif isinstance(raw_stop, list):
        for item in raw_stop:
            if not isinstance(item, str):
                raise ValueError("invalid_stop: each stop sequence must be a string")
        stop_tuple = tuple(raw_stop)
    elif raw_stop is not None:
        raise ValueError("invalid_stop: must be a string or array of strings")

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("invalid_stream: must be a boolean")

    return GenerationConfig(
        temperature=_opt_float("temperature"),
        top_p=_opt_float("top_p"),
        max_tokens=_opt_int("max_tokens"),
        presence_penalty=_opt_float("presence_penalty"),
        frequency_penalty=_opt_float("frequency_penalty"),
        stop=stop_tuple,
        seed=_opt_int("seed"),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatCompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_json(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ChatCompletionChoice:
    """One assistant choice in the response.

    ``finish_reason`` mirrors OpenAI's vocabulary:
      * ``"stop"`` — model emitted a stop token / completed naturally
      * ``"length"`` — hit ``max_tokens``
      * ``"content_filter"`` — adapter applied a safety / refusal layer

    The current adapter only emits ``"stop"`` / ``"length"`` since the
    lifeform's expression-layer refusal logic is opaque to us; future
    work can surface ``"content_filter"`` from the lifeform's response
    rationale tags (e.g. a ``risk=...`` / ``boundary`` tag).
    """

    index: int
    message: ChatMessage
    finish_reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_json(),
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class ChatCompletionResponse:
    """OpenAI ChatCompletion-shaped response envelope.

    Identifier conventions:

    * ``id`` is the lifeform session_id when the request used the
      sticky path; otherwise a freshly minted ``chatcmpl-<hex>``
      identifier so existing OpenAI clients keep working.
    * ``object`` is always the literal string ``"chat.completion"``.
    * ``created`` is a unix epoch second (matches OpenAI).
    * ``model`` echoes the request's ``model`` field unchanged.
    * ``system_fingerprint`` is empty by default; it is populated with
      a stable hash of (vertical, substrate model id, adapter version)
      once Packet 4 wires the substrate runtime in.
    """

    id: str
    object: str
    created: int
    model: str
    choices: tuple[ChatCompletionChoice, ...]
    usage: ChatCompletionUsage
    system_fingerprint: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.to_json() for choice in self.choices],
            "usage": self.usage.to_json(),
            "system_fingerprint": self.system_fingerprint,
        }
