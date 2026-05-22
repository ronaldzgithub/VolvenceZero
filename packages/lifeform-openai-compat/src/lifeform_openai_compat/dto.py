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
class ChatToolCallFunction:
    name: str
    arguments: str = "{}"

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ChatToolCall:
    id: str
    type: str
    function: ChatToolCallFunction

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_json(),
        }


@dataclass(frozen=True)
class OpenAIToolDefinition:
    type: str
    function: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "function": dict(self.function)}


@dataclass(frozen=True)
class ChatMessage:
    """A single message in the OpenAI chat history.

    ``role`` is one of ``"system"`` / ``"user"`` / ``"assistant"`` /
    ``"tool"`` / ``"developer"``. Assistant messages can carry
    OpenAI ``tool_calls``; tool-role messages carry ``tool_call_id``.
    """

    role: str
    content: str
    tool_calls: tuple[ChatToolCall, ...] = ()
    tool_call_id: str = ""
    name: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_json() for call in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


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
    tools: tuple[OpenAIToolDefinition, ...] = ()
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = False
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
            if content is None and role == "assistant" and "tool_calls" in item:
                content = ""
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
            parsed_messages.append(
                ChatMessage(
                    role=role,
                    content=content,
                    tool_calls=_parse_tool_calls(item.get("tool_calls"), index=index),
                    tool_call_id=_optional_string(item, "tool_call_id", index=index),
                    name=_optional_string(item, "name", index=index),
                )
            )

        gen = _parse_generation_config(payload)
        tools = _parse_tools(payload.get("tools", ()))
        tool_choice = _parse_tool_choice(payload.get("tool_choice"))
        parallel_tool_calls = payload.get("parallel_tool_calls", False)
        if not isinstance(parallel_tool_calls, bool):
            raise ValueError("invalid_parallel_tool_calls: must be a boolean")

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
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            user=user,
        )


def _optional_string(item: dict[str, Any], key: str, *, index: int) -> str:
    value = item.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"invalid_{key}_at_{index}: must be a string")
    return value


def _parse_tool_calls(raw: object, *, index: int) -> tuple[ChatToolCall, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"invalid_tool_calls_at_{index}: must be an array")
    calls: list[ChatToolCall] = []
    for call_index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"invalid_tool_call_at_{index}_{call_index}: each call must be an object"
            )
        call_id = item.get("id")
        call_type = item.get("type", "function")
        function = item.get("function")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError(
                f"invalid_tool_call_id_at_{index}_{call_index}: id must be non-empty"
            )
        if call_type != "function":
            raise ValueError(
                f"invalid_tool_call_type_at_{index}_{call_index}: only function is supported"
            )
        if not isinstance(function, dict):
            raise ValueError(
                f"invalid_tool_call_function_at_{index}_{call_index}: function is required"
            )
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"invalid_tool_call_name_at_{index}_{call_index}: name must be non-empty"
            )
        if not isinstance(arguments, str):
            raise ValueError(
                f"invalid_tool_call_arguments_at_{index}_{call_index}: arguments must be a JSON string"
            )
        calls.append(
            ChatToolCall(
                id=call_id,
                type="function",
                function=ChatToolCallFunction(name=name, arguments=arguments),
            )
        )
    return tuple(calls)


def _parse_tools(raw: object) -> tuple[OpenAIToolDefinition, ...]:
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list):
        raise ValueError("invalid_tools: tools must be an array")
    tools: list[OpenAIToolDefinition] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"invalid_tool_at_{index}: each tool must be an object")
        tool_type = item.get("type")
        function = item.get("function")
        if tool_type != "function":
            raise ValueError(f"invalid_tool_type_at_{index}: only function is supported")
        if not isinstance(function, dict):
            raise ValueError(f"invalid_tool_function_at_{index}: function is required")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid_tool_name_at_{index}: function.name must be non-empty")
        if not isinstance(parameters, dict):
            raise ValueError(
                f"invalid_tool_parameters_at_{index}: function.parameters must be an object"
            )
        tools.append(OpenAIToolDefinition(type="function", function=dict(function)))
    return tuple(tools)


def _parse_tool_choice(raw: object) -> str | dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw not in {"none", "auto", "required"}:
            raise ValueError("invalid_tool_choice: string must be none/auto/required")
        return raw
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError("invalid_tool_choice: must be a string or object")


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
