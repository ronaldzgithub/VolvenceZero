# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Agent backends for the CAMEL baseline.

The server owns session-boundary detection and durable memory persistence;
the *backend* owns two LLM-facing operations:

* :meth:`CamelBackend.respond` — produce the assistant reply for the current
  turn, given the prior cross-session memory records and the in-session
  transcript so far.
* :meth:`CamelBackend.compact` — compress a finished session's transcript into
  a :class:`SessionMemoryRecord` to be re-seeded into later sessions.

Two implementations:

* :class:`EchoCamelBackend` — deterministic, no network, no ``camel-ai``
  dependency. It demonstrably carries cross-session memory (the reply names the
  prior session topics) so the plumbing is end-to-end assertable in tests and
  ``--backend echo`` smoke runs.
* :class:`CamelChatAgentBackend` — production backend. Lazily imports
  ``camel-ai`` and builds a ``ChatAgent`` whose model is an OpenAI-compatible
  client pointed at the upstream substrate (the SAME substrate the ``raw`` track
  uses). No implicit model fallback: a misconfigured upstream fails loud.

Inlined prompt
==============

The compaction prompt the production backend sends is the module-level constant
:data:`COMPACTION_PROMPT_TEMPLATE`. It is inlined in source on purpose — the
reproducibility contract forbids hiding prompts in remote configs.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
from typing import Any, Awaitable, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from companion_camel_baseline.memory_store import SessionMemoryRecord


# ---------------------------------------------------------------------------
# Prompt (inlined per reproducibility contract)
# ---------------------------------------------------------------------------


def _normalize_upstream_openai_base_url(base_url: str) -> tuple[str, dict[str, str]]:
    """Strip ``mode`` query params into ``X-Compat-Mode`` for OpenAI SDK clients.

    The OpenAI Python SDK appends ``/chat/completions`` to ``base_url`` with a
    naive join that breaks when ``base_url`` carries a query string (e.g.
    ``http://127.0.0.1:8000/v1?mode=raw`` becomes
    ``.../v1?mode=raw/chat/completions``).
    """

    parts = urlsplit(base_url)
    query = parse_qs(parts.query, keep_blank_values=True)
    extra_headers: dict[str, str] = {}
    mode_values = query.pop("mode", None)
    if mode_values:
        extra_headers["X-Compat-Mode"] = mode_values[0]
    new_query = urlencode(query, doseq=True)
    normalized = urlunsplit(
        (parts.scheme, parts.netloc, parts.path.rstrip("/"), new_query, parts.fragment)
    )
    return normalized, extra_headers


COMPACTION_PROMPT_TEMPLATE: str = """You are the memory module of a long-running companion agent.
Read the session transcript below and emit ONE compact JSON object capturing
what this agent should remember for future sessions with the same person.

Schema:
{{
  "topic": "<one short sentence, <= 120 chars, no quotes>",
  "salient": ["<each item <= 120 chars, factual only>"]
}}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences, no preface.
- Do not invent facts. If the transcript does not state something, omit it.
- "salient" items are durable facts, preferences, commitments, or unresolved
  threads worth recalling next time (e.g. "user's daughter is named Mia",
  "user dislikes being upsold", "promised to follow up about the move").
- "salient" may be empty.

Transcript (chronological, alternating user/assistant):

{transcript_text}
"""

# Tag re-seeded into the agent's system context so collected transcripts can be
# grepped for the baseline's memory contribution. Public constant.
MEMORY_PREFIX_HEADER: str = "[camel-baseline · cross-session memory ↓]"
MEMORY_PREFIX_FOOTER: str = "[camel-baseline · end memory]"

# Maximum number of prior session records re-seeded into a new session.
# Public constant; part of the reproducibility contract.
MAX_PRIOR_RECORDS: int = 5

# Token budget for the CAMEL ChatHistoryMemory ScoreBasedContextCreator.
# Cross-session records + in-session turns are written as memory records and
# CAMEL's context creator prunes to this budget (token-aware) rather than the
# baseline hand-rolling a fixed preamble. Public constant.
MEMORY_TOKEN_LIMIT: int = 8192


# ---------------------------------------------------------------------------
# Reply value object
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AgentReply:
    """One assistant turn from a backend."""

    text: str
    model_id: str
    usage_prompt_tokens: int | None
    usage_completion_tokens: int | None
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CamelBackendError(RuntimeError):
    """Raised when the backend cannot produce a reply / compaction.

    The server maps this to a 502 with body ``{"error": {"code":
    "camel_baseline_internal", "message": "..."}}``. Fail-loud per the
    no-swallow-errors rule.
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CamelBackend(Protocol):
    """The surface the server invokes."""

    @property
    def model(self) -> str: ...

    def respond(
        self,
        *,
        scope_key: str,
        session_id: str,
        system_prompt: str,
        prior_memory: tuple[SessionMemoryRecord, ...],
        session_messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> Awaitable[AgentReply]: ...

    def compact(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> Awaitable[SessionMemoryRecord]: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_memory_preamble(
    prior_memory: tuple[SessionMemoryRecord, ...],
) -> str | None:
    """Render the prior-session memory block, or ``None`` if empty."""

    materialized = list(prior_memory)[-MAX_PRIOR_RECORDS:]
    if not materialized:
        return None
    lines: list[str] = [MEMORY_PREFIX_HEADER, ""]
    for r in materialized:
        lines.append(r.to_prompt_block())
    lines.append("")
    lines.append(MEMORY_PREFIX_FOOTER)
    return "\n".join(lines)


def build_openai_envelope(
    *,
    text: str,
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Wrap an assistant text in the OpenAI chat-completions response shape."""

    envelope: dict[str, Any] = {
        "id": "camel-baseline-stub",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    if prompt_tokens is not None or completion_tokens is not None:
        envelope["usage"] = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        }
    return envelope


def _format_transcript(transcript: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in transcript:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not role or not content or role == "system":
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(empty transcript)"


_JSON_BLOCK_RE: re.Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)


def parse_compaction_json(raw: str) -> dict[str, Any]:
    """Parse the compaction JSON; fail-loud on unrecoverable output."""

    raw_stripped = (raw or "").strip()
    try:
        return json.loads(raw_stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw_stripped)
    if match is None:
        raise CamelBackendError(
            f"compaction response contained no JSON object: {raw_stripped[:160]!r}"
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise CamelBackendError(
            f"compaction returned malformed JSON: {exc}; raw[:160]={raw_stripped[:160]!r}"
        ) from exc


def _coerce_salient(value: Any, *, limit: int = 8) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CamelBackendError(
            f"expected list for 'salient', got {type(value).__name__}"
        )
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise CamelBackendError(
                f"expected list of str, got element of type {type(item).__name__}"
            )
        s = item.strip()
        if not s:
            continue
        out.append(s[:120])
        if len(out) >= limit:
            break
    return tuple(out)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Deterministic echo backend (tests / smoke)
# ---------------------------------------------------------------------------


class EchoCamelBackend:
    """Deterministic backend with no network and no camel-ai dependency.

    Cross-session memory is *observable*: the reply names the prior session
    topics that were re-seeded, so a test can assert the agent carried memory
    across sessions. Records every call into :attr:`calls` for assertions.
    """

    def __init__(self, *, model: str = "camel-baseline/echo-agent-v1") -> None:
        self._model = model
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    async def respond(
        self,
        *,
        scope_key: str,
        session_id: str,
        system_prompt: str,
        prior_memory: tuple[SessionMemoryRecord, ...],
        session_messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> AgentReply:
        last_user = next(
            (m["content"] for m in reversed(session_messages) if m.get("role") == "user"),
            "",
        )
        recalled = [r.topic for r in prior_memory[-MAX_PRIOR_RECORDS:]]
        recall_note = (
            f" recalling[{'; '.join(recalled)}]" if recalled else ""
        )
        text = f"[camel-echo:{session_id}]{recall_note} {last_user}".strip()
        self.calls.append(
            {
                "scope_key": scope_key,
                "session_id": session_id,
                "system_prompt": system_prompt,
                "prior_memory_topics": recalled,
                "session_messages": [dict(m) for m in session_messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return AgentReply(
            text=text,
            model_id=self._model,
            usage_prompt_tokens=sum(
                len(m.get("content", "")) // 4 for m in session_messages
            ),
            usage_completion_tokens=max(1, len(text) // 4),
            raw=build_openai_envelope(text=text, model=self._model),
        )

    async def compact(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> SessionMemoryRecord:
        user_turns = [m["content"].strip() for m in transcript if m.get("role") == "user"]
        topic_seed = user_turns[0] if user_turns else "(empty)"
        salient = tuple(t[:120] for t in user_turns[:8])
        return SessionMemoryRecord(
            scope_key=scope_key,
            session_id=session_id,
            topic=topic_seed[:120] or "(empty)",
            salient=salient,
            extracted_at=_now_iso(),
            extractor_model=self._model,
        )

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# Production CAMEL backend (lazy camel-ai import)
# ---------------------------------------------------------------------------


class CamelChatAgentBackend:
    """Production backend: a CAMEL ``ChatAgent`` over an OpenAI-compatible model.

    The model is configured to call the upstream substrate directly, so the
    CAMEL track shares byte-identical weights with the ``raw`` track. No
    implicit model fallback: a missing / broken upstream raises
    :class:`CamelBackendError`.

    ``camel-ai`` is imported lazily so the wheel installs (and the echo path
    runs) without the heavy dependency. Memory is managed through CAMEL's own
    subsystem: a ``ChatHistoryMemory`` backed by a ``ScoreBasedContextCreator``
    holds both the in-session turns and the cross-session records (written as
    memory records), so context is pruned token-aware by CAMEL rather than by a
    hand-rolled fixed preamble.
    """

    def __init__(
        self,
        *,
        model_type: str,
        upstream_base_url: str,
        upstream_api_key: str,
        compaction_model_type: str,
        compaction_base_url: str,
        compaction_api_key: str,
        timeout_s: float = 120.0,
    ) -> None:
        self._model_type = model_type
        self._base_url, self._compat_headers = _normalize_upstream_openai_base_url(
            upstream_base_url
        )
        self._api_key = upstream_api_key
        # Separate cross-family model for memory compaction (the "extractor"),
        # so the CAMEL baseline does not summarise its own transcripts with the
        # same substrate family (same-family crib-notes bias).
        self._compaction_model_type = compaction_model_type
        self._compaction_base_url, self._compaction_headers = (
            _normalize_upstream_openai_base_url(compaction_base_url)
        )
        self._compaction_api_key = compaction_api_key
        self._timeout_s = timeout_s
        self._camel = _import_camel()

    @property
    def model(self) -> str:
        return self._model_type

    def _build_model(self, *, temperature: float | None, max_tokens: int | None):
        return self._build_named_model(
            model_type=self._model_type,
            base_url=self._base_url,
            api_key=self._api_key,
            headers=self._compat_headers,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_compaction_model(self, *, temperature: float | None, max_tokens: int | None):
        return self._build_named_model(
            model_type=self._compaction_model_type,
            base_url=self._compaction_base_url,
            api_key=self._compaction_api_key,
            headers=self._compaction_headers,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_named_model(
        self,
        *,
        model_type: str,
        base_url: str,
        api_key: str,
        headers: dict[str, str],
        temperature: float | None,
        max_tokens: int | None,
    ):
        camel = self._camel
        config: dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        factory_kwargs: dict[str, Any] = {}
        if headers:
            factory_kwargs["default_headers"] = headers
        return camel.ModelFactory.create(
            model_platform=camel.ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            url=base_url,
            api_key=api_key,
            model_config_dict=config or None,
            **factory_kwargs,
        )

    async def respond(
        self,
        *,
        scope_key: str,
        session_id: str,
        system_prompt: str,
        prior_memory: tuple[SessionMemoryRecord, ...],
        session_messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> AgentReply:
        import asyncio

        system_text = system_prompt.strip() or "You are a long-running companion AI."
        last_user = next(
            (m["content"] for m in reversed(session_messages) if m.get("role") == "user"),
            "",
        )
        if not last_user:
            raise CamelBackendError("no user message in session_messages")
        prior_turns = [m for m in session_messages if m is not session_messages[-1]]
        cross_session = build_memory_preamble(prior_memory)

        def _run() -> AgentReply:
            camel = self._camel
            model = self._build_model(temperature=temperature, max_tokens=max_tokens)
            memory = camel.ChatHistoryMemory(
                context_creator=camel.ScoreBasedContextCreator(
                    token_counter=model.token_counter,
                    token_limit=MEMORY_TOKEN_LIMIT,
                ),
            )
            agent = camel.ChatAgent(
                system_message=system_text,
                model=model,
                memory=memory,
            )
            # Cross-session memory: write the re-seeded prior-session records
            # into CAMEL memory as a SYSTEM record so the native context creator
            # owns pruning, instead of a hand-rolled system preamble.
            if cross_session:
                agent.memory.write_records(
                    [
                        camel.MemoryRecord(
                            message=camel.BaseMessage.make_assistant_message(
                                role_name="memory", content=cross_session,
                            ),
                            role_at_backend=camel.OpenAIBackendRole.SYSTEM,
                        )
                    ]
                )
            # In-session prior turns as CAMEL memory records.
            for m in prior_turns:
                if m.get("role") == "user":
                    agent.memory.write_records(
                        [
                            camel.MemoryRecord(
                                message=camel.BaseMessage.make_user_message(
                                    role_name="user", content=m["content"],
                                ),
                                role_at_backend=camel.OpenAIBackendRole.USER,
                            )
                        ]
                    )
                elif m.get("role") == "assistant":
                    agent.memory.write_records(
                        [
                            camel.MemoryRecord(
                                message=camel.BaseMessage.make_assistant_message(
                                    role_name="assistant", content=m["content"],
                                ),
                                role_at_backend=camel.OpenAIBackendRole.ASSISTANT,
                            )
                        ]
                    )
            user_msg = camel.BaseMessage.make_user_message(
                role_name="user", content=last_user,
            )
            response = agent.step(user_msg)
            msgs = getattr(response, "msgs", None) or []
            if not msgs:
                raise CamelBackendError("CAMEL ChatAgent returned no messages")
            text = str(msgs[0].content).strip()
            usage = _extract_camel_usage(response)
            return AgentReply(
                text=text,
                model_id=self._model_type,
                usage_prompt_tokens=usage[0],
                usage_completion_tokens=usage[1],
                raw=build_openai_envelope(
                    text=text,
                    model=self._model_type,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                ),
            )

        try:
            return await asyncio.to_thread(_run)
        except CamelBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 -- re-raise as typed backend error
            raise CamelBackendError(
                f"CAMEL respond failed for scope={scope_key} session={session_id}: {exc}"
            ) from exc

    async def compact(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> SessionMemoryRecord:
        import asyncio

        prompt = COMPACTION_PROMPT_TEMPLATE.format(
            transcript_text=_format_transcript(transcript),
        )

        def _run() -> str:
            camel = self._camel
            model = self._build_compaction_model(temperature=0.0, max_tokens=600)
            agent = camel.ChatAgent(
                system_message="You are a precise transcript memory compactor.",
                model=model,
            )
            user_msg = camel.BaseMessage.make_user_message(
                role_name="user", content=prompt,
            )
            response = agent.step(user_msg)
            msgs = getattr(response, "msgs", None) or []
            if not msgs:
                raise CamelBackendError("CAMEL compaction returned no messages")
            return str(msgs[0].content)

        try:
            raw = await asyncio.to_thread(_run)
        except CamelBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 -- re-raise as typed backend error
            raise CamelBackendError(
                f"CAMEL compaction failed for scope={scope_key} session={session_id}: {exc}"
            ) from exc
        parsed = parse_compaction_json(raw)
        topic = str(parsed.get("topic", "")).strip()[:120]
        if not topic:
            raise CamelBackendError(
                f"compaction returned empty topic for scope={scope_key} session={session_id}"
            )
        return SessionMemoryRecord(
            scope_key=scope_key,
            session_id=session_id,
            topic=topic,
            salient=_coerce_salient(parsed.get("salient")),
            extracted_at=_now_iso(),
            extractor_model=self._compaction_model_type,
        )

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# Lazy camel-ai import shim
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _CamelShim:
    """Bundles the camel-ai symbols the backend needs."""

    ChatAgent: Any
    ModelFactory: Any
    ModelPlatformType: Any
    BaseMessage: Any
    ChatHistoryMemory: Any
    ScoreBasedContextCreator: Any
    MemoryRecord: Any
    OpenAIBackendRole: Any


def _import_camel() -> _CamelShim:
    try:
        from camel.agents import ChatAgent
        from camel.memories import (
            ChatHistoryMemory,
            MemoryRecord,
            ScoreBasedContextCreator,
        )
        from camel.messages import BaseMessage
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, OpenAIBackendRole
    except ImportError as exc:  # pragma: no cover - exercised only without camel-ai
        raise CamelBackendError(
            "camel-ai is not installed. Install the optional extra: "
            "pip install companion-camel-baseline[camel] (or use --backend echo "
            "for the deterministic smoke path)."
        ) from exc
    return _CamelShim(
        ChatAgent=ChatAgent,
        ModelFactory=ModelFactory,
        ModelPlatformType=ModelPlatformType,
        BaseMessage=BaseMessage,
        ChatHistoryMemory=ChatHistoryMemory,
        ScoreBasedContextCreator=ScoreBasedContextCreator,
        MemoryRecord=MemoryRecord,
        OpenAIBackendRole=OpenAIBackendRole,
    )


def _extract_camel_usage(response: Any) -> tuple[int | None, int | None]:
    """Best-effort token usage extraction from a CAMEL response object.

    CAMEL exposes usage under ``response.info['usage']`` in recent versions.
    Missing usage is reported as ``None`` (never silently billed at 0).
    """

    info = getattr(response, "info", None)
    if not isinstance(info, dict):
        return (None, None)
    usage = info.get("usage")
    if not isinstance(usage, dict):
        return (None, None)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    prompt_i = int(prompt) if isinstance(prompt, (int, float)) else None
    completion_i = int(completion) if isinstance(completion, (int, float)) else None
    return (prompt_i, completion_i)
