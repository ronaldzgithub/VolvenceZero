# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Session summary component — extracts and stores a per-session summary.

The summary is **structured**: ``topic`` (one short sentence),
``commitments`` (things the user or assistant committed to do), and
``open_loops`` (questions or threads left unresolved at session end).
Structure is deliberately small so the prompt-fragment that injects
it into session N+1 stays compact.

This module defines:

* :class:`SessionSummary` — the typed value object (frozen dataclass).
* :class:`SummaryExtractor` — the Protocol that the server invokes
  at session boundaries.
* :class:`StubSummaryExtractor` — deterministic fake used by tests
  and by ``--components ""`` passthrough mode.
* :class:`LLMSummaryExtractor` — production extractor that calls an
  upstream chat-completions endpoint and parses strict JSON output.

LLM extractor prompt
====================

The exact prompt the LLM extractor sends is defined as the module-
level constant :data:`SUMMARY_EXTRACTOR_PROMPT_TEMPLATE` below. It
is inlined into source on purpose — the reproducibility contract in
the package README forbids hiding prompts in remote configs.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Prompt (inlined per reproducibility contract)
# ---------------------------------------------------------------------------


SUMMARY_EXTRACTOR_PROMPT_TEMPLATE: str = """You are a strict transcript summariser for a long-running companion AI.
Read the conversation transcript below and emit ONE compact JSON object
that captures what would be useful for the next session.

Schema:
{{
  "topic": "<one short sentence, <= 120 chars, no quotes>",
  "commitments": ["<each item <= 120 chars, factual only>"],
  "open_loops": ["<each item <= 120 chars, factual only>"]
}}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences, no preface.
- Do not invent facts. If the transcript does not state something, omit it.
- "commitments" are things the user OR the assistant explicitly said they
  would do (with the speaker prefix, e.g., "user: will return Tuesday").
- "open_loops" are unresolved questions or threads at the END of the
  transcript (e.g., "assistant has not yet replied about the move date").
- Both arrays may be empty.
- The summary will be shown to the SAME assistant at the start of the
  next session; write it as a note-to-self, not as a third-person report.

Transcript (chronological, alternating user/assistant):

{transcript_text}
"""


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SessionSummary:
    """One per-session summary, immutable after extraction."""

    scope_key: str
    session_id: str
    topic: str
    commitments: tuple[str, ...]
    open_loops: tuple[str, ...]
    extracted_at: str  # ISO-8601 UTC
    extractor_model: str

    def to_payload(self) -> dict[str, Any]:
        """Stable JSON shape stored in SQLite and returned by API."""
        return {
            "topic": self.topic,
            "commitments": list(self.commitments),
            "open_loops": list(self.open_loops),
        }

    def to_prompt_block(self) -> str:
        """Markdown fragment injected into the next session's system prefix."""

        lines: list[str] = [f"- topic: {self.topic}"]
        if self.commitments:
            lines.append("- commitments:")
            for c in self.commitments:
                lines.append(f"    - {c}")
        if self.open_loops:
            lines.append("- open loops:")
            for o in self.open_loops:
                lines.append(f"    - {o}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SummaryExtractor(Protocol):
    """Surface the server invokes at session-close.

    Implementations may be synchronous or async; the harness server
    awaits an awaitable returned by :meth:`extract` either way.
    """

    @property
    def model(self) -> str: ...

    def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> Awaitable[SessionSummary]: ...


# ---------------------------------------------------------------------------
# Deterministic stub (tests / passthrough)
# ---------------------------------------------------------------------------


class StubSummaryExtractor:
    """Produces a deterministic summary derived only from transcript shape.

    Used by unit tests and as the default ``SummaryExtractor`` when
    the harness is booted with ``--components ""`` (passthrough) so
    that any code path that asks for the extractor still gets a
    valid implementation rather than a ``None``.
    """

    def __init__(self, *, model: str = "ref-harness/stub-summary-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> SessionSummary:
        user_turns = [m for m in transcript if m.get("role") == "user"]
        assistant_turns = [m for m in transcript if m.get("role") == "assistant"]
        topic_seed = (user_turns[0]["content"] if user_turns else "(empty)").strip()
        topic = topic_seed[:120] if topic_seed else "(empty)"
        commitments: tuple[str, ...] = ()
        open_loops: tuple[str, ...] = ()
        if user_turns and not assistant_turns:
            open_loops = ("assistant has not replied yet",)
        return SessionSummary(
            scope_key=scope_key,
            session_id=session_id,
            topic=topic,
            commitments=commitments,
            open_loops=open_loops,
            extracted_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            extractor_model=self._model,
        )


# ---------------------------------------------------------------------------
# LLM-backed extractor (production)
# ---------------------------------------------------------------------------


# The "call this upstream and get an assistant message" hook. Concrete
# instantiation lives in upstream_client.py; we inject a callable here
# so this module never imports the upstream client transitively.
UpstreamChatCall = Callable[
    [list[dict[str, str]]], Awaitable[str]
]


class LLMSummaryExtractor:
    """Production summary extractor calling an upstream chat endpoint.

    The upstream call signature is intentionally minimal: one
    ``messages`` list in, the assistant text out. The caller is
    expected to pre-configure model / temperature / max_tokens on
    the upstream client.
    """

    def __init__(
        self,
        *,
        model: str,
        upstream_call: UpstreamChatCall,
        max_topic_chars: int = 120,
        max_list_items: int = 8,
    ) -> None:
        self._model = model
        self._upstream_call = upstream_call
        self._max_topic_chars = max_topic_chars
        self._max_list_items = max_list_items

    @property
    def model(self) -> str:
        return self._model

    async def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> SessionSummary:
        prompt = SUMMARY_EXTRACTOR_PROMPT_TEMPLATE.format(
            transcript_text=_format_transcript(transcript),
        )
        # Use a single-message conversation: the prompt itself is the user
        # message. The upstream's system prompt (if any) is set by the
        # caller via the upstream client config.
        messages = [{"role": "user", "content": prompt}]
        raw_response = await self._upstream_call(messages)
        parsed = _parse_extractor_json(raw_response)
        topic = str(parsed.get("topic", "")).strip()[: self._max_topic_chars]
        if not topic:
            # Fail loud per .cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc:
            # an empty topic means the extractor produced unusable output and
            # we should surface it rather than silently storing a blank.
            raise ValueError(
                f"summary extractor returned empty topic for "
                f"scope_key={scope_key!r} session_id={session_id!r}"
            )
        commitments = _coerce_string_list(
            parsed.get("commitments"), limit=self._max_list_items,
        )
        open_loops = _coerce_string_list(
            parsed.get("open_loops"), limit=self._max_list_items,
        )
        return SessionSummary(
            scope_key=scope_key,
            session_id=session_id,
            topic=topic,
            commitments=commitments,
            open_loops=open_loops,
            extracted_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            extractor_model=self._model,
        )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _format_transcript(transcript: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in transcript:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not role or not content:
            continue
        if role == "system":
            # System prompts often carry persona instructions that are
            # already known to the next session by injection; summaries
            # do not need to repeat them.
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(empty transcript)"


_JSON_BLOCK_RE: re.Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)


def _parse_extractor_json(raw: str) -> dict[str, Any]:
    """Parse JSON out of the extractor response.

    Some upstream models like to wrap JSON in markdown fences or
    add a preamble despite our explicit prompt instruction. We
    extract the first balanced ``{...}`` block. Parsing failure is
    fail-loud (per no-swallow-errors rule).
    """

    raw_stripped = (raw or "").strip()
    try:
        return json.loads(raw_stripped)
    except json.JSONDecodeError:
        # Fall through to regex recovery
        pass
    match = _JSON_BLOCK_RE.search(raw_stripped)
    if match is None:
        raise ValueError(
            f"summary extractor response contained no JSON object: "
            f"{raw_stripped[:160]!r}"
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"summary extractor returned malformed JSON: {exc}; "
            f"raw[:160]={raw_stripped[:160]!r}"
        ) from exc


def _coerce_string_list(value: Any, *, limit: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"expected list for commitments/open_loops, got {type(value).__name__}"
        )
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"expected list of str, got element of type {type(item).__name__}"
            )
        item_stripped = item.strip()
        if not item_stripped:
            continue
        out.append(item_stripped[:120])
        if len(out) >= limit:
            break
    return tuple(out)
