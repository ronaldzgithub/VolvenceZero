# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Episodic-memory component (H-C): salient events from past sessions.

Where the user-model captures *stable* facts, episodic memory captures *events*
— things that happened ("user adopted a kitten", "user started a new job") —
with enough recency ordering that the wrapped model can refer back to them.

* :class:`EpisodicEvent` — one event.
* :class:`EpisodicExtractor` — protocol invoked at session close.
* :class:`StubEpisodicExtractor` — deterministic, no-LLM, used by tests / smoke.
* :class:`LLMEpisodicExtractor` — production extractor, strict-JSON output.

Prompt inlined per the reproducibility contract.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


EPISODIC_EXTRACTOR_PROMPT_TEMPLATE: str = """You maintain an episodic memory for a long-running companion AI.
Read the session transcript and emit ONE JSON object listing salient EVENTS
(things that happened or changed) worth recalling in future sessions.

Schema:
{{
  "events": ["<each event <= 140 chars, factual, past-tense>"]
}}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences.
- Events are happenings/changes, not stable traits (those are the user-model's
  job). E.g. "user adopted a kitten named Mia", "user moved to Berlin".
- Do not invent. If nothing notable happened, return {{"events": []}}.

Transcript (chronological):

{transcript_text}
"""

MAX_EVENTS_IN_PREFIX: int = 6


@dataclasses.dataclass(frozen=True)
class EpisodicEvent:
    """One salient past event."""

    scope_key: str
    event_id: str
    summary: str
    source_turn: str
    ts: str  # ISO-8601 UTC

    def to_payload(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "summary": self.summary}

    def to_prompt_line(self) -> str:
        return f"- {self.summary}"


@runtime_checkable
class EpisodicExtractor(Protocol):
    @property
    def model(self) -> str: ...

    def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> Awaitable[tuple[EpisodicEvent, ...]]: ...


def _event_id(scope_key: str, session_id: str, summary: str) -> str:
    digest = hashlib.blake2b(
        f"{scope_key}|{session_id}|{summary}".encode("utf-8"), digest_size=8,
    ).hexdigest()
    return f"ev-{digest}"


class StubEpisodicExtractor:
    """Deterministic extractor: treats each user turn as a candidate event.

    Test/smoke stand-in only. Production uses :class:`LLMEpisodicExtractor`.
    """

    def __init__(self, *, model: str = "ref-harness/stub-episodic-v1") -> None:
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
    ) -> tuple[EpisodicEvent, ...]:
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        events: list[EpisodicEvent] = []
        for msg in transcript:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            summary = content[:140]
            events.append(
                EpisodicEvent(
                    scope_key=scope_key,
                    event_id=_event_id(scope_key, session_id, summary),
                    summary=summary,
                    source_turn=session_id,
                    ts=ts,
                )
            )
            if len(events) >= MAX_EVENTS_IN_PREFIX:
                break
        return tuple(events)


UpstreamChatCall = Callable[[list[dict[str, str]]], Awaitable[str]]


class LLMEpisodicExtractor:
    """Production episodic extractor calling an upstream chat endpoint."""

    def __init__(
        self,
        *,
        model: str,
        upstream_call: UpstreamChatCall,
        max_events: int = MAX_EVENTS_IN_PREFIX,
    ) -> None:
        self._model = model
        self._upstream_call = upstream_call
        self._max_events = max_events

    @property
    def model(self) -> str:
        return self._model

    async def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> tuple[EpisodicEvent, ...]:
        prompt = EPISODIC_EXTRACTOR_PROMPT_TEMPLATE.format(
            transcript_text=_format_transcript(transcript),
        )
        raw = await self._upstream_call([{"role": "user", "content": prompt}])
        parsed = _parse_json(raw)
        raw_events = parsed.get("events")
        if raw_events is None:
            return ()
        if not isinstance(raw_events, list):
            raise ValueError(
                f"episodic extractor 'events' must be a list; got {type(raw_events).__name__}"
            )
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        out: list[EpisodicEvent] = []
        for item in raw_events:
            if not isinstance(item, str):
                raise ValueError("each event must be a string")
            summary = item.strip()[:140]
            if not summary:
                continue
            out.append(
                EpisodicEvent(
                    scope_key=scope_key,
                    event_id=_event_id(scope_key, session_id, summary),
                    summary=summary,
                    source_turn=session_id,
                    ts=ts,
                )
            )
            if len(out) >= self._max_events:
                break
        return tuple(out)


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


def _parse_json(raw: str) -> dict[str, Any]:
    raw_stripped = (raw or "").strip()
    try:
        return json.loads(raw_stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw_stripped)
    if match is None:
        raise ValueError(
            f"episodic extractor response contained no JSON object: {raw_stripped[:160]!r}"
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"episodic extractor returned malformed JSON: {exc}; raw[:160]={raw_stripped[:160]!r}"
        ) from exc
