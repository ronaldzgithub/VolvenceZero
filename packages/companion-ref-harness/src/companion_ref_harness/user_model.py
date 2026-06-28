# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""User-model component (H-C): durable key/value facts about the user.

At each session boundary the harness extracts a small set of stable facts about
the user (name, preferences, constraints, recurring people/topics) and persists
them keyed by ``scope_key``. They are re-injected into later sessions so the
wrapped model can personalise without re-asking.

* :class:`UserFact` — one extracted fact (key/value + provenance).
* :class:`UserFactExtractor` — the protocol invoked at session close.
* :class:`StubUserFactExtractor` — deterministic, no-LLM, used by tests / smoke.
* :class:`LLMUserFactExtractor` — production extractor, strict-JSON output.

The extractor prompt is the module-level constant
:data:`USER_FACT_EXTRACTOR_PROMPT_TEMPLATE`, inlined per the reproducibility
contract.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


USER_FACT_EXTRACTOR_PROMPT_TEMPLATE: str = """You maintain a durable profile of a user for a long-running companion AI.
Read the session transcript and emit ONE JSON object of STABLE facts worth
remembering across future sessions.

Schema:
{{
  "facts": [
    {{"key": "<short snake_case key>", "value": "<= 120 chars", "confidence": <0..1>}}
  ]
}}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences.
- Only durable facts: names, relationships, stable preferences, constraints,
  recurring topics. NOT one-off chit-chat.
- Do not invent. If the transcript states nothing durable, return {{"facts": []}}.
- "key" is a stable identifier you would reuse next time (e.g. "user_name",
  "child_name", "dislikes").

Transcript (chronological):

{transcript_text}
"""

MAX_FACTS_IN_PREFIX: int = 12


@dataclasses.dataclass(frozen=True)
class UserFact:
    """One durable user fact."""

    scope_key: str
    key: str
    value: str
    source_turn: str
    confidence: float
    ts: str  # ISO-8601 UTC

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
        }

    def to_prompt_line(self) -> str:
        return f"- {self.key}: {self.value}"


@runtime_checkable
class UserFactExtractor(Protocol):
    @property
    def model(self) -> str: ...

    def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> Awaitable[tuple[UserFact, ...]]: ...


class StubUserFactExtractor:
    """Deterministic extractor: keys off simple ``my name is`` / ``i like`` cues.

    This is a *test/smoke* stand-in, not a production NLP system. It exists so
    the H-C plumbing can be exercised end-to-end without an LLM. Production runs
    use :class:`LLMUserFactExtractor`.
    """

    def __init__(self, *, model: str = "ref-harness/stub-userfact-v1") -> None:
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
    ) -> tuple[UserFact, ...]:
        facts: list[UserFact] = []
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        for msg in transcript:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", "")).strip()
            low = content.lower()
            name_match = re.search(r"\bmy name is ([a-z][a-z '\-]{0,40})", low)
            if name_match:
                facts.append(
                    UserFact(
                        scope_key=scope_key,
                        key="user_name",
                        value=name_match.group(1).strip().title()[:120],
                        source_turn=session_id,
                        confidence=0.9,
                        ts=ts,
                    )
                )
            like_match = re.search(r"\bi (?:like|love|prefer) ([a-z0-9][^.!?]{0,80})", low)
            if like_match:
                facts.append(
                    UserFact(
                        scope_key=scope_key,
                        key="preference",
                        value=like_match.group(1).strip()[:120],
                        source_turn=session_id,
                        confidence=0.6,
                        ts=ts,
                    )
                )
        return tuple(facts)


# The "call upstream, get assistant text" hook (same shape the summary
# extractor uses) so this module never imports the upstream client.
UpstreamChatCall = Callable[[list[dict[str, str]]], Awaitable[str]]


class LLMUserFactExtractor:
    """Production user-fact extractor calling an upstream chat endpoint."""

    def __init__(
        self,
        *,
        model: str,
        upstream_call: UpstreamChatCall,
        max_facts: int = MAX_FACTS_IN_PREFIX,
    ) -> None:
        self._model = model
        self._upstream_call = upstream_call
        self._max_facts = max_facts

    @property
    def model(self) -> str:
        return self._model

    async def extract(
        self,
        *,
        scope_key: str,
        session_id: str,
        transcript: list[dict[str, str]],
    ) -> tuple[UserFact, ...]:
        prompt = USER_FACT_EXTRACTOR_PROMPT_TEMPLATE.format(
            transcript_text=_format_transcript(transcript),
        )
        raw = await self._upstream_call([{"role": "user", "content": prompt}])
        parsed = _parse_json(raw)
        raw_facts = parsed.get("facts")
        if raw_facts is None:
            return ()
        if not isinstance(raw_facts, list):
            raise ValueError(
                f"user-fact extractor 'facts' must be a list; got {type(raw_facts).__name__}"
            )
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        out: list[UserFact] = []
        for item in raw_facts:
            if not isinstance(item, dict):
                raise ValueError("each fact must be an object")
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()
            if not key or not value:
                continue
            confidence = item.get("confidence", 0.5)
            try:
                conf = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"confidence must be a number; got {confidence!r}") from exc
            out.append(
                UserFact(
                    scope_key=scope_key,
                    key=key[:60],
                    value=value[:120],
                    source_turn=session_id,
                    confidence=max(0.0, min(1.0, conf)),
                    ts=ts,
                )
            )
            if len(out) >= self._max_facts:
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
            f"user-fact extractor response contained no JSON object: {raw_stripped[:160]!r}"
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"user-fact extractor returned malformed JSON: {exc}; raw[:160]={raw_stripped[:160]!r}"
        ) from exc
