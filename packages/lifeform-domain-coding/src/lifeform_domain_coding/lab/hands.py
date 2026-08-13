"""Hands (the executing coder) for coding-lab episodes.

Two implementations share one protocol:

* :class:`ScriptedHand` — deterministic generator-side hand with
  controllable error modes. It is allowed to know task ground truth;
  its purpose is machinery calibration (oracle teeth, pass-rate knobs,
  trajectory plumbing), never capability claims.
* :class:`OpenAICompatHand` — a real API coder behind any
  OpenAI-compatible endpoint (DashScope / OpenRouter / Moonshot...).
  Temperature 0, provider pinned via ``extra_body``, per-call usage and
  model fingerprints surfaced for lineage. API hands are inherently
  non-deterministic; replay relies on trajectory logs, not reruns.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol

from lifeform_domain_coding.lab.tasks import ChainTask, FileEdit

MODE_CORRECT = "correct"
MODE_ACCEPTANCE_SABOTAGE = "acceptance_sabotage"
MODE_INVARIANT_SABOTAGE = "invariant_sabotage"


@dataclass(frozen=True)
class TranscriptEntry:
    """One completed tool call visible to the hand."""

    tool_name: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    succeeded: bool


@dataclass(frozen=True)
class HandContext:
    """Everything the hand may condition on for its next action.

    ``context_preamble`` is the arm-controlled context block (Packet 2):
    brain-arm memory digest, steelman-arm full transcript, or empty for
    the stateless arm. Hands that use context must treat it as advisory
    input; the task description remains the task's SSOT.
    """

    task_id: str
    task_description: str
    package_name: str
    step_index: int
    max_steps: int
    transcript: tuple[TranscriptEntry, ...]
    context_preamble: str = ""


@dataclass(frozen=True)
class HandAction:
    """The hand's next move: one tool call or a final submission."""

    kind: str  # "tool" | "submit"
    tool_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("tool", "submit"):
            raise ValueError(f"HandAction.kind must be 'tool' or 'submit', got {self.kind!r}")
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("tool actions require tool_name")


@dataclass(frozen=True)
class HandDecision:
    """Action plus per-call metadata (token usage, model fingerprint)."""

    action: HandAction
    metadata: dict[str, Any] = field(default_factory=dict)


class Hand(Protocol):
    """Protocol every hand implementation satisfies."""

    def hand_id(self) -> str: ...

    async def decide(self, context: HandContext) -> HandDecision: ...


def apply_edit_to_text(content: str, edit: FileEdit) -> str:
    """String-level twin of :func:`workspace.apply_edit` (fail loudly)."""

    if not edit.old:
        return content + edit.new
    occurrences = content.count(edit.old)
    if occurrences != 1:
        raise ValueError(
            f"edit anchor must occur exactly once in {edit.path!r}, found {occurrences}: {edit.old[:80]!r}"
        )
    return content.replace(edit.old, edit.new, 1)


# ---------------------------------------------------------------------------
# Scripted hand
# ---------------------------------------------------------------------------

_FAST_TEST_BY_MODULE: dict[str, str] = {
    "config.py": "tests/fast/test_config.py",
    "store.py": "tests/fast/test_store.py",
    "pricing.py": "tests/fast/test_pricing.py",
}


class ScriptedHand:
    """Deterministic hand with seeded error-mode mixture.

    ``mode_rates`` maps mode name to probability mass; remaining mass
    goes to :data:`MODE_CORRECT`. Mode is drawn once per episode from
    ``hand_seed`` and the episode's position, so a chain replay with the
    same seed reproduces the same behaviour byte-for-byte.
    """

    def __init__(
        self,
        *,
        tasks_by_id: dict[str, ChainTask],
        episode_index_by_task_id: dict[str, int],
        hand_seed: int,
        invariant_sabotage_rate: float = 0.2,
        acceptance_sabotage_rate: float = 0.2,
    ) -> None:
        if invariant_sabotage_rate < 0 or acceptance_sabotage_rate < 0:
            raise ValueError("sabotage rates must be non-negative")
        if invariant_sabotage_rate + acceptance_sabotage_rate > 1.0:
            raise ValueError("sabotage rates must sum to <= 1.0")
        self._tasks_by_id = dict(tasks_by_id)
        self._episode_index_by_task_id = dict(episode_index_by_task_id)
        self._hand_seed = hand_seed
        self._invariant_rate = invariant_sabotage_rate
        self._acceptance_rate = acceptance_sabotage_rate

    def hand_id(self) -> str:
        return (
            f"scripted(seed={self._hand_seed},inv={self._invariant_rate},acc={self._acceptance_rate})"
        )

    def episode_mode(self, task_id: str) -> str:
        """The (deterministic) error mode this hand uses for ``task_id``."""

        task = self._tasks_by_id[task_id]
        episode_index = self._episode_index_by_task_id[task_id]
        draw = Random(self._hand_seed * 1_000_003 + episode_index * 7_919).random()
        if draw < self._invariant_rate and task.invariant_sabotage_edits:
            return MODE_INVARIANT_SABOTAGE
        if draw < self._invariant_rate + self._acceptance_rate:
            return MODE_ACCEPTANCE_SABOTAGE
        return MODE_CORRECT

    def _edits_for_mode(self, task: ChainTask, mode: str) -> tuple[FileEdit, ...]:
        if mode == MODE_CORRECT:
            return task.reference_edits
        if mode == MODE_INVARIANT_SABOTAGE:
            return task.invariant_sabotage_edits
        return task.acceptance_sabotage_edits

    def _effective_mode(self, context: HandContext, task: ChainTask, drawn_mode: str) -> str:
        """Hook for context-conditioned variants; base hand ignores context."""

        del context, task
        return drawn_mode

    async def decide(self, context: HandContext) -> HandDecision:
        task = self._tasks_by_id[context.task_id]
        mode = self._effective_mode(context, task, self.episode_mode(context.task_id))
        edits = self._edits_for_mode(task, mode)
        edited_paths = tuple(dict.fromkeys(edit.path for edit in edits))
        metadata = {"scripted_mode": mode}
        reads = len(edited_paths)
        step = context.step_index

        if step < reads:
            return HandDecision(
                action=HandAction(
                    kind="tool", tool_name="read_file", parameters={"path": edited_paths[step]}
                ),
                metadata=metadata,
            )
        if step < 2 * reads:
            path = edited_paths[step - reads]
            read_entry = next(
                (
                    entry
                    for entry in context.transcript
                    if entry.tool_name == "read_file" and entry.parameters.get("path") == path
                ),
                None,
            )
            if read_entry is None or not read_entry.succeeded:
                raise RuntimeError(f"scripted hand needs a completed read of {path!r} before writing")
            content = str(read_entry.result["content"])
            for edit in edits:
                if edit.path == path:
                    content = apply_edit_to_text(content, edit)
            return HandDecision(
                action=HandAction(
                    kind="tool",
                    tool_name="write_file",
                    parameters={"path": path, "content": content, "mode": "overwrite"},
                ),
                metadata=metadata,
            )
        module_name = edited_paths[0].rsplit("/", 1)[-1] if edited_paths else ""
        fast_test = _FAST_TEST_BY_MODULE.get(module_name)
        if fast_test is not None and step == 2 * reads:
            return HandDecision(
                action=HandAction(
                    kind="tool", tool_name="run_test", parameters={"test_path": fast_test}
                ),
                metadata=metadata,
            )
        return HandDecision(
            action=HandAction(kind="submit", note=f"scripted:{mode}"), metadata=metadata
        )


class MemoryAwareScriptedHand(ScriptedHand):
    """Scripted hand whose error mode is context-conditioned (smoke only).

    Instrument-calibration device for the Packet 2 measurement spine: it
    injects a KNOWN effect (avoid a failure mode when the context
    preamble mentions the category's needle string) so the slope
    statistics can be validated against a ground-truth direction. It is
    NOT evidence of memory value — both the brain digest and the full
    steelman transcript naturally contain the needles, which is exactly
    why the smoke's quality gate on brain-vs-steelman is expected null
    while brain-vs-stateless must be positive.
    """

    def __init__(self, *, needles_by_category: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._needles_by_category = dict(needles_by_category)

    def hand_id(self) -> str:
        return f"memory-aware-{super().hand_id()}"

    def _effective_mode(self, context: HandContext, task: ChainTask, drawn_mode: str) -> str:
        needle = self._needles_by_category.get(task.category, "")
        if not needle:
            return drawn_mode
        # Deterministic on the needle category: the known effect must not
        # be diluted by the stochastic sabotage draw, otherwise a lucky
        # no-sabotage draw in the stateless arm erases the ground-truth
        # direction the smoke is calibrating against (2026-08-12 smoke:
        # one whole chain drew zero sabotages and contributed zero
        # signal). Non-needle categories keep the drawn mode so the
        # slope statistics still see baseline variance.
        if needle in context.context_preamble:
            return MODE_CORRECT
        return MODE_ACCEPTANCE_SABOTAGE


# ---------------------------------------------------------------------------
# OpenAI-compatible API hand
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a literal string across workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "subdir": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file (mode: create | overwrite | append).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["create", "overwrite", "append"]},
                },
                "required": ["path", "content", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "Run pytest on one test file or node id inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"test_path": {"type": "string"}},
                "required": ["test_path"],
            },
        },
    },
)

_SYSTEM_PROMPT = (
    "You are a careful software engineer working inside a sandboxed Python repository. "
    "Use the provided tools to inspect the code, implement exactly what the task asks, and verify "
    "with tests where useful. The full test suite (including tests you may not have run) must keep "
    "passing. When you are confident the change is complete, reply WITHOUT any tool call and briefly "
    "state what you changed."
)


@dataclass(frozen=True)
class APIHandConfig:
    """Frozen lineage-bearing configuration of an API hand."""

    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be http(s), got {self.base_url!r}")
        if not self.model:
            raise ValueError("model must be non-empty")


class OpenAICompatHand:
    """Chat-completions tool-calling hand for OpenAI-compatible APIs."""

    def __init__(self, config: APIHandConfig) -> None:
        self._config = config
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"API hand requires environment variable {config.api_key_env!r} to be set"
            )
        self._api_key = api_key

    def hand_id(self) -> str:
        return f"api({self._config.model}@{self._config.base_url})"

    def _messages(self, context: HandContext) -> list[dict[str, Any]]:
        preamble_block = (
            f"Project context from prior work:\n{context.context_preamble}\n\n"
            if context.context_preamble
            else ""
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{preamble_block}"
                    f"Task in package `{context.package_name}`:\n\n{context.task_description}\n\n"
                    f"You have at most {context.max_steps} tool calls."
                ),
            },
        ]
        for index, entry in enumerate(context.transcript):
            call_id = f"call-{index:03d}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": entry.tool_name,
                                "arguments": json.dumps(entry.parameters, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(entry.result, ensure_ascii=False)[:20_000],
                }
            )
        return messages

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with bounded retry on transient transport failures.

        Long calibration runs (tens of minutes of serial calls) will hit
        occasional SSL EOFs / timeouts / 429s; those are provider-side
        noise, not contract violations. Retries are bounded with
        exponential backoff and the last error is re-raised with context
        — 4xx protocol errors (other than 429) fail immediately.
        """

        data = json.dumps(payload).encode("utf-8")
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        attempts = 4
        last_error: Exception | None = None
        for attempt in range(attempts):
            request = urllib.request.Request(
                url=url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self._config.timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code != 429 and error.code < 500:
                    raise
                last_error = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
            if attempt < attempts - 1:
                time.sleep(2.0 * (2**attempt))
        raise RuntimeError(
            f"API hand POST failed after {attempts} attempts: {last_error!r}"
        ) from last_error

    async def decide(self, context: HandContext) -> HandDecision:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
            "messages": self._messages(context),
            "tools": list(_TOOL_SCHEMAS),
        }
        payload.update(self._config.extra_body)
        response = await asyncio.to_thread(self._post, payload)
        choice = response["choices"][0]
        message = choice["message"]
        metadata = {
            "response_model": response.get("model", ""),
            "system_fingerprint": response.get("system_fingerprint", ""),
            "usage": response.get("usage", {}),
            "finish_reason": choice.get("finish_reason", ""),
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            arguments_raw = call["function"].get("arguments", "{}")
            parameters = json.loads(arguments_raw) if arguments_raw else {}
            if not isinstance(parameters, dict):
                raise ValueError(f"tool arguments must be an object, got {type(parameters).__name__}")
            return HandDecision(
                action=HandAction(
                    kind="tool",
                    tool_name=str(call["function"]["name"]),
                    parameters=parameters,
                ),
                metadata=metadata,
            )
        return HandDecision(
            action=HandAction(kind="submit", note=str(message.get("content") or "")[:2000]),
            metadata=metadata,
        )


__all__ = [
    "APIHandConfig",
    "Hand",
    "HandAction",
    "HandContext",
    "HandDecision",
    "MODE_ACCEPTANCE_SABOTAGE",
    "MODE_CORRECT",
    "MODE_INVARIANT_SABOTAGE",
    "MemoryAwareScriptedHand",
    "OpenAICompatHand",
    "ScriptedHand",
    "TranscriptEntry",
    "apply_edit_to_text",
]
