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
import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol

from lifeform_domain_coding.lab.junctions import (
    ACTION_EDIT,
    ACTION_INVESTIGATE,
    ACTION_SUBMIT,
    ACTION_TEST,
    FORCED_ASSIGNMENT_METADATA_KEY,
    FORCED_ASSIGNMENT_SCHEMA_VERSION,
    JUNCTION_ACTIONS,
    action_class_for_tool,
    state_key_for,
    transcript_protocol_state,
)
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
# Forced-action wrapper (Packet 3.5 junction RCT)
# ---------------------------------------------------------------------------

_FORCED_DIRECTIVE_MARKER = "[FORCED-ACTION DIRECTIVE]"
_FORCED_DIRECTIVE_TOOL_HINTS: dict[str, str] = {
    ACTION_EDIT: "write_file",
    ACTION_TEST: "run_test",
}


def _forced_directive(action_class: str) -> str:
    tool_hint = _FORCED_DIRECTIVE_TOOL_HINTS[action_class]
    return (
        f"\n\n{_FORCED_DIRECTIVE_MARKER} For your NEXT action only, you MUST call a "
        f"tool of class {action_class} ({tool_hint}). Choose the specific target "
        "and content yourself; afterwards continue the task normally."
    )


def _action_class_of(action: HandAction) -> str:
    if action.kind == "submit":
        return ACTION_SUBMIT
    return action_class_for_tool(action.tool_name)


@dataclass(frozen=True)
class ForcedActionAssignment:
    """Randomized per-episode assignment, drawn and logged by the runner.

    ``assigned_action=None`` marks a control episode: the wrapper only
    annotates the first target-state decision, never changes behaviour.
    """

    target_state_keys: tuple[str, ...]
    assigned_action: str | None
    assignment_id: str

    def __post_init__(self) -> None:
        if not self.target_state_keys:
            raise ValueError("assignment requires at least one target state key")
        if self.assigned_action is not None and self.assigned_action not in JUNCTION_ACTIONS:
            raise ValueError(
                f"assigned_action must be one of {JUNCTION_ACTIONS} or None, "
                f"got {self.assigned_action!r}"
            )
        if not self.assignment_id:
            raise ValueError("assignment_id must be non-empty")


class ForcedActionHand:
    """One-shot junction intervention wrapper (Packet 3.5 RCT machinery).

    Tracks the protocol state implied by the completed-call transcript
    (same SSOT state machine as the junction corpus). At the FIRST
    decision whose state key is in the assignment's target set it
    realizes the assigned action class exactly once:

    * ``submit`` / ``investigate`` — realized directly by the wrapper
      (deterministic compliance, no ground-truth leakage: submit takes no
      parameters, investigate is ``list_dir "."``);
    * ``edit`` / ``test`` — realized by constraining the inner hand via a
      directive appended to ``context_preamble`` (the hand picks its own
      target/content, so no oracle knowledge enters). Compliance is
      checked against the protocol action class with bounded retries;
      the analysis unit is the ASSIGNMENT (intention-to-treat), so a
      noncompliant episode keeps its natural action and its record.

    Control assignments annotate the natural decision without changing
    it. Every trigger writes one metadata record into the trajectory's
    ``hand_decision`` event — the trajectory stays the analysis SSOT.
    """

    def __init__(
        self,
        *,
        inner: Hand,
        category: str,
        assignment: ForcedActionAssignment,
        max_constraint_attempts: int = 2,
    ) -> None:
        if max_constraint_attempts < 1:
            raise ValueError("max_constraint_attempts must be >= 1")
        self._inner = inner
        self._category = category
        self._assignment = assignment
        self._max_constraint_attempts = max_constraint_attempts
        self._triggered = False

    def hand_id(self) -> str:
        label = self._assignment.assigned_action or "control"
        return f"forced[{label}]+{self._inner.hand_id()}"

    @property
    def triggered(self) -> bool:
        return self._triggered

    def _base_record(self, *, state_key: str, step_index: int) -> dict[str, Any]:
        return {
            "schema_version": FORCED_ASSIGNMENT_SCHEMA_VERSION,
            "assignment_id": self._assignment.assignment_id,
            "state_key": state_key,
            "step_index": step_index,
        }

    async def decide(self, context: HandContext) -> HandDecision:
        if self._triggered:
            return await self._inner.decide(context)
        state = transcript_protocol_state(
            tuple(
                (entry.tool_name, entry.succeeded, entry.result.get("exit_code"))
                for entry in context.transcript
            )
        )
        state_key = state_key_for(
            category=self._category,
            investigate_count=int(state["investigate_count"]),
            has_edited=bool(state["has_edited"]),
            test_state=str(state["test_state"]),
        )
        if state_key not in self._assignment.target_state_keys:
            return await self._inner.decide(context)
        self._triggered = True
        record = self._base_record(state_key=state_key, step_index=context.step_index)
        assigned = self._assignment.assigned_action

        if assigned is None:
            decision = await self._inner.decide(context)
            record.update(
                {
                    "arm": "control",
                    "assigned_action": None,
                    "realized_action": _action_class_of(decision.action),
                    "compliant": True,
                    "decide_attempts": 1,
                    "realization": "natural",
                }
            )
            return HandDecision(
                action=decision.action,
                metadata={**decision.metadata, FORCED_ASSIGNMENT_METADATA_KEY: record},
            )

        if assigned == ACTION_SUBMIT:
            record.update(
                {
                    "arm": "intervention",
                    "assigned_action": assigned,
                    "realized_action": ACTION_SUBMIT,
                    "compliant": True,
                    "decide_attempts": 0,
                    "realization": "direct",
                }
            )
            return HandDecision(
                action=HandAction(kind="submit", note="forced:submit"),
                metadata={FORCED_ASSIGNMENT_METADATA_KEY: record},
            )

        if assigned == ACTION_INVESTIGATE:
            record.update(
                {
                    "arm": "intervention",
                    "assigned_action": assigned,
                    "realized_action": ACTION_INVESTIGATE,
                    "compliant": True,
                    "decide_attempts": 0,
                    "realization": "direct",
                }
            )
            return HandDecision(
                action=HandAction(
                    kind="tool", tool_name="list_dir", parameters={"path": "."}
                ),
                metadata={FORCED_ASSIGNMENT_METADATA_KEY: record},
            )

        # edit / test: constrain the inner hand, intention-to-treat.
        directive = _forced_directive(assigned)
        constrained = dataclasses.replace(
            context, context_preamble=context.context_preamble + directive
        )
        decision: HandDecision | None = None
        attempts = 0
        compliant = False
        for _ in range(self._max_constraint_attempts):
            attempts += 1
            decision = await self._inner.decide(constrained)
            if _action_class_of(decision.action) == assigned:
                compliant = True
                break
        assert decision is not None  # max_constraint_attempts >= 1
        record.update(
            {
                "arm": "intervention",
                "assigned_action": assigned,
                "realized_action": _action_class_of(decision.action),
                "compliant": compliant,
                "decide_attempts": attempts,
                "realization": "constraint",
            }
        )
        return HandDecision(
            action=decision.action,
            metadata={**decision.metadata, FORCED_ASSIGNMENT_METADATA_KEY: record},
        )


class ConstraintAwareScriptedHand(ScriptedHand):
    """Closed-loop scripted hand that obeys forced-action directives (smoke only).

    Calibration device for the Packet 3.5 machinery. Two deliberate
    differences from :class:`ScriptedHand`:

    * it obeys ``[FORCED-ACTION DIRECTIVE]`` constraints, letting the
      compliant edit/test realization path be exercised without API
      cost. The forced-edit realization appends a trailing newline to
      the task's first reference path — a syntactically inert write
      chosen WITHOUT consulting the edit's ground-truth content;
    * its plan is CLOSED-LOOP: progress is derived from the transcript
      (which reads/writes actually completed) rather than from
      ``step_index`` arithmetic, so an injected forced step shifts the
      plan instead of derailing it — mirroring how a real API hand
      adapts to its own history.
    """

    def hand_id(self) -> str:
        return f"constraint-aware-{super().hand_id()}"

    async def decide(self, context: HandContext) -> HandDecision:
        task = self._tasks_by_id[context.task_id]
        if _FORCED_DIRECTIVE_MARKER in context.context_preamble:
            directive_class = (
                ACTION_EDIT if "class edit" in context.context_preamble else ACTION_TEST
            )
            first_path = (
                task.reference_edits[0].path if task.reference_edits else "src/config.py"
            )
            if directive_class == ACTION_EDIT:
                # mode=append so the closed-loop plan below does not count
                # this inert write as the task's real (overwrite) edit.
                action = HandAction(
                    kind="tool",
                    tool_name="write_file",
                    parameters={"path": first_path, "content": "\n", "mode": "append"},
                )
            else:
                module_name = first_path.rsplit("/", 1)[-1]
                test_path = _FAST_TEST_BY_MODULE.get(module_name, "tests/fast/test_config.py")
                action = HandAction(
                    kind="tool", tool_name="run_test", parameters={"test_path": test_path}
                )
            return HandDecision(
                action=action, metadata={"scripted_mode": "constraint_compliant"}
            )

        mode = self._effective_mode(context, task, self.episode_mode(context.task_id))
        edits = self._edits_for_mode(task, mode)
        edited_paths = tuple(dict.fromkeys(edit.path for edit in edits))
        metadata = {"scripted_mode": mode}
        reads_completed: dict[str, str] = {}
        overwrites_done: set[str] = set()
        test_run = False
        for entry in context.transcript:
            if entry.tool_name == "read_file" and entry.succeeded:
                reads_completed[str(entry.parameters.get("path"))] = str(
                    entry.result.get("content", "")
                )
            elif entry.tool_name == "write_file" and entry.parameters.get("mode") == "overwrite":
                overwrites_done.add(str(entry.parameters.get("path")))
            elif entry.tool_name == "run_test":
                test_run = True
        for path in edited_paths:
            if path not in reads_completed:
                return HandDecision(
                    action=HandAction(
                        kind="tool", tool_name="read_file", parameters={"path": path}
                    ),
                    metadata=metadata,
                )
        for path in edited_paths:
            if path not in overwrites_done:
                content = reads_completed[path]
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
        if fast_test is not None and not test_run:
            return HandDecision(
                action=HandAction(
                    kind="tool", tool_name="run_test", parameters={"test_path": fast_test}
                ),
                metadata=metadata,
            )
        return HandDecision(
            action=HandAction(kind="submit", note=f"scripted:{mode}"), metadata=metadata
        )


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
            # Provider-noise boundary (same class as the transport retries in
            # ``_post``): a truncated/malformed arguments blob — typically a
            # long write_file cut at max_output_tokens (finish_reason=length)
            # — must not kill the whole run. It becomes a structurally
            # malformed tool call whose backend failure lands in the
            # transcript, so the model sees its own truncation and adapts.
            try:
                parameters = json.loads(arguments_raw) if arguments_raw else {}
            except json.JSONDecodeError as error:
                parameters = {
                    "malformed_arguments_prefix": str(arguments_raw)[:2000],
                    "malformed_arguments_error": f"{type(error).__name__}: {error}",
                }
                metadata["arguments_parse_error"] = True
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
    "ConstraintAwareScriptedHand",
    "ForcedActionAssignment",
    "ForcedActionHand",
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
