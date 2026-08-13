"""Single-episode runner (coding-lab Packet 0).

Drives one hand through the sandboxed coding-affordance backends inside
an episode worktree, logs the full trajectory, then settles the final
workspace state with the oracle. Budget exhaustion without a submit is
still graded — an unfinished change is an outcome, not an error.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any

from lifeform_domain_coding.coding_affordances.backends import (
    SandboxPathError,
    build_coding_affordance_backends,
)
from lifeform_domain_coding.lab.generation import EnvSpec
from lifeform_domain_coding.lab.hands import Hand, HandContext, TranscriptEntry
from lifeform_domain_coding.lab.oracle import OracleOutcome, evaluate_episode
from lifeform_domain_coding.lab.tasks import ChainTask
from lifeform_domain_coding.lab.trajectory import TrajectoryRecord, TrajectoryWriter


@dataclass(frozen=True)
class EpisodeBudget:
    max_steps: int = 24
    max_wall_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.max_steps < 2:
            raise ValueError("max_steps must be >= 2 (one action plus submit)")
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")


@dataclass(frozen=True)
class EpisodeResult:
    """Everything one episode produced (pre-merge)."""

    episode_index: int
    task_id: str
    hand_id: str
    submitted: bool
    steps_used: int
    wall_seconds: float
    outcome: OracleOutcome
    trajectory: TrajectoryRecord
    prompt_tokens: int
    completion_tokens: int


def _aggregate_usage(usages: list[dict[str, Any]]) -> tuple[int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for usage in usages:
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
    return prompt_tokens, completion_tokens


async def run_episode(
    *,
    spec: EnvSpec,
    task: ChainTask,
    episode_index: int,
    worktree: pathlib.Path,
    hand: Hand,
    trajectory_path: pathlib.Path,
    budget: EpisodeBudget | None = None,
    oracle_python: str | None = None,
    context_preamble: str = "",
) -> EpisodeResult:
    """Run ``hand`` on ``task`` inside ``worktree`` and settle the outcome."""

    effective_budget = budget or EpisodeBudget()
    backends = build_coding_affordance_backends(worktree)
    writer = TrajectoryWriter(trajectory_path)
    writer.append(
        "task_presented",
        {
            "task_id": task.task_id,
            "category": task.category,
            "description": task.description,
            "episode_index": episode_index,
            "hand_id": hand.hand_id(),
        },
    )
    transcript: list[TranscriptEntry] = []
    usages: list[dict[str, Any]] = []
    submitted = False
    started = time.monotonic()
    step_index = 0
    while step_index < effective_budget.max_steps:
        if time.monotonic() - started > effective_budget.max_wall_seconds:
            writer.append("budget_exhausted", {"reason": "wall_clock", "step_index": step_index})
            break
        context = HandContext(
            task_id=task.task_id,
            task_description=task.description,
            package_name=spec.package_name,
            step_index=step_index,
            max_steps=effective_budget.max_steps,
            transcript=tuple(transcript),
            context_preamble=context_preamble,
        )
        decision = await hand.decide(context)
        if decision.metadata.get("usage"):
            usages.append(dict(decision.metadata["usage"]))
        writer.append(
            "hand_decision",
            {
                "step_index": step_index,
                "kind": decision.action.kind,
                "tool_name": decision.action.tool_name,
                "parameters": decision.action.parameters,
                "note": decision.action.note,
                "metadata": decision.metadata,
            },
        )
        if decision.action.kind == "submit":
            submitted = True
            step_index += 1
            break
        tool_name = decision.action.tool_name
        backend = backends.get(tool_name)
        if backend is None:
            result: dict[str, Any] = {
                "error_class": "UnknownTool",
                "error_detail": f"tool {tool_name!r} is not available",
            }
            succeeded = False
        else:
            try:
                raw = await backend(decision.action.parameters)
                result = dict(raw)
                succeeded = True
            except (SandboxPathError, UnicodeDecodeError, KeyError, ValueError, OSError) as exc:
                result = {"error_class": type(exc).__name__, "error_detail": str(exc)[:800]}
                succeeded = False
        writer.append(
            "tool_result",
            {
                "step_index": step_index,
                "tool_name": tool_name,
                "succeeded": succeeded,
                "result_keys": sorted(result.keys()),
                "result": _bounded_result_for_log(result),
            },
        )
        transcript.append(
            TranscriptEntry(
                tool_name=tool_name,
                parameters=dict(decision.action.parameters),
                result=result,
                succeeded=succeeded,
            )
        )
        step_index += 1

    outcome = evaluate_episode(
        spec=spec,
        task=task,
        workspace_root=worktree,
        python_executable=oracle_python,
    )
    wall_seconds = time.monotonic() - started
    writer.append(
        "oracle_outcome",
        {
            "task_id": outcome.task_id,
            "passed": outcome.passed,
            "acceptance_passed": outcome.acceptance_passed,
            "regression_passed": outcome.regression_passed,
            "failed_test_ids": list(outcome.failed_test_ids),
            "error_test_ids": list(outcome.error_test_ids),
            "invariant_violations": list(outcome.invariant_violations),
            "submitted": submitted,
        },
    )
    trajectory = writer.close()
    prompt_tokens, completion_tokens = _aggregate_usage(usages)
    return EpisodeResult(
        episode_index=episode_index,
        task_id=task.task_id,
        hand_id=hand.hand_id(),
        submitted=submitted,
        steps_used=step_index,
        wall_seconds=wall_seconds,
        outcome=outcome,
        trajectory=trajectory,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _bounded_result_for_log(result: dict[str, Any]) -> dict[str, Any]:
    """Trim large payload fields so trajectories stay disk-friendly."""

    bounded: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, str) and len(value) > 4000:
            bounded[key] = value[:4000] + f"... [{len(value) - 4000} chars trimmed]"
        else:
            bounded[key] = value
    return bounded


__all__ = ["EpisodeBudget", "EpisodeResult", "run_episode"]
