from __future__ import annotations

import argparse
import asyncio
import os
from typing import Callable, TextIO

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.memory import MemorySnapshot
from volvence_zero.reflection import ReflectionSnapshot
from volvence_zero.substrate import SubstrateFallbackMode


def render_trial_turn_result(result: AgentTurnResult) -> str:
    memory_retrievals = 0
    memory_snapshot = result.active_snapshots.get("memory")
    if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot):
        memory_retrievals = len(memory_snapshot.value.retrieved_entries)

    reflection_lessons = 0
    reflection_tensions = 0
    primary_lesson = None
    primary_tension = None
    reflection_snapshot = result.active_snapshots.get("reflection") or result.shadow_snapshots.get("reflection")
    if reflection_snapshot is not None and isinstance(reflection_snapshot.value, ReflectionSnapshot):
        reflection_lessons = len(reflection_snapshot.value.lessons_extracted)
        reflection_tensions = len(reflection_snapshot.value.tensions_identified)
        primary_lesson = next(iter(reflection_snapshot.value.lessons_extracted), None)
        primary_tension = next(iter(reflection_snapshot.value.tensions_identified), None)

    switch_gate = 0.0
    if result.metacontroller_state is not None:
        switch_gate = result.metacontroller_state.latest_switch_gate

    lines = [
        f"[{result.wave_id}] {result.response.text}",
        f"  regime: {result.active_regime or 'none'}",
        f"  temporal: {result.active_abstract_action or 'none'}",
        f"  switch_gate: {switch_gate:.2f}",
        f"  memory_retrievals: {memory_retrievals}",
        f"  reflection_lessons: {reflection_lessons}",
        f"  reflection_tensions: {reflection_tensions}",
        f"  joint_schedule: {result.joint_schedule_action}",
        f"  writeback_source: {result.writeback_source or 'none'}",
        f"  writeback_applied: {'yes' if result.bounded_writeback_applied else 'no'}",
        f"  acceptance: {'pass' if result.acceptance_passed else 'fail'}",
    ]
    if primary_lesson is not None:
        lines.append(f"  primary_lesson: {primary_lesson}")
    if primary_tension is not None:
        lines.append(f"  primary_tension: {primary_tension}")
    if result.writeback_operations:
        lines.append(f"  writeback_ops: {', '.join(result.writeback_operations)}")
    if result.writeback_blocks:
        lines.append(f"  writeback_blocks: {', '.join(result.writeback_blocks)}")
    if result.evaluation_alerts:
        lines.append(f"  alerts: {', '.join(result.evaluation_alerts)}")
    if result.acceptance_issues:
        lines.append(f"  issues: {', '.join(result.acceptance_issues)}")
    lines.append(f"  rationale: {result.response.rationale}")
    lines.append(f"  learning: {result.joint_learning_summary}")
    return "\n".join(lines)


async def run_trial_repl(
    *,
    runner: AgentSessionRunner,
    reader: Callable[[], str],
    writer: Callable[[str], None],
    prompt: str = "kernel> ",
) -> None:
    writer("VolvenceZero Kernel Trial REPL. Type 'exit' or 'quit' to stop.")
    while True:
        writer(prompt)
        user_input = reader().strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            writer("bye")
            return
        result = await runner.run_turn(user_input)
        writer(render_trial_turn_result(result))


def _stdin_reader(stdin: TextIO) -> Callable[[], str]:
    return lambda: stdin.readline()


def _stdout_writer(stdout: TextIO) -> Callable[[str], None]:
    def _write(text: str) -> None:
        stdout.write(text)
        if not text.endswith("\n"):
            stdout.write("\n")
        stdout.flush()

    return _write


def build_trial_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VolvenceZero kernel trial REPL.")
    default_fallback_mode = os.getenv("VOLVENCE_SUBSTRATE_FALLBACK_MODE", SubstrateFallbackMode.ALLOW_BUILTIN.value)
    parser.add_argument(
        "--session-id",
        default="trial-session",
        help="Session identifier used by the runtime graph.",
    )
    parser.add_argument(
        "--substrate-model-id",
        default="distilgpt2",
        help="Preferred Hugging Face causal LM model id for the default real substrate runtime.",
    )
    parser.add_argument(
        "--substrate-device",
        default="auto",
        help="Execution device for the substrate runtime, e.g. auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--substrate-local-files-only",
        action="store_true",
        help="Load the preferred Hugging Face model only from local files.",
    )
    parser.add_argument(
        "--disable-substrate-fallback",
        action="store_true",
        help="Disable fallback to the bundled tiny transformers runtime when the preferred model is unavailable.",
    )
    parser.add_argument(
        "--substrate-fallback-mode",
        choices=tuple(mode.value for mode in SubstrateFallbackMode),
        default=default_fallback_mode,
        help=(
            "Fallback policy for substrate loading. "
            "Use 'allow-builtin' for local dev and 'deny' for evaluation or production-like runs."
        ),
    )
    return parser


def trial_main() -> int:
    parser = build_trial_arg_parser()
    args = parser.parse_args()
    runner = AgentSessionRunner(
        session_id=args.session_id,
        substrate_model_id=args.substrate_model_id,
        substrate_device=args.substrate_device,
        substrate_local_files_only=args.substrate_local_files_only,
        substrate_fallback_mode=(
            SubstrateFallbackMode.DENY.value
            if args.disable_substrate_fallback
            else args.substrate_fallback_mode
        ),
    )
    asyncio.run(
        run_trial_repl(
            runner=runner,
            reader=_stdin_reader(__import__("sys").stdin),
            writer=_stdout_writer(__import__("sys").stdout),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(trial_main())
