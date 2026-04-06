from __future__ import annotations

import argparse
import asyncio
from typing import Callable, TextIO

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult, default_active_runner


def render_turn_result(result: AgentTurnResult) -> str:
    lines = [
        f"[{result.wave_id}] {result.response.text}",
        f"  regime: {result.active_regime or 'none'}",
        f"  temporal: {result.active_abstract_action or 'none'}",
        f"  acceptance: {'pass' if result.acceptance_passed else 'fail'}",
    ]
    if result.evaluation_alerts:
        lines.append(f"  alerts: {', '.join(result.evaluation_alerts)}")
    if result.acceptance_issues:
        lines.append(f"  issues: {', '.join(result.acceptance_issues)}")
    lines.append(f"  rationale: {result.response.rationale}")
    return "\n".join(lines)


async def run_repl(
    *,
    runner: AgentSessionRunner,
    reader: Callable[[], str],
    writer: Callable[[str], None],
    prompt: str = "you> ",
) -> None:
    writer("VolvenceZero REPL. Type 'exit' or 'quit' to stop.")
    while True:
        writer(prompt)
        user_input = reader().strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            writer("bye")
            return
        result = await runner.run_turn(user_input)
        writer(render_turn_result(result))


def _stdin_reader(stdin: TextIO) -> Callable[[], str]:
    return lambda: stdin.readline()


def _stdout_writer(stdout: TextIO) -> Callable[[str], None]:
    def _write(text: str) -> None:
        stdout.write(text)
        if not text.endswith("\n"):
            stdout.write("\n")
        stdout.flush()

    return _write


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal VolvenceZero agent REPL.")
    parser.add_argument(
        "--session-id",
        default="cli-session",
        help="Session identifier used by the runtime graph.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    runner = default_active_runner()
    if args.session_id:
        runner = AgentSessionRunner(session_id=args.session_id)
    asyncio.run(
        run_repl(
            runner=runner,
            reader=_stdin_reader(__import__("sys").stdin),
            writer=_stdout_writer(__import__("sys").stdout),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
