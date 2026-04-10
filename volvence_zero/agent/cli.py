from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Callable, TextIO

from volvence_zero.agent.response import LLMResponseSynthesizer
from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.substrate import SubstrateFallbackMode


def render_turn_result(result: AgentTurnResult) -> str:
    lines = [
        f"[{result.wave_id}] {result.response.text}",
        f"  regime: {result.active_regime or 'none'}",
        f"  temporal: {result.active_abstract_action or 'none'}",
        f"  acceptance: {'pass' if result.acceptance_passed else 'fail'}",
    ]
    if result.prediction_error is not None:
        lines.append(
            "  prediction_error: "
            f"reward={result.prediction_error.signed_reward:.2f} "
            f"magnitude={result.prediction_error.magnitude:.2f}"
        )
    if result.evaluation_alerts:
        lines.append(f"  alerts: {', '.join(result.evaluation_alerts)}")
    if result.acceptance_issues:
        lines.append(f"  issues: {', '.join(result.acceptance_issues)}")
    if result.reflection_promotion_eligible:
        lines.append(f"  reflection_promotion: eligible ({result.reflection_promotion_reason})")
    if result.imagination_result is not None:
        lines.append(
            f"  imagination: {result.imagination_result.selected_trajectory.candidate_id} "
            f"reward={result.imagination_result.selected_trajectory.cumulative_reward:.2f} "
            f"({len(result.imagination_result.trajectories)} candidates)"
        )
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
        t0 = time.perf_counter()
        result = await runner.run_turn(user_input)
        elapsed = time.perf_counter() - t0
        output = render_turn_result(result)
        output += f"\n  time: {elapsed:.2f}s"
        writer(output)


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
    default_fallback_mode = os.getenv("VOLVENCE_SUBSTRATE_FALLBACK_MODE", SubstrateFallbackMode.ALLOW_BUILTIN.value)
    parser.add_argument(
        "--session-id",
        default="cli-session",
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
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-backed response generation instead of template synthesis.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate when --llm is enabled.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM generation.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    from volvence_zero.substrate import build_transformers_runtime_with_fallback

    fallback_mode = (
        SubstrateFallbackMode.DENY.value
        if args.disable_substrate_fallback
        else args.substrate_fallback_mode
    )
    runtime = build_transformers_runtime_with_fallback(
        model_id=args.substrate_model_id,
        device=args.substrate_device,
        local_files_only=args.substrate_local_files_only,
        fallback_mode=fallback_mode,
    )
    synthesizer = None
    if args.llm:
        synthesizer = LLMResponseSynthesizer(
            runtime=runtime,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    runner = AgentSessionRunner(
        session_id=args.session_id,
        default_residual_runtime=runtime,
        response_synthesizer=synthesizer,
        substrate_fallback_mode=fallback_mode,
    )
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
