from __future__ import annotations

import argparse
import asyncio
import os
import time
from functools import partial
from typing import Callable, TextIO

from volvence_zero.agent.dialogue_benchmark import (
    OpenDialogueREPLReader,
    build_deterministic_user_simulator,
)
from volvence_zero.agent.response import LLMResponseSynthesizer
from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.substrate import SubstrateFallbackMode

DEFAULT_CHAT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


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


def _prefix_chat_text(*, speaker: str, text: str) -> str:
    lines = text.splitlines() or [""]
    prefixed_lines = [f"{speaker}{lines[0]}"]
    indent = " " * len(speaker)
    prefixed_lines.extend(f"{indent}{line}" for line in lines[1:])
    return "\n".join(prefixed_lines)


def render_chat_turn_result(result: AgentTurnResult, *, include_meta: bool = False) -> str:
    lines = [_prefix_chat_text(speaker="ai> ", text=result.response.text)]
    if not include_meta:
        return "\n".join(lines)

    lines.extend(
        [
            f"  regime: {result.active_regime or 'none'}",
            f"  temporal: {result.active_abstract_action or 'none'}",
            f"  joint_schedule: {result.joint_schedule_action}",
            f"  acceptance: {'pass' if result.acceptance_passed else 'fail'}",
        ]
    )
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
    return "\n".join(lines)


async def run_repl(
    *,
    runner: AgentSessionRunner,
    reader: Callable[[], str],
    writer: Callable[[str], None],
    prompt: str = "you> ",
    banner: str = "VolvenceZero REPL. Type 'exit' or 'quit' to stop.",
    render_result: Callable[[AgentTurnResult], str] = render_turn_result,
    render_input: Callable[[str], str] | None = None,
    after_result: Callable[[AgentTurnResult], None] | None = None,
    show_timing: bool = True,
) -> None:
    writer(banner)
    while True:
        if prompt:
            writer(prompt)
        try:
            user_input = reader().strip()
        except EOFError:
            writer("bye")
            return
        except KeyboardInterrupt:
            writer("bye")
            return
        if not user_input:
            continue
        if render_input is not None:
            writer(render_input(user_input))
        if user_input.lower() in {"exit", "quit"}:
            writer("bye")
            return
        t0 = time.perf_counter()
        try:
            result = await runner.run_turn(user_input)
        except KeyboardInterrupt:
            writer("bye")
            return
        elapsed = time.perf_counter() - t0
        if after_result is not None:
            after_result(result)
        output = render_result(result)
        if show_timing:
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


def should_use_llm(*, chat_mode: bool, llm_flag: bool) -> bool:
    return chat_mode or llm_flag


def should_prefer_local_files(
    *,
    local_files_only_flag: bool,
    allow_remote_model_fetch: bool,
) -> bool:
    if local_files_only_flag:
        return True
    return not allow_remote_model_fetch


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
        default=DEFAULT_CHAT_MODEL_ID,
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
        "--allow-remote-model-fetch",
        action="store_true",
        help="Allow the CLI to fetch model files from the Hugging Face Hub when they are not already cached locally.",
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
        "--chat",
        action="store_true",
        help="Run a simple chat view that prints only the assistant reply by default.",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="When used with --chat, also print compact runtime metadata after each reply.",
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
    parser.add_argument(
        "--open-sim",
        action="store_true",
        help="Run an open-environment simulated user episode instead of waiting for stdin.",
    )
    parser.add_argument(
        "--open-scenario-id",
        default="open_repair",
        help="Open dialogue scenario id used when --open-sim is enabled.",
    )
    parser.add_argument(
        "--open-max-turns",
        type=int,
        default=None,
        help="Optional max turn override for the open simulated episode.",
    )
    parser.add_argument(
        "--open-seed",
        type=int,
        default=0,
        help="Deterministic seed used by the open simulated user episode.",
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
    use_llm = should_use_llm(chat_mode=args.chat, llm_flag=args.llm)
    prefer_local_files = should_prefer_local_files(
        local_files_only_flag=args.substrate_local_files_only,
        allow_remote_model_fetch=args.allow_remote_model_fetch,
    )
    runtime = build_transformers_runtime_with_fallback(
        model_id=args.substrate_model_id,
        device=args.substrate_device,
        local_files_only=prefer_local_files,
        fallback_mode=fallback_mode,
    )
    synthesizer = None
    if use_llm:
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
    banner = "VolvenceZero REPL. Type 'exit' or 'quit' to stop."
    render_result = render_turn_result
    render_input = None
    after_result = None
    prompt = "you> "
    reader = _stdin_reader(__import__("sys").stdin)
    show_timing = True
    if args.chat:
        banner = "VolvenceZero Chat. Type 'exit' or 'quit' to stop."
        render_result = partial(render_chat_turn_result, include_meta=args.show_meta)
        show_timing = False
    if args.open_sim:
        simulator = build_deterministic_user_simulator(
            scenario_id=args.open_scenario_id,
            seed=args.open_seed,
            max_turns=args.open_max_turns,
        )
        open_reader = OpenDialogueREPLReader(turn_source=simulator)
        reader = open_reader
        prompt = ""
        render_input = lambda text: f"sim> {text}"
        after_result = open_reader.observe_result
        banner = (
            f"VolvenceZero Open Dialogue ({simulator.scenario.scenario_id}). "
            f"Simulated user episode with max_turns={simulator.scenario.max_turns}."
        )
    asyncio.run(
        run_repl(
            runner=runner,
            reader=reader,
            writer=_stdout_writer(__import__("sys").stdout),
            prompt=prompt,
            banner=banner,
            render_result=render_result,
            render_input=render_input,
            after_result=after_result,
            show_timing=show_timing,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
