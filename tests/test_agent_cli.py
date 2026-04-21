from __future__ import annotations

import asyncio
from functools import partial

from volvence_zero.agent import AgentSessionRunner, render_turn_result, run_repl
from volvence_zero.agent.cli import (
    DEFAULT_CHAT_MODEL_ID,
    build_arg_parser,
    render_chat_turn_result,
    should_prefer_local_files,
    should_use_llm,
)
from volvence_zero.agent.trial import build_trial_arg_parser, render_trial_turn_result, run_trial_repl


def test_render_turn_result_contains_core_fields():
    runner = AgentSessionRunner(session_id="render-test")
    result = asyncio.run(runner.run_turn("Help me think this through carefully."))
    rendered = render_turn_result(result)

    assert result.wave_id in rendered
    assert "regime:" in rendered
    assert "temporal:" in rendered
    assert "prediction_error:" in rendered
    assert "rationale:" in rendered
    assert "time:" not in rendered


def test_run_repl_handles_single_turn_and_exit():
    runner = AgentSessionRunner(session_id="repl-test")
    inputs = iter(
        [
            "I need a careful response.\n",
            "exit\n",
        ]
    )
    outputs: list[str] = []

    asyncio.run(
        run_repl(
            runner=runner,
            reader=lambda: next(inputs),
            writer=outputs.append,
        )
    )

    rendered_output = "\n".join(outputs)
    assert "VolvenceZero REPL" in rendered_output
    assert "[wave-1]" in rendered_output
    assert "bye" in rendered_output


def test_run_repl_skips_empty_lines():
    runner = AgentSessionRunner(session_id="empty-line-test")
    inputs = iter(
        [
            "\n",
            "Continue with a steady answer.\n",
            "quit\n",
        ]
    )
    outputs: list[str] = []

    asyncio.run(
        run_repl(
            runner=runner,
            reader=lambda: next(inputs),
            writer=outputs.append,
        )
    )

    rendered_output = "\n".join(outputs)
    assert "[wave-1]" in rendered_output


def test_render_chat_turn_result_hides_debug_fields_by_default():
    runner = AgentSessionRunner(session_id="chat-render-test")
    result = asyncio.run(runner.run_turn("Talk to me naturally."))

    rendered = render_chat_turn_result(result)

    assert rendered.startswith("ai> ")
    assert "regime:" not in rendered
    assert "prediction_error:" not in rendered
    assert "rationale:" not in rendered


def test_render_chat_turn_result_can_include_compact_meta():
    runner = AgentSessionRunner(session_id="chat-meta-render-test")
    result = asyncio.run(runner.run_turn("Give me the short answer, but keep context."))

    rendered = render_chat_turn_result(result, include_meta=True)

    assert rendered.startswith("ai> ")
    assert "regime:" in rendered
    assert "joint_schedule:" in rendered
    assert "acceptance:" in rendered
    assert "prediction_error:" in rendered


def test_run_repl_chat_mode_handles_single_turn_and_exit():
    runner = AgentSessionRunner(session_id="chat-repl-test")
    inputs = iter(
        [
            "Can we just talk directly?\n",
            "exit\n",
        ]
    )
    outputs: list[str] = []

    asyncio.run(
        run_repl(
            runner=runner,
            reader=lambda: next(inputs),
            writer=outputs.append,
            banner="VolvenceZero Chat. Type 'exit' or 'quit' to stop.",
            render_result=partial(render_chat_turn_result, include_meta=False),
            show_timing=False,
        )
    )

    rendered_output = "\n".join(outputs)
    assert "VolvenceZero Chat" in rendered_output
    assert "ai> " in rendered_output
    assert "time:" not in rendered_output
    assert "bye" in rendered_output


def test_run_repl_exits_cleanly_on_keyboard_interrupt():
    runner = AgentSessionRunner(session_id="interrupt-repl-test")
    outputs: list[str] = []

    asyncio.run(
        run_repl(
            runner=runner,
            reader=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            writer=outputs.append,
        )
    )

    rendered_output = "\n".join(outputs)
    assert "VolvenceZero REPL" in rendered_output
    assert "bye" in rendered_output


def test_cli_parser_accepts_real_substrate_arguments():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--session-id",
            "cli-test",
            "--substrate-model-id",
            "distilgpt2",
            "--substrate-device",
            "cpu",
            "--substrate-local-files-only",
        ]
    )

    assert args.session_id == "cli-test"
    assert args.substrate_model_id == "distilgpt2"
    assert args.substrate_device == "cpu"
    assert args.substrate_local_files_only is True
    assert args.disable_substrate_fallback is False
    assert args.substrate_fallback_mode == "allow-builtin"


def test_cli_parser_defaults_to_qwen_chat_model():
    parser = build_arg_parser()

    args = parser.parse_args([])

    assert args.substrate_model_id == DEFAULT_CHAT_MODEL_ID
    assert args.substrate_local_files_only is False
    assert args.allow_remote_model_fetch is False


def test_cli_parser_accepts_chat_arguments():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--chat",
            "--show-meta",
        ]
    )

    assert args.chat is True
    assert args.show_meta is True


def test_should_use_llm_defaults_chat_mode_to_generation():
    assert should_use_llm(chat_mode=True, llm_flag=False) is True
    assert should_use_llm(chat_mode=False, llm_flag=True) is True
    assert should_use_llm(chat_mode=False, llm_flag=False) is False


def test_should_prefer_local_files_defaults_to_local_first():
    assert should_prefer_local_files(local_files_only_flag=False, allow_remote_model_fetch=False) is True
    assert should_prefer_local_files(local_files_only_flag=True, allow_remote_model_fetch=False) is True
    assert should_prefer_local_files(local_files_only_flag=False, allow_remote_model_fetch=True) is False


def test_cli_parser_accepts_llm_generation_arguments():
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--llm",
            "--max-new-tokens",
            "128",
            "--temperature",
            "0.5",
        ]
    )

    assert args.llm is True
    assert args.max_new_tokens == 128
    assert abs(args.temperature - 0.5) < 1e-6


def test_cli_parser_accepts_remote_fetch_override():
    parser = build_arg_parser()

    args = parser.parse_args(["--allow-remote-model-fetch"])

    assert args.allow_remote_model_fetch is True


def test_render_trial_turn_result_contains_kernel_fields():
    runner = AgentSessionRunner(session_id="trial-render-test")
    result = asyncio.run(runner.run_turn("Please help me stabilize this planning session."))
    rendered = render_trial_turn_result(result)

    assert result.wave_id in rendered
    assert "switch_gate:" in rendered
    assert "joint_schedule:" in rendered
    assert "writeback_source:" in rendered
    assert "learning:" in rendered
    assert "primary_lesson:" in rendered


def test_run_trial_repl_handles_single_turn_and_exit():
    runner = AgentSessionRunner(session_id="trial-repl-test")
    inputs = iter(
        [
            "Keep this steady and continuous.\n",
            "exit\n",
        ]
    )
    outputs: list[str] = []

    asyncio.run(
        run_trial_repl(
            runner=runner,
            reader=lambda: next(inputs),
            writer=outputs.append,
        )
    )

    rendered_output = "\n".join(outputs)
    assert "Kernel Trial REPL" in rendered_output
    assert "[wave-1]" in rendered_output
    assert "switch_gate:" in rendered_output
    assert "bye" in rendered_output


def test_trial_cli_parser_accepts_real_substrate_arguments():
    parser = build_trial_arg_parser()

    args = parser.parse_args(
        [
            "--session-id",
            "trial-test",
            "--substrate-model-id",
            "distilgpt2",
            "--substrate-device",
            "cpu",
            "--substrate-local-files-only",
        ]
    )

    assert args.session_id == "trial-test"
    assert args.substrate_model_id == "distilgpt2"
    assert args.substrate_device == "cpu"
    assert args.substrate_local_files_only is True
    assert args.disable_substrate_fallback is False
    assert args.substrate_fallback_mode == "allow-builtin"
