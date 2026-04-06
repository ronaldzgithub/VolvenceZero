from __future__ import annotations

import asyncio

from volvence_zero.agent import AgentSessionRunner, render_turn_result, run_repl


def test_render_turn_result_contains_core_fields():
    runner = AgentSessionRunner(session_id="render-test")
    result = asyncio.run(runner.run_turn("Help me think this through carefully."))
    rendered = render_turn_result(result)

    assert result.wave_id in rendered
    assert "regime:" in rendered
    assert "temporal:" in rendered
    assert "rationale:" in rendered


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
