"""End-to-end smoke test for the CLI driver.

We boot a real ``lifeform-service`` aiohttp app on a free port, point
the CLI driver's ``run_arc`` at it, and assert that:

* the driver completes an arc (turns_run >= 2);
* the JSONL trace file exists and contains the expected event kinds
  (``arc_begin`` / ``session_create`` / ``user_turn`` / ``bot_turn`` /
  ``arc_end``);
* per-turn governance fields land in the trace (``active_regime``,
  ``pe_magnitude``, ``commitment_count``, ``open_loop_count``).

The user-side utterance backend is forced to the deterministic
``DeterministicFakeUtteranceClient`` so the test never reaches an
external LLM. The companion vertical is used because every install
ships it and it supports alpha mode (which is what the CLI binds to).

The test is gated on a free TCP port being available locally; if the
environment forbids ad-hoc listeners (some CI sandboxes) the runner
skips.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import socket
import threading
from typing import Any

import pytest

from companion_bench.spec import load_scenario_yaml
from companion_bench.user_simulator import DeterministicFakeUtteranceClient

# Mirrors the resource-style scenario lookup in
# scripts/governance_demo/drive_session_arc.py so the smoke arc covers
# the exact same loader path.
import importlib.resources as res


def _free_tcp_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _scenarios_dir() -> pathlib.Path:
    return pathlib.Path(
        str(res.files("companion_bench") / "scenarios" / "public")
    )


@pytest.fixture
def service_thread(tmp_path):
    """Start the lifeform-service in a background asyncio loop.

    We can't use ``aiohttp_client`` here because the CLI driver opens
    its own urllib HTTP connections — it needs a real TCP listener.
    """

    from aiohttp import web
    from lifeform_service.alpha import AlphaServiceConfig
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    verticals = discover_verticals()
    if "companion" not in verticals:
        pytest.skip("companion vertical not installed")

    port = _free_tcp_port()
    base_url = f"http://127.0.0.1:{port}"
    app = create_app(
        verticals=verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        templates_root_dir=str(tmp_path / "templates"),
        utterance_backend=DeterministicFakeUtteranceClient(),
        alpha_config=AlphaServiceConfig(
            enabled=True,
            memory_scope_root_dir=str(tmp_path / "memory"),
            evidence_root_dir=str(tmp_path / "evidence"),
            alpha_users=frozenset({"smoke-user"}),
        ),
    )

    loop = asyncio.new_event_loop()
    runner_holder: list[web.AppRunner] = []
    ready = threading.Event()
    stop_event = asyncio.Event()

    async def _start() -> None:
        runner = web.AppRunner(app)
        runner_holder.append(runner)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        ready.set()
        # Wait until the test signals shutdown; ``stop_event`` is
        # cleared once we've also cleaned up the runner so the loop
        # exits cleanly without leaving a "Future never completed"
        # warning behind.
        await stop_event.wait()
        await runner.cleanup()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start())

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    assert ready.wait(timeout=10.0), "service did not start in time"
    try:
        yield base_url
    finally:
        loop.call_soon_threadsafe(stop_event.set)
        thread.join(timeout=10.0)
        loop.close()


def _pick_short_scenario():
    """Pick the public scenario with the smallest schedule.

    Smaller schedules make the smoke test cheap; the simulator
    backend is deterministic so semantics don't matter.
    """

    best_path = None
    best_size = None
    for yaml_path in sorted(_scenarios_dir().rglob("*.yaml")):
        spec = load_scenario_yaml(yaml_path)
        lo, _ = spec.session_turn_range
        approx = spec.arc_length_sessions * lo
        if best_size is None or approx < best_size:
            best_size = approx
            best_path = yaml_path
    assert best_path is not None
    return best_path


def test_run_arc_smoke(service_thread, tmp_path) -> None:
    base_url = service_thread

    # Load the CLI driver directly from its file path so the test does
    # not depend on ``scripts/`` being a regular Python package (the
    # other ``scripts/*`` subdirs are intentionally not packages).
    # We must register the module in ``sys.modules`` BEFORE
    # ``exec_module`` because the driver uses
    # ``@dataclasses.dataclass(frozen=True)`` on classes with
    # ``from __future__ import annotations`` — dataclass evaluation
    # resolves the (string) field annotations via
    # ``sys.modules[cls.__module__].__dict__`` and crashes when the
    # module is not registered.
    import importlib.util
    import sys as _sys

    driver_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "governance_demo"
        / "drive_session_arc.py"
    )
    spec_load = importlib.util.spec_from_file_location(
        "_governance_demo_driver", driver_path
    )
    assert spec_load is not None and spec_load.loader is not None
    driver_module = importlib.util.module_from_spec(spec_load)
    _sys.modules["_governance_demo_driver"] = driver_module
    try:
        spec_load.loader.exec_module(driver_module)
        ServiceClient = driver_module.ServiceClient
        _AnsiPalette = driver_module._AnsiPalette
        _JsonlTrace = driver_module._JsonlTrace
        run_arc = driver_module.run_arc
    finally:
        _sys.modules.pop("_governance_demo_driver", None)

    scenario_path = _pick_short_scenario()
    spec = load_scenario_yaml(scenario_path)
    trace_path = tmp_path / "smoke.jsonl"
    trace = _JsonlTrace(trace_path)
    try:
        stats = run_arc(
            spec=spec,
            paraphrase_seed=0,
            backend=DeterministicFakeUtteranceClient(),
            service=ServiceClient(
                base_url=base_url, user_id="smoke-user"
            ),
            vertical="companion",
            trace=trace,
            pal=_AnsiPalette(enabled=False),
            max_turns=3,
        )
    finally:
        trace.close()

    assert stats["turns_run"] >= 2
    text = trace_path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    kinds = {row["kind"] for row in rows}
    assert {"arc_begin", "session_create", "user_turn", "bot_turn", "arc_end"} <= kinds
    # Every bot_turn must carry the governance fields the panel
    # consumes — if these regress the chat UI silently loses signal.
    bot_turns = [row for row in rows if row["kind"] == "bot_turn"]
    assert bot_turns
    for row in bot_turns:
        assert "active_regime" in row
        assert "pe_magnitude" in row
        assert "commitment_count" in row
        assert "open_loop_count" in row
    # FSM steps must surface in at least one user_turn for any
    # scenario that scripts an FSM action; the picked scenarios all
    # do so by design.
    user_turns = [row for row in rows if row["kind"] == "user_turn"]
    assert any(row.get("fsm_action") for row in user_turns)
