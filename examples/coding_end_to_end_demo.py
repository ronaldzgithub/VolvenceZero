"""End-to-end demo: coding vertical + thinking adapter + affordance scorer.

Runs a scripted multi-turn session that exercises every
affordance tier on a disposable sandbox. Prints a structured
audit trail so the components' interaction is visible at a
glance.

**Import discipline:** this script imports only from public
``lifeform-*`` wheel APIs. No ``volvence_zero.*`` (kernel) imports.
The session ``Snapshot`` objects are opaque here \u2014 every
transformation that needs kernel types (context building, score
computation, readout) is encapsulated inside the wheels.

Run:

    python examples/coding_end_to_end_demo.py

Layout of the printed trail:

    [setup] ...
    [turn 1] ...
        scorer.picks: ...
        interlocutor: ...
    [turn 2] ...
        invoker.result: ...
    ...
    [scene-close] ...
        thinking.artifacts: ...
    [summary] ...
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Public wheel imports only \u2014 NO kernel (volvence_zero.*) imports here.
# If you are tempted to reach for a kernel type, either (a) there's a
# public wheel re-export already, or (b) the wheel is missing a
# capability and we should add it rather than leak kernel internals.
# ---------------------------------------------------------------------------

from lifeform_affordance import (
    AffordanceInvocationStatus,
    build_scored_snapshot,
    build_scoring_context_from_snapshots,
)
from lifeform_domain_coding import (
    CONSENT_FILESYSTEM_READ,
    CONSENT_FILESYSTEM_WRITE,
    CONSENT_RUN_SHELL_COMMANDS,
    build_coding_affordance_invoker,
    build_coding_affordance_registry,
    build_coding_lifeform,
)
from lifeform_thinking import (
    ThinkingWiringLevel,
    build_default_thinking_adapter,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_section(tag: str, text: str) -> None:
    print(f"  [{tag}] {text}")


def _print_bullet(text: str) -> None:
    print(f"    - {text}")


# ---------------------------------------------------------------------------
# Sandbox bootstrap
# ---------------------------------------------------------------------------


_SAMPLE_MAIN = """def greet(name: str) -> str:
    if not name:
        raise ValueError("name must be non-empty")
    return f"hello, {name}"


def main() -> None:
    print(greet("world"))


if __name__ == "__main__":
    main()
"""


_SAMPLE_UTIL = """def doubled(value: int) -> int:
    return value * 2


def bounded(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
"""


def _bootstrap_sandbox(root: pathlib.Path) -> None:
    """Seed the sandbox with a tiny Python project.

    Real product usage would point at an existing repo; we build
    one here so the demo is self-contained and reproducible.
    """
    (root / "src").mkdir()
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "main.py").write_text(_SAMPLE_MAIN, encoding="utf-8")
    (root / "src" / "util.py").write_text(_SAMPLE_UTIL, encoding="utf-8")
    (root / "tests").mkdir()


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


@dataclass
class DemoAudit:
    """Lightweight counters the demo prints in its summary phase."""

    turns_run: int = 0
    affordance_invocations: int = 0
    succeeded_invocations: int = 0
    denied_invocations: int = 0
    thinking_artifacts: int = 0


async def _run_demo(sandbox: pathlib.Path) -> DemoAudit:
    audit = DemoAudit()

    # ---- Phase 1: setup ----

    _print_header("Phase 1: setup")
    _print_section("sandbox", f"root={sandbox}")

    # Build the coding lifeform with a thinking adapter factory wired
    # in. Thinking runs in SHADOW: workers execute + publish artifacts
    # for observability, but no downstream consumer has to read them.
    lifeform = build_coding_lifeform().with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.SHADOW,
        ),
    )
    session = lifeform.create_session(session_id="coding-end-to-end-demo")
    registry = build_coding_affordance_registry()
    invoker = build_coding_affordance_invoker(sandbox_root=sandbox)
    granted = frozenset(
        {
            CONSENT_FILESYSTEM_READ,
            CONSENT_FILESYSTEM_WRITE,
            CONSENT_RUN_SHELL_COMMANDS,
        }
    )
    _print_section(
        "affordances", f"registered={registry.names()}"
    )
    _print_section(
        "thinking",
        f"wiring={session.thinking_adapter_snapshot.wiring_level.value}",
    )

    # ---- Phase 2: turn 1 \u2014 user opens the scene ----

    _print_header("Phase 2: turn 1 \u2014 user opens scene")
    await session.run_turn(
        "Hi, I'm working on a small Python project. Can you help me poke "
        "at the code and add a test?"
    )
    audit.turns_run += 1
    _print_section(
        "regime", f"active={_active_regime_id(session)}"
    )

    # Score the 5 coding affordances from the live session snapshots.
    # The context builder is duck-typed; it reads session.latest_active_snapshots
    # (opaque to this script) and produces an AffordanceScoringContext.
    scoring_ctx = _build_scoring_context(session)
    scored = build_scored_snapshot(registry, scoring_ctx)
    _print_section(
        "scorer.evidence",
        f"{scoring_ctx.evidence:.2f} (regime={scoring_ctx.active_regime_id!r}, "
        f"depth={scoring_ctx.cognitive_depth!r})",
    )
    for candidate in sorted(
        scored.candidates_for_turn, key=lambda c: c.score, reverse=True
    ):
        tag = "BLOCKED" if candidate.is_blocked else f"score={candidate.score:.2f}"
        _print_bullet(f"{candidate.descriptor_name:<12}{tag}")
    if scored.selected is not None:
        _print_section(
            "scorer.pick",
            f"selected={scored.selected.descriptor_name} "
            f"(score={scored.selected.score:.2f})",
        )
    else:
        _print_section(
            "scorer.pick",
            "no single candidate stood out (below selection threshold); "
            "demo will drive invocations explicitly",
        )

    # Emit the interlocutor-state readout for visibility.
    _print_interlocutor_state(session)

    # ---- Phase 3: turn 2 \u2014 read the code ----

    _print_header("Phase 3: turn 2 \u2014 read src/main.py")
    await session.run_turn("Can you show me what's in src/main.py?")
    audit.turns_run += 1

    result = await invoker.invoke(
        "read_file",
        {"path": "src/main.py"},
        session=session.brain_session,
        event_id="demo-read-main",
        granted_consents=granted,
        active_regime_id=_active_regime_id(session),
    )
    audit.affordance_invocations += 1
    _print_invocation(result)
    if result.status is AffordanceInvocationStatus.SUCCEEDED:
        audit.succeeded_invocations += 1
        preview = (result.payload or {}).get("content", "")[:80].replace("\n", "\\n")
        _print_bullet(f"content preview: {preview!r}")

    # ---- Phase 4: turn 3 \u2014 write a test file ----

    _print_header("Phase 4: turn 3 \u2014 write a new test file")
    await session.run_turn("Please add a test for the greet() function.")
    audit.turns_run += 1

    test_body = (
        "from src.main import greet\n"
        "\n"
        "\n"
        "def test_greet_with_name() -> None:\n"
        "    assert greet('world') == 'hello, world'\n"
        "\n"
        "\n"
        "def test_greet_empty_raises() -> None:\n"
        "    import pytest\n"
        "\n"
        "    with pytest.raises(ValueError):\n"
        "        greet('')\n"
    )
    # write_file requires user_confirmation=True (safety gate). The
    # demo passes True because in a real product that flag flips after
    # the user explicitly clicks "approve" in the UI.
    result = await invoker.invoke(
        "write_file",
        {
            "path": "tests/test_greet.py",
            "content": test_body,
            "mode": "create",
        },
        session=session.brain_session,
        event_id="demo-write-test",
        granted_consents=granted,
        active_regime_id=_active_regime_id(session),
        user_confirmed=True,
    )
    audit.affordance_invocations += 1
    _print_invocation(result)
    if result.status is AffordanceInvocationStatus.SUCCEEDED:
        audit.succeeded_invocations += 1
    elif result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY:
        audit.denied_invocations += 1

    # Also demo a DENIED invocation: same write_file call without
    # user_confirmed=True \u2014 the invoker's confirmation gate should
    # fire before any file touches happen.
    denied = await invoker.invoke(
        "write_file",
        {
            "path": "tests/sneaky.py",
            "content": "# should not be written\n",
            "mode": "create",
        },
        session=session.brain_session,
        event_id="demo-write-denied",
        granted_consents=granted,
        active_regime_id=_active_regime_id(session),
        user_confirmed=False,
    )
    audit.affordance_invocations += 1
    _print_invocation(denied)
    if denied.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY:
        audit.denied_invocations += 1
    sneaky_path = sandbox / "tests" / "sneaky.py"
    _print_bullet(
        f"sneaky file actually on disk: {sneaky_path.is_file()} (should be False)"
    )

    # ---- Phase 5: turn 4 \u2014 run the test ----

    _print_header("Phase 5: turn 4 \u2014 run pytest on the new test")
    await session.run_turn("Now run the test and tell me how it went.")
    audit.turns_run += 1

    result = await invoker.invoke(
        "run_test",
        {"test_path": "tests/test_greet.py"},
        session=session.brain_session,
        event_id="demo-run-test",
        granted_consents=granted,
        active_regime_id=_active_regime_id(session),
    )
    audit.affordance_invocations += 1
    _print_invocation(result)
    if result.status is AffordanceInvocationStatus.SUCCEEDED:
        audit.succeeded_invocations += 1
        payload = result.payload or {}
        _print_bullet(
            f"pytest exit_code={payload.get('exit_code')} "
            f"duration={payload.get('duration_seconds')}s "
            f"timed_out={payload.get('timed_out')}"
        )
        # Preview the first couple lines of stdout for human readability.
        stdout_preview = (payload.get("stdout") or "").splitlines()[:3]
        for line in stdout_preview:
            _print_bullet(f"stdout: {line[:100]!r}")

    # ---- Phase 6: scene close + thinking drain ----

    _print_header("Phase 6: close scene (drain thinking adapter)")
    await session.run_turn(
        "That was a useful pair-programming round. I'm done for now."
    )
    audit.turns_run += 1
    closed_scene = await session.end_scene(reason="demo-complete")
    _print_section(
        "scene",
        f"closed scene_id={closed_scene.scene_id if closed_scene else '<none>'}",
    )
    thinking_snapshot = session.thinking_adapter_snapshot
    if thinking_snapshot is not None:
        sched = thinking_snapshot.scheduler_snapshot
        _print_section(
            "thinking.totals",
            f"submitted={sched.total_submitted} completed={sched.total_completed} "
            f"stale={sched.total_stale} failed={sched.total_failed}",
        )
        for consumer, artifact in (
            session.latest_thinking_artifacts_by_consumer.items()
        ):
            audit.thinking_artifacts += 1
            payload = artifact.payload
            desc = getattr(payload, "rationale", str(payload)[:60])
            _print_bullet(f"{consumer}: {desc}")

    # ---- Phase 7: summary ----

    _print_header("Phase 7: summary")
    _print_section(
        "turns", f"{audit.turns_run} run (all via LifeformSession.run_turn)"
    )
    _print_section(
        "affordances",
        f"{audit.affordance_invocations} invocations / "
        f"{audit.succeeded_invocations} succeeded / "
        f"{audit.denied_invocations} denied by boundary",
    )
    _print_section(
        "thinking", f"{audit.thinking_artifacts} mid-reflection artifacts collected"
    )
    case_reconcile = session.latest_case_memory_reconcile
    if case_reconcile is not None:
        _print_section(
            "case_memory.reconcile",
            f"promoted={len(case_reconcile.promoted)} "
            f"retired={len(case_reconcile.retired)} "
            f"expired={len(case_reconcile.expired)}",
        )
    _print_interlocutor_state(session, tag="interlocutor.final")

    return audit


# ---------------------------------------------------------------------------
# Small helpers (still lifeform-layer only)
# ---------------------------------------------------------------------------


def _active_regime_id(session: Any) -> str:
    """Extract the active regime_id from the session without leaking
    kernel types into user code.

    We only touch ``session.latest_active_snapshots`` and the
    exposed ``.value.active_regime.regime_id`` chain. The demo
    treats these values as opaque strings.
    """
    snap = session.latest_active_snapshots.get("regime")
    if snap is None:
        return ""
    active = getattr(snap.value, "active_regime", None)
    if active is None:
        return ""
    return str(getattr(active, "regime_id", ""))


def _build_scoring_context(session: Any) -> Any:
    """Thin helper around the wheel's context builder.

    Kept here (not inlined) so the demo reads top-down without
    stuffing the ``build_scoring_context_from_snapshots`` call into
    every turn.
    """
    snaps = session.latest_active_snapshots
    return build_scoring_context_from_snapshots(
        regime_snapshot=snaps.get("regime"),
        dual_track_snapshot=snaps.get("dual_track"),
    )


def _print_invocation(result: Any) -> None:
    if result.status is AffordanceInvocationStatus.SUCCEEDED:
        _print_section(
            "invoker",
            f"{result.descriptor_name} -> SUCCEEDED (kernel event_ids="
            f"{len(result.tool_event_ids)})",
        )
        return
    _print_section(
        "invoker",
        f"{result.descriptor_name} -> {result.status.value} "
        f"({result.error_class}: {result.error_detail[:100]})",
    )


def _print_interlocutor_state(session: Any, *, tag: str = "interlocutor") -> None:
    state = session.interlocutor_state
    axes = (
        ("engagement", state.engagement_intensity),
        ("task_focus", state.task_focus_level),
        ("rapport", state.rapport_warmth),
        ("resistance", state.resistance_level),
        ("openness", state.openness_to_guidance),
        ("directness", state.directness),
    )
    brief = " ".join(f"{name}={value:.2f}" for name, value in axes)
    _print_section(
        tag,
        f"{brief} trust={state.trust_signal:+.2f} "
        f"conf={state.readout_confidence:.2f}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> DemoAudit:
    """Run the demo end-to-end and return the audit struct.

    Uses a ``TemporaryDirectory`` so the sandbox is wiped on
    exit. Product code would point at a real workspace directory.
    """
    with tempfile.TemporaryDirectory(prefix="vz-coding-demo-") as tmp:
        sandbox = pathlib.Path(tmp)
        _bootstrap_sandbox(sandbox)
        return await _run_demo(sandbox)


def _cli() -> int:
    audit = asyncio.run(main())
    # Exit non-zero only when nothing happened \u2014 a defensive
    # check so a broken environment (missing wheels, etc.) is
    # caught by CI instead of silently passing.
    if audit.turns_run == 0:
        print("FAIL: demo ran no turns", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
