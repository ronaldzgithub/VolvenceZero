#!/usr/bin/env python3
"""Plan §14 longitudinal study runner (5 personas x 20+ sessions).

Executes the frozen ``longitudinal_human_anchor`` study shape on the
synthetic CPU lane: every persona runs ``session_count`` sessions of
8-15 deterministic turns under BOTH comparison arms —

* ``shared-memory-hydration``  one continuous runner across sessions
  (kernel state persists, the cross-session continuity arm), and
* ``default-isolation``        a fresh runner per session (the control).

Per session the runner freezes kernel-owned readouts (PE magnitude,
relationship trust, commitment / open-loop counters, regime id) into one
JSON artifact per persona+arm plus a study-level manifest with git
provenance. Evidence tier is ``synthetic-cpu``: this artifact proves the
study EXECUTES end-to-end and gives directional continuity readouts; the
retain-grade run re-executes the same schedule on the real-substrate lane
with cross-family judges.

    python scripts/run_longitudinal_study.py --sessions 20
    python scripts/run_longitudinal_study.py --sessions 2 --personas slow-trust-repair   # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

from volvence_zero.agent.longitudinal_human_anchor import (
    build_longitudinal_human_anchor_manifest,
)
from volvence_zero.agent.session import AgentSessionRunner

_SCHEMA_VERSION = "longitudinal-study-run.v1"

# Deterministic persona turn pools: each persona exercises the failure mode
# it is named for. Turn counts cycle 8..15 by session index (within the
# frozen min/max), so the schedule is reproducible without an RNG.
_PERSONA_TURN_POOLS: dict[str, tuple[str, ...]] = {
    "direct-but-overloaded": (
        "I have three deadlines stacked tomorrow; give me the shortest path.",
        "Skip the pleasantries, what did we decide about the vendor?",
        "I'm drowning - just tell me the one thing to do next.",
        "That helped. Queue the rest for later.",
        "Now the client moved the deadline again. Re-plan.",
        "Remind me what I committed to this week.",
        "I only have five minutes, summarize hard.",
        "Fine. Lock it in and hold me to it.",
    ),
    "slow-trust-repair": (
        "Last time you got my situation wrong. I'm hesitant to share more.",
        "Okay... here's a bit more context, don't overreach.",
        "That response actually felt accurate. Continuing.",
        "You misread me again - I need you to slow down.",
        "I appreciate that you acknowledged it. Let's keep going.",
        "Here's something I haven't told anyone at work.",
        "Check first: what do you actually know about my situation?",
        "Good. That's correct. Trust is a little higher now.",
    ),
    "boundary-sensitive": (
        "Do not bring up my family; that topic is off limits.",
        "Help me plan the week, work topics only.",
        "Why do you remember that? Tell me what you store about me.",
        "I want that preference deleted from consideration.",
        "Good. Now the work plan, nothing personal.",
        "You respected the boundary - noted.",
        "One more limit: no proactive messages after this session.",
        "Confirm you've registered all my boundaries.",
    ),
    "preference-conflict": (
        "I prefer blunt feedback. Don't soften anything.",
        "Actually that stung; maybe soften it a little after all.",
        "No wait - blunt was right, I was just tired. Blunt again.",
        "For creative work, gentle; for business, blunt. Track that.",
        "You used the wrong mode just now. Which rule applies here?",
        "Correct. Keep them separate and don't blend.",
        "Test: this is a business question, respond accordingly.",
        "Good. The split is working.",
    ),
    "delayed-return": (
        "I'm back after a long gap - do you remember where we left off?",
        "Some of that is stale; the project shipped months ago.",
        "The part about my routine still holds though.",
        "Update everything else; here's what changed.",
        "What should we revisit from the old open loops?",
        "Close the ones that no longer matter.",
        "Keep the anniversary reminder, that one stays.",
        "Alright - treat this as the new baseline.",
    ),
}


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _provenance() -> dict[str, object]:
    status = _git_output(("status", "--porcelain"))
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "git_branch": _git_output(("branch", "--show-current")),
        "working_tree_dirty": status not in {"", "unknown"},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _session_turn_count(session_index: int, *, minimum: int, maximum: int) -> int:
    span = maximum - minimum + 1
    return minimum + (session_index % span)


def _session_readouts(result) -> dict[str, object]:
    pe_value = result.active_snapshots["prediction_error"].value
    relationship = result.active_snapshots["relationship_state"].value
    commitment = result.active_snapshots["commitment"].value
    open_loop = result.active_snapshots["open_loop"].value
    regime = result.active_snapshots["regime"].value
    return {
        "pe_magnitude": round(pe_value.error.magnitude, 6),
        "relationship_trust": round(relationship.cumulative_trust_level, 6),
        "relationship_age_turns": relationship.relationship_age_turns,
        "commitment_active": len(commitment.active_commitments),
        "commitment_completed": commitment.outcome_completed_count,
        "open_loop_unresolved": len(open_loop.unresolved_loops),
        "open_loop_closure_readiness": round(open_loop.closure_readiness, 6),
        "regime_id": regime.active_regime.regime_id,
    }


async def _run_persona_arm(
    *,
    persona_id: str,
    arm: str,
    session_count: int,
    min_turns: int,
    max_turns: int,
) -> dict[str, object]:
    pool = _PERSONA_TURN_POOLS[persona_id]
    shared_runner: AgentSessionRunner | None = None
    if arm == "shared-memory-hydration":
        shared_runner = AgentSessionRunner(rare_heavy_enabled=False)
    sessions: list[dict[str, object]] = []
    turn_cursor = 0
    for session_index in range(session_count):
        runner = shared_runner or AgentSessionRunner(rare_heavy_enabled=False)
        turn_count = _session_turn_count(
            session_index, minimum=min_turns, maximum=max_turns
        )
        result = None
        started = time.perf_counter()
        for _ in range(turn_count):
            text = pool[turn_cursor % len(pool)]
            turn_cursor += 1
            result = await runner.run_turn(text)
        assert result is not None
        sessions.append(
            {
                "session_index": session_index,
                "turn_count": turn_count,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "readouts": _session_readouts(result),
            }
        )
    # Continuity readout: does relationship age / trust accumulate across
    # sessions under hydration but reset under isolation?
    ages = [s["readouts"]["relationship_age_turns"] for s in sessions]
    return {
        "persona_id": persona_id,
        "arm": arm,
        "session_count": session_count,
        "sessions": sessions,
        "continuity": {
            "relationship_age_final": ages[-1] if ages else 0,
            "relationship_age_monotone_nondecreasing": all(
                b >= a for a, b in zip(ages, ages[1:])
            ),
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    manifest = build_longitudinal_human_anchor_manifest()
    persona_plans = {p.persona_id: p for p in manifest.persona_plans}
    selected = (
        tuple(args.personas)
        if args.personas
        else tuple(persona_plans)
    )
    unknown = [p for p in selected if p not in persona_plans]
    if unknown:
        raise SystemExit(f"unknown personas: {unknown}; known: {sorted(persona_plans)}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, object]] = []
    for persona_id in selected:
        plan = persona_plans[persona_id]
        session_count = args.sessions or plan.session_count
        for arm in plan.comparison_arms:
            print(
                f"[longitudinal] persona={persona_id} arm={arm} "
                f"sessions={session_count}",
                flush=True,
            )
            payload = await _run_persona_arm(
                persona_id=persona_id,
                arm=arm,
                session_count=session_count,
                min_turns=plan.min_turns_per_session,
                max_turns=plan.max_turns_per_session,
            )
            path = output_dir / f"{persona_id}__{arm}.json"
            data = json.dumps(payload, indent=2, ensure_ascii=False)
            path.write_text(data, encoding="utf-8")
            artifacts.append(
                {
                    "persona_id": persona_id,
                    "arm": arm,
                    "path": str(path),
                    "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
                }
            )
    study = {
        "schema_version": _SCHEMA_VERSION,
        "evidence_tier": "synthetic-cpu",
        "note": (
            "Execution scaffold + directional continuity readouts only. "
            "Retain-grade longitudinal evidence re-runs this schedule on the "
            "real-substrate lane with cross-family judges (plan section 14)."
        ),
        "study_manifest_schema": manifest.schema_version,
        "sessions_per_persona_requested": args.sessions or "manifest-default",
        "personas": list(selected),
        "artifacts": artifacts,
        "provenance": _provenance(),
    }
    study_path = output_dir / "longitudinal_study_manifest.json"
    study_path.write_text(
        json.dumps(study, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[longitudinal] study manifest -> {study_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        type=int,
        default=None,
        help="Sessions per persona (default: manifest value, 20).",
    )
    parser.add_argument(
        "--personas",
        nargs="*",
        default=None,
        help="Subset of persona ids (default: all five).",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/longitudinal_study"
    )
    args = parser.parse_args(argv)
    if args.sessions is not None and args.sessions <= 0:
        parser.error("--sessions must be positive")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
