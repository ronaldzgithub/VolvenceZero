#!/usr/bin/env python3
"""CLI driver: run a companion-bench scenario against ``lifeform-service``.

This script reuses two libraries directly and stitches them together
over HTTP:

* ``companion_bench.user_simulator.UserSimulator`` — Apache 2.0,
  drives the FSM + LLM-backed user utterance generation.
* ``lifeform-service`` HTTP endpoints (``/v1/sessions/{sid}/turns``
  etc.) — proprietary, exposes per-turn governance signals
  (``active_regime`` / ``pe_magnitude`` / ``commitment_count`` /
  ``open_loop_count`` / ``response_rationale_tags`` / ``safety``).

Why this driver exists alongside the in-server simulator routes
================================================================

The in-server ``/v1/sessions/{sid}/simulator/*`` routes orchestrate the
exact same arc when the chat UI's "Run Simulator" button is pressed.
This driver is the CLI sibling for batch / evidence / scripted demos:

* it can run headless on CI / a screen-less workstation;
* it writes a JSONL trace under
  ``artifacts/governance_demo/<user_id>/<scenario>-<seed>-<ts>.jsonl``
  that downstream tooling can replay;
* it produces a coloured live console summary of every governance
  signal, useful for "look at this terminal output, governance is
  actually a thing" investor demos.

Usage::

    python scripts/governance_demo/drive_session_arc.py \
        --base-url http://127.0.0.1:8765 \
        --user-id alice \
        --scenario F2-repair-002 \
        --paraphrase-seed 0 \
        --backend openrouter  # openrouter | fake

The default ``--backend openrouter`` reads ``PROTOCOL_LLM_*`` env vars the
same way the JSON-mode protocol-uptake client does (see
``packages/lifeform-service/src/lifeform_service/openai_utterance_client.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.resources as res
import json
import os
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

# companion-bench is Apache 2.0; consuming its public API from this
# proprietary workspace is allowed by direction (the reverse is
# enforced against by tests/contracts/test_companion_bench_no_internal_imports.py).
from companion_bench.spec import ScenarioSpec, load_scenario_yaml
from companion_bench.user_simulator import (
    DeterministicFakeUtteranceClient,
    TurnContext,
    UserSimulator,
    UtteranceClient,
)

from lifeform_service.openai_utterance_client import (
    OpenAiUtteranceClient,
    build_utterance_client_from_env,
)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------


class _AnsiPalette:
    """Minimal ANSI palette; degrades to no-op when stdout isn't a TTY."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def dim(self, text: str) -> str:
        return self.wrap("2", text)

    def bold(self, text: str) -> str:
        return self.wrap("1", text)

    def green(self, text: str) -> str:
        return self.wrap("32", text)

    def amber(self, text: str) -> str:
        return self.wrap("33", text)

    def red(self, text: str) -> str:
        return self.wrap("31", text)

    def cyan(self, text: str) -> str:
        return self.wrap("36", text)

    def magenta(self, text: str) -> str:
        return self.wrap("35", text)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ServiceClient:
    base_url: str
    user_id: str
    timeout_seconds: float = 120.0

    def _req(
        self,
        path: str,
        *,
        method: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Alpha-User": self.user_id,
        }
        request = urllib.request.Request(
            url, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{method} {path} -> HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"{method} {path} -> URL error: {exc.reason}"
            ) from exc
        if not payload_text:
            return {}
        return json.loads(payload_text)

    def create_session(
        self,
        *,
        session_id: str | None,
        vertical: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if vertical:
            body["vertical"] = vertical
        return self._req("/v1/sessions", method="POST", body=body)

    def post_turn(self, sid: str, user_input: str) -> dict[str, Any]:
        return self._req(
            f"/v1/sessions/{sid}/turns",
            method="POST",
            body={"user_input": user_input},
        )

    def end_scene(self, sid: str, reason: str) -> dict[str, Any]:
        return self._req(
            f"/v1/sessions/{sid}/end-scene",
            method="POST",
            body={"reason": reason, "drain_slow_loop": True},
        )

    def close_session(self, sid: str) -> dict[str, Any]:
        return self._req(f"/v1/sessions/{sid}", method="DELETE")

    def relationship_summary(self) -> dict[str, Any]:
        return self._req(
            "/v1/users/me/relationship-summary",
            method="GET",
        )


# ---------------------------------------------------------------------------
# Scenario resolution
# ---------------------------------------------------------------------------


def _public_scenarios_dir() -> pathlib.Path:
    return pathlib.Path(
        str(res.files("companion_bench") / "scenarios" / "public")
    )


def _resolve_scenario(arg: str) -> ScenarioSpec:
    """Accept either a scenario_id (F2-repair-002) or a YAML path."""

    candidate = pathlib.Path(arg)
    if candidate.suffix.lower() == ".yaml" and candidate.exists():
        return load_scenario_yaml(candidate)
    yaml_path = _public_scenarios_dir() / f"{arg}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"scenario {arg!r} not found in {yaml_path.parent} "
            f"(also tried interpreting as a file path: {candidate})"
        )
    return load_scenario_yaml(yaml_path)


def _build_schedule(
    spec: ScenarioSpec, paraphrase_seed: int
) -> list[tuple[int, int, int]]:
    """Return ``[(session_index, turn_index, gap_days_before_turn), ...]``.

    Mirrors the schedule built server-side in ``simulator_routes._build_schedule``
    so CLI and UI runs against the same (scenario_id, paraphrase_seed)
    produce identical arc lengths.
    """

    rng = random.Random(f"{spec.scenario_id}|{paraphrase_seed}|schedule")
    lo, hi = spec.session_turn_range
    rows: list[tuple[int, int, int]] = []
    for s_idx in range(1, spec.arc_length_sessions + 1):
        turn_count = rng.randint(lo, hi)
        gap_days = (
            0 if s_idx == 1 else spec.inter_session_gap_days[s_idx - 2]
        )
        for t_idx in range(1, turn_count + 1):
            rows.append((s_idx, t_idx, gap_days if t_idx == 1 else 0))
    return rows


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _resolve_backend(name: str) -> UtteranceClient:
    if name == "fake":
        return DeterministicFakeUtteranceClient()
    if name in {"openrouter", "qwen", "auto"}:
        client = build_utterance_client_from_env()
        if client is not None:
            return client
        if name in {"openrouter", "qwen"}:
            raise RuntimeError(
                f"--backend {name} requires PROTOCOL_LLM_API_KEY to be set "
                "(OpenRouter by default). Either export the env "
                "var or pass --backend fake."
            )
        # ``auto`` falls back silently to the deterministic fake when
        # no credentials are present; callers that need to detect this
        # should pass --backend openrouter explicitly.
        return DeterministicFakeUtteranceClient()
    raise ValueError(
        f"unknown --backend {name!r}; expected openrouter|fake|auto"
    )


# ---------------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------------


class _JsonlTrace:
    """Append-only JSONL writer for the run trace."""

    def __init__(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self._path = path

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def write(self, kind: str, **fields: Any) -> None:
        record = {
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _summarise_turn(
    pal: _AnsiPalette, turn_payload: dict[str, Any]
) -> str:
    regime = turn_payload.get("active_regime") or "none"
    abstract = turn_payload.get("active_abstract_action") or "-"
    pe = turn_payload.get("pe_magnitude")
    pe_str = f"{pe:.3f}" if isinstance(pe, (int, float)) else "?"
    commitments = turn_payload.get("commitment_count", 0)
    open_loops = turn_payload.get("open_loop_count", 0)
    tags = turn_payload.get("response_rationale_tags") or ()
    tag_str = ",".join(tags) if tags else "-"
    safety = turn_payload.get("safety") or {}
    safety_chips = []
    for k, v in safety.items():
        if v:
            safety_chips.append(f"{k}=on")
    safety_str = ",".join(safety_chips) if safety_chips else "-"
    return (
        f"{pal.cyan('regime=')}{regime} "
        f"{pal.dim('z=')}{abstract} "
        f"{pal.amber('pe=')}{pe_str} "
        f"{pal.green('com=')}{commitments} "
        f"{pal.green('open=')}{open_loops} "
        f"{pal.magenta('tags=')}{tag_str} "
        f"{pal.red('safety=')}{safety_str}"
    )


def run_arc(
    *,
    spec: ScenarioSpec,
    paraphrase_seed: int,
    backend: UtteranceClient,
    service: ServiceClient,
    vertical: str | None,
    trace: _JsonlTrace,
    pal: _AnsiPalette,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Run one full arc; return a final stats dict.

    ``max_turns`` truncates the arc after the given number of POSTed
    user turns (default: run the whole schedule). Useful for the smoke
    test which wants to keep wall-clock cost bounded.
    """

    schedule = _build_schedule(spec, paraphrase_seed)
    if max_turns is not None and max_turns > 0:
        schedule = schedule[:max_turns]
    if not schedule:
        raise RuntimeError(
            "schedule is empty; check scenario session_turn_range"
        )
    simulator = UserSimulator(
        spec=spec,
        paraphrase_seed=paraphrase_seed,
        backend=backend,
    )
    trace.write(
        "arc_begin",
        scenario_id=spec.scenario_id,
        paraphrase_seed=paraphrase_seed,
        arc_length_sessions=spec.arc_length_sessions,
        session_turn_range=list(spec.session_turn_range),
        inter_session_gap_days=list(spec.inter_session_gap_days),
        schedule_length=len(schedule),
        identity_name=simulator.identity.name,
        identity_occupation=simulator.identity.occupation,
    )

    print(
        pal.bold(
            f"== arc: scenario={spec.scenario_id} seed={paraphrase_seed} "
            f"sessions={spec.arc_length_sessions} schedule_len={len(schedule)}"
        )
    )
    print(
        pal.dim(
            f"   identity={simulator.identity.name}, "
            f"persona={spec.user_simulator.persona!r}"
        )
    )

    current_session_id: str | None = None
    current_arc_session: int = 0
    bot_text_for_next_turn = ""
    turns_run = 0

    for s_idx, t_idx, gap_days in schedule:
        if s_idx != current_arc_session:
            # Close the prior HTTP session, mint a new one.
            if current_session_id is not None:
                try:
                    end_payload = service.end_scene(
                        current_session_id, reason="cli-arc-session-end"
                    )
                    trace.write(
                        "scene_close",
                        session_id=current_session_id,
                        evidence_artifact_ref=end_payload.get(
                            "evidence_artifact_ref"
                        ),
                    )
                except RuntimeError as exc:
                    trace.write(
                        "scene_close_failed",
                        session_id=current_session_id,
                        error=str(exc),
                    )
                # Cross-session governance banner.
                try:
                    summary = service.relationship_summary()
                    trace.write(
                        "relationship_summary",
                        session_id=current_session_id,
                        **summary,
                    )
                    banner = (
                        f"ruptures={summary.get('rupture_repair_count', 0)} "
                        f"repaired={summary.get('observed_repair_count', 0)} "
                        f"kinds={','.join(summary.get('rupture_kinds', []) or []) or '-'}"
                    )
                    print(pal.green(f"   [relationship] {banner}"))
                except RuntimeError as exc:
                    trace.write(
                        "relationship_summary_failed",
                        error=str(exc),
                    )
                try:
                    service.close_session(current_session_id)
                except RuntimeError as exc:
                    trace.write(
                        "close_session_failed",
                        session_id=current_session_id,
                        error=str(exc),
                    )
                current_session_id = None
            # Visual gap.
            if s_idx > 1:
                print(
                    pal.amber(
                        f"\n--- gap: {gap_days} day(s) "
                        f"(arc session {s_idx}/{spec.arc_length_sessions}) ---"
                    )
                )
            sid_request = (
                f"{spec.scenario_id}-{paraphrase_seed}-s{s_idx}-{int(time.time())}"
            )
            created = service.create_session(
                session_id=sid_request, vertical=vertical
            )
            current_session_id = created["session_id"]
            current_arc_session = s_idx
            trace.write(
                "session_create",
                session_id=current_session_id,
                vertical=created.get("vertical"),
                arc_session_index=s_idx,
                gap_days=gap_days,
            )
            print(
                pal.bold(
                    f"-> arc session {s_idx}: HTTP session "
                    f"{current_session_id} created"
                )
            )

        assert current_session_id is not None
        if bot_text_for_next_turn:
            simulator.append_assistant(bot_text_for_next_turn)
            bot_text_for_next_turn = ""

        generated = simulator.next_turn(
            TurnContext(
                session_index=s_idx,
                turn_index=t_idx,
                inter_session_gap_days=gap_days,
            )
        )
        fsm_action = generated.fsm_step.action if generated.fsm_step else None

        trace.write(
            "user_turn",
            session_id=current_session_id,
            arc_session_index=s_idx,
            turn_index=t_idx,
            gap_days=gap_days,
            user_text=generated.text,
            fsm_action=fsm_action,
            fsm_payload=(
                generated.fsm_step.payload if generated.fsm_step else None
            ),
            paraphrase_seed=paraphrase_seed,
            rng_draw_count=generated.rng_draw_count,
        )
        fsm_label = f" [{fsm_action}]" if fsm_action else ""
        print(
            pal.cyan(f"\n[user · s{s_idx}t{t_idx}{fsm_label}] ")
            + generated.text
        )

        turn_payload = service.post_turn(current_session_id, generated.text)
        bot_text_for_next_turn = turn_payload.get("response_text", "") or ""
        turns_run += 1

        # The service's TurnResponse echoes ``session_id`` and
        # ``turn_index``; we want our trace-meta values (which carry
        # the *arc-relative* indices) at the top level. Project the
        # governance-relevant fields out explicitly so they stay
        # easy to grep, and stash the rest under ``service_payload``
        # for full provenance.
        trace.write(
            "bot_turn",
            session_id=current_session_id,
            arc_session_index=s_idx,
            arc_turn_index=t_idx,
            response_text=turn_payload.get("response_text"),
            active_regime=turn_payload.get("active_regime"),
            active_abstract_action=turn_payload.get(
                "active_abstract_action"
            ),
            expression_intent=turn_payload.get("expression_intent"),
            pe_magnitude=turn_payload.get("pe_magnitude"),
            open_loop_count=turn_payload.get("open_loop_count"),
            commitment_count=turn_payload.get("commitment_count"),
            response_rationale_tags=list(
                turn_payload.get("response_rationale_tags") or ()
            ),
            safety=turn_payload.get("safety"),
            service_payload=turn_payload,
        )
        print(pal.dim(f"[bot]  ") + bot_text_for_next_turn)
        print("       " + _summarise_turn(pal, turn_payload))

    # Close the last session.
    if current_session_id is not None:
        try:
            end_payload = service.end_scene(
                current_session_id, reason="cli-arc-end"
            )
            trace.write(
                "scene_close",
                session_id=current_session_id,
                evidence_artifact_ref=end_payload.get(
                    "evidence_artifact_ref"
                ),
            )
        except RuntimeError as exc:
            trace.write(
                "scene_close_failed",
                session_id=current_session_id,
                error=str(exc),
            )
        try:
            summary = service.relationship_summary()
            trace.write(
                "relationship_summary",
                session_id=current_session_id,
                **summary,
            )
            banner = (
                f"ruptures={summary.get('rupture_repair_count', 0)} "
                f"repaired={summary.get('observed_repair_count', 0)} "
                f"kinds={','.join(summary.get('rupture_kinds', []) or []) or '-'}"
            )
            print(pal.green(f"\n[final relationship] {banner}"))
        except RuntimeError as exc:
            trace.write(
                "relationship_summary_failed",
                error=str(exc),
            )
        try:
            service.close_session(current_session_id)
        except RuntimeError as exc:
            trace.write(
                "close_session_failed",
                session_id=current_session_id,
                error=str(exc),
            )

    trace.write("arc_end", turns_run=turns_run)
    return {"turns_run": turns_run, "trace_path": str(trace.path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive a lifeform-service session with a companion-bench "
            "user simulator. Streams per-turn governance signals to "
            "stdout and writes a JSONL trace under artifacts/."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "GOVERNANCE_DEMO_BASE_URL", "http://127.0.0.1:8765"
        ),
        help="lifeform-service base URL (default %(default)s)",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help=(
            "X-Alpha-User to bind the arc to. Same user across runs "
            "compounds cross-session memory."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help=(
            "Scenario id (e.g. F2-repair-002) or path to a custom YAML."
        ),
    )
    parser.add_argument(
        "--paraphrase-seed",
        type=int,
        default=0,
        help="paraphrase_seed in [0, scenario.paraphrase_seed_count)",
    )
    parser.add_argument(
        "--backend",
        choices=("openrouter", "qwen", "fake", "auto"),
        default="openrouter",
        help=(
            "User-utterance backend. ``openrouter`` reads PROTOCOL_LLM_*; "
            "``fake`` is the deterministic hash-based stub; ``auto`` "
            "uses OpenRouter when keys are set, fake otherwise. ``qwen`` "
            "is a compatibility alias that still reads PROTOCOL_LLM_*."
        ),
    )
    parser.add_argument(
        "--vertical",
        default=None,
        help="vertical name to bind every created session to (optional)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/governance_demo",
        help="root dir for the JSONL trace (default %(default)s)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI colours in console output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    pal = _AnsiPalette(
        enabled=(not args.no_color) and sys.stdout.isatty()
    )
    spec = _resolve_scenario(args.scenario)
    if args.paraphrase_seed < 0 or args.paraphrase_seed >= spec.paraphrase_seed_count:
        parser.error(
            f"--paraphrase-seed must be in "
            f"[0, {spec.paraphrase_seed_count}); got {args.paraphrase_seed}"
        )
    backend = _resolve_backend(args.backend)
    service = ServiceClient(
        base_url=args.base_url, user_id=args.user_id
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = (
        pathlib.Path(args.artifacts_dir)
        / args.user_id
        / f"{spec.scenario_id}-{args.paraphrase_seed}-{ts}.jsonl"
    )
    trace = _JsonlTrace(trace_path)
    print(pal.dim(f"trace: {trace_path}"))
    try:
        stats = run_arc(
            spec=spec,
            paraphrase_seed=args.paraphrase_seed,
            backend=backend,
            service=service,
            vertical=args.vertical,
            trace=trace,
            pal=pal,
        )
    finally:
        trace.close()
    print(
        pal.bold(
            f"\nturns_run={stats['turns_run']} "
            f"trace={stats['trace_path']}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
