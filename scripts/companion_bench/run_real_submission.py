#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Real-API submission runner.

Wires :func:`companion_bench.submission.run_submission` to the production
clients (OpenAI-compatible SUT, LLM-backed user simulator, LLM-backed
per-turn + arc judges) and emits the artifact bundle the leaderboard
consumes.

This script is the only place we couple companion-bench to specific real
endpoints; the wheel itself stays system-agnostic. Pricing for cost
telemetry comes from ``companion_bench.cost.default_pricing``.

Usage::

    python scripts/companion_bench/run_real_submission.py \\
        --submission packages/companion-bench/examples/submission.yaml \\
        --user-sim-base-url https://api.anthropic.com/v1 \\
        --user-sim-model anthropic/claude-3.7-sonnet \\
        --user-sim-key-env ANTHROPIC_API_KEY \\
        --perturn-base-url https://api.anthropic.com/v1 \\
        --perturn-model anthropic/claude-3.7-sonnet \\
        --perturn-key-env ANTHROPIC_API_KEY \\
        --arc-base-url https://api.openai.com/v1 \\
        --arc-model openai/gpt-5 \\
        --arc-key-env OPENAI_API_KEY \\
        --paraphrase-seeds 0,1,2 \\
        --include-heldout \\
        --artifact-dir artifacts/companion-bench/example-2026-q2/

The user-simulator and per-turn judge MAY share a model family; the
arc judge MUST come from a different family per RFC §6.3.
"""

from __future__ import annotations

import argparse
import importlib.resources as res
import json
import logging
import os
import pathlib
import sys
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_HELDOUT_DIR = REPO_ROOT / "external" / "lscb-heldout" / "scenarios"

_LOG = logging.getLogger("run_real_submission")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_real_submission")
    p.add_argument("--submission", type=pathlib.Path, required=True)
    p.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    p.add_argument("--summary", type=pathlib.Path, default=None)

    p.add_argument("--user-sim-base-url", required=True)
    p.add_argument("--user-sim-model", required=True)
    p.add_argument("--user-sim-key-env", required=True)

    p.add_argument("--perturn-base-url", required=True)
    p.add_argument("--perturn-model", required=True)
    p.add_argument("--perturn-key-env", required=True)

    p.add_argument("--arc-base-url", required=True)
    p.add_argument("--arc-model", required=True)
    p.add_argument("--arc-key-env", required=True)

    p.add_argument("--paraphrase-seeds", default="0,1,2")
    p.add_argument("--include-heldout", action="store_true")
    p.add_argument(
        "--heldout-dir", type=pathlib.Path, default=DEFAULT_HELDOUT_DIR
    )
    p.add_argument(
        "--require-heldout",
        action="store_true",
        help="Fail if held-out submodule is missing (release tier).",
    )
    p.add_argument(
        "--family",
        default=None,
        help="Optional: restrict run to one family (F1..F6).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _resolve_seeds(spec: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def _read_env(env_var: str, label: str) -> str:
    val = os.environ.get(env_var, "").strip()
    if not val:
        raise SystemExit(f"error: {label} requires env var {env_var} to be set")
    return val


def _maybe_openrouter_headers(base_url: str) -> dict[str, str]:
    """Return OpenRouter attribution headers when the endpoint is OpenRouter.

    ``OPENROUTER_HTTP_REFERER`` + ``OPENROUTER_X_TITLE`` are read from env.
    Both optional; missing values are simply omitted. Only injected when the
    target ``base_url`` looks like OpenRouter so direct vendor calls (e.g.
    api.openai.com / api.anthropic.com / localhost VZ SUT) are unaffected.
    """

    if "openrouter.ai" not in base_url:
        return {}
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_X_TITLE", "").strip()
    if title:
        headers["X-Title"] = title
    return headers


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from companion_bench.heldout_loader import load_heldout_scenarios
    from companion_bench.judge_arc import LLMArcJudge
    from companion_bench.judge_perturn import LLMPerTurnJudge
    from companion_bench.spec import load_scenarios_dir
    from companion_bench.submission import (
        load_manifest,
        run_submission,
        write_submission_summary,
    )
    from companion_bench.sut_client import OpenAIChatClient
    from companion_bench.user_simulator import OpenAIUtteranceClient

    manifest = load_manifest(args.submission)
    sut_key = _read_env(manifest.api_key_env, "SUT")
    sut = OpenAIChatClient(
        base_url=manifest.base_url,
        api_key=sut_key,
        model=manifest.model_identifier,
        extra_headers=_maybe_openrouter_headers(manifest.base_url),
    )

    user_sim_key = _read_env(args.user_sim_key_env, "user simulator")
    user_sim = OpenAIUtteranceClient(
        base_url=args.user_sim_base_url,
        api_key=user_sim_key,
        model=args.user_sim_model,
        extra_headers=_maybe_openrouter_headers(args.user_sim_base_url),
    )

    pt_key = _read_env(args.perturn_key_env, "per-turn judge")
    arc_key = _read_env(args.arc_key_env, "arc judge")

    def make_completer(base_url: str, api_key: str, model: str):
        client = OpenAIUtteranceClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=1024,
            extra_headers=_maybe_openrouter_headers(base_url),
        )

        def complete(prompt: str, *, seed: int, system: str = "") -> str:
            return client.complete(
                system_prompt=system,
                user_prompt=prompt,
                temperature=0.0,
                seed=seed,
            )

        return complete

    perturn_judge = LLMPerTurnJudge(
        client_complete=make_completer(
            args.perturn_base_url, pt_key, args.perturn_model
        ),
        model=args.perturn_model,
    )
    arc_judge = LLMArcJudge(
        client_complete=make_completer(args.arc_base_url, arc_key, args.arc_model),
        model=args.arc_model,
    )

    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = list(load_scenarios_dir(public_dir, include_held_out=False))
    if args.include_heldout:
        held_out = load_heldout_scenarios(
            heldout_dir=pathlib.Path(args.heldout_dir),
            require=args.require_heldout,
        )
        specs.extend(held_out)
    if args.family:
        specs = [s for s in specs if s.family.value == args.family]
        if not specs:
            raise SystemExit(f"error: no scenarios for family {args.family!r}")

    _LOG.info(
        "running submission %s (%s) on %d scenarios",
        manifest.submission_id, manifest.system_name, len(specs),
    )

    paraphrase_seeds = _resolve_seeds(args.paraphrase_seeds)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = args.artifact_dir / "arcs"
    result = run_submission(
        manifest=manifest,
        specs=specs,
        sut_client=sut,
        user_backend=user_sim,
        perturn_judge=perturn_judge,
        arc_judge=arc_judge,
        paraphrase_seeds=paraphrase_seeds,
        artifact_dir=bundle_dir,
        user_simulator_model=args.user_sim_model,
    )
    summary_path = args.summary or args.artifact_dir / "summary.json"
    write_submission_summary(result, summary_path)
    _LOG.info(
        "submission %s complete; mean score = %.2f (n=%d arcs); cost USD = %s; summary → %s",
        manifest.submission_id,
        result.aggregate.final_mean,
        len(result.arc_bundles),
        result.cost.total_usd,
        summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
