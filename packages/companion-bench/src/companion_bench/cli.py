# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""``python -m companion_bench`` / ``companion-bench`` console entry point.

Subcommands:

* ``run`` — score one submission against the full public scenario set.
* ``smoke`` — same shape as ``run`` but with the deterministic fakes
  for SUT and judges; no API calls. Used by CI.
* ``hashes`` — emit the canonical scenario hash table.
* ``list-scenarios`` — print the loaded scenario set with their hashes.
"""

from __future__ import annotations

import argparse
import importlib.resources as res
import logging
import os
import pathlib
import sys

from companion_bench.heldout_loader import load_heldout_scenarios
from companion_bench.spec import load_scenarios_dir, scenario_hash
from companion_bench.submission import (
    SubmissionResult,
    dry_run_with_fakes,
    load_manifest,
    run_submission,
    write_submission_summary,
)


_LOG = logging.getLogger("companion-bench")


def _public_dir() -> pathlib.Path:
    return pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run the deterministic-fake pipeline end-to-end."""
    from companion_bench.sut_client import EchoFakeSUTClient
    from companion_bench.user_simulator import DeterministicFakeUtteranceClient
    from companion_bench.submission import SubmissionManifest, SubmissionAttestation

    specs = load_scenarios_dir(_public_dir(), include_held_out=False)
    if args.scenario_id:
        specs = tuple(s for s in specs if s.scenario_id == args.scenario_id)
        if not specs:
            print(f"error: no public scenario with id {args.scenario_id!r}", file=sys.stderr)
            return 2
    elif args.family:
        specs = tuple(s for s in specs if s.family.value == args.family)

    manifest = SubmissionManifest(
        submission_id="smoke-test",
        system_name="companionbench-smoke-fakes",
        model_identifier="fake-sut/echo",
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt="(none)",
        generation_config={},
        attestation=SubmissionAttestation(True, True, True, True),
        leaderboard_category="bespoke",
    )
    out_dir = pathlib.Path(args.artifact_dir) if args.artifact_dir else None
    result = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        paraphrase_seeds=(0,),
        artifact_dir=out_dir,
    )
    summary_path = (
        pathlib.Path(args.summary or (out_dir / "summary.json" if out_dir else "summary.json"))
    )
    write_submission_summary(result, summary_path)
    print(f"smoke OK: {len(result.arc_bundles)} arcs, mean score = {result.aggregate.final_mean:.2f}")
    print(f"summary → {summary_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a real submission against the public set + (optional) held-out set."""
    manifest = load_manifest(args.submission)
    api_key = os.environ.get(manifest.api_key_env, "")
    if not api_key:
        print(
            f"error: API key env var {manifest.api_key_env!r} not set", file=sys.stderr,
        )
        return 2
    # Defer the real-SUT path: building OpenAIChatClient + LLM judges
    # requires per-team configuration. We expose this as a Python API
    # rather than a fragile CLI flag explosion; the v1.0 release ships
    # `scripts/companion_bench/run_real_submission.py` for callers that want a
    # one-line invocation pattern.
    print(
        "companion-bench run is the orchestrator; for real SUT/judge wiring, "
        "use scripts/companion_bench/run_real_submission.py which composes "
        "companion_bench.run_submission(...) with the production clients.",
        file=sys.stderr,
    )
    return 1


def cmd_hashes(args: argparse.Namespace) -> int:
    """Emit the canonical scenario hash table."""
    specs = load_scenarios_dir(_public_dir(), include_held_out=False)
    if args.include_heldout:
        heldout = load_heldout_scenarios(
            heldout_dir=pathlib.Path(args.heldout_dir),
            require=False,
        )
        specs = tuple(specs) + tuple(heldout)
    sorted_specs = sorted(specs, key=lambda s: s.scenario_id)
    for s in sorted_specs:
        print(f"{s.scenario_id}\t{scenario_hash(s)}")
    return 0


def cmd_list_scenarios(args: argparse.Namespace) -> int:
    """Print public scenarios with families + hashes."""
    specs = load_scenarios_dir(_public_dir(), include_held_out=False)
    print(f"{'scenario_id':<36} {'family':<6} {'sessions':<10} {'hash (8)':<10}")
    print("-" * 70)
    for s in sorted(specs, key=lambda x: x.scenario_id):
        h = scenario_hash(s)[:8]
        print(f"{s.scenario_id:<36} {s.family.value:<6} {s.arc_length_sessions:<10} {h:<10}")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="companion-bench")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser(
        "smoke", help="Deterministic-fake end-to-end run for CI / dry-run."
    )
    smoke.add_argument("--scenario-id", default=None)
    smoke.add_argument("--family", default=None)
    smoke.add_argument("--artifact-dir", default=None)
    smoke.add_argument("--summary", default=None)
    smoke.set_defaults(func=cmd_smoke)

    run = sub.add_parser("run", help="Run a real submission manifest.")
    run.add_argument("--submission", required=True, type=pathlib.Path)
    run.add_argument("--include-heldout", action="store_true")
    run.add_argument(
        "--heldout-dir",
        type=pathlib.Path,
        default=pathlib.Path("external/companionbench-heldout/scenarios"),
    )
    run.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    run.add_argument("--summary", type=pathlib.Path, default=None)
    run.set_defaults(func=cmd_run)

    hashes = sub.add_parser(
        "hashes", help="Emit the canonical scenario hash table."
    )
    hashes.add_argument("--include-heldout", action="store_true")
    hashes.add_argument(
        "--heldout-dir",
        default="external/companionbench-heldout/scenarios",
    )
    hashes.set_defaults(func=cmd_hashes)

    listing = sub.add_parser(
        "list-scenarios", help="List public scenarios + summaries."
    )
    listing.set_defaults(func=cmd_list_scenarios)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
