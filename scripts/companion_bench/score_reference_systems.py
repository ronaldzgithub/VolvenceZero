#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Score the 10 reference systems on the Companion Bench public set (RFC §6.5).

Reads ``scripts/companion_bench/reference_systems.yaml`` and dispatches one
``run_real_submission.py`` invocation per system, then aggregates
the per-system summary JSONs into a single ``aggregate_results.json``
the leaderboard site consumes.

Cost guard: by default the script runs with ``--paraphrase-seeds 0``
(one seed per scenario) which keeps the per-system cost in the
``$40-115`` band published in RFC §6.7. Use ``--paraphrase-seeds 0,1,2``
once budget is approved for the full triple-seed protocol.

Usage::

    python scripts/companion_bench/score_reference_systems.py \\
        --user-sim-model anthropic/claude-3.7-sonnet \\
        --user-sim-key-env ANTHROPIC_API_KEY \\
        --perturn-model anthropic/claude-3.7-sonnet \\
        --perturn-key-env ANTHROPIC_API_KEY \\
        --arc-model openai/gpt-5 \\
        --arc-key-env OPENAI_API_KEY \\
        --systems openai/gpt-5,anthropic/claude-opus-4.6 \\
        --output-dir artifacts/companion-bench/reference/

Add ``--include-heldout --require-heldout`` for release-tier scoring.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import tempfile

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "scripts" / "companion_bench" / "reference_systems.yaml"
RUNNER = REPO_ROOT / "scripts" / "companion_bench" / "run_real_submission.py"

_LOG = logging.getLogger("score_reference_systems")


def _load_roster() -> list[dict]:
    with ROSTER_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return list(data["systems"])


def _build_manifest(system: dict, tmpdir: pathlib.Path) -> pathlib.Path:
    manifest = {
        "submission_id": system["submission_id"],
        "system_name": system["system_name"],
        "model_identifier": system["model_identifier"],
        "base_url": system["base_url"],
        "api_key_env": system["api_key_env"],
        "system_prompt": system.get("system_prompt", ""),
        "generation_config": {"temperature": 0.0, "max_tokens": 512},
        "attestation": {
            "no_lscb_derivative_in_training": True,
            "no_scenario_specific_prompt": True,
            "no_public_test_set_tuning": True,
            "cross_user_memory_isolation": True,
        },
        "leaderboard_category": system["leaderboard_category"],
    }
    p = tmpdir / f"{system['submission_id']}.yaml"
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True)
    return p


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="score_reference_systems")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--user-sim-model", required=True)
    p.add_argument("--user-sim-key-env", required=True)
    p.add_argument("--user-sim-base-url", default="https://api.anthropic.com/v1")
    p.add_argument("--perturn-model", required=True)
    p.add_argument("--perturn-key-env", required=True)
    p.add_argument("--perturn-base-url", default="https://api.anthropic.com/v1")
    p.add_argument("--arc-model", required=True)
    p.add_argument("--arc-key-env", required=True)
    p.add_argument("--arc-base-url", default="https://api.openai.com/v1")
    p.add_argument("--paraphrase-seeds", default="0")
    p.add_argument("--systems", default=None,
                   help="Comma-separated subset of model_identifiers; default = all 10.")
    p.add_argument("--include-heldout", action="store_true")
    p.add_argument("--require-heldout", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without executing.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roster = _load_roster()
    if args.systems:
        wanted = {s.strip() for s in args.systems.split(",") if s.strip()}
        roster = [s for s in roster if s["model_identifier"] in wanted]
        if not roster:
            print(f"error: no roster systems matched {args.systems}", file=sys.stderr)
            return 2

    aggregate: dict = {"systems": []}
    with tempfile.TemporaryDirectory(prefix="companion-bench-roster-") as tmpdir:
        tmp = pathlib.Path(tmpdir)
        for system in roster:
            manifest_path = _build_manifest(system, tmp)
            sys_artifact_dir = args.output_dir / system["submission_id"]
            cmd = [
                sys.executable,
                str(RUNNER),
                "--submission", str(manifest_path),
                "--artifact-dir", str(sys_artifact_dir),
                "--user-sim-base-url", args.user_sim_base_url,
                "--user-sim-model", args.user_sim_model,
                "--user-sim-key-env", args.user_sim_key_env,
                "--perturn-base-url", args.perturn_base_url,
                "--perturn-model", args.perturn_model,
                "--perturn-key-env", args.perturn_key_env,
                "--arc-base-url", args.arc_base_url,
                "--arc-model", args.arc_model,
                "--arc-key-env", args.arc_key_env,
                "--paraphrase-seeds", args.paraphrase_seeds,
            ]
            if args.include_heldout:
                cmd.append("--include-heldout")
            if args.require_heldout:
                cmd.append("--require-heldout")

            _LOG.info("[%s] %s", system["submission_id"], " ".join(cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd, check=False, cwd=REPO_ROOT)
            if result.returncode != 0:
                _LOG.error("[%s] runner exited %d", system["submission_id"], result.returncode)
                continue
            summary_path = sys_artifact_dir / "summary.json"
            if not summary_path.exists():
                _LOG.warning(
                    "[%s] no summary.json at %s; skipping aggregate row",
                    system["submission_id"], summary_path,
                )
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            aggregate["systems"].append(
                {
                    "submission_id": system["submission_id"],
                    "system_name": system["system_name"],
                    "leaderboard_category": system["leaderboard_category"],
                    "summary": payload,
                }
            )
    out = args.output_dir / "aggregate_results.json"
    if not args.dry_run:
        with out.open("w", encoding="utf-8") as fh:
            json.dump(aggregate, fh, indent=2, ensure_ascii=False)
        _LOG.info("aggregate → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
