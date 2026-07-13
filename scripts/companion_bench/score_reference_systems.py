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
import concurrent.futures
import json
import logging
import pathlib
import subprocess
import sys
import tempfile
import time

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from proxy_support import configure_companion_bench_proxy  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ROSTER_PATH = REPO_ROOT / "scripts" / "companion_bench" / "reference_systems.yaml"
RUNNER = REPO_ROOT / "scripts" / "companion_bench" / "run_real_submission.py"

_LOG = logging.getLogger("score_reference_systems")


def _load_roster(path: pathlib.Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
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
            "no_companionbench_derivative_in_training": True,
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
    p.add_argument(
        "--roster",
        type=pathlib.Path,
        default=DEFAULT_ROSTER_PATH,
        help=(
            "Path to roster YAML (default: reference_systems.yaml; "
            "smoke runs use reference_systems.smoke.yaml)."
        ),
    )
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
                   help="Comma-separated subset of model_identifiers; default = all roster entries.")
    p.add_argument(
        "--family",
        default=None,
        help=(
            "Optional: restrict run to one or more families (F1..F6), single "
            "or comma-separated (e.g. 'F1' or 'F1,F2,F3'). Passed through to "
            "run_real_submission."
        ),
    )
    p.add_argument("--include-heldout", action="store_true")
    p.add_argument("--require-heldout", action="store_true")
    p.add_argument(
        "--per-system-timeout-min",
        type=int,
        default=30,
        help=(
            "Per-system subprocess wallclock timeout in minutes "
            "(default 30). On timeout, the runner kills the subprocess "
            "and continues with the next system; the timed-out system "
            "is logged with no aggregate row. Prevents the kind of "
            "silent hang seen when a SUT endpoint is unreachable and "
            "urllib's default 120s timeout × N turns ≈ many hours."
        ),
    )
    p.add_argument(
        "--parallel-sut",
        type=int,
        default=1,
        help=(
            "Number of SUT subprocesses to run concurrently (debt #34). "
            "Default 1 keeps deterministic behaviour for reproducibility "
            "audits. Bump up to N to fan out across N SUTs in parallel "
            "(useful for sweep runs with abundant API quota); each SUT "
            "still gets its own subprocess + per-system-timeout, so a "
            "stuck SUT cannot starve the others."
        ),
    )
    p.add_argument(
        "--per-system-retries",
        type=int,
        default=0,
        help=(
            "Number of automatic retries per SUT on non-zero subprocess "
            "exit (debt #34, complements per-arc retry inside "
            "submission.run_submission). Default 0 = single attempt. "
            "Each retry re-invokes the same subprocess command; "
            "transient HTTP 429 / 5xx surface as non-zero exit codes "
            "the runner can retry without manual intervention. Timeouts "
            "do NOT retry (suspect endpoint config, not transient)."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without executing.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse an existing valid summary.json instead of rerunning that system",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_companion_bench_proxy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roster_path: pathlib.Path = args.roster
    if not roster_path.exists():
        print(f"error: roster YAML not found at {roster_path}", file=sys.stderr)
        return 2
    _LOG.info("loading roster from %s", roster_path)
    roster = _load_roster(roster_path)
    if args.systems:
        wanted = {s.strip() for s in args.systems.split(",") if s.strip()}
        roster = [s for s in roster if s["model_identifier"] in wanted]
        if not roster:
            print(f"error: no roster systems matched {args.systems}", file=sys.stderr)
            return 2

    aggregate: dict = {"systems": []}
    timeout_sec = max(60, int(args.per_system_timeout_min) * 60)
    retries = max(0, int(args.per_system_retries))
    parallelism = max(1, int(args.parallel_sut))

    def _run_one(system: dict, manifest_path: pathlib.Path) -> dict | None:
        sys_artifact_dir = args.output_dir / system["submission_id"]
        summary_path = sys_artifact_dir / "summary.json"
        if args.resume and summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            _LOG.info("[%s] resume: reusing %s", system["submission_id"], summary_path)
            return {
                "submission_id": system["submission_id"],
                "system_name": system["system_name"],
                "leaderboard_category": system["leaderboard_category"],
                "summary": payload,
            }
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
        if args.family:
            cmd.extend(["--family", args.family])

        _LOG.info("[%s] %s", system["submission_id"], " ".join(cmd))
        if args.dry_run:
            return None

        attempt = 0
        max_attempts = retries + 1
        while attempt < max_attempts:
            attempt += 1
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    cwd=REPO_ROOT,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                # Timeout = endpoint config issue, not transient.
                # Don't retry; the next pass would just hit the same
                # wall and burn more time.
                _LOG.error(
                    "[%s] runner exceeded %d-min wallclock timeout (attempt %d/%d); "
                    "subprocess killed; not retrying — fix endpoint config",
                    system["submission_id"],
                    args.per_system_timeout_min,
                    attempt,
                    max_attempts,
                )
                return None
            if result.returncode == 0:
                break
            _LOG.warning(
                "[%s] runner exited %d (attempt %d/%d)",
                system["submission_id"],
                result.returncode,
                attempt,
                max_attempts,
            )
            if attempt < max_attempts:
                # Exponential backoff (capped) between retries so a
                # transient API rate-limit clears.
                backoff_s = min(60, 2 ** attempt)
                _LOG.info(
                    "[%s] backing off %ds before retry",
                    system["submission_id"], backoff_s,
                )
                time.sleep(backoff_s)

        if result.returncode != 0:
            _LOG.error(
                "[%s] runner failed after %d attempts; "
                "no aggregate row recorded",
                system["submission_id"], max_attempts,
            )
            return None
        if not summary_path.exists():
            _LOG.warning(
                "[%s] no summary.json at %s; skipping aggregate row",
                system["submission_id"], summary_path,
            )
            return None
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "submission_id": system["submission_id"],
            "system_name": system["system_name"],
            "leaderboard_category": system["leaderboard_category"],
            "summary": payload,
        }

    with tempfile.TemporaryDirectory(prefix="companion-bench-roster-") as tmpdir:
        tmp = pathlib.Path(tmpdir)
        manifest_paths = {s["submission_id"]: _build_manifest(s, tmp) for s in roster}
        if parallelism == 1:
            # Serial path: preserve previous deterministic ordering for
            # audit / reproducibility runs.
            for system in roster:
                row = _run_one(system, manifest_paths[system["submission_id"]])
                if row is not None:
                    aggregate["systems"].append(row)
        else:
            _LOG.info("running %d SUTs with parallelism=%d", len(roster), parallelism)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parallelism,
                thread_name_prefix="companion-bench-sut",
            ) as executor:
                futures = {
                    executor.submit(_run_one, s, manifest_paths[s["submission_id"]]): s
                    for s in roster
                }
                for fut in concurrent.futures.as_completed(futures):
                    row = fut.result()
                    if row is not None:
                        aggregate["systems"].append(row)
            # Sort post-hoc so the aggregate row order matches roster
            # order even when threads completed out of sequence.
            order = {s["submission_id"]: i for i, s in enumerate(roster)}
            aggregate["systems"].sort(key=lambda r: order.get(r["submission_id"], 1_000))

    out = args.output_dir / "aggregate_results.json"
    if not args.dry_run:
        with out.open("w", encoding="utf-8") as fh:
            json.dump(aggregate, fh, indent=2, ensure_ascii=False)
        _LOG.info("aggregate → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
