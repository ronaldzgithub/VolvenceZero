#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Phased driver for the same-substrate Companion Bench ablation (debt #87).

Phases (see docs/specs/companion-ablation.md):

* ``p0-smoke``       In-process, no network, no GPU. Runs all five tracks with
                     deterministic fakes, writes one summary.json per track, and
                     runs the verdict comparator. Proves the wiring + artifact
                     shape + comparator end-to-end. NOTE: with identical fakes
                     the tracks do not separate, so the verdict will not be a
                     positive result — P0 only certifies the flow runs.
* ``judge-evidence`` Runs the judge robustness + calibration scaffolds so the
                     evidence bundle records judge variance/calibration before
                     trusting any scores (#48 / #71). Defaults to ``--dry-run``.
* ``p1``             Directional: real Qwen + cross-family judges on the 24
                     public scenarios, 1 seed. Builds + runs
                     score_reference_systems on the same-substrate roster, then
                     the comparator.
* ``p2``             Retain: 24 public + 96 held-out, 3 seeds. Same as p1 with
                     ``--include-heldout --require-heldout --paraphrase-seeds 0,1,2``.

Cross-family judge invariant (#71 / #72): the substrate is Qwen, so the
user-simulator and BOTH judges MUST be a different family. The defaults below
use Claude (user-sim + per-turn) and GPT-5 (arc) — none are Qwen. Override with
the ``--*-model`` / ``--*-key-env`` flags if you use a different non-Qwen vendor.

P1/P2 require the three-process serving topology to be live: one
``lifeform-serve --ablation-bundle`` owner on :8000 plus ref-harness on :8500
and camel on :8600 (see serve_same_substrate_ablation.sh). Use ``--dry-run``
to print the exact commands without executing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "companion_bench"
ABLATION_ROSTER = SCRIPTS / "reference_systems.same_substrate_ablation.yaml"
SCORE_CMD = SCRIPTS / "score_reference_systems.py"
COMPARATOR = SCRIPTS / "compare_companion_ablation.py"

# Ablation track <-> roster submission_id. SSOT for both phases.
TRACK_TO_SUBMISSION: dict[str, str] = {
    "raw": "abl-raw",
    "ref-harness": "abl-ref-harness",
    "camel": "abl-camel",
    "volvence-cold": "abl-volvence-cold",
    "volvence": "abl-volvence",
}

# Component-causal arms (claim_component_causal_contribution; frozen registry
# in docs/specs/human-world-model-ablation.md). P0 smoke always exercises them
# so the full matrix wiring + verdict schema is proven end-to-end; P1/P2 only
# include an arm when the roster actually declares its submission_id (served
# endpoints for these arms are still pending: eta-off/pe-off profile serving,
# active-learning-off #90, LoRA bake #41).
COMPONENT_TRACK_TO_SUBMISSION: dict[str, str] = {
    "pe-off": "abl-pe-off",
    "eta-off": "abl-eta-off",
    "active-learning-off": "abl-active-learning-off",
    "lora-adapter": "abl-lora-adapter",
}

TRACK_TO_VERTICAL: dict[str, str] = {
    "volvence": "companion",
    "volvence-cold": "companion-cold",
    "pe-off": "companion-pe-drive-off",
    "eta-off": "companion-eta-off",
    "active-learning-off": "companion-active-learning-off",
    "lora-adapter": "companion-lora-adapter",
}


def _roster_submission_ids(roster_path: pathlib.Path) -> frozenset[str]:
    """Read the submission ids declared in the roster YAML (fail loud)."""

    import yaml

    payload = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    systems = payload["systems"]
    return frozenset(str(entry["submission_id"]) for entry in systems)


# ---------------------------------------------------------------------------
# Cross-family judge defaults (#71 / #72 — NONE are Qwen)
# ---------------------------------------------------------------------------


DEFAULT_USER_SIM_MODEL = "anthropic/claude-3.7-sonnet"
DEFAULT_USER_SIM_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_USER_SIM_KEY_ENV = "ANTHROPIC_API_KEY"

DEFAULT_PERTURN_MODEL = "anthropic/claude-3.7-sonnet"
DEFAULT_PERTURN_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_PERTURN_KEY_ENV = "ANTHROPIC_API_KEY"

DEFAULT_ARC_MODEL = "openai/gpt-5"
DEFAULT_ARC_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ARC_KEY_ENV = "OPENAI_API_KEY"


# ---------------------------------------------------------------------------
# Phase: P0 in-process wiring smoke
# ---------------------------------------------------------------------------


def phase_p0_smoke(*, output_dir: pathlib.Path, family: str | None) -> int:
    try:
        from companion_bench.spec import load_scenarios_dir
        from companion_bench.submission import (
            SubmissionAttestation,
            SubmissionManifest,
            dry_run_with_fakes,
            write_submission_summary,
        )
        from companion_bench.sut_client import EchoFakeSUTClient
        from companion_bench.user_simulator import DeterministicFakeUtteranceClient
        import importlib.resources as res
    except ImportError as exc:
        print(
            "error: companion_bench is not importable. Install it with "
            "`pip install -e packages/companion-bench` before running p0-smoke.\n"
            f"  ({exc})",
            file=sys.stderr,
        )
        return 2

    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = list(load_scenarios_dir(public_dir, include_held_out=False))
    if family:
        specs = [s for s in specs if s.family.value == family]
    if not specs:
        print(f"error: no scenarios for family {family!r}", file=sys.stderr)
        return 1

    attestation = SubmissionAttestation(
        no_companionbench_derivative_in_training=True,
        no_scenario_specific_prompt=True,
        no_public_test_set_tuning=True,
        cross_user_memory_isolation=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths: dict[str, pathlib.Path] = {}
    smoke_tracks = {**TRACK_TO_SUBMISSION, **COMPONENT_TRACK_TO_SUBMISSION}
    for track, submission_id in smoke_tracks.items():
        manifest = SubmissionManifest(
            submission_id=submission_id,
            system_name=f"smoke-{track}",
            model_identifier=f"smoke-{track}",
            base_url="http://127.0.0.1:0/v1",
            api_key_env="UNUSED_SMOKE_KEY",
            system_prompt="You are a long-running companion AI.",
            generation_config={"temperature": 0.0, "max_tokens": 256},
            attestation=attestation,
            leaderboard_category="bespoke" if track != "raw" else "open-weight",
        )
        track_dir = output_dir / track
        result = dry_run_with_fakes(
            manifest=manifest,
            specs=specs,
            sut_client=EchoFakeSUTClient(),
            user_backend=DeterministicFakeUtteranceClient(),
            paraphrase_seeds=(0,),
            artifact_dir=track_dir / "arcs",
        )
        summary_path = track_dir / "summary.json"
        write_submission_summary(result, summary_path)
        summary_paths[track] = summary_path
        print(f"  [p0] {track:<14} final_mean={result.aggregate.final_mean:.2f} -> {summary_path}")

    comparator = _load_comparator()
    argv: list[str] = []
    for track, path in summary_paths.items():
        argv += ["--track", f"{track}={path}"]
    verdict_path = output_dir / "verdict.json"
    argv += ["--output", str(verdict_path)]
    print("\n  [p0] running comparator (identical fakes => no separation expected)\n")
    rc = comparator.main(argv)
    if rc != 0:
        print("error: comparator failed on p0 summaries", file=sys.stderr)
        return rc
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    valid_states = {
        comparator.STATE_KILL, comparator.STATE_WIRING,
        comparator.STATE_WEAK, comparator.STATE_FIRST_STAGE,
    }
    if verdict["state"] not in valid_states:
        print(f"error: unexpected verdict state {verdict['state']!r}", file=sys.stderr)
        return 1
    print(
        f"\n  [p0] WIRING OK. {len(smoke_tracks)} tracks scored (5 base + "
        f"{len(COMPONENT_TRACK_TO_SUBMISSION)} component arms), verdict emitted "
        f"(state={verdict['state']}). Real signal comes from p1/p2 on a real substrate."
    )
    return 0


# ---------------------------------------------------------------------------
# Phase: judge evidence
# ---------------------------------------------------------------------------


def phase_judge_evidence(*, output_dir: pathlib.Path, execute: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweeps = [
        ("judge_robustness_sweep.py", SCRIPTS / "judge_robustness_sweep.py"),
        ("calibration_sweep.py", SCRIPTS / "calibration_sweep.py"),
    ]
    rc_total = 0
    for label, path in sweeps:
        cmd = [
            sys.executable, str(path),
            "--output-dir", str(output_dir),
        ]
        if not execute:
            cmd.append("--dry-run")
        print(f"[judge-evidence] {label}: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode
        rc_total = rc_total or rc
    print(
        "[judge-evidence] NOTE: the sweep scaffolds are SHADOW (#48 / #71). A real "
        "cross-family judge sweep on Qwen transcripts must be run before any score "
        "is quoted externally; --dry-run records the intended config + placeholders."
    )
    return rc_total


# ---------------------------------------------------------------------------
# Phase: P1 / P2 real scoring + verdict
# ---------------------------------------------------------------------------


def phase_real(*, args: argparse.Namespace, tier: str) -> int:
    scores_dir = args.output_dir / "scores"
    paraphrase_seeds = "0" if tier == "p1" else "0,1,2"
    score_cmd = [
        sys.executable, str(SCORE_CMD),
        "--roster", str(ABLATION_ROSTER),
        "--output-dir", str(scores_dir),
        "--user-sim-base-url", args.user_sim_base_url,
        "--user-sim-model", args.user_sim_model,
        "--user-sim-key-env", args.user_sim_key_env,
        "--perturn-base-url", args.perturn_base_url,
        "--perturn-model", args.perturn_model,
        "--perturn-key-env", args.perturn_key_env,
        "--arc-base-url", args.arc_base_url,
        "--arc-model", args.arc_model,
        "--arc-key-env", args.arc_key_env,
        "--paraphrase-seeds", paraphrase_seeds,
    ]
    if tier == "p2":
        score_cmd += ["--include-heldout", "--require-heldout"]
    if args.parallel_sut > 1:
        score_cmd += ["--parallel-sut", str(args.parallel_sut)]
    if args.per_system_timeout_min is not None:
        score_cmd += ["--per-system-timeout-min", str(args.per_system_timeout_min)]
    if args.per_system_retries > 0:
        score_cmd += ["--per-system-retries", str(args.per_system_retries)]
    if args.resume:
        score_cmd.append("--resume")

    _assert_non_qwen_judges(args)
    if not args.dry_run:
        _assert_real_run_ready(args, tier=tier)

    print(f"[{tier}] score command:\n  {' '.join(score_cmd)}")
    if not args.dry_run:
        rc = subprocess.run(score_cmd, cwd=str(REPO_ROOT), check=False).returncode
        if rc != 0:
            print(f"error: score_reference_systems exited {rc}", file=sys.stderr)
            return rc
        roster_ids = _roster_submission_ids(ABLATION_ROSTER)
        expected_submission_ids = tuple(TRACK_TO_SUBMISSION.values()) + tuple(
            submission_id
            for submission_id in COMPONENT_TRACK_TO_SUBMISSION.values()
            if submission_id in roster_ids
        )
        missing_summaries = tuple(
            submission_id
            for submission_id in expected_submission_ids
            if not (scores_dir / submission_id / "summary.json").is_file()
        )
        if missing_summaries:
            print(
                "error: scoring returned without required summaries: "
                + ", ".join(missing_summaries),
                file=sys.stderr,
            )
            return 1

    # Build the comparator command from the produced per-system summaries.
    # Component arms join automatically once their submission ids appear in
    # the roster (until then they stay out and the verdict reports the
    # component claim as insufficient_data instead of crashing on a missing
    # summary path).
    roster_ids = _roster_submission_ids(ABLATION_ROSTER)
    real_tracks = dict(TRACK_TO_SUBMISSION)
    for track, submission_id in COMPONENT_TRACK_TO_SUBMISSION.items():
        if submission_id in roster_ids:
            real_tracks[track] = submission_id
    comp_cmd = [sys.executable, str(COMPARATOR)]
    for track, submission_id in real_tracks.items():
        summary = scores_dir / submission_id / "summary.json"
        comp_cmd += ["--track", f"{track}={summary}"]
    for track in real_tracks:
        fp = args.output_dir / track / "substrate_fingerprint.json"
        comp_cmd += ["--fingerprint-file", f"{track}={fp}"]
    verdict_path = args.output_dir / f"verdict_{tier}.json"
    comp_cmd += ["--output", str(verdict_path)]
    if tier == "p1":
        # On a single seed + public-only, stability cannot be claimed; relax the
        # arc floor so the verdict reports 'weak-positive' instead of crashing.
        comp_cmd += ["--min-arcs-for-stability", "9999"]

    print(f"\n[{tier}] verdict command:\n  {' '.join(comp_cmd)}")
    if not args.dry_run:
        return subprocess.run(comp_cmd, cwd=str(REPO_ROOT), check=False).returncode
    return 0


def _assert_non_qwen_judges(args: argparse.Namespace) -> None:
    for label, model in (
        ("user-sim", args.user_sim_model),
        ("per-turn judge", args.perturn_model),
        ("arc judge", args.arc_model),
    ):
        if "qwen" in model.lower():
            raise SystemExit(
                f"refusing to run: {label} model {model!r} is Qwen, but the substrate "
                "is Qwen. Judges/user-sim MUST be a different family (#71 / #72)."
            )


def _assert_real_run_ready(args: argparse.Namespace, *, tier: str) -> None:
    if tier != "p1":
        return
    run_manifest = args.output_dir / "run_manifest.json"
    if not run_manifest.is_file():
        raise SystemExit(
            f"P1 readiness manifest missing: {run_manifest}; run the Windows "
            "preflight/serve wrapper first"
        )
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "companion-p1-run-manifest.v1":
        raise SystemExit(f"unsupported P1 run manifest schema: {manifest['schema_version']!r}")
    if not manifest["temporal_bootstrap_sha256"] or not manifest["regime_bootstrap_sha256"]:
        raise SystemExit("P1 run manifest does not identify the trained bootstrap pair")

    scores_dir = args.output_dir / "scores"
    if scores_dir.exists() and any(scores_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"{scores_dir} already contains results; pass --resume or choose a new output dir"
        )
    if manifest.get("serving_topology") != "single-lifeform-ablation-bundle":
        raise SystemExit(
            "P1 run manifest does not declare the single-lifeform ablation topology"
        )
    if tuple(manifest.get("ablation_verticals", ())) != tuple(TRACK_TO_VERTICAL.values()):
        raise SystemExit("P1 run manifest ablation vertical set is not the reviewed roster")

    health_urls = {
        "lifeform-ablation-bundle": "http://127.0.0.1:8000/v1/health",
        "ref-harness": "http://127.0.0.1:8500/healthz",
        "camel": "http://127.0.0.1:8600/healthz",
    }
    failures: list[str] = []
    for label, url in health_urls.items():
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if not 200 <= response.status < 300:
                    failures.append(f"{label}=HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            failures.append(f"{label}={exc}")
    if failures:
        raise SystemExit("P1 endpoints are not healthy: " + "; ".join(failures))

    topology_path = args.output_dir / "serve_topology.json"
    if not topology_path.is_file():
        raise SystemExit(f"P1 serving topology missing: {topology_path}")
    # utf-8-sig: Windows PowerShell 5.1 writers may emit a UTF-8 BOM.
    topology = json.loads(topology_path.read_text(encoding="utf-8-sig"))
    if topology.get("schema_version") != "companion-ablation-serving-topology.v1":
        raise SystemExit("P1 serving topology has an unsupported schema")
    if topology.get("serving_topology") != "single-lifeform-ablation-bundle":
        raise SystemExit("P1 serving topology is not single-lifeform-ablation-bundle")
    if topology.get("process_count") != 3:
        raise SystemExit("P1 serving topology must contain exactly three processes")
    if tuple(topology.get("ablation_verticals", ())) != tuple(TRACK_TO_VERTICAL.values()):
        raise SystemExit("P1 serving topology vertical set is not the reviewed roster")

    probe_failures = _probe_lifeform_verticals()
    if probe_failures:
        raise SystemExit(
            "P1 lifeform vertical route probes failed: " + "; ".join(probe_failures)
        )

    fingerprint_cmd = [
        sys.executable,
        str(SCRIPTS / "assert_same_substrate.py"),
        "--require-weights-sha256",
    ]
    for track in tuple(TRACK_TO_SUBMISSION) + tuple(COMPONENT_TRACK_TO_SUBMISSION):
        path = args.output_dir / track / "substrate_fingerprint.json"
        fingerprint_cmd += ["--fingerprint-file", f"{track}={path}"]
    result = subprocess.run(fingerprint_cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise SystemExit("P1 same-substrate fingerprint gate failed")


def _probe_lifeform_verticals() -> list[str]:
    api_key = os.environ.get("LIFEFORM_LOCAL_API_KEY", "").strip()
    if not api_key:
        return ["LIFEFORM_LOCAL_API_KEY is empty"]
    failures: list[str] = []
    for track, vertical in TRACK_TO_VERTICAL.items():
        body = {
            "model": f"lifeform-{vertical}",
            "messages": [{"role": "user", "content": "p1 readiness probe"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:8000/v1/chat/completions?vertical={vertical}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{track}/{vertical}={exc}")
            continue
        expected = f"lifeform:{vertical}"
        if payload.get("system_fingerprint") != expected:
            failures.append(
                f"{track}/{vertical}=fingerprint {payload.get('system_fingerprint')!r}"
            )
    return failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_comparator():
    spec = importlib.util.spec_from_file_location("compare_companion_ablation", COMPARATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_same_substrate_ablation", description=__doc__)
    p.add_argument(
        "--phase",
        required=True,
        choices=["p0-smoke", "judge-evidence", "p1", "p2"],
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion-ablation" / "local",
    )
    p.add_argument("--family", default=None, help="restrict p0-smoke to one family (F1..F6).")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing (p1/p2).")
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-track summaries in the selected output directory",
    )
    p.add_argument("--execute", action="store_true", help="actually run sweeps in judge-evidence.")
    p.add_argument("--parallel-sut", type=int, default=1)
    p.add_argument(
        "--per-system-timeout-min",
        type=int,
        default=None,
        help=(
            "Per-track subprocess wallclock timeout passed to score_reference_systems "
            "(default: 240 for p1, 480 for p2 when omitted)."
        ),
    )
    p.add_argument(
        "--per-system-retries",
        type=int,
        default=0,
        help="Automatic retries per track on non-zero exit (not on timeout).",
    )
    # Cross-family judge config (defaults are non-Qwen).
    p.add_argument("--user-sim-model", default=DEFAULT_USER_SIM_MODEL)
    p.add_argument("--user-sim-base-url", default=DEFAULT_USER_SIM_BASE_URL)
    p.add_argument("--user-sim-key-env", default=DEFAULT_USER_SIM_KEY_ENV)
    p.add_argument("--perturn-model", default=DEFAULT_PERTURN_MODEL)
    p.add_argument("--perturn-base-url", default=DEFAULT_PERTURN_BASE_URL)
    p.add_argument("--perturn-key-env", default=DEFAULT_PERTURN_KEY_ENV)
    p.add_argument("--arc-model", default=DEFAULT_ARC_MODEL)
    p.add_argument("--arc-base-url", default=DEFAULT_ARC_BASE_URL)
    p.add_argument("--arc-key-env", default=DEFAULT_ARC_KEY_ENV)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.phase == "p0-smoke":
        return phase_p0_smoke(output_dir=args.output_dir, family=args.family)
    if args.phase == "judge-evidence":
        return phase_judge_evidence(output_dir=args.output_dir, execute=args.execute)
    if args.phase in ("p1", "p2"):
        if args.per_system_timeout_min is None:
            args.per_system_timeout_min = 240 if args.phase == "p1" else 480
        return phase_real(args=args, tier=args.phase)
    raise SystemExit(f"unhandled phase {args.phase!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
