#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""No-GPU / no-key minimal run of the NON-Volvence ablation tracks.

This boots the three baseline tracks that do NOT need a GPU — ``raw`` (bare
substrate stand-in), ``ref-harness`` (standard memory wrapper), and ``camel``
(standard agent framework) — as **real HTTP servers** with echo/stub backends,
then drives them with Companion Bench's real HTTP SUT client and deterministic
fake judges. No API keys, no spend, no GPU.

What it actually exercises (beyond the in-process p0-smoke):

* the real aiohttp servers booted via their CLIs on real localhost ports,
* the real ``OpenAIChatClient`` HTTP path including ``metadata.session_id`` /
  ``user_id`` propagation (the channel cross-session memory rides on),
* the ref-harness four-component memory layer + camel agent memory over HTTP,
* summary.json shape -> substrate guard -> verdict comparator.

The ``volvence`` / ``volvence-cold`` tracks are intentionally absent (they need
the in-process substrate + GPU), so the verdict will be ``wiring-ready`` — the
core claims compare against volvence and are therefore ``insufficient_data``.
That is the expected, honest outcome: "everything except Volvence runs".

Usage::

    python scripts/companion_bench/smoke_no_gpu_baselines.py --family F1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "companion_bench"
COMPARATOR = SCRIPTS / "compare_companion_ablation.py"
GUARD = SCRIPTS / "assert_same_substrate.py"

# A subprocess shim that runs a wheel CLI's main() regardless of whether the
# console-script shim is on PATH. argv after the -c string lands in sys.argv[1:].
_REF_HARNESS_SHIM = "from companion_ref_harness.cli import main; import sys; sys.exit(main())"
_CAMEL_SHIM = "from companion_camel_baseline.cli import main; import sys; sys.exit(main())"


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthz(port: int, timeout_s: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}/healthz"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return False


def _spawn(shim: str, args: list[str], log_path: pathlib.Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "wb")
    cmd = [sys.executable, "-c", shim, *args]
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=4)


def _load(path: pathlib.Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke_no_gpu_baselines")
    parser.add_argument("--family", default="F1", help="scenario family (F1..F6); default F1.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion-ablation" / "no_gpu_baselines",
    )
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args(argv)

    try:
        import importlib.resources as res

        from companion_bench.judge_arc import DeterministicFakeArcJudge
        from companion_bench.judge_perturn import DeterministicFakePerTurnJudge
        from companion_bench.spec import load_scenarios_dir
        from companion_bench.submission import (
            SubmissionAttestation,
            SubmissionManifest,
            run_submission,
            write_submission_summary,
        )
        from companion_bench.sut_client import OpenAIChatClient
        from companion_bench.user_simulator import DeterministicFakeUtteranceClient
    except ImportError as exc:
        print(f"error: companion_bench not importable: {exc}", file=sys.stderr)
        return 2

    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = [s for s in load_scenarios_dir(public_dir, include_held_out=False)
             if s.family.value == args.family]
    if not specs:
        print(f"error: no scenarios for family {args.family!r}", file=sys.stderr)
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ports = {"raw": _free_port(), "ref-harness": _free_port(), "camel": _free_port()}

    # CLI args for each baseline server (echo/stub backends; no keys, no GPU).
    boots = {
        "raw": (
            _REF_HARNESS_SHIM,
            ["serve", "--port", str(ports["raw"]), "--upstream-family", "passthrough",
             "--components", "", "--use-stub-summary-extractor", "--store-mode", "memory"],
        ),
        "ref-harness": (
            _REF_HARNESS_SHIM,
            ["serve", "--port", str(ports["ref-harness"]), "--upstream-family", "passthrough",
             "--components", "summary,embed,user_model,episodic",
             "--use-stub-summary-extractor", "--store-mode", "memory"],
        ),
        "camel": (
            _CAMEL_SHIM,
            ["serve", "--port", str(ports["camel"]), "--backend", "echo", "--store-mode", "memory"],
        ),
    }

    procs: dict[str, subprocess.Popen] = {}
    summary_paths: dict[str, pathlib.Path] = {}
    rc = 0
    try:
        for track, (shim, server_args) in boots.items():
            procs[track] = _spawn(shim, server_args, out / track / "server.log")
        print("[no-gpu] waiting for /healthz on all three servers...")
        for track, port in ports.items():
            if not _wait_healthz(port):
                print(f"error: {track} server did not become healthy on :{port}; "
                      f"see {out / track / 'server.log'}", file=sys.stderr)
                return 1
            print(f"  [no-gpu] {track:<12} healthy on :{port}")

        attestation = SubmissionAttestation(
            no_companionbench_derivative_in_training=True,
            no_scenario_specific_prompt=True,
            no_public_test_set_tuning=True,
            cross_user_memory_isolation=True,
        )
        for track, port in ports.items():
            manifest = SubmissionManifest(
                submission_id=f"nogpu-{track}",
                system_name=f"no-gpu-{track}",
                model_identifier="echo-substrate",
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key_env="UNUSED",
                system_prompt="You are a long-running companion AI.",
                generation_config={"temperature": 0.0, "max_tokens": 256},
                attestation=attestation,
                leaderboard_category="open-weight" if track == "raw" else "bespoke",
            )
            sut = OpenAIChatClient(
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="local-no-auth",
                model="echo-substrate",
            )
            result = run_submission(
                manifest=manifest,
                specs=specs,
                sut_client=sut,
                user_backend=DeterministicFakeUtteranceClient(),
                perturn_judge=DeterministicFakePerTurnJudge(),
                arc_judge=DeterministicFakeArcJudge(),
                paraphrase_seeds=(0,),
                artifact_dir=out / track / "arcs",
                user_simulator_model="fake/user-sim",
            )
            summary_path = out / track / "summary.json"
            write_submission_summary(result, summary_path)
            summary_paths[track] = summary_path
            # All three share the same echo "substrate" stand-in.
            (out / track).mkdir(parents=True, exist_ok=True)
            (out / track / "substrate_fingerprint.json").write_text(
                json.dumps({"track": track, "substrate_model_id": "echo-substrate"}),
                encoding="utf-8",
            )
            print(f"  [no-gpu] {track:<12} final_mean={result.aggregate.final_mean:.2f} "
                  f"arcs={result.aggregate.arc_count}")
    finally:
        for proc in procs.values():
            _stop(proc)

    # Same-substrate guard over the three echo fingerprints.
    guard = _load(GUARD, "assert_same_substrate")
    fps = [guard.fingerprint_from_inline(t, "echo-substrate") for t in ports]
    guard.assert_consistent(fps)
    print("\n[no-gpu] same-substrate guard: OK (echo-substrate across raw/ref-harness/camel)")

    # Verdict over the three available tracks (volvence absent -> wiring-ready).
    comparator = _load(COMPARATOR, "compare_companion_ablation")
    comp_argv: list[str] = []
    for track, path in summary_paths.items():
        comp_argv += ["--track", f"{track}={path}"]
    comp_argv += ["--output", str(out / "verdict.json")]
    print()
    rc = comparator.main(comp_argv)

    if not args.keep_artifacts:
        print(f"\n[no-gpu] (artifacts kept at {out}; pass --keep-artifacts to suppress cleanup hint)")
    print(
        "\n[no-gpu] DONE. This validated the non-Volvence tracks end-to-end over real HTTP "
        "with zero GPU/keys. Volvence tracks need a served substrate (P1/P2 on a GPU box)."
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
