#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Hosted, no-GPU formal run of the non-Volvence ablation tracks.

Substrate = a hosted Qwen (DashScope, OpenAI-compatible) — no GPU needed.
Judges + user-simulator = OpenRouter, cross-family non-Qwen (#71/#72).

Tracks (all on the SAME hosted Qwen substrate):
  * raw          — DashScope Qwen directly
  * ref-harness  — companion-ref-harness wrapping the same DashScope Qwen
                   (4 memory components; cross-family LLM memory extractor)
  * camel        — only with --with-camel (requires camel-ai installed)

Reads keys from .local/llm.env (gitignored). Never prints key values.

This is the real-judge counterpart to smoke_no_gpu_baselines.py: it produces an
actual raw-vs-wrapper comparison on a real model, judged by real non-Qwen LLMs.
Volvence / volvence-cold still need a served substrate on a GPU box (P1/P2).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "companion_bench"
ENV_FILE = REPO_ROOT / ".local" / "llm.env"
COMPARATOR = SCRIPTS / "compare_companion_ablation.py"
GUARD = SCRIPTS / "assert_same_substrate.py"

_REF_HARNESS_SHIM = "from companion_ref_harness.cli import main; import sys; sys.exit(main())"

_OPENROUTER_HEADERS = {"HTTP-Referer": "https://volvence.zero", "X-Title": "companion-ablation"}


def _load_env() -> None:
    if not ENV_FILE.exists():
        raise SystemExit(f"error: {ENV_FILE} not found (paste keys there first)")
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip()


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthz(port: int, timeout_s: float = 40.0) -> bool:
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


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=4)


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _openrouter_headers(base_url: str) -> dict[str, str]:
    return dict(_OPENROUTER_HEADERS) if "openrouter.ai" in base_url else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_hosted_ablation")
    parser.add_argument("--family", default="F1", help="scenario family (F1..F6); default F1.")
    parser.add_argument("--max-scenarios", type=int, default=0, help="cap scenarios (0=all); use 1 for a cheap real-wiring check.")
    parser.add_argument("--paraphrase-seeds", default="0")
    parser.add_argument("--with-camel", action="store_true", help="include camel track (needs camel-ai).")
    parser.add_argument(
        "--output-dir", type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion-ablation" / "hosted",
    )
    args = parser.parse_args(argv)

    _load_env()

    from companion_bench.judge_arc import LLMArcJudge
    from companion_bench.judge_perturn import LLMPerTurnJudge
    from companion_bench.spec import load_scenarios_dir
    from companion_bench.submission import (
        SubmissionAttestation, SubmissionManifest, run_submission, write_submission_summary,
    )
    from companion_bench.sut_client import OpenAIChatClient
    from companion_bench.user_simulator import OpenAIUtteranceClient
    import importlib.resources as res

    dash_base = os.environ["DASHSCOPE_BASE_URL"].strip()
    dash_key = os.environ["PROTOCOL_LLM_API_KEY"].strip()
    substrate_model = os.environ["ABLATION_SUBSTRATE_MODEL"].strip()
    or_base = os.environ["OPENROUTER_BASE_URL"].strip()
    or_key = os.environ["OPENROUTER_API_KEY"].strip()
    user_sim_model = os.environ["ABLATION_USER_SIM_MODEL"].strip()
    perturn_model = os.environ["ABLATION_PERTURN_MODEL"].strip()
    arc_model = os.environ["ABLATION_ARC_MODEL"].strip()
    refh_extractor_model = os.environ.get("ABLATION_REFH_EXTRACTOR_MODEL", arc_model).strip()

    for label, m in (("user-sim", user_sim_model), ("perturn", perturn_model), ("arc", arc_model)):
        if "qwen" in m.lower():
            raise SystemExit(f"refusing: {label} model {m!r} is Qwen; substrate is Qwen (#71/#72).")

    seeds = tuple(int(x) for x in args.paraphrase_seeds.split(",") if x.strip())
    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = [s for s in load_scenarios_dir(public_dir, include_held_out=False)
             if s.family.value == args.family]
    if not specs:
        print(f"error: no scenarios for family {args.family!r}", file=sys.stderr)
        return 1
    if args.max_scenarios and args.max_scenarios > 0:
        specs = specs[: args.max_scenarios]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- shared judge + user-sim wiring (OpenRouter, cross-family) ---
    def make_completer(base_url: str, api_key: str, model: str):
        client = OpenAIUtteranceClient(
            base_url=base_url, api_key=api_key, model=model, max_tokens=1024,
            extra_headers=_openrouter_headers(base_url),
        )

        def complete(prompt: str, *, seed: int, system: str = "") -> str:
            return client.complete(system_prompt=system, user_prompt=prompt, temperature=0.0, seed=seed)
        return complete

    user_sim = OpenAIUtteranceClient(
        base_url=or_base, api_key=or_key, model=user_sim_model,
        extra_headers=_openrouter_headers(or_base),
    )
    perturn_judge = LLMPerTurnJudge(client_complete=make_completer(or_base, or_key, perturn_model), model=perturn_model)
    arc_judge = LLMArcJudge(client_complete=make_completer(or_base, or_key, arc_model), model=arc_model)

    attestation = SubmissionAttestation(
        no_companionbench_derivative_in_training=True,
        no_scenario_specific_prompt=True,
        no_public_test_set_tuning=True,
        cross_user_memory_isolation=True,
    )
    system_prompt = (
        "You are a long-running companion AI. Maintain a stable persona across "
        "sessions and reference past conversations when relevant. Never reveal "
        "information the user did not actually say."
    )

    # --- boot ref-harness (+camel) servers pointed at the SAME DashScope Qwen ---
    procs: dict[str, subprocess.Popen] = {}
    track_endpoints: dict[str, tuple[str, str, str]] = {}  # track -> (base_url, key, model)
    track_endpoints["raw"] = (dash_base, dash_key, substrate_model)

    refh_port = _free_port()
    refh_args = [
        "serve", "--port", str(refh_port),
        "--upstream-family", "openai-compat",
        "--upstream-base-url", dash_base,
        "--upstream-model", substrate_model,
        "--upstream-key-env", "PROTOCOL_LLM_API_KEY",
        "--components", "summary,embed,user_model,episodic",
        "--summary-extractor-family", "openai-compat",
        "--summary-extractor-base-url", or_base,
        "--summary-extractor-model", refh_extractor_model,
        "--summary-extractor-key-env", "OPENROUTER_API_KEY",
        "--store-mode", "memory",
    ]
    (out / "ref-harness").mkdir(parents=True, exist_ok=True)
    procs["ref-harness"] = subprocess.Popen(
        [sys.executable, "-c", _REF_HARNESS_SHIM, *refh_args],
        stdout=open(out / "ref-harness" / "server.log", "wb"),
        stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=os.environ.copy(),
    )
    track_endpoints["ref-harness"] = (f"http://127.0.0.1:{refh_port}/v1", "local-no-auth", substrate_model)

    if args.with_camel:
        if importlib.util.find_spec("camel") is None:
            print("error: --with-camel requires camel-ai (pip install camel-ai)", file=sys.stderr)
            for p in procs.values():
                _stop(p)
            return 2
        camel_port = _free_port()
        camel_args = [
            "serve", "--port", str(camel_port), "--backend", "camel",
            "--upstream-base-url", dash_base, "--upstream-model", substrate_model,
            "--upstream-key-env", "PROTOCOL_LLM_API_KEY", "--store-mode", "memory",
        ]
        (out / "camel").mkdir(parents=True, exist_ok=True)
        procs["camel"] = subprocess.Popen(
            [sys.executable, "-c",
             "from companion_camel_baseline.cli import main; import sys; sys.exit(main())", *camel_args],
            stdout=open(out / "camel" / "server.log", "wb"),
            stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=os.environ.copy(),
        )
        track_endpoints["camel"] = (f"http://127.0.0.1:{camel_port}/v1", "local-no-auth", substrate_model)

    summary_paths: dict[str, pathlib.Path] = {}
    rc = 0
    try:
        # Wait for wrapper servers.
        for track in ("ref-harness", "camel"):
            if track in procs:
                port = int(track_endpoints[track][0].rsplit(":", 1)[1].split("/")[0])
                if not _wait_healthz(port):
                    print(f"error: {track} server unhealthy on :{port}; see {out/track/'server.log'}",
                          file=sys.stderr)
                    return 1
                print(f"[hosted] {track} healthy on :{port}")

        print(f"[hosted] substrate={substrate_model} (DashScope) | judges: user-sim={user_sim_model}, "
              f"perturn={perturn_model}, arc={arc_model} | family={args.family} seeds={seeds}")

        for track, (base_url, key, model) in track_endpoints.items():
            print(f"\n[hosted] === running track '{track}' ({base_url}) ===")
            sut = OpenAIChatClient(
                base_url=base_url, api_key=key, model=model,
                extra_headers=_openrouter_headers(base_url),
            )
            manifest = SubmissionManifest(
                submission_id=f"hosted-{track}",
                system_name=f"hosted-{track} ({substrate_model})",
                model_identifier=model,
                base_url=base_url,
                api_key_env="PROTOCOL_LLM_API_KEY",
                system_prompt=system_prompt,
                generation_config={"temperature": 0.0, "max_tokens": 512},
                attestation=attestation,
                leaderboard_category="open-weight" if track == "raw" else "bespoke",
            )
            result = run_submission(
                manifest=manifest, specs=specs, sut_client=sut, user_backend=user_sim,
                perturn_judge=perturn_judge, arc_judge=arc_judge,
                paraphrase_seeds=seeds, artifact_dir=out / track / "arcs",
                user_simulator_model=user_sim_model,
            )
            summary_path = out / track / "summary.json"
            write_submission_summary(result, summary_path)
            summary_paths[track] = summary_path
            (out / track).mkdir(parents=True, exist_ok=True)
            (out / track / "substrate_fingerprint.json").write_text(
                json.dumps({"track": track, "substrate_model_id": substrate_model}), encoding="utf-8")
            print(f"[hosted] {track:<12} final_mean={result.aggregate.final_mean:.2f} "
                  f"arcs={result.aggregate.arc_count} cost_usd={result.cost.total_usd}")
    finally:
        for proc in procs.values():
            _stop(proc)

    # Same-substrate guard (all qwen-turbo).
    guard = _load_module(GUARD, "assert_same_substrate")
    guard.assert_consistent([guard.fingerprint_from_inline(t, substrate_model) for t in summary_paths])
    print(f"\n[hosted] same-substrate guard OK ({substrate_model} across {', '.join(summary_paths)})")

    comparator = _load_module(COMPARATOR, "compare_companion_ablation")
    comp_argv: list[str] = []
    for track, path in summary_paths.items():
        comp_argv += ["--track", f"{track}={path}"]
    comp_argv += ["--min-arcs-for-stability", "9999", "--output", str(out / "verdict.json")]
    print()
    rc = comparator.main(comp_argv)
    print(f"\n[hosted] DONE. artifacts -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
