#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Cross-platform smoke run entry point (Linux / macOS / Windows).

Mirrors ``run_companion_bench_smoke.sh`` so Windows users (no bash)
can run the same smoke pipeline:

    python scripts/companion_bench/run_companion_bench_smoke.py

What it does:

1. Source ``.local/llm.env`` (simple KEY=VALUE parser; gitignored)
2. Start local VZ SUT (``lifeform-serve --enable-openai-compat``) in
   background, wait for ``/v1/health``
3. Run ``score_reference_systems.py`` against the chosen smoke roster
4. Stop VZ SUT
5. Build ``site/data/`` from artifacts
6. Print aggregate summary

Provider selection (``SMOKE_PROVIDER`` env or ``--provider`` flag):

* ``openrouter`` — cross-vendor via OpenRouter (one key proxies many)
* ``qwen``       — DashScope Qwen-only (uses existing PROTOCOL_LLM_API_KEY)

Refs:
    docs/external/companion-bench-openrouter-setup.md
    docs/moving forward/companion-bench-public-launch-packet.md §3
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
ENV_FILE = REPO_ROOT / ".local" / "llm.env"
SCORE_CMD = REPO_ROOT / "scripts" / "companion_bench" / "score_reference_systems.py"
BUILD_SITE_CMD = REPO_ROOT / "scripts" / "companion_bench" / "build_site.py"


@dataclasses.dataclass(frozen=True)
class ProviderConfig:
    label: str
    roster: pathlib.Path
    user_sim_base_url: str
    user_sim_model: str
    user_sim_key_env: str
    perturn_base_url: str
    perturn_model: str
    perturn_key_env: str
    arc_base_url: str
    arc_model: str
    arc_key_env: str
    required_key_var: str


PROVIDERS: dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        label="openrouter",
        roster=REPO_ROOT / "scripts" / "companion_bench" / "reference_systems.smoke.yaml",
        user_sim_base_url="https://openrouter.ai/api/v1",
        user_sim_model="openai/gpt-5-mini",
        user_sim_key_env="OPENROUTER_API_KEY",
        perturn_base_url="https://openrouter.ai/api/v1",
        perturn_model="openai/gpt-5-mini",
        perturn_key_env="OPENROUTER_API_KEY",
        arc_base_url="https://openrouter.ai/api/v1",
        arc_model="anthropic/claude-3.7-sonnet",
        arc_key_env="OPENROUTER_API_KEY",
        required_key_var="OPENROUTER_API_KEY",
    ),
    "qwen": ProviderConfig(
        label="qwen",
        roster=REPO_ROOT / "scripts" / "companion_bench" / "reference_systems.smoke_qwen.yaml",
        user_sim_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        user_sim_model="qwen3-max",
        user_sim_key_env="PROTOCOL_LLM_API_KEY",
        perturn_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        perturn_model="qwen3-max",
        perturn_key_env="PROTOCOL_LLM_API_KEY",
        arc_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        arc_model="qwen-plus",
        arc_key_env="PROTOCOL_LLM_API_KEY",
        required_key_var="PROTOCOL_LLM_API_KEY",
    ),
}


def _source_env(env_file: pathlib.Path) -> None:
    """Parse a simple KEY=VALUE .env file into ``os.environ``."""
    if not env_file.exists():
        raise SystemExit(f"ERROR: {env_file} not found")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _port_in_use(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_health(port: int, timeout_s: int = 30) -> bool:
    url = f"http://127.0.0.1:{port}/v1/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1.0)
    return False


def _start_vz_sut(
    port: int,
    vertical: str,
    substrate: str,
    log_path: pathlib.Path,
) -> subprocess.Popen[bytes] | None:
    if _port_in_use(port):
        print(
            f"port {port} already in use — assuming an existing VZ SUT is running; "
            "set SKIP_VZ_SUT=1 to suppress this auto-detection",
            file=sys.stderr,
        )
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "wb")
    cmd = [
        "lifeform-serve",
        "--vertical", vertical,
        "--substrate-mode", substrate,
        "--enable-openai-compat",
        "--port", str(port),
    ]
    print(f"Starting VZ SUT: {' '.join(cmd)}")
    print(f"  log → {log_path}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    except FileNotFoundError as exc:
        log_handle.close()
        raise SystemExit(
            "ERROR: 'lifeform-serve' command not found on PATH. "
            "Install with `pip install -e packages/lifeform-service` or set SKIP_VZ_SUT=1."
        ) from exc
    return proc


def _stop_vz_sut(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"Stopping VZ SUT (pid={proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_scoring(args: argparse.Namespace, cfg: ProviderConfig) -> int:
    cmd = [
        sys.executable,
        str(SCORE_CMD),
        "--roster", str(cfg.roster),
        "--output-dir", str(args.output_dir),
        "--user-sim-base-url", cfg.user_sim_base_url,
        "--user-sim-model", cfg.user_sim_model,
        "--user-sim-key-env", cfg.user_sim_key_env,
        "--perturn-base-url", cfg.perturn_base_url,
        "--perturn-model", cfg.perturn_model,
        "--perturn-key-env", cfg.perturn_key_env,
        "--arc-base-url", cfg.arc_base_url,
        "--arc-model", cfg.arc_model,
        "--arc-key-env", cfg.arc_key_env,
        "--paraphrase-seeds", "0",
        "--family", args.family,
    ]
    print("\n[2/4] Running score_reference_systems on family", args.family)
    print("  ", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def _run_build_site(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(BUILD_SITE_CMD),
        "--artifact-dir", str(args.output_dir),
        "--site-dir", str(REPO_ROOT / "site"),
    ]
    print("\n[3/4] Building site/data...")
    print("  ", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def _print_aggregate_summary(output_dir: pathlib.Path) -> None:
    agg_path = output_dir / "aggregate_results.json"
    if not agg_path.exists():
        print(f"WARNING: {agg_path} not found")
        return
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    rows = data.get("systems", [])
    print(f"\n[4/4] Aggregate summary: {len(rows)} systems")
    for row in rows:
        sid = row.get("submission_id", "?")
        summary = row.get("summary", {})
        agg = summary.get("aggregate", {})
        cost = summary.get("cost", {})
        final = agg.get("final_mean")
        arc_count = agg.get("arc_count", 0)
        total_usd = cost.get("total_usd")
        final_str = f"{final:6.2f}" if isinstance(final, (int, float)) else "  n/a"
        cost_str = f"${total_usd:.4f}" if isinstance(total_usd, (int, float)) else "n/a"
        print(f"  {sid:42s}  final={final_str}  arcs={arc_count}  cost={cost_str}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_companion_bench_smoke")
    p.add_argument(
        "--provider",
        choices=tuple(PROVIDERS.keys()),
        default=os.environ.get("SMOKE_PROVIDER", "openrouter"),
        help="LLM provider for SUT/judge/simulator (default: env SMOKE_PROVIDER or openrouter).",
    )
    p.add_argument(
        "--family",
        default=os.environ.get("SMOKE_FAMILY", "F1"),
        help="Scenario family (F1..F6); default F1 (4 scenarios).",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion_bench_smoke",
    )
    p.add_argument("--vz-port", type=int, default=int(os.environ.get("VZ_SUT_PORT", "8000")))
    p.add_argument("--vz-vertical", default=os.environ.get("VZ_SUT_VERTICAL", "companion"))
    p.add_argument(
        "--vz-substrate",
        default=os.environ.get("VZ_SUT_SUBSTRATE", "synthetic"),
    )
    p.add_argument(
        "--skip-vz-sut",
        action="store_true",
        default=os.environ.get("SKIP_VZ_SUT", "0") == "1",
        help="Do not start a local VZ SUT (e.g. SUT already running externally).",
    )
    p.add_argument(
        "--skip-build-site",
        action="store_true",
        default=os.environ.get("SKIP_BUILD_SITE", "0") == "1",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PROVIDERS[args.provider]

    print("=== Companion Bench smoke run ===")
    print(f"Provider:   {cfg.label}")
    print(f"Roster:     {cfg.roster.relative_to(REPO_ROOT)}")
    print(f"Family:     {args.family}")
    print(f"Output:     {args.output_dir.relative_to(REPO_ROOT)}")
    print(f"User-sim:   {cfg.user_sim_model} @ {cfg.user_sim_base_url}")
    print(f"Per-turn:   {cfg.perturn_model} @ {cfg.perturn_base_url}")
    print(f"Arc judge:  {cfg.arc_model} @ {cfg.arc_base_url}")
    print()

    print("[1/4] Sourcing env + checking required keys...")
    _source_env(ENV_FILE)
    if not os.environ.get(cfg.required_key_var, "").strip():
        print(
            f"ERROR: env var {cfg.required_key_var} is empty/missing. "
            f"Add to {ENV_FILE.relative_to(REPO_ROOT)}.",
            file=sys.stderr,
        )
        return 1
    if not os.environ.get("LIFEFORM_LOCAL_API_KEY", "").strip():
        os.environ["LIFEFORM_LOCAL_API_KEY"] = "local-dev-no-auth"
        print("  WARN: LIFEFORM_LOCAL_API_KEY not set; defaulting to 'local-dev-no-auth'")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    proc = None
    if not args.skip_vz_sut:
        log_path = args.output_dir / "vz_sut.log"
        proc = _start_vz_sut(args.vz_port, args.vz_vertical, args.vz_substrate, log_path)
        print(f"  Waiting for /v1/health on port {args.vz_port} (timeout 30s)...")
        if not _wait_health(args.vz_port, timeout_s=30):
            print(
                f"ERROR: VZ SUT did not become healthy within 30s. "
                f"See {log_path} for details.",
                file=sys.stderr,
            )
            _stop_vz_sut(proc)
            return 1
        print("  VZ SUT healthy.")
    else:
        print("[1/4] Skipping VZ SUT start (--skip-vz-sut)")

    try:
        rc = _run_scoring(args, cfg)
        if rc != 0:
            print(f"WARNING: score_reference_systems exited with code {rc}", file=sys.stderr)
    finally:
        _stop_vz_sut(proc)

    if not args.skip_build_site:
        _run_build_site(args)
    else:
        print("\n[3/4] Skipping build_site (--skip-build-site)")

    _print_aggregate_summary(args.output_dir)
    print(f"\nDone. Site preview: open {REPO_ROOT / 'site' / 'index.html'}")
    print(f"Bundle inspection: ls {args.output_dir}/*/arcs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
