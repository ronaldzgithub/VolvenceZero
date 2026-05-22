#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Run Volvence Zero's default companion entry through Companion Bench.

This is the smallest practical "our system as an OpenAI endpoint" path:

1. Load `.local/llm.env`.
2. Start `lifeform-serve --vertical companion --enable-openai-compat`.
3. Run `run_real_submission.py` with `vz_companion_eq_submission.yaml`.

By default the service uses the synthetic substrate so the endpoint is
cheap to validate. Pass `--substrate-mode hf-shared` for the local-real
Qwen-backed track.
"""

from __future__ import annotations

import argparse
import contextlib
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
SUBMISSION = REPO_ROOT / "scripts" / "companion_bench" / "vz_companion_eq_submission.yaml"
RUNNER = REPO_ROOT / "scripts" / "companion_bench" / "run_real_submission.py"


def _source_env(env_file: pathlib.Path) -> None:
    if not env_file.exists():
        raise SystemExit(f"ERROR: {env_file} not found")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _port_in_use(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_health(port: int, timeout_s: int) -> bool:
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


def _start_service(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    if args.skip_service:
        print("[1/3] Skipping service start (--skip-service)")
        return None
    if _port_in_use(args.port):
        print(f"[1/3] Port {args.port} already in use; assuming service is running.")
        return None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "vz_companion_service.log"
    log_handle = open(log_path, "wb")
    cmd = [
        "lifeform-serve",
        "--vertical", "companion",
        "--substrate-mode", args.substrate_mode,
        "--enable-openai-compat",
        "--port", str(args.port),
    ]
    if args.substrate_mode == "hf-shared":
        cmd.extend([
            "--substrate-model-id", args.substrate_model_id,
            "--substrate-device", args.substrate_device,
        ])
        if args.substrate_local_files_only:
            cmd.append("--substrate-local-files-only")

    print("[1/3] Starting VZ companion endpoint:")
    print("  " + " ".join(cmd))
    print(f"  log -> {log_path}")
    try:
        return subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    except FileNotFoundError as exc:
        log_handle.close()
        raise SystemExit(
            "ERROR: 'lifeform-serve' command not found. Install packages/lifeform-service."
        ) from exc


def _stop_service(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"Stopping VZ companion endpoint (pid={proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_bench(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--submission", str(SUBMISSION),
        "--artifact-dir", str(args.output_dir / "vz-companion-eq-default"),
        "--user-sim-base-url", args.judge_base_url,
        "--user-sim-model", args.user_sim_model,
        "--user-sim-key-env", args.judge_key_env,
        "--perturn-base-url", args.judge_base_url,
        "--perturn-model", args.perturn_model,
        "--perturn-key-env", args.judge_key_env,
        "--arc-base-url", args.judge_base_url,
        "--arc-model", args.arc_model,
        "--arc-key-env", args.judge_key_env,
        "--paraphrase-seeds", args.paraphrase_seeds,
        "--family", args.family,
    ]
    print("[2/3] Running Companion Bench:")
    print("  " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vz_companion_eq_smoke")
    parser.add_argument("--family", default=os.environ.get("SMOKE_FAMILY", "F1"))
    parser.add_argument("--paraphrase-seeds", default=os.environ.get("SMOKE_SEEDS", "0"))
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion_bench_vz_companion_eq_smoke",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("VZ_SUT_PORT", "8000")))
    parser.add_argument("--skip-service", action="store_true", default=os.environ.get("SKIP_VZ_SUT", "0") == "1")
    parser.add_argument(
        "--substrate-mode",
        choices=("synthetic", "hf-shared"),
        default=os.environ.get("VZ_SUT_SUBSTRATE", "synthetic"),
    )
    parser.add_argument(
        "--substrate-model-id",
        default=os.environ.get("VZ_SUBSTRATE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--substrate-device", default=os.environ.get("VZ_SUBSTRATE_DEVICE", "auto"))
    parser.add_argument(
        "--substrate-local-files-only",
        action="store_true",
        default=os.environ.get("VZ_SUBSTRATE_LOCAL_FILES_ONLY", "0") == "1",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument("--judge-key-env", default=os.environ.get("QWEN_KEY_ENV", "PROTOCOL_LLM_API_KEY"))
    parser.add_argument("--user-sim-model", default=os.environ.get("USER_SIM_MODEL", "qwen3-max"))
    parser.add_argument("--perturn-model", default=os.environ.get("PERTURN_MODEL", "qwen3-max"))
    parser.add_argument("--arc-model", default=os.environ.get("ARC_MODEL", "qwen-plus"))
    parser.add_argument("--health-timeout-s", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _source_env(ENV_FILE)
    if not os.environ.get("LIFEFORM_LOCAL_API_KEY", "").strip():
        os.environ["LIFEFORM_LOCAL_API_KEY"] = "local-dev-no-auth"
        print("WARN: LIFEFORM_LOCAL_API_KEY not set; defaulting to local-dev-no-auth")
    if not os.environ.get(args.judge_key_env, "").strip():
        print(f"ERROR: required judge key env var {args.judge_key_env} is missing", file=sys.stderr)
        return 1

    proc = _start_service(args)
    try:
        print(f"  Waiting for /v1/health on port {args.port}...")
        if not _wait_health(args.port, timeout_s=args.health_timeout_s):
            print("ERROR: VZ companion endpoint did not become healthy.", file=sys.stderr)
            return 1
        rc = _run_bench(args)
    finally:
        _stop_service(proc)

    print(f"[3/3] Done. Artifacts -> {args.output_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
