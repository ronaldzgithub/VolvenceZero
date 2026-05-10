#!/usr/bin/env python3
"""Three-track EmpathyBench / generic-external-harness runner.

Status caveat (2026-05-10):

The leaderboard at https://empathybench.com is **closed-source** —
it advertises results from three published psychometric instruments
(RMET / EQ-26 / IRI) but does not publish a runnable harness, and
its model-submission flow is manual / out-of-band as of this packet.
We cannot exactly reproduce its scoring pipeline without that
harness.

This script is therefore intentionally generic: it spins up the
same three lifeform-service tracks as ``run_eqbench3.py`` and
delegates scoring to whatever ``--harness-command`` the caller
provides. Two practical use cases for it today:

1. **EmotionBench** (CUHK-ARISE/EmotionBench, MIT-licensed,
   GitHub-hosted) — a published, reproducible psychometric eval that
   covers a sibling axis to empathybench.com's instruments.
   ``--harness-command`` points at its CLI; the runner takes care of
   spinning up the three tracks and collecting transcripts.

2. **Manual EmpathyBench submission** — for the closed leaderboard,
   run this script with a ``--harness-command`` that simply records
   transcripts to disk in the upload format empathybench.com
   expects, then upload manually.

Until an open EmpathyBench harness lands (or we publish our own),
this script's primary value is the orchestration scaffolding, not
the scoring itself.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import json
import logging
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "external_bench"
DEFAULT_SUBSTRATE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_LOG = logging.getLogger("run_empathybench")


@dataclasses.dataclass(frozen=True)
class TrackConfig:
    name: str
    vertical: str
    request_mode: str
    description: str


_TRACK_CATALOG: dict[str, TrackConfig] = {
    "companion": TrackConfig(
        name="companion",
        vertical="companion",
        request_mode="lifeform",
        description="Full lifeform pipeline + companion bootstraps.",
    ),
    "companion-cold": TrackConfig(
        name="companion-cold",
        vertical="companion-cold",
        request_mode="lifeform",
        description="Lifeform pipeline without trained bootstraps.",
    ),
    "raw": TrackConfig(
        name="raw",
        vertical="companion",
        request_mode="raw",
        description="Bare substrate via mode=raw passthrough.",
    ),
}


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_health(port: int, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    url = f"http://127.0.0.1:{port}/v1/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1.5)
    raise TimeoutError(
        f"Service on port {port} did not become healthy within "
        f"{timeout_s:.0f}s (last error: {last_error!r})"
    )


@contextlib.contextmanager
def _spawn_lifeform_serve(*, track: TrackConfig, port: int, args: argparse.Namespace):
    cmd = [
        sys.executable,
        "-m",
        "lifeform_service.cli",
        "--vertical",
        track.vertical,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--substrate-mode",
        args.substrate_mode,
        "--substrate-model-id",
        args.substrate_model_id,
        "--substrate-device",
        args.substrate_device,
        "--enable-openai-compat",
        "--max-sessions",
        str(args.max_sessions),
        "--idle-eviction-seconds",
        "0",
        "--log-level",
        args.service_log_level,
    ]
    _LOG.info("[track=%s] spawning: %s", track.name, " ".join(cmd))
    log_path = args.artifact_root / f"empathybench_service_{track.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(  # noqa: S603 - controlled args
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
        )
        try:
            _wait_for_health(port, timeout_s=args.service_boot_timeout_s)
            _LOG.info("[track=%s] service healthy on port %d", track.name, port)
            yield proc
        finally:
            _LOG.info("[track=%s] terminating service (pid=%s)", track.name, proc.pid)
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=30)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                    proc.wait(timeout=10)


def _attestation_block(*, args: argparse.Namespace, track: TrackConfig) -> dict[str, object]:
    return {
        "frozen_substrate": True,
        "no_kernel_modification": True,
        "no_benchmark_text_in_system_prompt": True,
        "no_internal_architecture_terms_in_model_card": True,
        "track": track.name,
        "track_description": track.description,
        "vertical": track.vertical,
        "request_mode": track.request_mode,
        "substrate_model_id": args.substrate_model_id,
        "harness_command": args.harness_command,
        "harness_label": args.harness_label,
        "runner_version": "packet-8/v0.1",
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _run_harness(
    *,
    track: TrackConfig,
    port: int,
    args: argparse.Namespace,
) -> pathlib.Path:
    if track.request_mode == "raw":
        base_url = f"http://127.0.0.1:{port}/v1?mode=raw"
    else:
        base_url = f"http://127.0.0.1:{port}/v1"
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.artifact_root / f"empathybench_{track.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TEST_API_URL"] = base_url
    env["TEST_API_KEY"] = "lifeform-openai-compat-no-auth"
    env["EMPATHYBENCH_TRACK"] = track.name
    env["EMPATHYBENCH_OUTPUT_DIR"] = str(output_dir)
    if args.judge_api_key is not None:
        env["JUDGE_API_KEY"] = args.judge_api_key
    if args.judge_api_url is not None:
        env["JUDGE_API_URL"] = args.judge_api_url

    cmd = shlex.split(args.harness_command)
    if not cmd:
        raise RuntimeError(
            "--harness-command parsed to an empty argv. Pass a non-empty "
            "shell-quoted command string."
        )
    _LOG.info("[track=%s] running harness: %s", track.name, " ".join(shlex.quote(part) for part in cmd))
    log_path = output_dir / "harness.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(  # noqa: S603 - controlled args
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"harness for track={track.name} exited with code "
            f"{result.returncode}; check {log_path}"
        )

    summary_path = output_dir / "summary.json"
    summary = {
        "track": track.name,
        "attestation": _attestation_block(args=args, track=track),
        "harness_output_dir": str(output_dir),
        "rubric_average": None,  # harness-specific; comparator handles None
        "notes": (
            "Generic empathybench-style runner. Rubric extraction is harness-specific; "
            "if your harness emits a 'rubric_average' field somewhere, post-process this "
            "summary to fill it in before running compare_ablation.py."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_empathybench",
        description=(
            "Three-track ablation runner for generic empathy / EQ harnesses. "
            "Use --harness-command to plug in EmotionBench, a future open-source "
            "EmpathyBench harness, or a custom transcript-collection script."
        ),
    )
    p.add_argument("--tracks", default="companion,companion-cold,raw")
    p.add_argument("--substrate-model-id", default=DEFAULT_SUBSTRATE_MODEL_ID)
    p.add_argument("--substrate-mode", default="hf-shared")
    p.add_argument("--substrate-device", default="auto")
    p.add_argument("--max-sessions", type=int, default=64)
    p.add_argument(
        "--harness-command",
        required=True,
        help=(
            "Command line to invoke the scoring harness, as a single "
            "shell-quoted string (parsed via shlex). The runner sets "
            "TEST_API_URL / EMPATHYBENCH_TRACK / EMPATHYBENCH_OUTPUT_DIR in "
            "the subprocess environment. Examples: \n"
            "  --harness-command 'python external/emotionbench/emotionbench_run.py --cycles 1'\n"
            "  --harness-command 'python scripts/external_bench/transcripts_to_disk.py'\n"
            "Why a single string: argparse's nargs handling makes "
            "embedded flags (like --cycles) ambiguous; shlex round-trips "
            "the quoting cleanly."
        ),
    )
    p.add_argument(
        "--harness-label",
        default="generic",
        help=(
            "Free-form label for the harness (recorded in attestation). "
            "Examples: 'emotionbench', 'empathybench-manual-transcripts'."
        ),
    )
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--judge-api-url", default=None)
    p.add_argument("--service-boot-timeout-s", type=float, default=600.0)
    p.add_argument("--service-log-level", default="INFO")
    p.add_argument("--artifact-root", type=pathlib.Path, default=DEFAULT_ARTIFACT_ROOT)
    return p


def _resolve_tracks(spec: str) -> list[TrackConfig]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise SystemExit("--tracks must list at least one track")
    unknown = [n for n in names if n not in _TRACK_CATALOG]
    if unknown:
        raise SystemExit(
            f"Unknown track(s): {unknown}. "
            f"Choose from: {sorted(_TRACK_CATALOG.keys())}"
        )
    return [_TRACK_CATALOG[n] for n in names]


def _verify_python() -> None:
    try:
        import lifeform_openai_compat  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "lifeform_openai_compat is not importable. Install it first:\n"
            "  pip install -e packages/lifeform-openai-compat\n"
            f"Underlying error: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    args.artifact_root = pathlib.Path(args.artifact_root)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    _verify_python()
    tracks = _resolve_tracks(args.tracks)
    _LOG.info(
        "running %d empathybench-style track(s) on substrate=%s, harness_label=%s",
        len(tracks),
        args.substrate_model_id,
        args.harness_label,
    )

    summaries: list[pathlib.Path] = []
    for track in tracks:
        port = _free_port()
        with _spawn_lifeform_serve(track=track, port=port, args=args):
            summary_path = _run_harness(track=track, port=port, args=args)
            summaries.append(summary_path)

    _LOG.info("all tracks complete; summaries:")
    for path in summaries:
        _LOG.info("  %s", path)
    _LOG.info(
        "next: python scripts/external_bench/compare_ablation.py "
        "--summaries %s",
        " ".join(str(p) for p in summaries),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
