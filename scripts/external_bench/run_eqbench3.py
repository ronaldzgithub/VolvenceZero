#!/usr/bin/env python3
"""Three-track EQ-Bench 3 ablation runner.

Boots a ``lifeform-serve`` instance for each ablation track and
runs ``external/eqbench3/eqbench3.py`` against it, capturing the
per-track results into ``artifacts/external_bench/`` for downstream
``compare_ablation.py`` analysis (Packet 7).

Usage:

    python scripts/external_bench/run_eqbench3.py \\
        --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --tracks companion,companion-cold,raw \\
        --judge-model anthropic/claude-3.7-sonnet \\
        --no-elo

What this script does NOT do:

* Train / fine-tune any model. Substrate is frozen at all times.
* Modify any vz-* or lifeform-* package. The only seam used is the
  ``--enable-openai-compat`` CLI flag added in Packet 5.
* Embed any EQ-Bench scenario text in the system prompt. The
  attestation block in each track's output JSON declares this.

Why a Python wrapper instead of a single shell script:

Per-track lifecycle (boot service → wait healthy → run harness →
collect → tear down) is easier to manage with subprocess + asyncio
than with shell + traps, and the attestation / artifact-naming
plumbing is tighter as code.
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
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "external_bench"
DEFAULT_HARNESS_DIR = REPO_ROOT / "external" / "eqbench3"
DEFAULT_SUBSTRATE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_VERTICAL_PER_TRACK: dict[str, str] = {
    "companion": "companion",
    "companion-cold": "companion-cold",
    "raw": "companion",
}

_LOG = logging.getLogger("run_eqbench3")


# ---------------------------------------------------------------------------
# Track config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrackConfig:
    """One row in the ablation matrix."""

    name: str
    vertical: str
    request_mode: str  # "lifeform" or "raw"
    description: str

    def model_id(self, substrate_model_id: str) -> str:
        """Public-facing OpenAI ``model`` field. Echoed in artifacts."""
        suffix = substrate_model_id.replace("/", "-")
        return f"lifeform-{self.name}@{suffix}"


_TRACK_CATALOG: dict[str, TrackConfig] = {
    "companion": TrackConfig(
        name="companion",
        vertical="companion",
        request_mode="lifeform",
        description=(
            "Full lifeform pipeline (PromptPlanner + ResponseSynthesizer + "
            "memory + regime + adaptive controllers) with the trained "
            "companion bootstraps loaded."
        ),
    ),
    "companion-cold": TrackConfig(
        name="companion-cold",
        vertical="companion-cold",
        request_mode="lifeform",
        description=(
            "Same lifeform pipeline as 'companion' but without the trained "
            "regime / temporal bootstraps. Isolates the score contribution "
            "of the trained artifacts."
        ),
    ),
    "raw": TrackConfig(
        name="raw",
        vertical="companion",  # vertical irrelevant for raw mode
        request_mode="raw",
        description=(
            "Bypass the lifeform; call runtime.generate(...) directly via "
            "the OpenAI-compat router's mode=raw path. Bare-Qwen baseline."
        ),
    ),
}


def _attestation_block(*, args: argparse.Namespace, track: TrackConfig) -> dict[str, object]:
    """Embedded in every track's output JSON; verifier reads this."""
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
        "judge_model": args.judge_model,
        "with_elo": args.with_elo,
        "iterations": args.iterations,
        "threads": args.threads,
        "harness": "EQ-bench/eqbench3",
        "harness_dir": str(args.harness_dir),
        "runner_version": "packet-6/v0.1",
        "git_sha": _git_sha_or_none(),
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _git_sha_or_none() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind a socket to port 0 and read back the OS-assigned port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for_health(port: int, *, timeout_s: float = 600.0) -> None:
    """Poll /v1/health until the service responds 200 (or time out)."""
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
def _spawn_lifeform_serve(
    *,
    track: TrackConfig,
    port: int,
    args: argparse.Namespace,
):
    """Run ``lifeform-serve`` in a subprocess; tear down on exit."""

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
        "0",  # disable; the runner controls lifecycle
        "--log-level",
        args.service_log_level,
    ]
    _LOG.info("[track=%s] spawning: %s", track.name, " ".join(cmd))
    log_path = args.artifact_root / f"service_{track.name}.log"
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


# ---------------------------------------------------------------------------
# Harness invocation
# ---------------------------------------------------------------------------


def _run_eqbench3_harness(
    *,
    track: TrackConfig,
    port: int,
    args: argparse.Namespace,
) -> pathlib.Path:
    """Run upstream eqbench3.py against the running service for ``track``.

    Returns the path of the runs JSON the harness produced (we store
    it under ``artifacts/external_bench/`` to keep all our outputs
    co-located).
    """

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_path = args.artifact_root / f"eqbench3_{track.name}_{timestamp}.runs.json"
    elo_path = args.artifact_root / f"eqbench3_{track.name}_{timestamp}.elo.json"

    if track.request_mode == "raw":
        # The OpenAI-compat router supports query-string mode selection,
        # but eqbench3 does not pass through query params on every call.
        # Use the per-track header default by exposing a track-local
        # "base_url" that routes via header. We achieve this by
        # pointing TEST_API_URL at our adapter's chat endpoint with a
        # dedicated path-to-mode mapping the runner sets via env.
        # Practically: we set OPENAI_COMPAT_MODE so the server picks
        # it up via X-Compat-Mode middleware. eqbench3 supports
        # passing arbitrary headers via the OpenAI client, but its
        # CLI does not surface that — we therefore use the
        # ``mode=raw`` query param baked into the URL (the router
        # accepts both header and query param).
        base_url = f"http://127.0.0.1:{port}/v1?mode=raw"
    else:
        base_url = f"http://127.0.0.1:{port}/v1"

    test_model = track.model_id(args.substrate_model_id)
    cmd = [
        sys.executable,
        str(args.harness_dir / "eqbench3.py"),
        "--test-model",
        test_model,
        "--model-name",
        f"vz-{track.name}-{timestamp}",
        "--judge-model",
        args.judge_model,
        "--runs-file",
        str(runs_path),
        "--elo-results-file",
        str(elo_path),
        "--threads",
        str(args.threads),
        "--iterations",
        str(args.iterations),
        "--ignore-canonical",
    ]
    if not args.with_elo:
        cmd.append("--no-elo")

    env = os.environ.copy()
    env["TEST_API_URL"] = base_url
    env["TEST_API_KEY"] = "lifeform-openai-compat-no-auth"
    env["JUDGE_API_KEY"] = args.judge_api_key or env.get("JUDGE_API_KEY", "")
    env["JUDGE_API_URL"] = args.judge_api_url or env.get(
        "JUDGE_API_URL", "https://api.anthropic.com/v1/"
    )

    if not env.get("JUDGE_API_KEY"):
        raise RuntimeError(
            "JUDGE_API_KEY is required (set --judge-api-key or export "
            "JUDGE_API_KEY before running). The eqbench3 judge cannot "
            "score transcripts without it."
        )

    _LOG.info("[track=%s] running harness: %s", track.name, " ".join(cmd))
    harness_log = args.artifact_root / f"harness_{track.name}_{timestamp}.log"
    with harness_log.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(  # noqa: S603 - controlled args
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=args.harness_dir,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"eqbench3 harness for track={track.name} exited with "
            f"code {result.returncode}; check {harness_log}"
        )
    return runs_path


def _emit_track_summary(
    *,
    track: TrackConfig,
    runs_path: pathlib.Path,
    args: argparse.Namespace,
) -> pathlib.Path:
    """Distill the upstream runs JSON + attestation into a summary file."""

    summary_path = runs_path.with_suffix(".summary.json")
    summary: dict[str, object] = {
        "track": track.name,
        "attestation": _attestation_block(args=args, track=track),
        "runs_file": str(runs_path),
    }

    # Attempt to read the upstream run file and pull the rubric score
    # if present. eqbench3's runs file is dict[run_key] -> per-task
    # results; the average rubric score is sometimes stored at
    # ``results.rubric_average``. We do not fail the run if the format
    # changed upstream; we just record an empty score block and leave
    # the diagnostic details for ``compare_ablation.py`` to recover.
    rubric_average: float | None = None
    try:
        with runs_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        rubric_average = _try_extract_rubric_average(raw)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _LOG.warning(
            "[track=%s] could not parse %s: %s", track.name, runs_path, exc
        )
    summary["rubric_average"] = rubric_average

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _LOG.info("[track=%s] summary → %s", track.name, summary_path)
    return summary_path


def _try_extract_rubric_average(runs_json: object) -> float | None:
    """Best-effort rubric-score extraction from upstream runs JSON.

    The upstream eqbench3 runs file format is not strictly versioned;
    we walk a few known shapes and return the first hit. The sole
    consumer is ``compare_ablation.py``, which itself re-derives
    scores when needed and only uses this number as a quick eyeball.
    """

    if not isinstance(runs_json, dict):
        return None
    # Shape A: top-level dict[run_key] -> {"results": {"rubric_average": float}}
    for value in runs_json.values():
        if isinstance(value, dict):
            results = value.get("results")
            if isinstance(results, dict):
                rubric = results.get("rubric_average")
                if isinstance(rubric, (int, float)):
                    return float(rubric)
    # Shape B: top-level "rubric_average"
    rubric = runs_json.get("rubric_average")
    if isinstance(rubric, (int, float)):
        return float(rubric)
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eqbench3",
        description="Run EQ-Bench 3 across the three Volvence Zero ablation tracks.",
    )
    p.add_argument(
        "--tracks",
        default="companion,companion-cold,raw",
        help=(
            "Comma-separated list of tracks to run. Choices: "
            f"{sorted(_TRACK_CATALOG.keys())}. Default: all three."
        ),
    )
    p.add_argument(
        "--substrate-model-id",
        default=DEFAULT_SUBSTRATE_MODEL_ID,
        help=f"HF model id for the shared runtime (default: {DEFAULT_SUBSTRATE_MODEL_ID}).",
    )
    p.add_argument("--substrate-mode", default="hf-shared")
    p.add_argument("--substrate-device", default="auto")
    p.add_argument("--max-sessions", type=int, default=64)
    p.add_argument(
        "--harness-dir",
        type=pathlib.Path,
        default=DEFAULT_HARNESS_DIR,
        help=(
            f"Path to the eqbench3 checkout (default: {DEFAULT_HARNESS_DIR}). "
            "Clone EQ-bench/eqbench3 into this directory before running."
        ),
    )
    p.add_argument(
        "--judge-model",
        default="anthropic/claude-3.7-sonnet",
        help="Judge model identifier for eqbench3 (default: claude-3.7-sonnet).",
    )
    p.add_argument(
        "--judge-api-key",
        default=None,
        help="Judge API key (default: read JUDGE_API_KEY env var).",
    )
    p.add_argument(
        "--judge-api-url",
        default=None,
        help="Judge API base URL (default: read JUDGE_API_URL env var or anthropic).",
    )
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "eqbench3 --threads. Keep at 1 for serial-inference safety; "
            "the lifeform service serialises generation on the asyncio "
            "loop and concurrent requests just queue."
        ),
    )
    p.add_argument(
        "--with-elo",
        action="store_true",
        help=(
            "Include ELO pass (more expensive judge cost). Default off; "
            "Packet 7 verdict gates this on."
        ),
    )
    p.add_argument(
        "--no-elo",
        dest="with_elo",
        action="store_false",
        help="Explicitly disable ELO pass (default).",
    )
    p.set_defaults(with_elo=False)
    p.add_argument(
        "--service-boot-timeout-s",
        type=float,
        default=600.0,
        help=(
            "How long to wait for /v1/health to come up after spawning "
            "lifeform-serve. The Qwen 1.5B load takes 1-2 min on a fresh "
            "cache; 600s default leaves headroom for slower disks."
        ),
    )
    p.add_argument("--service-log-level", default="INFO")
    p.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"Where to drop runs / summaries (default: {DEFAULT_ARTIFACT_ROOT}).",
    )
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


def _verify_harness_exists(harness_dir: pathlib.Path) -> None:
    if not harness_dir.exists():
        raise SystemExit(
            f"eqbench3 harness not found at {harness_dir}. Clone it:\n"
            f"  git clone https://github.com/EQ-bench/eqbench3.git {harness_dir}\n"
            f"Then install its requirements and copy "
            f"scripts/external_bench/.env.example to {harness_dir}/.env"
        )
    entry = harness_dir / "eqbench3.py"
    if not entry.exists():
        raise SystemExit(f"Expected {entry} to exist; harness checkout looks broken.")


def _verify_python() -> None:
    """Make sure the lifeform-openai-compat wheel is importable."""
    if shutil.which("python") is None and shutil.which("python3") is None:
        raise SystemExit("python interpreter not found on PATH")
    try:
        import lifeform_openai_compat  # noqa: F401  - import-only check
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
    args.harness_dir = pathlib.Path(args.harness_dir)
    _verify_harness_exists(args.harness_dir)
    _verify_python()

    tracks = _resolve_tracks(args.tracks)
    _LOG.info(
        "running %d track(s) on substrate=%s",
        len(tracks),
        args.substrate_model_id,
    )

    summaries: list[pathlib.Path] = []
    for track in tracks:
        port = _free_port()
        _LOG.info("[track=%s] selected port %d", track.name, port)
        with _spawn_lifeform_serve(track=track, port=port, args=args):
            runs_path = _run_eqbench3_harness(
                track=track, port=port, args=args
            )
            summaries.append(
                _emit_track_summary(track=track, runs_path=runs_path, args=args)
            )
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
