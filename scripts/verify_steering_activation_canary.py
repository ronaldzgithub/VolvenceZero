#!/usr/bin/env python3
"""Run one bounded B3 service canary and seal its rollout-chain receipt."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from companion_test_plan_common import exclusive_mps_lock, require_mps
from lifeform_service.steering_activation import (
    build_steering_activation_canary_receipt,
    load_steering_activation_authorization,
    write_steering_activation_canary_receipt,
)
from volvence_zero.steering_contracts import SteeringArtifactBundle


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "steering-b3-activation-canary.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_environment() -> dict[str, str]:
    environment = dict(os.environ)
    sources = tuple(
        str(path.resolve())
        for path in sorted((REPOSITORY_ROOT / "packages").glob("*/src"))
        if path.is_dir()
    )
    inherited = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        (*sources, *((inherited,) if inherited else ()))
    )
    return environment


def _service_command(args: argparse.Namespace) -> tuple[str, ...]:
    command = (
        str(args.python.resolve()),
        "-m",
        "lifeform_service.cli",
        "--vertical",
        "companion",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--substrate-mode",
        "hf-shared",
        "--substrate-local-files-only",
        "--substrate-model-id",
        args.substrate_model_id,
        "--substrate-model-source",
        str(args.substrate_model_source.resolve()),
        "--substrate-device",
        args.substrate_device,
        "--substrate-expected-weights-sha256",
        args.substrate_expected_weights_sha256,
        "--substrate-layer-indices",
        *(str(value) for value in args.substrate_layer_indices),
        "--substrate-activation-width",
        str(args.substrate_activation_width),
        "--substrate-max-length",
        str(args.substrate_max_length),
        "--steering-artifact-bundle",
        str(args.steering_artifact_bundle.resolve()),
        "--steering-promotion-manifest",
        str(args.steering_promotion_manifest.resolve()),
        "--steering-activation-plan",
        str(args.steering_activation_plan.resolve()),
        "--steering-activation-step",
        str(args.steering_activation_step),
    )
    if args.previous_activation_receipt is not None:
        command = (
            *command,
            "--steering-previous-activation-receipt",
            str(args.previous_activation_receipt.resolve()),
        )
    return command


def _health_payload(url: str) -> dict[str, object]:
    try:
        with urlopen(url, timeout=2.0) as response:  # noqa: S310 - localhost only
            if response.status != 200:
                raise RuntimeError(
                    f"B3 canary health returned HTTP {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("B3 canary health returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("B3 canary health root must be an object")
    return payload


def _wait_for_health(
    *,
    process: subprocess.Popen[bytes],
    url: str,
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "health endpoint not yet reachable"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"B3 canary exited before health check: return_code={return_code}"
            )
        try:
            payload = _health_payload(url)
        except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
        else:
            if payload.get("status") == "ok" and payload.get("vertical") == "companion":
                return payload
            last_error = f"unexpected health payload: {payload!r}"
        time.sleep(0.5)
    raise TimeoutError(
        f"B3 canary did not become healthy within {timeout_seconds:.1f}s: "
        f"{last_error}"
    )


def _assert_canary_endpoint_available(*, host: str, port: int) -> None:
    if host != "127.0.0.1":
        raise ValueError("B3 canary host must be exactly 127.0.0.1")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError(
            f"B3 canary endpoint is already occupied: {host}:{port}"
        ) from exc


def _stop_canary(
    process: subprocess.Popen[bytes],
    *,
    require_running: bool = False,
) -> int | None:
    return_code = process.poll()
    if return_code is not None:
        if require_running:
            raise RuntimeError(
                "B3 canary exited before the intentional post-health stop: "
                f"return_code={return_code}"
            )
        return return_code
    process.terminate()
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)
    if process.returncode is None:  # pragma: no cover - subprocess invariant
        raise RuntimeError("B3 canary stop did not produce a return code")
    return process.returncode


def run_canary(args: argparse.Namespace) -> dict[str, object]:
    receipt_path = args.receipt_output.resolve()
    stdout_path = (
        args.stdout_log.resolve()
        if args.stdout_log is not None
        else receipt_path.with_suffix(".stdout.log")
    )
    stderr_path = (
        args.stderr_log.resolve()
        if args.stderr_log is not None
        else receipt_path.with_suffix(".stderr.log")
    )
    existing = tuple(
        path for path in (receipt_path, stdout_path, stderr_path) if path.exists()
    )
    if existing:
        raise ValueError(f"B3 canary outputs already exist: {existing!r}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    bundle_payload = args.steering_artifact_bundle.resolve().read_bytes()
    bundle = SteeringArtifactBundle.from_json(bundle_payload.decode("utf-8"))
    bundle_sha256 = hashlib.sha256(bundle_payload).hexdigest()
    authorization = load_steering_activation_authorization(
        bundle=bundle,
        bundle_sha256=bundle_sha256,
        promotion_manifest=args.steering_promotion_manifest,
        activation_plan=args.steering_activation_plan,
        rollout_step=args.steering_activation_step,
        substrate_model_id=args.substrate_model_id,
        substrate_expected_weights_sha256=(
            args.substrate_expected_weights_sha256
        ),
        substrate_layer_indices=tuple(args.substrate_layer_indices),
        substrate_activation_width=args.substrate_activation_width,
        substrate_max_length=args.substrate_max_length,
        previous_activation_receipt=args.previous_activation_receipt,
    )
    command = _service_command(args)
    health_url = f"http://{args.host}:{args.port}/v1/health"
    environment = _workspace_environment()
    if args.substrate_device == "mps":
        environment["VZ_COMPANION_MPS_LOCK_HELD"] = "1"
        environment["VZ_COMPANION_MPS_LOCK_PATH"] = str(
            args.mps_lock.resolve()
        )
    lock_context = (
        exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID)
        if args.substrate_device == "mps"
        else nullcontext()
    )
    with lock_context:
        if args.substrate_device == "mps":
            require_mps()
        _assert_canary_endpoint_available(host=args.host, port=args.port)
        process: subprocess.Popen[bytes] | None = None
        try:
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                process = subprocess.Popen(  # noqa: S603 - exact argv, no shell
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                health = _wait_for_health(
                    process=process,
                    url=health_url,
                    timeout_seconds=args.startup_timeout,
                )
                service_exit_code = _stop_canary(
                    process,
                    require_running=True,
                )
                if service_exit_code is None:  # pragma: no cover - invariant
                    raise RuntimeError("B3 canary stop result is missing")
            receipt = build_steering_activation_canary_receipt(
                authorization=authorization,
                canary_health=health,
                service_pid=process.pid,
                service_exit_code=service_exit_code,
                service_command=command,
                stdout_log_path=stdout_path,
                stderr_log_path=stderr_path,
            )
            write_steering_activation_canary_receipt(
                path=receipt_path,
                receipt=receipt,
            )
            return receipt
        finally:
            if process is not None:
                _stop_canary(process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering-artifact-bundle", type=Path, required=True)
    parser.add_argument("--steering-promotion-manifest", type=Path, required=True)
    parser.add_argument("--steering-activation-plan", type=Path, required=True)
    parser.add_argument("--steering-activation-step", type=int, required=True)
    parser.add_argument("--previous-activation-receipt", type=Path)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--substrate-model-source", type=Path, required=True)
    parser.add_argument(
        "--substrate-model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--substrate-expected-weights-sha256",
        required=True,
    )
    parser.add_argument(
        "--substrate-layer-indices",
        type=int,
        nargs="+",
        default=(11, 12, 13, 20),
    )
    parser.add_argument("--substrate-activation-width", type=int, default=896)
    parser.add_argument("--substrate-max-length", type=int, default=768)
    parser.add_argument("--substrate-device", default="mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        raise ValueError("B3 canary port must be within 1..65535")
    if args.startup_timeout <= 0.0:
        raise ValueError("B3 canary startup timeout must be positive")
    return args


def main() -> int:
    receipt = run_canary(parse_args())
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
