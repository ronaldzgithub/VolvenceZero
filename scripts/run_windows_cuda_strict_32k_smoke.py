#!/usr/bin/env python3
"""Run or offline-validate the pinned Windows/CUDA strict 32K smoke."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import pathlib
import sys
import types


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
_VOLVENCE_NAMESPACE_LOCATIONS = tuple(
    (_REPOSITORY_ROOT / "packages" / wheel / "src" / "volvence_zero").resolve()
    for wheel in ("vz-runtime", "vz-substrate", "vz-contracts")
)
for _namespace_location in _VOLVENCE_NAMESPACE_LOCATIONS:
    if not _namespace_location.is_dir():
        raise RuntimeError(f"required frozen source root is missing: {_namespace_location}")

# Build the namespace explicitly so neither packages/* glob growth nor an
# installed regular ``volvence_zero`` package can shadow the three reviewed
# source roots used by this diagnostic.
_namespace = types.ModuleType("volvence_zero")
_namespace.__path__ = [str(location) for location in _VOLVENCE_NAMESPACE_LOCATIONS]
_namespace.__package__ = "volvence_zero"
_namespace.__spec__ = importlib.machinery.ModuleSpec(
    "volvence_zero",
    loader=None,
    is_package=True,
)
_namespace.__spec__.submodule_search_locations = list(_namespace.__path__)
sys.modules["volvence_zero"] = _namespace

from volvence_zero.offline_evidence import windows_cuda_strict_32k_smoke as _strict_smoke  # noqa: E402

_EXPECTED_STRICT_SMOKE_PATH = (
    _REPOSITORY_ROOT
    / "packages"
    / "vz-runtime"
    / "src"
    / "volvence_zero"
    / "offline_evidence"
    / "windows_cuda_strict_32k_smoke.py"
).resolve()
if pathlib.Path(_strict_smoke.__file__).resolve() != _EXPECTED_STRICT_SMOKE_PATH:
    raise RuntimeError("strict 32K implementation import origin drift")

from volvence_zero.offline_evidence.windows_cuda_strict_32k_smoke import (  # noqa: E402
    run_windows_cuda_strict_32k_smoke,
    validate_windows_cuda_strict_32k_smoke,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create or offline-validate the pinned Qwen2.5-1.5B Windows/CUDA strict 32767+1 engineering smoke")
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--output-dir", type=pathlib.Path, required=True)
    run.add_argument(
        "--outer-attempt-lease-id",
        required=True,
        help=(
            "SHA-256 lease preregistered by the outer host campaign; an "
            "arbitrary value does not create physical-execution evidence"
        ),
    )
    run.add_argument("--protocol", type=pathlib.Path)
    validate = subcommands.add_parser("validate-existing")
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate.add_argument(
        "--outer-attempt-lease-id",
        required=True,
        help="Expected preregistered outer host-campaign lease SHA-256",
    )
    validate.add_argument("--protocol", type=pathlib.Path)
    return parser.parse_args(argv)


def _require_frozen_offline_environment() -> None:
    required = {
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    drifted = tuple(name for name, expected in required.items() if os.environ.get(name) != expected)
    if drifted:
        raise RuntimeError(
            "strict 32K smoke run requires the frozen offline environment: "
            + ", ".join(f"{name}=1" for name in drifted)
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "validate-existing":
        result = validate_windows_cuda_strict_32k_smoke(
            output_dir=args.output_dir,
            expected_outer_attempt_lease_id=args.outer_attempt_lease_id,
            protocol_path=args.protocol,
        )
    elif args.command == "run":
        _require_frozen_offline_environment()
        result = run_windows_cuda_strict_32k_smoke(
            output_dir=args.output_dir,
            outer_attempt_lease_id=args.outer_attempt_lease_id,
            protocol_path=args.protocol,
            progress=lambda message: print(
                f"[strict-32k-smoke] {message}",
                flush=True,
            ),
        )
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "attempt_id": result.attempt_id,
                "outer_attempt_lease_id": result.outer_attempt_lease_id,
                "protocol_id": result.protocol_id,
                "execution_attestation_id": result.execution_attestation_id,
                "passed": result.passed,
                "verdict": result.verdict,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if not result.passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
