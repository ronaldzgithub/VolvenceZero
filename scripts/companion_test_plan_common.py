"""Shared process controls for the two companion evidence test plans.

This module is deliberately limited to execution concerns.  It does not own
experiment variables, readouts, or verdicts; those remain in the respective
preregistration and evidence owners.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterator, Mapping, Sequence, TextIO


MPS_LOCK_SCHEMA_VERSION = "companion-evidence-mps-lock.v1"
SEVEN_DAY_SMOKE_MANIFEST_SCHEMA_VERSION = "seven-day-formal-smoke-gate.v2"
_MPS_LOCK_HELD_ENV = "VZ_COMPANION_MPS_LOCK_HELD"
_MPS_LOCK_PATH_ENV = "VZ_COMPANION_MPS_LOCK_PATH"


@dataclass(frozen=True)
class MPSAvailability:
    torch_version: str
    built: bool
    available: bool
    fallback_disabled: bool


class MPSUnavailableError(RuntimeError):
    """Raised when a plan explicitly requiring MPS cannot use it."""


class MPSLockBusyError(RuntimeError):
    """Raised when another evidence plan already owns the shared MPS lock."""


def inspect_mps() -> MPSAvailability:
    """Inspect and exercise the local torch MPS backend without a CPU fallback."""

    try:
        import torch
    except ImportError as exc:
        raise MPSUnavailableError("torch is required for an MPS evidence run") from exc
    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    if built and available:
        probe = torch.tensor((1.0, 2.0), dtype=torch.float32, device="mps")
        result = float(probe.sum().to("cpu").item())
        torch.mps.synchronize()
        if result != 3.0:
            raise MPSUnavailableError("MPS arithmetic probe returned an invalid result")
    return MPSAvailability(
        torch_version=str(torch.__version__),
        built=built,
        available=available,
        fallback_disabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0",
    )


def require_mps() -> MPSAvailability:
    status = inspect_mps()
    if not status.built or not status.available:
        raise MPSUnavailableError(
            "this test plan requires Apple MPS, but torch reports "
            f"built={status.built}, available={status.available}"
        )
    if not status.fallback_disabled:
        raise MPSUnavailableError(
            "PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0 so the MPS run "
            "cannot silently fall back to CPU"
        )
    return status


def execution_environment(execution_root: Path) -> dict[str, str]:
    """Build a subprocess environment that imports only the selected source root."""

    root = execution_root.resolve()
    source_roots = tuple(
        str(path.resolve())
        for path in sorted((root / "packages").glob("*/src"))
        if path.is_dir()
    )
    if not source_roots:
        raise FileNotFoundError(f"execution root has no workspace wheel sources: {root}")
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        (*source_roots, *((inherited,) if inherited else ()))
    )
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    environment["TRANSFORMERS_VERBOSITY"] = "error"
    return environment


def run_plan_command(
    argv: Sequence[str],
    *,
    execution_root: Path,
    environment: Mapping[str, str],
) -> int:
    """Run a plan step without a shell and return its exact exit status."""

    if not argv:
        raise ValueError("plan command argv must not be empty")
    completed = subprocess.run(
        tuple(argv),
        cwd=execution_root,
        env=dict(environment),
        check=False,
    )
    return int(completed.returncode)


@contextmanager
def exclusive_mps_lock(lock_path: Path, *, plan_id: str) -> Iterator[None]:
    """Prevent the two long-running plans from sharing MPS memory."""

    if not plan_id.strip():
        raise ValueError("MPS lock plan_id must be non-empty")
    target = lock_path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise MPSLockBusyError(
                f"MPS evidence lock is already held at {target}: {owner}"
            ) from exc
        _write_lock_record(handle, plan_id=plan_id, state="held")
        try:
            yield
        finally:
            # The advisory flock is released when this process exits, so a
            # leftover "held" record would misreport a free device as busy.
            _write_lock_record(handle, plan_id=plan_id, state="released")
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_lock_record(handle: TextIO, *, plan_id: str, state: str) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "schema_version": MPS_LOCK_SCHEMA_VERSION,
                "plan_id": plan_id,
                "pid": os.getpid(),
                "python": sys.executable,
                "state": state,
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()


def canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def write_seven_day_smoke_manifest(
    *,
    output_root: Path,
    preregistration: Mapping[str, object],
    campaign: str,
    gate_id: int | None,
    evidence_file: str,
    evidence_sha256: str,
    checks: Mapping[str, bool],
) -> dict[str, object]:
    if not evidence_file or Path(evidence_file).is_absolute() or ".." in Path(evidence_file).parts:
        raise ValueError("smoke evidence_file must stay under output_root")
    if len(evidence_sha256) != 64:
        raise ValueError("smoke evidence_sha256 must be SHA-256")
    payload: dict[str, object] = {
        "schema_version": SEVEN_DAY_SMOKE_MANIFEST_SCHEMA_VERSION,
        "campaign": campaign,
        "gate_id": gate_id,
        "preregistration_sha256": canonical_sha256(preregistration),
        "evidence_file": evidence_file,
        "evidence_sha256": evidence_sha256,
        "checks": dict(checks),
        "passed": bool(checks) and all(checks.values()),
        "formal_claim_allowed": False,
        "production_promotion_authorized": False,
    }
    path = output_root / "smoke_manifest.json"
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"smoke manifest is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(encoded)
    return payload


def validate_seven_day_smoke_manifest(
    *,
    smoke_root: Path,
    preregistration: Mapping[str, object],
    campaign: str,
    gate_id: int | None,
) -> Mapping[str, object]:
    path = smoke_root / "smoke_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("smoke manifest must be an object")
    if (
        payload.get("schema_version")
        != SEVEN_DAY_SMOKE_MANIFEST_SCHEMA_VERSION
        or payload.get("campaign") != campaign
        or payload.get("gate_id") != gate_id
        or payload.get("preregistration_sha256")
        != canonical_sha256(preregistration)
        or payload.get("passed") is not True
        or payload.get("formal_claim_allowed") is not False
        or payload.get("production_promotion_authorized") is not False
    ):
        raise ValueError("smoke manifest contract drift")
    evidence_file = payload.get("evidence_file")
    if not isinstance(evidence_file, str):
        raise ValueError("smoke manifest lacks evidence_file")
    relative = Path(evidence_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("smoke evidence_file escapes smoke_root")
    evidence_path = smoke_root / relative
    if (
        not evidence_path.is_file()
        or hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        != payload.get("evidence_sha256")
    ):
        raise ValueError("smoke evidence artifact digest drift")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("smoke manifest checks are not all true")
    return payload


def guarded_mps_runner_entrypoint(
    main_callable: Callable[[], int],
    *,
    plan_id: str,
    argv: Sequence[str],
) -> int:
    """Acquire the shared lock when a formal runner is invoked directly."""

    requested_device = None
    for index, value in enumerate(argv[:-1]):
        if value == "--device":
            requested_device = argv[index + 1]
            break
    if requested_device != "mps":
        return int(main_callable())
    if os.environ.get(_MPS_LOCK_HELD_ENV) == "1":
        require_mps()
        return int(main_callable())
    lock_path = Path(
        os.environ.get(
            _MPS_LOCK_PATH_ENV,
            "artifacts/.companion-evidence-mps.lock",
        )
    )
    with exclusive_mps_lock(lock_path, plan_id=plan_id):
        require_mps()
        return int(main_callable())


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def mps_payload(status: MPSAvailability) -> dict[str, object]:
    return asdict(status)


__all__ = [
    "MPSAvailability",
    "MPSLockBusyError",
    "MPSUnavailableError",
    "SEVEN_DAY_SMOKE_MANIFEST_SCHEMA_VERSION",
    "canonical_sha256",
    "exclusive_mps_lock",
    "execution_environment",
    "guarded_mps_runner_entrypoint",
    "inspect_mps",
    "mps_payload",
    "print_json",
    "require_mps",
    "run_plan_command",
    "validate_seven_day_smoke_manifest",
    "write_seven_day_smoke_manifest",
]
