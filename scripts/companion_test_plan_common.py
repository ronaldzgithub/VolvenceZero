"""Shared process controls for the two companion evidence test plans.

This module is deliberately limited to execution concerns.  It does not own
experiment variables, readouts, or verdicts; those remain in the respective
preregistration and evidence owners.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator, Mapping, Sequence, TextIO


MPS_LOCK_SCHEMA_VERSION = "companion-evidence-mps-lock.v1"


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


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def mps_payload(status: MPSAvailability) -> dict[str, object]:
    return asdict(status)


__all__ = [
    "MPSAvailability",
    "MPSLockBusyError",
    "MPSUnavailableError",
    "exclusive_mps_lock",
    "execution_environment",
    "inspect_mps",
    "mps_payload",
    "print_json",
    "require_mps",
    "run_plan_command",
]
