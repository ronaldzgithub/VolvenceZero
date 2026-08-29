"""Exact-bound command delegation for the local Research Lab portal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collector import ResearchLabCollector
from .models import ArtifactRef, ResearchLabItem, ResearchLabSnapshot

_TASK_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_MAX_ACTOR_LENGTH = 160
_MAX_REASON_LENGTH = 2_000
_REQUEST_KIND = "research request"


@dataclass(frozen=True, slots=True)
class OwnerCommandResult:
    """Bounded subprocess result without exposing a mutable process object."""

    returncode: int
    stdout: str
    stderr: str


OwnerCommandRunner = Callable[[Sequence[str]], OwnerCommandResult]


class PortalCommandError(RuntimeError):
    """Typed command rejection suitable for a stable local HTTP response."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class SubprocessForgeCommandRunner:
    """Invoke the checked-in Forge CLI through one explicit Python runtime."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        *,
        python_executable: str | os.PathLike[str],
        timeout_seconds: float = 180.0,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.forge_source_root = self.repo_root / "forge" / "src"
        if not self.python_executable.is_file() or not os.access(self.python_executable, os.X_OK):
            raise ValueError(f"Forge Python runtime is not executable: {self.python_executable}")
        if not (self.forge_source_root / "volvence_forge" / "cli.py").is_file():
            raise ValueError(f"Forge source package is missing: {self.forge_source_root}")
        if timeout_seconds <= 0:
            raise ValueError("Forge command timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def __call__(self, arguments: Sequence[str]) -> OwnerCommandResult:
        environment = os.environ.copy()
        existing_python_path = environment.get("PYTHONPATH")
        source_path = str(self.forge_source_root)
        environment["PYTHONPATH"] = (
            f"{source_path}{os.pathsep}{existing_python_path}" if existing_python_path else source_path
        )
        try:
            completed = subprocess.run(
                [
                    str(self.python_executable),
                    "-m",
                    "volvence_forge.cli",
                    *arguments,
                ],
                cwd=self.repo_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise PortalCommandError(
                "owner_command_timeout",
                "Forge command exceeded the bounded execution timeout",
                status_code=504,
            ) from exc
        except OSError as exc:
            raise PortalCommandError(
                "owner_command_unavailable",
                f"Forge command could not be started: {exc}",
                status_code=502,
            ) from exc
        return OwnerCommandResult(completed.returncode, completed.stdout, completed.stderr)


class ResearchLabCommandService:
    """Validate portal commands against a fresh snapshot, then delegate to Forge."""

    supported_actions = ("review_a0", "reconcile")

    def __init__(
        self,
        collector: ResearchLabCollector,
        *,
        runner: OwnerCommandRunner,
    ) -> None:
        self.collector = collector
        self.repo_root = collector.repo_root
        self.runner = runner

    def review_a0(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "task_id",
            "artifact_id",
            "artifact_sha256",
            "actor",
            "reason",
            "decision",
        }
        _require_exact_fields(payload, allowed_fields)
        snapshot, item, request_ref, request_path, actor, reason = self._resolve_command(
            payload,
            action="review_a0",
        )
        decision = _required_string(payload, "decision").lower()
        if decision not in {"approve", "reject"}:
            raise PortalCommandError(
                "invalid_decision",
                "decision must be either approve or reject",
                status_code=400,
            )
        arguments = [
            "--repo-root",
            str(self.repo_root),
            "research-approve",
            str(request_path),
            "--approved-by",
            actor,
            "--reason",
            reason,
        ]
        if decision == "reject":
            arguments.append("--reject")
        owner_result = self._run_owner(arguments)
        current = self.collector.collect()
        current_item = _required_task(current, item.task_id)
        if "review_a0" in current_item.available_actions:
            raise PortalCommandError(
                "owner_transition_missing",
                "Forge returned success but no exact A0 review became visible",
                status_code=502,
            )
        return _command_response(
            action="review_a0",
            task_id=item.task_id,
            outcome="approved" if decision == "approve" else "rejected",
            message=_first_output_line(owner_result.stdout, fallback=f"A0 {decision} completed"),
            previous=snapshot,
            current=current,
            binding=request_ref,
        )

    def reconcile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "task_id",
            "artifact_id",
            "artifact_sha256",
            "actor",
            "reason",
        }
        _require_exact_fields(payload, allowed_fields)
        snapshot, item, request_ref, request_path, _, _ = self._resolve_command(
            payload,
            action="reconcile",
        )
        owner_result = self._run_owner(
            [
                "--repo-root",
                str(self.repo_root),
                "research-reconcile",
                "--once",
                "--request",
                str(request_path),
                "--json",
            ]
        )
        current = self.collector.collect()
        _required_task(current, item.task_id)
        return _command_response(
            action="reconcile",
            task_id=item.task_id,
            outcome="reconciled",
            message=_reconcile_message(owner_result.stdout),
            previous=snapshot,
            current=current,
            binding=request_ref,
        )

    def _resolve_command(
        self,
        payload: Mapping[str, Any],
        *,
        action: str,
    ) -> tuple[ResearchLabSnapshot, ResearchLabItem, ArtifactRef, Path, str, str]:
        requested_revision = _required_string(payload, "snapshot_revision")
        task_id = _required_string(payload, "task_id")
        if not _TASK_ID.fullmatch(task_id):
            raise PortalCommandError("invalid_task_id", "task_id has an invalid shape", status_code=400)
        actor = _bounded_text(payload, "actor", _MAX_ACTOR_LENGTH)
        reason = _bounded_text(payload, "reason", _MAX_REASON_LENGTH)

        snapshot = self.collector.collect()
        if snapshot.revision != requested_revision:
            raise PortalCommandError(
                "stale_snapshot",
                "Research Lab snapshot changed; refresh and review the exact artifacts again",
                status_code=409,
            )
        item = _required_task(snapshot, task_id)
        if action not in item.available_actions:
            raise PortalCommandError(
                "action_not_available",
                f"{action} is not available while task is {item.lifecycle.stage.value}",
                status_code=409,
            )
        request_ref = next((ref for ref in item.bindings if ref.kind == _REQUEST_KIND), None)
        if request_ref is None or request_ref.artifact_id is None:
            raise PortalCommandError(
                "request_binding_missing",
                "the current task has no exact ResearchRequest binding",
                status_code=409,
            )
        if _required_string(payload, "artifact_id") != request_ref.artifact_id:
            raise PortalCommandError(
                "artifact_identity_mismatch",
                "submitted artifact_id does not match the current ResearchRequest",
                status_code=409,
            )
        if _required_sha256(payload, "artifact_sha256") != request_ref.sha256:
            raise PortalCommandError(
                "artifact_digest_mismatch",
                "submitted artifact SHA-256 does not match the current ResearchRequest",
                status_code=409,
            )
        request_path = _resolve_repo_artifact(self.repo_root, request_ref.locator)
        if _sha256_file(request_path) != request_ref.sha256:
            raise PortalCommandError(
                "artifact_drift",
                "ResearchRequest bytes changed after snapshot collection",
                status_code=409,
            )
        return snapshot, item, request_ref, request_path, actor, reason

    def _run_owner(self, arguments: Sequence[str]) -> OwnerCommandResult:
        result = self.runner(tuple(arguments))
        if result.returncode != 0:
            detail = _first_output_line(result.stderr, fallback=f"Forge exited with {result.returncode}")
            raise PortalCommandError("owner_command_failed", detail, status_code=409)
        return result


def _required_task(snapshot: ResearchLabSnapshot, task_id: str) -> ResearchLabItem:
    item = snapshot.get_task(task_id)
    if item is None:
        raise PortalCommandError("task_not_found", "task is not present in the current snapshot", status_code=404)
    return item


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise PortalCommandError("invalid_request", f"{field} must be a non-empty string", status_code=400)
    if "\x00" in value:
        raise PortalCommandError("invalid_request", f"{field} contains a forbidden NUL byte", status_code=400)
    return value


def _bounded_text(payload: Mapping[str, Any], field: str, maximum: int) -> str:
    value = _required_string(payload, field).strip()
    if not value:
        raise PortalCommandError("invalid_request", f"{field} may not be blank", status_code=400)
    if len(value) > maximum:
        raise PortalCommandError("invalid_request", f"{field} exceeds {maximum} characters", status_code=400)
    return value


def _required_sha256(payload: Mapping[str, Any], field: str) -> str:
    value = _required_string(payload, field)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PortalCommandError("invalid_request", f"{field} must be a lowercase SHA-256", status_code=400)
    return value


def _require_exact_fields(payload: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append(f"missing={','.join(missing)}")
        if unexpected:
            detail.append(f"unexpected={','.join(unexpected)}")
        raise PortalCommandError("invalid_request", "; ".join(detail), status_code=400)


def _resolve_repo_artifact(repo_root: Path, locator: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute():
        raise PortalCommandError(
            "unsafe_artifact_locator",
            "command artifacts must be repository-relative",
            status_code=409,
        )
    resolved = (repo_root / relative).resolve()
    if not resolved.is_relative_to(repo_root):
        raise PortalCommandError(
            "unsafe_artifact_locator",
            "command artifact escapes the repository root",
            status_code=409,
        )
    if not resolved.is_file():
        raise PortalCommandError("artifact_missing", "command artifact is missing", status_code=409)
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_output_line(value: str, *, fallback: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1][:1_000] if lines else fallback


def _reconcile_message(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return _first_output_line(stdout, fallback="Forge reconciliation completed")
    if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        state = payload[0].get("state")
        run_id = payload[0].get("run_id")
        if isinstance(state, str):
            return f"Forge reconciliation state: {state}; run_id={run_id or '-'}"
    return "Forge reconciliation completed"


def _command_response(
    *,
    action: str,
    task_id: str,
    outcome: str,
    message: str,
    previous: ResearchLabSnapshot,
    current: ResearchLabSnapshot,
    binding: ArtifactRef,
) -> dict[str, Any]:
    return {
        "schema_version": "volvence-research-lab-command-result.v1",
        "action": action,
        "task_id": task_id,
        "outcome": outcome,
        "message": message,
        "previous_revision": previous.revision,
        "current_revision": current.revision,
        "binding": {
            "kind": binding.kind,
            "artifact_id": binding.artifact_id,
            "sha256": binding.sha256,
        },
    }
