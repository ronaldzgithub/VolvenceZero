"""Exact-bound command delegation for the local Research Lab portal."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .collector import ResearchLabCollector
from .models import ArtifactRef, ResearchLabItem, ResearchLabSnapshot

_TASK_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_DOMAIN_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_MAX_ACTOR_LENGTH = 160
_MAX_REASON_LENGTH = 2_000
_TASK_KIND = "task"
_REQUEST_KIND = "research request"
_HANDOFF_KIND = "praxist handoff"
_EXTERNAL_DESCRIPTOR_KIND = "external descriptor"
_EXTERNAL_HANDOFF_KIND = "external handoff"
_CANDIDATE_KIND = "candidate"
_VALIDATION_KIND = "validation"
_GATE_KIND = "gate"
_RECEIPT_KIND = "receipt"


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

    supported_actions = (
        "submit_external",
        "review_a0",
        "reconcile",
        "record_external_handoff",
        "import_candidate",
        "authorize_shadow",
        "authorize_active",
        "rollback",
    )

    def __init__(
        self,
        collector: ResearchLabCollector,
        *,
        runner: OwnerCommandRunner,
        external_domain_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> None:
        self.collector = collector
        self.repo_root = collector.repo_root
        self.runner = runner
        roots: dict[str, Path] = {}
        for domain_id, raw_root in (external_domain_roots or {}).items():
            if not _DOMAIN_ID.fullmatch(domain_id):
                raise ValueError(f"invalid external domain id: {domain_id!r}")
            candidate = Path(raw_root).expanduser()
            if candidate.is_symlink():
                raise ValueError(f"external domain root may not be a symlink: {candidate}")
            resolved = candidate.resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError(f"external domain root must be a directory: {resolved}")
            roots[domain_id] = resolved
        self.external_domain_roots = roots
        if not roots:
            self.supported_actions = tuple(
                action for action in type(self).supported_actions if action != "submit_external"
            )

    def submit_external(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "domain_id",
            "descriptor_locator",
            "descriptor_id",
            "descriptor_sha256",
            "actor",
            "reason",
        }
        _require_exact_fields(payload, allowed_fields)
        previous = self._require_current_revision(payload)
        actor = _bounded_text(payload, "actor", _MAX_ACTOR_LENGTH)
        reason = _bounded_text(payload, "reason", _MAX_REASON_LENGTH)
        domain_id = _required_string(payload, "domain_id")
        descriptor_id = _required_string(payload, "descriptor_id")
        descriptor_sha256 = _required_sha256(payload, "descriptor_sha256")
        descriptor_path = self._resolve_external_descriptor(
            domain_id=domain_id,
            locator=_required_string(payload, "descriptor_locator"),
            descriptor_id=descriptor_id,
            descriptor_sha256=descriptor_sha256,
        )
        owner_result = self._run_owner(
            [
                "--repo-root",
                str(self.repo_root),
                "research-submit-external",
                str(descriptor_path),
                "--requested-by",
                actor,
                "--reason",
                reason,
                "--json",
            ]
        )
        owner_payload = _owner_json_object(owner_result.stdout, context="external submission")
        if (
            owner_payload.get("descriptor_id") != descriptor_id
            or owner_payload.get("domain_id") != domain_id
            or not isinstance(owner_payload.get("request_id"), str)
            or not isinstance(owner_payload.get("request_sha256"), str)
        ):
            raise PortalCommandError(
                "owner_transition_mismatch",
                "Forge external submission result does not bind the reviewed descriptor",
                status_code=502,
            )
        current = self.collector.collect()
        matches = [
            item
            for item in current.items
            if any(
                ref.kind == _REQUEST_KIND
                and ref.artifact_id == owner_payload["request_id"]
                and ref.sha256 == owner_payload["request_sha256"]
                for ref in item.bindings
            )
        ]
        if len(matches) != 1:
            raise PortalCommandError(
                "owner_transition_missing",
                "Forge returned success but one exact external Request is not visible",
                status_code=502,
            )
        item = matches[0]
        request_ref = _required_binding(item, _REQUEST_KIND)
        descriptor_ref = ArtifactRef(
            kind=_EXTERNAL_DESCRIPTOR_KIND,
            locator=str(descriptor_path),
            sha256=descriptor_sha256,
            artifact_id=descriptor_id,
        )
        return _command_response(
            action="submit_external",
            task_id=item.task_id,
            outcome="awaiting_a0",
            message="External simulation Request sealed; exact named-human A0 review is required",
            previous=previous,
            current=current,
            binding=request_ref,
            input_bindings=(descriptor_ref,),
        )

    def record_external_handoff(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "task_id",
            "artifact_id",
            "artifact_sha256",
            "actor",
            "reason",
        }
        _require_exact_fields(payload, allowed_fields)
        snapshot, item, request_ref, request_path, actor, reason = self._resolve_command(
            payload,
            action="record_external_handoff",
        )
        if item.research_mode != "external_simulation":
            raise PortalCommandError(
                "wrong_research_mode",
                "external handoff is only available for an external simulation Request",
                status_code=409,
            )
        owner_result = self._run_owner(
            [
                "--repo-root",
                str(self.repo_root),
                "research-handoff-external",
                str(request_path),
                "--recorded-by",
                actor,
                "--reason",
                reason,
                "--json",
            ]
        )
        owner_payload = _owner_json_object(owner_result.stdout, context="external handoff")
        current = self.collector.collect()
        current_item = _required_task(current, item.task_id)
        handoff_ref = _required_binding(current_item, _EXTERNAL_HANDOFF_KIND)
        if (
            owner_payload.get("handoff_id") != handoff_ref.artifact_id
            or owner_payload.get("handoff_sha256") != handoff_ref.sha256
        ):
            raise PortalCommandError(
                "owner_transition_mismatch",
                "Forge external handoff result does not match the visible immutable handoff",
                status_code=502,
            )
        return _command_response(
            action="record_external_handoff",
            task_id=item.task_id,
            outcome="handed_off_for_external_review",
            message="Simulation evidence sealed for Foundry-owned review; no Volvence promotion was created",
            previous=snapshot,
            current=current,
            binding=handoff_ref,
            input_bindings=(request_ref,),
        )

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

    def import_candidate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "task_id",
            "task_artifact_id",
            "task_sha256",
            "handoff_sha256",
            "run_id",
            "actor",
            "reason",
        }
        _require_exact_fields(payload, allowed_fields)
        snapshot, item, _, _ = self._resolve_context(payload, action="import_candidate")
        task_ref, task_path = self._resolve_binding(
            item,
            kind=_TASK_KIND,
            submitted_id=_required_string(payload, "task_artifact_id"),
            submitted_sha256=_required_sha256(payload, "task_sha256"),
        )
        handoff_ref, handoff_path = self._resolve_binding(
            item,
            kind=_HANDOFF_KIND,
            submitted_sha256=_required_sha256(payload, "handoff_sha256"),
        )
        run_id = _required_string(payload, "run_id")
        if item.run is None or item.run.run_id != run_id or item.run.run_dir is None:
            raise PortalCommandError(
                "run_binding_mismatch",
                "submitted run_id does not match the completed Praxist run",
                status_code=409,
            )
        run_dir = _resolve_repo_directory(self.repo_root, item.run.run_dir)
        if handoff_path.parent != run_dir:
            raise PortalCommandError(
                "handoff_run_mismatch",
                "canonical handoff is not located at the exact completed run root",
                status_code=409,
            )

        owner_result = self._run_owner(
            [
                "--repo-root",
                str(self.repo_root),
                "research-import-praxist",
                str(task_path),
                str(handoff_path),
                "--run-dir",
                str(run_dir),
            ]
        )
        current = self.collector.collect()
        current_item = _required_task(current, item.task_id)
        candidate_ref = _required_binding(current_item, _CANDIDATE_KIND)
        if candidate_ref.sha256 == _binding_sha(item, _CANDIDATE_KIND):
            raise PortalCommandError(
                "owner_transition_missing",
                "Forge returned success but no new exact Candidate became visible",
                status_code=502,
            )
        return _command_response(
            action="import_candidate",
            task_id=item.task_id,
            outcome="sealed",
            message=_first_output_line(owner_result.stdout, fallback="Candidate import completed"),
            previous=snapshot,
            current=current,
            binding=candidate_ref,
            input_bindings=(task_ref, handoff_ref),
        )

    def authorize_shadow(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._authorize(payload, action="authorize_shadow", to_wiring="shadow")

    def authorize_active(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._authorize(payload, action="authorize_active", to_wiring="active")

    def rollback(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "task_id",
            "receipt_id",
            "receipt_sha256",
            "actor",
            "reason",
        }
        _require_exact_fields(payload, allowed_fields)
        snapshot, item, actor, reason = self._resolve_context(payload, action="rollback")
        receipt_ref, receipt_path = self._resolve_binding(
            item,
            kind=_RECEIPT_KIND,
            submitted_id=_required_string(payload, "receipt_id"),
            submitted_sha256=_required_sha256(payload, "receipt_sha256"),
        )
        receipt = _read_json_object(receipt_path, context="promotion receipt")
        transition = receipt.get("transition")
        resulting = transition.get("resulting_wiring") if isinstance(transition, Mapping) else None
        if receipt.get("outcome") != "AUTHORIZED" or resulting not in {"shadow", "active"}:
            raise PortalCommandError(
                "receipt_not_rollback_boundary",
                "only an AUTHORIZED SHADOW or ACTIVE receipt can be rolled back",
                status_code=409,
            )
        to_wiring = "disabled" if resulting == "shadow" else "shadow"
        owner_result = self.runner(
            (
                "--repo-root",
                str(self.repo_root),
                "research-rollback",
                str(receipt_path),
                "--to-wiring",
                to_wiring,
                "--authorized-by",
                actor,
                "--reason",
                reason,
            )
        )
        current, new_receipt_ref, new_receipt = self._resolve_new_receipt(
            previous=snapshot,
            task_id=item.task_id,
            previous_receipt_sha256=receipt_ref.sha256,
            owner_result=owner_result,
        )
        if owner_result.returncode != 0:
            detail = _first_output_line(owner_result.stderr, fallback=f"Forge exited with {owner_result.returncode}")
            raise PortalCommandError("owner_command_failed", detail, status_code=409)
        new_transition = new_receipt.get("transition")
        if (
            new_receipt.get("action") != "rollback"
            or not isinstance(new_transition, Mapping)
            or new_transition.get("resulting_wiring") != to_wiring
        ):
            raise PortalCommandError(
                "owner_transition_mismatch",
                "Forge rollback receipt does not match the requested adjacent downgrade",
                status_code=502,
            )
        return _command_response(
            action="rollback",
            task_id=item.task_id,
            outcome="authorized",
            message=_first_output_line(owner_result.stdout, fallback=f"Rollback authorized to {to_wiring}"),
            previous=snapshot,
            current=current,
            binding=new_receipt_ref,
            input_bindings=(receipt_ref,),
        )

    def _authorize(
        self,
        payload: Mapping[str, Any],
        *,
        action: str,
        to_wiring: str,
    ) -> dict[str, Any]:
        allowed_fields = {
            "snapshot_revision",
            "task_id",
            "task_artifact_id",
            "task_sha256",
            "candidate_artifact_id",
            "candidate_sha256",
            "validation_sha256",
            "gate_sha256",
            "previous_receipt_id",
            "previous_receipt_sha256",
            "actor",
            "reason",
        }
        _require_exact_fields(payload, allowed_fields)
        snapshot, item, actor, reason = self._resolve_context(payload, action=action)
        task_ref, task_path = self._resolve_binding(
            item,
            kind=_TASK_KIND,
            submitted_id=_required_string(payload, "task_artifact_id"),
            submitted_sha256=_required_sha256(payload, "task_sha256"),
        )
        candidate_ref, candidate_path = self._resolve_binding(
            item,
            kind=_CANDIDATE_KIND,
            submitted_id=_required_string(payload, "candidate_artifact_id"),
            submitted_sha256=_required_sha256(payload, "candidate_sha256"),
        )
        validation_ref, validation_path = self._resolve_binding(
            item,
            kind=_VALIDATION_KIND,
            submitted_sha256=_required_sha256(payload, "validation_sha256"),
        )
        gate_ref, gate_path = self._resolve_binding(
            item,
            kind=_GATE_KIND,
            submitted_sha256=_required_sha256(payload, "gate_sha256"),
        )
        previous_ref, previous_path = self._resolve_previous_receipt(
            item,
            payload,
            to_wiring=to_wiring,
        )

        arguments = [
            "--repo-root",
            str(self.repo_root),
            "research-authorize",
            str(task_path),
            str(candidate_path),
            str(validation_path),
            str(gate_path),
            "--to-wiring",
            to_wiring,
        ]
        if previous_path is not None:
            arguments.extend(("--previous-receipt", str(previous_path)))
        arguments.extend(("--authorized-by", actor, "--reason", reason))

        owner_result = self.runner(tuple(arguments))
        current, new_receipt_ref, receipt = self._resolve_new_receipt(
            previous=snapshot,
            task_id=item.task_id,
            previous_receipt_sha256=previous_ref.sha256 if previous_ref is not None else None,
            owner_result=owner_result,
        )
        outcome = receipt.get("outcome")
        expected_returncode = 0 if outcome == "AUTHORIZED" else 2 if outcome == "BLOCKED" else None
        if expected_returncode is None or owner_result.returncode != expected_returncode:
            detail = _first_output_line(owner_result.stderr, fallback=f"Forge exited with {owner_result.returncode}")
            raise PortalCommandError("owner_command_failed", detail, status_code=409)
        bindings = receipt.get("bindings")
        transition = receipt.get("transition")
        if (
            receipt.get("candidate_id") != candidate_ref.artifact_id
            or not isinstance(bindings, Mapping)
            or bindings.get("candidate_sha256") != candidate_ref.sha256
            or bindings.get("validation_sha256") != validation_ref.sha256
            or bindings.get("gate_sha256") != gate_ref.sha256
            or bindings.get("previous_receipt_sha256") != (previous_ref.sha256 if previous_ref is not None else None)
            or not isinstance(transition, Mapping)
            or transition.get("requested_wiring") != to_wiring
        ):
            raise PortalCommandError(
                "owner_transition_mismatch",
                "Forge receipt does not bind the exact reviewed authorization inputs",
                status_code=502,
            )
        return _command_response(
            action=action,
            task_id=item.task_id,
            outcome=str(outcome).lower(),
            message=_first_output_line(owner_result.stdout, fallback=f"{to_wiring} authorization completed"),
            previous=snapshot,
            current=current,
            binding=new_receipt_ref,
            input_bindings=tuple(
                ref for ref in (task_ref, candidate_ref, validation_ref, gate_ref, previous_ref) if ref is not None
            ),
        )

    def _resolve_command(
        self,
        payload: Mapping[str, Any],
        *,
        action: str,
    ) -> tuple[ResearchLabSnapshot, ResearchLabItem, ArtifactRef, Path, str, str]:
        snapshot, item, actor, reason = self._resolve_context(payload, action=action)
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

    def _require_current_revision(self, payload: Mapping[str, Any]) -> ResearchLabSnapshot:
        requested_revision = _required_string(payload, "snapshot_revision")
        snapshot = self.collector.collect()
        if snapshot.revision != requested_revision:
            raise PortalCommandError(
                "stale_snapshot",
                "Research Lab snapshot changed; refresh before submitting the external descriptor",
                status_code=409,
            )
        return snapshot

    def _resolve_external_descriptor(
        self,
        *,
        domain_id: str,
        locator: str,
        descriptor_id: str,
        descriptor_sha256: str,
    ) -> Path:
        if not _DOMAIN_ID.fullmatch(domain_id):
            raise PortalCommandError(
                "invalid_domain_id",
                "domain_id has an invalid shape",
                status_code=400,
            )
        root = self.external_domain_roots.get(domain_id)
        if root is None:
            raise PortalCommandError(
                "external_domain_not_registered",
                "external domain root is not registered on this Research Lab server",
                status_code=409,
            )
        relative = PurePosixPath(locator)
        if (
            not locator
            or "\\" in locator
            or relative.is_absolute()
            or "." in relative.parts
            or ".." in relative.parts
        ):
            raise PortalCommandError(
                "unsafe_external_descriptor_locator",
                "descriptor_locator must be a safe path below the registered domain root",
                status_code=400,
            )
        candidate = root / Path(*relative.parts)
        if candidate.is_symlink():
            raise PortalCommandError(
                "unsafe_external_descriptor_locator",
                "external descriptor may not be a symlink",
                status_code=409,
            )
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PortalCommandError(
                "external_descriptor_missing",
                "external descriptor is missing",
                status_code=409,
            ) from exc
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise PortalCommandError(
                "unsafe_external_descriptor_locator",
                "external descriptor must be a regular file below the registered domain root",
                status_code=409,
            )
        if _sha256_file(resolved) != descriptor_sha256:
            raise PortalCommandError(
                "artifact_digest_mismatch",
                "external descriptor bytes do not match descriptor_sha256",
                status_code=409,
            )
        descriptor = _read_json_object(resolved, context="external research descriptor")
        domain = descriptor.get("domain")
        if (
            descriptor.get("schema_version") != "forge-external-research-descriptor.v1"
            or descriptor.get("descriptor_id") != descriptor_id
            or not isinstance(domain, Mapping)
            or domain.get("domain_id") != domain_id
        ):
            raise PortalCommandError(
                "artifact_identity_mismatch",
                "external descriptor identity or domain does not match the submitted binding",
                status_code=409,
            )
        return resolved

    def _resolve_context(
        self,
        payload: Mapping[str, Any],
        *,
        action: str,
    ) -> tuple[ResearchLabSnapshot, ResearchLabItem, str, str]:
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
        return snapshot, item, actor, reason

    def _resolve_binding(
        self,
        item: ResearchLabItem,
        *,
        kind: str,
        submitted_sha256: str,
        submitted_id: str | None = None,
    ) -> tuple[ArtifactRef, Path]:
        ref = _required_binding(item, kind)
        if submitted_id is not None and ref.artifact_id != submitted_id:
            raise PortalCommandError(
                "artifact_identity_mismatch",
                f"submitted artifact id does not match the current {kind}",
                status_code=409,
            )
        if ref.sha256 != submitted_sha256:
            raise PortalCommandError(
                "artifact_digest_mismatch",
                f"submitted artifact SHA-256 does not match the current {kind}",
                status_code=409,
            )
        path = _resolve_repo_artifact(self.repo_root, ref.locator)
        if _sha256_file(path) != ref.sha256:
            raise PortalCommandError(
                "artifact_drift",
                f"{kind} bytes changed after snapshot collection",
                status_code=409,
            )
        return ref, path

    def _resolve_previous_receipt(
        self,
        item: ResearchLabItem,
        payload: Mapping[str, Any],
        *,
        to_wiring: str,
    ) -> tuple[ArtifactRef | None, Path | None]:
        submitted_id = _optional_string(payload, "previous_receipt_id")
        submitted_sha256 = _optional_sha256(payload, "previous_receipt_sha256")
        if (submitted_id is None) is not (submitted_sha256 is None):
            raise PortalCommandError(
                "invalid_request",
                "previous receipt id and SHA-256 must either both be null or both be strings",
                status_code=400,
            )
        ref = next((value for value in item.bindings if value.kind == _RECEIPT_KIND), None)
        eligible = False
        if ref is not None:
            path = _resolve_repo_artifact(self.repo_root, ref.locator)
            receipt = _read_json_object(path, context="previous promotion receipt")
            transition = receipt.get("transition")
            resulting = transition.get("resulting_wiring") if isinstance(transition, Mapping) else None
            eligible = receipt.get("outcome") == "AUTHORIZED" and (
                (to_wiring == "active" and resulting == "shadow") or (to_wiring == "shadow" and resulting == "disabled")
            )
        if not eligible:
            if to_wiring == "active":
                raise PortalCommandError(
                    "previous_receipt_missing",
                    "ACTIVE authorization requires the current AUTHORIZED SHADOW receipt",
                    status_code=409,
                )
            if submitted_id is not None or submitted_sha256 is not None:
                raise PortalCommandError(
                    "unexpected_previous_receipt",
                    "this SHADOW authorization does not accept the submitted previous receipt",
                    status_code=409,
                )
            return None, None
        if ref is None:  # pragma: no cover - narrowed by eligible
            raise AssertionError("eligible previous receipt must have a binding")
        if submitted_id is None or submitted_sha256 is None:
            raise PortalCommandError(
                "previous_receipt_binding_missing",
                "the exact previous receipt id and SHA-256 must be confirmed",
                status_code=400,
            )
        return self._resolve_binding(
            item,
            kind=_RECEIPT_KIND,
            submitted_id=submitted_id,
            submitted_sha256=submitted_sha256,
        )

    def _resolve_new_receipt(
        self,
        *,
        previous: ResearchLabSnapshot,
        task_id: str,
        previous_receipt_sha256: str | None,
        owner_result: OwnerCommandResult,
    ) -> tuple[ResearchLabSnapshot, ArtifactRef, Mapping[str, Any]]:
        current = self.collector.collect()
        current_item = _required_task(current, task_id)
        receipt_ref = _required_binding(current_item, _RECEIPT_KIND)
        if receipt_ref.sha256 == _binding_sha(previous.get_task(task_id), _RECEIPT_KIND):
            if owner_result.returncode != 0:
                detail = _first_output_line(
                    owner_result.stderr,
                    fallback=f"Forge exited with {owner_result.returncode}",
                )
                raise PortalCommandError("owner_command_failed", detail, status_code=409)
            raise PortalCommandError(
                "owner_transition_missing",
                "Forge returned success but no new exact promotion receipt became visible",
                status_code=502,
            )
        receipt_path = _resolve_repo_artifact(self.repo_root, receipt_ref.locator)
        receipt = _read_json_object(receipt_path, context="new promotion receipt")
        bindings = receipt.get("bindings")
        if not isinstance(bindings, Mapping) or bindings.get("previous_receipt_sha256") != previous_receipt_sha256:
            raise PortalCommandError(
                "owner_transition_mismatch",
                "new promotion receipt does not extend the reviewed authorization boundary",
                status_code=502,
            )
        return current, receipt_ref, receipt

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


def _required_binding(item: ResearchLabItem, kind: str) -> ArtifactRef:
    matches = [ref for ref in item.bindings if ref.kind == kind]
    if len(matches) != 1:
        raise PortalCommandError(
            "artifact_binding_missing" if not matches else "artifact_binding_ambiguous",
            f"task must expose exactly one current {kind} binding",
            status_code=409,
        )
    return matches[0]


def _binding_sha(item: ResearchLabItem | None, kind: str) -> str | None:
    if item is None:
        return None
    matches = [ref.sha256 for ref in item.bindings if ref.kind == kind]
    return matches[0] if len(matches) == 1 else None


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise PortalCommandError("invalid_request", f"{field} must be a non-empty string", status_code=400)
    if "\x00" in value:
        raise PortalCommandError("invalid_request", f"{field} contains a forbidden NUL byte", status_code=400)
    return value


def _optional_string(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PortalCommandError(
            "invalid_request",
            f"{field} must be null or a non-empty string",
            status_code=400,
        )
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


def _optional_sha256(payload: Mapping[str, Any], field: str) -> str | None:
    value = _optional_string(payload, field)
    if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PortalCommandError(
            "invalid_request",
            f"{field} must be null or a lowercase SHA-256",
            status_code=400,
        )
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


def _resolve_repo_directory(repo_root: Path, locator: str) -> Path:
    path = Path(locator).expanduser()
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise PortalCommandError(
            "unsafe_run_directory",
            "Praxist run directory escapes the repository root",
            status_code=409,
        )
    if not resolved.is_dir():
        raise PortalCommandError("run_directory_missing", "Praxist run directory is missing", status_code=409)
    return resolved


def _read_json_object(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortalCommandError(
            "artifact_unreadable",
            f"{context} is not readable canonical JSON",
            status_code=409,
        ) from exc
    if not isinstance(value, Mapping):
        raise PortalCommandError(
            "artifact_malformed",
            f"{context} must contain a JSON object",
            status_code=409,
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_output_line(value: str, *, fallback: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1][:1_000] if lines else fallback


def _owner_json_object(value: str, *, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PortalCommandError(
            "owner_response_malformed",
            f"Forge {context} response is not valid JSON",
            status_code=502,
        ) from exc
    if not isinstance(payload, Mapping):
        raise PortalCommandError(
            "owner_response_malformed",
            f"Forge {context} response must be one JSON object",
            status_code=502,
        )
    return payload


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
    input_bindings: Sequence[ArtifactRef] = (),
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
        "input_bindings": [
            {
                "kind": value.kind,
                "artifact_id": value.artifact_id,
                "sha256": value.sha256,
            }
            for value in input_bindings
        ],
    }
