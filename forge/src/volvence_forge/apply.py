"""Human-reviewed proposal application and append-only decision ledger."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ForgeConfig
from .foundation import ForgeError, SchemaStore, read_json, sha256_text, utc_now


class ApplyError(ForgeError):
    """Raised when a proposal cannot be safely recorded or applied."""


@dataclass(frozen=True)
class ApplyResult:
    proposal_id: str
    decision: str
    ledger_path: Path
    target: str


def apply_proposal(
    *,
    config: ForgeConfig,
    proposal_dir: Path,
    validation_report_path: Path,
    human_approved_by: str,
) -> ApplyResult:
    reviewer = _reviewer(human_approved_by)
    proposal_dir = proposal_dir.resolve()
    manifesto_path = proposal_dir / "manifesto.json"
    patch_path = proposal_dir / "patch.diff"
    manifesto_text = manifesto_path.read_text(encoding="utf-8")
    patch = patch_path.read_text(encoding="utf-8")
    manifesto = read_json(manifesto_path)
    validation_text = validation_report_path.read_text(encoding="utf-8")
    validation = read_json(validation_report_path)
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    schema_store.validate(manifesto, "proposal_manifesto.schema.json")
    schema_store.validate(validation, "validation_report.schema.json")
    if validation["status"] != "PASS":
        raise ApplyError(f"Proposal {manifesto['proposal_id']} is blocked by validation")
    if validation["proposal_id"] != manifesto["proposal_id"]:
        raise ApplyError("Validation report belongs to a different proposal")
    if validation["patch_sha256"] != sha256_text(patch):
        raise ApplyError("Patch changed after validation")
    if validation["manifesto_sha256"] != sha256_text(manifesto_text):
        raise ApplyError("Manifesto changed after validation")
    target = manifesto["target"]
    if config.editable_entry_for(target) is None:
        raise ApplyError(f"Target is no longer editable: {target}")
    target_path = config.resolve_target(target, must_exist=True)
    if sha256_text(target_path.read_text(encoding="utf-8")) != manifesto["target_preimage_sha256"]:
        raise ApplyError("Target changed after proposal generation; regenerate and revalidate")
    _ensure_not_decided(config.paths.ledger_path, manifesto["proposal_id"])
    _run_git_apply(config.paths.repo_root, patch, check_only=True)
    _run_git_apply(config.paths.repo_root, patch, check_only=False)
    event = _decision_event(
        manifesto=manifesto,
        validation_text=validation_text,
        patch=patch,
        reviewer=reviewer,
        decision="applied",
        reason="human review approved after loop-external validation",
    )
    try:
        _append_ledger(config.paths.ledger_path, event)
    except ApplyError as exc:
        try:
            _run_git_apply(config.paths.repo_root, patch, check_only=False, reverse=True)
        except ApplyError as rollback_exc:
            raise ApplyError(
                f"Ledger append failed and automatic reverse patch also failed: {rollback_exc}"
            ) from exc
        raise ApplyError("Ledger append failed; target patch was automatically reversed") from exc
    return ApplyResult(
        proposal_id=manifesto["proposal_id"],
        decision="applied",
        ledger_path=config.paths.ledger_path,
        target=target,
    )


def reject_proposal(
    *,
    config: ForgeConfig,
    proposal_dir: Path,
    human_approved_by: str,
    reason: str,
) -> ApplyResult:
    reviewer = _reviewer(human_approved_by)
    if not reason.strip():
        raise ApplyError("A rejection reason is required")
    proposal_dir = proposal_dir.resolve()
    manifesto_path = proposal_dir / "manifesto.json"
    patch_path = proposal_dir / "patch.diff"
    manifesto = read_json(manifesto_path)
    patch = patch_path.read_text(encoding="utf-8")
    SchemaStore(config.paths.forge_root / "schemas").validate(manifesto, "proposal_manifesto.schema.json")
    _ensure_not_decided(config.paths.ledger_path, manifesto["proposal_id"])
    event = _decision_event(
        manifesto=manifesto,
        validation_text="",
        patch=patch,
        reviewer=reviewer,
        decision="rejected",
        reason=reason.strip(),
    )
    _append_ledger(config.paths.ledger_path, event)
    return ApplyResult(
        proposal_id=manifesto["proposal_id"],
        decision="rejected",
        ledger_path=config.paths.ledger_path,
        target=manifesto["target"],
    )


def _decision_event(
    *,
    manifesto: dict[str, Any],
    validation_text: str,
    patch: str,
    reviewer: str,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": "forge-ledger.v1",
        "event": "proposal_decision",
        "proposal_id": manifesto["proposal_id"],
        "pattern_id": manifesto["pattern_id"],
        "target": manifesto["target"],
        "decision": decision,
        "reason": reason,
        "reviewer": reviewer,
        "timestamp": utc_now(),
        "patch_sha256": sha256_text(patch),
        "manifesto_sha256": sha256_text(json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True) + "\n"),
        "validation_sha256": sha256_text(validation_text) if validation_text else None,
        "proposal_summary": f"{manifesto['target']}: {manifesto['targeted_fix']}",
    }
    if decision == "applied":
        impact = manifesto["predicted_impact"]
        event["prediction"] = {
            "pattern_id": manifesto["pattern_id"],
            "metric": impact["metric"],
            "baseline_value": impact["baseline_value"],
            "expected_delta": impact["expected_delta"],
            "evaluation_window": impact["evaluation_window"],
        }
    return event


def _append_ledger(path: Path, event: dict[str, Any]) -> None:
    payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    except OSError as exc:
        raise ApplyError(f"Cannot open ledger {path}: {exc}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except OSError as exc:
        raise ApplyError(f"Cannot append ledger {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _ensure_not_decided(path: Path, proposal_id: str) -> None:
    if not path.exists():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ApplyError(f"Invalid ledger JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ApplyError(f"Ledger event must be an object at {path}:{line_number}")
        if event.get("event") == "proposal_decision" and event.get("proposal_id") == proposal_id:
            raise ApplyError(f"Proposal {proposal_id} already has a ledger decision")


def _run_git_apply(repo_root: Path, patch: str, *, check_only: bool, reverse: bool = False) -> None:
    argv = ["git", "apply"]
    if check_only:
        argv.append("--check")
    if reverse:
        argv.append("--reverse")
    argv.extend(("--recount", "-"))
    try:
        completed = subprocess.run(
            tuple(argv),
            cwd=repo_root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ApplyError(f"git executable unavailable: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ApplyError("git apply timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ApplyError(f"{' '.join(argv)} failed: {detail}")


def _reviewer(value: str) -> str:
    reviewer = value.strip()
    if not reviewer:
        raise ApplyError("A named human reviewer is required")
    return reviewer
