"""Persistent, approval-gated control plane for Praxist research runs.

The module owns only development-plane lifecycle artifacts.  Praxist remains
the run/process owner, while Volvence's formal validator, ModificationGate,
and target adapters remain the only production-promotion path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import yaml

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    SchemaStore,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
    utc_now,
)
from .research_promotion import validate_research_task

SCHEMA_NAME = "research_control.schema.json"
EXTERNAL_SCHEMA_NAME = "external_research.schema.json"
FOUNDRY_HANDOFF_SCHEMA_NAME = "foundry_research_handoff.schema.json"

_REQUEST_VERSION = "forge-research-request.v1"
_EXTERNAL_DESCRIPTOR_VERSION = "forge-external-research-descriptor.v1"
_EXTERNAL_REQUEST_VERSION = "forge-external-research-request.v1"
_EXTERNAL_HANDOFF_VERSION = "forge-foundry-research-handoff.v1"
_FOUNDRY_SEAM_VERSION = "foundry-research-lab-seam.v1"
_APPROVAL_VERSION = "forge-research-approval.v1"
_SUPERSESSION_VERSION = "forge-research-request-supersession.v1"
_DIRECTIVE_VERSION = "forge-research-control-directive.v1"
_EVENT_VERSION = "forge-research-control-event.v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_TERMINAL_STATES = frozenset(
    {"RUN_COMPLETED", "RUN_FAILED", "BLOCKED", "PAUSED", "CANCELLED"}
)
_LIVE_STATES = frozenset({"running", "starting", "status_inconsistent"})
_FAILED_STATES = frozenset({"failed", "stopped", "stale"})

_CODEX_NATIVE_UNSET_ENV = frozenset(
    {
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "OPENAI_BASE_URL",
        "PRAXIST_CODEX_BIN",
        "MODEL",
        "PRAXIST_MODEL",
    }
)

_SKIPPED_MANIFEST_DIR_NAMES = frozenset({".git", "__pycache__", ".pytest_cache"})
_SKIPPED_MANIFEST_TOP_LEVEL_DIRS = frozenset(
    {
        "data",
        "datasets",
        "experiments",
        "experiments_tracking",
        "frontier",
        "logs",
        "research_memory",
        "results",
        "runs",
        "outputs",
        "shared_findings",
        "variants",
    }
)
_SKIPPED_MANIFEST_RELATIVE_DIRS = frozenset({("docs", "praxist_reports")})


class ResearchControlError(ForgeError):
    """Raised when a research-control artifact or lifecycle boundary is unsafe."""


class ResearchRequestBindingDriftError(ResearchControlError):
    """Raised when a frozen Request binding no longer matches its exact input."""

    def __init__(self, binding: str, message: str) -> None:
        super().__init__(message)
        self.binding = binding


@dataclass(frozen=True)
class ResearchRequestResult:
    request_id: str
    request_path: Path


@dataclass(frozen=True)
class ResearchApprovalResult:
    approval_id: str
    approval_path: Path
    decision: str


@dataclass(frozen=True)
class ResearchSupersessionResult:
    supersession_id: str
    supersession_path: Path
    replacement_request_id: str
    replacement_request_path: Path
    reused: bool


@dataclass(frozen=True)
class ResearchDirectiveResult:
    directive_id: str
    directive_path: Path
    action: str


@dataclass(frozen=True)
class ExternalResearchHandoffResult:
    handoff_id: str
    handoff_path: Path
    result_path: Path


@dataclass(frozen=True)
class ResearchControlStatus:
    request_id: str
    task_id: str
    state: str
    request_path: Path
    approval_path: Path | None
    latest_event_path: Path | None
    run_id: str | None
    run_dir: str | None
    monitor_command: str | None
    latest_event_sha256: str | None = None
    supersession_path: Path | None = None
    replacement_request_id: str | None = None
    replacement_request_path: Path | None = None


@dataclass(frozen=True)
class CommandExecution:
    """Bounded subprocess result used by the real and fake Praxist runners."""

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class CapacityObservation:
    execution: CommandExecution
    active_rows: tuple[dict[str, Any], ...]
    details: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not self.blockers


class PraxistCommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecution: ...


@dataclass(frozen=True)
class SubprocessPraxistRunner:
    """Invoke Praxist without a shell or persisting provider credentials."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecution:
        command = tuple(str(value) for value in argv)
        environment = None
        if "--codex-native" in command:
            environment = os.environ.copy()
            for name in _CODEX_NATIVE_UNSET_ENV:
                environment.pop(name, None)
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandExecution(
                argv=command,
                returncode=None,
                stdout=_timeout_text(exc.stdout),
                stderr=_timeout_text(exc.stderr),
                timed_out=True,
            )
        except OSError as exc:
            raise ResearchControlError(
                f"cannot execute Praxist command {command[1]!r}: {exc}"
            ) from exc
        return CommandExecution(
            argv=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def submit_research_request(
    *,
    config: ForgeConfig,
    task_manifest_path: Path,
    task_project_path: Path,
    praxist_executable: Path,
    run_dir: Path,
    requested_by: str,
    reason: str,
    trigger_kind: str = "human",
    evidence_paths: Sequence[Path] = (),
    config_file: Path | None = None,
    agent_system: str | None = None,
    runtime: str | None = None,
    codex_native: bool = False,
    model_provider: str | None = None,
    model: str | None = None,
    strategy: str = "auto",
    cohort: int | None = None,
    generations: int | None = None,
    startup_timeout_seconds: int = 30,
) -> ResearchRequestResult:
    """Seal one exact, non-authorizing request for a future Praxist start."""

    submitter = _nonempty(requested_by, "research submission requires a named submitter")
    rationale = _nonempty(reason, "research submission requires a non-empty reason")
    if trigger_kind not in {"human", "forge_failure_pattern", "typed_signal"}:
        raise ResearchControlError(f"unsupported research trigger kind: {trigger_kind!r}")
    if codex_native:
        agent_system = agent_system or "codex_sdk"
        runtime = runtime or "agent_runtime:codex_sdk"
        model_provider = model_provider or "model_provider:openai_compatible"
    _validate_launch_profile_values(
        agent_system=agent_system,
        runtime=runtime,
        codex_native=codex_native,
        model_provider=model_provider,
        model=model,
        strategy=strategy,
        cohort=cohort,
        generations=generations,
        startup_timeout_seconds=startup_timeout_seconds,
    )

    task = validate_research_task(config=config, task_path=task_manifest_path)
    task_path = _resolve_regular_file(task_manifest_path, context="research task manifest")
    task_project = _snapshot_task_project(task_project_path)
    if task_project["task_id"] != task["praxist"]["task_project_id"]:
        raise ResearchControlError(
            "Praxist task project task_id does not match the Volvence research Task"
        )
    if task_project["manifest_sha256"] != task["praxist"]["task_project_manifest_sha256"]:
        raise ResearchControlError(
            "Praxist task project manifest digest does not match the Volvence research Task"
        )

    executable = _resolve_executable(praxist_executable)
    source_checkout = _snapshot_praxist_source_checkout(executable)
    normalized_run_dir = _normalize_fresh_run_dir(run_dir)
    if source_checkout is not None:
        source_root = Path(source_checkout["root"])
        if normalized_run_dir.is_relative_to(source_root):
            raise ResearchControlError("Praxist run output may not be inside its source checkout")
        if Path(task_project["root"]) == source_root:
            raise ResearchControlError("the Praxist source checkout is not a task project")

    config_ref = (
        _absolute_content_ref(config_file, context="Praxist config file")
        if config_file
        else None
    )
    evidence = [
        _content_ref(config, path, context=f"research trigger evidence {index}")
        for index, path in enumerate(evidence_paths)
    ]
    payload: dict[str, Any] = {
        "schema_version": _REQUEST_VERSION,
        "task_id": task["task_id"],
        "claim_id": task["claim_id"],
        "owner": task["owner"],
        "trigger": {
            "kind": trigger_kind,
            "submitted_by": submitter,
            "rationale": rationale,
            "evidence": evidence,
        },
        "bindings": {
            "research_task": _content_ref(
                config,
                task_path,
                context="research task manifest",
            ),
            "task_project": task_project,
            "praxist": {
                "executable": _absolute_content_ref(
                    executable,
                    context="Praxist executable",
                ),
                "source_checkout": source_checkout,
            },
        },
        "launch": {
            "run_dir": str(normalized_run_dir),
            "run_id": normalized_run_dir.name,
            "daemonize": True,
            "status_scope": "targeted_run_id",
            "profile": {
                "config_file": config_ref,
                "agent_system": _optional_text(agent_system),
                "runtime": _optional_text(runtime),
                "codex_native": codex_native,
                "model_provider": _optional_text(model_provider),
                "model": _optional_text(model),
                "strategy": strategy,
                "cohort": cohort,
                "generations": generations,
                "startup_timeout_seconds": startup_timeout_seconds,
            },
        },
        "authority": {
            "human_research_approval_required": True,
            "research_start_authorized": False,
            "production_promotion_authorized": False,
            "formal_validation_required_after_run": True,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    payload["request_id"] = _artifact_id("research-request", payload, "request_id")
    _validate_schema(config, payload, _REQUEST_VERSION)
    digest = str(payload["request_id"]).partition(":")[2]
    destination = (
        config.paths.artifacts_root
        / "research_control"
        / str(task["task_id"])
        / digest
        / "request.json"
    )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=payload,
        expected_version=_REQUEST_VERSION,
        identity_field="request_id",
    )
    return ResearchRequestResult(
        request_id=str(payload["request_id"]),
        request_path=destination,
    )


def validate_external_research_descriptor(
    *,
    config: ForgeConfig,
    descriptor_path: Path,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    """Validate one external-domain envelope without treating it as a Volvence Task."""

    descriptor, resolved = _load_external_descriptor(config, descriptor_path)
    if verify_bindings:
        _resolve_external_descriptor_bindings(
            config,
            descriptor,
            descriptor_path=resolved,
        )
    return descriptor


def submit_external_research_request(
    *,
    config: ForgeConfig,
    descriptor_path: Path,
    requested_by: str,
    reason: str,
) -> ResearchRequestResult:
    """Map one exact Foundry Intent onto the shared approval-gated lifecycle."""

    submitter = _nonempty(requested_by, "external research submission requires a named submitter")
    rationale = _nonempty(reason, "external research submission requires a non-empty reason")
    descriptor, resolved_descriptor = _load_external_descriptor(config, descriptor_path)
    resolved = _resolve_external_descriptor_bindings(
        config,
        descriptor,
        descriptor_path=resolved_descriptor,
    )

    project = resolved["task_project"]
    executable = resolved["executable"]
    source_checkout = _snapshot_praxist_source_checkout(executable)
    normalized_run_dir = _normalize_fresh_run_dir(resolved["run_dir"])
    if source_checkout is not None:
        source_root = Path(source_checkout["root"])
        if normalized_run_dir.is_relative_to(source_root):
            raise ResearchControlError("Praxist run output may not be inside its source checkout")
        if Path(project["root"]) == source_root:
            raise ResearchControlError("the Praxist source checkout is not a task project")

    intent = resolved["intent"]
    domain = descriptor["domain"]
    external_task_id = _external_control_task_id(domain)
    profile = {
        "config_file": resolved["config_file"],
        **intent["launch"],
    }
    payload: dict[str, Any] = {
        "schema_version": _EXTERNAL_REQUEST_VERSION,
        "task_id": external_task_id,
        "claim_id": f"foundry:{intent['intent_id']}",
        "owner": "external-domain:foundry",
        "objective": intent["objective"],
        "external_domain": {
            "descriptor_id": descriptor["descriptor_id"],
            "adapter_id": descriptor["adapter"]["adapter_id"],
            "intent_schema_version": descriptor["adapter"]["intent_schema_version"],
            "domain_id": domain["domain_id"],
            "task_id": domain["task_id"],
            "intent_id": domain["intent_id"],
            "ownership": _foundry_ownership(),
            "result_policy": descriptor["result_policy"],
        },
        "trigger": {
            "kind": "external_domain_descriptor",
            "submitted_by": submitter,
            "rationale": rationale,
            "evidence": [
                {"locator": ref["locator"], "sha256": ref["sha256"]}
                for ref in resolved["evidence"]
            ],
        },
        "bindings": {
            "external_descriptor": _absolute_content_ref(
                resolved_descriptor,
                context="external research descriptor",
            ),
            "external_intent": resolved["intent_ref"],
            "external_budget": resolved["intent_ref"],
            "external_evidence": resolved["evidence"],
            "task_project": project,
            "praxist": {
                "executable": _absolute_content_ref(
                    executable,
                    context="Praxist executable",
                ),
                "source_checkout": source_checkout,
            },
        },
        "launch": {
            "run_dir": str(normalized_run_dir),
            "run_id": normalized_run_dir.name,
            "daemonize": True,
            "status_scope": "targeted_run_id",
            "profile": profile,
        },
        "authority": {
            "human_research_approval_required": True,
            "research_start_authorized": False,
            "external_domain_adoption_required": True,
            "external_domain_human_apply_required": True,
            "result_evidence_class": "simulation",
            "external_actions_allowed": False,
            "foundry_checkout_write_allowed": False,
            "foundry_ledger_write_allowed": False,
            "direct_apply_allowed": False,
            "productzero_start_allowed": False,
            "production_promotion_authorized": False,
            "modification_gate_applicable": False,
            "runtime_wiring_applicable": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    payload["request_id"] = _artifact_id("research-request", payload, "request_id")
    _validate_schema(config, payload, _EXTERNAL_REQUEST_VERSION)
    digest = str(payload["request_id"]).partition(":")[2]
    destination = (
        config.paths.artifacts_root
        / "research_control"
        / external_task_id
        / digest
        / "request.json"
    )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=payload,
        expected_version=_EXTERNAL_REQUEST_VERSION,
        identity_field="request_id",
    )
    return ResearchRequestResult(
        request_id=str(payload["request_id"]),
        request_path=destination,
    )


def record_external_research_handoff(
    *,
    config: ForgeConfig,
    request_path: Path,
    recorded_by: str,
    reason: str,
) -> ExternalResearchHandoffResult:
    """Seal completed simulation evidence for Foundry-owned review and adoption."""

    recorder = _nonempty(recorded_by, "external handoff requires a named recorder")
    rationale = _nonempty(reason, "external handoff requires a non-empty reason")
    request, resolved_request, request_sha256 = _load_request(config, request_path)
    if request["schema_version"] != _EXTERNAL_REQUEST_VERSION:
        raise ResearchControlError("only an external-domain Request can produce an external handoff")
    _verify_request_bindings(config, request)
    approval_record = _find_approval(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    )
    if approval_record is None or approval_record[0]["decision"] != "APPROVE":
        raise ResearchControlError("external handoff requires the exact approved A0 Request")
    approval, _, approval_sha256 = approval_record
    events = _load_events(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
        approval=approval,
        approval_sha256=approval_sha256,
    )
    if not events or events[-1][0]["state"] != "RUN_COMPLETED":
        raise ResearchControlError("external handoff requires an exact RUN_COMPLETED terminal event")
    terminal_event, _, terminal_sha256 = events[-1]
    run = terminal_event["run"]
    if run is None:
        raise ResearchControlError("RUN_COMPLETED terminal event must bind the completed run")

    result_locator = request["external_domain"]["result_policy"]["result_locator"]
    result_relative = _safe_relative_locator(result_locator, context="external result locator")
    run_dir = _resolve_directory(Path(request["launch"]["run_dir"]), context="Praxist run directory")
    result_path = _resolve_regular_file(
        run_dir / Path(*result_relative.parts),
        context="external simulation result",
    )
    if not result_path.is_relative_to(run_dir):
        raise ResearchControlError("external simulation result escapes the exact run directory")

    external = request["external_domain"]
    descriptor_ref = request["bindings"]["external_descriptor"]
    result_ref = _absolute_content_ref(
        result_path,
        context="external simulation result",
    )
    contract_schema_ref = _absolute_content_ref(
        config.paths.forge_root / "schemas" / FOUNDRY_HANDOFF_SCHEMA_NAME,
        context="Foundry research handoff schema",
    )
    hash_chain: dict[str, Any] = {
        "request": {
            "artifact_id": request["request_id"],
            "sha256": request_sha256,
        },
        "approval": {
            "artifact_id": approval["approval_id"],
            "sha256": approval_sha256,
        },
        "run_completion": {
            "artifact_id": terminal_event["event_id"],
            "sha256": terminal_sha256,
        },
        "result": result_ref,
    }
    hash_chain["chain_sha256"] = sha256_text(canonical_json(hash_chain))
    handoff: dict[str, Any] = {
        "schema_version": _EXTERNAL_HANDOFF_VERSION,
        "contract": {
            "contract_version": _FOUNDRY_SEAM_VERSION,
            "schema": contract_schema_ref,
            "producer_owner": "volvence_forge.research_control",
            "consumer_domain": "foundry",
        },
        "descriptor": {
            "descriptor_id": external["descriptor_id"],
            "artifact": descriptor_ref,
        },
        "request": {
            "request_id": request["request_id"],
            "sha256": request_sha256,
        },
        "approval": {
            "approval_id": approval["approval_id"],
            "sha256": approval_sha256,
            "scope": approval["scope"],
            "decision": approval["decision"],
            "reviewed_by": approval["review"]["reviewed_by"],
        },
        "terminal_event": {
            "event_id": terminal_event["event_id"],
            "sha256": terminal_sha256,
            "state": terminal_event["state"],
        },
        "external_domain": {
            "adapter_id": external["adapter_id"],
            "domain": {
                "domain_id": external["domain_id"],
                "task_id": external["task_id"],
                "intent_id": external["intent_id"],
            },
            "ownership": external["ownership"],
        },
        "run": {
            "run_id": run["run_id"],
            "run_dir": run["run_dir"],
        },
        "result": {
            "artifact": result_ref,
            "evidence_class": "simulation",
            "adoption_mode": "proposal_only",
            "market_validation_claimed": False,
            "adoption_status": "pending_external_human_review",
        },
        "review": {
            "recorded_by": recorder,
            "reason": rationale,
        },
        "hash_chain": hash_chain,
        "consumer_permissions": _foundry_consumer_permissions(),
        "authority": _foundry_handoff_authority(),
        "created_at": utc_now(),
    }
    handoff["handoff_id"] = _artifact_id(
        "external-research-handoff",
        handoff,
        "handoff_id",
    )
    _validate_schema(
        config,
        handoff,
        _EXTERNAL_HANDOFF_VERSION,
        schema_name=FOUNDRY_HANDOFF_SCHEMA_NAME,
    )
    handoff_digest = str(handoff["handoff_id"]).partition(":")[2]
    destination = resolved_request.parent / "handoffs" / f"{handoff_digest}.json"
    existing = sorted((resolved_request.parent / "handoffs").glob("*.json"))
    if existing and destination not in existing:
        raise ResearchControlError(
            f"external Request already has an immutable handoff: {existing[0]}"
        )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=handoff,
        expected_version=_EXTERNAL_HANDOFF_VERSION,
        identity_field="handoff_id",
        schema_name=FOUNDRY_HANDOFF_SCHEMA_NAME,
    )
    return ExternalResearchHandoffResult(
        handoff_id=str(handoff["handoff_id"]),
        handoff_path=destination,
        result_path=result_path,
    )


def validate_research_request(
    *,
    config: ForgeConfig,
    request_path: Path,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    """Validate request identity and, by default, every approved input byte."""

    request, _, _ = _load_request(config, request_path)
    if verify_bindings:
        _verify_request_bindings(config, request)
    return request


def supersede_unreviewed_research_request(
    *,
    config: ForgeConfig,
    request_path: Path,
    replacement_request_path: Path,
    superseded_by: str,
    reason: str,
) -> ResearchSupersessionResult:
    """Retire one pre-A0 Request after exact Praxist source-checkout drift.

    The old Request remains immutable and auditable.  Its replacement is a new
    exact Request and therefore requires a new, separate A0 decision.
    """

    actor = _nonempty(
        superseded_by,
        "research Request supersession requires a named lifecycle actor",
    )
    rationale = _nonempty(
        reason,
        "research Request supersession requires a non-empty reason",
    )
    old_request, resolved_old, old_sha256 = _load_request(config, request_path)
    replacement, resolved_replacement, _ = _load_request(
        config,
        replacement_request_path,
    )
    lock_path = resolved_old.parent / ".supersession.lock"
    with _exclusive_lock(lock_path):
        existing = _find_supersession(
            config=config,
            request=old_request,
            request_path=resolved_old,
            request_sha256=old_sha256,
        )
        if existing is not None:
            artifact, path, _, _ = existing
            if (
                artifact["replacement_request"]["artifact_id"]
                != replacement["request_id"]
                or artifact["replacement_request"]["artifact"]
                != _content_ref(
                    config,
                    resolved_replacement,
                    context="replacement research Request",
                )
                or artifact["record"]["superseded_by"] != actor
                or artifact["record"]["reason"] != rationale
            ):
                raise ResearchControlError(
                    "research Request already has a different immutable supersession"
                )
            return ResearchSupersessionResult(
                supersession_id=str(artifact["supersession_id"]),
                supersession_path=path,
                replacement_request_id=str(replacement["request_id"]),
                replacement_request_path=resolved_replacement,
                reused=True,
            )

        _assert_unreviewed_request_can_be_superseded(
            config=config,
            request=old_request,
            request_path=resolved_old,
            request_sha256=old_sha256,
        )
        _verify_request_bindings(config, replacement)
        old_source, replacement_source = _assert_source_refresh_pair(
            old_request=old_request,
            replacement_request=replacement,
        )
        supersession: dict[str, Any] = {
            "schema_version": _SUPERSESSION_VERSION,
            "superseded_request": {
                "artifact_id": old_request["request_id"],
                "artifact": _content_ref(
                    config,
                    resolved_old,
                    context="superseded research Request",
                ),
            },
            "replacement_request": {
                "artifact_id": replacement["request_id"],
                "artifact": _content_ref(
                    config,
                    resolved_replacement,
                    context="replacement research Request",
                ),
            },
            "change": {
                "binding": "bindings.praxist.source_checkout",
                "reason_code": "praxist_source_checkout_changed",
                "before": old_source,
                "after": replacement_source,
            },
            "record": {
                "superseded_by": actor,
                "reason": rationale,
            },
            "authority": {
                "pre_a0_only": True,
                "human_a0_carried_forward": False,
                "research_start_authorized": False,
                "formal_validation_authorized": False,
                "production_promotion_authorized": False,
                "runtime_wiring_changed": False,
            },
            "created_at": utc_now(),
        }
        supersession["supersession_id"] = _artifact_id(
            "research-request-supersession",
            supersession,
            "supersession_id",
        )
        _validate_schema(config, supersession, _SUPERSESSION_VERSION)
        digest = str(supersession["supersession_id"]).partition(":")[2]
        destination = resolved_old.parent / "supersessions" / f"{digest}.json"
        _write_immutable_artifact(
            config=config,
            destination=destination,
            payload=supersession,
            expected_version=_SUPERSESSION_VERSION,
            identity_field="supersession_id",
        )
        return ResearchSupersessionResult(
            supersession_id=str(supersession["supersession_id"]),
            supersession_path=destination,
            replacement_request_id=str(replacement["request_id"]),
            replacement_request_path=resolved_replacement,
            reused=False,
        )


def review_research_request(
    *,
    config: ForgeConfig,
    request_path: Path,
    reviewed_by: str,
    reason: str,
    decision: str = "APPROVE",
    portfolio_path: Path | None = None,
    portfolio_study_id: str | None = None,
) -> ResearchApprovalResult:
    """Write the single exact-bound A0 approval or rejection for a Request."""

    reviewer = _nonempty(reviewed_by, "research review requires a named human reviewer")
    review_reason = _nonempty(reason, "research review requires a non-empty reason")
    if decision not in {"APPROVE", "REJECT"}:
        raise ResearchControlError("research review decision must be APPROVE or REJECT")
    if (portfolio_path is None) != (portfolio_study_id is None):
        raise ResearchControlError(
            "portfolio-bound A0 requires both portfolio_path and portfolio_study_id"
        )
    if decision == "REJECT" and portfolio_path is not None:
        raise ResearchControlError(
            "a rejected A0 may not grant a portfolio execution policy"
        )
    request, resolved_request, request_sha256 = _load_request(config, request_path)
    if _find_supersession(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    ) is not None:
        raise ResearchControlError(
            "a superseded ResearchRequest cannot receive A0; review its exact replacement"
        )
    _verify_request_bindings(config, request)

    execution_policy: dict[str, Any] | None = None
    if decision == "APPROVE" and portfolio_path is None:
        from .research_portfolio import find_research_portfolio_memberships

        memberships = find_research_portfolio_memberships(
            config=config,
            request_path=resolved_request,
        )
        if len(memberships) > 1:
            choices = [
                f"{item.portfolio_id}/{item.study_id}" for item in memberships
            ]
            raise ResearchControlError(
                "ResearchRequest belongs to multiple registered Portfolio studies; "
                f"an exact --portfolio/--study-id selection is required: {choices}"
            )
        if memberships:
            portfolio_path = memberships[0].portfolio_path
            portfolio_study_id = memberships[0].study_id
    if portfolio_path is not None and portfolio_study_id is not None:
        from .research_portfolio import build_research_execution_policy

        execution_policy = build_research_execution_policy(
            config=config,
            portfolio_path=portfolio_path,
            study_id=portfolio_study_id,
            request_path=resolved_request,
        )

    approval: dict[str, Any] = {
        "schema_version": _APPROVAL_VERSION,
        "request_id": request["request_id"],
        "request_sha256": request_sha256,
        "scope": "praxist_research_start",
        "decision": decision,
        "review": {
            "reviewed_by": reviewer,
            "reason": review_reason,
        },
        "authority": {
            "research_start_authorized": decision == "APPROVE",
            "formal_validation_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_authorized": False,
        },
        "created_at": utc_now(),
    }
    if execution_policy is not None:
        approval["execution_policy"] = execution_policy
    approval["approval_id"] = _artifact_id(
        "research-approval",
        approval,
        "approval_id",
    )
    _validate_schema(config, approval, _APPROVAL_VERSION)
    approval_digest = str(approval["approval_id"]).partition(":")[2]
    destination = resolved_request.parent / "approvals" / f"{approval_digest}.json"

    existing = sorted((resolved_request.parent / "approvals").glob("*.json"))
    if existing and destination not in existing:
        raise ResearchControlError(
            f"research Request already has an immutable review decision: {existing[0]}"
        )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=approval,
        expected_version=_APPROVAL_VERSION,
        identity_field="approval_id",
    )
    return ResearchApprovalResult(
        approval_id=str(approval["approval_id"]),
        approval_path=destination,
        decision=decision,
    )


def issue_research_control_directive(
    *,
    config: ForgeConfig,
    request_path: Path,
    action: str,
    expected_event_sha256: str,
    requested_by: str,
    reason: str,
    grace_seconds: int = 300,
) -> ResearchDirectiveResult:
    """Seal one revision-bound PAUSE, RESUME, or CANCEL lifecycle action."""

    normalized_action = action.upper()
    if normalized_action not in {"PAUSE", "RESUME", "CANCEL"}:
        raise ResearchControlError(
            "research control action must be PAUSE, RESUME, or CANCEL"
        )
    actor = _nonempty(
        requested_by,
        "research control directive requires a named operator",
    )
    rationale = _nonempty(
        reason,
        "research control directive requires a non-empty reason",
    )
    if (
        not isinstance(grace_seconds, int)
        or isinstance(grace_seconds, bool)
        or not 1 <= grace_seconds <= 600
    ):
        raise ResearchControlError("stop grace_seconds must be an integer from 1 to 600")

    request, resolved_request, request_sha256 = _load_request(config, request_path)
    approval_record = _find_approval(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    )
    if approval_record is None or approval_record[0]["decision"] != "APPROVE":
        raise ResearchControlError(
            "research lifecycle directives require an exact APPROVE A0 artifact"
        )
    approval, approval_path, approval_sha256 = approval_record
    events = _load_events(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
        approval=approval,
        approval_sha256=approval_sha256,
    )
    if not events or _latest_event(events, "START_CONFIRMED") is None:
        raise ResearchControlError(
            "research lifecycle directives require an already started exact run"
        )
    if expected_event_sha256 != events[-1][2]:
        raise ResearchControlError(
            "research control directive event revision is stale"
        )
    pending = _pending_directive(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
        approval=approval,
        approval_sha256=approval_sha256,
        events=events,
    )
    if pending is not None:
        raise ResearchControlError(
            f"research Request already has a pending directive: {pending[0]['directive_id']}"
        )
    state = str(events[-1][0]["state"])
    allowed_states = {
        "PAUSE": {"STARTING", "RUNNING"},
        "RESUME": {"PAUSED", "RUN_FAILED"},
        "CANCEL": {"STARTING", "RUNNING", "PAUSED", "RUN_FAILED"},
    }
    if state not in allowed_states[normalized_action]:
        raise ResearchControlError(
            f"cannot {normalized_action} a research Request in state {state}"
        )
    if normalized_action == "RESUME":
        _verify_request_bindings(config, request)
        run_dir = Path(request["launch"]["run_dir"])
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ResearchControlError(
                f"resume requires the exact existing non-symlink run_dir: {run_dir}"
            )

    directive: dict[str, Any] = {
        "schema_version": _DIRECTIVE_VERSION,
        "request_id": request["request_id"],
        "request_sha256": request_sha256,
        "approval_id": approval["approval_id"],
        "approval_sha256": approval_sha256,
        "expected_event_sha256": expected_event_sha256,
        "action": normalized_action,
        "parameters": {
            "grace_seconds": grace_seconds,
            "resume_policy": "completed_generation",
        },
        "review": {"requested_by": actor, "reason": rationale},
        "authority": {
            "named_operator_action": True,
            "research_lifecycle_only": True,
            "new_research_start_authorized": False,
            "force_resume_authorized": False,
            "artifact_cropping_authorized": False,
            "formal_validation_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
        },
        "created_at": utc_now(),
    }
    directive["directive_id"] = _artifact_id(
        "research-control-directive",
        directive,
        "directive_id",
    )
    _validate_schema(config, directive, _DIRECTIVE_VERSION)
    digest = str(directive["directive_id"]).partition(":")[2]
    destination = resolved_request.parent / "directives" / f"{digest}.json"
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=directive,
        expected_version=_DIRECTIVE_VERSION,
        identity_field="directive_id",
    )
    del approval_path
    return ResearchDirectiveResult(
        directive_id=str(directive["directive_id"]),
        directive_path=destination,
        action=normalized_action,
    )


def list_research_inbox(*, config: ForgeConfig) -> tuple[ResearchControlStatus, ...]:
    """Return the immutable projection for every registered research Request."""

    return tuple(
        inspect_research_request(config=config, request_path=path)
        for path in _request_paths(config)
    )


def inspect_research_request(
    *,
    config: ForgeConfig,
    request_path: Path,
) -> ResearchControlStatus:
    """Project one Request's current state without invoking Praxist."""

    request, resolved_request, request_sha256 = _load_request(config, request_path)
    supersession_record = _find_supersession(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    )
    if supersession_record is not None:
        return _status_from_parts(
            request,
            resolved_request,
            None,
            (),
            "SUPERSEDED",
            supersession=supersession_record,
        )
    approval_record = _find_approval(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    )
    if approval_record is None:
        if _event_paths(resolved_request):
            raise ResearchControlError("an unreviewed Request may not have control events")
        return _status_from_parts(request, resolved_request, None, (), "AWAITING_RESEARCH_APPROVAL")
    approval, approval_path, approval_sha256 = approval_record
    events = _load_events(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
        approval=approval,
        approval_sha256=approval_sha256,
    )
    if approval["decision"] == "REJECT":
        if events:
            raise ResearchControlError("a rejected Request may not have lifecycle events")
        if _directive_paths(resolved_request):
            raise ResearchControlError("a rejected Request may not have control directives")
        state = "REJECTED"
    else:
        _pending_directive(
            config=config,
            request=request,
            request_path=resolved_request,
            request_sha256=request_sha256,
            approval=approval,
            approval_sha256=approval_sha256,
            events=events,
        )
        state = str(events[-1][0]["state"]) if events else "APPROVED"
    return _status_from_parts(request, resolved_request, approval_path, events, state)


def reconcile_research_control(
    *,
    config: ForgeConfig,
    request_path: Path | None = None,
    runner: PraxistCommandRunner | None = None,
) -> tuple[ResearchControlStatus, ...]:
    """Run one bounded, globally serialized reconciliation pass."""

    selected = (request_path,) if request_path is not None else _request_paths(config)
    command_runner = runner or SubprocessPraxistRunner()
    lock_path = config.paths.artifacts_root / "research_control" / ".reconcile.lock"
    with _exclusive_lock(lock_path):
        return tuple(
            _reconcile_one(
                config=config,
                request_path=path,
                runner=command_runner,
            )
            for path in selected
        )


def _reconcile_one(
    *,
    config: ForgeConfig,
    request_path: Path,
    runner: PraxistCommandRunner,
) -> ResearchControlStatus:
    request, resolved_request, request_sha256 = _load_request(config, request_path)
    supersession_record = _find_supersession(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    )
    if supersession_record is not None:
        return _status_from_parts(
            request,
            resolved_request,
            None,
            (),
            "SUPERSEDED",
            supersession=supersession_record,
        )
    approval_record = _find_approval(
        config=config,
        request=request,
        request_path=resolved_request,
        request_sha256=request_sha256,
    )
    if approval_record is None:
        return _status_from_parts(request, resolved_request, None, (), "AWAITING_RESEARCH_APPROVAL")
    approval, approval_path, approval_sha256 = approval_record
    if approval["decision"] == "REJECT":
        return _status_from_parts(request, resolved_request, approval_path, (), "REJECTED")

    with _exclusive_lock(resolved_request.parent / ".control.lock"):
        events = _load_events(
            config=config,
            request=request,
            request_path=resolved_request,
            request_sha256=request_sha256,
            approval=approval,
            approval_sha256=approval_sha256,
        )
        pending_directive = _pending_directive(
            config=config,
            request=request,
            request_path=resolved_request,
            request_sha256=request_sha256,
            approval=approval,
            approval_sha256=approval_sha256,
            events=events,
        )
        if pending_directive is not None:
            events = _reconcile_directive(
                config=config,
                request=request,
                request_path=resolved_request,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                directive=pending_directive[0],
                runner=runner,
            )
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                str(events[-1][0]["state"]),
            )
        if events and events[-1][0]["state"] in _TERMINAL_STATES:
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                str(events[-1][0]["state"]),
            )

        start_confirmed = _latest_event(events, "START_CONFIRMED")
        start_intent = _latest_event(events, "START_INTENT")
        if start_confirmed is not None:
            events = _poll_started_run(
                config=config,
                request=request,
                request_path=resolved_request,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                runner=runner,
            )
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                str(events[-1][0]["state"]),
            )

        if start_intent is not None:
            events = _recover_or_finish_start(
                config=config,
                request=request,
                request_path=resolved_request,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                runner=runner,
            )
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                str(events[-1][0]["state"]),
            )

        try:
            _verify_request_bindings(config, request)
            _assert_run_dir_available(request)
        except ResearchControlError as exc:
            events = _append_event(
                config=config,
                request=request,
                request_path=resolved_request,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="BLOCKED",
                command=None,
                run=None,
                details=[str(exc)],
            )
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                "BLOCKED",
            )

        capacity = _run_capacity_check(config, request, approval, runner)
        if not _command_succeeded(capacity.execution):
            events = _append_event(
                config=config,
                request=request,
                request_path=resolved_request,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="BLOCKED",
                command=_command_receipt("capacity", capacity.execution),
                run=None,
                details=[
                    _command_failure_detail("capacity check", capacity.execution)
                ],
            )
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                "BLOCKED",
            )
        capacity_state = (
            "WAITING_FOR_CAPACITY" if not capacity.available else "APPROVED"
        )
        events = _append_observation_if_changed(
            config=config,
            request=request,
            request_path=resolved_request,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CAPACITY_OBSERVED",
            state=capacity_state,
            command=_command_receipt("capacity", capacity.execution),
            run=None,
            details=list(capacity.details),
        )
        if not capacity.available:
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                "WAITING_FOR_CAPACITY",
            )

        if _latest_event(events, "DOCTOR_SUCCEEDED") is None:
            doctor = _run_doctor(request, runner)
            if not _command_succeeded(doctor):
                events = _append_event(
                    config=config,
                    request=request,
                    request_path=resolved_request,
                    request_sha256=request_sha256,
                    approval=approval,
                    approval_path=approval_path,
                    approval_sha256=approval_sha256,
                    events=events,
                    kind="CONTROL_BLOCKED",
                    state="BLOCKED",
                    command=_command_receipt("doctor", doctor),
                    run=None,
                    details=[_command_failure_detail("Praxist doctor", doctor)],
                )
                return _status_from_parts(
                    request,
                    resolved_request,
                    approval_path,
                    events,
                    "BLOCKED",
                )
            doctor_report = _parse_json_object(doctor, context="Praxist doctor")
            if doctor_report.get("ok") is not True:
                events = _append_event(
                    config=config,
                    request=request,
                    request_path=resolved_request,
                    request_sha256=request_sha256,
                    approval=approval,
                    approval_path=approval_path,
                    approval_sha256=approval_sha256,
                    events=events,
                    kind="CONTROL_BLOCKED",
                    state="BLOCKED",
                    command=_command_receipt("doctor", doctor),
                    run=None,
                    details=["Praxist doctor did not report ok=true"],
                )
                return _status_from_parts(
                    request,
                    resolved_request,
                    approval_path,
                    events,
                    "BLOCKED",
                )
            events = _append_event(
                config=config,
                request=request,
                request_path=resolved_request,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="DOCTOR_SUCCEEDED",
                state="APPROVED",
                command=_command_receipt("doctor", doctor),
                run=None,
                details=["Praxist runtime readiness check passed"],
            )

        events = _ensure_resolved(
            config=config,
            request=request,
            request_path=resolved_request,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            runner=runner,
        )
        if events[-1][0]["state"] == "BLOCKED":
            return _status_from_parts(
                request,
                resolved_request,
                approval_path,
                events,
                "BLOCKED",
            )
        events = _start_after_resolve(
            config=config,
            request=request,
            request_path=resolved_request,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            runner=runner,
        )
        return _status_from_parts(
            request,
            resolved_request,
            approval_path,
            events,
            str(events[-1][0]["state"]),
        )


def _reconcile_directive(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    directive: dict[str, Any],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    if directive["action"] == "RESUME":
        return _reconcile_resume_directive(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            directive=directive,
            runner=runner,
        )
    return _reconcile_stop_directive(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        directive=directive,
        runner=runner,
    )


def _reconcile_stop_directive(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    directive: dict[str, Any],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    directive_id = str(directive["directive_id"])
    action = str(directive["action"])
    current_state = str(events[-1][0]["state"])
    if action == "CANCEL" and current_state in {"PAUSED", "RUN_FAILED"}:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="STOP_CONFIRMED",
            state="CANCELLED",
            command=None,
            run=_stopped_run_snapshot(request, _latest_run(events)),
            details=["cancelled an already non-live exact run without a process command"],
            directive_id=directive_id,
        )

    status = _run_target_status(request, runner)
    if not _command_succeeded(status):
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="RUNNING" if current_state == "STOPPING" else current_state,
            command=_command_receipt("status", status),
            run=_latest_run(events),
            details=[_command_failure_detail("pre-stop targeted status", status)],
            directive_id=directive_id,
        )
    try:
        rows = _parse_json_array(status, context="Praxist pre-stop targeted status")
        observed_run = _exact_status_run(request, rows)
    except ResearchControlError as exc:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="RUNNING" if current_state == "STOPPING" else current_state,
            command=_command_receipt("status", status),
            run=_latest_run(events),
            details=[str(exc)],
            directive_id=directive_id,
        )

    if observed_run is not None:
        observed_state = _control_state_for_run(observed_run)
        if observed_state == "RUN_COMPLETED":
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="RUN_COMPLETED",
                command=_command_receipt("status", status),
                run=observed_run,
                details=[f"exact run completed before {action.lower()} could be applied"],
                directive_id=directive_id,
            )
        if observed_state == "RUN_FAILED":
            return _append_stop_confirmed(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                directive=directive,
                command=_command_receipt("status", status),
                run=observed_run,
                detail="exact run was already non-live at the stop boundary",
            )
    else:
        return _append_stop_confirmed(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            directive=directive,
            command=_command_receipt("status", status),
            run=_latest_run(events),
            detail="targeted status confirmed that the exact run is no longer registered",
        )

    if _directive_event(events, directive_id, "STOP_INTENT") is None:
        events = _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="STOP_INTENT",
            state="STOPPING",
            command=None,
            run=observed_run,
            details=[f"durable {action.lower()} boundary recorded before Praxist stop"],
            directive_id=directive_id,
        )

    stop = _run_stop(request, directive, runner)
    stop_detail = "Praxist stop returned a successful exact receipt"
    if _command_succeeded(stop):
        try:
            _validate_stop_result(request, _parse_json_object(stop, context="Praxist stop"))
        except ResearchControlError as exc:
            stop_detail = str(exc)
    else:
        stop_detail = _command_failure_detail("Praxist stop", stop)

    recovery = _run_target_status(request, runner)
    if not _command_succeeded(recovery):
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="RUNNING",
            command=_command_receipt("stop", stop),
            run=observed_run,
            details=[stop_detail, _command_failure_detail("post-stop targeted status", recovery)],
            directive_id=directive_id,
        )
    try:
        recovered_rows = _parse_json_array(
            recovery,
            context="Praxist post-stop targeted status",
        )
        recovered_run = _exact_status_run(request, recovered_rows)
    except ResearchControlError as exc:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="RUNNING",
            command=_command_receipt("stop", stop),
            run=observed_run,
            details=[stop_detail, str(exc)],
            directive_id=directive_id,
        )
    if recovered_run is None or _control_state_for_run(recovered_run) == "RUN_FAILED":
        return _append_stop_confirmed(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            directive=directive,
            command=_command_receipt("stop", stop),
            run=recovered_run or observed_run,
            detail=stop_detail,
        )
    if _control_state_for_run(recovered_run) == "RUN_COMPLETED":
        blocked_state = "RUN_COMPLETED"
        detail = f"exact run completed while {action.lower()} was being applied"
    else:
        blocked_state = "RUNNING"
        detail = "exact run remained live after the bounded stop command"
    return _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="CONTROL_BLOCKED",
        state=blocked_state,
        command=_command_receipt("stop", stop),
        run=recovered_run,
        details=[stop_detail, detail],
        directive_id=directive_id,
    )


def _append_stop_confirmed(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    directive: dict[str, Any],
    command: dict[str, Any] | None,
    run: dict[str, Any] | None,
    detail: str,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    action = str(directive["action"])
    return _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="STOP_CONFIRMED",
        state="CANCELLED" if action == "CANCEL" else "PAUSED",
        command=command,
        run=_stopped_run_snapshot(request, run),
        details=[detail],
        directive_id=str(directive["directive_id"]),
    )


def _reconcile_resume_directive(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    directive: dict[str, Any],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    directive_id = str(directive["directive_id"])
    try:
        _verify_request_bindings(config, request)
        run_dir = Path(request["launch"]["run_dir"])
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ResearchControlError(
                f"resume requires the exact existing non-symlink run_dir: {run_dir}"
            )
    except ResearchControlError as exc:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="PAUSED",
            command=None,
            run=_latest_run(events),
            details=[str(exc)],
            directive_id=directive_id,
        )

    status = _run_target_status(request, runner)
    if not _command_succeeded(status):
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="PAUSED",
            command=_command_receipt("status", status),
            run=_latest_run(events),
            details=[_command_failure_detail("pre-resume targeted status", status)],
            directive_id=directive_id,
        )
    try:
        rows = _parse_json_array(status, context="Praxist pre-resume targeted status")
        observed_run = _exact_status_run(request, rows)
    except ResearchControlError as exc:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="PAUSED",
            command=_command_receipt("status", status),
            run=_latest_run(events),
            details=[str(exc)],
            directive_id=directive_id,
        )
    if observed_run is not None:
        observed_state = _control_state_for_run(observed_run)
        if observed_state in {"STARTING", "RUNNING"}:
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="RESUME_CONFIRMED",
                state=observed_state,
                command=_command_receipt("status", status),
                run=observed_run,
                details=["recovered an already-live exact run without repeating resume"],
                directive_id=directive_id,
            )
        if observed_state == "RUN_COMPLETED":
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="RUN_COMPLETED",
                command=_command_receipt("status", status),
                run=observed_run,
                details=["completed exact run cannot be resumed"],
                directive_id=directive_id,
            )

    if _directive_event(events, directive_id, "RESUME_INTENT") is not None:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="PAUSED",
            command=_command_receipt("status", status),
            run=observed_run or _latest_run(events),
            details=[
                "ambiguous resume boundary: no live run after durable RESUME_INTENT; "
                "a new revision-bound directive is required"
            ],
            directive_id=directive_id,
        )

    events = _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="RESUME_INTENT",
        state="RESUMING",
        command=None,
        run=_resuming_run_snapshot(request, observed_run or _latest_run(events)),
        details=["durable resume boundary recorded before Praxist resume"],
        directive_id=directive_id,
    )
    resume = _run_resume(request, directive, runner)
    resumed_run: dict[str, Any] | None = None
    resume_detail: str
    if _command_succeeded(resume):
        try:
            resumed_run = _normalize_start_result(
                request,
                _parse_json_object(resume, context="Praxist resume"),
            )
            resume_detail = "Praxist resume returned an exact registry entry"
        except ResearchControlError as exc:
            resume_detail = str(exc)
    else:
        resume_detail = _command_failure_detail("Praxist resume", resume)
    if resumed_run is None:
        recovery = _run_target_status(request, runner)
        if _command_succeeded(recovery):
            try:
                recovery_rows = _parse_json_array(
                    recovery,
                    context="Praxist targeted status after resume",
                )
                recovered_run = _exact_status_run(request, recovery_rows)
                if recovered_run is not None and _control_state_for_run(recovered_run) in {
                    "STARTING",
                    "RUNNING",
                }:
                    resumed_run = recovered_run
                    resume_detail = (
                        f"{resume_detail}; exact run recovered through targeted status"
                    )
            except ResearchControlError as exc:
                resume_detail = f"{resume_detail}; {exc}"
        if resumed_run is None:
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="PAUSED",
                command=_command_receipt("resume", resume),
                run=observed_run or _latest_run(events),
                details=[resume_detail],
                directive_id=directive_id,
            )
    events = _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="RESUME_CONFIRMED",
        state=_control_state_for_run(resumed_run),
        command=_command_receipt("resume", resume),
        run=resumed_run,
        details=[resume_detail],
        directive_id=directive_id,
    )
    if events[-1][0]["state"] in {"RUN_COMPLETED", "RUN_FAILED"}:
        return events
    return _poll_started_run(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        runner=runner,
    )


def _ensure_resolved(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    if _latest_event(events, "RESOLVE_SUCCEEDED") is not None:
        return events
    preflight_dir = request_path.parent / "preflight"
    resolve_intent = _latest_event(events, "RESOLVE_INTENT")
    if resolve_intent is None:
        if preflight_dir.exists():
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="BLOCKED",
                command=None,
                run=None,
                details=["preflight directory existed before RESOLVE_INTENT"],
            )
        events = _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="RESOLVE_INTENT",
            state="APPROVED",
            command=None,
            run=None,
            details=[f"preflight_dir={preflight_dir}"],
        )
    elif preflight_dir.exists():
        try:
            details = _validate_preflight(request, preflight_dir, result=None)
        except ResearchControlError as exc:
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="BLOCKED",
                command=None,
                run=None,
                details=[f"ambiguous preflight recovery: {exc}"],
            )
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="RESOLVE_SUCCEEDED",
            state="PREFLIGHT_RESOLVED",
            command=None,
            run=None,
            details=["recovered complete preflight after RESOLVE_INTENT", *details],
        )

    resolve = _run_resolve(request, preflight_dir, runner)
    if not _command_succeeded(resolve):
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="BLOCKED",
            command=_command_receipt("resolve", resolve),
            run=None,
            details=[_command_failure_detail("Praxist resolve", resolve)],
        )
    result = _parse_json_object(resolve, context="Praxist resolve")
    try:
        details = _validate_preflight(request, preflight_dir, result=result)
    except ResearchControlError as exc:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="BLOCKED",
            command=_command_receipt("resolve", resolve),
            run=None,
            details=[str(exc)],
        )
    return _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="RESOLVE_SUCCEEDED",
        state="PREFLIGHT_RESOLVED",
        command=_command_receipt("resolve", resolve),
        run=None,
        details=details,
    )


def _start_after_resolve(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    capacity = _run_capacity_check(config, request, approval, runner)
    if not _command_succeeded(capacity.execution):
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="BLOCKED",
            command=_command_receipt("capacity", capacity.execution),
            run=None,
            details=[
                _command_failure_detail(
                    "pre-start capacity check",
                    capacity.execution,
                )
            ],
        )
    capacity_state = (
        "WAITING_FOR_CAPACITY" if not capacity.available else "PREFLIGHT_RESOLVED"
    )
    events = _append_observation_if_changed(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="CAPACITY_OBSERVED",
        state=capacity_state,
        command=_command_receipt("capacity", capacity.execution),
        run=None,
        details=["pre-start recheck", *capacity.details],
    )
    if not capacity.available:
        return events
    if _latest_event(events, "START_INTENT") is None:
        events = _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="START_INTENT",
            state="STARTING",
            command=None,
            run=_request_run_snapshot(request),
            details=["durable start boundary recorded before Praxist launch"],
        )
    return _invoke_start_and_poll(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        runner=runner,
    )


def _recover_or_finish_start(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    status = _run_target_status(request, runner)
    if not _command_succeeded(status):
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="BLOCKED",
            command=_command_receipt("recovery", status),
            run=_request_run_snapshot(request),
            details=[
                _command_failure_detail(
                    "Praxist targeted status at the START_INTENT recovery boundary",
                    status,
                )
            ],
        )
    rows = _parse_json_array(status, context="Praxist targeted status")
    run = _exact_status_run(request, rows)
    if run is not None:
        events = _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="START_CONFIRMED",
            state=_control_state_for_run(run),
            command=_command_receipt("recovery", status),
            run=run,
            details=["recovered exact Praxist run after durable START_INTENT"],
        )
        return events

    run_dir = Path(request["launch"]["run_dir"])
    if run_dir.exists():
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="BLOCKED",
            command=_command_receipt("recovery", status),
            run=_request_run_snapshot(request),
            details=["run_dir exists without an exact status row after START_INTENT"],
        )
    try:
        _verify_request_bindings(config, request)
        _assert_run_dir_available(request)
    except ResearchControlError as exc:
        return _append_event(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="CONTROL_BLOCKED",
            state="BLOCKED",
            command=_command_receipt("recovery", status),
            run=_request_run_snapshot(request),
            details=[str(exc)],
        )
    return _invoke_start_and_poll(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        runner=runner,
    )


def _invoke_start_and_poll(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    start = _run_start(request, runner)
    run: dict[str, Any] | None = None
    details: list[str]
    if _command_succeeded(start):
        result = _parse_json_object(start, context="Praxist start")
        run = _normalize_start_result(request, result)
        details = ["Praxist detached launch returned an exact registry entry"]
    else:
        recovery = _run_target_status(request, runner)
        if _command_succeeded(recovery):
            rows = _parse_json_array(recovery, context="Praxist targeted status after start")
            run = _exact_status_run(request, rows)
        if run is None:
            return _append_event(
                config=config,
                request=request,
                request_path=request_path,
                request_sha256=request_sha256,
                approval=approval,
                approval_path=approval_path,
                approval_sha256=approval_sha256,
                events=events,
                kind="CONTROL_BLOCKED",
                state="BLOCKED",
                command=_command_receipt("start", start),
                run=_request_run_snapshot(request),
                details=[_command_failure_detail("Praxist start", start)],
            )
        details = [
            _command_failure_detail("Praxist start response", start),
            "exact run recovered through targeted status; start was not repeated",
        ]
    events = _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="START_CONFIRMED",
        state=_control_state_for_run(run),
        command=_command_receipt("start", start),
        run=run,
        details=details,
    )
    if events[-1][0]["state"] in {"RUN_COMPLETED", "RUN_FAILED"}:
        return events
    return _poll_started_run(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        runner=runner,
    )


def _poll_started_run(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    runner: PraxistCommandRunner,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    prior_run = _latest_run(events) or _request_run_snapshot(request)
    prior_state = str(events[-1][0]["state"]) if events else "STARTING"
    status = _run_target_status(request, runner)
    if not _command_succeeded(status):
        return _append_observation_if_changed(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="STATUS_OBSERVED",
            state=prior_state if prior_state in {"STARTING", "RUNNING"} else "RUNNING",
            command=_command_receipt("status", status),
            run=prior_run,
            details=[_command_failure_detail("Praxist targeted status", status)],
        )
    rows = _parse_json_array(status, context="Praxist targeted status")
    run = _exact_status_run(request, rows)
    if run is None:
        return _append_observation_if_changed(
            config=config,
            request=request,
            request_path=request_path,
            request_sha256=request_sha256,
            approval=approval,
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            events=events,
            kind="STATUS_OBSERVED",
            state="STARTING" if prior_state == "STARTING" else "RUNNING",
            command=_command_receipt("status", status),
            run=prior_run,
            details=["targeted status returned no exact row; launch state remains provisional"],
        )
    return _append_observation_if_changed(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind="STATUS_OBSERVED",
        state=_control_state_for_run(run),
        command=_command_receipt("status", status),
        run=run,
        details=["exact targeted Praxist status observed"],
    )


def _run_capacity_check(
    config: ForgeConfig,
    request: dict[str, Any],
    approval: dict[str, Any],
    runner: PraxistCommandRunner,
) -> CapacityObservation:
    executable = _request_executable(request)
    execution = runner.run(
        (str(executable), "status", "--active", "--json"),
        cwd=Path(request["bindings"]["task_project"]["root"]),
        timeout_seconds=30,
    )
    if not _command_succeeded(execution):
        return CapacityObservation(execution, (), (), ("capacity_status_failed",))
    raw_rows = _parse_json_array(execution, context="Praxist active status")
    details = _active_run_details(raw_rows)
    rows = tuple(raw_rows)
    policy = approval.get("execution_policy")
    if policy is None:
        blockers = ("legacy_host_wide_active_run_limit",) if rows else ()
        return CapacityObservation(
            execution=execution,
            active_rows=rows,
            details=tuple(["capacity_policy=host_wide_serial", *details, *blockers]),
            blockers=blockers,
        )

    known: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unknown_ids: list[str] = []
    for row in rows:
        active_policy = _active_run_execution_policy(config, row)
        if (
            active_policy is None
            or active_policy["portfolio"] != policy["portfolio"]
        ):
            unknown_ids.append(str(row.get("run_id") or "<unidentified>"))
            continue
        known.append((row, active_policy))

    lane = str(policy["concurrency_lane"])
    lane_count = sum(
        active_policy["concurrency_lane"] == lane
        for _, active_policy in known
    )
    global_count = len(known)
    blocker_list: list[str] = []
    if unknown_ids:
        blocker_list.append(f"unknown_active_run_ids={','.join(sorted(unknown_ids))}")
    if global_count >= int(policy["max_active_runs_global"]):
        blocker_list.append(
            "portfolio_global_limit_reached="
            f"{global_count}/{policy['max_active_runs_global']}"
        )
    if lane_count >= int(policy["max_active_runs_lane"]):
        blocker_list.append(
            f"portfolio_lane_limit_reached={lane}:{lane_count}/"
            f"{policy['max_active_runs_lane']}"
        )
    policy_details = [
        f"capacity_policy=portfolio:{policy['portfolio']['artifact_id']}",
        f"portfolio_active_run_count={global_count}",
        f"portfolio_lane_active_run_count={lane}:{lane_count}",
    ]
    return CapacityObservation(
        execution=execution,
        active_rows=rows,
        details=tuple([*policy_details, *details, *blocker_list]),
        blockers=tuple(blocker_list),
    )


def _active_run_execution_policy(
    config: ForgeConfig,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    run_id = row.get("run_id")
    run_dir = row.get("run_dir")
    if not isinstance(run_id, str) or not run_id:
        return None
    if not isinstance(run_dir, str) or not run_dir:
        return None
    matches: list[dict[str, Any]] = []
    for request_path in _request_paths(config):
        candidate, resolved, request_sha256 = _load_request(config, request_path)
        if candidate["launch"]["run_id"] != run_id:
            continue
        if _normalized_path(run_dir, context="active Praxist run_dir") != candidate["launch"]["run_dir"]:
            continue
        approval_record = _find_approval(
            config=config,
            request=candidate,
            request_path=resolved,
            request_sha256=request_sha256,
        )
        if approval_record is None or approval_record[0]["decision"] != "APPROVE":
            continue
        candidate_approval, _, approval_sha256 = approval_record
        events = _load_events(
            config=config,
            request=candidate,
            request_path=resolved,
            request_sha256=request_sha256,
            approval=candidate_approval,
            approval_sha256=approval_sha256,
        )
        if _latest_event(events, "START_CONFIRMED") is None or not events:
            continue
        if events[-1][0]["state"] not in {"STARTING", "RUNNING", "RESUMING"}:
            continue
        candidate_policy = candidate_approval.get("execution_policy")
        if candidate_policy is not None:
            matches.append(candidate_policy)
    if len(matches) > 1:
        raise ResearchControlError(
            f"active Praxist run maps to multiple Forge execution policies: {run_id}"
        )
    return matches[0] if matches else None


def _run_doctor(
    request: dict[str, Any],
    runner: PraxistCommandRunner,
) -> CommandExecution:
    executable = _request_executable(request)
    task_root = Path(request["bindings"]["task_project"]["root"])
    argv = [str(executable), "doctor", "--json", "--task-path", str(task_root)]
    argv.extend(_profile_argv(request, command="doctor"))
    return runner.run(argv, cwd=task_root, timeout_seconds=120)


def _run_resolve(
    request: dict[str, Any],
    preflight_dir: Path,
    runner: PraxistCommandRunner,
) -> CommandExecution:
    executable = _request_executable(request)
    task_root = Path(request["bindings"]["task_project"]["root"])
    argv = [
        str(executable),
        "resolve",
        str(task_root),
        "--run-dir",
        str(preflight_dir),
    ]
    argv.extend(_profile_argv(request, command="resolve"))
    return runner.run(argv, cwd=task_root, timeout_seconds=300)


def _run_start(
    request: dict[str, Any],
    runner: PraxistCommandRunner,
) -> CommandExecution:
    executable = _request_executable(request)
    task_root = Path(request["bindings"]["task_project"]["root"])
    profile = request["launch"]["profile"]
    argv = [
        str(executable),
        "start",
        "--task-path",
        str(task_root),
        "--run-dir",
        request["launch"]["run_dir"],
        "--daemonize",
        "--json",
        "--startup-timeout",
        str(profile["startup_timeout_seconds"]),
    ]
    argv.extend(_profile_argv(request, command="start"))
    return runner.run(
        argv,
        cwd=task_root,
        timeout_seconds=float(profile["startup_timeout_seconds"] + 30),
    )


def _run_stop(
    request: dict[str, Any],
    directive: dict[str, Any],
    runner: PraxistCommandRunner,
) -> CommandExecution:
    executable = _request_executable(request)
    task_root = Path(request["bindings"]["task_project"]["root"])
    return runner.run(
        (
            str(executable),
            "stop",
            request["launch"]["run_id"],
            "--grace",
            str(directive["parameters"]["grace_seconds"]),
            "--json",
        ),
        cwd=task_root,
        timeout_seconds=float(directive["parameters"]["grace_seconds"] + 30),
    )


def _run_resume(
    request: dict[str, Any],
    directive: dict[str, Any],
    runner: PraxistCommandRunner,
) -> CommandExecution:
    executable = _request_executable(request)
    task_root = Path(request["bindings"]["task_project"]["root"])
    profile = request["launch"]["profile"]
    argv = [
        str(executable),
        "resume",
        request["launch"]["run_dir"],
        "--daemonize",
        "--json",
        "--resume-policy",
        directive["parameters"]["resume_policy"],
        "--startup-timeout",
        str(profile["startup_timeout_seconds"]),
    ]
    argv.extend(_profile_argv(request, command="resume"))
    return runner.run(
        argv,
        cwd=task_root,
        timeout_seconds=float(profile["startup_timeout_seconds"] + 30),
    )


def _run_target_status(
    request: dict[str, Any],
    runner: PraxistCommandRunner,
) -> CommandExecution:
    executable = _request_executable(request)
    task_root = Path(request["bindings"]["task_project"]["root"])
    return runner.run(
        (
            str(executable),
            "status",
            "--run-id",
            request["launch"]["run_id"],
            "--json",
        ),
        cwd=task_root,
        timeout_seconds=30,
    )


def _profile_argv(request: dict[str, Any], *, command: str) -> list[str]:
    profile = request["launch"]["profile"]
    argv: list[str] = []
    config_ref = profile["config_file"]
    if config_ref is not None:
        argv.extend(("--config-file", str(_content_ref_path(config_ref))))
    if profile["agent_system"] is not None:
        argv.extend(("--agent-system", profile["agent_system"]))
    if command in {"resolve", "start", "resume"} and profile["runtime"] is not None:
        argv.extend(("--runtime", profile["runtime"]))
    if profile["codex_native"] is True:
        argv.append("--codex-native")
    if profile["model_provider"] is not None:
        argv.extend(("--model-provider", profile["model_provider"]))
    if profile["model"] is not None:
        argv.extend(("--model", profile["model"]))
    if command in {"start", "resume"}:
        argv.extend(("--strategy", profile["strategy"]))
        if profile["cohort"] is not None:
            argv.extend(("--cohort", str(profile["cohort"])))
        if profile["generations"] is not None:
            argv.extend(("--generations", str(profile["generations"])))
    return argv


def _validate_preflight(
    request: dict[str, Any],
    preflight_dir: Path,
    *,
    result: dict[str, Any] | None,
) -> list[str]:
    expected_root = str(preflight_dir.resolve())
    if result is not None:
        if result.get("status") != "resolved":
            raise ResearchControlError("Praxist resolve result must report status='resolved'")
        if result.get("run_id") != preflight_dir.name:
            raise ResearchControlError("Praxist resolve run_id does not match the preflight directory")
        if _normalized_path(result.get("run_dir"), context="Praxist resolve run_dir") != expected_root:
            raise ResearchControlError("Praxist resolve run_dir does not match the approved preflight path")

    manifest_path = preflight_dir / "task_project_manifest.json"
    run_metadata_path = preflight_dir / "run.json"
    startup_config_path = preflight_dir / "startup_config.json"
    plugin_resolution_path = preflight_dir / "plugin_resolution.json"
    effective_task_spec_path = preflight_dir / "effective_task_spec.yaml"
    manifest = _read_json_file(manifest_path, context="Praxist task project manifest")
    run_metadata = _read_json_file(run_metadata_path, context="Praxist preflight run metadata")
    startup_config = _read_json_file(
        startup_config_path,
        context="Praxist preflight startup config",
    )
    _read_json_file(
        plugin_resolution_path,
        context="Praxist preflight plugin resolution",
    )
    if not effective_task_spec_path.is_file() or effective_task_spec_path.is_symlink():
        raise ResearchControlError(
            f"missing canonical Praxist preflight effective task spec: {effective_task_spec_path}"
        )
    expected_project = request["bindings"]["task_project"]
    if manifest.get("schema_version") != "task_project_manifest.v1":
        raise ResearchControlError("Praxist preflight emitted an unsupported task manifest")
    if manifest.get("task_id") != expected_project["task_id"]:
        raise ResearchControlError("Praxist preflight task_id does not match the Request")
    if manifest.get("sha256") != expected_project["manifest_sha256"]:
        raise ResearchControlError("Praxist preflight task project digest does not match the Request")
    if manifest.get("path") != expected_project["root"]:
        raise ResearchControlError("Praxist preflight task project path does not match the Request")
    if manifest.get("files") != expected_project["files"]:
        raise ResearchControlError("Praxist preflight task project file manifest does not match the Request")
    if run_metadata.get("run_id") != preflight_dir.name:
        raise ResearchControlError("Praxist preflight run metadata has the wrong run_id")
    run_project = run_metadata.get("task_project")
    if not isinstance(run_project, dict):
        raise ResearchControlError("Praxist preflight run metadata lacks task_project")
    if run_project.get("manifest_sha256") != expected_project["manifest_sha256"]:
        raise ResearchControlError("Praxist preflight run metadata has the wrong manifest digest")
    if startup_config.get("schema_version") != "praxist.startup.v1":
        raise ResearchControlError("Praxist preflight emitted an unsupported startup config")
    canonical_args = startup_config.get("canonical_args")
    if not isinstance(canonical_args, dict):
        raise ResearchControlError("Praxist preflight startup config lacks canonical_args")
    if _normalized_path(
        canonical_args.get("task_path"),
        context="Praxist preflight startup task_path",
    ) != expected_project["root"]:
        raise ResearchControlError("Praxist preflight startup task_path does not match the Request")
    if _normalized_path(
        canonical_args.get("run_dir"),
        context="Praxist preflight startup run_dir",
    ) != expected_root:
        raise ResearchControlError("Praxist preflight startup run_dir does not match the Request")
    resume_identity = startup_config.get("resume_identity")
    if not isinstance(resume_identity, dict):
        raise ResearchControlError("Praxist preflight startup config lacks resume_identity")
    if resume_identity.get("task_project_manifest_sha256") != expected_project["manifest_sha256"]:
        raise ResearchControlError("Praxist preflight startup config has the wrong manifest digest")
    return [f"task_project_manifest_sha256={expected_project['manifest_sha256']}"]


def _normalize_start_result(
    request: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    expected_run_id = request["launch"]["run_id"]
    expected_run_dir = request["launch"]["run_dir"]
    expected_task = request["bindings"]["task_project"]["root"]
    if result.get("run_id") != expected_run_id:
        raise ResearchControlError("Praxist start returned a different run_id")
    if _normalized_path(result.get("run_dir"), context="Praxist start run_dir") != expected_run_dir:
        raise ResearchControlError("Praxist start returned a different run_dir")
    if _normalized_path(result.get("task_path"), context="Praxist start task_path") != expected_task:
        raise ResearchControlError("Praxist start returned a different task_path")
    pid = result.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ResearchControlError("Praxist start returned an invalid pid")
    state = result.get("state")
    if not isinstance(state, str) or not state:
        raise ResearchControlError("Praxist start returned an invalid state")
    extra = result.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    return {
        "run_id": expected_run_id,
        "run_dir": expected_run_dir,
        "pid": pid,
        "log_file": _nullable_text(result.get("log_file")),
        "monitor_command": _nullable_text(extra.get("monitor_command")),
        "source": "registry",
        "praxist_state": str(extra.get("startup_state") or state),
        "generation": None,
        "findings_total": None,
        "updated_at": _nullable_text(result.get("started_at")),
    }


def _validate_stop_result(request: dict[str, Any], result: dict[str, Any]) -> None:
    if result.get("run_id") != request["launch"]["run_id"]:
        raise ResearchControlError("Praxist stop returned a different run_id")
    if result.get("dry_run") is not False:
        raise ResearchControlError("Praxist stop did not confirm dry_run=false")
    for field in (
        "matched_pids",
        "descendant_pids",
        "terminated_pids",
        "killed_pids",
        "remaining_pids",
    ):
        values = result.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ResearchControlError(f"Praxist stop returned invalid {field}")
    failed_run_ids = result.get("failed_run_ids")
    if not isinstance(failed_run_ids, list) or any(
        not isinstance(value, str) or not value for value in failed_run_ids
    ):
        raise ResearchControlError("Praxist stop returned invalid failed_run_ids")
    if result["remaining_pids"] or failed_run_ids:
        raise ResearchControlError("Praxist stop left the exact run partially active")


def _exact_status_run(
    request: dict[str, Any],
    rows: list[Any],
) -> dict[str, Any] | None:
    expected_id = request["launch"]["run_id"]
    matches = [row for row in rows if isinstance(row, dict) and row.get("run_id") == expected_id]
    if not matches:
        return None
    if len(matches) != 1:
        raise ResearchControlError(f"targeted Praxist status returned {len(matches)} exact run rows")
    row = matches[0]
    expected_dir = request["launch"]["run_dir"]
    if _normalized_path(row.get("run_dir"), context="Praxist status run_dir") != expected_dir:
        raise ResearchControlError("Praxist status row run_dir does not match the Request")
    pid = row.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
        raise ResearchControlError("Praxist status row has an invalid pid")
    state = row.get("state")
    if not isinstance(state, str) or not state:
        raise ResearchControlError("Praxist status row has an invalid state")
    generation = _nullable_nonnegative_int(row.get("generation"), "Praxist status generation")
    findings = _nullable_nonnegative_int(
        row.get("findings_total"),
        "Praxist status findings_total",
    )
    return {
        "run_id": expected_id,
        "run_dir": expected_dir,
        "pid": pid,
        "log_file": None,
        "monitor_command": f"praxist --monitor --run-id {expected_id}",
        "source": _nullable_text(row.get("source")),
        "praxist_state": state,
        "generation": generation,
        "findings_total": findings,
        "updated_at": _nullable_text(row.get("updated_at")),
    }


def _control_state_for_run(run: dict[str, Any]) -> str:
    state = str(run["praxist_state"]).lower()
    if state == "completed":
        return "RUN_COMPLETED"
    if state in _FAILED_STATES:
        return "RUN_FAILED"
    if state == "starting":
        return "STARTING"
    if state in _LIVE_STATES:
        return "RUNNING"
    raise ResearchControlError(f"unsupported Praxist lifecycle state: {state!r}")


def _active_run_details(rows: list[Any]) -> list[str]:
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ResearchControlError(f"Praxist active status row {index} must be an object")
        state = row.get("state")
        if state not in _LIVE_STATES:
            raise ResearchControlError(
                f"Praxist --active returned non-live state at row {index}: {state!r}"
            )
        source = row.get("source")
        if source not in {"registry", "ps-only", "remote"}:
            raise ResearchControlError(
                f"Praxist active status row {index} has an invalid source: {source!r}"
            )
        pid = row.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
            raise ResearchControlError(f"Praxist active status row {index} has an invalid pid")
    details = [f"active_run_count={len(rows)}"]
    run_ids = sorted(
        str(row["run_id"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("run_id"), str) and row["run_id"]
    )
    if run_ids:
        details.append(f"active_run_ids={','.join(run_ids)}")
    return details


def _request_run_snapshot(request: dict[str, Any]) -> dict[str, Any]:
    run_id = request["launch"]["run_id"]
    return {
        "run_id": run_id,
        "run_dir": request["launch"]["run_dir"],
        "pid": None,
        "log_file": None,
        "monitor_command": f"praxist --monitor --run-id {run_id}",
        "source": None,
        "praxist_state": "starting",
        "generation": None,
        "findings_total": None,
        "updated_at": None,
    }


def _stopped_run_snapshot(
    request: dict[str, Any],
    run: dict[str, Any] | None,
) -> dict[str, Any]:
    snapshot = dict(run or _request_run_snapshot(request))
    snapshot.update(
        {
            "pid": 0,
            "praxist_state": "stopped",
            "updated_at": utc_now(),
        }
    )
    return snapshot


def _resuming_run_snapshot(
    request: dict[str, Any],
    run: dict[str, Any] | None,
) -> dict[str, Any]:
    snapshot = dict(run or _request_run_snapshot(request))
    snapshot.update(
        {
            "pid": None,
            "praxist_state": "starting",
            "updated_at": utc_now(),
        }
    )
    return snapshot


def _append_observation_if_changed(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    kind: str,
    state: str,
    command: dict[str, Any] | None,
    run: dict[str, Any] | None,
    details: list[str],
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    if events:
        latest = events[-1][0]
        if (
            latest["kind"] == kind
            and latest["state"] == state
            and latest["run"] == run
            and latest["details"] == details
        ):
            return events
    return _append_event(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        events=events,
        kind=kind,
        state=state,
        command=command,
        run=run,
        details=details,
    )


def _append_event(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_path: Path,
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    kind: str,
    state: str,
    command: dict[str, Any] | None,
    run: dict[str, Any] | None,
    details: list[str],
    directive_id: str | None = None,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    del approval_path
    sequence = len(events) + 1
    event: dict[str, Any] = {
        "schema_version": _EVENT_VERSION,
        "sequence": sequence,
        "request_id": request["request_id"],
        "request_sha256": request_sha256,
        "approval_id": approval["approval_id"],
        "approval_sha256": approval_sha256,
        "previous_event_sha256": events[-1][2] if events else None,
        "kind": kind,
        "state": state,
        "command": command,
        "run": run,
        "details": details,
        "authority": {
            "research_lifecycle_only": True,
            "formal_validation_performed": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    if directive_id is not None:
        event["directive_id"] = directive_id
    event["event_id"] = _artifact_id(
        "research-control-event",
        event,
        "event_id",
    )
    _validate_schema(config, event, _EVENT_VERSION)
    digest = str(event["event_id"]).partition(":")[2]
    destination = request_path.parent / "events" / f"{sequence:06d}-{digest}.json"
    _write_create_only_json(destination, event)
    file_sha256 = _sha256_file(destination, context=f"research control event {sequence}")
    return (*events, (event, destination, file_sha256))


def _load_events(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_sha256: str,
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    loaded: list[tuple[dict[str, Any], Path, str]] = []
    previous_sha256: str | None = None
    for expected_sequence, path in enumerate(_event_paths(request_path), start=1):
        event = read_json(path)
        _validate_schema(config, event, _EVENT_VERSION)
        if event["event_id"] != _artifact_id(
            "research-control-event",
            event,
            "event_id",
        ):
            raise ResearchControlError(f"research control event identity is invalid: {path}")
        if event["sequence"] != expected_sequence:
            raise ResearchControlError(f"research control event sequence is not contiguous: {path}")
        if event["request_id"] != request["request_id"]:
            raise ResearchControlError(f"research control event belongs to another Request: {path}")
        if event["request_sha256"] != request_sha256:
            raise ResearchControlError(f"research control event request digest mismatch: {path}")
        if event["approval_id"] != approval["approval_id"]:
            raise ResearchControlError(f"research control event belongs to another Approval: {path}")
        if event["approval_sha256"] != approval_sha256:
            raise ResearchControlError(f"research control event approval digest mismatch: {path}")
        if event["previous_event_sha256"] != previous_sha256:
            raise ResearchControlError(f"research control event hash chain is broken: {path}")
        expected_prefix = f"{expected_sequence:06d}-"
        expected_suffix = str(event["event_id"]).partition(":")[2] + ".json"
        if not path.name.startswith(expected_prefix) or not path.name.endswith(expected_suffix):
            raise ResearchControlError(f"research control event filename is not canonical: {path}")
        file_sha256 = _sha256_file(path, context=f"research control event {expected_sequence}")
        loaded.append((event, path, file_sha256))
        previous_sha256 = file_sha256
    return tuple(loaded)


def _pending_directive(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
) -> tuple[dict[str, Any], Path, str] | None:
    directives = _load_directives(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        approval=approval,
        approval_sha256=approval_sha256,
        events=events,
    )
    completed_ids = {
        str(event["directive_id"])
        for event, _, _ in events
        if event.get("directive_id") is not None
        and event["kind"] in {"STOP_CONFIRMED", "RESUME_CONFIRMED", "CONTROL_BLOCKED"}
    }
    pending = [
        directive
        for directive in directives
        if str(directive[0]["directive_id"]) not in completed_ids
    ]
    if len(pending) > 1:
        raise ResearchControlError(
            f"research Request has {len(pending)} concurrent pending directives"
        )
    return pending[0] if pending else None


def _load_directives(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    approval: dict[str, Any],
    approval_sha256: str,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
) -> tuple[tuple[dict[str, Any], Path, str], ...]:
    event_positions = {digest: index for index, (_, _, digest) in enumerate(events)}
    loaded: list[tuple[dict[str, Any], Path, str]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for path in _directive_paths(request_path):
        directive = read_json(path)
        _validate_schema(config, directive, _DIRECTIVE_VERSION)
        if directive["directive_id"] != _artifact_id(
            "research-control-directive",
            directive,
            "directive_id",
        ):
            raise ResearchControlError(
                f"research control directive identity is invalid: {path}"
            )
        directive_id = str(directive["directive_id"])
        if directive_id in by_id:
            raise ResearchControlError(
                f"research control directive is duplicated: {directive_id}"
            )
        if directive["request_id"] != request["request_id"]:
            raise ResearchControlError(
                f"research control directive belongs to another Request: {path}"
            )
        if directive["request_sha256"] != request_sha256:
            raise ResearchControlError(
                f"research control directive Request digest mismatch: {path}"
            )
        if directive["approval_id"] != approval["approval_id"]:
            raise ResearchControlError(
                f"research control directive belongs to another Approval: {path}"
            )
        if directive["approval_sha256"] != approval_sha256:
            raise ResearchControlError(
                f"research control directive Approval digest mismatch: {path}"
            )
        expected_revision = str(directive["expected_event_sha256"])
        if expected_revision not in event_positions:
            raise ResearchControlError(
                f"research control directive references an unknown event revision: {path}"
            )
        expected_name = directive_id.partition(":")[2] + ".json"
        if path.name != expected_name:
            raise ResearchControlError(
                f"research control directive filename is not canonical: {path}"
            )
        file_sha256 = _sha256_file(path, context="research control directive")
        loaded.append((directive, path, file_sha256))
        by_id[directive_id] = directive

    associated: dict[str, list[tuple[dict[str, Any], int]]] = {
        directive_id: [] for directive_id in by_id
    }
    for position, (event, path, _) in enumerate(events):
        directive_id = event.get("directive_id")
        if directive_id is None:
            continue
        if directive_id not in by_id:
            raise ResearchControlError(
                f"research control event references an unknown directive: {path}"
            )
        associated[str(directive_id)].append((event, position))

    for directive, path, _ in loaded:
        directive_id = str(directive["directive_id"])
        expected_position = event_positions[str(directive["expected_event_sha256"])]
        action = str(directive["action"])
        allowed_kinds = (
            {"STOP_INTENT", "STOP_CONFIRMED", "CONTROL_BLOCKED"}
            if action in {"PAUSE", "CANCEL"}
            else {"RESUME_INTENT", "RESUME_CONFIRMED", "CONTROL_BLOCKED"}
        )
        action_events = associated[directive_id]
        for event, position in action_events:
            if position <= expected_position:
                raise ResearchControlError(
                    f"research control directive event precedes its expected revision: {path}"
                )
            if event["kind"] not in allowed_kinds:
                raise ResearchControlError(
                    f"research control directive has an incompatible event kind: {path}"
                )
        intent_count = sum(
            event["kind"] in {"STOP_INTENT", "RESUME_INTENT"}
            for event, _ in action_events
        )
        completion_count = sum(
            event["kind"] in {"STOP_CONFIRMED", "RESUME_CONFIRMED", "CONTROL_BLOCKED"}
            for event, _ in action_events
        )
        if intent_count > 1 or completion_count > 1:
            raise ResearchControlError(
                f"research control directive has a non-canonical event sequence: {path}"
            )
        if completion_count and action_events[-1][0]["kind"] not in {
            "STOP_CONFIRMED",
            "RESUME_CONFIRMED",
            "CONTROL_BLOCKED",
        }:
            raise ResearchControlError(
                f"research control directive has events after completion: {path}"
            )
    return tuple(loaded)


def _find_supersession(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
) -> tuple[dict[str, Any], Path, str, Path] | None:
    paths = sorted((request_path.parent / "supersessions").glob("*.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ResearchControlError(
            f"research Request has conflicting supersession artifacts: {request_path}"
        )
    if request["schema_version"] != _REQUEST_VERSION:
        raise ResearchControlError(
            "external-domain ResearchRequests do not use Volvence pre-A0 supersession"
        )
    path = paths[0]
    supersession = read_json(path)
    _validate_schema(config, supersession, _SUPERSESSION_VERSION)
    if supersession["supersession_id"] != _artifact_id(
        "research-request-supersession",
        supersession,
        "supersession_id",
    ):
        raise ResearchControlError(
            f"research Request supersession identity is invalid: {path}"
        )
    expected_name = str(supersession["supersession_id"]).partition(":")[2] + ".json"
    if path.name != expected_name:
        raise ResearchControlError(
            f"research Request supersession filename is not canonical: {path}"
        )
    expected_old_ref = {
        "artifact_id": request["request_id"],
        "artifact": {
            "locator": _portable_locator(config, request_path),
            "sha256": request_sha256,
        },
    }
    if supersession["superseded_request"] != expected_old_ref:
        raise ResearchControlError(
            "research Request supersession does not bind its exact predecessor"
        )
    replacement_ref = supersession["replacement_request"]
    replacement_path = _verify_content_ref(
        config,
        replacement_ref["artifact"],
        context="replacement research Request",
    )
    replacement, resolved_replacement, _ = _load_request(config, replacement_path)
    if replacement["request_id"] != replacement_ref["artifact_id"]:
        raise ResearchControlError(
            "research Request supersession replacement identity mismatch"
        )
    if resolved_replacement == request_path:
        raise ResearchControlError("research Request cannot supersede itself")
    before, after = _assert_source_refresh_pair(
        old_request=request,
        replacement_request=replacement,
    )
    change = supersession["change"]
    if change["before"] != before or change["after"] != after:
        raise ResearchControlError(
            "research Request supersession source-checkout snapshots are not exact"
        )
    _assert_unreviewed_request_can_be_superseded(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
        allow_existing_supersession=True,
    )
    return (
        supersession,
        path.resolve(strict=True),
        _sha256_file(path, context="research Request supersession"),
        resolved_replacement,
    )


def _assert_unreviewed_request_can_be_superseded(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
    allow_existing_supersession: bool = False,
) -> None:
    if request["schema_version"] != _REQUEST_VERSION:
        raise ResearchControlError(
            "only a Volvence ResearchRequest may use pre-A0 source refresh"
        )
    approval = _find_approval(
        config=config,
        request=request,
        request_path=request_path,
        request_sha256=request_sha256,
    )
    if approval is not None:
        raise ResearchControlError(
            "an A0-reviewed ResearchRequest cannot be superseded automatically"
        )
    if _event_paths(request_path):
        raise ResearchControlError(
            "a ResearchRequest with lifecycle events cannot be superseded automatically"
        )
    if _directive_paths(request_path):
        raise ResearchControlError(
            "a ResearchRequest with control directives cannot be superseded automatically"
        )
    if sorted((request_path.parent / "handoffs").glob("*.json")):
        raise ResearchControlError(
            "a ResearchRequest with a handoff cannot be superseded automatically"
        )
    if not allow_existing_supersession and sorted(
        (request_path.parent / "supersessions").glob("*.json")
    ):
        raise ResearchControlError(
            "research Request already has an immutable supersession"
        )


def _assert_source_refresh_pair(
    *,
    old_request: dict[str, Any],
    replacement_request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if old_request["schema_version"] != _REQUEST_VERSION or replacement_request[
        "schema_version"
    ] != _REQUEST_VERSION:
        raise ResearchControlError(
            "pre-A0 source refresh requires two Volvence ResearchRequests"
        )
    old_body = {
        key: value
        for key, value in old_request.items()
        if key not in {"request_id", "created_at"}
    }
    replacement_body = {
        key: value
        for key, value in replacement_request.items()
        if key not in {"request_id", "created_at"}
    }
    old_bindings = dict(old_body["bindings"])
    replacement_bindings = dict(replacement_body["bindings"])
    old_praxist = dict(old_bindings["praxist"])
    replacement_praxist = dict(replacement_bindings["praxist"])
    old_source = old_praxist.pop("source_checkout")
    replacement_source = replacement_praxist.pop("source_checkout")
    old_bindings["praxist"] = old_praxist
    replacement_bindings["praxist"] = replacement_praxist
    old_body["bindings"] = old_bindings
    replacement_body["bindings"] = replacement_bindings
    if old_body != replacement_body:
        raise ResearchControlError(
            "automatic Request supersession may change only the Praxist source checkout"
        )
    if not isinstance(old_source, dict) or not isinstance(replacement_source, dict):
        raise ResearchControlError(
            "automatic Request supersession requires source-checkout-backed Praxist inputs"
        )
    if old_source["root"] != replacement_source["root"]:
        raise ResearchControlError(
            "automatic Request supersession cannot switch Praxist source roots"
        )
    if old_source == replacement_source:
        raise ResearchControlError(
            "replacement ResearchRequest does not contain source-checkout drift"
        )
    return dict(old_source), dict(replacement_source)


def _find_approval(
    *,
    config: ForgeConfig,
    request: dict[str, Any],
    request_path: Path,
    request_sha256: str,
) -> tuple[dict[str, Any], Path, str] | None:
    paths = sorted((request_path.parent / "approvals").glob("*.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ResearchControlError(f"research Request has conflicting approval artifacts: {request_path}")
    path = paths[0]
    approval = read_json(path)
    _validate_schema(config, approval, _APPROVAL_VERSION)
    if approval["approval_id"] != _artifact_id(
        "research-approval",
        approval,
        "approval_id",
    ):
        raise ResearchControlError(f"research approval identity is invalid: {path}")
    if approval["request_id"] != request["request_id"]:
        raise ResearchControlError("research approval belongs to a different Request")
    if approval["request_sha256"] != request_sha256:
        raise ResearchControlError("research approval Request digest mismatch")
    if "execution_policy" in approval:
        from .research_portfolio import validate_research_execution_policy

        try:
            validate_research_execution_policy(
                config=config,
                execution_policy=approval["execution_policy"],
                request=request,
            )
        except ForgeError as exc:
            raise ResearchControlError(
                f"research approval execution policy is invalid: {exc}"
            ) from exc
    expected_name = str(approval["approval_id"]).partition(":")[2] + ".json"
    if path.name != expected_name:
        raise ResearchControlError(f"research approval filename is not canonical: {path}")
    return approval, path, _sha256_file(path, context="research approval")


def _status_from_parts(
    request: dict[str, Any],
    request_path: Path,
    approval_path: Path | None,
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    state: str,
    *,
    supersession: tuple[dict[str, Any], Path, str, Path] | None = None,
) -> ResearchControlStatus:
    run = _latest_run(events)
    supersession_artifact = supersession[0] if supersession is not None else None
    return ResearchControlStatus(
        request_id=str(request["request_id"]),
        task_id=str(request["task_id"]),
        state=state,
        request_path=request_path,
        approval_path=approval_path,
        latest_event_path=events[-1][1] if events else None,
        latest_event_sha256=events[-1][2] if events else None,
        run_id=str(run["run_id"]) if run is not None else None,
        run_dir=str(run["run_dir"]) if run is not None else None,
        monitor_command=(
            str(run["monitor_command"])
            if run is not None and run["monitor_command"] is not None
            else None
        ),
        supersession_path=supersession[1] if supersession is not None else None,
        replacement_request_id=(
            str(supersession_artifact["replacement_request"]["artifact_id"])
            if supersession_artifact is not None
            else None
        ),
        replacement_request_path=supersession[3] if supersession is not None else None,
    )


def _latest_run(
    events: tuple[tuple[dict[str, Any], Path, str], ...],
) -> dict[str, Any] | None:
    for event, _, _ in reversed(events):
        if event["run"] is not None:
            return event["run"]
    return None


def _latest_event(
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    kind: str,
) -> dict[str, Any] | None:
    for event, _, _ in reversed(events):
        if event["kind"] == kind:
            return event
    return None


def _directive_event(
    events: tuple[tuple[dict[str, Any], Path, str], ...],
    directive_id: str,
    kind: str,
) -> dict[str, Any] | None:
    for event, _, _ in reversed(events):
        if event.get("directive_id") == directive_id and event["kind"] == kind:
            return event
    return None


def _load_request(
    config: ForgeConfig,
    request_path: Path,
) -> tuple[dict[str, Any], Path, str]:
    resolved = _resolve_regular_file(request_path, context="research Request")
    request = read_json(resolved)
    version = request.get("schema_version")
    if version not in {_REQUEST_VERSION, _EXTERNAL_REQUEST_VERSION}:
        raise ResearchControlError(f"unsupported research Request schema_version: {version!r}")
    _validate_schema(config, request, str(version))
    if request["request_id"] != _artifact_id(
        "research-request",
        request,
        "request_id",
    ):
        raise ResearchControlError("research request_id does not match its canonical payload")
    expected_digest = str(request["request_id"]).partition(":")[2]
    if resolved.name != "request.json" or resolved.parent.name != expected_digest:
        raise ResearchControlError("research Request is not stored at its canonical registry path")
    if resolved.parent.parent.name != request["task_id"]:
        raise ResearchControlError("research Request registry task directory does not match task_id")
    artifacts_root = config.paths.artifacts_root.resolve(strict=False)
    control_root = (artifacts_root / "research_control").resolve(strict=False)
    if not resolved.is_relative_to(control_root):
        raise ResearchControlError("research Request must be stored below artifacts/research_control")
    return request, resolved, _sha256_file(resolved, context="research Request")


def _verify_request_bindings(config: ForgeConfig, request: dict[str, Any]) -> None:
    task: dict[str, Any] | None = None
    if request["schema_version"] == _REQUEST_VERSION:
        task_ref = request["bindings"]["research_task"]
        task_path = _verify_content_ref(config, task_ref, context="research task manifest")
        task = validate_research_task(config=config, task_path=task_path)
        expected_task_fields = {
            "task_id": task["task_id"],
            "claim_id": task["claim_id"],
            "owner": task["owner"],
        }
        for name, value in expected_task_fields.items():
            if request[name] != value:
                raise ResearchControlError(f"research Request {name} no longer matches its Task")
    elif request["schema_version"] == _EXTERNAL_REQUEST_VERSION:
        _verify_external_request_bindings(config, request)
    else:  # pragma: no cover - request loading rejects unknown versions first.
        raise ResearchControlError(
            f"unsupported research Request schema_version: {request['schema_version']!r}"
        )

    expected_project = request["bindings"]["task_project"]
    current_project = _snapshot_task_project(Path(expected_project["root"]))
    if current_project != expected_project:
        raise ResearchControlError("Praxist task project changed after Request submission")
    if task is not None:
        if current_project["task_id"] != task["praxist"]["task_project_id"]:
            raise ResearchControlError("Praxist task project task_id no longer matches the Task")
        if current_project["manifest_sha256"] != task["praxist"]["task_project_manifest_sha256"]:
            raise ResearchControlError("Praxist task project digest no longer matches the Task")

    executable_ref = request["bindings"]["praxist"]["executable"]
    executable = _verify_content_ref(config, executable_ref, context="Praxist executable")
    _resolve_executable(executable)
    current_source = _snapshot_praxist_source_checkout(executable)
    if current_source != request["bindings"]["praxist"]["source_checkout"]:
        raise ResearchRequestBindingDriftError(
            "bindings.praxist.source_checkout",
            "Praxist source checkout changed after Request submission",
        )

    profile = request["launch"]["profile"]
    if profile["config_file"] is not None:
        _verify_content_ref(config, profile["config_file"], context="Praxist config file")
    for index, evidence_ref in enumerate(request["trigger"]["evidence"]):
        _verify_content_ref(
            config,
            evidence_ref,
            context=f"research trigger evidence {index}",
        )
    _validate_launch_profile_values(
        agent_system=profile["agent_system"],
        runtime=profile["runtime"],
        codex_native=profile["codex_native"],
        model_provider=profile["model_provider"],
        model=profile["model"],
        strategy=profile["strategy"],
        cohort=profile["cohort"],
        generations=profile["generations"],
        startup_timeout_seconds=profile["startup_timeout_seconds"],
    )
    run_dir = Path(request["launch"]["run_dir"])
    if run_dir.name != request["launch"]["run_id"] or not _RUN_ID_RE.fullmatch(run_dir.name):
        raise ResearchControlError("research Request has an invalid deterministic run identity")
    source_checkout = request["bindings"]["praxist"]["source_checkout"]
    if source_checkout is not None and run_dir.is_relative_to(Path(source_checkout["root"])):
        raise ResearchControlError("Praxist run output may not be inside its source checkout")


def _load_external_descriptor(
    config: ForgeConfig,
    descriptor_path: Path,
) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_regular_file(descriptor_path, context="external research descriptor")
    descriptor = read_json(resolved)
    _validate_schema(
        config,
        descriptor,
        _EXTERNAL_DESCRIPTOR_VERSION,
        schema_name=EXTERNAL_SCHEMA_NAME,
    )
    if descriptor["descriptor_id"] != _artifact_id(
        "external-research-descriptor",
        descriptor,
        "descriptor_id",
    ):
        raise ResearchControlError(
            "external research descriptor_id does not match its canonical payload"
        )
    return descriptor, resolved


def _resolve_external_descriptor_bindings(
    config: ForgeConfig,
    descriptor: dict[str, Any],
    *,
    descriptor_path: Path,
) -> dict[str, Any]:
    base = descriptor_path.parent
    raw_bindings = descriptor["bindings"]
    if raw_bindings["intent"] != raw_bindings["budget"]:
        raise ResearchControlError(
            "Foundry launch budget must bind the same exact Intent via intent:/launch"
        )
    intent_ref = _external_absolute_ref(
        raw_bindings["intent"],
        base=base,
        context="Foundry Research Lab Intent",
    )
    intent_path = Path(intent_ref["locator"])
    intent = _read_json_file(intent_path, context="Foundry Research Lab Intent")
    _validate_schema(
        config,
        intent,
        "foundry-research-lab-intent.v1",
        schema_name=EXTERNAL_SCHEMA_NAME,
    )
    return _finish_external_descriptor_resolution(
        descriptor,
        descriptor_path=descriptor_path,
        intent=intent,
        intent_ref=intent_ref,
    )


def _finish_external_descriptor_resolution(
    descriptor: dict[str, Any],
    *,
    descriptor_path: Path,
    intent: dict[str, Any],
    intent_ref: dict[str, str],
) -> dict[str, Any]:
    identity_payload = {
        key: value
        for key, value in intent.items()
        if key not in {"intent_id", "created_at"}
    }
    expected_intent_id = f"rli_{sha256_text(canonical_json(identity_payload))[:16]}"
    if intent["intent_id"] != expected_intent_id:
        raise ResearchControlError("Foundry Research Lab Intent content does not match intent_id")
    domain = descriptor["domain"]
    if domain != {
        "domain_id": "foundry",
        "task_id": intent["opportunity_id"],
        "intent_id": intent["intent_id"],
    }:
        raise ResearchControlError("external descriptor domain does not match the Foundry Intent")

    task_binding = intent["task_project"]
    task_root = _resolve_directory(
        Path(task_binding["root"]),
        context="Foundry Praxist task project",
    )
    task_yaml = _resolve_regular_file(
        task_root / "task.yaml",
        context="Foundry Praxist task.yaml",
    )
    if _sha256_file(task_yaml, context="Foundry Praxist task.yaml") != task_binding["task_yaml_sha256"]:
        raise ResearchControlError("Foundry Praxist task.yaml changed after Intent publication")
    project = _snapshot_task_project(task_root)
    if project["task_id"] != task_binding["task_id"]:
        raise ResearchControlError("Foundry Intent task_id no longer matches Praxist task.yaml")

    evidence: list[dict[str, str]] = []
    for index, raw_ref in enumerate(intent["trigger"]["evidence_refs"]):
        normalized = _external_absolute_ref(
            raw_ref,
            base=Path(intent_ref["locator"]).parent,
            context=f"Foundry trigger evidence {index}",
        )
        evidence.append({**normalized, "evidence_class": raw_ref["evidence_class"]})

    profile = intent["launch"]
    _validate_launch_profile_values(
        agent_system=profile["agent_system"],
        runtime=profile["runtime"],
        codex_native=profile["codex_native"],
        model_provider=profile["model_provider"],
        model=profile["model"],
        strategy=profile["strategy"],
        cohort=profile["cohort"],
        generations=profile["generations"],
        startup_timeout_seconds=profile["startup_timeout_seconds"],
    )
    control = descriptor["control"]
    base = descriptor_path.parent
    executable = _resolve_executable(
        _external_path(
            str(control["praxist_executable"]),
            base=base,
            context="Praxist executable",
        )
    )
    run_dir = _external_path(
        str(control["run_dir"]),
        base=base,
        context="Praxist run directory",
        must_exist=False,
    )
    config_file = (
        _external_absolute_ref(
            control["config_file"],
            base=base,
            context="Praxist config file",
        )
        if control["config_file"] is not None
        else None
    )
    _safe_relative_locator(
        descriptor["result_policy"]["result_locator"],
        context="external result locator",
    )
    return {
        "intent": intent,
        "intent_ref": intent_ref,
        "evidence": evidence,
        "task_project": project,
        "executable": executable,
        "run_dir": run_dir,
        "config_file": config_file,
    }


def _verify_external_request_bindings(config: ForgeConfig, request: dict[str, Any]) -> None:
    descriptor_path = _verify_content_ref(
        config,
        request["bindings"]["external_descriptor"],
        context="external research descriptor",
    )
    descriptor, resolved_descriptor = _load_external_descriptor(config, descriptor_path)
    resolved = _resolve_external_descriptor_bindings(
        config,
        descriptor,
        descriptor_path=resolved_descriptor,
    )
    intent = resolved["intent"]
    domain = descriptor["domain"]
    expected_external = {
        "descriptor_id": descriptor["descriptor_id"],
        "adapter_id": descriptor["adapter"]["adapter_id"],
        "intent_schema_version": descriptor["adapter"]["intent_schema_version"],
        "domain_id": domain["domain_id"],
        "task_id": domain["task_id"],
        "intent_id": domain["intent_id"],
        "ownership": _foundry_ownership(),
        "result_policy": descriptor["result_policy"],
    }
    if request["external_domain"] != expected_external:
        raise ResearchControlError("external Request no longer matches its domain descriptor")
    expected_identity = {
        "task_id": _external_control_task_id(domain),
        "claim_id": f"foundry:{intent['intent_id']}",
        "owner": "external-domain:foundry",
        "objective": intent["objective"],
    }
    for name, value in expected_identity.items():
        if request[name] != value:
            raise ResearchControlError(
                f"external research Request {name} no longer matches its descriptor"
            )
    expected_refs = {
        "external_intent": resolved["intent_ref"],
        "external_budget": resolved["intent_ref"],
        "external_evidence": resolved["evidence"],
        "task_project": resolved["task_project"],
    }
    for name, value in expected_refs.items():
        if request["bindings"][name] != value:
            raise ResearchControlError(f"external research Request {name} binding changed")
    expected_trigger_evidence = [
        {"locator": ref["locator"], "sha256": ref["sha256"]}
        for ref in resolved["evidence"]
    ]
    if request["trigger"]["evidence"] != expected_trigger_evidence:
        raise ResearchControlError("external research trigger evidence changed")
    expected_profile = {"config_file": resolved["config_file"], **intent["launch"]}
    if request["launch"]["profile"] != expected_profile:
        raise ResearchControlError("external research launch profile changed")
    expected_run_dir = str(_normalize_bound_run_dir(resolved["run_dir"]))
    if request["launch"]["run_dir"] != expected_run_dir:
        raise ResearchControlError("external research run directory changed")


def _external_control_task_id(domain: dict[str, Any]) -> str:
    return f"external_{sha256_text(canonical_json(domain))[:16]}"


def _foundry_ownership() -> dict[str, str]:
    return {
        "research_intent": "foundry",
        "budget": "foundry",
        "evidence_classification": "foundry",
        "result_adoption": "foundry",
        "human_application": "foundry",
    }


def _external_authority() -> dict[str, bool]:
    return {
        "external_actions_allowed": False,
        "foundry_checkout_write_allowed": False,
        "foundry_ledger_write_allowed": False,
        "direct_apply_allowed": False,
        "productzero_start_allowed": False,
        "volvence_promotion_eligible": False,
        "modification_gate_applicable": False,
        "runtime_wiring_applicable": False,
    }


def _foundry_consumer_permissions() -> dict[str, Any]:
    return {
        "allowed_operations": ["import_simulation_handoff"],
        "prohibited_operations": [
            "approve_research_request",
            "reconcile_research_control",
            "start_praxist",
            "import_volvence_candidate",
            "modify_runtime_wiring",
        ],
    }


def _foundry_handoff_authority() -> dict[str, bool]:
    return {
        **_external_authority(),
        "foundry_approve_allowed": False,
        "foundry_reconcile_allowed": False,
        "foundry_praxist_start_allowed": False,
    }


def _external_absolute_ref(
    content_ref: dict[str, Any],
    *,
    base: Path,
    context: str,
) -> dict[str, str]:
    path = _external_path(str(content_ref["locator"]), base=base, context=context)
    resolved = _resolve_regular_file(path, context=context)
    actual = _sha256_file(resolved, context=context)
    if actual != content_ref["sha256"]:
        raise ResearchControlError(
            f"{context} digest mismatch: declared {content_ref['sha256']}, actual {actual}"
        )
    return {"locator": str(resolved), "sha256": actual}


def _external_path(
    locator: str,
    *,
    base: Path,
    context: str,
    must_exist: bool = True,
) -> Path:
    candidate = Path(locator).expanduser()
    if not candidate.is_absolute():
        relative = _safe_relative_locator(locator, context=context)
        candidate = base / Path(*relative.parts)
    if candidate.is_symlink():
        raise ResearchControlError(f"{context} may not be a symlink: {candidate}")
    try:
        return candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise ResearchControlError(f"missing {context}: {candidate}") from exc


def _safe_relative_locator(locator: str, *, context: str) -> PurePosixPath:
    relative = PurePosixPath(locator)
    if (
        not locator
        or "\\" in locator
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise ResearchControlError(f"unsafe {context}: {locator!r}")
    return relative


def _normalize_bound_run_dir(run_dir: Path) -> Path:
    expanded = run_dir.expanduser()
    if not expanded.is_absolute():
        raise ResearchControlError("Praxist run directory must be absolute")
    resolved = expanded.resolve(strict=False)
    if not _RUN_ID_RE.fullmatch(resolved.name):
        raise ResearchControlError(
            "Praxist run directory basename must be a deterministic safe run_id"
        )
    return resolved


def _snapshot_task_project(task_project_path: Path) -> dict[str, Any]:
    root = _resolve_directory(task_project_path, context="Praxist task project")
    descriptor_path = root / "task.yaml"
    descriptor = _read_yaml_mapping(descriptor_path)
    plugins = descriptor.get("praxist_plugins")
    if not isinstance(plugins, dict):
        raise ResearchControlError("Praxist task.yaml must contain praxist_plugins")
    workflow = plugins.get("workflow")
    stage = workflow.get("stage") if isinstance(workflow, dict) else plugins.get("workflow_stage")
    if stage != "workflow_stage:research_loop":
        raise ResearchControlError(
            "Praxist task.yaml must select workflow_stage:research_loop"
        )
    task_id = descriptor.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ResearchControlError("Praxist task.yaml must contain a non-empty task_id")

    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in _iter_task_project_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ResearchControlError(f"cannot read Praxist task project file {path}: {exc}") from exc
        file_digest = sha256_bytes(content)
        files.append({"path": relative, "sha256": file_digest, "bytes": len(content)})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    if not files:
        raise ResearchControlError("Praxist task project manifest cannot be empty")
    return {
        "root": str(root),
        "task_id": task_id.strip(),
        "manifest_schema_version": "task_project_manifest.v1",
        "manifest_sha256": digest.hexdigest(),
        "snapshot_sha256": sha256_text(canonical_json(files)),
        "file_count": len(files),
        "files": files,
    }


def _iter_task_project_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        relative_parts = () if relative_dir == Path(".") else relative_dir.parts
        retained_dirs: list[str] = []
        for dirname in sorted(dirnames):
            child_parts = (*relative_parts, dirname)
            if dirname in _SKIPPED_MANIFEST_DIR_NAMES:
                continue
            if len(child_parts) == 1 and dirname in _SKIPPED_MANIFEST_TOP_LEVEL_DIRS:
                continue
            if child_parts in _SKIPPED_MANIFEST_RELATIVE_DIRS:
                continue
            child = current / dirname
            if child.is_symlink():
                raise ResearchControlError(f"Praxist task project may not contain symlink {child}")
            retained_dirs.append(dirname)
        dirnames[:] = retained_dirs
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise ResearchControlError(f"Praxist task project may not contain symlink {path}")
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ResearchControlError(f"Praxist task project file escapes its root: {path}")
            yield path


def _snapshot_praxist_source_checkout(executable: Path) -> dict[str, Any] | None:
    root = _detect_praxist_source_checkout(executable)
    if root is None:
        return None
    paths = [root / "pyproject.toml"]
    for dirpath, dirnames, filenames in os.walk(root / "praxist", topdown=True, followlinks=False):
        current = Path(dirpath)
        retained: list[str] = []
        for dirname in sorted(dirnames):
            child = current / dirname
            if dirname in {"__pycache__", ".pytest_cache"}:
                continue
            if child.is_symlink():
                raise ResearchControlError(f"Praxist source package may not contain symlink {child}")
            retained.append(dirname)
        dirnames[:] = retained
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise ResearchControlError(f"Praxist source package may not contain symlink {path}")
            if path.is_file():
                paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ResearchControlError(f"cannot read Praxist source file {path}: {exc}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {
        "root": str(root),
        "tree_sha256": digest.hexdigest(),
        "file_count": len(paths),
    }


def _detect_praxist_source_checkout(executable: Path) -> Path | None:
    for parent in (executable.parent, *executable.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "praxist").is_dir():
            return parent.resolve()
    return None


def _normalize_fresh_run_dir(run_dir: Path) -> Path:
    expanded = run_dir.expanduser()
    if not expanded.is_absolute():
        raise ResearchControlError("Praxist run directory must be absolute")
    if expanded.is_symlink():
        raise ResearchControlError(f"Praxist run directory may not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=False)
    if not _RUN_ID_RE.fullmatch(resolved.name):
        raise ResearchControlError(
            "Praxist run directory basename must be a deterministic safe run_id"
        )
    if resolved.exists():
        if not resolved.is_dir():
            raise ResearchControlError(f"Praxist run path is not a directory: {resolved}")
        try:
            nonempty = next(resolved.iterdir(), None) is not None
        except OSError as exc:
            raise ResearchControlError(f"cannot inspect Praxist run directory {resolved}: {exc}") from exc
        if nonempty:
            raise ResearchControlError(f"fresh Praxist run directory is not empty: {resolved}")
    return resolved


def _assert_run_dir_available(request: dict[str, Any]) -> None:
    _normalize_fresh_run_dir(Path(request["launch"]["run_dir"]))


def _resolve_executable(path: Path) -> Path:
    resolved = _resolve_regular_file(path, context="Praxist executable")
    if not os.access(resolved, os.X_OK):
        raise ResearchControlError(f"Praxist executable is not executable: {resolved}")
    return resolved


def _resolve_regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ResearchControlError(f"{context} may not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchControlError(f"missing {context}: {expanded}") from exc
    if not resolved.is_file():
        raise ResearchControlError(f"{context} must be a regular file: {resolved}")
    return resolved


def _resolve_directory(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ResearchControlError(f"{context} may not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchControlError(f"missing {context}: {expanded}") from exc
    if not resolved.is_dir():
        raise ResearchControlError(f"{context} must be a directory: {resolved}")
    return resolved


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchControlError(f"missing Praxist task descriptor: {path}") from exc
    except yaml.YAMLError as exc:
        raise ResearchControlError(f"invalid YAML in Praxist task descriptor {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResearchControlError(f"Praxist task descriptor must contain a mapping: {path}")
    return raw


def _read_json_file(path: Path, *, context: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchControlError(f"missing {context}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchControlError(f"invalid JSON in {context} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResearchControlError(f"{context} must contain a JSON object: {path}")
    return raw


def _content_ref(config: ForgeConfig, path: Path, *, context: str) -> dict[str, str]:
    resolved = _resolve_regular_file(path, context=context)
    return {
        "locator": _portable_locator(config, resolved),
        "sha256": _sha256_file(resolved, context=context),
    }


def _absolute_content_ref(path: Path, *, context: str) -> dict[str, str]:
    resolved = _resolve_regular_file(path, context=context)
    return {
        "locator": str(resolved),
        "sha256": _sha256_file(resolved, context=context),
    }


def _verify_content_ref(
    config: ForgeConfig,
    content_ref: dict[str, Any],
    *,
    context: str,
) -> Path:
    path = _content_ref_path(content_ref, repo_root=config.paths.repo_root)
    resolved = _resolve_regular_file(path, context=context)
    actual = _sha256_file(resolved, context=context)
    if actual != content_ref["sha256"]:
        raise ResearchControlError(
            f"{context} digest mismatch: declared {content_ref['sha256']}, actual {actual}"
        )
    return resolved


def _content_ref_path(
    content_ref: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> Path:
    locator = str(content_ref["locator"])
    path = Path(locator).expanduser()
    if path.is_absolute():
        return path
    if repo_root is None:
        raise ResearchControlError(f"relative content locator needs a repository root: {locator!r}")
    relative = PurePosixPath(locator)
    if not locator or "\\" in locator or ".." in relative.parts or "." in relative.parts:
        raise ResearchControlError(f"unsafe content locator: {locator!r}")
    return repo_root / Path(*relative.parts)


def _portable_locator(config: ForgeConfig, path: Path) -> str:
    repo_root = config.paths.repo_root.resolve()
    return path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)


def _request_executable(request: dict[str, Any]) -> Path:
    ref = request["bindings"]["praxist"]["executable"]
    locator = Path(str(ref["locator"])).expanduser()
    if not locator.is_absolute():
        raise ResearchControlError(
            "Praxist executable locator must be absolute when executing a Request"
        )
    return locator


def _parse_json_object(execution: CommandExecution, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(execution.stdout)
    except json.JSONDecodeError as exc:
        raise ResearchControlError(f"{context} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ResearchControlError(f"{context} must return one JSON object")
    return value


def _parse_json_array(execution: CommandExecution, *, context: str) -> list[Any]:
    try:
        value = json.loads(execution.stdout)
    except json.JSONDecodeError as exc:
        raise ResearchControlError(f"{context} returned invalid JSON") from exc
    if not isinstance(value, list):
        raise ResearchControlError(f"{context} must return one JSON array")
    return value


def _command_succeeded(execution: CommandExecution) -> bool:
    return execution.returncode == 0 and execution.timed_out is False


def _command_receipt(phase: str, execution: CommandExecution) -> dict[str, Any]:
    return {
        "phase": phase,
        "argv": list(execution.argv),
        "exit_code": execution.returncode,
        "timed_out": execution.timed_out,
        "stdout_sha256": sha256_text(execution.stdout),
        "stderr_sha256": sha256_text(execution.stderr),
    }


def _command_failure_detail(label: str, execution: CommandExecution) -> str:
    if execution.timed_out:
        return f"{label} timed out; stderr_sha256={sha256_text(execution.stderr)}"
    return (
        f"{label} exited with code {execution.returncode}; "
        f"stderr_sha256={sha256_text(execution.stderr)}"
    )


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _normalized_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResearchControlError(f"{context} must be a non-empty path")
    return str(Path(value).expanduser().resolve(strict=False))


def _nullable_nonnegative_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResearchControlError(f"{context} must be null or a non-negative integer")
    return value


def _nullable_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _nonempty(value: str, message: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ResearchControlError(message)
    return stripped


def _validate_launch_profile_values(
    *,
    agent_system: str | None,
    runtime: str | None,
    codex_native: bool,
    model_provider: str | None,
    model: str | None,
    strategy: str,
    cohort: int | None,
    generations: int | None,
    startup_timeout_seconds: int,
) -> None:
    if agent_system not in {None, "claude_sdk", "codex_sdk"}:
        raise ResearchControlError("agent_system must be claude_sdk, codex_sdk, or omitted")
    if strategy not in {"auto", "mixed", "explore", "exploit"}:
        raise ResearchControlError(f"unsupported Praxist strategy: {strategy!r}")
    for name, value in (("cohort", cohort), ("generations", generations)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ResearchControlError(f"{name} must be a positive integer when supplied")
    if (
        not isinstance(startup_timeout_seconds, int)
        or isinstance(startup_timeout_seconds, bool)
        or not 0 <= startup_timeout_seconds <= 300
    ):
        raise ResearchControlError("startup_timeout_seconds must be an integer from 0 to 300")
    for name, value in (("runtime", runtime), ("model_provider", model_provider), ("model", model)):
        if value is not None and not value.strip():
            raise ResearchControlError(f"{name} must be non-empty when supplied")
    if model is None:
        raise ResearchControlError("an explicit model is required for an approval-bound research run")
    if codex_native:
        if agent_system not in {None, "codex_sdk"}:
            raise ResearchControlError("codex_native is incompatible with claude_sdk")
        if runtime not in {None, "agent_runtime:codex_sdk"}:
            raise ResearchControlError("codex_native requires agent_runtime:codex_sdk")
        if model_provider not in {None, "model_provider:openai_compatible"}:
            raise ResearchControlError(
                "codex_native cannot be combined with a non-OpenAI model provider"
            )
    elif agent_system is None or runtime is None or model_provider is None:
        raise ResearchControlError(
            "non-Codex-native research requires explicit agent_system, runtime, and model_provider"
        )
    builtin_runtime_agent = {
        "agent_runtime:claude_sdk": "claude_sdk",
        "agent_runtime:codex_sdk": "codex_sdk",
    }
    if runtime in builtin_runtime_agent and agent_system != builtin_runtime_agent[runtime]:
        raise ResearchControlError(
            f"{runtime} requires agent_system={builtin_runtime_agent[runtime]}"
        )


def _artifact_id(prefix: str, payload: dict[str, Any], identity_field: str) -> str:
    identity_payload = {
        key: value
        for key, value in payload.items()
        if key not in {identity_field, "created_at"}
    }
    return f"{prefix}:{sha256_text(canonical_json(identity_payload))}"


def _validate_schema(
    config: ForgeConfig,
    payload: dict[str, Any],
    expected_version: str,
    *,
    schema_name: str = SCHEMA_NAME,
) -> None:
    SchemaStore(config.paths.forge_root / "schemas").validate(payload, schema_name)
    if payload.get("schema_version") != expected_version:
        raise ResearchControlError(
            f"expected schema_version {expected_version!r}, got {payload.get('schema_version')!r}"
        )


def _write_immutable_artifact(
    *,
    config: ForgeConfig,
    destination: Path,
    payload: dict[str, Any],
    expected_version: str,
    identity_field: str,
    schema_name: str = SCHEMA_NAME,
) -> None:
    artifacts_root = config.paths.artifacts_root.resolve(strict=False)
    target = destination.expanduser().resolve(strict=False)
    control_root = (artifacts_root / "research_control").resolve(strict=False)
    if not target.is_relative_to(control_root):
        raise ResearchControlError(
            "research-control artifacts may only be written below artifacts/research_control"
        )
    if target.exists():
        existing = read_json(target)
        _validate_schema(config, existing, expected_version, schema_name=schema_name)
        if existing[identity_field] != payload[identity_field]:
            raise ResearchControlError(f"refusing to overwrite another immutable artifact: {target}")
        existing_body = {
            key: value
            for key, value in existing.items()
            if key not in {identity_field, "created_at"}
        }
        payload_body = {
            key: value
            for key, value in payload.items()
            if key not in {identity_field, "created_at"}
        }
        if existing_body != payload_body:
            raise ResearchControlError(f"refusing to overwrite changed immutable artifact: {target}")
        return
    _write_create_only_json(target, payload)


def _write_create_only_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ResearchControlError(f"refusing to overwrite create-only artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise ResearchControlError(f"cannot open artifact directory for fsync {path}: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ResearchControlError(f"cannot fsync artifact directory {path}: {exc}") from exc
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Forge control currently targets POSIX hosts.
        raise ResearchControlError("research reconciliation requires POSIX file locking") from exc
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ResearchControlError(f"cannot open research control lock {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _request_paths(config: ForgeConfig) -> tuple[Path, ...]:
    root = config.paths.artifacts_root / "research_control"
    return tuple(sorted(root.glob("*/*/request.json")))


def _event_paths(request_path: Path) -> tuple[Path, ...]:
    return tuple(sorted((request_path.parent / "events").glob("*.json")))


def _directive_paths(request_path: Path) -> tuple[Path, ...]:
    return tuple(sorted((request_path.parent / "directives").glob("*.json")))


def _sha256_file(path: Path, *, context: str) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ResearchControlError(f"cannot read {context} at {path}: {exc}") from exc
