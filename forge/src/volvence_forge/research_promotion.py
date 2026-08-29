"""Offline contracts for importing and authorizing research candidates.

The module deliberately stops at immutable authorization receipts.  It never
imports a Volvence runtime wheel, invokes ModificationGate, applies candidate
files, or changes runtime wiring.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    SchemaStore,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
    utc_now,
)

SCHEMA_NAME = "research_promotion.schema.json"

_MANDATORY_PROTECTED_ROOTS = (
    ".git",
    ".github",
    "AGENTS.md",
    "forge",
    "docs/specs",
    "docs/DATA_CONTRACT.md",
    "pyproject.toml",
    "scripts",
    "tests",
    "packages/vz-cognition/src/volvence_zero/credit",
)
_RESULT_SUMMARY_NAMES = {
    "summary.json",
    "evaluation_summary.json",
    "eval_summary.json",
    "tiered_eval_summary.json",
    "result_summary.json",
}


class ResearchPipelineError(ForgeError):
    """Raised when a research-promotion boundary is malformed or unsafe."""


@dataclass(frozen=True)
class CandidateImportResult:
    candidate_id: str
    candidate_path: Path


@dataclass(frozen=True)
class PromotionReceiptResult:
    receipt_id: str
    receipt_path: Path
    outcome: str
    resulting_wiring: str


def validate_research_task(*, config: ForgeConfig, task_path: Path) -> dict[str, Any]:
    """Validate a frozen Volvence-owned research task contract."""

    task, resolved, _ = _load_artifact(config, task_path, "forge-research-task.v1")
    repo_root = config.paths.repo_root.resolve()
    if not resolved.is_relative_to(repo_root):
        raise ResearchPipelineError("research task manifest must be stored inside the Volvence repository")
    task_relative = resolved.relative_to(repo_root).as_posix()

    baseline_path = _verify_repo_content_ref(config, task["baseline"], context="task baseline")
    protocol_path = _verify_repo_content_ref(
        config,
        task["validation"]["formal_protocol"],
        context="formal validation protocol",
    )

    editable_roots = _normalize_roots(task["sandbox"]["editable_roots"], context="editable root")
    declared_protected = _normalize_roots(task["sandbox"]["protected_roots"], context="protected root")
    protected_roots = tuple(dict.fromkeys((*declared_protected, *_MANDATORY_PROTECTED_ROOTS)))
    for editable in editable_roots:
        for protected in protected_roots:
            if _paths_overlap(editable, protected):
                raise ResearchPipelineError(
                    f"research editable root {editable!r} overlaps protected root {protected!r}"
                )

    protected_contract_paths = (
        task_relative,
        baseline_path.relative_to(repo_root).as_posix(),
        protocol_path.relative_to(repo_root).as_posix(),
    )
    for contract_path in protected_contract_paths:
        if any(_is_under(contract_path, editable) for editable in editable_roots):
            raise ResearchPipelineError(
                f"research contract input {contract_path!r} must remain outside every editable root"
            )

    validation = task["validation"]
    if validation["development_evaluator_id"] == validation["formal_validator_id"]:
        raise ResearchPipelineError("development evaluator and formal validator must be different identities")
    shadow_checks = set(validation["shadow_required_checks"])
    active_checks = set(validation["active_required_checks"])
    if not shadow_checks.issubset(active_checks):
        raise ResearchPipelineError("active_required_checks must include every shadow_required_check")
    if task["release"]["mode"] == "runtime_wiring" and (not shadow_checks or not active_checks):
        raise ResearchPipelineError("runtime_wiring tasks require non-empty SHADOW and ACTIVE check sets")
    return task


def import_praxist_candidate(
    *,
    config: ForgeConfig,
    task_path: Path,
    handoff_path: Path,
    run_dir: Path,
    output_path: Path | None = None,
) -> CandidateImportResult:
    """Import one exact Praxist candidate without granting deployment authority."""

    task = validate_research_task(config=config, task_path=task_path)
    task_resolved = _resolve_regular_file(task_path, context="research task manifest")
    task_sha256 = _sha256_file(task_resolved, context="research task manifest")
    handoff, handoff_resolved, handoff_sha256 = _load_artifact(
        config,
        handoff_path,
        "forge-praxist-candidate-handoff.v1",
    )
    run_root = _normalize_run_root(run_dir)
    _validate_handoff(config=config, task=task, handoff=handoff, run_root=run_root)

    payload: dict[str, Any] = {
        "schema_version": "forge-research-candidate.v1",
        "task_id": task["task_id"],
        "claim_id": task["claim_id"],
        "owner": task["owner"],
        "capability_axes": list(task["capability_axes"]),
        "source_base_revision": task["source_base_revision"],
        "baseline": dict(task["baseline"]),
        "inputs": {
            "task_manifest": {
                "locator": _portable_locator(config, task_resolved),
                "sha256": task_sha256,
            },
            "praxist_handoff": {
                "locator": _portable_locator(config, handoff_resolved),
                "sha256": handoff_sha256,
            },
        },
        "source": {
            "kind": "praxist",
            "run_root": str(run_root),
            "run_id": handoff["run_id"],
            "generation_id": handoff["generation_id"],
            "variant_id": handoff["candidate"]["variant_id"],
            "refs": _copy_refs(handoff["refs"]),
            "files": _copy_candidate_files(handoff["candidate"]["files"]),
        },
        "research": {
            "maturity": "candidate_sealed",
            "source_retention": dict(handoff["candidate"]["retention"]),
            "parent_candidate_ids": list(handoff["candidate"]["parent_candidate_ids"]),
        },
        "release": {
            "mode": task["release"]["mode"],
            "target": task["release"]["target"],
            "gate_decision": "not_evaluated",
            "wiring_level": "disabled",
        },
        "authority": {
            "praxist_retention_is_release_authority": False,
            "evaluation_is_learning_source": False,
            "production_promotion_authorized": False,
        },
        "created_at": utc_now(),
    }
    payload["candidate_id"] = _artifact_id("research-candidate", payload, "candidate_id")
    _validate_schema(config, payload, "forge-research-candidate.v1")

    digest = str(payload["candidate_id"]).partition(":")[2]
    default_path = (
        config.paths.artifacts_root
        / "research_promotion"
        / str(task["task_id"])
        / digest
        / "candidate.json"
    )
    destination = _resolve_output_path(config, output_path or default_path)
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=payload,
        expected_version="forge-research-candidate.v1",
        identity_field="candidate_id",
    )
    return CandidateImportResult(candidate_id=str(payload["candidate_id"]), candidate_path=destination)


def validate_research_candidate(
    *,
    config: ForgeConfig,
    task_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    """Revalidate candidate identity, cross-bindings, and every referenced Praxist byte."""

    task = validate_research_task(config=config, task_path=task_path)
    task_resolved = _resolve_regular_file(task_path, context="research task manifest")
    task_sha256 = _sha256_file(task_resolved, context="research task manifest")
    candidate, _, _ = _load_artifact(config, candidate_path, "forge-research-candidate.v1")
    if candidate["candidate_id"] != _artifact_id("research-candidate", candidate, "candidate_id"):
        raise ResearchPipelineError("research candidate_id does not match its canonical payload")

    task_ref_path = _verify_external_content_ref(
        config,
        candidate["inputs"]["task_manifest"],
        context="candidate task manifest",
    )
    if task_ref_path != task_resolved or candidate["inputs"]["task_manifest"]["sha256"] != task_sha256:
        raise ResearchPipelineError("candidate task manifest binding does not match the supplied Task")

    expected_fields = {
        "task_id": task["task_id"],
        "claim_id": task["claim_id"],
        "owner": task["owner"],
        "capability_axes": task["capability_axes"],
        "source_base_revision": task["source_base_revision"],
        "baseline": task["baseline"],
    }
    for name, expected in expected_fields.items():
        if candidate[name] != expected:
            raise ResearchPipelineError(f"candidate {name} does not match the supplied Task")
    if candidate["release"]["mode"] != task["release"]["mode"]:
        raise ResearchPipelineError("candidate release mode does not match the supplied Task")
    if candidate["release"]["target"] != task["release"]["target"]:
        raise ResearchPipelineError("candidate release target does not match the supplied Task")

    handoff_path = _verify_external_content_ref(
        config,
        candidate["inputs"]["praxist_handoff"],
        context="candidate Praxist handoff",
    )
    handoff, _, _ = _load_artifact(
        config,
        handoff_path,
        "forge-praxist-candidate-handoff.v1",
    )
    run_root = _normalize_run_root(Path(candidate["source"]["run_root"]))
    _validate_handoff(config=config, task=task, handoff=handoff, run_root=run_root)
    _assert_candidate_matches_handoff(candidate, handoff, run_root)
    return candidate


def authorize_research_candidate(
    *,
    config: ForgeConfig,
    task_path: Path,
    candidate_path: Path,
    validation_path: Path,
    gate_path: Path,
    to_wiring: str,
    authorized_by: str,
    reason: str,
    previous_receipt_path: Path | None = None,
    output_path: Path | None = None,
) -> PromotionReceiptResult:
    """Issue an offline SHADOW/ACTIVE authorization receipt or a BLOCKED receipt."""

    _require_named_authorization(authorized_by, reason)
    if to_wiring not in {"shadow", "active"}:
        raise ResearchPipelineError("authorization target must be shadow or active")

    task = validate_research_task(config=config, task_path=task_path)
    if task["release"]["mode"] != "runtime_wiring":
        raise ResearchPipelineError("only runtime_wiring tasks can request SHADOW/ACTIVE authorization")
    task_resolved = _resolve_regular_file(task_path, context="research task manifest")
    task_sha256 = _sha256_file(task_resolved, context="research task manifest")

    candidate = validate_research_candidate(
        config=config,
        task_path=task_path,
        candidate_path=candidate_path,
    )
    candidate_resolved = _resolve_regular_file(candidate_path, context="research candidate")
    candidate_sha256 = _sha256_file(candidate_resolved, context="research candidate")
    validation, validation_resolved, validation_sha256 = _load_artifact(
        config,
        validation_path,
        "forge-research-validation.v1",
    )
    run_root = _normalize_run_root(Path(candidate["source"]["run_root"]))
    _assert_outside_run_root(validation_resolved, run_root, context="formal validation artifact")
    _validate_external_validation(
        config=config,
        task=task,
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        validation=validation,
        run_root=run_root,
    )
    gate, gate_resolved, gate_sha256 = _load_artifact(config, gate_path, "forge-research-gate.v1")
    _assert_outside_run_root(gate_resolved, run_root, context="ModificationGate artifact")
    _validate_gate(
        config=config,
        task=task,
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        validation_sha256=validation_sha256,
        gate=gate,
        run_root=run_root,
    )

    previous_receipt: dict[str, Any] | None = None
    previous_receipt_sha256: str | None = None
    if previous_receipt_path is not None:
        previous_receipt, _, previous_receipt_sha256 = _load_receipt(
            config,
            previous_receipt_path,
        )
        _assert_previous_receipt_binding(
            previous_receipt=previous_receipt,
            task=task,
            task_sha256=task_sha256,
            candidate=candidate,
            candidate_sha256=candidate_sha256,
        )
    if to_wiring == "shadow":
        from_wiring = "disabled"
        if (
            previous_receipt is not None
            and previous_receipt["transition"]["resulting_wiring"] != "disabled"
        ):
            raise ResearchPipelineError("SHADOW re-authorization requires an AUTHORIZED DISABLED receipt")
    else:
        if previous_receipt is None:
            raise ResearchPipelineError("SHADOW to ACTIVE authorization requires the previous SHADOW receipt")
        if previous_receipt["transition"]["resulting_wiring"] != "shadow":
            raise ResearchPipelineError("ACTIVE authorization requires an AUTHORIZED SHADOW receipt")
        if previous_receipt["bindings"]["validation_sha256"] == validation_sha256:
            raise ResearchPipelineError("ACTIVE authorization requires new loop-external validation evidence")
        if previous_receipt["bindings"]["gate_sha256"] == gate_sha256:
            raise ResearchPipelineError("ACTIVE authorization requires a new exact-bound gate review")
        from_wiring = "shadow"
    if to_wiring == "shadow" and previous_receipt is not None:
        if previous_receipt["bindings"]["validation_sha256"] == validation_sha256:
            raise ResearchPipelineError("SHADOW re-authorization requires new loop-external validation evidence")
        if previous_receipt["bindings"]["gate_sha256"] == gate_sha256:
            raise ResearchPipelineError("SHADOW re-authorization requires a new exact-bound gate review")

    blockers, check_results = _promotion_blockers(
        task=task,
        candidate=candidate,
        validation=validation,
        gate=gate,
        to_wiring=to_wiring,
    )
    outcome = "BLOCKED" if blockers else "AUTHORIZED"
    resulting_wiring = from_wiring if blockers else to_wiring
    gate_decision = str(gate["decision"]).lower()
    research_maturity = "externally_validated" if validation["status"] == "PASS" else "rejected"

    receipt: dict[str, Any] = {
        "schema_version": "forge-research-promotion-receipt.v1",
        "task_id": task["task_id"],
        "candidate_id": candidate["candidate_id"],
        "release": {
            "mode": task["release"]["mode"],
            "target": task["release"]["target"],
        },
        "action": "authorize",
        "outcome": outcome,
        "transition": {
            "from_wiring": from_wiring,
            "requested_wiring": to_wiring,
            "resulting_wiring": resulting_wiring,
        },
        "state": {
            "research_maturity": research_maturity,
            "gate_decision": gate_decision,
            "wiring_level": resulting_wiring,
        },
        "bindings": {
            "task_manifest_sha256": task_sha256,
            "candidate_sha256": candidate_sha256,
            "validation_sha256": validation_sha256,
            "gate_sha256": gate_sha256,
            "previous_receipt_sha256": previous_receipt_sha256,
        },
        "check_results": check_results,
        "blocking_reasons": blockers,
        "authorization": {
            "authorized_by": authorized_by.strip(),
            "reason": reason.strip(),
        },
        "rollback": {
            "to_wiring": from_wiring,
            "instructions": task["release"]["rollback_instructions"],
        },
        "authority": _receipt_authority(),
        "created_at": utc_now(),
    }
    receipt["receipt_id"] = _artifact_id("research-receipt", receipt, "receipt_id")
    _validate_receipt_payload(config, receipt)
    destination = _receipt_destination(
        config=config,
        task_id=str(task["task_id"]),
        candidate_id=str(candidate["candidate_id"]),
        receipt_id=str(receipt["receipt_id"]),
        output_path=output_path,
    )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=receipt,
        expected_version="forge-research-promotion-receipt.v1",
        identity_field="receipt_id",
    )
    return PromotionReceiptResult(
        receipt_id=str(receipt["receipt_id"]),
        receipt_path=destination,
        outcome=outcome,
        resulting_wiring=resulting_wiring,
    )


def rollback_research_candidate(
    *,
    config: ForgeConfig,
    previous_receipt_path: Path,
    to_wiring: str,
    authorized_by: str,
    reason: str,
    output_path: Path | None = None,
) -> PromotionReceiptResult:
    """Authorize one adjacent downgrade without depending on current research evidence."""

    _require_named_authorization(authorized_by, reason)
    previous, _, previous_sha256 = _load_receipt(config, previous_receipt_path)
    from_wiring = previous["transition"]["resulting_wiring"]
    legal_downgrades = {"active": "shadow", "shadow": "disabled"}
    if legal_downgrades.get(from_wiring) != to_wiring:
        raise ResearchPipelineError(
            f"illegal rollback transition: {from_wiring} to {to_wiring}; rollback must be one adjacent downgrade"
        )

    receipt: dict[str, Any] = {
        "schema_version": "forge-research-promotion-receipt.v1",
        "task_id": previous["task_id"],
        "candidate_id": previous["candidate_id"],
        "release": dict(previous["release"]),
        "action": "rollback",
        "outcome": "AUTHORIZED",
        "transition": {
            "from_wiring": from_wiring,
            "requested_wiring": to_wiring,
            "resulting_wiring": to_wiring,
        },
        "state": {
            "research_maturity": previous["state"]["research_maturity"],
            "gate_decision": previous["state"]["gate_decision"],
            "wiring_level": to_wiring,
        },
        "bindings": {
            "task_manifest_sha256": previous["bindings"]["task_manifest_sha256"],
            "candidate_sha256": previous["bindings"]["candidate_sha256"],
            "validation_sha256": previous["bindings"]["validation_sha256"],
            "gate_sha256": previous["bindings"]["gate_sha256"],
            "previous_receipt_sha256": previous_sha256,
        },
        "check_results": [],
        "blocking_reasons": [],
        "authorization": {
            "authorized_by": authorized_by.strip(),
            "reason": reason.strip(),
        },
        "rollback": {
            "to_wiring": "disabled",
            "instructions": previous["rollback"]["instructions"],
        },
        "authority": _receipt_authority(),
        "created_at": utc_now(),
    }
    receipt["receipt_id"] = _artifact_id("research-receipt", receipt, "receipt_id")
    _validate_receipt_payload(config, receipt)
    destination = _receipt_destination(
        config=config,
        task_id=str(previous["task_id"]),
        candidate_id=str(previous["candidate_id"]),
        receipt_id=str(receipt["receipt_id"]),
        output_path=output_path,
    )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=receipt,
        expected_version="forge-research-promotion-receipt.v1",
        identity_field="receipt_id",
    )
    return PromotionReceiptResult(
        receipt_id=str(receipt["receipt_id"]),
        receipt_path=destination,
        outcome="AUTHORIZED",
        resulting_wiring=to_wiring,
    )


def _validate_handoff(
    *,
    config: ForgeConfig,
    task: dict[str, Any],
    handoff: dict[str, Any],
    run_root: Path,
) -> None:
    if handoff["task_id"] != task["task_id"]:
        raise ResearchPipelineError("Praxist handoff task_id does not match the Volvence Task")
    if handoff["source_base_revision"] != task["source_base_revision"]:
        raise ResearchPipelineError("Praxist handoff source_base_revision does not match the Volvence Task")

    generation_id = handoff["generation_id"]
    refs = handoff["refs"]
    expected_locators = {
        "run_metadata": "run.json",
        "task_project_manifest": "task_project_manifest.json",
        "generation_boundary": f"gen_{generation_id}/generation_boundary.json",
    }
    for name, expected in expected_locators.items():
        actual = _strict_relative(refs[name]["locator"], context=f"handoff {name} locator")
        if actual != expected:
            raise ResearchPipelineError(f"handoff {name} must point to {expected!r}")
    result_locator = _strict_relative(refs["result_summary"]["locator"], context="handoff result locator")
    if not _is_under(result_locator, "results") or not _recognized_result_name(PurePosixPath(result_locator).name):
        raise ResearchPipelineError("handoff result_summary must be a recognized JSON summary below results/")

    run_paths = {
        name: _verify_run_content_ref(run_root, ref, context=f"handoff {name}")
        for name, ref in refs.items()
    }
    run_metadata = _read_json_object_bytes(run_paths["run_metadata"].read_bytes(), run_paths["run_metadata"])
    task_manifest = _read_json_object_bytes(
        run_paths["task_project_manifest"].read_bytes(),
        run_paths["task_project_manifest"],
    )
    boundary = _read_json_object_bytes(
        run_paths["generation_boundary"].read_bytes(),
        run_paths["generation_boundary"],
    )
    _read_json_object_bytes(run_paths["result_summary"].read_bytes(), run_paths["result_summary"])

    if run_metadata.get("schema_version") != "praxist.run.v1":
        raise ResearchPipelineError("handoff run.json must use schema_version praxist.run.v1")
    if run_metadata.get("run_id") != handoff["run_id"]:
        raise ResearchPipelineError("handoff run_id does not match run.json")
    run_task_project = run_metadata.get("task_project")
    if not isinstance(run_task_project, dict):
        raise ResearchPipelineError("handoff run.json must contain task_project metadata")
    if run_task_project.get("manifest_sha256") != task["praxist"]["task_project_manifest_sha256"]:
        raise ResearchPipelineError("run.json task project digest does not match the Volvence Task")

    if task_manifest.get("schema_version") != "task_project_manifest.v1":
        raise ResearchPipelineError("task_project_manifest.json has an unsupported schema_version")
    if task_manifest.get("task_id") != task["praxist"]["task_project_id"]:
        raise ResearchPipelineError("Praxist task project id does not match the Volvence Task")
    if task_manifest.get("sha256") != task["praxist"]["task_project_manifest_sha256"]:
        raise ResearchPipelineError("Praxist task project manifest digest does not match the Volvence Task")

    if boundary.get("schema_version") != "praxist.generation_boundary.v1":
        raise ResearchPipelineError("generation boundary has an unsupported schema_version")
    if boundary.get("generation_id") != generation_id:
        raise ResearchPipelineError("generation boundary id does not match the handoff")
    semantics = boundary.get("artifact_semantics")
    if not isinstance(semantics, dict):
        raise ResearchPipelineError("generation boundary is missing artifact_semantics")
    if semantics.get("role") != "canonical_state" or semantics.get("status") != "committed":
        raise ResearchPipelineError("generation boundary must be committed canonical_state")
    if semantics.get("generation_id") != generation_id:
        raise ResearchPipelineError("generation boundary semantics do not bind the handoff generation")
    if semantics.get("runtime_fact_source") is not True:
        raise ResearchPipelineError("generation boundary must be marked as a runtime fact source")

    editable_roots = _normalize_roots(task["sandbox"]["editable_roots"], context="editable root")
    declared_protected = _normalize_roots(
        task["sandbox"]["protected_roots"],
        context="protected root",
    )
    protected_roots = tuple(dict.fromkeys((*declared_protected, *_MANDATORY_PROTECTED_ROOTS)))
    target_paths: set[str] = set()
    for index, file_mapping in enumerate(handoff["candidate"]["files"]):
        target = _strict_relative(file_mapping["target_path"], context=f"candidate target {index}")
        if target in target_paths:
            raise ResearchPipelineError(f"duplicate candidate target path: {target}")
        target_paths.add(target)
        if not any(_is_under(target, root) for root in editable_roots):
            raise ResearchPipelineError(f"candidate target {target!r} is outside the Task editable roots")
        if any(_is_under(target, root) for root in protected_roots):
            raise ResearchPipelineError(f"candidate target {target!r} enters a protected root")
        source_locator = _strict_relative(
            file_mapping["source"]["locator"],
            context=f"candidate source {index}",
        )
        if not (_is_under(source_locator, "variants") or _is_under(source_locator, "results")):
            raise ResearchPipelineError("candidate sources must be below Praxist variants/ or results/")
        _verify_run_content_ref(run_root, file_mapping["source"], context=f"candidate source {index}")


def _assert_candidate_matches_handoff(
    candidate: dict[str, Any],
    handoff: dict[str, Any],
    run_root: Path,
) -> None:
    source = candidate["source"]
    expected = {
        "run_root": str(run_root),
        "run_id": handoff["run_id"],
        "generation_id": handoff["generation_id"],
        "variant_id": handoff["candidate"]["variant_id"],
        "refs": handoff["refs"],
        "files": handoff["candidate"]["files"],
    }
    for name, value in expected.items():
        if source[name] != value:
            raise ResearchPipelineError(f"candidate source {name} does not match its Praxist handoff")
    research = candidate["research"]
    if research["source_retention"] != handoff["candidate"]["retention"]:
        raise ResearchPipelineError("candidate source retention does not match its Praxist handoff")
    if research["parent_candidate_ids"] != handoff["candidate"]["parent_candidate_ids"]:
        raise ResearchPipelineError("candidate parent lineage does not match its Praxist handoff")


def _validate_external_validation(
    *,
    config: ForgeConfig,
    task: dict[str, Any],
    candidate: dict[str, Any],
    candidate_sha256: str,
    validation: dict[str, Any],
    run_root: Path,
) -> None:
    if validation["task_id"] != task["task_id"]:
        raise ResearchPipelineError("formal validation task_id does not match the Task")
    if validation["candidate_id"] != candidate["candidate_id"]:
        raise ResearchPipelineError("formal validation candidate_id does not match the candidate")
    if validation["candidate_sha256"] != candidate_sha256:
        raise ResearchPipelineError("formal validation candidate digest does not match the candidate bytes")
    if validation["validator_id"] != task["validation"]["formal_validator_id"]:
        raise ResearchPipelineError("formal validation was not issued by the Task validator")
    if validation["formal_protocol"] != task["validation"]["formal_protocol"]:
        raise ResearchPipelineError("formal validation protocol does not match the frozen Task protocol")
    _verify_repo_content_ref(config, validation["formal_protocol"], context="formal validation protocol")

    names: set[str] = set()
    all_passed = True
    for check in validation["checks"]:
        name = check["name"]
        if name in names:
            raise ResearchPipelineError(f"formal validation contains duplicate check {name!r}")
        names.add(name)
        all_passed = all_passed and check["passed"] is True
        for evidence_index, evidence in enumerate(check["evidence"]):
            evidence_path = _verify_external_content_ref(
                config,
                evidence,
                context=f"formal validation check {name!r} evidence {evidence_index}",
            )
            _assert_outside_run_root(
                evidence_path,
                run_root,
                context=f"formal validation check {name!r} evidence {evidence_index}",
            )
    expected_status = "PASS" if all_passed else "BLOCK"
    if validation["status"] != expected_status:
        raise ResearchPipelineError(
            f"formal validation status must be {expected_status} for its declared check results"
        )


def _validate_gate(
    *,
    config: ForgeConfig,
    task: dict[str, Any],
    candidate: dict[str, Any],
    candidate_sha256: str,
    validation_sha256: str,
    gate: dict[str, Any],
    run_root: Path,
) -> None:
    expected = {
        "task_id": task["task_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate_sha256,
        "validation_sha256": validation_sha256,
        "target": task["release"]["target"],
        "authority": task["release"]["gate_authority"],
    }
    for name, value in expected.items():
        if gate[name] != value:
            raise ResearchPipelineError(f"gate {name} does not match the frozen authorization inputs")
    if gate["decision"] == "ALLOW" and gate["reasons"]:
        raise ResearchPipelineError("an ALLOW gate artifact must not contain blocking reasons")
    if gate["decision"] == "BLOCK" and not gate["reasons"]:
        raise ResearchPipelineError("a BLOCK gate artifact must contain at least one reason")
    gate_review_path = _verify_external_content_ref(
        config,
        gate["gate_review"],
        context="ModificationGate review",
    )
    _assert_outside_run_root(gate_review_path, run_root, context="ModificationGate review")


def _promotion_blockers(
    *,
    task: dict[str, Any],
    candidate: dict[str, Any],
    validation: dict[str, Any],
    gate: dict[str, Any],
    to_wiring: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    blockers: list[str] = []
    retention = candidate["research"]["source_retention"]
    if retention["maturity"] != "mature":
        blockers.append("Praxist source retention is not mature")
    if retention["late_after_generation_boundary"] is True:
        blockers.append("Praxist source was written after the committed generation boundary")
    if validation["status"] != "PASS":
        blockers.append("loop-external formal validation blocked the candidate")

    checks_by_name = {check["name"]: check["passed"] for check in validation["checks"]}
    required_key = "shadow_required_checks" if to_wiring == "shadow" else "active_required_checks"
    check_results: list[dict[str, Any]] = []
    for name in task["validation"][required_key]:
        passed = checks_by_name.get(name) is True
        check_results.append({"name": name, "passed": passed})
        if not passed:
            blockers.append(f"required {to_wiring.upper()} check missing or failed: {name}")
    if gate["decision"] != "ALLOW":
        blockers.extend(f"ModificationGate blocked: {reason}" for reason in gate["reasons"])
    return blockers, check_results


def _assert_previous_receipt_binding(
    *,
    previous_receipt: dict[str, Any],
    task: dict[str, Any],
    task_sha256: str,
    candidate: dict[str, Any],
    candidate_sha256: str,
) -> None:
    if previous_receipt["outcome"] != "AUTHORIZED":
        raise ResearchPipelineError("a BLOCKED receipt cannot establish the current wiring state")
    if previous_receipt["task_id"] != task["task_id"]:
        raise ResearchPipelineError("previous receipt belongs to a different Task")
    if previous_receipt["candidate_id"] != candidate["candidate_id"]:
        raise ResearchPipelineError("previous receipt belongs to a different candidate")
    if previous_receipt["release"] != {
        "mode": task["release"]["mode"],
        "target": task["release"]["target"],
    }:
        raise ResearchPipelineError("previous receipt binds a different release target")
    if previous_receipt["bindings"]["task_manifest_sha256"] != task_sha256:
        raise ResearchPipelineError("previous receipt binds a different Task revision")
    if previous_receipt["bindings"]["candidate_sha256"] != candidate_sha256:
        raise ResearchPipelineError("previous receipt binds different candidate bytes")


def _load_receipt(
    config: ForgeConfig,
    receipt_path: Path,
) -> tuple[dict[str, Any], Path, str]:
    receipt, resolved, digest = _load_artifact(
        config,
        receipt_path,
        "forge-research-promotion-receipt.v1",
    )
    _validate_receipt_payload(config, receipt)
    if receipt["outcome"] != "AUTHORIZED":
        raise ResearchPipelineError("only an AUTHORIZED receipt can be used as a transition boundary")
    return receipt, resolved, digest


def _validate_receipt_payload(config: ForgeConfig, receipt: dict[str, Any]) -> None:
    _validate_schema(config, receipt, "forge-research-promotion-receipt.v1")
    if receipt["receipt_id"] != _artifact_id("research-receipt", receipt, "receipt_id"):
        raise ResearchPipelineError("research receipt_id does not match its canonical payload")
    transition = receipt["transition"]
    transition_pair = (transition["from_wiring"], transition["requested_wiring"])
    if receipt["action"] == "authorize" and transition_pair not in {
        ("disabled", "shadow"),
        ("shadow", "active"),
    }:
        raise ResearchPipelineError("authorization receipt must contain one adjacent forward transition")
    if receipt["action"] == "rollback" and transition_pair not in {
        ("active", "shadow"),
        ("shadow", "disabled"),
    }:
        raise ResearchPipelineError("rollback receipt must contain one adjacent downgrade")
    if receipt["action"] == "rollback" and receipt["outcome"] != "AUTHORIZED":
        raise ResearchPipelineError("rollback receipts cannot be BLOCKED by research evidence")
    if receipt["state"]["wiring_level"] != transition["resulting_wiring"]:
        raise ResearchPipelineError("receipt state wiring does not match its transition result")
    if receipt["outcome"] == "AUTHORIZED":
        if receipt["blocking_reasons"]:
            raise ResearchPipelineError("AUTHORIZED receipt cannot contain blocking reasons")
        if transition["resulting_wiring"] != transition["requested_wiring"]:
            raise ResearchPipelineError("AUTHORIZED receipt must reach its requested wiring")
    else:
        if not receipt["blocking_reasons"]:
            raise ResearchPipelineError("BLOCKED receipt must contain blocking reasons")
        if transition["resulting_wiring"] != transition["from_wiring"]:
            raise ResearchPipelineError("BLOCKED receipt must preserve the prior wiring")


def _load_artifact(
    config: ForgeConfig,
    path: Path,
    expected_version: str,
) -> tuple[dict[str, Any], Path, str]:
    resolved = _resolve_regular_file(path, context=expected_version)
    payload = read_json(resolved)
    _validate_schema(config, payload, expected_version)
    return payload, resolved, _sha256_file(resolved, context=expected_version)


def _validate_schema(config: ForgeConfig, payload: dict[str, Any], expected_version: str) -> None:
    SchemaStore(config.paths.forge_root / "schemas").validate(payload, SCHEMA_NAME)
    if payload.get("schema_version") != expected_version:
        raise ResearchPipelineError(
            f"expected schema_version {expected_version!r}, got {payload.get('schema_version')!r}"
        )


def _verify_repo_content_ref(
    config: ForgeConfig,
    content_ref: dict[str, Any],
    *,
    context: str,
) -> Path:
    locator = _strict_relative(content_ref["locator"], context=f"{context} locator")
    repo_root = config.paths.repo_root.resolve()
    path = _resolve_below_root(repo_root, locator, context=context)
    _assert_digest(path, content_ref["sha256"], context=context)
    return path


def _verify_external_content_ref(
    config: ForgeConfig,
    content_ref: dict[str, Any],
    *,
    context: str,
) -> Path:
    locator = content_ref["locator"]
    raw_path = Path(locator).expanduser()
    if raw_path.is_absolute():
        if ".." in raw_path.parts:
            raise ResearchPipelineError(f"{context} locator is not canonical: {locator!r}")
        path = _resolve_regular_file(raw_path, context=context)
    else:
        path = _resolve_below_root(
            config.paths.repo_root.resolve(),
            _strict_relative(locator, context=f"{context} locator"),
            context=context,
        )
    _assert_digest(path, content_ref["sha256"], context=context)
    return path


def _verify_run_content_ref(
    run_root: Path,
    content_ref: dict[str, Any],
    *,
    context: str,
) -> Path:
    locator = _strict_relative(content_ref["locator"], context=f"{context} locator")
    path = _resolve_below_root(run_root, locator, context=context)
    _assert_digest(path, content_ref["sha256"], context=context)
    return path


def _resolve_below_root(root: Path, locator: str, *, context: str) -> Path:
    current = root
    for part in PurePosixPath(locator).parts:
        current = current / part
        if current.is_symlink():
            raise ResearchPipelineError(f"{context} may not traverse symlink {current}")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchPipelineError(f"missing {context}: {current}") from exc
    if not resolved.is_relative_to(root):
        raise ResearchPipelineError(f"{context} escapes its declared root: {locator!r}")
    if not resolved.is_file():
        raise ResearchPipelineError(f"{context} must be a regular file: {resolved}")
    return resolved


def _resolve_regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ResearchPipelineError(f"{context} may not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchPipelineError(f"missing {context}: {expanded}") from exc
    if not resolved.is_file():
        raise ResearchPipelineError(f"{context} must be a regular file: {resolved}")
    return resolved


def _assert_outside_run_root(path: Path, run_root: Path, *, context: str) -> None:
    if path.is_relative_to(run_root):
        raise ResearchPipelineError(f"{context} must be produced outside the Praxist run root")


def _normalize_run_root(run_dir: Path) -> Path:
    expanded = run_dir.expanduser()
    if expanded.is_symlink():
        raise ResearchPipelineError(f"Praxist run root may not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchPipelineError(f"missing Praxist run root: {expanded}") from exc
    if not resolved.is_dir():
        raise ResearchPipelineError(f"Praxist run root must be a directory: {resolved}")
    return resolved


def _strict_relative(value: str, *, context: str) -> str:
    if "\\" in value:
        raise ResearchPipelineError(f"{context} must use canonical POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ResearchPipelineError(f"unsafe {context}: {value!r}")
    normalized = path.as_posix()
    if value != normalized:
        raise ResearchPipelineError(f"{context} is not canonical: {value!r}")
    return normalized


def _normalize_roots(values: list[str], *, context: str) -> tuple[str, ...]:
    return tuple(_strict_relative(value, context=context) for value in values)


def _is_under(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


def _paths_overlap(left: str, right: str) -> bool:
    return _is_under(left, right) or _is_under(right, left)


def _recognized_result_name(name: str) -> bool:
    return name in _RESULT_SUMMARY_NAMES or fnmatch.fnmatchcase(name, "custom_*_tiered_eval_summary.json")


def _read_json_object_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchPipelineError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchPipelineError(f"expected JSON object in {path}")
    return payload


def _assert_digest(path: Path, declared: str, *, context: str) -> None:
    actual = _sha256_file(path, context=context)
    if actual != declared:
        raise ResearchPipelineError(f"{context} digest mismatch: declared {declared}, actual {actual}")


def _sha256_file(path: Path, *, context: str) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ResearchPipelineError(f"cannot read {context} at {path}: {exc}") from exc


def _portable_locator(config: ForgeConfig, path: Path) -> str:
    repo_root = config.paths.repo_root.resolve()
    return path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)


def _copy_refs(refs: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        name: {"locator": str(ref["locator"]), "sha256": str(ref["sha256"])}
        for name, ref in refs.items()
    }


def _copy_candidate_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_path": str(item["target_path"]),
            "source": {
                "locator": str(item["source"]["locator"]),
                "sha256": str(item["source"]["sha256"]),
            },
        }
        for item in files
    ]


def _artifact_id(prefix: str, payload: dict[str, Any], identity_field: str) -> str:
    identity_payload = {
        key: value
        for key, value in payload.items()
        if key not in {identity_field, "created_at"}
    }
    return f"{prefix}:{sha256_text(canonical_json(identity_payload))}"


def _resolve_output_path(config: ForgeConfig, output_path: Path) -> Path:
    expanded = output_path.expanduser()
    if not expanded.is_absolute():
        expanded = config.paths.repo_root / expanded
    if expanded.is_symlink():
        raise ResearchPipelineError(f"research artifacts may not overwrite a symlink: {expanded}")
    destination = expanded.resolve(strict=False)
    artifacts_root = config.paths.artifacts_root.resolve(strict=False)
    if not destination.is_relative_to(artifacts_root):
        raise ResearchPipelineError("research promotion artifacts may only be written below artifacts/")
    return destination


def _write_immutable_artifact(
    *,
    config: ForgeConfig,
    destination: Path,
    payload: dict[str, Any],
    expected_version: str,
    identity_field: str,
) -> None:
    if destination.exists():
        existing = read_json(destination)
        _validate_schema(config, existing, expected_version)
        existing_identity = _artifact_id(
            str(existing[identity_field]).partition(":")[0],
            existing,
            identity_field,
        )
        if existing[identity_field] != existing_identity:
            raise ResearchPipelineError(f"existing artifact identity is invalid: {destination}")
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
        if existing[identity_field] != payload[identity_field] or existing_body != payload_body:
            raise ResearchPipelineError(f"refusing to overwrite a different immutable artifact: {destination}")
        return
    atomic_write_json(destination, payload)


def _receipt_destination(
    *,
    config: ForgeConfig,
    task_id: str,
    candidate_id: str,
    receipt_id: str,
    output_path: Path | None,
) -> Path:
    candidate_digest = candidate_id.partition(":")[2]
    receipt_digest = receipt_id.partition(":")[2]
    default_path = (
        config.paths.artifacts_root
        / "research_promotion"
        / task_id
        / candidate_digest
        / "receipts"
        / f"{receipt_digest}.json"
    )
    return _resolve_output_path(config, output_path or default_path)


def _require_named_authorization(authorized_by: str, reason: str) -> None:
    if not authorized_by.strip():
        raise ResearchPipelineError("authorization requires a named human operator")
    if not reason.strip():
        raise ResearchPipelineError("authorization requires a non-empty reason")


def _receipt_authority() -> dict[str, bool]:
    return {
        "runtime_mutated": False,
        "production_default_changed": False,
        "target_adapter_apply_required": True,
        "evaluation_is_learning_source": False,
    }
