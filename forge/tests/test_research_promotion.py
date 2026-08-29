from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from volvence_forge.cli import main
from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import ForgeError, sha256_bytes
from volvence_forge.research_promotion import (
    authorize_research_candidate,
    import_praxist_candidate,
    rollback_research_candidate,
    validate_research_candidate,
    validate_research_task,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResearchFixture:
    config: ForgeConfig
    task_path: Path
    handoff_path: Path
    run_dir: Path
    candidate_source: Path
    runtime_target: Path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _ref(path: Path, *, root: Path) -> dict[str, str]:
    return {"locator": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def _config(tmp_path: Path) -> ForgeConfig:
    repo = tmp_path / "repo"
    forge = repo / "forge"
    forge.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge)
    shutil.copytree(REPO_ROOT / "forge" / "schemas", forge / "schemas")
    return ForgeConfig.load(ForgePaths.discover(repo_root=repo, transcripts_root=repo / "transcripts"))


def _fixture(
    tmp_path: Path,
    *,
    maturity: str = "mature",
    late_after_boundary: bool = False,
) -> ResearchFixture:
    config = _config(tmp_path)
    repo = config.paths.repo_root
    baseline = _write_json(repo / "research/contracts/baseline.json", {"baseline": "v1"})
    protocol = _write_json(repo / "research/contracts/formal_protocol.json", {"protocol": "sealed-v1"})
    task_path = repo / "research/tasks/memory_inheritance/task.json"
    task = {
        "schema_version": "forge-research-task.v1",
        "task_id": "memory_inheritance",
        "claim_id": "claim:memory-inheritance",
        "owner": "vz-memory",
        "objective": "Improve inheritance without changing the formal evaluator or runtime wiring.",
        "capability_axes": ["appendable", "readable", "learnable"],
        "source_base_revision": "a" * 40,
        "baseline": _ref(baseline, root=repo),
        "praxist": {
            "task_project_id": "memory_inheritance_project",
            "task_project_manifest_sha256": "b" * 64,
        },
        "sandbox": {
            "editable_roots": ["research_surface"],
            "protected_roots": ["research/contracts", "research/tasks"],
            "praxist_can_modify_task_contract": False,
            "praxist_can_modify_formal_evaluator": False,
            "praxist_can_read_sealed_holdout": False,
            "praxist_can_change_production_wiring": False,
            "praxist_can_access_production_credentials": False,
        },
        "validation": {
            "development_evaluator_id": "praxist:development-evaluator",
            "formal_validator_id": "volvence:sealed-validator",
            "formal_protocol": _ref(protocol, root=repo),
            "shadow_required_checks": ["formal_quality", "rollback_drill"],
            "active_required_checks": [
                "formal_quality",
                "rollback_drill",
                "shadow_observation",
                "canary",
            ],
        },
        "release": {
            "mode": "runtime_wiring",
            "target": "memory_inheritance_policy",
            "initial_wiring": "disabled",
            "gate_authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
            "rollback_instructions": "Restore the previous content hash and lower wiring one level.",
        },
        "authority": {
            "praxist_is_research_retention_authority_only": True,
            "evaluation_is_learning_source": False,
            "production_promotion_authorized": False,
        },
    }
    _write_json(task_path, task)

    runtime_target = repo / "research_surface/candidate.py"
    runtime_target.parent.mkdir(parents=True)
    runtime_target.write_text("BASELINE = True\n", encoding="utf-8")

    run_dir = tmp_path / "praxist-run"
    candidate_source = run_dir / "variants/variant_a/candidate.py"
    candidate_source.parent.mkdir(parents=True)
    candidate_source.write_text("CANDIDATE = 'a'\n", encoding="utf-8")
    result_summary = _write_json(
        run_dir / "results/variant_a/tiered_eval_summary.json",
        {"variant_id": "variant_a", "score": 0.9},
    )
    task_project_manifest = _write_json(
        run_dir / "task_project_manifest.json",
        {
            "schema_version": "task_project_manifest.v1",
            "task_id": "memory_inheritance_project",
            "sha256": "b" * 64,
        },
    )
    run_metadata = _write_json(
        run_dir / "run.json",
        {
            "schema_version": "praxist.run.v1",
            "run_id": "run-memory-001",
            "task_project": {"manifest_sha256": "b" * 64},
        },
    )
    boundary = _write_json(
        run_dir / "gen_0/generation_boundary.json",
        {
            "schema_version": "praxist.generation_boundary.v1",
            "generation_id": 0,
            "artifact_semantics": {
                "role": "canonical_state",
                "status": "committed",
                "generation_id": 0,
                "runtime_fact_source": True,
            },
        },
    )
    handoff = {
        "schema_version": "forge-praxist-candidate-handoff.v1",
        "task_id": "memory_inheritance",
        "run_id": "run-memory-001",
        "generation_id": 0,
        "source_base_revision": "a" * 40,
        "refs": {
            "run_metadata": _ref(run_metadata, root=run_dir),
            "task_project_manifest": _ref(task_project_manifest, root=run_dir),
            "generation_boundary": _ref(boundary, root=run_dir),
            "result_summary": _ref(result_summary, root=run_dir),
        },
        "candidate": {
            "variant_id": "variant_a",
            "parent_candidate_ids": ["baseline"],
            "retention": {
                "lane": "frontier",
                "maturity": maturity,
                "parent_eligible": True,
                "late_after_generation_boundary": late_after_boundary,
            },
            "files": [
                {
                    "target_path": "research_surface/candidate.py",
                    "source": _ref(candidate_source, root=run_dir),
                }
            ],
        },
        "authority": {
            "research_retention_only": True,
            "production_promotion_authorized": False,
            "requested_wiring": "disabled",
            "formal_validation_performed": False,
        },
    }
    handoff_path = _write_json(run_dir / "volvence_handoff.json", handoff)
    return ResearchFixture(
        config=config,
        task_path=task_path,
        handoff_path=handoff_path,
        run_dir=run_dir,
        candidate_source=candidate_source,
        runtime_target=runtime_target,
    )


def _import(fixture: ResearchFixture) -> Path:
    result = import_praxist_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        handoff_path=fixture.handoff_path,
        run_dir=fixture.run_dir,
    )
    return result.candidate_path


def _validation(
    fixture: ResearchFixture,
    candidate_path: Path,
    *,
    name: str,
    checks: dict[str, bool],
) -> Path:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    task = json.loads(fixture.task_path.read_text(encoding="utf-8"))
    evidence = _write_json(
        fixture.config.paths.artifacts_root / "research_evidence" / f"{name}.json",
        {"phase": name, "checks": checks},
    )
    payload = {
        "schema_version": "forge-research-validation.v1",
        "task_id": task["task_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": _sha(candidate_path),
        "validator_id": task["validation"]["formal_validator_id"],
        "formal_protocol": task["validation"]["formal_protocol"],
        "status": "PASS" if all(checks.values()) else "BLOCK",
        "checks": [
            {"name": check_name, "passed": passed, "evidence": [_ref(evidence, root=fixture.config.paths.repo_root)]}
            for check_name, passed in checks.items()
        ],
        "sealed_holdout": {"used": True, "visible_to_praxist": False},
        "authority": {
            "loop_external": True,
            "evaluation_is_learning_source": False,
            "production_promotion_authorized": False,
        },
        "created_at": "2026-08-29T00:00:00Z" if name == "shadow" else "2026-08-29T01:00:00Z",
    }
    return _write_json(
        fixture.config.paths.artifacts_root / "research_validation" / f"{name}.json",
        payload,
    )


def _gate(
    fixture: ResearchFixture,
    candidate_path: Path,
    validation_path: Path,
    *,
    name: str,
    decision: str = "ALLOW",
    authority: str = "volvence_zero.credit.gate.evaluate_gate_reasons",
) -> Path:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    task = json.loads(fixture.task_path.read_text(encoding="utf-8"))
    gate_review = _write_json(
        fixture.config.paths.artifacts_root / "research_gate" / f"{name}_review.json",
        {"decision": decision, "phase": name},
    )
    payload = {
        "schema_version": "forge-research-gate.v1",
        "task_id": task["task_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": _sha(candidate_path),
        "validation_sha256": _sha(validation_path),
        "target": task["release"]["target"],
        "desired_gate": "offline",
        "decision": decision,
        "reasons": [] if decision == "ALLOW" else ["capacity guard blocked"],
        "gate_review": _ref(gate_review, root=fixture.config.paths.repo_root),
        "rollback_evidence_present": True,
        "authority": authority,
        "production_promotion_authorized": False,
        "created_at": "2026-08-29T00:10:00Z" if name == "shadow" else "2026-08-29T01:10:00Z",
    }
    return _write_json(
        fixture.config.paths.artifacts_root / "research_gate" / f"{name}.json",
        payload,
    )


def _shadow_checks() -> dict[str, bool]:
    return {"formal_quality": True, "rollback_drill": True}


def _active_checks() -> dict[str, bool]:
    return {
        "formal_quality": True,
        "rollback_drill": True,
        "shadow_observation": True,
        "canary": True,
    }


def test_task_and_import_seal_disabled_candidate_without_touching_target(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    task = validate_research_task(config=fixture.config, task_path=fixture.task_path)
    candidate_path = _import(fixture)
    candidate = validate_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
    )

    assert task["authority"]["production_promotion_authorized"] is False
    assert candidate["research"]["source_retention"]["parent_eligible"] is True
    assert candidate["release"] == {
        "mode": "runtime_wiring",
        "target": "memory_inheritance_policy",
        "gate_decision": "not_evaluated",
        "wiring_level": "disabled",
    }
    assert candidate["authority"]["production_promotion_authorized"] is False
    assert fixture.runtime_target.read_text(encoding="utf-8") == "BASELINE = True\n"


def test_import_rejects_target_escape_and_revalidation_detects_source_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    handoff = json.loads(fixture.handoff_path.read_text(encoding="utf-8"))
    handoff["candidate"]["files"][0]["target_path"] = "../escape.py"
    _write_json(fixture.handoff_path, handoff)

    with pytest.raises(ForgeError, match="unsafe candidate target"):
        _import(fixture)

    handoff["candidate"]["files"][0]["target_path"] = "research_surface/candidate.py"
    _write_json(fixture.handoff_path, handoff)
    candidate_path = _import(fixture)
    fixture.candidate_source.write_text("CANDIDATE = 'tampered'\n", encoding="utf-8")

    with pytest.raises(ForgeError, match="candidate source 0 digest mismatch"):
        validate_research_candidate(
            config=fixture.config,
            task_path=fixture.task_path,
            candidate_path=candidate_path,
        )


def test_preliminary_frontier_candidate_stays_blocked_despite_pass_and_allow(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, maturity="preliminary")
    candidate_path = _import(fixture)
    validation_path = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    gate_path = _gate(fixture, candidate_path, validation_path, name="shadow")

    result = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=validation_path,
        gate_path=gate_path,
        to_wiring="shadow",
        authorized_by="reviewer@example.com",
        reason="Exercise the source maturity gate.",
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))

    assert result.outcome == "BLOCKED"
    assert result.resulting_wiring == "disabled"
    assert receipt["transition"]["resulting_wiring"] == "disabled"
    assert "Praxist source retention is not mature" in receipt["blocking_reasons"]
    assert receipt["authority"]["runtime_mutated"] is False
    assert fixture.runtime_target.read_text(encoding="utf-8") == "BASELINE = True\n"


def test_frontier_cannot_impersonate_modification_gate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_path = _import(fixture)
    validation_path = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    gate_path = _gate(
        fixture,
        candidate_path,
        validation_path,
        name="shadow",
        authority="praxist.frontier",
    )

    with pytest.raises(ForgeError, match="research_promotion.schema.json validation failed"):
        authorize_research_candidate(
            config=fixture.config,
            task_path=fixture.task_path,
            candidate_path=candidate_path,
            validation_path=validation_path,
            gate_path=gate_path,
            to_wiring="shadow",
            authorized_by="reviewer@example.com",
            reason="This must not be accepted.",
        )

    valid_gate = _gate(fixture, candidate_path, validation_path, name="valid_shadow")
    forged_inside_run = _write_json(
        fixture.run_dir / "results/variant_a/forged_gate.json",
        json.loads(valid_gate.read_text(encoding="utf-8")),
    )
    with pytest.raises(ForgeError, match="outside the Praxist run root"):
        authorize_research_candidate(
            config=fixture.config,
            task_path=fixture.task_path,
            candidate_path=candidate_path,
            validation_path=validation_path,
            gate_path=forged_inside_run,
            to_wiring="shadow",
            authorized_by="reviewer@example.com",
            reason="A correctly named in-loop gate is still not external.",
        )


def test_shadow_then_active_requires_adjacent_receipt_and_fresh_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_path = _import(fixture)
    shadow_validation = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    shadow_gate = _gate(fixture, candidate_path, shadow_validation, name="shadow")
    shadow = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=shadow_validation,
        gate_path=shadow_gate,
        to_wiring="shadow",
        authorized_by="reviewer@example.com",
        reason="Authorize bounded shadow observation.",
    )
    assert shadow.outcome == "AUTHORIZED"
    assert shadow.resulting_wiring == "shadow"

    active_validation = _validation(fixture, candidate_path, name="active", checks=_active_checks())
    active_gate = _gate(fixture, candidate_path, active_validation, name="active")
    with pytest.raises(ForgeError, match="requires the previous SHADOW receipt"):
        authorize_research_candidate(
            config=fixture.config,
            task_path=fixture.task_path,
            candidate_path=candidate_path,
            validation_path=active_validation,
            gate_path=active_gate,
            to_wiring="active",
            authorized_by="reviewer@example.com",
            reason="Direct activation is forbidden.",
        )

    with pytest.raises(ForgeError, match="requires new loop-external validation evidence"):
        authorize_research_candidate(
            config=fixture.config,
            task_path=fixture.task_path,
            candidate_path=candidate_path,
            validation_path=shadow_validation,
            gate_path=shadow_gate,
            to_wiring="active",
            previous_receipt_path=shadow.receipt_path,
            authorized_by="reviewer@example.com",
            reason="Reusing SHADOW evidence is forbidden.",
        )

    active = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=active_validation,
        gate_path=active_gate,
        to_wiring="active",
        previous_receipt_path=shadow.receipt_path,
        authorized_by="reviewer@example.com",
        reason="Authorize active adapter processing after canary evidence.",
    )
    receipt = json.loads(active.receipt_path.read_text(encoding="utf-8"))
    assert active.outcome == "AUTHORIZED"
    assert active.resulting_wiring == "active"
    assert receipt["bindings"]["previous_receipt_sha256"] == _sha(shadow.receipt_path)
    assert fixture.runtime_target.read_text(encoding="utf-8") == "BASELINE = True\n"


def test_cross_candidate_previous_receipt_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first_candidate = _import(fixture)
    shadow_validation = _validation(fixture, first_candidate, name="shadow", checks=_shadow_checks())
    shadow_gate = _gate(fixture, first_candidate, shadow_validation, name="shadow")
    shadow = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=first_candidate,
        validation_path=shadow_validation,
        gate_path=shadow_gate,
        to_wiring="shadow",
        authorized_by="reviewer@example.com",
        reason="First candidate shadow.",
    )

    second_source = fixture.run_dir / "variants/variant_b/candidate.py"
    second_source.parent.mkdir(parents=True)
    second_source.write_text("CANDIDATE = 'b'\n", encoding="utf-8")
    handoff = json.loads(fixture.handoff_path.read_text(encoding="utf-8"))
    handoff["candidate"]["variant_id"] = "variant_b"
    handoff["candidate"]["files"][0]["source"] = _ref(second_source, root=fixture.run_dir)
    second_handoff = _write_json(fixture.run_dir / "volvence_handoff_b.json", handoff)
    second_candidate = import_praxist_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        handoff_path=second_handoff,
        run_dir=fixture.run_dir,
    ).candidate_path
    active_validation = _validation(fixture, second_candidate, name="active", checks=_active_checks())
    active_gate = _gate(fixture, second_candidate, active_validation, name="active")

    with pytest.raises(ForgeError, match="different candidate"):
        authorize_research_candidate(
            config=fixture.config,
            task_path=fixture.task_path,
            candidate_path=second_candidate,
            validation_path=active_validation,
            gate_path=active_gate,
            to_wiring="active",
            previous_receipt_path=shadow.receipt_path,
            authorized_by="reviewer@example.com",
            reason="Cross-candidate chains are forbidden.",
        )


def test_gate_block_writes_negative_receipt_instead_of_mutating_runtime(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_path = _import(fixture)
    validation_path = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    gate_path = _gate(fixture, candidate_path, validation_path, name="shadow", decision="BLOCK")

    result = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=validation_path,
        gate_path=gate_path,
        to_wiring="shadow",
        authorized_by="reviewer@example.com",
        reason="Retain the negative gate result.",
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))

    assert result.outcome == "BLOCKED"
    assert receipt["blocking_reasons"] == ["ModificationGate blocked: capacity guard blocked"]
    assert fixture.runtime_target.read_text(encoding="utf-8") == "BASELINE = True\n"


def test_rollback_survives_missing_task_candidate_and_praxist_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_path = _import(fixture)
    validation_path = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    gate_path = _gate(fixture, candidate_path, validation_path, name="shadow")
    shadow = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=validation_path,
        gate_path=gate_path,
        to_wiring="shadow",
        authorized_by="reviewer@example.com",
        reason="Prepare rollback drill.",
    )

    fixture.task_path.unlink()
    candidate_path.unlink()
    shutil.rmtree(fixture.run_dir)
    rollback = rollback_research_candidate(
        config=fixture.config,
        previous_receipt_path=shadow.receipt_path,
        to_wiring="disabled",
        authorized_by="operator@example.com",
        reason="Lower authority despite unavailable research inputs.",
    )
    receipt = json.loads(rollback.receipt_path.read_text(encoding="utf-8"))

    assert rollback.outcome == "AUTHORIZED"
    assert rollback.resulting_wiring == "disabled"
    assert receipt["release"] == {"mode": "runtime_wiring", "target": "memory_inheritance_policy"}
    assert receipt["bindings"]["previous_receipt_sha256"] == _sha(shadow.receipt_path)
    assert receipt["check_results"] == []
    assert receipt["authority"]["runtime_mutated"] is False


def test_shadow_reauthorization_links_the_authorized_disabled_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_path = _import(fixture)
    first_validation = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    first_gate = _gate(fixture, candidate_path, first_validation, name="shadow")
    first_shadow = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=first_validation,
        gate_path=first_gate,
        to_wiring="shadow",
        authorized_by="reviewer@example.com",
        reason="Initial shadow authorization.",
    )
    disabled = rollback_research_candidate(
        config=fixture.config,
        previous_receipt_path=first_shadow.receipt_path,
        to_wiring="disabled",
        authorized_by="operator@example.com",
        reason="Complete the rollback before re-authorization.",
    )
    second_validation = _validation(fixture, candidate_path, name="reshadow", checks=_shadow_checks())
    second_gate = _gate(fixture, candidate_path, second_validation, name="reshadow")

    second_shadow = authorize_research_candidate(
        config=fixture.config,
        task_path=fixture.task_path,
        candidate_path=candidate_path,
        validation_path=second_validation,
        gate_path=second_gate,
        to_wiring="shadow",
        previous_receipt_path=disabled.receipt_path,
        authorized_by="reviewer@example.com",
        reason="Re-authorize from the latest disabled boundary with fresh evidence.",
    )
    receipt = json.loads(second_shadow.receipt_path.read_text(encoding="utf-8"))

    assert second_shadow.outcome == "AUTHORIZED"
    assert receipt["bindings"]["previous_receipt_sha256"] == _sha(disabled.receipt_path)


def test_cli_runs_validate_import_authorize_and_rollback_lifecycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)

    validate_status = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "research-validate-task",
            str(fixture.task_path),
        ]
    )
    assert validate_status == 0
    assert "VALID: task=memory_inheritance" in capsys.readouterr().out

    candidate_path = fixture.config.paths.artifacts_root / "cli/candidate.json"
    import_status = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "research-import-praxist",
            str(fixture.task_path),
            str(fixture.handoff_path),
            "--run-dir",
            str(fixture.run_dir),
            "--output",
            str(candidate_path),
        ]
    )
    assert import_status == 0
    assert "SEALED:" in capsys.readouterr().out

    validation_path = _validation(fixture, candidate_path, name="shadow", checks=_shadow_checks())
    gate_path = _gate(fixture, candidate_path, validation_path, name="shadow")
    shadow_receipt = fixture.config.paths.artifacts_root / "cli/shadow_receipt.json"
    authorize_status = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "research-authorize",
            str(fixture.task_path),
            str(candidate_path),
            str(validation_path),
            str(gate_path),
            "--to-wiring",
            "shadow",
            "--authorized-by",
            "reviewer@example.com",
            "--reason",
            "CLI lifecycle test.",
            "--output",
            str(shadow_receipt),
        ]
    )
    assert authorize_status == 0
    assert "AUTHORIZED:" in capsys.readouterr().out

    rollback_receipt = fixture.config.paths.artifacts_root / "cli/rollback_receipt.json"
    rollback_status = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "research-rollback",
            str(shadow_receipt),
            "--to-wiring",
            "disabled",
            "--authorized-by",
            "operator@example.com",
            "--reason",
            "CLI rollback test.",
            "--output",
            str(rollback_receipt),
        ]
    )

    assert rollback_status == 0
    assert "resulting_wiring=disabled" in capsys.readouterr().out
    assert fixture.runtime_target.read_text(encoding="utf-8") == "BASELINE = True\n"
