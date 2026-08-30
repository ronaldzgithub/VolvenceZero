from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.cli import main
from volvence_forge.foundation import SchemaStore, canonical_json, sha256_bytes, sha256_text
from volvence_forge.research_control import (
    CommandExecution,
    ResearchControlError,
    SubprocessPraxistRunner,
    inspect_research_request,
    issue_research_control_directive,
    list_research_inbox,
    record_external_research_handoff,
    reconcile_research_control,
    review_research_request,
    submit_external_research_request,
    submit_research_request,
    supersede_unreviewed_research_request,
    validate_external_research_descriptor,
    validate_research_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_foundry_public_handoff_fixture_freezes_import_only_hash_chain() -> None:
    schema_path = REPO_ROOT / "forge/schemas/foundry_research_handoff.schema.json"
    fixture_path = REPO_ROOT / "forge/contracts/foundry_research_lab_seam/v1/handoff.fixture.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    SchemaStore(schema_path.parent).validate(payload, schema_path.name)
    assert payload["contract"]["schema"]["sha256"] == _sha(schema_path)
    chain = payload["hash_chain"]
    chain_body = {key: value for key, value in chain.items() if key != "chain_sha256"}
    assert chain["chain_sha256"] == sha256_text(canonical_json(chain_body))
    identity_body = {
        key: value for key, value in payload.items() if key not in {"handoff_id", "created_at"}
    }
    assert payload["handoff_id"] == (
        f"external-research-handoff:{sha256_text(canonical_json(identity_body))}"
    )
    assert payload["approval"]["reviewed_by"] == "Named A0 Reviewer"
    assert payload["terminal_event"]["state"] == "RUN_COMPLETED"
    assert payload["consumer_permissions"]["allowed_operations"] == [
        "import_simulation_handoff"
    ]


def test_subprocess_runner_sanitizes_codex_native_provider_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_run(_command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    blocked = (
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "OPENAI_BASE_URL",
        "PRAXIST_CODEX_BIN",
        "MODEL",
        "PRAXIST_MODEL",
    )
    for name in blocked:
        monkeypatch.setenv(name, "must-not-reach-codex-native")
    monkeypatch.setenv("VOLVENCE_TEST_PRESERVED", "preserved")
    monkeypatch.setattr(
        "volvence_forge.research_control.subprocess.run",
        fake_run,
    )

    runner = SubprocessPraxistRunner()
    runner.run(
        ("praxist", "doctor", "--codex-native"),
        cwd=tmp_path,
        timeout_seconds=1,
    )
    codex_environment = captured[-1]["env"]
    assert codex_environment["VOLVENCE_TEST_PRESERVED"] == "preserved"
    assert all(name not in codex_environment for name in blocked)

    runner.run(
        ("praxist", "doctor"),
        cwd=tmp_path,
        timeout_seconds=1,
    )
    assert captured[-1]["env"] is None


@dataclass(frozen=True)
class ControlFixture:
    config: ForgeConfig
    task_manifest: Path
    task_project: Path
    executable: Path
    run_dir: Path


class FakePraxistRunner:
    def __init__(self, fixture: ControlFixture) -> None:
        self.fixture = fixture
        self.calls: list[tuple[str, ...]] = []
        self.started = False
        self.state = "running"
        self.external_active = False
        self.known_active_rows: list[dict[str, Any]] = []
        self.activate_external_after_resolve = False
        self.corrupt_resolved_manifest = False
        self.fail_phase: str | None = None
        self.timeout_after_launch = False
        self.start_count = 0
        self.stop_count = 0
        self.resume_count = 0

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecution:
        del timeout_seconds
        command = tuple(argv)
        self.calls.append(command)
        assert cwd == self.fixture.task_project
        phase = command[1]
        if phase == "status":
            return self._status(command)
        if phase == "doctor":
            if self.fail_phase == phase:
                return self._failure(command, "doctor unavailable: secret-value-must-not-persist")
            return self._success(command, {"ok": True, "checks": []})
        if phase == "resolve":
            if self.fail_phase == phase:
                return self._failure(command, "resolve failed: secret-value-must-not-persist")
            preflight = Path(_option(command, "--run-dir"))
            preflight.mkdir(parents=True)
            manifest = _project_manifest(self.fixture.task_project)
            if self.corrupt_resolved_manifest:
                manifest["sha256"] = "f" * 64
            _write_json(preflight / "task_project_manifest.json", manifest)
            _write_json(
                preflight / "run.json",
                {
                    "schema_version": "praxist.run.v1",
                    "run_id": preflight.name,
                    "run_dir": str(preflight),
                    "task_project": {"manifest_sha256": manifest["sha256"]},
                },
            )
            _write_json(
                preflight / "startup_config.json",
                {
                    "schema_version": "praxist.startup.v1",
                    "canonical_args": {
                        "task_path": str(self.fixture.task_project),
                        "run_dir": str(preflight),
                    },
                    "resume_identity": {
                        "task_project_manifest_sha256": manifest["sha256"]
                    },
                },
            )
            _write_json(
                preflight / "plugin_resolution.json",
                {"schema_version": "praxist.plugin-resolution.v1"},
            )
            (preflight / "effective_task_spec.yaml").write_text(
                "schema_version: 1\n",
                encoding="utf-8",
            )
            if self.activate_external_after_resolve:
                self.external_active = True
            return self._success(
                command,
                {"run_id": preflight.name, "run_dir": str(preflight), "status": "resolved"},
            )
        if phase == "start":
            self.start_count += 1
            if self.fail_phase == phase:
                return self._failure(command, "start failed: secret-value-must-not-persist")
            run_dir = Path(_option(command, "--run-dir"))
            run_dir.mkdir(parents=True)
            self.started = True
            if self.timeout_after_launch:
                return CommandExecution(
                    argv=command,
                    returncode=None,
                    stdout="",
                    stderr="startup deadline crossed: secret-value-must-not-persist",
                    timed_out=True,
                )
            return self._success(command, self._start_payload(run_dir))
        if phase == "stop":
            self.stop_count += 1
            if self.fail_phase == phase:
                return self._failure(command, "stop failed: secret-value-must-not-persist")
            self.started = False
            self.state = "stopped"
            return self._success(
                command,
                {
                    "run_id": self.fixture.run_dir.name,
                    "matched_pids": [4242],
                    "descendant_pids": [],
                    "terminated_pids": [4242],
                    "killed_pids": [],
                    "remaining_pids": [],
                    "failed_run_ids": [],
                    "monitor_sessions": [],
                    "monitor_stopped_sessions": [],
                    "dry_run": False,
                    "warnings": [],
                },
            )
        if phase == "resume":
            self.resume_count += 1
            if self.fail_phase == phase:
                return self._failure(command, "resume failed: secret-value-must-not-persist")
            run_dir = Path(command[2])
            self.started = True
            self.state = "running"
            return self._success(command, self._start_payload(run_dir))
        raise AssertionError(f"unexpected fake Praxist command: {command}")

    def _status(self, command: tuple[str, ...]) -> CommandExecution:
        rows: list[dict[str, Any]] = []
        if "--active" in command:
            rows.extend(self.known_active_rows)
            if self.external_active:
                rows.append(
                    {
                        "run_id": "external-run",
                        "run_dir": "/tmp/external-run",
                        "pid": 999,
                        "source": "registry",
                        "state": "running",
                    }
                )
            if self.started and self.state in {"running", "starting"}:
                rows.append(self._status_row())
        elif self.started:
            rows.append(self._status_row())
        return self._success(command, rows)

    def _start_payload(self, run_dir: Path) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_dir.name,
            "pid": 4242,
            "parent_pid": 1,
            "run_dir": str(run_dir),
            "log_file": str(run_dir / "logs" / "launcher.nohup.log"),
            "task_path": str(self.fixture.task_project),
            "model": "fake-model",
            "model_provider_ref": "model_provider:fake",
            "runtime_ref": "agent_runtime:fake",
            "command": ["python", "-m", "praxist.run", "run"],
            "command_prefix": "python -m praxist.run run",
            "started_at": "2026-08-29T00:00:00Z",
            "state": "running",
            "stopped_at": None,
            "extra": {
                "startup_state": "running",
                "monitor_command": f"praxist --monitor --run-id {run_dir.name}",
            },
        }

    def _status_row(self) -> dict[str, Any]:
        return {
            "pid": 4242,
            "ppid": 1,
            "etime": "00:01",
            "command": "python -m praxist.run run",
            "run_dir": str(self.fixture.run_dir),
            "source": "registry" if self.state == "running" else "stale",
            "state": self.state,
            "run_id": self.fixture.run_dir.name,
            "task_path": str(self.fixture.task_project),
            "model": "fake-model",
            "model_provider_ref": "model_provider:fake",
            "started_at": "2026-08-29T00:00:00Z",
            "generation": 2,
            "findings_total": 7,
            "updated_at": "2026-08-29T00:02:00Z",
            "peer_health_summary": None,
            "peers": [],
            "extras": {},
        }

    @staticmethod
    def _success(command: tuple[str, ...], payload: Any) -> CommandExecution:
        return CommandExecution(
            argv=command,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    @staticmethod
    def _failure(command: tuple[str, ...], message: str) -> CommandExecution:
        return CommandExecution(
            argv=command,
            returncode=3,
            stdout="",
            stderr=message,
        )


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
    return ForgeConfig.load(
        ForgePaths.discover(repo_root=repo, transcripts_root=repo / "transcripts")
    )


def _fixture(tmp_path: Path) -> ControlFixture:
    config = _config(tmp_path)
    repo = config.paths.repo_root
    task_project = repo / "research" / "praxist_tasks" / "memory_inheritance_project"
    task_project.mkdir(parents=True)
    (task_project / "task.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "task_id: memory_inheritance_project",
                "praxist_plugins:",
                "  task_ref: task:memory_inheritance_project",
                "  workflow:",
                "    stage: workflow_stage:research_loop",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = _project_manifest(task_project)
    baseline = _write_json(repo / "research/contracts/baseline.json", {"baseline": "v1"})
    protocol = _write_json(repo / "research/contracts/protocol.json", {"protocol": "sealed-v1"})
    task_manifest = _write_json(
        repo / "research/tasks/memory_inheritance/task.json",
        {
            "schema_version": "forge-research-task.v1",
            "task_id": "memory_inheritance",
            "claim_id": "claim:memory-inheritance",
            "owner": "vz-memory",
            "objective": "Improve inheritance through a bounded Praxist research task.",
            "capability_axes": ["appendable", "readable", "learnable"],
            "source_base_revision": "a" * 40,
            "baseline": _ref(baseline, root=repo),
            "praxist": {
                "task_project_id": "memory_inheritance_project",
                "task_project_manifest_sha256": manifest["sha256"],
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
                ],
            },
            "release": {
                "mode": "runtime_wiring",
                "target": "memory_inheritance_policy",
                "initial_wiring": "disabled",
                "gate_authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
                "rollback_instructions": "Restore the previous content hash.",
            },
            "authority": {
                "praxist_is_research_retention_authority_only": True,
                "evaluation_is_learning_source": False,
                "production_promotion_authorized": False,
            },
        },
    )
    executable = tmp_path / "tools" / "praxist"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)
    return ControlFixture(
        config=config,
        task_manifest=task_manifest,
        task_project=task_project,
        executable=executable,
        run_dir=(tmp_path / "runs" / "run_memory_001").resolve(),
    )


def _source_backed_fixture(fixture: ControlFixture) -> tuple[ControlFixture, Path]:
    source_root = fixture.executable.parent.parent / "praxist-source"
    package = source_root / "praxist"
    package.mkdir(parents=True)
    runtime = package / "runtime.py"
    runtime.write_text("REVISION = 1\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text(
        "[project]\nname = 'praxist-fixture'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    executable = source_root / "bin/praxist"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)
    return (
        ControlFixture(
            config=fixture.config,
            task_manifest=fixture.task_manifest,
            task_project=fixture.task_project,
            executable=executable,
            run_dir=fixture.run_dir,
        ),
        runtime,
    )


def _identity(prefix: str, payload: dict[str, Any], field: str) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {field, "created_at"}
    }
    return f"{prefix}:{sha256_text(canonical_json(body))}"


def _clone_fixture(fixture: ControlFixture) -> ControlFixture:
    repo = fixture.config.paths.repo_root
    task_project = repo / "research/praxist_tasks/memory_inheritance_project_b"
    task_project.mkdir(parents=True)
    (task_project / "task.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "task_id: memory_inheritance_project_b",
                "praxist_plugins:",
                "  task_ref: task:memory_inheritance_project_b",
                "  workflow:",
                "    stage: workflow_stage:research_loop",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = _project_manifest(task_project)
    task = json.loads(fixture.task_manifest.read_text(encoding="utf-8"))
    task.update(
        {
            "task_id": "memory_inheritance_b",
            "claim_id": "claim:memory-inheritance-b",
            "objective": "Improve a second bounded inheritance mechanism.",
        }
    )
    task["praxist"] = {
        "task_project_id": "memory_inheritance_project_b",
        "task_project_manifest_sha256": manifest["sha256"],
    }
    task["release"] = {**task["release"], "target": "memory_inheritance_policy_b"}
    task_manifest = _write_json(
        repo / "research/tasks/memory_inheritance_b/task.json",
        task,
    )
    return ControlFixture(
        config=fixture.config,
        task_manifest=task_manifest,
        task_project=task_project,
        executable=fixture.executable,
        run_dir=(fixture.run_dir.parent / "run_memory_002").resolve(),
    )


def _portfolio_for_fixtures(
    first: ControlFixture,
    second: ControlFixture,
    *,
    max_active_runs: int = 2,
) -> tuple[Path, Path, Path]:
    config = first.config
    repo = config.paths.repo_root
    fixtures = (first, second)
    study_ids = ("memory_reader_a", "memory_reader_b")
    mapping_ids = ("memory_reader_a_v1", "memory_reader_b_v1")
    demands: list[Path] = []
    mappings: list[dict[str, Any]] = []
    studies: list[dict[str, Any]] = []
    study_authority = {
        "registration_only": True,
        "human_topic_binding_required": True,
        "human_a0_required": True,
        "human_outcome_decision_required": True,
        "research_start_authorized": False,
        "production_promotion_authorized": False,
        "runtime_wiring_changed": False,
        "evaluation_is_learning_source": False,
    }
    for index, (fixture, study_id, mapping_id) in enumerate(
        zip(fixtures, study_ids, mapping_ids, strict=True)
    ):
        task = json.loads(fixture.task_manifest.read_text(encoding="utf-8"))
        source = repo / f"research/sources/{study_id}.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"Frozen evidence for {study_id}.\n", encoding="utf-8")
        demand: dict[str, Any] = {
            "schema_version": "forge-volvence-research-demand.v1",
            "claim_id": task["claim_id"],
            "title": f"Research {study_id}",
            "objective": f"Resolve the bounded {study_id} research gap.",
            "owner": task["owner"],
            "capability_axes": task["capability_axes"],
            "need": {
                "current_gap": "The current mechanism lacks decisive evidence.",
                "required_outcome": "Produce a bounded falsifiable result.",
                "success_criteria": ["The preregistered primary gate passes."],
                "falsification_criteria": ["Matched controls explain the result."],
                "protected_boundaries": [
                    "Evaluation is not a learning source.",
                    "No production wiring changes are authorized.",
                ],
            },
            "evidence": [_ref(source, root=repo)],
            "discovery": {
                "source_roots": [source.relative_to(repo).as_posix()],
                "max_source_files": 2,
                "max_source_bytes": 4096,
                "max_topics": 1,
            },
            "routing": {"requested_mapping_id": mapping_id},
            "status": "OPEN",
            "authority": {
                "discovery_only": True,
                "human_topic_binding_required": True,
                "human_a0_required": True,
                "research_start_authorized": False,
                "formal_validation_performed": False,
                "production_promotion_authorized": False,
                "runtime_wiring_changed": False,
                "evaluation_is_learning_source": False,
            },
            "created_at": "2026-08-30T00:00:00Z",
        }
        demand["demand_id"] = _identity("research-demand", demand, "demand_id")
        demand_path = _write_json(repo / f"research/demands/{study_id}.json", demand)
        demands.append(demand_path)
        mappings.append(
            {
                "mapping_id": mapping_id,
                "binding_revision": f"test-{index}",
                "match": {
                    "editable_component": f"test_component_{index}",
                    "editable_target": f"research/candidate_{index}.json",
                },
                "task_manifest": str(fixture.task_manifest),
                "task_project": str(fixture.task_project),
                "praxist_executable": str(fixture.executable),
                "run_root": str(fixture.run_dir.parent),
                "launch": {
                    "config_file": None,
                    "agent_system": "claude_sdk",
                    "runtime": "agent_runtime:claude_sdk",
                    "codex_native": False,
                    "model_provider": "model_provider:fake",
                    "model": "fake-model",
                    "strategy": "auto",
                    "cohort": 2,
                    "generations": 3,
                    "startup_timeout_seconds": 30,
                },
            }
        )
        studies.append(
            {
                "study_id": study_id,
                "title": f"Study {study_id}",
                "objective": demand["objective"],
                "claim_id": task["claim_id"],
                "owner": task["owner"],
                "capability_axes": task["capability_axes"],
                "priority": 10 + index,
                "depends_on": [],
                "concurrency_lane": "bounded_parallel",
                "demand": {
                    "artifact_id": demand["demand_id"],
                    "artifact": _ref(demand_path, root=repo),
                },
                "mapping_id": mapping_id,
                "task_id": task["task_id"],
                "readiness": "RUNNABLE_MAPPING",
                "required_completion_decision": "PROCEED",
                "authority": study_authority,
            }
        )
    _write_json(
        config.paths.forge_root / "research_task_registry.yaml",
        {
            "schema_version": "forge-research-task-registry.v1",
            "policy": {"max_new_requests_per_scan": 2},
            "mappings": mappings,
        },
    )
    portfolio: dict[str, Any] = {
        "schema_version": "forge-research-portfolio.v1",
        "title": "Bounded parallel test portfolio",
        "objective": "Run two independent exact studies under one bounded budget.",
        "owner": "volvence-research-program",
        "scheduling": {
            "ordering_strategy": "dependency_then_priority",
            "max_active_runs_global": max_active_runs,
            "unknown_active_run_policy": "BLOCK",
            "dependency_gate": "NAMED_HUMAN_OUTCOME",
            "resume_policy": "completed_generation",
            "lanes": [
                {
                    "name": "bounded_parallel",
                    "max_active_runs": max_active_runs,
                }
            ],
        },
        "studies": studies,
        "authority": {
            "portfolio_scheduling_only": True,
            "automatic_human_gates_authorized": False,
            "automatic_candidate_import_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": "2026-08-30T00:00:00Z",
    }
    portfolio["portfolio_id"] = _identity(
        "research-portfolio",
        portfolio,
        "portfolio_id",
    )
    portfolio_path = _write_json(
        repo / "research/portfolios/bounded_parallel.json",
        portfolio,
    )
    return portfolio_path, demands[0], demands[1]


def _project_manifest(task_project: Path) -> dict[str, Any]:
    task_yaml = task_project / "task.yaml"
    content = task_yaml.read_bytes()
    lines = task_yaml.read_text(encoding="utf-8").splitlines()
    task_id = next(
        line.partition(":")[2].strip()
        for line in lines
        if line.startswith("task_id:")
    )
    task_ref = next(
        line.partition(":")[2].strip()
        for line in lines
        if line.strip().startswith("task_ref:")
    )
    digest = hashlib.sha256()
    digest.update(b"task.yaml\0")
    digest.update(content)
    digest.update(b"\0")
    return {
        "schema_version": "task_project_manifest.v1",
        "source": "external_task_project",
        "task_id": task_id,
        "task_ref": task_ref,
        "path": str(task_project),
        "descriptor_path": str(task_yaml),
        "sha256": digest.hexdigest(),
        "files": [
            {
                "path": "task.yaml",
                "sha256": sha256_bytes(content),
                "bytes": len(content),
            }
        ],
    }


def _submit(fixture: ControlFixture, **overrides: Any):
    values: dict[str, Any] = {
        "config": fixture.config,
        "task_manifest_path": fixture.task_manifest,
        "task_project_path": fixture.task_project,
        "praxist_executable": fixture.executable,
        "run_dir": fixture.run_dir,
        "requested_by": "detector:typed-fixture",
        "reason": "Investigate a frozen memory inheritance gap.",
        "trigger_kind": "typed_signal",
        "agent_system": "claude_sdk",
        "runtime": "agent_runtime:claude_sdk",
        "model_provider": "model_provider:fake",
        "model": "fake-model",
        "cohort": 2,
        "generations": 3,
    }
    values.update(overrides)
    return submit_research_request(**values)


def _external_descriptor(fixture: ControlFixture, *, result_locator: str = "result.json") -> Path:
    root = fixture.config.paths.repo_root / "external-foundry"
    evidence = _write_json(root / "artifacts/evidence.json", {"source": "public"})
    intent: dict[str, Any] = {
        "schema_version": "foundry-research-lab-intent.v1",
        "opportunity_id": "opp_research_lab_001",
        "title": "Improve one bounded Foundry mechanism",
        "objective": "Explore evaluator-backed variants without changing Foundry governance surfaces.",
        "trigger": {
            "kind": "factory_diagnostic",
            "submitted_by": "Foundry detector",
            "rationale": "A bounded mechanism has measurable alternatives.",
            "evidence_refs": [
                {
                    "locator": str(evidence.resolve()),
                    "sha256": _sha(evidence),
                    "evidence_class": "public_source",
                }
            ],
        },
        "task_project": {
            "root": str(fixture.task_project.resolve()),
            "task_id": "memory_inheritance_project",
            "task_yaml_sha256": _sha(fixture.task_project / "task.yaml"),
        },
        "launch": {
            "agent_system": "claude_sdk",
            "runtime": "agent_runtime:claude_sdk",
            "codex_native": False,
            "model_provider": "model_provider:fake",
            "model": "fake-model",
            "strategy": "auto",
            "cohort": 2,
            "generations": 3,
            "startup_timeout_seconds": 30,
        },
        "adoption_policy": {
            "mode": "proposal_only",
            "formal_validation_required": True,
            "named_human_apply_required": True,
            "foundry_gate_is_final_authority": True,
        },
        "execution_policy": {
            "evidence_class": "simulation",
            "execution_mode": "research_lab_delegated",
            "external_actions_allowed": False,
            "deployment_allowed": False,
            "market_contact_allowed": False,
            "payment_allowed": False,
            "contract_acceptance_allowed": False,
            "secret_material_allowed": False,
            "foundry_checkout_write_allowed": False,
            "foundry_ledger_write_allowed": False,
            "direct_apply_allowed": False,
            "productzero_start_allowed": False,
            "named_human_research_approval_required": True,
        },
        "created_at": "2026-08-30T00:00:00Z",
    }
    intent_identity = {
        key: value for key, value in intent.items() if key not in {"intent_id", "created_at"}
    }
    intent["intent_id"] = f"rli_{sha256_text(canonical_json(intent_identity))[:16]}"
    intent_path = _write_json(root / "artifacts/research_lab/intent.json", intent)
    intent_ref = {"locator": str(intent_path.resolve()), "sha256": _sha(intent_path)}
    descriptor: dict[str, Any] = {
        "schema_version": "forge-external-research-descriptor.v1",
        "adapter": {
            "adapter_id": "foundry-research-lab-intent.v1",
            "intent_schema_version": "foundry-research-lab-intent.v1",
        },
        "domain": {
            "domain_id": "foundry",
            "task_id": intent["opportunity_id"],
            "intent_id": intent["intent_id"],
        },
        "bindings": {
            "intent": intent_ref,
            "budget": intent_ref,
            "budget_source": "intent:/launch",
        },
        "control": {
            "praxist_executable": str(fixture.executable.resolve()),
            "run_dir": str(fixture.run_dir),
            "config_file": None,
        },
        "result_policy": {
            "result_locator": result_locator,
            "evidence_class": "simulation",
            "adoption_mode": "proposal_only",
            "market_validation_claimed": False,
            "adoption_status": "pending_external_human_review",
            "domain_human_review_required": True,
        },
        "authority": {
            "external_actions_allowed": False,
            "foundry_checkout_write_allowed": False,
            "foundry_ledger_write_allowed": False,
            "direct_apply_allowed": False,
            "productzero_start_allowed": False,
            "volvence_promotion_eligible": False,
            "modification_gate_applicable": False,
            "runtime_wiring_applicable": False,
        },
        "created_at": "2026-08-30T00:01:00Z",
    }
    descriptor_identity = {
        key: value
        for key, value in descriptor.items()
        if key not in {"descriptor_id", "created_at"}
    }
    descriptor["descriptor_id"] = (
        f"external-research-descriptor:{sha256_text(canonical_json(descriptor_identity))}"
    )
    return _write_json(root / "artifacts/research_lab/descriptor.json", descriptor)


def _approve(fixture: ControlFixture, request_path: Path):
    return review_research_request(
        config=fixture.config,
        request_path=request_path,
        reviewed_by="human@example.com",
        reason="Approve the exact bounded research run only.",
    )


def _option(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_submit_seals_non_authorizing_request_and_inbox_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    request = validate_research_request(
        config=fixture.config,
        request_path=result.request_path,
    )

    assert request["authority"]["research_start_authorized"] is False
    assert request["authority"]["production_promotion_authorized"] is False
    assert request["bindings"]["task_project"]["manifest_sha256"] == json.loads(
        fixture.task_manifest.read_text(encoding="utf-8")
    )["praxist"]["task_project_manifest_sha256"]
    assert request["launch"]["run_id"] == fixture.run_dir.name
    assert inspect_research_request(
        config=fixture.config,
        request_path=result.request_path,
    ).state == "AWAITING_RESEARCH_APPROVAL"
    assert [item.request_id for item in list_research_inbox(config=fixture.config)] == [
        result.request_id
    ]


def test_pre_a0_source_drift_seals_exact_supersession_without_carrying_approval(
    tmp_path: Path,
) -> None:
    fixture, runtime = _source_backed_fixture(_fixture(tmp_path))

    old = _submit(fixture)
    runtime.write_text("REVISION = 2\n", encoding="utf-8")
    replacement = _submit(fixture)
    assert old.request_id != replacement.request_id

    sealed = supersede_unreviewed_research_request(
        config=fixture.config,
        request_path=old.request_path,
        replacement_request_path=replacement.request_path,
        superseded_by="forge:demand-discovery-loop.v1",
        reason="The exact source checkout changed before A0.",
    )
    old_status = inspect_research_request(
        config=fixture.config,
        request_path=old.request_path,
    )
    replacement_status = inspect_research_request(
        config=fixture.config,
        request_path=replacement.request_path,
    )
    artifact = json.loads(sealed.supersession_path.read_text(encoding="utf-8"))

    assert old_status.state == "SUPERSEDED"
    assert old_status.supersession_path == sealed.supersession_path
    assert old_status.replacement_request_id == replacement.request_id
    assert old_status.replacement_request_path == replacement.request_path.resolve()
    assert replacement_status.state == "AWAITING_RESEARCH_APPROVAL"
    assert artifact["change"]["before"]["tree_sha256"] != artifact["change"]["after"][
        "tree_sha256"
    ]
    assert artifact["authority"]["human_a0_carried_forward"] is False

    runner = FakePraxistRunner(fixture)
    assert reconcile_research_control(
        config=fixture.config,
        request_path=old.request_path,
        runner=runner,
    )[0].state == "SUPERSEDED"
    assert runner.calls == []
    with pytest.raises(ResearchControlError, match="superseded ResearchRequest"):
        _approve(fixture, old.request_path)


def test_pre_a0_source_supersession_refuses_an_a0_reviewed_request(
    tmp_path: Path,
) -> None:
    fixture, runtime = _source_backed_fixture(_fixture(tmp_path))
    old = _submit(fixture)
    _approve(fixture, old.request_path)
    runtime.write_text("REVISION = 2\n", encoding="utf-8")
    replacement = _submit(fixture)

    with pytest.raises(ResearchControlError, match="A0-reviewed"):
        supersede_unreviewed_research_request(
            config=fixture.config,
            request_path=old.request_path,
            replacement_request_path=replacement.request_path,
            superseded_by="forge:demand-discovery-loop.v1",
            reason="The exact source checkout changed after A0.",
        )


def test_submit_requires_exact_runtime_selection_and_normalizes_codex_native(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ResearchControlError, match="explicit model"):
        _submit(fixture, model=None)

    result = _submit(
        fixture,
        codex_native=True,
        agent_system=None,
        runtime=None,
        model_provider=None,
        model="catalog-approved-model",
    )
    request = json.loads(result.request_path.read_text(encoding="utf-8"))
    profile = request["launch"]["profile"]
    assert profile["agent_system"] == "codex_sdk"
    assert profile["runtime"] == "agent_runtime:codex_sdk"
    assert profile["model_provider"] == "model_provider:openai_compatible"


def test_reconcile_never_calls_praxist_without_approval_or_after_rejection(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    runner = FakePraxistRunner(fixture)

    awaiting = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert awaiting.state == "AWAITING_RESEARCH_APPROVAL"
    assert runner.calls == []

    review_research_request(
        config=fixture.config,
        request_path=result.request_path,
        reviewed_by="human@example.com",
        reason="Do not spend this research budget.",
        decision="REJECT",
    )
    rejected = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert rejected.state == "REJECTED"
    assert runner.calls == []


def test_approved_request_starts_once_and_polls_exact_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture, model="model; touch should-not-run")
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)

    launched = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert launched.state == "RUNNING"
    assert launched.run_id == fixture.run_dir.name
    assert launched.monitor_command == f"praxist --monitor --run-id {fixture.run_dir.name}"
    assert [call[1] for call in runner.calls] == [
        "status",
        "doctor",
        "resolve",
        "status",
        "start",
        "status",
    ]
    start_command = next(call for call in runner.calls if call[1] == "start")
    assert "--daemonize" in start_command
    assert "--json" in start_command
    assert _option(start_command, "--run-dir") == str(fixture.run_dir)
    assert _option(start_command, "--model") == "model; touch should-not-run"

    event_paths = sorted((result.request_path.parent / "events").glob("*.json"))
    event_count = len(event_paths)
    again = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert again.state == "RUNNING"
    assert runner.start_count == 1
    assert runner.calls[-1][1:] == (
        "status",
        "--run-id",
        fixture.run_dir.name,
        "--json",
    )
    assert len(list((result.request_path.parent / "events").glob("*.json"))) == event_count


def test_control_directive_rejects_a_stale_event_revision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    launched = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]

    assert launched.latest_event_sha256 is not None
    with pytest.raises(ResearchControlError, match="event revision is stale"):
        issue_research_control_directive(
            config=fixture.config,
            request_path=result.request_path,
            action="PAUSE",
            expected_event_sha256="0" * 64,
            requested_by="human@example.com",
            reason="This old browser revision must not control the current run.",
        )


def test_pause_and_resume_use_exact_run_and_completed_generation_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(
        fixture,
        codex_native=True,
        agent_system=None,
        runtime=None,
        model_provider=None,
    )
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    launched = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert launched.latest_event_sha256 is not None

    pause = issue_research_control_directive(
        config=fixture.config,
        request_path=result.request_path,
        action="PAUSE",
        expected_event_sha256=launched.latest_event_sha256,
        requested_by="human@example.com",
        reason="Pause at a durable generation boundary.",
        grace_seconds=42,
    )
    paused = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert pause.action == "PAUSE"
    assert paused.state == "PAUSED"
    assert paused.latest_event_sha256 is not None
    stop_command = next(call for call in runner.calls if call[1] == "stop")
    assert stop_command == (
        str(fixture.executable),
        "stop",
        fixture.run_dir.name,
        "--grace",
        "42",
        "--json",
    )

    resume = issue_research_control_directive(
        config=fixture.config,
        request_path=result.request_path,
        action="RESUME",
        expected_event_sha256=paused.latest_event_sha256,
        requested_by="human@example.com",
        reason="Continue from the last completed generation.",
    )
    running = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert resume.action == "RESUME"
    assert running.state == "RUNNING"
    assert runner.resume_count == 1
    resume_command = next(call for call in runner.calls if call[1] == "resume")
    assert resume_command[2] == str(fixture.run_dir)
    assert _option(resume_command, "--resume-policy") == "completed_generation"
    assert "--codex-native" in resume_command
    assert "--force" not in resume_command

    reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )
    assert runner.resume_count == 1


def test_cancel_from_paused_is_terminal_without_a_second_stop(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    launched = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert launched.latest_event_sha256 is not None
    issue_research_control_directive(
        config=fixture.config,
        request_path=result.request_path,
        action="PAUSE",
        expected_event_sha256=launched.latest_event_sha256,
        requested_by="human@example.com",
        reason="Pause before deciding whether to cancel.",
    )
    paused = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert paused.latest_event_sha256 is not None
    assert runner.stop_count == 1

    issue_research_control_directive(
        config=fixture.config,
        request_path=result.request_path,
        action="CANCEL",
        expected_event_sha256=paused.latest_event_sha256,
        requested_by="human@example.com",
        reason="Close the paused research run without promotion.",
    )
    cancelled = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert cancelled.state == "CANCELLED"
    assert runner.stop_count == 1


def test_running_request_reaches_terminal_completion_without_promotion_authority(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )

    runner.state = "completed"
    completed = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    call_count = len(runner.calls)
    assert completed.state == "RUN_COMPLETED"
    last_event = json.loads(completed.latest_event_path.read_text(encoding="utf-8"))
    assert last_event["authority"]["production_promotion_authorized"] is False
    assert last_event["authority"]["runtime_wiring_changed"] is False

    terminal_again = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert terminal_again.state == "RUN_COMPLETED"
    assert len(runner.calls) == call_count


def test_foundry_intent_uses_shared_lifecycle_and_seals_simulation_handoff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    descriptor_path = _external_descriptor(fixture)
    descriptor = validate_external_research_descriptor(
        config=fixture.config,
        descriptor_path=descriptor_path,
    )
    submitted = submit_external_research_request(
        config=fixture.config,
        descriptor_path=descriptor_path,
        requested_by="lab-operator@example.com",
        reason="Submit the exact Foundry Intent for A0 review.",
    )
    request = validate_research_request(
        config=fixture.config,
        request_path=submitted.request_path,
    )

    assert descriptor["adapter"]["intent_schema_version"] == (
        "foundry-research-lab-intent.v1"
    )
    assert request["schema_version"] == "forge-external-research-request.v1"
    assert request["external_domain"]["domain_id"] == "foundry"
    assert request["bindings"]["external_intent"] == request["bindings"]["external_budget"]
    assert request["launch"]["profile"]["cohort"] == 2
    assert request["authority"]["result_evidence_class"] == "simulation"
    assert request["authority"]["foundry_checkout_write_allowed"] is False
    assert request["authority"]["foundry_ledger_write_allowed"] is False
    assert request["authority"]["direct_apply_allowed"] is False
    assert request["authority"]["productzero_start_allowed"] is False
    assert request["authority"]["production_promotion_authorized"] is False
    assert request["authority"]["modification_gate_applicable"] is False
    assert request["authority"]["runtime_wiring_applicable"] is False

    _approve(fixture, submitted.request_path)
    runner = FakePraxistRunner(fixture)
    running = reconcile_research_control(
        config=fixture.config,
        request_path=submitted.request_path,
        runner=runner,
    )[0]
    assert running.state == "RUNNING"
    assert runner.start_count == 1

    _write_json(fixture.run_dir / "result.json", {"best_variant": "candidate-a"})
    runner.state = "completed"
    completed = reconcile_research_control(
        config=fixture.config,
        request_path=submitted.request_path,
        runner=runner,
    )[0]
    assert completed.state == "RUN_COMPLETED"

    handoff_result = record_external_research_handoff(
        config=fixture.config,
        request_path=submitted.request_path,
        recorded_by="lab-operator@example.com",
        reason="Return simulation evidence to Foundry for its own review.",
    )
    handoff = json.loads(handoff_result.handoff_path.read_text(encoding="utf-8"))
    assert handoff["schema_version"] == "forge-foundry-research-handoff.v1"
    assert handoff["contract"]["contract_version"] == "foundry-research-lab-seam.v1"
    schema_path = fixture.config.paths.forge_root / "schemas/foundry_research_handoff.schema.json"
    assert handoff["contract"]["schema"]["sha256"] == _sha(schema_path)
    assert handoff["request"]["request_id"] == submitted.request_id
    assert handoff["descriptor"]["descriptor_id"] == descriptor["descriptor_id"]
    assert handoff["approval"]["decision"] == "APPROVE"
    assert handoff["approval"]["scope"] == "praxist_research_start"
    assert handoff["approval"]["reviewed_by"] == "human@example.com"
    assert handoff["terminal_event"]["state"] == "RUN_COMPLETED"
    assert handoff["result"]["evidence_class"] == "simulation"
    assert handoff["result"]["adoption_mode"] == "proposal_only"
    chain = handoff["hash_chain"]
    chain_body = {key: value for key, value in chain.items() if key != "chain_sha256"}
    assert chain["chain_sha256"] == sha256_text(canonical_json(chain_body))
    assert chain["request"]["sha256"] == handoff["request"]["sha256"]
    assert chain["approval"]["sha256"] == handoff["approval"]["sha256"]
    assert chain["run_completion"]["sha256"] == handoff["terminal_event"]["sha256"]
    assert chain["result"] == handoff["result"]["artifact"]
    assert handoff["consumer_permissions"] == {
        "allowed_operations": ["import_simulation_handoff"],
        "prohibited_operations": [
            "approve_research_request",
            "reconcile_research_control",
            "start_praxist",
            "import_volvence_candidate",
            "modify_runtime_wiring",
        ],
    }
    assert handoff["authority"]["foundry_approve_allowed"] is False
    assert handoff["authority"]["foundry_reconcile_allowed"] is False
    assert handoff["authority"]["foundry_praxist_start_allowed"] is False
    assert handoff["authority"]["volvence_promotion_eligible"] is False
    assert handoff["authority"]["modification_gate_applicable"] is False
    assert handoff["authority"]["runtime_wiring_applicable"] is False

    exit_code = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "--transcripts-root",
            str(fixture.config.paths.transcripts_root),
            "research-handoff-external",
            str(submitted.request_path),
            "--recorded-by",
            "lab-operator@example.com",
            "--reason",
            "Return simulation evidence to Foundry for its own review.",
            "--json",
        ]
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert cli_payload["schema_version"] == "forge-foundry-research-handoff-result.v1"
    assert cli_payload["contract_version"] == "foundry-research-lab-seam.v1"
    assert cli_payload["contract_schema_sha256"] == _sha(schema_path)
    assert cli_payload["hash_chain"] == handoff["hash_chain"]
    assert cli_payload["consumer_permissions"] == handoff["consumer_permissions"]

    with pytest.raises(ResearchControlError, match="already has an immutable handoff"):
        record_external_research_handoff(
            config=fixture.config,
            request_path=submitted.request_path,
            recorded_by="different-operator@example.com",
            reason="A changed handoff must not overwrite the sealed artifact.",
        )


def test_external_descriptor_fails_closed_on_budget_or_result_escape(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor_path = _external_descriptor(fixture)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["bindings"]["budget"] = {
        "locator": str(fixture.task_manifest),
        "sha256": _sha(fixture.task_manifest),
    }
    descriptor_identity = {
        key: value
        for key, value in descriptor.items()
        if key not in {"descriptor_id", "created_at"}
    }
    descriptor["descriptor_id"] = (
        f"external-research-descriptor:{sha256_text(canonical_json(descriptor_identity))}"
    )
    _write_json(descriptor_path, descriptor)
    with pytest.raises(ResearchControlError, match="same exact Intent"):
        submit_external_research_request(
            config=fixture.config,
            descriptor_path=descriptor_path,
            requested_by="lab-operator@example.com",
            reason="This mismatched budget binding must fail.",
        )

    unsafe_path = _external_descriptor(fixture, result_locator="../outside.json")
    with pytest.raises(ResearchControlError, match="unsafe external result locator"):
        validate_external_research_descriptor(
            config=fixture.config,
            descriptor_path=unsafe_path,
        )


def test_external_intent_drift_after_a0_blocks_before_praxist(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    descriptor_path = _external_descriptor(fixture)
    submitted = submit_external_research_request(
        config=fixture.config,
        descriptor_path=descriptor_path,
        requested_by="lab-operator@example.com",
        reason="Submit an exact Intent that must remain immutable.",
    )
    _approve(fixture, submitted.request_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    intent_path = Path(descriptor["bindings"]["intent"]["locator"])
    intent_path.write_text(intent_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    runner = FakePraxistRunner(fixture)

    status = reconcile_research_control(
        config=fixture.config,
        request_path=submitted.request_path,
        runner=runner,
    )[0]
    assert status.state == "BLOCKED"
    assert runner.calls == []


def test_cli_external_submit_returns_stable_machine_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    descriptor_path = _external_descriptor(fixture)
    exit_code = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "--transcripts-root",
            str(fixture.config.paths.transcripts_root),
            "research-submit-external",
            str(descriptor_path),
            "--requested-by",
            "lab-operator@example.com",
            "--reason",
            "Submit exact Foundry Intent through the machine CLI.",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "forge-external-research-submit-result.v1"
    assert payload["state"] == "AWAITING_RESEARCH_APPROVAL"
    assert payload["domain_id"] == "foundry"
    assert payload["evidence_class"] == "simulation"
    assert payload["praxist_started"] is False
    assert payload["volvence_promotion_eligible"] is False
    assert len(payload["request_sha256"]) == 64


def test_active_host_capacity_queues_approved_request(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    runner.external_active = True

    waiting = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert waiting.state == "WAITING_FOR_CAPACITY"
    assert runner.start_count == 0
    assert [call[1] for call in runner.calls] == ["status"]

    runner.external_active = False
    launched = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert launched.state == "RUNNING"
    assert runner.start_count == 1


def test_portfolio_a0_allows_known_bounded_parallel_runs(tmp_path: Path) -> None:
    first = _fixture(tmp_path)
    second = _clone_fixture(first)
    portfolio, first_demand, second_demand = _portfolio_for_fixtures(first, second)
    first_request = _submit(first, evidence_paths=(first_demand,))
    second_request = _submit(second, evidence_paths=(second_demand,))
    inferred_approval = review_research_request(
        config=first.config,
        request_path=first_request.request_path,
        reviewed_by="human@example.com",
        reason="Approve study A under the exact portfolio capacity budget.",
    )
    inferred_payload = json.loads(
        inferred_approval.approval_path.read_text(encoding="utf-8")
    )
    assert inferred_payload["execution_policy"]["portfolio"]["artifact_id"] == (
        json.loads(portfolio.read_text(encoding="utf-8"))["portfolio_id"]
    )
    assert inferred_payload["execution_policy"]["study_id"] == "memory_reader_a"
    review_research_request(
        config=second.config,
        request_path=second_request.request_path,
        reviewed_by="human@example.com",
        reason="Approve study B under the exact portfolio capacity budget.",
        portfolio_path=portfolio,
        portfolio_study_id="memory_reader_b",
    )

    first_runner = FakePraxistRunner(first)
    first_status = reconcile_research_control(
        config=first.config,
        request_path=first_request.request_path,
        runner=first_runner,
    )[0]
    assert first_status.state == "RUNNING"

    second_runner = FakePraxistRunner(second)
    second_runner.known_active_rows = [first_runner._status_row()]
    second_status = reconcile_research_control(
        config=second.config,
        request_path=second_request.request_path,
        runner=second_runner,
    )[0]
    assert second_status.state == "RUNNING"
    assert second_runner.start_count == 1
    capacity_events = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (second_request.request_path.parent / "events").glob("*.json")
        )
        if json.loads(path.read_text(encoding="utf-8"))["kind"]
        == "CAPACITY_OBSERVED"
    ]
    assert any(
        "portfolio_active_run_count=1" in event["details"]
        for event in capacity_events
    )


def test_portfolio_a0_enforces_exact_global_and_lane_limit(tmp_path: Path) -> None:
    first = _fixture(tmp_path)
    second = _clone_fixture(first)
    portfolio, first_demand, second_demand = _portfolio_for_fixtures(
        first,
        second,
        max_active_runs=1,
    )
    first_request = _submit(first, evidence_paths=(first_demand,))
    second_request = _submit(second, evidence_paths=(second_demand,))
    for fixture, request, study_id in (
        (first, first_request, "memory_reader_a"),
        (second, second_request, "memory_reader_b"),
    ):
        review_research_request(
            config=fixture.config,
            request_path=request.request_path,
            reviewed_by="human@example.com",
            reason="Approve only the exact single-run portfolio capacity.",
            portfolio_path=portfolio,
            portfolio_study_id=study_id,
        )

    first_runner = FakePraxistRunner(first)
    reconcile_research_control(
        config=first.config,
        request_path=first_request.request_path,
        runner=first_runner,
    )
    second_runner = FakePraxistRunner(second)
    second_runner.known_active_rows = [first_runner._status_row()]
    waiting = reconcile_research_control(
        config=second.config,
        request_path=second_request.request_path,
        runner=second_runner,
    )[0]
    assert waiting.state == "WAITING_FOR_CAPACITY"
    assert second_runner.start_count == 0
    event = json.loads(waiting.latest_event_path.read_text(encoding="utf-8"))
    assert "portfolio_global_limit_reached=1/1" in event["details"]
    assert "portfolio_lane_limit_reached=bounded_parallel:1/1" in event["details"]


def test_capacity_is_rechecked_after_resolve_before_start(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    runner.activate_external_after_resolve = True

    waiting = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert waiting.state == "WAITING_FOR_CAPACITY"
    assert runner.start_count == 0
    assert [call[1] for call in runner.calls] == ["status", "doctor", "resolve", "status"]

    runner.external_active = False
    runner.activate_external_after_resolve = False
    launched = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert launched.state == "RUNNING"
    assert runner.start_count == 1


def test_task_tampering_after_approval_blocks_before_praxist(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    (fixture.task_project / "task.yaml").write_text(
        (fixture.task_project / "task.yaml").read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    runner = FakePraxistRunner(fixture)

    blocked = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert blocked.state == "BLOCKED"
    assert runner.calls == []
    event = json.loads(blocked.latest_event_path.read_text(encoding="utf-8"))
    assert event["details"] == ["Praxist task project changed after Request submission"]


def test_resolve_failure_blocks_start_and_does_not_persist_raw_stderr(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    runner.fail_phase = "resolve"

    blocked = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert blocked.state == "BLOCKED"
    assert runner.start_count == 0
    encoded = blocked.latest_event_path.read_text(encoding="utf-8")
    assert "secret-value-must-not-persist" not in encoded
    event = json.loads(encoded)
    assert event["command"]["stderr_sha256"] == sha256_text(
        "resolve failed: secret-value-must-not-persist"
    )


def test_resolve_manifest_mismatch_blocks_before_start(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    runner.corrupt_resolved_manifest = True

    blocked = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert blocked.state == "BLOCKED"
    assert runner.start_count == 0
    event = json.loads(blocked.latest_event_path.read_text(encoding="utf-8"))
    assert event["details"] == [
        "Praxist preflight task project digest does not match the Request"
    ]


def test_start_timeout_recovers_exact_run_without_duplicate_launch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    runner.timeout_after_launch = True

    recovered = reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )[0]
    assert recovered.state == "RUNNING"
    assert runner.start_count == 1
    reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )
    assert runner.start_count == 1


def test_review_revalidates_bindings_and_request_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    payload = json.loads(result.request_path.read_text(encoding="utf-8"))
    payload["launch"]["profile"]["generations"] = 99
    result.request_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ResearchControlError, match="request_id does not match"):
        _approve(fixture, result.request_path)


def test_event_chain_tampering_fails_loudly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    _approve(fixture, result.request_path)
    runner = FakePraxistRunner(fixture)
    reconcile_research_control(
        config=fixture.config,
        request_path=result.request_path,
        runner=runner,
    )
    first_event = sorted((result.request_path.parent / "events").glob("*.json"))[0]
    payload = json.loads(first_event.read_text(encoding="utf-8"))
    payload["details"] = ["tampered"]
    first_event.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ResearchControlError, match="event identity is invalid"):
        inspect_research_request(
            config=fixture.config,
            request_path=result.request_path,
        )


def test_task_snapshot_digest_matches_canonical_file_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _submit(fixture)
    request = json.loads(result.request_path.read_text(encoding="utf-8"))
    files = request["bindings"]["task_project"]["files"]
    assert request["bindings"]["task_project"]["snapshot_sha256"] == sha256_text(
        canonical_json(files)
    )


def test_cli_submit_inbox_and_human_review(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    common = [
        "--repo-root",
        str(fixture.config.paths.repo_root),
        "--transcripts-root",
        str(fixture.config.paths.transcripts_root),
    ]
    assert (
        main(
            [
                *common,
                "research-submit",
                str(fixture.task_manifest),
                "--task-project",
                str(fixture.task_project),
                "--praxist-executable",
                str(fixture.executable),
                "--run-dir",
                str(fixture.run_dir),
                "--requested-by",
                "detector:cli-test",
                "--reason",
                "Exercise the typed A0 command surface.",
                "--agent-system",
                "claude_sdk",
                "--runtime",
                "agent_runtime:claude_sdk",
                "--model-provider",
                "model_provider:fake",
                "--model",
                "fake-model",
            ]
        )
        == 0
    )
    assert "AWAITING_RESEARCH_APPROVAL" in capsys.readouterr().out
    request_path = next(
        fixture.config.paths.artifacts_root.glob("research_control/*/*/request.json")
    )

    assert main([*common, "research-inbox", "--json"]) == 0
    inbox = json.loads(capsys.readouterr().out)
    assert inbox[0]["state"] == "AWAITING_RESEARCH_APPROVAL"

    assert (
        main(
            [
                *common,
                "research-approve",
                str(request_path),
                "--approved-by",
                "human@example.com",
                "--reason",
                "Approve only the exact research-start scope.",
            ]
        )
        == 0
    )
    assert "APPROVE" in capsys.readouterr().out
