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
from volvence_forge.foundation import canonical_json, sha256_bytes, sha256_text
from volvence_forge.research_control import (
    CommandExecution,
    ResearchControlError,
    SubprocessPraxistRunner,
    inspect_research_request,
    list_research_inbox,
    reconcile_research_control,
    review_research_request,
    submit_research_request,
    validate_research_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        self.activate_external_after_resolve = False
        self.corrupt_resolved_manifest = False
        self.fail_phase: str | None = None
        self.timeout_after_launch = False
        self.start_count = 0

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
        raise AssertionError(f"unexpected fake Praxist command: {command}")

    def _status(self, command: tuple[str, ...]) -> CommandExecution:
        rows: list[dict[str, Any]] = []
        if "--active" in command and self.external_active:
            rows.append(
                {
                    "run_id": "external-run",
                    "run_dir": "/tmp/external-run",
                    "pid": 999,
                    "source": "registry",
                    "state": "running",
                }
            )
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


def _project_manifest(task_project: Path) -> dict[str, Any]:
    task_yaml = task_project / "task.yaml"
    content = task_yaml.read_bytes()
    digest = hashlib.sha256()
    digest.update(b"task.yaml\0")
    digest.update(content)
    digest.update(b"\0")
    return {
        "schema_version": "task_project_manifest.v1",
        "source": "external_task_project",
        "task_id": "memory_inheritance_project",
        "task_ref": "task:memory_inheritance_project",
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
