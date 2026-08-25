from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import os
import pathlib
import sys
from typing import Mapping

import pytest

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads

from lifeform_evolution import (
    relationship_condition_reader_qualification_execution as execution,
)
from lifeform_evolution import (
    relationship_condition_reader_qualification_execution_protocol as execution_protocol,
)


_QUALIFICATION_ID = hashlib.sha256(b"qualification").hexdigest()
_EXECUTION_ID = hashlib.sha256(b"execution").hexdigest()
_RUN_NONCE = hashlib.sha256(b"run-nonce").hexdigest()
_PREVIOUS_INTEGRITY_ID = hashlib.sha256(b"post-prediction-child-2").hexdigest()
_SOURCE_TREE_ID = hashlib.sha256(b"source-tree").hexdigest()
_BGE_TREE_ID = hashlib.sha256(b"bge-tree").hexdigest()
_RUNTIME_ID = hashlib.sha256(b"runtime").hexdigest()
_SCORER_MODULE_REPOSITORY_PATHS = {
    "lifeform_evolution.relationship_condition_reader_qualification_runtime_binding": (
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_runtime_binding.py"
    ),
    "lifeform_evolution.relationship_condition_reader_qualification_scorer": (
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_scorer.py"
    ),
    "volvence_zero.canonical_json": ("packages/vz-contracts/src/volvence_zero/canonical_json.py"),
    "volvence_zero.social_cognition": ("packages/vz-contracts/src/volvence_zero/social_cognition.py"),
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    return {
        **core,
        "artifact_id": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }


def _write_artifact(
    path: pathlib.Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    raw = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": path.name,
        "artifact_id": payload["artifact_id"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
    }


def _scoring_request(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, object]]:
    root = tmp_path.resolve()
    request = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-scoring-request.v1"),
            "qualification_protocol_id": _QUALIFICATION_ID,
            "execution_protocol_id": _EXECUTION_ID,
            "run_nonce": _RUN_NONCE,
            "prediction_ledger_path": str(root / "commit" / "prediction_ledger.json"),
            "prediction_ledger_artifact_id": _digest("ledger"),
            "prediction_ledger_raw_sha256": _digest("ledger-raw"),
            "prediction_ledger_raw_bytes": 101,
            "commit_receipt_path": str(root / "commit" / "commit_receipt.json"),
            "commit_receipt_artifact_id": _digest("commit"),
            "challenge_labels_path": str(root / "sealed" / "challenge_labels.json"),
            "challenge_labels_artifact_id": _digest("challenge"),
            "challenge_labels_raw_sha256": _digest("challenge-raw"),
            "challenge_labels_raw_bytes": 202,
            "group_split_path": str(root / "sealed" / "group_split.json"),
            "group_split_artifact_id": _digest("groups"),
            "group_split_raw_sha256": _digest("groups-raw"),
            "group_split_raw_bytes": 303,
            "minimum_normalized_margin_hex": (0.01).hex(),
        }
    )
    path = root / "scoring_request.json"
    _write_artifact(path, request)
    return path, request


class _FakeIntegrityGuard:
    def __init__(self, *, drift_phase: str | None = None) -> None:
        self.drift_phase = drift_phase
        self.calls: list[tuple[str, str | None]] = []
        self.receipts: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        phase: str,
        previous_integrity_receipt_artifact_id: str | None,
    ) -> Mapping[str, object]:
        self.calls.append((phase, previous_integrity_receipt_artifact_id))
        source_tree_id = _digest("drifted-source") if phase == self.drift_phase else _SOURCE_TREE_ID
        receipt = _with_artifact_id(
            {
                "schema_version": (execution.RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION),
                "execution_protocol_id": _EXECUTION_ID,
                "phase": phase,
                "phase_ordinal": execution._INTEGRITY_PHASE_ORDINALS[phase],
                "previous_integrity_receipt_artifact_id": (previous_integrity_receipt_artifact_id),
                "source_tree_artifact_id": source_tree_id,
                "source_tree_entry_count": 41,
                "bge_snapshot_tree_artifact_id": _BGE_TREE_ID,
                "bge_snapshot_entry_count": 11,
                "runtime_identity_artifact_id": _RUNTIME_ID,
                "source_tree_exact": True,
                "bge_snapshot_tree_exact": True,
                "runtime_identity_exact": True,
                "observer_model_or_cuda_execution_used": False,
                "torch_imported": False,
                "sentence_transformers_imported": False,
                "os_security_boundary": False,
                "windows_directory_entry_durability_attested": False,
            }
        )
        self.receipts.append(receipt)
        return receipt


def _capture(raw: bytes) -> execution._BoundedStreamCapture:
    digest = hashlib.sha256(raw).hexdigest()
    return execution._BoundedStreamCapture(
        raw_sha256=digest,
        total_bytes=len(raw),
        retained_prefix=raw,
        retained_prefix_sha256=digest,
        retained_prefix_bytes=len(raw),
        prefix_truncated=False,
    )


class _FakeScorerLauncher:
    def __init__(
        self,
        *,
        admitted: bool = True,
        exit_code: int = 0,
        write_outputs: bool = True,
        report_overrides: Mapping[str, object] | None = None,
        attestation_overrides: Mapping[str, object] | None = None,
        result_overrides: Mapping[str, object] | None = None,
        tamper_manifest_receipt: bool = False,
        extra_output: bool = False,
        contaminate_capsule: bool = False,
        tamper_frozen_source: bool = False,
        swap_module_origin: bool = False,
    ) -> None:
        self.admitted = admitted
        self.exit_code = exit_code
        self.write_outputs = write_outputs
        self.report_overrides = dict(report_overrides or {})
        self.attestation_overrides = dict(attestation_overrides or {})
        self.result_overrides = dict(result_overrides or {})
        self.tamper_manifest_receipt = tamper_manifest_receipt
        self.extra_output = extra_output
        self.contaminate_capsule = contaminate_capsule
        self.tamper_frozen_source = tamper_frozen_source
        self.swap_module_origin = swap_module_origin
        self.calls: list[execution._ScorerLaunchSpec] = []

    def __call__(
        self,
        spec: execution._ScorerLaunchSpec,
    ) -> execution._ScorerLaunchResult:
        self.calls.append(spec)
        process_id = os.getpid() + 10_000
        request = strict_json_loads(
            spec.scoring_request_path.read_bytes(),
            max_bytes=2_000_000,
        )
        assert isinstance(request, dict)
        if self.tamper_frozen_source:
            source_path = spec.import_binding.repository_root / pathlib.PurePosixPath(
                _SCORER_MODULE_REPOSITORY_PATHS["volvence_zero.social_cognition"]
            )
            source_path.write_bytes(b"drifted after source freeze\n")
        if self.write_outputs:
            self._write_outputs(spec, request=request, process_id=process_id)
        if self.contaminate_capsule:
            (spec.capsule_root / "unexpected.txt").write_text(
                "unexpected",
                encoding="utf-8",
            )
        result = execution._ScorerLaunchResult(
            process_id=process_id,
            exit_code=self.exit_code,
            process_exited=True,
            job_object_empty=True,
            environment_contract_id=spec.environment_contract_id,
            environment_projection=spec.environment_projection,
            creation_flags=execution._FORMAL_SCORER_CREATION_FLAGS,
            shell=False,
            close_fds=True,
            process_created_suspended=True,
            job_assigned_before_resume=True,
            initial_thread_resume_previous_count=1,
            job_limit_flags=execution._FORMAL_SCORER_JOB_LIMIT_FLAGS,
            job_active_process_limit=1,
            stdout_capture=_capture(b""),
            stderr_capture=_capture(b"fixture scorer failed" if self.exit_code else b""),
            source_capsule_used=False,
            repository_import_path_used=True,
        )
        return replace(result, **self.result_overrides)

    def _write_outputs(
        self,
        spec: execution._ScorerLaunchSpec,
        *,
        request: Mapping[str, object],
        process_id: int,
    ) -> None:
        output_root = spec.output_root
        output_root.mkdir()
        successful_rows = 224 if self.admitted else 223
        successful_groups = 28 if self.admitted else 27
        report = _with_artifact_id(
            {
                "schema_version": ("relationship-condition-reader-qualification-report.v1"),
                "qualification_protocol_id": request["qualification_protocol_id"],
                "execution_protocol_id": request["execution_protocol_id"],
                "scoring_request_artifact_id": request["artifact_id"],
                "prediction_ledger_artifact_id": request["prediction_ledger_artifact_id"],
                "challenge_labels_artifact_id": request["challenge_labels_artifact_id"],
                "group_split_artifact_id": request["group_split_artifact_id"],
                "row_count": 224,
                "effective_group_count": 28,
                "rows_per_group": 8,
                "correct_row_count": successful_rows,
                "margin_passing_row_count": successful_rows,
                "passing_row_count": successful_rows,
                "passing_group_count": successful_groups,
                "minimum_normalized_margin_hex": request["minimum_normalized_margin_hex"],
                "exact_source_reader_development_admitted": self.admitted,
                "verdict": (
                    "exact_source_reader_development_admitted"
                    if self.admitted
                    else "exact_source_reader_development_not_admitted"
                ),
                "statistical_independence_claim": False,
                "campaign_execution_admitted": False,
                "readable_product_effect": False,
                "appendable_product_effect": False,
                "learnable_product_effect": False,
                "steerable_product_effect": False,
                "four_able_complete": False,
                "formal_evidence_authorized": False,
                "human_product_validation": False,
                "production_active": False,
                **self.report_overrides,
            }
        )
        report_receipt = _write_artifact(output_root / "report.json", report)

        origins = [
            {
                "module_name": module_name,
                "origin": str(spec.import_binding.repository_root / pathlib.PurePosixPath(repository_path)),
            }
            for module_name, repository_path in sorted(_SCORER_MODULE_REPOSITORY_PATHS.items())
        ]
        if self.swap_module_origin:
            origins[-1] = {
                **origins[-1],
                "origin": origins[-2]["origin"],
            }
        attestation = _with_artifact_id(
            {
                "schema_version": ("relationship-condition-reader-qualification-scorer-attestation.v3"),
                "qualification_protocol_id": request["qualification_protocol_id"],
                "execution_protocol_id": request["execution_protocol_id"],
                "scoring_request_artifact_id": request["artifact_id"],
                "prediction_ledger_commit_artifact_id": request["commit_receipt_artifact_id"],
                "process_pid": process_id,
                "parent_pid": os.getpid(),
                "run_nonce": request["run_nonce"],
                "process_executable": str(spec.python_executable.resolve()),
                "process_argv": execution._expected_scorer_attestation_argv(spec),
                "process_cwd": str(spec.capsule_root.resolve()),
                "process_sys_path": list(
                    execution.expected_child_sys_path(
                        spec.import_binding,
                        python_version=(f"{sys.version_info.major}.{sys.version_info.minor}"),
                    )
                ),
                "process_runtime_flags": {
                    "dont_write_bytecode": True,
                    "no_site": 1,
                    "pycache_prefix": str(spec.pycache_prefix),
                    "safe_path": True,
                    "utf8_mode": 1,
                },
                "environment_key_names": [item.key for item in spec.environment_projection],
                "environment_value_sha256s": {item.key: item.value_sha256 for item in spec.environment_projection},
                "unlisted_environment_variables_recorded": True,
                "loaded_file_backed_module_origins": origins,
                "volvence_zero_namespace_search_locations": [
                    str(path) for path in spec.import_binding.volvence_zero_namespace_search_locations
                ],
                "event_sequence": list(execution._SCORER_EVENT_SEQUENCE),
                "challenge_labels_first_open_after_commit_validation": True,
                "model_or_cuda_used": False,
                "torch_imported": False,
                "sentence_transformers_imported": False,
                "os_security_boundary": False,
                "windows_directory_entry_durability_attested": False,
                **self.attestation_overrides,
            }
        )
        attestation_receipt = _write_artifact(
            output_root / "scorer_attestation.json",
            attestation,
        )
        if self.tamper_manifest_receipt:
            report_receipt["raw_sha256"] = _digest("wrong-report-bytes")
        manifest = _with_artifact_id(
            {
                "schema_version": ("relationship-condition-reader-qualification-scorer-manifest.v1"),
                "qualification_protocol_id": request["qualification_protocol_id"],
                "execution_protocol_id": request["execution_protocol_id"],
                "scoring_request_artifact_id": request["artifact_id"],
                "files": [report_receipt, attestation_receipt],
                "file_count": 2,
                "model_or_cuda_used": False,
            }
        )
        _write_artifact(output_root / "manifest.json", manifest)
        if self.extra_output:
            (output_root / "extra.json").write_text("{}\n", encoding="utf-8")


def _environment_source(system_root: pathlib.Path) -> dict[str, str]:
    return {
        "PATH": str(system_root / "poison-ambient-path"),
        "SystemRoot": str(system_root),
        "PYTHONPATH": r"D:\volvence\packages\lifeform-evolution\src",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_TOKEN": "must-not-cross",
        "AWS_SECRET_ACCESS_KEY": "must-not-cross",
    }


def _source_binding_fixture(
    root: pathlib.Path,
) -> tuple[
    pathlib.Path,
    tuple[pathlib.Path, ...],
    Mapping[str, Mapping[str, object]],
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
]:
    repository_root = root.resolve() / "repository"
    frozen_entries: dict[str, Mapping[str, object]] = {}
    for module_name, repository_path in _SCORER_MODULE_REPOSITORY_PATHS.items():
        source_path = repository_root / pathlib.PurePosixPath(repository_path)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"# frozen fixture for {module_name}\n".encode("utf-8")
        source_path.write_bytes(raw)
        frozen_entries[repository_path] = {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_bytes": len(raw),
        }
    source_roots = tuple(
        sorted(
            {
                repository_root / pathlib.PurePosixPath(*pathlib.PurePosixPath(path).parts[:3])
                for path in frozen_entries
            },
            key=lambda path: str(path).encode("utf-8"),
        )
    )
    python_home = root.resolve() / "python"
    python_executable = python_home / "python.exe"
    python_executable.parent.mkdir(parents=True, exist_ok=True)
    python_executable.write_bytes(b"fixture python executable\n")
    (python_home / "DLLs").mkdir()
    (python_home / "Library" / "bin").mkdir(parents=True)
    site_packages_root = python_home / "Lib" / "site-packages"
    site_packages_root.mkdir(parents=True, exist_ok=True)
    system_root = root.resolve() / "windows"
    (system_root / "System32").mkdir(parents=True)
    return (
        repository_root,
        source_roots,
        frozen_entries,
        site_packages_root,
        python_executable,
        system_root,
    )


def _execute(
    tmp_path: pathlib.Path,
    *,
    launcher: _FakeScorerLauncher | None = None,
    guard: _FakeIntegrityGuard | None = None,
    expected_request_id: str | None = None,
) -> tuple[
    Mapping[str, object],
    pathlib.Path,
    _FakeScorerLauncher,
    _FakeIntegrityGuard,
]:
    request_path, request = _scoring_request(tmp_path)
    fake_launcher = launcher or _FakeScorerLauncher()
    fake_guard = guard or _FakeIntegrityGuard()
    stage_root = tmp_path.resolve() / "stage"
    (
        repository_root,
        source_roots,
        source_entries,
        site_packages_root,
        python_executable,
        system_root,
    ) = _source_binding_fixture(tmp_path)
    result = execution._execute_scoring_stage_with_launcher(
        scoring_request_path=request_path,
        expected_scoring_request_artifact_id=(expected_request_id or str(request["artifact_id"])),
        stage_root=stage_root,
        python_executable=python_executable,
        environment_source=_environment_source(system_root),
        launcher=fake_launcher,
        integrity_guard=fake_guard,
        previous_integrity_receipt_artifact_id=_PREVIOUS_INTEGRITY_ID,
        expected_source_tree_artifact_id=_SOURCE_TREE_ID,
        expected_bge_snapshot_tree_artifact_id=_BGE_TREE_ID,
        expected_runtime_identity_artifact_id=_RUNTIME_ID,
        repository_root=repository_root,
        repository_source_roots=source_roots,
        frozen_source_entries=source_entries,
        frozen_site_packages_root=site_packages_root,
    )
    return result, stage_root, fake_launcher, fake_guard


def test_fixture_scorer_stage_binds_outputs_and_remains_unauthorized(
    tmp_path: pathlib.Path,
) -> None:
    result, stage_root, launcher, guard = _execute(tmp_path)

    assert len(launcher.calls) == 1
    assert [phase for phase, _previous in guard.calls] == [
        "pre_scorer",
        "post_scorer",
    ]
    assert guard.calls[0][1] == _PREVIOUS_INTEGRITY_ID
    assert guard.calls[1][1] == guard.receipts[0]["artifact_id"]
    assert result["last_integrity_receipt_artifact_id"] == guard.receipts[1]["artifact_id"]
    assert result["exact_source_reader_development_admitted"] is True
    assert result["scorer_model_free"] is True
    assert result["source_tree_artifact_id_verified"] is True
    assert result["bge_snapshot_tree_artifact_id_verified"] is True
    assert result["runtime_identity_artifact_id_verified"] is True
    for field_name in (
        "source_capsule_used",
        "external_execution_anchor_verified",
        "qualification_execution_authorized",
        "formal_evidence_authorized",
        "campaign_execution_admitted",
        "readable_product_effect",
        "appendable_product_effect",
        "learnable_product_effect",
        "steerable_product_effect",
        "four_able_complete",
        "human_product_validation",
        "production_active",
        "os_security_boundary",
        "windows_directory_entry_durability_attested",
    ):
        assert result[field_name] is False

    stage_manifest_path = stage_root / "scorer_stage_manifest.json"
    on_disk = strict_json_loads(
        stage_manifest_path.read_bytes(),
        max_bytes=2_000_000,
    )
    assert on_disk == result
    assert (
        result["artifact_id"]
        == hashlib.sha256(
            canonical_json_bytes({key: value for key, value in result.items() if key != "artifact_id"})
        ).hexdigest()
    )


def test_negative_scorer_verdict_is_a_valid_unpromoted_stage(
    tmp_path: pathlib.Path,
) -> None:
    result, _root, _launcher, _guard = _execute(
        tmp_path,
        launcher=_FakeScorerLauncher(admitted=False),
    )

    assert result["exact_source_reader_development_admitted"] is False
    assert result["verdict"] == "exact_source_reader_development_not_admitted"
    assert result["formal_evidence_authorized"] is False


def test_environment_is_built_from_empty_exact_allowlist(
    tmp_path: pathlib.Path,
) -> None:
    repo, source_roots, entries, site_packages, python_executable, system_root = _source_binding_fixture(tmp_path)
    import_binding = execution.build_qualification_child_import_binding(
        python_executable=python_executable,
        repository_root=repo,
        repository_source_roots=source_roots,
        frozen_source_entries=entries,
        frozen_site_packages_root=site_packages,
    )
    pycache_prefix = tmp_path / "isolated-pycache"
    environment, projection, contract_id = execution._build_formal_scorer_environment(
        _environment_source(system_root),
        import_binding=import_binding,
        pycache_prefix=pycache_prefix,
    )

    assert set(environment) == {
        "PATH",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "PYTHONUTF8",
        "SYSTEMROOT",
    }
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert "HF_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["PYTHONPATH"] == os.pathsep.join(str(path) for path in import_binding.import_roots)
    assert environment["PYTHONPYCACHEPREFIX"] == str(pycache_prefix)
    assert environment["PATH"] == os.pathsep.join(
        str(path)
        for path in execution.controlled_child_path(
            import_binding,
            system_root=system_root,
        )
    )
    assert [item.key for item in projection] == sorted(environment)
    assert contract_id == execution._environment_contract_id(projection)


def test_request_identity_mismatch_stops_before_guard_launch_or_stage_creation(
    tmp_path: pathlib.Path,
) -> None:
    launcher = _FakeScorerLauncher()
    guard = _FakeIntegrityGuard()
    with pytest.raises(ValueError, match="scoring request external artifact identity"):
        _execute(
            tmp_path,
            launcher=launcher,
            guard=guard,
            expected_request_id=_digest("wrong-request"),
        )

    assert launcher.calls == []
    assert guard.calls == []
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize(
    ("result_overrides", "message"),
    [
        ({"process_exited": False}, "containment"),
        ({"job_object_empty": False}, "containment"),
        ({"initial_thread_resume_previous_count": 0}, "containment"),
        ({"job_active_process_limit": 2}, "containment"),
        ({"creation_flags": 0}, "containment"),
    ],
)
def test_launcher_contract_drift_is_rejected(
    tmp_path: pathlib.Path,
    result_overrides: Mapping[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _execute(
            tmp_path,
            launcher=_FakeScorerLauncher(result_overrides=result_overrides),
        )
    assert not (tmp_path / "stage" / "scorer_stage_manifest.json").exists()


def test_nonzero_exit_reports_bounded_stderr_without_accepting_outputs(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(RuntimeError, match="fixture scorer failed"):
        _execute(
            tmp_path,
            launcher=_FakeScorerLauncher(exit_code=7, write_outputs=False),
        )
    assert not (tmp_path / "stage" / "scorer_stage_manifest.json").exists()


@pytest.mark.parametrize(
    "attestation_overrides",
    [
        {"parent_pid": 1},
        {"run_nonce": _digest("wrong-nonce")},
        {"event_sequence": list(reversed(execution._SCORER_EVENT_SEQUENCE))},
        {"torch_imported": True},
        {"environment_key_names": ["PATH"]},
        {"loaded_file_backed_module_origins": []},
        {"process_sys_path": []},
        {
            "process_runtime_flags": {
                "dont_write_bytecode": True,
                "no_site": 1,
                "pycache_prefix": "wrong",
                "safe_path": True,
                "utf8_mode": 1,
            }
        },
        {"volvence_zero_namespace_search_locations": []},
    ],
)
def test_attestation_lineage_model_free_and_event_contract_are_strict(
    tmp_path: pathlib.Path,
    attestation_overrides: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        _execute(
            tmp_path,
            launcher=_FakeScorerLauncher(attestation_overrides=attestation_overrides),
        )
    assert not (tmp_path / "stage" / "scorer_stage_manifest.json").exists()


@pytest.mark.parametrize(
    "launcher",
    [
        _FakeScorerLauncher(tamper_frozen_source=True),
        _FakeScorerLauncher(swap_module_origin=True),
    ],
)
def test_frozen_source_bytes_and_module_name_path_mapping_are_strict(
    tmp_path: pathlib.Path,
    launcher: _FakeScorerLauncher,
) -> None:
    with pytest.raises(ValueError):
        _execute(tmp_path, launcher=launcher)
    assert not (tmp_path / "stage" / "scorer_stage_manifest.json").exists()


def test_report_cannot_upgrade_formal_or_product_claims(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="formal_evidence_authorized=false"):
        _execute(
            tmp_path,
            launcher=_FakeScorerLauncher(report_overrides={"formal_evidence_authorized": True}),
        )


@pytest.mark.parametrize(
    "launcher",
    [
        _FakeScorerLauncher(tamper_manifest_receipt=True),
        _FakeScorerLauncher(extra_output=True),
        _FakeScorerLauncher(contaminate_capsule=True),
    ],
)
def test_manifest_extra_output_and_capsule_contamination_are_rejected(
    tmp_path: pathlib.Path,
    launcher: _FakeScorerLauncher,
) -> None:
    with pytest.raises(ValueError):
        _execute(tmp_path, launcher=launcher)
    assert not (tmp_path / "stage" / "scorer_stage_manifest.json").exists()


def test_post_scorer_integrity_drift_blocks_stage_manifest(
    tmp_path: pathlib.Path,
) -> None:
    guard = _FakeIntegrityGuard(drift_phase="post_scorer")
    with pytest.raises(ValueError, match="expected source_tree_artifact_id"):
        _execute(tmp_path, guard=guard)

    assert [phase for phase, _previous in guard.calls] == [
        "pre_scorer",
        "post_scorer",
    ]
    assert not (tmp_path / "stage" / "scorer_stage_manifest.json").exists()


def test_pre_scorer_integrity_drift_stops_before_stage_or_launch(
    tmp_path: pathlib.Path,
) -> None:
    launcher = _FakeScorerLauncher()
    guard = _FakeIntegrityGuard(drift_phase="pre_scorer")
    with pytest.raises(ValueError, match="expected source_tree_artifact_id"):
        _execute(tmp_path, launcher=launcher, guard=guard)

    assert launcher.calls == []
    assert not (tmp_path / "stage").exists()


def test_scorer_bootstrap_only_lazy_imports_the_existing_scorer_api(
    tmp_path: pathlib.Path,
) -> None:
    result, _root, launcher, _guard = _execute(tmp_path)
    spec = launcher.calls[0]
    argv = execution._scorer_process_argv(spec)

    assert argv[1:10] == [
        "-P",
        "-S",
        "-B",
        "-u",
        "-X",
        "utf8",
        "-X",
        f"pycache_prefix={spec.pycache_prefix}",
        "-c",
    ]
    assert "relationship_condition_reader_qualification_scorer" in argv[10]
    assert "score_relationship_condition_reader_qualification as run" in argv[10]
    assert "torch" not in argv[10]
    assert "sentence_transformers" not in argv[10]
    assert result["scorer_process_exited"] is True


def test_suspended_job_constants_freeze_one_process_kill_on_close() -> None:
    assert execution._FORMAL_SCORER_CREATION_FLAGS & execution._CREATE_SUSPENDED
    assert execution._FORMAL_SCORER_CREATION_FLAGS & execution._CREATE_NO_WINDOW
    assert execution._FORMAL_SCORER_CREATION_FLAGS & execution._EXTENDED_STARTUPINFO_PRESENT
    assert execution._FORMAL_SCORER_JOB_LIMIT_FLAGS == (
        execution._JOB_OBJECT_LIMIT_ACTIVE_PROCESS | execution._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )


_ANCHOR_ID = _digest("public-anchor-receipt")


def _outer_protocol(
    execution_root: pathlib.Path,
    *,
    frozen_source_entries: Mapping[str, Mapping[str, object]],
    frozen_site_packages_root: pathlib.Path,
) -> dict[str, object]:
    return {
        "qualification_preflight": {
            "qualification_protocol_id": _QUALIFICATION_ID,
            "artifact_id": _digest("preflight-binding"),
            "files": [
                {
                    "path": "manifest.json",
                    "artifact_id": _digest("preflight-manifest"),
                },
                {
                    "path": "public/publication_request.json",
                    "artifact_id": _digest("publication-request"),
                },
            ],
        },
        "execution_source_tree": {
            "artifact_id": _SOURCE_TREE_ID,
            "entries": [{"path": path, **dict(receipt)} for path, receipt in sorted(frozen_source_entries.items())],
        },
        "bge_snapshot_tree": {"artifact_id": _BGE_TREE_ID},
        "runtime_identity": {
            "artifact_id": _RUNTIME_ID,
            "distributions": [{"dist_info_path": str(frozen_site_packages_root / "fixture-1.0.dist-info")}],
        },
        "proposed_execution_root": str(execution_root.resolve()),
    }


class _FakePredictionStage:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls: list[Mapping[str, object]] = []

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        self.order.append("prediction")
        self.calls.append(dict(kwargs))
        root = pathlib.Path(str(kwargs["execution_root"]))
        root.mkdir(parents=True)
        guard = kwargs["integrity_guard"]
        assert callable(guard)
        previous_id = str(kwargs["previous_integrity_receipt_artifact_id"])
        receipts = []
        for phase in (
            "pre_prediction_child_1",
            "post_prediction_child_1",
            "pre_prediction_child_2",
            "post_prediction_child_2",
        ):
            receipt = guard(
                phase=phase,
                previous_integrity_receipt_artifact_id=previous_id,
            )
            receipts.append(receipt)
            previous_id = str(receipt["artifact_id"])

        scoring_request_path, scoring_request = _scoring_request(root)
        child_request_id = _digest("outer-child-request")
        expected_ids = {
            "source_tree_artifact_id": kwargs["expected_source_tree_artifact_id"],
            "bge_snapshot_tree_artifact_id": kwargs["expected_bge_snapshot_tree_artifact_id"],
            "runtime_identity_artifact_id": kwargs["expected_runtime_identity_artifact_id"],
        }
        launcher_attestation = _with_artifact_id(
            {
                "schema_version": ("relationship-condition-reader-qualification-launcher-attestation.v1"),
                "qualification_protocol_id": _QUALIFICATION_ID,
                "execution_protocol_id": _EXECUTION_ID,
                "child_request_artifact_id": child_request_id,
                "run_nonce": _RUN_NONCE,
                "runs": [{"ordinal": 1}, {"ordinal": 2}],
                "run_count": 2,
                "integrity_receipts": receipts,
                "integrity_receipt_count": 4,
                "integrity_phase_order": [
                    "pre_prediction_child_1",
                    "post_prediction_child_1",
                    "pre_prediction_child_2",
                    "post_prediction_child_2",
                ],
                "previous_integrity_receipt_artifact_id": kwargs["previous_integrity_receipt_artifact_id"],
                "last_integrity_receipt_artifact_id": previous_id,
                "expected_integrity_artifact_ids": expected_ids,
                "processes_created_suspended": True,
                "job_assigned_before_initial_thread_resume": True,
                "job_kill_on_close": True,
                "job_active_process_limit": 1,
                "shell": False,
                "close_fds": True,
                "environment_built_from_empty_allowlist": True,
                "source_capsule_used": False,
                "repository_import_path_used": True,
                "bge_snapshot_tree_verified_by_launcher": False,
                "external_execution_anchor_verified": False,
                "qualification_execution_authorized": False,
                "os_security_boundary": False,
                "windows_directory_entry_durability_attested": False,
            }
        )
        _write_artifact(root / "launcher_attestation.json", launcher_attestation)
        return {
            "schema_version": ("relationship-condition-reader-qualification-executor-result.v1"),
            "qualification_protocol_id": _QUALIFICATION_ID,
            "execution_protocol_id": _EXECUTION_ID,
            "preflight_manifest_artifact_id": _digest("preflight-manifest"),
            "publication_request_artifact_id": _digest("publication-request"),
            "launcher_attestation_artifact_id": launcher_attestation["artifact_id"],
            "last_integrity_receipt_artifact_id": previous_id,
            "training_corpus_artifact_id": _digest("outer-training-corpus"),
            "child_request_artifact_id": child_request_id,
            "prediction_ledger_artifact_id": scoring_request["prediction_ledger_artifact_id"],
            "commit_receipt_artifact_id": scoring_request["commit_receipt_artifact_id"],
            "scoring_request_artifact_id": scoring_request["artifact_id"],
            "scoring_request_path": str(scoring_request_path.resolve()),
            "fresh_process_count": 2,
            "predictor_processes_exited": True,
            "predictor_job_objects_empty": True,
            "deterministic_outputs_byte_exact": True,
            "parent_opened_challenge_labels": False,
            "parent_opened_group_split": False,
            "scorer_launched": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
            "qualification_scored": False,
            "external_execution_anchor_verified": False,
            "qualification_execution_authorized": False,
            "reader_development_admission": False,
            "readable_claim_proven": False,
            "four_able_claim_proven": False,
        }


class _FakeScoringStage:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls: list[Mapping[str, object]] = []

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        self.order.append("scoring")
        self.calls.append(dict(kwargs))
        repository_root = pathlib.Path(str(kwargs["repository_root"]))
        return execution._execute_scoring_stage_with_launcher(
            scoring_request_path=pathlib.Path(str(kwargs["scoring_request_path"])),
            expected_scoring_request_artifact_id=str(kwargs["expected_scoring_request_artifact_id"]),
            stage_root=pathlib.Path(str(kwargs["stage_root"])),
            python_executable=pathlib.Path(str(kwargs["python_executable"])),
            environment_source=_environment_source(repository_root.parent / "windows"),
            launcher=_FakeScorerLauncher(),
            integrity_guard=kwargs["integrity_guard"],
            previous_integrity_receipt_artifact_id=str(kwargs["previous_integrity_receipt_artifact_id"]),
            expected_source_tree_artifact_id=str(kwargs["expected_source_tree_artifact_id"]),
            expected_bge_snapshot_tree_artifact_id=str(kwargs["expected_bge_snapshot_tree_artifact_id"]),
            expected_runtime_identity_artifact_id=str(kwargs["expected_runtime_identity_artifact_id"]),
            repository_root=repository_root,
            repository_source_roots=tuple(kwargs["repository_source_roots"]),
            frozen_source_entries=kwargs["frozen_source_entries"],
            frozen_site_packages_root=pathlib.Path(str(kwargs["frozen_site_packages_root"])),
        )


def _run_fake_outer(
    tmp_path: pathlib.Path,
    *,
    guard: _FakeIntegrityGuard | None = None,
    anchor_error: BaseException | None = None,
) -> tuple[Mapping[str, object], list[str], _FakeIntegrityGuard, pathlib.Path]:
    root = tmp_path.resolve() / "authorized_execution"
    (
        repository_root,
        _source_roots,
        frozen_source_entries,
        frozen_site_packages_root,
        python_executable,
        _system_root,
    ) = _source_binding_fixture(tmp_path)
    protocol = _outer_protocol(
        root,
        frozen_source_entries=frozen_source_entries,
        frozen_site_packages_root=frozen_site_packages_root,
    )
    protocol_raw = canonical_json_bytes(protocol)
    order: list[str] = []
    fake_guard = guard or _FakeIntegrityGuard()
    prediction = _FakePredictionStage(order)
    scoring = _FakeScoringStage(order)

    def validate_anchor(*_args: object, **_kwargs: object) -> str:
        order.append("anchor")
        assert not root.exists()
        if anchor_error is not None:
            raise anchor_error
        return _ANCHOR_ID

    def guard_factory() -> _FakeIntegrityGuard:
        order.append("guard_factory")
        return fake_guard

    result = execution._execute_authorized_qualification_with_stages(
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=_EXECUTION_ID,
        public_anchor_receipt_payload={"artifact_id": _ANCHOR_ID},
        expected_public_anchor_receipt_artifact_id=_ANCHOR_ID,
        repository_root=repository_root,
        preflight_root=tmp_path / "preflight",
        bge_snapshot_root=tmp_path / "snapshot",
        execution_root=root,
        run_nonce=_RUN_NONCE,
        python_executable=python_executable,
        prediction_timeout_seconds=30,
        scorer_timeout_seconds=30,
        anchor_validator=validate_anchor,
        integrity_guard_factory=guard_factory,
        prediction_stage=prediction,
        scoring_stage=scoring,
    )
    return result, order, fake_guard, root


def test_outer_runner_orders_anchor_all_integrity_phases_and_final_manifest(
    tmp_path: pathlib.Path,
) -> None:
    result, order, guard, root = _run_fake_outer(tmp_path)

    assert order == ["anchor", "guard_factory", "prediction", "scoring"]
    assert [phase for phase, _previous in guard.calls] == list(execution._INTEGRITY_PHASE_ORDINALS)
    assert result["integrity_receipt_count"] == 8
    assert result["external_execution_anchor_verified"] is True
    assert result["qualification_execution_authorized"] is True
    assert result["exact_source_reader_development_admitted"] is True
    for field_name in (
        "formal_evidence_authorized",
        "campaign_execution_admitted",
        "readable_product_effect",
        "appendable_product_effect",
        "learnable_product_effect",
        "steerable_product_effect",
        "four_able_complete",
        "human_product_validation",
        "production_active",
        "os_security_boundary",
        "windows_directory_entry_durability_attested",
    ):
        assert result[field_name] is False
    final_path = root / "final_manifest.json"
    assert strict_json_loads(final_path.read_bytes(), max_bytes=2_000_000) == result


def test_outer_runner_anchor_failure_precedes_guard_and_root_creation(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="anchor rejected"):
        _run_fake_outer(tmp_path, anchor_error=ValueError("anchor rejected"))

    assert not (tmp_path / "authorized_execution").exists()


def test_outer_runner_final_integrity_drift_blocks_final_manifest(
    tmp_path: pathlib.Path,
) -> None:
    guard = _FakeIntegrityGuard(drift_phase="final_validation")
    with pytest.raises(ValueError, match="expected source_tree_artifact_id"):
        _run_fake_outer(tmp_path, guard=guard)

    assert not (tmp_path / "authorized_execution" / "final_manifest.json").exists()


def _synthetic_prediction_values(
    condition_label: str,
) -> tuple[str, str, list[dict[str, object]]]:
    labels = ("agency_displacement", "belonging_erasure")
    scores = (0.8, 0.2) if condition_label == labels[0] else (0.2, 0.8)
    maximum = max(scores)
    exponentials = tuple(math.exp(score - maximum) for score in scores)
    confidence = max(exponentials) / math.fsum(exponentials)
    return (
        confidence.hex(),
        ((0.8 - 0.2) / 2.0).hex(),
        [{"label": label, "score_hex": score.hex()} for label, score in zip(labels, scores, strict=True)],
    )


def _persist_synthetic_scorer_fixture(
    root: pathlib.Path,
) -> tuple[pathlib.Path, Mapping[str, object]]:
    labels = ("agency_displacement", "belonging_erasure")
    public_corpus_id = _digest("smoke-public-corpus")
    child_request_id = _digest("smoke-child-request")
    predictor_request_id = _digest("smoke-predictor-request")
    embedding_table_id = _digest("smoke-embedding-table")
    reader_id = _digest("smoke-reader")
    challenge_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for group_index in range(28):
        group_id = _digest(f"smoke-group-{group_index:02d}")
        condition_label = labels[0] if group_index < 14 else labels[1]
        item_ids: list[str] = []
        for variant_index in range(8):
            item_id = _digest(f"smoke-item-{group_index:02d}-{variant_index:02d}")
            text_sha256 = _digest(f"smoke-text-{group_index:02d}-{variant_index:02d}")
            item_ids.append(item_id)
            challenge_rows.append(
                {
                    "item_id": item_id,
                    "text_sha256": text_sha256,
                    "condition_label": condition_label,
                    "group_id": group_id,
                    "subject_index": variant_index,
                    "surface_kind": ("onboarding" if group_index < 4 else "decision"),
                    "source_position": group_index,
                    "source_session_id": (f"smoke-session-{group_index:02d}-{variant_index:02d}"),
                }
            )
            confidence, margin, candidate_scores = _synthetic_prediction_values(condition_label)
            ledger_rows.append(
                {
                    "item_id": item_id,
                    "text_sha256": text_sha256,
                    "condition_label": condition_label,
                    "confidence_hex": confidence,
                    "normalized_margin_hex": margin,
                    "candidate_scores": candidate_scores,
                    "reader_artifact_id": reader_id,
                    "source_observation_sha256": text_sha256,
                }
            )
        group_rows.append(
            {
                "group_id": group_id,
                "item_ids": sorted(item_ids),
                "row_count": 8,
                "condition_label": condition_label,
            }
        )
    challenge_rows.sort(key=lambda row: str(row["item_id"]))
    ledger_rows.sort(key=lambda row: str(row["item_id"]))
    group_rows.sort(key=lambda row: str(row["group_id"]))

    ledger = _with_artifact_id(
        {
            "schema_version": "relationship-condition-reader-prediction-ledger.v1",
            "protocol_id": _QUALIFICATION_ID,
            "execution_protocol_id": _EXECUTION_ID,
            "child_request_artifact_id": child_request_id,
            "predictor_request_artifact_id": predictor_request_id,
            "embedding_table_artifact_id": embedding_table_id,
            "reader_artifact_id": reader_id,
            "rows": ledger_rows,
            "row_count": 224,
            "challenge_labels_present": False,
            "qualification_scored": False,
        }
    )
    challenge = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-challenge-labels.v1"),
            "protocol_id": _QUALIFICATION_ID,
            "public_corpus_artifact_id": public_corpus_id,
            "rows": challenge_rows,
            "row_count": 224,
            "label_release_condition": "prediction_ledger_create_only_fsynced",
        }
    )
    groups = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-group-split.v1"),
            "protocol_id": _QUALIFICATION_ID,
            "training_item_ids": sorted(_digest(f"smoke-training-item-{index}") for index in range(4)),
            "challenge_item_ids": [str(row["item_id"]) for row in challenge_rows],
            "challenge_groups": group_rows,
            "challenge_group_count": 28,
            "rows_per_challenge_group": 8,
            "training_challenge_text_overlap_count": 0,
            "statistical_independence_claim": False,
            "grouping_owner": "qualification_preflight",
            "grouping_contract": ("surface_kind_and_source_position_across_voice_variants.v1"),
            "group_level_evaluation_unit_count": 28,
        }
    )
    ledger_path = root / "prediction" / "prediction_ledger.json"
    challenge_path = root / "sealed" / "challenge_labels.json"
    groups_path = root / "sealed" / "group_split.json"
    ledger_receipt = _write_artifact(ledger_path, ledger)
    challenge_receipt = _write_artifact(challenge_path, challenge)
    groups_receipt = _write_artifact(groups_path, groups)
    commit = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-prediction-ledger-commit.v1"),
            "qualification_protocol_id": _QUALIFICATION_ID,
            "execution_protocol_id": _EXECUTION_ID,
            "child_request_artifact_id": child_request_id,
            "predictor_request_artifact_id": predictor_request_id,
            "prediction_ledger_artifact_id": ledger["artifact_id"],
            "prediction_ledger_raw_sha256": ledger_receipt["raw_sha256"],
            "prediction_ledger_raw_bytes": ledger_receipt["raw_bytes"],
            "prediction_run_manifest_artifact_ids": [
                _digest("smoke-run-manifest-1"),
                _digest("smoke-run-manifest-2"),
            ],
            "prediction_run_attestation_artifact_ids": [
                _digest("smoke-run-attestation-1"),
                _digest("smoke-run-attestation-2"),
            ],
            "fresh_process_count": 2,
            "predictor_processes_exited": True,
            "predictor_job_objects_empty": True,
            "embedding_tables_byte_exact": True,
            "reader_artifacts_byte_exact": True,
            "prediction_ledgers_byte_exact": True,
            "ledger_file_fsync_completed": True,
            "ledger_same_descriptor_readback": True,
            "ledger_closed_reopen_readback": True,
            "windows_directory_entry_durability_attested": False,
        }
    )
    commit_path = root / "prediction" / "commit_receipt.json"
    _write_artifact(commit_path, commit)
    request = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-scoring-request.v1"),
            "qualification_protocol_id": _QUALIFICATION_ID,
            "execution_protocol_id": _EXECUTION_ID,
            "run_nonce": _RUN_NONCE,
            "prediction_ledger_path": str(ledger_path.resolve()),
            "prediction_ledger_artifact_id": ledger["artifact_id"],
            "prediction_ledger_raw_sha256": ledger_receipt["raw_sha256"],
            "prediction_ledger_raw_bytes": ledger_receipt["raw_bytes"],
            "commit_receipt_path": str(commit_path.resolve()),
            "commit_receipt_artifact_id": commit["artifact_id"],
            "challenge_labels_path": str(challenge_path.resolve()),
            "challenge_labels_artifact_id": challenge["artifact_id"],
            "challenge_labels_raw_sha256": challenge_receipt["raw_sha256"],
            "challenge_labels_raw_bytes": challenge_receipt["raw_bytes"],
            "group_split_path": str(groups_path.resolve()),
            "group_split_artifact_id": groups["artifact_id"],
            "group_split_raw_sha256": groups_receipt["raw_sha256"],
            "group_split_raw_bytes": groups_receipt["raw_bytes"],
            "minimum_normalized_margin_hex": (0.01).hex(),
        }
    )
    request_path = root / "scoring_request.json"
    _write_artifact(request_path, request)
    return request_path, request


@pytest.mark.skipif(os.name != "nt", reason="requires the Windows Job Object API")
def test_windows_real_job_smoke_with_only_synthetic_scorer_inputs(
    tmp_path: pathlib.Path,
) -> None:
    request_path, request = _persist_synthetic_scorer_fixture(tmp_path / "synthetic_inputs")
    guard = _FakeIntegrityGuard()
    repository_root = pathlib.Path(__file__).resolve().parents[3]
    source_manifest = execution_protocol.build_relationship_condition_reader_execution_source_tree_manifest(
        repository_root=repository_root,
    )
    frozen_source_entries = {
        str(row["path"]): {
            "raw_sha256": row["raw_sha256"],
            "raw_bytes": row["raw_bytes"],
        }
        for row in source_manifest["entries"]
        if isinstance(row, Mapping)
    }
    repository_source_roots = tuple(
        sorted(
            {
                repository_root / pathlib.PurePosixPath(*pathlib.PurePosixPath(path).parts[:3])
                for path in frozen_source_entries
                if pathlib.PurePosixPath(path).parts[:1] == ("packages",)
            },
            key=lambda path: str(path).encode("utf-8"),
        )
    )
    frozen_site_packages_root = pathlib.Path(sys.executable).resolve().parent / "Lib" / "site-packages"

    result = execution.execute_relationship_condition_reader_qualification_scoring_stage(
        scoring_request_path=request_path,
        expected_scoring_request_artifact_id=str(request["artifact_id"]),
        stage_root=tmp_path / "real_windows_scorer_stage",
        integrity_guard=guard,
        previous_integrity_receipt_artifact_id=_PREVIOUS_INTEGRITY_ID,
        expected_source_tree_artifact_id=_SOURCE_TREE_ID,
        expected_bge_snapshot_tree_artifact_id=_BGE_TREE_ID,
        expected_runtime_identity_artifact_id=_RUNTIME_ID,
        repository_root=repository_root,
        repository_source_roots=repository_source_roots,
        frozen_source_entries=frozen_source_entries,
        frozen_site_packages_root=frozen_site_packages_root,
        python_executable=pathlib.Path(sys.executable),
        scorer_timeout_seconds=60,
    )

    assert result["scorer_process_id"] != os.getpid()
    assert result["process_created_suspended"] is True
    assert result["job_assigned_before_resume"] is True
    assert result["initial_thread_resume_previous_count"] == 1
    assert result["job_active_process_limit"] == 1
    assert result["scorer_job_object_empty"] is True
    assert result["scorer_process_exited"] is True
    assert result["scorer_exit_code"] == 0
    assert result["scorer_model_free"] is True
    assert result["model_or_cuda_used"] is False
    assert result["torch_imported"] is False
    assert result["sentence_transformers_imported"] is False
    assert result["exact_source_reader_development_admitted"] is True
