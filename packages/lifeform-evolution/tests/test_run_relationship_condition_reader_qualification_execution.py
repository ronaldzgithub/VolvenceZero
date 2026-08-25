from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import stat
from types import ModuleType, SimpleNamespace

import pytest

from volvence_zero.canonical_json import canonical_json_bytes


_REPOSITORY_ROOT = pathlib.Path(__file__).parents[3]
_SCRIPT_PATH = _REPOSITORY_ROOT / "scripts" / "run_relationship_condition_reader_qualification_execution.py"


@pytest.fixture
def cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_relationship_condition_reader_qualification_execution_cli",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("qualification execution CLI import spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(core: dict[str, object]) -> dict[str, object]:
    return {**core, "artifact_id": hashlib.sha256(canonical_json_bytes(core)).hexdigest()}


def _protocol_fixture(
    *,
    execution_root: pathlib.Path,
    anchor_relative_path: str,
) -> dict[str, object]:
    return {
        "schema_version": ("relationship-condition-reader-qualification-execution-protocol.v1"),
        "qualification_preflight": {
            "qualification_protocol_id": "1" * 64,
            "artifact_id": "2" * 64,
        },
        "execution_source_tree": {
            "artifact_id": "3" * 64,
            "entry_count": 17,
        },
        "bge_snapshot_tree": {
            "artifact_id": "4" * 64,
            "entry_count": 11,
        },
        "runtime_identity": {"artifact_id": "5" * 64},
        "proposed_execution_root": str(execution_root),
        "external_public_anchor": {
            "receipt_path": anchor_relative_path,
            "gist_owner": "ronaldzgithub",
            "filename": ("relationship_condition_reader_qualification_execution_v1.json"),
        },
    }


def _write_canonical(path: pathlib.Path, payload: dict[str, object]) -> bytes:
    raw = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def test_parser_is_lazy_requires_external_execution_authority_and_has_no_force(
    cli_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "_install_workspace_source_roots", lambda: ())

    def forbidden_import() -> object:
        raise AssertionError("argument parsing must not import qualification modules")

    monkeypatch.setattr(cli_module, "_load_protocol_module", forbidden_import)
    monkeypatch.setattr(cli_module, "_load_execution_module", forbidden_import)

    with pytest.raises(SystemExit) as help_exit:
        cli_module.main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "freeze-protocol" in help_text
    assert "record-anchor" in help_text
    assert "validate-anchor" in help_text
    assert "execute" in help_text
    assert "--force" not in help_text

    parser = cli_module._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "execute",
                "--execution-protocol-path",
                str(_SCRIPT_PATH),
                "--expected-execution-protocol-id",
                "a" * 64,
            ]
        )

    args = parser.parse_args(
        [
            "execute",
            "--execution-protocol-path",
            str(_SCRIPT_PATH),
            "--expected-execution-protocol-id",
            "a" * 64,
            "--anchor-receipt-path",
            str(_SCRIPT_PATH),
            "--expected-anchor-receipt-artifact-id",
            "b" * 64,
            "--expected-execution-root",
            str(_REPOSITORY_ROOT / "future-output"),
            "--preflight-root",
            str(_REPOSITORY_ROOT),
            "--bge-snapshot-root",
            str(_REPOSITORY_ROOT),
            "--run-nonce",
            "fixed-external-run-nonce",
        ]
    )
    assert args.expected_execution_protocol_id == "a" * 64
    assert args.expected_anchor_receipt_artifact_id == "b" * 64
    assert not hasattr(args, "force")


def test_source_root_bootstrap_is_sorted_and_rejects_reparse_entries(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    first = repository / "packages" / "alpha" / "src"
    second = repository / "packages" / "zeta" / "src"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    monkeypatch.setattr(cli_module, "_SCRIPT_PATH", repository / "scripts" / "cli.py")
    (repository / "scripts").mkdir()
    (repository / "scripts" / "cli.py").write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr(cli_module.sys, "path", ["tail-entry"])

    roots = cli_module._install_workspace_source_roots(repository)

    assert roots == (first, second)
    assert cli_module.sys.path[:3] == [str(first), str(second), "tail-entry"]

    reparse_repository = tmp_path / "reparse-repository"
    (reparse_repository / "packages").mkdir(parents=True)
    (reparse_repository / "scripts").mkdir()
    reparse_script = reparse_repository / "scripts" / "cli.py"
    reparse_script.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_SCRIPT_PATH", reparse_script)
    try:
        os.symlink(
            repository / "packages" / "alpha",
            reparse_repository / "packages" / "linked-package",
            target_is_directory=True,
        )
    except OSError:
        linked_package = reparse_repository / "packages" / "linked-package"
        (linked_package / "src").mkdir(parents=True)
        real_lstat = cli_module.os.lstat

        def simulated_reparse_lstat(path: object) -> object:
            if pathlib.Path(path) == linked_package:
                return SimpleNamespace(
                    st_mode=stat.S_IFLNK,
                    st_file_attributes=0,
                )
            return real_lstat(path)

        monkeypatch.setattr(cli_module.os, "lstat", simulated_reparse_lstat)
    with pytest.raises(ValueError, match="symlink or reparse"):
        cli_module._install_workspace_source_roots(reparse_repository)


def test_freeze_protocol_builds_in_frozen_order_and_writes_create_only_canonical_lf(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    bge = tmp_path / "bge"
    bge.mkdir()
    execution_root = tmp_path / "execution"
    output = tmp_path / "protocol.json"
    anchor_relative = "artifacts/relationship_lab/qualification-anchor.json"
    protocol = _protocol_fixture(
        execution_root=execution_root,
        anchor_relative_path=anchor_relative,
    )
    calls: list[str] = []

    fake_protocol_module = SimpleNamespace(
        build_relationship_condition_reader_execution_preflight_binding=(
            lambda **_kwargs: calls.append("preflight") or protocol["qualification_preflight"]
        ),
        build_relationship_condition_reader_execution_source_tree_manifest=(
            lambda **_kwargs: calls.append("source") or protocol["execution_source_tree"]
        ),
        build_bge_m3_snapshot_tree_manifest=(lambda **_kwargs: calls.append("bge") or protocol["bge_snapshot_tree"]),
        build_relationship_condition_reader_qualification_runtime_identity=(
            lambda: calls.append("runtime") or protocol["runtime_identity"]
        ),
        build_relationship_condition_reader_qualification_execution_protocol=(
            lambda **_kwargs: calls.append("compose") or protocol
        ),
        relationship_condition_reader_qualification_execution_protocol_id=(lambda _payload: "a" * 64),
        validate_relationship_condition_reader_qualification_execution_protocol=(
            lambda _payload, **_kwargs: calls.append("validate") or "a" * 64
        ),
    )
    monkeypatch.setattr(cli_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(cli_module, "_install_workspace_source_roots", lambda: ())
    monkeypatch.setattr(
        cli_module,
        "_load_protocol_module",
        lambda: fake_protocol_module,
    )

    def forbidden_execution_import() -> object:
        raise AssertionError("freeze-protocol must not import execution/model code")

    monkeypatch.setattr(
        cli_module,
        "_load_execution_module",
        forbidden_execution_import,
    )
    argv = [
        "freeze-protocol",
        "--preflight-root",
        str(preflight),
        "--expected-qualification-protocol-id",
        "1" * 64,
        "--bge-snapshot-root",
        str(bge),
        "--proposed-execution-root",
        str(execution_root),
        "--anchor-receipt-relative-path",
        anchor_relative,
        "--protocol-output-path",
        str(output),
    ]

    assert cli_module.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert calls == ["preflight", "source", "bge", "runtime", "compose", "validate"]
    assert output.read_bytes() == canonical_json_bytes(protocol) + b"\n"
    assert summary["execution_protocol_id"] == "a" * 64
    assert summary["qualification_execution_authorized"] is False
    assert summary["model_or_cuda_execution_used"] is False
    assert summary["windows_directory_entry_durability_attested"] is False

    with pytest.raises(FileExistsError, match="create-only"):
        cli_module.main(argv)


def test_record_anchor_requires_byte_exact_external_observation_and_writes_bound_path(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    anchor_relative = "artifacts/relationship_lab/qualification-anchor.json"
    anchor_path = repository / pathlib.PurePosixPath(anchor_relative)
    anchor_path.parent.mkdir(parents=True)
    execution_root = tmp_path / "future-execution"
    protocol = _protocol_fixture(
        execution_root=execution_root,
        anchor_relative_path=anchor_relative,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_raw = _write_canonical(protocol_path, protocol)
    observed_path = tmp_path / "downloaded-public-protocol.json"
    observed_path.write_bytes(protocol_raw)
    receipt = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-execution-public-anchor-receipt.v1"),
            "execution_protocol_id": "a" * 64,
            "execution_root": str(execution_root),
        }
    )
    builder_calls: list[dict[str, object]] = []

    def build_receipt(**kwargs: object) -> dict[str, object]:
        builder_calls.append(kwargs)
        return receipt

    fake_protocol_module = SimpleNamespace(
        load_relationship_condition_reader_qualification_execution_protocol=(
            lambda _path, **_kwargs: (protocol, protocol_raw)
        ),
        build_relationship_condition_reader_qualification_public_anchor_receipt=(build_receipt),
        validate_relationship_condition_reader_qualification_public_anchor_receipt=(
            lambda payload, **_kwargs: payload["artifact_id"]
        ),
    )
    monkeypatch.setattr(cli_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(cli_module, "_install_workspace_source_roots", lambda: ())
    monkeypatch.setattr(
        cli_module,
        "_load_protocol_module",
        lambda: fake_protocol_module,
    )
    argv = [
        "record-anchor",
        "--execution-protocol-path",
        str(protocol_path),
        "--expected-execution-protocol-id",
        "a" * 64,
        "--observed-protocol-raw-path",
        str(observed_path),
        "--gist-id",
        "b" * 32,
        "--history-version",
        "c" * 40,
        "--created-at",
        "2026-08-24T12:00:00Z",
        "--updated-at",
        "2026-08-24T12:00:00Z",
        "--api-raw-url",
        "https://gist.githubusercontent.com/example/raw/file",
        "--revision-raw-url",
        "https://gist.githubusercontent.com/example/raw/revision/file",
        "--observed-at-utc",
        "2026-08-24T12:01:00Z",
        "--expected-execution-root",
        str(execution_root),
    ]

    assert cli_module.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert builder_calls[0]["observed_protocol_raw"] == protocol_raw
    assert builder_calls[0]["gist_owner"] == "ronaldzgithub"
    assert builder_calls[0]["gist_url"] == (f"https://gist.github.com/ronaldzgithub/{'b' * 32}")
    assert builder_calls[0]["history_revision_count"] == 1
    assert builder_calls[0]["first_revision"] is True
    assert builder_calls[0]["observation_transport"] == ("unauthenticated_github_rest_api_and_raw_http")
    real_protocol_module = importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_execution_protocol"
    )
    assert set(builder_calls[0]) == set(
        inspect.signature(
            real_protocol_module.build_relationship_condition_reader_qualification_public_anchor_receipt
        ).parameters
    )
    assert anchor_path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert summary["anchor_receipt_artifact_id"] == receipt["artifact_id"]
    assert summary["caller_supplied_github_observation_contract_validated"] is True
    assert summary["github_reobservation_performed_by_cli"] is False
    assert summary["external_public_anchor_independently_verified_by_cli"] is False
    assert summary["qualification_execution_authorized"] is False
    assert "--expected-anchor-receipt-artifact-id" in summary["next_required_external_input"]

    observed_path.write_bytes(protocol_raw + b" ")
    with pytest.raises(ValueError, match="do not match"):
        cli_module.main(argv)


def test_validate_and_execute_require_bound_receipt_and_forward_external_ids(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    anchor_relative = "artifacts/relationship_lab/qualification-anchor.json"
    anchor_path = repository / pathlib.PurePosixPath(anchor_relative)
    execution_root = tmp_path / "future-execution"
    protocol = _protocol_fixture(
        execution_root=execution_root,
        anchor_relative_path=anchor_relative,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_raw = _write_canonical(protocol_path, protocol)
    receipt = _artifact({"schema_version": "test-anchor.v1"})
    receipt_raw = _write_canonical(anchor_path, receipt)
    anchor_validations: list[dict[str, object]] = []
    runner_calls: list[dict[str, object]] = []

    def validate_anchor(_receipt: object, **kwargs: object) -> str:
        anchor_validations.append(kwargs)
        return "b" * 64

    fake_protocol_module = SimpleNamespace(
        load_relationship_condition_reader_qualification_execution_protocol=(
            lambda _path, **_kwargs: (protocol, protocol_raw)
        ),
        validate_relationship_condition_reader_qualification_public_anchor_receipt=(validate_anchor),
    )

    def run_outer(**kwargs: object) -> dict[str, object]:
        runner_calls.append(kwargs)
        return {
            "schema_version": ("relationship-condition-reader-qualification-authorized-execution-manifest.v1"),
            "artifact_id": "f" * 64,
        }

    fake_execution_module = SimpleNamespace(
        execute_authorized_relationship_condition_reader_qualification_execution=(run_outer)
    )
    monkeypatch.setattr(cli_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(cli_module, "_install_workspace_source_roots", lambda: ())
    monkeypatch.setattr(
        cli_module,
        "_load_protocol_module",
        lambda: fake_protocol_module,
    )
    monkeypatch.setattr(
        cli_module,
        "_load_execution_module",
        lambda: fake_execution_module,
    )
    shared = [
        "--execution-protocol-path",
        str(protocol_path),
        "--expected-execution-protocol-id",
        "a" * 64,
        "--anchor-receipt-path",
        str(anchor_path),
        "--expected-anchor-receipt-artifact-id",
        "b" * 64,
        "--expected-execution-root",
        str(execution_root),
    ]

    assert cli_module.main(["validate-anchor", *shared]) == 0
    validation_summary = json.loads(capsys.readouterr().out)
    assert validation_summary["public_anchor_receipt_contract_validated"] is True
    assert validation_summary["github_reobservation_performed_by_cli"] is False
    assert validation_summary["external_public_anchor_independently_verified_by_cli"] is False
    assert validation_summary["qualification_execution_performed"] is False
    assert validation_summary["anchor_receipt_raw_sha256"] == hashlib.sha256(receipt_raw).hexdigest()

    preflight = tmp_path / "preflight"
    preflight.mkdir()
    bge = tmp_path / "bge"
    bge.mkdir()
    assert (
        cli_module.main(
            [
                "execute",
                *shared,
                "--preflight-root",
                str(preflight),
                "--bge-snapshot-root",
                str(bge),
                "--run-nonce",
                "externally-fixed-run-nonce",
                "--prediction-timeout-seconds",
                "73",
                "--scorer-timeout-seconds",
                "19",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["artifact_id"] == "f" * 64
    assert len(anchor_validations) == 2
    call = runner_calls[0]
    assert call["expected_execution_protocol_id"] == "a" * 64
    assert call["expected_public_anchor_receipt_artifact_id"] == "b" * 64
    assert call["public_anchor_receipt_payload"] == receipt
    assert call["execution_protocol_raw"] == protocol_raw
    assert call["execution_root"] == execution_root
    assert call["run_nonce"] == "externally-fixed-run-nonce"
    assert call["prediction_timeout_seconds"] == 73
    assert call["scorer_timeout_seconds"] == 19
    real_execution_module = importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_execution"
    )
    assert set(call) == set(
        inspect.signature(
            real_execution_module.execute_authorized_relationship_condition_reader_qualification_execution
        ).parameters
    )

    wrong_receipt = tmp_path / "wrong-receipt.json"
    _write_canonical(wrong_receipt, receipt)
    wrong_shared = [value if value != str(anchor_path) else str(wrong_receipt) for value in shared]
    with pytest.raises(ValueError, match="pre-registered"):
        cli_module.main(["validate-anchor", *wrong_shared])


def test_canonical_artifact_loader_rejects_noncanonical_json(
    cli_module: ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{"value": 1}\n')
    with pytest.raises(ValueError, match="canonical LF-terminated"):
        cli_module._load_canonical_json_object(
            noncanonical,
            label="fixture",
            max_bytes=100,
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"value":1,"value":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        cli_module._load_canonical_json_object(
            duplicate,
            label="fixture",
            max_bytes=100,
        )
